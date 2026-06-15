"""Standalone VAE encode: load HuggingFace VAE weights and encode image/video to latents.

Implements the Wan2.2 encoder architecture (inverse of decoder):
- Spatial patchification (pixel-space → patched input)
- WanResidualDownBlock: resnets + downsample + AvgDown3D residual shortcut
- Mid-block self-attention
- quant_conv → mean extraction (argmax mode)
- Per-channel normalization
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .decode_vae import (
    _transpose_conv3d_weight,
    _transpose_conv2d_weight,
    _conv3d_forward,
    _rms_norm,
    _resnet_block,
    _attention_block,
)


def _nearest_downsample_2x(x_2d: mx.array) -> mx.array:
    """Strided 2x spatial downsampling for a 2D tensor [B, H, W, C]."""
    return x_2d[:, ::2, ::2, :]


def _wan_resample_downsample2d(x: mx.array, weights: dict, prefix: str) -> mx.array:
    """WanResample downsample2d: learned Conv2d with stride 2 per frame.

    Args:
        x: [B, T, H, W, C] channels-last
        weights: dict with keys like "{prefix}.resample.1.weight", "{prefix}.resample.1.bias"
    """
    b, t, h, w, c = x.shape

    frames = []
    for ti in range(t):
        frame = x[:, ti]  # [B, H, W, C]
        # HF uses ZeroPad2d((0,1,0,1)) → asymmetric padding: right and bottom
        # In channels-last: pad W (right) and H (bottom) by 1
        frame = mx.pad(frame, [(0, 0), (0, 1), (0, 1), (0, 0)])
        conv_w = weights[f"{prefix}.resample.1.weight"]
        conv_b = weights.get(f"{prefix}.resample.1.bias")
        frame = mx.conv2d(frame, conv_w, stride=(2, 2), padding=(0, 0))
        if conv_b is not None:
            frame = frame + conv_b
        frames.append(frame)

    return mx.stack(frames, axis=1)


def _wan_resample_downsample3d(x: mx.array, weights: dict, prefix: str) -> mx.array:
    """WanResample downsample3d: spatial downsample only (single-pass mode).

    In HF's WanResample, the temporal time_conv is only applied during cached/chunked
    encoding (feat_cache is not None). For single-pass encoding (our case), only the
    spatial Conv2d stride-2 downsample runs. The temporal conv is skipped.

    Args:
        x: [B, T, H, W, C] channels-last
        weights: dict with keys for time_conv and resample
    """
    # Only spatial downsample — temporal conv skipped in single-pass mode
    return _wan_resample_downsample2d(x, weights, prefix)


def _avg_down_3d(x: mx.array, in_channels: int, out_channels: int,
                 factor_t: int, factor_s: int) -> mx.array:
    """AvgDown3D: parameter-free channel-reshaping average-pool residual shortcut.

    Matches HF's AvgDown3D exactly. Reshapes input by interleaving spatial/temporal
    factors into channels, groups, and averages to produce the target channel count
    at downsampled resolution.

    Args:
        x: [B, T, H, W, C] channels-last
        in_channels: input channel count
        out_channels: output channel count
        factor_t: temporal downsampling factor (1 or 2)
        factor_s: spatial downsampling factor (1 or 2)

    Returns:
        [B, T//factor_t, H//factor_s, W//factor_s, out_channels]
    """
    factor = factor_t * factor_s * factor_s
    group_size = in_channels * factor // out_channels

    b, t, h, w, c = x.shape

    # Pad temporal if needed
    pad_t = (factor_t - t % factor_t) % factor_t
    if pad_t > 0:
        x = mx.pad(x, [(0, 0), (pad_t, 0), (0, 0), (0, 0), (0, 0)])
        t = t + pad_t

    # HF layout is [B, C, T, H, W]. We work in [B, T, H, W, C].
    # HF reshapes: [B, C, T//ft, ft, H//fs, fs, W//fs, fs]
    # then permutes: [B, C, ft, fs, fs, T//ft, H//fs, W//fs]
    # then views: [B, C*factor, T//ft, H//fs, W//fs]
    # then views: [B, out_channels, group_size, T//ft, H//fs, W//fs]
    # then means over group_size dim.
    #
    # In channels-last, the equivalent is:
    # 1. Reshape to expose factors
    # 2. Move factors next to channels
    # 3. Reshape to [B, T', H', W', C*factor]
    # 4. Reshape to [B, T', H', W', out_channels, group_size]
    # 5. Mean over group_size

    t_out = t // factor_t
    h_out = h // factor_s
    w_out = w // factor_s

    # [B, T, H, W, C] -> [B, T//ft, ft, H//fs, fs, W//fs, fs, C]
    x = x.reshape(b, t_out, factor_t, h_out, factor_s, w_out, factor_s, c)
    # Move factors next to C: [B, T', H', W', C, ft, fs, fs]
    x = mx.transpose(x, (0, 1, 3, 5, 7, 2, 4, 6))
    # Collapse factors into channels: [B, T', H', W', C*factor]
    x = x.reshape(b, t_out, h_out, w_out, c * factor)
    # Group and average: [B, T', H', W', out_channels, group_size]
    x = x.reshape(b, t_out, h_out, w_out, out_channels, group_size)
    x = mx.mean(x, axis=-1)

    return x


def _patchify_input(x: mx.array, patch_size: int = 2) -> mx.array:
    """Patchify pixel-space input: pack spatial patches into channels.

    Input: [B, T, H, W, 3]
    Output: [B, T, H//p, W//p, 3*p*p]

    Matches HF's patchify() exactly:
    HF: [B,C,T,H,W] -> view [B,C,T,H//p,p,W//p,p] -> permute [B,C,p,p,T,H//p,W//p]
        -> view [B, C*p*p, T, H//p, W//p]

    In channels-last: [B,T,H,W,C] -> [B,T,H//p,p,W//p,p,C]
        -> permute to [B,T,H//p,W//p,C,p,p] -> [B,T,H//p,W//p,C*p*p]
    """
    b, t, h, w, c = x.shape
    p = patch_size
    # [B, T, H, W, C] -> [B, T, H//p, p, W//p, p, C]
    x = x.reshape(b, t, h // p, p, w // p, p, c)
    # HF permute is (0,1,6,4,2,3,5) on [B,C,T,H//p,p,W//p,p]
    # which gives [B,C,p_w,p_h,T,H//p,W//p] — i.e. channel order is C,p_w,p_h
    # In channels-last: [B,T,H//p,W//p,C,p_w,p_h]
    # From [B,T,H//p,p_h,W//p,p_w,C] -> [B,T,H//p,W//p,C,p_w,p_h]
    x = mx.transpose(x, (0, 1, 2, 4, 6, 5, 3))
    x = x.reshape(b, t, h // p, w // p, c * p * p)
    return x


def encode_video(
    video: np.ndarray | mx.array,
    vae_dir: str,
) -> mx.array:
    """Encode a video (or single image) to normalized VAE latents.

    Args:
        video: [T, H, W, 3] or [H, W, 3] uint8/float32 in [0, 1].
               For multi-frame input, temporal causal convolutions produce
               per-frame latents that depend on preceding frames.
        vae_dir: path to vae/ directory with config.json and safetensors

    Returns:
        [1, T_lat, H//16, W//16, z_dim] normalized latents (channels-last)
        where T_lat = T for single-pass encoding.
    """
    vae_path = Path(vae_dir)

    with open(vae_path / "config.json") as f:
        config = json.load(f)

    # Load and prepare weights
    raw_weights = mx.load(str(vae_path / "diffusion_pytorch_model.safetensors"))

    weights = {}
    qc_weight = None
    qc_bias = None
    for k, v in raw_weights.items():
        if k == "quant_conv.weight":
            qc_weight = _transpose_conv3d_weight(v).astype(mx.bfloat16)
            continue
        if k == "quant_conv.bias":
            qc_bias = v.astype(mx.bfloat16)
            continue
        if not k.startswith("encoder."):
            continue
        name = k[len("encoder."):]

        if "conv" in name and name.endswith(".weight") and v.ndim == 5:
            v = _transpose_conv3d_weight(v)
        elif name.endswith(".weight") and v.ndim == 4:
            v = _transpose_conv2d_weight(v)

        weights[name] = v.astype(mx.bfloat16)

    # Prepare input: normalize to [-1, 1], shape [1, T, H, W, 3]
    if isinstance(video, np.ndarray):
        if video.dtype == np.uint8:
            video = video.astype(np.float32) / 255.0
        x = mx.array(video)
    else:
        x = video

    if x.ndim == 3:
        x = mx.expand_dims(mx.expand_dims(x, 0), 0)  # [H,W,3] -> [1, 1, H, W, 3]
    elif x.ndim == 4:
        x = mx.expand_dims(x, 0)  # [T,H,W,3] -> [1, T, H, W, 3]
    x = x * 2.0 - 1.0  # [0,1] -> [-1,1]
    x = x.astype(mx.bfloat16)

    # Patchify: [1, 1, H, W, 3] -> [1, 1, H//2, W//2, 12]
    patch_size = config.get("patch_size", 2)
    if patch_size is not None and patch_size > 1:
        x = _patchify_input(x, patch_size)
    print(f"    After patchify: {x.shape}")

    # conv_in
    x = _conv3d_forward(x, weights["conv_in.weight"], weights.get("conv_in.bias"))
    mx.eval(x)
    print(f"    After conv_in: {x.shape}")

    # Architecture config
    is_residual = config.get("is_residual", False)
    temporal_downsample = config.get("temperal_downsample", [False, True, True])
    dim_mult = config.get("dim_mult", [1, 2, 4, 4])
    base_dim = config.get("base_dim", 160)

    # Channel dimensions: [base_dim*1, base_dim*1, base_dim*2, base_dim*4, base_dim*4]
    dims = [base_dim * u for u in [1] + dim_mult]

    # down_blocks
    num_down_blocks = len(dim_mult)

    for block_idx in range(num_down_blocks):
        prefix = f"down_blocks.{block_idx}"
        in_dim = dims[block_idx]
        out_dim = dims[block_idx + 1]

        # Save input for residual shortcut (WanResidualDownBlock)
        x_copy = x if is_residual else None

        # Count resnets
        resnet_ids = set()
        for k in weights:
            if k.startswith(f"{prefix}.resnets."):
                ri = int(k.split(".")[3])
                resnet_ids.add(ri)
        num_resnets = len(resnet_ids)

        for ri in range(num_resnets):
            x = _resnet_block(x, weights, f"{prefix}.resnets.{ri}")
            mx.eval(x)

        # Downsampling if this block has a downsampler
        has_downsampler = any(k.startswith(f"{prefix}.downsampler.") for k in weights)
        down_flag = block_idx != len(dim_mult) - 1
        t_down = temporal_downsample[block_idx] if block_idx < len(temporal_downsample) and down_flag else False

        if has_downsampler:
            has_time_conv = f"{prefix}.downsampler.time_conv.weight" in weights
            if has_time_conv and t_down:
                x = _wan_resample_downsample3d(x, weights, f"{prefix}.downsampler")
            else:
                x = _wan_resample_downsample2d(x, weights, f"{prefix}.downsampler")
            mx.eval(x)

        # AvgDown3D residual shortcut (WanResidualDownBlock)
        if is_residual:
            factor_t = 2 if t_down else 1
            factor_s = 2 if down_flag else 1
            shortcut = _avg_down_3d(x_copy, in_dim, out_dim, factor_t, factor_s)
            x = x + shortcut
            mx.eval(x)

        print(f"    After down_block {block_idx}: {x.shape}")

    # mid_block: resnet[0] -> attention[0] -> resnet[1]
    x = _resnet_block(x, weights, "mid_block.resnets.0")
    mx.eval(x)

    attn_key = "mid_block.attentions.0.to_qkv.weight"
    if attn_key in weights:
        x = _attention_block(x, weights, "mid_block.attentions.0")
        mx.eval(x)

    x = _resnet_block(x, weights, "mid_block.resnets.1")
    mx.eval(x)
    print(f"    After mid_block: {x.shape}")

    # norm_out + silu + conv_out
    x = _rms_norm(x, weights["norm_out.gamma"])
    x = nn.silu(x)
    x = _conv3d_forward(x, weights["conv_out.weight"], weights.get("conv_out.bias"))
    mx.eval(x)
    print(f"    After conv_out: {x.shape}")

    # quant_conv: [1, 1, H_lat, W_lat, 96] -> [1, 1, H_lat, W_lat, 96]
    if qc_weight is not None:
        x = _conv3d_forward(x, qc_weight, qc_bias,
                            stride=(1, 1, 1), padding=(0, 0, 0), causal=False)
        mx.eval(x)

    # Extract mean (argmax mode): first z_dim channels
    z_dim = config.get("z_dim", 48)
    mu = x[..., :z_dim]
    print(f"    Raw latent mu: mean={mx.mean(mu).item():.3f}, std={mx.std(mu).item():.3f}")

    # Normalize: z_norm = (mu - mean) * inv_std
    latents_mean = config.get("latents_mean", [0.0] * z_dim)
    latents_std = config.get("latents_std", [1.0] * z_dim)
    mean = mx.array(latents_mean, dtype=mu.dtype)
    inv_std = 1.0 / mx.array(latents_std, dtype=mu.dtype)
    z_norm = (mu - mean) * inv_std
    print(f"    Normalized latent: mean={mx.mean(z_norm).item():.3f}, std={mx.std(z_norm).item():.3f}")

    return z_norm


# Backward-compatible alias
encode_image = encode_video
