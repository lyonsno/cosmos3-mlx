"""Standalone VAE decode: load HuggingFace VAE weights and decode latents to pixels.

This bypasses our WanDecoder model class and directly loads/runs the HF decoder
weights, handling the Conv3D weight transposition and naming differences.

For the first smoke, this is faster than rewriting the VAE model to match HF exactly.
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def _transpose_conv3d_weight(w: mx.array) -> mx.array:
    """Transpose Conv3D weight from PyTorch [O,I,D,H,W] to MLX-friendly [O,D,H,W,I]."""
    return mx.transpose(w, (0, 2, 3, 4, 1))


def _conv3d_forward(x: mx.array, weight: mx.array, bias: mx.array = None,
                    stride=(1,1,1), padding=(1,1,1), causal=True) -> mx.array:
    """Run Conv3D via per-frame 2D decomposition.

    Args:
        x: [B, T, H, W, C] channels-last
        weight: [O, kD, kH, kW, I] MLX layout
        bias: [O] optional
        stride: (sD, sH, sW)
        padding: (pD, pH, pW)
        causal: if True, use causal temporal padding
    """
    b, t, h, w, c = x.shape
    o_ch, kd, kh, kw, i_ch = weight.shape
    sd, sh, sw = stride

    # Causal temporal padding
    if causal:
        pad_t = 2 * padding[0]
        pad_zeros = mx.zeros((b, pad_t, h, w, c), dtype=x.dtype)
        x = mx.concatenate([pad_zeros, x], axis=1)
    else:
        # Symmetric padding
        pad_t = padding[0]
        if pad_t > 0:
            pad_zeros = mx.zeros((b, pad_t, h, w, c), dtype=x.dtype)
            x = mx.concatenate([pad_zeros, x, pad_zeros], axis=1)

    t_padded = x.shape[1]
    t_out = (t_padded - kd) // sd + 1

    outputs = []
    for ti in range(t_out):
        t_start = ti * sd
        accum = None
        for d in range(kd):
            frame = x[:, t_start + d]  # [B, H, W, C]
            w_2d = weight[:, d, :, :, :]  # [O, kH, kW, I]
            conv_out = mx.conv2d(frame, w_2d, stride=(sh, sw),
                                 padding=(padding[1], padding[2]))
            accum = conv_out if accum is None else accum + conv_out
        if bias is not None:
            accum = accum + bias
        outputs.append(accum)

    return mx.stack(outputs, axis=1)


def _rms_norm(x: mx.array, gamma: mx.array, eps: float = 1e-6) -> mx.array:
    """RMS norm with gamma. Gamma may have trailing singleton dims."""
    # Squeeze gamma to 1D
    g = gamma.reshape(-1)
    rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)
    return x / rms * g


def _resnet_block(x: mx.array, weights: dict, prefix: str) -> mx.array:
    """Run one residual block."""
    residual = x

    x = _rms_norm(x, weights[f"{prefix}.norm1.gamma"])
    x = nn.silu(x)
    x = _conv3d_forward(x,
                        weights[f"{prefix}.conv1.weight"],
                        weights.get(f"{prefix}.conv1.bias"))
    mx.eval(x)

    x = _rms_norm(x, weights[f"{prefix}.norm2.gamma"])
    x = nn.silu(x)
    x = _conv3d_forward(x,
                        weights[f"{prefix}.conv2.weight"],
                        weights.get(f"{prefix}.conv2.bias"))
    mx.eval(x)

    # Skip connection (conv_shortcut for channel changes)
    skip_key = f"{prefix}.conv_shortcut.weight"
    if skip_key in weights:
        residual = _conv3d_forward(residual,
                                   weights[skip_key],
                                   weights.get(f"{prefix}.conv_shortcut.bias"),
                                   padding=(0, 0, 0))

    return x + residual


def _dup_up_3d(x: mx.array, temporal: bool = False) -> mx.array:
    """Duplicate-based upsampling."""
    x = mx.repeat(x, 2, axis=2)  # H
    x = mx.repeat(x, 2, axis=3)  # W
    if temporal:
        x = mx.repeat(x, 2, axis=1)  # T
    return x


def decode_latents(
    latents: mx.array,
    vae_dir: str,
    latents_mean: list[float] = None,
    latents_std: list[float] = None,
) -> mx.array:
    """Decode latents to video frames using HuggingFace VAE weights directly.

    Args:
        latents: [B, T, H, W, z_dim] denoised latents (channels-last)
        vae_dir: path to vae/ directory with config.json and safetensors
        latents_mean: per-channel mean for denormalization
        latents_std: per-channel std for denormalization

    Returns:
        [B, T_out, H_out, W_out, 3] decoded video frames in [0, 1]
    """
    vae_path = Path(vae_dir)

    # Load config
    with open(vae_path / "config.json") as f:
        config = json.load(f)

    # Load weights
    raw_weights = mx.load(str(vae_path / "diffusion_pytorch_model.safetensors"))

    # Extract decoder weights and transpose Conv3D
    weights = {}
    for k, v in raw_weights.items():
        if not k.startswith("decoder."):
            continue
        # Strip "decoder." prefix
        name = k[len("decoder."):]

        # Transpose Conv3D weights: [O,I,D,H,W] -> [O,D,H,W,I]
        if "conv" in name and name.endswith(".weight") and v.ndim == 5:
            v = _transpose_conv3d_weight(v)

        weights[name] = v.astype(mx.bfloat16)

    # Denormalize latents
    if latents_mean is None:
        latents_mean = config.get("latents_mean", [0.0] * latents.shape[-1])
    if latents_std is None:
        latents_std = config.get("latents_std", [1.0] * latents.shape[-1])

    mean = mx.array(latents_mean, dtype=latents.dtype)
    std = mx.array(latents_std, dtype=latents.dtype)
    inv_std = 1.0 / std
    z = latents / inv_std + mean

    print(f"    Denormalized latent stats: mean={mx.mean(z).item():.3f}, std={mx.std(z).item():.3f}")

    # conv_in
    x = _conv3d_forward(z, weights["conv_in.weight"], weights.get("conv_in.bias"))
    mx.eval(x)
    print(f"    After conv_in: {x.shape}")

    # mid_block resnets
    for i in range(2):
        x = _resnet_block(x, weights, f"mid_block.resnets.{i}")
        mx.eval(x)
    print(f"    After mid_block: {x.shape}")

    # up_blocks
    temporal_upsample = list(reversed(config.get("temperal_downsample", [False, True, True])))

    # Count up_blocks from weights
    up_block_ids = set()
    for k in weights:
        if k.startswith("up_blocks."):
            idx = int(k.split(".")[1])
            up_block_ids.add(idx)
    num_up_blocks = len(up_block_ids)

    for block_idx in range(num_up_blocks):
        # Count resnets in this block
        resnet_ids = set()
        for k in weights:
            if k.startswith(f"up_blocks.{block_idx}.resnets."):
                ri = int(k.split(".")[3])
                resnet_ids.add(ri)
        num_resnets = len(resnet_ids)

        for ri in range(num_resnets):
            x = _resnet_block(x, weights, f"up_blocks.{block_idx}.resnets.{ri}")
            mx.eval(x)

        # Upsample
        t_up = temporal_upsample[block_idx] if block_idx < len(temporal_upsample) else False
        x = _dup_up_3d(x, temporal=t_up)
        mx.eval(x)
        print(f"    After up_block {block_idx}: {x.shape}")

    # norm_out + silu + conv_out
    x = _rms_norm(x, weights["norm_out.gamma"])
    x = nn.silu(x)
    x = _conv3d_forward(x, weights["conv_out.weight"], weights.get("conv_out.bias"))
    mx.eval(x)
    print(f"    After conv_out: {x.shape}")

    # Unpatchify if needed (patch_size=2)
    patch_size = config.get("patch_size", 2)
    if patch_size > 1:
        b, t, h, w, c_patch = x.shape
        c = c_patch // (patch_size * patch_size)
        x = x.reshape(b, t, h, w, c, patch_size, patch_size)
        x = mx.transpose(x, (0, 1, 2, 5, 3, 6, 4))  # [B, T, H, p, W, p, C]
        x = x.reshape(b, t, h * patch_size, w * patch_size, c)
        print(f"    After unpatchify: {x.shape}")

    # Clamp to [0, 1]
    x = (mx.clip(x, -1.0, 1.0) + 1.0) / 2.0

    return x
