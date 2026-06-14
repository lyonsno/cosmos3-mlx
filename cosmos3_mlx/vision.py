"""Qwen3-VL Vision Encoder for Cosmos 3.

The Cosmos 3 vision encoder is a Qwen3VLVisionModel — a 27-layer ViT
with 3D patch embedding, rotary position embeddings, and PatchMerger.

Architecture adapted from the mlx-vlm Qwen3-VL implementation.
"""

import math
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass
class VisionConfig:
    """Configuration for the Qwen3-VL vision encoder."""

    depth: int = 27
    hidden_size: int = 1152
    num_heads: int = 16
    intermediate_size: int = 4304
    patch_size: int = 16
    temporal_patch_size: int = 2
    in_channels: int = 3
    out_hidden_size: int = 4096  # Must match transformer hidden_size
    spatial_merge_size: int = 2
    rope_theta: float = 10000.0


class VisionRotaryEmbedding(nn.Module):
    """Rotary position embeddings for vision patches."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        self._inv_freq = inv_freq

    def __call__(self, seq_len: int) -> mx.array:
        """Compute cos/sin embeddings for given sequence length."""
        t = mx.arange(seq_len, dtype=mx.float32)
        freqs = mx.outer(t, self._inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb), mx.sin(emb)


def rotate_half(x: mx.array) -> mx.array:
    """Rotate halves for rotary embeddings."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb_vision(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array):
    """Apply rotary embeddings to vision Q and K.

    Args:
        q, k: [seq_len, num_heads, head_dim]
        cos, sin: [seq_len, head_dim]
    """
    # Expand cos/sin for heads: [seq_len, 1, head_dim]
    cos = mx.expand_dims(cos, 1)
    sin = mx.expand_dims(sin, 1)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class PatchEmbed(nn.Module):
    """Convert image/video to patch embeddings.

    Uses a 3D convolution to handle both spatial and temporal dimensions.
    For MLX (which doesn't have Conv3d), we decompose into temporal and spatial convolutions.
    """

    def __init__(
        self,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size

        # Flatten 3D conv into a linear projection over the patch volume
        patch_volume = temporal_patch_size * patch_size * patch_size * in_channels
        self.proj = nn.Linear(patch_volume, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Convert pixel values to patch embeddings.

        Args:
            x: [batch, channels, temporal, height, width]

        Returns:
            [num_patches, hidden_size]
        """
        batch, c, t, h, w = x.shape
        tp = self.temporal_patch_size
        sp = self.patch_size

        # Reshape into patches: [batch, t//tp, h//sp, w//sp, tp*sp*sp*c]
        nt = t // tp
        nh = h // sp
        nw = w // sp

        # Rearrange: (b, c, t, h, w) -> (b, nt, tp, nh, sp, nw, sp, c) -> (b*nt*nh*nw, tp*sp*sp*c)
        x = x.reshape(batch, c, nt, tp, nh, sp, nw, sp)
        x = mx.transpose(x, (0, 2, 4, 6, 3, 5, 7, 1))  # (b, nt, nh, nw, tp, sp, sp, c)
        x = x.reshape(-1, tp * sp * sp * c)

        return self.proj(x)


class PatchMerger(nn.Module):
    """Merge spatial patches to reduce sequence length.

    Merges spatial_merge_size x spatial_merge_size patches into one.
    """

    def __init__(
        self,
        hidden_size: int,
        out_hidden_size: int,
        spatial_merge_size: int = 2,
    ):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        merge_dim = hidden_size * spatial_merge_size * spatial_merge_size

        self.ln_q = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = [
            nn.Linear(merge_dim, merge_dim, bias=True),
            nn.GELU(),
            nn.Linear(merge_dim, out_hidden_size, bias=True),
        ]

    def __call__(self, x: mx.array, grid_thw: mx.array) -> mx.array:
        """Merge patches.

        Args:
            x: [total_patches, hidden_size]
            grid_thw: [batch, 3] with (temporal, height, width) grid dims

        Returns:
            [total_merged_patches, out_hidden_size]
        """
        x = self.ln_q(x)

        merge = self.spatial_merge_size
        merged_patches = []

        offset = 0
        for i in range(grid_thw.shape[0]):
            t, h, w = grid_thw[i].tolist()
            t, h, w = int(t), int(h), int(w)
            num_patches = t * h * w
            seq = x[offset : offset + num_patches]
            offset += num_patches

            # Reshape to spatial grid and merge
            seq = seq.reshape(t, h, w, -1)

            # Merge spatial patches
            h_merged = h // merge
            w_merged = w // merge
            seq = seq.reshape(t, h_merged, merge, w_merged, merge, -1)
            seq = mx.transpose(seq, (0, 1, 3, 2, 4, 5))  # (t, h_m, w_m, merge, merge, hidden)
            seq = seq.reshape(t * h_merged * w_merged, -1)

            merged_patches.append(seq)

        x = mx.concatenate(merged_patches, axis=0)

        # MLP
        for layer in self.mlp:
            x = layer(x)

        return x


class VisionAttention(nn.Module):
    """Multi-head attention for vision transformer blocks."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def __call__(
        self,
        x: mx.array,
        rotary_pos_emb: Optional[tuple] = None,
    ) -> mx.array:
        seq_len, hidden = x.shape

        # QKV projection
        qkv = self.qkv(x)
        qkv = qkv.reshape(seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Apply rotary embeddings
        if rotary_pos_emb is not None:
            cos, sin = rotary_pos_emb
            q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # Reshape for attention: [1, num_heads, seq_len, head_dim]
        q = mx.transpose(q[None, :, :, :], (0, 2, 1, 3))
        k = mx.transpose(k[None, :, :, :], (0, 2, 1, 3))
        v = mx.transpose(v[None, :, :, :], (0, 2, 1, 3))

        scale = self.head_dim ** -0.5
        attn_out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)

        # Reshape back: [seq_len, hidden]
        attn_out = mx.transpose(attn_out, (0, 2, 1, 3))
        attn_out = attn_out.reshape(seq_len, hidden)

        return self.proj(attn_out)


class VisionMLP(nn.Module):
    """Feed-forward network for vision transformer."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.act(self.fc1(x)))


class VisionBlock(nn.Module):
    """Single vision transformer block."""

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = VisionAttention(hidden_size, num_heads)
        self.mlp = VisionMLP(hidden_size, intermediate_size)

    def __call__(self, x: mx.array, rotary_pos_emb: Optional[tuple] = None) -> mx.array:
        x = x + self.attn(self.norm1(x), rotary_pos_emb)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionModel(nn.Module):
    """Qwen3-VL Vision Encoder for Cosmos 3.

    27-layer ViT with 3D patch embedding, rotary position embeddings,
    and PatchMerger for spatial compression.
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            hidden_size=config.hidden_size,
        )

        self.rotary_pos_emb = VisionRotaryEmbedding(
            config.hidden_size // config.num_heads,
            theta=config.rope_theta,
        )

        self.blocks = [
            VisionBlock(config.hidden_size, config.num_heads, config.intermediate_size)
            for _ in range(config.depth)
        ]

        self.merger = PatchMerger(
            hidden_size=config.hidden_size,
            out_hidden_size=config.out_hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )

    def __call__(self, pixel_values: mx.array, grid_thw: mx.array) -> mx.array:
        """Encode images/video to vision embeddings.

        Args:
            pixel_values: [batch, channels, temporal, height, width]
            grid_thw: [batch, 3] grid dimensions after patching

        Returns:
            [total_merged_patches, out_hidden_size]
        """
        # Patch embedding
        x = self.patch_embed(pixel_values)

        # Compute rotary embeddings for the full sequence
        total_patches = x.shape[0]
        cos, sin = self.rotary_pos_emb(total_patches)

        # Expand cos/sin for heads: [seq_len, head_dim] -> [seq_len, num_heads, head_dim]
        # Actually they broadcast fine as [seq_len, head_dim]

        # Forward through transformer blocks
        for block in self.blocks:
            x = block(x, rotary_pos_emb=(cos, sin))

        # Merge patches
        x = self.merger(x, grid_thw)

        return x
