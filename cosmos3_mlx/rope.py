"""3D Multi-dimensional Rotary Position Embeddings for Cosmos 3.

Cosmos 3 uses interleaved mRoPE with three axes (temporal, height, width)
following the Qwen3-VL design. Each axis gets a section of the head dimensions:
  - temporal: mrope_section[0] dims (default 24)
  - height:   mrope_section[1] dims (default 20)
  - width:    mrope_section[2] dims (default 20)
Total = 64 half-dims = 128 head_dim / 2.

The interleaved layout mixes [T, H, W] frequencies so that adjacent
dimensions alternate axes, preserving frequency continuity.
"""

import math
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn


class Cosmos3RotaryEmbedding(nn.Module):
    """Compute 3D interleaved mRoPE cos/sin embeddings."""

    def __init__(
        self,
        head_dim: int = 128,
        mrope_section: list[int] | None = None,
        rope_theta: float = 5_000_000.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.mrope_section = mrope_section or [24, 20, 20]
        self.rope_theta = rope_theta

        # Validate: sections must sum to head_dim // 2
        half_dim = head_dim // 2
        assert sum(self.mrope_section) == half_dim, (
            f"mrope_section {self.mrope_section} sums to {sum(self.mrope_section)}, "
            f"expected {half_dim}"
        )

    def _compute_inv_freq(self, section_dim: int) -> mx.array:
        """Compute inverse frequency bands for a section.

        section_dim is already in half-dim space (e.g. 24 out of 64 total).
        We produce section_dim frequencies.
        """
        freqs = mx.arange(0, section_dim, dtype=mx.float32) / section_dim
        inv_freq = 1.0 / (self.rope_theta ** freqs)
        return inv_freq

    def __call__(
        self,
        position_ids: mx.array,
        seq_len: int,
    ) -> Tuple[mx.array, mx.array]:
        """Compute cos/sin embeddings from 3-axis position IDs.

        Args:
            position_ids: [3, batch, seq_len] — position IDs per axis
            seq_len: sequence length

        Returns:
            cos: [batch, seq_len, head_dim]
            sin: [batch, seq_len, head_dim]
        """
        # Compute per-axis frequencies and combine
        cos_parts = []
        sin_parts = []

        for axis_idx, section_dim in enumerate(self.mrope_section):
            inv_freq = self._compute_inv_freq(section_dim)  # [section_dim]

            # position_ids for this axis: [batch, seq_len]
            pos = position_ids[axis_idx].astype(mx.float32)  # [batch, seq_len]

            # Outer product: [batch, seq_len, section_dim]
            freqs = mx.expand_dims(pos, -1) * mx.expand_dims(
                mx.expand_dims(inv_freq, 0), 0
            )

            cos_parts.append(mx.cos(freqs))
            sin_parts.append(mx.sin(freqs))

        # Concatenate sections: [batch, seq_len, half_dim]
        cos_cat = mx.concatenate(cos_parts, axis=-1)
        sin_cat = mx.concatenate(sin_parts, axis=-1)

        return cos_cat, sin_cat


def rotate_half(x: mx.array) -> mx.array:
    """Rotate halves: [x1, x2] -> [-x2, x1] where x1, x2 are each half_dim."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> Tuple[mx.array, mx.array]:
    """Apply rotary position embeddings to query and key tensors.

    Args:
        q: [batch, seq_len, num_heads, head_dim]
        k: [batch, seq_len, num_kv_heads, head_dim]
        cos: [batch, seq_len, head_dim]
        sin: [batch, seq_len, head_dim]

    Returns:
        q_rotated, k_rotated with same shapes as inputs
    """
    # cos/sin are [batch, seq_len, half_dim]
    # Duplicate to match full head_dim: [cos, cos] for the two halves
    cos_full = mx.concatenate([cos, cos], axis=-1)  # [batch, seq_len, head_dim]
    sin_full = mx.concatenate([sin, sin], axis=-1)  # [batch, seq_len, head_dim]

    # Expand for broadcasting over heads: [batch, seq_len, 1, head_dim]
    cos_full = mx.expand_dims(cos_full, 2)
    sin_full = mx.expand_dims(sin_full, 2)

    q_rot = q * cos_full + rotate_half(q) * sin_full
    k_rot = k * cos_full + rotate_half(k) * sin_full

    return q_rot, k_rot
