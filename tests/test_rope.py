"""Tests for 3D multi-dimensional rotary position embeddings."""

import math

import mlx.core as mx
import numpy as np
import pytest

from cosmos3_mlx.rope import Cosmos3RotaryEmbedding, apply_rotary_pos_emb


class TestRotaryEmbedding:
    """Test the 3D mRoPE implementation."""

    def test_output_shape(self):
        """Rotary embedding output shape matches input."""
        rope = Cosmos3RotaryEmbedding(
            head_dim=128,
            mrope_section=[24, 20, 20],
            rope_theta=5_000_000.0,
        )
        # position_ids: [3, batch, seq_len] — one per axis (temporal, height, width)
        position_ids = mx.zeros((3, 1, 16), dtype=mx.int32)
        cos, sin = rope(position_ids, seq_len=16)
        # cos/sin should be [1, seq_len, head_dim // 2]
        # (half_dim because rotate_half splits the head into two halves)
        assert cos.shape == (1, 16, 64), f"Expected (1, 16, 64), got {cos.shape}"
        assert sin.shape == (1, 16, 64), f"Expected (1, 16, 64), got {sin.shape}"

    def test_cos_sin_range(self):
        """cos and sin values should be in [-1, 1]."""
        rope = Cosmos3RotaryEmbedding(
            head_dim=128,
            mrope_section=[24, 20, 20],
            rope_theta=5_000_000.0,
        )
        position_ids = mx.broadcast_to(mx.arange(32)[None, None, :], (3, 1, 32)).astype(mx.int32)
        cos, sin = rope(position_ids, seq_len=32)
        mx.eval(cos, sin)
        assert mx.all(cos >= -1.0).item() and mx.all(cos <= 1.0).item()
        assert mx.all(sin >= -1.0).item() and mx.all(sin <= 1.0).item()

    def test_text_positions_share_axes(self):
        """For text tokens, all 3 axes should have the same position IDs.
        When all axes match, the result should be consistent."""
        rope = Cosmos3RotaryEmbedding(
            head_dim=128,
            mrope_section=[24, 20, 20],
            rope_theta=5_000_000.0,
        )
        seq_len = 8
        ids = mx.arange(seq_len)[None, :]  # [1, seq_len]
        # All 3 axes share same positions (text case)
        position_ids = mx.stack([ids, ids, ids])  # [3, 1, seq_len]
        cos, sin = rope(position_ids, seq_len=seq_len)
        mx.eval(cos, sin)
        # Should produce valid embeddings without NaN
        assert not mx.any(mx.isnan(cos)).item()
        assert not mx.any(mx.isnan(sin)).item()

    def test_different_axes_produce_different_embeddings(self):
        """Different position IDs across axes should produce different embeddings."""
        rope = Cosmos3RotaryEmbedding(
            head_dim=128,
            mrope_section=[24, 20, 20],
            rope_theta=5_000_000.0,
        )
        seq_len = 8
        # All same positions
        same_ids = mx.stack([
            mx.arange(seq_len)[None, :],
            mx.arange(seq_len)[None, :],
            mx.arange(seq_len)[None, :],
        ])
        cos_same, sin_same = rope(same_ids, seq_len=seq_len)

        # Different height/width positions (vision case)
        diff_ids = mx.stack([
            mx.arange(seq_len)[None, :],
            mx.zeros((1, seq_len), dtype=mx.int32),  # height = 0
            mx.arange(seq_len)[None, :] * 2,  # width varies
        ])
        cos_diff, sin_diff = rope(diff_ids, seq_len=seq_len)
        mx.eval(cos_same, cos_diff)

        # The embeddings should differ
        assert not mx.allclose(cos_same, cos_diff, atol=1e-6).item()


class TestApplyRotary:
    """Test applying rotary embeddings to query/key tensors."""

    def test_apply_preserves_shape(self):
        """apply_rotary_pos_emb should preserve tensor shapes."""
        batch, seq_len, num_heads, head_dim = 1, 16, 32, 128
        q = mx.random.normal((batch, seq_len, num_heads, head_dim))
        k = mx.random.normal((batch, seq_len, 8, head_dim))  # GQA: 8 KV heads

        rope = Cosmos3RotaryEmbedding(
            head_dim=128,
            mrope_section=[24, 20, 20],
            rope_theta=5_000_000.0,
        )
        position_ids = mx.stack([
            mx.arange(seq_len)[None, :],
            mx.arange(seq_len)[None, :],
            mx.arange(seq_len)[None, :],
        ])
        cos, sin = rope(position_ids, seq_len=seq_len)

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_apply_changes_values(self):
        """Rotary embedding should change values (not identity)."""
        batch, seq_len, num_heads, head_dim = 1, 8, 4, 128
        q = mx.ones((batch, seq_len, num_heads, head_dim))
        k = mx.ones((batch, seq_len, 2, head_dim))

        rope = Cosmos3RotaryEmbedding(
            head_dim=128,
            mrope_section=[24, 20, 20],
            rope_theta=5_000_000.0,
        )
        position_ids = mx.stack([
            mx.arange(seq_len)[None, :],
            mx.arange(seq_len)[None, :],
            mx.arange(seq_len)[None, :],
        ])
        cos, sin = rope(position_ids, seq_len=seq_len)

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        mx.eval(q, q_rot)
        # Position 0 should be identity (sin=0, cos=1) but others should differ
        assert not mx.allclose(q[:, 1:], q_rot[:, 1:], atol=1e-6).item()
