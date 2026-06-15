"""Tests for Cosmos 3 dual-pathway MoT attention."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from cosmos3_mlx.attention import Cosmos3Attention


class TestCosmos3Attention:
    """Test the dual-pathway packed attention mechanism."""

    @pytest.fixture
    def attn(self):
        """Create a small attention module for testing."""
        return Cosmos3Attention(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=32,
            mrope_section=[6, 5, 5],  # sums to 16 = head_dim/2
        )

    def test_understanding_only_output_shape(self, attn):
        """Understanding-only forward should return correct shape."""
        batch, seq_len = 1, 16
        x = mx.random.normal((batch, seq_len, 256))
        # position_ids for text: all axes same
        position_ids = mx.stack([
            mx.arange(seq_len)[None, :],
            mx.arange(seq_len)[None, :],
            mx.arange(seq_len)[None, :],
        ])

        out_und, out_gen, _, _ = attn(
            hidden_states=x,
            position_ids=position_ids,
            understanding_mask=None,
            generation_tokens=None,
        )
        assert out_und.shape == (batch, seq_len, 256)
        assert out_gen is None  # No generation tokens provided

    def test_dual_pathway_output_shapes(self, attn):
        """Both pathways should produce correct output shapes."""
        batch = 1
        und_len, gen_len = 16, 8
        und_tokens = mx.random.normal((batch, und_len, 256))
        gen_tokens = mx.random.normal((batch, gen_len, 256))
        position_ids = mx.stack([
            mx.arange(und_len + gen_len)[None, :],
            mx.arange(und_len + gen_len)[None, :],
            mx.arange(und_len + gen_len)[None, :],
        ])

        out_und, out_gen, _, _ = attn(
            hidden_states=und_tokens,
            position_ids=position_ids,
            understanding_mask=None,
            generation_tokens=gen_tokens,
        )
        assert out_und.shape == (batch, und_len, 256)
        assert out_gen.shape == (batch, gen_len, 256)

    def test_understanding_is_causal(self, attn):
        """Understanding pathway should use causal attention.
        Future tokens should not influence past tokens."""
        batch, seq_len = 1, 8
        x = mx.random.normal((batch, seq_len, 256))
        position_ids = mx.stack([
            mx.arange(seq_len)[None, :],
            mx.arange(seq_len)[None, :],
            mx.arange(seq_len)[None, :],
        ])

        # Full sequence
        out_full, _, __, ___ = attn(
            hidden_states=x,
            position_ids=position_ids,
            understanding_mask=None,
            generation_tokens=None,
        )

        # Truncated to first 4 tokens — should match if causal
        out_trunc, _, __, ___ = attn(
            hidden_states=x[:, :4],
            position_ids=position_ids[:, :, :4],
            understanding_mask=None,
            generation_tokens=None,
        )
        mx.eval(out_full, out_trunc)

        # First 4 tokens of full output should match truncated output
        assert mx.allclose(out_full[:, :4], out_trunc, atol=1e-5).item(), \
            "Understanding pathway is not causal — future tokens influenced past"

    def test_gqa_head_ratio(self, attn):
        """GQA should work with 4:1 head ratio (8 query, 2 kv)."""
        # This is implicitly tested by the forward pass working,
        # but let's verify the projection shapes
        assert attn.to_q.weight.shape == (256, 256)  # 8 heads * 32 dim
        assert attn.to_k.weight.shape == (64, 256)   # 2 kv heads * 32 dim
        assert attn.to_v.weight.shape == (64, 256)   # 2 kv heads * 32 dim


class TestCosmos3AttentionKVCache:
    """Test KV cache for autoregressive generation."""

    def test_kv_cache_consistent(self):
        """Generating token-by-token with KV cache should match full forward."""
        attn = Cosmos3Attention(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            mrope_section=[6, 5, 5],
        )
        seq_len = 6
        x = mx.random.normal((1, seq_len, 128))
        position_ids = mx.stack([
            mx.arange(seq_len)[None, :],
            mx.arange(seq_len)[None, :],
            mx.arange(seq_len)[None, :],
        ])

        # Full forward
        out_full, _, __, ___ = attn(
            hidden_states=x,
            position_ids=position_ids,
            understanding_mask=None,
            generation_tokens=None,
        )
        mx.eval(out_full)

        # Token-by-token with KV cache
        cache = None
        outputs = []
        for i in range(seq_len):
            pos_ids = position_ids[:, :, i : i + 1]
            out_i, _, cache, _ = attn(
                hidden_states=x[:, i : i + 1],
                position_ids=pos_ids,
                understanding_mask=None,
                generation_tokens=None,
                cache=cache,
            )
            outputs.append(out_i)
            mx.eval(out_i, *cache)

        out_cached = mx.concatenate(outputs, axis=1)
        mx.eval(out_cached)

        assert mx.allclose(out_full, out_cached, atol=1e-4).item(), \
            "KV cache output diverges from full forward pass"
