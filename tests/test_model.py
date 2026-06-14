"""Tests for the full Cosmos 3 MoT transformer model."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from cosmos3_mlx.model import Cosmos3Config, Cosmos3Model


class TestCosmos3Config:
    """Test model configuration."""

    def test_default_nano_config(self):
        """Default config should match Cosmos3-Nano architecture."""
        cfg = Cosmos3Config()
        assert cfg.hidden_size == 4096
        assert cfg.num_hidden_layers == 36
        assert cfg.num_attention_heads == 32
        assert cfg.num_key_value_heads == 8
        assert cfg.head_dim == 128
        assert cfg.intermediate_size == 12288
        assert cfg.vocab_size == 151936
        assert cfg.rms_norm_eps == 1e-6
        assert cfg.rope_theta == 5_000_000.0
        assert cfg.mrope_section == [24, 20, 20]

    def test_small_config(self):
        """Small config for testing should be constructable."""
        cfg = Cosmos3Config(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            intermediate_size=256,
            vocab_size=1000,
            mrope_section=[6, 5, 5],
        )
        assert cfg.hidden_size == 128
        assert cfg.num_hidden_layers == 2


class TestCosmos3Model:
    """Test the full transformer model."""

    @pytest.fixture
    def small_model(self):
        """Create a small model for testing."""
        cfg = Cosmos3Config(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            intermediate_size=256,
            vocab_size=1000,
            mrope_section=[6, 5, 5],
        )
        return Cosmos3Model(cfg)

    def test_forward_shape(self, small_model):
        """Forward pass should produce logits with vocab_size last dim."""
        batch, seq_len = 1, 8
        input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        logits = small_model(input_ids)
        assert logits.shape == (batch, seq_len, 1000)

    def test_forward_no_nan(self, small_model):
        """Forward pass should not produce NaN values."""
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        logits = small_model(input_ids)
        mx.eval(logits)
        assert not mx.any(mx.isnan(logits)).item(), "Model produced NaN logits"

    def test_causal_masking(self, small_model):
        """Adding tokens to the end should not change earlier logits."""
        short_ids = mx.array([[1, 2, 3, 4]])
        long_ids = mx.array([[1, 2, 3, 4, 5, 6]])

        logits_short = small_model(short_ids)
        logits_long = small_model(long_ids)
        mx.eval(logits_short, logits_long)

        # First 4 positions should be identical
        assert mx.allclose(logits_short, logits_long[:, :4], atol=1e-5).item(), \
            "Causal masking violated — future tokens affected past logits"

    def test_different_inputs_different_outputs(self, small_model):
        """Different input sequences should produce different logits."""
        ids_a = mx.array([[1, 2, 3, 4]])
        ids_b = mx.array([[5, 6, 7, 8]])
        logits_a = small_model(ids_a)
        logits_b = small_model(ids_b)
        mx.eval(logits_a, logits_b)
        assert not mx.allclose(logits_a, logits_b, atol=1e-6).item()

    def test_batch_independence(self, small_model):
        """Batched forward should give same results as individual forwards."""
        ids_a = mx.array([[1, 2, 3]])
        ids_b = mx.array([[4, 5, 6]])
        ids_batch = mx.array([[1, 2, 3], [4, 5, 6]])

        logits_a = small_model(ids_a)
        logits_b = small_model(ids_b)
        logits_batch = small_model(ids_batch)
        mx.eval(logits_a, logits_b, logits_batch)

        assert mx.allclose(logits_a, logits_batch[:1], atol=1e-5).item()
        assert mx.allclose(logits_b, logits_batch[1:], atol=1e-5).item()


class TestCosmos3ModelGenerate:
    """Test autoregressive generation with KV cache."""

    @pytest.fixture
    def small_model(self):
        cfg = Cosmos3Config(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            intermediate_size=256,
            vocab_size=1000,
            mrope_section=[6, 5, 5],
        )
        return Cosmos3Model(cfg)

    def test_generate_produces_tokens(self, small_model):
        """Generate should produce the requested number of tokens."""
        prompt = mx.array([[1, 2, 3]])
        tokens = small_model.generate(prompt, max_tokens=5)
        mx.eval(tokens)
        # Should return prompt + 5 generated tokens
        assert tokens.shape[1] == 8  # 3 prompt + 5 generated

    def test_generate_deterministic_with_temp_zero(self, small_model):
        """Generation with temperature=0 should be deterministic."""
        prompt = mx.array([[1, 2, 3]])
        tokens_a = small_model.generate(prompt, max_tokens=4, temperature=0.0)
        tokens_b = small_model.generate(prompt, max_tokens=4, temperature=0.0)
        mx.eval(tokens_a, tokens_b)
        assert mx.array_equal(tokens_a, tokens_b).item()
