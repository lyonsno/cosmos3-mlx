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
        logits, _ = small_model(input_ids)
        assert logits.shape == (batch, seq_len, 1000)

    def test_forward_no_nan(self, small_model):
        """Forward pass should not produce NaN values."""
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        logits, _ = small_model(input_ids)
        mx.eval(logits)
        assert not mx.any(mx.isnan(logits)).item(), "Model produced NaN logits"

    def test_causal_masking(self, small_model):
        """Adding tokens to the end should not change earlier logits."""
        short_ids = mx.array([[1, 2, 3, 4]])
        long_ids = mx.array([[1, 2, 3, 4, 5, 6]])

        logits_short, _ = small_model(short_ids)
        logits_long, _ = small_model(long_ids)
        mx.eval(logits_short, logits_long)

        # First 4 positions should be identical
        assert mx.allclose(logits_short, logits_long[:, :4], atol=1e-5).item(), \
            "Causal masking violated — future tokens affected past logits"

    def test_different_inputs_different_outputs(self, small_model):
        """Different input sequences should produce different logits."""
        ids_a = mx.array([[1, 2, 3, 4]])
        ids_b = mx.array([[5, 6, 7, 8]])
        logits_a, _ = small_model(ids_a)
        logits_b, _ = small_model(ids_b)
        mx.eval(logits_a, logits_b)
        assert not mx.allclose(logits_a, logits_b, atol=1e-6).item()

    def test_batch_independence(self, small_model):
        """Batched forward should give same results as individual forwards."""
        ids_a = mx.array([[1, 2, 3]])
        ids_b = mx.array([[4, 5, 6]])
        ids_batch = mx.array([[1, 2, 3], [4, 5, 6]])

        logits_a, _ = small_model(ids_a)
        logits_b, _ = small_model(ids_b)
        logits_batch, _ = small_model(ids_batch)
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


class TestDiffusionForward:
    """Test the diffusion generation forward pass."""

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
        model = Cosmos3Model(cfg)
        # Match patch_latent_dim = z_dim(8) * patch_size(2)^2 = 32
        model.proj_in = nn.Linear(32, 128, bias=True)
        model.proj_out = nn.Linear(128, 32, bias=True)
        return model

    def test_diffusion_forward_shape(self, small_model):
        """Velocity prediction should match input patch shape."""
        text_ids = mx.array([[1, 2, 3, 4, 5]])
        gen_tokens = mx.random.normal((1, 4, 32))  # 4 patches
        t = mx.array([0.5])
        velocity, _ = small_model.diffusion_forward(
            text_ids, gen_tokens, t, grid_t=1, grid_h=2, grid_w=2
        )
        mx.eval(velocity)
        assert velocity.shape == (1, 4, 32)

    def test_diffusion_forward_no_nan(self, small_model):
        """Diffusion forward should not produce NaN."""
        text_ids = mx.array([[1, 2, 3]])
        gen_tokens = mx.random.normal((1, 8, 32))  # 2x2x2 grid
        t = mx.array([0.5])
        velocity, _ = small_model.diffusion_forward(
            text_ids, gen_tokens, t, grid_t=2, grid_h=2, grid_w=2
        )
        mx.eval(velocity)
        assert not mx.any(mx.isnan(velocity)).item()

    def test_spatial_position_ids_affect_output(self, small_model):
        """Different grid shapes with same num_patches should produce different outputs.

        This verifies that the spatial grid mRoPE positions are actually being used,
        not just flat sequential positions.
        """
        text_ids = mx.array([[1, 2, 3]])
        gen_tokens = mx.random.normal((1, 4, 32))
        t = mx.array([0.5])

        # Same 4 patches, but different spatial arrangements
        v_2x2, _ = small_model.diffusion_forward(
            text_ids, gen_tokens, t, grid_t=1, grid_h=2, grid_w=2
        )
        v_1x4, _ = small_model.diffusion_forward(
            text_ids, gen_tokens, t, grid_t=1, grid_h=1, grid_w=4
        )
        v_4x1, _ = small_model.diffusion_forward(
            text_ids, gen_tokens, t, grid_t=1, grid_h=4, grid_w=1
        )
        mx.eval(v_2x2, v_1x4, v_4x1)

        # All should be different because spatial positions differ
        assert not mx.allclose(v_2x2, v_1x4, atol=1e-4).item(), \
            "2x2 and 1x4 grids produced identical output — spatial positions not working"
        assert not mx.allclose(v_2x2, v_4x1, atol=1e-4).item(), \
            "2x2 and 4x1 grids produced identical output — spatial positions not working"

    def test_different_timesteps_different_velocity(self, small_model):
        """Different timesteps should produce different velocities."""
        text_ids = mx.array([[1, 2, 3]])
        gen_tokens = mx.random.normal((1, 4, 32))

        # Use scheduler-scale timesteps (sigma * 1000)
        v_early, _ = small_model.diffusion_forward(
            text_ids, gen_tokens, mx.array([900.0]), grid_t=1, grid_h=2, grid_w=2
        )
        v_late, _ = small_model.diffusion_forward(
            text_ids, gen_tokens, mx.array([100.0]), grid_t=1, grid_h=2, grid_w=2
        )
        mx.eval(v_early, v_late)
        assert not mx.allclose(v_early, v_late, atol=1e-4).item()
