"""Tests for diffusion generation components (scheduler, timestep embedding)."""

import mlx.core as mx
import numpy as np
import pytest

from cosmos3_mlx.scheduler import UniPCScheduler
from cosmos3_mlx.timestep import TimestepEmbedding, apply_timestep_to_noisy_tokens


class TestUniPCScheduler:
    """Test the UniPC scheduler."""

    def test_set_timesteps(self):
        """Timesteps should be decreasing (high noise to clean)."""
        sched = UniPCScheduler()
        sched.set_timesteps(10)
        assert sched.timesteps.shape == (10,)
        mx.eval(sched.timesteps)
        # Should be decreasing
        for i in range(9):
            assert sched.timesteps[i].item() > sched.timesteps[i + 1].item()

    def test_step_moves_toward_clean(self):
        """Each step should move the sample closer to the prediction."""
        sched = UniPCScheduler()
        sample = mx.ones((1, 4, 64))  # noisy sample
        velocity = mx.full((1, 4, 64), -0.5)  # predicted velocity
        t_current = mx.array(0.8)
        t_next = mx.array(0.6)

        result = sched.step(velocity, t_current, sample, t_next)
        mx.eval(result)
        # dt = 0.6 - 0.8 = -0.2, result = sample + (-0.2) * (-0.5) = 1.0 + 0.1 = 1.1
        assert mx.allclose(result, mx.full((1, 4, 64), 1.1), atol=1e-5).item()

    def test_add_noise_at_t0_is_clean(self):
        """At t=0, noisy sample should equal original."""
        sched = UniPCScheduler()
        original = mx.random.normal((1, 4, 64))
        noise = mx.random.normal((1, 4, 64))
        result = sched.add_noise(original, noise, mx.array(0.0))
        mx.eval(result, original)
        assert mx.allclose(result, original, atol=1e-6).item()

    def test_add_noise_at_t1_is_pure_noise(self):
        """At t=1, noisy sample should equal pure noise."""
        sched = UniPCScheduler()
        original = mx.random.normal((1, 4, 64))
        noise = mx.random.normal((1, 4, 64))
        result = sched.add_noise(original, noise, mx.array(1.0))
        mx.eval(result, noise)
        assert mx.allclose(result, noise, atol=1e-6).item()


class TestTimestepEmbedding:
    """Test timestep embedding."""

    def test_output_shape(self):
        """Should produce [batch, hidden_size] embeddings."""
        emb = TimestepEmbedding(hidden_size=128, freq_dim=64)
        t = mx.array([0.5, 0.3])
        out = emb(t)
        assert out.shape == (2, 128)

    def test_different_timesteps_different_embeddings(self):
        """Different timesteps should produce different embeddings."""
        emb = TimestepEmbedding(hidden_size=128, freq_dim=64)
        out_a = emb(mx.array([0.1]))
        out_b = emb(mx.array([0.9]))
        mx.eval(out_a, out_b)
        assert not mx.allclose(out_a, out_b, atol=1e-6).item()

    def test_no_nan(self):
        """Should not produce NaN values."""
        emb = TimestepEmbedding(hidden_size=128, freq_dim=64)
        out = emb(mx.array([0.0, 0.5, 1.0]))
        mx.eval(out)
        assert not mx.any(mx.isnan(out)).item()


class TestApplyTimestep:
    """Test selective timestep application to noisy tokens."""

    def test_only_noisy_tokens_affected(self):
        """Clean tokens should be unchanged, noisy tokens get timestep emb."""
        hidden = mx.zeros((1, 4, 64))
        t_emb = mx.ones((1, 64))
        mask = mx.array([[False, True, True, False]])  # tokens 1,2 are noisy

        result = apply_timestep_to_noisy_tokens(hidden, t_emb, mask)
        mx.eval(result)

        # Clean tokens (0, 3) should still be zero
        assert mx.allclose(result[0, 0], mx.zeros(64), atol=1e-6).item()
        assert mx.allclose(result[0, 3], mx.zeros(64), atol=1e-6).item()
        # Noisy tokens (1, 2) should have timestep embedding
        assert mx.allclose(result[0, 1], mx.ones(64), atol=1e-6).item()
        assert mx.allclose(result[0, 2], mx.ones(64), atol=1e-6).item()
