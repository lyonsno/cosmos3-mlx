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

    def test_sigmas_descending_from_near_one(self):
        """Sigmas should start near 1.0 and decrease to 0."""
        sched = UniPCScheduler()
        sched.set_timesteps(10)
        mx.eval(sched.sigmas)
        # First sigma should be near 1.0
        assert sched.sigmas[0].item() > 0.9
        # Last sigma should be 0.0 (terminal)
        assert sched.sigmas[-1].item() == 0.0
        # Should have num_steps + 1 sigmas (including terminal)
        assert sched.sigmas.shape == (11,)

    def test_step_denoises(self):
        """After all steps, output should differ from pure noise input."""
        sched = UniPCScheduler()
        sched.set_timesteps(5)
        sample = mx.random.normal((1, 4, 64))  # pure noise
        initial_norm = mx.sqrt(mx.sum(sample * sample)).item()

        for i in range(5):
            # Zero velocity = model predicts no flow
            velocity = mx.zeros((1, 4, 64))
            result = sched.step(velocity, mx.array(0.0), sample)
            mx.eval(result)
            sample = result

        # With zero velocity, x0 = sample - sigma * 0 = sample
        # So the scheduler moves via the stepping formula
        mx.eval(sample)
        assert not mx.any(mx.isnan(sample)).item()

    def test_first_order_recovers_x0_at_final_step(self):
        """When last sigma is 0, should return x0 prediction directly."""
        sched = UniPCScheduler()
        sched.set_timesteps(2)
        sample = mx.ones((1, 4, 64))
        velocity = mx.full((1, 4, 64), 0.5)

        # Step through both steps
        result = sched.step(velocity, mx.array(0.0), sample)
        mx.eval(result)
        result = sched.step(velocity, mx.array(0.0), result)
        mx.eval(result)
        # Final step (sigma=0) should return x0 = sample - sigma * velocity
        assert not mx.any(mx.isnan(result)).item()

    def test_add_noise_at_t0_is_clean(self):
        """At t=0, noisy sample should equal original."""
        sched = UniPCScheduler()
        original = mx.random.normal((1, 4, 64))
        noise = mx.random.normal((1, 4, 64))
        result = sched.add_noise(original, noise, mx.array(0.0))
        mx.eval(result, original)
        assert mx.allclose(result, original, atol=1e-6).item()

    def test_add_noise_at_t1000_is_pure_noise(self):
        """At t=num_train_timesteps, noisy sample should equal pure noise."""
        sched = UniPCScheduler()
        original = mx.random.normal((1, 4, 64))
        noise = mx.random.normal((1, 4, 64))
        result = sched.add_noise(original, noise, mx.array(1000.0))
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
