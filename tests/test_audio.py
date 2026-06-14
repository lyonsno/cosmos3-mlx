"""Tests for the Cosmos3 Audio VAE decoder."""

import mlx.core as mx
import pytest

from cosmos3_mlx.audio import (
    AudioDecoderConfig,
    AudioDecoder,
    AudioDecoderBlock,
    AudioResidualUnit,
    Snake1d,
)


class TestSnake1d:
    """Test Snake activation."""

    def test_output_shape(self):
        """Should preserve shape."""
        snake = Snake1d(32)
        x = mx.random.normal((1, 32, 100))
        out = snake(x)
        assert out.shape == x.shape

    def test_not_identity(self):
        """Should modify values (not identity)."""
        snake = Snake1d(16)
        x = mx.random.normal((1, 16, 50))
        out = snake(x)
        mx.eval(x, out)
        assert not mx.allclose(x, out, atol=1e-6).item()

    def test_no_nan(self):
        """Should not produce NaN."""
        snake = Snake1d(64)
        x = mx.random.normal((1, 64, 200))
        out = snake(x)
        mx.eval(out)
        assert not mx.any(mx.isnan(out)).item()


class TestAudioResidualUnit:
    """Test audio residual blocks."""

    def test_output_shape(self):
        """Should preserve shape (same in/out channels)."""
        unit = AudioResidualUnit(32, dilation=1)
        x = mx.random.normal((1, 32, 100))
        out = unit(x)
        mx.eval(out)
        assert out.shape == x.shape

    def test_dilation_3(self):
        """Should work with dilation=3."""
        unit = AudioResidualUnit(16, dilation=3)
        x = mx.random.normal((1, 16, 100))
        out = unit(x)
        mx.eval(out)
        assert out.shape == x.shape


class TestAudioDecoderBlock:
    """Test upsampling decoder blocks."""

    def test_upsample_by_stride(self):
        """Should upsample temporal dimension by stride factor."""
        block = AudioDecoderBlock(64, 32, stride=4)
        x = mx.random.normal((1, 64, 25))
        out = block(x)
        mx.eval(out)
        assert out.shape == (1, 32, 100)  # 25 * 4 = 100

    def test_stride_2(self):
        """Stride 2 upsample."""
        block = AudioDecoderBlock(32, 16, stride=2)
        x = mx.random.normal((1, 32, 50))
        out = block(x)
        mx.eval(out)
        assert out.shape == (1, 16, 100)  # 50 * 2 = 100


class TestAudioDecoder:
    """Test the full audio decoder."""

    @pytest.fixture
    def small_decoder(self):
        """Small decoder for testing."""
        cfg = AudioDecoderConfig(
            input_dim=8,
            dim=16,
            channel_mults=[1, 2, 4],
            strides=[2, 4, 5],
            out_channels=2,
        )
        return AudioDecoder(cfg)

    def test_output_shape(self, small_decoder):
        """Should produce stereo audio from latents with ~expected upsample."""
        # Input: [batch, latent_channels, latent_time]
        z = mx.random.normal((1, 8, 10))
        out = small_decoder(z)
        mx.eval(out)
        assert out.shape[0] == 1   # batch
        assert out.shape[1] == 2   # stereo
        # Temporal upsample ~10 * 2 * 4 * 5 = 400 (±small padding artifacts)
        assert 390 <= out.shape[2] <= 410

    def test_output_clamped(self, small_decoder):
        """Output should be in [-1, 1] range."""
        z = mx.random.normal((1, 8, 10)) * 5.0
        out = small_decoder(z)
        mx.eval(out)
        assert mx.all(out >= -1.0).item() and mx.all(out <= 1.0).item()

    def test_no_nan(self, small_decoder):
        """Should not produce NaN."""
        z = mx.random.normal((1, 8, 10))
        out = small_decoder(z)
        mx.eval(out)
        assert not mx.any(mx.isnan(out)).item()

    def test_default_config_matches_cosmos3(self):
        """Default config should match Cosmos3 audio tokenizer."""
        cfg = AudioDecoderConfig()
        assert cfg.input_dim == 64
        assert cfg.dim == 320
        assert cfg.channel_mults == [1, 2, 4, 8, 16]
        assert cfg.strides == [2, 4, 5, 6, 8]
        assert cfg.out_channels == 2
