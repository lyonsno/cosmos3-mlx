"""Tests for the Wan2.2 VAE decoder."""

import mlx.core as mx
import pytest

from cosmos3_mlx.vae import VAEConfig, WanDecoder, WanResidualBlock, WanRMSNorm, dup_up_3d


class TestVAEConfig:
    """Test VAE configuration."""

    def test_default_matches_cosmos3(self):
        """Default config should match Cosmos3-Nano VAE."""
        cfg = VAEConfig()
        assert cfg.z_dim == 48
        assert cfg.decoder_base_dim == 256
        assert cfg.dim_mult == [1, 2, 4, 4]
        assert cfg.num_res_blocks == 2
        assert cfg.patch_size == 2


class TestWanRMSNorm:
    """Test RMS normalization."""

    def test_output_shape(self):
        """RMSNorm should preserve shape."""
        norm = WanRMSNorm(64)
        x = mx.random.normal((1, 4, 8, 8, 64))
        out = norm(x)
        assert out.shape == x.shape

    def test_normalizes(self):
        """Output should have approximately unit RMS."""
        norm = WanRMSNorm(64)
        x = mx.random.normal((1, 4, 8, 8, 64)) * 10.0
        out = norm(x)
        mx.eval(out)
        rms = mx.sqrt(mx.mean(out * out, axis=-1))
        mx.eval(rms)
        # RMS should be approximately 1.0
        assert mx.all(rms < 2.0).item() and mx.all(rms > 0.5).item()


class TestDupUp3D:
    """Test duplicate-based upsampling."""

    def test_spatial_only(self):
        """Spatial upsample should double H and W."""
        x = mx.random.normal((1, 4, 8, 8, 32))
        out = dup_up_3d(x, temporal=False)
        assert out.shape == (1, 4, 16, 16, 32)

    def test_spatiotemporal(self):
        """With temporal=True, should double T, H, and W."""
        x = mx.random.normal((1, 4, 8, 8, 32))
        out = dup_up_3d(x, temporal=True)
        assert out.shape == (1, 8, 16, 16, 32)


class TestWanResidualBlock:
    """Test residual blocks."""

    def test_same_channels(self):
        """Same in/out channels: no skip conv needed."""
        block = WanResidualBlock(32, 32)
        x = mx.random.normal((1, 2, 4, 4, 32))
        out = block(x)
        mx.eval(out)
        assert out.shape == x.shape
        assert block.skip is None

    def test_different_channels(self):
        """Different in/out channels: skip conv used."""
        block = WanResidualBlock(32, 64)
        x = mx.random.normal((1, 2, 4, 4, 32))
        out = block(x)
        mx.eval(out)
        assert out.shape == (1, 2, 4, 4, 64)
        assert block.skip is not None

    def test_no_nan(self):
        """Should not produce NaN."""
        block = WanResidualBlock(16, 16)
        x = mx.random.normal((1, 2, 4, 4, 16))
        out = block(x)
        mx.eval(out)
        assert not mx.any(mx.isnan(out)).item()


class TestWanDecoder:
    """Test the full VAE decoder."""

    @pytest.fixture
    def small_decoder(self):
        """Create a small decoder for testing."""
        cfg = VAEConfig(
            z_dim=8,
            decoder_base_dim=16,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=1,
            temporal_upsample=[False, True, True],
            out_channels=3,
            patch_size=2,
        )
        return WanDecoder(cfg)

    def test_output_shape(self, small_decoder):
        """Decoder should produce spatially upsampled output."""
        # Latent: [1, 2, 4, 4, 8]
        z = mx.random.normal((1, 2, 4, 4, 8))
        out = small_decoder(z)
        mx.eval(out)

        # 3 upsample blocks each double spatial dims: 4 -> 8 -> 16 -> 32
        # Temporal: [False, True, True] reversed = [True, True, False]
        # So temporal: 2 -> 4 -> 8 -> 8
        # Patch unpatchify doubles spatial again: 32 -> 64
        # Output: [1, 8, 64, 64, 3]
        assert out.shape[0] == 1  # batch
        assert out.shape[-1] == 3  # RGB channels
        # Spatial should be upsampled
        assert out.shape[2] > 4 and out.shape[3] > 4

    def test_no_nan(self, small_decoder):
        """Decoder should not produce NaN."""
        z = mx.random.normal((1, 1, 2, 2, 8))
        out = small_decoder(z)
        mx.eval(out)
        assert not mx.any(mx.isnan(out)).item()
