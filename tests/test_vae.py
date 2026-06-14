"""Tests for the Wan2.2 VAE decoder."""

import mlx.core as mx
import pytest

from cosmos3_mlx.vae import VAEConfig, WanDecoder, WanResidualBlock, WanRMSNorm, dup_up_3d
from cosmos3_mlx.decode_vae import (
    _conv3d_forward, _transpose_conv3d_weight, decode_latents,
)


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


class TestPostQuantConv:
    """Test post_quant_conv in the functional decoder."""

    def test_post_quant_conv_transforms_latents(self):
        """post_quant_conv should be a non-trivial 1x1x1 channel transform."""
        # Simulate a post_quant_conv weight: [O, I, 1, 1, 1] -> transposed [O, 1, 1, 1, I]
        z_dim = 8
        pqc_w_pt = mx.random.normal((z_dim, z_dim, 1, 1, 1)) * 0.1
        pqc_w = _transpose_conv3d_weight(pqc_w_pt)
        pqc_b = mx.zeros((z_dim,))

        z = mx.random.normal((1, 1, 4, 4, z_dim))
        mx.eval(z)

        z_transformed = _conv3d_forward(z, pqc_w, pqc_b,
                                        stride=(1, 1, 1), padding=(0, 0, 0), causal=False)
        mx.eval(z_transformed)

        # Shape preserved
        assert z_transformed.shape == z.shape
        # Not identity (random weights should change the values)
        diff = mx.mean(mx.abs(z - z_transformed)).item()
        assert diff > 0.01, f"post_quant_conv had no effect: diff={diff}"

    def test_post_quant_conv_is_linear_channel_mix(self):
        """1x1x1 conv should be equivalent to a linear transform on channels."""
        z_dim = 4
        # Create a known weight matrix
        W = mx.array([[1, 0, 0, 0],
                       [0, 0, 1, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1]], dtype=mx.float32)
        # Shape as Conv3D weight: [O, I, 1, 1, 1]
        w_pt = W.reshape(z_dim, z_dim, 1, 1, 1)
        w_mlx = _transpose_conv3d_weight(w_pt)

        z = mx.array([[[[[1.0, 2.0, 3.0, 4.0]]]]]) # [1, 1, 1, 1, 4]
        out = _conv3d_forward(z, w_mlx, None,
                              stride=(1,1,1), padding=(0,0,0), causal=False)
        mx.eval(out)
        # W swaps channels 1 and 2
        expected = mx.array([[[[[1.0, 3.0, 2.0, 4.0]]]]])
        assert mx.allclose(out, expected, atol=1e-5).item()
