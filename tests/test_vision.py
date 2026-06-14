"""Tests for the Qwen3-VL vision encoder."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from cosmos3_mlx.vision import VisionConfig, VisionModel


class TestVisionConfig:
    """Test vision encoder configuration."""

    def test_default_config_matches_cosmos3(self):
        """Default config should match Cosmos3-Nano vision encoder."""
        cfg = VisionConfig()
        assert cfg.depth == 27
        assert cfg.hidden_size == 1152
        assert cfg.num_heads == 16
        assert cfg.intermediate_size == 4304
        assert cfg.patch_size == 16
        assert cfg.temporal_patch_size == 2
        assert cfg.out_hidden_size == 4096  # matches transformer hidden_size
        assert cfg.spatial_merge_size == 2


class TestVisionModel:
    """Test the vision encoder."""

    @pytest.fixture
    def small_model(self):
        """Create a small vision model for testing."""
        cfg = VisionConfig(
            depth=2,
            hidden_size=64,
            num_heads=4,
            intermediate_size=128,
            patch_size=8,
            temporal_patch_size=1,
            in_channels=3,
            out_hidden_size=128,
            spatial_merge_size=2,
        )
        return VisionModel(cfg)

    def test_output_shape(self, small_model):
        """Vision model should produce embeddings matching out_hidden_size."""
        # Simulate a single 64x64 image: [batch, channels, temporal, height, width]
        # After patch embedding with patch_size=8: 8x8=64 patches
        # After PatchMerger with spatial_merge_size=2: 64/4=16 merged patches
        pixel_values = mx.random.normal((1, 3, 1, 64, 64))
        grid_thw = mx.array([[1, 8, 8]])  # temporal=1, h_patches=8, w_patches=8

        out = small_model(pixel_values, grid_thw)
        mx.eval(out)

        # After PatchMerger: 64 patches / (2*2) = 16 merged patches
        assert out.shape[-1] == 128  # out_hidden_size
        assert out.shape[0] == 16   # merged patches

    def test_no_nan(self, small_model):
        """Vision model should not produce NaN values."""
        pixel_values = mx.random.normal((1, 3, 1, 64, 64))
        grid_thw = mx.array([[1, 8, 8]])

        out = small_model(pixel_values, grid_thw)
        mx.eval(out)
        assert not mx.any(mx.isnan(out)).item(), "Vision model produced NaN"

    def test_different_images_different_embeddings(self, small_model):
        """Different images should produce different embeddings."""
        img_a = mx.random.normal((1, 3, 1, 64, 64))
        img_b = mx.random.normal((1, 3, 1, 64, 64))
        grid_thw = mx.array([[1, 8, 8]])

        out_a = small_model(img_a, grid_thw)
        out_b = small_model(img_b, grid_thw)
        mx.eval(out_a, out_b)

        assert not mx.allclose(out_a, out_b, atol=1e-6).item()
