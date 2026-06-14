"""Tests for config loading and weight mapping."""

import json
import tempfile
from pathlib import Path

import mlx.core as mx
import pytest

from cosmos3_mlx.load import (
    load_transformer_config,
    load_vision_config,
    _sanitize_vision_weights,
)
from cosmos3_mlx.model import Cosmos3Config
from cosmos3_mlx.vision import VisionConfig


class TestLoadTransformerConfig:
    """Test loading transformer config from JSON."""

    def test_loads_from_json(self, tmp_path):
        """Should correctly parse the HuggingFace config format."""
        config = {
            "hidden_size": 4096,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 12288,
            "vocab_size": 151936,
            "rms_norm_eps": 1e-6,
            "rope_theta": 5000000,
            "max_position_embeddings": 262144,
            "rope_scaling": {
                "mrope_interleaved": True,
                "mrope_section": [24, 20, 20],
                "rope_type": "default",
            },
        }
        transformer_dir = tmp_path / "transformer"
        transformer_dir.mkdir()
        with open(transformer_dir / "config.json", "w") as f:
            json.dump(config, f)

        cfg = load_transformer_config(tmp_path)
        assert isinstance(cfg, Cosmos3Config)
        assert cfg.hidden_size == 4096
        assert cfg.num_hidden_layers == 36
        assert cfg.mrope_section == [24, 20, 20]
        assert cfg.rope_theta == 5000000


class TestLoadVisionConfig:
    """Test loading vision encoder config from JSON."""

    def test_loads_from_json(self, tmp_path):
        """Should correctly parse the HuggingFace vision config format."""
        config = {
            "depth": 27,
            "hidden_size": 1152,
            "num_heads": 16,
            "intermediate_size": 4304,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "in_channels": 3,
            "out_hidden_size": 4096,
            "spatial_merge_size": 2,
        }
        vision_dir = tmp_path / "vision_encoder"
        vision_dir.mkdir()
        with open(vision_dir / "config.json", "w") as f:
            json.dump(config, f)

        cfg = load_vision_config(tmp_path)
        assert isinstance(cfg, VisionConfig)
        assert cfg.depth == 27
        assert cfg.hidden_size == 1152
        assert cfg.out_hidden_size == 4096


class TestSanitizeVisionWeights:
    """Test vision weight name sanitization."""

    def test_mlp_renaming(self):
        """Should rename linear_fc1/fc2 to fc1/fc2."""
        weights = {
            "blocks.0.mlp.linear_fc1.weight": mx.ones((128, 64)),
            "blocks.0.mlp.linear_fc2.weight": mx.ones((64, 128)),
            "blocks.0.norm1.weight": mx.ones((64,)),
        }
        sanitized = _sanitize_vision_weights(weights)
        assert "blocks.0.mlp.fc1.weight" in sanitized
        assert "blocks.0.mlp.fc2.weight" in sanitized
        assert "blocks.0.norm1.weight" in sanitized
        assert "blocks.0.mlp.linear_fc1.weight" not in sanitized

    def test_preserves_attention_weights(self):
        """Attention weights should pass through unchanged."""
        weights = {
            "blocks.0.attn.qkv.weight": mx.ones((192, 64)),
            "blocks.0.attn.proj.weight": mx.ones((64, 64)),
        }
        sanitized = _sanitize_vision_weights(weights)
        assert "blocks.0.attn.qkv.weight" in sanitized
        assert "blocks.0.attn.proj.weight" in sanitized


class TestImagePreprocessing:
    """Test image preprocessing for vision encoder."""

    def test_preprocess_creates_correct_shape(self, tmp_path):
        """Preprocessed image should have correct dimensions."""
        from PIL import Image
        from cosmos3_mlx.generate import preprocess_image

        # Create a test image
        img = Image.new("RGB", (256, 256), color=(128, 64, 32))
        img_path = tmp_path / "test.png"
        img.save(img_path)

        pixel_values, grid_thw = preprocess_image(
            str(img_path), patch_size=16, temporal_patch_size=2
        )
        mx.eval(pixel_values, grid_thw)

        # Should be [1, 3, 2, 256, 256]
        assert pixel_values.shape == (1, 3, 2, 256, 256)
        # Grid: [1, 3] with (1, 16, 16)
        assert grid_thw.shape == (1, 3)
        assert grid_thw[0, 0].item() == 1   # temporal patches
        assert grid_thw[0, 1].item() == 16  # height patches
        assert grid_thw[0, 2].item() == 16  # width patches

    def test_preprocess_normalizes(self, tmp_path):
        """Preprocessed pixels should be normalized (not in [0, 255])."""
        from PIL import Image
        from cosmos3_mlx.generate import preprocess_image

        img = Image.new("RGB", (64, 64), color=(255, 255, 255))
        img_path = tmp_path / "white.png"
        img.save(img_path)

        pixel_values, _ = preprocess_image(str(img_path), patch_size=16, temporal_patch_size=2)
        mx.eval(pixel_values)

        # White pixel (255) normalized with ImageNet stats should be around 2.6
        assert mx.max(pixel_values).item() < 10.0
        assert mx.max(pixel_values).item() > 1.0  # Definitely not in [0, 1]
