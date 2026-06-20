"""Tests for the weight load audit guard in load_transformer.

These tests exercise the full load path — config parsing, safetensors
loading, weight conversion, and the missing/skipped parameter audit —
using a tiny synthetic model. This catches namespace bugs (like
mx.utils vs mlx.utils) that import-only or --help tests miss.
"""

import json
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from cosmos3_mlx.load import load_transformer
from cosmos3_mlx.model import Cosmos3Config, Cosmos3Model


# Tiny config that creates a valid but small model
# head_dim=16 → half_dim=8, mrope_section must sum to 8
TINY_CONFIG = {
    "hidden_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "intermediate_size": 64,
    "vocab_size": 128,
    "rms_norm_eps": 1e-6,
    "rope_theta": 5000000,
    "max_position_embeddings": 256,
    "rope_scaling": {
        "mrope_interleaved": True,
        "mrope_section": [4, 2, 2],
        "rope_type": "default",
    },
}


def _create_fake_model_dir(tmp_path, config=None, drop_keys=None, extra_keys=None):
    """Create a fake model directory with config and matching weights."""
    cfg = config or TINY_CONFIG
    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir(parents=True)

    with open(transformer_dir / "config.json", "w") as f:
        json.dump(cfg, f)

    # Build a real model to get the exact parameter names
    model_config = Cosmos3Config(
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        intermediate_size=cfg["intermediate_size"],
        vocab_size=cfg["vocab_size"],
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=cfg["rope_theta"],
        mrope_section=cfg["rope_scaling"]["mrope_section"],
        max_position_embeddings=cfg["max_position_embeddings"],
    )
    model = Cosmos3Model(model_config)

    # Extract all parameter names and create matching fake weights
    weights = {}
    for name, param in tree_flatten(model.parameters()):
        weights[name] = mx.zeros(param.shape, dtype=mx.bfloat16)

    # Add expected extra keys (action modality projections)
    weights["action_proj_in.fc.weight"] = mx.zeros((cfg["hidden_size"],))
    weights["action_proj_out.weight"] = mx.zeros((cfg["hidden_size"],))

    if drop_keys:
        for k in drop_keys:
            weights.pop(k, None)

    if extra_keys:
        for k, shape in extra_keys.items():
            weights[k] = mx.zeros(shape)

    mx.save_safetensors(str(transformer_dir / "weights.safetensors"), weights)
    return tmp_path


class TestLoadTransformerAudit:
    """Test the full load_transformer path including weight audit."""

    def test_successful_load(self, tmp_path):
        """Full load should succeed with complete matching weights."""
        model_dir = _create_fake_model_dir(tmp_path)
        model = load_transformer(model_dir, reasoner_only=False, dtype=mx.float32)
        assert isinstance(model, Cosmos3Model)

    def test_missing_params_raises(self, tmp_path):
        """Should raise RuntimeError when required model params are missing."""
        # Drop a weight that the model needs
        model_dir = _create_fake_model_dir(
            tmp_path, drop_keys=["layers.0.self_attn.to_q.weight"]
        )
        with pytest.raises(RuntimeError, match="Missing.*required model parameters"):
            load_transformer(model_dir, reasoner_only=False, dtype=mx.float32)

    def test_expected_extra_keys_tolerated(self, tmp_path):
        """action_proj_in/out are expected extras and should not cause errors."""
        model_dir = _create_fake_model_dir(tmp_path)
        # Should not raise — action_proj keys are in the allowlist
        model = load_transformer(model_dir, reasoner_only=False, dtype=mx.float32)
        assert isinstance(model, Cosmos3Model)

    def test_missing_config_raises(self, tmp_path):
        """Should raise FileNotFoundError with helpful message when config missing."""
        with pytest.raises(FileNotFoundError, match="Download the model first"):
            load_transformer(tmp_path, reasoner_only=False)

    def test_empty_weights_raises(self, tmp_path):
        """Should raise RuntimeError when no weights match model params."""
        transformer_dir = tmp_path / "transformer"
        transformer_dir.mkdir(parents=True)
        with open(transformer_dir / "config.json", "w") as f:
            json.dump(TINY_CONFIG, f)
        # Empty safetensors → all params missing → RuntimeError
        mx.save_safetensors(str(transformer_dir / "empty.safetensors"), {})
        with pytest.raises(RuntimeError, match="Missing.*required model parameters"):
            load_transformer(tmp_path, reasoner_only=False)
