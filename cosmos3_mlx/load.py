"""Load Cosmos 3 Nano weights from HuggingFace format into MLX models.

Handles:
- Loading config.json to build model configs
- Loading sharded safetensors into MLX arrays
- Filtering for reasoner-only weights
- Mapping weights into Cosmos3Model and VisionModel
"""

import json
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .convert import convert_weights
from .model import Cosmos3Config, Cosmos3Model
from .vision import VisionConfig, VisionModel


def load_transformer_config(model_dir: str | Path) -> Cosmos3Config:
    """Load transformer config from HuggingFace model directory."""
    config_path = Path(model_dir) / "transformer" / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Model config not found at {config_path}. "
            f"Download the model first: hf download nvidia/Cosmos3-Nano --local-dir {model_dir}"
        )
    with open(config_path) as f:
        cfg = json.load(f)

    rope_scaling = cfg.get("rope_scaling", {})
    mrope_section = rope_scaling.get("mrope_section", [24, 20, 20])

    return Cosmos3Config(
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        intermediate_size=cfg["intermediate_size"],
        vocab_size=cfg["vocab_size"],
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=cfg["rope_theta"],
        mrope_section=mrope_section,
        max_position_embeddings=cfg["max_position_embeddings"],
    )


def load_vision_config(model_dir: str | Path) -> VisionConfig:
    """Load vision encoder config from HuggingFace model directory."""
    config_path = Path(model_dir) / "vision_encoder" / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)

    return VisionConfig(
        depth=cfg["depth"],
        hidden_size=cfg["hidden_size"],
        num_heads=cfg["num_heads"],
        intermediate_size=cfg["intermediate_size"],
        patch_size=cfg["patch_size"],
        temporal_patch_size=cfg["temporal_patch_size"],
        in_channels=cfg["in_channels"],
        out_hidden_size=cfg["out_hidden_size"],
        spatial_merge_size=cfg["spatial_merge_size"],
    )


def _load_safetensors_shards(directory: Path) -> dict[str, mx.array]:
    """Load all safetensors shards from a directory."""
    weights = {}
    for shard in sorted(directory.glob("*.safetensors")):
        shard_weights = mx.load(str(shard))
        weights.update(shard_weights)
    return weights


def load_transformer(
    model_dir: str | Path,
    reasoner_only: bool = False,
    dtype: mx.Dtype = mx.bfloat16,
) -> Cosmos3Model:
    """Load Cosmos 3 transformer with weights.

    Args:
        model_dir: path to HuggingFace model directory
        reasoner_only: strip generation weights for smaller memory
        dtype: target dtype (default bfloat16)

    Returns:
        Cosmos3Model with loaded weights
    """
    model_dir = Path(model_dir)

    # Load config and create model
    config = load_transformer_config(model_dir)
    model = Cosmos3Model(config)

    # Load and convert weights
    raw_weights = _load_safetensors_shards(model_dir / "transformer")
    weights = convert_weights(raw_weights, reasoner_only=reasoner_only)

    # Cast to target dtype
    weights = {k: v.astype(dtype) for k, v in weights.items()}

    # Weight keys that are in the checkpoint but have no matching nn.Module
    # parameter (action modality projections use a non-standard structure)
    EXPECTED_EXTRA_KEYS = {"action_proj_in.fc.weight", "action_proj_out.weight"}

    model_params = set(k for k, _ in mx.utils.tree_flatten(model.parameters()))
    weight_keys = set(weights.keys())
    skipped = weight_keys - model_params
    missing = model_params - weight_keys
    unexpected_extra = skipped - EXPECTED_EXTRA_KEYS

    model.load_weights(list(weights.items()), strict=False)

    if unexpected_extra:
        print(f"  Note: {len(unexpected_extra)} unexpected extra weight keys")
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} required model parameters in weights. "
            f"First 5: {sorted(missing)[:5]}. "
            f"The checkpoint may be incomplete or from an incompatible model version."
        )

    return model


def load_vision_encoder(
    model_dir: str | Path,
    dtype: mx.Dtype = mx.bfloat16,
) -> VisionModel:
    """Load Qwen3-VL vision encoder with weights.

    Args:
        model_dir: path to HuggingFace model directory
        dtype: target dtype (default bfloat16)

    Returns:
        VisionModel with loaded weights
    """
    model_dir = Path(model_dir)

    # Load config and create model
    config = load_vision_config(model_dir)
    model = VisionModel(config)

    # Load weights
    raw_weights = _load_safetensors_shards(model_dir / "vision_encoder")

    # Vision encoder weights may need sanitization
    # HF uses "blocks.N.attn.qkv" style, we use similar
    sanitized = _sanitize_vision_weights(raw_weights)

    # Cast to target dtype
    sanitized = {k: v.astype(dtype) for k, v in sanitized.items()}

    # Load into model
    model.load_weights(list(sanitized.items()))

    return model


def _sanitize_vision_weights(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Map HuggingFace vision encoder weight names to our model structure.

    HF naming: blocks.N.attn.qkv.weight, blocks.N.mlp.linear_fc1.weight, etc.
    Our naming: blocks.N.attn.qkv.weight, blocks.N.mlp.fc1.weight, etc.
    """
    sanitized = {}
    for name, tensor in weights.items():
        # Map MLP naming
        new_name = name
        new_name = new_name.replace("mlp.linear_fc1", "mlp.fc1")
        new_name = new_name.replace("mlp.linear_fc2", "mlp.fc2")

        sanitized[new_name] = tensor

    return sanitized


def load_tokenizer(model_dir: str | Path):
    """Load the Qwen2 tokenizer.

    Returns a HuggingFace tokenizer — we use it directly since
    tokenization is CPU-only and doesn't need MLX.
    """
    from transformers import AutoTokenizer

    model_dir = Path(model_dir)

    # Try text_tokenizer subdirectory first (HF Cosmos3 layout),
    # then fall back to root directory
    tokenizer_dir = model_dir / "text_tokenizer"
    if not tokenizer_dir.exists():
        tokenizer_dir = model_dir

    return AutoTokenizer.from_pretrained(
        str(tokenizer_dir),
        trust_remote_code=False,
    )
