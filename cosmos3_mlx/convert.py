"""Convert Cosmos 3 weights from HuggingFace safetensors to MLX format.

Handles:
- Weight name mapping (to_out.0 -> to_out, etc.)
- Reasoner-only mode: strips generation/diffusion weights
- Loading sharded safetensors from HuggingFace Hub
"""

import re
from pathlib import Path
from typing import Optional

import mlx.core as mx

# Patterns for generation/diffusion-only weights (skipped in reasoner mode)
GENERATION_PATTERNS = [
    r".*_moe_gen.*",           # MoE generation layers
    r".*add_q_proj.*",         # Generation Q projections
    r".*add_k_proj.*",         # Generation K projections
    r".*add_v_proj.*",         # Generation V projections
    r".*to_add_out.*",         # Generation output projections
    r".*norm_added_.*",        # Generation QK norms
    r".*norm_moe_gen.*",       # Generation final norm
    r"proj_in\..*",            # Diffusion input projection
    r"proj_out\..*",           # Diffusion output projection
    r"audio_proj_in\..*",      # Audio input projection
    r"audio_proj_out\..*",     # Audio output projection
    r"action_proj_in\..*",     # Action input projection
    r"action_proj_out\..*",    # Action output projection
    r"time_embedder\..*",      # Timestep embedder
    r"action_modality_embed",  # Action modality embedding
    r"audio_modality_embed",   # Audio modality embedding
]

_GENERATION_RE = [re.compile(p) for p in GENERATION_PATTERNS]

# Weight name remapping: HuggingFace -> MLX
# Most names stay the same. The main change is to_out.0.weight -> to_out.weight
WEIGHT_NAME_MAP = {
    # The HF diffusers model wraps output projection in nn.ModuleList
    # so it's to_out.0.weight instead of to_out.weight
}


def map_weight_name(name: str) -> str:
    """Map a HuggingFace weight name to the MLX model name.

    The main transformation: to_out.0.weight -> to_out.weight
    (and to_add_out.0.weight -> to_add_out.weight)
    """
    # Strip the .0 index from output projections
    name = re.sub(r"\.to_out\.0\.", ".to_out.", name)
    name = re.sub(r"\.to_add_out\.0\.", ".to_add_out.", name)
    return name


def _is_generation_weight(name: str) -> bool:
    """Check if a weight belongs to the generation/diffusion pathway."""
    return any(p.search(name) for p in _GENERATION_RE)


def convert_weights(
    weights: dict[str, mx.array],
    reasoner_only: bool = True,
) -> dict[str, mx.array]:
    """Convert a dict of weights from HF naming to MLX naming.

    Args:
        weights: dict mapping HF weight names to arrays
        reasoner_only: if True, strip generation/diffusion weights

    Returns:
        dict mapping MLX weight names to arrays
    """
    converted = {}
    for name, tensor in weights.items():
        # Skip generation weights in reasoner mode
        if reasoner_only and _is_generation_weight(name):
            continue

        # Map the name
        mlx_name = map_weight_name(name)
        converted[mlx_name] = tensor

    return converted


def load_and_convert_from_hub(
    model_id: str = "nvidia/Cosmos3-Nano",
    output_dir: Optional[str] = None,
    reasoner_only: bool = True,
    component: str = "transformer",
) -> dict[str, mx.array]:
    """Download and convert weights from HuggingFace Hub.

    Args:
        model_id: HuggingFace model ID
        output_dir: directory to save converted weights (optional)
        reasoner_only: strip generation weights
        component: which component to convert ("transformer", "vision_encoder", etc.)

    Returns:
        dict of converted weights
    """
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    # Download model
    model_path = Path(snapshot_download(model_id))
    component_path = model_path / component

    # Load all safetensors shards
    weights = {}
    for shard in sorted(component_path.glob("*.safetensors")):
        with safe_open(str(shard), framework="numpy") as f:
            for key in f.keys():
                weights[key] = mx.array(f.get_tensor(key))

    # Convert
    converted = convert_weights(weights, reasoner_only=reasoner_only)

    # Optionally save
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(output_path / "weights.safetensors"), converted)

    return converted
