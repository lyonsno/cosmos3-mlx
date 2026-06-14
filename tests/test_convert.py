"""Tests for weight conversion from HuggingFace safetensors to MLX."""

import mlx.core as mx
import pytest

from cosmos3_mlx.convert import (
    map_weight_name,
    convert_weights,
    WEIGHT_NAME_MAP,
)


class TestWeightNameMapping:
    """Test weight name mapping from HuggingFace to MLX."""

    def test_embedding(self):
        """embed_tokens maps correctly."""
        assert map_weight_name("embed_tokens.weight") == "embed_tokens.weight"

    def test_lm_head(self):
        """lm_head maps correctly."""
        assert map_weight_name("lm_head.weight") == "lm_head.weight"

    def test_layer_norm(self):
        """Layer norms map correctly."""
        assert map_weight_name("layers.0.input_layernorm.weight") == \
            "layers.0.input_layernorm.weight"

    def test_attention_q(self):
        """Understanding Q projection uses to_q naming."""
        assert map_weight_name("layers.5.self_attn.to_q.weight") == \
            "layers.5.self_attn.to_q.weight"

    def test_attention_k(self):
        """Understanding K projection uses to_k naming."""
        assert map_weight_name("layers.10.self_attn.to_k.weight") == \
            "layers.10.self_attn.to_k.weight"

    def test_generation_q(self):
        """Generation Q projection maps correctly."""
        assert map_weight_name("layers.0.self_attn.add_q_proj.weight") == \
            "layers.0.self_attn.add_q_proj.weight"

    def test_mlp(self):
        """MLP projections map correctly."""
        assert map_weight_name("layers.0.mlp.gate_proj.weight") == \
            "layers.0.mlp.gate_proj.weight"

    def test_generation_mlp(self):
        """Generation MLP maps correctly."""
        assert map_weight_name("layers.0.mlp_moe_gen.gate_proj.weight") == \
            "layers.0.mlp_moe_gen.gate_proj.weight"

    def test_final_norm(self):
        """Final norms map correctly."""
        assert map_weight_name("norm.weight") == "norm.weight"

    def test_to_out_projection(self):
        """Output projection maps to to_out.weight (with list index)."""
        mapped = map_weight_name("layers.0.self_attn.to_out.0.weight")
        assert mapped == "layers.0.self_attn.to_out.weight"


class TestConvertWeights:
    """Test the weight conversion pipeline."""

    def test_skip_moe_gen_in_reasoner_mode(self):
        """In reasoner-only mode, _moe_gen weights should be skipped."""
        fake_weights = {
            "layers.0.input_layernorm.weight": mx.ones((128,)),
            "layers.0.input_layernorm_moe_gen.weight": mx.ones((128,)),
            "layers.0.mlp_moe_gen.gate_proj.weight": mx.ones((256, 128)),
        }
        converted = convert_weights(fake_weights, reasoner_only=True)
        assert "layers.0.input_layernorm.weight" in converted
        assert "layers.0.input_layernorm_moe_gen.weight" not in converted
        assert "layers.0.mlp_moe_gen.gate_proj.weight" not in converted

    def test_keep_moe_gen_in_full_mode(self):
        """In full mode, _moe_gen weights should be kept."""
        fake_weights = {
            "layers.0.input_layernorm.weight": mx.ones((128,)),
            "layers.0.input_layernorm_moe_gen.weight": mx.ones((128,)),
        }
        converted = convert_weights(fake_weights, reasoner_only=False)
        assert "layers.0.input_layernorm.weight" in converted
        assert "layers.0.input_layernorm_moe_gen.weight" in converted

    def test_to_out_renaming(self):
        """to_out.0.weight should become to_out.weight."""
        fake_weights = {
            "layers.0.self_attn.to_out.0.weight": mx.ones((128, 128)),
        }
        converted = convert_weights(fake_weights, reasoner_only=False)
        assert "layers.0.self_attn.to_out.weight" in converted
        assert "layers.0.self_attn.to_out.0.weight" not in converted

    def test_preserves_tensor_values(self):
        """Conversion should not modify tensor values."""
        val = mx.random.normal((64, 128))
        fake_weights = {"embed_tokens.weight": val}
        converted = convert_weights(fake_weights, reasoner_only=False)
        assert mx.array_equal(converted["embed_tokens.weight"], val).item()

    def test_skip_diffusion_projections_in_reasoner_mode(self):
        """Diffusion-specific projections should be skipped in reasoner mode."""
        fake_weights = {
            "embed_tokens.weight": mx.ones((1000, 128)),
            "proj_in.weight": mx.ones((128, 192)),
            "proj_out.weight": mx.ones((192, 128)),
            "audio_proj_in.weight": mx.ones((128, 64)),
            "audio_proj_out.weight": mx.ones((64, 128)),
            "time_embedder.linear_1.weight": mx.ones((128, 128)),
            "action_proj_in.fc.weight": mx.ones((128, 64)),
        }
        converted = convert_weights(fake_weights, reasoner_only=True)
        assert "embed_tokens.weight" in converted
        assert "proj_in.weight" not in converted
        assert "proj_out.weight" not in converted
        assert "audio_proj_in.weight" not in converted
        assert "time_embedder.linear_1.weight" not in converted
        assert "action_proj_in.fc.weight" not in converted
