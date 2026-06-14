"""Tests for the generation pipeline."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from cosmos3_mlx.model import Cosmos3Config, Cosmos3Model
from cosmos3_mlx.pipeline import Cosmos3GenerationPipeline
from cosmos3_mlx.vae import VAEConfig, WanDecoder


class FakeTokenizer:
    """Minimal tokenizer stub for testing."""

    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        # Return content from last non-system message
        for msg in reversed(messages):
            if msg["role"] != "system":
                return msg["content"]
        return ""

    def encode(self, text, **kwargs):
        return [1, 2, 3, 4, 5]  # Fixed tokens

    def convert_tokens_to_ids(self, token):
        return 99  # Fake vision_start token


class TestGenerationPipeline:
    """Test the end-to-end generation pipeline."""

    @pytest.fixture
    def small_pipeline(self):
        """Create a small pipeline for testing."""
        cfg = Cosmos3Config(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            intermediate_size=256,
            vocab_size=1000,
            mrope_section=[6, 5, 5],
        )
        model = Cosmos3Model(cfg)
        # Override proj_in/out to match test VAE dims (z_dim=8, patch=2 → 32)
        model.proj_in = nn.Linear(32, 128, bias=True)
        model.proj_out = nn.Linear(128, 32, bias=True)

        vae_cfg = VAEConfig(
            z_dim=8,
            decoder_base_dim=16,
            dim_mult=[1, 2],
            num_res_blocks=1,
            temporal_upsample=[False],
            out_channels=3,
            patch_size=2,
        )
        vae = WanDecoder(vae_cfg)

        return Cosmos3GenerationPipeline(
            model=model,
            tokenizer=FakeTokenizer(),
            vae_decoder=vae,
            vae_config=vae_cfg,
        )

    def test_prepare_noise_latents(self, small_pipeline):
        """Should create correctly shaped noise."""
        latents = small_pipeline._prepare_noise_latents(
            num_frames=4, height=64, width=64, z_dim=8
        )
        assert latents.shape == (1, 1, 4, 4, 8)

    def test_patchify_roundtrip(self, small_pipeline):
        """Patchify then unpatchify should recover original shape."""
        small_pipeline.vae_config.patch_size = 2
        small_pipeline.vae_config.z_dim = 8
        latents = mx.random.normal((1, 1, 4, 4, 8))

        patches = small_pipeline._patchify_latents(latents)
        assert patches.shape == (1, 4, 32)  # 1*2*2 patches, 2*2*8=32 patch_dim

        recovered = small_pipeline._unpatchify_latents(patches, t=1, h_p=2, w_p=2)
        mx.eval(latents, recovered)
        assert mx.allclose(latents, recovered, atol=1e-6).item()

    def test_generate_returns_latents(self, small_pipeline):
        """Generate should return latents even without VAE."""
        pipeline_no_vae = Cosmos3GenerationPipeline(
            model=small_pipeline.model,
            tokenizer=FakeTokenizer(),
            vae_config=small_pipeline.vae_config,
        )
        result = pipeline_no_vae.generate(
            prompt="test",
            num_frames=1,
            height=64,
            width=64,
            num_inference_steps=2,
            seed=0,
        )
        assert "latents" in result
        mx.eval(result["latents"])
        assert not mx.any(mx.isnan(result["latents"])).item()

    def test_generate_with_vae(self, small_pipeline):
        """Generate with VAE should return video frames."""
        result = small_pipeline.generate(
            prompt="test",
            num_frames=1,
            height=64,
            width=64,
            num_inference_steps=2,
            seed=0,
        )
        assert "video" in result
        mx.eval(result["video"])
        assert result["video"].shape[-1] == 3  # RGB
        assert not mx.any(mx.isnan(result["video"])).item()

    def test_deterministic_with_seed(self, small_pipeline):
        """Same seed should produce same latents."""
        r1 = small_pipeline.generate(
            prompt="test", height=64, width=64,
            num_inference_steps=2, seed=42,
        )
        r2 = small_pipeline.generate(
            prompt="test", height=64, width=64,
            num_inference_steps=2, seed=42,
        )
        mx.eval(r1["latents"], r2["latents"])
        assert mx.allclose(r1["latents"], r2["latents"], atol=1e-6).item()
