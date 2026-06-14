"""Generation pipeline for Cosmos 3 Nano on MLX.

Wires the MoT transformer (generation path), scheduler, timestep
embedding, video VAE decoder, and audio decoder into a complete
text-to-image/video generation flow.
"""

import time
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .model import Cosmos3Config, Cosmos3Model
from .scheduler import UniPCScheduler
from .timestep import TimestepEmbedding, apply_timestep_to_noisy_tokens
from .vae import VAEConfig, WanDecoder
from .audio import AudioDecoderConfig, AudioDecoder


class Cosmos3GenerationPipeline:
    """End-to-end generation pipeline: text → image/video (+ audio).

    Orchestrates:
    1. Text tokenization and embedding
    2. Noise latent preparation
    3. Denoising loop (MoT generation path + scheduler)
    4. VAE decode to pixels
    5. Optional audio decode
    """

    def __init__(
        self,
        model: Cosmos3Model,
        tokenizer,
        vae_decoder: Optional[WanDecoder] = None,
        audio_decoder: Optional[AudioDecoder] = None,
        vae_config: Optional[VAEConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.vae_decoder = vae_decoder
        self.audio_decoder = audio_decoder
        self.vae_config = vae_config or VAEConfig()
        self.scheduler = UniPCScheduler()

    def _prepare_noise_latents(
        self,
        num_frames: int = 1,
        height: int = 512,
        width: int = 512,
        z_dim: int = 48,
        dtype: mx.Dtype = mx.bfloat16,
    ) -> mx.array:
        """Prepare initial noise latents.

        Returns:
            [1, T_lat, H_lat, W_lat, z_dim] noise tensor (channels-last)
        """
        # Compute latent dimensions
        t_lat = max(1, num_frames // 4)  # 4x temporal compression
        h_lat = height // 16  # 16x spatial compression
        w_lat = width // 16

        noise = mx.random.normal((1, t_lat, h_lat, w_lat, z_dim)).astype(dtype)
        return noise

    def _patchify_latents(self, latents: mx.array) -> mx.array:
        """Convert VAE latents to patch tokens for the transformer.

        Input: [batch, T, H, W, z_dim]
        Output: [batch, num_patches, patch_latent_dim]

        With patch_size=2: each 2x2 spatial region becomes one token.
        patch_latent_dim = z_dim * patch_size * patch_size = 48 * 4 = 192
        """
        batch, t, h, w, z = latents.shape
        p = self.vae_config.patch_size

        # Reshape into patches
        h_p = h // p
        w_p = w // p
        # [B, T, H//p, p, W//p, p, z] -> [B, T*H_p*W_p, p*p*z]
        x = latents.reshape(batch, t, h_p, p, w_p, p, z)
        x = mx.transpose(x, (0, 1, 2, 4, 3, 5, 6))  # [B, T, H_p, W_p, p, p, z]
        x = x.reshape(batch, t * h_p * w_p, p * p * z)

        return x

    def _unpatchify_latents(
        self, tokens: mx.array, t: int, h_p: int, w_p: int
    ) -> mx.array:
        """Convert patch tokens back to VAE latent shape.

        Input: [batch, num_patches, patch_latent_dim]
        Output: [batch, T, H, W, z_dim]
        """
        batch = tokens.shape[0]
        p = self.vae_config.patch_size
        z = self.vae_config.z_dim

        x = tokens.reshape(batch, t, h_p, w_p, p, p, z)
        x = mx.transpose(x, (0, 1, 2, 4, 3, 5, 6))  # [B, T, H_p, p, W_p, p, z]
        x = x.reshape(batch, t, h_p * p, w_p * p, z)

        return x

    def generate(
        self,
        prompt: str,
        num_frames: int = 1,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 6.0,
        seed: Optional[int] = None,
    ) -> dict:
        """Generate image/video from text prompt.

        Args:
            prompt: text description
            num_frames: number of video frames (1 = single image)
            height: output height in pixels
            width: output width in pixels
            num_inference_steps: denoising steps
            guidance_scale: classifier-free guidance strength
            seed: random seed for reproducibility

        Returns:
            dict with 'latents', 'video' (if VAE available), 'audio' (if enabled)
        """
        if seed is not None:
            mx.random.seed(seed)

        dtype = mx.bfloat16

        # 1. Tokenize prompt
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        cond_ids = mx.array([self.tokenizer.encode(text)])

        # Unconditional prompt for CFG
        uncond_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            tokenize=False, add_generation_prompt=True,
        )
        uncond_ids = mx.array([self.tokenizer.encode(uncond_text)])

        # 2. Prepare noise latents
        z_dim = self.vae_config.z_dim
        latents = self._prepare_noise_latents(
            num_frames, height, width, z_dim, dtype
        )
        t_lat = latents.shape[1]
        h_lat = latents.shape[2]
        w_lat = latents.shape[3]

        # Patchify: [1, T_lat, H_lat, W_lat, z_dim] -> [1, num_patches, patch_dim]
        p = self.vae_config.patch_size
        h_p = h_lat // p
        w_p = w_lat // p
        num_patches = t_lat * h_p * w_p

        # 3. Set up scheduler
        self.scheduler.set_timesteps(num_inference_steps)

        print(f"  Latent shape: ({t_lat}, {h_lat}, {w_lat}, {z_dim})")
        print(f"  Patches: {num_patches} ({t_lat}×{h_p}×{w_p})")
        print(f"  Denoising steps: {num_inference_steps}")

        # 4. Denoising loop
        for i in range(num_inference_steps):
            sigma = float(self.scheduler.sigmas[i].item())

            # Patchify current latents
            gen_tokens = self._patchify_latents(latents).astype(dtype)

            # Timestep tensor (sigma * num_train_timesteps)
            t_tensor = mx.array([sigma * self.scheduler.num_train_timesteps]).astype(dtype)

            # Conditional forward: get velocity prediction
            cond_velocity = self.model.diffusion_forward(
                cond_ids, gen_tokens, t_tensor,
                grid_t=t_lat, grid_h=h_p, grid_w=w_p,
            )

            # Classifier-free guidance
            if guidance_scale != 1.0:
                # Eval conditional velocity first to free its computation graph
                # before building the unconditional graph (halves peak memory)
                mx.eval(cond_velocity)
                uncond_velocity = self.model.diffusion_forward(
                    uncond_ids, gen_tokens, t_tensor,
                    grid_t=t_lat, grid_h=h_p, grid_w=w_p,
                )
                velocity_patches = uncond_velocity + guidance_scale * (
                    cond_velocity - uncond_velocity
                )
            else:
                velocity_patches = cond_velocity

            # Unpatchify velocity back to latent shape
            velocity = self._unpatchify_latents(
                velocity_patches, t_lat, h_p, w_p
            )

            # Scheduler step (uses internal step_index)
            latents = self.scheduler.step(velocity, t_tensor, latents)
            mx.eval(latents)

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Step {i+1}/{num_inference_steps} (σ={sigma:.4f})")

        # 5. Decode latents
        result = {"latents": latents}

        if self.vae_decoder is not None:
            print("  Decoding video...")
            video = self.vae_decoder(latents)
            mx.eval(video)
            # Convert to numpy [T, H, W, C] in [0, 1]
            video = (mx.clip(video[0], -1, 1) + 1) / 2  # [-1,1] -> [0,1]
            result["video"] = video

        print("  Done!")
        return result


def run_generation_smoke(model_dir: str):
    """Quick smoke test for generation pipeline."""
    from .load import load_transformer, load_tokenizer

    print("=== Cosmos 3 Nano Generation Smoke ===\n")

    # Load model + tokenizer
    print("Loading model...")
    t0 = time.time()
    model = load_transformer(model_dir, reasoner_only=False)
    tokenizer = load_tokenizer(model_dir)
    t1 = time.time()
    print(f"  Loaded in {t1-t0:.1f}s\n")

    # Create pipeline (no VAE decoder for now — just test the loop)
    pipeline = Cosmos3GenerationPipeline(
        model=model,
        tokenizer=tokenizer,
    )

    # Generate
    print("Generating 64x64 image from text...")
    result = pipeline.generate(
        prompt="A beautiful sunset over the ocean",
        num_frames=1,
        height=64,
        width=64,
        num_inference_steps=5,
        guidance_scale=6.0,
        seed=42,
    )

    latents = result["latents"]
    print(f"\n  Output latent shape: {latents.shape}")
    print(f"  Latent stats: mean={mx.mean(latents).item():.4f}, "
          f"std={mx.std(latents).item():.4f}")
    print("\n=== Generation loop completed! ===")
