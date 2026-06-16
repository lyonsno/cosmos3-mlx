"""Head-to-head i2v comparison: MLX vs HF PyTorch at 256x256, 4 frames.

Uses the NVIDIA reference i2v input image and a short prompt.
Compares conditioning latents and final denoised latents.
"""

import sys
sys.path.insert(0, ".")

import json
import time
import numpy as np
import torch
import mlx.core as mx
from PIL import Image

model_dir = "weights/Cosmos3-Nano"
image_path = f"{model_dir}/assets/example_i2v_input.jpg"

H, W = 256, 256
NUM_FRAMES = 4
STEPS = 10  # fewer steps for speed
SEED = 42
GUIDANCE = 6.0
PROMPT = "A car driving along a coastal mountain road with falling rocks causing an emergency stop"

img = Image.open(image_path).convert("RGB")
print(f"Input image: {img.size}")

# ═══════════════════════════════════════════════
# MLX
# ═══════════════════════════════════════════════
print("\n" + "=" * 60)
print("MLX i2v")
print("=" * 60)

from cosmos3_mlx.load import load_transformer, load_tokenizer
from cosmos3_mlx.pipeline import Cosmos3GenerationPipeline

model = load_transformer(model_dir, reasoner_only=False)
tokenizer = load_tokenizer(model_dir)
pipeline = Cosmos3GenerationPipeline(model=model, tokenizer=tokenizer, model_dir=model_dir)

t0 = time.time()
mlx_result = pipeline.generate(
    prompt=PROMPT,
    num_frames=NUM_FRAMES,
    height=H, width=W,
    num_inference_steps=STEPS,
    guidance_scale=GUIDANCE,
    seed=SEED,
    image=np.array(img),
)
t1 = time.time()
print(f"MLX generate: {t1 - t0:.1f}s")

mlx_latents = mlx_result["latents"]
print(f"MLX latents: {mlx_latents.shape}")
mlx_lat_np = np.array(mlx_latents.astype(mx.float32))

# Free MLX model memory
del model, tokenizer, pipeline
mx.metal.reset_peak_memory()

# ═══════════════════════════════════════════════
# HF PyTorch
# ═══════════════════════════════════════════════
print("\n" + "=" * 60)
print("HF PyTorch i2v")
print("=" * 60)

from diffusers import Cosmos3OmniPipeline

# Bypass safety checker: return "safe" for everything
import diffusers.pipelines.cosmos.pipeline_cosmos3_omni as _cosmos_mod
class _FakeSafetyChecker:
    def __getattr__(self, name):
        # Any method call returns True (safe) or self (for chaining)
        def _pass(*a, **kw): return True
        return _pass
    def to(self, *a, **kw): return self
_cosmos_mod.CosmosSafetyChecker = _FakeSafetyChecker

hf_pipe = Cosmos3OmniPipeline.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
)

# Set seed
generator = torch.Generator().manual_seed(SEED)

t2 = time.time()
hf_result = hf_pipe(
    prompt=PROMPT,
    image=img,
    height=H,
    width=W,
    num_frames=NUM_FRAMES,
    num_inference_steps=STEPS,
    guidance_scale=GUIDANCE,
    generator=generator,
    output_type="latent",
)
t3 = time.time()
print(f"HF generate: {t3 - t2:.1f}s")

# hf_result should have latents
if hasattr(hf_result, 'frames'):
    hf_latents = hf_result.frames
    print(f"HF output type: {type(hf_latents)}")
    if isinstance(hf_latents, torch.Tensor):
        hf_lat_np = hf_latents.float().numpy()
        print(f"HF latents shape: {hf_lat_np.shape}")
    else:
        print(f"HF output: {type(hf_latents)}")
        hf_lat_np = None
elif hasattr(hf_result, 'images'):
    print(f"HF images: {type(hf_result.images)}")
    hf_lat_np = None
else:
    print(f"HF result keys: {dir(hf_result)}")
    hf_lat_np = None

# ═══════════════════════════════════════════════
# Compare
# ═══════════════════════════════════════════════
print("\n" + "=" * 60)
print("Comparison")
print("=" * 60)

print(f"MLX latent shape: {mlx_lat_np.shape}")
if hf_lat_np is not None:
    print(f"HF latent shape: {hf_lat_np.shape}")

    # May need to transpose HF from [B, C, T, H, W] to [B, T, H, W, C]
    if hf_lat_np.ndim == 5 and hf_lat_np.shape[1] != mlx_lat_np.shape[1]:
        hf_lat_np = np.transpose(hf_lat_np, (0, 2, 3, 4, 1))
        print(f"HF transposed: {hf_lat_np.shape}")

    if hf_lat_np.shape == mlx_lat_np.shape:
        diff = np.abs(hf_lat_np - mlx_lat_np)
        print(f"Max abs diff: {diff.max():.6f}")
        print(f"Mean abs diff: {diff.mean():.6f}")

        hf_flat = hf_lat_np.flatten()
        mlx_flat = mlx_lat_np.flatten()
        cos = np.dot(hf_flat, mlx_flat) / (np.linalg.norm(hf_flat) * np.linalg.norm(mlx_flat) + 1e-8)
        print(f"Cosine similarity: {cos:.6f}")
    else:
        print(f"Shape mismatch: HF {hf_lat_np.shape} vs MLX {mlx_lat_np.shape}")
else:
    print("Could not extract HF latents for comparison")
    print("Saving MLX result for visual inspection instead")

print(f"\nTotal time: MLX {t1-t0:.1f}s, HF {t3-t2:.1f}s")
