"""Bisect i2v denoising: compare MLX vs HF after step 0.

Extracts: initial mixed latents, velocity after step 0, latents after step 0.
"""

import sys
sys.path.insert(0, ".")

import json
import numpy as np
import torch
import mlx.core as mx
from PIL import Image

model_dir = "weights/Cosmos3-Nano"
img = Image.open(f"{model_dir}/assets/example_i2v_input.jpg").convert("RGB")
H, W, T = 256, 256, 4
SEED = 42
GUIDANCE = 6.0
PROMPT = "A car driving along a coastal mountain road with falling rocks causing an emergency stop"

# ═══════════════════════════════════════════════
# MLX: 1 step, capture intermediates
# ═══════════════════════════════════════════════
print("=" * 60)
print("MLX: 1 denoising step")
print("=" * 60)

from cosmos3_mlx.load import load_transformer, load_tokenizer
from cosmos3_mlx.pipeline import Cosmos3GenerationPipeline

model = load_transformer(model_dir, reasoner_only=False)
tokenizer = load_tokenizer(model_dir)
pipeline = Cosmos3GenerationPipeline(model=model, tokenizer=tokenizer, model_dir=model_dir)

# Run with 1 step to get first-step output
mlx_1step = pipeline.generate(
    prompt=PROMPT, num_frames=T, height=H, width=W,
    num_inference_steps=1, guidance_scale=GUIDANCE, seed=SEED,
    image=np.array(img),
)
mlx_lat_1 = np.array(mlx_1step["latents"].astype(mx.float32))
print(f"MLX after step 0: shape={mlx_lat_1.shape}, mean={mlx_lat_1.mean():.6f}, std={mlx_lat_1.std():.6f}")

# Also run 2 steps to see accumulation
mlx_2step = pipeline.generate(
    prompt=PROMPT, num_frames=T, height=H, width=W,
    num_inference_steps=2, guidance_scale=GUIDANCE, seed=SEED,
    image=np.array(img),
)
mlx_lat_2 = np.array(mlx_2step["latents"].astype(mx.float32))
print(f"MLX after step 1: shape={mlx_lat_2.shape}, mean={mlx_lat_2.mean():.6f}, std={mlx_lat_2.std():.6f}")

del model, tokenizer, pipeline

# ═══════════════════════════════════════════════
# HF: 1 step, capture intermediates via callback
# ═══════════════════════════════════════════════
print()
print("=" * 60)
print("HF: 1 denoising step")
print("=" * 60)

import diffusers.pipelines.cosmos.pipeline_cosmos3_omni as _cm
class _F:
    def __getattr__(self, n):
        def _p(*a, **kw): return True
        return _p
    def to(self, *a, **kw): return self
_cm.CosmosSafetyChecker = _F

from diffusers import Cosmos3OmniPipeline
hf_pipe = Cosmos3OmniPipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16)

generator1 = torch.Generator().manual_seed(SEED)
hf_1step = hf_pipe(
    prompt=PROMPT, image=img, height=H, width=W, num_frames=T,
    num_inference_steps=1, guidance_scale=GUIDANCE,
    generator=generator1, output_type="latent",
)
hf_lat_1 = hf_1step.video
if isinstance(hf_lat_1, torch.Tensor):
    hf_lat_1 = hf_lat_1.float().numpy()
    if hf_lat_1.shape[1] == 48:  # [B, C, T, H, W] -> [B, T, H, W, C]
        hf_lat_1 = np.transpose(hf_lat_1, (0, 2, 3, 4, 1))
print(f"HF after step 0: shape={hf_lat_1.shape}, mean={hf_lat_1.mean():.6f}, std={hf_lat_1.std():.6f}")

generator2 = torch.Generator().manual_seed(SEED)
hf_2step = hf_pipe(
    prompt=PROMPT, image=img, height=H, width=W, num_frames=T,
    num_inference_steps=2, guidance_scale=GUIDANCE,
    generator=generator2, output_type="latent",
)
hf_lat_2 = hf_2step.video
if isinstance(hf_lat_2, torch.Tensor):
    hf_lat_2 = hf_lat_2.float().numpy()
    if hf_lat_2.shape[1] == 48:
        hf_lat_2 = np.transpose(hf_lat_2, (0, 2, 3, 4, 1))
print(f"HF after step 1: shape={hf_lat_2.shape}, mean={hf_lat_2.mean():.6f}, std={hf_lat_2.std():.6f}")

# ═══════════════════════════════════════════════
# Compare
# ═══════════════════════════════════════════════
print()
print("=" * 60)
print("Comparison")
print("=" * 60)

def compare(name, a, b):
    if a.shape != b.shape:
        print(f"{name}: SHAPE MISMATCH {a.shape} vs {b.shape}")
        return
    diff = np.abs(a - b)
    cos = np.dot(a.flatten(), b.flatten()) / (np.linalg.norm(a.flatten()) * np.linalg.norm(b.flatten()) + 1e-8)
    print(f"{name}: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}, cosine={cos:.6f}")

compare("After step 0 (1-step run)", mlx_lat_1, hf_lat_1)
compare("After step 1 (2-step run)", mlx_lat_2, hf_lat_2)

# Check if step 0 output is basically identical (meaning divergence is later)
print()
print("Stats comparison:")
print(f"  Step 0: MLX std={mlx_lat_1.std():.4f}, HF std={hf_lat_1.std():.4f}")
print(f"  Step 1: MLX std={mlx_lat_2.std():.4f}, HF std={hf_lat_2.std():.4f}")
