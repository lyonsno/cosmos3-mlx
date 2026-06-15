"""Verify chunked VAE encoder for 5 frames (2 chunks) at 64x64."""

import sys
sys.path.insert(0, ".")

import json
import numpy as np
import torch
import mlx.core as mx

np.random.seed(42)
T, H, W = 5, 64, 64
video_np = np.random.rand(T, H, W, 3).astype(np.float32)

vae_dir = "weights/Cosmos3-Nano/vae"
with open(f"{vae_dir}/config.json") as f:
    config = json.load(f)

z_dim = config["z_dim"]

# ── HF ──
print(f"HF _encode: {T} frames at {H}x{W}")
from diffusers import AutoencoderKLWan

hf_vae = AutoencoderKLWan.from_pretrained(
    "weights/Cosmos3-Nano", subfolder="vae",
    torch_dtype=torch.bfloat16,
)
hf_vae.eval()

video_pt = torch.from_numpy(video_np).permute(0, 3, 1, 2).unsqueeze(0).permute(0, 2, 1, 3, 4)
video_pt = (video_pt * 2.0 - 1.0).to(torch.bfloat16)
print(f"Input: {video_pt.shape}")

# Trace HF chunking
print(f"  iter_ = 1 + ({T}-1)//4 = {1 + (T-1)//4}")
print(f"  Chunk 0: frame 0")
if T > 1:
    for i in range(1, 1 + (T-1)//4):
        s = 1 + 4*(i-1)
        e = 1 + 4*i
        print(f"  Chunk {i}: frames {s}:{e}")

with torch.no_grad():
    hf_enc = hf_vae._encode(video_pt)

hf_mu = hf_enc[:, :z_dim]
hf_mu_cl = hf_mu.permute(0, 2, 3, 4, 1).float().numpy()
print(f"HF output: {hf_mu.shape}")
print(f"HF mu: mean={hf_mu.mean().item():.4f}, std={hf_mu.std().item():.4f}")

del hf_vae

# ── MLX ──
print()
print(f"MLX chunked encode: {T} frames at {H}x{W}")

from cosmos3_mlx.encode_vae import encode_video

mlx_result = encode_video(video_np, vae_dir)
mx.eval(mlx_result)

latents_mean_mx = mx.array(config["latents_mean"], dtype=mx.bfloat16)
latents_std_mx = mx.array(config["latents_std"], dtype=mx.bfloat16)
mlx_mu = mlx_result * latents_std_mx + latents_mean_mx
mx.eval(mlx_mu)
mlx_mu_np = np.array(mlx_mu.astype(mx.float32))

print(f"MLX output: {mlx_mu_np.shape}")
print(f"MLX mu: mean={np.mean(mlx_mu_np):.4f}, std={np.std(mlx_mu_np):.4f}")

# ── Compare ──
print()
print(f"HF shape: {hf_mu_cl.shape}")
print(f"MLX shape: {mlx_mu_np.shape}")

if hf_mu_cl.shape != mlx_mu_np.shape:
    print(f"SHAPE MISMATCH: HF {hf_mu_cl.shape} vs MLX {mlx_mu_np.shape}")
    sys.exit(1)

diff = np.abs(hf_mu_cl - mlx_mu_np)
print(f"Max abs diff: {diff.max():.6f}")
print(f"Mean abs diff: {diff.mean():.6f}")

hf_flat = hf_mu_cl.flatten()
mlx_flat = mlx_mu_np.flatten()
cos = np.dot(hf_flat, mlx_flat) / (np.linalg.norm(hf_flat) * np.linalg.norm(mlx_flat) + 1e-8)
print(f"Cosine similarity: {cos:.6f}")

for t in range(hf_mu_cl.shape[1]):
    hf_t = hf_mu_cl[0, t].flatten()
    mlx_t = mlx_mu_np[0, t].flatten()
    cos_t = np.dot(hf_t, mlx_t) / (np.linalg.norm(hf_t) * np.linalg.norm(mlx_t) + 1e-8)
    max_t = np.abs(hf_t - mlx_t).max()
    print(f"  Frame {t}: cosine={cos_t:.6f}, max_diff={max_t:.4f}")

if cos > 0.999:
    print("\nPASS: Chunked encoder matches HF")
else:
    print(f"\nFAIL: Cosine {cos:.6f} below threshold 0.999")
