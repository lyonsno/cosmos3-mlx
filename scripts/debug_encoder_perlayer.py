"""Per-layer comparison: MLX encoder vs fresh HF VAE encoder.

Key: instantiate a FRESH HF VAE for each comparison to avoid WanCausalConv3d
cache contamination. The HF VAE has internal state (feat_cache) that persists
between calls and contaminates manual per-layer comparisons.
"""

import sys
sys.path.insert(0, ".")

import json
import numpy as np
import torch
import mlx.core as mx

# ── Create a deterministic test image ──
np.random.seed(42)
H, W = 256, 256
image_np = np.random.rand(H, W, 3).astype(np.float32)

vae_dir = "weights/Cosmos3-Nano/vae"
with open(f"{vae_dir}/config.json") as f:
    config = json.load(f)

# ── HF reference: full encode ──
print("=" * 60)
print("HF full encode (fresh model)")
print("=" * 60)

from diffusers import AutoencoderKLWan

hf_vae = AutoencoderKLWan.from_pretrained(
    "weights/Cosmos3-Nano", subfolder="vae",
    torch_dtype=torch.bfloat16,
)
hf_vae.eval()

# HF expects [B, C, T, H, W] in [-1, 1]
image_pt = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)  # [1, 3, 1, H, W]
image_pt = image_pt * 2.0 - 1.0
image_pt = image_pt.to(torch.bfloat16)

with torch.no_grad():
    hf_result = hf_vae.encode(image_pt)
    hf_mu = hf_result.latent_dist.mean  # [1, z_dim, T_lat, H_lat, W_lat]

print(f"HF latent shape: {hf_mu.shape}")
print(f"HF latent stats: mean={hf_mu.mean().item():.4f}, std={hf_mu.std().item():.4f}")

# Normalize HF latents same way as generation pipeline
latents_mean_t = torch.tensor(config["latents_mean"], dtype=torch.bfloat16).view(1, -1, 1, 1, 1)
latents_std_t = torch.tensor(config["latents_std"], dtype=torch.bfloat16).view(1, -1, 1, 1, 1)
hf_mu_norm = (hf_mu - latents_mean_t) / latents_std_t
print(f"HF normalized latent stats: mean={hf_mu_norm.mean().item():.4f}, std={hf_mu_norm.std().item():.4f}")

# Convert to channels-last for comparison
hf_mu_cl = hf_mu.permute(0, 2, 3, 4, 1).float().numpy()  # [1, T, H, W, z_dim]

del hf_vae
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ── MLX encode ──
print()
print("=" * 60)
print("MLX encode")
print("=" * 60)

from cosmos3_mlx.encode_vae import encode_image

mlx_result = encode_image(image_np, vae_dir)
mx.eval(mlx_result)

# mlx_result is normalized; denormalize to compare raw mu
latents_mean_mx = mx.array(config["latents_mean"], dtype=mx.bfloat16)
latents_std_mx = mx.array(config["latents_std"], dtype=mx.bfloat16)
mlx_mu = mlx_result * latents_std_mx + latents_mean_mx
mx.eval(mlx_mu)

mlx_mu_np = np.array(mlx_mu.astype(mx.float32))

print(f"MLX latent shape: {mlx_mu_np.shape}")
print(f"MLX latent stats: mean={np.mean(mlx_mu_np):.4f}, std={np.std(mlx_mu_np):.4f}")

# ── Compare ──
print()
print("=" * 60)
print("Comparison (raw mu, before normalization)")
print("=" * 60)

diff = np.abs(hf_mu_cl - mlx_mu_np)
print(f"Max abs diff: {diff.max():.6f}")
print(f"Mean abs diff: {diff.mean():.6f}")

# Cosine similarity
hf_flat = hf_mu_cl.flatten()
mlx_flat = mlx_mu_np.flatten()
cos_sim = np.dot(hf_flat, mlx_flat) / (np.linalg.norm(hf_flat) * np.linalg.norm(mlx_flat) + 1e-8)
print(f"Cosine similarity: {cos_sim:.6f}")

# Per-channel correlation
print("\nPer-channel correlation:")
z_dim = config["z_dim"]
for ch in range(min(z_dim, 8)):
    hf_ch = hf_mu_cl[..., ch].flatten()
    mlx_ch = mlx_mu_np[..., ch].flatten()
    corr = np.corrcoef(hf_ch, mlx_ch)[0, 1]
    print(f"  ch{ch:2d}: corr={corr:.4f}, max_diff={np.abs(hf_ch - mlx_ch).max():.4f}")
