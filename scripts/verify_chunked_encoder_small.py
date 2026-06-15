"""Verify chunked VAE encoder matches HF for 2-frame input at 64x64.

Smaller test that runs in reasonable time on CPU.
"""

import sys
sys.path.insert(0, ".")

import json
import numpy as np
import torch
import mlx.core as mx

# Deterministic 2-frame "video" at tiny resolution
np.random.seed(42)
T, H, W = 2, 64, 64
video_np = np.random.rand(T, H, W, 3).astype(np.float32)

vae_dir = "weights/Cosmos3-Nano/vae"
with open(f"{vae_dir}/config.json") as f:
    config = json.load(f)

# ── HF reference ──
print("=" * 60)
print(f"HF _encode: {T} frames at {H}x{W}")
print("=" * 60)

from diffusers import AutoencoderKLWan

hf_vae = AutoencoderKLWan.from_pretrained(
    "weights/Cosmos3-Nano", subfolder="vae",
    torch_dtype=torch.bfloat16,
)
hf_vae.eval()

# HF expects [B, C, T, H, W] in [-1, 1]
video_pt = torch.from_numpy(video_np).permute(0, 3, 1, 2)  # [T, C, H, W]
video_pt = video_pt.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, C, T, H, W]
video_pt = video_pt * 2.0 - 1.0
video_pt = video_pt.to(torch.bfloat16)

print(f"Input shape: {video_pt.shape}")

with torch.no_grad():
    hf_enc = hf_vae._encode(video_pt)

# hf_enc is the raw quant_conv output [1, 2*z_dim, T_lat, H_lat, W_lat]
z_dim = config["z_dim"]
hf_mu = hf_enc[:, :z_dim]
hf_mu_cl = hf_mu.permute(0, 2, 3, 4, 1).float().numpy()  # [1, T_lat, H_lat, W_lat, z_dim]

print(f"HF output shape: {hf_mu.shape}")
print(f"HF mu stats: mean={hf_mu.mean().item():.4f}, std={hf_mu.std().item():.4f}")

del hf_vae
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ── MLX chunked encode ──
print()
print("=" * 60)
print(f"MLX chunked encode: {T} frames at {H}x{W}")
print("=" * 60)

from cosmos3_mlx.encode_vae import encode_video

mlx_result = encode_video(video_np, vae_dir)
mx.eval(mlx_result)

# Denormalize to compare raw mu
latents_mean_mx = mx.array(config["latents_mean"], dtype=mx.bfloat16)
latents_std_mx = mx.array(config["latents_std"], dtype=mx.bfloat16)
mlx_mu = mlx_result * latents_std_mx + latents_mean_mx
mx.eval(mlx_mu)
mlx_mu_np = np.array(mlx_mu.astype(mx.float32))

print(f"MLX output shape: {mlx_mu_np.shape}")
print(f"MLX mu stats: mean={np.mean(mlx_mu_np):.4f}, std={np.std(mlx_mu_np):.4f}")

# ── Also test single-frame to verify no regression ──
print()
print("=" * 60)
print("Single-frame regression check")
print("=" * 60)

# Re-load HF for single frame
hf_vae2 = AutoencoderKLWan.from_pretrained(
    "weights/Cosmos3-Nano", subfolder="vae",
    torch_dtype=torch.bfloat16,
)
hf_vae2.eval()

single_np = video_np[0]  # [H, W, 3]
single_pt = torch.from_numpy(single_np).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)  # [1, 3, 1, H, W]
single_pt = single_pt * 2.0 - 1.0
single_pt = single_pt.to(torch.bfloat16)

with torch.no_grad():
    hf_single = hf_vae2._encode(single_pt)

hf_single_mu = hf_single[:, :z_dim].permute(0, 2, 3, 4, 1).float().numpy()
del hf_vae2

mlx_single = encode_video(single_np, vae_dir)
mx.eval(mlx_single)
mlx_single_mu = np.array((mlx_single * latents_std_mx + latents_mean_mx).astype(mx.float32))

single_diff = np.abs(hf_single_mu - mlx_single_mu)
hf_flat = hf_single_mu.flatten()
mlx_flat = mlx_single_mu.flatten()
cos_single = np.dot(hf_flat, mlx_flat) / (np.linalg.norm(hf_flat) * np.linalg.norm(mlx_flat) + 1e-8)
print(f"Single-frame: max_diff={single_diff.max():.4f}, cosine={cos_single:.6f}")

# ── Compare multi-frame ──
print()
print("=" * 60)
print("Multi-frame comparison")
print("=" * 60)

print(f"HF shape: {hf_mu_cl.shape}")
print(f"MLX shape: {mlx_mu_np.shape}")

if hf_mu_cl.shape != mlx_mu_np.shape:
    print(f"SHAPE MISMATCH! HF {hf_mu_cl.shape} vs MLX {mlx_mu_np.shape}")
    min_t = min(hf_mu_cl.shape[1], mlx_mu_np.shape[1])
    diff = np.abs(hf_mu_cl[:, :min_t] - mlx_mu_np[:, :min_t])
else:
    diff = np.abs(hf_mu_cl - mlx_mu_np)

print(f"Max abs diff: {diff.max():.6f}")
print(f"Mean abs diff: {diff.mean():.6f}")

if hf_mu_cl.shape == mlx_mu_np.shape:
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
