"""Quick i2v smoke test at small resolution."""

import sys
sys.path.insert(0, ".")

import time
import numpy as np
from PIL import Image

model_dir = "weights/Cosmos3-Nano"

# Load the conditioning image
img = Image.open(f"{model_dir}/assets/example_i2v_input.jpg").convert("RGB")
print(f"Input image: {img.size}")

# Load model
print("Loading model...")
t0 = time.time()
from cosmos3_mlx.load import load_transformer, load_tokenizer
model = load_transformer(model_dir, reasoner_only=False)
tokenizer = load_tokenizer(model_dir)
t1 = time.time()
print(f"  Loaded in {t1-t0:.1f}s")

# Create pipeline (no VAE decoder for now — just test the conditioning loop)
from cosmos3_mlx.pipeline import Cosmos3GenerationPipeline

pipeline = Cosmos3GenerationPipeline(
    model=model,
    tokenizer=tokenizer,
    model_dir=model_dir,
)

# Generate: 16 frames at 256x256 with i2v conditioning
print("\nGenerating 256x256 16-frame video with i2v conditioning...")
result = pipeline.generate(
    prompt="A car driving along a coastal mountain road with falling rocks causing an emergency stop",
    num_frames=16,
    height=256,
    width=256,
    num_inference_steps=10,  # Quick smoke
    guidance_scale=6.0,
    seed=42,
    image=np.array(img),
)

latents = result["latents"]
print(f"\n  Output latent shape: {latents.shape}")

import mlx.core as mx
print(f"  Frame 0 (conditioned) stats: mean={mx.mean(latents[:, 0]).item():.4f}, std={mx.std(latents[:, 0]).item():.4f}")
print(f"  Frame 1 (denoised) stats: mean={mx.mean(latents[:, 1]).item():.4f}, std={mx.std(latents[:, 1]).item():.4f}")
print(f"  Frame 2 (denoised) stats: mean={mx.mean(latents[:, 2]).item():.4f}, std={mx.std(latents[:, 2]).item():.4f}")
print(f"  Frame 3 (denoised) stats: mean={mx.mean(latents[:, 3]).item():.4f}, std={mx.std(latents[:, 3]).item():.4f}")

# Check that frame 0 is different from the noise-generated frames
# (it should be close to the encoded image latent)
frame0_std = mx.std(latents[:, 0]).item()
frame1_std = mx.std(latents[:, 1]).item()
print(f"\n  Frame 0 vs frame 1 std ratio: {frame0_std / frame1_std:.3f}")
print("  (Should be different if conditioning is working)")

print("\n=== i2v smoke test completed! ===")
