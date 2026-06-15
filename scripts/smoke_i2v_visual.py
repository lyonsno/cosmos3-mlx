"""Full i2v visual smoke: encode conditioning image, denoise, VAE decode, save MP4."""

import sys
sys.path.insert(0, ".")

import time
import numpy as np
from PIL import Image
from pathlib import Path

import mlx.core as mx

model_dir = "weights/Cosmos3-Nano"
image_path = Path.home() / "dev" / "evil_orb.png"
output_path = "/tmp/cosmos3_i2v_evil_orb.mp4"

# Load the conditioning image (convert RGBA→RGB)
img = Image.open(image_path).convert("RGB")
print(f"Input image: {img.size}")

# Load model
print("Loading model...")
t0 = time.time()
from cosmos3_mlx.load import load_transformer, load_tokenizer
model = load_transformer(model_dir, reasoner_only=False)
tokenizer = load_tokenizer(model_dir)
t1 = time.time()
print(f"  Loaded in {t1-t0:.1f}s")

# Create pipeline with model_dir for VAE access
from cosmos3_mlx.pipeline import Cosmos3GenerationPipeline

pipeline = Cosmos3GenerationPipeline(
    model=model,
    tokenizer=tokenizer,
    model_dir=model_dir,
)

# Generate: 16 frames at 256x256 with i2v conditioning — turntable prompt
height, width = 256, 256
num_frames = 16
print(f"\nGenerating {width}x{height} {num_frames}-frame i2v video...")
print("  Prompt: turntable rotation of the evil orb")

result = pipeline.generate(
    prompt="A slow smooth turntable rotation of a dark metallic orb with glowing orange energy core, rotating against a dark background, cinematic lighting",
    num_frames=num_frames,
    height=height,
    width=width,
    num_inference_steps=30,
    guidance_scale=6.0,
    seed=42,
    image=np.array(img),
)

latents = result["latents"]
print(f"\n  Output latent shape: {latents.shape}")
print(f"  Frame 0 stats: mean={mx.mean(latents[:, 0]).item():.4f}, std={mx.std(latents[:, 0]).item():.4f}")
print(f"  Frame 1 stats: mean={mx.mean(latents[:, 1]).item():.4f}, std={mx.std(latents[:, 1]).item():.4f}")

# Decode through VAE
print("\n  Decoding through VAE...")
t_dec = time.time()
from cosmos3_mlx.decode_vae import decode_latents

vae_dir = str(Path(model_dir) / "vae")
video = decode_latents(latents, vae_dir)
mx.eval(video)
t_dec_done = time.time()
print(f"  VAE decode: {t_dec_done - t_dec:.1f}s")

# Convert to uint8 frames
video_np = np.array(video[0].astype(mx.float32))  # [T, H, W, 3] in [0, 1]
video_np = (video_np * 255).clip(0, 255).astype(np.uint8)
print(f"  Video frames: {video_np.shape}")

# Save as MP4
from cosmos3_mlx.pipeline import save_video
save_video(video_np, output_path, fps=25)
print(f"\n  Saved to {output_path}")

# Also save frame 0 and frame 1 as PNGs for quick visual check
Image.fromarray(video_np[0]).save("/tmp/cosmos3_i2v_frame0.png")
Image.fromarray(video_np[1]).save("/tmp/cosmos3_i2v_frame1.png")
Image.fromarray(video_np[-1]).save("/tmp/cosmos3_i2v_frame_last.png")
print("  Saved frame PNGs: /tmp/cosmos3_i2v_frame0.png, frame1.png, frame_last.png")

print("\n=== i2v visual smoke completed! ===")
