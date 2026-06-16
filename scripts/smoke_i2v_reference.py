"""Run the NVIDIA reference i2v demo: their input image + prompt at 720p, 16 frames."""

import sys
sys.path.insert(0, ".")

import json
import time
import numpy as np
from PIL import Image
from pathlib import Path

import mlx.core as mx

model_dir = "weights/Cosmos3-Nano"

# Load reference prompt
with open(f"{model_dir}/assets/example_i2v_prompt.json") as f:
    prompt_data = json.load(f)
prompt = prompt_data["temporal_caption"]
print(f"Prompt: {prompt[:120]}...")

# Load reference image
img = Image.open(f"{model_dir}/assets/example_i2v_input.jpg").convert("RGB")
print(f"Input image: {img.size}")

# Load model
print("Loading model...")
t0 = time.time()
from cosmos3_mlx.load import load_transformer, load_tokenizer
model = load_transformer(model_dir, reasoner_only=False)
tokenizer = load_tokenizer(model_dir)
t_load = time.time()
print(f"  Loaded in {t_load - t0:.1f}s")

from cosmos3_mlx.pipeline import Cosmos3GenerationPipeline
pipeline = Cosmos3GenerationPipeline(
    model=model, tokenizer=tokenizer, model_dir=model_dir,
)

height, width = 720, 1280
num_frames = 16
print(f"\nGenerating {width}x{height} {num_frames}-frame i2v (30 steps)...")
print(f"Reference prompt from NVIDIA example_i2v_prompt.json")

t_gen = time.time()
result = pipeline.generate(
    prompt=prompt,
    num_frames=num_frames,
    height=height,
    width=width,
    num_inference_steps=30,
    guidance_scale=6.0,
    seed=42,
    image=np.array(img),
)
t_gen_done = time.time()
print(f"  Generation: {t_gen_done - t_gen:.1f}s")

latents = result["latents"]
print(f"  Latent shape: {latents.shape}")

# Decode
print("  Decoding through VAE...")
t_dec = time.time()
from cosmos3_mlx.decode_vae import decode_latents
video = decode_latents(latents, f"{model_dir}/vae")
mx.eval(video)
t_dec_done = time.time()
print(f"  VAE decode: {t_dec_done - t_dec:.1f}s")

video_np = np.array(video[0].astype(mx.float32))
video_np = (video_np * 255).clip(0, 255).astype(np.uint8)
print(f"  Video frames: {video_np.shape}")

from cosmos3_mlx.pipeline import save_video
save_video(video_np, "/tmp/cosmos3_i2v_reference.mp4", fps=24)
print("  Saved /tmp/cosmos3_i2v_reference.mp4")

# Save key frames for inspection
Image.fromarray(video_np[0]).save("/tmp/i2v_ref_frame0.png")
Image.fromarray(video_np[4]).save("/tmp/i2v_ref_frame4.png")
Image.fromarray(video_np[8]).save("/tmp/i2v_ref_frame8.png")
Image.fromarray(video_np[-1]).save("/tmp/i2v_ref_frame_last.png")
print("  Saved frame PNGs")

# Also extract first 16 frames from NVIDIA reference for comparison
import subprocess
subprocess.run([
    "ffmpeg", "-y", "-i", f"{model_dir}/assets/example_i2v_output.mp4",
    "-frames:v", "1", "/tmp/nvidia_ref_frame0.png",
], capture_output=True)
subprocess.run([
    "ffmpeg", "-y", "-i", f"{model_dir}/assets/example_i2v_output.mp4",
    "-vf", f"select=eq(n\\,15)", "-frames:v", "1", "/tmp/nvidia_ref_frame15.png",
], capture_output=True)
print("  Extracted NVIDIA reference frames 0 and 15")

total = t_dec_done - t0
print(f"\n  Total wall time: {total:.1f}s")
print("=== Reference i2v smoke completed! ===")
