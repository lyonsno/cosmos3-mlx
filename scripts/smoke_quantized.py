"""Quick quantization viability test: does the model still produce coherent video at 4-bit and 8-bit?"""

import sys
sys.path.insert(0, ".")

import time
import numpy as np
from PIL import Image
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn

model_dir = "weights/Cosmos3-Nano"
vae_dir = str(Path(model_dir) / "vae")
image_path = f"{model_dir}/assets/example_i2v_input.jpg"

img = Image.open(image_path).convert("RGB")

bits = int(sys.argv[1]) if len(sys.argv) > 1 else 4
group_size = int(sys.argv[2]) if len(sys.argv) > 2 else 64

print(f"=== Quantization test: {bits}-bit, group_size={group_size} ===")
print(f"Input image: {img.size}")

# Load model at bf16
print("Loading model (bf16)...")
t0 = time.time()
from cosmos3_mlx.load import load_transformer, load_tokenizer
model = load_transformer(model_dir, reasoner_only=False)
tokenizer = load_tokenizer(model_dir)
t_load = time.time() - t0
print(f"  Loaded in {t_load:.1f}s")

# Count params before quantization
from mlx.utils import tree_flatten
total_params = sum(p.size for _, p in tree_flatten(model.parameters()))
print(f"  Parameters: {total_params:,}")

# Quantize
print(f"  Quantizing to {bits}-bit...")
t_q = time.time()
nn.quantize(model, bits=bits, group_size=group_size)
mx.eval(model.parameters())
t_quant = time.time() - t_q
print(f"  Quantized in {t_quant:.1f}s")

# Estimate memory
bits_per_param = bits + 32 / group_size  # weight bits + scale bits
est_gb = total_params * bits_per_param / 8 / 1e9
print(f"  Estimated model size: ~{est_gb:.1f} GB")

# Create pipeline
from cosmos3_mlx.pipeline import Cosmos3GenerationPipeline, save_video
from cosmos3_mlx.decode_vae import decode_latents

pipeline = Cosmos3GenerationPipeline(
    model=model,
    tokenizer=tokenizer,
    model_dir=model_dir,
)

prompt = "A car driving along a winding coastal mountain road, approaching a left turn, with ocean visible in the distance and rocky cliffs on the right"

out_dir = Path(f"/tmp/cosmos3_q{bits}_g{group_size}")
out_dir.mkdir(exist_ok=True)

print(f"\nGenerating 256x256 16-frame i2v...")
t_start = time.time()
result = pipeline.generate(
    prompt=prompt,
    num_frames=16,
    height=256,
    width=256,
    num_inference_steps=30,
    guidance_scale=6.0,
    seed=42,
    image=np.array(img),
)
t_gen = time.time() - t_start
print(f"  Generation: {t_gen:.1f}s")

latents = result["latents"]
print(f"  Latent stats: mean={mx.mean(latents).item():.4f}, std={mx.std(latents).item():.4f}")

# Decode
t_dec = time.time()
video = decode_latents(latents, vae_dir)
mx.eval(video)
t_dec_done = time.time() - t_dec
print(f"  VAE decode: {t_dec_done:.1f}s")

video_np = np.array(video[0].astype(mx.float32))
video_np = (video_np * 255).clip(0, 255).astype(np.uint8)

save_video(video_np, str(out_dir / "output.mp4"), fps=25)
for i in [0, 1, len(video_np)//2, len(video_np)-1]:
    Image.fromarray(video_np[i]).save(str(out_dir / f"frame_{i:02d}.png"))

print(f"  Saved to {out_dir}/")
print(f"  Total: {t_gen + t_dec_done:.1f}s")
print(f"\n=== {bits}-bit quantization test done ===")
