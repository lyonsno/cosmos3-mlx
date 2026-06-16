"""480p i2v smoke: evil orb turntable at 640x480, 24 frames, with audio."""

import sys
sys.path.insert(0, ".")

import time
import numpy as np
from PIL import Image
from pathlib import Path

import mlx.core as mx

model_dir = "weights/Cosmos3-Nano"
image_path = Path.home() / "dev" / "evil_orb.png"
output_path = "/tmp/cosmos3_i2v_480p_evil_orb.mp4"

img = Image.open(image_path).convert("RGB")
print(f"Input image: {img.size}")

print("Loading model...")
t0 = time.time()
from cosmos3_mlx.load import load_transformer, load_tokenizer
model = load_transformer(model_dir, reasoner_only=False)
tokenizer = load_tokenizer(model_dir)
t1 = time.time()
print(f"  Loaded in {t1-t0:.1f}s")

from cosmos3_mlx.pipeline import Cosmos3GenerationPipeline
pipeline = Cosmos3GenerationPipeline(
    model=model, tokenizer=tokenizer, model_dir=model_dir,
)

height, width = 480, 640
num_frames = 24  # 6 latent frames
print(f"\nGenerating {width}x{height} {num_frames}-frame i2v video (30 steps, with audio)...")

t_gen = time.time()
result = pipeline.generate(
    prompt="A slow smooth turntable rotation of a dark metallic orb with a glowing orange energy core, rotating against a dark background, cinematic lighting, high detail, subtle mechanical humming sounds",
    num_frames=num_frames,
    height=height,
    width=width,
    num_inference_steps=30,
    guidance_scale=6.0,
    seed=42,
    image=np.array(img),
    enable_audio=True,
)
t_gen_done = time.time()
print(f"  Generation: {t_gen_done - t_gen:.1f}s")

latents = result["latents"]
print(f"  Latent shape: {latents.shape}")

# Decode through VAE
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

# Decode audio if present
audio_np = None
if "audio_latents" in result and result["audio_latents"] is not None:
    print("  Decoding audio...")
    try:
        from cosmos3_mlx.audio import AudioDecoder, AudioDecoderConfig
        import json
        audio_cfg = AudioDecoderConfig()
        audio_decoder = AudioDecoder(audio_cfg)
        # Load audio weights
        audio_weights = mx.load(f"{model_dir}/audio_decoder/audio_decoder.safetensors")
        audio_decoder.load_weights(list(audio_weights.items()))
        audio_latents = result["audio_latents"]
        audio_np = audio_decoder.decode(audio_latents)
        mx.eval(audio_np)
        audio_np = np.array(audio_np.astype(mx.float32))
        print(f"  Audio shape: {audio_np.shape}")
    except Exception as e:
        print(f"  Audio decode failed: {e}")

from cosmos3_mlx.pipeline import save_video
save_video(video_np, output_path, fps=25, audio_waveform=audio_np, audio_sample_rate=48000)
print(f"  Saved to {output_path}")

# Save key frames
Image.fromarray(video_np[0]).save("/tmp/i2v_480p_frame0.png")
Image.fromarray(video_np[num_frames // 2]).save("/tmp/i2v_480p_frame_mid.png")
Image.fromarray(video_np[-1]).save("/tmp/i2v_480p_frame_last.png")
print("  Saved frame PNGs")

total = t_dec_done - t0
print(f"\n  Total wall time: {total:.1f}s")
print("=== 480p i2v smoke completed! ===")
