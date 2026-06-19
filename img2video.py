"""Image-to-video generation with Cosmos 3 Nano on MLX."""

import argparse
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from cosmos3_mlx.load import load_transformer, load_tokenizer
from cosmos3_mlx.pipeline import Cosmos3GenerationPipeline, save_video
from cosmos3_mlx.decode_vae import decode_latents


def main():
    parser = argparse.ArgumentParser(description="Cosmos 3 Nano image-to-video generation")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--image", type=str, required=True, help="Conditioning image path")
    parser.add_argument("--model-dir", type=str, default="weights/Cosmos3-Nano",
                        help="Path to Cosmos3-Nano weights directory")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output path")
    parser.add_argument("--height", type=int, default=256, help="Output height")
    parser.add_argument("--width", type=int, default=256, help="Output width")
    parser.add_argument("--num-frames", type=int, default=16, help="Number of frames")
    parser.add_argument("--steps", type=int, default=30, help="Denoising steps")
    parser.add_argument("--guidance", type=float, default=6.0, help="CFG scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--quantize", type=int, choices=[4, 8], default=None,
                        help="Quantize model to N bits (reduces memory)")
    parser.add_argument("--fps", type=int, default=25, help="Output video FPS")
    parser.add_argument("--enable-audio", action="store_true", help="Generate audio")
    args = parser.parse_args()

    img = np.array(Image.open(args.image).convert("RGB"))
    print(f"Input image: {args.image} ({img.shape[1]}x{img.shape[0]})")

    print(f"Loading model from {args.model_dir}...")
    t0 = time.time()
    model = load_transformer(args.model_dir, reasoner_only=False)
    tokenizer = load_tokenizer(args.model_dir)

    if args.quantize:
        print(f"Quantizing to {args.quantize}-bit...")
        nn.quantize(model, bits=args.quantize, group_size=64)
        mx.eval(model.parameters())

    print(f"  Loaded in {time.time()-t0:.1f}s")

    pipeline = Cosmos3GenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        model_dir=args.model_dir,
    )

    print(f"\nGenerating {args.width}x{args.height} {args.num_frames}-frame i2v...")
    print(f"  Prompt: {args.prompt}")
    t_start = time.time()

    result = pipeline.generate(
        prompt=args.prompt,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        image=img,
        enable_audio=args.enable_audio,
    )
    t_gen = time.time() - t_start
    print(f"  Generation: {t_gen:.1f}s")

    # Decode video
    vae_dir = str(Path(args.model_dir) / "vae")
    video = decode_latents(result["latents"], vae_dir)
    mx.eval(video)
    video_np = np.array(video[0].astype(mx.float32))
    video_np = (video_np * 255).clip(0, 255).astype(np.uint8)

    # Handle audio if generated
    audio_np = None
    if "audio_latents" in result:
        from cosmos3_mlx.decode_audio import decode_audio
        snd_dir = str(Path(args.model_dir) / "sound_tokenizer")
        audio_waveform = decode_audio(result["audio_latents"], snd_dir)
        mx.eval(audio_waveform)
        audio_np = np.array(audio_waveform[0].astype(mx.float32))

    save_video(video_np, args.output, fps=args.fps, audio_waveform=audio_np)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
