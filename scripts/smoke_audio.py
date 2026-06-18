"""Audio generation smoke test — post latent-norm fix."""

import sys
sys.path.insert(0, ".")

import time
import numpy as np
from pathlib import Path
import mlx.core as mx

model_dir = "weights/Cosmos3-Nano"

print("Loading model...")
t0 = time.time()
from cosmos3_mlx.load import load_transformer, load_tokenizer
model = load_transformer(model_dir, reasoner_only=False)
tokenizer = load_tokenizer(model_dir)
print(f"  Loaded in {time.time()-t0:.1f}s")

from cosmos3_mlx.pipeline import Cosmos3GenerationPipeline, save_video
from cosmos3_mlx.decode_vae import decode_latents
from cosmos3_mlx.decode_audio import decode_audio

pipeline = Cosmos3GenerationPipeline(
    model=model,
    tokenizer=tokenizer,
    model_dir=model_dir,
)

prompt = "A robot arm picks up a red apple from a wooden table in a bright kitchen, with a satisfying thud sound"

out_dir = Path("/tmp/cosmos3_audio_smoke")
out_dir.mkdir(exist_ok=True)

print(f"\nGenerating 256x256 16-frame video WITH audio...")
t_start = time.time()
result = pipeline.generate(
    prompt=prompt,
    num_frames=16,
    height=256,
    width=256,
    num_inference_steps=30,
    guidance_scale=6.0,
    seed=42,
    enable_audio=True,
)
t_gen = time.time() - t_start
print(f"  Generation: {t_gen:.1f}s")

# Check what we got
print(f"\n  Keys: {list(result.keys())}")
latents = result["latents"]
print(f"  Video latent shape: {latents.shape}")

if "audio_latents" in result:
    audio_lat = result["audio_latents"]
    print(f"  Audio latent shape: {audio_lat.shape}")
    print(f"  Audio latent stats: mean={mx.mean(audio_lat).item():.4f}, std={mx.std(audio_lat).item():.4f}")

    # Decode audio
    print("\n  Decoding audio...")
    snd_dir = str(Path(model_dir) / "sound_tokenizer")
    audio_waveform = decode_audio(audio_lat, snd_dir)
    mx.eval(audio_waveform)
    print(f"  Audio waveform shape: {audio_waveform.shape}")
    audio_np = np.array(audio_waveform[0].astype(mx.float32))
    print(f"  Audio stats: min={audio_np.min():.4f}, max={audio_np.max():.4f}, mean={audio_np.mean():.4f}")
    print(f"  Audio duration: {audio_np.shape[1] / 48000:.2f}s at 48kHz")

    # Decode video
    print("\n  Decoding video...")
    vae_dir = str(Path(model_dir) / "vae")
    video = decode_latents(latents, vae_dir)
    mx.eval(video)
    video_np = np.array(video[0].astype(mx.float32))
    video_np = (video_np * 255).clip(0, 255).astype(np.uint8)

    # Save with audio
    save_video(video_np, str(out_dir / "output_with_audio.mp4"), fps=25,
               audio_waveform=audio_np, audio_sample_rate=48000)

    # Also save audio-only WAV for inspection
    import wave
    wav_path = str(out_dir / "audio_only.wav")
    audio_int16 = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(wav_path, 'w') as wf:
        wf.setnchannels(audio_np.shape[0])  # stereo
        wf.setsampwidth(2)
        wf.setframerate(48000)
        # Interleave channels for WAV: [2, N] -> [N, 2] -> flat
        interleaved = audio_int16.T.flatten()
        wf.writeframes(interleaved.tobytes())
    print(f"  Saved WAV: {wav_path}")

    from PIL import Image
    Image.fromarray(video_np[0]).save(str(out_dir / "frame_00.png"))
    Image.fromarray(video_np[-1]).save(str(out_dir / "frame_last.png"))

    print(f"\n  Saved to {out_dir}/")
else:
    print("\n  NO audio latents in result! Audio generation may not be wired.")

print(f"\n=== Audio smoke done ===")
