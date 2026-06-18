# cosmos3-mlx

NVIDIA Cosmos 3 running natively on Apple Silicon via MLX.

## What is Cosmos 3?

[NVIDIA Cosmos 3](https://github.com/NVIDIA/cosmos) is an omnimodal world foundation model for physical AI. It jointly handles text, image, video, audio, and physical actions in a single Mixture-of-Transformers (MoT) architecture. Designed for robotics, autonomous driving, and smart space simulation. Released June 2026 under the OpenMDW 1.1 license.

This project ports the Cosmos3-Nano (16B) model to run on Mac using Apple's [MLX](https://github.com/ml-explore/mlx) framework — no NVIDIA GPU required.

## What works

- **Text-to-video generation** — 256p, 480p, 720p at 16–128+ frames
- **Image-to-video generation** — condition on a first frame, generate forward motion
- **Text-to-image generation** — single-frame output
- **Audio generation** — joint video+audio denoising with Oobleck decoder
- **Text KV caching** — up to 9.8× speedup (text tokens are constant across denoising steps)
- **Full VAE pipeline** — encoder (chunked multi-frame with feat_cache) and decoder, both numerically verified against HuggingFace PyTorch reference

### Performance (M4 Max, 128GB)

| Resolution | Frames | Time | Notes |
|------------|--------|------|-------|
| 256×256 | 16 | ~38s | With text KV cache |
| 256×256 | 32 | ~131s | Longer video |
| 480p (832×480) | 16 | ~252s | |
| 720p (1280×720) | 16 | ~591s | |

### Numerical parity with HuggingFace reference

Every component has been verified against the HF PyTorch implementation:

- **VAE decoder:** max pixel diff 0.000016, PSNR 122 dB
- **VAE encoder (single-frame):** cosine 0.9998
- **VAE encoder (chunked multi-frame):** cosine 0.9999
- **Scheduler (UniPC):** max diff 0.0000019 across 35 steps
- **Transformer (t2v):** cosine 0.99992 (256p), 0.99984 (720p)
- **Transformer (i2v):** cosine 0.99981–0.99990 per frame (720p)

### Important: training distribution

Cosmos 3 is a **physical AI** model, not a general-purpose video generator. It was trained primarily on robotics manipulation, autonomous driving, and industrial/factory environments. It produces strong physical motion for on-distribution inputs (dashcam driving, robot arms, factory floors) but does not generalize well to arbitrary creative prompts or subjects outside this domain.

## Architecture

Cosmos 3 uses a Mixture-of-Transformers design with two pathways sharing the same backbone:

- **Understanding (reasoner):** Causal self-attention for text/vision comprehension (Qwen3-VL text backbone)
- **Generation (diffuser):** Full bidirectional attention for image/video/audio synthesis

Video VAE: Wan2.2 AutoencoderKL (8× spatial, 4× temporal downsampling).
Audio: Cosmos3 AVAEAudioTokenizer (Oobleck decoder, stereo 48kHz).

### Key specs (Cosmos3-Nano)

| Parameter | Value |
|-----------|-------|
| Parameters | 16B |
| Hidden size | 4096 |
| Layers | 36 |
| Attention heads | 32 (8 KV heads, GQA) |
| Head dim | 128 |
| FFN intermediate | 12288 |
| Vocab size | 151936 |
| Max context | 262144 (256K) |
| Position encoding | 3D mRoPE (interleaved) |

## Requirements

- Apple Silicon Mac (M1 or later, 32GB+ recommended)
- Python 3.10+
- MLX 0.24+

## Install

```bash
git clone https://github.com/lyonsno/cosmos3-mlx.git
cd cosmos3-mlx
uv venv && uv pip install -e ".[dev]"
```

## Usage

Download the model weights (~32GB at bf16):

```bash
huggingface-cli download nvidia/Cosmos3-Nano --local-dir weights/Cosmos3-Nano
```

### Text-to-video

```python
from cosmos3_mlx.load import load_transformer, load_tokenizer
from cosmos3_mlx.pipeline import Cosmos3GenerationPipeline

model = load_transformer("weights/Cosmos3-Nano", reasoner_only=False)
tokenizer = load_tokenizer("weights/Cosmos3-Nano")

pipeline = Cosmos3GenerationPipeline(
    model=model,
    tokenizer=tokenizer,
    model_dir="weights/Cosmos3-Nano",
)

result = pipeline.generate(
    prompt="A car driving through a suburban intersection on a sunny day",
    num_frames=16,
    height=256,
    width=256,
    num_inference_steps=30,
    guidance_scale=6.0,
    seed=42,
)

# result["video"] contains decoded frames [T, H, W, 3] in [0, 1]
```

### Image-to-video

```python
import numpy as np
from PIL import Image

img = np.array(Image.open("first_frame.jpg").convert("RGB"))

result = pipeline.generate(
    prompt="A car driving forward along a winding coastal road",
    num_frames=16,
    height=256,
    width=256,
    num_inference_steps=30,
    guidance_scale=6.0,
    seed=42,
    image=img,
)
```

### Save as MP4

```python
from cosmos3_mlx.pipeline import save_video

video_np = (np.array(result["video"]) * 255).clip(0, 255).astype(np.uint8)
save_video(video_np, "output.mp4", fps=25)
```

## Test

```bash
python -m pytest tests/ -v  # 105 tests
```

## License

MIT (this port). Model weights are under [NVIDIA OpenMDW 1.1](https://developer.nvidia.com/cosmos/license).
