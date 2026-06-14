# cosmos3-mlx

NVIDIA Cosmos 3 running natively on Apple Silicon via MLX.

## What is Cosmos 3?

[NVIDIA Cosmos 3](https://github.com/NVIDIA/cosmos) is an omnimodal world foundation model that jointly handles text, image, video, audio, and physical actions in a single Mixture-of-Transformers (MoT) architecture. Released June 2026 under the OpenMDW 1.1 license.

This project ports the Cosmos3-Nano (16B) model to run on Mac using Apple's [MLX](https://github.com/ml-explore/mlx) framework — no NVIDIA GPU required.

## Status

**Work in progress.** Phase 1: AR reasoner (text + vision understanding).

- [x] 3D multi-dimensional rotary position embeddings (mRoPE)
- [x] Dual-pathway MoT attention with GQA (32 heads / 8 KV heads)
- [x] Full transformer backbone (36 layers, 4096 hidden, RMSNorm, GLU/SiLU FFN)
- [x] Autoregressive generation with KV cache
- [x] Test suite (20 tests passing)
- [ ] Weight conversion from HuggingFace safetensors
- [ ] Qwen3-VL vision encoder
- [ ] End-to-end inference (image in → text out)
- [ ] Quantization (4-bit, 8-bit)

## Architecture

Cosmos 3 uses a Mixture-of-Transformers design with two pathways sharing the same backbone:

- **Understanding (reasoner):** Causal self-attention for text/vision comprehension
- **Generation (diffuser):** Full attention for image/video/audio/action synthesis

Phase 1 implements the understanding pathway. The generation pathway is defined but not yet wired for inference.

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
| Text backbone | Qwen3-VL derived |

## Requirements

- Apple Silicon Mac (M1 or later)
- Python 3.10+
- MLX 0.24+

## Install

```bash
git clone https://github.com/lyonsno/cosmos3-mlx.git
cd cosmos3-mlx
uv venv && uv pip install -e ".[dev]"
```

## Test

```bash
python -m pytest tests/ -v
```

## License

MIT
