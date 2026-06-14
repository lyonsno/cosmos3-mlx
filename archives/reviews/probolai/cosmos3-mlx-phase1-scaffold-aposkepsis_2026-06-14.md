# Probolē: cosmos3-mlx Phase 1 scaffold review

**Date:** 2026-06-14
**Target:** `lyonsno/cosmos3-mlx` — MLX port of NVIDIA Cosmos 3 Nano
**Worktree:** `/private/tmp/cosmos3-mlx-initial-scaffold-0614`
**Branch:** `cc/initial-scaffold-0614` (landed on `main`)
**Review context mode:** target code only, no inherited implementation thread

## Target range

All source files in `cosmos3_mlx/` and `tests/`:
- `cosmos3_mlx/rope.py` — 3D mRoPE implementation
- `cosmos3_mlx/attention.py` — dual-pathway MoT attention with GQA
- `cosmos3_mlx/model.py` — full transformer model + generation
- `cosmos3_mlx/convert.py` — weight conversion (HF safetensors → MLX)
- `cosmos3_mlx/vision.py` — Qwen3-VL vision encoder
- `cosmos3_mlx/load.py` — config + weight loading from HF directory
- `cosmos3_mlx/generate.py` — end-to-end inference pipeline
- `tests/test_rope.py`, `tests/test_attention.py`, `tests/test_model.py`, `tests/test_convert.py`, `tests/test_vision.py`, `tests/test_load.py`

## Review scope

1. **Correctness:** Does the architecture match the reference Cosmos3OmniTransformer from HuggingFace diffusers? Are there bugs in the attention, RoPE, model, or vision encoder that would cause wrong outputs when real weights are loaded?
2. **Weight compatibility:** Will the weight name mapping and loading work with the actual HuggingFace Cosmos3-Nano safetensors? Are there missing or mismatched weight names?
3. **Test quality:** Are the tests actually testing meaningful invariants, or are they trivially passing? Are there missing test cases for critical paths?
4. **MLX idioms:** Is the code using MLX correctly and efficiently? Any anti-patterns?
5. **Architectural risks:** Anything that will obviously break when real weights are loaded and real inference is attempted?

## What is NOT in scope

- Phase 2 generation pathway (diffusion, VAE, audio, actions)
- Performance optimization
- Quantization
- Documentation quality
