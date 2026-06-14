# Probolē: cosmos3-mlx Phase 2 Generation Pipeline Review

**Created:** 2026-06-14
**Target:** `lyonsno/cosmos3-mlx` main branch
**Range:** `c906136..85f5bb7` (6 commits, ~1500 LOC added)
**Review context mode:** Code-only, no inherited implementation thread context

## Scope

Review the Phase 2 generation pipeline implementation:
- `cosmos3_mlx/vae.py` — AutoencoderKLWan (Wan2.2 VAE) decoder
- `cosmos3_mlx/audio.py` — Cosmos3 Audio VAE decoder (Oobleck architecture)
- `cosmos3_mlx/decode_vae.py` — VAE decode utilities
- `cosmos3_mlx/pipeline.py` — Full generation pipeline (text → noise → denoise → VAE → pixels)
- `cosmos3_mlx/model.py` — Dual-pathway generation forward pass changes
- `cosmos3_mlx/load.py` — Weight loading updates
- `tests/test_vae.py`, `tests/test_audio.py`, `tests/test_pipeline.py` — Test coverage
- `pyproject.toml` — Dependency updates

## Review focus

1. **Correctness of MoT dual-pathway attention:** Understanding vs generation attention paths should be correctly separated; generation path uses full (non-causal) attention to concatenated understanding+generation KVs
2. **VAE decoder fidelity:** Weight loading from HF safetensors, Conv3D decomposition via CausalConv3d, correct spatial/temporal upsampling
3. **CFG implementation:** Conditional + unconditional forward passes, guidance scale blending
4. **Scheduler:** UniPC or Euler step correctness, noise schedule, timestep handling
5. **Memory safety:** 16B model at BF16 on 128GB — any OOM risks in generation pipeline?
6. **Test quality:** Are tests actually testing meaningful invariants, or are they tautological?
7. **Weight conversion correctness:** Any name mapping gaps between HF and MLX weight names?

## Out of scope

- Phase 1 AR reasoner (already reviewed in passes 1 and 2)
- Image quality tuning (known to need scheduler upgrade, mRoPE fix, CFG tuning)
- Quantization
- Audio generation wiring
