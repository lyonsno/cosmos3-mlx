# Probolē: cosmos3-mlx Phase 2 diffusion components review

**Date:** 2026-06-14
**Target:** `lyonsno/cosmos3-mlx` — Phase 2 diffusion components
**Worktree:** `/private/tmp/cosmos3-mlx-initial-scaffold-0614`
**Branch:** `cc/initial-scaffold-0614` (landed on `main`)
**Review context mode:** target code only, no inherited implementation thread

## Target range

Phase 2 diffusion generation components only:
- `cosmos3_mlx/scheduler.py` — UniPC rectified flow scheduler
- `cosmos3_mlx/timestep.py` — Sinusoidal timestep embedding + masked application
- `cosmos3_mlx/conv3d.py` — CausalConv3d decomposed into per-frame 2D convolutions
- `tests/test_diffusion.py` — Scheduler and timestep tests
- `tests/test_conv3d.py` — CausalConv3d tests

## Review scope

1. **Scheduler correctness:** Does the UniPC scheduler correctly implement rectified flow? Is the velocity prediction step formula `x_{t-1} = x_t + (t_{i-1} - t_i) * v` correct? Are the noise interpolation endpoints right (t=0 is clean, t=1 is noise)?

2. **Conv3D decomposition correctness:** Is the per-frame 2D conv accumulation mathematically equivalent to a full 3D convolution? Is the causal temporal padding correct (2×pad left, 0 right)? Does the temporal cache produce identical results to full-sequence processing?

3. **Timestep embedding:** Is the sinusoidal embedding standard? Does the masked application correctly leave clean tokens unchanged?

4. **Test quality:** Are the tests catching real bugs? Is the causal padding test actually verifying causality? Is the cache consistency test meaningful?

5. **MLX idioms:** Correct use of MLX ops? Any issues with the conv2d decomposition approach?

## What is NOT in scope

- Phase 1 components (already reviewed in passes 1 and 2)
- VAE decoder (not yet implemented)
- Audio decoder (not yet implemented)
- Performance optimization
