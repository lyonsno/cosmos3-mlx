# Probolē: cosmos3-mlx Generation Pipeline Boundary Audit

**Created:** 2026-06-14
**Target:** `lyonsno/cosmos3-mlx` main branch at `fbe0693`
**Review context mode:** Code-only with HF reference comparison
**Motivation:** Text generation works perfectly but image generation produces
semantically incoherent outputs (blobs with some structure, no recognizable
objects). All obvious single-point failures fixed (mRoPE, scheduler, timestep
scale, temporal margin). Need systematic boundary-by-boundary audit.

## Reference

HuggingFace Diffusers Cosmos 3 implementation:
- Pipeline: `diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py`
- Transformer: `diffusers/models/transformers/transformer_cosmos3.py`
- Scheduler: `diffusers/schedulers/scheduling_unipc_multistep.py`
- Available at: https://github.com/huggingface/diffusers/tree/main/src/diffusers/

## Boundaries to audit

### B1: Tokenization and chat template (pipeline.py:139-150)
- Does `apply_chat_template` with `add_generation_prompt=True` produce the
  same token sequence as the HF reference for the conditional pass?
- Does the unconditional (empty prompt) pass produce the same tokens?
- Are there special tokens (e.g., `start_of_generation`) that the reference
  adds but we don't?
- HF ref uses `_add_special_tokens()` after tokenization — what does it add?

### B2: Noise initialization and patchification (pipeline.py:48-89)
- Our noise is `[B, T, H, W, z_dim]` channels-last. HF is `[B, C, T, H, W]`
  channels-first. Is the patchification order correct for channels-last?
- HF patchifies inside the transformer. We patchify in the pipeline. Are the
  patch token orderings identical? (t-major vs h-major vs w-major flattening)
- Does patchification preserve the same channel interleaving?

### B3: Timestep embedding numerical match (timestep.py + model.py:341-345)
- HF uses `Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)` →
  `TimestepEmbedding(256, hidden_size)`. Our code uses a single class.
- For the same timestep value (e.g., 999.0 * 0.001 = 0.999), do we produce
  the same sinusoidal features? Check `flip_sin_to_cos` and frequency formula.
- Are the MLP weight names mapped correctly from HF checkpoint?

### B4: Dual-pathway attention wiring (attention.py:174-224)
- Generation queries attend to concatenated [und_keys, gen_keys].
- Are the understanding keys the correct post-RoPE, post-QK-norm keys?
- Is GQA repeat applied correctly for the concatenated keys?
- Does the reference use the same attention pattern (full, non-causal) for
  generation?

### B5: Velocity → spatial latent conversion (pipeline.py:206-222)
- We unpatchify velocity tokens back to `[B, T, H, W, z_dim]`.
- The scheduler then steps in spatial space: `x0 = sample - sigma * velocity`.
- HF scheduler also steps in spatial space (no patchification in scheduler).
- Is the unpatchify → step → repatchify loop equivalent to the HF approach
  of patchify → transformer → unpatchify within a single spatial-space step?

### B6: Scheduler numerical match (scheduler.py)
- For the same sigma, velocity, and sample values, does our UniPC step
  produce the same output as HF's UniPCMultistepScheduler?
- Are the flow sigma values identical? (linspace from 1 to 1/1000,
  flow_shift=1.0 means no shift)
- Is the second-order correction coefficient correct?

### B7: VAE decode correctness (decode_vae.py)
- For a known latent tensor, does our decoder produce the same output as HF?
- Is the Conv3D decomposition (per-frame 2D) numerically equivalent?
- Are the upsampling block orderings correct?
- Is the unpatchify after conv_out correct?

## What to report

For each boundary, report:
- Whether the MLX code matches the HF reference
- If not, the exact discrepancy with code locations
- Severity: whether the discrepancy could explain semantic incoherence
- Smallest fix or test to verify
