# Anaphora: cosmos3-mlx Phase 2 Generation Pipeline

**Reviewer:** Claude Opus 4.6 (1M context), fresh Aposkepsis agent
**Date:** 2026-06-14
**Probole:** `archives/reviews/probolai/cosmos3-mlx-phase2-generation-pipeline-aposkepsis_2026-06-14.md`
**Range:** `c906136..85f5bb7` (6 commits, ~1500 LOC)
**Review context mode:** fresh (code-only, no implementation thread context)
**Independence:** Satisfied. No inherited implementation-thread narrative.

---

## Summary

The Phase 2 diff adds a generation pipeline (text to denoised latents to pixels), VAE decoder (Wan2.2), audio decoder (Oobleck), dual-pathway MoT forward pass, direct-from-HF-weights VAE decode utility, and tests. The architecture is structurally sound. The dual-pathway attention wiring, scheduler step logic, and VAE decomposition are correct in principle. There are material correctness bugs in the VAE denormalization and weight conversion, and several important gaps.

---

## Findings

### P1 — Material

#### F1. VAE latent denormalization is inverted (`decode_vae.py:155-157`)

```python
inv_std = 1.0 / std
z = latents / inv_std + mean
```

This computes `z = latents * std + mean`, which is the correct denormalization formula for `latents = (z - mean) / std`. However, the variable name `inv_std` is misleading and the double-negation makes it fragile. More critically, the HuggingFace diffusers `AutoencoderKLWan` uses `latents = (latents - mean) * (1/std)` for normalization in its `encode` path, meaning denormalization should be `z = latents * std + mean`. The math happens to be correct by accident of the double inversion, but the code reads as wrong and will break silently if anyone "fixes" the naming. The naming should be clarified and a comment should document the normalization convention.

**Severity:** P1 — the code produces correct output today but is a high-probability silent regression target.
**Smallest test:** Unit test that denormalizes a known vector with known mean/std and asserts the output matches `latents * std + mean`.

#### F2. `strict=False` on transformer weight loading silently drops mismatched weights (`load.py:104`)

Changing `load_weights` from strict to non-strict means any weight name mismatch (typo in `convert.py`, new weight added to model but not present in checkpoint, or vice versa) will silently succeed and leave the parameter at random initialization. The comment says this is for `action_proj_in.fc` which has a "non-standard structure" — but the proper fix is to handle that mapping in `convert.py` or explicitly list the expected missing keys, not to disable all strictness.

For a 16B model, a silently missing weight at random init will produce garbage that might not be immediately obvious (especially for generation-pathway weights that were previously unused in Phase 1).

**Severity:** P1 — silent correctness failure mode.
**Smallest test:** After loading with `reasoner_only=False`, enumerate model parameters and assert none remain at their random init (e.g., check that proj_in, proj_out, time_embedder weights differ from the nn.Linear default init, or compare a checksum against expected values).

#### F3. Position IDs for generation tokens use a continuous sequence with text (`model.py:340-342`)

```python
total_len = text_len + num_patches
pos = mx.arange(total_len)[None, :]
position_ids = mx.stack([pos, pos, pos])
```

The generation pathway position IDs are a flat `[0, 1, 2, ..., text_len + num_patches - 1]`. This means generation tokens get position IDs that continue from the text sequence. However, the Cosmos 3 architecture uses multimodal RoPE (mRoPE) with 3 axes: the first axis is temporal, the second is spatial-H, the third is spatial-W. For generation tokens (which represent a 3D latent grid), all three axes should carry the corresponding T/H/W coordinates, not a flat 1D sequence position.

The understanding pathway uses the same flat positions and that is correct for text. But the generation pathway receiving flat 1D positions for a spatiotemporal grid means mRoPE is not providing the intended spatial/temporal encoding. The attention will still work (RoPE just provides relative position bias), but it will be a degenerate 1D positional encoding rather than the 3D encoding the model was trained with. This will materially degrade generation quality and may explain any downstream image quality issues.

**Severity:** P1 — affects model output quality. The model was trained with 3D position IDs for generation tokens.
**Smallest test:** Assert that generation position_ids have shape `[3, batch, total_len]` where the three axes carry distinct T/H/W coordinates for the generation token segment, not identical flat sequences.

### P2 — Important

#### F4. No `mx.eval` between conditional and unconditional forward passes in CFG (`pipeline.py:172-180`)

Each `diffusion_forward` call runs the full 36-layer transformer. With CFG, two full forward passes happen back-to-back without an intermediate `mx.eval`. For a 16B BF16 model, this means the computation graph for both passes is held in memory simultaneously before any evaluation. At ~32GB for model weights alone, plus two full activation graphs, this could approach or exceed 128GB on a Mac with unified memory.

**Severity:** P2 — potential OOM on target hardware (128GB Mac). Adding `mx.eval(cond_velocity)` before the unconditional pass would halve peak graph memory.
**Smallest test:** Memory profiling test with the full-size model, or at minimum a comment/TODO acknowledging this.

#### F5. `WanDecoder` upsampling block order vs. `decode_vae.py` direct decode differ in refinement conv placement

In `vae.py` `WanUpBlock.__call__`, the order is: residual blocks -> upsample -> conv_after_up. In `decode_vae.py`, the order is: residual blocks -> upsample (no post-upsample conv). The `decode_vae.py` path also applies the upsample after all resnets within a block rather than interleaving. These are two different decode paths and they will produce different outputs from the same weights.

If `decode_vae.py` is the path intended for real inference (loading HF weights directly), and `vae.py` is the structured model class intended to eventually be weight-compatible, they need to agree on architecture. Currently `decode_vae.py` does not apply a post-upsample convolution, while `vae.py` adds one. This means `vae.py` has weights (`conv_after_up`) that have no counterpart in the HF checkpoint and will remain at random init.

**Severity:** P2 — the structured `WanDecoder` class cannot produce correct results from HF weights as-is.
**Smallest test:** Load HF VAE weights into `WanDecoder`, compare output against `decode_vae.py` for the same input latent. They should match.

#### F6. Audio decoder dilated convolution is O(T * K) with Python loops (`audio.py:108-121`)

The manual dilated convolution implementation constructs patches with a Python for-loop over output timesteps and kernel positions. For audio with tens of thousands of timesteps, this will be extremely slow. Each iteration creates a new MLX array, and `out_len` can be large (e.g., 48000 samples / stride for intermediate representations).

**Severity:** P2 — performance, not correctness. Audio decode will be impractically slow at real sequence lengths.
**Smallest test:** Benchmark `AudioResidualUnit` with `dilation=3` at sequence length 4800 (1 second of intermediate audio). If it takes more than a few seconds, the loop approach is not viable.

#### F7. Pipeline hardcodes `patch_latent_dim = 192` for proj_in/proj_out (`model.py:198-199`)

```python
self.proj_in = nn.Linear(192, config.hidden_size, bias=True)
self.proj_out = nn.Linear(config.hidden_size, 192, bias=True)
```

The value 192 = 48 * 2 * 2 is correct for the default Cosmos3-Nano config (z_dim=48, patch_size=2). But this is hardcoded in the model constructor rather than derived from a config field. If anyone changes `z_dim` or `patch_size` (as the tests do for small configs), the proj_in/proj_out dimensions will be wrong. The test fixture in `test_pipeline.py` works around this by manually replacing `model.proj_in` and `model.proj_out` (line 49-50), which shows the hardcoding is already a friction point.

**Severity:** P2 — config/hardcode inconsistency. Not a bug at production scale but makes testing fragile.
**Smallest test:** Already partially covered by the test fixture workaround, but a proper fix would derive from config.

#### F8. Understanding pathway attention in `_generation_forward` uses un-RoPE'd keys for generation cross-attention (`attention.py:178-180, 203-205`)

The understanding keys (`und_keys`) passed to `_generation_forward` are the post-RoPE keys from the understanding pathway. The generation keys are also post-RoPE. When concatenated as `k_full = concat([und_keys, k_gen])`, the generation queries attend to understanding keys with their RoPE-encoded positions.

This is architecturally correct for the MoT design: generation tokens see understanding tokens at their text positions. However, there is a subtle issue: the understanding keys were computed with `und_position_ids` (covering only `[:und_len]`), while generation tokens use `gen_position_ids` (covering `[und_len:total_len]`). The RoPE relative positions between generation queries and understanding keys will correctly encode the cross-modal offset. This finding is downgraded from the initial concern; the wiring appears correct.

**Severity:** Informational (initially suspected P2, verified correct on closer inspection).

### P3 — Minor

#### F9. `_prepare_noise_latents` temporal compression factor is hardcoded to 4 (`pipeline.py:63`)

```python
t_lat = max(1, num_frames // 4)  # 4x temporal compression
```

The VAE config has `temporal_upsample` which determines temporal compression, but the pipeline hardcodes `4`. For `num_frames=1`, this gives `t_lat=1`, but for `num_frames=2` or `3` it gives `0` which is clamped to `1`, meaning 1-3 frames all produce the same latent shape.

**Severity:** P3 — not a bug for the current single-image use case, but will be wrong for video.

#### F10. `test_deterministic_with_seed` may be flaky (`test_pipeline.py:113-126`)

The test sets `mx.random.seed(42)` via the `generate` method, but any MLX random state mutation between the two calls (from model initialization, parameter init, etc.) could cause divergence. This depends on MLX's global RNG state semantics.

**Severity:** P3 — test robustness.

#### F11. Audio decoder test uses approximate shape assertion (`test_audio.py:103`)

```python
assert 390 <= out.shape[2] <= 410
```

The expected output length for `strides=[2,4,5]` with input length 10 should be exactly deterministic given the ConvTranspose1d formula. The range assertion hides potential off-by-one errors in padding. A tighter assertion with the exact expected value would be more useful.

**Severity:** P3 — test quality.

#### F12. `decode_vae.py` reads `temperal_downsample` (typo in HF config key) (`decode_vae.py:180`)

```python
temporal_upsample = list(reversed(config.get("temperal_downsample", [False, True, True])))
```

This reads a misspelled key from the HF config (`temperal` instead of `temporal`). If this is the actual key name in the HF config, it should have a comment noting the upstream typo. If not, it will silently fall back to the default.

**Severity:** P3 — but should be documented either way.

---

## Test Quality Assessment

The tests are **reasonable but not strong**.

**Good:**
- Shape assertions for all major components
- NaN checks (important for BF16 pipelines)
- Patchify/unpatchify roundtrip test
- Small configs to keep tests fast
- Clamping assertion for audio output

**Weak:**
- No numerical regression tests (no golden values from the reference implementation)
- No test for weight loading from actual HF checkpoint (even a single-layer subset)
- VAE decoder test asserts shapes but not spatial relationships (e.g., upsampling should produce locally correlated values, not just any non-NaN tensor)
- Pipeline test with `num_inference_steps=2` and random weights is not testing meaningful denoising — it is testing that the code does not crash, which is valuable but distinct from correctness
- Audio dilated conv has no test comparing against a reference non-dilated implementation
- `decode_vae.py` has zero test coverage

**Tautological risk:** The config default tests (F11-style "default config matches Cosmos3") are near-tautological since they test that a hardcoded default equals the same hardcoded value. They would only catch accidental edits.

---

## Weight Conversion Completeness

The `convert.py` `GENERATION_PATTERNS` list covers the major generation-pathway weight families. Reviewing against the model definition:

- `proj_in`, `proj_out` — covered by `r"proj_in\..*"`, `r"proj_out\..*"`
- `audio_proj_in`, `audio_proj_out` — covered
- `time_embedder` — covered
- `norm_moe_gen` — covered by `r".*norm_moe_gen.*"`
- `add_q_proj`, `add_k_proj`, `add_v_proj`, `to_add_out` — covered
- `norm_added_q`, `norm_added_k` — covered by `r".*norm_added_.*"`
- `input_layernorm_moe_gen`, `post_attention_layernorm_moe_gen` — covered by `r".*_moe_gen.*"`
- `mlp_moe_gen` — covered by `r".*_moe_gen.*"`
- `audio_modality_embed`, `action_modality_embed` — covered

**Gap identified:** `action_proj_in` is listed in the generation patterns but the comment in `load.py` says it has a "non-standard structure" (`action_proj_in.fc`). This means the HF checkpoint has `action_proj_in.fc.weight` but the model defines `action_proj_in` as a plain `nn.Linear` (no `.fc` sub-module). With `strict=False`, this weight is silently dropped. The action projection is not used in the current pipeline, but it is a known mapping gap.

The `map_weight_name` function only handles `to_out.0` and `to_add_out.0` remapping. If any other HF weight names have numeric indices (e.g., from `nn.ModuleList` wrappers), they would be silently mismatched. This is hard to verify without the actual checkpoint, but the risk is mitigated by the fact that Phase 1 reasoning works correctly, suggesting the understanding-pathway mappings are complete.

---

## Memory Safety Assessment

For 16B BF16 on 128GB unified memory:
- Model weights: ~32GB
- Single forward pass activations: ~8-16GB (estimated, depends on sequence length and graph retention)
- CFG doubles the activation cost per step (F4 above)
- VAE decode adds ~2-4GB (much smaller model)
- `mx.eval` calls are placed at reasonable points in the VAE decoder

**Risk:** The CFG double-forward without intermediate eval (F4) is the primary OOM risk. At 30 steps with `mx.eval(latents)` after each step, the denoising loop itself should be fine — but each step's peak includes both forward passes.

---

## Commands Run

- `git log --oneline c906136..85f5bb7` — 6 commits confirmed
- `git diff c906136..85f5bb7 --stat` — 1500 LOC confirmed
- `git diff c906136..85f5bb7 -- <each file>` — all diffs read
- All source files in the diff read in full
- All dependency files (`attention.py`, `conv3d.py`, `scheduler.py`, `timestep.py`, `convert.py`, `load.py`) read in full
- No commands failed

---

## Disposition

Three P1 findings (F1, F2, F3), four P2 findings (F4, F5, F6, F7), four P3 findings (F9-F12). F3 (mRoPE position IDs) is the most impactful for generation quality. F2 (strict=False) is the most dangerous for silent regression. F1 is correct-by-accident and should be clarified before it becomes a real bug.

No findings require blocking the current work — the pipeline runs end-to-end and produces output — but F2 and F3 should be addressed before trusting generation quality results.
