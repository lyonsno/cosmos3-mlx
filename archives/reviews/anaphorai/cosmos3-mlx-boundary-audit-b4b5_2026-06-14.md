# Anaphora: cosmos3-mlx Generation Boundary Audit (B4, B5)

**Probolē:** `archives/reviews/probolai/cosmos3-mlx-generation-boundary-audit_2026-06-14.md`
**Target:** `cosmos3-mlx` main at `fbe0693`
**Review context mode:** fresh (code-only with HF reference comparison)
**Reviewer:** Claude Opus 4.6 (Epistaxis Aposkepsis agent)
**Date:** 2026-06-14
**Reference:** HuggingFace `diffusers` main, `transformer_cosmos3.py` and `pipeline_cosmos3_omni.py`

---

## B4: Dual-pathway attention wiring

**File:** `cosmos3_mlx/attention.py`, `_generation_forward` (lines 174-224)

### Finding B4.1: Attention wiring is structurally correct — NO material discrepancy

The MLX `_generation_forward` follows the same pattern as the HF `Cosmos3AttnProcessor`:

| Step | HF reference (lines 46-99) | MLX (lines 189-224) | Match? |
|------|---------------------------|---------------------|--------|
| Project gen Q/K/V | `attn.add_q_proj(gen_seq)` etc. | `self.add_q_proj`, `self.add_k_proj`, `self.add_v_proj` | Yes |
| QK norm | `attn.norm_added_q(q_gen)`, `attn.norm_added_k(k_gen)` | `self.norm_added_q(q_gen)`, `self.norm_added_k(k_gen)` | Yes |
| RoPE on gen Q/K | `q_gen * cos_gen + rotate_half(q_gen) * sin_gen` | `apply_rotary_pos_emb(q_gen, k_gen, cos, sin)` | Yes |
| Concat und+gen K/V | `torch.cat([k_und, k_gen], dim=0)` | `mx.concatenate([und_keys, k_gen], axis=1)` | Yes (axis differs due to layout: HF uses `[seq, heads, dim]`, MLX uses `[batch, seq, heads, dim]`) |
| Full (non-causal) attention | `is_causal=False` | No mask passed to SDPA | Yes |
| GQA expansion | `enable_gqa=True` (handled inside dispatch) | Explicit `mx.repeat(k_full, repeat_factor, axis=2)` | Yes |
| Output projection | `attn.to_add_out(full_out)` | `self.to_add_out(attn_out)` | Yes |

**Understanding keys reuse:** The `und_keys` passed to `_generation_forward` (line 168-169) are `k_unexpanded` — the post-QK-norm, post-RoPE keys saved at line 121 before GQA expansion. This is correct: the HF reference also concatenates pre-GQA-expansion keys and then lets `dispatch_attention_fn` handle GQA with `enable_gqa=True`. The MLX code applies GQA expansion after concatenation (lines 208-210), which is mathematically equivalent.

**Understanding values reuse:** Same pattern — `v_unexpanded` at line 121 is the pre-GQA values, correctly reused.

**Attention masking:** The HF reference uses `is_causal=False` for the generation pathway. The MLX code calls `scaled_dot_product_attention` with no mask (line 218-219), which defaults to full (non-causal) attention. This is correct.

**Severity:** None. B4 is not the source of the image generation incoherence.

---

## B5: Velocity conversion (patchify/unpatchify/scheduler step loop)

**Files:** `cosmos3_mlx/pipeline.py` (lines 174-212), `cosmos3_mlx/model.py` (`diffusion_forward`, lines 305-385)

### Finding B5.1: Patchify/unpatchify loop is structurally correct — NO material discrepancy

The MLX pipeline's denoising loop follows this sequence per step:

1. `_patchify_latents(latents)` — spatial latents to patch tokens
2. `model.diffusion_forward(...)` — transformer produces velocity in patch token space
3. `_unpatchify_latents(velocity_patches, ...)` — velocity back to spatial latent space
4. `scheduler.step(velocity, t_tensor, latents)` — scheduler steps in spatial latent space

The HF reference does the equivalent:
1. `_patchify_and_pack_latents(vision_tokens)` — inside `transformer.forward()`
2. Transformer produces velocity predictions in packed patch space
3. `_unpatchify_and_unpack_latents(preds_vision_packed, ...)` — inside `transformer.forward()`
4. `scheduler.step(velocity_vision.unsqueeze(0), t, latents.unsqueeze(0))` — scheduler steps in spatial `[C, T, H, W]` space

The key difference is WHERE patchify/unpatchify happens:
- **HF:** patchify and unpatchify are inside the transformer's `forward()` method
- **MLX:** patchify is in the pipeline, unpatchify is in the pipeline

This is a structural refactoring, not a semantic difference. The operations themselves are mathematically equivalent (verified by tracing the einsum/reshape/transpose patterns — see below).

### Finding B5.2: Patchify token ordering matches — NO discrepancy

HF patchify (einsum `"cthpwq->thwpqc"`): `[C,T,H_p,p,W_p,p] -> [T,H_p,W_p,p,p,C] -> [T*H_p*W_p, p*p*C]`

MLX patchify (reshape + transpose):
```
[B,T,H_p,p,W_p,p,z] -> transpose(0,1,2,4,3,5,6) -> [B,T,H_p,W_p,p,p,z] -> [B,T*H_p*W_p, p*p*z]
```

Token ordering: T-major, then H-major, then W-major. Identical to HF.

MLX unpatchify correctly inverts this (transpose is its own inverse for swapping adjacent dims).

### Finding B5.3: Scheduler stepping domain matches — NO discrepancy

Both HF and MLX step the scheduler in spatial latent space (post-unpatchify), not in patch token space. The velocity is unpatchified first, then the scheduler computes `x_0 = x_t - sigma * velocity` and steps in that space.

### Finding B5.4: Latent normalization — POTENTIAL discrepancy (low confidence)

The HF pipeline's postprocessing (line ~1669-1672 of `pipeline_cosmos3_omni.py`) applies VAE latent normalization before decoding:

```python
mean = self._vae_latents_mean.to(...)
inv_std = self._vae_latents_inv_std.to(...)
z_raw = latents.to(dtype) / inv_std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)
```

The MLX pipeline does NOT apply any latent normalization before VAE decode (lines 220-226). If the checkpoint was trained with normalized latents (which is standard for latent diffusion), the denoised latents must be denormalized before VAE decoding. However, this is a B7 (VAE decode) boundary issue rather than B5, and it would produce washed-out or color-shifted output rather than semantic incoherence. Noted for completeness but out of scope.

**Severity:** B5 overall is not the source of the image generation incoherence.

---

## Out-of-scope finding flagged during B4/B5 audit

### Critical: mRoPE frequency computation is fundamentally different (B3 boundary, not B4/B5)

**Severity: HIGH — this is the most likely cause of semantic incoherence in generation.**

**File:** `cosmos3_mlx/rope.py`, `_compute_inv_freq` (lines 42-49) vs HF `Cosmos3VLTextRotaryEmbedding` (lines 108-134)

The HF reference computes RoPE frequencies as a single shared set of 64 values across all three axes, then interleaves them:

```python
# HF: ONE shared inv_freq for all axes
inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2) / head_dim))
# = theta^(-{0, 2, 4, ..., 126}/128) -> 64 frequencies

# Then per-axis position IDs multiply the SAME 64 frequencies
# Then apply_interleaved_mrope mixes T/H/W at indices {0,3,6,...}, {1,4,7,...}, {2,5,8,...}
```

The MLX code computes SEPARATE inv_freq per section with DIFFERENT frequency bases:

```python
# MLX: SEPARATE inv_freq per axis section
# Temporal (section_dim=24): theta^(-{0,1,...,23}/24)
# Height (section_dim=20):   theta^(-{0,1,...,19}/20)
# Width (section_dim=20):    theta^(-{0,1,...,19}/20)
```

These produce completely different frequency patterns:
- HF temporal frequency 0: `theta^(0/128) = 1.0`
- MLX temporal frequency 0: `theta^(0/24) = 1.0` (same)
- HF temporal frequency 1: `theta^(-2/128) = theta^(-0.0156)`
- MLX temporal frequency 1: `theta^(-1/24) = theta^(-0.0417)` (2.67x faster decay)

Additionally, the HF uses **interleaved** layout (T,H,W frequencies are mixed at indices `{0,3,6,...}`, `{1,4,7,...}`, `{2,5,8,...}`), while MLX uses **contiguous** sections (T at `[0:24]`, H at `[24:44]`, W at `[44:64]`).

Both the frequency values AND the layout are wrong. This means every attention layer receives incorrect positional information for generation tokens, which would directly cause the spatial structure to be destroyed — explaining semantically incoherent output (blobs with some structure).

The understanding pathway shares the same RoPE bug, but for text tokens all 3 axes have identical position IDs (`pos, pos, pos`), so the interleaving has no practical effect and the frequency difference is compensated by the model's tolerance. For generation tokens where T, H, W positions differ, this bug is devastating.

**Smallest fix:** Rewrite `Cosmos3RotaryEmbedding` to match the HF pattern:
1. Compute a single shared `inv_freq` of size `head_dim//2` using `theta^(-arange(0, head_dim, 2) / head_dim)`
2. Multiply each axis's position IDs by all 64 frequencies to get `[3, B, N, 64]`
3. Apply interleaved mRoPE mixing per the HF `apply_interleaved_mrope` pattern

**Smallest test:** For a fixed set of position IDs (e.g., T=[15000], H=[0], W=[0]), compute cos/sin from both the HF `Cosmos3VLTextRotaryEmbedding` and the MLX `Cosmos3RotaryEmbedding`. They should match element-wise. They currently will not.

---

## Summary

| Boundary | Verdict | Severity |
|----------|---------|----------|
| B4: Dual-pathway attention | **PASS** — structurally correct, matching HF reference | None |
| B5: Velocity conversion | **PASS** — patchify/unpatchify/scheduler loop is correct | None |
| (Out of scope) mRoPE frequencies | **FAIL** — fundamentally wrong frequency computation and layout | **HIGH** — likely root cause of generation incoherence |

B4 and B5 are clean. The generation pipeline's attention wiring and velocity conversion are structurally sound and match the HF reference. The semantic incoherence is almost certainly caused by the mRoPE frequency computation (a B3 boundary issue), which produces completely wrong positional embeddings for generation tokens where the three spatial axes have different position IDs.

---

## Commands run

- WebFetch of HF reference `transformer_cosmos3.py` (main branch) — success
- WebFetch of HF reference `pipeline_cosmos3_omni.py` (main branch) — success
- Read of all target files (`attention.py`, `pipeline.py`, `model.py`, `scheduler.py`, `rope.py`, `timestep.py`) — success
- No destructive commands executed
