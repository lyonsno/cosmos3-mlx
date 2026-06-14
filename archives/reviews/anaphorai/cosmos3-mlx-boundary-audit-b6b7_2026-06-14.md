# Anaphora: cosmos3-mlx Boundary Audit B6 (Scheduler) + B7 (VAE Decode)

**Probole:** `archives/reviews/probolai/cosmos3-mlx-generation-boundary-audit_2026-06-14.md`
**Target commit:** `fbe0693` (main)
**Review context mode:** Code-only with HF reference comparison (fresh)
**Reviewer:** Claude Opus 4.6 (Epistaxis Aposkepsis agent)
**Date:** 2026-06-14

## Summary

The scheduler (B6) is numerically correct. The VAE decoder (B7) has three
critical structural defects that together fully explain the observed semantic
incoherence in generated images.

---

## B6: Scheduler Numerical Match

### B6.1: Flow sigma computation — PASS

MLX (`scheduler.py:52`):
```python
sigmas = np.linspace(1, 1 / self.num_train_timesteps, num_inference_steps + 1)[:-1]
```

HF reference (`scheduling_unipc_multistep.py`, flow sigma branch):
```python
sigmas = np.linspace(1, 1 / self.config.num_train_timesteps, num_inference_steps + 1)[:-1]
```

Identical. Flow shift formula (line 56) also matches HF (line 434). The
sigma[0] epsilon clamp (line 60) matches HF (line 438). Terminal sigma=0
appended (line 64) matches HF `final_sigmas_type="zero"` (line 444-450).

### B6.2: x0 prediction formula — PASS

MLX (`scheduler.py:83`):
```python
return sample - sigma * model_output
```

HF (`scheduling_unipc_multistep.py`, line 805-807):
```python
elif self.config.prediction_type == "flow_prediction":
    sigma_t = self.sigmas[self.step_index]
    x0_pred = sample - sigma_t * model_output
```

Identical for `prediction_type="flow_prediction"`.

### B6.3: First-order step — PASS

MLX (`scheduler.py:150-159`):
```python
lambda_s = np.log(alpha_s / max(sigma_s, 1e-10))
lambda_t = np.log(max(alpha_t, 1e-10) / max(sigma_t, 1e-10))
h = lambda_t - lambda_s
h_phi_1 = np.expm1(-h)
x_t = (sigma_t / sigma_s) * sample - alpha_t * h_phi_1 * x0_pred
```

HF (`scheduling_unipc_multistep.py`, lines 886-945):
```python
lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
h = lambda_t - lambda_s0
hh = -h  # because predict_x0=True
h_phi_1 = torch.expm1(hh)  # = expm1(-h)
x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
```

Numerically identical. The MLX code uses `np.log(alpha/sigma)` which equals
`np.log(alpha) - np.log(sigma)`. The HF code computes `hh = -h` for predict_x0
mode and `h_phi_1 = expm1(-h)`. Both produce the same result.

For flow sigmas, `_sigma_to_alpha_sigma_t` returns `alpha_t = 1 - sigma`,
`sigma_t = sigma`, matching the MLX `alpha = 1 - sigma` (line 115-116).

### B6.4: Second-order correction coefficients — PASS

After algebraic analysis of sign conventions:

- HF uses `rk = (lambda_{prev} - lambda_{current}) / h` (negative, since
  lambda increases as sigma decreases), `D1 = (m_prev - m0) / rk`,
  `rhos_p = 0.5` (hardcoded for order 2), correction = `-alpha_t * B_h * 0.5 * D1`.
- MLX uses `r = (lambda_s - lambda_prev) / h = -rk`, `rho = 1/(2*r)`,
  `D1 = m0 - m_prev`, correction = `-alpha_t * B_h * rho * D1`.

The double sign flip (negated D1 and negated rho denominator) produces the
same numerical result. `B_h = expm1(-h)` matches HF's `B_h = expm1(hh)` with
`hh = -h`.

### B6.5: Final step (sigma=0) — PASS

MLX (`scheduler.py:118-119`):
```python
if sigma_t == 0.0:
    result = x0_pred
```

HF does not have an explicit sigma=0 branch in `multistep_uni_p_bh_update`.
However, when `sigma_t = 0`, `alpha_t = 1`, `lambda_t -> +inf`, `h -> +inf`,
`h_phi_1 = expm1(-inf) = -1`, so `x_t = 0 * x - 1 * (-1) * x0 = x0`. The
MLX shortcut produces the same result and avoids the numerical edge case.

**B6 Verdict: PASS. No discrepancies found.**

---

## B7: VAE Decode

### Finding B7.1: MISSING MID-BLOCK ATTENTION — CRITICAL

**Severity:** CRITICAL — likely contributes to semantic incoherence
**File:** `/Users/noahlyons/dev/cosmos3-mlx/cosmos3_mlx/decode_vae.py:177-179`

The MLX decoder runs `mid_block` as two sequential resnet blocks:
```python
for i in range(2):
    x = _resnet_block(x, weights, f"mid_block.resnets.{i}")
```

The HF `WanMidBlock` with `num_layers=1` (which is what the decoder uses, line
833) constructs:
- `resnets[0]`: first residual block
- `attentions[0]`: one self-attention block (`WanAttentionBlock`)
- `resnets[1]`: second residual block

The forward pass is: `resnet[0]` -> `attention[0]` -> `resnet[1]`.

The MLX code completely omits the attention block between the two resnets. The
attention weights (`mid_block.attentions.0.*`) are loaded from the checkpoint
but never applied. This means the mid-block's self-attention, which provides
global spatial coherence at the bottleneck resolution, is entirely skipped.

**Impact:** Self-attention at the bottleneck is essential for global coherence.
Without it, the decoder can only rely on local convolutional receptive fields,
which explains why generated images have local texture but no global semantic
structure.

**Smallest fix:** Insert attention block execution between resnets:
```python
x = _resnet_block(x, weights, "mid_block.resnets.0")
x = _attention_block(x, weights, "mid_block.attentions.0")  # MISSING
x = _resnet_block(x, weights, "mid_block.resnets.1")
```

**Smallest test:** Compare mid_block output with and without attention for a
known input; the output should differ significantly.

### Finding B7.2: WRONG UPSAMPLING ARCHITECTURE — CRITICAL

**Severity:** CRITICAL — structurally incorrect decoder
**File:** `/Users/noahlyons/dev/cosmos3-mlx/cosmos3_mlx/decode_vae.py:107-113, 193-210`

The actual Cosmos 3 VAE config (`weights/Cosmos3-Nano/vae/config.json`) specifies
`"is_residual": true`. This means the HF decoder uses `WanResidualUpBlock`, not
`WanUpBlock`. The MLX code uses a simple `_dup_up_3d` function that does bare
`mx.repeat` duplication:

```python
def _dup_up_3d(x, temporal=False):
    x = mx.repeat(x, 2, axis=2)  # H
    x = mx.repeat(x, 2, axis=3)  # W
    if temporal:
        x = mx.repeat(x, 2, axis=1)  # T
    return x
```

The HF `WanResidualUpBlock` does three things the MLX code does not:

1. **`DupUp3D` residual shortcut:** A channel-reshuffling duplication upsample
   (`repeat_interleave` on channels, then spatial reshape) that creates a skip
   connection from the block input. This is NOT the same as `mx.repeat` on
   spatial dims — it operates on the channel dimension and reshapes.

2. **`WanResample` spatial upsample with learned Conv2d:** The main path uses
   `nn.functional.interpolate(mode="nearest-exact")` followed by a learned
   `Conv2d(dim, out_dim, 3, padding=1)`. The MLX code uses `mx.repeat` (which
   is nearest-neighbor duplication, close but not identical to nearest-exact
   interpolation) and has NO post-upsample convolution.

3. **`WanCausalConv3d` temporal upsample (for upsample3d mode):** For blocks
   with temporal upsampling, a `WanCausalConv3d(dim, dim*2, (3,1,1))` doubles
   the channel count, then the output is reshaped to interleave frames. The MLX
   code uses `mx.repeat(x, 2, axis=1)` for temporal upsampling.

4. **Residual addition:** The block output is `x + avg_shortcut(x_copy)`. The
   MLX code has no residual connection at all.

**Impact:** Every upsampling step is wrong. The learned convolutions and
residual connections in the upsampling path are essential for producing sharp,
semantically meaningful spatial structure. Without them, the upsampled features
are just duplicated blobs.

**Smallest fix:** Implement `WanResidualUpBlock` with DupUp3D shortcut,
WanResample (interpolation + conv), temporal conv, and residual add.

**Smallest test:** For a given input tensor and checkpoint weights, compare the
output of one up_block between MLX and HF PyTorch.

### Finding B7.3: POST-UPSAMPLE CONVOLUTION WEIGHTS IGNORED — CRITICAL

**Severity:** CRITICAL (consequence of B7.2)
**File:** `/Users/noahlyons/dev/cosmos3-mlx/cosmos3_mlx/decode_vae.py:186-210`

The weight-loading loop at line 143-154 loads all decoder weights including
upsampler convolution weights (e.g., `up_blocks.N.upsampler.resample.1.weight`,
`up_blocks.N.upsampler.time_conv.weight`, `up_blocks.N.avg_shortcut.*`). These
weights are loaded into the `weights` dict but never used, because the
upsampling is done by bare `_dup_up_3d` with no learned operations.

This means a significant fraction of the VAE decoder parameters (all upsampler
convolutions, temporal convolutions, and DupUp3D parameters) are completely
ignored during decoding.

**Impact:** Same as B7.2 — the learned upsampling transformations are skipped.

### B7.4: Unpatchify ordering — PASS

The MLX unpatchify (`decode_vae.py:222-226`) correctly converts channels-last
`[B, T, H, W, C*p*p]` by reshaping to `[B, T, H, W, C, p, p]` and transposing
to interleave patches: `[B, T, H, p, W, p, C]` -> `[B, T, H*p, W*p, C]`.

This matches the HF `unpatchify` function (which does the same in
channels-first format). The channel-to-patch mapping is consistent.

### B7.5: Output range conversion — PASS

MLX (`decode_vae.py:230`): `(mx.clip(x, -1.0, 1.0) + 1.0) / 2.0`

HF: VAE `_decode` returns `torch.clamp(out, -1.0, 1.0)`, then
`VaeImageProcessor.denormalize` applies `(images * 0.5 + 0.5).clamp(0, 1)`.

Both produce `[0, 1]` output from `[-1, 1]` VAE output. Numerically identical.

### B7.6: Block count and resnet count — PASS (conditional)

The MLX code dynamically counts up_blocks and resnets from weight keys
(lines 186-199). This correctly handles any number of blocks/resnets present
in the checkpoint. For the Cosmos 3 config with `num_res_blocks=2` and
`is_residual=true`, each WanResidualUpBlock has `num_res_blocks + 1 = 3`
resnets, which the weight-counting approach will discover correctly.

However, the resnets are the only operations applied per block — the upsampler
convolutions, temporal convolutions, and residual shortcuts are all missing
(as documented in B7.2).

---

## Severity Summary

| Boundary | Finding | Severity | Could explain incoherence? |
|----------|---------|----------|---------------------------|
| B6.1 | Flow sigma computation | PASS | N/A |
| B6.2 | x0 prediction formula | PASS | N/A |
| B6.3 | First-order step | PASS | N/A |
| B6.4 | Second-order correction | PASS | N/A |
| B6.5 | Final step (sigma=0) | PASS | N/A |
| B7.1 | Missing mid-block attention | **CRITICAL** | Yes — destroys global coherence |
| B7.2 | Wrong upsampling architecture | **CRITICAL** | Yes — all spatial upsample paths are structurally wrong |
| B7.3 | Upsampler weights ignored | **CRITICAL** | Yes — consequence of B7.2 |
| B7.4 | Unpatchify ordering | PASS | N/A |
| B7.5 | Output range conversion | PASS | N/A |
| B7.6 | Block/resnet count | PASS | N/A |

## Root Cause Assessment

The VAE decoder is the most likely root cause of the observed semantic
incoherence. The scheduler is numerically correct and should not be causing
issues.

The three VAE findings (B7.1, B7.2, B7.3) together mean that:

1. The decoder has no global spatial attention at the bottleneck.
2. Every spatial and temporal upsampling step uses bare duplication instead of
   learned convolutions, interpolation, temporal convolutions, and residual
   shortcuts.
3. A large fraction of the decoder's learned parameters are loaded but never
   applied.

Even if the transformer produces perfect latents, this decoder cannot produce
semantically meaningful images. The fix requires implementing the full
`WanResidualUpBlock` and `WanMidBlock` (with attention) architectures.

## Commands Run

- Read: `cosmos3_mlx/scheduler.py` (full)
- Read: `cosmos3_mlx/decode_vae.py` (full)
- Read: `weights/Cosmos3-Nano/vae/config.json` (full)
- WebFetch: HF `scheduling_unipc_multistep.py` (full)
- WebFetch: HF `autoencoder_kl_wan.py` (full)
- All reads completed successfully.
