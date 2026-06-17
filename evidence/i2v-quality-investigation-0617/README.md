# i2v Quality Investigation — 2026-06-17

Diaulos: nvidia-omni-mlx-port-research
Session: cc-cosmos3-quality-0617
Branch: cc/gen-quality-review-0614
Attractor: port-nvidia-cosmos3-nano-to-mlx-for-mac-native-omnimodal-inference

## Problem Statement

Operator reports i2v generation at 720p produces qualitatively wrong output
compared to HF diffusers reference: fewer details resolved, specific objects
missing, things wrong. Not just softness — structurally different quality.

## Experiments Run

### 1. t2v Transformer Parity (identical inputs)

**Script:** `scripts/bisect_transformer_720p.py`

Fed identical noise, tokens, and timestep to both MLX and HF transformers
for t2v (all frames noisy). Compared raw velocity output.

| Resolution | Cosine | Max diff | Mean diff | p99.9 |
|---|---|---|---|---|
| 256x256 | 0.99992 | 0.056 | 0.011 | 0.045 |
| 720p | 0.99984 | 0.119 | 0.015 | 0.062 |

**Conclusion:** t2v transformer output matches within bf16 tolerance.

### 2. i2v Transformer Parity (identical inputs, frame 0 conditioned)

**Script:** `scripts/bisect_i2v_transformer.py`
**Numerical data:** `cosmos3_i2v_xfmr_256_256.npz`, `cosmos3_i2v_xfmr_720p.npz`

Fed identical mixed latents (frame 0 = fake conditioning, frames 1-3 = noise)
to both transformers with noisy_frame_indexes=[1,2,3].

**256x256 i2v:**
| Frame | Cosine | Max diff | Mean diff |
|---|---|---|---|
| 0 (COND) | 0.000 (MLX=garbage, HF=zero) | 5.85 | 0.74 |
| 1 (noisy) | 0.99988 | 0.105 | 0.013 |
| 2 (noisy) | 0.99991 | 0.059 | 0.012 |
| 3 (noisy) | 0.99991 | 0.068 | 0.011 |

**720p i2v:**
| Frame | Cosine | Max diff | Mean diff |
|---|---|---|---|
| 0 (COND) | 0.000 (MLX=garbage, HF=zero) | 6.86 | 0.71 |
| 1 (noisy) | 0.99981 | 0.220 | 0.016 |
| 2 (noisy) | 0.99989 | 0.092 | 0.013 |
| 3 (noisy) | 0.99990 | 0.095 | 0.012 |

**Frame 0 difference explained:** HF runs proj_out only on noisy-frame tokens
via vision_mse_loss_indexes, producing exact zeros for conditioned frames.
MLX runs proj_out on all tokens (garbage for conditioned frame) then zeros
in pipeline. Both produce zero velocity for frame 0 in the final pipeline.

**Conclusion:** i2v noisy frames match at cosine 0.9998+ at 720p. The
transformer forward pass is faithful for i2v.

### 3. Frame 0 Latent Preservation Through Denoising

Instrumented scheduler.step() to track frame 0 latent across all steps.

```
Step  0: vel_f0_mag=0.000000 | in: mean=-0.0179 std=0.6344 | out: mean=-0.0179 std=0.6344 | diff=0.000000
Step  1: vel_f0_mag=0.000000 | in: mean=-0.0179 std=0.6344 | out: mean=-0.0179 std=0.6344 | diff=0.000000
...
Step  9: vel_f0_mag=0.000000 | in: mean=-0.0179 std=0.6344 | out: mean=-0.0179 std=0.6344 | diff=0.000000
```

**Conclusion:** Frame 0 conditioning latent is perfectly preserved through
all denoising steps. Velocity is exactly zero, latent unchanged.

### 4. VAE Encode-Decode Roundtrip

| Condition | PSNR | Mean pixel diff | Max pixel diff |
|---|---|---|---|
| Single frame encode→decode | 28.1 dB | 6.3 | 118 |
| 16-frame encode→decode, frame 0 | 23.5 dB | 13.5 | 135 |

**Visual evidence:**
- `cosmos3_vae_roundtrip_input.png` — original input resized to 256
- `cosmos3_vae_roundtrip_256.png` — single-frame roundtrip (good quality)
- `cosmos3_vae_roundtrip_16f_frame0.png` — 16-frame roundtrip frame 0 (softer)

**Conclusion:** 16-frame VAE roundtrip degrades frame 0 due to temporal causal
convolutions in the decoder blending it with neighboring frame latents.
This is expected VAE behavior, not a pipeline bug.

### 5. Full i2v Pipeline Output (visual inspection)

**Generated with:** seed=42, guidance=6.0, steps=30, prompt="A car driving
along a coastal mountain road with falling rocks causing an emergency stop",
conditioning image=example_i2v_input.jpg

**256x256 output:**
- `cosmos3_i2v_256_input.png` — input resized
- `cosmos3_i2v_256_frame00.png` through `_frame15.png`
- Frame 0: heavily degraded — stippling artifacts, barely recognizable
- Frames 1+: soft/hazy but scene-coherent, no prompt-specific events

**720p output:**
- `cosmos3_i2v_720p_input.png` — input resized
- `cosmos3_i2v_720p_frame00.png` through `_frame15.png`
- Frame 0: degraded but recognizable (road, mountain visible)
- Frames 1+: washed out, hazy, scene-coherent forward motion
- No falling rocks or emergency stop (prompt-specific content missing)

**Frame 0 degradation source:** VAE decoder temporal convolutions, not
denoising corruption. Frame 0 latent is perfectly preserved through
denoising; the degradation happens during video decode.

## Eliminated Causes (Exhaustive)

- VAE encoder: cosine 0.9999 vs HF (single and multi-frame)
- VAE decoder: pixel-identical for identical latent input
- Transformer t2v: cosine 0.99984 at 720p
- Transformer i2v noisy frames: cosine 0.99981-0.99990 at 720p
- Timestep embedding: mathematically equivalent (t2v and i2v)
- mRoPE position IDs: identical formulas, verified config values
- Token sequence packing: no zero gap for video-only
- Patchify: matches HF zero-padding
- Conditioning latents: cosine 0.9999
- `.at[].add()` mask corruption: tested correct at 720p sizes
- Velocity zeroing: structurally correct, frame 0 diff=0 at every step
- Scheduler: max diff 0.0000019 (synthetic test), frame 0 preserved exactly

## ROOT CAUSE FOUND: Double Denormalization

**Bug:** `pipeline.generate()` returned already-denormalized latents
(`latents * std + mean`) in `result["latents"]`. All external decode paths
called `decode_latents(result["latents"], vae_dir)` which denormalized
AGAIN: `(latents * std + mean) * std + mean`. This corrupted the entire
latent space, producing the washed-out, stippled, detail-free output.

**Fix:** Pipeline now returns normalized latents. `decode_latents()` handles
denormalization as its contract specifies. One-line change in `pipeline.py`.

**Visual proof:**
- `cosmos3_DOUBLE_DENORM_frame0.png` — before fix (stippled mess)
- `cosmos3_CORRECT_DENORM_frame0.png` — after fix (crisp, matches input)
- `cosmos3_FIXED_i2v_256_frame*.png` — full 16-frame fixed output

**Why this only affected i2v:** t2v visual smoke tests were evaluated from
a different code path or with different expectations. The double denorm
affects ALL decode, but frame 0 of i2v (which should match the input image)
made the corruption obvious. t2v output has no ground-truth frame to compare
against, so the degradation was less noticeable.

**Operator observation (confirmed, durable):** The quality problem is specific
to image-conditioned generation (i2v). t2v output was visually confirmed good
at all resolutions in a prior session. This is consistent with the double
denormalization being the sole cause — both t2v and i2v were affected, but
only i2v has a ground-truth conditioning frame that makes the corruption
immediately visible.

## Remaining: Early-Frame Quality (post-fix)

After the double-denorm fix, the first 3-4 pixel frames (latent frame 0 =
conditioned, decoded into pixel frames 0-3) are still slightly hazy and more
static than frames 4+. The NVIDIA reference `example_i2v_output.mp4` does NOT
show this — early frames are crisp and in motion from frame 1.

**Latent statistics show a sharp discontinuity at the conditioning boundary:**
```
frame 0 (cond):  std=0.626  range [-3.0, 3.0]
frame 1 (gen):   std=0.869  range [-4.2, 3.6]
frame 2 (gen):   std=0.947  range [-4.4, 4.1]
frame 3 (gen):   std=0.953  range [-4.3, 4.4]
cos(f0, f1) = 0.68  (big jump)
cos(f1, f2) = 0.85  (smooth)
cos(f2, f3) = 0.86  (smooth)
```

The conditioning latent (frame 0) has much lower variance than generated
frames. The VAE decoder's temporal convolutions cross this boundary, causing
visible quality degradation in the early decoded frames.

**Not yet determined:** whether HF produces the same latent statistics at
the boundary, or whether their conditioning/mixing approach avoids this
discontinuity. The NVIDIA reference output proves clean early frames are
possible with this model.

**Tested and eliminated:**
- Step count (30 vs 35): no visible difference at 256
- KV caching vs full recompute: architecturally correct (und pathway is
  causal self-attention only, doesn't attend to gen tokens)

**Remaining suspects:**
- fps=25 vs HF's fps=24 (affects mRoPE temporal positions and text prompt)
- NVIDIA reference may use cosmos-framework, not diffusers pipeline
- The reference uses a vastly more detailed structured prompt
- Need HF pipeline on GPU to properly baseline early-frame quality
