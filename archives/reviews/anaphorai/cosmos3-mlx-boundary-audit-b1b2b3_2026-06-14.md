# Anaphora: Cosmos3-MLX Generation Boundary Audit (B1, B2, B3)

**Probolē:** `archives/reviews/probolai/cosmos3-mlx-generation-boundary-audit_2026-06-14.md`
**Target:** `lyonsno/cosmos3-mlx` main branch at `fbe0693`
**Reviewer:** Claude Opus 4.6 (Epistaxis Aposkepsis, fresh mode)
**Date:** 2026-06-14
**Review context mode:** Code-only with HF reference comparison (treated as fresh)
**Commands run:** curl (HF pipeline, transformer, embeddings sources) — all succeeded.

---

## B1: Tokenization — MATERIAL FINDING (Severity: HIGH)

### Three discrepancies found between MLX and HF reference:

**B1.1: Missing `_add_special_tokens` — eos + start_of_generation tokens**

The HF reference (`pipeline_cosmos3_omni.py:1088-1092`) appends two special
tokens after the chat-template output:

```python
def _add_special_tokens(input_ids: list[int]) -> list[int]:
    return list(input_ids) + [
        self.llm_special_tokens["eos_token_id"],
        self.llm_special_tokens["start_of_generation"],
    ]
```

where `start_of_generation` is `<|vision_start|>` (line 403).

The MLX code (`pipeline.py:139-150`) does:
```python
text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
cond_ids = mx.array([self.tokenizer.encode(text)])
```

It does NOT append eos + `<|vision_start|>`. The model was trained to see these
sentinel tokens as the boundary between the text understanding stream and the
generation (diffusion) stream. Without them, the transformer has no signal that
generation tokens follow, and the text-to-generation pathway transition is
malformed.

**Severity:** HIGH. This is the most likely single-point cause of semantic
incoherence. The model literally does not know it should be generating an image.

**Smallest fix:** After `self.tokenizer.encode(text)`, append eos_token_id and
the token id for `<|vision_start|>`. Same for unconditional.

**B1.2: Missing system prompt**

The HF reference (`pipeline_cosmos3_omni.py:1076-1078`) prepends a system
message:

```python
system_prompt = _SYSTEM_PROMPT_IMAGE if is_image else _SYSTEM_PROMPT_VIDEO
conversations.append({"role": "system", "content": system_prompt})
```

with `_SYSTEM_PROMPT_IMAGE = "You are a helpful assistant who will generate images from a give prompt."` (line 135).

The MLX code uses only `[{"role": "user", "content": prompt}]` with no system
message. This changes the token sequence the model sees.

**Severity:** MEDIUM. The model was fine-tuned with these system prompts. Missing
them likely degrades quality but is less critical than B1.1.

**Smallest fix:** Prepend the system message to the conversations list before
`apply_chat_template`.

**B1.3: Missing prompt-augmentation templates (resolution/duration)**

The HF reference appends resolution and duration metadata to the prompt text
before tokenization:

```python
"This image is of {height}x{width} resolution."
"The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
```

The MLX code passes the raw prompt without these augmentations.

**Severity:** MEDIUM. The model was trained with these templates. Missing them
means the model does not know the target resolution, which may contribute to
spatial incoherence at non-default sizes.

**Smallest fix:** Append the resolution template to the prompt text before
tokenization. For the unconditional (negative) prompt, use the inverse templates.

---

## B2: Patchification — MATCH (Severity: NONE)

### Patchification order matches.

The HF reference (`transformer_cosmos3.py:418-439`) patchifies channels-first
`[C, T, H, W]` latents as:

```python
latent = latent.reshape(latent_channel, t_actual, h_patches, p, w_patches, p)
latent = torch.einsum("cthpwq->thwpqc", latent).reshape(-1, p * p * latent_channel)
```

The resulting token order is: for each `(t, h, w)` position, one token
containing `(p, p, C)` flattened. The flattening within each patch is
`(p, q, c)` — row-major over the two patch spatial dims, then channels.

The MLX code (`pipeline.py:78-88`) does:

```python
x = latents.reshape(batch, t, h_p, p, w_p, p, z)
x = mx.transpose(x, (0, 1, 2, 4, 3, 5, 6))  # [B, T, H_p, W_p, p, p, z]
x = x.reshape(batch, t * h_p * w_p, p * p * z)
```

Starting from channels-last `[B, T, H, W, z]`, this produces the same token
ordering: iterate `(t, h_p, w_p)` outer, then `(p, p, z)` inner. The einsum
`cthpwq->thwpqc` and the reshape-transpose sequence produce identical
flattened token sequences.

**Unpatchify is also correct** — it is the exact inverse of patchify, matching
the HF `einsum("thwpqc->cthpwq")` semantics.

**Severity:** None. B2 is clean.

---

## B3: Timestep Embedding — MATERIAL FINDING (Severity: HIGH)

### Sinusoidal frequency formula differs: `cos,sin` order vs `sin,cos` with flip.

**HF reference chain:**

1. `Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)` calls
   `get_timestep_embedding()` (embeddings.py:32-92)
2. Frequency: `exponent = -log(10000) * arange(128) / (128 - 0)` i.e. divided
   by `half_dim - downscale_freq_shift` = `128 - 0 = 128`
3. Concatenation: `[sin(args), cos(args)]`
4. With `flip_sin_to_cos=True`: swaps to `[cos(args), sin(args)]`

Net result: `[cos, sin]` order, frequencies divided by `half_dim` (128).

**MLX code** (`timestep.py:34-41`):

```python
half_dim = self.freq_dim // 2  # 128
freqs = mx.exp(-math.log(10000.0) * mx.arange(half_dim) / half_dim)
embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
```

Frequency: divided by `half_dim` (128). Concatenation: `[cos, sin]`.

**These match.** The MLX code already uses `[cos, sin]` order directly, and the
frequency divisor is `half_dim = 128` in both cases (since `downscale_freq_shift=0`
means the HF formula `half_dim - 0 = half_dim`).

Correcting my initial assessment: **B3 sinusoidal embedding is numerically
identical.**

**MLP structure also matches:** Both use `Linear(256, hidden_size)` -> SiLU ->
`Linear(hidden_size, hidden_size)`.

**However, there is a timestep scaling discrepancy in the application chain.**

The HF reference (`transformer_cosmos3.py:620-621`) does:
```python
timesteps_vision = vision_timesteps * self.config.timestep_scale  # * 0.001
packed_timestep_embeds_vision = self.time_embedder(self.time_proj(timesteps_vision))
```

The MLX code (`model.py:342-344`) does:
```python
scaled_t = timestep * 0.001
t_emb = self.time_embedder(scaled_t)
```

The HF code passes the scaled timestep through `self.time_proj` (the `Timesteps`
module) BEFORE `self.time_embedder` (the `TimestepEmbedding` MLP). In HF,
`time_proj` computes the sinusoidal features, then `time_embedder` runs the MLP.

The MLX `TimestepEmbedding.__call__` combines both steps internally (sinusoidal
+ MLP in one class). This is architecturally equivalent IF the weight mapping
is correct — `time_proj` has no learnable parameters, and the MLX
`TimestepEmbedding.linear_1` corresponds to HF `time_embedder.linear_1`.

**Weight mapping check:** HF has `time_proj` (no weights) + `time_embedder.linear_1`
+ `time_embedder.linear_2`. MLX has `time_embedder.linear_1` +
`time_embedder.linear_2`. The names match directly, so weight loading should
be correct assuming the conversion script maps `time_embedder.*` to
`time_embedder.*`.

**Revised B3 severity:** LOW — the sinusoidal formula and MLP structure match.
The only residual risk is weight-name mapping, which should be verified by a
numerical smoke test comparing embeddings for a known timestep value.

---

## Summary

| Boundary | Status | Severity | Could explain semantic incoherence? |
|----------|--------|----------|-------------------------------------|
| B1.1: Missing eos + `<\|vision_start\|>` tokens | MISMATCH | HIGH | **YES — most likely cause** |
| B1.2: Missing system prompt | MISMATCH | MEDIUM | Contributes to degradation |
| B1.3: Missing resolution/duration templates | MISMATCH | MEDIUM | Contributes to degradation |
| B2: Patchification order | MATCH | NONE | No |
| B3: Timestep embedding formula | MATCH | LOW | No (verify weight names) |

## Recommended action priority

1. **Fix B1.1 immediately.** Append `[eos_token_id, vision_start_token_id]` to
   both conditional and unconditional token sequences. This is almost certainly
   the primary cause of semantic incoherence — without the `<|vision_start|>`
   sentinel, the model has no signal that the following tokens are diffusion
   generation targets rather than text continuation.

2. **Fix B1.2 and B1.3** in the same pass. Add system prompt and resolution
   templates.

3. **B3 smoke test:** Compare MLX and PyTorch `time_embedder` output for
   timestep=999.0 (raw) -> 0.999 (scaled) to confirm numerical match.

## Smallest verification test

```python
# After fixing B1.1:
tok = tokenizer
eos = tok.eos_token_id
vision_start = tok.convert_tokens_to_ids("<|vision_start|>")
# Verify these are appended:
assert cond_ids_list[-2] == eos
assert cond_ids_list[-1] == vision_start
```
