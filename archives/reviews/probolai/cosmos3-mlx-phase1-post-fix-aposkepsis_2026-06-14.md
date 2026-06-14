# Probolē: cosmos3-mlx Phase 1 post-fix review

**Date:** 2026-06-14
**Target:** `lyonsno/cosmos3-mlx` — MLX port of NVIDIA Cosmos 3 Nano
**Worktree:** `/private/tmp/cosmos3-mlx-initial-scaffold-0614`
**Branch:** `cc/initial-scaffold-0614` (landed on `main`)
**Review context mode:** target code only, no inherited implementation thread
**Prior review:** A previous aposkepsis found 3 material + 3 important findings, all fixed. This is a fresh pass on the post-fix codebase.

## Target range

All source files in `cosmos3_mlx/` and `tests/`:
- `cosmos3_mlx/rope.py` — 3D mRoPE implementation
- `cosmos3_mlx/attention.py` — dual-pathway MoT attention with GQA, KV cache, causal masking
- `cosmos3_mlx/model.py` — full transformer model, generate() with EOS and KV cache
- `cosmos3_mlx/convert.py` — weight conversion (HF safetensors → MLX, reasoner-only filtering)
- `cosmos3_mlx/vision.py` — Qwen3-VL vision encoder with 3D grid-aware RoPE
- `cosmos3_mlx/load.py` — config + weight loading from HF directory
- `cosmos3_mlx/generate.py` — end-to-end inference pipeline
- `tests/test_rope.py`, `tests/test_attention.py`, `tests/test_model.py`, `tests/test_convert.py`, `tests/test_vision.py`, `tests/test_load.py`

## Review scope

1. **Correctness of the fixes:** Were the material findings from the prior review actually fixed correctly? Look especially at:
   - `__call__` now returns `(logits, caches)` — is the cache plumbing correct through generate()?
   - `re.search` instead of `re.match` — any edge cases where this over-matches?
   - Causal mask with cache: `full_mask[-q_len:]` — does this produce correct masking for all q_len/k_len combinations?
   - Vision 3D RoPE: `_compute_3d_rotary_pos_emb` — is the height/width frequency split correct? Does it match Qwen3-VL reference?

2. **New bugs introduced by fixes:** Did fixing the material findings break anything or introduce new issues?

3. **Weight loading readiness:** The actual Cosmos3-Nano safetensors are about to be loaded. The weight index shows names like:
   - `embed_tokens.weight`, `lm_head.weight`, `norm.weight`
   - `layers.N.self_attn.to_q.weight`, `layers.N.self_attn.to_k.weight`, `layers.N.self_attn.to_v.weight`
   - `layers.N.self_attn.to_out.weight` (NOT `to_out.0.weight`)
   - `layers.N.self_attn.norm_q.weight`, `layers.N.self_attn.norm_k.weight`
   - `layers.N.mlp.gate_proj.weight`, `layers.N.mlp.up_proj.weight`, `layers.N.mlp.down_proj.weight`
   - `layers.N.input_layernorm.weight`, `layers.N.post_attention_layernorm.weight`

   Will these map correctly to the MLX model's parameter tree? Check every layer's expected weight names against the model structure.

4. **Shape mismatches:** At full Cosmos3-Nano scale (4096 hidden, 32 heads, 8 kv heads, 128 head_dim, 12288 intermediate, 151936 vocab), will any tensor operations fail due to shape assumptions?

5. **Anything else** that would prevent real 16B weight loading and first text generation from working.

## What is NOT in scope

- Phase 2 generation pathway
- Performance optimization
- Quantization
- Style/documentation
