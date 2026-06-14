"""Cosmos 3 Mixture-of-Transformers model for MLX.

Phase 1: AR reasoner pathway only (text + vision understanding).
The generation (diffusion) pathway is defined but not wired for inference yet.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .attention import Cosmos3Attention


@dataclass
class Cosmos3Config:
    """Configuration for Cosmos 3 Nano."""

    hidden_size: int = 4096
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 12288
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5_000_000.0
    mrope_section: list[int] = field(default_factory=lambda: [24, 20, 20])
    max_position_embeddings: int = 262144


class Cosmos3MLP(nn.Module):
    """GLU/SiLU gated feed-forward network."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Cosmos3DecoderLayer(nn.Module):
    """Single MoT decoder layer with understanding + generation pathways."""

    def __init__(self, config: Cosmos3Config):
        super().__init__()

        # Understanding pathway norms
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Dual-pathway attention
        self.self_attn = Cosmos3Attention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            mrope_section=config.mrope_section,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
        )

        # Shared MLP (understanding pathway)
        self.mlp = Cosmos3MLP(config.hidden_size, config.intermediate_size)

        # Generation pathway norms and MLP (Phase 2)
        self.input_layernorm_moe_gen = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm_moe_gen = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp_moe_gen = Cosmos3MLP(config.hidden_size, config.intermediate_size)

    def __call__(
        self,
        hidden_states: mx.array,
        position_ids: mx.array,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        """Forward pass for understanding pathway only.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            position_ids: [3, batch, seq_len]
            cache: optional KV cache

        Returns:
            (output, updated_cache)
        """
        # Pre-norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention (understanding pathway only)
        attn_out, _, new_cache = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            understanding_mask=None,
            generation_tokens=None,
            cache=cache,
        )

        # Residual
        hidden_states = residual + attn_out

        # Post-attention norm + MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_cache

    def forward_with_generation(
        self,
        und_hidden: mx.array,
        gen_hidden: mx.array,
        position_ids: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Forward pass with both understanding and generation pathways.

        Args:
            und_hidden: [batch, und_len, hidden_size] understanding tokens
            gen_hidden: [batch, gen_len, hidden_size] generation tokens
            position_ids: [3, batch, und_len + gen_len]

        Returns:
            (und_output, gen_output)
        """
        # Understanding pre-norm
        und_residual = und_hidden
        und_normed = self.input_layernorm(und_hidden)

        # Generation pre-norm
        gen_residual = gen_hidden
        gen_normed = self.input_layernorm_moe_gen(gen_hidden)

        # Dual-pathway attention
        und_attn, gen_attn, _ = self.self_attn(
            hidden_states=und_normed,
            position_ids=position_ids,
            understanding_mask=None,
            generation_tokens=gen_normed,
        )

        # Understanding residual + MLP
        und_hidden = und_residual + und_attn
        und_residual = und_hidden
        und_hidden = self.post_attention_layernorm(und_hidden)
        und_hidden = self.mlp(und_hidden)
        und_hidden = und_residual + und_hidden

        # Generation residual + MLP
        gen_hidden = gen_residual + gen_attn
        gen_residual = gen_hidden
        gen_hidden = self.post_attention_layernorm_moe_gen(gen_hidden)
        gen_hidden = self.mlp_moe_gen(gen_hidden)
        gen_hidden = gen_residual + gen_hidden

        return und_hidden, gen_hidden


class Cosmos3Model(nn.Module):
    """Cosmos 3 Mixture-of-Transformers model.

    Phase 1: AR reasoner for text understanding and generation.
    """

    def __init__(self, config: Cosmos3Config):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer layers
        self.layers = [
            Cosmos3DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ]

        # Final norms (understanding + generation)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_moe_gen = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Generation pathway: latent projections
        # patch_latent_dim = latent_channel(48) * latent_patch_size(2)^2 = 192
        # These are loaded from weights; default matches Cosmos3-Nano
        self.proj_in = nn.Linear(192, config.hidden_size, bias=True)
        self.proj_out = nn.Linear(config.hidden_size, 192, bias=True)

        # Audio projections
        sound_dim = 64
        self.audio_proj_in = nn.Linear(sound_dim, config.hidden_size, bias=True)
        self.audio_proj_out = nn.Linear(config.hidden_size, sound_dim, bias=True)

        # Timestep embedder
        from .timestep import TimestepEmbedding
        self.time_embedder = TimestepEmbedding(config.hidden_size)

        # Modality embeddings
        self.audio_modality_embed = mx.zeros((config.hidden_size,))
        self.action_modality_embed = mx.zeros((config.hidden_size,))

    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[list] = None,
    ) -> tuple[mx.array, list]:
        """Forward pass.

        Args:
            input_ids: [batch, seq_len] token IDs
            cache: optional list of KV caches per layer

        Returns:
            (logits, new_caches): logits [batch, seq_len, vocab_size]
                and list of KV cache tuples per layer
        """
        batch, seq_len = input_ids.shape

        # Embed tokens
        h = self.embed_tokens(input_ids)

        # Build position IDs
        if cache is not None and cache[0] is not None:
            # During generation, offset by cache length
            cache_len = cache[0][0].shape[1]
            pos = mx.arange(cache_len, cache_len + seq_len)[None, :]
        else:
            pos = mx.arange(seq_len)[None, :]

        # All 3 axes share the same position IDs for text tokens
        position_ids = mx.stack([pos, pos, pos])  # [3, 1, seq_len]

        # Forward through layers
        new_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            h, new_cache = layer(h, position_ids, cache=layer_cache)
            new_caches.append(new_cache)

        # Final norm + LM head
        h = self.norm(h)
        logits = self.lm_head(h)

        return logits, new_caches

    def generate(
        self,
        input_ids: mx.array,
        max_tokens: int = 100,
        temperature: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> mx.array:
        """Autoregressive text generation with KV cache.

        Args:
            input_ids: [batch, prompt_len] prompt token IDs
            max_tokens: number of tokens to generate
            temperature: sampling temperature (0 = greedy)
            eos_token_id: stop generation when this token is produced

        Returns:
            [batch, prompt_len + generated] full token sequence
        """
        # Prefill: single forward pass, get logits + cache
        logits, caches = self.__call__(input_ids)
        mx.eval(logits, *[c for cache_pair in caches for c in cache_pair])

        tokens = [input_ids]

        for step in range(max_tokens):
            # Sample next token from last position
            next_logits = logits[:, -1, :]

            if temperature == 0.0:
                next_token = mx.argmax(next_logits, axis=-1, keepdims=True)
            else:
                next_token = mx.random.categorical(next_logits / temperature)
                next_token = next_token[:, None]

            tokens.append(next_token)

            # Check EOS (batch=1 only)
            if eos_token_id is not None and next_token.shape[0] == 1:
                if next_token[0, 0].item() == eos_token_id:
                    break

            # Forward single token with cache
            logits, caches = self.__call__(next_token, cache=caches)
            mx.eval(logits, *[c for cache_pair in caches for c in cache_pair])

        return mx.concatenate(tokens, axis=1)

    def diffusion_forward(
        self,
        input_ids: mx.array,
        gen_tokens: mx.array,
        timestep: mx.array,
        grid_t: int = 1,
        grid_h: int = 1,
        grid_w: int = 1,
    ) -> mx.array:
        """Forward pass for diffusion generation (one denoising step).

        Runs both understanding (text) and generation (latent) pathways
        through the dual-pathway MoT transformer. Returns velocity prediction
        for the generation tokens.

        Args:
            input_ids: [batch, text_len] text token IDs
            gen_tokens: [batch, num_patches, patch_latent_dim] patchified latents
            timestep: [batch] current diffusion timestep
            grid_t: temporal grid size (number of latent frames)
            grid_h: height grid size (latent height / patch_size)
            grid_w: width grid size (latent width / patch_size)

        Returns:
            [batch, num_patches, patch_latent_dim] velocity prediction
        """
        batch = input_ids.shape[0]
        text_len = input_ids.shape[1]
        num_patches = gen_tokens.shape[1]

        # Embed text tokens
        und_h = self.embed_tokens(input_ids)

        # Project generation tokens into hidden space
        gen_h = self.proj_in(gen_tokens)

        # Add timestep embedding to generation tokens
        # Reference applies timestep_scale=0.001 before sinusoidal encoding
        scaled_t = timestep * 0.001
        t_emb = self.time_embedder(scaled_t)  # [batch, hidden_size]
        gen_h = gen_h + mx.expand_dims(t_emb, 1)

        # Build 3D mRoPE position IDs
        # Text tokens: all 3 axes share monotonically increasing IDs
        text_pos = mx.arange(text_len)[None, :]  # [1, text_len]
        text_position_ids = mx.stack([text_pos, text_pos, text_pos])  # [3, 1, text_len]

        # Generation tokens: proper spatial grid positions
        # Temporal axis: frame index, repeated across spatial grid
        # Height axis: row index within each frame (reset to 0 per frame)
        # Width axis: column index within each frame (reset to 0 per frame)
        temporal_offset = text_len
        t_idx = (mx.arange(grid_t) + temporal_offset).reshape(-1, 1)  # [T, 1]
        t_idx = mx.broadcast_to(t_idx, (grid_t, grid_h * grid_w)).reshape(1, -1)  # [1, T*H*W]

        h_idx = mx.arange(grid_h).reshape(1, -1, 1)  # [1, H, 1]
        h_idx = mx.broadcast_to(h_idx, (grid_t, grid_h, grid_w)).reshape(1, -1)  # [1, T*H*W]

        w_idx = mx.arange(grid_w).reshape(1, 1, -1)  # [1, 1, W]
        w_idx = mx.broadcast_to(w_idx, (grid_t, grid_h, grid_w)).reshape(1, -1)  # [1, T*H*W]

        gen_position_ids = mx.stack([t_idx, h_idx, w_idx])  # [3, 1, num_patches]

        # Concatenate text + generation position IDs
        position_ids = mx.concatenate([text_position_ids, gen_position_ids], axis=2)

        # Forward through all layers with both pathways
        for layer in self.layers:
            und_h, gen_h = layer.forward_with_generation(
                und_h, gen_h, position_ids
            )

        # Apply generation final norm
        gen_h = self.norm_moe_gen(gen_h)

        # Project back to patch latent space
        velocity = self.proj_out(gen_h)

        return velocity
