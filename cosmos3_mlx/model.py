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

        # Final norm
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # LM head (tied with embeddings in the reference, but separate here)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[list] = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            input_ids: [batch, seq_len] token IDs
            cache: optional list of KV caches per layer

        Returns:
            logits: [batch, seq_len, vocab_size]
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

        return logits

    def generate(
        self,
        input_ids: mx.array,
        max_tokens: int = 100,
        temperature: float = 1.0,
    ) -> mx.array:
        """Autoregressive text generation with KV cache.

        Args:
            input_ids: [batch, prompt_len] prompt token IDs
            max_tokens: number of tokens to generate
            temperature: sampling temperature (0 = greedy)

        Returns:
            [batch, prompt_len + max_tokens] full token sequence
        """
        # Prefill
        cache = [None] * len(self.layers)
        logits = self.__call__(input_ids, cache=cache)

        # Initialize cache from prefill
        # The cache is already populated by the forward pass
        # We need to collect it properly
        h = self.embed_tokens(input_ids)
        batch, seq_len = input_ids.shape
        pos = mx.arange(seq_len)[None, :]
        position_ids = mx.stack([pos, pos, pos])

        caches = []
        for i, layer in enumerate(self.layers):
            h, layer_cache = layer(h, position_ids, cache=None)
            caches.append(layer_cache)

        h = self.norm(h)
        logits = self.lm_head(h)
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

            # Forward single token with cache
            h = self.embed_tokens(next_token)
            cache_len = caches[0][0].shape[1]
            pos = mx.array([[cache_len]])
            position_ids = mx.stack([pos, pos, pos])

            new_caches = []
            for i, layer in enumerate(self.layers):
                h, layer_cache = layer(h, position_ids, cache=caches[i])
                new_caches.append(layer_cache)
            caches = new_caches

            h = self.norm(h)
            logits = self.lm_head(h)
            mx.eval(logits, *[c for cache_pair in caches for c in cache_pair])

        return mx.concatenate(tokens, axis=1)
