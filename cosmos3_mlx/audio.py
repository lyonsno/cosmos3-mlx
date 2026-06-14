"""Cosmos 3 Audio VAE Decoder (Oobleck architecture) for MLX.

Decodes 64-channel audio latents at 25 FPS into 48kHz stereo waveforms.
Uses Snake1d activations, weight-normalized Conv1d/ConvTranspose1d,
and dilated residual blocks.

Architecture follows the Cosmos3AVAEAudioTokenizer decoder from HuggingFace diffusers.
"""

import math
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn


@dataclass
class AudioDecoderConfig:
    """Configuration for the Cosmos 3 audio decoder."""

    input_dim: int = 64          # Latent channels
    dim: int = 320               # Base decoder channels
    channel_mults: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    strides: list[int] = field(default_factory=lambda: [2, 4, 5, 6, 8])
    out_channels: int = 2        # Stereo


def _conv1d_cf(conv: nn.Conv1d, x: mx.array) -> mx.array:
    """Apply MLX Conv1d to channels-first input [B, C, T] -> [B, C_out, T_out]."""
    # MLX Conv1d expects [B, T, C]
    x = mx.transpose(x, (0, 2, 1))
    x = conv(x)
    return mx.transpose(x, (0, 2, 1))


def _convt1d_cf(conv: nn.ConvTranspose1d, x: mx.array) -> mx.array:
    """Apply MLX ConvTranspose1d to channels-first input."""
    x = mx.transpose(x, (0, 2, 1))
    x = conv(x)
    return mx.transpose(x, (0, 2, 1))


class Snake1d(nn.Module):
    """Learnable Snake activation: x + (1/β) * sin²(αx).

    Periodic activation that adapts frequency per channel.
    Input/output: [batch, channels, time] (channels-first, PyTorch convention).
    Internally we keep this convention for the audio path since the
    reference model uses it. We transpose at Conv1d boundaries.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.ones((1, channels, 1))
        self.beta = mx.ones((1, channels, 1))

    def __call__(self, x: mx.array) -> mx.array:
        alpha_x = self.alpha * x
        return x + (1.0 / (self.beta + 1e-9)) * mx.sin(alpha_x) ** 2


class AudioResidualUnit(nn.Module):
    """Residual block with dilated Conv1d and Snake1d activations."""

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        # Dilated conv: kernel=7, padding = dilation * (kernel-1) // 2
        pad = dilation * 3  # (7-1)//2 * dilation
        self.snake1 = Snake1d(channels)
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=7,
            stride=1, padding=pad,
        )
        self.snake2 = Snake1d(channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=1,
        )
        self.dilation = dilation

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass. Input/output: [batch, channels, time]."""
        residual = x

        x = self.snake1(x)

        # Dilated conv: for dilation > 1, we manually gather dilated positions
        # For dilation=1, use standard conv
        if self.dilation > 1:
            x = self._dilated_conv1d(x, self.conv1, self.dilation)
        else:
            x = _conv1d_cf(self.conv1, x)

        x = self.snake2(x)
        x = _conv1d_cf(self.conv2, x)

        # Trim to match residual length if needed
        if x.shape[-1] > residual.shape[-1]:
            x = x[..., :residual.shape[-1]]
        elif x.shape[-1] < residual.shape[-1]:
            residual = residual[..., :x.shape[-1]]

        return x + residual

    def _dilated_conv1d(
        self, x: mx.array, conv: nn.Conv1d, dilation: int
    ) -> mx.array:
        """Apply dilated 1D convolution manually.

        Input/output: [batch, channels, time] (channels-first).
        """
        batch, channels, length = x.shape
        kernel_size = 7
        pad = dilation * 3

        # Pad input along time axis
        x_padded = mx.pad(x, [(0, 0), (0, 0), (pad, pad)])

        # Extract patches with dilation spacing
        patches = []
        out_len = length
        for i in range(out_len):
            indices = [i + d * dilation for d in range(kernel_size)]
            patch = mx.stack([x_padded[:, :, idx] for idx in indices], axis=-1)
            patches.append(patch)

        # [batch, channels, out_len, kernel_size]
        stacked = mx.stack(patches, axis=2)

        # MLX Conv1d weight: [out_ch, kernel, in_ch]
        weight = conv.weight
        out = mx.einsum("bctk,okc->bot", stacked, weight)

        if conv.bias is not None:
            out = out + conv.bias[:, None]

        return out


class AudioDecoderBlock(nn.Module):
    """Upsampling block: Snake + ConvTranspose1d + 3 residual units."""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.snake = Snake1d(in_channels)
        # ConvTranspose1d for upsampling
        self.upsample = nn.ConvTranspose1d(
            in_channels, out_channels,
            kernel_size=2 * stride, stride=stride, padding=stride // 2,
        )
        # Three residual units with increasing dilation
        self.res_units = [
            AudioResidualUnit(out_channels, dilation=1),
            AudioResidualUnit(out_channels, dilation=3),
            AudioResidualUnit(out_channels, dilation=9),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass. Input/output: [batch, channels, time]."""
        x = self.snake(x)
        x = _convt1d_cf(self.upsample, x)
        for unit in self.res_units:
            x = unit(x)
        return x


class AudioDecoder(nn.Module):
    """Cosmos 3 Audio VAE Decoder.

    Converts audio latents [B, 64, T_latent] to stereo waveforms [B, 2, T_audio].
    Total temporal upsample factor = product of strides (default: 2*4*5*6*8 = 1920).
    At 25 latent FPS → 48000 Hz audio.
    """

    def __init__(self, config: AudioDecoderConfig):
        super().__init__()
        self.config = config

        # Channel progression: input → channels * max_mult → ... → channels * 1
        channels = [config.dim * m for m in reversed(config.channel_mults)]
        # channels = [16*dim, 8*dim, 4*dim, 2*dim, 1*dim]

        # Initial conv from latent dim
        self.conv_in = nn.Conv1d(
            config.input_dim, channels[0], kernel_size=7, padding=3,
        )

        # Decoder blocks (progressive upsampling)
        self.blocks = []
        for i, stride in enumerate(config.strides):
            ch_in = channels[i]
            ch_out = channels[i + 1] if i + 1 < len(channels) else channels[-1]
            self.blocks.append(AudioDecoderBlock(ch_in, ch_out, stride))

        # Final output
        final_ch = channels[-1]
        self.snake_out = Snake1d(final_ch)
        self.conv_out = nn.Conv1d(
            final_ch, config.out_channels, kernel_size=7, padding=3,
        )

    def __call__(self, z: mx.array) -> mx.array:
        """Decode audio latents to waveform.

        Args:
            z: [batch, latent_channels, latent_time]

        Returns:
            [batch, 2, audio_samples] stereo waveform clamped to [-1, 1]
        """
        x = _conv1d_cf(self.conv_in, z)

        for block in self.blocks:
            x = block(x)
            mx.eval(x)  # Control memory

        x = self.snake_out(x)
        x = _conv1d_cf(self.conv_out, x)

        # Clamp to [-1, 1]
        x = mx.clip(x, -1.0, 1.0)

        return x
