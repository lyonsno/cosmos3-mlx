"""AutoencoderKLWan (Wan2.2) Video VAE Decoder for MLX.

Decodes 48-dimensional latent tensors into video frames.
16x spatial compression, 4x temporal compression.

Architecture: CausalConv3d residual blocks with RMS normalization,
SiLU activation, and DupUp3D upsampling.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .conv3d import CausalConv3d


@dataclass
class VAEConfig:
    """Configuration for the Wan2.2 VAE."""

    z_dim: int = 48
    decoder_base_dim: int = 256
    dim_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2
    temporal_upsample: list[bool] = field(default_factory=lambda: [False, True, True])
    out_channels: int = 3
    patch_size: int = 2
    dropout: float = 0.0


class WanRMSNorm(nn.Module):
    """RMS normalization with learnable scale."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x / rms * self.weight


class WanResidualBlock(nn.Module):
    """Residual block with two CausalConv3d layers and RMSNorm."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = WanRMSNorm(in_channels)
        self.conv1 = CausalConv3d(
            in_channels, out_channels,
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),
        )
        self.norm2 = WanRMSNorm(out_channels)
        self.conv2 = CausalConv3d(
            out_channels, out_channels,
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),
        )

        # Skip connection if channel dims differ
        if in_channels != out_channels:
            self.skip = CausalConv3d(
                in_channels, out_channels,
                kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0),
            )
        else:
            self.skip = None

    def __call__(self, x: mx.array) -> mx.array:
        residual = x

        x = self.norm1(x)
        x = nn.silu(x)
        x, _ = self.conv1(x)

        x = self.norm2(x)
        x = nn.silu(x)
        x, _ = self.conv2(x)

        if self.skip is not None:
            residual, _ = self.skip(residual)

        return x + residual


def dup_up_3d(x: mx.array, temporal: bool = False) -> mx.array:
    """Duplicate-based 3D upsampling (parameter-free).

    Doubles spatial dimensions by repeating. Optionally doubles temporal.

    Args:
        x: [batch, T, H, W, C] input
        temporal: whether to upsample temporal dimension too

    Returns:
        upsampled tensor
    """
    # Spatial: repeat H and W by 2
    x = mx.repeat(x, 2, axis=2)  # H
    x = mx.repeat(x, 2, axis=3)  # W

    if temporal:
        x = mx.repeat(x, 2, axis=1)  # T

    return x


class WanUpBlock(nn.Module):
    """Upsampling block: DupUp3D + residual blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int = 2,
        temporal_upsample: bool = False,
    ):
        super().__init__()
        self.temporal_upsample = temporal_upsample

        # Residual blocks
        self.res_blocks = []
        for i in range(num_res_blocks):
            ch_in = in_channels if i == 0 else out_channels
            self.res_blocks.append(WanResidualBlock(ch_in, out_channels))

        # Post-upsample conv to refine
        self.conv_after_up = CausalConv3d(
            out_channels, out_channels,
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),
        )

    def __call__(self, x: mx.array) -> mx.array:
        for block in self.res_blocks:
            x = block(x)
            mx.eval(x)  # Aggressive eval to control memory

        # Upsample
        x = dup_up_3d(x, temporal=self.temporal_upsample)

        # Refine
        x, _ = self.conv_after_up(x)
        mx.eval(x)

        return x


class WanMidBlock(nn.Module):
    """Middle block: residual blocks (no attention for simplicity)."""

    def __init__(self, channels: int, num_res_blocks: int = 2):
        super().__init__()
        self.res_blocks = [
            WanResidualBlock(channels, channels) for _ in range(num_res_blocks)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for block in self.res_blocks:
            x = block(x)
        return x


class WanDecoder(nn.Module):
    """Wan2.2 VAE Decoder.

    Decodes latent [batch, T_lat, H_lat, W_lat, z_dim] to
    video [batch, T, H, W, 3].
    """

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config

        dim = config.decoder_base_dim
        dim_mult = config.dim_mult
        n_stages = len(dim_mult)

        # Channel progression (reversed for decoder): [4*dim, 4*dim, 2*dim, 1*dim]
        channels = [dim * m for m in dim_mult]
        dec_channels = list(reversed(channels))

        # Initial projection from latent dim
        # Unpatchify: z_dim -> z_dim * patch_size^2, then conv to top channel
        self.patch_size = config.patch_size
        initial_ch = dec_channels[0]
        self.conv_in = CausalConv3d(
            config.z_dim, initial_ch,
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),
        )

        # Middle block
        self.mid_block = WanMidBlock(initial_ch, num_res_blocks=config.num_res_blocks)

        # Upsample blocks (reversed temporal_upsample for decoder)
        temporal_up = list(reversed(config.temporal_upsample))
        self.up_blocks = []
        for i in range(n_stages - 1):
            ch_in = dec_channels[i]
            ch_out = dec_channels[i + 1]
            t_up = temporal_up[i] if i < len(temporal_up) else False
            self.up_blocks.append(
                WanUpBlock(ch_in, ch_out, config.num_res_blocks, temporal_upsample=t_up)
            )

        # Final output conv
        final_ch = dec_channels[-1]
        self.norm_out = WanRMSNorm(final_ch)
        self.conv_out = CausalConv3d(
            final_ch, config.out_channels * config.patch_size ** 2,
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),
        )

    def _unpatchify(self, x: mx.array) -> mx.array:
        """Reverse spatial patchification: expand each spatial position.

        HF packs channels as [C, p1, p2] and interleaves H with p2, W with p1.
        In channels-last: reshape to [B, T, H, W, C, p1, p2], then permute so
        p2 (dim6) interleaves with H and p1 (dim5) interleaves with W.

        Input: [B, T, H, W, C * p * p]
        Output: [B, T, H*p, W*p, C]
        """
        b, t, h, w, _ = x.shape
        p = self.config.patch_size
        c = self.config.out_channels
        x = x.reshape(b, t, h, w, c, p, p)
        x = mx.transpose(x, (0, 1, 2, 6, 3, 5, 4))  # [B, T, H, p2, W, p1, C]
        x = x.reshape(b, t, h * p, w * p, c)
        return x

    def __call__(self, z: mx.array) -> mx.array:
        """Decode latent to video.

        Args:
            z: [batch, T_lat, H_lat, W_lat, z_dim] latent tensor

        Returns:
            [batch, T, H, W, 3] decoded video (channels-last)
        """
        # Initial conv
        x, _ = self.conv_in(z)
        mx.eval(x)

        # Middle block
        x = self.mid_block(x)
        mx.eval(x)

        # Upsample blocks
        for block in self.up_blocks:
            x = block(x)

        # Output
        x = self.norm_out(x)
        x = nn.silu(x)
        x, _ = self.conv_out(x)

        # Unpatchify
        x = self._unpatchify(x)

        return x
