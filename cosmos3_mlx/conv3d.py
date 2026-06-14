"""Causal 3D convolution decomposed into per-frame 2D convolutions for MLX.

MLX doesn't have efficient native Conv3D. Following the mlx-video approach,
we decompose 3D convolutions into per-frame 2D convolutions summed over the
temporal kernel dimension. This is mathematically equivalent.

Causal padding ensures convolutions only see past and current frames.
"""

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class CausalConv3d(nn.Module):
    """3D convolution with causal temporal padding, decomposed for MLX.

    Decomposes Conv3D(O, I, kd, kh, kw) into kd Conv2D operations per output frame,
    accumulated (summed) to produce the final result.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (1, 1, 1),
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        kd, kh, kw = kernel_size

        # Weight: [out_channels, kd, kh, kw, in_channels] (MLX layout)
        self.weight = mx.random.normal(
            (out_channels, kd, kh, kw, in_channels)
        ) * 0.02

        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

        # Causal temporal padding: 2*pad_d on left, 0 on right
        self._causal_pad_t = 2 * padding[0]
        self._spatial_padding = (padding[1], padding[2])

    def __call__(
        self,
        x: mx.array,
        cache: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """Forward pass.

        Args:
            x: [batch, T, H, W, C] input tensor (channels-last for MLX)
            cache: optional [batch, cache_t, H, W, C] temporal cache

        Returns:
            (output, new_cache): output [batch, T_out, H_out, W_out, C_out]
                and optional cache for next chunk
        """
        batch, t, h, w, c = x.shape
        kd, kh, kw = self.kernel_size
        sd, sh, sw = self.stride

        # Apply causal temporal padding
        if cache is not None:
            x = mx.concatenate([cache, x], axis=1)
        elif self._causal_pad_t > 0:
            pad = mx.zeros((batch, self._causal_pad_t, h, w, c), dtype=x.dtype)
            x = mx.concatenate([pad, x], axis=1)

        # Save cache for next chunk (last causal_pad_t frames)
        new_cache = None
        if self._causal_pad_t > 0:
            new_cache = x[:, -self._causal_pad_t:]

        t_padded = x.shape[1]

        # Pointwise (1x1x1) kernel shortcut
        if kd == 1 and kh == 1 and kw == 1:
            # Simple linear projection per spatiotemporal position
            if sd > 1:
                x = x[:, ::sd, ::sh, ::sw, :]
            out = x @ self.weight[0, 0, 0].T  # [O, I].T
            if self.bias is not None:
                out = out + self.bias
            return out, new_cache

        # Per-frame 2D conv accumulation
        t_out = (t_padded - kd) // sd + 1
        outputs = []

        for t_idx in range(t_out):
            t_start = t_idx * sd
            accum = None

            for d in range(kd):
                frame = x[:, t_start + d]  # [batch, H, W, C]

                # 2D conv with this temporal slice of the weight
                # weight[:, d, :, :, :] is [O, kh, kw, I]
                w_2d = self.weight[:, d, :, :, :]

                conv_out = mx.conv2d(
                    frame,
                    w_2d,
                    stride=(sh, sw),
                    padding=(self._spatial_padding[0], self._spatial_padding[1]),
                )

                if accum is None:
                    accum = conv_out
                else:
                    accum = accum + conv_out

            if self.bias is not None:
                accum = accum + self.bias

            outputs.append(accum)

        out = mx.stack(outputs, axis=1)  # [batch, T_out, H_out, W_out, C_out]
        return out, new_cache
