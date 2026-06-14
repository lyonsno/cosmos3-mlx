"""Tests for CausalConv3d decomposition."""

import mlx.core as mx
import pytest

from cosmos3_mlx.conv3d import CausalConv3d


class TestCausalConv3d:
    """Test the Conv3D decomposition."""

    def test_output_shape_same_padding(self):
        """Output spatial dims should match input with same padding."""
        conv = CausalConv3d(
            in_channels=16,
            out_channels=32,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
        )
        x = mx.random.normal((1, 4, 8, 8, 16))
        out, _ = conv(x)
        mx.eval(out)
        assert out.shape == (1, 4, 8, 8, 32), f"Expected (1, 4, 8, 8, 32), got {out.shape}"

    def test_output_shape_stride(self):
        """Strided conv should reduce spatial dims."""
        conv = CausalConv3d(
            in_channels=16,
            out_channels=32,
            kernel_size=(3, 3, 3),
            stride=(1, 2, 2),
            padding=(1, 1, 1),
        )
        x = mx.random.normal((1, 4, 8, 8, 16))
        out, _ = conv(x)
        mx.eval(out)
        assert out.shape == (1, 4, 4, 4, 32), f"Expected (1, 4, 4, 4, 32), got {out.shape}"

    def test_pointwise_conv(self):
        """1x1x1 kernel should work as pointwise projection."""
        conv = CausalConv3d(
            in_channels=16,
            out_channels=32,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
        )
        x = mx.random.normal((1, 4, 8, 8, 16))
        out, _ = conv(x)
        mx.eval(out)
        assert out.shape == (1, 4, 8, 8, 32)

    def test_causal_padding(self):
        """Causal conv should not look into future frames.
        Output for frame t should be the same regardless of frames t+1, t+2, etc."""
        conv = CausalConv3d(
            in_channels=4,
            out_channels=4,
            kernel_size=(3, 1, 1),
            stride=(1, 1, 1),
            padding=(1, 0, 0),
        )

        # Process 4 frames
        x_full = mx.random.normal((1, 4, 2, 2, 4))
        out_full, _ = conv(x_full)

        # Process only first 2 frames
        x_short = x_full[:, :2]
        out_short, _ = conv(x_short)
        mx.eval(out_full, out_short)

        # First 2 output frames should be identical (causal = no future leakage)
        assert mx.allclose(out_full[:, :2], out_short, atol=1e-5).item(), \
            "Causal violation: future frames affected past output"

    def test_no_nan(self):
        """Should not produce NaN values."""
        conv = CausalConv3d(
            in_channels=8,
            out_channels=8,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
        )
        x = mx.random.normal((1, 4, 8, 8, 8))
        out, _ = conv(x)
        mx.eval(out)
        assert not mx.any(mx.isnan(out)).item()

    def test_temporal_cache(self):
        """Chunked processing with cache should match full processing."""
        conv = CausalConv3d(
            in_channels=4,
            out_channels=4,
            kernel_size=(3, 1, 1),
            stride=(1, 1, 1),
            padding=(1, 0, 0),
        )

        # Full sequence
        x = mx.random.normal((1, 6, 2, 2, 4))
        out_full, _ = conv(x)
        mx.eval(out_full)

        # Chunked: first 3 frames, then next 3 with cache
        x1 = x[:, :3]
        x2 = x[:, 3:]
        out1, cache = conv(x1)
        mx.eval(out1, cache)
        out2, _ = conv(x2, cache=cache)
        mx.eval(out2)

        out_chunked = mx.concatenate([out1, out2], axis=1)
        mx.eval(out_chunked)

        assert mx.allclose(out_full, out_chunked, atol=1e-5).item(), \
            "Chunked processing with cache diverges from full processing"
