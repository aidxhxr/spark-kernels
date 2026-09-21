"""CPU-only tests for the point placement in scripts/roofline.py (matplotlib is not needed)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import roofline  # noqa: E402


def test_bandwidth_copy_is_plotted_on_the_dram_roof_scale():
    # 1 GiB of f32 copied in 10 ms: 2 passes (read + write) -> 2 GiB of traffic
    r = {"kernel": "bandwidth", "dtype": "f32", "shape": "n=268435456", "median_ms": 10.0}
    assert roofline.plot_intensity(r) == 1.0  # not 0, which the plot would drop
    # at 1 FLOP/byte the height is bytes/s, so it compares directly against the bandwidth roof
    assert roofline.achieved_tflops(r) * 1e12 == pytest.approx(2 * 2**30 / 10e-3)
    # same point from the abbreviated shape the bench writes
    assert roofline.achieved_tflops({**r, "shape": "n=256M"}) == roofline.achieved_tflops(r)


def test_compute_kernels_use_their_arithmetic_intensity():
    r = {"kernel": "sgemm", "dtype": "f32", "shape": "M1024_N1024_K1024", "median_ms": 5.0}
    assert roofline.plot_intensity(r) == pytest.approx(2 * 1024 / 12)
    assert roofline.achieved_tflops(r) == pytest.approx(2 * 1024**3 / 5e-3 / 1e12)


def test_zero_time_rows_do_not_divide_by_zero():
    r = {"kernel": "rmsnorm", "dtype": "bf16", "shape": "4096x8192", "median_ms": 0.0}
    assert roofline.achieved_tflops(r) == 0.0
