"""CPU-only tests for scripts/shape_utils.py (the math behind docs/RESULTS.md and the roofline)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import shape_utils as su  # noqa: E402


@pytest.mark.parametrize(
    "kernel,shape,expected",
    [
        ("sgemm", "M4096_N4096_K11008", {"M": 4096, "N": 4096, "K": 11008}),
        ("hgemm", "4096", {"M": 4096, "N": 4096, "K": 4096}),
        ("rmsnorm", "4096x8192", {"rows": 4096, "cols": 8192}),
        ("softmax", "rows16_cols4096", {"rows": 16, "cols": 4096}),
        ("swiglu", "14336", {"rows": 1, "cols": 14336}),
        ("bandwidth", "n=268435456", {"n": 268435456}),
        ("bandwidth", "", {"n": 0}),
    ],
)
def test_parse_shape(kernel, shape, expected):
    assert su.parse_shape(kernel, shape) == expected


def test_traffic_bytes_counts_every_pass():
    dims = {"rows": 4096, "cols": 8192}
    elems = 4096 * 8192
    assert su.traffic_bytes("rmsnorm", "bf16", dims) == 2 * elems * 2
    assert su.traffic_bytes("rmsnorm", "f32", dims) == 2 * elems * 4
    assert su.traffic_bytes("swiglu", "bf16", dims) == 3 * elems * 2
    # read x, read + write resid, write out
    assert su.traffic_bytes("add_rmsnorm", "bf16", dims) == 4 * elems * 2
    assert su.traffic_bytes("bandwidth", "f32", {"n": 1024}) == 2 * 1024 * 4


def test_gemm_flops_and_intensity():
    dims = {"M": 1024, "N": 1024, "K": 1024}
    assert su.flops("sgemm", dims) == 2 * 1024**3
    # 2*n^3 FLOP over 3*n^2 elements of 4 bytes
    assert su.arithmetic_intensity("sgemm", "f32", "1024") == pytest.approx(2 * 1024 / 12)


def test_elementwise_kernels_sit_left_of_the_ridge():
    ridge = su.PEAKS["bf16_tflops"] * 1e12 / (su.PEAKS["bw_gbps"] * 1e9)
    assert ridge == pytest.approx(780, rel=0.01)  # the figure quoted in the README
    for kernel in ("rmsnorm", "add_rmsnorm", "softmax", "swiglu"):
        ai = su.arithmetic_intensity(kernel, "bf16", "4096x8192")
        assert 0 < ai < ridge
        assert not su.is_compute_bound_kernel(kernel)
    assert su.arithmetic_intensity("bandwidth", "f32", "1024") == 0.0


def test_load_bench_rows_skips_noise_and_torch_comparison(tmp_path):
    (tmp_path / "rmsnorm.json").write_text(
        'device: GB10\n{"kernel":"rmsnorm","variant":0}\n\n{truncated\n'
        '{"kernel":"rmsnorm","variant":1}\n'
    )
    (tmp_path / su.TORCH_COMPARISON).write_text('{"kernel":"rmsnorm","speedup":2.0}\n')
    rows = su.load_bench_rows(tmp_path)
    assert [r["variant"] for r in rows] == [0, 1]
