"""CPU-only checks on scripts/bench_torch.py: it must time the shapes the C++ benches time,
because the results table joins the two on (kernel, dtype, shape)."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import bench_torch  # noqa: E402


def ints_after(bench: str, pattern: str) -> list[int]:
    """Integers of the first brace list that follows `pattern` in src/bench/<bench>.cu."""
    src = (ROOT / "src" / "bench" / f"{bench}.cu").read_text()
    m = re.search(pattern + r"\s*(\{[^;]*?\})\s*[;)]", src)
    assert m, f"default shape list not found in {bench}.cu; update this test and bench_torch.py"
    return [int(t) for t in re.findall(r"\d+", m.group(1))]


def test_row_kernel_widths_match_the_cpp_defaults():
    assert ints_after("bench_rmsnorm", r"col_list =") == bench_torch.RMSNORM_COLS
    assert ints_after("bench_softmax", r"cols_list =") == bench_torch.SOFTMAX_COLS
    assert ints_after("bench_swiglu", r"int cols :") == bench_torch.SWIGLU_COLS


def test_gemm_shapes_match_the_cpp_defaults():
    for bench, shapes in (("bench_sgemm", bench_torch.SGEMM_SHAPES),
                          ("bench_hgemm", bench_torch.HGEMM_SHAPES)):
        flat = ints_after(bench, r"shapes =")
        assert [tuple(flat[i:i + 3]) for i in range(0, len(flat), 3)] == shapes
