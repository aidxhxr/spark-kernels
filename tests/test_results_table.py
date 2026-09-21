"""CPU-only end-to-end test of scripts/make_results_table.py on rows shaped exactly like the
ones the C++ benches print (kernel names, reference rows, shape strings)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import make_results_table as mrt  # noqa: E402

DEVICE = "NVIDIA GeForce RTX 5090"


def bench_row(kernel, dtype, variant, shape, median_ms, **kw):
    return {"device": DEVICE, "kernel": kernel, "dtype": dtype, "variant": variant,
            "shape": shape, "median_ms": median_ms, "min_ms": median_ms * 0.98, "gbps": 0.0,
            "tflops": 0.0, "ref_ms": 0.0, "max_abs_err": 0.0, "max_rel_err": 0.0, "ok": True, **kw}


@pytest.fixture
def results(tmp_path, monkeypatch):
    monkeypatch.setattr(mrt, "RESULTS", tmp_path)
    monkeypatch.setattr(mrt, "OUT_MD", tmp_path / "RESULTS.md")
    monkeypatch.setattr(mrt, "OUT_HEADLINE", tmp_path / "headline.md")
    monkeypatch.setattr(sys, "argv", ["make_results_table.py"])

    def write(name, rows):
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in rows))

    return tmp_path, write


def test_cublas_row_is_a_labelled_reference_and_never_the_best_variant(results):
    out, write = results
    write("sgemm.json", [
        bench_row("sgemm_cublas", "f32", -1, "4096x4096x4096", 2.0, tflops=68.7, ref_ms=2.0),
        bench_row("sgemm", "f32", 2, "4096x4096x4096", 4.0, tflops=34.4, ref_ms=2.0),
        bench_row("sgemm", "f32", 3, "4096x4096x4096", 3.0, tflops=45.8, ref_ms=2.0),
    ])
    assert mrt.main() == 0
    md = (out / "RESULTS.md").read_text()
    assert "## sgemm_cublas" not in md  # filed under sgemm, not a kernel of its own
    assert "| f32 | 4096x4096x4096 | cuBLAS | 2.0000 |" in md
    headline = (out / "headline.md").read_text()
    assert "| sgemm | f32 | 4096x4096x4096 | v3 |" in headline  # cuBLAS is faster, still not "best"
    assert "66.7%" in headline  # 2.0 ms cuBLAS / 3.0 ms ours


def test_hgemm_rows_are_reported_in_tflops(results):
    out, write = results
    write("hgemm.json", [
        bench_row("hgemm_bf16", "bf16", 2, "4096x4096x4096", 0.5, tflops=274.9, ref_ms=0.4),
    ])
    assert mrt.main() == 0
    md = (out / "RESULTS.md").read_text()
    assert "## hgemm\n" in md and "TFLOPS" in md and "% of cuBLAS" in md
    assert "| hgemm | bf16 | 4096x4096x4096 | v2 | 0.5000 | 274.9 TFLOPS | — | 80.0% |" in (
        out / "headline.md").read_text()
