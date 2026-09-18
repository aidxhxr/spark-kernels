#!/usr/bin/env python3
"""Turn results/*.json (from run_all_benches.sh) and results/torch_comparison.json (from
bench_torch.py) into docs/RESULTS.md plus a short results/headline.md for the README."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shape_utils import (  # noqa: E402
    TORCH_COMPARISON,
    arithmetic_intensity,
    is_compute_bound_kernel,
    load_bench_rows,
    load_jsonl,
    parse_shape,
    peaks_for_rows,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT_MD = ROOT / "docs" / "RESULTS.md"
OUT_HEADLINE = RESULTS / "headline.md"

SHEETS = {"RTX 5090": "RTX5090.md", "GB10": "GB10.md"}
KERNEL_ORDER = ["bandwidth", "rmsnorm", "add_rmsnorm", "swiglu", "softmax", "sgemm", "hgemm"]


def load_torch_rows() -> dict[tuple, dict]:
    p = RESULTS / TORCH_COMPARISON
    if not p.exists():
        return {}
    return {(r["kernel"], r["dtype"], r["shape"]): r for r in load_jsonl(p)}


def shape_size(kernel: str, shape: str) -> float:
    d = parse_shape(kernel, shape)
    if "M" in d:
        return d["M"] * d["N"] * d["K"]
    if "rows" in d:
        return d["rows"] * d["cols"]
    return d.get("n", 0)


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}%" if x > 0 else "—"


def fmt(x: float, nd=3) -> str:
    return f"{x:.{nd}f}" if x else "—"


def vs_ref(kernel: str, r: dict) -> str:
    """ref_ms is cuBLAS for the GEMMs (shown as % of cuBLAS throughput) and the naive variant
    or cudaMemcpy for everything else (shown as a speedup)."""
    ref_ms, ms = r.get("ref_ms", 0), r["median_ms"]
    if ref_ms <= 0 or ms <= 0:
        return "—"
    return pct(ref_ms / ms) if is_compute_bound_kernel(kernel) else f"{ref_ms / ms:.2f}×"


def peak_for(kernel: str, peaks: dict) -> tuple[str, float | None]:
    """(unit, ceiling) for a kernel; the ceiling is None when it is not known for the device."""
    if is_compute_bound_kernel(kernel):
        if kernel == "hgemm":
            return "TFLOPS", peaks["bf16_tflops"]
        return "TFLOPS", peaks["fp32_tflops"]
    return "GB/s", peaks["bw_gbps"]


def table_for(kernel: str, rows: list[dict], torch_rows: dict, peaks: dict) -> str:
    unit, peak = peak_for(kernel, peaks)
    peak_label = f"{peak:g} {unit}" if peak else "peak TBD"
    has_ref = any(r.get("ref_ms", 0) > 0 for r in rows)
    has_torch = any((kernel, r["dtype"], r["shape"]) in torch_rows for r in rows)
    ref_label = "% of cuBLAS" if is_compute_bound_kernel(kernel) else "speedup vs ref"

    hdr = ["dtype", "shape", "variant", "median ms", "min ms", unit, f"% of peak ({peak_label})"]
    if has_ref:
        hdr.append(ref_label)
    if has_torch:
        hdr.append("speedup vs torch")
    hdr.append("ok")
    lines = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]

    rows = sorted(rows, key=lambda r: (r["dtype"], shape_size(kernel, r["shape"]), r["variant"]))
    for r in rows:
        val = r.get("tflops", 0) if unit == "TFLOPS" else r.get("gbps", 0)
        cells = [
            r["dtype"],
            r["shape"],
            str(r["variant"]),
            fmt(r["median_ms"], 4),
            fmt(r.get("min_ms", 0), 4),
            fmt(val, 2),
            pct(val / peak if peak else 0),
        ]
        if has_ref:
            cells.append(vs_ref(kernel, r))
        if has_torch:
            t = torch_rows.get((kernel, r["dtype"], r["shape"]))
            # torch comparison is for the fastest variant only
            best_variant = max(
                x["variant"] for x in rows if x["dtype"] == r["dtype"] and x["shape"] == r["shape"]
            )
            cells.append(f"{t['speedup']:.2f}×" if t and r["variant"] == best_variant else "—")
        cells.append("✅" if r.get("ok", True) else "❌")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def headline(by_kernel: dict[str, list[dict]], torch_rows: dict, peaks: dict) -> str:
    out = [
        "| kernel | dtype | shape | best variant | median ms | achieved | % of peak "
        "| vs cuBLAS / naive | vs torch |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for kernel in KERNEL_ORDER:
        rows = by_kernel.get(kernel)
        if not rows:
            continue
        unit, peak = peak_for(kernel, peaks)
        for dtype in sorted({r["dtype"] for r in rows}):
            drows = [r for r in rows if r["dtype"] == dtype and r.get("ok", True)]
            if not drows:
                continue
            biggest = max(shape_size(kernel, r["shape"]) for r in drows)
            cands = [r for r in drows if shape_size(kernel, r["shape"]) == biggest]
            best = min(cands, key=lambda r: r["median_ms"])
            val = best.get("tflops", 0) if unit == "TFLOPS" else best.get("gbps", 0)
            t = torch_rows.get((kernel, dtype, best["shape"]))
            out.append(
                "| {k} | {d} | {s} | v{v} | {ms:.4f} | {val:.1f} {u} | {p} | {c} | {t} |".format(
                    k=kernel, d=dtype, s=best["shape"], v=best["variant"], ms=best["median_ms"],
                    val=val, u=unit, p=pct(val / peak if peak else 0),
                    c=vs_ref(kernel, best),
                    t=f"{t['speedup']:.2f}×" if t else "—",
                )
            )
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bf16-peak", type=float, default=None, metavar="TFLOPS",
                    help="measured dense bf16 tensor-core peak, for devices where it is not "
                         "in shape_utils.DEVICE_PEAKS yet (RTX 5090)")
    args = ap.parse_args()
    rows = load_bench_rows(RESULTS)
    if not rows:
        print(f"no results in {RESULTS}/ — run ./scripts/run_all_benches.sh first", file=sys.stderr)
        return 1
    try:
        device, peaks = peaks_for_rows(rows, args.bf16_peak)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    torch_rows = load_torch_rows()
    by_kernel: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_kernel[r["kernel"]].append(r)

    bf16 = (f"{peaks['bf16_tflops']:g} TFLOPS bf16 tensor" if peaks["bf16_tflops"]
            else "bf16 tensor peak TBD (pass --bf16-peak)")
    md = ["# Benchmark results", "",
          "Generated by `scripts/make_results_table.py` from `results/*.json` "
          "(C++ benches, CUDA-event median of 100 iterations) and `results/torch_comparison.json` "
          "(PyTorch eager on the same shapes).", "",
          f"Device: **{device}**. Peaks used: {peaks['bw_gbps']:g} GB/s DRAM, "
          f"{peaks['fp32_tflops']:g} TFLOPS fp32, {bf16} "
          f"(see docs/{SHEETS.get(device, 'RTX5090.md')}).", "",
          "## Headline", "", headline(by_kernel, torch_rows, peaks), ""]
    for kernel in KERNEL_ORDER + sorted(k for k in by_kernel if k not in KERNEL_ORDER):
        if kernel not in by_kernel:
            continue
        krows = by_kernel[kernel]
        md += [f"## {kernel}", ""]
        ai_examples = sorted({(r["dtype"], r["shape"]) for r in krows}, key=lambda x: x[1])[:1]
        if ai_examples and not is_compute_bound_kernel(kernel):
            d, s = ai_examples[0]
            md.append(f"Arithmetic intensity ≈ {arithmetic_intensity(kernel, d, s):.2f} FLOP/byte "
                      f"({d}, {s}) — memory-bound; the ceiling is DRAM bandwidth.")
            md.append("")
        md += [table_for(kernel, krows, torch_rows, peaks), ""]
    OUT_MD.parent.mkdir(exist_ok=True)
    OUT_MD.write_text("\n".join(md))
    OUT_HEADLINE.write_text(headline(by_kernel, torch_rows, peaks) + "\n")
    print(f"wrote {OUT_MD} and {OUT_HEADLINE}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
