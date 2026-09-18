#!/usr/bin/env python3
"""Roofline plot for the GB10 with every kernel's best measured point.

Reads results/*.json (from run_all_benches.sh) and writes results/roofline.png.
Ceilings: 273 GB/s DRAM, 31 TFLOPS fp32 CUDA cores, 213 TFLOPS bf16 tensor cores.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shape_utils import (  # noqa: E402
    PEAKS,
    arithmetic_intensity,
    flops,
    parse_shape,
    traffic_bytes,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "roofline.png"

MARKERS = {"bandwidth": "P", "rmsnorm": "o", "add_rmsnorm": "D", "swiglu": "s", "softmax": "^",
           "sgemm": "v", "hgemm": "*"}


def load_rows() -> list[dict]:
    rows = []
    for p in sorted(RESULTS.glob("*.json")):
        if p.name == "torch_comparison.json":
            continue
        with p.open() as f:
            for line in f:
                line = line.strip()
                if line.startswith("{"):
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return rows


def achieved_tflops(r: dict) -> float:
    """Recompute from time so memory-bound kernels get a FLOP/s too."""
    dims = parse_shape(r["kernel"], r["shape"])
    fl = flops(r["kernel"], dims)
    if fl <= 0 and r["kernel"] == "bandwidth":
        # copy has no FLOPs; plot it at its byte-throughput as if 1 FLOP/byte for visibility
        fl = traffic_bytes(r["kernel"], r["dtype"], dims)
    return fl / (r["median_ms"] * 1e-3) / 1e12 if r["median_ms"] > 0 else 0.0


def main() -> int:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: pip install matplotlib", file=sys.stderr)
        return 1

    rows = [r for r in load_rows() if r.get("ok", True)]
    if not rows:
        print(f"no results in {RESULTS}/ — run ./scripts/run_all_benches.sh first", file=sys.stderr)
        return 1

    # best (fastest) row per (kernel, dtype, shape), then keep every shape so the plot shows
    # the trend along arithmetic intensity
    best: dict[tuple, dict] = {}
    for r in rows:
        key = (r["kernel"], r["dtype"], r["shape"])
        if key not in best or r["median_ms"] < best[key]["median_ms"]:
            best[key] = r

    bw = PEAKS["bw_gbps"] * 1e9
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    ai = [2.0**k for k in range(-4, 13)]
    ai_f = [x for x in ai]
    ax.plot(ai_f, [min(bw * x, PEAKS["fp32_tflops"] * 1e12) / 1e12 for x in ai_f], color="#444",
            lw=1.5, label=f"fp32 CUDA cores: {PEAKS['fp32_tflops']:g} TFLOPS")
    ax.plot(ai_f, [min(bw * x, PEAKS["bf16_tflops"] * 1e12) / 1e12 for x in ai_f], color="#76b900",
            lw=1.5, label=f"bf16 tensor cores: {PEAKS['bf16_tflops']:g} TFLOPS")
    ax.axvline(PEAKS["bf16_tflops"] * 1e12 / bw, color="#76b900", ls=":", lw=0.8)
    ax.axvline(PEAKS["fp32_tflops"] * 1e12 / bw, color="#444", ls=":", lw=0.8)

    by_kernel: dict[str, list[dict]] = defaultdict(list)
    for r in best.values():
        by_kernel[r["kernel"]].append(r)
    for kernel, krows in sorted(by_kernel.items()):
        xs = [arithmetic_intensity(r["kernel"], r["dtype"], r["shape"]) for r in krows]
        ys = [achieved_tflops(r) for r in krows]
        pts = [(x, y) for x, y in zip(xs, ys, strict=True) if x > 0 and y > 0]
        if not pts:
            continue
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], marker=MARKERS.get(kernel, "o"), s=55,
                   label=kernel, zorder=3, edgecolors="black", linewidths=0.4)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xlabel("arithmetic intensity (FLOP / DRAM byte)")
    ax.set_ylabel("achieved TFLOP/s")
    ax.set_title(f"GB10 (DGX Spark) roofline — DRAM {PEAKS['bw_gbps']:g} GB/s")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
