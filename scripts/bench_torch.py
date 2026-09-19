#!/usr/bin/env python3
"""Time spark_kernels ops against PyTorch eager on the same shapes as the C++ benches.

Writes results/torch_comparison.json (one JSON object per row) and prints a table.
Run on the GPU box (RTX 5090 or DGX Spark) after `pip install -e . --no-build-isolation`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "torch_comparison.json"

ROW_SHAPES = [(4096, 1024), (4096, 4096), (4096, 8192), (4096, 11008), (16384, 4096)]
GEMM_SHAPES = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096), (4096, 11008, 4096)]
WARMUP, ITERS = 10, 100


def time_ms(fn, warmup=None, iters=None) -> float:
    # read the globals at call time so --warmup/--iters apply to every call site
    warmup = WARMUP if warmup is None else warmup
    iters = ITERS if iters is None else iters
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop))
    return statistics.median(samples)


def bytes_rowop(rows, cols, itemsize, n_in, n_out):
    return rows * cols * itemsize * (n_in + n_out)


def bench_row_ops(sk, add, dtype, rows, cols):
    dev = "cuda"
    dname = "f32" if dtype == torch.float32 else "bf16"
    isz = torch.tensor([], dtype=dtype).element_size()
    x = torch.randn(rows, cols, device=dev, dtype=dtype)
    w = torch.ones(cols, device=dev, dtype=dtype)
    shape = f"{rows}x{cols}"

    def gbps(n_in, n_out, ms):
        return bytes_rowop(rows, cols, isz, n_in, n_out) / ms / 1e6

    ours = time_ms(lambda: sk.rmsnorm(x, w))
    ref = time_ms(lambda: F.rms_norm(x, (cols,), w, 1e-6))
    add("rmsnorm", dname, shape, ours, ref, gbps=gbps(1, 1, ours))

    ours = time_ms(lambda: sk.softmax(x))
    ref = time_ms(lambda: torch.softmax(x, dim=-1))
    add("softmax", dname, shape, ours, ref, gbps=gbps(1, 1, ours))

    up = torch.randn(rows, cols, device=dev, dtype=dtype)
    ours = time_ms(lambda: sk.swiglu(x, up))
    ref = time_ms(lambda: F.silu(x) * up)
    add("swiglu", dname, shape, ours, ref, gbps=gbps(2, 1, ours))

    if dtype == torch.bfloat16:
        resid = torch.randn(rows, cols, device=dev, dtype=dtype)
        ours = time_ms(lambda: sk.add_rmsnorm_(x, resid, w))

        def unfused():
            resid.add_(x)
            return F.rms_norm(resid, (cols,), w, 1e-6)

        ref = time_ms(unfused)
        # traffic: read x, read+write resid, write out = 4 passes
        add("add_rmsnorm", dname, shape, ours, ref, gbps=gbps(3, 1, ours))


def bench_gemms(sk, add, M, N, K):
    dev = "cuda"
    shape = f"M{M}_N{N}_K{K}"
    flops = 2.0 * M * N * K
    a = torch.randn(M, K, device=dev)
    b = torch.randn(K, N, device=dev)
    ours = time_ms(lambda: sk.sgemm(a, b))
    ref = time_ms(lambda: a @ b)
    add("sgemm", "f32", shape, ours, ref, tflops=flops / ours / 1e9)

    ah, bh = a.to(torch.bfloat16), b.to(torch.bfloat16)
    ours = time_ms(lambda: sk.hgemm(ah, bh))
    ref = time_ms(lambda: ah @ bh)
    add("hgemm", "bf16", shape, ours, ref, tflops=flops / ours / 1e9)


def positive_int(text: str) -> int:
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")  # median of no samples is undefined
    return n


def non_negative_int(text: str) -> int:
    n = int(text)
    if n < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return n


def main() -> int:
    global WARMUP, ITERS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=positive_int, default=ITERS,
                    help="timed iterations per op (default: %(default)s, same as the C++ benches)")
    ap.add_argument("--warmup", type=non_negative_int, default=WARMUP,
                    help="untimed iterations before each measurement (default: %(default)s)")
    args = ap.parse_args()
    WARMUP, ITERS = args.warmup, args.iters
    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 1
    import spark_kernels as sk

    rows_out: list[dict] = []
    device = torch.cuda.get_device_name()
    print(f"device: {device} | cc {torch.cuda.get_device_capability()}", file=sys.stderr)

    def add(kernel, dtype, shape, ours_ms, torch_ms, gbps=None, tflops=None):
        r = {
            "device": device,
            "kernel": kernel,
            "dtype": dtype,
            "shape": shape,
            "spark_ms": ours_ms,
            "torch_ms": torch_ms,
            "speedup": torch_ms / ours_ms if ours_ms > 0 else 0.0,
        }
        if gbps is not None:
            r["gbps"] = gbps
        if tflops is not None:
            r["tflops"] = tflops
        rows_out.append(r)
        print(
            f"{kernel:12s} {dtype:5s} {shape:22s} spark {ours_ms:9.4f} ms  torch {torch_ms:9.4f} ms"
            f"  x{r['speedup']:.2f}",
            file=sys.stderr,
        )

    for dtype in (torch.float32, torch.bfloat16):
        for rows, cols in ROW_SHAPES:
            bench_row_ops(sk, add, dtype, rows, cols)
    for M, N, K in GEMM_SHAPES:
        bench_gemms(sk, add, M, N, K)

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
