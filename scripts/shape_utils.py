"""Helpers shared by make_results_table.py and roofline.py: shape parsing, peaks, traffic."""

from __future__ import annotations

import re

# GB10 (DGX Spark) ceilings used for "% of peak". See docs/GB10.md for provenance:
#   273 GB/s  LPDDR5X (256-bit @ 8533 MT/s, NVIDIA spec)
#   31 TFLOPS fp32 CUDA cores (6144 cores x 2 FLOP x 2.42 GHz, theoretical)
#   213 TFLOPS bf16/fp16 tensor cores, fp32 accumulate, dense (community mmapeak measurement)
PEAKS = {
    "bw_gbps": 273.0,
    "fp32_tflops": 31.0,
    "bf16_tflops": 213.0,
}

ITEMSIZE = {"f32": 4, "bf16": 2, "fp32": 4, "float32": 4, "bfloat16": 2}


def ints_in(s: str) -> list[int]:
    return [int(t) for t in re.findall(r"\d+", s)]


def parse_shape(kernel: str, shape: str) -> dict:
    """Interpret a shape string by kernel family.

    Accepts "4096x4096", "rows4096_cols4096", "M4096_N4096_K11008", "4096x4096x4096",
    "n=1048576" ... anything: we take the integers in order.
    """
    v = ints_in(shape)
    k = kernel.lower()
    if k in ("sgemm", "hgemm", "gemm"):
        if len(v) >= 3:
            return {"M": v[0], "N": v[1], "K": v[2]}
        if len(v) == 1:
            return {"M": v[0], "N": v[0], "K": v[0]}
    if k in ("rmsnorm", "add_rmsnorm", "softmax", "swiglu"):
        if len(v) >= 2:
            return {"rows": v[0], "cols": v[1]}
        if len(v) == 1:
            return {"rows": 1, "cols": v[0]}
    if k == "bandwidth":
        return {"n": v[0] if v else 0}
    return {"raw": v}


def traffic_bytes(kernel: str, dtype: str, dims: dict) -> float:
    """Minimum DRAM traffic for the op (used for GB/s and arithmetic intensity)."""
    isz = ITEMSIZE.get(dtype, 4)
    k = kernel.lower()
    if k in ("rmsnorm", "softmax"):
        return 2.0 * dims["rows"] * dims["cols"] * isz
    if k == "add_rmsnorm":
        return 4.0 * dims["rows"] * dims["cols"] * isz
    if k == "swiglu":
        return 3.0 * dims["rows"] * dims["cols"] * isz
    if k == "bandwidth":
        return 2.0 * dims["n"] * isz
    if k in ("sgemm", "hgemm", "gemm"):
        M, N, K = dims["M"], dims["N"], dims["K"]
        return float(M * K + K * N + M * N) * isz
    return 0.0


def flops(kernel: str, dims: dict) -> float:
    k = kernel.lower()
    if k in ("sgemm", "hgemm", "gemm"):
        return 2.0 * dims["M"] * dims["N"] * dims["K"]
    if k in ("rmsnorm", "add_rmsnorm"):
        return 4.0 * dims["rows"] * dims["cols"]
    if k == "softmax":
        return 5.0 * dims["rows"] * dims["cols"]
    if k == "swiglu":
        return 6.0 * dims["rows"] * dims["cols"]
    return 0.0


def arithmetic_intensity(kernel: str, dtype: str, shape: str) -> float:
    dims = parse_shape(kernel, shape)
    b = traffic_bytes(kernel, dtype, dims)
    return flops(kernel, dims) / b if b > 0 else 0.0


def compute_peak_tflops(kernel: str, dtype: str) -> float:
    if kernel.lower() == "hgemm" or dtype in ("bf16", "bfloat16"):
        if kernel.lower() in ("hgemm",):
            return PEAKS["bf16_tflops"]
    return PEAKS["fp32_tflops"]


def is_compute_bound_kernel(kernel: str) -> bool:
    return kernel.lower() in ("sgemm", "hgemm", "gemm")
