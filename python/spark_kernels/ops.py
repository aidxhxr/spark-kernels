"""Thin Python wrappers over the compiled extension (spark_kernels._C).

The C++ side flattens all leading dims of row-wise ops into `rows`; these wrappers only add
argument checking with friendlier messages and keep the output shape equal to the input.
"""

from __future__ import annotations

import torch

from . import _C

KERNELS = ("bandwidth", "rmsnorm", "swiglu", "softmax", "sgemm", "hgemm")


def num_variants(name: str) -> int:
    """Number of implementation variants for kernel `name` (see docs/DESIGN.md)."""
    if name not in KERNELS:
        raise ValueError(f"unknown kernel {name!r}; expected one of {KERNELS}")
    return _C.num_variants(name)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6, variant: int = -1) -> torch.Tensor:
    """out = x * rsqrt(mean(x**2, dim=-1) + eps) * w, computed in fp32.

    Args:
        x: [..., cols] float32 or bfloat16 CUDA tensor (contiguous).
        w: [cols] weight, same dtype as x.
        eps: numerical epsilon inside the rsqrt.
        variant: implementation index; -1 = fastest. Vectorized variants need cols % 4 == 0
            (float32) or cols % 8 == 0 (bfloat16).
    """
    return _C.rmsnorm(x, w, eps, variant)


def add_rmsnorm_(
    x: torch.Tensor, resid: torch.Tensor, w: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Fused decoder-block prologue: `resid += x` in place, then return rmsnorm(resid) * w.

    bfloat16 only; cols must be a multiple of 8. The residual stream is updated in place so
    the next block can consume it without another kernel launch.
    """
    return _C.add_rmsnorm_(x, resid, w, eps)


def swiglu(gate: torch.Tensor, up: torch.Tensor, variant: int = -1) -> torch.Tensor:
    """silu(gate) * up, elementwise, fp32 math. Any shape; gate and up must match."""
    return _C.swiglu(gate, up, variant)


def softmax(x: torch.Tensor, variant: int = -1) -> torch.Tensor:
    """Softmax over the last dim with fp32 max/sum (online softmax for variants >= 1)."""
    return _C.softmax(x, variant)


def sgemm(a: torch.Tensor, b: torch.Tensor, variant: int = -1) -> torch.Tensor:
    """fp32 GEMM: a[M,K] @ b[K,N] -> [M,N]. Any M, N, K."""
    return _C.sgemm(a, b, variant)


def hgemm(a: torch.Tensor, b: torch.Tensor, variant: int = -1) -> torch.Tensor:
    """bf16 tensor-core GEMM with fp32 accumulation: a[M,K] @ b[K,N] -> [M,N] (bf16).

    Requires M, N, K multiples of 16; variant 2 additionally requires M, N multiples of 128
    and K a multiple of 32.
    """
    return _C.hgemm(a, b, variant)
