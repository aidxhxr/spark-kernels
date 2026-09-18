"""Plain-PyTorch reference implementations. Used by tests and by scripts/bench_torch.py.

They deliberately mirror the numerics of the CUDA kernels (fp32 math, cast at the end).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * inv * w.float()).to(x.dtype)


def add_rmsnorm(
    x: torch.Tensor, resid: torch.Tensor, w: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (new_resid, out) without mutating inputs. Matches the fused kernel's rounding:
    the residual is rounded to the storage dtype before normalization."""
    new_resid = (resid.float() + x.float()).to(x.dtype)
    return new_resid, rmsnorm(new_resid, w, eps)


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return (F.silu(gate.float()) * up.float()).to(gate.dtype)


def softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x.float(), dim=-1).to(x.dtype)


def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() @ b.float()).to(a.dtype)
