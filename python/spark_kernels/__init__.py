"""spark-kernels: hand-written CUDA kernels for LLM inference on the NVIDIA DGX Spark.

Ops (all run on the current CUDA stream, all accept float32 or bfloat16 unless noted):

    rmsnorm(x, w, eps=1e-6, variant=-1)      RMSNorm over the last dim
    add_rmsnorm_(x, resid, w, eps=1e-6)      resid += x; return rmsnorm(resid)  (bf16, in place)
    swiglu(gate, up, variant=-1)             silu(gate) * up
    softmax(x, variant=-1)                   softmax over the last dim, fp32 math
    sgemm(a, b, variant=-1)                  fp32 GEMM (a @ b)
    hgemm(a, b, variant=-1)                  bf16 tensor-core GEMM (a @ b), fp32 accumulate
    num_variants(name)                       how many implementations exist for `name`

`variant` selects a rung on the optimization ladder described in docs/DESIGN.md; -1 picks
the fastest. `spark_kernels.reference` holds plain-PyTorch implementations used by tests.
"""

from . import reference
from .ops import (
    add_rmsnorm_,
    hgemm,
    num_variants,
    rmsnorm,
    sgemm,
    softmax,
    swiglu,
)

__all__ = [
    "add_rmsnorm_",
    "hgemm",
    "num_variants",
    "reference",
    "rmsnorm",
    "sgemm",
    "softmax",
    "swiglu",
]
__version__ = "0.1.0"
