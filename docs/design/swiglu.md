# SwiGLU (`swiglu_f32`, `swiglu_bf16`)

`out[i] = silu(gate[i]) * up[i]`, where `silu(x) = x * sigmoid(x) = x / (1 + e^-x)`.

This is the gated activation between the gate/up projections and the down projection in every
Llama- and Qwen-style MLP. For a 4096-token prefill through a 14336-wide intermediate in bf16
the tensors are 117 MB each, so the op is purely memory-bound and the whole game is bytes.

## The fusion win

PyTorch eager evaluates `F.silu(gate) * up` as two kernels:

| step | reads | writes | passes over an activation-sized tensor |
|------|-------|--------|----------------------------------------|
| `F.silu(gate)` | gate | tmp | 2 |
| `tmp * up` | tmp, up | out | 3 |
| **eager total** | | | **5** (+ 2 launches, + a temporary allocation) |
| **fused** | gate, up | out | **3** (1 launch, no temporary) |

Ideal speedup from traffic alone is 5/3 = 1.67x; in practice the gap is larger because the two
eager kernels each pay launch latency and neither is guaranteed to be at the bandwidth roofline
(`torch.compile` fuses this too, which is why the Python bench compares against both eager and
compiled).

Traffic formula used for GB/s in the bench: `3 * n * sizeof(T) / time`.

## Variants

| # | Design | Notes |
|---|--------|-------|
| 0 | scalar, one element per thread, `cdiv(n, 256)` blocks | Baseline. bf16 lanes load 2 bytes each: a warp moves 64 B per load instruction, far below what the LSU pipe can issue, so the kernel is instruction-bound before it is DRAM-bound. |
| 1 | 16-byte vectorized, grid-stride, tail in-kernel | `f32x4` (4 floats) or `bf16x8` (4 x `__nv_bfloat162`) per thread per operand, one 128-bit load each. Math is done in fp32 via `__bfloat1622float2` / `__float22bfloat162_rn`. The grid is fixed at 8 blocks x 256 threads per SM and each thread strides over the array; the `n % VEC` remainder is handled by a second loop in the same kernel so there is no extra launch. |

`silu` uses `__expf`, the fast SFU exponential. Its error is well under bf16 resolution, and
for f32 it is within 2 ulp over the activation range that matters here. For `x -> -inf`,
`__expf(-x) -> inf` and `x / inf -> -0`, which is the correct limit.

Variant 1 requires 16-byte-aligned pointers (`cudaMalloc` and PyTorch allocations satisfy this;
sliced tensors may not, and the host wrapper throws `std::invalid_argument` rather than
silently misreading).

## Validation

Both variants are checked against a CPU reference computed from the *rounded* inputs (the host
converts to bf16 first, so the reference sees exactly the values the GPU sees). Pass criterion
is `|got - ref| <= atol + rtol * |ref|` with `1e-5 / 1e-5` for f32 and `2e-2 / 2e-2` for bf16.

## Nsight Compute

```
ncu --set full --kernel-name regex:swiglu_ ./build/bench_swiglu --n=58720256 --iters=3
```

Look at `dram__throughput.avg.pct_of_peak_sustained_elapsed` (should approach the bandwidth
probe's number for variant 1) and `smsp__inst_executed.sum` (should fall ~8x for bf16 between
variants 0 and 1). If `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` is more than
`3 * n * sizeof(T) / 32`, loads are not coalesced.

## Results (GB10)

Fill in from `results/swiglu.json` after `make bench`; PyTorch comparison from
`scripts/bench_torch.py`.

| dtype | shape | variant | median ms | GB/s | % of bandwidth probe | vs PyTorch eager |
|-------|-------|---------|-----------|------|----------------------|------------------|
| f32  | 4096x2048  | 0 |  |  |  |  |
| f32  | 4096x2048  | 1 |  |  |  |  |
| f32  | 4096x14336 | 0 |  |  |  |  |
| f32  | 4096x14336 | 1 |  |  |  |  |
| bf16 | 4096x2048  | 0 |  |  |  |  |
| bf16 | 4096x2048  | 1 |  |  |  |  |
| bf16 | 4096x14336 | 0 |  |  |  |  |
| bf16 | 4096x14336 | 1 |  |  |  |  |
