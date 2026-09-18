# Design overview

## Conventions

- **Row-major everywhere.** GEMMs compute `C[M,N] = A[M,K] · B[K,N]`. Row ops treat the input as
  `rows × cols` with `cols` contiguous.
- **`variant` is the ladder.** Variant 0 is always the naive baseline. Numbers are stable so the
  benchmarks, tests, and design docs can refer to them. The Python bindings default to the
  highest variant.
- **fp32 accumulation** for every reduction and every GEMM, regardless of I/O dtype.
- **Host validation throws** `std::invalid_argument`; CUDA errors throw `std::runtime_error`.
  The PyTorch bindings surface these as Python exceptions.
- **One 128-bit transaction per thread** is the target memory access pattern for the
  memory-bound kernels (`bf16x8`, `f32x4` in `common.cuh`).

## Target

GB10 (`sm_121`). What that means for kernel design:

| Fact | Consequence |
|---|---|
| 273 GB/s LPDDR5X | Memory-bound kernels are judged as % of this. Fusion matters more than on HBM parts. |
| ~213 TFLOPS bf16 dense (measured) | Ridge point ≈ 780 FLOP/byte; a GEMM block tile must reuse operands heavily through smem and L2. |
| 24 MB L2 | Large enough that a 4096×4096 bf16 operand (32 MB) does *not* fit, so tile-order matters. |
| `mma.sync` yes, `tcgen05`/TMA no | Tensor-core GEMMs use the WMMA API and `cp.async`, not CUTLASS 3.x SM100 pipelines. |
| 48 SMs | Grid sizes for grid-stride kernels are set from `multiProcessorCount` at runtime. |

## Measuring

- CUDA events around each launch, 10 warmup iterations, median of 100.
- Achieved bandwidth = bytes that *must* cross DRAM for the op (documented per kernel) ÷ time.
  This is a lower bound on true traffic and therefore a conservative efficiency number.
- Achieved TFLOPS = `2·M·N·K ÷ time`.
- `% of cuBLAS` = `cuBLAS time ÷ our time`, same shape, same stream, same timing loop.
- Reference correctness: CPU double precision for row ops, cuBLAS for GEMMs. Tolerances are
  stated in each bench source.

## Profiling

`scripts/profile_ncu.sh` runs Nsight Compute with `--set full`. The metrics referenced in the
design docs:

| Metric | Tells you |
|---|---|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | how close to the memory roofline |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | how close to the compute roofline |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | achieved occupancy |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` | shared-memory bank conflicts |
| `smsp__inst_executed_pipe_tensor.sum` | tensor-core utilization |
| `launch__registers_per_thread` | register pressure (occupancy limiter for register-tiled GEMM) |

## Per-kernel notes

- [bandwidth](design/bandwidth.md)
- [rmsnorm / add_rmsnorm](design/rmsnorm.md)
- [swiglu](design/swiglu.md)
- [softmax](design/softmax.md)
- [sgemm](design/sgemm.md)
- [hgemm](design/hgemm.md)
