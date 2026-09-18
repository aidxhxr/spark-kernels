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

## Targets

Primary: RTX 5090 (`sm_120`, [RTX5090.md](RTX5090.md)). Secondary: GB10 / DGX Spark (`sm_121`,
[GB10.md](GB10.md)), to be benchmarked when the machine arrives. Same source for both; only the
arch flag and the tuning constants differ. What the two machines mean for kernel design:

| Fact | RTX 5090 | GB10 | Consequence |
|---|---|---|---|
| Memory bandwidth | 1,792 GB/s GDDR7, discrete | 273 GB/s LPDDR5X, unified | Memory-bound kernels are judged as % of this. Fusion removes whole passes and launches on both; the time saved per pass is ~6.5× larger on the GB10, where it matters more than on HBM parts. GDDR7 is much closer to HBM-class. |
| fp32 peak | ≈ 104.8 TFLOPS | ≈ 31 TFLOPS | fp32 ridge ≈ 58.5 vs 114 FLOP/byte. A 4096³ SGEMM (≈ 683 FLOP/byte) is compute-bound on both, with more margin on the 5090. |
| bf16 dense peak | TBD — not published, not measured yet | ~213 TFLOPS (community measurement) | GB10 ridge ≈ 780 FLOP/byte; a GEMM block tile must reuse operands heavily through smem and L2. 5090 ridge TBD; "% of peak" for `hgemm` stays "—" until `--bf16-peak=<TFLOPS>` is passed to the results scripts. The GB10 number is not scaled. |
| L2 | TBD — read from the bench banner / `deviceQuery` | 24 MB | GB10: a 4096×4096 bf16 operand (32 MB) does *not* fit, so tile order matters. Whether it fits on the 5090 is open until the L2 size is known. |
| `mma.sync` yes, `tcgen05`/TMA no | same | same | Tensor-core GEMMs use the WMMA API and `cp.async`, not CUTLASS 3.x SM100 pipelines. |
| SMs | 170 | 48 | Grid sizes for grid-stride kernels are set from `multiProcessorCount` at runtime. Fixed-size launches (one block per GEMM tile, one block per row) need 3.5× more blocks to fill the 5090. |

The tile sizes and crossovers in the kernels (`hgemm` 128×128×32 with +8 padding and 8 warps per
tile, `sgemm` 128×128×8 with an 8×8 register tile, `cols > 8192` / `cols > 4096` for the
block-per-row rmsnorm / softmax) were reasoned for 48 SMs and 273 GB/s. They are correct on
both machines and get re-derived for the 5090 after the first benchmark run; each per-kernel
note has an "RTX 5090 notes" section saying what I expect to move. Expectations, not results.

## Measuring

- CUDA events around each launch, 5–10 warmup launches per timing loop (`--warmup=N`
  overrides all of them), median of 100 (`--iters=N`). The first kernel of each bench process is
  also spun for ~300 ms so a cold 5090 has reached its boost clocks; `--warmup=0` disables that
  too. The minimum is recorded next to the median (`min_ms`): the 5090 boosts and
  throttles, and a median far above the min means the clocks moved during the run.
- Every JSON row carries a `device` field. The results scripts pick that device's ceilings
  from `scripts/shape_utils.py` and refuse a `results/` directory that mixes machines.
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

- hardware: [RTX 5090](RTX5090.md), [GB10](GB10.md)
- [bandwidth](design/bandwidth.md)
- [rmsnorm / add_rmsnorm](design/rmsnorm.md)
- [swiglu](design/swiglu.md)
- [softmax](design/softmax.md)
- [sgemm](design/sgemm.md)
- [hgemm](design/hgemm.md)
