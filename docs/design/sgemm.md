# SGEMM (fp32) — the optimization ladder

`C[M,N] = A[M,K] · B[K,N]`, row-major, fp32 in/out, fp32 accumulate. Every rung accepts arbitrary
`M, N, K ≥ 1`. Source: `src/kernels/sgemm.cu`; bench: `src/bench/bench_sgemm.cu`.

The point of this ladder is not to beat cuBLAS. It is to show, one change at a time, *which*
bottleneck each classic GEMM optimization removes, and to measure it on the GB10.

## Ceiling

GB10 has 48 SMs × 128 FP32 lanes at ≈2.42 GHz → **≈ 31 TFLOPS fp32 peak** (CUDA cores, no tensor
cores). Memory is 273 GB/s LPDDR5X, so the fp32 ridge point is 31e12 / 273e9 ≈ **114 FLOP/byte**.
A GEMM does `2MNK` FLOPs over `4(MK + KN + MN)` bytes of unique traffic — at 4096³ that is
≈ 680 FLOP/byte, comfortably compute-bound *if* the kernel reuses data well enough. Naive kernels do
not, and that is what the rungs fix.

## The metric

```
% of cuBLAS = cuBLAS median time / our median time × 100
```

cuBLAS is run with `CUBLAS_DEFAULT_MATH` (no TF32) so both sides do real fp32 FMAs. Both sides are
timed identically (10 warmup, median of 100, CUDA events on the same stream).

## Rung 0 — naive (one thread per output)

Each thread walks a full row of A and a full column of B from global memory. Per FMA it loads two
floats → **0.25 FLOP/byte** from L1/L2's point of view. Reads of `B[k][col]` across a warp are
coalesced, reads of `A[row][k]` are broadcast, but every value is re-fetched `N` (resp. `M`) times.
Pure latency-bound.

## Rung 1 — shared-memory tiling (32×32×32)

A block loads a 32×32 tile of A and B into shared memory once, then each thread does 32 FMAs from
smem per tile. Global traffic drops by 32× per operand. Arithmetic intensity vs global memory:
`2·32·32·32 / (4·(32·32 + 32·32))` = **8 FLOP/byte**. Still far from the ridge; each FMA also needs
two `LDS` instructions, so the kernel is now bound by shared-memory instruction throughput.

## Rung 2 — register tiling (128×128×8, 8×8 micro-tile per thread)

The key idea: **reuse operands in registers**. A thread loads 8 values of A and 8 values of B from
smem per `k`, then does 64 FMAs. That is 64 FMAs per 16 `LDS` — 4 FMAs per smem load, versus 0.5 in
rung 1. Global-memory intensity becomes `2·128·128·8 / (4·(128·8 + 8·128))` = **32 FLOP/byte** and
the smem-instruction bottleneck is gone.

Implementation notes:

* **Layout.** 256 threads → a 16×16 grid of 8×8 micro-tiles. Each thread's 8 rows are split
  `{ty·4+i}` and `{64+ty·4+i}` (same for columns). This makes a warp's fragment reads of `Bs[k][·]`
  contiguous 128-bit accesses (16 threads × 16 B = 256 B, conflict-free per 8-thread phase) and
  makes the epilogue two `float4` stores per row instead of eight scalars.
* **A stored transposed** (`As[k][m]`). The inner loop then reads 4 consecutive `m` for a fixed `k`
  as one `float4`. Cost: the tile fill does 4 scalar smem stores per thread with a mild 2-way
  bank conflict (threads with adjacent `k0` land 4·128 floats apart, same bank). It is paid once
  per tile and is dwarfed by the 512 FMAs the tile buys.
* **Registers.** 64 accumulators + 16 fragment registers + addressing ≈ 100 registers/thread. At
  256 threads/block that caps occupancy at 2 blocks/SM (65 536 regs / (256×~110)), i.e. 16 warps —
  plenty for a kernel whose loop is FMA-dense.
* **Edges.** Any chunk that is not fully in-bounds *and* 16-byte aligned falls back to guarded
  scalar loads with zero fill; stores are guarded the same way. Interior tiles take the vector
  path, so odd shapes cost only a partial-tile penalty.

## Rung 3 — cp.async double buffering

Rung 2 serialises "load tile t+1" and "compute tile t": while a tile is being fetched the FMA units
idle. `cp.async` (Ampere+, available on sm_121) copies global → shared *without* passing through
registers, so the next tile's copy can be in flight while the current tile is consumed. Two smem
stages; the loop is

```
issue(t+1 → stage (t+1)&1); commit; wait_group 1; __syncthreads();
compute(stage t&1);            __syncthreads();
```

`cp.async` cannot transpose, so the A tile is stored **untransposed** in this rung (`As[m][k]`).
Fragment reads become 8 scalar `LDS` at stride `BK` instead of two `float4`. Within a warp the two
thread-rows (`ty = 0, 1`) read rows `r` and `r + 4`; with an unpadded 8-float stride those sit
`32` floats apart — the same bank, a 2-way conflict on every fragment load. Padding the row to
12 floats (48 B, still 16-byte aligned for `cp.async`) moves `r + 4` to bank `+16` and removes the
conflict. The trade is 8 `LDS.32` vs 2 `LDS.128` per `k` for A in exchange for full load/compute
overlap; on a 273 GB/s part the overlap wins.

Shared memory: 2 stages × (128×12 + 8×128) × 4 B = 20.5 KB (static, well under 48 KB).

## What Nsight Compute should show

| Rung | Expected limiter (`ncu --section SpeedOfLight`) |
|---|---|
| 0 | very low SM throughput, `lg_throttle` / long-scoreboard stalls |
| 1 | `MIO throttle` — shared-memory instruction issue |
| 2 | FMA pipe busy, but `stall_barrier` and global-load latency exposed between tiles |
| 3 | FMA pipe busy, stalls shift to `stall_math_pipe_throttle` |

Profile a rung with `scripts/profile_ncu.sh build bench_sgemm --m=4096 --variant=<v>`.

## Results (GB10, 4096×4096×4096) — fill in with `make bench`

| Variant | ms | TFLOPS | % of cuBLAS |
|---|---|---|---|
| cuBLAS SGEMM | | | 100 |
| 0 naive | | | |
| 1 smem tiled | | | |
| 2 register tiled | | | |
| 3 cp.async double buffered | | | |

Also reported by the bench: 512³, 1024³, 2048³, and the Llama/Qwen MLP shapes 4096×4096×11008 and
4096×11008×4096.
