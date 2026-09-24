# SGEMM (fp32): the optimization ladder

`C[M,N] = A[M,K] · B[K,N]`, row-major, fp32 in/out, fp32 accumulate. Every rung accepts arbitrary
`M, N, K ≥ 1`. Source: `src/kernels/sgemm.cu`; bench: `src/bench/bench_sgemm.cu`.

The point of this ladder is not to beat cuBLAS. It is to show, one change at a time, *which*
bottleneck each classic GEMM optimization removes, and to measure it on the RTX 5090 (primary
target) and later on the GB10.

## Ceiling

| | RTX 5090 | GB10 |
|---|---|---|
| FP32 lanes | 170 SMs × 128 | 48 SMs × 128 at ≈ 2.42 GHz |
| fp32 peak (CUDA cores, no tensor cores) | **123.4 TFLOPS measured** (`bench_peak`, FMA chains at 2,976 MHz; the 104.8 TFLOPS spec figure assumes 2.41 GHz) | **≈ 31 TFLOPS** |
| memory | 1,792 GB/s GDDR7 | 273 GB/s LPDDR5X |
| fp32 ridge point | 123.4e12 / 1,792e9 ≈ **69 FLOP/byte** | 31e12 / 273e9 ≈ **114 FLOP/byte** |

A GEMM does `2MNK` FLOPs over `4(MK + KN + MN)` bytes of unique traffic, at 4096³ that is
≈ 683 FLOP/byte, comfortably compute-bound on both *if* the kernel reuses data well enough. Naive
kernels do not, and that is what the rungs fix.

The ceiling that actually binds on sm_120 is not DRAM but **shared-memory bandwidth**: an SM
does 128 FMAs per clock and reads 128 B per clock from shared memory. A thread computing an
8×8 micro-tile loads 8 + 8 floats = 64 B per `k` for 64 FMAs, exactly 1 B per FMA, so the
FMA and LDS pipes have to be perfectly overlapped just to reach peak, and in practice neither
gets there. cuBLAS SGEMM sits at the same wall: 67 TFLOPS is 54% of the measured 123.4.

## The metric

```
% of cuBLAS = cuBLAS median time / our median time × 100
```

cuBLAS is run with `CUBLAS_DEFAULT_MATH` (no TF32) so both sides do real fp32 FMAs. Both sides are
timed identically (10 warmup, median of 100, CUDA events on the same stream).

## Rung 0: naive (one thread per output)

Each thread walks a full row of A and a full column of B from global memory. Per FMA it loads two
floats → **0.25 FLOP/byte** from L1/L2's point of view. Reads of `B[k][col]` across a warp are
coalesced, reads of `A[row][k]` are broadcast, but every value is re-fetched `N` (resp. `M`) times.
Pure latency-bound. (The bench skips it at 4096³ and above; at 2048³ it is 7.6 TFLOPS.)

## Rung 1: shared-memory tiling (32×32×32)

A block loads a 32×32 tile of A and B into shared memory once, then each thread does 32 FMAs from
smem per tile. Global traffic drops by 32× per operand. Arithmetic intensity vs global memory:
`2·32·32·32 / (4·(32·32 + 32·32))` = **8 FLOP/byte**. Still far from the ridge; each FMA also needs
two `LDS` instructions, so the kernel is now bound by shared-memory instruction throughput.
Measured: 9.3 TFLOPS at 4096³, 14% of cuBLAS.

## Rung 2: register tiling (128×128×8, 8×8 micro-tile per thread)

The key idea: **reuse operands in registers**. A thread loads 8 values of A and 8 values of B from
smem per `k`, then does 64 FMAs. That is 64 FMAs per 16 `LDS`, 4 FMAs per smem load, versus 0.5 in
rung 1. Global-memory intensity becomes `2·128·128·8 / (4·(128·8 + 8·128))` = **32 FLOP/byte** and
the smem-instruction bottleneck is gone. Measured: 51.0 TFLOPS at 4096³, 76% of cuBLAS, the
single biggest step on the ladder, 5.5× over rung 1.

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
  256 threads/block that caps occupancy at 2 blocks/SM (65 536 regs / (256×~110)), i.e. 16 warps ,
  plenty for a kernel whose loop is FMA-dense.
* **Edges.** Any chunk that is not fully in-bounds *and* 16-byte aligned falls back to guarded
  scalar loads with zero fill; stores are guarded the same way. Interior tiles take the vector
  path, so odd shapes cost only a partial-tile penalty.

## Rung 3: cp.async double buffering

Rung 2 serialises "load tile t+1" and "compute tile t": while a tile is being fetched the FMA units
idle. `cp.async` (Ampere+, available on sm_120 and sm_121) copies global → shared *without* passing through
registers, so the next tile's copy can be in flight while the current tile is consumed. Two smem
stages; the loop is

```
issue(t+1 → stage (t+1)&1); commit; wait_group 1; __syncthreads();
compute(stage t&1);            __syncthreads();
```

`cp.async` cannot transpose, so the A tile is stored **untransposed** in this rung (`As[m][k]`).
Fragment reads become 8 scalar `LDS` at stride `BK` instead of two `float4`. Within a warp the two
thread-rows (`ty = 0, 1`) read rows `r` and `r + 4`; with an unpadded 8-float stride those sit
`32` floats apart, the same bank, a 2-way conflict on every fragment load. Padding the row to
12 floats (48 B, still 16-byte aligned for `cp.async`) moves `r + 4` to bank `+16` and removes the
conflict. The trade is 8 `LDS.32` vs 2 `LDS.128` per `k` for A in exchange for full load/compute
overlap.

Shared memory: 2 stages × (128×12 + 8×128) × 4 B = 20.5 KB (static, well under 48 KB).

On the RTX 5090 this rung **loses** to rung 2 at 4096³ (49.0 vs 51.0 TFLOPS) and only wins at
1024³. The overlap it buys is worth little on a 1,792 GB/s bus with a 96 MB L2 (the tile fetch
is short), and the 8 scalar `LDS` per `k` are exactly the wrong direction for a kernel bound by
shared-memory bandwidth. Nsight: 130 registers and a shared-memory block limit of 1, so one
8-warp block per SM (16.7% achieved occupancy), 52.6% SM throughput.

## Rung 4: register-prefetch double buffering, transposed A, BK = 16

Rung 3's problem was that `cp.async` cannot transpose. Prefetching the next tile into
*registers* can: each thread issues its global loads for tile *t+1* before the FMAs of tile *t*
(so they are in flight across the whole compute phase, which is what the overlap needs), and
stores them to shared memory after the FMAs, transposing A on the way. That gives rung 2's inner
loop (4 × `LDS.128` per `k`) with rung 3's latency hiding, at the cost of 16 prefetch registers.
Two other changes:

* **`BK = 16`** halves the number of `__syncthreads` per FLOP (two barriers per tile step: one
  before the tile is overwritten, one after). Shared memory is a single 16 KB stage.
* **XOR swizzle of the transposed A tile.** With `BK = 16` four neighbouring threads store the
  four `k` chunks of one row, and `As[k][m]` rows are 128 floats = 4 × 32 banks, so all four
  stores hit the same bank: a 4-way conflict on every transposed store, 50 M conflicts over
  489 M shared wavefronts in the first profile. Swizzling `m ^= (k/4)·8` puts them in four
  different banks; the XOR operand is a multiple of 8, so the `float4` fragment reads stay
  contiguous and aligned, and `k` is a compile-time constant in the unrolled loop so the XOR
  folds into the address.
* **`__launch_bounds__(256, 2)`.** Left alone the compiler used 153 registers, which fits one
  block per SM: 8 warps, 16.7% achieved occupancy, issue slot active 68% of cycles, FMA pipe
  64%. Capping at 128 registers (no spills) puts two blocks on an SM; the full profile then
  shows 32% occupancy, FMA pipe 68.4%, IPC 2.85, 70% SM throughput.

Measured: 59.3 TFLOPS at 4096³, **88.5% of cuBLAS**, with the tail split-K described under
rung 5 (without it: 54.0 / 80%).

## Rung 5: 256×128 tile, 16×8 micro-tiles

The remaining lever against the shared-memory wall is bytes of smem per FMA. A 16×8 micro-tile
reads 16 + 8 floats = 96 B per `k` for 128 FMAs: **0.75 B per FMA** instead of 1. The block tile
grows to 256×128 (256 threads, 16 × 16 grid of 16×8 micro-tiles; the row groups become four
64-row quarters instead of two halves), the A stage is 256×16 floats and the B stage 16×128, and
the prefetch is 4 + 2 `float4` per thread. Accumulators alone are 128 registers; the kernel
takes 245 with `__launch_bounds__(256, 1)`, no spills, one block per SM. That is 8 warps, but
each of them has 128 independent FMAs per `k` to issue, and the profile says that is enough.

Measured: 63.3 TFLOPS at 2048³ (90.6% of cuBLAS), 56.4 at 4096³ (84%), 56.7–57.3 on the two
Llama MLP shapes (85%). The 4096³ number is *below* rung 4's, for the reason in the next
section; at 2048³ (128 tiles, under one wave) the tile pays off in full.

### Wave quantization and the split-K tail (rungs 4 and 5)

The 5090 has 170 SMs. Rung 4 runs 2 blocks per SM, so 340 tiles are resident at once; a 4096³
GEMM has 32 × 32 = 1,024 tiles = **3.01 waves**, and the 4 tiles of the fourth wave run alone
for a full tile's time: 4 waves of time for 3.01 of work, 75% efficiency. Rung 5 has 16 × 32 =
512 tiles on 170 slots: also 3.01 waves, 2 leftover tiles. 8192-class shapes see 12.05 waves
paid as 13.

Both rungs therefore split the tiles of the last partial wave along K across the idle blocks:
`tail = tiles mod resident`, `split = min(K-tiles, resident / tail)` (85 at 4096³ for both
rungs, 10 for 4096×11008×4096), each slice does `[s·KT/split, (s+1)·KT/split)` of the K loop,
and the extra wave lasts 1/`split` of a tile. For fp32 output no workspace is needed: the tail
tiles of C are zeroed with one `cudaMemset2DAsync` each on the stream, and every slice
`atomicAdd`s its partial sums into C (the adds are performed at L2, so their order does not
matter). Full-wave tiles store directly and are unaffected. `resident` comes from
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` × SM count, computed once per kernel. Those
tail tiles are not bitwise reproducible run to run; the bench's cuBLAS comparison (`1e-3`
relative) never sees the difference.

The fix took rung 4 from 49.6 to 60.6 TFLOPS at 4096³ in the tuning run, but rung 5 only from
55.0 to 56.9: with 245 registers and one block per SM there is no second block to soak up the
issue slots a straggling slice leaves idle, and 4096³ at one block per SM is more sensitive to
per-block latency than the 2048³ shape. I have not profiled the tail phase itself yet.

## What Nsight Compute shows

| Rung | Measured limiter (`--set full`, 4096³) |
|---|---|
| 0 | latency-bound, `lg_throttle` / long-scoreboard stalls |
| 1 | `MIO throttle` — shared-memory instruction issue |
| 2 | FMA pipe busy; 2 blocks/SM |
| 3 | 130 registers, smem-limited to 1 block/SM (16.7% occupancy), SM throughput 52.6% |
| 4 | 128 registers, 2 blocks/SM (32% occupancy), FMA pipe 68%, SM throughput 70%, IPC 2.85, dominant stall "waiting for the micro scheduler" |
| 5 | 245 registers, 1 block/SM, FMA-dense inner loop with 128 independent FMAs per `k` |

Profile a rung with `scripts/profile_ncu.sh build` (it runs rungs 3 and 4 at 4096³) or directly
with `ncu --set full ./build/bench_sgemm --variant=<v> --m=4096 --iters=1 --warmup=0`.

## RTX 5090 notes (measured)

The block and register tile sizes (128×128×8, 8×8) were reasoned for 48 SMs and 273 GB/s. They
are correct on the 5090 and rung 2 alone reaches 76% of cuBLAS with them; the last 12 points came
from the memory-system-specific rungs above.

* **The binding ceiling is shared memory, not DRAM.** sm_120 issues 128 FMAs and reads 128 B of
  shared memory per SM per clock; an 8×8 micro-tile needs 1 B per FMA. cuBLAS SGEMM stops at
  the same wall, 67 TFLOPS = 54% of the measured 123.4 TFLOPS fp32 peak, on every shape from
  2048³ up. Rung 5's 16×8 micro-tile is the only way past it without changing the numerics.
* **Rung 3 vs rung 2** was the open question and the answer is "rung 3 loses": the tile fetch
  is short on a 1,792 GB/s bus with a 96 MB L2 (94–96% L2 hit rate, 4–6% DRAM throughput on
  every register-tiled rung), so there is little to hide, and the 8 scalar `LDS` per `k` cost
  more than the overlap gains. Rung 4 keeps the overlap and drops the scalar loads.
* **The fp32 peak is a clock number.** 123.4 TFLOPS is 21,760 lanes × 2 FLOP × 2.84 GHz
  equivalent; the card boosts to ~2.98 GHz for the short FMA loop and sits at 2.72–2.78 GHz
  under a sustained GEMM at its 600 W limit. "% of peak" in the tables uses the measured 123.4.
* **Small shapes underfill 170 SMs.** 16 tiles at 512³ and 64 at 1024³ (32 for rung 5), so
  those rows are occupancy numbers, not kernel-quality numbers: 33% and 47% of cuBLAS at best.
  cuBLAS presumably switches to a smaller tile or split-K there.
* **The Llama MLP shapes are ~10% slower per FLOP than 4096³** on rungs 2–4 (e.g. rung 4: 59.3
  TFLOPS at 4096³ vs 51.7–52.6 with N or K = 11008), while cuBLAS is flat across them. An
  11008-float row is 44 KB, so a 128-row A tile spans 5.6 MB of address space per load instead
  of 2 MB; I have not profiled it, TLB reach is the first suspect. Rung 5, with its 256-row tiles,
  is the one rung that does not show the effect (56.4 vs 56.7–57.3).
* **Occupancy cap** confirmed: the register file is 64 K per SM, `launch__registers_per_thread`
  is 128 for rung 4 (2 blocks) and 245 for rung 5 (1 block).
* `min_ms` and the median agree to within 1–2% on every row; the 600 W limit is a steady-state
  clock, not jitter.

## Results (RTX 5090, sm_120, CUDA 13.2, driver 595.58)

From `results/sgemm.json`, median of 100 iterations, cuBLAS timed identically. A GB10 table is
added when the Spark has been benchmarked.

4096×4096×4096:

| Variant | ms | TFLOPS | % of cuBLAS |
|---|---|---|---|
| cuBLAS SGEMM | 2.0518 | 66.98 | 100 |
| 1 smem tiled | 14.7207 | 9.34 | 13.9 |
| 2 register tiled | 2.6926 | 51.04 | 76.2 |
| 3 cp.async double buffered | 2.8076 | 48.95 | 73.1 |
| 4 register prefetch, transposed A, BK=16, tail split-K | **2.3189** | **59.27** | **88.5** |
| 5 256×128 tile, 16×8 micro-tiles, tail split-K | 2.4357 | 56.43 | 84.2 |

All shapes, best rung per shape in bold (ms / TFLOPS / % of cuBLAS):

| shape (M×N×K) | cuBLAS ms / TFLOPS | v2 | v3 | v4 | v5 |
|---|---|---|---|---|---|
| 512³ | 0.0146 / 18.40 | 0.0585 / 4.59 / 25.0% | 0.0484 / 5.55 / 30.2% | 0.0545 / 4.92 / 26.8% | 0.0443 / 6.06 / 32.9% (v1 is best here: 0.0382 / 7.03 / 38.2%) |
| 1024³ | 0.0443 / 48.45 | 0.1118 / 19.21 / 39.6% | **0.0935 / 22.97 / 47.4%** | 0.1355 / 15.85 / 32.7% | 0.0950 / 22.60 / 46.6% |
| 2048³ | 0.2458 / 69.90 | 0.3228 / 53.22 / 76.1% | 0.3781 / 45.44 / 65.0% | 0.3082 / 55.75 / 79.8% | **0.2714 / 63.30 / 90.6%** |
| 4096³ | 2.0518 / 66.98 | 2.6926 / 51.04 / 76.2% | 2.8076 / 48.95 / 73.1% | **2.3189 / 59.27 / 88.5%** | 2.4357 / 56.43 / 84.2% |
| 4096×4096×11008 | 5.5698 / 66.32 | 7.4008 / 49.91 / 75.3% | 7.6101 / 48.54 / 73.2% | 7.1478 / 51.68 / 77.9% | **6.5172 / 56.68 / 85.5%** |
| 4096×11008×4096 | 5.4645 / 67.59 | 7.3249 / 50.43 / 74.6% | 7.2536 / 50.92 / 75.3% | 7.0252 / 52.58 / 77.8% | **6.4446 / 57.31 / 84.8%** |

Rungs 0 and 1 at the small shapes: v0 5.82 / 7.46 / 7.58 TFLOPS and v1 7.03 / 8.55 / 9.59 at
512³ / 1024³ / 2048³. The Python default (`variant=-1`) is rung 5. Against PyTorch eager
(`torch.matmul` in fp32, `results/torch_comparison.json`) that is 0.95× at 2048³, 0.90× at
4096³ and 0.86–0.87× on the MLP shapes; torch's cuBLAS call is a little faster than the bench's
`cublasSgemm` on the large shapes.

## What remains

* **The 11008-stride slowdown** on rungs 2–4: profile with `--section MemoryWorkloadAnalysis`
  and the L1/TLB counters, and try a grouped (L2-aware) tile order.
* **Rung 5 at 4096³** should not trail rung 4; the tail phase with one block per SM is the
  place to look (a smaller `split`, or letting rung 5's tail use rung 4's tile).
* **Small shapes**: a 64×64 tile or whole-problem split-K for 512³ / 1024³.
* **TF32 tensor cores** would roughly triple the ceiling, but that is a different precision
  contract (10-bit mantissa inputs), which is why cuBLAS is run without it here; it would be a
  separate ladder, not a rung on this one.
