# HGEMM: bf16 tensor-core GEMM

`C[M,N] = A[M,K] · B[K,N]`, row-major, bf16 inputs and outputs, fp32 accumulation.
Source: `src/kernels/hgemm.cu`. Bench: `bench_hgemm` (validates every variant against cuBLAS).

## Why WMMA / mma.sync on these GPUs

Both targets are the consumer/workstation Blackwell lineage: the RTX 5090 is compute capability
12.0 (sm_120, primary target) and the GB10 is 12.1 (sm_121, secondary). They have
fifth-generation tensor cores driven by the classic `mma.sync` warp-level instruction, which is
what the WMMA C++ API compiles to. The datacenter Blackwell parts (sm_100, B200/GB200) add
`tcgen05` instructions and thread-block clusters; those do not exist on either machine (TMA,
`cp.async.bulk.tensor`, does exist on sm_120, but with `cp.async` already hiding the copies
behind the tensor cores it is not used here). So `mma.sync` is not a compromise, it is the
native path, and the same source builds for both.

A `wmma::fragment` is a warp-distributed register tile. One `mma_sync` on
`fragment<..., 16, 16, 16, __nv_bfloat16, ...>` performs a 16×16×16 matrix multiply-accumulate
(8,192 FLOPs) per warp with fp32 accumulators. Each lane holds a compiler-defined slice of the
tile; `load_matrix_sync` / `store_matrix_sync` move whole tiles between memory and fragments
(pointer must be 32-byte aligned, `ldm` in elements and a multiple of 8 for bf16).

Variant 3 drops below WMMA to the PTX it compiles to, `mma.sync.m16n8k16` with `ldmatrix`, so
the fragment layout is explicit and the kernel controls every shared-memory access itself.

## The ladder

| variant | idea | what it fixes |
|---|---|---|
| 0 | one warp per 16×16 C tile, fragments loaded straight from global memory | baseline: correct use of tensor cores, zero data reuse |
| 1 | 128×128×32 block tile, 8 warps (2×4), each warp owns a 64×32 sub-tile (4×2 fragments), tile staged in shared memory with +8 padding | global traffic ÷ 8 vs v0 for the same FLOPs, no bank conflicts on fragment loads |
| 2 | v1 + two-stage `cp.async` pipeline | overlaps the global→shared copy of tile *k+1* with the tensor-core work on tile *k*; copies bypass registers |
| 3 | raw `mma.sync.m16n8k16` + `ldmatrix`, XOR-swizzled smem, 3-stage `cp.async` pipeline, register-direct epilogue, split-K on the last partial wave, tile picked per call (128×128, 64×128 or 64×64) with zero-filled rows past M | fragment loads in one instruction each, no padding bytes, DRAM latency covered two tiles ahead, no epilogue staging, the wave-quantization tail on 170 SMs, and small / decode shapes (any M % 16 == 0) that a 128-row tile could not fill the card with |

### Arithmetic intensity of the block tile

Per K-step the block loads an A slab of 128×32 and a B slab of 32×128 bf16 = 16 KB, and performs
2·128·128·32 = 1,048,576 FLOPs:

```
AI_block = 2·128·128·32 / ((128·32 + 32·128) · 2 B) = 64 FLOP/byte
```

The RTX 5090's measured bf16 ridge point is

```
258.7 TFLOPS / 1,792 GB/s ≈ 144 FLOP/byte      (GB10: 213 / 273 ≈ 780)
```

so a 128×128 tile fed from DRAM alone would be memory-bound by a factor of ~2.3 on the 5090 and
~12 on the GB10. The reason it still works: neighbouring blocks share A rows and B columns, and
the L2 (96 MB on the 5090, 24 MB on the GB10) serves those re-reads. A 4096² bf16 operand is
32 MB, so on the 5090 both operands of a 4096³ GEMM sit in L2 at once; Nsight Compute reports a
96.5% L2 hit rate and ~5% of DRAM throughput for v3 even at 8192³, where they do not. Per-SM
the tile's smem-side intensity is the number that matters: each fragment loaded from smem is
reused across 4 (A) or 2 (B) `mma_sync` calls in v1/v2, and 4 (A) or 4 (B) `mma.sync` calls in
v3.

### Padding

WMMA loads a 16×16 bf16 sub-tile whose rows are `ldm` elements apart. With `ldm = 32` (64 B)
rows map to the same shared-memory banks every two rows; with `ldm = 40` (80 B, +16 B pad) the
16 row starts hit 16 different bank groups. The same trick with `ldm = 136` for B. Cost: 1.5 KB
of smem. Nsight metric to watch: `l1tex__data_bank_conflicts_pipe_lsu_mem_shared`. Measured on
the 5090: the v2 fragment *loads* are conflict-free; the only conflicts Nsight reports for v2
are on the epilogue's fp32 staging *stores* (2.2 M conflicts over 1.05 M store requests), which
v3 removes altogether.

### cp.async pipeline (v2)

```
issue(tile 0); commit
for t:
    if t+1 < T: issue(tile t+1 into other stage); commit; wait_group 1   # tile t landed
    else:       wait_group 0
    __syncthreads()
    mma over stage t&1
    __syncthreads()          # nobody still reads the stage that t+2 will overwrite
```

`cp.async.cg` copies 16 B global→shared without staging through registers, so the load
instructions do not occupy the warp's issue slots while the tensor cores run. Two stages ×
(10,240 + 8,704) B + an 8 KB fp32 epilogue staging buffer = 46,080 B, under the 48 KB static
limit. v2 requires M, N % 128 == 0 and K % 32 == 0 so every asynchronous copy is in bounds;
v1 zero-fills partial tiles and handles any multiples of 16.

### Epilogue

Accumulators are fp32 and WMMA cannot store bf16 accumulators directly, so each warp stores one
16×16 fp32 fragment to a per-warp 1 KB smem scratch, converts, and writes 8 bf16 (16 B) per lane
with a single vector store.

### Variant 3: mma.sync + ldmatrix, swizzle, three stages, tail split-K

Same 128×128×32 block tile and 2×4 warp grid as v1/v2 (warp tile 64×32, now 4×4 `m16n8`
accumulators per warp) on the large shapes; the kernel is templated on `BM`, `BN`, `BK` and
the stage count, and a 64-row tile takes over for small problems (next subsection). Four things
changed in the tile itself:

* **Raw `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` with `ldmatrix`.** The PTX
  fragment layout is documented, so each lane knows which 32-bit words it holds. One
  `ldmatrix.x4` fills a whole 16×16 A fragment from smem (four 8×8 matrices, one row address
  per lane), and one `ldmatrix.x4.trans` fills the B fragments of two adjacent `n8` tiles from
  the k-major B slab. Per k16 step per warp: 4 + 2 = 6 `ldmatrix` for 16 `mma.sync`, versus 6
  WMMA `load_matrix_sync` (each several `LDS`) for 8 `mma_sync` before.
* **XOR swizzle instead of padding.** The 16-byte chunk index of an A row (64 B rows, 4 chunks)
  is XORed with `(row/2) % 4`; of a B row (256 B rows, 16 chunks) with `row % 8`. The eight row
  addresses an `ldmatrix` touches then fall in eight different bank groups, with zero padding
  bytes: 16 KB per stage instead of 18.9 KB. Measured: 247 K bank conflicts over 403 M
  shared-load wavefronts at 8192³ (0.06%).
* **Three-stage `cp.async` pipeline** in 49 KB of dynamic shared memory (opted in with
  `cudaFuncSetAttribute`): tiles *k+1* and *k+2* are in flight while the tensor cores work on
  tile *k*, one `__syncthreads` per BK step, and the group count stays uniform by always
  committing (possibly empty) groups. A `wait_group STAGES-2` before the barrier is what makes
  the depth a compile-time constant.
* **Register-direct epilogue.** Each lane owns `(row g, cols 2c..2c+1)` and `(row g+8, same)`
  of every 16×8 accumulator, so the output is two `__nv_bfloat162` stores per tile straight
  from registers: no fp32 staging buffer, no barriers after the K loop.

The fourth change is a scheduling one, and it is the one that moved the 4096³ number the most.

#### Wave quantization and the split-K tail

The 5090 has 170 SMs and v3 fits 2 blocks per SM (121–128 registers, 49 KB smem), so 340
tiles run at once. A 4096³ GEMM has 32 × 32 = 1,024 tiles = **3.01 waves**: three full waves,
then 4 tiles run alone for as long as a full wave would. That is 4 waves of time for 3.01 waves
of work, 75% efficiency. 8192³ has 4,096 tiles = 12.05 waves, paid as 13 (93%). Before the tail
fix v3 measured 200 TFLOPS at 4096³ against cuBLAS's 226, and the arithmetic said exactly why.

The fix is split-K on the tail only, in the spirit of Stream-K: the `tiles mod 340` leftover
tiles are each split `split = min(KT, 340 / tail)` ways along K over the otherwise idle blocks
(4 tiles × 85 slices at 4096³, 16 × 21 at 8192³), so the extra wave lasts 1/`split` of a tile.
Each slice accumulates its K-range into an fp32 workspace tile with `atomicAdd` (performed at
L2, order-independent), fences, and bumps a per-tile arrival counter; the last slice to arrive
converts the finished fp32 tile to bf16 in-kernel, reading the workspace with `__ldcg` so it
sees L2 and not a stale L1 line. The workspace (`tail × 64 KB` + counters) is a per-device
static grown on demand and zeroed with `cudaMemsetAsync` on the same stream before the launch;
the full-wave tiles store bf16 directly and never touch it. Two consequences worth stating:
those tail tiles are not bitwise reproducible run to run (fp32 atomics in varying order, then
one bf16 rounding), and the tile order is row-major over N with `blockIdx.x` flattened, so the
DP tiles are exactly the first `tiles - tail` in that order.

Configuration sweep on the 5090 (BN, BK, stages), 4096³ / 8192³ TFLOPS, before the tail fix:
(128, 32, 3) **201 / 221**, shipped; (128, 32, 4) 182 / 201, the compiler dropped to 93
registers and lost throughput; (128, 64, 3) 185 / 205; (256, 32, 3) and (256, 32, 4) 166 / 203
with 186–188 registers and one block per SM; (256, 64, 3) needs 147 KB of smem and does not
fit in the 100 KB configuration.

#### Tile selection and decode shapes

The tail split fixes the *last* wave. It does nothing for a problem that never fills the
*first* one: 1024³ is 8 × 8 = 64 tiles of 128×128 on 340 slots, 0.19 of a wave, and splitting
64 tiles five ways along K leaves every block with six K-steps of work and a three-stage
prologue to amortize over them. v3 measured 52% of cuBLAS there with the 128-row tile. A
decode step is worse: with 16 tokens in flight, `M = 16`, a 128×128 tile could not even be
launched (v2 requires M % 128 == 0) and the default fell back to v1.

So `launch_auto` picks the tile per call, from the same kernel source:

| condition | tile | stages | why |
|---|---|---|---|
| M > 64 and the 128×128 grid is ≥ one full wave (resident blocks from `cudaOccupancyMaxActiveBlocksPerMultiprocessor` × SM count) | 128×128×32 | 3 | the large-shape config above |
| the 64×128 grid is ≥ one full wave | 64×128×32 | 3 | twice the tiles, half the A reuse |
| everything else, M > 64 | 64×64×64 | 3 | 1024³ becomes 256 tiles; `BK = 64` halves the per-step pipeline overhead that dominates when a block owns a dozen K-steps |
| M ≤ 64 (decode) | 64×64×64 | 4 | bound by streaming B; a fourth stage keeps more of it in flight |

Rows past M are handled without a branch in the inner loop: the A-tile copies for rows
`≥ M - bm` are issued as `cp.async` with a source size of 0 (`cp_async_16_zfill`), which reads
nothing and writes 16 zero bytes to smem, so the `mma.sync` work on those rows is on zeros and
the epilogue and the split-K paths skip them. Any M % 16 == 0 works; N % 64 == 0 and
K % 64 == 0 are required. The split-K tail applies to every tile size.

Sweep that picked the two small configs, as % of cuBLAS on 16×4096×4096 / 64×4096×4096 /
16×11008×4096 / 1024³:

| tile, BK, stages | 16×4096² | 64×4096² | 16×11008×4096 | 1024³ |
|---|---|---|---|---|
| 64×64, 32, 3 | 97.0 | 95.9 | 96.9 | 101.1 |
| 64×64, 64, 3 | 96.9 | 97.2 | 100.4 | **101.5** |
| 64×64, 32, 4 | 96.6 | 90.8 | 96.9 | 101.5 |
| 64×64, 64, 4 | **104.3** | **104.1** | 97.0 | 90.3 |
| 64×128, 32, 3 | 100.3 | 83.5 | 94.3 | 63.6 |
| 64×128, 64, 3 | 96.6 | 91.2 | 103.9 | 101.1 |

`BK = 64` with four stages wins the decode shapes and loses 1024³ to its lower occupancy
(48 KB of smem per block), hence the split by `M ≤ 64`.

A decode GEMM is a bandwidth problem: at `M = 16`, `N = K = 4096` there are 2·16·4096·4096 =
537 MFLOP against 32 MB of weights, 16 FLOP/byte, a ninth of the ridge. The kernel's job is to
stream B once at full bandwidth, and the benchmark has to let it. B alone fits in the 96 MB L2,
so timing one B back to back measured 1,730–1,880 GB/s on these shapes, above the DRAM spec,
for cuBLAS and for us alike: the weights never left L2 between iterations. A real decode step
touches every layer's weights once per token, so `bench_hgemm` now rotates through enough
copies of B to exceed L2 (256 MB+) whenever M ≤ 64, and both sides are timed the same way.
Every `hgemm` row now records `gbps = 2(MK + KN + MN) / time`, the traffic floor.

## Correctness

Inputs uniform in [-1, 1]. cuBLAS (`cublasGemmEx`, `CUBLAS_COMPUTE_32F`, bf16 in/out) is the
reference. Both results are a single bf16 rounding of an fp32 sum, so the check is
`max|C − C_ref| ≤ 0.02·max|C_ref| + 1e-3`, which admits one bf16 ulp on each side plus fp32
summation-order noise. The split-K tail tiles are covered by the same check (their sum is fp32
throughout, rounded once at the end).

## RTX 5090 notes (measured)

The constants (128×128×32 block tile, 8 warps per tile) were reasoned for 48 SMs, 273 GB/s and
a 24 MB L2. They turned out to be the right tile for the 5090 too; what the 5090 needed on top
was the mma.sync/ldmatrix rung and the tail scheduling.

- **The roof.** `bench_peak` (`results/peak.json`) measures a dense bf16 `mma.sync` peak of
  **258.7 TFLOPS at 2,976 MHz** with register-resident operands. That is the number the "% of
  peak" column uses. But a sustained GEMM hits the card's 600 W power limit within a
  millisecond or two and settles at 2.72–2.78 GHz (`nvidia-smi` sampled during the 8192³
  loop: 600 W, throttle reason 0x4), so the *practical* roof for a long GEMM is ≈ 239 TFLOPS.
  cuBLAS lands there: 233–241 TFLOPS on the large shapes. The remaining gap between v3 and
  cuBLAS at 8192³ (220 vs 233) is at least partly a perf-per-watt gap, not an instruction one:
  Nsight (which runs at a fixed 2.55 GHz) shows v3's tensor pipe **94.9% active** with the
  dominant stall `math_pipe_throttle`, i.e. the tensor cores are the bottleneck and the kernel
  is issuing to them as fast as they take work.
- **Memory side.** 64 FLOP/byte at 1,792 GB/s would allow 115 TFLOPS from DRAM alone; the
  measured 96 MB L2 (96.5% hit rate at 8192³, 81.6% in the `--set full` run with its cold
  replays) is what makes 220+ possible. DRAM throughput during v3 is ~5% of peak. `BK = 64` was
  tried and lost (see the sweep), so the smem-side intensity of `BK = 32` is enough here.
- **What each rung bought at 8192³** (TFLOPS, % of cuBLAS): v0 24.0 (10%), v1 168.1 (72%),
  v2 202.9 (87%), v3 220.0 (95%). The `cp.async` pipeline (v1→v2) was worth 21%, about as much
  as I expected it to be worth on the GB10; the tensor cores got faster by more than the bus
  did, so there was still plenty of copy to hide. In the `--set full` profiles v2 is 88.6%
  tensor-pipe utilized at 124 registers and 33% occupancy; v3 91.2% at 121 registers.
- **Small shapes.** With the 128-row tile 1024³ was 64 tiles on 170 SMs and v3 reached 52% of
  cuBLAS (63 vs 122 TFLOPS). The 64×64×64 tile makes it 256 tiles and **100.9%** (122.5
  TFLOPS). 2048³ is 256 tiles of 128×128, under one wave, and v2 and v3 both sit at 96%; the
  64×128 tile does not beat that (the sweep above), so it stays as is.
- **Decode shapes stream B at 1,160–1,470 GB/s** against a 1,532 GB/s `cudaMemcpy` roof,
  with B rotated past L2: 16×4096×4096 in 27.7 µs (104.5% of cuBLAS), 64×4096×4096 in 29.8 µs
  (102.5%), 16×11008×4096 in 62.6 µs (97.0%), 64×4096×11008 in 62.7 µs (101.7%). At 30–60 µs
  per launch the pipeline prologue and launch latency are a visible share of the time, which
  is why the 4096² weights reach a lower GB/s than the 11008-wide ones. M = 16 and M = 64
  cost the same: the tensor cores are idle either way and the zero-filled rows are free.
- **Timing hygiene.** `min_ms` and the median agree to within 1% on every row, so the 300 ms
  clock ramp and per-loop warmup are doing their job; the power-limit clock drop is a
  steady state, not jitter. Nsight Compute needs `sudo` on this GeForce card.

## Results (RTX 5090, sm_120, CUDA 13.2, driver 595.58)

From `results/hgemm.json` (median of 50 iterations; cuBLAS `cublasGemmEx` timed identically on
the same stream; for M ≤ 64 both stream B from DRAM through copies that exceed L2). v2 has no
rows for the decode shapes: it requires M % 128 == 0. A GB10 (sm_121) table is added when the
Spark has been benchmarked.

| shape (M×N×K) | cuBLAS ms / TFLOPS | v0 ms / TFLOPS / % | v1 | v2 | v3 |
|---|---|---|---|---|---|
| 1024³ | 0.0177 / 121.6 | 0.0707 / 30.4 / 25.0% | 0.0442 / 48.6 / 40.1% | 0.0400 / 53.7 / 44.2% | **0.0175 / 122.5 / 100.9%** |
| 2048³ | 0.0996 / 172.6 | 0.6581 / 26.1 / 15.1% | 0.1344 / 127.9 / 74.0% | 0.1035 / 166.0 / 96.2% | **0.1035 / 166.1 / 96.3%** |
| 4096³ | 0.6091 / 225.7 | 5.4217 / 25.4 / 11.2% | 0.9091 / 151.2 / 67.0% | 0.7564 / 181.7 / 80.5% | **0.6111 / 224.9 / 99.7%** |
| 8192³ | 4.7306 / 232.4 | 45.7073 / 24.1 / 10.3% | 6.5359 / 168.2 / 72.5% | 5.4208 / 202.8 / 87.3% | **4.9877 / 220.4 / 94.9%** |
| 4096×4096×11008 | 1.6267 / 227.1 | 15.7354 / 23.5 / 10.3% | 2.3978 / 154.0 / 67.8% | 2.0425 / 180.8 / 79.7% | **1.6433 / 224.8 / 98.8%** |
| 4096×11008×4096 | 1.5427 / 239.4 | 15.2551 / 24.2 / 10.1% | 2.2055 / 167.5 / 70.0% | 1.8501 / 199.6 / 83.5% | **1.6453 / 224.5 / 93.8%** |
| 16×4096×4096 | 0.0298 / 18.0 | 0.1035 / 5.2 / 28.8% | 0.1794 / 3.0 / 16.0% | — | **0.0277 / 19.4 / 104.5%** |
| 64×4096×4096 | 0.0307 / 70.1 | 0.1058 / 20.3 / 29.0% | 0.1815 / 11.8 / 16.5% | — | **0.0298 / 72.2 / 102.5%** |
| 16×11008×4096 | 0.0607 / 23.8 | 0.1156 / 12.5 / 52.5% | 0.1812 / 8.0 / 33.5% | — | **0.0626 / 23.1 / 97.0%** |
| 64×4096×11008 | 0.0664 / 86.9 | 0.2815 / 20.5 / 23.6% | 0.4822 / 12.0 / 13.2% | — | **0.0627 / 92.0 / 101.7%** |

"%" is cuBLAS time ÷ our time. The decode rows in GB/s (2(MK + KN + MN) ÷ time): 1,220,
1,163, 1,449 and 1,469 respectively, against 1,532 GB/s for `cudaMemcpy`. The PyTorch eager
comparison (`torch.matmul` in bf16, which calls cuBLAS/cuBLASLt through its own heuristics) is
in `results/torch_comparison.json` and `docs/RESULTS.md`; on the earlier run of the same
kernel v3 was 1.01× at 4096³, 1.02× at 4096×4096×11008 and 0.97× at 8192³.

## What was done and what remains

Done in v3, from the list I had written before the first run:

- WMMA → raw `mma.sync.m16n8k16` + `ldmatrix`, explicit fragment layout, fewer and wider
  shared-memory instructions.
- Dynamic shared memory with a 3-stage pipeline. Larger `BK` (64) was tried and did not help.
- XOR-swizzled smem instead of padding.
- Split-K, in the tail-only form that the wave arithmetic on 170 SMs actually calls for.
- Small-shape tiles: 64×128 and 64×64×64 picked per call, which took 1024³ from 52% to 101% of
  cuBLAS.
- Decode shapes: zero-filled rows past M, so M = 16..64 runs on the same kernel at 97–105% of
  cuBLAS with the weights streamed from DRAM.

Still open:

- **Stream-K proper**, i.e. persistent blocks each owning an equal share of the flattened
  (tile, k-step) space, would handle every shape uniformly instead of only the last partial
  wave, and would replace the per-call tile heuristic with one schedule.
- **A 16- or 32-row tile for M ≤ 32.** The 64-row tile does 4× the `mma.sync` work at M = 16;
  it is free today because the tensor cores are idle, but it costs registers and smem that
  could go to a deeper pipeline on the B stream.
- **Perf per watt** on the long shapes: a 64×64 warp tile (4 warps per 128×128 block, or 8 warps
  on 128×256) halves the `ldmatrix` count per `mma.sync`. My 128×256 attempt lost with the
  current 2×4 warp grid; a 2×2 grid with 4 warps and 2–3 blocks per SM is the untested layout.
  8192³ at 95% of cuBLAS is the row it would move.
- **Persistent scheduling with L2-aware rasterization** (grouped tile order), which matters once
  operands exceed the 96 MB L2 (8192³ and up).
