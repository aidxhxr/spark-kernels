# HGEMM: bf16 tensor-core GEMM

`C[M,N] = A[M,K] · B[K,N]`, row-major, bf16 inputs and outputs, fp32 accumulation.
Source: `src/kernels/hgemm.cu`. Bench: `bench_hgemm` (validates every variant against cuBLAS).

## Why WMMA / mma.sync on these GPUs

Both targets are the consumer/workstation Blackwell lineage: the RTX 5090 is compute capability
12.0 (sm_120, primary target) and the GB10 is 12.1 (sm_121, secondary). They have
fifth-generation tensor cores driven by the classic `mma.sync` warp-level instruction, which is
what the WMMA C++ API compiles to. The datacenter Blackwell parts (sm_100, B200/GB200) add
`tcgen05` instructions, TMA and thread-block clusters; those do not exist on either machine. So
WMMA / `mma.sync` is not a compromise here, it is the native path, and the same source builds
for both.

A `wmma::fragment` is a warp-distributed register tile. One `mma_sync` on
`fragment<..., 16, 16, 16, __nv_bfloat16, ...>` performs a 16×16×16 matrix multiply-accumulate
(8,192 FLOPs) per warp with fp32 accumulators. Each lane holds a compiler-defined slice of the
tile; `load_matrix_sync` / `store_matrix_sync` move whole tiles between memory and fragments
(pointer must be 32-byte aligned, `ldm` in elements and a multiple of 8 for bf16).

## The ladder

| variant | idea | what it fixes |
|---|---|---|
| 0 | one warp per 16×16 C tile, fragments loaded straight from global memory | baseline: correct use of tensor cores, zero data reuse |
| 1 | 128×128×32 block tile, 8 warps (2×4), each warp owns a 64×32 sub-tile (4×2 fragments), tile staged in shared memory with +8 padding | global traffic ÷ 8 vs v0 for the same FLOPs, no bank conflicts on fragment loads |
| 2 | v1 + two-stage `cp.async` pipeline | overlaps the global→shared copy of tile *k+1* with the tensor-core work on tile *k*; copies bypass registers |

### Arithmetic intensity of the block tile

Per K-step the block loads an A slab of 128×32 and a B slab of 32×128 bf16 = 16 KB, and performs
2·128·128·32 = 1,048,576 FLOPs:

```
AI_block = 2·128·128·32 / ((128·32 + 32·128) · 2 B) = 64 FLOP/byte
```

GB10's ridge point for bf16 tensor math is

```
213 TFLOPS / 273 GB/s ≈ 780 FLOP/byte
```

so a 128×128 tile fed from DRAM alone would be memory-bound by a factor of ~12. (The tile was
sized with the GB10 in mind; the RTX 5090's bf16 ridge is TBD, see the notes below.) The reason
it still works: neighbouring blocks share A rows and B columns, and the GB10's 24 MB L2 serves
those re-reads. For a 4096² problem the A and B matrices are 32 MB each, so L2 hit rate is what
Nsight Compute should show climbing between v0 and v1 (section *Memory Workload Analysis*,
`lts__t_sector_hit_rate`). Per-SM the tile's smem-side intensity is the number that matters:
each fragment loaded from smem is reused across 4 (A) or 2 (B) `mma_sync` calls.

### Padding

WMMA loads a 16×16 bf16 sub-tile whose rows are `ldm` elements apart. With `ldm = 32` (64 B)
rows map to the same shared-memory banks every two rows; with `ldm = 40` (80 B, +16 B pad) the
16 row starts hit 16 different bank groups. The same trick with `ldm = 136` for B. Cost: 1.5 KB
of smem. Nsight metric to watch: `l1tex__data_bank_conflicts_pipe_lsu_mem_shared`.

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

## Correctness

Inputs uniform in [-1, 1]. cuBLAS (`cublasGemmEx`, `CUBLAS_COMPUTE_32F`, bf16 in/out) is the
reference. Both results are a single bf16 rounding of an fp32 sum, so the check is
`max|C − C_ref| ≤ 0.02·max|C_ref| + 1e-3`, which admits one bf16 ulp on each side plus fp32
summation-order noise.

## RTX 5090 notes (expectations, nothing measured yet)

The constants here — 128×128×32 block tile, +8-element padding, 8 warps per tile — were reasoned
for 48 SMs, 273 GB/s and a 24 MB L2. They are correct on the 5090 and get re-derived after the
first run.

- **The roof is unknown.** NVIDIA publishes no dense bf16 tensor-core peak for the 5090 and I
  have not measured one. The GB10's 213 TFLOPS is not scaled. "% of peak" stays "—" in the
  generated tables until `--bf16-peak=<TFLOPS>` is passed to `scripts/make_results_table.py` /
  `scripts/roofline.py`; "% of cuBLAS" is the number to read until then.
- **What DRAM alone can feed.** At 64 FLOP/byte, a tile stream with zero L2 reuse supports up
  to 64 × 1,792 GB/s ≈ 115 TFLOPS at spec bandwidth (GB10: 64 × 273 ≈ 17.5). Whether that is
  above or below the tensor-core roof is exactly the TBD above, but the memory-side pressure on
  this tile size is 6.5× lower than where it was designed. If v1 turns out compute-side
  limited, `BK = 32` is already enough; if not, `BK = 64` (128 FLOP/byte, needs dynamic smem)
  is the first thing to try.
- **L2.** Size TBD (bench banner / `deviceQuery`). The "A and B do not fit" argument is
  GB10-specific; `lts__t_sector_hit_rate` between v0 and v1 shows how much reuse the 5090's L2
  actually delivers.
- **Padding.** Bank geometry is part of the sm_12x programming model, so I expect `ldm = 40` /
  `ldm = 136` to stay conflict-free. Verify: `l1tex__data_bank_conflicts_pipe_lsu_mem_shared`
  should be ≈ 0 for v1 and v2.
- **v2 vs v1.** The `cp.async` pipeline hides the global→shared tile copy behind tensor-core
  work. The copy (≈ 19 KB per tile step) is shorter on a 1,792 GB/s bus, so there is less to
  hide unless the tensor cores are faster by a similar factor. I expect a smaller v2 gain than
  on the GB10. In Nsight Compute compare the stall breakdown of v1 between tiles with
  `smsp__inst_executed_pipe_tensor.sum` per unit time.
- **Small shapes underfill 170 SMs.** One block per 128×128 tile of C: 64 blocks at 1024³,
  256 at 2048³, 1,024 at 4096³. 1024³ cannot occupy the card, and the decode-time shapes in the
  v3 list (small M, large K) are worse; split-K matters more here than on 48 SMs. Check
  `sm__warps_active.avg.pct_of_peak_sustained_active` before blaming the tile.
- 575 W card: compare `min_ms` with the median, lock clocks if 8192³ numbers wander, and run
  Nsight Compute with admin rights (see [../RTX5090.md](../RTX5090.md#measuring-on-a-geforce-card)).

## Results (RTX 5090, sm_120, CUDA 13) — fill in from `results/hgemm.json`

A GB10 (sm_121) table is added when the Spark has been benchmarked.

| shape (M×N×K) | variant | ms | TFLOPS | cuBLAS ms | % of cuBLAS |
|---|---|---|---|---|---|
| 1024³ | 0 / 1 / 2 | | | | |
| 2048³ | 0 / 1 / 2 | | | | |
| 4096³ | 0 / 1 / 2 | | | | |
| 8192³ | 0 / 1 / 2 | | | | |
| 4096×4096×11008 | 0 / 1 / 2 | | | | |
| 4096×11008×4096 | 0 / 1 / 2 | | | | |

## What a v3 would do

- Drop from WMMA to raw `mma.sync.m16n8k16` PTX with `ldmatrix` so fragment layout is explicit
  and A/B can be loaded with fewer, wider shared-memory instructions.
- Larger `BK` (64) and 3–4 pipeline stages via dynamic shared memory
  (`cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`).
- XOR-swizzled smem layout instead of padding (no wasted bytes, still conflict-free).
- Split-K for the decode-time shapes (small M, large K) where a 128×128 grid cannot fill 48 SMs,
  let alone 170.
- Persistent-block scheduling with a tile rasterization that maximises L2 reuse of A/B.
