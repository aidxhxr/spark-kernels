# HGEMM: bf16 tensor-core GEMM on GB10

`C[M,N] = A[M,K] · B[K,N]`, row-major, bf16 inputs and outputs, fp32 accumulation.
Source: `src/kernels/hgemm.cu`. Bench: `bench_hgemm` (validates every variant against cuBLAS).

## Why WMMA / mma.sync on this GPU

GB10 is compute capability 12.1, the consumer/workstation Blackwell lineage (same family as
sm_120 RTX cards). It has fifth-generation tensor cores driven by the classic `mma.sync`
warp-level instruction, which is what the WMMA C++ API compiles to. The datacenter Blackwell
parts (sm_100, B200/GB200) add `tcgen05` instructions, TMA and thread-block clusters; those do
not exist on GB10. So WMMA / `mma.sync` is not a compromise here, it is the native path.

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

so a 128×128 tile fed from DRAM alone would be memory-bound by a factor of ~12. The reason it
still works: neighbouring blocks share A rows and B columns, and the 24 MB L2 serves those
re-reads. For a 4096² problem the A and B matrices are 32 MB each, so L2 hit rate is what
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

## Results (GB10, sm_121, CUDA 13) — fill in from `results/hgemm.json`

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
- Split-K for the decode-time shapes (small M, large K) where a 128×128 grid cannot fill 48 SMs.
- Persistent-block scheduling with a tile rasterization that maximises L2 reuse of A/B.
