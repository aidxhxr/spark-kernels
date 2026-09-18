# Softmax

`softmax_f32` / `softmax_bf16` in `src/kernels/softmax.cu`. Row-wise softmax over a `rows x cols`
row-major matrix, fp32 math regardless of storage type.

## Why softmax matters here

Softmax is the normalization inside every attention head: for a query the scores against
`cols = sequence length` keys are exponentiated and normalized. Numerically it must be computed as
`exp(x - max) / sum(exp(x - max))`, otherwise `exp` overflows in fp32 past `x ≈ 88`. That subtraction
is what makes a naive implementation *three* passes over the row: max, sum, normalize.

## The optimization ladder

| Variant | Mapping | Passes over the row | Loads |
|---|---|---|---|
| 0 naive | one thread per row | 3 (max, sum, write) | scalar, uncoalesced |
| 1 warp | one warp per row | 2 (online (m, s), write) | 16-byte vectors |
| 2 block | 256 threads per row | 2 (online (m, s), write) | 16-byte vectors |

### Variant 0: three-pass, one thread per row

The reference implementation. Adjacent threads read addresses `cols` elements apart, so a warp's
32 loads hit 32 different cache lines: every byte of DRAM traffic is amplified by up to 32x, and
each thread walks its row three times. It exists only to be measured against.

### Variant 1: online softmax, one warp per row

The trick from Milakov & Gimelshein, *Online normalizer calculation for softmax* (2018): keep a
running max `m` and a running sum `s = Σ exp(x_i − m)` and update both in a single sweep. When a
new max `m'` arrives, the old sum is rescaled instead of recomputed:

```
s ← s · exp(m − m') + exp(x − m')
m ← m'
```

The same rule merges two partial states, which is what makes the algorithm parallel:

```
merge((m1, s1), (m2, s2)):
    m = max(m1, m2)
    s = s1 · exp(m1 − m) + s2 · exp(m2 − m)
```

`merge` is associative and commutative. Each lane of the warp accumulates a private `(m, s)` over
a strided slice of the row, then a five-step `__shfl_xor_sync` butterfly merges the 32 lane states
so every lane holds the row result. Lanes read 16 bytes at a time (`float4` or 8 x bf16), so one
warp instruction moves 512 contiguous bytes: full coalescing and the fewest possible load
instructions per byte, which is what a memory-bound kernel needs on GB10's 273 GB/s bus.

The write pass re-reads the row. That second read is served from L2 (a 16384-wide bf16 row is
32 KB; the GB10 has 24 MB of L2), so DRAM sees roughly one read and one write of the matrix.

**Edge case**: merging two `-inf` maxima gives `exp(-inf − (-inf)) = NaN`. The merge returns
`s = 0` when the merged max is `-inf`, which is the mathematically consistent answer for "no
finite elements seen yet" and what masked attention rows produce.

This is exactly the primitive FlashAttention builds on: because `(m, s)` partial states merge, the
attention softmax can be computed block by block over the key dimension without ever
materializing the full `rows x cols` score matrix. Variant 1 is that inner loop, isolated.

### Variant 2: online softmax, one block per row

A single warp can only keep so many 16-byte loads in flight. For long rows (attention over long
contexts, `cols ≥ 4096`) 8 warps share one row, each warp reduces its own `(m, s)` with shuffles,
lane 0 of each warp writes the pair to shared memory, and warp 0 merges the 8 pairs with the same
butterfly. Two `__syncthreads()` and 64 floats of shared memory are the whole cost. For short rows
this mapping wastes threads (256 threads for a 128-element row), so the bench reports both and
the Python binding picks by `cols`.

## Memory traffic and the roofline

Softmax does ~5 flops per element, so it is purely bandwidth-bound. Minimum traffic is one read and
one write:

```
bytes = 2 · rows · cols · sizeof(T)
GB/s  = bytes / time
```

The bench reports `gbps` against that formula, so the number to compare with is the GB10's
273 GB/s peak (see `docs/GB10.md`), or better, the achieved copy bandwidth from `bench_bandwidth`.
bf16 moves half the bytes of f32 for the same shape, so at the same GB/s it is twice as fast.

## What to look at in Nsight Compute

```
ncu --set full --kernel-name regex:softmax_ ./build/bench_softmax --cols=4096 --iters=20
```

- **Memory Workload Analysis → DRAM Throughput (% of peak)**: the headline. Variant 0 sits near
  zero because of the uncoalesced access; variants 1 and 2 should approach the copy-kernel ceiling.
- **Memory Workload Analysis → L1/TEX sectors per request**: 32 for variant 0 (one sector per
  lane), 4 for the vectorized variants (a warp's 512 bytes are 16 sectors of 32 B over 4 requests).
- **Occupancy → Achieved**: variant 1 launches 4 warps per block with a handful of registers,
  so occupancy is high; if the second pass shows L2 misses, raise `kWarpsPerBlock`.
- **Compute Workload Analysis → Pipe utilization (XU)**: `__expf` runs on the SFU; on short rows
  it can become visible, which is the reason to accumulate a local `(max, sum)` over each 16-byte
  vector before merging rather than merging element by element.
- **Warp State → Stall Long Scoreboard**: memory latency. Variant 2 exists to hide it on long
  rows by putting more loads in flight per row.

## Results (GB10)

Filled by `make bench` (`results/softmax.json`) and `scripts/make_results_table.py`.

| dtype | shape | variant | median ms | GB/s | % of 273 GB/s | speedup vs naive |
|---|---|---|---|---|---|---|
| | | | | | | |
