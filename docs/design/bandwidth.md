# Bandwidth probe (`bandwidth_copy`)

`y = x` over `n` floats. No math, so every byte the kernel moves is a byte the DRAM system
had to deliver. Its throughput is the **practical roofline ceiling** for every memory-bound
kernel in this repo (RMSNorm, fused add+RMSNorm, SwiGLU, softmax): they are reported as a
percentage of this number, not of the spec sheet.

## Why measure instead of trusting 273 GB/s

GB10 pairs its Blackwell GPU with 128 GB of LPDDR5X on a 256-bit bus at 8533 MT/s, which
works out to **273 GB/s theoretical**. Nothing achieves the theoretical figure: DRAM refresh,
page-open overhead, read/write turnaround and the unified-memory fabric all take a cut, and a
copy pays for both a read and a write stream. The number that matters for a kernel writer is
the best any streaming kernel can do, so the bench times `cudaMemcpyAsync` device-to-device
on the same buffers as `ref_ms` and reports each variant next to it.

Throughput is computed as `2 * n * 4 bytes / time` (read + write).

## Variants

| # | Design | What it teaches |
|---|--------|-----------------|
| 0 | one float per thread, one block per 256 elements | Baseline. Each lane issues a 4-byte load: a warp moves 128 B per instruction, so the SM's load/store pipeline is the limiter before DRAM is. |
| 1 | one `float4` per thread | One 128-bit transaction per lane, a warp moves 512 B per instruction. Four times fewer instructions for the same bytes, which is what lets a single SM saturate its share of DRAM bandwidth. |
| 2 | `float4` + grid-stride loop, grid fixed at ~4 blocks x 256 threads per SM | Same instruction mix as 1, but the grid no longer scales with `n`. Fewer, longer-lived blocks avoid block-scheduling overhead and the partially-filled final wave ("tail effect") that a `cdiv(n, 256)`-sized grid always has. |

Both vectorized variants require 16-byte-aligned pointers (`cudaMalloc` returns 256-byte
aligned memory) and handle `n % 4` with a one-block scalar tail launch.

### Grid-stride sizing

With 48 SMs and 4 blocks each, the grid is 192 blocks and every thread loops over the array.
The launch is "occupancy-sized" rather than "problem-sized": there is exactly one wave, every
SM stays busy until the last few iterations, and the same kernel handles any `n` up to
`int64_t` range without hitting the 2^31 grid limit. The block count is a tunable; on a
memory-bound kernel the goal is only enough loads in flight to cover DRAM latency, and 4 x 256
threads x 16 B = 16 KB in flight per SM is comfortably past that on GB10.

## Reading this in Nsight Compute

```
ncu --set full --kernel-name regex:copy_ ./build/bench_bandwidth --n=67108864 --iters=3
```

* `dram__throughput.avg.pct_of_peak_sustained_elapsed` is the headline: how close the kernel
  sits to the DRAM ceiling. Variant 0 should be visibly lower than 1 and 2.
* `dram__bytes_read.sum` / `dram__bytes_write.sum` should each equal `n * 4`; if reads are
  higher than that, something is being fetched twice.
* `smsp__inst_executed.sum` drops ~4x from variant 0 to 1: the same bytes, a quarter of the
  instructions.
* `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` vs `..._requests.sum`: the sectors-per-request
  ratio should be 4 (fully coalesced 128-byte lines) for every variant here.

## Results (GB10)

Fill in from `results/bandwidth.json` after `make bench` on the Spark.

| variant | n | median ms | GB/s | % of cudaMemcpy D2D | % of 273 GB/s |
|---------|---|-----------|------|---------------------|---------------|
| memcpy  | 64M  |  |  | 100 |  |
| 0       | 64M  |  |  |  |  |
| 1       | 64M  |  |  |  |  |
| 2       | 64M  |  |  |  |  |
| memcpy  | 256M |  |  | 100 |  |
| 0       | 256M |  |  |  |  |
| 1       | 256M |  |  |  |  |
| 2       | 256M |  |  |  |  |
