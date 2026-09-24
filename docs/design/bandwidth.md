# Bandwidth probe (`bandwidth_copy`)

`y = x` over `n` floats. No math, so every byte the kernel moves is a byte the DRAM system
had to deliver. Its throughput is the **practical roofline ceiling** for every memory-bound
kernel in this repo (RMSNorm, fused add+RMSNorm, SwiGLU, softmax): they are reported as a
percentage of this number, not of the spec sheet.

## Why measure instead of trusting the spec sheet

The RTX 5090 (primary target) has 32 GB of GDDR7 on a 512-bit bus at 28 Gbps, which works out
to **1,792 GB/s theoretical**. The GB10 (secondary target) pairs its Blackwell GPU with 128 GB
of LPDDR5X on a 256-bit bus at 8533 MT/s: **273 GB/s theoretical**. Nothing achieves the
theoretical figure on either: DRAM refresh, page-open overhead, read/write turnaround and (on
the GB10) the unified-memory fabric all take a cut, and a copy pays for both a read and a
write stream. The number that matters for a kernel writer is
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

The SM count comes from `multiProcessorCount` at runtime. With 4 blocks per SM the grid is
680 blocks on the RTX 5090 (170 SMs) and 192 on the GB10 (48 SMs), and every thread loops over
the array.
The launch is "occupancy-sized" rather than "problem-sized": there is exactly one wave, every
SM stays busy until the last few iterations, and the same kernel handles any `n` up to
`int64_t` range without hitting the 2^31 grid limit. The block count is a tunable; on a
memory-bound kernel the goal is only enough loads in flight to cover DRAM latency, and 4 x 256
threads x 16 B = 16 KB in flight per SM is comfortably past that on GB10, and turned out to be
enough on GDDR7 too (see the RTX 5090 notes).

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

## RTX 5090 notes (measured)

* **`cudaMemcpy` D2D lands at 1,527–1,530 GB/s, 85.3% of the 1,792 GB/s spec.** That is the
  number every memory-bound kernel in this repo is judged against: the row kernels reach
  1,500–1,580 GB/s on their DRAM-bound shapes, i.e. 98–103% of it (SwiGLU exceeds the copy
  slightly because two reads per write turn the bus around less often than one read per write).
* **Variant 0 is not instruction-bound here.** The prediction was a wider gap between the
  scalar and vectorized variants than on the GB10; the measurement is a tie: 1,540 vs
  1,538 GB/s at 64M elements, 1,532 vs 1,532 at 256M. A warp's 128 B scalar loads are issued
  fast enough on sm_120 to keep the SM's ≈ 10.5 GB/s share of the bus busy, so the win from
  `float4` shows up in instruction counts, not in time. (It does show up in time for the
  bf16 row kernels, whose scalar loads are 64 B per warp.)
* **The grid-stride variant is 2–3% slower** (1,494 GB/s). 4 × 256 threads × 16 B = 16 KB in
  flight per SM is enough to cover GDDR7 latency, so the "blocks per SM" constant is not the
  problem; the problem is that 680 long-lived blocks over 170 SMs finish unevenly at the end
  of the array, while the problem-sized grid of variant 1 (262,144 blocks of 256 `float4`
  threads at 256M elements) lets the block scheduler balance the tail at a finer grain. The
  blocks-per-SM sweep was not needed and was not run.
* **The L2 is 96 MB.** Both bench sizes (256 MB and 1 GiB per buffer) are well past it, so
  these are DRAM numbers; the row-kernel benches include shapes that are *not* and report L2
  bandwidth there (2–6 TB/s), which is called out in each of those docs.
* `min_ms` is within 0.7% of the median on every row: the copy never reaches the 600 W power
  limit and the 300 ms ramp is enough.

## Results (RTX 5090)

From `results/bandwidth.json`: median and minimum of 50 iterations, CUDA 13.2, driver 595.58,
no display attached, power limit 600 W. A GB10 table (% of 273 GB/s) is added when the Spark has
been benchmarked.

| variant | n | median ms | min ms | GB/s | % of cudaMemcpy D2D | % of 1,792 GB/s |
|---------|---|-----------|--------|------|---------------------|-----------------|
| memcpy  | 64M  | 0.3508 | 0.3488 | 1530 | 100.0% | 85.4% |
| 0       | 64M  | 0.3486 | 0.3464 | 1540 | 100.6% | 85.9% |
| 1       | 64M  | 0.3491 | 0.3483 | 1538 | 100.5% | 85.8% |
| 2       | 64M  | 0.3594 | 0.3570 | 1494 | 97.6% | 83.4% |
| memcpy  | 256M | 1.4068 | 1.4016 | 1527 | 100.0% | 85.2% |
| 0       | 256M | 1.4017 | 1.3996 | 1532 | 100.4% | 85.5% |
| 1       | 256M | 1.4017 | 1.3988 | 1532 | 100.4% | 85.5% |
| 2       | 256M | 1.4386 | 1.4281 | 1493 | 97.8% | 83.3% |
