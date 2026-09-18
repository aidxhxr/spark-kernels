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
threads x 16 B = 16 KB in flight per SM is comfortably past that on GB10. Whether it is on
GDDR7 is an open question, see below.

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

## RTX 5090 notes (expectations, nothing measured yet)

* Per-SM share of the bus is higher: 1,792 GB/s / 170 SMs ≈ 10.5 GB/s per SM, against
  273 / 48 ≈ 5.7 on the GB10. Variant 0 is limited by the SM's load/store pipeline, not by
  DRAM, so I expect the gap between variant 0 and variants 1/2 to be *wider* on the 5090: the
  same instruction-bound lane now has almost twice the bandwidth it fails to use.
* The "4 blocks x 256 threads per SM" constant was sized so that 16 KB in flight per SM covers
  LPDDR5X latency. I do not know GDDR7's latency from the SM's point of view; at 6.5x the
  bandwidth, the same latency window holds 6.5x more bytes. If variant 2 lands clearly below
  `cudaMemcpy` D2D, sweep the blocks-per-SM constant (4, 8, 16) before anything else.
* In Nsight Compute, `dram__throughput.avg.pct_of_peak_sustained_elapsed` and
  "Warp State → Stall Long Scoreboard" tell the two cases apart: low throughput with few
  scoreboard stalls means not enough loads in flight; low throughput with many means the
  memory system itself is the limit.
* The card boosts and throttles: compare `min_ms` with the median, and see
  [../RTX5090.md](../RTX5090.md#measuring-on-a-geforce-card) before trusting a run. GeForce
  needs admin rights for the hardware counters.
* How far below 1,792 GB/s `cudaMemcpy` D2D lands: TBD — this bench is what measures it.

## Results (RTX 5090)

Fill in from `results/bandwidth.json` after `make bench`. A GB10 table (% of 273 GB/s) is added
when the Spark has been benchmarked.

| variant | n | median ms | GB/s | % of cudaMemcpy D2D | % of 1,792 GB/s |
|---------|---|-----------|------|---------------------|-----------------|
| memcpy  | 64M  |  |  | 100 |  |
| 0       | 64M  |  |  |  |  |
| 1       | 64M  |  |  |  |  |
| 2       | 64M  |  |  |  |  |
| memcpy  | 256M |  |  | 100 |  |
| 0       | 256M |  |  |  |  |
| 1       | 256M |  |  |  |  |
| 2       | 256M |  |  |  |  |
