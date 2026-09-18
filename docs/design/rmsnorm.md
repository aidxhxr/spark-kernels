# RMSNorm and fused add+RMSNorm

`out[r, c] = x[r, c] · rsqrt(mean_c(x[r, :]²) + ε) · w[c]`

RMSNorm runs twice per decoder layer in Llama/Qwen-style models, and it is a pure
bandwidth problem: every element is read, squared, reduced, and written back.
The arithmetic intensity is ~1 FLOP/byte, far below the fp32 ridge point of either
target (≈ 58.5 FLOP/byte on the RTX 5090 at 1,792 GB/s GDDR7, 114 on the GB10 at
273 GB/s LPDDR5X), so the only lever is **moving bytes at peak bandwidth**. The ladder below is about
removing everything that stops the memory system from streaming.

## Optimization ladder

| Variant | Mapping | What it fixes |
|---|---|---|
| 0 | one thread per row | baseline |
| 1 | one warp per row, lane-strided scalar loads | coalescing + parallel reduction |
| 2 | one warp per row, 128-bit vector loads | fewer, wider memory instructions |
| 3 | one 256-thread block per row | parallelism for very wide rows |
| fused | add+RMSNorm, warp per row, 128-bit vectors | removes a full pass over the activations |

### Variant 0: one thread per row (why it is slow)

Thread `t` walks row `t` from column 0 to `cols-1`. Adjacent threads in a warp
therefore touch addresses `cols · sizeof(T)` bytes apart on every iteration, so a
32-lane warp issues 32 separate cache-line requests per load instruction instead of
one. Each thread also performs a serial `cols`-length reduction, and with only
`rows / 256` blocks the GPU is badly under-occupied for typical `rows = 4096`
(16 blocks for 48 SMs on the GB10, for 170 on the RTX 5090). Nsight Compute shows this as low
`dram__throughput`, a huge `l1tex__t_sectors_pipe_lsu_mem_global_op_ld` to
requests ratio, and `sm__warps_active` near the floor.

### Variant 1: one warp per row

Lane `l` reads columns `l, l+32, l+64, …`. Consecutive lanes now read consecutive
elements, so every warp-wide load is a contiguous 128 B (f32) or 64 B (bf16)
request. The sum of squares is reduced with `__shfl_xor_sync` in five steps
(`warp_reduce_sum` from `common.cuh`) instead of a serial loop. Eight warps per
block gives `rows / 8` blocks, plenty of parallelism.

### Variant 2: 128-bit vectorized loads

A warp-wide load of 4-byte scalars is 128 B; the LSU can issue 16 B per lane per
instruction, i.e. 512 B per warp. Loading `float4` / eight packed `bf16` per lane
cuts load instruction count 4–8× and lets the memory pipeline keep more bytes in
flight, which matters on LPDDR5X where latency is high relative to bandwidth
(GB10 reasoning; for GDDR7 see the RTX 5090 notes below).
This requires `cols % 4 == 0` (f32) or `cols % 8 == 0` (bf16); every real
hidden size satisfies this.

The kernel makes two passes over the row: pass 1 reads to compute the sum of
squares, pass 2 re-reads and writes. A row is at most 32 KB for the shapes here,
and the GB10 has a 24 MB L2, so the second read is an L2 hit and DRAM traffic
stays at `2·rows·cols·sizeof(T)`. The RTX 5090's L2 size is TBD (bench banner /
`deviceQuery`); a 32 KB row is small enough that I expect the same to hold.

### Variant 3: one block per row

For rows wider than ~8k elements a single warp cannot keep enough loads in flight.
A 256-thread block per row, with `block_reduce_sum` (warp shuffle + one shared
memory exchange), spreads the row over 8 warps. For narrow rows this is worse than
variant 2 because each block does one `__syncthreads`-heavy reduction for little
work.

### Fused residual add + RMSNorm

The decoder block computes `resid = resid + x; h = rmsnorm(resid) · w`. Unfused, that is: read x,
read resid, write resid (add), then read resid, write h (norm) — five passes over an `rows × cols`
tensor. The fused kernel reads `x` and `resid` once, writes the updated `resid` and `h`: four
passes. On a 273 GB/s machine (GB10) a saved pass over a `4096 × 4096` bf16 tensor is ~32 MB ≈
0.12 ms, which is more than the whole norm costs. At the RTX 5090's 1,792 GB/s spec the same pass
is ≈ 0.018 ms: still a whole pass and a whole launch removed, but 6.5× less wall time. The sum is
rounded to bf16 before it is squared so the stored residual and the normalized value agree
bit-for-bit, matching PyTorch's `(resid + x).rms_norm(...)` in bf16.

## Memory-traffic formulas (used for the GB/s column)

```
rmsnorm      bytes = 2 · rows · cols · sizeof(T) + cols · sizeof(T)
add_rmsnorm  bytes = 4 · rows · cols · sizeof(T) + cols · sizeof(T)
GB/s         = bytes / (median_ms · 1e-3) / 1e9
% of peak    = GB/s / 1792     (RTX 5090; 273 on the GB10)
```

The bench validates every variant against a double-precision CPU reference
(tolerance 1e-4 for f32, 2e-2 for bf16 outputs) before timing it, and exits
non-zero if any check fails.

## What to look at in Nsight Compute

```
ncu --set full -k regex:rmsnorm ./build/bench_rmsnorm --cols=4096 --iters=3
```

| Metric | Meaning | Expected trend v0 → v2 |
|---|---|---|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | share of peak DRAM bandwidth actually used | ~5% → 80%+ (reasoned for the GB10) |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | achieved occupancy | low → high |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` / `…_requests` | sectors per request (coalescing) | 32 → 4 (bf16 vec) |
| `smsp__inst_executed_op_global_ld.sum` | load instruction count | drops 4–8× at v2 |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` | shared-memory bank conflicts | only relevant to v3 (should be ~0) |

## RTX 5090 notes (expectations, nothing measured yet)

- **Variant 0 gets relatively worse.** 16 blocks cannot occupy 170 SMs, and an
  uncoalesced lane wastes a per-SM bandwidth share that is almost twice the GB10's
  (1,792 / 170 ≈ 10.5 GB/s per SM vs 273 / 48 ≈ 5.7). I expect a larger v2/v0
  speedup than on the GB10.
- **The `cols > 8192` crossover to variant 3 will probably move.** It was placed
  where one warp per row stops keeping enough loads in flight on LPDDR5X. With more
  bandwidth behind each SM, a single warp should run out earlier, so I expect the
  crossover to move *down*. Sweep `--cols` from 2048 to 16384 for v2 and v3 and
  put the constant where the curves cross; it is correct at any value.
- **Fusion pays less in milliseconds.** The pass count (5 → 4) is the same, the
  saved pass is ≈ 0.018 ms instead of ≈ 0.12 ms at 4096×4096 bf16. The removed
  launch may matter as much as the removed pass; `scripts/bench_torch.py` reports
  the end-to-end ratio against eager, which is the number to quote.
- **Nsight Compute:** `dram__throughput…pct_of_peak` first. If it is low for v2,
  look at "Warp State → Stall Long Scoreboard" (memory latency: not enough loads
  in flight per row → try v3 / more warps) against `sm__warps_active`
  (occupancy: not enough rows per launch). Check `lts__t_sector_hit_rate` on the
  second pass to confirm the re-read is an L2 hit once the L2 size is known.
- The 5090 boosts and throttles; compare `min_ms` with the median before reading
  anything into a small difference between variants.

## Results (RTX 5090)

Filled from `results/rmsnorm.json` by `scripts/make_results_table.py`.
`rows = 4096`, median of 100 iterations, `% peak` relative to 1,792 GB/s. GB10
tables (relative to 273 GB/s) are added when the Spark has been benchmarked.

| dtype | cols | v0 ms | v1 ms | v2 ms | v3 ms | best GB/s | % peak | speedup v2/v0 |
|---|---|---|---|---|---|---|---|---|
| f32 | 1024 | | | | | | | |
| f32 | 2048 | | | | | | | |
| f32 | 4096 | | | | | | | |
| f32 | 8192 | | | | | | | |
| bf16 | 1024 | | | | | | | |
| bf16 | 2048 | | | | | | | |
| bf16 | 4096 | | | | | | | |
| bf16 | 8192 | | | | | | | |

| fused add+RMSNorm (bf16) | cols | ms | GB/s | % peak |
|---|---|---|---|---|
| | 1024 | | | |
| | 2048 | | | |
| | 4096 | | | |
| | 8192 | | | |
