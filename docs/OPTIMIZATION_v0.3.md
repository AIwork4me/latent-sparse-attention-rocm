# v0.3.0 optimization — measured bottleneck attribution (T2)

Generated from `results/campaigns/campaign-20261106T231500Z/profile/`
(`profile_stages.py --mode cold|warm`, 32k ctx, bf16, seed 20261105, gfx1151,
ROCm 10.1.0 / torch 2.14.0+rocm10.1.0 / Triton 3.8.0+git669b31ac).
GPU state during the pass: 49–52 °C edge, sclk level 1 (1416–1460 MHz), no
thermal throttling; host idle apart from this campaign.

Method: absolute stage costs are synchronized wall-clock p50 over 20 reps
(warmup 5 excluded); kernel identity and relative shares come from a
`torch.profiler` pass (8 iters). Profiler-inflated "GPU totals" exceed the
unprofiled wall on this ROCm build (CUPTI/HIP overhead inflates kernel
durations), so **shares are the attribution signal; p50 wall is the cost**.
Cold first-call times come from a fresh process with an isolated empty
`TRITON_CACHE_DIR`, measured inside the dependency-ordered build sequence (no
stage pre-warmed by setup).

## a. Stage waterfall @ 32768 bf16 (warm p50 ms, launches/iter)

| stage | p50 ms | launches/iter | verdict |
|---|---:|---:|---|
| dense prefill (SDPA `attn_fwd.kd`) | 74.8 | 4 | baseline |
| dense decode | 1.41 | 2 | baseline |
| index_regroup | 2.0 | 20 | amortizable (per index build) |
| ward_cluster | 6.8 | 86 | amortizable |
| hierarchy_build | 13.1 | **1535** | python-composed, overhead-bound |
| flat_select prefill | 95.0 | 42 | sort/copy-chain bound |
| flat_select decode | 0.27 | **42** | launch-bound (42 launches, ~139 µs GPU) |
| tree_select prefill | 288.6 | **19940** | massively launch-bound |
| tree_select decode | 1.12 | **353** | launch-bound |
| **sparse_read prefill (flat)** | **299.3** | 1 | single `_sparse_read_kernel`, dominant |
| sparse_read decode (flat) | 0.14 | 1 | 8 programs, latency-bound |
| sparse_read prefill (tree) | 229.0 | 1 | same kernel, shorter lists |

## b. Kernel attribution (shares within each stage's profiled GPU time)

### sparse_read prefill — the #1 cost (299 ms of the 390 ms flat step)
- `_sparse_read_kernel.kd` — **100.0%** (single launch; 262 144 programs =
  32 768 rows × 8 heads, each independently gathering its 4 620-id read list
  in 64-id blocks, `num_stages=1`, `num_warps=4`).
- No python overhead in the stage; the cost is the gather pattern itself:
  per-program K/V gathers with no cross-query tile sharing — rows in one
  cluster select largely the same blocks but each program re-reads them.

### flat_select prefill (95 ms)
| kernel | share |
|---|---:|
| `radixSortKVInPlace<2,-1,512,8,long,long,…>` (aten::sort of top_p int64 keys) | 19.4% |
| `aten::copy_` (64×; direct_copy elementwise) + `__amd_rocclr_copyBuffer.kd` | 16.7% + 12.6% |
| `Cijk_Alik_Bljk_BSS_BH_Bias_HA_S…_ISA1151…` (bf16 centroid GEMM, rocBLAS) | 5.6% |
| `_pack_scores.kd` (upstream Triton top-k) | 4.4% |
| `_expand_in_order.kd` (upstream Triton expansion) | 3.0% |
| `aten::bitwise_and` + others | remainder |
→ the sort + its copy chain ≈ **48%** of select; the actual scoring GEMM is cheap.

### tree_select prefill (289 ms) — T4 target
| kernel | share |
|---|---:|
| `Cijk_Alik_Bljk_S_B_Bias…_MT40x64x8…` (aten::bmm, 1872 tiny bmms) | 22.3% |
| `vectorized_gather_kernel<16,long>` (aten::index, 4264×) | 9.7% |
| `radixSortKVInPlace<2,-1,128,8,…>` (aten::sort, 7488×) | 9.2% + 7.0% |
| `aten::copy_` (24 864×) + `__amd_rocclr_copyBuffer.kd` (17 368×) | 3.2% + 1.9% |
→ **19 940 launches/iter**: a python beam composing thousands of
milli-second-scale-nothing ops; per-launch overhead (incl.
`hipModuleLaunchKernel`, `hipPointerGetAttribute` rows) is the cost, not FLOPs.

### hierarchy_build (13 ms)
`_nearest_open_kernel.kd` 34.8%, `hipPointerGetAttribute` 15.8% (6272 calls —
API overhead), `aten::mul`/`aten::sum` reduces ~11%: python-composed pairing
loop, 1535 launches. Amortizable per index build, but the same fusion pattern
as T4.

### dense decode baseline (1.41 ms)
`attn_fwd.kd`-family flash decode: 2 launches. (Receipt p50 1.27 ms; the
0.14 ms gap is profiler-mode overhead on this pass.)

## c. Cold (first-call) costs — one-time, per process

| stage | first call | note |
|---|---:|---|
| ward_cluster | **9 702 ms** | Triton Ward merge compile |
| hierarchy_build | 2 032 ms | node pairing compile |
| sparse_read prefill | 487 ms | `_sparse_read_kernel` compile (decode variant reuses: 0.4 ms) |
| flat_select prefill | 552 ms | top-k + expansion compiles |
| tree_select prefill | 550 ms | shares most compiles with flat_select |
| dense prefill / decode | 95 / 16 ms | rocBLAS/attention library init |
Second calls are launch-scale (≤ 17 ms) except the real-compute stages —
compile is fully amortized by the existing warmup=5 contract; no steady-state
effect on receipts.

## d. Priorities derived (what T3/T4 attack, in measured order)

1. **T3 prefill sparse read** — 299 ms = 77% of the 390 ms flat prefill step;
   tiled multi-query programs sharing gathered K/V blocks across a query tile
   (cluster rows select the same blocks), `tl.dot` QK^T, coalesced loads.
   Even 2× on this stage takes prefill 0.19× → ~0.35×.
2. **T3b select sort chain** — sort+copies ≈ 45 ms of the 95 ms prefill
   select; int32 keys or a fused pack→sort→expand Triton pass.
3. **T4 tree beam kernelization** — 19 940 launches/iter at 289 ms; fusing the
   per-level beam into one kernel per level (or 3) removes the per-launch tax.
   Decode tree select: 353 launches for 1.12 ms — same lever, smaller prize.
4. **Decode select fusion (new, measured)** — flat decode select is 42
   launches / ~139 µs GPU for 0.27 ms wall: fusing pack+sort+expand for the
   single-row decode case is the cheapest decode-step win, and it compounds
   with the read-kernel work because both sit on the decode critical path.
5. Not worth it now: index/hierarchy build (amortizable, 20 ms combined per
   index), dense baselines (library kernels).

## e. Non-goals re-confirmed by measurement

- The sparse read kernel already excludes compile (c) and runs as a single
  launch — its cost is the gather pattern, not launch or python overhead.
- Decode's sparse read (72 µs GPU) is latency-bound at 8 programs; batching
  (multi-query decode) changes occupancy, not the algorithm.
