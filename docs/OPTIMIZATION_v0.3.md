# v0.3.0 optimization — measured bottleneck attribution (T2) and outcomes (T3/T4)

Campaign `campaign-20261106T231500Z`. T2 profiling from
`results/campaigns/campaign-20261106T231500Z/profile/`
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

## f. T3 outcome — tiled sparse read (banked, audited)

`read_kernel_tiled.py`: one program reads TM query rows of one head (TM=4/CN=32
prefill, TM=1/CN64 decode via `sparse_read_auto`), same per-row numeric path as
v1 (fp32 scores/softmax, -1 padding, GQA, per-occurrence duplicates); `tl.dot`
deliberately not used (per-row expanded lists give no shared K matrix for the
M-axis — a forced tl.dot would change semantics).

A/B on identical production inputs, every config correctness-gated before its
timing counts (`scripts/ab_read_kernel.py`, JSON in the campaign dir):

| context | phase | v1 p50 | best v2 | config | speedup |
|---:|---|---:|---:|---|---:|
| 8192 | prefill | 60.5 ms | 48.9 ms | TM4/CN32 | **1.24×** |
| 16384 | prefill | 133.4 ms | 113.0 ms | TM4/CN32 | **1.18×** |
| 32768 | prefill | 286.8 ms | 249.1 ms | TM2/CN64 (TM4/CN32 1.14×) | **1.15×** |
| 8192–32768 | decode | 0.14 ms | parity | TM1/CN64 | ~1.0× |

Step-level (fresh 17/17-gate receipts at HEAD 4988672, auditor-APPROVED,
72 comparisons recomputed): 32k bf16 prefill select+read 389.1 ms → **345.7 ms
(0.19× → 0.207× vs dense)**; decode unchanged as measured (TM1 dispatch).
Gate H added: production path == v1 kernel across fp32/fp16/bf16 ×
prefill/decode launches + the zero-length contract (v1's 0/0 NaN → defined
zero output). 16→17 gates; v0.2.1's 16-gate receipts stay valid for their time.

Multi-seed re-check on the new kernel (9 fresh receipts, auditor-APPROVED):
32k decode flat band 1.67–2.00× (med 1.89×), block-sparse 3.93–4.19× (med
3.99×), tree 0.77–0.95× — consistent with the v1-kernel T1 study (1.69–2.79
med 1.74; 4.14–4.38; 0.80–0.92): the flat decode advantage is real but
seed-sensitive; block-sparse is robust.

## g. T4 outcome — tree beam kernelization (negative result, documented)

Hypothesis from T2: 19,940 launches/iter made the python page loop the cost.
Experiment: page-batched walk (`_tree_select_pages` — one batched op set per
level, chunked to bound the gathered-centroid einsum; growth tables replayed
on CPU; set-equal to the per-page walk on every seed tested — enforced inside
the A/B before timing counts — and gate C PASS with `LSA_TREE_IMPL=batched`).
Measured by `scripts/ab_tree_select.py` (receipt methodology: warmup 5, reps
20, synchronized; artifact `evidence/receipts-v0.3.0/ab-tree-select.json`):

| context | phase | per-page p50 | page-batched p50 | batched/per-page |
|---:|---|---:|---:|---:|
| 8192 | prefill select | 48.06 ms | 55.98 ms | 1.16× slower |
| 32768 | prefill select | 288.70 ms | 373.28 ms | 1.29× slower |
| 8192 | decode select | 1.07 ms | 1.91 ms | 1.78× slower |
| 32768 | decode select | 1.14 ms | 2.12 ms | 1.86× slower |

(An earlier ad-hoc measurement of the same comparison quoted 0.81–0.87× with
under-warmed baselines and was replaced by this committed, set-equality-gated
artifact — the review pass caught it.)

**The launch tax is not the dominant cost** — each launch is tiny; the beam's
GPU work (gathered-centroid scores, per-row sorts, dedup) is. Torch-level
batching forces batch-max candidate widths and extra copies that cost more
than the loop saves (and single-row decode loses the per-page walk's short
`torch.unique` path). Production keeps the per-page walk (default;
`LSA_TREE_IMPL=batched` selects the experimental path, held set-equal by test
and by the A/B's own gate). The remaining lever is a true per-level Triton
beam (upstream's `beam.py` layout: fp8 node centroids, packed keys, children
tables) — a layout-adapting port, estimated multi-day, deferred with this
measurement as its justification.

## h. What a v0.3.1+ attacker does next, in measured order

1. **Partially delivered (v0.3.1, `60c004b`)**: `best_first_keys` replaces the
   select's full-width int64 sort + strided slice with `torch.topk` — provably
   identical on upstream's unique packed keys (5 equivalence tests). Measured
   in-process on identical 32k inputs: **decode select 0.266 → 0.214 ms
   (1.24×)**, prefill neutral (0.99×) — the radix sort was already efficient at
   [32768, 4096]; what remains of this lever is a fused pack→expand Triton pass
   (the copies, not the ordering).
2. The Triton beam port above.
3. Decode select fusion (42 launches / ~139 µs GPU for ~0.21 ms wall after 1).
4. Sparse-read shared-block dedup across cluster rows (needs a grouping pass;
   the ~10× traffic cut case, research-grade).
