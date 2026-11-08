# v0.4.0 / B1 — v0.3.1 Baseline and Bottlenecks (gfx1151)

**Purpose:** the uncontaminated reference against which every v0.4.0 candidate is judged.
**Production code measured:** `1ab0e0b` (v0.3.1 release tree, `src/` clean — recorded per artifact), upstream pin `5ebea732`. **Instrumentation:** `scripts/v040_select_stages.py` (new, times the *production* callables — no reimplementation) + `scripts/profile_stages.py` (existing T2 profiler).
**Evidence:** `evidence/v0.4.0/baseline/` — `select-stages-matrix.json` (sha256 `8200be85…`, raw samples included), `profile-warm-32k-bf16.json`, `profile-warm-16k-bf16.json`, `profile-cold-32k-bf16.json`.

## Environment (frozen at measurement, 2026-11-08 ~02:23–02:26 UTC)

gfx1151 (Radeon 8060S, 32 GiB UMA) · idle before campaign (no KFD PIDs; verified) · edge 51 °C · mclk L2 1000 MHz / sclk L1 ~1500 MHz · torch 2.14.0+rocm10.1.0 · hip 7.16.26385 · Triton 3.8.0+git669b31ac.rocm10.1.0 · seed 20261105 · warmup 5 / reps 20 synchronized (receipt methodology) · profiler passes excluded from p50s. Correctness pre-flight on this branch: **17/17 gates** (default tree); full 192-test suite + 17/17 batched + SHA256SUMS 0 non-OK + receipt recompute 153/153 verified the same morning on the identical tree (R1.1 PR #6 verification log).

## Where the time goes — production flat select (`TopK.listed → best_first_keys → unpack_ids → expand_in_order`)

### bf16, p50 ms (launches/iter), stage × context

| ctx | phase (rows) | score GEMM | pack (`_pack_scores`) | **2nd topk (sorted)** | **unpack** | expand | select total | select+read |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 512 | prefill (512) | 0.032 (4) | 0.066 (7) | 0.091 (3) | 0.032 (13) | 0.068 (3) | 0.230 (34) | 2.584 (37) |
| 512 | decode B=1 | 0.030 (4) | 0.054 (7) | 0.049 (3) | 0.060 (13) | 0.041 (3) | 0.182 (34) | 0.283 (37) |
| 2048 | prefill (2048) | 0.070 (4) | 0.105 (7) | 0.468 (3) | 0.092 (13) | 0.235 (3) | 0.981 (34) | 10.730 (37) |
| 8192 | prefill (8192) | 0.679 (4) | 1.269 (7) | **7.013 (21)** | 2.605 (13) | 1.548 (3) | 12.471 (52) | 62.542 (55) |
| 16384 | prefill (16384) | 2.402 (4) | 4.341 (7) | **20.120 (21)** | 5.028 (13) | 3.195 (3) | 32.969 (52) | 145.075 (55) |
| 32768 | prefill (32768) | 10.414 (4) | 18.176 (7) | **63.251 (21)** | 10.237 (13) | 7.067 (3) | **97.219 (52)** | 348.690 (55) |
| 32768 | decode B=1 | 0.046 (4) | 0.090 (7) | 0.068 (3) | 0.035 (13) | 0.038 (3) | 0.193 (34) | 0.275 (37) |
| 32768 | decode ragged B=32 | 0.071 (4) | 0.106 (7) | 0.115 (3) | 0.036 (13) | 0.061 (3) | 0.253 (34) | 0.294 (37) |

fp16 mirrors bf16 within noise (32k prefill select 97.662 ms; 2nd topk 62.896; decode B=1 select 0.147, select+read 0.229). Ragged B=32 rows are in the matrix JSON for every context.

### Kernel attribution (32k bf16, warm, device-only rows)

- `flat_select_prefill` 96.6 ms p50, 52 launches: **`aten::topk` umbrella 34.8 % + `mbtopk::computeBlockDigitCounts` 23.2 % + `gatherTopK` 4.7 % + `radixSortKVInPlace` (in decode row) — the int64 top-k fan-out is ~60–65 % of the whole select**; `aten::mm` (the score GEMM) 5.6 %; `_pack_scores` 4.5 %.
- `flat_select_decode` 0.210 ms p50 but only 126 µs GPU time across 34 launches — **launch-gap dominated**; topk family ≈ 45 % of the GPU time, `aten::copy_` 40×/iter (allocator/counts churn) 7 %.
- Cold start (isolated Triton cache): first `flat_select_prefill` 652 ms (compile), first decode select 127 ms; second calls 135 ms / 1.5 ms → compile is one-off, excluded from all warm claims.

## Reading of the bottlenecks (what v0.4.0 must attack)

1. **The second, order-giving top-k (`best_first_keys`) is the single dominant cost at scale** — 65 % of the 32k prefill select (63.3 ms of 97.2 ms), and its ATen int64 radix fan-out is 21 launches at wide rows. The v0.3.1 change (sort→topk) reduced this stage 0.528→0.466 ms at decode; at prefill scale it remains the wall.
2. **`unpack_ids` is pure overhead**: 13 launches of elementwise torch (10.5 % of 32k prefill select; comparable to the GEMM itself) to turn ordered int64 keys into int32 ids — information the next kernel could extract itself.
3. **Decode select is launch-bound**: 34 launches ≈ 0.19–0.26 ms for µs-scale device work; every removed launch is direct latency.
4. The score GEMM (5.6 % at 32k) and the pack kernel (4.5 %) are small; `_pack_scores`' [R, N] int64 materialization (1 GiB at 32k prefill) is the one pack-side lever a fully fused kernel would also remove.
5. The upstream contract check (mission B0): the *whole* listing path (`columns ≤ 4·top_p`) is what every benchmark shape takes (4096 ≤ 4·1024 at 32k) — the sampled-listing path is out of scope and must gate any fused dispatch.

**Candidate targeting:** F1 (unpack elimination) attacks (2)+(3); F2 (fused select+expand, one kernel after the GEMM) attacks (1)+(2)+(3)+(4) at once. Details in `EXPERIMENT_DECISIONS.md`.

## Provenance notes

- Absolute decode-select p50s here (tight-loop, hot allocator) run lower than the v0.3.1 receipt conditions (e.g. 32k bf16 decode select 0.466 ms receipt vs 0.193 ms here): different measurement context, not a contradiction — every v0.4.0 comparison in this campaign is same-conditions reference-vs-candidate (identical tensors, interleaved), and no cross-context speedup is claimed.
- No simulated numbers; every row above is a real synchronized measurement on the physical gfx1151 host.
