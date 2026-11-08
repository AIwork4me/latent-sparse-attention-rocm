# v0.4.0 — Measured Select Fusion on ROCm gfx1151

**Status: correctness-complete, measured, review-ready (Draft PR; production promotion decision at the end).**
**Implementation SHA `9fb32f90462c20f8508fa6059a1b91e8cccb0111`** (branch `feature/v0.4.0-measured-select-fusion`); every receipt below pins it. Upstream pin `5ebea732` (untouched). Baseline: v0.3.1 (`1ab0e0b`) production chain, measured unmodified (`docs/v0.4.0/BASELINE_AND_BOTTLENECKS.md`).

## 1. What was built

The production flat select was `GEMM scores → TopK.listed (pack ALL columns to int64) → torch.topk(sorted) [best_first_keys] → unpack_ids → expand_in_order`. Two port-side Triton kernels (`src/lsa_rocm_lab/port/select_fused.py`) replace everything after the GEMM:

- **Kernel A — `select_topk_scores`**: loads the fp32 score row, packs score-bits+column in-register (upstream's exact `ordered_bits`/pack layout, imported), keeps the `min(top_p, N)` greatest keys sorted descending via `tl.topk` (probe: exact vs `torch.topk` on int64 at every width). The `[rows, columns]` int64 key matrix is **never materialized**.
- **Kernel B — `expand_in_order_from_keys`**: upstream's `_expand_in_order` walk verbatim (same loop, masks, `_append_window_and_sink`, `_row_budget` — imported, upstream unmodified), reading the ordered int64 keys and computing `key_bucket` in-register: the whole `unpack_ids` chain (13 launches) disappears into the walk.

Selected by `LSA_SELECT_IMPL=reference|fused|fused_unpack` (default `reference` = the immutable v0.3.1 oracle, extracted verbatim as `flat_select_reference`). `fused_unpack` = F1 alone (reference pack+topk, fused walk) kept as a measured alternative. Wide score matrices (`columns > 4·top_p`, the sampled-listing contract — no benchmark shape) dispatch loudly to the reference composition (`UserWarning`, tested).

## 2. Exactness (correctness before performance, fail-closed)

- **34 new adversarial tests** (`tests/test_select_fusion.py`): byte-exact lists+lengths vs the oracle for tied scores (column tie-break), EMPTY keys/short rows/zero-visibility rows, k == width, k > width, ragged B=32, window/sink boundaries, tiny window/sink, non-power-of-two widths, production widths to 4096, fp32/bf16 scoring, determinism (3 repeats), GQA select+read numerical equivalence; CPU-executable contract tests (switch validation, wide fallback) run on GitHub-hosted CI.
- **Suite 226/226** (192 existing + 34 new) · **17/17 gates × 3 trees** (default, `LSA_TREE_IMPL=batched`, `LSA_SELECT_IMPL=fused` — the fused path passes the fail-closed gate manifest itself).
- **Every A/B receipt row re-verifies byte-exactness before and after its timing block**; the campaign's verdict: **360/360 rows exact, 0 failures**.

## 3. Performance (interleaved A/B, warmup 5 / reps 20, synchronized; 3 seeds × 3 independent process runs per seed; speedup = same-run reference-p50 / candidate-p50)

Full tables: `evidence/v0.4.0/ab/aggregate-fused.md` (+ `.json`, raw samples in every receipt). Ranges are **observed ranges** (min–max over the stated runs), not confidence intervals.

### select+read p50, `fused` vs reference (the mission's acceptance stage)

| dtype | ctx | decode B=1 | prefill |
|---|---:|---|---|
| bf16 | 512 | **1.199×** (seeds 1.187–1.206; all runs 1.155–1.263) | 1.047× (1.033–1.067) |
| bf16 | 2048 | **1.200×** (1.146–1.234) | 1.048× (1.045–1.052) |
| bf16 | 8192 | **1.176×** (1.153–1.246) | 1.129× (1.127–1.130) |
| bf16 | 16384 | **1.211×** (1.203–1.224) | **1.178×** (1.177–1.179) |
| bf16 | 32768 | **1.199×** (1.192–1.238) | **1.226×** (1.225–1.228) |
| fp16 | 16384 | **1.202×** (1.194–1.208) | **1.184×** (1.183–1.184) |
| fp16 | 32768 | **1.218×** (1.189–1.277) | **1.222×** (1.222–1.227) |

Decode ragged B=32 select+read: 1.08–1.20× across the ladder (see aggregate). **Against the ≥5% acceptance bar for a 16k/32k select+read scenario: 16k bf16 decode 1.211×, 32k bf16 prefill 1.226× — met with ~4× margin. Zero regression cells (<0.98×) across all 120 aggregated (dtype, ctx, phase, stage) cells.**

### Selection stage alone (bottleneck attack confirmed)

- 32k bf16 prefill select: **2.96× pooled** (98.0 → 33.1 ms; seeds 2.952–2.969) — the 65%-share second topk is gone. 16k: 3.55×.
- 32k bf16 decode select: 1.38× (0.512 → 0.369 ms); the reference p50 here matches the v0.3.1 receipt conditions (0.528 → 0.466 ms after the v0.3.1 topk change) — v0.4.0 stacks on top: sort-era 0.528 → fused 0.369 ms.
- Launches: select 34→16 (decode) / 52→16 (32k prefill); select+read 37→19 / 55→19.

### Memory, compile, cold-start

- Peak select allocation @32k prefill: **reference 2304 MiB → fused 1345 MiB (−959 MiB)** — the `[32768, 4096]` int64 key matrix (1 GiB) is never written. `fused_unpack` peaks at 2369 MiB (no saving — expected: it keeps the full chain).
- Kernel A compile: 33.9 ms once per process at BLOCK=4096 (Triton cache makes it per-process-only); decode variants reuse the binary (first-call 0.2–0.6 ms = launch cost). Kernel B compiles in the same first select call.
- Cold-start of the whole step is unchanged in structure (one extra kernel binary to compile, ~34 ms one-off, vs 13+ launches removed on every call).

### Seed/run variance

Within-seed run-to-run spread of the pooled statistic is ≤ ±3% on decode cells and ≤ ±0.5% on the big prefill cells (see `speedup_all_runs_min_max` in the aggregate JSON); per-seed medians never cross the regression line in any cell. No sample was aggregated across seeds — per-seed medians are computed first, the pooled median over run-level p50s second.

### The F1 alternative (kept, measured, not selected)

`fused_unpack` (unpack fusion only): 32k bf16 decode s+r 1.105–1.135×, prefill s+r 1.026–1.028×, zero regressions — a real but strictly smaller win than `fused` in **every** cell. Kept behind the switch as the low-complexity fallback and as evidence the unpack chain alone was worth ~10% at decode.

## 4. What is NOT claimed (NOT_RUN / out of scope)

- Contexts > 32768 and `columns > 4·top_p` (sampled-listing path) — NOT_RUN; dispatches to reference (tested).
- Tree select, batched tree, index build — untouched by this change (gates re-run green).
- End-to-end SGLang serving on gfx1151 — still BLOCKED_ON_REASSESS (unchanged v0.3.1 posture); **no model-level or serving-level claim is made** — all numbers are stage-level (select / select+read) on the port's production path.
- W7900/other GPUs — not run (gfx1151 is the only evidence lane, per mission).
- F2b (digit-radix in-kernel select, upstream `_search_rows`-style) — recorded as follow-on research in `EXPERIMENT_DECISIONS.md`, not implemented (F2a already removes the bottleneck; complexity budget unspent deliberately).

## 5. Reproduction

```bash
git checkout feature/v0.4.0-measured-select-fusion   # 9fb32f9
PYTHONPATH=src python -m lsa_rocm_lab.port.gates                 # 17/17 (reference)
LSA_SELECT_IMPL=fused PYTHONPATH=src python -m lsa_rocm_lab.port.gates   # 17/17
python -m pytest tests/ -q                                        # 226 passed
# A/B (one seed, one process run):
python scripts/ab_select_fusion.py --seed 424243 --reps 20 --dtypes bf16 \
  --candidate fused --out /tmp/fused.json
python scripts/aggregate_v040_ab.py --receipts evidence/v0.4.0/ab/fused-s*.json \
  --out /tmp/agg.json --markdown /tmp/agg.md
```

Hardware/software: AMD Ryzen AI MAX+ PRO 395 / Radeon 8060S (gfx1151) · torch 2.14.0+rocm10.1.0 · hip 7.16.26385 · Triton 3.8.0+git669b31ac.rocm10.1.0 · idle-host discipline (no KFD PIDs before each run; GPU state embedded in every receipt).

## 6. Production-readiness assessment (input to the PR's GO/NO-GO)

| Criterion | Verdict |
|---|---|
| Correctness (all tests/gates, no semantic regression) | **PASS** — byte-exact oracle equality, 360/360 receipt rows, 17/17 gates on the fused tree |
| Performance (≥5% select+read at 16k/32k, beyond noise) | **PASS** — 17.6–22.6% observed, bands tight and seed-stable |
| Regression screening (>2% any cell) | **PASS** — zero cells < 0.98× |
| Memory (no peak regression / hidden GiB) | **PASS** — −959 MiB peak at 32k prefill |
| Complexity (maintenance burden) | **ACCEPTABLE** — two bounded kernels, upstream walk reused verbatim, explicit dispatch with loud fallback; default stays `reference` until the owner promotes |
