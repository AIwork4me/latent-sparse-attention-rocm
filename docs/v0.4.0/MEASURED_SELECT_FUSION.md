# v0.4.0 — Measured Select Fusion on ROCm gfx1151

**Status: code INTEGRATED via PR #7 (main merge `6d17a7d`), opt-in fused path; v0.4.0 RELEASE CANDIDATE only (not tagged/published). Default promotion remains Draft PR #8, owner + GPU gated.**
**Implementation chain `9fb32f9` (campaign) → `33d9293` (audit fixes) → `45a0ed1` (final numerical code, wide-row correctness fix + whole-path skip)** on branch `feature/v0.4.0-measured-select-fusion`; every receipt pins its own SHA. Upstream pin `5ebea732` (untouched). Baseline: v0.3.1 (`1ab0e0b`) production chain, measured unmodified (`docs/v0.4.0/BASELINE_AND_BOTTLENECKS.md`). The campaign receipts at `9fb32f9` remain applicable to `45a0ed1`: the fused kernels' code path is identical, and the reference oracle's op sequence is preserved by the provably-no-op whole-path mask skip — re-confirmed by the final-SHA receipt `fused-final-s424243.json` (§3).

## 1. What was built

The production flat select was `GEMM scores → TopK.listed (pack ALL columns to int64) → torch.topk(sorted) [best_first_keys] → unpack_ids → expand_in_order`. Two port-side Triton kernels (`src/lsa_rocm_lab/port/select_fused.py`) replace everything after the GEMM:

- **Kernel A — `select_topk_scores`**: loads the fp32 score row, packs score-bits+column in-register (upstream's exact `ordered_bits`/pack layout, imported), keeps the `min(top_p, N)` greatest keys sorted descending via `tl.topk` (probe: exact vs `torch.topk` on int64 at every width). The `[rows, columns]` int64 key matrix is **never materialized**.
- **Kernel B — `expand_in_order_from_keys`**: upstream's `_expand_in_order` walk verbatim (same loop, masks, `_append_window_and_sink`, `_row_budget` — imported, upstream unmodified), reading the ordered int64 keys and computing `key_bucket` in-register: the whole `unpack_ids` chain (13 launches) disappears into the walk.

Selected by `LSA_SELECT_IMPL=reference|fused|fused_unpack` (default `reference` = the immutable v0.3.1 oracle, extracted verbatim as `flat_select_reference`). `fused_unpack` = F1 alone (reference pack+topk, fused walk) kept as a measured alternative. Wide score matrices (`columns > 4·top_p`, the sampled-listing contract — no benchmark or production shape reaches it: the ladder tops out at exactly `columns == 4·top_p`) dispatch loudly (`UserWarning`, tested) to the **length-respecting pack+topk with upstream `TopK.best()` semantics** — which, since the `best_first_keys` fix, is byte-equal to the reference chain on those shapes too (§4.2).

## 2. Exactness (correctness before performance, fail-closed)

- **34 fusion tests** (`tests/test_select_fusion.py`): byte-exact lists+lengths vs the oracle for tied scores (column tie-break), EMPTY keys/short rows/zero-visibility rows, k == width, k > width, ragged B=32, window/sink boundaries, tiny window/sink, non-power-of-two widths, production widths to 4096, fp32/bf16 scoring, determinism (3 repeats), GQA select+read numerical equivalence; CPU-executable contract tests (switch validation, wide fallback) run on GitHub-hosted CI.
- **31 wide-row adversarial tests** (`tests/test_select_wide_adversarial.py`, §4.2's fix): poisoned unlisted slots never leak, sampled listings match the exact best-k, determinism under three arbitrary garbage patterns, fused fallback == fixed reference on wide shapes, kernel-B expansion at bucket_cap {8,32,64} + composition-level non-default caps, 65536-context end-to-end through the switch (fused + fused_unpack), whole-path op-sequence pin, non-pow2 cap loud failure.
- **Suite 258/258** on gfx1151 · **17/17 gates × 3 trees** (default, `LSA_TREE_IMPL=batched`, `LSA_SELECT_IMPL=fused` — the fused path passes the fail-closed gate manifest itself).
- **Every A/B receipt row re-verifies byte-exactness before and after its timing block**; the campaign's verdict: **360/360 rows exact, 0 failures**; the final-SHA confirming receipt: 3/3 exact.
- **Byte-exact equivalence demonstrated for**: contexts 512–32768 (columns 64–4096, whole path; lists+lengths, fused and fused_unpack, bf16 and fp32 scoring, prefill/decode/ragged) and, since the fix, sampled shapes `columns > 4·top_p` (synthetic 8192-column listings and the 65536-context composition, where columns reach 8128) plus bucket_cap {8,16,32,64} (pow2 caps; non-pow2 caps fail loudly by contract — same Triton arange constraint upstream carries).

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

- Peak select allocation @32k prefill, machine-receipted (`evidence/v0.4.0/ab/memory-probe.json`, sha256 `7efffbcb…`: **clean tree at final SHA `45a0ed1`**, 5 raw samples per impl, device/runtime/checksums embedded; original dirty-tree receipt archived as `memory-probe-superseded-dirty.json`, sha256 `c8f961df…`): **reference 2304.2 MiB → fused 1344.9 MiB (−959.4 MiB, 41.6%)** — reduction independently recomputed from the raw samples. The `[32768, 4096]` int64 key matrix (1 GiB) is never written. `fused_unpack` 2369.0 MiB (+64.8 vs reference — expected: it keeps the full chain; consistent with the B1 baseline matrix's 2312 MiB standalone-stage figure). The clean rerun reproduces the superseded receipt's raw bytes exactly (2416181248 / 1410203648 / 2484076544) — the original numbers were valid; only its provenance (dirty tree, evidence-commit SHA) was broken, now replaced.
- Final-SHA confirming A/B (`evidence/v0.4.0/ab/fused-final-s424243.json`, sha256 `39e82d03…`, clean at `45a0ed1`, 3/3 rows exact pre+post): 32k bf16 prefill select **2.977×** (ref 97.08 → cand 32.61 ms — inside the campaign band 2.952–2.969 at equal conditions), prefill s+r **1.225×**, decode s+r 1.224×, ragged s+r 1.156× — all within the campaign bands ⇒ **the `9fb32f9` campaign receipts remain applicable to the final numerical code**. (The intermediate receipt `fused-7fdb364-maskcost-s424243.json` documents why the whole-path mask skip exists: masking unconditionally cost the reference arm +28% at 32k prefill select, 97.3 → 124.2 ms — unacceptable drift of the published baseline.)
- Kernel A compile: 33.9 ms once per process at BLOCK=4096 (Triton cache makes it per-process-only); decode variants reuse the binary (first-call 0.2–0.6 ms = launch cost). Kernel B compiles in the same first select call.
- Cold-start of the whole step is unchanged in structure (one extra kernel binary to compile, ~34 ms one-off, vs 13+ launches removed on every call).

### Seed/run variance

Within-seed run-to-run spread of the pooled statistic is ≤ ±3% on decode cells and ≤ ±0.5% on the big prefill cells (see `speedup_all_runs_min_max` in the aggregate JSON); per-seed medians never cross the regression line in any cell. No sample was aggregated across seeds — per-seed medians are computed first, the pooled median over run-level p50s second.

### The F1 alternative (kept, measured, not selected)

`fused_unpack` (unpack fusion only): 32k bf16 decode s+r 1.105–1.135×, prefill s+r 1.026–1.028×, zero regressions — a real but strictly smaller win than `fused` in **every** cell. Kept behind the switch as the low-complexity fallback and as evidence the unpack chain alone was worth ~10% at decode.

## 4. Known issues and audit findings (independent adversarial review, 2026-11-08)

An independent fresh-context adversarial audit re-verified the walk kernel as token-identical to upstream (modulo the fused load), the pack layout against `pack_key`, recomputed every headline speedup from raw samples (all exact), and confirmed zero regression cells (min 0.9986 over 120 cells). Its findings and the dispositions:

1. **[HIGH, FIXED] Wide-row fallback semantics.** The first implementation's `columns > 4·top_p` fallback packed all columns ignoring `lengths` while its docs called it "the reference composition" — false on both counts (proven by the auditor with a zero-visibility row). Fixed: the fallback is now the length-respecting pack+topk (upstream `TopK.best()` semantics; a zero-visibility row returns all EMPTY), its test pins exactly that, and every docstring states what it is.
2. **[HIGH, PRE-EXISTING — NOW FIXED at `7fdb364`, refined at `45a0ed1`] Port `best_first_keys` dropped upstream's `~listed` mask.** Upstream's `TopK.best()` masks non-listed slots to EMPTY before its top-k (`top_k.py` `masked_fill(~listed, EMPTY)`); the port's `best_first_keys` (v0.2.1→v0.3.1) did not, so on a wide sampled listing the *reference chain itself* read uninitialized key-buffer slots (poison values beat real keys; a zeroed buffer unpacks to out-of-range columns). Fix: `best_first_keys(keys, top_p, counts=...)` masks unlisted slots exactly as upstream; both listing-holding call sites pass the counts. **The historical oracle is preserved two ways, proven not asserted:** (a) on the whole path (every benchmarked shape, columns ≤ 4·top_p) the mask is a provable no-op — pinned byte-exact against the old op sequence (`test_whole_path_byte_freeze`, `test_whole_path_reference_is_the_v031_op_sequence`) and the gates run byte-identically; (b) an unconditional mask was measured to cost the reference arm +28% at 32k prefill select (receipt `fused-7fdb364-maskcost-s424243.json`) — so the production chain skips the mask exactly where it is provably a no-op (`_mask_for`) and keeps the v0.3.1 op sequence, timings and allocations; the sampled path always masks (correctness before comparability). The fused fallback and the fixed reference are now byte-equal on wide shapes (the former divergence note is obsolete).
3. **[MEDIUM, FIXED; provenance re-established 2026-11-08] Memory claim receipt.** The first receipt measured a dirty tree (the probe script itself was uncommitted at run time, hence `production_src_dirty: true` and an evidence-commit SHA `8e5e1e5` ≠ benchmark impl SHA). Corrected: probe hardened (clean-tree enforcement, `--repo` pinning, device/runtime versions, 5 raw samples + canonical checksum), rerun on the clean final implementation commit `45a0ed1` (`memory-probe.json`, sha256 `7efffbcb…`); original archived verbatim (`memory-probe-superseded-dirty.json`, sha256 `c8f961df…`); reduction independently recomputed from raw samples: 2304.2 → 1344.9 MiB, −959.4 MiB (41.6%). The clean rerun reproduces the original raw bytes exactly.
4. **[LOW, FIXED] A/B receipts pinned HEAD without a dirty-tree flag** (the tree *was* clean — impl commit `9fb32f9` preceded the campaign, evidence commits followed); `ab_select_fusion.py` now records `production_src_dirty` for future runs.
5. **[LOW, CLOSED 2026-11-08] Coverage gaps:** `bucket_cap != 16` kernel-B variants now tested ({8,32,64} byte-exact + composition-level; non-pow2 caps fail loudly with a clear `ValueError` — the same Triton arange constraint upstream's `_expand_in_order` carries); sampled-listing determinism now tested under three arbitrary garbage patterns; wide/sampled path covered end-to-end at 65536 context. Remaining accepted as documented: `lengths > width` defensive validation (unreachable via `visible_counts` clamp), non-contiguous `scores` rows (same stride assumption as upstream `_pack_scores`), thin timing-determinism coverage (3 repeats at one shape). Timing nits accepted as-is and documented: fixed ref-first interleave order, `p95 == max` at n=20, env-var mutation inside both arms equally.

## 5. What is NOT claimed (NOT_RUN / out of scope)

- Contexts > 32768 in the *timed campaign*: performance numbers at `columns > 4·top_p` are NOT_RUN (the sampled path is now CORRECTNESS-covered — adversarial tests + 65536-context byte-exact end-to-end — but no speedup is claimed there; the fused wide fallback is untimed torch, deliberately).
- Tree select, batched tree, index build — untouched by this change (gates re-run green).
- End-to-end SGLang serving on gfx1151 — still BLOCKED_ON_REASSESS (unchanged v0.3.1 posture); **no model-level or serving-level claim is made** — all numbers are stage-level (select / select+read) on the port's production path.
- W7900/other GPUs — not run (gfx1151 is the only evidence lane, per mission).
- F2b (digit-radix in-kernel select, upstream `_search_rows`-style) — recorded as follow-on research in `EXPERIMENT_DECISIONS.md`, not implemented (F2a already removes the bottleneck; complexity budget unspent deliberately).

## 6. Reproduction

```bash
git checkout feature/v0.4.0-measured-select-fusion   # final numerical code 45a0ed1
PYTHONPATH=src python -m lsa_rocm_lab.port.gates                 # 17/17 (reference)
LSA_SELECT_IMPL=fused PYTHONPATH=src python -m lsa_rocm_lab.port.gates   # 17/17
python -m pytest tests/ -q                                        # 258 passed
# A/B (one seed, one process run, clean tree enforced):
python scripts/ab_select_fusion.py --seed 424243 --reps 20 --dtypes bf16 --contexts 32768 \
  --candidate fused --out /tmp/fused.json
python scripts/aggregate_v040_ab.py --receipts evidence/v0.4.0/ab/fused-s*.json \
  --out /tmp/agg.json --markdown /tmp/agg.md
# memory receipt (clean tree enforced, raw samples + checksums):
python scripts/v040_memory_probe.py --samples 5 --out /tmp/memprobe.json
```

Hardware/software: AMD Ryzen AI MAX+ PRO 395 / Radeon 8060S (gfx1151) · torch 2.14.0+rocm10.1.0 · hip 7.16.26385 · Triton 3.8.0+git669b31ac.rocm10.1.0 · idle-host discipline (no KFD PIDs before each run; GPU state embedded in every receipt).

## 7. Production-readiness assessment (input to the PR's GO/NO-GO)

| Criterion | Verdict |
|---|---|
| Correctness (all tests/gates, no semantic regression) | **PASS** — byte-exact oracle equality, 360/360 campaign + 3/3 final-SHA receipt rows, 17/17 gates on the fused tree, wide-row defect FIXED with proven oracle freeze |
| Performance (≥5% select+read at 16k/32k, beyond noise) | **PASS** — 17.6–22.6% observed, bands tight and seed-stable; re-confirmed at the final SHA (2.977× / 1.225×) |
| Regression screening (>2% any cell) | **PASS** — zero cells < 0.98× |
| Memory (no peak regression / hidden GiB) | **PASS** — −959.4 MiB peak at 32k prefill, clean-tree receipt, independently recomputed |
| Complexity (maintenance burden) | **ACCEPTABLE** — two bounded kernels, upstream walk reused verbatim, explicit dispatch with loud fallback; default stays `reference` until the owner promotes |
