# V0.4.0 Independent Adversarial Review — Measured Select Fusion

**Reviewer:** independent fresh-context review agent (no shared session state with the implementation; read-only access to the repository, CPU-only probes, no GPU benchmarks, no files modified).
**Date:** 2026-11-08 · **Audited tree:** `feature/v0.4.0-measured-select-fusion` at `8e5e1e5` (impl SHA `9fb32f9`).
**Method:** mechanical diff of the port kernel against upstream `expansion.py`; pack-layout verification against `pack_key`/`_pack_scores`; CPU probes for tie/NaN/EMPTY orderings; recomputation of every headline speedup from the receipts' raw `samples_ms`; statistical review of the aggregation; coverage and memory-safety review; CPU-CI compatibility walkthrough.

## Verdict summary (auditor's own words, abbreviated)

| # | Finding | Severity | Disposition (by the engineering session, after the audit) |
|---|---|---|---|
| 1 | Wide-row fallback (`columns > 4·top_p`) ignored `lengths` and docs called it "the reference composition" — false on both counts (zero-visibility row returned live keys) | **HIGH** | **FIXED** at `33d9293`: length-respecting pack+topk with upstream `TopK.best()` semantics; test pins the corrected oracle; docs rewritten |
| 2 | Pre-existing (v0.3.1, not a v0.4.0 regression): port `best_first_keys` drops upstream `best()`'s `masked_fill(~listed, EMPTY)` — the reference chain can read uninitialized listing slots on wide sampled listings | **HIGH (latent, out-of-change)** | **DOCUMENTED, deliberately NOT changed** (oracle byte-frozen per mission; unreachable at every benchmark/production shape — `columns ≤ 4·top_p` takes the whole path); recorded in `MEASURED_SELECT_FUSION.md` §4 |
| 3 | −959 MiB memory claim had no machine receipt; promised command/outputs absent | **MEDIUM** | **FIXED**: `evidence/v0.4.0/ab/memory-probe.json` (sha256 `c8f961df…`); §3 quotes it |
| 4 | A/B receipts pin HEAD without a dirty-tree flag | **LOW** | **FIXED** in `ab_select_fusion.py` (`production_src_dirty`); historical receipts' trees were clean (impl commit preceded the campaign) |
| 5 | Timing nits: fixed ref-first interleave order; env mutation inside both arms equally; `p95 == max` at n=20 | **LOW** | Accepted as-is, documented |
| 6 | Coverage gaps: NaN, `lengths > width`, non-contiguous scores, `bucket_cap ≠ 16`, rows=0/keep=0, thin determinism | **LOW** | NaN/±inf/−0 test added; remainder documented as open LOW (all unreachable via `visible_counts`/production configs) |

## Items the auditor PASSED outright (evidence cited in its report)

- Kernel B is **token-identical** to upstream `_expand_in_order` except the fused int64 load + in-register `key_bucket` (masks, cumsum/store, `filled` accounting, `_append_window_and_sink`, tile/num_warps rules all match).
- Kernel A's pack layout is byte-identical to `_pack_scores`/`pack_key`; EMPTY padding for dead columns and K2 tail verified; `K2 ≤ BLOCK` always.
- No tie/NaN/±0 counterexample constructible (unique keys per column; both sides apply the same `ordered_bits` to the same bits).
- Timing methodology clean: true sample-interleave, sync placement correct, warmup both arms, correctness pre+post, profiler outside p50s.
- **All headline numbers recompute exactly from raw samples**: 32k bf16 decode s+r pooled **1.1986** (per-seed 1.192/1.238/1.207), 32k prefill select **2.9622** (98.05→33.12 ms), prefill s+r **1.2255**, 16k prefill select 3.5449; zero-regression verified (min cell 0.9986 of 120).
- Kernel B memory safety: masked lanes never dereference; no overflow; kernel A fully writes all K2 slots.
- SHA binding: all 12 receipts pin `9fb32f9`; timeline consistent (impl 11:13 → receipts 11:16–11:37 → evidence commit 11:42).
- CPU CI: 31 GPU tests skip loudly, 3 CPU contract tests execute on the hosted runner; no import-time crash risk.

## Headline-claim survival

- 1.20× decode select+read @32k bf16 — **SURVIVES**
- 1.23× prefill select+read @32k bf16 — **SURVIVES**
- 2.96× 32k prefill select — **SURVIVES**
- Zero regressions — **SURVIVES**
- −959 MiB — initially **did not survive as evidenced** (doc-only) → now receipted (finding 3)
- Byte-exactness — **SURVIVES on every measured/production shape**; the wide-row fallback divergence (finding 1) fixed to upstream semantics; the reference's own wide-row latent unsoundness (finding 2) documented.

*Post-audit confirmation: after the fixes (`33d9293`), the full suite (227/227), gates (17/17 reference and fused trees) re-ran green, and a confirming A/B receipt at the new SHA reproduced the headline bands (2.975× prefill select, 1.155–1.210× select+read) — `evidence/v0.4.0/ab/fused-postfix-s424243.json`.*
