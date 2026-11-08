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

---

## Closure-session addendum (2026-11-08, final numerical code `45a0ed1`)

A second independent pass (fresh reviewer context, R1.1/v0.4.0 closure mission) audited the
handoff, provenance and the open HIGH finding. Dispositions:

1. **Finding 2 (open HIGH) — FIXED.** `best_first_keys` now applies upstream `TopK.best()`'s
   `~listed` mask (`counts=` parameter; both listing-holding call sites pass it). Oracle freeze
   PROVEN, not asserted: whole-path byte-freeze vs the old op sequence across the ladder
   (incl. both `columns == 4·top_p` and `columns < top_p` boundaries), gates byte-identical
   (T=512 geometry sits exactly on the whole-path boundary), 31 new adversarial tests incl.
   poison-injection, garbage-pattern determinism, 65536-context end-to-end byte-exactness
   through the switch, and non-default (pow2) bucket capacities; non-pow2 caps now fail loudly.
2. **New measured finding — reference-arm mask cost.** Unconditional masking cost the v0.3.1
   oracle +28% at 32k prefill select (97.3 → 124.2 ms, receipt `fused-7fdb364-maskcost-s424243.json`).
   Resolution: the production chain skips the mask exactly where it is provably a no-op
   (`_mask_for`: whole path), keeping the oracle's measured timings/allocations comparable to
   every published receipt; the sampled path always masks. Re-confirmed at the final SHA:
   ref 97.08 → cand 32.61 ms = **2.977×** (`fused-final-s424243.json`, 3/3 rows exact).
3. **Memory-claim provenance re-established.** The receipt had measured a dirty tree at an
   evidence-commit SHA (`8e5e1e5`, `production_src_dirty: true`) — cause: the probe script was
   itself uncommitted when it ran. Probe hardened (clean-tree fail-closed, `--repo` pinning,
   device/runtime versions, 5 raw samples + canonical checksum); rerun clean at `45a0ed1`;
   raw bytes reproduce the original exactly; reduction independently recomputed:
   **−959.4 MiB (41.6%)**. Original archived verbatim; checksums in EVIDENCE_INDEX.md.
4. **Runner-migration handoff defect (outside this review's scope, same session):** the kept
   queued run `42381769814` was bound to the pre-migration workflow file (`$HOME`-derived venv)
   and could never validate the migrated runner — cancelled; see `docs/r1.1/ACTIONS_QUEUE_AUDIT.md`.

**Verdict: all headline claims SURVIVE at the final numerical code `45a0ed1`; finding 2 closed;
no unverified performance claim remains in the v0.4.0 documents.** Default stays `reference`;
promotion remains a separate owner decision.
