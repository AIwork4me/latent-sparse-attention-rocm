# v0.4.0 Evidence Index

Every v0.4.0 claim in `MEASURED_SELECT_FUSION.md` / `BASELINE_AND_BOTTLENECKS.md` / `EXPERIMENT_DECISIONS.md` traces to an entry here. Append-only: nothing under `evidence/v0.4.0/` is ever edited or regenerated after commit.

| ID | What | Path / anchor | Integrity |
|---|---|---|---|
| V040-A | B1 substage baseline matrix: production select stages, 5 ctx × {bf16,fp16} × {prefill, decode B=1, ragged B=32}, warm p50/p95 + launch counts + peak alloc + raw samples; production code `1ab0e0b` (src clean, recorded in-artifact) | `evidence/v0.4.0/baseline/select-stages-matrix.json` | sha256 `8200be85…` (printed at write) |
| V040-B | B1 kernel-level profiles (warm, device-only attribution): 32k + 16k bf16; topk family ≈60–65% of prefill select GPU time | `evidence/v0.4.0/baseline/profile-warm-32k-bf16.json`, `profile-warm-16k-bf16.json` | in-repo |
| V040-C | B1 cold-start profile (isolated Triton cache): first-call compile costs | `evidence/v0.4.0/baseline/profile-cold-32k-bf16.json` | in-repo |
| V040-D | tl.topk/tl.sort int64 exactness + single-CTA timing probe (design input, session artifact — reproduced by any run of the documented snippet in `EXPERIMENT_DECISIONS.md` §probe) | `docs/v0.4.0/EXPERIMENT_DECISIONS.md` §Probe | doc-recorded |
| V040-E | B4 A/B receipts, `fused`: 9 process runs = 3 seeds (`424243`, `777001`, `20261105`) × 3 reps, full matrix, interleaved, warmup 5 / reps 20; per-row byte-exact correctness pre+post; impl SHA `9fb32f9`, upstream `5ebea732`, GPU state embedded | `evidence/v0.4.0/ab/fused-s{seed}-rep{1,2,3}.json` (run ids `rep<N>-<ts>Z`) | sha256s printed at write; per-file in commit message trail |
| V040-F | B4 A/B receipts, `fused_unpack` (F1 alternative): 3 runs (one per seed) | `evidence/v0.4.0/ab/fused-unpack-s{seed}-rep1.json` | as above |
| V040-G | B4 aggregation (observed ranges, per-seed medians, pooled medians, run spreads) — inputs V040-E / V040-F | `evidence/v0.4.0/ab/aggregate-fused.{json,md}`, `aggregate-fused-unpack.{json,md}` | derived, regenerable by `scripts/aggregate_v040_ab.py` |
| V040-H | Peak-memory measurement (32k bf16 select: reference 2304 MiB → fused 1345 MiB, fused_unpack 2369 MiB) — command in `MEASURED_SELECT_FUSION.md` §3 | session-measured, command + outputs recorded there | doc-recorded |
| V040-I | Correctness: 34-test byte-exact equivalence suite + 226/226 full suite + 17/17 gates × {default, batched, fused} (run log in PR description) | `tests/test_select_fusion.py` | CI-rerunnable |

## Immutability rules

- Historical areas (`evidence/v0.2.0-archive`, `receipts-v0.3.0*`, `receipts-v0.3.1-final`, `evidence/SHA256SUMS`) are untouched by v0.4.0; v0.4.0 adds only `evidence/v0.4.0/`.
- The v0.3.x `SHA256SUMS` manifest is NOT extended in place; a v0.4.0 manifest lands with the release-tagged tree if/when the owner promotes (pre-tag hygiene, same as A8 in R1).
- Every receipt pins `production_code_sha`; the measured production tree is `9fb32f9` (impl) — documentation-only commits after it are distinguished in the branch history.
