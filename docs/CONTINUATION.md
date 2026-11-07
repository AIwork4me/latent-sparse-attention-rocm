# CONTINUATION — exact resume point for the next session

Updated: 2026-11-07 (release-readiness pass on PR #5; v0.3.0 + v0.3.1 complete)
Branch: `v0.3.0-kernel-optimization` · code head `0908842` (src = `60c004b` + a docstring-only T4 comment fix) · test-fix `5a13bf0` · docs-only commits above those
PR: #5 (Draft, stacked behind PR #1 — do not merge before it; title/body now describe the implemented v0.3.0+v0.3.1, not the plan) · PR #1: still Draft at `79412f8` (owner has NOT acted)
main: `5f2918e3` · upstream pin: `5ebea732` (== upstream main)

## SHA bookkeeping (explicit, per the mission contract)

- **Production code SHA** (what receipts/gates describe): `60c004b` = v0.3.1 topk select. `0908842` = `60c004b` + `_tree.py` docstring correction ONLY — proven non-behavioral (gates 17/17 both impls re-run at that tree; 186 unit tests pass). Perf receipts stay pinned at `60c004b`; **no new performance measurement was claimed this session.**
- **Test-only commit**: `5a13bf0` (tests/test_select_topk.py device portability; no src change).
- **Doc-only commits**: everything else on the branch above `60c004b` (`849a80e`, `47e63f7`, `2f3002c`, and this session's docs).
- **Eventual merged SHA**: NONE YET — merge condition remains a fresh 17/17 receipt at whatever SHA merges.

## What just happened (verified this session; do not re-run without cause)

- **GitHub CPU CI repaired**: runs 42260195840 / 42260187512 failed at `2f3002c` with exactly 5 failures in `tests/test_select_topk.py` (`Torch not compiled with CUDA enabled` — hardcoded `device="cuda"` on the CPU-only wheel). Fix `5a13bf0`: device-parametrized tests (`cpu` everywhere + visible ROCm device when present). No skip/xfail/relaxed assertions; tied-score, EMPTY, k==width, ragged, production-shape coverage all still execute. Green run: 42371335910 (and 42371339224). Local CPU-wheel venv reproduction: pre-fix 5 failed/130 passed/46 skipped (identical to CI), post-fix **135 passed / 46 skipped / 0 failed**. On gfx1151 the suite is now **186 passed** (was 181; each equivalence case runs cpu+cuda).
- **Evidence verified, not re-measured**: v0.3.1 final receipts at `60c004b` — 11/11 auditor-APPROVED, 153 comparisons recomputed, 17/17 gates, upstream `5ebea732`, gfx1151. Indexed as **EV-2026-11-07-J** in docs/EVIDENCE_INDEX.md.
- **`evidence/SHA256SUMS` repaired**: 2 of 71 lines had an erroneous `evidence/` path prefix (files exist, hashes matched) — prefixes fixed, hashes untouched; `cd evidence && sha256sum -c SHA256SUMS` now 71/71 OK. Historical evidence bytes immutable.
- **Stale T4 commentary fixed**: `_tree.py` docstring said "0.81-0.87x the per-page walk" (superseded early number); corrected to the audited artifact (`evidence/receipts-v0.3.0/ab-tree-select.json`): batched 1.16–1.29× slower prefill / 1.78–1.86× slower decode.
- **Gates re-verified at the reconciled tree**: 17/17 PASS, default AND `LSA_TREE_IMPL=batched`, on gfx1151 (this is a correctness re-run, not a new perf claim).
- PR #5 title + Description rewritten to the implemented state (see the PR); still Draft.

## Awaiting owner (unchanged)

- PR #1 merge + tag v0.2.1 (after a trusted GPU run on the post-merge main SHA).
- Runner hardening per `docs/SECURITY_RUNNER_POLICY.md` §4 (issue #3).
- PR #5 review; **merge condition**: fresh 17/17 receipt at whatever SHA merges.

## ONE next most important action

Owner review of PR #5 (title/body now describe the implemented work, positive
AND negative results, tested configs, evidence links, limitations, merge
conditions). Engineering next lever (only on owner go-ahead, measured order):
fused pack→expand select pass — `docs/OPTIMIZATION_v0.3.md` §h. If the owner
merges PR #1 first: retarget PR #5 per §13 of the mission doc.

## Resume commands

```bash
cd ~/Desktop/lsa-rocm-eval-starter && git status --porcelain=v1 && git log --oneline -5
gh pr view 1 --json state,isDraft,headRefOid   # has the owner acted?
gh pr view 5 --json state,isDraft,headRefOid
.venv/bin/python -m pytest tests/ -q           # baseline: 186 passed on gfx1151
PYTHONPATH=src .venv/bin/python -m lsa_rocm_lab.port.gates | tail -1   # 17/17
python3 scripts/check_workflow_security.py     # routing guard must stay green
gh run list --limit 5                          # CPU CI posture (green at/above 5a13bf0)
cd evidence && sha256sum -c SHA256SUMS | grep -cv ": OK$"   # must be 0
```

## Risks / gotchas (carried + fresh)

- Profiler-inflated "GPU totals" exceed unprofiled wall on this ROCm build —
  attribute by kernel-table SHARES, cost by synchronized p50.
- `pkill -f <pattern>` matches your own heredoc's command line — kill by PID.
- Full-batch tree einsum gathers are multi-GiB at 32k (UMA thrash) — chunked
  PAGE_CHUNK=4; any batched-tree work must re-run `scripts/ab_tree_select.py`.
- A/B scripts must use receipt methodology (warmup 5 / reps 20) and gate
  equivalence before timing.
- Kill leftover GPU python from prior campaigns before benchmarking.
- NEW: tests that are device-independent contracts should parametrize
  `["cpu"] + (["cuda"] if ROCm visible else [])` — never hardcode
  `device="cuda"` in files without the `LSA_SKIP_GPU_TESTS` module guard; the
  GitHub CPU wheel raises at allocation, not collection, so CI catches it
  only at run time.
- NEW: pip from download.pytorch.org can stall (0 B/s) for long stretches on
  this host — the aliyun mirror is ~7× faster for generic packages; torch CPU
  wheel must still come from the pytorch index.
