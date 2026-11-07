# R1 PR #1 Review — scientific-validity release audit

Date: 2026-11-07 · PR #1: `v0.2.1-scientific-validity` @ `79412f8a8b4fcb09f09f4a41b6eb07d075dbe3b1` (Draft, base main, MERGEABLE/CLEAN, CPU CI SUCCESS ×2).
Method: two independent passes — (i) mechanical (this session): detached worktree `~/wt-pr1-audit`, fresh CPU suite, fresh GPU gate run on gfx1151, independent receipt-arithmetic recompute; (ii) adversarial code review by a fresh-context reviewer agent (full report preserved in §4 summary; every scope item confirmed with file:line evidence, re-executed checks, no prior report trusted).

## Verdict: **CONDITIONAL GO** (content GO; merge awaits explicit owner approval of the exact SHA; two tiny pre-tag hygiene items listed in §3)

## 1. Mechanical verification (fresh, this session)

| Check | Result |
|---|---|
| CPU CI suite on the PR tree (torch 2.15.0+cpu, `LSA_SKIP_GPU_TESTS=1`, PYTHONPATH=src) | **117 passed / 16 skipped / 0 failed** (identical counts expected: 133 collected; the 16 skips are loud GPU-availability module skips) |
| Static workflow guard + guard unit tests | PASS · 10/10 |
| **GPU 16-gate manifest on gfx1151 (trusted local path, worktree src)** | **16/16 PASS** — A, A2, B1, B2, C, D1/D2 ×{fp32,fp16,bf16}, E, F, G ×{fp32,fp16,bf16} (module confirmed loaded from the worktree's src, not the main tree's 17-gate version) |
| Independent receipt recompute (r1_verify_receipts) | overnight-bf16 36/36, overnight-fp16 36/36, pilot 12/12, post-fix 12/12, repetitions 27/27 — **every stored speedup == baseline/candidate recompute (1e-9)**; audit-v0.2.1.json 6/6 rows APPROVED, **111/111 recomputed match** |
| Provenance | all receipts: `repo_head_sha` `82c891e2…`/`2139758e…` are ancestors of `79412f8` (verified `git cat-file`/`merge-base`); `upstream_sha=5ebea732…` == pin; `gpu_arch=gfx1151`; benchmark path byte-identical between receipt SHAs and HEAD (only auditor/run.py changed after 82c891e) |

## 2. The original baseline defect (documented, not rewritten)

v0.2.0's dense-decode baseline ran `is_causal=True` on a single-token query → top-left mask → attended only key 0, running ~2.8× slower than a correct dispatch and inflating every decode speedup. PR #1's `dense_read` (pipeline.py:246-276) handles square/single-row/chunk shapes correctly with an explicit bottom-right-aligned chunk mask; gate G pins decode-row and 16-row-chunk == square-prefill tail per dtype; gates D2+E close the baseline-equals-oracle chain. All v0.2.0 decode claims are superseded in-tree (SUPERSEDED banner, strike-throughs, no carry-forward found by grep).

## 3. Findings & dispositions (independent reviewer + this session)

| ID | Sev | Finding | Disposition |
|---|---|---|---|
| F10 | LOW | `bench.py:3-4` docstring says "CUDA events" but timing is synchronized `perf_counter` | Pre-existing, in-tree-acknowledged; measurement itself sound. Fold into the optional pre-merge hygiene commit (§5) or fix post-merge on main. |
| F11 | LOW | PR #1 tree's `evidence/SHA256SUMS` lines 23-24 are repo-root-relative while 1-22 are `evidence/`-relative (digests all correct; auditor resolves relative to the sums file, so the 6 claim receipts are covered) | Already fixed on the PR #5 branch (which carries these files forward); main gets the fix when PR #5 merges. Optional: include in the pre-merge hygiene commit so the v0.2.1 tag window is clean. |
| F12 | LOW | `audit_receipts.py --expect-repo-sha` defaults to None — receipt-SHA∈branch membership not machine-enforced at audit time | Process note: R1's verifier enforces membership independently (§1); propose passing `--expect-repo-sha $(git rev-parse HEAD)` in the GPU workflow's audit step post-migration (workflow change, reviewable, owner-gated). |
| F13 | LOW | `pyproject.toml` version = "0.2.0" while the tree/docs/CHANGELOG are v0.2.1 | **Pre-tag fix required** (one line). See §5 options. |
| F14 | INFO | Disclosed limitations: v0.2.1's 3 decode repetitions share one seed (superseded by v0.3.0's 9-receipt multi-seed studies); p95==max at 20 samples (conservative); upstream junit `<skipped>` counted as passed (0 skips in the recorded run); `LSA_SKIP_GPU_TESTS` module-skips 2 CPU-runnable tests | All in-tree disclosed; no action for v0.2.1; the general test-portability pattern was already improved in v0.3.1 (`5a13bf0`). |

No CRITICAL/HIGH findings. No silent skips or suppressed failures (grep: only GPU-availability skipifs; zero xfail; zero body `pytest.skip`). Historical review `docs/reviews/PR1_FINAL_INDEPENDENT_REVIEW.md` claims (133 tests, 16/16, 203/7/0 upstream, 6 receipts/111 comparisons) all reproduced by this audit; its extra-commit drift (e243ba4→79412f8) verified docs-only.

## 4. Acceptance-gate checklist (mission §3)

- Complete code review: ✅ (independent agent, 9/9 scope confirmed)
- No unresolved critical correctness defects: ✅ (none found; LOW items dispositioned)
- CPU CI green on reviewed code: ✅ (117/16/0 local + GitHub SUCCESS ×2)
- Required v0.2.1 gates passing: ✅ 16/16 fresh on gfx1151
- Valid receipt provenance: ✅ (SHAs in-branch, upstream pin, gfx1151, arithmetic recomputed)
- Acceptable runner security: ⚠️ **RUNNER HARDENING BLOCKED** — merge does not require the runner (post-merge validation uses the trusted local path; GitHub-dispatched GPU jobs stay off until R1_RUNNER_SECURITY §3 items land)
- No silent skips/suppressed failures: ✅
- Explicit owner approval for merge: ⏳ **pending** (R1_MERGE_PLAN §1)

## 5. Pre-merge hygiene options (owner picks; R1 does not execute without approval)

- **Option A (recommended): one hygiene commit on the PR #1 branch** — `pyproject` version → 0.2.1 (F13), normalize the 2 SHA256SUMS path lines (F11), bench.py docstring (F10). ~4 lines, CPU CI re-runs, owner then approves the new head SHA for merge. Keeps the v0.2.1 tag window fully clean.
- **Option B: merge `79412f8` as-is** and land F10/F11/F13 on main immediately post-merge (F13 must land before the `v0.2.1` tag; F11 before PR #5 merge anyway).

GPU validation note: fresh trusted-local GPU execution at this exact tree = the 16/16 gate run in §1 (provenance: this session, gfx1151, worktree src, exit 0). No new performance measurement was taken or claimed.
