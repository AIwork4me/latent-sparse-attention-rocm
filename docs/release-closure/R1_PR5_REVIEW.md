# R1 PR #5 Review — independent release audit (v0.3.0 + v0.3.1)

Date: 2026-11-07 · PR #5: `v0.3.0-kernel-optimization` @ `261805c98c41230c2bfc32533a78873857058fdf` (Draft, base `v0.2.1-scientific-validity`, MERGEABLE, CPU CI SUCCESS ×2 — runs 42371502625/42371506680).

## Verdict: **CONDITIONAL GO** (content and evidence GO; merge awaits retarget-after-PR#1, owner approval, and the fresh 17/17 merged-tree receipt that is the standing merge condition)

## 1. Correctness (§6.1) — all fresh at the exact HEAD `261805c`

| Check | Result |
|---|---|
| Gate manifest (`src/lsa_rocm_lab/gate_manifest.py`) | **exactly 17 required ids** (A, A2, B1, B2, C, D1/D2 ×{fp32,fp16,bf16}, E, F, G ×{fp32,fp16,bf16}, H); `verdict_of` fail-closed: missing/unexpected/duplicate/failed all break `ok` — vacuous lists can never pass |
| Gates, default tree impl, gfx1151 | **17/17 PASS** (exit 0) |
| Gates, `LSA_TREE_IMPL=batched` | **17/17 PASS** (exit 0) |
| Local pytest on gfx1151 | **186 passed** (topk equivalence tests parametrized cpu+cuda) |
| GitHub-hosted CPU CI | **135 passed / 46 skipped / 0 failed** — 46 are loud GPU-availability skips; no hidden failures; the five former `test_select_topk.py` failures (runs 42260195840/42260187512, hardcoded `device="cuda"`) fixed by device parametrization at `5a13bf0` — every equivalence case still executes on CPU (tied scores, EMPTY keys, k==width, ragged rows, production shapes) |
| Tiled sparse-read dispatch contracts | gate H every receipt run; ragged/zero-length/GQA/boundary covered by gates D1/D2/F/H + `tests/test_read_kernel_tiled.py` (incl. the B>1 row-mask regression case) |
| Dense baseline correct | unchanged from PR #1 (gate G family PASS in both runs above) |
| topk ≡ sort+slice | `best_first_keys` = `torch.topk(sorted=True)` on column-unique packed keys; equivalence pinned by 10 GPU + 5 CPU executions of the 5 contract cases |
| GPU tests run on ROCm when available | yes — same test ids parametrize `cuda` when `_gpu_available()` (torch.cuda.is_available() ∧ torch.version.hip) |

Historical reference values from the mission (186 / 135+46 / 17+17) — **reproduced exactly at this HEAD**.

## 2. Evidence integrity (§6.2) — see R1_EVIDENCE_AUDIT.md for the full table

Independent recompute (not trusting any stored APPROVED): receipts-v0.3.1-final bf16+fp16 (36+36) + seed-study 9×9 = **153/153 speedups recompute**; audit-v031 11/11 rows APPROVED with 153 recomputed (matches); receipts-v0.3.0-final 153/153 likewise; `sha256sum -c evidence/SHA256SUMS` **71/71 OK**; every receipt `repo_head_sha` (60c004b/1e73c3a) in-branch; upstream pin + gfx1151 on all. Historical evidence untouched; receipts stay pinned at their SHAs and are never presented as executions at `261805c`.

## 3. Performance claims (§6.3) — README / BENCHMARKS / PR body / release notes

All mandated distinctions verified present: prefill vs decode; select-only vs read-only vs select+read; stage-level speedups never conflated with dense-baseline ratios; per-seed medians + cross-seed bands + pooled median ("cite the band, not one run"); bf16/fp16 sections; `LSA_TREE_IMPL=batched` labeled experimental behind env; device/torch/rocm/triton pinned per table; explicit not-claimed section (no E2E SGLang — BLOCKED_ON_REASSESS). Preserved findings intact: flat decode 1.94× (1.85–2.06), block-sparse 4.29× (4.24–4.38), tree 0.93× (0.92–0.99, not faster), flat prefill 0.210×, tiled read 1.14–1.24× prefill-stage (A/B artifact `ab-read-kernel.json`, recomputed exact), topk decode-select-stage — **receipt-backed 1.13× (0.528→0.466 ms)**; the widely-quoted 1.24× (0.266→0.214 ms) was an in-process development A/B with no retained artifact (adversarial finding F4; public docs relabeled), page-batched tree 1.16–1.86× slower (negative result). BENCHMARKS.md is script-generated from audited receipts only.

## 4. SHA ledger (code vs docs — the mission's explicit distinction)

- Production code: `60c004b` (all perf receipts) → `0908842` (docstring-only T4 comment fix; proven non-behavioral by the fresh 17/17+17/17 above).
- Test-only: `5a13bf0`. Docs-only: `849a80e`, `47e63f7`, `2f3002c`, `261805c`(+R1 docs).
- Eventual merged SHA: TBD — **hard merge condition**: fresh 17/17 gates (both impls) at the merged tree per R1_MERGE_PLAN §3.

## 5. Findings

| ID | Sev | Finding | Disposition |
|---|---|---|---|
| P5-1 | LOW | `_tree.py`'s stale T4 docstring was fixed at `0908842` (was 0.81–0.87× vs audited 1.16–1.86× slower) — **initially marked "closed" here, which the adversarial review disproved (F5): the same stale figure survived in `tests/test_tree_batched_pages.py` and `docs/DECISION_LOG.md`** | corrected in the R1 adversarial-fixes commit (both locations now cite the audited artifact); grep-clean at the new HEAD |
| P5-2 | INFO | v0.2.1 16-gate receipts intentionally fail the current 17-gate manifest (fail-closed by design, documented) | by-design; noted in release notes |
| P5-3 | INFO | Runner-blocked: GitHub-dispatched GPU regression cannot run until RUNNER HARDENING BLOCKED lifts; post-merge validation will use the trusted local path (provenance-labeled) unless the owner completes A1–A3 first | tracked in R1_RUNNER_SECURITY/R1_MERGE_PLAN |
| P5-4..P5-12 | — | Adversarial review of the R1 package itself (fresh-context reviewer, GO-WITH-FINDINGS): F1 HIGH guard bypass (matrix/custom-label `runs-on`, `workflow_call` chains), F2 wf-level `env:` secrets, F3 guard self-reference in fork PRs, F4 topk provenance, F5 above, F6 version/CHANGELOG gate, F7–F9 release-notes consistency, F10 test-parametrization nuances | **F1/F2 fixed** (guard hardening: hosted-label allowlist + R5 file-scope + new R6, 16/16 unit tests incl. 6 new attack fixtures; all reviewer bypasses now caught), **F3 fixed** (cpu-pr.yml re-runs the guard from `origin/main`), **F4 fixed** (receipt-backed 1.13× leads; 1.24× relabeled), **F5 fixed**, **F6 partially** (CHANGELOG entries added; version bump is an owner pre-tag step A8), F7–F9 fixed in these notes/audit text, F10 documented in CONTINUATION. Full table in R1_FINAL_REPORT §adversarial. |

No CRITICAL/HIGH findings in code, evidence, or claims.

## 6. Acceptance checklist

Content ✅ · evidence ✅ (recomputed) · claims hygiene ✅ · CPU CI ✅ · gates 17/17 ×2 ✅ · no hidden skips ✅ · stacking intact ✅ · owner approval ⏳ · retarget to main ⏳ (after PR #1) · merged-tree fresh receipt ⏳ (hard gate).
