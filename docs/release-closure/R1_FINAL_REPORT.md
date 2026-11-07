# R1 Final Report — production release closure (audit + hardening phase)

Date: 2026-11-07 · Branch head: `7907dae` (v0.3.0-kernel-optimization, PR #5, Draft) · main `5f2918e3` unchanged · PR #1 `79412f8` unchanged (Draft) · **0 tags, 0 releases, no merges, no host changes, no history rewrites, no secrets touched.**

## 1. What R1 completed (all evidence fresh this session)

| Phase | Deliverable | Result |
|---|---|---|
| §1 state freeze | R1_BASELINE.md | all 7 reference SHAs unchanged; stacking `60c004b`⊂`0908842`⊂`261805c`, `79412f8`⊂PR5 verified; drift-free |
| §2 runner security | R1_RUNNER_SECURITY.md | read-only preflight (runner OFFLINE, 3 installs, amd-user + live gh cred present, sudo password-required, device groups mapped); repo-control gap table; minimum-privilege migration design with exact commands, offline-verified systemd unit, dry-run, rollback; guard adversarially validated then **hardened after the review found real bypasses**; verdict **RUNNER HARDENING BLOCKED** → zero GitHub-dispatched GPU jobs in R1 |
| §3 PR #1 audit | R1_PR1_REVIEW.md | **CONDITIONAL GO**: fresh 16/16 gates on gfx1151 at `79412f8`; CPU CI 117/16/0; 141 receipt comparisons + 111/111 audit recomputes all match; independent reviewer confirmed 9/9 scope items incl. the v0.2.0 baseline defect mechanism; residuals are LOW (version bump, 2 manifest path lines, docstring) |
| §6 PR #5 audit | R1_PR5_REVIEW.md, R1_EVIDENCE_AUDIT.md | **CONDITIONAL GO**: 17/17 gates × both tree impls at audit HEAD; 186 (now 192) gfx1151 tests; CPU CI 135→141 passed after R1's own fixtures; **153/153 + 153/153 + 141+111 comparisons independently recomputed**; `sha256sum -c` 71/71; claims hygiene verified with two provenance exceptions found and corrected (F4) |
| §4/5/7/8 merge/release prep | R1_MERGE_PLAN.md, R1_RELEASE_NOTES_v0.2.1.md, R1_RELEASE_NOTES_v0.3.1.md | approval ledger A1–A8(+6), controlled merge-commit runbooks with pre-flight drift checks, retarget procedure, post-merge hard gates, release-notes drafts with provenance slots — **nothing executed** |
| §9 adversarial review | findings F1–F10 below | independent fresh-context reviewer; **GO-WITH-FINDINGS**; every code/docs fixable finding fixed and re-verified this session |
| §10 deliverables | this directory, 9 files + verifier script | committed `c9a792b` (package), `2af35b9` (guard fix), `58b80a3` (docs dispositions), `7907dae` (guard-pin CI fix); CPU CI green at `7907dae` ([run 42377333669](https://github.com/AIwork4me/latent-sparse-attention-rocm/actions/runs/42377333669), [run 42377338235](https://github.com/AIwork4me/latent-sparse-attention-rocm/actions/runs/42377338235)); the intermediate `58b80a3` failure is recorded honestly (guard-from-`origin/main` assumed the script exists on pre-PR#1 main; fixed by pinning the reviewed commit SHA) |

## 2. Adversarial findings and dispositions (§9 — full table)

| ID | Sev | Finding (verified) | Disposition (status) |
|---|---|---|---|
| F1 | **HIGH** | Guard bypassable: custom label sets (`[linux, rocm, gfx1151]`), `${{ matrix.runner }}`, `workflow_call` composition all passed `_is_self_hosted`'s literal-string check — the exact untrusted-PR→persistent-host routes it exists to block; repo-internal owner actions (env reviewers, SHA pinning) do not close it | **FIXED** `2af35b9`: hosted-label allowlist (only `ubuntu-*`/`macos-*`/`windows-*`; expressions/unknown/missing = persistent-host), new R6 (workflow_call + non-hosted ⇒ reviewed-marker required). All reproduced bypasses now caught; 16/16 unit tests (6 new fixtures); live workflows pass. Residual (repo code can't fix): runner-group scoping → owner packet item 6 |
| F2 | MEDIUM | R5 secret scan missed workflow-level `env:`/`defaults:` (repro: wf-level `secrets.GH_PAT` + self-hosted job → exit 0) | **FIXED** `2af35b9`: file-scope R5 when any non-hosted job exists; fixture added |
| F3 | MEDIUM | Guard self-referential in fork PRs (ran from the PR's own checkout — a fork PR can neuter it in the same PR) | **FIXED** `7907dae`: cpu-pr.yml re-runs the guard from **pinned commit `2af35b9`** (first attempt used `origin/main` and correctly failed CI — script not on pre-PR#1 main; the pinned-SHA approach verified from a fresh clone) |
| F4 | MEDIUM | Public claim "decode select 1.24× (0.266→0.214 ms)" has **no retained artifact** (exhaustive search); receipts' own stage data gives 0.528→0.466 ms (1.13×); R1_EVIDENCE_AUDIT's universal-traceability sentence was false | **FIXED** `58b80a3` + PR-body patch: receipt-backed **1.13×** now leads in release notes/PR5/audit/OPTIMIZATION doc; 1.24× explicitly labeled a development in-process A/B (artifact not retained); audit sentence corrected with both artifact-class exceptions (incl. tiled A/B = `ab-read-kernel.json` class) |
| F5 | MEDIUM | Discredited 0.81–0.87× T4 figure survived in `tests/test_tree_batched_pages.py` docstring + `docs/DECISION_LOG.md` (0908842 fixed only `_tree.py`); PR5-review P5-1 "closed" was wrong | **FIXED** `58b80a3`: both locations corrected to audited 1.16–1.86×-slower; grep-clean (only historical/methodology-rejection contexts remain); P5-1 disposition corrected in R1_PR5_REVIEW |
| F6 | MEDIUM | v0.3.1 tag hygiene gap: `pyproject` still `0.2.0`, no CHANGELOG 0.3.x entries; package flagged this class for v0.2.1 but not v0.3.1 | **PARTIALLY FIXED** `58b80a3`: CHANGELOG `[0.3.0]`/`[0.3.1]` entries added; version bump is now an explicit pre-tag gate (A5/A8) — owner-approved one-liner before each tag (kept out of R1 to avoid SHA churn under review) |
| F7 | LOW | v0.2.1 release notes referenced `scripts/r1_verify_receipts.py`, absent at that tag | **FIXED**: note added (script lives on the v0.3.x branch; hand-recompute rule stated) |
| F8 | LOW | "38 receipts" count untraceable | **FIXED**: counting rule stated — 42 unique audited (6 matrix + 36 seed-study) + 2 A/B artifacts; "38" was the pre-final-matrix count |
| F9 | LOW | v0.3.1 notes pinned PR5 head to audit-time `261805c` | **FIXED**: provenance now says tag targets the branch head at merge time |
| F10 | INFO | cuda-params vanish silently when GPU invisible at collection; `LSA_SKIP_GPU_TESTS=1` doesn't gate that file | **Documented** (CONTINUATION gotcha); cpu params can never vanish — no hidden-failure risk |

**No CRITICAL findings; the one HIGH (F1) is fixed in code with the residual assigned to the owner packet. All fixable findings are fixed and re-verified; remaining owner-gated items are explicit (below).**

## 3. Fresh verification at final branch head `7907dae`

- GitHub-hosted CPU CI: **141 passed / 46 skipped / 0 failed** (both trigger events; security-guard steps PASS incl. the pinned-source step)
- Local CPU-wheel CI: 141 passed / 46 skipped / 0 failed (identical)
- gfx1151 full suite: **192 passed**; guard unit suite 16/16 (both pythons); live workflows pass the hardened guard; all adversarial fixtures (reviewer's + R1's) caught
- Evidence: `sha256sum -c evidence/SHA256SUMS` 71/71; verifier reproduces "ALL VERIFIED" on v0.3.1 + v0.3.0-final + v0.2.1 sets
- Host: LSA runner still OFFLINE; no GPU jobs dispatched; sibling runners untouched

## 4. Stage verdicts

| Stage | Verdict | Condition to advance |
|---|---|---|
| PR #1 (v0.2.1) | **CONDITIONAL GO** | owner approves exact merge SHA (+ optional Option-A hygiene commit: version bump, manifest paths, docstring) |
| v0.2.1 Release | **CONDITIONAL GO** | PR #1 merged → post-merge CPU CI + 16/16 gates at merge SHA → version bump → tag+draft-release (owner) |
| PR #5 (v0.3.0+v0.3.1) | **CONDITIONAL GO** | PR #1 merged → retarget to main → owner merge approval → **fresh 17/17×2 gates at merged SHA (hard gate)** |
| v0.3.1 Release | **CONDITIONAL GO** | §7 merged-tree validation green → version bump + CHANGELOG check → tag+draft-release (owner) |
| Runner hardening | **BLOCKED (owner)** | A1 env reviewers · A2 SHA-pin enforcement · A3 migration (§2.1) · A4 cred rotation · A6 runner-group scoping — until then no GitHub-dispatched GPU jobs |

## 5. Owner approval request (immediate, in order)

1. **A1+A2+A6** (settings, minutes, reversible): env required-reviewers, `sha_pinning_required=true`, runner-group scoping + external-contributor approval.
2. **PR #1 merge decision** on `79412f8` (or Option-A hygiene head) — merge commit.
3. **A3 runner migration** execute-or-defer (packet in R1_RUNNER_SECURITY §2; GPU dispatch stays blocked meanwhile; R1 needs no runner for any planned validation).
4. After PR #1: **PR #5 retarget + merge decision** on the then-head, followed by the §7 merged-tree validation R1 will run locally.

## 6. Reproducibility

Every R1 claim is re-checkable: `scripts/r1_verify_receipts.py` (committed), `sha256sum -c evidence/SHA256SUMS`, the gate commands in the release notes, CI run links above, and the adversarial-review artifacts (bypass fixtures reproducible from the findings text; guard tests in `tests/test_workflow_security.py`).
