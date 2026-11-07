# CONTINUATION — exact resume point for the next session

Updated: 2026-11-06 (session: F10/F11/F12 closure + independent review)
Branch: `v0.2.1-scientific-validity` · head `e243ba4e03f996e8650b4eb143dd678eff13c643`
PR: #1 (Draft, review-ready) · main: `5f2918e3` · upstream pin: `5ebea732` (== upstream main)

## What just happened (verified, do not re-run)

- F10/F11/F12 closed, committed `2139758` + CI fixes `e243ba4`; pushed.
- Live validation: PR events → CPU-only workflow (runs 42139138701→42139629501);
  trusted GPU dispatch on the PR branch succeeded (run 42139297335, receipt
  APPROVED); mismatched `confirm_ref` dispatch failed closed in 20s (run 42139650668).
- All 6 v0.2.1 claim receipts APPROVED by the new independent auditor
  (`evidence/audit-v0.2.1.json`, 111 comparisons recomputed).
- Fresh local evidence at head: gates 16/16, upstream 203/7/0, pytest 133 passed,
  post-fix receipt `evidence/receipt-2139758-postfix-verification.json`.
- Independent review written: `docs/reviews/PR1_FINAL_INDEPENDENT_REVIEW.md` → **GO
  (merge-ready) for v0.2.1 validity scope; NO-GO beyond**; L1–L5 open LOW findings.
- Ledgers created: `tracking/PROJECT_STATE.json`, `tracking/MILESTONES.yaml`,
  `docs/DECISION_LOG.md`, `docs/EVIDENCE_INDEX.md`.

## Awaiting owner

- AWAITING_OWNER_APPROVAL: merge PR #1, then tag `v0.2.1` only after a trusted GPU
  run on the post-merge main SHA is green. Runner hardening items:
  `docs/SECURITY_RUNNER_POLICY.md` §4.

## ONE next most important action

Create the v0.3.0 work-plan branch: `git checkout -b v0.3.0-kernel-optimization
v0.2.1-scientific-validity` … write `docs/OPTIMIZATION_PLAN_v0.3.md` (start with the
seed-study + read-kernel profiling tasks from `tracking/MILESTONES.yaml`), open a
Draft PR against `main` stating it is staged behind PR #1. No GPU campaign until the
plan's profiling scripts exist; the seed-study (≥3 seeds, 32k bf16 decode) is the
first safe GPU task if the machine is idle.

## Resume commands

```bash
cd ~/Desktop/lsa-rocm-eval-starter && git status --porcelain=v1 && git log --oneline -3
gh pr view 1 --json state,isDraft,headRefOid   # has the owner acted?
.venv/bin/python -m pytest tests/ -q           # baseline must stay 133 passed
python3 scripts/check_workflow_security.py     # routing guard must stay green
gh run list --limit 5                          # CI posture
```

## Risks / gotchas (fresh this session)

- GitHub-hosted CPU runners need `PYTHONPATH=src`, a `triton` wheel and the pinned
  upstream checkout or the port import chain fails collection (fixed in cpu-pr.yml).
- `checkout_upstream.sh` now pins the reviewed SHA from tracking/UPSTREAM_BASELINE.json;
  moving the pin requires an explicit `UPSTREAM_REF` + review.
- Unquoted YAML step names containing ": " break parsing (again).
- Do not let the auditor's run-id regex drift: accepted shapes are
  `<ts>Z`, `<ts>Z-adapter`, `gh-<id>`, `rep<N>-<ts>Z`.
