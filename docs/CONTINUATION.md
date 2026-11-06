# CONTINUATION — exact resume point for the next session

Updated: 2026-11-07 ~04:20 local (campaign `campaign-20261106T231500Z`, v0.3.0 complete)
Branch: `v0.3.0-kernel-optimization` · head `25c7bbc` (docs-only on top of code head `1e73c3a`)
PR: #5 (Draft, stacked behind PR #1 — do not merge before it) · PR #1: still Draft at `79412f8` (owner has NOT acted)
main: `5f2918e3` · upstream pin: `5ebea732` (== upstream main)

## What just happened (verified, do not re-run)

- **T1 seed study** (issue #2): 3 studies, 27 receipts total, all auditor-APPROVED.
  Honest bands replace the v0.2.1 single-seed "2.1×" flat-decode headline.
  Final (at `1e73c3a`): 32k flat 1.98× med (1.76–2.01), block-sparse 4.20×
  (4.02–4.21), tree 0.95×. Dense baseline is seed-stable; the spread is
  genuine sparse-path input sensitivity.
- **T2 profiling**: `docs/OPTIMIZATION_v0.3.md` §a–e. sparse_read prefill =
  299 ms = 77% of the flat step (single launch, gather-bound); tree select
  19,940 launches/iter; select sort+copy ≈ 48% of the 95 ms prefill select.
- **T3 tiled read (banked)**: `read_kernel_tiled.sparse_read_auto` is the
  production path (TM4/CN32 prefill = 1.24/1.18/1.14× at 8k/16k/32k on
  identical inputs; TM1/CN64 decode parity). New gate H (manifest now 17
  gates; v0.2.1 16-gate receipts grandfathered). 32k prefill flat
  0.19× → **0.211×**. Zero-length NaN hazard fixed + gated. B>1 row-mask bug
  caught by the ragged-tile test.
- **T4 tree beam (negative result)**: page-batched torch walk is set-equal
  (after fixing a picked-validity bug the re-measure exposed) but 1.16–1.29×
  slower prefill / 1.78–1.86× decode → launch tax is NOT the cost; production
  keeps the per-page walk; batched stays behind `LSA_TREE_IMPL=batched`.
  Artifact: `scripts/ab_tree_select.py` → `evidence/receipts-v0.3.0/ab-tree-select.json`.
- **Independent adversarial review**: GO-WITH-FINDINGS; F1–F7 all closed at
  `1e73c3a` with fresh 11/11 APPROVED receipts (153 comparisons). Disposition:
  `docs/reviews/V0_3_INDEPENDENT_REVIEW.md`.
- Tests 176 passed; gates 17/17 (both tree impls); security guard green;
  pushed through `25c7bbc`.

## Awaiting owner (unchanged)

- PR #1 merge + tag v0.2.1 (after a trusted GPU run on the post-merge main SHA).
- Runner hardening per `docs/SECURITY_RUNNER_POLICY.md` §4 (issue #3).
- PR #5 review; **merge condition**: fresh 17/17 receipt at whatever SHA merges.

## ONE next most important action

Start v0.3.1 on a branch off `v0.3.0-kernel-optimization` (or continue in PR
#5 if the owner prefers): **fuse/retype the flat-select sort chain** — the
measured #1 remaining lever (sort+copies ≈ 48% of the 95 ms prefill select;
int32 keys instead of int64 radix sort, or a fused pack→sort→expand Triton
pass). Alternatives in measured order: `docs/OPTIMIZATION_v0.3.md` §h.
If the owner merges PR #1 first: retarget PR #5 per §13 of the mission doc.

## Resume commands

```bash
cd ~/Desktop/lsa-rocm-eval-starter && git status --porcelain=v1 && git log --oneline -3
gh pr view 1 --json state,isDraft,headRefOid   # has the owner acted?
gh pr view 5 --json state,isDraft,headRefOid
.venv/bin/python -m pytest tests/ -q           # baseline must stay 176 passed
PYTHONPATH=src .venv/bin/python -m lsa_rocm_lab.port.gates | tail -1   # 17/17
python3 scripts/check_workflow_security.py     # routing guard must stay green
gh run list --limit 5                          # CI posture
```

## Risks / gotchas (fresh this session)

- Profiler-inflated "GPU totals" exceed unprofiled wall on this ROCm build —
  attribute by kernel-table SHARES, cost by synchronized p50 (see
  OPTIMIZATION_v0.3.md method note).
- `pkill -f <pattern>` matches your own heredoc's command line — kill by PID.
- Full-batch tree einsum gathers are multi-GiB at 32k (UMA thrash) — chunked
  PAGE_CHUNK=4; any batched-tree work must re-run `scripts/ab_tree_select.py`.
- An ad-hoc benchmark with warmup=3 measured tree decode ~2× the receipt
  baseline; A/B scripts must use receipt methodology (warmup 5 / reps 20) and
  gate equivalence before timing — the reviewer caught exactly this (F2).
- Kill leftover GPU python from prior campaigns before benchmarking (this
  session found one at 100% CPU for ~25 h contaminating UMA timings).
