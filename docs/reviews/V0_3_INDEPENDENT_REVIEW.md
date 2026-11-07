# v0.3.0 independent adversarial review — disposition record

Reviewed: PR #5 branch `v0.3.0-kernel-optimization` at `40526c9` (post-T1/T2/T3/T4).
Reviewer: independent adversarial agent pass (fresh context, GPU access, ran its
own negative tests and re-audited every receipt). Verdict: **GO-WITH-FINDINGS**.
Disposition of each finding below; fixes landed at `1e73c3a` + final receipts.

| Finding | Severity | Reviewer's point | Disposition |
|---|---|---|---|
| F1 | MEDIUM (merge condition) | Receipts pinned `4988672`; the docs commit `40526c9` carried `_tree.py` source changes they never ran against; v0.2.1's re-run-at-final-SHA precedent was dropped | Fixed: source fixes (incl. F2–F7) committed as `1e73c3a`; **fresh full-matrix bf16+fp16 receipts AND a fresh 9-receipt seed study re-run at `1e73c3a`**, all 11 APPROVED (153 comparisons recomputed, `evidence/receipts-v0.3.0-final/`, `evidence/seed-study-20261106T170105Z/`); only docs-only commits after. A re-run at the eventual merge SHA remains an owner-side merge condition |
| F2 | MEDIUM | T4 negative-result table had no backing artifact and its decode baselines contradicted the receipts ~2× | Fixed: committed `scripts/ab_tree_select.py` (set-equality-gated A/B, receipt methodology) → `evidence/receipts-v0.3.0/ab-tree-select.json`; doc table replaced (batched 1.16–1.29× slower prefill, 1.78–1.86× decode), under-warmed ad-hoc numbers retracted in-doc. Re-measuring exposed a real batched-walk validity bug (sentinel picks ≥0 flowed as real ids) — fixed by gathering picked validity from candidate validity; set-equality now verified per-run by the A/B itself |
| F3 | LOW | `lengths > width` → illegal memory access in BOTH read kernels; wrappers validate nothing | Fixed: `sparse_read_tiled` clamps `lengths` to `width` (no-op for production inputs; reviewer verified max length 4096 ≤ width 4612) |
| F4 | LOW | Gate H's zero-length leg only exercised the TM1 decode config | Fixed: H now checks zero-length on both dispatch regimes (8 rows TM1/CN64, 64 rows TM4/CN32). Reviewer independently ran the ragged-tile and TM4-zero-length cases on GPU — all equal v1 |
| F5 | LOW | README "1.15–1.24×" excluded the shipped config's 1.14× at 32k | Fixed: README says 1.14–1.24×, production config |
| F6 | LOW | Single bf16 receipt's decode rows sat below the published band | Addressed: README/BENCHMARKS now carry the explicit caveat "single-receipt runs can land below the band — cite the band" |
| F7 | LOW | Stale "16-gate"/"16/16" labels vs the 17-gate manifest | Fixed in README ×2 and the workflow step name |

What the reviewer tried to break and could not (their words, spot-verified here):
tiled-kernel equivalence on adversarial inputs (ragged B=2/Tq=13, GQA 6:2,
ids at TK−1/0, −1 inside counted length, over-counted lengths ≤ width, all
three dtypes; max diff 2.4e-4 bf16 / 7.5e-8 fp32); dispatch boundary
rows_total ∈ {31,32,33,34}; zero-length mixed batches (tiled exactly 0.0,
v1 NaN confirmed); gate suites 17/17 default AND `LSA_TREE_IMPL=batched`;
pytest 176/176; SHA256SUMS all OK; every BENCHMARKS cell traced to an
APPROVED receipt; prefill 0.207× (now 0.211× at final) recomputed from raw rows.

Post-fix verification at `1e73c3a` (this session, after the fixes):
pytest **176 passed**; gates **17/17 PASS** default and batched; 11 final
receipts APPROVED. Remaining owner-side conditions: merge decision on PR #1
(unchanged, Draft), PR #5 review, and a fresh receipt at whatever SHA merges.

**v0.3.0 verdict: GO (as Draft PR #5 content, for owner review — no merge
without owner approval, per policy).**
