# v0.4.0 release GO/NO-GO memo (overnight_v1, 2026-11-08/09)

Advisory only. The tag/Release decision is the owner's; this memo separates what IS
verified tonight on the merged tree from what remains owner-gated. Nothing was published.

## Verified tonight (merged tree `83e410b` + ledger-only `614becc`, physical gfx1151)

| Check | Result |
|---|---|
| Full GPU suite | **314/314 PASS** (0 skip) — 258 from PR7 head + 56 release-integrity cases from PR9; GPU execution proven in-run (Triton fused wide-fallback warning from `pipeline.py:305`) |
| 17-gate manifest × {default, `LSA_TREE_IMPL=batched`, `LSA_SELECT_IMPL=fused`} | **17/17 PASS × 3** (receipts `m1-gates-{default,batched,fused}.txt`) |
| Evidence integrity | `evidence/v0.4.0/SHA256SUMS` **28/28 OK** |
| Independent aggregate recompute | **60/60 cells reproduced** from 9 per-seed raw receipts (second implementation, tol 5e-4); all pre+post byte-exact flags true; min pooled speedup 1.0469 — zero regression cells |
| Single-run A/B confirmation 16k+32k bf16 | **PASS** — 12/12 rows byte-exact pre+post; 32k prefill select 2.951×, prefill s+r 1.226×, 16k prefill s+r 1.180× — at/inside campaign bands; decode cells 1.16–1.26× (within/just below ≤3–4% run-to-run decode noise; no cell <1.0). Receipt sha256 `9b2026b2…` — explicitly a single-seed single-process confirmation, NOT a fresh 3×3 campaign |
| Workflow security guard | PASS (`check_workflow_security.py`) |

## Verdict

- **Engineering GO** for v0.4.0 release candidacy on the evidence that exists: merged-tree
  correctness fully green locally, sealed evidence fully reproducible, performance bands hold
  on a fresh confirmation run.
- **Release remains NO-GO to execute tonight** — owner gates unchanged:
  1. GitHub post-merge GPU run `42483727525` still `pending` (= NOT RUN; runner migration is
     `OWNER_ACTION_REQUIRED`; no environment approval attempted by this campaign).
  2. Tag/Release requires explicit owner authorization (RC review).
  3. PR #8 default promotion stays DRAFT (needs release + owner approval + real GPU regression).

## Honest caveats

- The 32k decode B1 select+read single-run value (1.179×) sits ~1% below the campaign's
  all-runs minimum (1.192×); prefill cells are inside bands. No regression cell anywhere.
- PR9 added 56 release-integrity tests; they pin "reference default + unreleased" wording —
  consistent with the current default (`reference`) checked tonight.
