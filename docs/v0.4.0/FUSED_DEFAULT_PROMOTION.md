# v0.4.0 — Fused-Default Promotion (minimal PR, owner-gated)

**Status: PREPARED, NOT MERGED.** This branch flips ONE production line — the
`LSA_SELECT_IMPL` default in `pipeline.flat_select` from `reference` to `fused` — plus its
docstring, the default-dispatch test, and this runbook. **Merge only after** PR #7 is merged,
v0.4.0 is released, and the owner has explicitly approved promoting the measured fusion to the
served default (the standing mission rule: `reference` default is preserved through PR #7's
integration unless promotion is separately approved).

## What changes

| | before | after |
|---|---|---|
| unset `LSA_SELECT_IMPL` | v0.3.1 reference chain | v0.4.0 fused select (both kernels) |
| explicit `LSA_SELECT_IMPL=reference` | reference chain | unchanged (the oracle stays one env var away) |
| outputs | — | byte-identical on every tested configuration (see below) |

## Why it is safe to promote (evidence, all clean-tree receipts)

- Byte-exact equality with the oracle: contexts 512–32768 (bf16+fp32, prefill/decode/ragged),
  sampled shapes `columns > 4·top_p` (incl. 65536-context end-to-end), bucket_cap {8,16,32,64}
  — `tests/test_select_fusion.py`, `tests/test_select_wide_adversarial.py`, 258/258 suite,
  17/17 gates with the fused tree, every A/B row re-verified exact pre+post (363/363 total).
- Measured, inside campaign bands at the final numerical code `45a0ed1`
  (`evidence/v0.4.0/ab/fused-final-s424243.json`): 32k bf16 prefill select 2.977×,
  prefill select+read 1.225×, decode select+read 1.224×; zero regression cells campaign-wide.
- Peak memory at 32k prefill: −959.4 MiB vs reference (clean receipt `memory-probe.json`).
- Wide (sampled) shapes dispatch loudly to the length-respecting fallback == fixed reference.

## Rollback (three independent layers, pick the fastest)

1. **Runtime, no rebuild (seconds):** set `LSA_SELECT_IMPL=reference` in the serving
   environment — `flat_select` returns to the v0.3.1 chain verbatim. This is the documented
   incident response; it requires no code change and can be applied per-process.
2. **Code (one revert):** `git revert <promotion-commit>` — restores the `reference` default
   for every consumer that does not pin the env var.
3. **Full retreat:** revert PR #7's merge (last resort; unnecessary given 1–2).

## Required GPU regression before + after merging this PR

On the physical gfx1151 host (or the migrated runner via the GPU workflow):

```bash
git checkout chore/v0.4.0-fused-default-promotion
PYTHONPATH=src python -m pytest tests/ -q                  # 259/259 (incl. default-dispatch pin)
PYTHONPATH=src python -m lsa_rocm_lab.port.gates           # 17/17 — now via the FUSED default
LSA_SELECT_IMPL=reference PYTHONPATH=src python -m lsa_rocm_lab.port.gates   # 17/17 — rollback path live
python scripts/ab_select_fusion.py --seed 424243 --reps 20 --dtypes bf16 --contexts 32768 \
  --candidate fused --out /tmp/post-promotion-ab.json      # bands: prefill select ~2.97×, s+r ~1.22×
```

Post-merge, on the migrated runner: the merge push fires the GPU workflow (approve in
`gpu-regression`) — 17/17 gates + full suite + receipt validation end-to-end on the default
(fused) tree is the release-level confirmation.

## Owner checklist

1. v0.4.0 released and soak-decided (this PR waits).
2. GPU regression above green (host or runner).
3. Merge this PR (rebase to `main` after PR #7 lands).
4. Post-merge GPU workflow run green.
5. Rollback rehearsal (optional but recommended): once, set `LSA_SELECT_IMPL=reference` in a
   scratch process and confirm gates 17/17 — proves layer 1 works before you need it.
