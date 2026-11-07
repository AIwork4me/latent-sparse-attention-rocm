# Release Notes — v0.3.1 (DRAFT — publication requires owner approval; tag target = PR #5 merge commit, to be recorded)

## What this release is

v0.3.1 = **v0.3.0 kernel optimization + v0.3.1 top-k selection** on top of the v0.2.1 scientific-validity closure. One release covers the whole PR #5 range (no separate v0.3.0 tag). All performance claims come from auditor-APPROVED receipts pinned at code SHA `60c004b5630771082ac9ac3a2877262cae1f4ed4`; every speedup is independently recomputable (`scripts/r1_verify_receipts.py`).

## Correctness guarantees (17-gate manifest, all PASS on gfx1151; both tree implementations)

v0.2.1's 16 gates plus **gate H** (tiled production read == v1 kernel + zero-length contract). Fresh R1 verification at PR #5 HEAD `261805c`: **17/17 default AND 17/17 `LSA_TREE_IMPL=batched`**; gfx1151 suite 186 passed; GitHub-hosted CPU CI 135 passed / 46 loud GPU skips; static security guard PASS. Fail-closed manifest (`lsa_rocm_lab.gate_manifest`): a receipt is claimable only with exactly these 17 ids, all PASS — v0.2.1's 16-gate receipts intentionally no longer validate (documented, by design).

## Improvements (measured, multi-seed where variance matters)

- **Tiled sparse read (v0.3.0/T3, production path)**: row-tiled Triton kernel; prefill-stage 1.14–1.24× over v1 across 8k–32k on identical inputs; 32k prefill flat 0.19× → **0.210×** vs dense. Zero-length NaN hazard fixed and gated; B>1 row-mask bug found by the ragged-tile test before release.
- **Top-k selection (v0.3.1)**: `best_first_keys` replaces the select's full-width sort + strided slice (~19% of prefill-select profile) — packed keys are column-unique so `topk(sorted=True)` ≡ `sort()[:, :k]`; equivalence pinned by CPU+GPU tests (tied scores, EMPTY keys, k==width, ragged rows, production shapes). **Receipt-backed: 32k bf16 decode select stage 0.528→0.466 ms (1.13×) across the final receipts** (independent runs, same methodology); an identical-input in-process A/B during development measured 1.24× (0.266→0.214 ms) — development measurement, artifact not retained, superseded here by the receipt-backed figure; prefill neutral.
- **Multi-seed honesty (v0.3.0/T1)**: flat-decode "2.1×" shown to be seed luck; headline is now the band.

## Headline @ 32k ctx bf16 (decode select+read vs dense; cross-seed bands, 9 receipts/study)

- **Block-sparse 4.29× (4.24–4.38) — robust**
- **D-LSA Flat 1.94× median (1.85–2.06) — seed-sensitive; cite the band, not one run**
- **D-LSA Tree 0.93× (0.92–0.99) — still slower than dense**
- Prefill flat **0.210×** — still slower than dense, stated as-is

## Negative results (kept honest)

- **Page-batched tree beam (v0.3.0/T4): 1.16–1.29× slower prefill / 1.78–1.86× slower decode** than the per-page walk (audited A/B, set-equality-gated) — the launch tax is not the cost; production keeps the per-page walk; batched stays behind `LSA_TREE_IMPL=batched` (experimental).
- No end-to-end SGLang serving on gfx1151 — **BLOCKED_ON_REASSESS** (MI-class path; gfx1151 image lane is a separate v0.6.0 feasibility study). No model-level inference claims.

## Known limitations

- Prefill remains ~5× slower than dense; next measured levers: fused pack→expand select pass, Triton per-level beam port, decode select fusion (`docs/OPTIMIZATION_v0.3.md` §h).
- Flat-decode speedup varies by seed (band above); 16k-context flat results fluctuate most.
- Stage-level (select / read / select+read) microbenchmarks only — hardware/software pinned per table in `docs/BENCHMARKS.md`.

## Reproducibility

```bash
bash scripts/checkout_upstream.sh                                # pin 5ebea732
PYTHONPATH=src python -m lsa_rocm_lab.port.gates                 # 17/17 (default)
LSA_TREE_IMPL=batched PYTHONPATH=src python -m lsa_rocm_lab.port.gates   # 17/17
python -m pytest tests/ -v                                       # 186 on gfx1151
python3 -I scripts/r1_verify_receipts.py . \
  evidence/receipts-v0.3.1-final/kernel-receipt-v031-bf16.json \
  evidence/receipts-v0.3.1-final/kernel-receipt-v031-fp16.json \
  evidence/seed-study-20261106T171352Z/receipt-*.json \
  evidence/receipts-v0.3.1-final/audit-v031.json --expect-gates 17
cd evidence && sha256sum -c SHA256SUMS                           # 71/71 OK
```

Hardware/software: AMD Ryzen AI MAX+ PRO 395 / Radeon 8060S (gfx1151), 32 GiB UMA · ROCm 10.1.0 pip venv · torch 2.14.0+rocm10.1.0 · Triton 3.8.0+git669b31ac · Python 3.12.3 · upstream `Prefrontal-AI/latent-sparse-attention@5ebea732`.

## Source & evidence provenance (filled at tag time)

- PR #5 head at R1 audit: `261805c98c41230c2bfc32533a78873857058fdf`; the branch head is now above it (R1 audit + adversarial-review fixes) — **the tag must point at the branch head at merge time, recorded here then** · code `0908842` = `60c004b` + docstring-only; test fix `5a13bf0` · merge commit: `<RECORD AT MERGE>` · tree: `<RECORD>`
- Receipts: `evidence/receipts-v0.3.1-final/` (bf16 `20261106T171009Z-adapter`, fp16 `20261106T171202Z-adapter`) + `evidence/seed-study-20261106T171352Z/` (9) — audit `audit-v031.json` 11/11 APPROVED, 153/153 recomputed. Counting rule for cumulative v0.3.x: **42 unique audited receipts = 6 matrix (v0.3.0@`4988672` ×2, v0.3.0-final@`1e73c3a` ×2, v0.3.1@`60c004b` ×2) + 36 seed-study (4 studies × 9)**, plus 2 correctness-gated A/B artifacts (not receipts).
- Pre-tag hygiene gate (owner checklist A8): `pyproject.toml` version bump to `0.3.1` (currently `0.2.0`) and `CHANGELOG.md` `[0.3.0]`/`[0.3.1]` entries (added by R1; verify present).
- Post-merge fresh gate receipt at the merged SHA: `<RECORD — hard release gate §7>`
