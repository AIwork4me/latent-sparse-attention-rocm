# v0.2.0 benchmark results — **SUPERSEDED**

> **Status: SUPERSEDED by v0.2.1 (2026-11-06). Do not cite these numbers.**
>
> The v0.2.0 Dense-Decode baseline was defective: `pipeline.dense_read` passed
> `is_causal=True` for a single-token query against the full cache, so PyTorch
> built a top-left-aligned mask and the decode query attended **only key 0**
> (max abs error 2.44 vs the true last row at 4k bf16). The mis-dispatched causal
> path was also ~2.8× slower than the correct full-attention path at 32k
> (3.40 ms vs 1.22 ms p50), so every decode-phase speedup below divided by a
> baseline that was both semantically wrong and artificially slow — the speedups
> were **overstated**. Corrected, receipt-generated tables live in
> `docs/BENCHMARKS.md`; the repair is documented in `docs/REVIEW_v0.2.1.md`
> (finding F1) and CHANGELOG 0.2.1.
>
> The numbers are preserved verbatim below for the audit trail; the receipts they
> came from remain in `results/20261105T145658Z/` and `results/20261105T145917Z/`
> and `evidence/kernel-receipt-{quick,overnight}.json` (v0.2.0 copies).

Original headline (bf16, 8q/2kv×64d, top_k 4096 / top_p 1024 / window 512 / sink 4):

| Claim (v0.2.0) | Value | Why invalid |
|---|---|---|
| D-LSA Flat decode speedup @ 32k | 5.04× | baseline attended key 0 only; correct baseline is ~2.8× faster |
| Block-sparse decode speedup @ 32k | 7.86× | same defective baseline |
| D-LSA Tree decode speedup @ 32k | 2.18× | same defective baseline |
| Decode crossover vs dense | ~4k ctx | derived from the defective baseline curve |

Decode step p50 ms as published then:

| ctx | dense SDPA (defective) | block-sparse | D-LSA Flat | D-LSA Tree |
|---:|---:|---:|---:|---:|
| 512 | 0.173 | 0.367 | 0.295 | — |
| 1024 | 0.200 | 0.375 | 0.277 | — |
| 2048 | 0.239 | 0.398 | 0.282 | — |
| 4096 | 0.364 | 0.468 | 0.300 (1.21×) | 0.972 |
| 8192 | 0.943 | 0.500 (1.89×) | 0.335 (2.81×) | 1.124 |
| 16384 | 1.714 | 0.476 (3.60×) | 0.456 (3.76×) | 1.264 (1.36×) |
| 32768 | 3.408 | 0.433 (7.86×) | 0.676 (5.04×) | 1.561 (2.18×) |

Additional v0.2.0 reporting defects found by the v0.2.1 audit (all corrected in
v0.2.1; see `docs/REVIEW_v0.2.1.md`):

- §3 claimed "~40 min wall each" for the two campaign profiles; the run trees'
  `stages.jsonl` show ~2.5 min of stage time per profile.
- §6 claimed "fp16 matrix also recorded in receipts"; both v0.2.0 receipts are
  bf16-only.
- The "quick" profile receipt is a full 32k/reps-20 matrix identical in scope to
  overnight (both trees were direct `run.py --quick/--overnight` invocations, not
  the scripted quick pilot).
