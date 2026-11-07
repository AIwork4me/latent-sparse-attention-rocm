# EVIDENCE INDEX — pointers to receipts, hashes, sources and job URLs

Latest first. Every claim in README/FINAL_REPORT/BENCHMARKS must trace to an entry here.

| ID | What | Path / URL | Anchor |
|---|---|---|---|
| EV-2026-11-06-C | Post-fix verification receipt at head `2139758` (2048/reps10, 12 comparisons, gates 16/16) + audit | `evidence/receipt-2139758-postfix-verification.json`, `evidence/audit-2139758-postfix.json` | receipt `run_id 20261106T…Z-adapter`, repo HEAD `2139758` |
| EV-2026-11-06-B | Trusted GPU dispatch run on PR branch (success; artifact re-audited locally: APPROVED) | https://github.com/AIwork4me/latent-sparse-attention-rocm/actions/runs/42139297335 | artifact `ci-kernel-receipt-gh42139297335` |
| EV-2026-11-06-A | Fat-finger guard negative run (mismatched confirm_ref fails closed, 20s) | https://github.com/AIwork4me/latent-sparse-attention-rocm/actions/runs/42139650668 | guard step error line |
| EV-2026-11-06-9 | CPU-PR workflow green on GitHub-hosted runner (59s, zero secrets) | https://github.com/AIwork4me/latent-sparse-attention-rocm/actions/runs/42139623108 | — |
| EV-2026-11-05-8 | Independent audit: all 6 v0.2.1 claim receipts APPROVED, 111 comparisons recomputed | `evidence/audit-v0.2.1.json` (hash in `evidence/SHA256SUMS`) | verdict block |
| EV-2026-11-05-7 | 3× independent decode repetitions (8k/16k/32k bf16) + aggregated medians | `evidence/repetitions-20261105T225807Z/` | `REPETITION_REPORT.md` |
| EV-2026-11-05-6 | Overnight fp16 matrix receipt | `evidence/kernel-receipt-overnight-fp16.json` | run `20261105T225806Z-adapter` |
| EV-2026-11-05-5 | Overnight bf16 matrix receipt (36 comparisons) | `evidence/kernel-receipt-overnight-bf16.json` | run `20261105T230036Z` |
| EV-2026-11-05-4 | Pilot receipt (2048, reps 10) | `evidence/kernel-receipt-pilot-2048.json` | run `20261105T230021Z` |
| EV-2026-11-05-3 | Upstream test triage manifest (203 pass / 7 classified / 0 unclassified) + logs | `evidence/upstream-test-classification.json`, `evidence/upstream-kernel-tests-triaged.log` | — |
| EV-2026-11-05-2 | Gate verdicts (16) jsonl | `evidence/port-gates.jsonl` | — |
| EV-2026-11-05-1 | Superseded v0.2.0 evidence (preserved bytes) | `evidence/v0.2.0-archive/`, `docs/RESULTS_v0.2.0.md` | SUPERSEDED banner |

Integrity: `cd evidence && sha256sum -c SHA256SUMS` must report no failures.
