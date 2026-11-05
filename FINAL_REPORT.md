# FINAL REPORT v0.2.1 — Latent Sparse Attention on ROCm gfx1151

**Project:** `AIwork4me/latent-sparse-attention-rocm` (working tree: `lsa-rocm-eval-starter`)
**Date:** 2026-11-06 (UTC) · **Release:** v0.2.1 (scientific validity closure)
**Upstream:** Prefrontal-AI/latent-sparse-attention @ `5ebea732d639576d2a7579b0de40e6252f9001f5` (pinned, read-only; HEAD of upstream main is the same SHA as of this run)
**Audit:** the independent review of this release is [`docs/REVIEW_v0.2.1.md`](docs/REVIEW_v0.2.1.md); the v0.2.0 numbers it supersedes are preserved in [`docs/RESULTS_v0.2.0.md`](docs/RESULTS_v0.2.0.md).

---

## 1. Executive summary

v0.2.0 published decode speedups (5.04× flat / 7.86× block-sparse at 32k) that
were **invalid**: the Dense-Decode baseline timed a mis-masked kernel. v0.2.1
repairs the baseline semantics, adds a release-blocking parity gate, closes the
evidence chain (skipped or short gate suites can no longer produce claimable
receipts; kernel failures now fail the campaign), re-runs every benchmark on the
corrected baseline — bf16 + fp16 matrices plus three independent repetitions —
and classifies all 7 known upstream test failures with reproducible reasons.

**Corrected headline (decode select+read, 32k ctx, bf16, median of 3 independent runs):**

| Result | v0.2.0 (superseded) | **v0.2.1 (verified)** |
|---|---:|---:|
| Block-sparse decode speedup @ 32k | ~~7.86×~~ | **3.9×** (3.57–4.17 across runs) |
| D-LSA Flat decode speedup @ 32k | ~~5.04×~~ | **2.1×** (1.52–2.41 across runs) |
| D-LSA Tree decode speedup @ 32k | ~~2.18×~~ | **0.9× — not faster than dense** |
| D-LSA Flat decode @ 16k | ~~3.76×~~ | **0.9× (slower than dense)** |
| Block-sparse decode @ 16k | ~~3.60×~~ | **1.8×** |
| Prefill select+read @ 32k | 0.19× | **0.19× (unchanged: prefill deficit is real)** |
| Correctness gates | 13/13 | **16/16** (incl. new gate G, dense decode parity) |

Every number above is generated from validated receipts by
`scripts/render_tables.py` → [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md); nothing
in the tables is hand-typed. Upstream Triton kernel tests: **203 pass / 7
classified** (4 × NVIDIA-only tf32x3 precision the HIP compiler rejects; 3 ×
marginal RDNA tolerances on the unused shape-scoring path) — any *unclassified*
failure fails CI.

## 2. The v0.2.0 decode-baseline defect (why everything was re-run)

`pipeline.dense_read` passed `is_causal=True` unconditionally. For the decode
benchmark's single query row against the whole cache, PyTorch aligns that mask
to the **top-left**, so the query attended **only key 0**:

- **Semantics:** max abs error 2.44 vs the square prefill's last row (4k, bf16) —
  the baseline's *output* was wrong, not just its timing.
- **Timing:** the mis-dispatched causal path took 3.40 ms p50 at 32k where the
  correct full-attention decode takes ~1.2 ms — the defective baseline was
  ~2.8× **slower** than the truth, so every v0.2.0 decode speedup was overstated
  roughly twofold.

The repair (`pipeline.dense_read`): query rows are the **tail `q_len` positions**
of the cache — square shapes keep the fused causal flag, a single row attends
the whole cache unmasked, a multi-token chunk gets an explicit
bottom-right-aligned causal mask (`q_len > kv_len` raises). Gate G pins decode
and chunk rows to the square prefill's tail per dtype (release-blocking:
`tests/test_dense_read.py` + `lsa_rocm_lab.port.gates`).

## 3. Environment (unchanged from v0.2.0; re-verified by preflight)

| Item | Value |
|---|---|
| CPU / SoC | AMD Ryzen AI MAX+ PRO 395 (32C), HP ZBook Ultra |
| GPU | Radeon 8060S, gfx1151, 32 GiB dedicated VRAM (UMA APU) |
| Kernel / OS | 6.17.0-1033-oem, Ubuntu 24.04.4 |
| Runtime | `.venv` (python 3.12.3) + AMD pip wheels `rocm[...]==10.1.0` |
| torch / Triton | `2.14.0+rocm10.1.0` (HIP 7.16.26385) / `3.8.0+git669b31ac.rocm10.1.0` |

## 4. Campaign (v0.2.1 re-run, scripted end-to-end)

`bash scripts/campaign.sh` → pilot (2048, reps 10) + overnight bf16; plus a
separate `LSA_DTYPE=fp16 … --overnight` pass and `bash scripts/repeat_bench.sh`
(3 independent adapter processes at 8k/16k/32k decode).

| Run | Tree | Status | Receipt |
|---|---|---|---|
| Pilot (quick, 2048) | `results/20261105T230021Z/` | all PASS, claimable | `evidence/kernel-receipt-pilot-2048.json` |
| Overnight bf16 | `results/20261105T230036Z/` | all PASS, claimable | `evidence/kernel-receipt-overnight-bf16.json` |
| Overnight fp16 | `results/20261105T225549Z/` | all PASS, claimable | `evidence/kernel-receipt-overnight-fp16.json` |
| 3× repetitions | `results/repetitions-20261105T225807Z/` | 16/16 gates each | `evidence/repetitions-20261105T225807Z/` |

Every receipt validates against the gate manifest, the pinned upstream SHA, the
gfx1151 device check, and the ≥10-sample evidence bar; each records run id, repo
HEAD SHA, device, torch build, seed, exact command, contexts and phases.

**Reconciliation of the v0.2.0 report (all corrected here; details in the review):**
the "~40 min wall each" claim (actual stage time ≈2.5 min/profile), the
"fp16 matrix also recorded in receipts" claim (receipts were bf16-only — fp16 now
actually run), and the mislabeled "quick" profile (a full 32k/reps-20 matrix; the
pilot now genuinely runs 2048/reps-10 and validates).

## 5. Correctness evidence

16/16 gates PASS in every run above (gate ids: A, A2, B1, B2, C, D1/D2 ×3 dtypes,
E, F, **G ×3 dtypes**). Verdicts: `evidence/port-gates.jsonl` (from the bf16
overnight receipt). The evidence chain is fail-closed end-to-end:

- `lsa_rocm_lab.gate_manifest` — the required gate-id set, enforced at receipt
  **write** (adapter) and receipt **read** (`run.validate_receipt`); a skipped
  suite writes `status: UNVERIFIED, correctness_pass: false` and cannot validate.
- `run.campaign_exit_code` — any mandatory stage FAIL/TIMEOUT (kernel included,
  whenever an adapter ran) fails the campaign; BLOCKED without an adapter remains
  a legitimate, non-claimable outcome.
- Upstream failures are triaged (`evidence/upstream-test-classification.json`):
  classified ones stay visible with reasons; any new failure exits nonzero.

## 6. Performance — corrected, receipt-generated

See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for every table (bf16 + fp16
matrices, stage breakdown, repetition spread). Shape of the honest picture:

- **Decode:** with the correct full-attention baseline (~1.27 ms at 32k bf16),
  block-sparse leads (3.9× at 32k), D-LSA Flat wins only at the largest contexts
  (2.1–2.4× at 32k, run-to-run spread 1.5–2.4×), and is *slower* than dense at
  ≤16k; the unoptimized tree beam is slower than dense everywhere at these
  scales. Crossover moved to ~8k (block-sparse) / ~16k (flat).
- **Prefill:** unchanged conclusion — the sparse read is correctness-first v1;
  0.19× at 32k. Tiling the read kernel (ROADMAP Track 1) remains the top lever.
- **Repetition variance** at 32k flat (1.52–2.41×) is dominated by candidate-side
  timing noise, not the dense baseline (stable ±5%); the repetition report
  publishes per-run numbers rather than averaging them away.

## 7. BLOCKED — SGLang end-to-end (unchanged)

SGLang's AMD support targets MI-class Instinct parts via per-release Docker
builds; upstream's plugin arms NVIDIA-only attention backends; target checkpoints
exceed the APU's dedicated window. No E2E claim is made. See the review's §E2E
for the unblocking path.

## 8. CI and upstream tracking

- Self-hosted gfx1151 runner registered for this repository (labels
  `rocm,gfx1151`); the regression workflow runs preflight → triaged upstream
  tests → 16 gates → repo tests → small validated receipt, per PR.
- `scripts/upstream_watch.py` daily watch: upstream HEAD (still `5ebea732`) vs
  the reviewed baseline; a new SHA opens a deduplicated review issue.
- `tracking/UPSTREAM_BASELINE.json` status refreshed; the baseline SHA moves only
  after ROCm review with evidence.

## 9. GO/NO-GO

**GO for v0.2.1** as a *validity* release: correctness and evidence gates are
green, every published number is receipt-backed, and the defective claims are
superseded and explained. **NO-GO for any performance claim beyond the tables
above** until the ROADMAP tracks land (tiled read kernel, kernelized tree beam,
128k+ contexts) — the corrected data shows the v0.2.0 performance story was
roughly half overrun. Full verdict: [`docs/REVIEW_v0.2.1.md`](docs/REVIEW_v0.2.1.md).

## 10. Evidence index

| Artifact | Path |
|---|---|
| Corrected overnight receipt (bf16, 36 comparisons, 16 gates) | `evidence/kernel-receipt-overnight-bf16.json` |
| Overnight receipt (fp16) | `evidence/kernel-receipt-overnight-fp16.json` |
| Pilot receipt (2048) | `evidence/kernel-receipt-pilot-2048.json` |
| 3× independent repetitions + report | `evidence/repetitions-20261105T225807Z/` |
| Gate verdicts (16) | `evidence/port-gates.jsonl` |
| Upstream test triage manifest + log | `evidence/upstream-test-classification.json`, `evidence/upstream-kernel-tests-triaged.log` |
| Stage rows (JSONL) | `evidence/stage-rows.jsonl` |
| Generated benchmark tables | `docs/BENCHMARKS.md` |
| Superseded v0.2.0 results | `docs/RESULTS_v0.2.0.md`, `evidence/v0.2.0-archive/` |
| Full run trees | `results/20261105T230021Z/`, `results/20261105T230036Z/`, `results/20261105T225549Z/` |
| Checksums | `evidence/SHA256SUMS` |

## 11. Honest-claims statement

- Every performance number in this report comes from a receipt that passed
  `run.validate_receipt` (gate manifest included) on this GPU; tables are
  generated, never hand-edited.
- Decode-phase speedups at 32k are real but **modest** (block-sparse 3.9×,
  flat 2.1×); below 16k the sparse arms are slower than the corrected dense
  decode baseline, and the tree arm is slower everywhere at these scales.
- Prefill at ≤32k is slower than dense (0.19×) and always has been.
- No E2E/SGLang claim exists; no NVIDIA paper number is re-labeled as a Radeon
  result; the v0.2.0 claims are superseded, not deleted.
