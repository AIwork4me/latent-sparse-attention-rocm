# Changelog

All notable changes to this repository. Dates are UTC.

## [0.4.0] — UNRELEASED (release candidate prepared 2026-11-08)

**Scope: audited opt-in fused selection on Radeon gfx1151.** The version is
prepared for release but **no v0.4.0 tag or GitHub Release exists yet**. The
default `LSA_SELECT_IMPL` remains `reference`; `fused` is opt-in. The
owner-gated default change is tracked separately in PR #8.

### Added
- Fused Triton `select_topk_scores`: score-pack + ordered top-k in one kernel,
  omitting the large `[rows,columns]` int64 intermediate key matrix.
- `expand_in_order_from_keys`: upstream-equivalent expansion with packed-key
  unpacking fused into the kernel. `LSA_SELECT_IMPL=fused_unpack` is retained
  as a measured alternative.
- 31 adversarial sampled-wide-listing tests, on top of the 35 select-fusion
  cases; the gfx1151 physical test report was **258 PASS** and gates **17/17**
  in reference, batched-tree, and fused-select configurations.
- Clean-tree final-SHA confirmation and five-sample peak-memory receipt at
  numerical commit `45a0ed1`. All 25 v0.4.0 evidence files now have a
  separate `evidence/v0.4.0/SHA256SUMS` seal, checked on hosted CPU CI.
- R1.1 documentation/operations hardening from PR #6: dedicated-runner
  migration runbook, safer workflow path, device-policy runtime probes.
  **Owner must still execute and validate the actual systemd migration.**

### Fixed
- Sampled-wide selection previously let `best_first_keys` consider
  unwritten listing slots. The repaired path masks unlisted elements to EMPTY,
  while `_mask_for` skips an otherwise costly no-op mask on whole listings
  (retaining prior v0.3.1 semantics and measured latency).
- Historical v0.2.1/v0.3.1 Release descriptions were reconciled without
  moving historical tags or replacing raw evidence.

### Stage-level performance (gfx1151, 32k BF16)
- Measured opt-in `fused` versus v0.3.1 `reference`, **not Dense SDPA**:
  prefill select **2.96×**, prefill select+read **1.226×**, and B=1 decode
  select+read **1.199×** across the 3-seed × 3-process campaign.
  These are observed values/ranges, not confidence intervals.
- One clean final-code confirmation: prefill select **2.977×**, prefill
  select+read **1.225×**, decode select+read **1.224×** (single seed/process,
  not a replacement for the multi-seed campaign).
- 32k BF16 prefill select peak allocated memory: 2304.2 → 1344.9 MiB,
  **−959.4 MiB (41.6%)** in a clean-tree five-sample receipt.
- No more than 2% measured performance regressions across 120 aggregation
  cells (smallest observed speedup 0.9986×). No model-serving/E2E benefit claimed.

### Known limits / rollout
- Hardware evidence: Radeon 8060S (gfx1151), ROCm 10.1.0. W7900/gfx1100
  is NOT_TESTED. SGLang and model-level quality/E2E are NOT_VALIDATED.
- Release candidate must pass hosted CPU checks and receive owner approval
  before creating tag/Release. Post-merge trusted GPU CI requires the
  owner-executed runner migration and environment approval.
- Promotion of the `fused` **default** has its own explicit GPU/owner gate
  in PR #8 and is NOT part of this release candidate.

### Evidence and reproduction
- `docs/v0.4.0/MEASURED_SELECT_FUSION.md`
- `docs/reviews/V0_4_INDEPENDENT_REVIEW.md`
- `evidence/v0.4.0/ab/` and `evidence/v0.4.0/SHA256SUMS`
- `python -m pytest tests/test_v040_release_integrity.py -q` (CPU-only)
- `cd evidence/v0.4.0 && sha256sum -c SHA256SUMS` (no GPU required)

## [0.3.1] — 2026-11-07

Top-k key selection replaces the flat select's full-width sort + strided slice.
Full narrative: `docs/OPTIMIZATION_v0.3.md` §(v0.3.1); receipts at `60c004b`
(`evidence/receipts-v0.3.1-final/`, 11/11 auditor-APPROVED, 153 comparisons).

### Changed
- `best_first_keys` now uses `torch.topk(k, sorted=True)` — packed keys are
  unique per column (score bits high, `0x7FFFFFFF - column` low), so topk is
  exactly `sort(...).values[:, :k]`; the replaced sort ordered up to 4× more
  keys than needed (~19% of the prefill-select profile) and left a strided
  slice. Equivalence pinned by device-parametrized tests (tied scores, EMPTY
  keys, k == width, ragged rows, production shapes) on CPU everywhere plus
  the ROCm device when visible (suite 181 → 186 on gfx1151).
- Receipt-backed effect: 32k bf16 decode select stage 0.528 → 0.466 ms
  (1.13×); prefill neutral. (An identical-input in-process A/B measured
  1.24×; artifact not retained — the receipt-backed figure is the citable
  one.)

## [0.3.0] — 2026-11-07

Kernel optimization cycle T1–T4 with honest bands and a kept negative result.
Full narrative: `docs/OPTIMIZATION_v0.3.md`, campaign report, EV index.

### Added
- T1 multi-seed studies (issue #2): honest variance bands replace single-seed
  headlines — 32k bf16 decode: block-sparse 4.29× (4.24–4.38), flat 1.94×
  (1.85–2.06; the v0.2.1 "2.1×" was a favorable seed), tree 0.93× (not
  faster than dense); prefill flat 0.210× vs dense.
- T3 tiled sparse read (production): row-tiled Triton kernel, prefill-stage
  1.14–1.24× over v1 on identical inputs; new gate H (manifest 16 → 17);
  zero-length NaN contract fixed and gated.

### Fixed
- B>1 ragged-tile row-mask bug caught by the new equivalence tests pre-release.

### Changed
- T4 page-batched tree beam measured 1.16–1.86× SLOWER (audited A/B) —
  kept experimental behind `LSA_TREE_IMPL=batched`; production stays on the
  per-page walk (launch tax is not the cost; the beam is op-bound).

## [0.2.1] — 2026-11-06

Scientific validity closure: the decode baseline defect, fail-closed evidence,
and re-run benchmarks. **All v0.2.0 decode-phase speedups are superseded** — see
`docs/RESULTS_v0.2.0.md` for why and `docs/BENCHMARKS.md` for the corrected,
receipt-generated tables.

### Fixed
- **Dense decode baseline attended only key 0.** `dense_read` passed
  `is_causal=True` unconditionally; for a 1-row query against the full cache the
  top-left-aligned mask hides all but the first key (max abs error 2.44 vs the
  prefill's last row). Query rows are now the tail `q_len` positions of the
  cache: square → fused causal; one row → full attention; a chunk → explicit
  bottom-right-aligned mask; `q_len > kv_len` raises. The defective path was
  also ~2.8× slower than the correct dispatch at 32k, so v0.2.0 decode speedups
  were overstated on both counts.
- **A skipped gate suite could produce a claimable receipt.** `--skip-gates`
  computed `correctness_pass = all([]) = True`. Now a shared, torch-free gate
  manifest (`lsa_rocm_lab.gate_manifest`) is enforced at both ends: the adapter
  writes `status: UNVERIFIED` / `correctness_pass: false` and the runner's
  `validate_receipt` independently rejects UNVERIFIED, empty, short, renamed,
  duplicated or failed gate lists.
- **Kernel-stage failures no longer exit the campaign green.** The stage-chain
  exit code ignored the kernel stage entirely; `campaign_exit_code` now fails on
  any mandatory stage FAIL/TIMEOUT, kernel included whenever an adapter ran.

### Added
- Gate G (dense decode/chunk == prefill tail, per dtype) — release-blocking,
  13 → 16 gates, every gate carries a stable `gate_id`.
- Release-blocking parity tests (`tests/test_dense_read.py`) incl. non-square
  multi-token chunks with absolute-position semantics.
- Upstream-failure triage: all 7 known failures reproduced and classified with
  exact evidence (`evidence/upstream-test-classification.json`); unclassified
  failures fail CI, classified ones stay fully visible.
- Reproducibility: receipts record run id, repo HEAD SHA, device, exact command;
  run trees gain run_id/profile/hostname; `scripts/repeat_bench.sh` runs three
  independent repetitions for strategic shapes with a validated aggregation
  report; `scripts/render_tables.py` generates every headline table from
  validated receipts only.
- `bench_dense_decode(query_len=…)` — speculative-decode chunk timing on the
  same semantics; `kernel_adapter --contexts/--phases` for repetition runs.

### Changed
- v0.2.0 benchmark claims moved to `docs/RESULTS_v0.2.0.md`, marked SUPERSEDED.
- `schemas/metrics.schema.json`: `UNVERIFIED` status, boolean
  `correctness_pass`, `gate_id` required, `gates_skipped` optional.
- Benchmarks re-run end-to-end on the corrected baseline (bf16 + fp16 matrices,
  three independent repetitions) — do not compare v0.2.0 numbers with these.

## [0.2.0] — 2026-11-05

The first release with a genuine ROCm-native LSA path.

### Added
- **ROCm port layer** (`src/lsa_rocm_lab/port/`): bucket build and node-bucket
  hierarchy via upstream's Triton Ward/pairing kernels at `ieee` precision;
  flat select via the production short/medium-context composition (bf16 GEMM
  with fp32 out → upstream Triton top-k listing → packed-key sort → upstream
  Triton rank-order expansion); tree select as a GPU-torch beam of upstream's
  `read_hierarchy` semantics; positional block-sparse baseline arm.
- **Triton sparse gather-attention read kernel** (`port/read_kernel.py`) — the
  piece upstream delegates to NVIDIA-only backends. Online-softmax, GQA-aware,
  causal-list contract, `num_stages=1` (see Fixed).
- **13-gate correctness suite** (`port/gates.py`): build partition, hierarchy
  pairing (same-device), flat/tree selection set-equality, read-vs-oracle
  numerics per dtype, full-select == dense SDPA, decode-step equivalence, GQA.
- **Matched benchmark harness + receipt adapter** (`port/bench.py`,
  `kernel_adapter.py`) wiring the starter's `LSA_ROCM_KERNEL_CMD` hook with
  stage breakdowns, p50/p95/MAD, OOM capture, and `schemas/metrics.schema.json`.
- Upstream Triton kernel test runner (`scripts/run_upstream_kernel_tests.sh`),
  campaign script, ROCm regression + upstream-watch GitHub workflows,
  CONTRIBUTING/SECURITY/NOTICE/PORT_NOTES.

### Fixed
- Decode-phase benchmark passed the full query set with a single position,
  driving upstream's expansion kernel to out-of-bounds position reads
  (nondeterministic illegal-access faults). Decode now slices its query row;
  `flat_select` asserts a position per query.
- Triton-HIP default software pipelining miscompiles the read kernel's masked
  gather loop at low occupancy (decode): pinned `num_stages=1`; stress-verified
  200 iterations on both phases.

### Validated (gfx1151, ROCm 10.1 wheels, torch 2.14.0+rocm10.1.0)
- Upstream reference == dense SDPA on GPU (starter gate); Triton HIP JIT smoke.
- Upstream Triton kernel unit tests: all `ieee`-precision Ward/pairing cases
  pass exactly; tf32 cases fail to compile on HIP by construction (documented).
- Port gates: 13/13 PASS.

## [0.1.0] — 2026-11-05 (starter harness)

Preflight, reference-correctness, dense SDPA baseline, Triton probe, and the
BLOCKED-safe overnight stage controller with receipts.
