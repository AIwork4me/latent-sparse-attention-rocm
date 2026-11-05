# Changelog

All notable changes to this repository. Dates are UTC.

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
