# Contributing to latent-sparse-attention-rocm

Thank you for improving the ROCm LSA validation harness.

## The one rule that cannot be bent

**No performance claim without a real GPU measurement.** A pull request that
states a speedup must contain (or extend) a benchmark receipt produced by
`python -m lsa_rocm_lab.kernel_adapter` on the target GPU, with its correctness
gates passing. See `docs/PORT_CONTRACT.md` — it is the contract this repository
is built on, and it overrides convenience.

## Before you send a change

1. **Correctness first.** Every algorithm-touching change must keep
   `python -m lsa_rocm_lab.port.gates` at 13/13 PASS, and the upstream Triton
   kernel tests green:
   ```bash
   bash scripts/run_upstream_kernel_tests.sh
   ```
2. **Unit tests.** `.venv/bin/python -m pytest tests/ -v` (GPU tests skip loudly
   without a ROCm device; they never pass silently).
3. **Upstream is read-only.** Never edit files under `third_party/`. The port
   composes upstream's Triton kernels; changes to upstream behavior belong
   upstream, behind a pinned-SHA bump reviewed through
   `scripts/upstream_watch.py` + `tracking/UPSTREAM_BASELINE.json`.
4. **No NVIDIA-only dependencies in the ROCm path.** `cutlass`, CuTe DSL,
   `input_precision="tf32*"` and similar are NVIDIA-only; the port pins
   `ieee` precision (see `lsa_rocm_lab.port.HIP_PRECISION`). If you need a new
   kernel, write it in Triton and gate it against the upstream oracle.
5. **Evidence format.** Benchmark changes must keep the receipt JSON valid
   against `schemas/metrics.schema.json`.

## Benchmark methodology (non-negotiable details)

- Matched comparisons only: same context, dtype, heads, top-k, window, sink,
  seed, and GPU, warm ≥5, timed ≥20, median/p95/MAD reported.
- The dense baseline is torch SDPA on the same tensors; LSA rows time
  select+read on a built index with the index build timed separately.
- OOM/TIMEOUT rows are recorded, never silently dropped; a failed shape keeps
  its log.

## Reporting bugs

Include: GPU (gfx target), ROCm + torch versions, upstream SHA from
`third_party/upstream.sha`, the receipt or log under `results/`, and the
smallest reproducing script.
