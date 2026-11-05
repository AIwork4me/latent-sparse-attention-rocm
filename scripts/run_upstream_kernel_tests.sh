#!/usr/bin/env bash
# Run upstream's own Triton-kernel unit tests on this ROCm GPU and TRIAGE the
# result against evidence/upstream-test-classification.json.
#
# Known, reproducibly-classified failures (NVIDIA-only tf32x3 dot precision that
# HIP rejects at compile; marginal RDNA tolerances on the unused shape-scoring
# path) stay fully visible in the log but do not fail this script. Any failure
# NOT in the classification manifest fails it - a genuine regression on the
# kernels this port composes can never hide behind the known list. The cutedsl
# files need NVIDIA's cutlass and are excluded (their kernels are the
# NVIDIA-only path this port replaces); the expansion kernels they would cover
# are driven by this port's gates B1/B2/C against the torch oracle.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UPSTREAM="${LSA_UPSTREAM:-$ROOT/third_party/latent-sparse-attention}"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
export LSA_UPSTREAM="$UPSTREAM"
cd "$ROOT"
exec "$PY" -m lsa_rocm_lab.upstream_triage --upstream "$UPSTREAM" "$@"
