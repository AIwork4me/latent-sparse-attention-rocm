#!/usr/bin/env bash
# Run upstream's own Triton-kernel unit tests on this ROCm GPU, as the port's
# correctness evidence. The cutedsl files need NVIDIA's cutlass and are excluded
# (their kernels are the NVIDIA-only path this port replaces); everything here
# exercises upstream Triton kernels against upstream's torch oracles.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UPSTREAM="${LSA_UPSTREAM:-$ROOT/third_party/latent-sparse-attention}"
cd "$UPSTREAM"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONPATH="$UPSTREAM/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -m pytest \
  tests/unit/core/test_ward_kernel.py \
  tests/unit/core/test_ward.py \
  tests/unit/core/test_keys.py \
  tests/unit/core/test_top_k.py \
  tests/unit/core/test_beam.py \
  tests/unit/core/test_shape_kernels.py \
  tests/unit/core/test_paged.py \
  tests/unit/core/test_reference.py \
  tests/unit/core/test_attention.py \
  tests/unit/core/test_index_space.py \
  tests/unit/core/test_hierarchy.py \
  "$@"
# Excluded, with reasons recorded in the report:
#   test_scan_select.py, test_bucket_scan.py, test_dispatch.py, test_expansion.py -
#     import the CuTe DSL front doors (NVIDIA cutlass) at module level. The expansion
#     kernels themselves are covered by this port's gates B1/B2/C, which drive the
#     Triton expand_in_order against the torch oracle on this GPU; the fused scan
#     path starts at 32768 visible buckets, above this campaign's scale.
