#!/usr/bin/env bash
# The benchmark campaign: pilot first, then the bounded overnight profile.
# Every stage writes immutable evidence under results/<timestamp>/.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export LSA_ROCM_KERNEL_CMD="${LSA_ROCM_KERNEL_CMD:-$ROOT/.venv/bin/python -m lsa_rocm_lab.kernel_adapter}"
export LSA_UPSTREAM="$ROOT/third_party/latent-sparse-attention"
mkdir -p results

banner() { printf '\n=== %s ===\n' "$1"; }

banner "0. environment manifest (read-only)"
"$ROOT/.venv/bin/python" -m lsa_rocm_lab.preflight > results/preflight-manifest.json 2>results/preflight-manifest.err
echo "preflight rc=$? (see results/preflight-manifest.json)"

banner "1. upstream Triton kernel unit tests on ROCm"
bash scripts/run_upstream_kernel_tests.sh > results/upstream-kernel-tests.log 2>&1
echo "upstream tests rc=$? (see results/upstream-kernel-tests.log)"

banner "2. quick pilot (starter stages + adapter at small scale)"
LSA_MAX_CONTEXT=2048 LSA_REPS=5 "$ROOT/.venv/bin/python" -m lsa_rocm_lab.run --quick
echo "pilot rc=$?"

banner "3. overnight profile (bounded stages, failure-isolated)"
bash scripts/overnight.sh
echo "overnight rc=$?"
