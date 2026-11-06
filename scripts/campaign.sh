#!/usr/bin/env bash
# The benchmark campaign: pilot first, then the bounded overnight profile.
# Every stage writes immutable evidence under results/<timestamp>/.
#
# Fail-closed (F11): each phase records its exit code, every phase still runs
# (failure isolation), and the script exits nonzero if ANY phase failed - the
# v0.2.1 script echoed the codes and always exited 0.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
export LSA_ROCM_KERNEL_CMD="${LSA_ROCM_KERNEL_CMD:-$PY -m lsa_rocm_lab.kernel_adapter}"
export LSA_UPSTREAM="$ROOT/third_party/latent-sparse-attention"
mkdir -p results

FAILED=()

banner() { printf '\n=== %s ===\n' "$1"; }

record() { # record <label> <rc>
  local label="$1" rc="$2"
  echo "$label rc=$rc"
  if [ "$rc" -ne 0 ]; then
    FAILED+=("$label($rc)")
  fi
}

banner "0. environment manifest (read-only)"
"$PY" -m lsa_rocm_lab.preflight > results/preflight-manifest.json 2>results/preflight-manifest.err
record "preflight" $?

banner "1. upstream Triton kernel tests on ROCm"
bash scripts/run_upstream_kernel_tests.sh > results/upstream-kernel-tests.log 2>&1
record "upstream-tests" $?

banner "2. quick pilot (starter stages + adapter at small scale)"
# reps 10 = the receipt validator's minimum sample evidence; a pilot below it
# cannot produce a valid receipt (v0.2.0's pilot silently ran reps 20 instead)
LSA_MAX_CONTEXT=2048 LSA_REPS=10 "$PY" -m lsa_rocm_lab.run --quick
record "pilot" $?

banner "3. overnight profile (bounded stages, failure-isolated)"
bash scripts/overnight.sh
record "overnight" $?

if [ "${#FAILED[@]}" -gt 0 ]; then
  printf '\ncampaign FAILED: %s\n' "${FAILED[*]}" | tee results/campaign-status.txt
  exit 1
fi
echo "campaign OK: all phases exited 0" | tee results/campaign-status.txt
exit 0
