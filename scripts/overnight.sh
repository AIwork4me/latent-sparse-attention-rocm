#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p results
PY="${PYTHON:-$ROOT/.venv/bin/python}"
bash scripts/telemetry.sh > results/nightly-telemetry.log 2>&1 &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
# Safeguard: hard 10h job limit. Never auto-poweroff the host.
set +e
timeout --signal=TERM --kill-after=60s 10h "$PY" -m lsa_rocm_lab.run --overnight 2>&1 | tee results/nightly-launch.log
rc=${PIPESTATUS[0]}
set -e
printf 'run exit code: %s\n' "$rc"
exit "$rc"
