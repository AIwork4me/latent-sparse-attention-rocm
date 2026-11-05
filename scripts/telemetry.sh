#!/usr/bin/env bash
set -uo pipefail
# Samples once per minute. Optional tools are strictly best-effort.
while true; do
  printf '\n=== %s ===\n' "$(date -Is)"
  free -m | head -n 3
  if command -v amd-smi >/dev/null; then
    timeout 8 amd-smi monitor -u -p -t -m 2>&1 | head -n 45 || true
  elif command -v rocm-smi >/dev/null; then
    timeout 8 rocm-smi --showtemp --showpower --showuse --showmemuse 2>&1 || true
  else
    echo 'GPU telemetry CLI missing; temperature/power UNMEASURED'
  fi
  sleep 60
done
