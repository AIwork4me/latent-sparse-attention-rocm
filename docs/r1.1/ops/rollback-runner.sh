#!/usr/bin/env bash
# rollback-runner.sh — exact reversal of migrate-runner.sh (owner-executed with sudo).
# Safe at any point after migration; the ORIGINAL installs were never modified.
# Usage: sudo bash docs/r1.1/ops/rollback-runner.sh
set -euo pipefail

echo "== 1. Stop + remove the system service =="
systemctl disable --now github-runner-lsa.service 2>/dev/null || true
rm -f /etc/systemd/system/github-runner-lsa.service
systemctl daemon-reload

echo "== 2. Verify the old install is intact (rollback anchor) =="
test -f /home/amd/actions-runner-lsa/.credentials || { echo "ABORT: old install damaged — do NOT clean /opt yet"; exit 1; }
test -x /home/amd/Desktop/lsa-rocm-eval-starter/.venv/bin/python || { echo "ABORT: original venv missing"; exit 1; }

cat <<'EOF'
Host-level rollback complete (new runner stopped+removed; /opt copies remain as inert data).
To return to legacy operation (amd-user runner, pre-migration posture):
  cd /home/amd/actions-runner-lsa && nohup ./run.sh &
  (first restore the directory if it was retired: mv ~/actions-runner-lsa.retired-pre-migration ~/actions-runner-lsa)
  and git revert the one-line workflow PR (PY=... path back to $HOME/Desktop/...).
Optional cleanup only AFTER legacy operation is confirmed stable:
  sudo userdel -r lsa-runner; sudo rm -rf /opt/actions-runner-lsa /opt/lsa-rocm
NOTE: while the legacy runner runs, the migrated copy in /opt must never be started
      concurrently (same runner identity lsa-rocm-gfx1151 — duplicate listener conflict).
EOF
