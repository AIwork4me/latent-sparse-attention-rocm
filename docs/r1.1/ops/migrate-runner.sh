#!/usr/bin/env bash
# migrate-runner.sh — R1.1 runner migration, owner-executed with sudo (script is read-only until then).
# Usage: sudo bash docs/r1.1/ops/migrate-runner.sh
# Preconditions (verified nonprivileged 2026-11-08, see ../RUNNER_MIGRATION_VALIDATION.md):
#   user lsa-runner free · /opt >= 12 GiB · venv copy empirically validated · siblings are
#   amd's user-scoped units (untouched by this system-unit migration).
# Every step echoes what it does; aborts on first failure (set -e) leaving originals intact.
set -euo pipefail

OLD_RUNNER=/home/amd/actions-runner-lsa
OLD_VENV=/home/amd/Desktop/lsa-rocm-eval-starter/.venv
NEW_ROOT=/opt/actions-runner-lsa
NEW_VENV_ROOT=/opt/lsa-rocm
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "== 0. Pre-flight (repeat; abort on any failure) =="
id lsa-runner 2>/dev/null && { echo "ABORT: user lsa-runner already exists"; exit 1; } || echo "name free"
free_kb=$(df --output=avail /opt | tail -1)
[ "$free_kb" -ge $((12 * 1024 * 1024)) ] || { echo "ABORT: < 12 GiB free on /opt"; exit 1; }
[ -d "$OLD_RUNNER/.credentials" ] || [ -f "$OLD_RUNNER/.credentials" ] || { echo "ABORT: old runner credentials missing"; exit 1; }
[ -x "$OLD_VENV/bin/python" ] || { echo "ABORT: old venv missing"; exit 1; }
# siblings must be alive and are NOT touched by this script:
pgrep -f "/home/amd/actions-runner/bin/Runner.Listener" >/dev/null && echo "sibling actions-runner alive (untouched)"
pgrep -f "/home/amd/actions-runner-ovamd/bin/Runner.Listener" >/dev/null && echo "sibling actions-runner-ovamd alive (untouched)"

echo "== 1. Dedicated system user (nologin, video+render only — exactly the GPU groups) =="
useradd -r -m -d /home/lsa-runner -s /usr/sbin/nologin -G video,render lsa-runner
id lsa-runner

echo "== 2. Copy (originals untouched = rollback) =="
mkdir -p "$NEW_ROOT" "$NEW_VENV_ROOT"
rsync -a "$OLD_RUNNER/" "$NEW_ROOT/"
rsync -a "$OLD_VENV/" "$NEW_VENV_ROOT/.venv/"

echo "== 3. Strip stale amd-owned runtime state from the copy =="
rm -rf "$NEW_ROOT/_work" "$NEW_ROOT/_diag" "$NEW_ROOT/runner-console.log" "$NEW_ROOT/svc.sh"* 2>/dev/null || true

echo "== 4. Repair the relocated venv (recipe empirically validated 2026-11-08) =="
# 4a. console-script shebangs (45 files carry the old absolute prefix)
find "$NEW_VENV_ROOT/.venv/bin" -maxdepth 1 -type f -exec \
  sed -i "s|/home/amd/Desktop/lsa-rocm-eval-starter/.venv|$NEW_VENV_ROOT/.venv|g" {} +
# 4b. drop the editable install pointing into amd's checkout (workflow uses PYTHONPATH=src)
rm -f "$NEW_VENV_ROOT/.venv/lib/python3.12/site-packages/__editable__.lsa_rocm_lab"*.pth
# 4c. pyvenv.cfg needs no change (home=/usr/bin, executable=/usr/bin/python3.12 — absolute, host-stable)

echo "== 5. Ownership + tightened credential modes =="
chown -R lsa-runner:lsa-runner "$NEW_ROOT" "$NEW_VENV_ROOT"
chmod 600 "$NEW_ROOT/.runner" "$NEW_ROOT/.credentials" "$NEW_ROOT/.credentials_rsaparams" 2>/dev/null || true
mkdir -p "$NEW_ROOT/home" && chown lsa-runner:lsa-runner "$NEW_ROOT/home"   # sandbox HOME (unit sets it)

echo "== 6. System unit (corrected R1.1 draft in this repo) =="
cp "$HERE/github-runner-lsa.service" /etc/systemd/system/github-runner-lsa.service
systemd-analyze verify /etc/systemd/system/github-runner-lsa.service
systemctl daemon-reload
systemctl enable --now github-runner-lsa.service
sleep 5
systemctl status github-runner-lsa.service --no-pager | head -8

echo "== 7. Isolation + GPU validation (read-only, as lsa-runner) =="
sudo -u lsa-runner test ! -r /home/amd/.config/gh/hosts.yml && echo "credential isolation OK (hosts.yml unreadable)"
sudo -u lsa-runner test ! -r /home/amd/actions-runner && echo "isolation OK (old install unreadable)"
sudo -u lsa-runner env TRITON_CACHE_DIR=/opt/actions-runner-lsa/home/.triton \
  "$NEW_VENV_ROOT/.venv/bin/python" -c "import torch, triton; assert torch.cuda.is_available() and torch.version.hip; print('GPU OK:', torch.cuda.get_device_name(0), '| triton', triton.__version__)"

cat <<'EOF'
MIGRATION HOST STEPS DONE.
Remaining (in order):
  1. Merge the one-line workflow PR: rocm-regression.yml PY="/opt/lsa-rocm/.venv/bin/python"
     (normal reviewed PR; guard-covered).
  2. The intentionally-kept queued run 42381769814 (main @ 1ab0e0b) should now execute on the
     new runner — approve it in the gpu-regression environment; verify 17/17 gates + full suite
     (this is the end-to-end owner-approved GitHub GPU workflow validation).
  3. After validation: retire the old install so a duplicate listener is impossible:
        mv /home/amd/actions-runner-lsa /home/amd/actions-runner-lsa.retired-pre-migration
  4. Rotate the amd gh credential (gh auth logout / remove ~/.config/gh/hosts.yml) after
     confirming nothing legit still depends on it.
  5. Scope the runner group to this repository only (runner settings UI; R1 owner-packet item 6).
EOF
