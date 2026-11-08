#!/usr/bin/env bash
# migrate-runner.sh — R1.1 runner migration, owner-executed with sudo (script is read-only until then).
# Usage:
#   sudo bash docs/r1.1/ops/migrate-runner.sh            # fresh migration (aborts on ANY partial state)
#   sudo bash docs/r1.1/ops/migrate-runner.sh --resume   # resume a partially-completed migration
# Preconditions (verified nonprivileged 2026-11-08, see ../RUNNER_MIGRATION_VALIDATION.md):
#   /opt >= 12 GiB free · venv copy+repair empirically validated · siblings are amd's user-scoped
#   units (untouched by this system-unit migration).
# Partial-state policy: a fresh run aborts if the user OR /opt state already exists (never merges
# into a half-copy silently). Steps 2-7 are idempotent (rsync / sed / rm / chown / unit install),
# so --resume safely re-runs them from wherever a previous attempt aborted. To discard a partial
# state instead: rollback-runner.sh, then `userdel -r lsa-runner; rm -rf /opt/actions-runner-lsa /opt/lsa-rocm`.
# Validation (step 7) executes INSIDE the service security context — a transient systemd-run
# service transcribing the installed unit's sandbox directives — never via `sudo -u`: a plain
# sudo context sees all of /home and every device the unit's DevicePolicy=closed filters.
set -euo pipefail

OLD_RUNNER=/home/amd/actions-runner-lsa
OLD_VENV=/home/amd/Desktop/lsa-rocm-eval-starter/.venv
NEW_ROOT=/opt/actions-runner-lsa
NEW_VENV_ROOT=/opt/lsa-rocm
WORKFLOW_PY=/opt/lsa-rocm/.venv/bin/python   # must equal the PY= line in .github/workflows/rocm-regression.yml
HERE="$(cd "$(dirname "$0")" && pwd)"
RESUME=0
[ "${1:-}" = "--resume" ] && RESUME=1

echo "== 0. Pre-flight (repeat; abort on any failure) =="
free_kb=$(df --output=avail /opt | tail -1)
[ "$free_kb" -ge $((12 * 1024 * 1024)) ] || { echo "ABORT: < 12 GiB free on /opt"; exit 1; }
[ -d "$OLD_RUNNER/.credentials" ] || [ -f "$OLD_RUNNER/.credentials" ] || { echo "ABORT: old runner credentials missing"; exit 1; }
[ -x "$OLD_VENV/bin/python" ] || { echo "ABORT: old venv missing"; exit 1; }
# Cross-file consistency (the R1.1 handoff defect): the workflow must pin the migrated venv
# BEFORE the runner goes online, else the first push run dies at "Locate the ROCm venv".
# Run this script from a checkout that contains the workflow-fix commit (PR #6 branch or later main).
grep -Fq "PY=\"$WORKFLOW_PY\"" "$HERE/../../../.github/workflows/rocm-regression.yml" || {
  echo "ABORT: .github/workflows/rocm-regression.yml does not pin PY=\"$WORKFLOW_PY\" —"
  echo "       merge the workflow-fix commit first (its merge push is the validation vehicle; see handoff)."
  exit 1
}
if [ "$RESUME" = 1 ]; then
  id lsa-runner >/dev/null 2>&1 || { echo "ABORT: --resume but user lsa-runner absent (nothing to resume — run fresh)"; exit 1; }
  [ -d "$NEW_ROOT" ] || { echo "ABORT: --resume but $NEW_ROOT missing (run fresh)"; exit 1; }
  echo "resume mode: user + /opt state present, re-running idempotent steps 2-7"
else
  if id lsa-runner >/dev/null 2>&1; then
    echo "ABORT: user lsa-runner exists — a previous attempt may have partially completed."
    echo "       Inspect: ls -la $NEW_ROOT $NEW_VENV_ROOT ; systemctl status github-runner-lsa 2>/dev/null"
    echo "       Then either rerun with --resume, or clean: rollback-runner.sh && userdel -r lsa-runner && rm -rf $NEW_ROOT $NEW_VENV_ROOT"
    exit 1
  fi
  echo "name free"
  if [ -e "$NEW_ROOT" ]; then
    echo "ABORT: $NEW_ROOT already exists (stale or partial copy) — inspect, then --resume or clean it; never merge into it blind"
    exit 1
  fi
fi
# siblings must be alive and are NOT touched by this script:
pgrep -f "/home/amd/actions-runner/bin/Runner.Listener" >/dev/null && echo "sibling actions-runner alive (untouched)"
pgrep -f "/home/amd/actions-runner-ovamd/bin/Runner.Listener" >/dev/null && echo "sibling actions-runner-ovamd alive (untouched)"

if [ "$RESUME" = 0 ]; then
  echo "== 1. Dedicated system user (nologin, video+render only — exactly the GPU groups) =="
  useradd -r -m -d /home/lsa-runner -s /usr/sbin/nologin -G video,render lsa-runner
  id lsa-runner
fi

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

echo "== 7. Isolation + GPU validation IN THE SERVICE SECURITY CONTEXT =="
# Parity guard: the installed unit must still carry the R1.1 design sandbox, so the validation
# context below (transcribed from the live unit) can never silently weaken with an owner edit.
for line in "User=lsa-runner" "ProtectHome=true" "ProtectSystem=strict" "DevicePolicy=closed" \
            "PrivateTmp=true" "NoNewPrivileges=true"; do
  grep -q "^$line\$" /etc/systemd/system/github-runner-lsa.service || {
    echo "ABORT: unit no longer pins '$line' — reconcile this script with the unit before validating"; exit 1
  }
done
# Transcribe the live unit into a transient service: sandbox directives and Environment= come
# from systemctl show, so what the checks below see IS what the runner service sees.
props=(--wait --pipe --collect --quiet)
props+=(--property=User=lsa-runner --property=Group=lsa-runner)
props+=(--property=WorkingDirectory="$(systemctl show github-runner-lsa.service -p WorkingDirectory --value)")
for p in ProtectHome ProtectSystem PrivateTmp NoNewPrivileges DevicePolicy; do
  props+=(--property="$p=$(systemctl show github-runner-lsa.service -p "$p" --value)")
done
for p in $(systemctl show github-runner-lsa.service -p ReadWritePaths --value); do
  props+=(--property=ReadWritePaths="$p")
done
read -ra ENV_WORDS <<< "$(systemctl show github-runner-lsa.service -p Environment --value)"
for w in "${ENV_WORDS[@]}"; do props+=(--setenv="$w"); done
run_in_context() { systemd-run "${props[@]}" "$@"; }

echo "-- 7a. ProtectHome: ALL of /home (amd's gh credential, old install, siblings) invisible --"
run_in_context bash -c 'test ! -e /home/amd && test ! -e /home/lsa-runner' \
  || { echo "ABORT: /home visible inside the service context (ProtectHome not effective)"; exit 1; }
echo "credential isolation OK: /home entirely invisible in the service security context"

echo "-- 7b. DevicePolicy=closed allowlist: /dev/dri inside the sandbox = only the unit's nodes --"
allowed_dri="$(systemctl show github-runner-lsa.service -p DeviceAllow --value \
  | tr ' ' '\n' | grep '^/dev/dri/' | xargs -n1 basename | sort -u | tr '\n' ' ')"
echo "unit allowlist (dri): $allowed_dri"
run_in_context bash -c 'ls /dev/dri 2>/dev/null | sort' > /tmp/lsa-iso-dri.txt || true
echo "sandbox sees:         $(tr '\n' ' ' < /tmp/lsa-iso-dri.txt)"
while read -r node; do
  [ -z "$node" ] && continue
  case " $allowed_dri " in
    *" $node "*) ;;
    *) echo "ABORT: /dev/dri/$node visible in the sandbox but NOT in the unit allowlist (host renumbered? fix the unit's DeviceAllow, re-run with --resume)"; exit 1 ;;
  esac
done < /tmp/lsa-iso-dri.txt
echo "device allowlist OK: sandbox /dev/dri ⊆ unit allowlist"

echo "-- 7c. GPU visible + Triton JIT compiles and runs INSIDE the sandbox --"
cat > "$NEW_ROOT/home/iso_gpu_probe.py" <<'PYEOF'
import torch, triton, triton.language as tl

@triton.jit
def _double(x_ptr, o_ptr, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(o_ptr + i, tl.load(x_ptr + i) * 2)

assert torch.cuda.is_available() and torch.version.hip, "no ROCm GPU visible in the service context"
x = torch.arange(8, device="cuda")
y = torch.empty_like(x)
_double[(1,)](x, y, N=8)
assert bool((y == x * 2).all()), "triton kernel produced a wrong result in the service context"
print("ISO-GPU-PROBE-OK", torch.cuda.get_device_name(0), "| torch", torch.__version__, "| triton", triton.__version__)
PYEOF
chown lsa-runner:lsa-runner "$NEW_ROOT/home/iso_gpu_probe.py"
run_in_context "$NEW_VENV_ROOT/.venv/bin/python" "$NEW_ROOT/home/iso_gpu_probe.py" | tee /tmp/lsa-iso-gpu.log
grep -q "ISO-GPU-PROBE-OK" /tmp/lsa-iso-gpu.log || { echo "ABORT: in-service GPU/Triton probe failed (see /tmp/lsa-iso-gpu.log)"; exit 1; }
rm -f "$NEW_ROOT/home/iso_gpu_probe.py"

echo "-- 7d. sandbox HOME writable (Triton cache lands there per the unit's Environment) --"
run_in_context bash -c 'test -w "$HOME" && test -w "$TRITON_CACHE_DIR"' \
  || { echo "ABORT: sandbox HOME/TRITON_CACHE_DIR not writable"; exit 1; }

echo "-- 7e. service active, exactly one listener for the identity --"
systemctl is-active --quiet github-runner-lsa.service || { echo "ABORT: github-runner-lsa.service not active"; exit 1; }
listeners=$(pgrep -c -u lsa-runner -f Runner.Listener || true)
[ "$listeners" -eq 1 ] || { echo "ABORT: $listeners Runner.Listener processes under lsa-runner (duplicate-listener risk — stop the extra one before proceeding)"; exit 1; }
echo "unique listener OK (1 Runner.Listener under lsa-runner; confirm the single online session gh-side, handoff step 3)"

cat <<'EOF'
MIGRATION HOST STEPS DONE (service-context isolation + GPU validation passed).
Remaining (in order):
  1. Stale pre-migration queued run 42381769814: CANCELLED 2026-11-08 (docs/r1.1/ACTIONS_QUEUE_AUDIT.md).
     It was bound to the workflow file at 1ab0e0b ($HOME-derived venv) and could NEVER validate
     the migrated runner — do not re-approve any run predating the workflow-fix merge.
  2. Merge PR #6 (rocm-regression.yml PY="/opt/lsa-rocm/.venv/bin/python" + this runbook).
     The merge push itself fires a FRESH GPU run whose workflow resolves the migrated venv:
     approve it in the gpu-regression environment. THAT run — 17/17 gates + full suite +
     receipt validation on the new runner — is the end-to-end post-migration validation.
  3. Confirm exactly one session online for identity lsa-rocm-gfx1151:
     gh api repos/AIwork4me/latent-sparse-attention-rocm/actions/runners --jq '.runners[] | {name,status,busy}'
     (expect a single entry, status online; old install must be offline).
  4. After validation: retire the old install so a duplicate listener is impossible:
        mv /home/amd/actions-runner-lsa /home/amd/actions-runner-lsa.retired-pre-migration
  5. Rotate the amd gh credential (gh auth logout / remove ~/.config/gh/hosts.yml) after
     confirming nothing legit still depends on it.
  6. Scope the runner group to this repository only (runner settings UI; R1 owner-packet item 6).
Rollback at any point: docs/r1.1/ops/rollback-runner.sh (mutual exclusion: never run the legacy
runner and this service concurrently — one identity, one listener).
EOF
