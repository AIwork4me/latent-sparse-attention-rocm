#!/usr/bin/env bash
# migrate-runner.sh — R1.1 runner migration, owner-executed with sudo (script is read-only until then).
# Usage:
#   sudo bash docs/r1.1/ops/migrate-runner.sh            # fresh migration (aborts on ANY partial state)
#   sudo bash docs/r1.1/ops/migrate-runner.sh --resume   # resume a partially-completed migration
# Preconditions (verified nonprivileged 2026-11-08, see ../RUNNER_MIGRATION_VALIDATION.md):
#   /opt >= 12 GiB free · venv copy+repair empirically validated · siblings are amd's user-scoped
#   units (untouched by this system-unit migration).
# Partial-state policy: a fresh run aborts if the user OR /opt state already exists (never merges
# into a half-copy silently). Steps 2-8 are idempotent (rsync / sed / rm / chown / unit
# install / validation / enable+start),
# so --resume safely re-runs them from wherever a previous attempt aborted. To discard a partial
# state instead: rollback-runner.sh, then `userdel -r lsa-runner; rm -rf /opt/actions-runner-lsa /opt/lsa-rocm`.
# Validation (step 7) executes INSIDE the service security context — a transient systemd-run
# service transcribing the installed unit's COMPLETE effective sandbox: namespace directives AND
# the full device access policy (DevicePolicy + every DeviceAllow entry, via
# device-policy-check.sh parity/emit-entries) — never via `sudo -u`: a plain sudo context sees
# all of /home and every device the unit's DevicePolicy=closed filters. Device checks are
# ACCESS-based (structured os.open()/OSError.errno probes via device-open-probe.py — EPERM is
# the cgroup filter's signature, EACCES is DAC, never strerror string-matched), never
# visibility-based: DevicePolicy does not hide nodes from /dev, it gates open(2).
# SEQUENCING (fail-closed): step 6 installs the unit and reloads the manager but does NOT
# start the runner; §7 validates everything first; step 8 runs `systemctl enable --now` only
# after ALL checks pass. Any pre-start failure leaves the runner OFFLINE (unit installed but
# inactive) — recover with --resume or rollback-runner.sh.
# --resume refuses to run while the migrated service is active or any lsa-runner process
# (a GPU job can outlive `systemctl stop` under KillMode=process) is alive, and step 3 never
# deletes a _work tree any process still holds open.
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
  # Never resume into a LIVE system: steps 2-3 overwrite/strip the very trees the
  # service and its jobs own. A GPU job can outlive `systemctl stop` (KillMode=process
  # kills only the main process), so an explicit per-user process check is required
  # even when the service already reports inactive.
  if systemctl is-active --quiet github-runner-lsa.service 2>/dev/null; then
    echo "ABORT: --resume but github-runner-lsa.service is ACTIVE — the migration appears"
    echo "       complete/live. --resume is for ABORTED attempts only. To redo a completed"
    echo "       migration: rollback-runner.sh first, then run fresh. To re-run §7 validation"
    echo "       only: systemctl stop github-runner-lsa, wait for lsa-runner processes to"
    echo "       exit, then --resume (steps 2-7 are idempotent; §7 re-validates everything)."
    exit 1
  fi
  if pgrep -u lsa-runner >/dev/null 2>&1; then
    echo "ABORT: --resume but lsa-runner processes are still alive (GPU job or KillMode=process straggler):"
    pgrep -a -u lsa-runner || true
    echo "       let the job finish and/or stop the service, WAIT for exit, then re-run --resume."
    exit 1
  fi
  echo "resume mode: user + /opt state present and idle, re-running idempotent steps 2-7"
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
  if [ -e "$NEW_VENV_ROOT" ]; then
    echo "ABORT: $NEW_VENV_ROOT already exists (pre-existing destination venv state) — inspect it,"
    echo "       then either --resume (if it is our partial copy) or remove it; never rsync over an unknown tree"
    exit 1
  fi
  if [ -e /etc/systemd/system/github-runner-lsa.service ]; then
    echo "ABORT: /etc/systemd/system/github-runner-lsa.service already installed (partially"
    echo "       completed migration) — rollback-runner.sh removes it, or --resume if the /opt"
    echo "       state is a partial copy of ours"
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
# NEVER delete a live _work: a running job holds its checkout (cwd) and files
# (fds) under _work. Probe every lsa-runner process's cwd + open descriptors
# first — if anything still touches _work, abort instead of rm -rf.
if [ -d "$NEW_ROOT/_work" ]; then
  busy=""
  for pid in $(pgrep -u lsa-runner 2>/dev/null || true); do
    if readlink "/proc/$pid/cwd" 2>/dev/null | grep -qF "$NEW_ROOT/_work"; then
      busy="$busy $pid(cwd)"
    fi
    for fd in "/proc/$pid/fd"/*; do
      if readlink "$fd" 2>/dev/null | grep -qF "$NEW_ROOT/_work"; then
        busy="$busy $pid(fd)"; break
      fi
    done
  done
  [ -z "$busy" ] || {
    echo "ABORT: processes still hold $NEW_ROOT/_work open:$busy — never delete a live _work."
    echo "       Let the job finish / stop the service and wait for exit, then re-run with --resume."
    exit 1
  }
fi
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

echo "== 6. System unit install + manager reload (runner NOT started yet) =="
# Fail-closed sequencing: the runner must stay offline until EVERY pre-start check
# in §7 passes (device access, isolation, writability, Triton JIT). A §7 abort
# therefore leaves the unit installed-but-inactive; rollback or --resume from there.
cp "$HERE/github-runner-lsa.service" /etc/systemd/system/github-runner-lsa.service
systemd-analyze verify /etc/systemd/system/github-runner-lsa.service
systemctl daemon-reload

echo "== 7. Isolation + GPU validation IN THE SERVICE SECURITY CONTEXT =="
UNIT_FILE=/etc/systemd/system/github-runner-lsa.service
# Parity guard: the installed unit must still carry the R1.1 design sandbox, so the validation
# context below (transcribed from the live unit) can never silently weaken with an owner edit.
for line in "User=lsa-runner" "ProtectHome=true" "ProtectSystem=strict" "DevicePolicy=closed" \
            "PrivateTmp=true" "NoNewPrivileges=true"; do
  grep -q "^$line\$" "$UNIT_FILE" || {
    echo "ABORT: unit no longer pins '$line' — reconcile this script with the unit before validating"; exit 1
  }
done
# Effective device access policy — the pre-correction defect: the transient context carried
# DevicePolicy=closed but NO DeviceAllow entries, i.e. a zero-device sandbox in which the GPU
# probe below could never pass. Rebuild the COMPLETE policy from the installed unit and verify
# parity explicitly (unit file == what systemd actually loaded; catches drop-ins/reload lag)
# plus node compatibility (the unit's nodes must exist on THIS host — card renumbering after a
# reboot aborts loudly instead of failing later with a mystery EPERM).
SHOW_FILE=/tmp/lsa-unit-device-policy.txt
systemctl show github-runner-lsa.service -p DevicePolicy -p DeviceAllow > "$SHOW_FILE"
bash "$HERE/device-policy-check.sh" parity "$UNIT_FILE" "$SHOW_FILE" || {
  echo "ABORT: the live unit's device access policy diverges from $UNIT_FILE (details above)"; exit 1
}
bash "$HERE/device-policy-check.sh" nodes "$UNIT_FILE" || {
  echo "ABORT: the unit's DeviceAllow entries do not match this host's GPU device nodes (details above)"; exit 1
}
# Transcribe the live unit into a transient service: namespace directives, Environment= AND
# the full device access policy come from systemctl show / its capture, so what the checks
# below see IS what the runner service sees.
props=(--wait --pipe --collect --quiet)
props+=(--property=User=lsa-runner --property=Group=lsa-runner)
props+=(--property=WorkingDirectory="$(systemctl show github-runner-lsa.service -p WorkingDirectory --value)")
for p in ProtectHome ProtectSystem PrivateTmp NoNewPrivileges DevicePolicy; do
  props+=(--property="$p=$(systemctl show github-runner-lsa.service -p "$p" --value)")
done
for p in $(systemctl show github-runner-lsa.service -p ReadWritePaths --value); do
  props+=(--property=ReadWritePaths="$p")
done
# Fail-closed transcription capture: `mapfile < <(...)` cannot propagate a parser
# failure — a malformed capture would silently transcribe ZERO DeviceAllow entries
# (the defect-1 zero-device context). Capture to a file, check the exit code
# explicitly, and refuse an empty entry list.
DA_LIST=/tmp/lsa-da-entries.txt
bash "$HERE/device-policy-check.sh" emit-entries "$SHOW_FILE" > "$DA_LIST" || {
  echo "ABORT: DeviceAllow transcription (emit-entries) failed — the transient context"
  echo "       would be built without the unit's device allowlist; runner stays offline"
  exit 1
}
mapfile -t DA_ENTRIES < "$DA_LIST"
[ "${#DA_ENTRIES[@]}" -ge 1 ] || {
  echo "ABORT: emit-entries produced no DeviceAllow entries (zero-device transcription) — runner stays offline"
  exit 1
}
for da_entry in "${DA_ENTRIES[@]}"; do
  props+=(--property=DeviceAllow="$da_entry")
done
read -ra ENV_WORDS <<< "$(systemctl show github-runner-lsa.service -p Environment --value)"
for w in "${ENV_WORDS[@]}"; do props+=(--setenv="$w"); done
run_in_context() { systemd-run "${props[@]}" "$@"; }

echo "-- 7a. ProtectHome: ALL of /home (amd's gh credential, old install, siblings) invisible --"
run_in_context bash -c 'test ! -e /home/amd && test ! -e /home/lsa-runner' \
  || { echo "ABORT: /home visible inside the service context (ProtectHome not effective)"; exit 1; }
echo "credential isolation OK: /home entirely invisible in the service security context"

echo "-- 7b. DevicePolicy=closed allowlist enforced on open(2): allowed nodes OPEN, non-allowed EPERM-denied --"
# Access, not visibility: DevicePolicy is a cgroup eBPF filter on open(2) — every context
# still SEES all nodes under /dev/dri. Each allowlisted node is OPENED inside the transient
# context by device-open-probe.py: os.open() + OSError.errno, NEVER strerror string matching
# (EPERM(1)="Operation not permitted" is the cgroup filter's signature; EACCES(13)=
# "Permission denied" is DAC — indistinguishable by text). The probe is staged into
# $NEW_ROOT/home because ProtectHome=true hides the repo checkout the script runs from;
# probe stderr goes to a persistent log so every abort below is diagnosable post-mortem.
PROBE_ERR_LOG=/tmp/lsa-open-probe-stderr.log
: > "$PROBE_ERR_LOG"
cleanup_staged_probes() { rm -f "$NEW_ROOT/home/iso_gpu_probe.py" "$NEW_ROOT/home/device-open-probe.py"; }
trap cleanup_staged_probes EXIT   # staged probes are never part of the install, even on abort
OPEN_PROBE="$NEW_ROOT/home/device-open-probe.py"
VENV_PY="$NEW_VENV_ROOT/.venv/bin/python"
cp "$HERE/device-open-probe.py" "$OPEN_PROBE"
chown lsa-runner:lsa-runner "$OPEN_PROBE" && chmod 0755 "$OPEN_PROBE"
probe_mode_for() {  # unit mode (subset of r,w,m) -> open mode for the probe
  local om=""
  [[ "$1" == *r* ]] && om+="r"
  [[ "$1" == *w* ]] && om+="w"
  echo "${om:-}"
}
probed=0
for da_entry in "${DA_ENTRIES[@]}"; do
  read -r da_node da_mode <<<"$da_entry"
  case "$da_node" in char-*|block-*) continue ;; esac   # device-group globs: no node to open
  case "$da_mode" in
    *[rw]*) ;;                                          # has an open permission
    *) echo "   note: $da_node grants mknod only ('m') — no open permission to probe"; continue ;;
  esac
  om="$(probe_mode_for "$da_mode")"
  open_line="$(run_in_context "$VENV_PY" "$OPEN_PROBE" open "$da_node" "$om" 2>>"$PROBE_ERR_LOG")" || true
  if [ "$open_line" = "OPEN-OK $da_node" ]; then
    echo "   openable in-sandbox: $da_node ($da_mode)"
    probed=$((probed + 1))
  elif [[ "$open_line" == "OPEN-ERR $da_node "* ]]; then
    read -r _ _ p_errno p_name <<<"$open_line"
    echo "ABORT: allowlisted device $da_node ($da_mode) NOT accessible in the service context:"
    echo "       open() failed with $p_name ($p_errno) — EPERM means the DeviceAllow entry was"
    echo "       not transcribed into the transient unit; ENOENT/ENXIO mean the node is gone."
    echo "       Runner stays OFFLINE. Probe stderr: $PROBE_ERR_LOG"
    exit 1
  else
    echo "ABORT: unparseable probe output for $da_node: '$open_line' — runner stays offline."
    echo "       Probe stderr (venv python / systemd-run diagnostics): $PROBE_ERR_LOG"
    exit 1
  fi
done
[ "$probed" -gt 0 ] || { echo "ABORT: no /dev-path DeviceAllow entries with open permissions to probe — runner stays offline"; exit 1; }
# Negative control — STRONG TIER REQUIRED (adversarial review: weak-tier-only evidence must
# never be enough to start the runner). Strong = a non-allowed char/block device node that
# lsa-runner can open by DAC ALONE, proven OUTSIDE any device filter (setpriv as lsa-runner
# in our unrestricted root cgroup; BOTH rc==0 and OPEN-OK required). Inside the service
# context that same open MUST then fail with EPERM — the cgroup device filter's signature.
# EACCES there = DAC refused (node groups/mode changed? udev/ACL race?) — NOT filter
# evidence. ENOENT/ENXIO/anything else = ambiguous. Every non-EPERM outcome hard-aborts.
allowed_nodes="$(printf '%s\n' "${DA_ENTRIES[@]}" | awk '{print $1}')"
neg_control="" dac_line=""
for cand_glob in "/dev/dri/*" "/dev/video*"; do
  for n in $cand_glob; do
    { [ -c "$n" ] || [ -b "$n" ]; } || continue            # device nodes only (never dirs/symlink-dirs)
    grep -qx "$n" <<<"$allowed_nodes" && continue          # must NOT be allowlisted
    if command -v setpriv >/dev/null 2>&1; then
      dac_rc=0
      dac_line="$(setpriv --reuid lsa-runner --regid lsa-runner --init-groups \
                    "$VENV_PY" "$OPEN_PROBE" open "$n" r 2>>"$PROBE_ERR_LOG")" || dac_rc=$?
      if [ "$dac_rc" -eq 0 ] && [ "$dac_line" = "OPEN-OK $n" ]; then
        neg_control="$n"
        break 2
      fi
      echo "   note: candidate $n failed the DAC premise (rc=$dac_rc, $dac_line) — trying the next candidate"
    else
      echo "   note: setpriv unavailable — a DAC premise cannot be evidenced; the strong tier is impossible"
      break 2
    fi
  done
done
if [ -z "$neg_control" ]; then
  echo "ABORT: no STRONG negative control available: found no non-allowlisted device node that"
  echo "       lsa-runner can open by DAC alone (tried /dev/dri/* and /dev/video*; setpriv: $(command -v setpriv || echo missing))."
  echo "       Without it the cgroup device filter cannot be PROVEN to discriminate — the runner stays OFFLINE."
  echo "       Ensure a video/render-group device exists outside the allowlist (e.g. a camera node /dev/video0),"
  echo "       then re-run with --resume. Details: $PROBE_ERR_LOG"
  exit 1
fi
echo "   DAC premise (as lsa-runner, OUTSIDE any device filter): $dac_line"
neg_line="$(run_in_context "$VENV_PY" "$OPEN_PROBE" open "$neg_control" r 2>>"$PROBE_ERR_LOG")" || true
if [ "$neg_line" = "OPEN-OK $neg_control" ]; then
  echo "ABORT: non-allowed device $neg_control WAS openable in the service context —"
  echo "       DevicePolicy=closed is not filtering (transient context mis-built or policy"
  echo "       drifted); this is a security failure. Runner stays OFFLINE. Probe stderr: $PROBE_ERR_LOG"
  exit 1
elif [[ "$neg_line" == "OPEN-ERR $neg_control "* ]]; then
  read -r _ _ n_errno n_name <<<"$neg_line"
  case "$n_name" in
    EPERM)
      echo "device access OK: $probed allowlisted node(s) open in-sandbox; non-allowed $neg_control"
      echo "   denied with EPERM (cgroup device filter signature), DAC premise proven outside the filter" ;;
    EACCES)
      echo "ABORT: strong-tier control $neg_control was refused with EACCES (DAC), not EPERM —"
      echo "       the node's groups/mode changed since the premise probe (udev/ACL race?) or the premise"
      echo "       was misread. This is NOT cgroup-filter evidence. Runner stays OFFLINE. Probe stderr: $PROBE_ERR_LOG"
      exit 1 ;;
    *)
      echo "ABORT: non-allowed $neg_control refused for an unexpected reason: $n_name ($n_errno) —"
      echo "       ambiguous (node vanished / no driver?). Runner stays OFFLINE. Probe stderr: $PROBE_ERR_LOG"
      exit 1 ;;
  esac
else
  echo "ABORT: unparseable negative probe output: '$neg_line' — runner stays offline."
  echo "       Probe stderr (venv python / systemd-run diagnostics): $PROBE_ERR_LOG"
  exit 1
fi

echo "-- 7c. GPU visible + Triton JIT compiles and runs INSIDE the sandbox --"
# The GPU proof is only meaningful inside the CORRECT cgroup: attest the probe context is
# the transient systemd service itself (system.slice/run-*.service with an INVOCATION_ID),
# not some stray shell the checks accidentally escaped into.
cg_line="$(run_in_context bash -c 'cat /proc/self/cgroup')"
echo "probe cgroup: $cg_line"
echo "$cg_line" | grep -qE '/system\.slice/run-[0-9a-zA-Z]+\.service$' \
  || { echo "ABORT: probe context's /proc/self/cgroup is not a transient system.slice/run-*.service unit — the sandbox attestation failed"; exit 1; }
inv="$(run_in_context bash -c 'echo "$INVOCATION_ID"')"
[ -n "$inv" ] || { echo "ABORT: INVOCATION_ID unset inside the probe context — not running as a systemd unit"; exit 1; }
echo "cgroup attestation OK: probes execute inside the transient service unit (the runner's own sandbox semantics)"
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
run_in_context "$VENV_PY" "$NEW_ROOT/home/iso_gpu_probe.py" | tee /tmp/lsa-iso-gpu.log
grep -q "ISO-GPU-PROBE-OK" /tmp/lsa-iso-gpu.log || { echo "ABORT: in-service GPU/Triton probe failed (see /tmp/lsa-iso-gpu.log)"; exit 1; }

echo "-- 7d. sandbox HOME writable (Triton cache lands there per the unit's Environment) --"
run_in_context bash -c 'test -w "$HOME" && test -w "$TRITON_CACHE_DIR"' \
  || { echo "ABORT: sandbox HOME/TRITON_CACHE_DIR not writable"; exit 1; }

echo "== 8. ALL pre-start checks PASSED — start the persistent runner now =="
systemctl enable --now github-runner-lsa.service
sleep 5
systemctl status github-runner-lsa.service --no-pager | head -8
systemctl is-active --quiet github-runner-lsa.service || {
  echo "ABORT: github-runner-lsa.service not active after enable --now."
  echo "       The unit remains ENABLED (it will start on reboot) — keep it offline with"
  echo "       `systemctl disable github-runner-lsa` or run rollback-runner.sh."
  exit 1
}
listeners=$(pgrep -c -u lsa-runner -f Runner.Listener || true)
[ "$listeners" -eq 1 ] || {
  echo "ABORT: $listeners Runner.Listener processes under lsa-runner (duplicate-listener risk)."
  echo "       The unit remains enabled+running — stop the extra listener, or run rollback-runner.sh."
  exit 1
}
echo "unique listener OK (1 Runner.Listener under lsa-runner; confirm the single online session gh-side, handoff step 3)"

cat <<'EOF'
MIGRATION HOST STEPS DONE (validated FIRST, started LAST: §7 proved the COMPLETE unit
device policy on the transient context — every DeviceAllow node opens in-sandbox and the
non-allowed control is denied with EPERM, its DAC premise evidenced outside the filter —
plus credential isolation, writability and a real Triton JIT run in the service cgroup;
§8 only then enabled+started the persistent runner and verified exactly one Listener).
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
