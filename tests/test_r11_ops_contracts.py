"""R1.1 ops contracts - the runner migration handoff cannot silently rot again.

The 2026-11-08 handoff review found the GPU workflow still deriving its venv
from ``$HOME`` (pre-migration semantics) while the migration runbook copied the
venv to ``/opt/lsa-rocm/.venv`` and redefined the service ``HOME`` - so the
kept queued run 42381769814 (bound to the workflow file at ``1ab0e0b``) could
never have validated the migrated runner: it resolves ``$HOME`` inside the new
sandbox and dies at "Locate the ROCm venv". These static contracts pin the
correct invariants so the workflow, the migration script and the runbook
cannot drift apart again:

- the workflow venv path is the absolute migration target, never ``$HOME``;
- the migration script guards partially-completed states and can resume;
- isolation/GPU validation executes in the *service* security context
  (``systemd-run`` with the unit's sandbox directives), not ``sudo -u`` -
  a ``sudo -u`` process sees all of ``/home/amd`` and every device the unit's
  ``DevicePolicy=closed`` allowlist is supposed to filter;
- the post-migration validation vehicle is a FRESH push run of the fixed
  workflow, never the pre-migration queued run.

Pure static checks on committed files: they run on GitHub-hosted CI everywhere.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github/workflows/rocm-regression.yml"
MIGRATE = REPO / "docs/r1.1/ops/migrate-runner.sh"
ROLLBACK = REPO / "docs/r1.1/ops/rollback-runner.sh"
SERVICE = REPO / "docs/r1.1/ops/github-runner-lsa.service"

MIGRATION_VENV = "/opt/lsa-rocm/.venv/bin/python"


def test_files_exist():
    for path in (WORKFLOW, MIGRATE, ROLLBACK, SERVICE):
        assert path.is_file(), f"missing ops artifact {path}"


def test_workflow_venv_is_the_absolute_migration_target():
    """The GPU workflow must pin the migrated venv absolutely: the service
    unit redefines HOME, so any $HOME-derived path silently breaks."""
    text = WORKFLOW.read_text()
    assert MIGRATION_VENV in text, "workflow does not reference the migrated venv"
    venv_step = re.search(r'PY="([^"]+)"', text)
    assert venv_step, "workflow no longer locates the venv via PY"
    assert venv_step.group(1) == MIGRATION_VENV, (
        f"workflow venv {venv_step.group(1)!r} is not the migration target"
    )
    assert "$HOME" not in venv_step.group(1)


def test_migrate_script_targets_the_same_venv_the_workflow_pins():
    """Cross-file consistency: NEW_VENV_ROOT in the script must produce exactly
    the path the workflow executes (the handoff bug was these two drifting)."""
    script = MIGRATE.read_text()
    assert 'NEW_VENV_ROOT=/opt/lsa-rocm' in script
    assert f'"{MIGRATION_VENV}"' in script or MIGRATION_VENV in script


def test_migrate_script_guards_partial_states_and_resumes():
    """A partially-completed migration (user added, copies half-done, no
    service) must be detected - never silently merged into by a fresh run -
    and recoverable via --resume without manual surgery."""
    script = MIGRATE.read_text()
    assert "--resume" in script, "no resume path for partially-completed states"
    assert re.search(r'\[ -e "?\$NEW_ROOT"? \]', script), (
        "fresh run does not abort when /opt/actions-runner-lsa already exists"
    )


def test_validation_runs_in_the_service_security_context_not_sudo_u():
    """Isolation and GPU proof must execute under the unit's own sandbox
    (systemd-run mirroring its directives); `sudo -u` proves nothing about
    ProtectHome/ProtectSystem/DevicePolicy=closed."""
    script = MIGRATE.read_text()
    assert not re.search(r"^\s*sudo\s+-u", script, re.M), (
        "an executable `sudo -u` host-level check survived (documentation mentions are fine)"
    )
    assert "systemd-run" in script, "no service-context validation"
    for directive in ("ProtectHome=true", "ProtectSystem=strict", "DevicePolicy=closed"):
        assert directive in script, f"parity guard misses {directive}"


def test_systemd_run_directives_cannot_drift_from_the_unit():
    """The systemd-run check must take its sandbox directives from the
    installed unit, or at minimum verify parity with it - a hardcoded copy
    would rot the first time the unit changes."""
    script = MIGRATE.read_text()
    assert "systemctl show" in script or "systemd-analyze security" in script, (
        "no parity check between the validation context and the installed unit"
    )


def test_handoff_validation_vehicle_is_a_fresh_push_run():
    """The handoff must not present the pre-migration queued run 42381769814
    (workflow file frozen at 1ab0e0b, $HOME-derived venv) as post-migration
    proof; the vehicle is the push run the workflow-fix merge itself fires."""
    script = MIGRATE.read_text()
    assert "42381769814" not in script or "cancel" in script.lower(), (
        "queued run 42381769814 still referenced as a validation vehicle"
    )
    assert re.search(r"push[ -]run|new push|fresh", script, re.I), (
        "handoff does not name the fresh push run as the validation vehicle"
    )


def test_rollback_keeps_the_mutual_exclusion_rule():
    """Legacy runner and migrated copy share one identity - the rollback doc
    must keep forbidding running both at once."""
    script = ROLLBACK.read_text()
    assert "concurrent" in script or "mutually exclusive" in script


# --------------------------------------------------------------------------
# PR #6 final correction (owner review of 7f1c971): the transient validation
# context must carry the unit's COMPLETE effective device access policy, and
# device validation must exercise the cgroup filter (open(2)), not visibility.
# --------------------------------------------------------------------------

def test_transient_context_transcribes_the_unit_device_allow():
    """Defect 1: the systemd-run context copied DevicePolicy=closed but no
    DeviceAllow= entries - under `closed` that is a zero-device sandbox and
    the in-context GPU probe can never pass. The script must build
    --property=DeviceAllow= props FROM THE LIVE UNIT (via the extracted
    device-policy-check.sh parser), never hardcode them, and must verify
    parity between the unit file and what systemd actually loaded."""
    script = MIGRATE.read_text()
    assert "device-policy-check.sh" in script, (
        "validation does not use the extracted device-policy validator"
    )
    assert re.search(r"--property=DeviceAllow=", script), (
        "transient context is not given the unit's DeviceAllow entries"
    )
    assert re.search(r'parity\s+"?\$?\{?(UNIT_FILE|LSA_UNIT|--unit-file)', script) or \
           re.search(r'check\.sh parity', script), (
        "no explicit unit-file vs live-unit DeviceAllow parity verification"
    )


def test_device_validation_is_access_based_not_visibility_based():
    """Defect 2: `ls /dev/dri`-vs-allowlist compares VISIBILITY, but
    DevicePolicy is enforced by the cgroup eBPF filter on open(2) - every
    context still *sees* the nodes. The check must actually OPEN each
    allowlisted node inside the transient context (positive) and prove a
    non-allowed device is refused (negative control)."""
    script = MIGRATE.read_text()
    assert "⊆ unit allowlist" not in script, (
        "the vacuous filename-visibility check (ls /dev/dri subset) survived"
    )
    assert re.search(r"exec 3<>", script), (
        "no open(2)-based access probe for the allowlisted device nodes"
    )
    assert re.search(r"refused|denied|must NOT be open", script), (
        "no negative control proving non-allowed devices are inaccessible"
    )


def test_gpu_probe_attests_it_ran_in_the_transient_cgroup():
    """The GPU/Triton proof is only meaningful INSIDE the correct cgroup and
    service sandbox: the probe context must attest its own cgroup contains
    the transient systemd unit (run-*.service, INVOCATION_ID set) — not just
    any shell the checks might have escaped into."""
    script = MIGRATE.read_text()
    assert "/proc/self/cgroup" in script, (
        "no cgroup self-attestation inside the validation context"
    )
    assert re.search(r"run-\[0-9a-zA-Z\]\+\\\.service", script), (
        "cgroup attestation does not pin the transient run-*.service unit pattern"
    )
    assert "INVOCATION_ID" in script, (
        "no systemd unit attestation (INVOCATION_ID) inside the probe context"
    )


def test_resume_refuses_while_service_or_gpu_job_is_live():
    """--resume re-runs steps that strip/rmtree under /opt: it must refuse to
    run while the migrated service is active or ANY lsa-runner process (a GPU
    job can outlive `systemctl stop` under KillMode=process) is alive."""
    script = MIGRATE.read_text()
    resume_block = script.split('if [ "$RESUME" = 1 ]; then', 1)[1].split("\nelse", 1)[0]
    assert "is-active" in resume_block, "resume path has no service-active refusal"
    assert re.search(r"pgrep\s+-u\s+lsa-runner", resume_block), (
        "resume path has no liveness check for lsa-runner processes "
        "(GPU job / KillMode=process stragglers)"
    )


def test_never_deletes_a_live_work_directory():
    """Step 3 rm -rf's $NEW_ROOT/_work: the deletion must be guarded by a
    liveness probe (processes with cwd/fds under _work) right before it."""
    script = MIGRATE.read_text()
    step3 = script.split('== 3.', 1)[1].split('== 4.', 1)[0]
    assert "rm -rf" in step3 and "_work" in step3
    guard = script.split('== 3.', 1)[1].split('rm -rf "$NEW_ROOT/_work"', 1)[0]
    assert re.search(r"/proc/.*(/cwd|/fd)|fuser", guard), (
        "no per-process cwd/fd liveness probe guarding the _work deletion"
    )


def test_fresh_run_refuses_preexisting_destination_venv_and_unit():
    """Partial-state detection must also cover: a pre-existing /opt/lsa-rocm
    destination venv state and an already-installed unit file (an attempt that
    died between useradd and enable, or unrelated pre-existing /opt state)."""
    script = MIGRATE.read_text()
    fresh_block = script.split('if [ "$RESUME" = 0 ]', 1)[0]
    assert re.search(r'\[ -e "\$NEW_VENV_ROOT" \]', fresh_block), (
        "fresh run does not abort on pre-existing destination venv state"
    )
    assert "/etc/systemd/system/github-runner-lsa.service" in fresh_block, (
        "fresh run does not detect an already-installed unit file"
    )
