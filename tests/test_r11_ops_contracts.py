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
