"""Shell fail-closed exit propagation (F11): campaign.sh and overnight.sh.

The v0.2.1 campaign.sh printed ``pilot rc=1`` and still exited 0, so a failed
campaign looked green to any caller (CI, cron, operator script). These tests
run both scripts against a sandboxed repo skeleton with a stub interpreter -
no GPU, no real benchmark - and pin the contract: any failing phase must make
the script exit nonzero, and an all-green stub run must exit 0.
"""

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

FAKE_PY = """#!/usr/bin/env bash
# Stub interpreter: exits FAKE_RC (default 0) regardless of arguments.
echo "fakepy $*"
exit "${FAKE_RC:-0}"
"""

STUB_SH = """#!/usr/bin/env bash
# Stub helper: exits STUB_RC (default 0).
exit "${STUB_RC:-0}"
"""


@pytest.fixture()
def sandbox(tmp_path):
    """A minimal repo skeleton: real campaign/overnight scripts, stub the rest."""
    (tmp_path / "scripts").mkdir()
    shutil.copy2(SCRIPTS / "campaign.sh", tmp_path / "scripts" / "campaign.sh")
    shutil.copy2(SCRIPTS / "overnight.sh", tmp_path / "scripts" / "overnight.sh")
    for name in ("run_upstream_kernel_tests.sh", "telemetry.sh"):
        stub = tmp_path / "scripts" / name
        stub.write_text(STUB_SH)
        stub.chmod(0o755)
    fakepy = tmp_path / "fakepy"
    fakepy.write_text(FAKE_PY)
    fakepy.chmod(0o755)
    return tmp_path


def _run_script(sandbox, script, extra_env=None):
    env = {**os.environ,
           "PATH": f"{sandbox}{os.pathsep}{os.environ.get('PATH', '/usr/bin:/bin')}",
           "PYTHON": str(sandbox / "fakepy"),
           "STUB_RC": "0", "FAKE_RC": "0", **(extra_env or {})}
    return subprocess.run(["bash", str(sandbox / "scripts" / script)],
                          capture_output=True, text=True, env=env, cwd=sandbox,
                          timeout=120)


class TestCampaignSh:
    def test_all_phases_green_exits_zero(self, sandbox):
        result = _run_script(sandbox, "campaign.sh")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "campaign OK" in result.stdout

    def test_pilot_failure_fails_the_campaign(self, sandbox):
        result = _run_script(sandbox, "campaign.sh", {"FAKE_RC": "1"})
        assert result.returncode == 1
        assert "pilot(1)" in result.stdout and "overnight(1)" in result.stdout

    def test_upstream_tests_failure_fails_the_campaign(self, sandbox):
        result = _run_script(sandbox, "campaign.sh", {"STUB_RC": "1"})
        assert result.returncode == 1
        assert "upstream-tests(1)" in result.stdout

    def test_failure_isolation_all_phases_still_run(self, sandbox):
        result = _run_script(sandbox, "campaign.sh", {"FAKE_RC": "3", "STUB_RC": "0"})
        # preflight/pilot/overnight fail via the stub interpreter, upstream passes
        assert result.returncode == 1
        assert "preflight(3)" in result.stdout
        assert "upstream-tests" not in result.stdout.split("FAILED")[1]

    def test_status_file_written(self, sandbox):
        _run_script(sandbox, "campaign.sh", {"FAKE_RC": "1"})
        status = (sandbox / "results" / "campaign-status.txt").read_text()
        assert "campaign FAILED" in status


class TestOvernightSh:
    def test_success_propagates_zero(self, sandbox):
        result = _run_script(sandbox, "overnight.sh")
        assert result.returncode == 0
        assert "run exit code: 0" in result.stdout

    def test_run_failure_code_propagates_exactly(self, sandbox):
        result = _run_script(sandbox, "overnight.sh", {"FAKE_RC": "7"})
        assert result.returncode == 7
        assert "run exit code: 7" in result.stdout

    def test_missing_interpreter_fails_not_greens(self, sandbox):
        env = {**os.environ,
               "PATH": f"{sandbox}{os.pathsep}{os.environ.get('PATH', '/usr/bin:/bin')}",
               "PYTHON": str(sandbox / "absent-python")}
        result = subprocess.run(["bash", str(sandbox / "scripts" / "overnight.sh")],
                                capture_output=True, text=True, env=env, cwd=sandbox,
                                timeout=120)
        assert result.returncode != 0
