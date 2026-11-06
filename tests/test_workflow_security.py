"""Negative checks for the runner-routing guard (F10).

``scripts/check_workflow_security.py`` is the regression tripwire that keeps
untrusted PR code off the persistent gfx1151 host. These tests pin each rule
with adversarial YAML fixtures: the exact workflow shapes an attacker (or a
careless maintainer) would need to write must be caught, and the repository's
own live workflows must always pass.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_guard():
    spec = importlib.util.spec_from_file_location(
        "check_workflow_security", ROOT / "scripts" / "check_workflow_security.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = _load_guard()
check_all, check_workflow = guard.check_all, guard.check_workflow

SHA = "11d5960a326750d5838078e36cf38b85af677262"  # actions/checkout v4.3.0


def _write(tmp_path, body):
    path = tmp_path / ".github" / "workflows" / "wf.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path.parent


GOOD_GPU = f"""
name: trusted-gpu
on:
  push:
    branches: [main]
permissions:
  contents: read
jobs:
  gpu:
    runs-on: [self-hosted, linux, rocm, gfx1151]
    steps:
      - uses: actions/checkout@{SHA}
"""

EVIL_PR_GPU = f"""
name: evil-pr-gpu
on: pull_request
jobs:
  gpu:
    runs-on: [self-hosted, gfx1151]
    steps:
      - uses: actions/checkout@{SHA}
"""

EVIL_PR_TARGET = """
name: evil-prt
on: pull_request_target
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.sha }}
"""

EVIL_WORKFLOW_RUN = f"""
name: evil-wfr
on: workflow_run
jobs:
  gpu:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@{SHA}
"""

FLOATING_TAG = """
name: floating
on: push
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
"""

SELF_HOSTED_SECRET = """
name: secret-leak
on: push
jobs:
  gpu:
    runs-on: [self-hosted]
    steps:
      - uses: actions/checkout@""" + SHA + """
      - run: echo ${{ secrets.GPU_TOKEN }}
"""


class TestRules:
    def test_trusted_gpu_workflow_passes(self, tmp_path):
        assert check_all(_write(tmp_path, GOOD_GPU)) == []

    def test_r1_public_pr_to_self_hosted_is_caught(self, tmp_path):
        findings = check_all(_write(tmp_path, EVIL_PR_GPU))
        assert any(f.startswith("R1") and "'pull_request'" in f for f in findings)

    def test_r2_pull_request_target_is_caught_everywhere(self, tmp_path):
        findings = check_all(_write(tmp_path, EVIL_PR_TARGET))
        assert any(f.startswith("R2") for f in findings)
        assert any(f.startswith("R4") for f in findings)  # also floating tag

    def test_r3_workflow_run_to_self_hosted_is_caught(self, tmp_path):
        findings = check_all(_write(tmp_path, EVIL_WORKFLOW_RUN))
        assert any(f.startswith("R1") and "'workflow_run'" in f for f in findings)

    def test_r4_floating_action_tag_is_caught(self, tmp_path):
        findings = check_all(_write(tmp_path, FLOATING_TAG))
        assert any(f.startswith("R4") for f in findings)

    def test_r5_secrets_on_self_hosted_is_caught(self, tmp_path):
        findings = check_all(_write(tmp_path, SELF_HOSTED_SECRET))
        assert any(f.startswith("R5") and "secrets.GPU_TOKEN" in f for f in findings)

    def test_no_workflow_files_is_a_finding(self, tmp_path):
        assert check_all(tmp_path / "nowhere")

    def test_list_trigger_form_parsed(self, tmp_path):
        """`on: [push, pull_request]` must parse (never silently miss a trigger)."""
        body = EVIL_PR_GPU.replace("on: pull_request", "on: [push, pull_request]")
        findings = check_all(_write(tmp_path, body))
        assert any("'pull_request'" in f for f in findings)


def test_repository_workflows_pass_the_guard():
    """The live .github/workflows must always satisfy every rule."""
    assert check_all(ROOT / ".github" / "workflows") == []


def test_guard_cli_exits_nonzero_on_violation(tmp_path):
    workflows = _write(tmp_path, EVIL_PR_GPU)
    script = ROOT / "scripts" / "check_workflow_security.py"
    result = subprocess.run([sys.executable, str(script), str(workflows)],
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert "R1" in result.stderr
    good = _write(tmp_path, GOOD_GPU)
    result = subprocess.run([sys.executable, str(script), str(good)],
                            capture_output=True, text=True)
    assert result.returncode == 0
