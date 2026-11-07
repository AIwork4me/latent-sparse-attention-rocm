"""Negative tests for execute_stage - the fail-closed stage classifier (F11).

Every way a configured and attempted adapter can go wrong must classify as
something other than PASS: nonzero exit, BLOCKED exit 78, process crash,
timeout, missing/partial/malformed receipt, skipped gates, missing upstream
SHA, and pre-execution failures (log file cannot even be created). A verified
PASS needs exit 0 AND a receipt that survives validate_receipt.
"""

import json
import sys
from pathlib import Path

import pytest

from lsa_rocm_lab.gate_manifest import REQUIRED_GATE_IDS
from lsa_rocm_lab.run import execute_stage

SHA = "5ebea732d639576d2a7579b0de40e6252f9001f5"


def _valid_kernel_receipt() -> dict:
    return {
        "kind": "lsa_rocm_kernel",
        "status": "PASS",
        "correctness_pass": True,
        "gates_skipped": False,
        "run_id": "stub-run",
        "repo_head_sha": "0" * 40,
        "upstream_sha": SHA,
        "gpu_arch": "gfx1151",
        "gpu_name": "stub",
        "torch": "stub",
        "command": ["stub"],
        "gates": [{"gate_id": i, "gate": f"gate {i}", "status": "PASS"} for i in REQUIRED_GATE_IDS],
        "stages": [],
        "comparisons": [
            {"context": 8192, "dtype": "bf16", "phase": "decode", "method": "lsa_flat",
             "stage": "select+read", "baseline_ms": 1.0, "candidate_ms": 0.5,
             "samples": 20, "speedup": 2.0}
        ],
        "env": {"dtype": "bf16", "reps": 20, "warmup": 5, "seed": 20261105,
                "contexts": "all", "phases": ["prefill", "decode"]},
    }


def _cmd_writes_receipt(payload, exit_code=0):
    """A stub adapter that writes `payload` to $LSA_STAGE_RECEIPT, then exits."""
    return [sys.executable, "-c",
            "import json,os,sys;json.dump(json.loads(sys.argv[1]),"
            "open(os.environ['LSA_STAGE_RECEIPT'],'w'));sys.exit(int(sys.argv[2]))",
            json.dumps(payload), str(exit_code)]


@pytest.fixture()
def stage_dir(tmp_path):
    folder = tmp_path / "run"
    (folder / "receipts").mkdir(parents=True)
    return folder


def _run(folder, cmd, name="lsa_rocm_kernel", sha=SHA, tmax=30):
    return execute_stage(name, cmd, folder, upstream=Path("/nonexistent"),
                         sha=sha, run_id="test-run", tmax=tmax, base_env={})


class TestPassNeedsVerifiedEvidence:
    def test_exit0_with_valid_receipt_is_pass(self, stage_dir):
        status = _run(stage_dir, _cmd_writes_receipt(_valid_kernel_receipt()))
        assert status["status"] == "PASS"
        assert status["exit_code"] == 0
        assert status["command"] and status["started_utc"]

    def test_e2e_pass_needs_armed_server_receipt(self, stage_dir):
        receipt = {"kind": "lsa_rocm_end_to_end", "status": "PASS",
                   "correctness_pass": True, "upstream_sha": SHA, "gpu_arch": "gfx1151"}
        status = _run(stage_dir, _cmd_writes_receipt(receipt), name="lsa_rocm_end_to_end")
        assert status["status"] == "FAIL"  # no sglang_armed/model/metrics: not a PASS


class TestAttemptedAdapterFailures:
    def test_nonzero_exit_is_fail(self, stage_dir):
        assert _run(stage_dir, [sys.executable, "-c", "raise SystemExit(1)"])["status"] == "FAIL"

    def test_exit78_is_blocked_and_fails_the_claim(self, stage_dir):
        status = _run(stage_dir, _cmd_writes_receipt({"kind": "lsa_rocm_kernel",
                                                      "status": "BLOCKED"}, exit_code=78))
        assert status["status"] == "BLOCKED"

    def test_signal_crash_is_fail(self, stage_dir):
        assert _run(stage_dir, [sys.executable, "-c",
                                "import os,signal;os.kill(os.getpid(),signal.SIGKILL)"]
                    )["status"] == "FAIL"

    def test_timeout_reaps_and_classifies(self, stage_dir):
        status = _run(stage_dir, [sys.executable, "-c", "import time;time.sleep(30)"], tmax=1)
        assert status["status"] == "TIMEOUT"

    def test_zero_exit_missing_receipt_is_fail(self, stage_dir):
        assert _run(stage_dir, [sys.executable, "-c", "pass"])["status"] == "FAIL"

    def test_partial_truncated_receipt_is_fail(self, stage_dir):
        cmd = [sys.executable, "-c",
               "open(__import__('os').environ['LSA_STAGE_RECEIPT'],'w').write('{trunc')"
               ",sys.exit(0)"]
        assert _run(stage_dir, cmd)["status"] == "FAIL"

    def test_skipped_gates_receipt_is_fail_even_on_exit0(self, stage_dir):
        receipt = _valid_kernel_receipt()
        receipt.update(status="UNVERIFIED", correctness_pass=False, gates=[])
        assert _run(stage_dir, _cmd_writes_receipt(receipt))["status"] == "FAIL"

    def test_failed_gate_receipt_is_fail_even_on_exit0(self, stage_dir):
        receipt = _valid_kernel_receipt()
        receipt["gates"][3]["status"] = "FAIL"
        assert _run(stage_dir, _cmd_writes_receipt(receipt))["status"] == "FAIL"

    def test_missing_upstream_sha_is_fail(self, stage_dir):
        assert _run(stage_dir, _cmd_writes_receipt(_valid_kernel_receipt()),
                    sha="UNAVAILABLE")["status"] == "FAIL"

    def test_mismatched_upstream_sha_is_fail(self, stage_dir):
        assert _run(stage_dir, _cmd_writes_receipt(_valid_kernel_receipt()),
                    sha="f" * 40)["status"] == "FAIL"

    def test_wrong_gpu_arch_is_fail(self, stage_dir):
        receipt = _valid_kernel_receipt()
        receipt["gpu_arch"] = "gfx942:somecode"  # not the claimed device
        assert _run(stage_dir, _cmd_writes_receipt(receipt))["status"] == "FAIL"


class TestPreExecutionFailures:
    def test_unwritable_log_dir_is_fail_not_crash(self, stage_dir, tmp_path):
        stage_dir.chmod(0o555)  # log creation must fail -> FAIL with error recorded
        try:
            status = _run(stage_dir, [sys.executable, "-c", "pass"])
            assert status["status"] == "FAIL"
            assert "error" in status
        finally:
            stage_dir.chmod(0o755)

    def test_none_command_is_blocked_with_reason(self, stage_dir):
        status = _run(stage_dir, None)
        assert status["status"] == "BLOCKED"
        assert status["command"] is None
        assert "reason" in status
