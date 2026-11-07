"""Shell fail-closed exit propagation (F11 pattern) for seed_study.sh (v0.3.0/T1).

A failed adapter repetition or an INSUFFICIENT aggregation must make the study
exit nonzero - a thin or broken multi-seed study must never look green, because
T2-T4 gain claims are gated on its variance band. The sandbox stubs the python
interpreter (no GPU); one failing stub distinguishes the adapter calls from the
aggregator call by argument.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

FAKE_PY = """#!/usr/bin/env bash
# Stub: adapter invocations write an (empty) receipt at $LSA_STAGE_RECEIPT and
# exit ADAPTER_RC; the aggregator exits AGG_RC.
case "$*" in
  *aggregate_seed_study.py*) echo "stub-aggregate"; exit "${AGG_RC:-0}";;
  *) [ -n "${LSA_STAGE_RECEIPT:-}" ] && echo '{}' > "$LSA_STAGE_RECEIPT"
     echo "fakepy $*"; exit "${ADAPTER_RC:-0}";;
esac
"""


@pytest.fixture()
def sandbox(tmp_path):
    (tmp_path / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / "seed_study.sh", tmp_path / "scripts" / "seed_study.sh")
    upstream = tmp_path / "third_party"
    upstream.mkdir()
    (upstream / "upstream.sha").write_text("0" * 40 + "\n")
    fakepy = tmp_path / "fakepy"
    fakepy.write_text(FAKE_PY)
    fakepy.chmod(0o755)
    return tmp_path


def _run(sandbox, extra_env=None):
    env = {**os.environ, "PYTHON": str(sandbox / "fakepy"),
           "ADAPTER_RC": "0", "AGG_RC": "0", **(extra_env or {})}
    return subprocess.run(["bash", str(sandbox / "scripts" / "seed_study.sh")],
                          capture_output=True, text=True, env=env, cwd=sandbox, timeout=120)


def test_green_study_exits_zero_and_runs_nine_repetitions(sandbox):
    result = _run(sandbox)
    assert result.returncode == 0, result.stdout + result.stderr
    # 3 seeds x 3 reps: rep ids 1..9 under one stamp, receipts named after them
    for n in range(1, 10):
        assert f"rep{n}-" in result.stdout
    receipts = sorted((sandbox / "results").glob("seed-study-*/receipt-rep*.json"))
    assert len(receipts) == 9
    assert "aggregate rc=0" in result.stdout


def test_one_failed_repetition_fails_the_whole_study(sandbox):
    result = _run(sandbox, {"ADAPTER_RC": "1"})
    assert result.returncode == 1
    # failure isolation: all 9 repetitions still attempted and each reported
    rep_lines = [line for line in result.stdout.splitlines() if ") rc=" in line]
    assert len(rep_lines) == 9
    assert all(line.rstrip().endswith("rc=1") for line in rep_lines)


def test_insufficient_aggregation_fails_the_study(sandbox):
    result = _run(sandbox, {"AGG_RC": "1"})
    assert result.returncode == 1
    assert "aggregate rc=1" in result.stdout


def test_default_seed_set_has_three_distinct_seeds(sandbox):
    result = _run(sandbox)
    # the default seed list must stay >= 3 distinct seeds (issue #2 contract)
    seeds = set()
    for line in result.stdout.splitlines():
        if line.startswith("=== seed"):
            seeds.add(line.split()[2])
    assert len(seeds) == 3
