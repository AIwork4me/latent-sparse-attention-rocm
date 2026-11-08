"""R1.1 structured device-open probe - machine-parseable errno reporting.

The last-mile review (2026-11-08, PR #6) found the negative device control
matched strerror STRINGS ("permission denied") - but on Linux:

    EPERM  (1)  -> "Operation not permitted"   (cgroup device filter denies)
    EACCES (13) -> "Permission denied"          (DAC: mode bits / ACL denies)

so the string match (a) would have classified a real cgroup-filter denial
(EPERM) as an "unexpected reason" false abort, and (b) would happily accept a
plain DAC rejection (EACCES) as "filter active" evidence - exactly the
confusion the review forbids.

``docs/r1.1/ops/device-open-probe.py`` replaces string matching with
``os.open()`` + ``OSError.errno`` and prints one machine-parseable line:

    OPEN-OK  <path>                        rc 0
    OPEN-ERR <path> <errno> <ERRNAME>      rc 1
    usage errors -> stderr,                rc 2

These tests execute the REAL probe (no mocks) against fixture files and pin
the errno vocabulary (EPERM/EACCES/ENOENT/ENXIO/...) the migrate script's
strong/weak negative-control tiers dispatch on.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PROBE = REPO / "docs/r1.1/ops/device-open-probe.py"

IS_ROOT = os.geteuid() == 0


def run_probe(*args):
    proc = subprocess.run(
        [sys.executable, str(PROBE), *[str(a) for a in args]],
        capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def test_open_reports_ok_for_a_readable_node(tmp_path):
    node = tmp_path / "kfd"
    node.write_text("")
    rc, out, _ = run_probe("open", node, "r")
    assert rc == 0, out
    assert out == f"OPEN-OK {node}"


def test_open_reports_structured_enoent_for_missing_node(tmp_path):
    rc, out, _ = run_probe("open", tmp_path / "nosuch", "r")
    assert rc == 1
    assert out == f"OPEN-ERR {tmp_path / 'nosuch'} 2 ENOENT"


@pytest.mark.skipif(IS_ROOT, reason="as root, DAC never denies - run as a regular user")
def test_open_reports_structured_eacces_for_dac_denied_read(tmp_path):
    node = tmp_path / "rootonly"
    node.write_text("")
    node.chmod(0o000)
    rc, out, _ = run_probe("open", node, "r")
    assert rc == 1
    assert out == f"OPEN-ERR {node} 13 EACCES", "DAC denial must surface as EACCES(13), never as a string match"


@pytest.mark.skipif(IS_ROOT, reason="as root, DAC never denies - run as a regular user")
def test_open_write_mode_distinguishes_direction(tmp_path):
    node = tmp_path / "writeonly"
    node.write_text("")
    node.chmod(0o200)
    rc, out, _ = run_probe("open", node, "w")
    assert rc == 0 and out == f"OPEN-OK {node}"
    rc, out, _ = run_probe("open", node, "r")
    assert rc == 1 and out == f"OPEN-ERR {node} 13 EACCES"


def test_open_rw_mode_is_a_real_rdwr_open(tmp_path):
    node = tmp_path / "rdwr"
    node.write_text("")
    node.chmod(0o600)
    rc, out, _ = run_probe("open", node, "rw")
    assert rc == 0 and out == f"OPEN-OK {node}"


def test_errno_name_vocabulary():
    """The exact names the migrate script's tiers dispatch on."""
    for num, name in ((1, "EPERM"), (13, "EACCES"), (2, "ENOENT"),
                      (6, "ENXIO"), (16, "EBUSY"), (19, "ENODEV")):
        rc, out, _ = run_probe("errno-name", num)
        assert rc == 0, out
        assert out == name


def test_errno_name_rejects_unknown_numbers():
    rc, _, err = run_probe("errno-name", 99999)
    assert rc == 2
    assert err


def test_usage_errors_exit_2():
    assert run_probe()[0] == 2
    assert run_probe("open")[0] == 2
    assert run_probe("open", "/dev/null", "x")[0] == 2  # bad mode chars


def test_output_is_one_machine_parseable_line(tmp_path):
    node = tmp_path / "n"
    node.write_text("")
    rc, out, _ = run_probe("open", node, "r")
    assert rc == 0
    assert len(out.splitlines()) == 1
    assert out.startswith("OPEN-OK ")
    rc, out, _ = run_probe("open", tmp_path / "gone", "r")
    assert len(out.splitlines()) == 1
    # OPEN-ERR <path> <int-errno> <ERRNAME> - exactly 4 fields for abs paths
    assert len(out.split()) == 4
    assert out.split()[2].isdigit()
