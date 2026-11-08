"""R1.1 device-policy validator - negative regression tests for PR #6's final correction.

The 2026-11-08 owner review of the migration runbook found two defects in step 7:

1. The transient ``systemd-run`` validation context transcribed ``DevicePolicy=closed``
   from the installed unit but NOT its ``DeviceAllow=`` entries - under ``closed``
   policy a transient service with an empty allowlist has NO device access at all,
   so the in-sandbox GPU/Triton probe could never pass (a guaranteed false abort).
2. The old step 7b compared ``ls /dev/dri`` filenames against the allowlist.
   ``DevicePolicy`` is enforced by the cgroup eBPF device filter on open(2), NOT by
   hiding nodes: a plain ``sudo -u`` (or any context where the filter is inert)
   also "sees" the same nodes, so the filename check proved nothing about access.

``docs/r1.1/ops/device-policy-check.sh`` is the extracted, unit-testable validator
core. These tests execute the REAL script (no mocks) with tampered inputs and pin
that it fails loudly for every corruption class the review called out:

- a required DeviceAllow entry missing from the live unit;
- an entry altered (path or mode) vs the unit file;
- an extra unexpected entry;
- DevicePolicy drift between unit file and live unit;
- DeviceAllow paths incompatible with the actual GPU device nodes.

Pure file-in/file-out tests: they run on GitHub-hosted CPU CI everywhere.
"""

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SERVICE = REPO / "docs/r1.1/ops/github-runner-lsa.service"
CHECK = REPO / "docs/r1.1/ops/device-policy-check.sh"

# The exact device access policy the R1.1 unit ships with (gfx1151 host):
# /dev/kfd + the two amdgpu DRI nodes, read-write.
EXPECTED_ENTRIES = [
    ("/dev/kfd", "rw"),
    ("/dev/dri/renderD128", "rw"),
    ("/dev/dri/card1", "rw"),
]


def run_check(*args):
    """Execute the real validator script; return (rc, stdout+stderr)."""
    proc = subprocess.run(
        ["bash", str(CHECK), *[str(a) for a in args]],
        capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


def make_show(entries, policy="closed", per_line=False):
    """Render a `systemctl show -p DevicePolicy -p DeviceAllow` capture.

    Observed on this host's systemd 255: DeviceAllow serializes one line per
    entry, and an EMPTY allowlist is omitted entirely (a capture with only a
    DevicePolicy line means zero devices — parity must treat that as the
    defect-1 class). The space-joined single-line form (older systemd list
    serialization, still produced for some properties) is accepted too; the
    validator must survive both.
    """
    lines = [f"DevicePolicy={policy}"]
    if per_line:
        lines += [f"DeviceAllow={path} {mode}" for path, mode in entries]
    else:
        joined = " ".join(f"{path} {mode}" for path, mode in entries)
        lines.append(f"DeviceAllow={joined}" if entries else "DeviceAllow=")
    return "\n".join(lines) + "\n"


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def unit_file_text(entries=EXPECTED_ENTRIES, policy="closed"):
    svc = "[Service]\n" + f"DevicePolicy={policy}\n"
    svc += "".join(f"DeviceAllow={path} {mode}\n" for path, mode in entries)
    return svc


# ---------------------------------------------------------------- parity ----

def test_parity_passes_when_live_show_matches_unit_file(tmp_path):
    unit = write(tmp_path, "unit", unit_file_text())
    show = write(tmp_path, "show", make_show(EXPECTED_ENTRIES))
    rc, out = run_check("parity", unit, show)
    assert rc == 0, out


def test_parity_passes_for_per_line_show_format(tmp_path):
    unit = write(tmp_path, "unit", unit_file_text())
    show = write(tmp_path, "show", make_show(EXPECTED_ENTRIES, per_line=True))
    rc, out = run_check("parity", unit, show)
    assert rc == 0, out


def test_parity_passes_for_the_shipped_unit_against_its_own_show(tmp_path):
    """Self-consistency: a faithful `systemctl show` capture of the committed
    unit must parity-pass - pins the derivation both sides agree on."""
    show = write(tmp_path, "show", make_show(EXPECTED_ENTRIES))
    rc, out = run_check("parity", SERVICE, show)
    assert rc == 0, out


def test_parity_fails_when_a_required_entry_is_missing(tmp_path):
    unit = write(tmp_path, "unit", unit_file_text())
    tampered = [e for e in EXPECTED_ENTRIES if e[0] != "/dev/kfd"]
    show = write(tmp_path, "show", make_show(tampered))
    rc, out = run_check("parity", unit, show)
    assert rc == 1, out
    assert "/dev/kfd" in out, "failure must name the missing device entry"


def test_parity_fails_when_the_allowlist_is_entirely_absent(tmp_path):
    """The original defect: live unit carries DevicePolicy=closed but no
    DeviceAllow entries - the transient context would get zero devices."""
    unit = write(tmp_path, "unit", unit_file_text())
    show = write(tmp_path, "show", make_show([]))
    rc, out = run_check("parity", unit, show)
    assert rc == 1, out


def test_parity_fails_when_an_entry_path_is_altered(tmp_path):
    unit = write(tmp_path, "unit", unit_file_text())
    altered = [("/dev/dri/card0", m) if p == "/dev/dri/card1" else (p, m)
               for p, m in EXPECTED_ENTRIES]
    show = write(tmp_path, "show", make_show(altered))
    rc, out = run_check("parity", unit, show)
    assert rc == 1, out
    assert "/dev/dri/card1" in out or "/dev/dri/card0" in out, (
        "failure must name the diverging node"
    )


def test_parity_fails_when_an_entry_mode_is_altered(tmp_path):
    unit = write(tmp_path, "unit", unit_file_text())
    altered = [(p, "rwm" if p == "/dev/kfd" else m) for p, m in EXPECTED_ENTRIES]
    show = write(tmp_path, "show", make_show(altered))
    rc, out = run_check("parity", unit, show)
    assert rc == 1, out
    assert "/dev/kfd" in out


def test_parity_fails_when_an_extra_unexpected_entry_appears(tmp_path):
    unit = write(tmp_path, "unit", unit_file_text())
    show = write(tmp_path, "show", make_show(EXPECTED_ENTRIES + [("/dev/sda", "r")]))
    rc, out = run_check("parity", unit, show)
    assert rc == 1, out
    assert "/dev/sda" in out


@pytest.mark.parametrize("live_policy", ["auto", "strict"])
def test_parity_fails_when_device_policy_drifts(tmp_path, live_policy):
    """Catches an owner drop-in or daemon-reload lag weakening the sandbox."""
    unit = write(tmp_path, "unit", unit_file_text())
    show = write(tmp_path, "show", make_show(EXPECTED_ENTRIES, policy=live_policy))
    rc, out = run_check("parity", unit, show)
    assert rc == 1, out


def test_parity_fails_loudly_when_unit_file_lacks_device_policy(tmp_path):
    """Malformed input must exit 2 with SYNTAX-ABORT — NOT die silently with
    rc 1 under pipefail (regression: the pre-review version did exactly that)."""
    unit = write(tmp_path, "unit", "[Service]\nDeviceAllow=/dev/kfd rw\n")
    show = write(tmp_path, "show", make_show(EXPECTED_ENTRIES))
    rc, out = run_check("parity", unit, show)
    assert rc == 2, f"expected rc 2, got {rc}: {out!r}"
    assert "SYNTAX-ABORT" in out


def test_parity_fails_loudly_when_show_capture_lacks_device_policy(tmp_path):
    """The exact wrong-capture case (systemctl show without -p DevicePolicy):
    loud usage error, not a silent exit."""
    unit = write(tmp_path, "unit", unit_file_text())
    show = write(tmp_path, "show", "DeviceAllow=/dev/kfd rw\n")
    rc, out = run_check("parity", unit, show)
    assert rc == 2, f"expected rc 2, got {rc}: {out!r}"
    assert "SYNTAX-ABORT" in out


def test_parity_fails_loudly_on_missing_input_file(tmp_path):
    rc, out = run_check("parity", tmp_path / "nope", tmp_path / "nada")
    assert rc == 2, f"expected rc 2, got {rc}: {out!r}"
    assert "SYNTAX-ABORT" in out


def test_parity_normalizes_mode_order_and_omitted_modes(tmp_path):
    """'wr' == 'rw', and an omitted unit-file mode defaults to rwm: parity is
    about the effective policy, not textual spelling."""
    unit = write(tmp_path, "unit", "[Service]\nDevicePolicy=closed\n"
                                "DeviceAllow=/dev/kfd wr\nDeviceAllow=/dev/dri/card1\n")
    show = write(tmp_path, "show", "DevicePolicy=closed\n"
                                   "DeviceAllow=/dev/kfd rw /dev/dri/card1 rwm\n")
    rc, out = run_check("parity", unit, show)
    assert rc == 0, out


def test_emit_entries_refuses_an_empty_allowlist(tmp_path):
    """Emitting nothing would silently recreate the defect-1 zero-device
    transient context — must be a loud usage error."""
    show = write(tmp_path, "show", "DevicePolicy=closed\nDeviceAllow=\n")
    rc, out = run_check("emit-entries", show)
    assert rc == 2, f"expected rc 2, got {rc}: {out!r}"


# ---------------------------------------------------------- emit-entries ----

def test_emit_entries_reconstructs_path_mode_pairs(tmp_path):
    show = write(tmp_path, "show", make_show(EXPECTED_ENTRIES))
    rc, out = run_check("emit-entries", show)
    assert rc == 0, out
    emitted = sorted(l for l in out.splitlines() if l.strip())
    assert emitted == sorted(f"{p} {m}" for p, m in EXPECTED_ENTRIES)


def test_emit_entries_accepts_per_line_show_format(tmp_path):
    show = write(tmp_path, "show", make_show(EXPECTED_ENTRIES, per_line=True))
    rc, out = run_check("emit-entries", show)
    assert rc == 0, out
    emitted = sorted(l for l in out.splitlines() if l.strip())
    assert emitted == sorted(f"{p} {m}" for p, m in EXPECTED_ENTRIES)


def test_emit_entries_fails_loudly_on_a_dangling_mode_token(tmp_path):
    show = write(tmp_path, "show", "DevicePolicy=closed\nDeviceAllow= rw\n")
    rc, out = run_check("emit-entries", show)
    assert rc == 2, out


def test_emit_entries_fails_loudly_on_unknown_mode_chars(tmp_path):
    show = write(tmp_path, "show", "DevicePolicy=closed\nDeviceAllow=/dev/kfd rx\n")
    rc, out = run_check("emit-entries", show)
    assert rc == 2, out


# ------------------------------------------------------------------ nodes ----

def test_nodes_passes_when_every_allowed_path_exists(tmp_path):
    devroot = tmp_path / "dev"
    for sub in ("kfd", "dri/renderD128", "dri/card1"):
        node = devroot / sub
        node.parent.mkdir(parents=True, exist_ok=True)
        node.touch()
    unit = write(tmp_path, "unit", unit_file_text())
    rc, out = run_check("nodes", unit, "--dev-root", devroot)
    assert rc == 0, out


def test_nodes_fails_when_a_gpu_node_is_incompatible_with_the_host(tmp_path):
    """The unit pins card1 but the host renumbered to card0 after a reboot:
    the validator must abort naming the node that no longer exists."""
    devroot = tmp_path / "dev"
    for sub in ("kfd", "dri/renderD128", "dri/card0"):  # host has card0, unit wants card1
        node = devroot / sub
        node.parent.mkdir(parents=True, exist_ok=True)
        node.touch()
    unit = write(tmp_path, "unit", unit_file_text())
    rc, out = run_check("nodes", unit, "--dev-root", devroot)
    assert rc == 1, out
    assert "/dev/dri/card1" in out


def test_nodes_fails_when_kfd_is_missing(tmp_path):
    devroot = tmp_path / "dev"
    for sub in ("dri/renderD128", "dri/card1"):
        node = devroot / sub
        node.parent.mkdir(parents=True, exist_ok=True)
        node.touch()
    unit = write(tmp_path, "unit", unit_file_text())
    rc, out = run_check("nodes", unit, "--dev-root", devroot)
    assert rc == 1, out
    assert "/dev/kfd" in out


def test_nodes_skips_char_block_group_specifiers(tmp_path):
    """char-*/block-* globs allow device GROUPS, not paths - there is no node
    to stat; the existence check must not choke on them."""
    devroot = tmp_path / "dev"
    devroot.mkdir(parents=True)
    unit = write(tmp_path, "unit", unit_file_text(entries=[("char-drm", "rw")]))
    rc, out = run_check("nodes", unit, "--dev-root", devroot)
    assert rc == 0, out
