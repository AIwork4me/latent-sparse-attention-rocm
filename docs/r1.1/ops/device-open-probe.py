#!/usr/bin/env python3
"""device-open-probe.py — structured device-node open() probe (PR #6 last-mile hardening).

Replaces strerror STRING matching in migrate-runner.sh §7b. On Linux the two
permission denials have different errnos and DIFFERENT strings:

    EPERM  (1)  "Operation not permitted"  — the cgroup (eBPF) device filter's
                                            signature: this is what a DevicePolicy
                                            denial looks like inside the sandbox;
    EACCES (13) "Permission denied"        — DAC (mode bits / ACL) rejection.

A string match on "permission denied" therefore (a) misclassifies a genuine
cgroup-filter denial (EPERM) as an unexpected error and (b) accepts a plain DAC
rejection as "filter active" evidence. This probe does os.open() + OSError.errno
and prints exactly one machine-parseable line:

    open <path> [r|w|rw]   ->  "OPEN-OK <path>"            exit 0
                               "OPEN-ERR <path> <errno> <ERRNAME>"  exit 1
    errno-name <n>         ->  "<ERRNAME>" (e.g. EPERM)    exit 0 / 2 if unknown
    anything else          ->  usage message on stderr     exit 2

The migrate script runs it BOTH inside the transient service context (via the
migrated venv python — real open(2) under the unit's DevicePolicy) and on the
host under `setpriv --reuid lsa-runner` (outside any device filter) to evidence
the strong negative control's premise: DAC alone would permit the open.
Tested for real (no mocks) by tests/test_r11_device_open_probe.py on CPU CI.
"""

import errno
import os
import sys

USAGE = (
    "usage: device-open-probe.py open <path> [r|w|rw]\n"
    "       device-open-probe.py errno-name <int>\n"
    "output: OPEN-OK <path> | OPEN-ERR <path> <errno> <ERRNAME>"
)


def cmd_open(path: str, mode: str) -> int:
    flags = getattr(os, "O_CLOEXEC", 0)
    if mode == "rw":
        flags |= os.O_RDWR
    elif mode == "r":
        flags |= os.O_RDONLY
    elif mode == "w":
        flags |= os.O_WRONLY
    else:
        print(f"SYNTAX-ABORT: mode must be r, w or rw (got {mode!r})\n{USAGE}", file=sys.stderr)
        return 2
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        name = errno.errorcode.get(exc.errno, f"ERRNO_{exc.errno}")
        print(f"OPEN-ERR {path} {exc.errno} {name}")
        return 1
    os.close(fd)
    print(f"OPEN-OK {path}")
    return 0


def cmd_errno_name(num_text: str) -> int:
    try:
        num = int(num_text)
    except ValueError:
        print(f"SYNTAX-ABORT: errno-name needs an integer (got {num_text!r})\n{USAGE}", file=sys.stderr)
        return 2
    name = errno.errorcode.get(num)
    if name is None:
        print(f"SYNTAX-ABORT: no symbolic name for errno {num}", file=sys.stderr)
        return 2
    print(name)
    return 0


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "open":
        if len(argv) == 3:
            return cmd_open(argv[2], "r")
        if len(argv) == 4:
            return cmd_open(argv[2], argv[3])
    if len(argv) == 3 and argv[1] == "errno-name":
        return cmd_errno_name(argv[2])
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
