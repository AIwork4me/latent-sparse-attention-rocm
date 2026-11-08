# RESUME - exact next runnable step

Updated: 2026-11-08 ~22:30 +08:00

## Environment

```bash
export WT="$HOME/lsa-v1-overnight"
export ROCM_PYTHON="$HOME/Desktop/lsa-rocm-eval-starter/.venv/bin/python"
cd "$WT" && git branch --show-current   # exp/gfx1151-v1-night-20261108-2155
```

## State

M1 PASS (see STATE.json + V040_RELEASE_GO_NOGO.md). M2 is current.

## Exact next command (if resuming after interruption)

```bash
cd "$HOME/lsa-v1-overnight"
ls tracking/overnight_v1/receipts/ | grep m2 || true   # what M2 already produced
# then continue the M2 ladder from the highest completed size in the m2 receipt,
# respecting: decode-first, memory-probe gate before each step, BLOCKED_RESOURCE on exceed
```

## Before ANY heavy GPU step

```bash
fuser -v /dev/kfd 2>&1   # must be empty; if busy, do CPU/docs work instead
free -h | head -2        # keep >=6 GiB available for OS
```

