# RESUME - exact next runnable step

Updated: 2026-11-08 ~22:10 +08:00

## Environment

```bash
export WT="$HOME/lsa-v1-overnight"
export ROCM_PYTHON="$HOME/Desktop/lsa-rocm-eval-starter/.venv/bin/python"
cd "$WT" && git branch --show-current   # exp/gfx1151-v1-night-20261108-2155
```

## Exact next command (if resuming after interruption)

```bash
cd "$HOME/lsa-v1-overnight"
PYTHONPATH=src "$ROCM_PYTHON" -m pytest tests/ -q 2>&1 | tail -5
# then 17-gate manifest x {default, batched, fused}, see PLAN.md M1
```

## Before ANY heavy GPU step

```bash
fuser -v /dev/kfd 2>&1   # must be empty; if busy, do CPU/docs work instead
free -h | head -2        # keep >=6 GiB available for OS
```
