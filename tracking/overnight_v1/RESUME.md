# RESUME - exact next runnable step

Updated: 2026-11-09 ~00:55 +08:00

## Environment

```bash
export WT="$HOME/lsa-v1-overnight"
export ROCM_PYTHON="$HOME/Desktop/lsa-rocm-eval-starter/.venv/bin/python"
cd "$WT" && git branch --show-current   # exp/gfx1151-v1-night-20261108-2155
```

## State

M1 PASS · M2 PASS (findings) · M3 PILOT_DONE · M4 BLOCKED_WITH_ROOT_CAUSE · M6 soak
running/extended-pilots queued · M7 checklist delivered. Closeout in progress.

## Exact next commands (if resuming)

```bash
# 1. if soak receipt missing, rerun: $ROCM_PYTHON scripts/soak_gates.py --minutes 10 \
#      --out tracking/overnight_v1/receipts/m6-soak.json
# 2. extended pilots (preregistered m3-ppl-extended / m3-niah-extended):
HF_HUB_OFFLINE=1 PYTHONPATH=src:third_party/latent-sparse-attention/src \
  "$ROCM_PYTHON" scripts/m3_model_integration.py --model tinyllama --modes ppl \
  --arms dense,lsa_full,lsa_1024,lsa_256 --n-windows 48 \
  --out tracking/overnight_v1/receipts/m3-ppl-extended.json
# 3. fill OVERNIGHT_REPORT.md {{SOAK_*}} placeholders, final STATE.json, commit.
```

## Before ANY heavy GPU step

```bash
fuser -v /dev/kfd 2>&1   # must be empty; if busy, do CPU/docs work instead
free -h | head -2        # keep >=6 GiB available for OS
```



## Campaign CLOSED 2026-11-09 ~01:50 +08:00 (STOP.json written)

All workstreams terminal: M1 PASS(+fresh 3x3) / M2 PASS / M3 PILOT_DONE(+ext) /
M4 BLOCKED / M6 PASS / M7 packet NO-GO. Nothing pushed; branch awaits owner decision.
Next session should NOT rerun tonight's experiments - read OVERNIGHT_REPORT.md first.
