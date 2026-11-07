#!/usr/bin/env bash
# Three INDEPENDENT matched benchmark repetitions for the strategic decode shapes.
#
# Each repetition is a fresh adapter process (own gates run, own receipt, own run
# id) at the same seed and shapes, measuring run-to-run variance of the headline
# claims - never a repackaging of one run's rows. Strategic shapes: decode
# select+read at 8k/16k/32k (the contexts the headline speedups cite).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
CONTEXTS="${LSA_REPETITION_CONTEXTS:-8192,16384,32768}"
PHASES="${LSA_REPETITION_PHASES:-decode}"
REPS="${LSA_REPETITION_REPS:-20}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ROOT/results/repetitions-$STAMP"
mkdir -p "$OUT"
echo "repetitions -> $OUT (contexts $CONTEXTS, phases $PHASES, reps $REPS)"

rc_all=0
for i in 1 2 3; do
  run_id="rep$i-$STAMP"
  echo "=== repetition $i ($run_id) ==="
  LSA_RUN_ID="$run_id" \
  LSA_STAGE_RECEIPT="$OUT/receipt-$i.json" \
  LSA_UPSTREAM_SHA="$(cat "$ROOT/third_party/upstream.sha")" \
  LSA_UPSTREAM="$ROOT/third_party/latent-sparse-attention" \
    "$PY" -m lsa_rocm_lab.kernel_adapter \
      --contexts "$CONTEXTS" --phases "$PHASES" --reps "$REPS" \
      > "$OUT/run-$i.log" 2>&1
  rc=$?
  echo "repetition $i rc=$rc"
  [ "$rc" -ne 0 ] && rc_all=1
done

"$PY" "$ROOT/scripts/aggregate_repetitions.py" "$OUT"
agg_rc=$?
echo "aggregate rc=$agg_rc"
exit $(( rc_all || agg_rc ))
