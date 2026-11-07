#!/usr/bin/env bash
# T1 multi-seed decode study (issue #2): >=3 distinct seeds x 3 INDEPENDENT
# fresh adapter processes each, decode select+read at 8k/16k/32k bf16.
#
# Every repetition is a fresh process running its own 16 correctness gates and
# writing its own receipt (same contract as repeat_bench.sh); the only change
# is the seed, so cross-seed spread reflects input sensitivity, not process
# noise. Run-id mapping (auditor shape rep<N>-<ts>Z, dedup-checked): seed i
# (0-based, in $SEEDS order), repetition j (1..3) -> rep$((i*3+j))-$STAMP.
# Seeds default to 20261105 (the v0.2.1 audited seed) plus two fresh ones.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
SEEDS="${LSA_SEEDS:-20261105,777001,424243}"
CONTEXTS="${LSA_SEED_CONTEXTS:-8192,16384,32768}"
REPS="${LSA_REPS:-20}"
WARMUP="${LSA_WARMUP:-5}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ROOT/results/seed-study-$STAMP"
mkdir -p "$OUT"
echo "seed study -> $OUT (seeds $SEEDS, contexts $CONTEXTS, reps $REPS, decode bf16)"

rc_all=0
i=0
for seed in ${SEEDS//,/ }; do
  i=$((i + 1))
  for j in 1 2 3; do
    run_id="rep$(( (i-1)*3 + j ))-$STAMP"
    echo "=== seed $seed repetition $j ($run_id) ==="
    LSA_RUN_ID="$run_id" \
    LSA_STAGE_RECEIPT="$OUT/receipt-$run_id.json" \
    LSA_UPSTREAM_SHA="$(cat "$ROOT/third_party/upstream.sha")" \
    LSA_UPSTREAM="$ROOT/third_party/latent-sparse-attention" \
      "$PY" -m lsa_rocm_lab.kernel_adapter \
        --seed "$seed" --contexts "$CONTEXTS" --phases decode \
        --reps "$REPS" --warmup "$WARMUP" \
        > "$OUT/run-$run_id.log" 2>&1
    rc=$?
    echo "seed $seed rep $j ($run_id) rc=$rc"
    [ "$rc" -ne 0 ] && rc_all=1
  done
done

"$PY" "$ROOT/scripts/aggregate_seed_study.py" "$OUT"
agg_rc=$?
echo "aggregate rc=$agg_rc"
exit $(( rc_all || agg_rc ))
