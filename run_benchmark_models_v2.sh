#!/bin/bash
# ══════════════════════════════════════════════════════════════════
#  Model Stack Benchmark v2 — with XGBoost, 30d + 60d
# ══════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=".venv/bin/python"
SIM="run_fast_sim.py"
COMMON="--capital 5000 --leverage 3 --short-blocked --data trading_logs/frozen_raw.parquet"
OUTDIR="benchmark_results_v2"
mkdir -p "$OUTDIR"

declare -a CONFIGS=(
  "v6_solo|--model-dir results/production/lgb_v6_no_news --no-deriv-gate --no-xgb"
  "v7_solo|--model-dir results/production/lgb_v7_no_news --no-deriv-gate --no-xgb"
  "v6+deriv|--model-dir results/production/lgb_v6_no_news --deriv-gate --no-xgb"
  "v7+deriv|--model-dir results/production/lgb_v7_no_news --deriv-gate --no-xgb"
  "ens3|--ensemble --no-deriv-gate --no-xgb"
  "ens3+deriv|--ensemble --deriv-gate --no-xgb"
  "ens3+meta_lgb|--ensemble --meta-model auto --meta-variant lgb_minimal --no-deriv-gate --no-xgb"
  "ens3+deriv+meta_lgb|--ensemble --meta-model auto --meta-variant lgb_minimal --deriv-gate --no-xgb"
  "ens4|--ensemble --no-deriv-gate"
  "ens4+deriv|--ensemble --deriv-gate"
  "ens4+meta_lgb|--ensemble --meta-model auto --meta-variant lgb_minimal --no-deriv-gate"
  "ens4+deriv+meta_lgb|--ensemble --meta-model auto --meta-variant lgb_minimal --deriv-gate"
)

echo ""
echo "MODEL STACK BENCHMARK v2 - ${#CONFIGS[@]} configs x 30d+60d"
echo ""

for DAYS in 30 60; do
  run_idx=0
  total=${#CONFIGS[@]}
  echo "--- ${DAYS}d ---"
  for cfg_line in "${CONFIGS[@]}"; do
    IFS='|' read -r label extra_args <<< "$cfg_line"
    run_idx=$((run_idx + 1))
    echo "  [$run_idx/$total] ${label} (${DAYS}d) ..."
    LOG="$OUTDIR/${label}_${DAYS}d.log"
    if $PYTHON $SIM --days $DAYS $COMMON $extra_args > "$LOG" 2>&1; then
      echo "         done"
    else
      echo "         FAILED"
    fi
  done
  echo ""
done

echo "ALL SIMS COMPLETE - parsing..."
$PYTHON parse_benchmark_v2.py "$OUTDIR"
