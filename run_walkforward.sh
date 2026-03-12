#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Walk-Forward Validation — 6 windows × key configs
# ═══════════════════════════════════════════════════════════════
# Tests model stability across different market regimes.
# Each window is 90 days, non-overlapping, covering 2024-09 → 2026-03.
#
# Configs chosen to answer key questions:
#   1. ens4 vs v7_solo — is ensemble actually more stable?
#   2. deriv-gate — does it ever help?
#   3. meta ridge vs meta lgb vs no meta — which is stable?
# ═══════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=".venv/bin/python"
SIM="run_fast_sim.py"
COMMON="--capital 5000 --leverage 3 --short-blocked --data trading_logs/frozen_features_wf.parquet"
OUTDIR="walkforward_results"
mkdir -p "$OUTDIR"

# 6 non-overlapping 90d windows
declare -a WINDOWS=(
  "W1|2024-09-18|2024-12-17"
  "W2|2024-12-17|2025-03-17"
  "W3|2025-03-17|2025-06-15"
  "W4|2025-06-15|2025-09-13"
  "W5|2025-09-13|2025-12-12"
  "W6|2025-12-12|2026-03-12"
)

# Key configs to compare
declare -a CONFIGS=(
  "v7_solo|--model-dir results/production/lgb_v7_no_news --no-deriv-gate --no-xgb"
  "ens3|--ensemble --no-deriv-gate --no-xgb"
  "ens4|--ensemble --no-deriv-gate"
  "ens4+deriv|--ensemble --deriv-gate"
  "ens4+meta_lgb|--ensemble --meta-model auto --meta-variant lgb_minimal --no-deriv-gate"
  "ens4+meta_ridge|--ensemble --meta-model auto --meta-variant ridge --no-deriv-gate"
)

N_WINDOWS=${#WINDOWS[@]}
N_CONFIGS=${#CONFIGS[@]}
TOTAL=$((N_WINDOWS * N_CONFIGS))

echo ""
echo "WALK-FORWARD VALIDATION"
echo "  ${N_CONFIGS} configs × ${N_WINDOWS} windows = ${TOTAL} simulations"
echo "  Windows: 90d each, non-overlapping"
echo ""

run_idx=0
for win_line in "${WINDOWS[@]}"; do
  IFS='|' read -r wlabel wstart wend <<< "$win_line"
  echo "━━━ ${wlabel}: ${wstart} → ${wend} ━━━"

  for cfg_line in "${CONFIGS[@]}"; do
    IFS='|' read -r clabel extra_args <<< "$cfg_line"
    run_idx=$((run_idx + 1))
    echo "  [${run_idx}/${TOTAL}] ${clabel} ..."
    LOG="$OUTDIR/${clabel}_${wlabel}.log"
    if $PYTHON $SIM --days 600 --start-date "$wstart" --end-date "$wend" $COMMON $extra_args > "$LOG" 2>&1; then
      echo "         done"
    else
      echo "         FAILED"
    fi
  done
  echo ""
done

echo "ALL WALK-FORWARD SIMS COMPLETE — parsing..."
$PYTHON parse_walkforward.py "$OUTDIR"
