#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Walk-Forward: Position Sizing Experiments
# ═══════════════════════════════════════════════════════════════
# Tests sizing methods × position counts across 6 non-overlapping
# 90-day windows (same windows as original walk-forward).
#
# Question: does edge-boost / softmax / inv-vol improve Sharpe?
# Question: does 7+7 beat 10+10 when combined with sizing?
#
# All configs use: ens4 (v6+v7+CB+XGB), no deriv-gate, no meta
# (the stack validated by prior walk-forward).
# ═══════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=".venv/bin/python"
SIM="run_fast_sim.py"
COMMON="--capital 5000 --leverage 3 --short-blocked --data trading_logs/frozen_features_wf.parquet --ensemble --no-deriv-gate"
OUTDIR="walkforward_sizing"
mkdir -p "$OUTDIR"

# 6 non-overlapping 90d windows (identical to run_walkforward.sh)
declare -a WINDOWS=(
  "W1|2024-09-18|2024-12-17"
  "W2|2024-12-17|2025-03-17"
  "W3|2025-03-17|2025-06-15"
  "W4|2025-06-15|2025-09-13"
  "W5|2025-09-13|2025-12-12"
  "W6|2025-12-12|2026-03-12"
)

# ── Sizing configs ────────────────────────────────────────────
# Format: name|extra_args
#
# 10+10 configs (current default for n_pos)
declare -a CONFIGS_10=(
  "baseline_10|--npos 10"
  "edge_boost_10|--npos 10 --edge-boost"
  "edge_boost_vol_10|--npos 10 --edge-boost --vol-size"
  "softmax2.5_10|--npos 10 --softmax-temp 2.5"
  "softmax4.0_10|--npos 10 --softmax-temp 4.0"
  "softmax2.5_vol_10|--npos 10 --softmax-temp 2.5 --vol-size"
)

# 7+7 configs
declare -a CONFIGS_7=(
  "baseline_7|--npos 7"
  "edge_boost_7|--npos 7 --edge-boost"
  "edge_boost_vol_7|--npos 7 --edge-boost --vol-size"
  "softmax2.5_7|--npos 7 --softmax-temp 2.5"
  "softmax4.0_7|--npos 7 --softmax-temp 4.0"
  "softmax2.5_vol_7|--npos 7 --softmax-temp 2.5 --vol-size"
)

# Combine all configs
declare -a CONFIGS=("${CONFIGS_10[@]}" "${CONFIGS_7[@]}")

N_WINDOWS=${#WINDOWS[@]}
N_CONFIGS=${#CONFIGS[@]}
TOTAL=$((N_WINDOWS * N_CONFIGS))

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  WALK-FORWARD: POSITION SIZING EXPERIMENTS"
echo "  ${N_CONFIGS} configs × ${N_WINDOWS} windows = ${TOTAL} simulations"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  10+10 configs: baseline, edge-boost, edge-boost+vol,"
echo "                 softmax(2.5), softmax(4.0), softmax(2.5)+vol"
echo "  7+7 configs:   same 6 variants"
echo ""

run_idx=0
for win_line in "${WINDOWS[@]}"; do
  IFS='|' read -r wlabel wstart wend <<< "$win_line"
  echo "━━━ ${wlabel}: ${wstart} → ${wend} ━━━"

  for cfg_line in "${CONFIGS[@]}"; do
    IFS='|' read -r clabel extra_args <<< "$cfg_line"
    run_idx=$((run_idx + 1))
    LOG="$OUTDIR/${clabel}_${wlabel}.log"

    # Skip if already completed (allows restarts)
    if [ -f "$LOG" ] && grep -q "Sharpe HAC" "$LOG" 2>/dev/null; then
      echo "  [${run_idx}/${TOTAL}] ${clabel} ... CACHED"
      continue
    fi

    echo -n "  [${run_idx}/${TOTAL}] ${clabel} ..."
    if $PYTHON $SIM --days 600 --start-date "$wstart" --end-date "$wend" $COMMON $extra_args > "$LOG" 2>&1; then
      # Extract key metric
      sharpe=$(grep 'Sharpe HAC' "$LOG" 2>/dev/null | grep -oE '[+-][0-9.]+' | head -1 || echo "?")
      echo " Sharpe=${sharpe}"
    else
      echo " FAILED"
    fi
  done
  echo ""
done

echo "═══════════════════════════════════════════════════════════"
echo "ALL ${TOTAL} SIZING SIMS COMPLETE"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Parse results
$PYTHON parse_walkforward.py "$OUTDIR"
