#!/bin/bash
# ══════════════════════════════════════════════════════════════════
#  Model Stack Benchmark — find the best production configuration
#  Runs 60d backtests for all model combos, parses Sharpe/Return/DD
# ══════════════════════════════════════════════════════════════════

set -euo pipefail
cd "$(dirname "$0")"

PYTHON=".venv/bin/python"
SIM="run_fast_sim.py"
DAYS=60
COMMON="--days $DAYS --capital 5000 --leverage 3 --short-blocked"
OUTDIR="benchmark_results"
mkdir -p "$OUTDIR"

# ── Configs to benchmark ─────────────────────────────────────────
# Format: "label|extra_args"
declare -a CONFIGS=(
  # Single models (no ensemble)
  "v6_solo|--model-dir results/production/lgb_v6_no_news --no-deriv-gate"
  "v7_solo|--model-dir results/production/lgb_v7_no_news --no-deriv-gate"
  "v6_solo+deriv|--model-dir results/production/lgb_v6_no_news --deriv-gate"
  "v7_solo+deriv|--model-dir results/production/lgb_v7_no_news --deriv-gate"

  # Ensembles (v6 + v7 + CatBoost)
  "ensemble_no_deriv|--ensemble --no-deriv-gate"
  "ensemble+deriv|--ensemble --deriv-gate"

  # Ensemble + meta-model (no deriv gate)
  "ensemble+meta_lgb|--ensemble --meta-model auto --meta-variant lgb_minimal --no-deriv-gate"
  "ensemble+meta_ridge|--ensemble --meta-model auto --meta-variant ridge --no-deriv-gate"

  # Ensemble + meta-model + deriv gate (full stack)
  "ensemble+deriv+meta_lgb|--ensemble --meta-model auto --meta-variant lgb_minimal --deriv-gate"
  "ensemble+deriv+meta_ridge|--ensemble --meta-model auto --meta-variant ridge --deriv-gate"
)

RESULTS_FILE="$OUTDIR/benchmark_$(date +%Y%m%d_%H%M%S).csv"
echo "config,return_pct,return_ann,max_dd,sharpe,sharpe_hac,calmar,win_rate,pf,trades,costs_pct" > "$RESULTS_FILE"

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  MODEL STACK BENCHMARK — ${#CONFIGS[@]} configs × ${DAYS}d"
echo "══════════════════════════════════════════════════════════════════"
echo ""

run_idx=0
total=${#CONFIGS[@]}

for cfg_line in "${CONFIGS[@]}"; do
  IFS='|' read -r label extra_args <<< "$cfg_line"
  run_idx=$((run_idx + 1))

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  [$run_idx/$total] $label"
  echo "  cmd: $PYTHON $SIM $COMMON $extra_args"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  LOG="$OUTDIR/${label}.log"

  # Run sim, capture output
  if $PYTHON $SIM $COMMON $extra_args > "$LOG" 2>&1; then
    # Parse results from log
    ret=$(grep -oP 'Return:\s+\K[+-]?[\d.]+' "$LOG" | head -1 || echo "N/A")
    ret_ann=$(grep -oP 'ann\. ~\K[+-]?[\d.]+' "$LOG" | head -1 || echo "N/A")
    dd=$(grep -oP 'Max DD:\s+-?\K[\d.]+' "$LOG" | head -1 || echo "N/A")
    sharpe=$(grep -oP 'Sharpe:\s+\K[+-]?[\d.]+' "$LOG" | head -1 || echo "N/A")
    sharpe_hac=$(grep -oP 'Sharpe HAC:\s+\K[+-]?[\d.]+' "$LOG" | head -1 || echo "N/A")
    calmar=$(grep -oP 'Calmar:\s+\K[+-]?[\d.]+' "$LOG" | head -1 || echo "N/A")
    wr=$(grep -oP 'Win Rate:\s+\K[\d]+' "$LOG" | head -1 || echo "N/A")
    pf=$(grep -oP 'PF:\s+\K[\d.]+' "$LOG" | head -1 || echo "N/A")
    trades=$(grep -oP 'Trades:\s+\K[\d]+' "$LOG" | head -1 || echo "N/A")
    costs=$(grep -oP 'Costs:.*\(\K[\d.]+' "$LOG" | head -1 || echo "N/A")

    echo "$label,$ret,$ret_ann,$dd,$sharpe,$sharpe_hac,$calmar,$wr,$pf,$trades,$costs" >> "$RESULTS_FILE"
    echo "  ✅ Return: ${ret}%  Sharpe: ${sharpe}  DD: -${dd}%  WR: ${wr}%  PF: ${pf}"
  else
    echo "$label,ERR,ERR,ERR,ERR,ERR,ERR,ERR,ERR,ERR,ERR" >> "$RESULTS_FILE"
    echo "  ❌ FAILED — see $LOG"
  fi
  echo ""
done

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  BENCHMARK COMPLETE — Results saved to $RESULTS_FILE"
echo "══════════════════════════════════════════════════════════════════"
echo ""

# ── Pretty table ──────────────────────────────────────────────────
echo "┌──────────────────────────────┬─────────┬──────────┬────────┬─────────┬────────┬──────┬──────┐"
echo "│ Config                       │ Return  │ Sharpe   │ Max DD │ Calmar  │ WinR%  │  PF  │Trades│"
echo "├──────────────────────────────┼─────────┼──────────┼────────┼─────────┼────────┼──────┼──────┤"

tail -n +2 "$RESULTS_FILE" | while IFS=',' read -r config ret ret_ann dd sharpe sharpe_hac calmar wr pf trades costs; do
  printf "│ %-28s │ %+5s%%  │ %+7s  │ -%4s%% │ %6s  │  %3s%%  │ %4s │ %4s │\n" \
    "$config" "$ret" "$sharpe" "$dd" "$calmar" "$wr" "$pf" "$trades"
done

echo "└──────────────────────────────┴─────────┴──────────┴────────┴─────────┴────────┴──────┴──────┘"
echo ""
echo "📊 Full CSV: $RESULTS_FILE"
echo "📝 Individual logs: $OUTDIR/<config>.log"
