#!/bin/bash
# ============================================================
# OVERNIGHT RESEARCH — runs unattended on GPU cluster
#
# Experiments:
#   1. A/B Calendar: retrain without calendar → compare
#   2. Multi-horizon: LGB v6 on target_ret_4h
#   3. Multi-horizon: LGB v6 on target_ret_24h
#   4. Ensemble sims: 4-grp baseline vs 5-grp (+4h) vs 5-grp (+24h)
#   5. Correlation/IC analysis of all candidates
#
# Total time estimate: ~3-4 hours on GPU cluster
# ============================================================
set -euo pipefail

TRAIN_END="2026-02-01"
VAL_END="2026-03-07"

SIM_BASE="python run_fast_sim.py --data data/features/crypto_features_1h.parquet \
  --days 120 --start-date 2026-02-09 --end-date 2026-03-07 \
  --leverage 3 --kelly 0.8 --ensemble --edge-boost \
  --no-deriv-gate --no-ddstop"

RESULTS_DIR="overnight_results"
mkdir -p $RESULTS_DIR

# Safety trap — restore Gen#3 models on any failure
cleanup() {
  echo ""
  echo "⚠️  Script interrupted or failed! Restoring Gen#3 models..."
  for d in results_v6_prod results_v7_prod results_catboost_prod results_xgboost_prod; do
    [ -d "${d}_gen3_backup" ] && rm -rf "$d" && mv "${d}_gen3_backup" "$d" && echo "   restored $d"
  done
  # Restore any hidden dirs
  [ -d results_mlp_prod_bak_overnight ] && mv results_mlp_prod_bak_overnight results_mlp_prod
  [ -d results_v6_4h_prod_bak ] && mv results_v6_4h_prod_bak results_v6_4h_prod
  [ -d results_v6_24h_prod_bak ] && mv results_v6_24h_prod_bak results_v6_24h_prod
  unset SKIP_CALENDAR 2>/dev/null
  echo "   cleanup done."
}
trap cleanup ERR INT TERM

echo "============================================================"
echo "  OVERNIGHT RESEARCH — $(date)"
echo "  Train→${TRAIN_END}, Val→${VAL_END}"
echo "============================================================"

# ============================================================
# EXPERIMENT 1: A/B Calendar test
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  EXP 1: A/B Calendar Features Impact"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Back up Gen#3 (with calendar)
echo "📦 Backing up Gen#3 models..."
for d in results_v6_prod results_v7_prod results_catboost_prod results_xgboost_prod; do
  [ -d "$d" ] && cp -r "$d" "${d}_gen3_backup"
done

# Retrain WITHOUT calendar
echo ""
echo "🔧 Retraining 4 GBDT WITHOUT calendar (SKIP_CALENDAR=1)..."
export SKIP_CALENDAR=1

echo "━━━ v6 LGB (no-cal) ━━━"
python run_pipeline_v6.py --production --train-end $TRAIN_END --val-end $VAL_END \
  2>&1 | tee $RESULTS_DIR/train_v6_nocal.txt | tail -20

echo "━━━ v7 LGB (no-cal) ━━━"
python run_pipeline_v7.py --production --train-end $TRAIN_END --val-end $VAL_END \
  2>&1 | tee $RESULTS_DIR/train_v7_nocal.txt | tail -20

echo "━━━ CatBoost (no-cal) ━━━"
python run_pipeline_catboost.py --production --train-end $TRAIN_END --val-end $VAL_END \
  2>&1 | tee $RESULTS_DIR/train_cb_nocal.txt | tail -20

echo "━━━ XGBoost (no-cal) ━━━"
python run_pipeline_xgboost.py --production --train-end $TRAIN_END --val-end $VAL_END \
  2>&1 | tee $RESULTS_DIR/train_xgb_nocal.txt | tail -20

unset SKIP_CALENDAR

# Sim WITHOUT calendar (4 GBDT)
echo ""
echo "📊 Sim: 4-grp GBDT WITHOUT calendar..."
[ -d results_mlp_prod ] && mv results_mlp_prod results_mlp_prod_bak_overnight

export SKIP_CALENDAR=1
eval $SIM_BASE 2>&1 | tee $RESULTS_DIR/sim_no_calendar.txt
unset SKIP_CALENDAR

cp trading_logs/fast_sim_equity.csv $RESULTS_DIR/equity_no_calendar.csv
[ -d results_mlp_prod_bak_overnight ] && mv results_mlp_prod_bak_overnight results_mlp_prod

# Restore Gen#3 models
echo "📦 Restoring Gen#3 models..."
for d in results_v6_prod results_v7_prod results_catboost_prod results_xgboost_prod; do
  [ -d "${d}_gen3_backup" ] && rm -rf "$d" && mv "${d}_gen3_backup" "$d"
done

# Sim WITH calendar (4 GBDT) — gen#3
echo ""
echo "📊 Sim: 4-grp GBDT WITH calendar (gen#3)..."
[ -d results_mlp_prod ] && mv results_mlp_prod results_mlp_prod_bak_overnight
eval $SIM_BASE 2>&1 | tee $RESULTS_DIR/sim_with_calendar.txt
cp trading_logs/fast_sim_equity.csv $RESULTS_DIR/equity_with_calendar.csv
[ -d results_mlp_prod_bak_overnight ] && mv results_mlp_prod_bak_overnight results_mlp_prod

echo ""
echo "✅ EXP 1 done."

# ============================================================
# EXPERIMENT 2: Multi-horizon LGB (4h target)
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  EXP 2: LGB v6 with target_ret_4h"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "🔧 Training LGB v6 on 4h horizon..."
python run_pipeline_v6.py --production --train-end $TRAIN_END --val-end $VAL_END \
  --horizon 4 --results results_v6_4h_prod \
  2>&1 | tee $RESULTS_DIR/train_v6_4h.txt

echo ""
echo "✅ EXP 2 training done. Models → results_v6_4h_prod/"

# ============================================================
# EXPERIMENT 3: Multi-horizon LGB (24h target)
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  EXP 3: LGB v6 with target_ret_24h"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "🔧 Training LGB v6 on 24h horizon..."
python run_pipeline_v6.py --production --train-end $TRAIN_END --val-end $VAL_END \
  --horizon 24 --results results_v6_24h_prod \
  2>&1 | tee $RESULTS_DIR/train_v6_24h.txt

echo ""
echo "✅ EXP 3 training done. Models → results_v6_24h_prod/"

# ============================================================
# EXPERIMENT 4: Ensemble Sims
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  EXP 4: Ensemble Backtests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 4a. Baseline: 4-grp gen#3 (already have it from EXP 1)
# Copy from EXP 1 result
cp $RESULTS_DIR/sim_with_calendar.txt $RESULTS_DIR/sim_baseline_4grp.txt
cp $RESULTS_DIR/equity_with_calendar.csv $RESULTS_DIR/equity_baseline_4grp.csv

# 4b. 5-grp: baseline + 4h model ONLY
echo "📊 Sim: 5-grp (4 GBDT + LGB_4h)..."
[ -d results_v6_24h_prod ] && mv results_v6_24h_prod results_v6_24h_prod_bak
[ -d results_mlp_prod ] && mv results_mlp_prod results_mlp_prod_bak_overnight
eval $SIM_BASE 2>&1 | tee $RESULTS_DIR/sim_5grp_with_4h.txt
cp trading_logs/fast_sim_equity.csv $RESULTS_DIR/equity_5grp_with_4h.csv
[ -d results_v6_24h_prod_bak ] && mv results_v6_24h_prod_bak results_v6_24h_prod
[ -d results_mlp_prod_bak_overnight ] && mv results_mlp_prod_bak_overnight results_mlp_prod

# 4c. 5-grp: baseline + 24h model  
echo "📊 Sim: 5-grp (4 GBDT + LGB_24h)..."
# Temporarily hide 4h model, keep only 24h
[ -d results_v6_4h_prod ] && mv results_v6_4h_prod results_v6_4h_prod_bak
[ -d results_mlp_prod ] && mv results_mlp_prod results_mlp_prod_bak_overnight
eval $SIM_BASE 2>&1 | tee $RESULTS_DIR/sim_5grp_with_24h.txt
cp trading_logs/fast_sim_equity.csv $RESULTS_DIR/equity_5grp_with_24h.csv
[ -d results_v6_4h_prod_bak ] && mv results_v6_4h_prod_bak results_v6_4h_prod
[ -d results_mlp_prod_bak_overnight ] && mv results_mlp_prod_bak_overnight results_mlp_prod

# 4d. 6-grp: baseline + BOTH 4h and 24h
echo "📊 Sim: 6-grp (4 GBDT + LGB_4h + LGB_24h)..."
[ -d results_mlp_prod ] && mv results_mlp_prod results_mlp_prod_bak_overnight
eval $SIM_BASE 2>&1 | tee $RESULTS_DIR/sim_6grp_4h_24h.txt
cp trading_logs/fast_sim_equity.csv $RESULTS_DIR/equity_6grp_4h_24h.csv
[ -d results_mlp_prod_bak_overnight ] && mv results_mlp_prod_bak_overnight results_mlp_prod

echo ""
echo "✅ EXP 4 done."

# ============================================================
# EXPERIMENT 5: Correlation / IC analysis
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  EXP 5: Cross-model Correlation & IC Analysis"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python _analyze_multi_horizon.py 2>&1 | tee $RESULTS_DIR/analysis_multi_horizon.txt

echo ""
echo "✅ EXP 5 done."

# ============================================================
# FINAL SUMMARY
# ============================================================
echo ""
echo "============================================================"
echo "  FINAL SUMMARY — $(date)"
echo "============================================================"
echo ""
echo "--- EXP 1: Calendar A/B ---"
echo "WITHOUT calendar:"
grep -E "Return:|Sharpe:" $RESULTS_DIR/sim_no_calendar.txt | head -3
echo "WITH calendar:"
grep -E "Return:|Sharpe:" $RESULTS_DIR/sim_with_calendar.txt | head -3

echo ""
echo "--- EXP 4: Ensemble Comparison ---"
echo "4-grp baseline (gen#3):"
grep -E "Return:|Sharpe:|Max DD:|PF:" $RESULTS_DIR/sim_baseline_4grp.txt | head -4
echo ""
echo "5-grp (+LGB_4h):"
grep -E "Return:|Sharpe:|Max DD:|PF:" $RESULTS_DIR/sim_5grp_with_4h.txt | head -4
echo ""
echo "5-grp (+LGB_24h):"
grep -E "Return:|Sharpe:|Max DD:|PF:" $RESULTS_DIR/sim_5grp_with_24h.txt | head -4
echo ""
echo "6-grp (+4h+24h):"
grep -E "Return:|Sharpe:|Max DD:|PF:" $RESULTS_DIR/sim_6grp_4h_24h.txt | head -4

echo ""
echo "--- EXP 5: Correlation/IC (see full output) ---"
tail -30 $RESULTS_DIR/analysis_multi_horizon.txt

echo ""
echo "============================================================"
echo "  All results saved to: $RESULTS_DIR/"
echo "  $(date)"
echo "============================================================"
