#!/bin/bash
# ============================================================
# OVERNIGHT RESEARCH v3 — Portfolio Construction + Robust Loss
#
# Three experiments:
#   EXP A: Hysteresis + Turnover Budget (sim-only, grid search)
#   EXP B: Dynamic N + Min Z-score (sim-only, grid search)
#   EXP C: Huber Loss + Dead-zone Weighting (retrain + sim)
#
# No model changes in A+B — pure portfolio construction.
# C retrains v6+v7 with robust loss.
#
# Baseline: Gen#3 no-calendar 5-group (v6+v7+cb+xgb+24h)
# Total time estimate: ~1-2 hours (A+B are fast, C needs retraining)
# ============================================================
set -euo pipefail

TRAIN_END="2026-02-01"
VAL_END="2026-03-07"

# Base sim command — same as production config
SIM_BASE="python run_fast_sim.py --data data/features/crypto_features_1h.parquet \
  --days 120 --start-date 2026-02-09 --end-date 2026-03-07 \
  --leverage 3 --kelly 0.8 --ensemble --edge-boost \
  --no-deriv-gate --no-ddstop"

RESULTS_DIR="overnight_v3_results"
mkdir -p $RESULTS_DIR

export SKIP_CALENDAR=1

echo "============================================================"
echo "  OVERNIGHT RESEARCH v3 — Portfolio + Loss Experiments"
echo "  Started: $(date)"
echo "============================================================"

# ============================================================
# BASELINE: current production sim (for comparison)
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  BASELINE: Current config (no changes)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$SIM_BASE 2>&1 | tee $RESULTS_DIR/baseline.log

# ============================================================
# EXPERIMENT A: Hysteresis + Turnover Budget
# Grid search position stickiness parameters
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  EXP A: Hysteresis + Turnover Budget"
echo "  Hypothesis: reduce churn → lower costs → higher net Sharpe"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# A1: Hysteresis only (keep until rank > N+H)
for H in 3 5 7 10; do
  echo ""
  echo "📊 A1: hysteresis=$H ..."
  $SIM_BASE --hysteresis $H \
    2>&1 | tee $RESULTS_DIR/exp_a1_hyst_${H}.log
done

# A2: Turnover budget only (max replacements per side)
for TB in 2 3 5; do
  echo ""
  echo "📊 A2: turnover-budget=$TB ..."
  $SIM_BASE --turnover-budget $TB \
    2>&1 | tee $RESULTS_DIR/exp_a2_tb_${TB}.log
done

# A3: Best hysteresis + turnover budget combo
for H in 3 5; do
  for TB in 3 5; do
    echo ""
    echo "📊 A3: hysteresis=$H + turnover-budget=$TB ..."
    $SIM_BASE --hysteresis $H --turnover-budget $TB \
      2>&1 | tee $RESULTS_DIR/exp_a3_hyst${H}_tb${TB}.log
  done
done

echo "✅ EXP A done."

# ============================================================
# EXPERIMENT B: Dynamic N + Min Z-score
# Vary positions based on signal conviction
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  EXP B: Dynamic N + Min Z-score"
echo "  Hypothesis: avoid weak signals → higher WR, better PF"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# B1: Min z-score filter (skip weak signals)
for MZ in 0.3 0.5 0.7 1.0; do
  echo ""
  echo "📊 B1: min-zscore=$MZ ..."
  $SIM_BASE --min-zscore $MZ \
    2>&1 | tee $RESULTS_DIR/exp_b1_mz_${MZ}.log
done

# B2: Dynamic N (auto-adjust based on dispersion)
echo ""
echo "📊 B2: dynamic-n ..."
$SIM_BASE --dynamic-n \
  2>&1 | tee $RESULTS_DIR/exp_b2_dynN.log

# B3: Dynamic N + min z-score
for MZ in 0.3 0.5; do
  echo ""
  echo "📊 B3: dynamic-n + min-zscore=$MZ ..."
  $SIM_BASE --dynamic-n --min-zscore $MZ \
    2>&1 | tee $RESULTS_DIR/exp_b3_dynN_mz${MZ}.log
done

# B4: Best of A + best of B (if we had to guess)
echo ""
echo "📊 B4: hysteresis=5 + min-zscore=0.5 ..."
$SIM_BASE --hysteresis 5 --min-zscore 0.5 \
  2>&1 | tee $RESULTS_DIR/exp_b4_hyst5_mz05.log

echo ""
echo "📊 B5: hysteresis=5 + turnover-budget=3 + min-zscore=0.5 ..."
$SIM_BASE --hysteresis 5 --turnover-budget 3 --min-zscore 0.5 \
  2>&1 | tee $RESULTS_DIR/exp_b5_full_combo.log

echo "✅ EXP B done."

# ============================================================
# EXPERIMENT C: Huber Loss + Dead-zone Weighting (RETRAIN)
# Requires retraining v6+v7, then running sim
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  EXP C: Huber Loss + Dead-zone Weighting"
echo "  Hypothesis: robust loss → less gradient domination by tails"
echo "  ⚠️  Retraining v6+v7 (existing models backed up)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# C1: Huber loss only
echo ""
echo "🏋️ C1a: Training v6 Huber..."
python run_pipeline_v6.py \
  --production --train-end $TRAIN_END --val-end $VAL_END \
  --huber \
  --results results_v6_huber_prod \
  2>&1 | tee $RESULTS_DIR/exp_c1a_v6_huber_train.log

echo ""
echo "🏋️ C1b: Training v7 Huber..."
python run_pipeline_v7.py \
  --production --train-end $TRAIN_END --val-end $VAL_END \
  --huber \
  --results results_v7_huber_prod \
  2>&1 | tee $RESULTS_DIR/exp_c1b_v7_huber_train.log

# Sim with Huber models
echo ""
echo "📊 C1c: Sim with Huber v6+v7..."
cp -r results_v6_prod results_v6_prod_bak_v3
cp -r results_v7_prod results_v7_prod_bak_v3
cp -r results_v6_huber_prod/* results_v6_prod/
cp -r results_v7_huber_prod/* results_v7_prod/

$SIM_BASE 2>&1 | tee $RESULTS_DIR/exp_c1c_sim_huber.log

# Restore
rm -rf results_v6_prod results_v7_prod
mv results_v6_prod_bak_v3 results_v6_prod
mv results_v7_prod_bak_v3 results_v7_prod

# C2: Dead-zone weighting (τ=0.3%)
echo ""
echo "🏋️ C2a: Training v6 deadzone=0.3%..."
python run_pipeline_v6.py \
  --production --train-end $TRAIN_END --val-end $VAL_END \
  --deadzone-weight 0.3 \
  --results results_v6_dz03_prod \
  2>&1 | tee $RESULTS_DIR/exp_c2a_v6_dz03_train.log

echo ""
echo "🏋️ C2b: Training v7 deadzone=0.3%..."
python run_pipeline_v7.py \
  --production --train-end $TRAIN_END --val-end $VAL_END \
  --deadzone-weight 0.3 \
  --results results_v7_dz03_prod \
  2>&1 | tee $RESULTS_DIR/exp_c2b_v7_dz03_train.log

echo ""
echo "📊 C2c: Sim with deadzone v6+v7..."
cp -r results_v6_prod results_v6_prod_bak_v3
cp -r results_v7_prod results_v7_prod_bak_v3
cp -r results_v6_dz03_prod/* results_v6_prod/
cp -r results_v7_dz03_prod/* results_v7_prod/

$SIM_BASE 2>&1 | tee $RESULTS_DIR/exp_c2c_sim_dz03.log

rm -rf results_v6_prod results_v7_prod
mv results_v6_prod_bak_v3 results_v6_prod
mv results_v7_prod_bak_v3 results_v7_prod

# C3: Huber + dead-zone combo
echo ""
echo "🏋️ C3a: Training v6 Huber + deadzone=0.3%..."
python run_pipeline_v6.py \
  --production --train-end $TRAIN_END --val-end $VAL_END \
  --huber --deadzone-weight 0.3 \
  --results results_v6_huber_dz_prod \
  2>&1 | tee $RESULTS_DIR/exp_c3a_v6_huber_dz_train.log

echo ""
echo "🏋️ C3b: Training v7 Huber + deadzone=0.3%..."
python run_pipeline_v7.py \
  --production --train-end $TRAIN_END --val-end $VAL_END \
  --huber --deadzone-weight 0.3 \
  --results results_v7_huber_dz_prod \
  2>&1 | tee $RESULTS_DIR/exp_c3b_v7_huber_dz_train.log

echo ""
echo "📊 C3c: Sim with Huber+deadzone v6+v7..."
cp -r results_v6_prod results_v6_prod_bak_v3
cp -r results_v7_prod results_v7_prod_bak_v3
cp -r results_v6_huber_dz_prod/* results_v6_prod/
cp -r results_v7_huber_dz_prod/* results_v7_prod/

$SIM_BASE 2>&1 | tee $RESULTS_DIR/exp_c3c_sim_huber_dz.log

rm -rf results_v6_prod results_v7_prod
mv results_v6_prod_bak_v3 results_v6_prod
mv results_v7_prod_bak_v3 results_v7_prod

# C4: Best portfolio params from A+B + Huber (if Huber helps)
echo ""
echo "📊 C4: Huber + hysteresis=5 + turnover-budget=3 + min-zscore=0.5..."
cp -r results_v6_prod results_v6_prod_bak_v3
cp -r results_v7_prod results_v7_prod_bak_v3
cp -r results_v6_huber_prod/* results_v6_prod/
cp -r results_v7_huber_prod/* results_v7_prod/

$SIM_BASE --hysteresis 5 --turnover-budget 3 --min-zscore 0.5 \
  2>&1 | tee $RESULTS_DIR/exp_c4_huber_pc.log

rm -rf results_v6_prod results_v7_prod
mv results_v6_prod_bak_v3 results_v6_prod
mv results_v7_prod_bak_v3 results_v7_prod

echo "✅ EXP C done."

# ============================================================
# SUMMARY: Extract key metrics from all logs
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SUMMARY: Quick comparison"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
printf "%-45s %8s %8s %8s %8s %8s %10s\n" "Experiment" "Return" "Sharpe" "HAC" "MaxDD" "WinRate" "Turnover"
echo "────────────────────────────────────────────────────────────────────────────────────────────────────"

for logf in $RESULTS_DIR/*.log; do
  name=$(basename $logf .log)
  ret=$(grep -o 'Return:.*' $logf 2>/dev/null | head -1 | awk '{print $2}' || echo "—")
  sharpe=$(grep 'Sharpe:' $logf 2>/dev/null | grep -v HAC | head -1 | awk '{print $NF}' || echo "—")
  hac=$(grep 'Sharpe HAC:' $logf 2>/dev/null | head -1 | awk '{print $NF}' || echo "—")
  dd=$(grep 'Max DD:' $logf 2>/dev/null | head -1 | awk '{print $NF}' || echo "—")
  wr=$(grep 'Win Rate:' $logf 2>/dev/null | head -1 | awk '{print $NF}' || echo "—")
  turn=$(grep 'Turnover:' $logf 2>/dev/null | head -1 | awk '{print $2}' || echo "—")
  printf "%-45s %8s %8s %8s %8s %8s %10s\n" "$name" "$ret" "$sharpe" "$hac" "$dd" "$wr" "$turn"
done

echo ""
echo "============================================================"
echo "  OVERNIGHT RESEARCH v3 COMPLETE — $(date)"
echo "============================================================"
echo ""
echo "Results: $RESULTS_DIR/"
echo ""
echo "Key questions to answer:"
echo "  1. Does hysteresis reduce turnover AND keep/improve Sharpe?"
echo "  2. Does min-zscore improve WR from 63%?"
echo "  3. Does Huber loss improve WR / directional accuracy?"
echo "  4. Best combo of all three?"
echo ""
echo "If positive results: retrain CatBoost/XGBoost with Huber next."

unset SKIP_CALENDAR 2>/dev/null || true
