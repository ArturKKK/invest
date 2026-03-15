#!/bin/bash
set -euo pipefail

# ============================================================
# OVERNIGHT RESEARCH v4 — Retrain CatBoost + XGBoost with Huber
# ============================================================
# Purpose: Complete the 4-model Huber ensemble.
# v3 showed Huber-trained v6+v7 → HAC 7.56, WR 65%.
# Now retrain CatBoost + XGBoost with Huber, then sim full ensemble.
#
# Expected runtime: ~40-60 min (2 retrains + 1 sim)
# ============================================================

RESULTS_DIR="overnight_v4_results"
mkdir -p $RESULTS_DIR

# Base sim command — same OOS window as v3
SIM_BASE="python run_fast_sim.py --data data/features/crypto_features_1h.parquet \
  --days 120 --start-date 2026-02-09 --end-date 2026-03-07 \
  --leverage 3 --kelly 0.8 --ensemble --edge-boost \
  --no-deriv-gate --no-ddstop"

export SKIP_CALENDAR=1

echo "============================================================"
echo "  OVERNIGHT RESEARCH v4 — CatBoost + XGBoost Huber Retrain"
echo "  Started: $(date)"
echo "============================================================"
echo ""

# ============================================================
# STEP 0: Baseline (for comparison)
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 0: Baseline (current production models)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Baseline sim..."
$SIM_BASE 2>&1 | tee $RESULTS_DIR/baseline.log
echo ""

# ============================================================
# STEP 1: Retrain CatBoost with Huber
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 1: CatBoost Huber retrain"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Training CatBoost with Huber loss..."
python run_pipeline_catboost.py \
  --production --skip-hpo --gpu \
  --huber \
  --results results_catboost_huber_prod \
  2>&1 | tee $RESULTS_DIR/catboost_huber_train.log

echo ""

# ============================================================
# STEP 2: Retrain XGBoost with Huber
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 2: XGBoost Huber retrain"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Training XGBoost with Pseudo-Huber loss..."
python run_pipeline_xgboost.py \
  --production --skip-hpo --gpu \
  --huber \
  --results results_xgboost_huber_prod \
  2>&1 | tee $RESULTS_DIR/xgboost_huber_train.log

echo ""

# ============================================================
# STEP 3: Sim with full 4-model Huber ensemble
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 3: Full Huber ensemble simulation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Backup production models
echo "📦 Backing up production models..."
cp -r results_v6_prod results_v6_prod_bak_v4
cp -r results_v7_prod results_v7_prod_bak_v4
cp -r results_catboost_prod results_catboost_prod_bak_v4
cp -r results_xgboost_prod results_xgboost_prod_bak_v4

# Swap in Huber models
echo "🔄 Swapping in Huber models..."
cp -r results_v6_huber_prod/* results_v6_prod/
cp -r results_v7_huber_prod/* results_v7_prod/
cp -r results_catboost_huber_prod/* results_catboost_prod/
cp -r results_xgboost_huber_prod/* results_xgboost_prod/

echo "📊 Sim: 4-model Huber ensemble..."
$SIM_BASE 2>&1 | tee $RESULTS_DIR/huber_4model.log
RC_4M=$?

# Also sim with min-zscore=0.5 (best from v3)
echo ""
echo "📊 Sim: 4-model Huber + mz=0.5..."
$SIM_BASE --min-zscore 0.5 2>&1 | tee $RESULTS_DIR/huber_4model_mz05.log
RC_MZ=$?

# ALWAYS restore production models
echo ""
echo "📦 Restoring production models..."
rm -rf results_v6_prod results_v7_prod results_catboost_prod results_xgboost_prod
mv results_v6_prod_bak_v4 results_v6_prod
mv results_v7_prod_bak_v4 results_v7_prod
mv results_catboost_prod_bak_v4 results_catboost_prod
mv results_xgboost_prod_bak_v4 results_xgboost_prod
echo "✅ Production models restored."

if [[ ${RC_4M:-0} -ne 0 || ${RC_MZ:-0} -ne 0 ]]; then
  echo "⚠️  Some sims failed — check logs."
fi

echo ""

# ============================================================
# STEP 4: Sim with ONLY v6+v7 Huber (v3 winner, for comparison)
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 4: v6+v7 Huber only (v3 winner, for comparison)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Backup + swap only v6/v7
cp -r results_v6_prod results_v6_prod_bak_v4b
cp -r results_v7_prod results_v7_prod_bak_v4b
cp -r results_v6_huber_prod/* results_v6_prod/
cp -r results_v7_huber_prod/* results_v7_prod/

echo "📊 Sim: v6+v7 Huber + baseline CatBoost+XGBoost + mz=0.5..."
$SIM_BASE --min-zscore 0.5 2>&1 | tee $RESULTS_DIR/huber_v6v7_mz05.log

# Restore
rm -rf results_v6_prod results_v7_prod
mv results_v6_prod_bak_v4b results_v6_prod
mv results_v7_prod_bak_v4b results_v7_prod
echo "✅ Restored."
echo ""

# ============================================================
# SUMMARY
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SUMMARY: Quick comparison"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
printf "%-45s %8s %8s %8s %8s %8s %10s\n" "Experiment" "Return" "Sharpe" "HAC" "MaxDD" "WinRate" "Turnover"
printf "%-45s %8s %8s %8s %8s %8s %10s\n" "─────────────────────────────────────────────" "────────" "────────" "────────" "────────" "────────" "──────────"

for log in $RESULTS_DIR/*.log; do
  name=$(basename $log .log)
  # Skip training logs
  [[ "$name" == *"_train"* ]] && continue

  ret=$(grep -m1 "Return:" $log 2>/dev/null | awk '{print $2}' || echo "—")
  sharpe=$(grep -m1 "Sharpe:" $log 2>/dev/null | awk '{print $2}' || echo "—")
  hac=$(grep -m1 "Sharpe HAC:" $log 2>/dev/null | awk '{print $3}' || echo "—")
  dd=$(grep -m1 "Max DD:" $log 2>/dev/null | awk '{print $3}' || echo "—")
  wr=$(grep -m1 "Win Rate:" $log 2>/dev/null | awk '{print $3}' || echo "—")
  turnover=$(grep -m1 "Turnover:" $log 2>/dev/null | awk '{print $2}' || echo "—")

  printf "%-45s %8s %8s %8s %8s %8s %10s\n" "$name" "$ret" "$sharpe" "$hac" "$dd" "${wr}%" "$turnover"
done

echo ""
echo "============================================================"
echo "  OVERNIGHT RESEARCH v4 COMPLETE — $(date)"
echo "============================================================"
echo ""
echo "Results: $RESULTS_DIR/"
echo ""
echo "Key comparison:"
echo "  - baseline: all 4 models with RMSE (current production)"
echo "  - huber_4model: all 4 models retrained with Huber"
echo "  - huber_4model_mz05: 4-model Huber + min-zscore=0.5 filter"
echo "  - huber_v6v7_mz05: only v6+v7 Huber (v3 winner replication)"
echo ""
echo "If 4-model Huber > v6v7-only Huber → retrain all models with Huber for VPS."
