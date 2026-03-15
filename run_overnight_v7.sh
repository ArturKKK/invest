#!/bin/bash
set -euo pipefail

# ============================================================
# OVERNIGHT RESEARCH v7 — DVOL Features (Implied Volatility)
# ============================================================
# New feature: Deribit DVOL (BTC + ETH implied vol) — 10 market-level features.
# Data: data/sentiment/deribit_dvol.parquet (87K rows, 2021-04 → 2026-03).
# These features go into add_derivatives_features() → REGIME_COLS (unranked).
#
# This script:
#   1. Retrains v6 Huber with DVOL + news (new features auto-picked up)
#   2. Retrains v7 Huber with DVOL (no news, to isolate DVOL effect)
#   3. Retrain CatBoost Huber with DVOL
#   4. Retrain XGBoost Huber with DVOL
#   5. Sims: 4-model Huber+DVOL vs v4-best reference (no DVOL)
#
# NOTE: DVOL is added to add_derivatives_features() in v6 pipeline.
#       CB and XGB import from v6 → they also see DVOL.
#       v7 imports add_derivatives_features from v6 → also gets DVOL.
#       So ALL models automatically get DVOL features.
#
# Models from v4 (without DVOL) used as reference:
#   - results_v6_huber_prod (v3/v5, v6 Huber + news)
#   - results_v7_huber_prod (v4, v7 Huber no news)
#   - results_catboost_huber_prod (v4, CB Huber)
#   - results_xgboost_huber_prod (v4, XGB Huber)
#
# New models go to *_dvol_prod dirs.
# Expected runtime: ~40-50 min (4 retrains + 3 sims)
# ============================================================

TRAIN_END="2026-02-01"
VAL_END="2026-03-07"

RESULTS_DIR="overnight_v7_results"
mkdir -p $RESULTS_DIR

SIM_BASE="python run_fast_sim.py --data data/features/crypto_features_1h.parquet \
  --days 120 --start-date 2026-02-09 --end-date 2026-03-07 \
  --leverage 3 --kelly 0.8 --ensemble --edge-boost \
  --no-deriv-gate --no-ddstop"

export SKIP_CALENDAR=1

# Safety: restore production models even if script crashes
cleanup() {
  echo "🛡️ Cleanup: restoring production models..."
  for suffix in v6 v7 catboost xgboost; do
    bak="results_${suffix}_prod_bak_v7exp"
    prod="results_${suffix}_prod"
    if [ -d "$bak" ]; then
      rm -rf "$prod"
      mv "$bak" "$prod"
    fi
  done
  echo "✅ Restored."
}
trap cleanup EXIT

echo "============================================================"
echo "  OVERNIGHT RESEARCH v7 — DVOL Features"
echo "  Started: $(date)"
echo "============================================================"
echo ""

# ============================================================
# STEP 1: Retrain all 4 models with DVOL
# ============================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 1a: v6 Huber retrain (+ DVOL, + news)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python run_pipeline_v6.py \
  --production --skip-hpo \
  --train-end $TRAIN_END --val-end $VAL_END \
  --news-mode all \
  --huber \
  --results results_v6_dvol_prod \
  2>&1 | tee $RESULTS_DIR/v6_dvol_train.log

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 1b: v7 Huber retrain (+ DVOL, no news)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python run_pipeline_v7.py \
  --production --skip-hpo \
  --train-end $TRAIN_END --val-end $VAL_END \
  --huber \
  --results results_v7_dvol_prod \
  2>&1 | tee $RESULTS_DIR/v7_dvol_train.log

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 1c: CatBoost Huber retrain (+ DVOL)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python run_pipeline_catboost.py \
  --production --skip-hpo \
  --train-end $TRAIN_END --val-end $VAL_END \
  --huber \
  --results results_catboost_dvol_prod \
  2>&1 | tee $RESULTS_DIR/cb_dvol_train.log

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 1d: XGBoost Huber retrain (+ DVOL)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python run_pipeline_xgboost.py \
  --production --skip-hpo \
  --train-end $TRAIN_END --val-end $VAL_END \
  --huber --huber-slope 1.0 \
  --results results_xgboost_dvol_prod \
  2>&1 | tee $RESULTS_DIR/xgb_dvol_train.log

echo ""

# ============================================================
# STEP 2: Sim with DVOL models
# ============================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 2: 4-model Huber+DVOL sim"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Backup production models
echo "📦 Backing up production models..."
cp -r results_v6_prod results_v6_prod_bak_v7exp
cp -r results_v7_prod results_v7_prod_bak_v7exp
cp -r results_catboost_prod results_catboost_prod_bak_v7exp
cp -r results_xgboost_prod results_xgboost_prod_bak_v7exp

# Swap in DVOL models
echo "🔄 Swapping in DVOL models..."
cp -r results_v6_dvol_prod/* results_v6_prod/
cp -r results_v7_dvol_prod/* results_v7_prod/
cp -r results_catboost_dvol_prod/* results_catboost_prod/
cp -r results_xgboost_dvol_prod/* results_xgboost_prod/

echo "📊 Sim A: 4-model Huber+DVOL + mz=0.5..."
$SIM_BASE --min-zscore 0.5 2>&1 | tee $RESULTS_DIR/huber_dvol_4model_mz05.log || true

echo ""
echo "📊 Sim B: 4-model Huber+DVOL (no mz filter)..."
$SIM_BASE 2>&1 | tee $RESULTS_DIR/huber_dvol_4model.log || true

# ============================================================
# STEP 3: Reference sim (v4-best, no DVOL)
# ============================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 3: v4 reference (no DVOL) + mz=0.5"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Swap back to non-DVOL Huber models
echo "🔄 Swapping in non-DVOL Huber models (v4 reference)..."
cp -r results_v6_huber_prod/* results_v6_prod/
cp -r results_v7_huber_prod/* results_v7_prod/
cp -r results_catboost_huber_prod/* results_catboost_prod/
cp -r results_xgboost_huber_prod/* results_xgboost_prod/

echo "📊 Sim C: v4-best (no DVOL) + mz=0.5 [reference]..."
$SIM_BASE --min-zscore 0.5 2>&1 | tee $RESULTS_DIR/huber_nodvol_4model_mz05_ref.log || true

# ============================================================
# STEP 4: Restore and Summary
# ============================================================

echo ""
echo "📦 Restoring production models..."
for suffix in v6 v7 catboost xgboost; do
  bak="results_${suffix}_prod_bak_v7exp"
  prod="results_${suffix}_prod"
  if [ -d "$bak" ]; then
    rm -rf "$prod"
    mv "$bak" "$prod"
  fi
done
trap - EXIT
echo "✅ Production models restored."

echo ""
echo "============================================================"
echo "  OVERNIGHT v7 COMPLETE — $(date)"
echo "============================================================"
echo ""
echo "Results:"
echo "  A: huber_dvol_4model_mz05.log     (DVOL + mz=0.5)"
echo "  B: huber_dvol_4model.log           (DVOL only)"
echo "  C: huber_nodvol_4model_mz05_ref.log (v4 ref, no DVOL)"
echo ""
echo "Compare A vs C to see DVOL impact."
echo "Logs in: $RESULTS_DIR/"
