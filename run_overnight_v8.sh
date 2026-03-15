#!/bin/bash
set -euo pipefail

# ============================================================
# OVERNIGHT RESEARCH v8 — Macro + DVOL (Full Feature Set)
# ============================================================
# Adds FRED macro features (~38 cols) ON TOP of existing DVOL features.
# Data needed on cluster:
#   - data/sentiment/macro_daily.parquet  (75 KB, 9 FRED series)
#   - data/sentiment/deribit_dvol.parquet (already from v7)
#
# Macro features (add_macro_features in v6):
#   Raw:     vix_close, spx_close, dxy_close, gold_close, yield_10y_close,
#            hy_spread, breakeven_10y, yield_curve_10y2y, fed_funds_rate
#   Changes: *_chg_1d/5d/20d (7 series × 3 = 21)
#   Z-scores: vix_close_z20d, hy_spread_z20d, breakeven_10y_z20d,
#             yield_curve_10y2y_z20d
#   Cross:   risk_aversion, real_rate, risk_on_off_ratio, real_rate_chg_5d
#
# All macro+DVOL → REGIME_COLS (market-level, not CS-ranked).
# CB/XGB import add_macro_features from v6 → all 4 models get macro.
#
# Experiment design:
#   A: 4-model Huber + macro + DVOL + mz0.5      (full feature set)
#   B: 4-model Huber + macro + DVOL (no mz)       (check mz impact)
#   C: v4-best reference (no macro, no DVOL)       (HAC 8.14 baseline)
#   D: v7 DVOL-only reference                      (isolate macro delta)
#
# Compare A vs C → total macro+DVOL improvement
# Compare A vs D → marginal macro improvement over DVOL
#
# Expected runtime: ~50 min (4 retrains + 4 sims)
# ============================================================

TRAIN_END="2026-02-01"
VAL_END="2026-03-07"

RESULTS_DIR="overnight_v8_results"
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
    bak="results_${suffix}_prod_bak_v8exp"
    prod="results_${suffix}_prod"
    if [ -d "$bak" ]; then
      rm -rf "$prod"
      mv "$bak" "$prod"
    fi
  done
  echo "✅ Restored."
}
trap cleanup EXIT

# ── Pre-flight checks ──
echo "============================================================"
echo "  OVERNIGHT RESEARCH v8 — Macro + DVOL"
echo "  Started: $(date)"
echo "============================================================"
echo ""

MACRO_FILE="data/sentiment/macro_daily.parquet"
DVOL_FILE="data/sentiment/deribit_dvol.parquet"
if [ ! -f "$MACRO_FILE" ]; then
  echo "❌ Missing $MACRO_FILE — run: scp macro_daily.parquet to cluster"
  exit 1
fi
if [ ! -f "$DVOL_FILE" ]; then
  echo "❌ Missing $DVOL_FILE — should exist from v7"
  exit 1
fi
echo "✅ Data files present: macro + DVOL"
echo ""

# ============================================================
# STEP 1: Retrain all 4 models with macro + DVOL
# ============================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 1a: v6 Huber retrain (+ macro, + DVOL, + news)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python run_pipeline_v6.py \
  --production --skip-hpo \
  --train-end $TRAIN_END --val-end $VAL_END \
  --news-mode all \
  --huber \
  --results results_v6_macro_prod \
  2>&1 | tee $RESULTS_DIR/v6_macro_train.log

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 1b: v7 Huber retrain (+ macro, + DVOL, no news)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python run_pipeline_v7.py \
  --production --skip-hpo \
  --train-end $TRAIN_END --val-end $VAL_END \
  --huber \
  --results results_v7_macro_prod \
  2>&1 | tee $RESULTS_DIR/v7_macro_train.log

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 1c: CatBoost Huber retrain (+ macro, + DVOL)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python run_pipeline_catboost.py \
  --production --skip-hpo \
  --train-end $TRAIN_END --val-end $VAL_END \
  --huber \
  --results results_catboost_macro_prod \
  2>&1 | tee $RESULTS_DIR/cb_macro_train.log

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 1d: XGBoost Huber retrain (+ macro, + DVOL)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python run_pipeline_xgboost.py \
  --production --skip-hpo \
  --train-end $TRAIN_END --val-end $VAL_END \
  --huber --huber-slope 1.0 \
  --results results_xgboost_macro_prod \
  2>&1 | tee $RESULTS_DIR/xgb_macro_train.log

echo ""

# ============================================================
# STEP 2: Sim with macro+DVOL models
# ============================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 2: 4-model Huber + macro + DVOL sims"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Backup production models
echo "📦 Backing up production models..."
cp -r results_v6_prod results_v6_prod_bak_v8exp
cp -r results_v7_prod results_v7_prod_bak_v8exp
cp -r results_catboost_prod results_catboost_prod_bak_v8exp
cp -r results_xgboost_prod results_xgboost_prod_bak_v8exp

# Swap in macro+DVOL models
echo "🔄 Swapping in macro+DVOL models..."
cp -r results_v6_macro_prod/* results_v6_prod/
cp -r results_v7_macro_prod/* results_v7_prod/
cp -r results_catboost_macro_prod/* results_catboost_prod/
cp -r results_xgboost_macro_prod/* results_xgboost_prod/

echo "📊 Sim A: 4-model macro+DVOL + mz=0.5..."
$SIM_BASE --min-zscore 0.5 2>&1 | tee $RESULTS_DIR/macro_dvol_4model_mz05.log || true

echo ""
echo "📊 Sim B: 4-model macro+DVOL (no mz)..."
$SIM_BASE 2>&1 | tee $RESULTS_DIR/macro_dvol_4model.log || true

# ============================================================
# STEP 3: Reference sims
# ============================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 3: Reference sims"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Sim C: v4-best (no macro, no DVOL) ──
echo "🔄 Swapping in v4 Huber models (no macro, no DVOL)..."
cp -r results_v6_huber_prod/* results_v6_prod/
cp -r results_v7_huber_prod/* results_v7_prod/
cp -r results_catboost_huber_prod/* results_catboost_prod/
cp -r results_xgboost_huber_prod/* results_xgboost_prod/

echo "📊 Sim C: v4-best (no macro, no DVOL) + mz=0.5 [baseline]..."
$SIM_BASE --min-zscore 0.5 2>&1 | tee $RESULTS_DIR/v4_ref_nomacro_nodvol_mz05.log || true

# ── Sim D: DVOL-only (from v7, no macro) ──
echo ""
if [ -d "results_v6_dvol_prod" ] && [ -d "results_v7_dvol_prod" ] && \
   [ -d "results_catboost_dvol_prod" ] && [ -d "results_xgboost_dvol_prod" ]; then
  echo "🔄 Swapping in DVOL-only models (v7, no macro)..."
  cp -r results_v6_dvol_prod/* results_v6_prod/
  cp -r results_v7_dvol_prod/* results_v7_prod/
  cp -r results_catboost_dvol_prod/* results_catboost_prod/
  cp -r results_xgboost_dvol_prod/* results_xgboost_prod/

  echo "📊 Sim D: DVOL-only (v7) + mz=0.5 [DVOL baseline]..."
  $SIM_BASE --min-zscore 0.5 2>&1 | tee $RESULTS_DIR/dvol_only_4model_mz05_ref.log || true
else
  echo "⚠️  Skipping Sim D — DVOL-only models not found (run v7 first)"
fi

# ============================================================
# STEP 4: Restore and Summary
# ============================================================

echo ""
echo "📦 Restoring production models..."
for suffix in v6 v7 catboost xgboost; do
  bak="results_${suffix}_prod_bak_v8exp"
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
echo "  OVERNIGHT v8 COMPLETE — $(date)"
echo "============================================================"
echo ""
echo "Results (all with 4-model Huber ensemble):"
echo "  A: macro_dvol_4model_mz05.log         (macro + DVOL + mz=0.5)"
echo "  B: macro_dvol_4model.log               (macro + DVOL, no mz)"
echo "  C: v4_ref_nomacro_nodvol_mz05.log     (v4-best baseline, HAC 8.14)"
echo "  D: dvol_only_4model_mz05_ref.log       (DVOL-only from v7)"
echo ""
echo "Key comparisons:"
echo "  A vs C → total improvement from macro + DVOL"
echo "  A vs D → marginal improvement from macro alone"
echo "  D vs C → DVOL-only improvement (should match v7 results)"
echo ""
echo "Logs in: $RESULTS_DIR/"
