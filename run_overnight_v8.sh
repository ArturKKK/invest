#!/bin/bash
set -euo pipefail

# ============================================================
# OVERNIGHT RESEARCH v8 — Macro Features (± DVOL)
# ============================================================
# v7 showed DVOL HURT (HAC 7.73 vs 8.14 baseline).
# This tests FRED macro features (~38 cols) in isolation and combined.
#
# Data needed:
#   - data/sentiment/macro_daily.parquet  (75 KB, 9 FRED series)
#   - data/sentiment/deribit_dvol.parquet (for Sim B only)
#
# Experiment design (all with Huber, 4-model ensemble):
#   A: macro-only (NO DVOL) + mz0.5        ← key test: does macro help?
#   B: macro + DVOL + mz0.5                ← does macro rescue DVOL?
#   C: v4-best reference (HAC 8.14)         ← baseline
#
# Key comparison: A vs C → macro-only improvement
# Expected runtime: ~45 min (8 retrains + 3 sims)
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
  # Restore DVOL file if hidden
  if [ -f "data/sentiment/.deribit_dvol.parquet.bak_v8" ]; then
    mv "data/sentiment/.deribit_dvol.parquet.bak_v8" "data/sentiment/deribit_dvol.parquet"
    echo "📦 DVOL file restored from cleanup"
  fi
  echo "✅ Restored."
}
trap cleanup EXIT

# ── Pre-flight checks ──
echo "============================================================"
echo "  OVERNIGHT RESEARCH v8 — Macro Features (± DVOL)"
echo "  Started: $(date)"
echo "============================================================"
echo ""

MACRO_FILE="data/sentiment/macro_daily.parquet"
if [ ! -f "$MACRO_FILE" ]; then
  echo "❌ Missing $MACRO_FILE — scp it to cluster first"
  exit 1
fi
echo "✅ Macro data present"
echo ""

# ============================================================
# STEP 1: Retrain all 4 models — macro-only (no DVOL)
# ============================================================
# --no-derivatives skips add_derivatives_features() entirely
# (which includes DVOL). Macro is added by add_macro_features()
# which runs BEFORE derivatives. So macro-only = keep derivatives
# but DVOL file simply absent → DVOL block skipped gracefully.
#
# IMPORTANT: derivatives (OI, taker, basis etc.) are valuable —
# we only want to skip DVOL. Since DVOL is loaded from a separate
# file inside add_derivatives_features(), we keep derivatives ON
# but ensure deribit_dvol.parquet is temporarily hidden.

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 1: Retrain 4 models — macro only (DVOL hidden)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Hide DVOL file temporarily
DVOL_FILE="data/sentiment/deribit_dvol.parquet"
DVOL_HIDDEN="data/sentiment/.deribit_dvol.parquet.bak_v8"
if [ -f "$DVOL_FILE" ]; then
  mv "$DVOL_FILE" "$DVOL_HIDDEN"
  echo "📦 DVOL file hidden for macro-only retrains"
fi

echo ""
echo "  STEP 1a: v6 Huber (+ macro, NO DVOL, + news)"
python run_pipeline_v6.py \
  --production --skip-hpo \
  --train-end $TRAIN_END --val-end $VAL_END \
  --news-mode all \
  --huber \
  --results results_v6_macroonly_prod \
  2>&1 | tee $RESULTS_DIR/v6_macroonly_train.log

echo ""
echo "  STEP 1b: v7 Huber (+ macro, NO DVOL, no news)"
python run_pipeline_v7.py \
  --production --skip-hpo \
  --train-end $TRAIN_END --val-end $VAL_END \
  --huber --news-mode none \
  --results results_v7_macroonly_prod \
  2>&1 | tee $RESULTS_DIR/v7_macroonly_train.log

echo ""
echo "  STEP 1c: CatBoost Huber (+ macro, NO DVOL) [GPU]"
python run_pipeline_catboost.py \
  --production --skip-hpo \
  --train-end $TRAIN_END --val-end $VAL_END \
  --huber --gpu \
  --results results_catboost_macroonly_prod \
  2>&1 | tee $RESULTS_DIR/cb_macroonly_train.log

echo ""
echo "  STEP 1d: XGBoost Huber (+ macro, NO DVOL) [GPU]"
python run_pipeline_xgboost.py \
  --production --skip-hpo \
  --train-end $TRAIN_END --val-end $VAL_END \
  --huber --huber-slope 1.0 --gpu \
  --results results_xgboost_macroonly_prod \
  2>&1 | tee $RESULTS_DIR/xgb_macroonly_train.log

# Restore DVOL file
if [ -f "$DVOL_HIDDEN" ]; then
  mv "$DVOL_HIDDEN" "$DVOL_FILE"
  echo ""
  echo "📦 DVOL file restored"
fi

echo ""

# ============================================================
# STEP 2: Retrain all 4 models — macro + DVOL
# ============================================================

if [ -f "$DVOL_FILE" ]; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  STEP 2: Retrain 4 models — macro + DVOL"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  echo ""
  echo "  STEP 2a: v6 Huber (+ macro, + DVOL, + news)"
  python run_pipeline_v6.py \
    --production --skip-hpo \
    --train-end $TRAIN_END --val-end $VAL_END \
    --news-mode all \
    --huber \
    --results results_v6_macro_prod \
    2>&1 | tee $RESULTS_DIR/v6_macro_dvol_train.log

  echo ""
  echo "  STEP 2b: v7 Huber (+ macro, + DVOL, no news)"
  python run_pipeline_v7.py \
    --production --skip-hpo \
    --train-end $TRAIN_END --val-end $VAL_END \
    --huber --news-mode none \
    --results results_v7_macro_prod \
    2>&1 | tee $RESULTS_DIR/v7_macro_dvol_train.log

  echo ""
  echo "  STEP 2c: CatBoost Huber (+ macro, + DVOL) [GPU]"
  python run_pipeline_catboost.py \
    --production --skip-hpo \
    --train-end $TRAIN_END --val-end $VAL_END \
    --huber --gpu \
    --results results_catboost_macro_prod \
    2>&1 | tee $RESULTS_DIR/cb_macro_dvol_train.log

  echo ""
  echo "  STEP 2d: XGBoost Huber (+ macro, + DVOL) [GPU]"
  python run_pipeline_xgboost.py \
    --production --skip-hpo \
    --train-end $TRAIN_END --val-end $VAL_END \
    --huber --huber-slope 1.0 --gpu \
    --results results_xgboost_macro_prod \
    2>&1 | tee $RESULTS_DIR/xgb_macro_dvol_train.log
else
  echo "⚠️  No DVOL data — skipping macro+DVOL retrains (Step 2)"
fi

echo ""

# ============================================================
# STEP 3: Sims
# ============================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 3: Simulations"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Backup production models
echo "📦 Backing up production models..."
cp -r results_v6_prod results_v6_prod_bak_v8exp
cp -r results_v7_prod results_v7_prod_bak_v8exp
cp -r results_catboost_prod results_catboost_prod_bak_v8exp
cp -r results_xgboost_prod results_xgboost_prod_bak_v8exp

# ── Sim A: macro-only (no DVOL) + mz0.5 ──
echo ""
echo "🔄 Swapping in macro-only models..."
cp -r results_v6_macroonly_prod/* results_v6_prod/
cp -r results_v7_macroonly_prod/* results_v7_prod/
cp -r results_catboost_macroonly_prod/* results_catboost_prod/
cp -r results_xgboost_macroonly_prod/* results_xgboost_prod/

echo "📊 Sim A: macro-only (no DVOL) + mz=0.5..."
$SIM_BASE --min-zscore 0.5 2>&1 | tee $RESULTS_DIR/macro_only_4model_mz05.log || true

# ── Sim B: macro + DVOL + mz0.5 ──
echo ""
if [ -d "results_v6_macro_prod" ] && [ -d "results_v7_macro_prod" ] && \
   [ -d "results_catboost_macro_prod" ] && [ -d "results_xgboost_macro_prod" ]; then
  echo "🔄 Swapping in macro+DVOL models..."
  cp -r results_v6_macro_prod/* results_v6_prod/
  cp -r results_v7_macro_prod/* results_v7_prod/
  cp -r results_catboost_macro_prod/* results_catboost_prod/
  cp -r results_xgboost_macro_prod/* results_xgboost_prod/

  echo "📊 Sim B: macro + DVOL + mz=0.5..."
  $SIM_BASE --min-zscore 0.5 2>&1 | tee $RESULTS_DIR/macro_dvol_4model_mz05.log || true
else
  echo "⚠️  Skipping Sim B — macro+DVOL models not found"
fi

# ── Sim C: v4-best reference ──
echo ""
echo "🔄 Swapping in v4 Huber models (baseline)..."
cp -r results_v6_huber_prod/* results_v6_prod/
cp -r results_v7_huber_prod/* results_v7_prod/
cp -r results_catboost_huber_prod/* results_catboost_prod/
cp -r results_xgboost_huber_prod/* results_xgboost_prod/

echo "📊 Sim C: v4-best (no macro, no DVOL) + mz=0.5 [baseline]..."
$SIM_BASE --min-zscore 0.5 2>&1 | tee $RESULTS_DIR/v4_ref_mz05.log || true

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
echo "Results (all 4-model Huber, mz=0.5):"
echo "  A: macro_only_4model_mz05.log     (macro, NO DVOL)     ← KEY TEST"
echo "  B: macro_dvol_4model_mz05.log     (macro + DVOL)"
echo "  C: v4_ref_mz05.log                (baseline HAC 8.14)"
echo ""
echo "Key comparison: A vs C → does macro help?"
echo "  If A > C → macro adds alpha, ship it"
echo "  If A ≈ C → macro neutral, skip"
echo "  If A < C → macro hurts (like DVOL did)"
echo ""
echo "Secondary: B vs A → does DVOL add anything on top of macro?"
echo "  v7 showed DVOL alone hurts (HAC 7.73 vs 8.14)"
echo ""
echo "Logs in: $RESULTS_DIR/"
