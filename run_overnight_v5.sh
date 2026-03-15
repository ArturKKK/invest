#!/bin/bash
set -euo pipefail

# ============================================================
# OVERNIGHT RESEARCH v5 — v6 Huber WITHOUT news
# ============================================================
# v4 showed 4-model Huber = HAC 8.14 (best ever).
# But v6 was trained WITH news (bug — news hurt LGB by -47%).
# This script retrains v6 Huber without news, then re-sims.
#
# Models reused from v3/v4 (already on cluster):
#   - results_v7_huber_prod (v3, no news — correct)
#   - results_catboost_huber_prod (v4, with news — correct for CB)
#   - results_xgboost_huber_prod (v4, with news)
#
# Only retraining: v6 Huber --news-mode none
# Expected runtime: ~15-20 min (1 retrain + 3 sims)
# ============================================================

TRAIN_END="2026-02-01"
VAL_END="2026-03-07"

RESULTS_DIR="overnight_v5_results"
mkdir -p $RESULTS_DIR

SIM_BASE="python run_fast_sim.py --data data/features/crypto_features_1h.parquet \
  --days 120 --start-date 2026-02-09 --end-date 2026-03-07 \
  --leverage 3 --kelly 0.8 --ensemble --edge-boost \
  --no-deriv-gate --no-ddstop"

export SKIP_CALENDAR=1

echo "============================================================"
echo "  OVERNIGHT RESEARCH v5 — v6 Huber no-news"
echo "  Started: $(date)"
echo "============================================================"
echo ""

# ============================================================
# STEP 1: Retrain v6 Huber WITHOUT news
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 1: v6 Huber retrain (--news-mode none)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
python run_pipeline_v6.py \
  --production --skip-hpo \
  --train-end $TRAIN_END --val-end $VAL_END \
  --news-mode none \
  --huber \
  --results results_v6_huber_nonews_prod \
  2>&1 | tee $RESULTS_DIR/v6_huber_nonews_train.log

echo ""

# ============================================================
# STEP 2: Sim — 4-model Huber (v6 no-news)
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 2: Full 4-model Huber sim (v6 no-news)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Backup production models
echo "📦 Backing up production models..."
cp -r results_v6_prod results_v6_prod_bak_v5
cp -r results_v7_prod results_v7_prod_bak_v5
cp -r results_catboost_prod results_catboost_prod_bak_v5
cp -r results_xgboost_prod results_xgboost_prod_bak_v5

# Swap in Huber models (v6 = new no-news version)
echo "🔄 Swapping in Huber models (v6 = no-news)..."
cp -r results_v6_huber_nonews_prod/* results_v6_prod/
cp -r results_v7_huber_prod/* results_v7_prod/
cp -r results_catboost_huber_prod/* results_catboost_prod/
cp -r results_xgboost_huber_prod/* results_xgboost_prod/

echo "📊 Sim: 4-model Huber (v6 no-news)..."
$SIM_BASE 2>&1 | tee $RESULTS_DIR/huber_4model_nonews.log

echo ""
echo "📊 Sim: 4-model Huber (v6 no-news) + mz=0.5..."
$SIM_BASE --min-zscore 0.5 2>&1 | tee $RESULTS_DIR/huber_4model_nonews_mz05.log

# ============================================================
# STEP 3: Comparison — v6 WITH news (v4 result replication)
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 3: 4-model Huber (v6 WITH news, v4 replication)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Swap v6 back to WITH-news Huber
cp -r results_v6_huber_prod/* results_v6_prod/

echo "📊 Sim: 4-model Huber (v6 with-news) + mz=0.5..."
$SIM_BASE --min-zscore 0.5 2>&1 | tee $RESULTS_DIR/huber_4model_withnews_mz05.log

# ALWAYS restore production models
echo ""
echo "📦 Restoring production models..."
rm -rf results_v6_prod results_v7_prod results_catboost_prod results_xgboost_prod
mv results_v6_prod_bak_v5 results_v6_prod
mv results_v7_prod_bak_v5 results_v7_prod
mv results_catboost_prod_bak_v5 results_catboost_prod
mv results_xgboost_prod_bak_v5 results_xgboost_prod
echo "✅ Production models restored."
echo ""

# ============================================================
# SUMMARY
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
printf "%-45s %8s %8s %8s %8s %8s\n" "Experiment" "Return" "Sharpe" "HAC" "MaxDD" "WinRate"
printf "%-45s %8s %8s %8s %8s %8s\n" "─────────────────────────────────────────────" "────────" "────────" "────────" "────────" "────────"

# Reference from v4 (hardcoded for comparison)
printf "%-45s %8s %8s %8s %8s %8s\n" "[v4] huber_4model_mz05 (v6 WITH news)" "+18.8%" "+7.29" "+8.14" "-4.5%" "65%"

for log in $RESULTS_DIR/*.log; do
  name=$(basename $log .log)
  [[ "$name" == *"_train"* ]] && continue

  ret=$(grep -m1 "Return:" $log 2>/dev/null | awk '{print $2}' || echo "—")
  sharpe=$(grep -m1 "Sharpe:" $log 2>/dev/null | awk '{print $2}' || echo "—")
  hac=$(grep -m1 "Sharpe HAC:" $log 2>/dev/null | awk '{print $3}' || echo "—")
  dd=$(grep -m1 "Max DD:" $log 2>/dev/null | awk '{print $3}' || echo "—")
  wr=$(grep -m1 "Win Rate:" $log 2>/dev/null | awk '{print $3}' || echo "—")

  printf "%-45s %8s %8s %8s %8s %8s\n" "$name" "$ret" "$sharpe" "$hac" "$dd" "${wr}%"
done

echo ""
echo "============================================================"
echo "  OVERNIGHT RESEARCH v5 COMPLETE — $(date)"
echo "============================================================"
echo ""
echo "Key question: Does v6 no-news improve the 4-model Huber ensemble?"
echo "  If HAC(no-news) > HAC(with-news) → use no-news for production"
echo "  If HAC(no-news) ≈ HAC(with-news) → Huber neutralized news noise, either is fine"
echo "  If HAC(no-news) < HAC(with-news) → keep news (Huber made them useful)"
