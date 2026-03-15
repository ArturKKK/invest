#!/bin/bash
set -euo pipefail

# ============================================================
# OVERNIGHT RESEARCH v6 — v7 Huber WITH news
# ============================================================
# v5 showed: v6 WITH news + Huber = HAC 8.14 (news HELP under Huber).
# Question: does the same apply to v7? v7 never had news before.
#
# This script:
#   1. Retrains v7 Huber WITH news (--news-mode all)
#   2. Sims 4-model Huber (v6 with-news + v7 with-news + CB + XGB)
#   3. Compares vs v4 best (v7 no-news, HAC 8.14)
#
# Models reused from v3/v4 (already on cluster):
#   - results_v6_huber_prod (v3, with news)
#   - results_catboost_huber_prod (v4, with news)
#   - results_xgboost_huber_prod (v4, with news)
#
# Expected runtime: ~15-20 min (1 retrain + 3 sims)
# ============================================================

TRAIN_END="2026-02-01"
VAL_END="2026-03-07"

RESULTS_DIR="overnight_v6_results"
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
    bak="results_${suffix}_prod_bak_v6exp"
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
echo "  OVERNIGHT RESEARCH v6 — v7 Huber WITH news"
echo "  Started: $(date)"
echo "============================================================"
echo ""

# ============================================================
# STEP 1: Retrain v7 Huber WITH news
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 1: v7 Huber retrain (--news-mode all)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
python run_pipeline_v7.py \
  --production --skip-hpo \
  --train-end $TRAIN_END --val-end $VAL_END \
  --news-mode all \
  --huber \
  --results results_v7_huber_withnews_prod \
  2>&1 | tee $RESULTS_DIR/v7_huber_withnews_train.log

echo ""

# ============================================================
# STEP 2: Sim — 4-model Huber (v7 WITH news)
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 2: Full 4-model Huber sim (v7 with-news)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Backup production models
echo "📦 Backing up production models..."
cp -r results_v6_prod results_v6_prod_bak_v6exp
cp -r results_v7_prod results_v7_prod_bak_v6exp
cp -r results_catboost_prod results_catboost_prod_bak_v6exp
cp -r results_xgboost_prod results_xgboost_prod_bak_v6exp

# Swap in Huber models (v7 = new with-news version)
echo "🔄 Swapping in Huber models (v7 = with-news)..."
cp -r results_v6_huber_prod/* results_v6_prod/
cp -r results_v7_huber_withnews_prod/* results_v7_prod/
cp -r results_catboost_huber_prod/* results_catboost_prod/
cp -r results_xgboost_huber_prod/* results_xgboost_prod/

echo "📊 Sim A: 4-model Huber (v7 with-news)..."
$SIM_BASE 2>&1 | tee $RESULTS_DIR/huber_4model_v7news.log || true

echo ""
echo "📊 Sim B: 4-model Huber (v7 with-news) + mz=0.5..."
$SIM_BASE --min-zscore 0.5 2>&1 | tee $RESULTS_DIR/huber_4model_v7news_mz05.log || true

# ============================================================
# STEP 3: Reference — v7 WITHOUT news (v4 replication)
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 3: v4 replication (v7 no-news) + mz=0.5"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Swap v7 back to no-news Huber from v3
cp -r results_v7_huber_prod/* results_v7_prod/

echo "📊 Sim C: 4-model Huber (v7 no-news) + mz=0.5 [v4 ref]..."
$SIM_BASE --min-zscore 0.5 2>&1 | tee $RESULTS_DIR/huber_4model_v7nonews_mz05.log || true

# ALWAYS restore production models
echo ""
echo "📦 Restoring production models..."
rm -rf results_v6_prod results_v7_prod results_catboost_prod results_xgboost_prod
mv results_v6_prod_bak_v6exp results_v6_prod
mv results_v7_prod_bak_v6exp results_v7_prod
mv results_catboost_prod_bak_v6exp results_catboost_prod
mv results_xgboost_prod_bak_v6exp results_xgboost_prod
echo "✅ Production models restored."
# Clear trap since we restored manually
trap - EXIT
echo ""

# ============================================================
# SUMMARY
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
printf "%-50s %8s %8s %8s %8s %8s\n" "Experiment" "Return" "Sharpe" "HAC" "MaxDD" "WinRate"
printf "%-50s %8s %8s %8s %8s %8s\n" "──────────────────────────────────────────────────" "────────" "────────" "────────" "────────" "────────"

# Reference from v4 (hardcoded)
printf "%-50s %8s %8s %8s %8s %8s\n" "[v4 ref] v7 no-news + mz0.5 (HAC 8.14)" "+18.8%" "+7.29" "+8.14" "-4.5%" "65%"

for log in $RESULTS_DIR/*.log; do
  name=$(basename $log .log)
  [[ "$name" == *"_train"* ]] && continue

  ret=$(grep -m1 "Return:" $log 2>/dev/null | awk '{print $2}' || echo "—")
  sharpe=$(grep -m1 "Sharpe:" $log 2>/dev/null | awk '{print $2}' || echo "—")
  hac=$(grep -m1 "Sharpe HAC:" $log 2>/dev/null | awk '{print $3}' || echo "—")
  dd=$(grep -m1 "Max DD:" $log 2>/dev/null | awk '{print $3}' || echo "—")
  wr=$(grep -m1 "Win Rate:" $log 2>/dev/null | awk '{print $3}' || echo "—")

  printf "%-50s %8s %8s %8s %8s %8s\n" "$name" "$ret" "$sharpe" "$hac" "$dd" "${wr}%"
done

echo ""
echo "============================================================"
echo "  OVERNIGHT RESEARCH v6 COMPLETE — $(date)"
echo "============================================================"
echo ""
echo "Key question: Does v7 with news improve the 4-model Huber ensemble?"
echo "  Compare HAC(v7news) vs HAC(v7nonews=8.14)"
echo "  If better → use both v6+v7 with news for production"
echo "  If worse → keep v7 without news (current best)"
