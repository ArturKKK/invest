#!/bin/bash
# ============================================================
# Train PRODUCTION models (maximum data, no test holdout)
# ============================================================
# Usage:
#   ./train_production.sh                    # Default: train→2025-09, val→2026-03
#   ./train_production.sh --train-end 2025-10-01 --val-end 2026-03-15
#   ./train_production.sh --skip-hpo         # Skip HPO (use defaults)
#
# Research models (with test holdout) — for comparing ideas:
#   python run_pipeline_v6.py                # 3 walk-forward windows
#   python run_pipeline_v6.py --single-window # Quick: window 3 only
#
# Production models go to results_v6_prod/, results_v7_prod/, results_catboost_prod/
# run_trading.py automatically prefers *_prod dirs when they exist.
# ============================================================

set -e

EXTRA_ARGS="$@"

echo "============================================================"
echo "  PRODUCTION TRAINING — max data, models for live trading"
echo "============================================================"
echo ""

echo "━━━ [1/3] LGB v6 (production) ━━━"
python run_pipeline_v6.py --production --skip-hpo $EXTRA_ARGS
echo ""

echo "━━━ [2/3] LGB v7 (production) ━━━"
python run_pipeline_v7.py --production --skip-hpo $EXTRA_ARGS
echo ""

echo "━━━ [3/3] CatBoost (production) ━━━"
python run_pipeline_catboost.py --production --skip-hpo $EXTRA_ARGS
echo ""

echo "============================================================"
echo "  ✅ All production models trained!"
echo ""
echo "  Models saved to:"
echo "    results_v6_prod/"
echo "    results_v7_prod/"
echo "    results_catboost_prod/"
echo ""
echo "  run_trading.py will auto-detect and use these."
echo "  To deploy: scp -r results_*_prod/ root@185.42.163.63:/home/trader/invest/"
echo "============================================================"
