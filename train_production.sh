#!/bin/bash
# ============================================================
# Train PRODUCTION models (maximum data, no test holdout)
# ============================================================
# Usage:
#   ./train_production.sh                    # Default (Huber loss, no news for v7)
#   ./train_production.sh --gpu              # CatBoost + XGBoost on GPU
#   ./train_production.sh --train-end 2025-10-01 --val-end 2026-03-15
#
# Gen#4 config (v4-huber-4model-mz05, HAC 8.14):
#   - All 4 GBDT models use Huber loss
#   - v7 trained with --news-mode none (news hurt performance)
#   - XGBoost uses --huber-slope 1.0
#   - v6/v7 use default --huber-alpha 0.9
#   - CatBoost uses default --huber-delta 1.0
#   - MLP skipped (not in 4-model ensemble)
#
# Research models (with test holdout) — for comparing ideas:
#   python run_pipeline_v6.py                # 3 walk-forward windows
#   python run_pipeline_v6.py --single-window # Quick: window 3 only
#
# Production models go to results_v6_prod/, results_v7_prod/, results_catboost_prod/, results_xgboost_prod/
# run_trading.py automatically prefers *_prod dirs when they exist.
# ============================================================

set -e

# Parse --gpu separately (only CatBoost & XGBoost support it)
EXTRA_ARGS=""
GPU_FLAG=""

for arg in "$@"; do
    if [[ "$arg" == "--gpu" ]]; then
        GPU_FLAG="--gpu"
    else
        EXTRA_ARGS="$EXTRA_ARGS $arg"
    fi
done

echo "============================================================"
echo "  PRODUCTION TRAINING — Gen#4 Huber 4-model ensemble"
if [[ -n "$GPU_FLAG" ]]; then
echo "  GPU enabled for CatBoost + XGBoost"
fi
echo "============================================================"
echo ""

echo "━━━ [1/4] LGB v6 Huber (production) ━━━"
python run_pipeline_v6.py --production --huber $EXTRA_ARGS
echo ""

echo "━━━ [2/4] LGB v7 Huber (production, no news) ━━━"
python run_pipeline_v7.py --production --huber --news-mode none $EXTRA_ARGS
echo ""

echo "━━━ [3/4] CatBoost Huber (production) ━━━"
python run_pipeline_catboost.py --production --huber $GPU_FLAG $EXTRA_ARGS
echo ""

echo "━━━ [4/4] XGBoost Huber (production) ━━━"
python run_pipeline_xgboost.py --production --huber --huber-slope 1.0 $GPU_FLAG $EXTRA_ARGS
echo ""

echo "============================================================"
echo "  ✅ All production models trained (Huber loss)!"
echo ""
echo "  Models saved to:"
echo "    results_v6_prod/"
echo "    results_v7_prod/"
echo "    results_catboost_prod/"
echo "    results_xgboost_prod/"
echo ""
echo "  run_trading.py will auto-detect and use these."
echo "  To deploy: ./deploy/deploy.sh"
echo "============================================================"
