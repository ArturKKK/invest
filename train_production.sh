#!/bin/bash
# ============================================================
# Train PRODUCTION models (maximum data, no test holdout)
# ============================================================
# Usage:
#   ./train_production.sh                    # Default: train→2025-12, val→2026-03-07
#   ./train_production.sh --gpu              # CatBoost + XGBoost on GPU
#   ./train_production.sh --train-end 2025-10-01 --val-end 2026-03-15
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
echo "  PRODUCTION TRAINING — max data, models for live trading"
if [[ -n "$GPU_FLAG" ]]; then
echo "  GPU enabled for CatBoost + XGBoost"
fi
echo "============================================================"
echo ""

echo "━━━ [1/5] LGB v6 (production) ━━━"
python run_pipeline_v6.py --production $EXTRA_ARGS
echo ""

echo "━━━ [2/5] LGB v7 (production) ━━━"
python run_pipeline_v7.py --production $EXTRA_ARGS
echo ""

echo "━━━ [3/5] CatBoost (production) ━━━"
python run_pipeline_catboost.py --production $GPU_FLAG $EXTRA_ARGS
echo ""

echo "━━━ [4/5] XGBoost + News Interactions (production) ━━━"
python run_pipeline_xgboost.py --production $GPU_FLAG $EXTRA_ARGS
echo ""

echo "━━━ [5/5] MLP (production, GPU) ━━━"
python run_pipeline_mlp.py --production $GPU_FLAG $EXTRA_ARGS
echo ""

echo "============================================================"
echo "  ✅ All production models trained!"
echo ""
echo "  Models saved to:"
echo "    results_v6_prod/"
echo "    results_v7_prod/"
echo "    results_catboost_prod/"
echo "    results_xgboost_prod/"
echo "    results_mlp_prod/"
echo ""
echo "  run_trading.py will auto-detect and use these."
echo "  To deploy: scp -r results_*_prod/ root@185.42.163.63:/home/trader/invest/"
echo "============================================================"
