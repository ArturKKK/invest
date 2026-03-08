#!/bin/bash
# ============================================================
# Train RESEARCH models (walk-forward with held-out test set)
# ============================================================
# Usage:
#   ./train_research.sh              # Full: 3 windows × HPO 50 trials × 5 seeds
#   ./train_research.sh --gpu        # Same, but CatBoost on GPU (requires CUDA)
#   ./train_research.sh --skip-hpo   # Skip HPO (use default params, ~3× faster)
#   ./train_research.sh --single-window          # Quick: window 3 only
#   ./train_research.sh --single-window --gpu    # Quick + GPU
#
# Results go to: results_v6/, results_v7/, results_catboost/, results_xgboost/
# ============================================================

set -e

EXTRA_ARGS=""
GPU_FLAG=""

# Parse args: separate --gpu from EXTRA_ARGS
for arg in "$@"; do
    if [[ "$arg" == "--gpu" ]]; then
        GPU_FLAG="--gpu"
    else
        EXTRA_ARGS="$EXTRA_ARGS $arg"
    fi
done

echo "============================================================"
echo "  RESEARCH TRAINING — walk-forward, held-out test"
if [[ -n "$GPU_FLAG" ]]; then
echo "  GPU enabled for CatBoost"
fi
echo "============================================================"
echo ""

echo "━━━ [1/4] LGB v6 (research) ━━━"
python run_pipeline_v6.py $EXTRA_ARGS
echo ""

echo "━━━ [2/4] LGB v7 (research) ━━━"
python run_pipeline_v7.py $EXTRA_ARGS
echo ""

echo "━━━ [3/4] CatBoost (research) ━━━"
python run_pipeline_catboost.py $GPU_FLAG $EXTRA_ARGS
echo ""

echo "━━━ [4/4] XGBoost + News Interactions (research) ━━━"
python run_pipeline_xgboost.py $GPU_FLAG $EXTRA_ARGS
echo ""

echo "============================================================"
echo "  ✅ All research models trained!"
echo ""
echo "  Results saved to:"
echo "    results_v6/"
echo "    results_v7/"
echo "    results_catboost/"
echo "    results_xgboost/"
echo ""
echo "  Check test-period Sharpe / ICIR in the output above."
echo "  If metrics are good → run ./train_production.sh --gpu"
echo "============================================================"
