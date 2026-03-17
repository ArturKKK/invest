#!/bin/bash
# ============================================================
# Train RESEARCH models (proper OOS evaluation)
# ============================================================
# Usage:
#   ./train_research.sh              # Full: 2 research windows × HPO 50 trials × 5 seeds
#   ./train_research.sh --gpu        # Same, but CatBoost/XGBoost on GPU
#   ./train_research.sh --skip-hpo   # Skip HPO (use default params, ~3× faster)
#
# Mirrors train_production.sh config (Huber loss, v7 no-news, etc.)
# so research metrics are directly comparable to production models.
#
# Research windows:
#   R1: train→2024-12-31, val 2025-01→2025-09, test 2025-10→2025-12
#   R2: train→2025-06-30, val 2025-07→2025-12, test 2026-01→2026-03
#
# Results go to: results_v6_research/, results_v7_research/,
#                results_catboost_research/, results_xgboost_research/
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
echo "  RESEARCH TRAINING — proper OOS on 2025 & 2026"
echo "  Config: Huber loss, v7 no-news (mirrors production)"
if [[ -n "$GPU_FLAG" ]]; then
echo "  GPU enabled for all models (LGB + CatBoost + XGBoost)"
fi
echo "============================================================"
echo ""

echo "━━━ [1/4] LGB v6 Huber (research) ━━━"
python run_pipeline_v6.py --research --huber $GPU_FLAG $EXTRA_ARGS
echo ""

echo "━━━ [2/4] LGB v7 Huber (research, no news) ━━━"
python run_pipeline_v7.py --research --huber --news-mode none $GPU_FLAG $EXTRA_ARGS
echo ""

echo "━━━ [3/4] CatBoost Huber (research) ━━━"
python run_pipeline_catboost.py --research --huber $GPU_FLAG $EXTRA_ARGS
echo ""

echo "━━━ [4/4] XGBoost Huber (research) ━━━"
python run_pipeline_xgboost.py --research --huber --huber-slope 1.0 $GPU_FLAG $EXTRA_ARGS
echo ""

echo "============================================================"
echo "  ✅ All research models trained!"
echo ""
echo "  Results saved to:"
echo "    results_v6_research/"
echo "    results_v7_research/"
echo "    results_catboost_research/"
echo "    results_xgboost_research/"
echo ""
echo "  Check test-period Sharpe / ICIR in the output above."
echo "  If metrics are good → run ./train_production.sh --gpu"
echo "============================================================"

# ────────────────────────────────────────────────────────────
# Post-training analysis: ensemble sim + model correlations
# ────────────────────────────────────────────────────────────
echo ""
echo "━━━ [5/5] Post-training analysis ━━━"
python analyze_research.py 2>/dev/null || echo "  ⚠️  analyze_research.py not found or failed, skipping post-analysis"
