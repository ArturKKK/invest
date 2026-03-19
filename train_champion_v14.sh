#!/bin/bash
set -euo pipefail

# ============================================================
# Train v14 Champion: cb_market_noderiv_hpo
# ============================================================
# CatBoost, Huber loss (delta=1.0), HPO 50 trials, 5 seeds
# --news-mode market-only --no-derivatives
#
# Output: results_catboost_prod/ (ready for deploy)
#
# Usage (cluster with GPU):
#   nohup ./train_champion_v14.sh > train_champion.log 2>&1 &
#
# Usage (VPS/local without GPU — slower but works):
#   GPU="" ./train_champion_v14.sh
# ============================================================

GPU="${GPU:---gpu}"
RESULTS="results_catboost_prod"
TRAIN_END="${TRAIN_END:-2026-03-15}"
VAL_END="${VAL_END:-2026-03-19}"

echo "============================================================"
echo "  Training v14 Champion: cb_market_noderiv_hpo"
echo "  GPU: $GPU"
echo "  Train end: $TRAIN_END"
echo "  Val end:   $VAL_END"
echo "  Output:    $RESULTS/"
echo "============================================================"

# Archive old models
if [[ -d "$RESULTS" ]]; then
    BACKUP="${RESULTS}_bak_$(date +%Y%m%d_%H%M)"
    echo "Backing up old models to $BACKUP"
    mv "$RESULTS" "$BACKUP"
fi

export SKIP_CALENDAR=1

python run_pipeline_catboost.py \
    --production \
    --seeds 5 \
    --huber \
    --news-mode market-only \
    --no-derivatives \
    --hpo-trials 50 \
    --train-end "$TRAIN_END" \
    --val-end "$VAL_END" \
    --results "$RESULTS" \
    $GPU

echo ""
echo "============================================================"
echo "  Done! Models in: $RESULTS/"
echo "============================================================"
ls -la "$RESULTS/"/*.cbm "$RESULTS/feature_names.json" 2>/dev/null
