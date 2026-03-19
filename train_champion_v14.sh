#!/bin/bash
set -euo pipefail

# ============================================================
# Train v14 Champion: cb_market_noderiv_hpo (CatBoost solo)
# ============================================================
# Config from v14 experiment (won +143.8%, HAC 5.33):
#   CatBoost, Huber loss (delta=1.0), HPO 50 trials, 5 seeds
#   --news-mode market-only --no-derivatives
#
# PURGE_DAYS=8: gap between train_end and val_start.
# So train_end must be at least 9 days before latest data.
#
# Uses model_registry.py to archive before overwriting.
#
# Usage (cluster with GPU):
#   nohup ./train_champion_v14.sh > train_champion.log 2>&1 &
#
# Usage (VPS/local without GPU — slower but works):
#   GPU="" ./train_champion_v14.sh
#
# Custom dates:
#   TRAIN_END=2026-03-01 VAL_END=2026-03-18 ./train_champion_v14.sh
# ============================================================

GPU="${GPU:---gpu}"
RESULTS="results_catboost_prod"
PURGE=8

# Auto-compute dates: val_end = yesterday, train_end = val_end - 38 days
# PURGE_DAYS=8, so val_start = train_end + 8 → val window ~30 days
# Need enough val data for HPO not to overfit (min 3-4 weeks)
if command -v gdate &>/dev/null; then
    DATE_CMD=gdate  # macOS with coreutils
else
    DATE_CMD=date   # Linux
fi
VAL_END="${VAL_END:-$($DATE_CMD -u -d '1 day ago' +%Y-%m-%d 2>/dev/null || $DATE_CMD -u -v-1d +%Y-%m-%d)}"
TRAIN_END="${TRAIN_END:-$($DATE_CMD -u -d "$VAL_END - 38 days" +%Y-%m-%d 2>/dev/null || $DATE_CMD -u -j -f %Y-%m-%d -v-38d "$VAL_END" +%Y-%m-%d)}"

# Sanity check: val_start (train_end + purge) must be BEFORE val_end
VAL_START=$($DATE_CMD -u -d "$TRAIN_END + $PURGE days" +%Y-%m-%d 2>/dev/null || $DATE_CMD -u -j -f %Y-%m-%d -v+${PURGE}d "$TRAIN_END" +%Y-%m-%d)
if [[ "$VAL_START" > "$VAL_END" || "$VAL_START" == "$VAL_END" ]]; then
    echo "❌ FATAL: val_start ($VAL_START) >= val_end ($VAL_END)"
    echo "   train_end=$TRAIN_END + purge=${PURGE}d = val_start=$VAL_START"
    echo "   Fix: set TRAIN_END earlier or VAL_END later"
    echo "   Example: TRAIN_END=2026-02-08 VAL_END=2026-03-18 $0"
    exit 1
fi

echo "============================================================"
echo "  Training v14 Champion: cb_market_noderiv_hpo"
echo "  GPU:       $GPU"
echo "  Train end: $TRAIN_END"
echo "  Val:       $VAL_START → $VAL_END (purge=${PURGE}d)"
echo "  Output:    $RESULTS/"
echo "============================================================"

# Archive current prod models via model_registry
if [[ -d "$RESULTS" ]]; then
    echo ""
    echo "📦 Archiving current production models..."
    python model_registry.py archive --tag "pre-v14-champion" --notes "Before v14 champion deploy" 2>/dev/null || true
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
ls -la "$RESULTS/"*.cbm "$RESULTS/feature_names.json" 2>/dev/null

# Register new generation
echo ""
echo "📋 Registering new model generation..."
python model_registry.py register \
    --tag "v14-champion-cb-solo" \
    --notes "v14 cb_market_noderiv_hpo: CatBoost solo, Huber, HPO 50, market-only news, no derivs. Sim: +143.8% HAC 5.33, +147.5% with vol-size HAC 5.48" \
    2>/dev/null || true
