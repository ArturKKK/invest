#!/bin/bash
# ============================================================
# Experiment 15: New features (basis/premium, market-wide aggregates)
# + bugfixes (v7 features, deriv-gate, renorm)
# ============================================================
# All results go to results/exp15_new_features/<model>/
# All logs go to results/exp15_new_features/logs/
#
# Usage:
#   ./run_exp15_new_features.sh              # CPU only
#   ./run_exp15_new_features.sh --gpu        # CatBoost on GPU
#   ./run_exp15_new_features.sh --skip-hpo   # Skip HPO (faster)
#   ./run_exp15_new_features.sh --gpu --skip-hpo  # Fast + GPU
# ============================================================

set -e

EXP_DIR="results/exp15_new_features"
LOG_DIR="$EXP_DIR/logs"
mkdir -p "$LOG_DIR"

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
echo "  EXPERIMENT 15: New Features + Bugfixes"
echo "  Output: $EXP_DIR/"
echo "  Logs:   $LOG_DIR/"
if [[ -n "$GPU_FLAG" ]]; then
echo "  GPU enabled for CatBoost"
fi
echo "============================================================"
echo ""

# ── 1. LGB v6 ────────────────────────────────────────────────
echo "━━━ [1/5] LGB v6 ━━━"
python run_pipeline_v6.py --results "$EXP_DIR/v6" $EXTRA_ARGS \
    2>&1 | tee "$LOG_DIR/v6.log"
echo ""

# ── 2. LGB v7 ────────────────────────────────────────────────
echo "━━━ [2/5] LGB v7 ━━━"
python run_pipeline_v7.py --results "$EXP_DIR/v7" $EXTRA_ARGS \
    2>&1 | tee "$LOG_DIR/v7.log"
echo ""

# ── 3. CatBoost ──────────────────────────────────────────────
echo "━━━ [3/5] CatBoost ━━━"
python run_pipeline_catboost.py --results "$EXP_DIR/catboost" $GPU_FLAG $EXTRA_ARGS \
    2>&1 | tee "$LOG_DIR/catboost.log"
echo ""

# ── 4. Deriv-only mini-model ─────────────────────────────────
echo "━━━ [4/5] Derivatives-only model ━━━"
python run_pipeline_derivatives.py --results "$EXP_DIR/deriv_only" --skip-hpo $EXTRA_ARGS \
    2>&1 | tee "$LOG_DIR/deriv_only.log"
echo ""

# ── 5. XGBoost ───────────────────────────────────────────────
echo "━━━ [5/5] XGBoost ━━━"
python run_pipeline_xgboost.py --results "$EXP_DIR/xgboost" $GPU_FLAG $EXTRA_ARGS \
    2>&1 | tee "$LOG_DIR/xgboost.log"
echo ""

# ── Summary ───────────────────────────────────────────────────
echo "============================================================"
echo "  ✅ Experiment 15 complete!"
echo ""
echo "  Results:"
for d in v6 v7 catboost deriv_only xgboost; do
    if [[ -d "$EXP_DIR/$d" ]]; then
        n_models=$(ls "$EXP_DIR/$d"/*.txt 2>/dev/null | wc -l | tr -d ' ')
        echo "    $EXP_DIR/$d/ ($n_models models)"
    fi
done
echo ""
echo "  Logs:"
ls -la "$LOG_DIR"/*.log 2>/dev/null | awk '{print "    "$NF" ("$5" bytes)"}'
echo ""
echo "  Next steps:"
echo "    1. Check Sharpe/ICIR in logs above"
echo "    2. scp -r $EXP_DIR/ local:invest/results/"
echo "    3. python run_meta_stack.py --save-model  (locally)"
echo "============================================================"
