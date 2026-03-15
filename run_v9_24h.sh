#!/bin/bash
set -euo pipefail

# ============================================================
# v9 — Multi-Horizon Ensemble: 24h Model Evaluation
# ============================================================
#
# We have a trained 24h-horizon LGB v6 model (results_v6_24h_prod)
# that was NEVER tested in backtest. This experiment evaluates:
#
#   A: Baseline — current 4-model ensemble (v6+v7+CB+XGB, all 12h)
#   B: 5-model — add 24h model as 5th ensemble member
#   C: 24h solo — 24h model standalone (sanity check)
#   D: 6-model — add both 4h and 24h models
#
# The 24h model has 167 features (155 overlap with 12h).
# run_fast_sim.py already supports multi-horizon models via
# glob("results_v6_*h_prod") — we just need to ensure dirs exist.
#
# NO RETRAINING needed — models are already trained.
# Expected runtime: ~5 min (4 sims on cached features)
# ============================================================

RESULTS_DIR="results_v9_24h"
mkdir -p $RESULTS_DIR

DATA="data/features/crypto_features_1h.parquet"
if [ ! -f "$DATA" ]; then
  echo "❌ Feature file not found: $DATA"
  echo "   Run: python run_fast_sim.py --days 120 first to generate it"
  exit 1
fi

SIM_BASE="python run_fast_sim.py --data $DATA \
  --days 120 --start-date 2026-02-09 --end-date 2026-03-07 \
  --leverage 3 --kelly 0.8 --ensemble --edge-boost \
  --no-deriv-gate --no-ddstop --min-zscore 0.5"

export SKIP_CALENDAR=1

echo "============================================================"
echo "  v9 — Multi-Horizon Ensemble (24h model)"
echo "  Started: $(date)"
echo "============================================================"

# Verify models exist
echo ""
echo "📦 Model inventory:"
for d in results_v6_prod results_v7_prod results_catboost_prod results_xgboost_prod results_v6_24h_prod results_v6_4h_prod; do
  if [ -d "$d" ]; then
    n=$(ls "$d"/*.txt "$d"/*.cbm "$d"/*.json 2>/dev/null | grep -c 'seed' || true)
    echo "   ✅ $d ($n seeds)"
  else
    echo "   ❌ $d — MISSING"
  fi
done

echo ""

# ============================================================
# Sim A: Baseline — 4-model ensemble (no multi-horizon)
# ============================================================
# Temporarily hide multi-horizon dirs so run_fast_sim doesn't auto-load them

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Sim A: Baseline 4-model (v6+v7+CB+XGB, 12h)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Hide multi-horizon dirs
for d in results_v6_4h_prod results_v6_24h_prod; do
  [ -d "$d" ] && mv "$d" ".${d}_hidden_v9"
done

$SIM_BASE 2>&1 | tee $RESULTS_DIR/A_baseline_4model.log || true

# Restore
for d in results_v6_4h_prod results_v6_24h_prod; do
  [ -d ".${d}_hidden_v9" ] && mv ".${d}_hidden_v9" "$d"
done

echo ""

# ============================================================
# Sim B: 5-model — baseline + 24h model
# ============================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Sim B: 5-model (v6+v7+CB+XGB+v6_24h)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Hide 4h, keep 24h visible
[ -d "results_v6_4h_prod" ] && mv results_v6_4h_prod .results_v6_4h_prod_hidden_v9

$SIM_BASE 2>&1 | tee $RESULTS_DIR/B_5model_with_24h.log || true

[ -d ".results_v6_4h_prod_hidden_v9" ] && mv .results_v6_4h_prod_hidden_v9 results_v6_4h_prod

echo ""

# ============================================================
# Sim C: 24h model solo (5 seeds, sanity check)
# ============================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Sim C: 24h model solo"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Use single-model mode pointing to 24h dir
python run_fast_sim.py --data $DATA \
  --days 120 --start-date 2026-02-09 --end-date 2026-03-07 \
  --leverage 3 --kelly 0.8 --edge-boost \
  --no-deriv-gate --no-ddstop --min-zscore 0.5 \
  --model-dir results_v6_24h_prod \
  2>&1 | tee $RESULTS_DIR/C_24h_solo.log || true

echo ""

# ============================================================
# Sim D: 6-model — all horizons (v6+v7+CB+XGB+v6_4h+v6_24h)
# ============================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Sim D: 6-model (v6+v7+CB+XGB+v6_4h+v6_24h)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Both multi-horizon dirs visible → run_fast_sim globs them
$SIM_BASE 2>&1 | tee $RESULTS_DIR/D_6model_all_horizons.log || true

echo ""

# ============================================================
# Summary
# ============================================================

echo "============================================================"
echo "  v9 COMPLETE — $(date)"
echo "============================================================"
echo ""
echo "Results:"
echo "  A: A_baseline_4model.log       — 4-model 12h (reference HAC≈8.14)"
echo "  B: B_5model_with_24h.log       — + 24h model (5th member)"
echo "  C: C_24h_solo.log              — 24h model alone (sanity)"
echo "  D: D_6model_all_horizons.log   — + 4h + 24h (6 models)"
echo ""
echo "Key questions:"
echo "  B > A? → 24h horizon adds orthogonal alpha"
echo "  C decent? → 24h model is not garbage"
echo "  D > B? → 4h also adds value"
echo ""
echo "If B > A: add 24h to production ensemble (Gen#5)"
echo ""
echo "Logs in: $RESULTS_DIR/"
