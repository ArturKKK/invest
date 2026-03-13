#!/bin/bash
# ============================================================
# A/B test: Calendar features impact
#
# Compares Gen#3 (WITH calendar) vs retrained (WITHOUT calendar)
# Same train/val dates: train→2026-02-01, val→2026-02-09..03-07
# ============================================================
set -e

echo "============================================================"
echo "  A/B TEST: Calendar Features Impact"
echo "============================================================"

SIM_CMD="python run_fast_sim.py --data data/features/crypto_features_1h.parquet \
  --days 120 --start-date 2026-02-09 --end-date 2026-03-07 \
  --leverage 3 --kelly 0.8 --ensemble --edge-boost \
  --no-deriv-gate --no-ddstop"

# ── Step 1: Back up Gen#3 (with calendar) ──────────────────
echo ""
echo "📦 Step 1: Backing up Gen#3 models..."
for d in results_v6_prod results_v7_prod results_catboost_prod results_xgboost_prod; do
  if [ -d "$d" ]; then
    cp -r "$d" "${d}_gen3_cal"
    echo "   ✅ $d → ${d}_gen3_cal"
  fi
done

# ── Step 2: Retrain WITHOUT calendar ───────────────────────
echo ""
echo "🔧 Step 2: Retraining 4 GBDT models WITHOUT calendar..."
echo "   (SKIP_CALENDAR=1 disables add_calendar_features)"

export SKIP_CALENDAR=1

echo ""
echo "━━━ v6 LGB ━━━"
python run_pipeline_v6.py --production --train-end 2026-02-01 --val-end 2026-03-07

echo ""
echo "━━━ v7 LGB ━━━"
python run_pipeline_v7.py --production --train-end 2026-02-01 --val-end 2026-03-07

echo ""
echo "━━━ CatBoost ━━━"
python run_pipeline_catboost.py --production --train-end 2026-02-01 --val-end 2026-03-07

echo ""
echo "━━━ XGBoost ━━━"
python run_pipeline_xgboost.py --production --train-end 2026-02-01 --val-end 2026-03-07

unset SKIP_CALENDAR

# ── Step 3: Sim WITHOUT calendar ──────────────────────────
echo ""
echo "📊 Step 3: Running sim WITHOUT calendar (4 GBDT groups)..."
# Hide MLP (we only test GBDT)
[ -d results_mlp_prod ] && mv results_mlp_prod results_mlp_prod_bak

export SKIP_CALENDAR=1
eval $SIM_CMD 2>&1 | tee /tmp/ab_no_calendar.txt
unset SKIP_CALENDAR

[ -d results_mlp_prod_bak ] && mv results_mlp_prod_bak results_mlp_prod

# Save the no-cal equity curve
cp trading_logs/fast_sim_equity.csv /tmp/equity_no_calendar.csv

# ── Step 4: Restore Gen#3 + Sim WITH calendar ─────────────
echo ""
echo "📊 Step 4: Restoring Gen#3 and running sim WITH calendar..."
for d in results_v6_prod results_v7_prod results_catboost_prod results_xgboost_prod; do
  if [ -d "${d}_gen3_cal" ]; then
    rm -rf "$d"
    mv "${d}_gen3_cal" "$d"
    echo "   ✅ restored $d"
  fi
done

# Hide MLP for fair comparison
[ -d results_mlp_prod ] && mv results_mlp_prod results_mlp_prod_bak

eval $SIM_CMD 2>&1 | tee /tmp/ab_with_calendar.txt

[ -d results_mlp_prod_bak ] && mv results_mlp_prod_bak results_mlp_prod

# Save the cal equity curve
cp trading_logs/fast_sim_equity.csv /tmp/equity_with_calendar.csv

# ── Step 5: Compare ───────────────────────────────────────
echo ""
echo "============================================================"
echo "  COMPARISON"
echo "============================================================"
echo ""
echo "--- WITHOUT calendar ---"
grep -E "Return:|Sharpe:|Max DD:|PF:" /tmp/ab_no_calendar.txt | head -5
echo ""
echo "--- WITH calendar ---"
grep -E "Return:|Sharpe:|Max DD:|PF:" /tmp/ab_with_calendar.txt | head -5
echo ""
echo "============================================================"
echo "Done. Full logs: /tmp/ab_no_calendar.txt, /tmp/ab_with_calendar.txt"
echo "Equity curves: /tmp/equity_no_calendar.csv, /tmp/equity_with_calendar.csv"
