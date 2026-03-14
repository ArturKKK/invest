#!/bin/bash
# ============================================================
# OVERNIGHT RESEARCH v2 — LambdaRank + Residual Target + Meta-Label
#
# Three high-impact experiments from quant consultation:
#   EXP A: LambdaRank (ranking objective instead of MSE)
#   EXP B: Residual target (ret - beta×BTC)
#   EXP C: Meta-labeling (binary filter for marginal trades)
#
# All models trained with SKIP_CALENDAR=1 (calendar hurts per v1 results)
# Baseline: Gen#3 no-calendar 4-group + 24h = Sharpe 7.10
#
# Total time estimate: ~3-4 hours on GPU cluster
# ============================================================
set -euo pipefail

TRAIN_END="2026-02-01"
VAL_END="2026-03-07"

SIM_BASE="python run_fast_sim.py --data data/features/crypto_features_1h.parquet \
  --days 120 --start-date 2026-02-09 --end-date 2026-03-07 \
  --leverage 3 --kelly 0.8 --ensemble --edge-boost \
  --no-deriv-gate --no-ddstop"

RESULTS_DIR="overnight_v2_results"
mkdir -p $RESULTS_DIR

# Safety: ensure SKIP_CALENDAR is on
export SKIP_CALENDAR=1

# Safety trap — restore production models on any failure
cleanup() {
  echo ""
  echo "⚠️  Script interrupted or failed! Cleaning up..."
  # Restore any backed-up prod models
  for d in results_v6_prod results_v7_prod; do
    [ -d "${d}_pre_overnight_v2" ] && rm -rf "$d" && mv "${d}_pre_overnight_v2" "$d" && echo "   restored $d"
  done
  # Restore hidden 24h dir
  [ -d results_v6_24h_prod_hidden ] && mv results_v6_24h_prod_hidden results_v6_24h_prod && echo "   restored results_v6_24h_prod"
  # Remove experiment-specific dirs (safe to regen)
  for d in results_v6_rank_prod results_v7_rank_prod results_v6_resid_prod results_v7_resid_prod; do
    [ -d "$d" ] && echo "   leaving $d (experiment output)"
  done
  unset SKIP_CALENDAR 2>/dev/null || true
  echo "   cleanup done."
}
trap cleanup ERR INT TERM

echo "============================================================"
echo "  OVERNIGHT RESEARCH v2 — $(date)"
echo "  LambdaRank + Residual + Meta-Label"
echo "  Train→${TRAIN_END}, Val→${VAL_END}, SKIP_CALENDAR=1"
echo "============================================================"

# ============================================================
# EXPERIMENT A: LambdaRank (ranking objective)
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  EXP A: LambdaRank — LGBMRanker with NDCG objective"
echo "  Hypothesis: ranking loss → better top/bottom selection → higher Sharpe"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# A1: Train v6 with LambdaRank
echo ""
echo "🏋️ A1: Training v6 LambdaRank..."
python run_pipeline_v6.py \
  --production --train-end $TRAIN_END --val-end $VAL_END \
  --lambdarank \
  --results results_v6_rank_prod \
  2>&1 | tee $RESULTS_DIR/exp_a1_v6_rank_train.log

echo ""
echo "🏋️ A2: Training v7 LambdaRank..."
python run_pipeline_v7.py \
  --production --train-end $TRAIN_END --val-end $VAL_END \
  --lambdarank \
  --results results_v7_rank_prod \
  2>&1 | tee $RESULTS_DIR/exp_a2_v7_rank_train.log

# A3: Sim — 4-group with LambdaRank v6+v7 replacing original v6+v7
echo ""
echo "📊 A3: Sim with LambdaRank v6+v7 (4-group: rank_v6 + rank_v7 + cb + xgb)..."

# Temporarily swap v6/v7 with ranked versions
cp -r results_v6_prod results_v6_prod_pre_overnight_v2
cp -r results_v7_prod results_v7_prod_pre_overnight_v2
cp -r results_v6_rank_prod/* results_v6_prod/
cp -r results_v7_rank_prod/* results_v7_prod/

# Hide 24h group to get pure 4-group sim
[ -d results_v6_24h_prod ] && mv results_v6_24h_prod results_v6_24h_prod_hidden

$SIM_BASE 2>&1 | tee $RESULTS_DIR/exp_a3_sim_rank_4grp.log

# A4: Sim — 5-group (rank v6+v7 + cb + xgb + 24h)
echo ""
echo "📊 A4: Sim with LambdaRank 5-group (+ 24h)..."
# Restore 24h group
[ -d results_v6_24h_prod_hidden ] && mv results_v6_24h_prod_hidden results_v6_24h_prod

$SIM_BASE 2>&1 | tee $RESULTS_DIR/exp_a4_sim_rank_5grp.log

# Restore original v6/v7
rm -rf results_v6_prod results_v7_prod
mv results_v6_prod_pre_overnight_v2 results_v6_prod
mv results_v7_prod_pre_overnight_v2 results_v7_prod

echo "✅ EXP A done."

# ============================================================
# EXPERIMENT B: Residual Target (ret - beta×BTC)
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  EXP B: Residual Target — remove BTC beta before training"
echo "  Hypothesis: predicting residual → less correlated → better diversity"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# B1: Train v6 with residual target
echo ""
echo "🏋️ B1: Training v6 residual-target..."
python run_pipeline_v6.py \
  --production --train-end $TRAIN_END --val-end $VAL_END \
  --residual-target \
  --results results_v6_resid_prod \
  2>&1 | tee $RESULTS_DIR/exp_b1_v6_resid_train.log

echo ""
echo "🏋️ B2: Training v7 residual-target..."
python run_pipeline_v7.py \
  --production --train-end $TRAIN_END --val-end $VAL_END \
  --residual-target \
  --results results_v7_resid_prod \
  2>&1 | tee $RESULTS_DIR/exp_b2_v7_resid_train.log

# B3: Sim — 4-group with residual v6+v7
echo ""
echo "📊 B3: Sim with residual v6+v7 (4-group)..."
cp -r results_v6_prod results_v6_prod_pre_overnight_v2
cp -r results_v7_prod results_v7_prod_pre_overnight_v2
cp -r results_v6_resid_prod/* results_v6_prod/
cp -r results_v7_resid_prod/* results_v7_prod/

# Hide 24h group to get pure 4-group sim
[ -d results_v6_24h_prod ] && mv results_v6_24h_prod results_v6_24h_prod_hidden

$SIM_BASE 2>&1 | tee $RESULTS_DIR/exp_b3_sim_resid_4grp.log

# B4: Sim — 5-group (resid + 24h)
echo ""
echo "📊 B4: Sim with residual 5-group (+24h)..."
# Restore 24h group
[ -d results_v6_24h_prod_hidden ] && mv results_v6_24h_prod_hidden results_v6_24h_prod

$SIM_BASE 2>&1 | tee $RESULTS_DIR/exp_b4_sim_resid_5grp.log

# Restore
rm -rf results_v6_prod results_v7_prod
mv results_v6_prod_pre_overnight_v2 results_v6_prod
mv results_v7_prod_pre_overnight_v2 results_v7_prod

echo "✅ EXP B done."

# ============================================================
# EXPERIMENT C: Meta-Labeling (binary trade filter)
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  EXP C: Meta-Labeling — P(profitable) filter"
echo "  Hypothesis: skip marginal trades → WR 63% → 68-70%"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "🏋️ C1: Training meta-label classifier (threshold sweep)..."
python run_train_meta_label.py \
  --sweep \
  --n-pos 5 \
  --cost-bps 8 \
  --leverage 3 \
  --train-end $TRAIN_END \
  --output-dir $RESULTS_DIR/meta_label \
  2>&1 | tee $RESULTS_DIR/exp_c1_meta_label_train.log

echo "✅ EXP C done."

# ============================================================
# EXPERIMENT D: Combo — LambdaRank + Residual + 24h (best of A+B)
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  EXP D: Combo — LambdaRank v6 + Residual v7 + original cb/xgb + 24h"
echo "  Hypothesis: orthogonal objectives → maximum diversity"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "📊 D1: Sim with mixed objectives (rank_v6 + resid_v7 + cb + xgb + 24h)..."
cp -r results_v6_prod results_v6_prod_pre_overnight_v2
cp -r results_v7_prod results_v7_prod_pre_overnight_v2

# v6 → lambdarank, v7 → residual
cp -r results_v6_rank_prod/* results_v6_prod/
cp -r results_v7_resid_prod/* results_v7_prod/

$SIM_BASE 2>&1 | tee $RESULTS_DIR/exp_d1_sim_combo_5grp.log

# Restore
rm -rf results_v6_prod results_v7_prod
mv results_v6_prod_pre_overnight_v2 results_v6_prod
mv results_v7_prod_pre_overnight_v2 results_v7_prod

echo "✅ EXP D done."

# ============================================================
# ANALYSIS: IC + correlation across all model variants
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ANALYSIS: Cross-model IC & correlation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "📊 Running model comparison analysis..."
python _analyze_overnight_v2.py \
  --output $RESULTS_DIR/analysis_report.json \
  2>&1 | tee $RESULTS_DIR/analysis.log

echo ""
echo "============================================================"
echo "  OVERNIGHT RESEARCH v2 COMPLETE — $(date)"
echo "============================================================"
echo ""
echo "Results saved to: $RESULTS_DIR/"
echo ""
echo "Key files:"
echo "  - exp_a3_sim_rank_4grp.log    LambdaRank 4-group sim"
echo "  - exp_a4_sim_rank_5grp.log    LambdaRank 5-group sim"
echo "  - exp_b3_sim_resid_4grp.log   Residual 4-group sim"
echo "  - exp_b4_sim_resid_5grp.log   Residual 5-group sim"
echo "  - exp_c1_meta_label_train.log Meta-label training + threshold sweep"
echo "  - exp_d1_sim_combo_5grp.log   Combo (rank+resid) 5-group sim"
echo "  - analysis_report.json        IC/correlation analysis"
echo ""
echo "Quick comparison: grep 'Sharpe\|Return\|Win Rate\|Max DD' $RESULTS_DIR/exp_*_sim_*.log"

# Final cleanup
unset SKIP_CALENDAR 2>/dev/null || true
trap - ERR INT TERM
