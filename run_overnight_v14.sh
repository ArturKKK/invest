#!/bin/bash
set -uo pipefail

# ============================================================
# OVERNIGHT v14 — CatBoost is King
# ============================================================
#
# v13 KEY FINDINGS:
#   - cb_no_deriv SOLO = +131.5% (FULL period), absolute best
#   - cb_price_only (no news) = +103.7% — news ADD +27.8%
#   - Ensembles consistently WORSE than best solo CB
#   - DDstop does nothing, edge-boost helps ~1-2pp
#   - cb_market_no_deriv trained Sharpe=1.84 but no solo sim yet
#   - Training Sharpe does NOT predict sim performance
#     (1.66 Sharpe → +131.5% vs 1.78 Sharpe → +103.7%)
#
# v14 PLAN:
#   Phase 1: Training — CatBoost variations (6 experiments)
#     1. cb_no_deriv + HPO (v12 was skip-hpo, maybe HPO improves?)
#     2. cb_no_deriv + residual target
#     3. cb_no_deriv + huber delta variants (0.5, 1.5)
#     4. cb_all_features + HPO (re-baseline with HPO)
#     5. cb_market_news_no_deriv + HPO
#
#   Phase 2: Sim grid — all new + existing CB models on FULL (15+ sims)
#     Solo sim for each trained model + cb_market_no_deriv from v13
#     Compare to v12_cbnd benchmark (+131.5%)
#
#   Phase 3: Best model on R1, R2, FULL separately (stability check)
#
# Expected runtime: ~3-4h (5 CB trains ~20min each + sims ~3min each)
#
# Usage:
#   nohup ./run_overnight_v14.sh > overnight_v14.log 2>&1 &
# ============================================================

LOGDIR="results/overnight_v14"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG="$LOGDIR/run_${TIMESTAMP}.log"
SUMMARY="$LOGDIR/summary_${TIMESTAMP}.txt"

GPU="--gpu"
SEEDS="--seeds 5"
COMMON="--research $SEEDS"
HUBER="--huber"

SIM_DATA="--data data/features/crypto_features_1h.parquet"

export SKIP_CALENDAR=1
START_TIME=$(date +%s)

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

phase_start() {
  PHASE_T0=$(date +%s)
  log ""
  log "============================================================"
  log "  $*"
  log "============================================================"
}
phase_end() {
  local elapsed=$(( $(date +%s) - PHASE_T0 ))
  log "  ⏱️  Phase took $(( elapsed / 60 ))m $(( elapsed % 60 ))s"
}

# ─── Sim helper ───
SIM_N=0
run_sim() {
  local label="$1"
  shift
  SIM_N=$((SIM_N + 1))
  log "SIM #${SIM_N}: $label"
  log "CMD: python run_fast_sim.py $SIM_DATA $*"

  local output
  output=$(python run_fast_sim.py $SIM_DATA "$@" 2>&1) || true
  echo "$output" >> "$LOG"

  echo "=== SIM #${SIM_N}: $label ===" >> "$SUMMARY"
  echo "$output" | grep -E "Return:|Max DD:|Sharpe|Calmar:|Win Rate:" | head -6 >> "$SUMMARY"
  echo "" >> "$SUMMARY"

  local ret hac
  ret=$(echo "$output" | grep "Return:" | head -1 | awk '{print $2}')
  hac=$(echo "$output" | grep "Sharpe HAC:" | awk '{print $3}')
  log "  => Return=$ret  HAC=$hac"
}

# ─── Model swap helper ───
MODEL_DIRS=(
  results_v6_prod results_v7_prod results_catboost_prod results_xgboost_prod
  results_v6 results_v7 results_catboost results_xgboost
)

isolate_models() {
  for d in "${MODEL_DIRS[@]}"; do
    [[ -d "$d" ]] && mv "$d" "${d}_bak_v14"
  done
  [[ -d "results/production" ]] && mv "results/production" "results/production_bak_v14"
  for d in results_v6_*h_prod; do
    [[ -d "$d" ]] && mv "$d" "${d}_bak_v14"
  done
}

restore_models() {
  for d in "${MODEL_DIRS[@]}"; do
    rm -rf "$d" 2>/dev/null
    [[ -d "${d}_bak_v14" ]] && mv "${d}_bak_v14" "$d"
  done
  [[ -d "results/production_bak_v14" ]] && mv "results/production_bak_v14" "results/production"
  for d in results_v6_*h_prod_bak_v14; do
    [[ -d "$d" ]] && mv "$d" "${d%_bak_v14}"
  done
}

setup_models() {
  # Args: v6_src v7_src cb_src xgb_src
  local v6_src="$1" v7_src="$2" cb_src="$3" xgb_src="$4"
  for d in results_v6_prod results_v7_prod results_catboost_prod results_xgboost_prod; do
    rm -rf "$d" 2>/dev/null
  done
  for pair in "v6:$v6_src" "v7:$v7_src" "catboost:$cb_src" "xgboost:$xgb_src"; do
    IFS=: read -r suffix src <<< "$pair"
    if [[ "$src" != "SKIP" && -d "$src" ]]; then
      cp -r "$src" "results_${suffix}_prod"
    fi
  done
}

# ─── Training helper ───
EXP_N=0
train_experiment() {
  local label="$1"
  local script="$2"
  shift 2
  local extra_args="$*"
  EXP_N=$((EXP_N + 1))
  local results_dir="${label}"
  log ""
  log "────────────────────────────────────"
  log "EXP #${EXP_N}: $label"
  log "CMD: python $script $COMMON --results $results_dir $extra_args"
  log "────────────────────────────────────"
  local t0=$(date +%s)
  python "$script" $COMMON --results "$results_dir" $extra_args 2>&1 | tee -a "$LOG"
  local rc=$?
  local elapsed=$(( $(date +%s) - t0 ))
  if [[ $rc -eq 0 ]]; then
    log "  ✅ #${EXP_N} done in $(( elapsed / 60 ))m $(( elapsed % 60 ))s"
  else
    log "  ❌ #${EXP_N} FAILED (rc=$rc) after $(( elapsed / 60 ))m"
  fi
  echo "=== #${EXP_N}: $label ===" >> "$SUMMARY"
  local results_json=""
  for jname in all_results_v6.json all_results_v7.json all_results_catboost.json all_results_xgboost.json; do
    if [[ -f "$results_dir/$jname" ]]; then
      results_json="$results_dir/$jname"
      break
    fi
  done
  if [[ -n "$results_json" ]]; then
    python -c "
import json; d = json.load(open('$results_json')); avg = d.get('average', {})
print(f\"  ICIR={avg.get('Rank_ICIR','?')}  Sharpe={avg.get('LS_Sharpe_net','?')}  DD={avg.get('LS_MaxDD_net_%','?')}%  Ret={avg.get('LS_Total_net_%','?')}%\")
" 2>/dev/null >> "$SUMMARY"
  fi
  echo "" >> "$SUMMARY"
}

# ─── Reference model paths ───
# v12 champion (cb_no_deriv, Sharpe 1.66, sim +131.5%)
V12_CBND="results/overnight_v12/cb_no_deriv"
# v12 cb_price_only (Sharpe 1.78, sim +103.7%)
V12_CBPO="results/overnight_v12/cb_price_only"
# v13 cb_market_no_deriv (Sharpe 1.84, no solo sim yet)
V13_CBMKT="results/overnight_v13/cb_market_no_deriv"
# v11 baselines (for comparison)
V11_V6="results_v6_research"
V11_CB="results_catboost_research"

# Verify reference dirs
for d in "$V12_CBND" "$V12_CBPO" "$V13_CBMKT"; do
  if [[ ! -d "$d" ]]; then
    log "⚠️  Missing reference dir: $d"
  fi
done

# Isolate all model dirs to prevent fallback contamination
isolate_models

# Sim periods
R1="--days 92 --start-date 2025-10-01 --end-date 2025-12-31"
R2="--days 66 --start-date 2026-01-01 --end-date 2026-03-07"
FULL="--days 158 --start-date 2025-10-01 --end-date 2026-03-07"

# ╔══════════════════════════════════════════════════════════════╗
# ║  PHASE 1: CATBOOST TRAINING VARIATIONS (~2h)                 ║
# ║  All based on cb_no_deriv (the v13 sim champion)             ║
# ╚══════════════════════════════════════════════════════════════╝
phase_start "PHASE 1: CATBOOST TRAINING (~2h)"

# EXP 1: cb_no_deriv + HPO
# v12's cb_no_deriv was skip-hpo. HPO might find better params
train_experiment "$LOGDIR/cb_noderiv_hpo" \
  run_pipeline_catboost.py "$GPU $HUBER --no-derivatives --hpo-trials 50"

# EXP 2: cb_no_deriv + residual target
# Residual target predicts alpha over cross-sectional mean
train_experiment "$LOGDIR/cb_noderiv_residual" \
  run_pipeline_catboost.py "$GPU $HUBER --no-derivatives --residual-target --skip-hpo"

# EXP 3: cb_no_deriv + huber delta=0.5 (more robust to outliers)
train_experiment "$LOGDIR/cb_noderiv_hd05" \
  run_pipeline_catboost.py "$GPU $HUBER --no-derivatives --huber-delta 0.5 --skip-hpo"

# EXP 4: cb_no_deriv + huber delta=1.5 (less robust, closer to MSE)
train_experiment "$LOGDIR/cb_noderiv_hd15" \
  run_pipeline_catboost.py "$GPU $HUBER --no-derivatives --huber-delta 1.5 --skip-hpo"

# EXP 5: cb ALL features + HPO (full re-baseline — does HPO help even WITH derivs?)
train_experiment "$LOGDIR/cb_all_hpo" \
  run_pipeline_catboost.py "$GPU $HUBER --hpo-trials 50"

# EXP 6: cb market-only news + no-derivs + HPO
# v13's cb_market_no_deriv (skip-hpo) got 1.84 Sharpe. HPO may push higher.
train_experiment "$LOGDIR/cb_market_noderiv_hpo" \
  run_pipeline_catboost.py "$GPU $HUBER --news-mode market-only --no-derivatives --hpo-trials 50"

phase_end

# ╔══════════════════════════════════════════════════════════════╗
# ║  PHASE 2: SIM GRID — FULL PERIOD (~1h)                      ║
# ║  Solo sim for every CB model, compare to v12_cbnd benchmark  ║
# ╚══════════════════════════════════════════════════════════════╝
phase_start "PHASE 2: SIM GRID on FULL period (~1h)"

SIM_FLAGS="--leverage 3 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble"

# ── Reference benchmarks (reproduce v13 results) ──
# v12 cb_no_deriv solo (the champion: +131.5%)
setup_models "SKIP" "SKIP" "$V12_CBND" "SKIP"
run_sim "REF_v12_cbnd_solo" $FULL $SIM_FLAGS

# v12 cb_price_only solo (+103.7%)
setup_models "SKIP" "SKIP" "$V12_CBPO" "SKIP"
run_sim "REF_v12_cbpo_solo" $FULL $SIM_FLAGS

# v13 cb_market_no_deriv solo (untested!)
if [[ -d "$V13_CBMKT" ]]; then
  setup_models "SKIP" "SKIP" "$V13_CBMKT" "SKIP"
  run_sim "REF_v13_cbmkt_solo" $FULL $SIM_FLAGS
fi

# v11 cb baseline solo
setup_models "SKIP" "SKIP" "$V11_CB" "SKIP"
run_sim "REF_v11_cb_solo" $FULL $SIM_FLAGS

# ── New models from Phase 1 ──
for exp in cb_noderiv_hpo cb_noderiv_residual cb_noderiv_hd05 cb_noderiv_hd15 cb_all_hpo cb_market_noderiv_hpo; do
  if [[ -d "$LOGDIR/$exp" ]]; then
    setup_models "SKIP" "SKIP" "$LOGDIR/$exp" "SKIP"
    run_sim "NEW_${exp}" $FULL $SIM_FLAGS
  else
    log "⚠️  Skipping sim for $exp — dir not found"
  fi
done

phase_end

# ╔══════════════════════════════════════════════════════════════╗
# ║  PHASE 3: STABILITY CHECK — best across R1, R2, FULL        ║
# ║  Take top 2 from Phase 2 and test on each period separately  ║
# ╚══════════════════════════════════════════════════════════════╝
phase_start "PHASE 3: STABILITY CHECK (~30min)"

# Always test these known-good models + top Phase 1 candidates
# We'll test the reference champion and HPO variant for stability

# v12 cb_no_deriv on R1 and R2
setup_models "SKIP" "SKIP" "$V12_CBND" "SKIP"
run_sim "STAB_v12_cbnd_R1" $R1 $SIM_FLAGS
run_sim "STAB_v12_cbnd_R2" $R2 $SIM_FLAGS

# cb_noderiv_hpo on R1, R2, FULL
if [[ -d "$LOGDIR/cb_noderiv_hpo" ]]; then
  setup_models "SKIP" "SKIP" "$LOGDIR/cb_noderiv_hpo" "SKIP"
  run_sim "STAB_cb_noderiv_hpo_R1" $R1 $SIM_FLAGS
  run_sim "STAB_cb_noderiv_hpo_R2" $R2 $SIM_FLAGS
fi

# cb_all_hpo on R1, R2 (does HPO + all features beat no-deriv?)
if [[ -d "$LOGDIR/cb_all_hpo" ]]; then
  setup_models "SKIP" "SKIP" "$LOGDIR/cb_all_hpo" "SKIP"
  run_sim "STAB_cb_all_hpo_R1" $R1 $SIM_FLAGS
  run_sim "STAB_cb_all_hpo_R2" $R2 $SIM_FLAGS
fi

# cb_market_noderiv_hpo on R1, R2
if [[ -d "$LOGDIR/cb_market_noderiv_hpo" ]]; then
  setup_models "SKIP" "SKIP" "$LOGDIR/cb_market_noderiv_hpo" "SKIP"
  run_sim "STAB_cb_mkt_hpo_R1" $R1 $SIM_FLAGS
  run_sim "STAB_cb_mkt_hpo_R2" $R2 $SIM_FLAGS
fi

phase_end

# ╔══════════════════════════════════════════════════════════════╗
# ║  PHASE 4: LEVERAGE SENSITIVITY for top model (~15min)        ║
# ╚══════════════════════════════════════════════════════════════╝
phase_start "PHASE 4: LEVERAGE SENSITIVITY (~15min)"

# v12 cbnd champion across leverage levels
setup_models "SKIP" "SKIP" "$V12_CBND" "SKIP"
run_sim "LEV_cbnd_lev2" $FULL --leverage 2 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble
run_sim "LEV_cbnd_lev1" $FULL --leverage 1 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble

# cb_noderiv_hpo if it exists
if [[ -d "$LOGDIR/cb_noderiv_hpo" ]]; then
  setup_models "SKIP" "SKIP" "$LOGDIR/cb_noderiv_hpo" "SKIP"
  run_sim "LEV_hpo_lev2" $FULL --leverage 2 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble
  run_sim "LEV_hpo_lev1" $FULL --leverage 1 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble
fi

phase_end

# ╔══════════════════════════════════════════════════════════════╗
# ║  FINAL SUMMARY                                               ║
# ╚══════════════════════════════════════════════════════════════╝
phase_start "FINAL SUMMARY"

# Restore all model dirs
restore_models

log ""
log "============================================================"
log "  OVERNIGHT v14 — FINAL RESULTS"
log "============================================================"

# Print all sim results in order
{
  echo ""
  echo "=== ALL SIM RESULTS ==="
  echo ""
  printf "%-35s %10s %10s\n" "Config" "Return" "HAC"
  echo "────────────────────────────────────────────────────────────"
} >> "$SUMMARY"

grep -E 'SIM #|=> Return' "$LOG" | paste - - | while read -r line; do
  label=$(echo "$line" | sed 's/.*SIM #[0-9]*: //' | sed 's/\[.*//' | tr -d '\n')
  ret=$(echo "$line" | grep -o 'Return=[^ ]*' | head -1 | cut -d= -f2)
  hac=$(echo "$line" | grep -o 'HAC=[^ ]*' | head -1 | cut -d= -f2)
  printf "%-35s %10s %10s\n" "$label" "$ret" "$hac"
done | tee -a "$LOG" >> "$SUMMARY"

# Training comparison
{
  echo ""
  echo "=== TRAINING RESULTS ==="
  echo ""
} >> "$SUMMARY"

for exp in cb_noderiv_hpo cb_noderiv_residual cb_noderiv_hd05 cb_noderiv_hd15 cb_all_hpo cb_market_noderiv_hpo; do
  local_json="$LOGDIR/$exp/all_results_catboost.json"
  if [[ -f "$local_json" ]]; then
    python -c "
import json; d = json.load(open('$local_json'))
avg = d.get('average', {})
pw = d.get('per_window', {})
r1_s = 'N/A'; r2_s = 'N/A'
for wname, wdata in pw.items():
    s = wdata.get('LS_Sharpe_net', 'N/A')
    if '2025' in wname or 'R1' in wname:
        r1_s = s
    elif '2026' in wname or 'R2' in wname:
        r2_s = s
avg_s = avg.get('LS_Sharpe_net', 'N/A')
print(f'  {\"$exp\":<30s}  R1={r1_s:<6}  R2={r2_s:<6}  avg={avg_s}')
" 2>/dev/null >> "$SUMMARY"
  fi
done

TOTAL_TIME=$(( $(date +%s) - START_TIME ))
log ""
log "============================================================"
log "  TOTAL RUNTIME: $(( TOTAL_TIME / 3600 ))h $(( (TOTAL_TIME % 3600) / 60 ))m"
log "  SUMMARY: $SUMMARY"
log "============================================================"
log ""
log "Key questions to answer:"
log "  1. Does HPO improve cb_no_deriv (skip-hpo → hpo)?"
log "  2. Does residual target or huber delta help?"
log "  3. cb_all_hpo (with derivs+HPO) vs cb_noderiv — final answer on derivs"
log "  4. market-only vs all-news: which news mode is better for CB?"
log "  5. Is cb_no_deriv consistently best across R1, R2, FULL?"
log "  6. Optimal leverage for production deployment"
