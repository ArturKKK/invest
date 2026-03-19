#!/bin/bash
set -uo pipefail

# ============================================================
# OVERNIGHT v15 — Execution Layer Optimization
# ============================================================
#
# ZERO new training. Pure sim grid on execution flags.
#
# v14 FINDINGS:
#   - NEW CHAMPION: cb_market_noderiv_hpo = +143.8% HAC 5.33
#   - Old champion: v12 cb_no_deriv = +131.5% HAC 5.09
#   - v13 cb_market_no_deriv (skip-hpo) = +132.8% HAC 4.90
#   - HPO for cb_no_deriv hurt sim (-10pp), overfit validation
#   - residual target = flop (+92.2%)
#   - huber delta=1.5 = best training Sharpe (1.93) but sim +127.2%
#   - Training Sharpe ≠ sim performance (AGAIN confirmed)
#
# v15 INSIGHT:
#   We have 6+ sim flags that were NEVER TESTED:
#     --vol-target-ann, --hysteresis, --smooth-signal,
#     --turnover-budget, --vol-size, --regime-shorts
#   These are pure execution-layer improvements — no retraining.
#   If any flag adds +10-20% return, it's free alpha.
#
# PLAN:
#   Phase 1: Baseline reproduction (1 sim)
#   Phase 2: Single-flag sweeps — isolate each flag's effect (18 sims)
#   Phase 3: Best combo grid — combine winning flags (12 sims)
#   Phase 4: Best combo on R1/R2 stability check (6 sims)
#   Phase 5: Leverage sensitivity with best combo (4 sims)
#
# Expected: ~50 sims × ~3min = ~2.5h
#
# Usage:
#   nohup ./run_overnight_v15.sh > overnight_v15.log 2>&1 &
# ============================================================

LOGDIR="results/overnight_v15"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG="$LOGDIR/run_${TIMESTAMP}.log"
SUMMARY="$LOGDIR/summary_${TIMESTAMP}.txt"

SIM_DATA="--data data/features/crypto_features_1h.parquet"

# Champion model (v14 cb_market_noderiv_hpo: +143.8%, HAC 5.33)
CHAMPION="results/overnight_v14/cb_market_noderiv_hpo"
# Old champion for reference (v12 cb_no_deriv: +131.5%, HAC 5.09)
V12_CBND="results/overnight_v12/cb_no_deriv"

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
    [[ -d "$d" ]] && mv "$d" "${d}_bak_v15"
  done
  [[ -d "results/production" ]] && mv "results/production" "results/production_bak_v15"
  for d in results_v6_*h_prod; do
    [[ -d "$d" ]] && mv "$d" "${d}_bak_v15"
  done
}

restore_models() {
  for d in "${MODEL_DIRS[@]}"; do
    rm -rf "$d" 2>/dev/null
    [[ -d "${d}_bak_v15" ]] && mv "${d}_bak_v15" "$d"
  done
  [[ -d "results/production_bak_v15" ]] && mv "results/production_bak_v15" "results/production"
  for d in results_v6_*h_prod_bak_v15; do
    [[ -d "$d" ]] && mv "${d}" "${d%_bak_v15}"
  done
}

setup_models() {
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

# ─── Verify models exist ───
for d in "$CHAMPION" "$V12_CBND"; do
  if [[ ! -d "$d" ]]; then
    log "❌ FATAL: Model not found: $d"
    exit 1
  fi
done

# Isolate all model dirs
isolate_models

# Setup NEW champion for sims
setup_models "SKIP" "SKIP" "$CHAMPION" "SKIP"

# Sim periods
R1="--days 92 --start-date 2025-10-01 --end-date 2025-12-31"
R2="--days 66 --start-date 2026-01-01 --end-date 2026-03-07"
FULL="--days 158 --start-date 2025-10-01 --end-date 2026-03-07"

# Baseline sim flags (from v13 — what the champion was tested with)
BASE="--leverage 3 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble"

# ╔══════════════════════════════════════════════════════════════╗
# ║  PHASE 1: BASELINE REPRODUCTION (~6min)                      ║
# ║  New champion + old champion reference                       ║
# ╚══════════════════════════════════════════════════════════════╝
phase_start "PHASE 1: BASELINE REPRODUCTION"

# New champion: cb_market_noderiv_hpo (expect +143.8%)
run_sim "BASELINE_new_champ" $FULL $BASE

# Old champion reference
setup_models "SKIP" "SKIP" "$V12_CBND" "SKIP"
run_sim "BASELINE_old_champ" $FULL $BASE

# Switch back to new champion for all subsequent sims
setup_models "SKIP" "SKIP" "$CHAMPION" "SKIP"

phase_end

# ╔══════════════════════════════════════════════════════════════╗
# ║  PHASE 2: SINGLE-FLAG SWEEPS (~1h)                           ║
# ║  Test each flag in isolation vs baseline                     ║
# ╚══════════════════════════════════════════════════════════════╝
phase_start "PHASE 2: SINGLE-FLAG SWEEPS"

# ── 2A: Vol targeting (annualized target %, scale exposure inversely to vol) ──
log "--- Vol targeting sweep ---"
run_sim "VOLTGT_30pct" $FULL $BASE --vol-target-ann 0.30
run_sim "VOLTGT_40pct" $FULL $BASE --vol-target-ann 0.40
run_sim "VOLTGT_50pct" $FULL $BASE --vol-target-ann 0.50
run_sim "VOLTGT_60pct" $FULL $BASE --vol-target-ann 0.60

# ── 2B: Hysteresis (keep position until rank > N+K) ──
log "--- Hysteresis sweep ---"
run_sim "HYST_3"  $FULL $BASE --hysteresis 3
run_sim "HYST_5"  $FULL $BASE --hysteresis 5
run_sim "HYST_7"  $FULL $BASE --hysteresis 7
run_sim "HYST_10" $FULL $BASE --hysteresis 10

# ── 2C: Signal smoothing (EMA alpha on predictions) ──
log "--- Signal smoothing sweep ---"
run_sim "SMOOTH_02" $FULL $BASE --smooth-signal 0.2
run_sim "SMOOTH_03" $FULL $BASE --smooth-signal 0.3
run_sim "SMOOTH_04" $FULL $BASE --smooth-signal 0.4
run_sim "SMOOTH_05" $FULL $BASE --smooth-signal 0.5

# ── 2D: Turnover budget (max N replacements per side per rebalance) ──
log "--- Turnover budget sweep ---"
run_sim "TOBUD_3" $FULL $BASE --turnover-budget 3
run_sim "TOBUD_5" $FULL $BASE --turnover-budget 5
run_sim "TOBUD_8" $FULL $BASE --turnover-budget 8

# ── 2E: Vol-adjusted sizing (inverse vol weighting) ──
log "--- Vol sizing ---"
run_sim "VOLSIZE" $FULL $BASE --vol-size

# ── 2F: Regime short scaling (reduce shorts in bull regime) ──
log "--- Regime shorts ---"
run_sim "REGSHORT_05" $FULL $BASE --regime-shorts 0.5
run_sim "REGSHORT_03" $FULL $BASE --regime-shorts 0.3

phase_end

# ╔══════════════════════════════════════════════════════════════╗
# ║  PHASE 3: BEST COMBO GRID (~30min)                           ║
# ║  Combine top flags from Phase 2                              ║
# ║  (Pre-selected likely winners based on quant theory)         ║
# ╚══════════════════════════════════════════════════════════════╝
phase_start "PHASE 3: COMBO GRID"

# Combo A: hysteresis + vol-target (turnover reduction + risk management)
run_sim "COMBO_hyst5_vt50"  $FULL $BASE --hysteresis 5 --vol-target-ann 0.50
run_sim "COMBO_hyst5_vt40"  $FULL $BASE --hysteresis 5 --vol-target-ann 0.40

# Combo B: hysteresis + smoothing (double smoothing)
run_sim "COMBO_hyst5_sm03"  $FULL $BASE --hysteresis 5 --smooth-signal 0.3

# Combo C: hysteresis + vol-target + smoothing (triple)
run_sim "COMBO_hyst5_vt50_sm03" $FULL $BASE --hysteresis 5 --vol-target-ann 0.50 --smooth-signal 0.3
run_sim "COMBO_hyst5_vt40_sm03" $FULL $BASE --hysteresis 5 --vol-target-ann 0.40 --smooth-signal 0.3

# Combo D: add vol-size to best combos
run_sim "COMBO_hyst5_vt50_vs"  $FULL $BASE --hysteresis 5 --vol-target-ann 0.50 --vol-size
run_sim "COMBO_hyst5_vt50_sm03_vs" $FULL $BASE --hysteresis 5 --vol-target-ann 0.50 --smooth-signal 0.3 --vol-size

# Combo E: turnover budget + hysteresis (both turnover reducers)
run_sim "COMBO_hyst5_tobud5"  $FULL $BASE --hysteresis 5 --turnover-budget 5
run_sim "COMBO_hyst5_tobud3"  $FULL $BASE --hysteresis 5 --turnover-budget 3

# Combo F: kitchen sink (everything reasonable)
run_sim "COMBO_all_moderate" $FULL $BASE --hysteresis 5 --vol-target-ann 0.50 --smooth-signal 0.3 --turnover-budget 5 --vol-size
run_sim "COMBO_all_conservative" $FULL $BASE --hysteresis 7 --vol-target-ann 0.40 --smooth-signal 0.4 --turnover-budget 3 --vol-size

# Combo G: meta-risk overlay (model-agreement based risk scaler)
run_sim "COMBO_metarisk" $FULL $BASE --meta-risk
run_sim "COMBO_metarisk_hyst5_vt50" $FULL $BASE --meta-risk --hysteresis 5 --vol-target-ann 0.50

phase_end

# ╔══════════════════════════════════════════════════════════════╗
# ║  PHASE 4: STABILITY CHECK — best combos on R1 & R2 (~15min) ║
# ╚══════════════════════════════════════════════════════════════╝
phase_start "PHASE 4: STABILITY CHECK"

# New champion: baseline + best combos on R1, R2
run_sim "STAB_newchamp_R1" $R1 $BASE
run_sim "STAB_newchamp_R2" $R2 $BASE
run_sim "STAB_newchamp_hyst5_R1" $R1 $BASE --hysteresis 5
run_sim "STAB_newchamp_hyst5_R2" $R2 $BASE --hysteresis 5
run_sim "STAB_newchamp_h5vt50_R1" $R1 $BASE --hysteresis 5 --vol-target-ann 0.50
run_sim "STAB_newchamp_h5vt50_R2" $R2 $BASE --hysteresis 5 --vol-target-ann 0.50

# Old champion with best flags (do execution flags help BOTH models?)
setup_models "SKIP" "SKIP" "$V12_CBND" "SKIP"
run_sim "STAB_oldchamp_hyst5_R1" $R1 $BASE --hysteresis 5
run_sim "STAB_oldchamp_hyst5_R2" $R2 $BASE --hysteresis 5
run_sim "STAB_oldchamp_h5vt50_FULL" $FULL $BASE --hysteresis 5 --vol-target-ann 0.50

# Switch back to new champion
setup_models "SKIP" "SKIP" "$CHAMPION" "SKIP"

phase_end

# ╔══════════════════════════════════════════════════════════════╗
# ║  PHASE 5: LEVERAGE SENSITIVITY with best combo (~12min)      ║
# ╚══════════════════════════════════════════════════════════════╝
phase_start "PHASE 5: LEVERAGE SENSITIVITY"

BEST_EXTRA="--hysteresis 5 --vol-target-ann 0.50"

run_sim "LEV_combo_1x" $FULL --leverage 1 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble $BEST_EXTRA
run_sim "LEV_combo_2x" $FULL --leverage 2 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble $BEST_EXTRA
run_sim "LEV_combo_4x" $FULL --leverage 4 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble $BEST_EXTRA
run_sim "LEV_combo_5x" $FULL --leverage 5 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble $BEST_EXTRA

phase_end

# ╔══════════════════════════════════════════════════════════════╗
# ║  FINAL SUMMARY                                               ║
# ╚══════════════════════════════════════════════════════════════╝
phase_start "FINAL SUMMARY"

# Restore all model dirs
restore_models

log ""
log "============================================================"
log "  OVERNIGHT v15 — FINAL RESULTS"
log "============================================================"

{
  echo ""
  echo "=== ALL SIM RESULTS (sorted by HAC) ==="
  echo ""
  printf "%-40s %10s %10s\n" "Config" "Return" "HAC"
  echo "─────────────────────────────────────────────────────────────────"
} >> "$SUMMARY"

# Parse and sort results
grep -E 'SIM #|=> Return' "$LOG" | paste - - | while read -r line; do
  label=$(echo "$line" | sed 's/.*SIM #[0-9]*: //' | sed 's/ *\[.*//')
  ret=$(echo "$line" | grep -o 'Return=[^ ]*' | head -1 | cut -d= -f2)
  hac=$(echo "$line" | grep -o 'HAC=[^ ]*' | head -1 | cut -d= -f2)
  printf "%-40s %10s %10s\n" "$label" "$ret" "$hac"
done | sort -t'%' -k2 -rn 2>/dev/null | tee -a "$LOG" >> "$SUMMARY"

TOTAL_TIME=$(( $(date +%s) - START_TIME ))

{
  echo ""
  echo "============================================================"
  echo "  TOTAL: ${SIM_N} sims in $(( TOTAL_TIME / 3600 ))h $(( (TOTAL_TIME % 3600) / 60 ))m"
  echo "============================================================"
  echo ""
  echo "KEY QUESTIONS:"
  echo "  1. Does vol-targeting improve Sharpe/HAC vs baseline (+143.8%)?"
  echo "  2. Does hysteresis reduce turnover and improve returns?"
  echo "  3. Does signal smoothing help or hurt?"
  echo "  4. What is the optimal combo of execution flags?"
  echo "  5. Is the best combo stable across R1 and R2?"
  echo "  6. Do execution flags help old champion too (model-independent)?"
  echo "  7. What leverage is optimal with best execution config?"
  echo ""
  echo "NEXT STEPS:"
  echo "  - If any flag adds >5%: implement in production"
  echo "  - If vol-target helps: tune EWMA halflife (v16)"
  echo "  - If hysteresis helps: test K=3..10 more finely"
  echo "  - Add funding z-score overlay as code (Tier 2)"
} >> "$SUMMARY"

log ""
log "  TOTAL: ${SIM_N} sims in $(( TOTAL_TIME / 3600 ))h $(( (TOTAL_TIME % 3600) / 60 ))m"
log "  SUMMARY: $SUMMARY"
log ""
