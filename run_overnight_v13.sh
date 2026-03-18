#!/bin/bash
set -uo pipefail

# ============================================================
# OVERNIGHT v13 — Resolve the Sim Paradox
# ============================================================
#
# KEY PARADOX from v12:
#   - cb_price_only = best training Sharpe ever (1.78)
#   - v6_price_only_hpo = huge jump (1.45 vs 1.10 baseline)
#   - BUT in sim: v11 baseline (v6+CB with derivs) = +46.2%
#     v12 models (better training) = +35-40%
#
# HYPOTHESIS: models trained WITH derivative features capture
# timing signals (funding rate spikes, OI shifts) that help
# real trading even though they hurt cross-sectional ranking.
# Pure price models rank better but trade worse.
#
# This script tests:
#   Phase 1: Sim sensitivity — same models, different sim flags
#            (deriv-gate, ddstop, leverage, kelly, edge-boost)
#   Phase 2: Time stability — sim on R2 period (Q1 2026)
#   Phase 3: Mixed ensembles — v11 timing + v12 ranking
#   Phase 4: Correlation analysis between model predictions
#   Phase 5: 1-2 targeted training experiments
#
# ALL PHASES ARE SIM-ONLY except Phase 5 → very fast (~2-3h total)
#
# Usage:
#   nohup ./run_overnight_v13.sh > overnight_v13.log 2>&1 &
# ============================================================

LOGDIR="results/overnight_v13"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG="$LOGDIR/run_${TIMESTAMP}.log"
SUMMARY="$LOGDIR/summary_${TIMESTAMP}.txt"

GPU="--gpu"
SEEDS="--seeds 5"
COMMON="--research $GPU $SEEDS"
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
    [[ -d "$d" ]] && mv "$d" "${d}_bak_v13"
  done
  [[ -d "results/production" ]] && mv "results/production" "results/production_bak_v13"
  for d in results_v6_*h_prod; do
    [[ -d "$d" ]] && mv "$d" "${d}_bak_v13"
  done
}

restore_models() {
  for d in "${MODEL_DIRS[@]}"; do
    rm -rf "$d" 2>/dev/null
    [[ -d "${d}_bak_v13" ]] && mv "${d}_bak_v13" "$d"
  done
  [[ -d "results/production_bak_v13" ]] && mv "results/production_bak_v13" "results/production"
  for d in results_v6_*h_prod_bak_v13; do
    [[ -d "$d" ]] && mv "$d" "${d%_bak_v13}"
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

# ─── Model paths ───
# v11 baselines (with derivatives)
V11_V6="results_v6_research"
V11_CB="results_catboost_research"
V11_XGB="results_xgboost_research"
# v12 best (without derivatives)
V12_V6PO="results/overnight_v12/v6_price_only_hpo"
V12_CBPO="results/overnight_v12/cb_price_only"
V12_CBND="results/overnight_v12/cb_no_deriv"
V12_XGBND="results/overnight_v12/xgb_no_deriv"

# Verify model dirs exist
for d in "$V11_V6" "$V11_CB" "$V12_V6PO" "$V12_CBPO"; do
  if [[ ! -d "$d" ]]; then
    log "⚠️  Missing model dir: $d"
  fi
done

# Isolate all model dirs to prevent fallback contamination
isolate_models

# ╔══════════════════════════════════════════════════════════════╗
# ║  PHASE 1: SIM SENSITIVITY on Q4 2025 (R1 test period)       ║
# ║  Same models, vary sim flags → find what actually matters    ║
# ╚══════════════════════════════════════════════════════════════╝
phase_start "PHASE 1: SIM SENSITIVITY on Q4'25 (~40min)"

R1="--days 92 --start-date 2025-10-01 --end-date 2025-12-31"

# ── 1A: v11 baseline (v6+CB with derivs) — reproduce v12 result ──
setup_models "$V11_V6" "SKIP" "$V11_CB" "SKIP"

run_sim "v11_v6cb_NO_protect" \
  $R1 --leverage 3 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble

run_sim "v11_v6cb_WITH_ddstop" \
  $R1 --leverage 3 --kelly 0.8 --edge-boost --no-deriv-gate --ensemble

run_sim "v11_v6cb_lev2" \
  $R1 --leverage 2 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble

run_sim "v11_v6cb_lev1" \
  $R1 --leverage 1 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble

run_sim "v11_v6cb_kelly05" \
  $R1 --leverage 3 --kelly 0.5 --edge-boost --no-deriv-gate --no-ddstop --ensemble

run_sim "v11_v6cb_no_edgeboost" \
  $R1 --leverage 3 --kelly 0.8 --no-deriv-gate --no-ddstop --ensemble

# ── 1B: v12 cb_price_only solo — same flag variations ──
setup_models "SKIP" "SKIP" "$V12_CBPO" "SKIP"

run_sim "v12_cbpo_NO_protect" \
  $R1 --leverage 3 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble

run_sim "v12_cbpo_WITH_ddstop" \
  $R1 --leverage 3 --kelly 0.8 --edge-boost --no-deriv-gate --ensemble

run_sim "v12_cbpo_lev2" \
  $R1 --leverage 2 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble

run_sim "v12_cbpo_lev1" \
  $R1 --leverage 1 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble

run_sim "v12_cbpo_kelly05" \
  $R1 --leverage 3 --kelly 0.5 --edge-boost --no-deriv-gate --no-ddstop --ensemble

run_sim "v12_cbpo_no_edgeboost" \
  $R1 --leverage 3 --kelly 0.8 --no-deriv-gate --no-ddstop --ensemble

# ── 1C: v12 v6_price_only + cb_price_only — same variations ──
setup_models "$V12_V6PO" "SKIP" "$V12_CBPO" "SKIP"

run_sim "v12_v6po_cbpo_NO_protect" \
  $R1 --leverage 3 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble

run_sim "v12_v6po_cbpo_lev2" \
  $R1 --leverage 2 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble

run_sim "v12_v6po_cbpo_kelly05" \
  $R1 --leverage 3 --kelly 0.5 --edge-boost --no-deriv-gate --no-ddstop --ensemble

phase_end

# ╔══════════════════════════════════════════════════════════════╗
# ║  PHASE 2: TIME STABILITY — sim on R2 (Q1 2026)              ║
# ║  Is v11 > v12 consistent or period-specific?                 ║
# ╚══════════════════════════════════════════════════════════════╝
phase_start "PHASE 2: R2 PERIOD (Q1 2026) — ~20min"

R2="--days 66 --start-date 2026-01-01 --end-date 2026-03-07"

# v11 baseline
setup_models "$V11_V6" "SKIP" "$V11_CB" "SKIP"
run_sim "R2_v11_v6cb" \
  $R2 --leverage 3 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble

# v11 v6 solo
setup_models "$V11_V6" "SKIP" "SKIP" "SKIP"
run_sim "R2_v11_v6_solo" \
  $R2 --leverage 3 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble

# v12 cb_price_only solo
setup_models "SKIP" "SKIP" "$V12_CBPO" "SKIP"
run_sim "R2_v12_cbpo_solo" \
  $R2 --leverage 3 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble

# v12 v6po + cbpo
setup_models "$V12_V6PO" "SKIP" "$V12_CBPO" "SKIP"
run_sim "R2_v12_v6po_cbpo" \
  $R2 --leverage 3 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble

# Mix: v11_v6 (timing) + v12 cbpo (ranking)
setup_models "$V11_V6" "SKIP" "$V12_CBPO" "SKIP"
run_sim "R2_mix_v11v6_v12cbpo" \
  $R2 --leverage 3 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble

phase_end

# ╔══════════════════════════════════════════════════════════════╗
# ║  PHASE 3: FULL PERIOD (Q4'25 + Q1'26) — 5.5 months          ║
# ║  Most robust evaluation                                      ║
# ╚══════════════════════════════════════════════════════════════╝
phase_start "PHASE 3: FULL PERIOD Q4'25–Q1'26 (~20min)"

FULL="--days 158 --start-date 2025-10-01 --end-date 2026-03-07"

# v11 baseline
setup_models "$V11_V6" "SKIP" "$V11_CB" "SKIP"
run_sim "FULL_v11_v6cb" \
  $FULL --leverage 3 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble

# v11 v6 solo
setup_models "$V11_V6" "SKIP" "SKIP" "SKIP"
run_sim "FULL_v11_v6_solo" \
  $FULL --leverage 3 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble

# v12 cbpo solo
setup_models "SKIP" "SKIP" "$V12_CBPO" "SKIP"
run_sim "FULL_v12_cbpo_solo" \
  $FULL --leverage 3 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble

# v12 v6po + cbpo
setup_models "$V12_V6PO" "SKIP" "$V12_CBPO" "SKIP"
run_sim "FULL_v12_v6po_cbpo" \
  $FULL --leverage 3 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble

# Mix: v11_v6 + v12_cbpo (best hypothesis)
setup_models "$V11_V6" "SKIP" "$V12_CBPO" "SKIP"
run_sim "FULL_mix_v11v6_v12cbpo" \
  $FULL --leverage 3 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble

# Mix: v11_v6 + v12_cbpo + v12_xgbnd (3 models)
setup_models "$V11_V6" "SKIP" "$V12_CBPO" "$V12_XGBND"
run_sim "FULL_mix_3model" \
  $FULL --leverage 3 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble

# v12 cb_no_deriv solo (Sharpe 1.66 — second best training)
setup_models "SKIP" "SKIP" "$V12_CBND" "SKIP"
run_sim "FULL_v12_cbnd_solo" \
  $FULL --leverage 3 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble

# v11 3-model (v6+CB+XGB, no v7)
setup_models "$V11_V6" "SKIP" "$V11_CB" "$V11_XGB"
run_sim "FULL_v11_3model" \
  $FULL --leverage 3 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble

# Leverage sweep for best config (will use winner from above, but test v11 and mix)
setup_models "$V11_V6" "SKIP" "$V11_CB" "SKIP"
run_sim "FULL_v11_v6cb_lev2" \
  $FULL --leverage 2 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble
run_sim "FULL_v11_v6cb_lev1" \
  $FULL --leverage 1 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble

setup_models "$V11_V6" "SKIP" "$V12_CBPO" "SKIP"
run_sim "FULL_mix_lev2" \
  $FULL --leverage 2 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble
run_sim "FULL_mix_lev1" \
  $FULL --leverage 1 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble

phase_end

# ╔══════════════════════════════════════════════════════════════╗
# ║  PHASE 4: PREDICTION CORRELATION ANALYSIS                   ║
# ╚══════════════════════════════════════════════════════════════╝
phase_start "PHASE 4: CORRELATION ANALYSIS (~2min)"

python - <<'PYEOF' 2>&1 | tee -a "$LOG" >> "$SUMMARY"
import os, glob, json
import numpy as np

def load_preds(results_dir):
    """Load test predictions from a results directory."""
    preds = {}
    for csv_path in sorted(glob.glob(os.path.join(results_dir, "test_predictions*.csv"))):
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            # Find prediction column
            pred_col = [c for c in df.columns if c.startswith("pred_")]
            if pred_col:
                for col in pred_col:
                    key = os.path.basename(results_dir) + "/" + col
                    preds[key] = df[col].values
        except Exception:
            continue
    return preds

dirs = {
    "v11_v6": "results_v6_research",
    "v11_cb": "results_catboost_research",
    "v11_xgb": "results_xgboost_research",
    "v12_v6po": "results/overnight_v12/v6_price_only_hpo",
    "v12_cbpo": "results/overnight_v12/cb_price_only",
    "v12_cbnd": "results/overnight_v12/cb_no_deriv",
    "v12_xgbnd": "results/overnight_v12/xgb_no_deriv",
}

print("\n=== PREDICTION CORRELATION MATRIX ===\n")
all_preds = {}
for name, path in dirs.items():
    if os.path.isdir(path):
        p = load_preds(path)
        if p:
            # Average across seeds, take first pred column type
            first_key = list(p.keys())[0]
            all_preds[name] = p[first_key]

if len(all_preds) >= 2:
    names = sorted(all_preds.keys())
    min_len = min(len(all_preds[n]) for n in names)
    print(f"{'':>12}", end="")
    for n in names:
        print(f" {n:>10}", end="")
    print()
    for n1 in names:
        print(f"{n1:>12}", end="")
        for n2 in names:
            v1 = all_preds[n1][:min_len]
            v2 = all_preds[n2][:min_len]
            corr = np.corrcoef(v1, v2)[0,1]
            print(f" {corr:>10.3f}", end="")
        print()
else:
    print("Not enough prediction files found for correlation analysis")
    print(f"Found: {list(all_preds.keys())}")
PYEOF

phase_end

# ╔══════════════════════════════════════════════════════════════╗
# ║  PHASE 5: TARGETED TRAINING EXPERIMENTS                     ║
# ║  Test if we can get best of both worlds                      ║
# ╚══════════════════════════════════════════════════════════════╝
phase_start "PHASE 5: TARGETED TRAINING (~1.5h)"

# Experiment: v6 with ONLY derivative features (no price technicals)
# Rationale: if derivs help timing but hurt ranking, maybe a
# dedicated deriv-only model can provide timing signal as separate
# ensemble member alongside price_only models

# v6 with derivs but no news, no price features — just derivs + base
# Actually this won't work easily without code changes.
# Instead: test v6 with ALL features but MORE aggressive regularization
# to prevent overfitting to noisy derivs

# More practical: CatBoost with news but no derivs (market-only + no deriv)
train_experiment "$LOGDIR/cb_market_no_deriv" \
  run_pipeline_catboost.py "$HUBER --news-mode market-only --no-derivatives --skip-hpo"

# v6 no_deriv without HPO — check if HPO is causing the sim degradation
train_experiment "$LOGDIR/v6_no_deriv_skip_hpo" \
  run_pipeline_v6.py "$HUBER --no-derivatives --skip-hpo"

# Final sim: test new models from Phase 5
if [[ -d "$LOGDIR/cb_market_no_deriv" ]]; then
  setup_models "$V11_V6" "SKIP" "$LOGDIR/cb_market_no_deriv" "SKIP"
  run_sim "FULL_mix_v11v6_cbMKTnd" \
    $FULL --leverage 3 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --ensemble
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
log "  OVERNIGHT v13 — FINAL RESULTS"
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

TOTAL_TIME=$(( $(date +%s) - START_TIME ))
log ""
log "============================================================"
log "  TOTAL RUNTIME: $(( TOTAL_TIME / 3600 ))h $(( (TOTAL_TIME % 3600) / 60 ))m"
log "  SUMMARY: $SUMMARY"
log "============================================================"
log ""
log "Key questions answered:"
log "  1. Does v11 > v12 hold across R1, R2, and FULL period?"
log "  2. Is the gap due to leverage/kelly/edge-boost settings?"
log "  3. Does mixed ensemble (v11 timing + v12 ranking) beat both?"
log "  4. How correlated are v11 vs v12 predictions?"
