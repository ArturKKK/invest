#!/bin/bash
set -uo pipefail

# ============================================================
# OVERNIGHT DEEP RESEARCH v11 — 8-hour comprehensive experiment grid
# ============================================================
#
# Goal: find the REAL edge (or lack thereof) and optimal ensemble.
# All experiments use RESEARCH windows with TRUE out-of-sample test:
#   R1: train→2024-12-31, val 2025-01→2025-09, test 2025-Q4 (3 months)
#   R2: train→2025-06-30, val 2025-07→2025-12, test 2026-Q1 (2.5 months)
#
# Phase 1: Baseline research — 4 current models with HPO         (~5-7h)
# Phase 2: Horizon experiments — v6 at 4h and 24h                (~30min)
# Phase 3: Loss/target experiments — MSE, residual, lambdarank   (~1h)
# Phase 4: Feature ablation — no-news, no-deriv, news-only       (~30min)
# Phase 5: Decorrelation — v7 no-news, deadzone, hybrid-norm     (~30min)
# Phase 6: Sim grid on best research models                      (~15min)
# Phase 7: Summary & analysis                                    (~1min)
#
# Expected total: ~8-10 hours on GPU (Phase 1 dominates: HPO 50 trials × 4 models × 2 windows)
# 
# Usage:
#   nohup ./run_overnight_v11.sh > overnight_v11.log 2>&1 &
#   tail -f overnight_v11.log
# ============================================================

LOGDIR="results/overnight_v11"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG="$LOGDIR/run_${TIMESTAMP}.log"
SUMMARY="$LOGDIR/summary_${TIMESTAMP}.txt"

GPU="--gpu"
SKIP_HPO=""          # empty = do HPO; set "--skip-hpo" to go faster
SEEDS="--seeds 5"
COMMON="--research $GPU $SEEDS"   # base flags (no --huber, add per-experiment)
HUBER="--huber"                     # add explicitly where needed

# Sim config for Phase 6
SIM_DATA="--data data/features/crypto_features_1h.parquet"
SIM_PERIOD="--days 120 --start-date 2025-10-01 --end-date 2025-12-31"
SIM_BASE="$SIM_DATA $SIM_PERIOD --leverage 3 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop"

export SKIP_CALENDAR=1

START_TIME=$(date +%s)

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ─── Timing helper ───
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
  log "CMD: python $script $COMMON $SKIP_HPO --results $results_dir $extra_args"
  log "────────────────────────────────────"
  
  local t0=$(date +%s)
  python "$script" $COMMON $SKIP_HPO --results "$results_dir" $extra_args 2>&1 | tee -a "$LOG"
  local rc=$?
  local elapsed=$(( $(date +%s) - t0 ))
  
  if [[ $rc -eq 0 ]]; then
    log "  ✅ #${EXP_N} done in $(( elapsed / 60 ))m $(( elapsed % 60 ))s"
  else
    log "  ❌ #${EXP_N} FAILED (rc=$rc) after $(( elapsed / 60 ))m"
  fi
  
  # Extract key metrics to summary — check all possible JSON names
  echo "=== #${EXP_N}: $label ===" >> "$SUMMARY"
  echo "CMD: python $script $COMMON $SKIP_HPO --results $results_dir $extra_args" >> "$SUMMARY"
  local results_json=""
  for jname in all_results_v6.json all_results_v7.json all_results_catboost.json all_results_xgboost.json; do
    if [[ -f "$results_dir/$jname" ]]; then
      results_json="$results_dir/$jname"
      break
    fi
  done
  if [[ -n "$results_json" ]]; then
    python -c "
import json
d = json.load(open('$results_json'))
avg = d.get('average', {})
print(f\"  Rank_IC={avg.get('Rank_IC','?')}  ICIR={avg.get('Rank_ICIR','?')}  Sharpe={avg.get('LS_Sharpe_net','?')}  DD={avg.get('LS_MaxDD_net_%','?')}%  Ret={avg.get('LS_Total_net_%','?')}%\")
" 2>/dev/null >> "$SUMMARY"
  else
    echo "  (no results JSON found)" >> "$SUMMARY"
  fi
  echo "" >> "$SUMMARY"
}

# ─── Sim helper (Phase 6) ───
SIM_N=0
run_sim() {
  local label="$1"
  shift
  SIM_N=$((SIM_N + 1))
  log "SIM #${SIM_N}: $label"
  log "CMD: python run_fast_sim.py $SIM_BASE $*"
  
  local output
  output=$(python run_fast_sim.py $SIM_BASE "$@" 2>&1) || true
  echo "$output" >> "$LOG"
  
  echo "=== SIM #${SIM_N}: $label ===" >> "$SUMMARY"
  echo "$output" | grep -E "Return:|Max DD:|Sharpe|Calmar:|Win Rate:" | head -6 >> "$SUMMARY"
  echo "" >> "$SUMMARY"
  
  local ret hac
  ret=$(echo "$output" | grep "Return:" | head -1 | awk '{print $2}')
  hac=$(echo "$output" | grep "Sharpe HAC:" | awk '{print $3}')
  log "  => Return=$ret  HAC=$hac"
}

# ╔══════════════════════════════════════════════════════════════╗
# ║  PHASE 1: BASELINE — 4 models with HPO (current config)     ║
# ╚══════════════════════════════════════════════════════════════╝
phase_start "PHASE 1: BASELINE (4 models, HPO, Huber) — ~5-7h"

train_experiment "results_v6_research" \
  run_pipeline_v6.py "$HUBER"

train_experiment "results_v7_research" \
  run_pipeline_v7.py "$HUBER --news-mode none"

train_experiment "results_catboost_research" \
  run_pipeline_catboost.py "$HUBER"

train_experiment "results_xgboost_research" \
  run_pipeline_xgboost.py "$HUBER"

phase_end

# ╔══════════════════════════════════════════════════════════════╗
# ║  PHASE 2: HORIZON EXPERIMENTS — does 4h or 24h beat 12h?    ║
# ╚══════════════════════════════════════════════════════════════╝
phase_start "PHASE 2: HORIZON EXPERIMENTS (v6 at 4h, 24h) — ~40min"

# 4h horizon — faster signal, more noise
train_experiment "$LOGDIR/v6_horizon_4h" \
  run_pipeline_v6.py "$HUBER --horizon 4 --skip-hpo"

# 24h horizon — slower signal, less noise, fewer data points
train_experiment "$LOGDIR/v6_horizon_24h" \
  run_pipeline_v6.py "$HUBER --horizon 24 --skip-hpo"

phase_end

# ╔══════════════════════════════════════════════════════════════╗
# ║  PHASE 3: LOSS & TARGET EXPERIMENTS                          ║
# ╚══════════════════════════════════════════════════════════════╝
phase_start "PHASE 3: LOSS & TARGET (MSE, residual, lambdarank) — ~1.5h"

# 3a. MSE loss (instead of Huber) — is Huber actually helping?
train_experiment "$LOGDIR/v6_mse" \
  run_pipeline_v6.py "--skip-hpo"
# ^ no $HUBER → defaults to MSE

# 3b. Residual target (remove BTC beta) — pure alpha signal
train_experiment "$LOGDIR/v6_residual" \
  run_pipeline_v6.py "$HUBER --residual-target --skip-hpo"

# 3c. LambdaRank (learning to rank) — direct ranking optimization
train_experiment "$LOGDIR/v6_lambdarank" \
  run_pipeline_v6.py "$HUBER --lambdarank --skip-hpo"

# 3d. Huber with different alpha (0.7 = more robust, 0.95 = closer to MSE)
train_experiment "$LOGDIR/v6_huber_alpha07" \
  run_pipeline_v6.py "$HUBER --huber-alpha 0.7 --skip-hpo"

# 3e. CatBoost with MSE (is Huber helping for CatBoost too?)
train_experiment "$LOGDIR/cb_mse" \
  run_pipeline_catboost.py "--skip-hpo"
# ^ no $HUBER → defaults to MSE

phase_end

# ╔══════════════════════════════════════════════════════════════╗
# ║  PHASE 4: FEATURE ABLATION — what's actually helping?        ║
# ╚══════════════════════════════════════════════════════════════╝
phase_start "PHASE 4: FEATURE ABLATION — ~40min"

# 4a. No news — are news features adding noise or signal?
train_experiment "$LOGDIR/v6_no_news" \
  run_pipeline_v6.py "$HUBER --news-mode none --skip-hpo"

# 4b. No derivatives — funding/OI/LSR contributing?
train_experiment "$LOGDIR/v6_no_deriv" \
  run_pipeline_v6.py "$HUBER --no-derivatives --skip-hpo"

# 4c. No news AND no derivatives — pure price model
train_experiment "$LOGDIR/v6_price_only" \
  run_pipeline_v6.py "$HUBER --news-mode none --no-derivatives --skip-hpo"

phase_end

# ╔══════════════════════════════════════════════════════════════╗
# ║  PHASE 5: DECORRELATION — make v7 actually different         ║
# ╚══════════════════════════════════════════════════════════════╝
phase_start "PHASE 5: DECORRELATION EXPERIMENTS — ~40min"

# 5a. v7 with deadzone weighting — focus on larger moves
train_experiment "$LOGDIR/v7_deadzone" \
  run_pipeline_v7.py "$HUBER --news-mode none --deadzone-weight 0.3 --skip-hpo"

# 5b. v7 with hybrid normalization — different preprocessing
train_experiment "$LOGDIR/v7_hybrid" \
  run_pipeline_v7.py "$HUBER --news-mode none --hybrid-norm --skip-hpo"

# 5c. v6 with 24h + no news — maximally different from baseline v6
train_experiment "$LOGDIR/v6_24h_no_news" \
  run_pipeline_v6.py "$HUBER --horizon 24 --news-mode none --skip-hpo"

# 5d. XGBoost with market-only news (coin news might be noise)
train_experiment "$LOGDIR/xgb_market_news" \
  run_pipeline_xgboost.py "$HUBER --news-mode market-only --skip-hpo"

phase_end

# ╔══════════════════════════════════════════════════════════════╗
# ║  PHASE 6: SIM GRID — test best combos on R1 test period     ║
# ╚══════════════════════════════════════════════════════════════╝
phase_start "PHASE 6: SIM GRID on 2025-Q4 OOS — ~30min"

# run_fast_sim.py checks MANY fallback dirs for models:
#   LGB: results/production/lgb_v6_no_news, results_v6_prod, results_v6 (same for v7)
#   CB:  results/production/catboost_with_news, results_catboost_prod, results_catboost
#   XGB: results/production/xgboost, results_xgboost_prod, results_xgboost
#   Multi-Hz: results_v6_*h_prod
# We must suppress ALL of them so SKIP actually works.
MODEL_DIRS=(
  results_v6_prod results_v7_prod results_catboost_prod results_xgboost_prod
  results_v6 results_v7 results_catboost results_xgboost
)
for d in "${MODEL_DIRS[@]}"; do
  [[ -d "$d" ]] && mv "$d" "${d}_bak_v11"
done
# Also suppress results/production/ (another fallback path)
[[ -d "results/production" ]] && mv "results/production" "results/production_bak_v11"
# Suppress multi-horizon dirs
for d in results_v6_*h_prod; do
  [[ -d "$d" ]] && mv "$d" "${d}_bak_v11"
done

# Sim helper: clean slate → copy selected → run → clean
sim_with_models() {
  local label="$1"
  local v6_src="$2"
  local v7_src="$3"
  local cb_src="$4"
  local xgb_src="$5"
  shift 5
  local extra_sim_args="$*"
  
  # Clean slate: remove all _prod dirs
  for d in results_v6_prod results_v7_prod results_catboost_prod results_xgboost_prod; do
    rm -rf "$d" 2>/dev/null
  done
  
  # Copy only selected models into _prod
  for pair in "v6:$v6_src" "v7:$v7_src" "catboost:$cb_src" "xgboost:$xgb_src"; do
    IFS=: read -r suffix src <<< "$pair"
    if [[ "$src" != "SKIP" && -d "$src" ]]; then
      cp -r "$src" "results_${suffix}_prod"
    fi
  done
  
  run_sim "$label" --ensemble $extra_sim_args
}

# 6a. Baseline ensemble (all 4 research models)
sim_with_models "baseline_4model" \
  "results_v6_research" "results_v7_research" \
  "results_catboost_research" "results_xgboost_research" ""

# 6b. 3 models: v6 + CB + XGB (drop v7 since v6↔v7 corr ~0.95)
sim_with_models "3model_no_v7" \
  "results_v6_research" "SKIP" \
  "results_catboost_research" "results_xgboost_research" ""

# 6c. v6 solo
sim_with_models "v6_solo" \
  "results_v6_research" "SKIP" "SKIP" "SKIP" ""

# 6d. v6 + CB only (most decorrelated pair)
sim_with_models "v6_cb_only" \
  "results_v6_research" "SKIP" \
  "results_catboost_research" "SKIP" ""

# 6e. If 24h horizon models exist, test solo
if [[ -d "$LOGDIR/v6_horizon_24h" ]]; then
  sim_with_models "v6_24h_solo" \
    "$LOGDIR/v6_horizon_24h" "SKIP" "SKIP" "SKIP" "--rebal 24"
fi

# Restore all model directories from backup
for d in "${MODEL_DIRS[@]}"; do
  rm -rf "$d" 2>/dev/null
  [[ -d "${d}_bak_v11" ]] && mv "${d}_bak_v11" "$d"
done
[[ -d "results/production_bak_v11" ]] && mv "results/production_bak_v11" "results/production"
for d in results_v6_*h_prod_bak_v11; do
  [[ -d "$d" ]] && mv "$d" "${d%_bak_v11}"
done

phase_end

# ╔══════════════════════════════════════════════════════════════╗
# ║  PHASE 7: ANALYSIS & SUMMARY                                ║
# ╚══════════════════════════════════════════════════════════════╝
phase_start "PHASE 7: ANALYSIS & SUMMARY"

# Run analyze_research.py for correlations
python analyze_research.py 2>&1 | tee -a "$LOG" >> "$SUMMARY" || true

# Build final comparison table
log ""
log "============================================================"
log "  OVERNIGHT v11 — FINAL RESULTS"
log "============================================================"

# Collect all experiment results into a table
python - <<'PYEOF' 2>&1 | tee -a "$LOG" >> "$SUMMARY"
import json, os, glob

experiments = []
for results_dir in sorted(glob.glob("results/**/all_results_*.json", recursive=True)) + \
                   sorted(glob.glob("results_*_research/all_results_*.json")):
    try:
        d = json.load(open(results_dir))
        avg = d.get('average', {})
        meta = d.get('meta', {})
        label = os.path.dirname(results_dir)
        experiments.append({
            'label': label,
            'IC': avg.get('Rank_IC', 0),
            'ICIR': avg.get('Rank_ICIR', 0),
            'Sharpe': avg.get('LS_Sharpe_net', 0),
            'MaxDD': avg.get('LS_MaxDD_net_%', 0),
            'Return': avg.get('LS_Total_net_%', 0),
            'N_feat': meta.get('n_selected', '?'),
        })
    except Exception:
        continue

if not experiments:
    print("No results found yet.")
else:
    # Sort by Sharpe descending
    experiments.sort(key=lambda x: -x.get('Sharpe', 0) if isinstance(x.get('Sharpe', 0), (int, float)) else 0)
    
    print(f"\n{'Label':<45} {'IC':>6} {'ICIR':>7} {'Sharpe':>7} {'MaxDD':>7} {'Ret%':>7} {'Feats':>6}")
    print("─" * 90)
    for e in experiments:
        ic = f"{e['IC']:.4f}" if isinstance(e['IC'], float) else str(e['IC'])
        icir = f"{e['ICIR']:.3f}" if isinstance(e['ICIR'], float) else str(e['ICIR'])
        sharpe = f"{e['Sharpe']:.2f}" if isinstance(e['Sharpe'], (int, float)) else str(e['Sharpe'])
        dd = f"{e['MaxDD']:.1f}" if isinstance(e['MaxDD'], (int, float)) else str(e['MaxDD'])
        ret = f"{e['Return']:.1f}" if isinstance(e['Return'], (int, float)) else str(e['Return'])
        print(f"{e['label']:<45} {ic:>6} {icir:>7} {sharpe:>7} {dd:>7} {ret:>7} {e['N_feat']:>6}")
    
    # Highlight best
    best = experiments[0]
    print(f"\n🏆 Best by Sharpe: {best['label']} (Sharpe={best['Sharpe']}, IC={best['IC']})")
PYEOF

TOTAL_TIME=$(( $(date +%s) - START_TIME ))
log ""
log "============================================================"
log "  TOTAL RUNTIME: $(( TOTAL_TIME / 3600 ))h $(( (TOTAL_TIME % 3600) / 60 ))m"
log "  LOG: $LOG"
log "  SUMMARY: $SUMMARY"
log "============================================================"
log ""
log "Next steps:"
log "  1. Review $SUMMARY for best experiments"
log "  2. Compare Sharpe/ICIR across all configs"  
log "  3. Check correlations (analyze_research.py output above)"
log "  4. If clear winner → retrain production with those settings"
log "  5. scp overnight_v11.log to local and share with Copilot for analysis"
