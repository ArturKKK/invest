#!/bin/bash
set -uo pipefail

# ============================================================
# OVERNIGHT v12 — Focused follow-up from v11 results
# ============================================================
#
# Key findings from v11:
#   - CatBoost best single model (avg Sharpe 1.48)
#   - Derivatives features HURT LGB (+0.22 Sharpe when removed)
#   - News neutral for LGB, market-only better for XGB
#   - v7 is weakest, v6 solo beats 4-model ensemble in sim
#   - 4h horizon interesting (Sharpe 1.41, lowest MaxDD -61%)
#
# Open questions this script answers:
#   Q1: Does CatBoost improve without derivatives? (cb_no_deriv)
#   Q2: Does CatBoost improve without news+derivs? (cb_price_only)
#   Q3: How good is v6_price_only with HPO? (v6_price_only_hpo)
#   Q4: How good is v6_no_deriv with HPO? (v6_no_deriv_hpo)
#   Q5: CatBoost + market-only news? (cb_market_news)
#   Q6: 4h horizon CatBoost? (cb_4h) — needs --horizon support check
#   Q7: Best 2-3 model ensemble from new configs? (sim grid)
#   Q8: XGB without derivatives? (xgb_no_deriv)
#
# Phase 1: HPO experiments for promising configs       (~4h)
# Phase 2: Skip-HPO ablation combos                   (~1h)
# Phase 3: Sim grid — find optimal ensemble            (~20min)
#
# Expected total: ~5-6 hours
#
# Usage:
#   nohup ./run_overnight_v12.sh > overnight_v12.log 2>&1 &
#   tail -f overnight_v12.log
# ============================================================

LOGDIR="results/overnight_v12"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG="$LOGDIR/run_${TIMESTAMP}.log"
SUMMARY="$LOGDIR/summary_${TIMESTAMP}.txt"

GPU="--gpu"
SEEDS="--seeds 5"
COMMON="--research $GPU $SEEDS"
HUBER="--huber"

# Sim config — same period as v11 for comparability
SIM_DATA="--data data/features/crypto_features_1h.parquet"
SIM_PERIOD="--days 120 --start-date 2025-10-01 --end-date 2025-12-31"
SIM_BASE="$SIM_DATA $SIM_PERIOD --leverage 3 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop"

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

# ─── Training helper (reused from v11) ───
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
  echo "CMD: python $script $COMMON --results $results_dir $extra_args" >> "$SUMMARY"
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

# ─── Sim helpers (reused from v11) ───
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

MODEL_DIRS=(
  results_v6_prod results_v7_prod results_catboost_prod results_xgboost_prod
  results_v6 results_v7 results_catboost results_xgboost
)

sim_with_models() {
  local label="$1"
  local v6_src="$2"
  local v7_src="$3"
  local cb_src="$4"
  local xgb_src="$5"
  shift 5
  local extra_sim_args="$*"

  for d in results_v6_prod results_v7_prod results_catboost_prod results_xgboost_prod; do
    rm -rf "$d" 2>/dev/null
  done

  for pair in "v6:$v6_src" "v7:$v7_src" "catboost:$cb_src" "xgboost:$xgb_src"; do
    IFS=: read -r suffix src <<< "$pair"
    if [[ "$src" != "SKIP" && -d "$src" ]]; then
      cp -r "$src" "results_${suffix}_prod"
    fi
  done

  run_sim "$label" --ensemble $extra_sim_args
}

# ╔══════════════════════════════════════════════════════════════╗
# ║  PHASE 1: HPO on top configs from v11                       ║
# ╚══════════════════════════════════════════════════════════════╝
phase_start "PHASE 1: HPO on promising configs (~4h)"

# Q3: v6 price_only WITH HPO — v11 showed 1.33 without HPO, baseline with HPO was 1.10
train_experiment "$LOGDIR/v6_price_only_hpo" \
  run_pipeline_v6.py "$HUBER --news-mode none --no-derivatives"

# Q4: v6 no_deriv WITH HPO — v11 showed 1.32 without HPO
train_experiment "$LOGDIR/v6_no_deriv_hpo" \
  run_pipeline_v6.py "$HUBER --no-derivatives"

# Q1: CatBoost without derivatives — CB was 1.48 with all features
train_experiment "$LOGDIR/cb_no_deriv" \
  run_pipeline_catboost.py "$HUBER --no-derivatives"

# Q2: CatBoost price_only — no news, no derivs
train_experiment "$LOGDIR/cb_price_only" \
  run_pipeline_catboost.py "$HUBER --news-mode none --no-derivatives"

phase_end

# ╔══════════════════════════════════════════════════════════════╗
# ║  PHASE 2: Skip-HPO quick tests                              ║
# ╚══════════════════════════════════════════════════════════════╝
phase_start "PHASE 2: Quick ablation tests (~1h)"

# Q5: CatBoost market-only news (coin news might be noise like in XGB)
train_experiment "$LOGDIR/cb_market_news" \
  run_pipeline_catboost.py "$HUBER --news-mode market-only --skip-hpo"

# Q8: XGBoost without derivatives — derivs hurt LGB, do they hurt XGB?
train_experiment "$LOGDIR/xgb_no_deriv" \
  run_pipeline_xgboost.py "$HUBER --no-derivatives --skip-hpo"

# XGB price_only — the nuclear option for XGB too
train_experiment "$LOGDIR/xgb_price_only" \
  run_pipeline_xgboost.py "$HUBER --news-mode none --no-derivatives --skip-hpo"

# CB MSE no_deriv — v11 showed CB MSE ≈ CB Huber, test with no_deriv
train_experiment "$LOGDIR/cb_mse_no_deriv" \
  run_pipeline_catboost.py "--no-derivatives --skip-hpo"

phase_end

# ╔══════════════════════════════════════════════════════════════╗
# ║  PHASE 3: SIM GRID — find optimal ensemble from new models  ║
# ╚══════════════════════════════════════════════════════════════╝
phase_start "PHASE 3: SIM GRID — optimal ensemble (~20min)"

# Isolate model dirs (suppress fallbacks)
for d in "${MODEL_DIRS[@]}"; do
  [[ -d "$d" ]] && mv "$d" "${d}_bak_v12"
done
[[ -d "results/production" ]] && mv "results/production" "results/production_bak_v12"
for d in results_v6_*h_prod; do
  [[ -d "$d" ]] && mv "$d" "${d}_bak_v12"
done

# Use v11 baseline models for comparison where new ones don't exist
V6_BASE="results_v6_research"        # v11 baseline (Sharpe 1.10)
CB_BASE="results_catboost_research"  # v11 baseline (Sharpe 1.48)

# Determine best new v6 (will check after Phase 1)
V6_NEW="$LOGDIR/v6_price_only_hpo"
CB_NEW="$LOGDIR/cb_no_deriv"

# Sim 1: v11 baseline for reference (v6+CB from v11)
sim_with_models "v11_v6cb_baseline" \
  "$V6_BASE" "SKIP" "$CB_BASE" "SKIP" ""

# Sim 2: New v6_price_only + v11 CB
sim_with_models "v6new_cb_old" \
  "$V6_NEW" "SKIP" "$CB_BASE" "SKIP" ""

# Sim 3: v11 v6 + new CB_no_deriv
sim_with_models "v6old_cb_new" \
  "$V6_BASE" "SKIP" "$CB_NEW" "SKIP" ""

# Sim 4: Both new — v6_price_only + CB_no_deriv
sim_with_models "v6new_cb_new" \
  "$V6_NEW" "SKIP" "$CB_NEW" "SKIP" ""

# Sim 5: New v6_price_only solo
sim_with_models "v6new_solo" \
  "$V6_NEW" "SKIP" "SKIP" "SKIP" ""

# Sim 6: New CB_no_deriv solo
sim_with_models "cb_new_solo" \
  "SKIP" "SKIP" "$CB_NEW" "SKIP" ""

# Sim 7: CB_price_only solo (if it worked)
if [[ -d "$LOGDIR/cb_price_only" ]]; then
  sim_with_models "cb_priceonly_solo" \
    "SKIP" "SKIP" "$LOGDIR/cb_price_only" "SKIP" ""
fi

# Sim 8: 3-model new: v6_price + CB_no_deriv + XGB_no_deriv
if [[ -d "$LOGDIR/xgb_no_deriv" ]]; then
  sim_with_models "3model_new_no_deriv" \
    "$V6_NEW" "SKIP" "$CB_NEW" "$LOGDIR/xgb_no_deriv" ""
fi

# Restore
for d in "${MODEL_DIRS[@]}"; do
  rm -rf "$d" 2>/dev/null
  [[ -d "${d}_bak_v12" ]] && mv "${d}_bak_v12" "$d"
done
[[ -d "results/production_bak_v12" ]] && mv "results/production_bak_v12" "results/production"
for d in results_v6_*h_prod_bak_v12; do
  [[ -d "$d" ]] && mv "$d" "${d%_bak_v12}"
done

phase_end

# ╔══════════════════════════════════════════════════════════════╗
# ║  PHASE 4: ANALYSIS                                          ║
# ╚══════════════════════════════════════════════════════════════╝
phase_start "PHASE 4: ANALYSIS & SUMMARY"

log ""
log "============================================================"
log "  OVERNIGHT v12 — FINAL RESULTS"
log "============================================================"

python - <<'PYEOF' 2>&1 | tee -a "$LOG" >> "$SUMMARY"
import json, os, glob

experiments = []
# Scan v12 results + v11 baselines for comparison
for pattern in ["results/overnight_v12/*/all_results_*.json",
                "results/overnight_v11/*/all_results_*.json",
                "results_*_research/all_results_*.json"]:
    for fp in sorted(glob.glob(pattern)):
        try:
            d = json.load(open(fp))
            avg = d.get('average', {})
            meta = d.get('meta', {})
            label = os.path.dirname(fp)
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
    experiments.sort(key=lambda x: -x.get('Sharpe', 0) if isinstance(x.get('Sharpe', 0), (int, float)) else 0)

    print(f"\n{'Label':<50} {'IC':>6} {'ICIR':>7} {'Sharpe':>7} {'MaxDD':>7} {'Ret%':>7} {'Feats':>6}")
    print("─" * 95)
    for e in experiments:
        ic = f"{e['IC']:.4f}" if isinstance(e['IC'], float) else str(e['IC'])
        icir = f"{e['ICIR']:.3f}" if isinstance(e['ICIR'], float) else str(e['ICIR'])
        sharpe = f"{e['Sharpe']:.2f}" if isinstance(e['Sharpe'], (int, float)) else str(e['Sharpe'])
        dd = f"{e['MaxDD']:.1f}" if isinstance(e['MaxDD'], (int, float)) else str(e['MaxDD'])
        ret = f"{e['Return']:.1f}" if isinstance(e['Return'], (int, float)) else str(e['Return'])
        print(f"{e['label']:<50} {ic:>6} {icir:>7} {sharpe:>7} {dd:>7} {ret:>7} {e['N_feat']:>6}")

    best = experiments[0]
    print(f"\n🏆 Best by Sharpe: {best['label']} (Sharpe={best['Sharpe']})")
PYEOF

TOTAL_TIME=$(( $(date +%s) - START_TIME ))
log ""
log "============================================================"
log "  TOTAL RUNTIME: $(( TOTAL_TIME / 3600 ))h $(( (TOTAL_TIME % 3600) / 60 ))m"
log "  LOG: $LOG"
log "  SUMMARY: $SUMMARY"
log "============================================================"
