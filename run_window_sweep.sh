#!/bin/bash
set -uo pipefail

# ============================================================================
#  WINDOW SWEEP — Rolling vs Expanding Training Window Experiment
# ============================================================================
#
#  Phase 1: Cap-only sweep (5 configs × 3 WF windows = 15 training + 15 sims)
#    - expanding (baseline, all data from Dec 2021)
#    - cap 36 months
#    - cap 24 months
#    - cap 18 months
#    - cap 12 months
#
#  Model: CatBoost Huber solo (champion from mega3)
#  Sim config: 1x leverage, edge-boost, no-ddstop, vol-size, min-zscore 0.8
#
#  ЗАПУСК:
#    nohup ./run_window_sweep.sh > window_sweep.log 2>&1 &
#    tail -f window_sweep.log
# ============================================================================

LOGDIR="results/window_sweep"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG="$LOGDIR/run_${TIMESTAMP}.log"
SUMMARY="$LOGDIR/summary_${TIMESTAMP}.csv"
DETAIL_LOG="$LOGDIR/detail_${TIMESTAMP}.txt"
DATA="data/features"
DATA_FILE="data/features/crypto_features_1h.parquet"

USE_GPU="${USE_GPU:---gpu}"
SKIP_TRAINING="${SKIP_TRAINING:-0}"

START_TIME=$(date +%s)

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ─── CSV header ───
echo "window|cap_months|train_start|train_end|train_rows|return_pct|sharpe_hac|max_dd|win_rate|profit_factor" > "$SUMMARY"

# ─── Sim runner ───
SIM_N=0
run_sim() {
  local window="$1"
  local cap_label="$2"
  local sim_label="$3"
  shift 3

  SIM_N=$((SIM_N + 1))
  local label="${window}__${cap_label}"
  log "SIM #${SIM_N}: $label"

  local output
  output=$(python run_fast_sim.py --data "$DATA_FILE" "$@" 2>&1) || true
  echo "=== SIM #${SIM_N}: $label ===" >> "$DETAIL_LOG"
  echo "$output" >> "$DETAIL_LOG"
  echo "" >> "$DETAIL_LOG"

  local clean
  clean=$(echo "$output" | sed 's/\x1b\[[0-9;]*m//g')

  local ret hac maxdd wr pf
  ret=$(echo "$clean"    | grep "Return:"     | head -1 | awk '{print $2}' | tr -d '%')
  hac=$(echo "$clean"    | grep "Sharpe HAC:" | head -1 | awk '{print $3}')
  maxdd=$(echo "$clean"  | grep "Max DD:"     | head -1 | awk '{print $3}' | tr -d '%')
  wr=$(echo "$clean"     | grep "Win Rate:"   | head -1 | awk '{print $3}' | tr -d '%')
  pf=$(echo "$clean"     | grep "^   PF:"     | head -1 | awk '{print $2}')

  [[ -z "$ret" ]] && ret="N/A"
  [[ -z "$hac" ]] && hac="N/A"
  [[ -z "$maxdd" ]] && maxdd="N/A"
  [[ -z "$wr" ]] && wr="N/A"
  [[ -z "$pf" ]] && pf="N/A"

  echo "$window|$cap_label|$sim_label|||$ret|$hac|$maxdd|$wr|$pf" >> "$SUMMARY"
  log "  => Return=${ret}%  HAC=${hac}  MaxDD=${maxdd}%  WR=${wr}%  PF=${pf}"
}

# ─── Model dirs management ───
MODEL_DIRS=(results_catboost_prod)

backup_prod_models() {
  log "📦 Backing up production models..."
  mkdir -p "$LOGDIR/prod_backup"
  for d in "${MODEL_DIRS[@]}"; do
    if [[ -d "$d" ]]; then
      cp -r "$d" "$LOGDIR/prod_backup/"
    fi
  done
}

restore_prod_models() {
  log "🔄 Restoring production models..."
  for d in "${MODEL_DIRS[@]}"; do
    if [[ -d "$LOGDIR/prod_backup/$d" ]]; then
      rm -rf "$d" 2>/dev/null
      cp -r "$LOGDIR/prod_backup/$d" "$d"
    fi
  done
}

setup_sim_models() {
  local cb_src="$1"
  rm -rf results_v6_prod results_v7_prod results_catboost_prod results_xgboost_prod 2>/dev/null
  if [[ -d "$cb_src" ]]; then
    cp -r "$cb_src" results_catboost_prod
  fi
}

# ============================================================================
#  WALK-FORWARD WINDOWS (same as mega_comparison3)
# ============================================================================
declare -A WINDOWS

WINDOWS[A_train_end]="2024-04-30"
WINDOWS[A_val_end]="2024-06-24"
WINDOWS[A_test_start]="2024-07-01"
WINDOWS[A_test_end]="2024-12-31"
WINDOWS[A_label]="WinA"

WINDOWS[B_train_end]="2024-10-31"
WINDOWS[B_val_end]="2024-12-24"
WINDOWS[B_test_start]="2025-01-01"
WINDOWS[B_test_end]="2025-06-30"
WINDOWS[B_label]="WinB"

WINDOWS[C_train_end]="2025-04-30"
WINDOWS[C_val_end]="2025-06-24"
WINDOWS[C_test_start]="2025-07-01"
WINDOWS[C_test_end]="2025-12-31"
WINDOWS[C_label]="WinC"

# ============================================================================
#  CAP CONFIGURATIONS
# ============================================================================
# For each (window, cap), compute train_start = train_end - cap_months
# "expanding" = no --train-start (use all data)

compute_train_start() {
  local train_end="$1"
  local cap_months="$2"

  if [[ "$cap_months" == "0" ]]; then
    echo ""  # expanding — no cap
    return
  fi

  # Use python for reliable date math (no external deps)
  python3 -c "
from datetime import datetime
d = datetime.strptime('$train_end', '%Y-%m-%d')
y = d.year
m = d.month - $cap_months
while m <= 0:
    m += 12
    y -= 1
day = min(d.day, 28)
print(f'{y:04d}-{m:02d}-{day:02d}')
"
}

# Cap configs: 0=expanding, then 36, 24, 18, 12
CAP_MONTHS=(0 36 24 18 12)
CAP_LABELS=("expanding" "cap36m" "cap24m" "cap18m" "cap12m")

# ============================================================================
#  PHASE 1: TRAINING
# ============================================================================

log "=================================================================="
log "  WINDOW SWEEP EXPERIMENT"
log "  Configs: ${CAP_LABELS[*]}"
log "  Windows: A B C"
log "  Total trains: $(( ${#CAP_MONTHS[@]} * 3 ))"
log "  Total sims:   $(( ${#CAP_MONTHS[@]} * 3 ))"
log "=================================================================="

backup_prod_models

if [[ "$SKIP_TRAINING" != "1" ]]; then

log ""
log "################################################################"
log "  PHASE 1: TRAINING (${#CAP_MONTHS[@]} caps × 3 windows)"
log "################################################################"

PHASE1_T0=$(date +%s)

for WIN in A B C; do
  local_train_end="${WINDOWS[${WIN}_train_end]}"
  local_val_end="${WINDOWS[${WIN}_val_end]}"
  local_label="${WINDOWS[${WIN}_label]}"

  log ""
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  log "  WINDOW $WIN: train_end=$local_train_end"
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  for i in "${!CAP_MONTHS[@]}"; do
    cap="${CAP_MONTHS[$i]}"
    cap_label="${CAP_LABELS[$i]}"
    train_start=$(compute_train_start "$local_train_end" "$cap")

    outdir="$LOGDIR/models/${local_label}/${cap_label}"
    mkdir -p "$outdir"

    train_start_flag=""
    train_start_info="start"
    if [[ -n "$train_start" ]]; then
      train_start_flag="--train-start $train_start"
      train_start_info="$train_start"
    fi

    log "🧠 Training: ${local_label} / ${cap_label}"
    log "   train: $train_start_info → $local_train_end"

    t0=$(date +%s)
    python run_pipeline_catboost.py \
      --data "$DATA" \
      --results "$outdir" \
      --production \
      --train-end "$local_train_end" \
      --val-end "$local_val_end" \
      --skip-hpo \
      --seeds 5 \
      --huber \
      --no-news \
      $train_start_flag \
      $USE_GPU \
      2>&1 | tee -a "$LOG"

    elapsed=$(( $(date +%s) - t0 ))
    log "   ✅ Done: ${cap_label} (${elapsed}s)"
  done
done

PHASE1_ELAPSED=$(( $(date +%s) - PHASE1_T0 ))
log ""
log "✅ Phase 1 complete: $(( PHASE1_ELAPSED / 60 ))m $(( PHASE1_ELAPSED % 60 ))s"

fi  # end SKIP_TRAINING

# ============================================================================
#  PHASE 2: SIMULATIONS
# ============================================================================

log ""
log "################################################################"
log "  PHASE 2: SIMULATIONS"
log "################################################################"

PHASE2_T0=$(date +%s)

# Sim config = champion config from mega3 (cb_solo_huber, 1x, vol-size, min-zscore)
BASE_SIM="--leverage 1 --edge-boost --no-ddstop --vol-size --min-zscore 0.8"

for WIN in A B C; do
  local_label="${WINDOWS[${WIN}_label]}"
  test_start="${WINDOWS[${WIN}_test_start]}"
  test_end="${WINDOWS[${WIN}_test_end]}"

  # Compute --days to cover test window
  today_epoch=$(date +%s)
  start_epoch=$(date -j -f "%Y-%m-%d" "$test_start" "+%s" 2>/dev/null || date -d "$test_start" "+%s")
  days_from_start=$(( (today_epoch - start_epoch) / 86400 + 30 ))

  SIM_PERIOD="--start-date $test_start --end-date $test_end --days $days_from_start"

  log ""
  log "╔══════════════════════════════════════════════════════════════╗"
  log "║  SIMULATING $local_label: $test_start → $test_end"
  log "╚══════════════════════════════════════════════════════════════╝"

  for i in "${!CAP_MONTHS[@]}"; do
    cap="${CAP_MONTHS[$i]}"
    cap_label="${CAP_LABELS[$i]}"
    model_dir="$LOGDIR/models/${local_label}/${cap_label}"

    if [[ ! -d "$model_dir" ]]; then
      log "⚠️  Model dir not found: $model_dir, skipping"
      continue
    fi

    # Get train info from model
    train_start=$(compute_train_start "${WINDOWS[${WIN}_train_end]}" "$cap")
    ts_info="${train_start:-all_data}"

    setup_sim_models "$model_dir"
    run_sim "$local_label" "$cap_label" "$ts_info" $SIM_PERIOD $BASE_SIM --ensemble
  done
done

PHASE2_ELAPSED=$(( $(date +%s) - PHASE2_T0 ))
log ""
log "✅ Phase 2 complete: $(( PHASE2_ELAPSED / 60 ))m $(( PHASE2_ELAPSED % 60 ))s"

# ============================================================================
#  PHASE 3: RESULTS
# ============================================================================

log ""
log "################################################################"
log "  PHASE 3: RESULTS"
log "################################################################"

log ""
log "Raw results: $SUMMARY"
log ""

# Print formatted table
log "| Window | Cap | Return% | HAC | MaxDD% | WR% | PF |"
log "|--------|-----|---------|-----|--------|-----|-----|"
while IFS='|' read -r window cap ts _ _ ret hac maxdd wr pf; do
  [[ "$window" == "window" ]] && continue  # skip header
  printf "| %-6s | %-10s | %7s | %6s | %6s | %3s | %4s |\n" \
    "$window" "$cap" "$ret" "$hac" "$maxdd" "$wr" "$pf" | tee -a "$LOG"
done < "$SUMMARY"

# Restore production models
restore_prod_models

TOTAL_ELAPSED=$(( $(date +%s) - START_TIME ))
log ""
log "=================================================================="
log "  WINDOW SWEEP COMPLETE"
log "  Total time: $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
log "  Sims: $SIM_N"
log "  Results: $SUMMARY"
log "=================================================================="
