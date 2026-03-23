#!/bin/bash
# ══════════════════════════════════════════════════════════════════════
#  run_feature_comparison.sh — A/B тест: старые фичи vs новые фичи
#
#  Новые фичи: market-mode (avg_corr, pca1_share, beta_dispersion,
#               dispersion_regime) + liquidity (dollar_volume, amihud,
#               range_per_dv, vol_price_corr)
#
#  Тренируем CatBoost huber (чемпион mega3) на 3 окнах:
#    - WinA: train→2024-04-30, test 2024-07→2024-12
#    - WinB: train→2024-10-31, test 2025-01→2025-06
#    - WinC: train→2025-04-30, test 2025-07→2025-12
#
#  Baseline = mega_comparison3 результаты (cb_solo_huber)
#  Сравниваем 1x и 3x leverage
#
#  Время: ~3 обучения × ~20 мин + 6 симуляций × ~3 мин ≈ 1.5 часа
# ══════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M)
LOGDIR="results/feature_comparison"
mkdir -p "$LOGDIR/models"
LOG="$LOGDIR/feature_comparison_${TIMESTAMP}.log"
SUMMARY="$LOGDIR/summary_${TIMESTAMP}.csv"
DETAIL_LOG="$LOGDIR/detail_${TIMESTAMP}.log"
DATA_FILE="data/features/crypto_features_1h.parquet"

# Detect GPU
USE_GPU=""
if command -v nvidia-smi &>/dev/null; then
  USE_GPU="--gpu"
fi

# ── Logging ──
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ── CSV header ──
echo "window|model|sim_cfg|ret|hac|maxdd|wr|pf|calmar|trades|costs" > "$SUMMARY"

SIM_N=0

# ── Run sim function (from mega_comparison) ──
run_sim() {
  local window="$1"
  local model_config="$2"
  local sim_config="$3"
  shift 3

  SIM_N=$((SIM_N + 1))
  local label="${window}__${model_config}__${sim_config}"
  log "SIM #${SIM_N}: $label"
  log "CMD: python run_fast_sim.py --data $DATA_FILE $*"

  local output
  output=$(python run_fast_sim.py --data "$DATA_FILE" "$@" 2>&1) || true
  echo "=== SIM #${SIM_N}: $label ===" >> "$DETAIL_LOG"
  echo "$output" >> "$DETAIL_LOG"
  echo "" >> "$DETAIL_LOG"

  local clean
  clean=$(echo "$output" | sed 's/\x1b\[[0-9;]*m//g')

  local ret hac maxdd wr pf calmar trades costs
  ret=$(echo "$clean"    | grep "Return:"     | head -1 | awk '{print $2}' | tr -d '%')
  hac=$(echo "$clean"    | grep "Sharpe HAC:" | head -1 | awk '{print $3}')
  maxdd=$(echo "$clean"  | grep "Max DD:"     | head -1 | awk '{print $3}' | tr -d '%')
  wr=$(echo "$clean"     | grep "Win Rate:"   | head -1 | awk '{print $3}' | tr -d '%')
  pf=$(echo "$clean"     | grep "^   PF:"     | head -1 | awk '{print $2}')
  calmar=$(echo "$clean" | grep "Calmar:"     | head -1 | awk '{print $2}')
  trades=$(echo "$clean" | grep "Trades:"     | head -1 | awk '{print $2}')
  costs=$(echo "$clean"  | grep "Costs:"      | head -1 | awk '{print $NF}' | tr -d '()%')

  [[ -z "$ret" ]] && ret="N/A"
  [[ -z "$hac" ]] && hac="N/A"
  [[ -z "$maxdd" ]] && maxdd="N/A"
  [[ -z "$wr" ]] && wr="N/A"
  [[ -z "$pf" ]] && pf="N/A"
  [[ -z "$calmar" ]] && calmar="N/A"
  [[ -z "$trades" ]] && trades="N/A"
  [[ -z "$costs" ]] && costs="N/A"

  ret=$(echo "$ret" | tr -d '|')
  hac=$(echo "$hac" | tr -d '|')
  maxdd=$(echo "$maxdd" | tr -d '|')
  wr=$(echo "$wr" | tr -d '|')
  pf=$(echo "$pf" | tr -d '|')
  calmar=$(echo "$calmar" | tr -d '|')
  trades=$(echo "$trades" | tr -d '|')
  costs=$(echo "$costs" | tr -d '|')

  echo "$window|$model_config|$sim_config|$ret|$hac|$maxdd|$wr|$pf|$calmar|$trades|$costs" >> "$SUMMARY"
  log "  => Return=${ret}%  HAC=${hac}  MaxDD=${maxdd}%  WR=${wr}%  PF=${pf}"
}

# ── Model directory management ──
MODEL_DIRS=(results_v6_prod results_v7_prod results_catboost_prod results_xgboost_prod)

setup_sim_models() {
  local v6_src="$1" v7_src="$2" cb_src="$3" xgb_src="$4"
  for d in "${MODEL_DIRS[@]}"; do
    rm -rf "$d" 2>/dev/null
  done
  if [[ "$cb_src" != "SKIP" && -d "$cb_src" ]]; then
    cp -r "$cb_src" results_catboost_prod
  fi
}

backup_prod_models() {
  log "📦 Backing up production models..."
  mkdir -p "$LOGDIR/prod_backup"
  for d in "${MODEL_DIRS[@]}"; do
    if [[ -d "$d" ]]; then cp -r "$d" "$LOGDIR/prod_backup/" && log "   Backed up $d"; fi
  done
  if [[ -d "results/production" ]]; then cp -r "results/production" "$LOGDIR/prod_backup/"; fi
}

restore_prod_models() {
  log "♻️  Restoring production models..."
  for d in "${MODEL_DIRS[@]}"; do
    rm -rf "$d" 2>/dev/null || true
    if [[ -d "$LOGDIR/prod_backup/$d" ]]; then cp -r "$LOGDIR/prod_backup/$d" . && log "   Restored $d"; fi
  done
  if [[ -d "$LOGDIR/prod_backup/production" ]]; then cp -r "$LOGDIR/prod_backup/production" results/; fi
}

# ── Window definitions (same as mega_comparison3) ──
declare -A WINDOWS
WINDOWS[A_train_end]="2024-04-30"
WINDOWS[A_val_start]="2024-05-08"
WINDOWS[A_val_end]="2024-06-24"
WINDOWS[A_test_start]="2024-07-01"
WINDOWS[A_test_end]="2024-12-31"
WINDOWS[A_label]="WinA"

WINDOWS[B_train_end]="2024-10-31"
WINDOWS[B_val_start]="2024-11-08"
WINDOWS[B_val_end]="2024-12-24"
WINDOWS[B_test_start]="2025-01-01"
WINDOWS[B_test_end]="2025-06-30"
WINDOWS[B_label]="WinB"

WINDOWS[C_train_end]="2025-04-30"
WINDOWS[C_val_start]="2025-05-08"
WINDOWS[C_val_end]="2025-06-24"
WINDOWS[C_test_start]="2025-07-01"
WINDOWS[C_test_end]="2025-12-31"
WINDOWS[C_label]="WinC"

PURGE_DAYS=8

# ══════════════════════════════════════════════════════════════════════
log "╔══════════════════════════════════════════════════════════════╗"
log "║  FEATURE A/B TEST: old features vs new features            ║"
log "║  Model: CatBoost huber (mega3 champion)                    ║"
log "║  New: market-mode (8) + liquidity (7) features             ║"
log "╚══════════════════════════════════════════════════════════════╝"
log ""

backup_prod_models

# ══════════════════════════════════════════════════════════════════════
#  PHASE 1: ОБУЧЕНИЕ — CatBoost huber × 3 окна
# ══════════════════════════════════════════════════════════════════════
log "################################################################"
log "  PHASE 1: ОБУЧЕНИЕ (3 окна × CatBoost huber)"
log "################################################################"

T0=$(date +%s)

for WIN in A B C; do
  train_end="${WINDOWS[${WIN}_train_end]}"
  val_end="${WINDOWS[${WIN}_val_end]}"
  label="${WINDOWS[${WIN}_label]}"
  outdir="$LOGDIR/models/${label}/catboost_huber_new_feats"

  log ""
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  log "  WINDOW $WIN: train_end=$train_end  label=$label"
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  mkdir -p "$outdir"

  t0=$(date +%s)
  python run_pipeline_catboost.py \
    --data "$DATA_FILE" \
    --results "$outdir" \
    --production \
    --train-end "$train_end" \
    --val-end "$val_end" \
    --skip-hpo \
    --seeds 5 \
    --no-news \
    --huber --huber-delta 1.0 \
    $USE_GPU \
    2>&1 | tee -a "$LOG"

  elapsed=$(( $(date +%s) - t0 ))
  log "   ✅ Done: ${label}/catboost_huber_new_feats (${elapsed}s)"
done

T1=$(date +%s)
log ""
log "✅ Phase 1 complete: All models trained"
log "   ⏱️  Phase took $(( (T1 - T0) / 60 ))m $(( (T1 - T0) % 60 ))s"

# ══════════════════════════════════════════════════════════════════════
#  PHASE 2: СИМУЛЯЦИИ — 1x и 3x на каждом окне
# ══════════════════════════════════════════════════════════════════════
log ""
log "################################################################"
log "  PHASE 2: СИМУЛЯЦИИ (3 окна × 2 leverage = 6 симов)"
log "################################################################"

T2=$(date +%s)

BASE_SIM="--leverage 1 --edge-boost --no-ddstop"
LEV3_SIM="--leverage 3 --edge-boost --no-ddstop"

for WIN in A B C; do
  test_start="${WINDOWS[${WIN}_test_start]}"
  test_end="${WINDOWS[${WIN}_test_end]}"
  label="${WINDOWS[${WIN}_label]}"
  model_dir="$LOGDIR/models/${label}/catboost_huber_new_feats"

  # Calculate days for --days
  start_epoch=$(date -j -f "%Y-%m-%d" "$test_start" "+%s" 2>/dev/null || date -d "$test_start" "+%s")
  today_epoch=$(date +%s)
  days_from_start=$(( (today_epoch - start_epoch) / 86400 + 30 ))
  SIM_PERIOD="--start-date $test_start --end-date $test_end --days $days_from_start"

  log ""
  log "╔══════════════════════════════════════════════════════════════╗"
  log "║  SIMULATING $label: $test_start → $test_end"
  log "╚══════════════════════════════════════════════════════════════╝"

  setup_sim_models "SKIP" "SKIP" "$model_dir" "SKIP"

  run_sim "$label" "cb_huber_new_feats" "1x_base" $SIM_PERIOD $BASE_SIM --ensemble
  run_sim "$label" "cb_huber_new_feats" "3x_base" $SIM_PERIOD $LEV3_SIM --ensemble
done

T3=$(date +%s)
log ""
log "✅ Phase 2 complete"
log "   ⏱️  Phase took $(( (T3 - T2) / 60 ))m $(( (T3 - T2) % 60 ))s"

# ══════════════════════════════════════════════════════════════════════
#  PHASE 3: СРАВНЕНИЕ
# ══════════════════════════════════════════════════════════════════════
log ""
log "################################################################"
log "  PHASE 3: РЕЗУЛЬТАТЫ"
log "################################################################"
log ""

# Baseline from mega_comparison3 (cb_solo_huber, 1x)
log "╔══════════════════════════════════════════════════════════════╗"
log "║  BASELINE (mega3: cb_solo_huber)                            ║"
log "╠══════════════════════════════════════════════════════════════╣"
log "║  WinA 1x: Ret=+127.0%  HAC=+6.91  MaxDD=-2.3%  PF=3.31   ║"
log "║  WinB 1x: Ret=+133.6%  HAC=+9.22  MaxDD=-3.3%  PF=2.96   ║"
log "║  WinC 1x: Ret=+66.0%   HAC=+7.17  MaxDD=-4.6%  PF=2.23   ║"
log "╚══════════════════════════════════════════════════════════════╝"
log ""
log "📊 New features results:"
cat "$SUMMARY"
log ""
log "📄 Full log: $LOG"
log "📄 Summary CSV: $SUMMARY"
log "📄 Detail log: $DETAIL_LOG"

# ══════════════════════════════════════════════════════════════════════
#  PHASE 4: ВОССТАНОВЛЕНИЕ
# ══════════════════════════════════════════════════════════════════════
restore_prod_models

log ""
log "🏁 Feature comparison complete!"
log "   Total time: $(( ($(date +%s) - T0) / 60 )) minutes"
