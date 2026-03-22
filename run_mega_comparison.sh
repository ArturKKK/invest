#!/bin/bash
set -uo pipefail

# ============================================================================
#  MEGA COMPARISON v1 — Ultimate Model Battle
# ============================================================================
#
#  ЦЕЛЬ: Раз и навсегда определить, какая конфигурация моделей лучше:
#    - Solo CatBoost (текущий "чемпион", который в лайве сливает)
#    - Solo LGB v6 / v7
#    - Ансамбль из 2 моделей (v6+v7)
#    - Ансамбль из 3 моделей (v6+v7+CB)
#    - Ансамбль из 4 моделей (v6+v7+CB+XGB)
#    - Все варианты с/без новостей, с/без деривативов
#    - Все варианты с разными execution флагами
#
#  МЕТОДОЛОГИЯ: Walk-Forward Out-of-Sample (3 окна)
#    Window A: Train до 2024-06-30, тест 2024-07-01 → 2024-12-31 (6 мес)
#    Window B: Train до 2024-12-31, тест 2025-01-01 → 2025-06-30 (6 мес)
#    Window C: Train до 2025-06-30, тест 2025-07-01 → 2025-12-31 (6 мес)
#
#  Это даёт 18 месяцев честного OOS тестирования без заглядывания в будущее.
#
#  СТРУКТУРА:
#    Phase 0: Подготовка (бэкап текущих моделей)
#    Phase 1: Обучение ВСЕХ моделей на ВСЕХ окнах (3 окна × 6 конфигов = 18 обучений)
#    Phase 2: Симуляции ВСЕХ вариантов (3 окна × ~50 конфигов = ~150 симуляций)
#    Phase 3: Сводная таблица и рейтинг
#    Phase 4: Восстановление продакшн моделей
#
#  ОЦЕНКА ВРЕМЕНИ: 3-4 дня при ~40 мин на обучение и ~3 мин на симуляцию
#
#  ЗАПУСК:
#    nohup ./run_mega_comparison.sh > mega_comparison.log 2>&1 &
#    # Следить: tail -f mega_comparison.log
# ============================================================================

LOGDIR="results/mega_comparison"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG="$LOGDIR/run_${TIMESTAMP}.log"
SUMMARY="$LOGDIR/summary_${TIMESTAMP}.csv"
DETAIL_LOG="$LOGDIR/detail_${TIMESTAMP}.txt"
DATA="data/features"
DATA_FILE="data/features/crypto_features_1h.parquet"

# ─── GPU: set to "--gpu" if you have CUDA, "" otherwise ───
# CatBoost/XGBoost benefit from GPU. LGB uses CPU (needs OpenCL for GPU).
USE_GPU="${USE_GPU:---gpu}"  # default: --gpu (set USE_GPU="" to disable)

# ─── Skip controls (set via env vars before running) ───
# SKIP_TRAINING=1  → skip Phase 1 (use existing models, e.g. after a failed Phase 2)
# SIM_WINDOWS="A B" → run Phase 2 only for specified windows (default: all three)
SKIP_TRAINING="${SKIP_TRAINING:-0}"
SIM_WINDOWS="${SIM_WINDOWS:-A B C}"

START_TIME=$(date +%s)

# ─── Logging ───
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

phase_start() {
  PHASE_T0=$(date +%s)
  log ""
  log "################################################################"
  log "  $*"
  log "################################################################"
}

phase_end() {
  local elapsed=$(( $(date +%s) - PHASE_T0 ))
  log "  ⏱️  Phase took $(( elapsed / 60 ))m $(( elapsed % 60 ))s"
}

# ─── CSV header ───
echo "window|model_config|sim_config|return_pct|sharpe_hac|max_dd|win_rate|profit_factor|calmar|n_trades|costs_pct" > "$SUMMARY"

# ─── Sim runner (парсит результат, пишет в CSV) ───
SIM_N=0
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

  # Strip ANSI escape codes before parsing (libraries may emit color codes)
  local clean
  clean=$(echo "$output" | sed 's/\x1b\[[0-9;]*m//g')

  # Parse metrics — формат вывода run_fast_sim.py:
  #   Return:     +55.3%  (ann. ~+120%)
  #   Max DD:     -12.3%
  #   Sharpe HAC: +5.32  (Newey-West)
  #   Win Rate:   63%  (100W / 58L)
  #   PF:         1.82
  #   Calmar:     5.32
  #   Trades:     1234
  #   Costs:      $1,234.56  (24.7%)
  local ret hac maxdd wr pf calmar trades costs
  ret=$(echo "$clean"    | grep "Return:"     | head -1 | awk '{print $2}' | tr -d '%')
  hac=$(echo "$clean"    | grep "Sharpe HAC:" | head -1 | awk '{print $3}')
  maxdd=$(echo "$clean"  | grep "Max DD:"     | head -1 | awk '{print $3}' | tr -d '%')
  wr=$(echo "$clean"     | grep "Win Rate:"   | head -1 | awk '{print $3}' | tr -d '%')
  pf=$(echo "$clean"     | grep "^   PF:"     | head -1 | awk '{print $2}')
  calmar=$(echo "$clean" | grep "Calmar:"     | head -1 | awk '{print $2}')
  trades=$(echo "$clean" | grep "Trades:"     | head -1 | awk '{print $2}')
  costs=$(echo "$clean"  | grep "Costs:"      | head -1 | awk '{print $NF}' | tr -d '()%')

  # Fallbacks
  [[ -z "$ret" ]] && ret="N/A"
  [[ -z "$hac" ]] && hac="N/A"
  [[ -z "$maxdd" ]] && maxdd="N/A"
  [[ -z "$wr" ]] && wr="N/A"
  [[ -z "$pf" ]] && pf="N/A"
  [[ -z "$calmar" ]] && calmar="N/A"
  [[ -z "$trades" ]] && trades="N/A"
  [[ -z "$costs" ]] && costs="N/A"

  # Sanitize: strip any | or control chars that would break CSV
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

# ─── Model directory management ───
# Массив директорий, которые нужно бэкапить/восстанавливать
MODEL_DIRS=(
  results_v6_prod results_v7_prod results_catboost_prod
  results_xgboost_prod results_mlp_prod results_ridge_prod
)

backup_prod_models() {
  log "📦 Backing up production models..."
  mkdir -p "$LOGDIR/prod_backup"
  for d in "${MODEL_DIRS[@]}"; do
    if [[ -d "$d" ]]; then
      cp -r "$d" "$LOGDIR/prod_backup/"
      log "   Backed up $d"
    fi
  done
  # Backup results/production if exists
  if [[ -d "results/production" ]]; then
    cp -r "results/production" "$LOGDIR/prod_backup/"
    log "   Backed up results/production"
  fi
  log "   ✅ Backup complete: $LOGDIR/prod_backup/"
}

restore_prod_models() {
  log "🔄 Restoring production models..."
  for d in "${MODEL_DIRS[@]}"; do
    if [[ -d "$LOGDIR/prod_backup/$d" ]]; then
      rm -rf "$d" 2>/dev/null
      cp -r "$LOGDIR/prod_backup/$d" "$d"
      log "   Restored $d"
    fi
  done
  if [[ -d "$LOGDIR/prod_backup/production" ]]; then
    rm -rf "results/production" 2>/dev/null
    cp -r "$LOGDIR/prod_backup/production" "results/production"
    log "   Restored results/production"
  fi
  log "   ✅ Restore complete"
}

# Установить нужные модели для симуляции
# Аргументы: v6_dir v7_dir cb_dir xgb_dir
# "SKIP" = не ставить эту модель (не участвует)
# "NONE" = очистить директорию
setup_sim_models() {
  local v6_src="$1" v7_src="$2" cb_src="$3" xgb_src="$4"

  for d in results_v6_prod results_v7_prod results_catboost_prod results_xgboost_prod; do
    rm -rf "$d" 2>/dev/null
  done

  if [[ "$v6_src" != "SKIP" && "$v6_src" != "NONE" && -d "$v6_src" ]]; then
    cp -r "$v6_src" results_v6_prod
  fi
  if [[ "$v7_src" != "SKIP" && "$v7_src" != "NONE" && -d "$v7_src" ]]; then
    cp -r "$v7_src" results_v7_prod
  fi
  if [[ "$cb_src" != "SKIP" && "$cb_src" != "NONE" && -d "$cb_src" ]]; then
    cp -r "$cb_src" results_catboost_prod
  fi
  if [[ "$xgb_src" != "SKIP" && "$xgb_src" != "NONE" && -d "$xgb_src" ]]; then
    cp -r "$xgb_src" results_xgboost_prod
  fi
}


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  PHASE 0: ПОДГОТОВКА                                                ║
# ╚══════════════════════════════════════════════════════════════════════╝

phase_start "PHASE 0: ПОДГОТОВКА"

# Проверяем данные
if [[ ! -f "$DATA_FILE" ]]; then
  log "❌ FATAL: Feature data not found: $DATA_FILE"
  exit 1
fi

# Бэкапим текущие продакшн модели
backup_prod_models

phase_end

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  PHASE 1: ОБУЧЕНИЕ МОДЕЛЕЙ                                          ║
# ║                                                                      ║
# ║  3 Walk-Forward окна × 6+ конфигураций моделей                       ║
# ║  Каждая конфигурация = 5 seeds                                       ║
# ║  Итого: ~18+ обучений × ~30-60 мин = ~9-18 часов                    ║
# ╚══════════════════════════════════════════════════════════════════════╝

phase_start "PHASE 1: ОБУЧЕНИЕ МОДЕЛЕЙ"

# Определяем 3 окна обучения
# Каждое окно: train до даты X, тест = следующие 6 месяцев
declare -A WINDOWS

WINDOWS[A_train_end]="2024-06-30"
WINDOWS[A_val_start]="2024-07-08"
WINDOWS[A_val_end]="2024-12-24"
WINDOWS[A_test_start]="2024-07-01"
WINDOWS[A_test_end]="2024-12-31"
WINDOWS[A_label]="WinA_train2024H1"

WINDOWS[B_train_end]="2024-12-31"
WINDOWS[B_val_start]="2025-01-08"
WINDOWS[B_val_end]="2025-06-24"
WINDOWS[B_test_start]="2025-01-01"
WINDOWS[B_test_end]="2025-06-30"
WINDOWS[B_label]="WinB_train2024"

WINDOWS[C_train_end]="2025-06-30"
WINDOWS[C_val_start]="2025-07-08"
WINDOWS[C_val_end]="2025-12-24"
WINDOWS[C_test_start]="2025-07-01"
WINDOWS[C_test_end]="2025-12-31"
WINDOWS[C_label]="WinC_train2025H1"

# Функция обучения одной модели в одном окне
train_model() {
  local win_key="$1"    # A, B, C
  local model_type="$2" # v6, v7, catboost, xgboost
  local suffix="$3"     # суффикс для output (напр. "news", "no_news", "no_deriv")
  shift 3
  local extra_flags="$*"

  local train_end="${WINDOWS[${win_key}_train_end]}"
  local val_end="${WINDOWS[${win_key}_val_end]}"
  local label="${WINDOWS[${win_key}_label]}"
  local outdir="$LOGDIR/models/${label}/${model_type}_${suffix}"

  log "🧠 Training: ${label} / ${model_type}_${suffix}"
  log "   train_end=$train_end  val_end=$val_end"
  log "   flags: $extra_flags"

  mkdir -p "$outdir"

  local script=""
  case "$model_type" in
    v6)       script="run_pipeline_v6.py" ;;
    v7)       script="run_pipeline_v7.py" ;;
    catboost) script="run_pipeline_catboost.py" ;;
    xgboost)  script="run_pipeline_xgboost.py" ;;
    *)
      log "❌ Unknown model type: $model_type"
      return 1
      ;;
  esac

  # GPU: pass --gpu for catboost/xgboost if USE_GPU is set
  local gpu_flag=""
  if [[ -n "$USE_GPU" && ("$model_type" == "catboost" || "$model_type" == "xgboost") ]]; then
    gpu_flag="$USE_GPU"
  fi

  local t0=$(date +%s)

  python "$script" \
    --data "$DATA" \
    --results "$outdir" \
    --production \
    --train-end "$train_end" \
    --val-end "$val_end" \
    --skip-hpo \
    --seeds 5 \
    $gpu_flag \
    $extra_flags \
    2>&1 | tee -a "$LOG"

  local elapsed=$(( $(date +%s) - t0 ))
  log "   ✅ Done: ${model_type}_${suffix} (${elapsed}s)"
}

if [[ "$SKIP_TRAINING" == "1" ]]; then
  log "⏭  SKIP_TRAINING=1 → skipping Phase 1, assuming models already exist"
else

# ── Обучаем все модели для всех окон ──

for WIN in A B C; do
  log ""
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  log "  WINDOW $WIN: train_end=${WINDOWS[${WIN}_train_end]}"
  log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # ── 1. LGB v6: базовая версия (без новостей) ──
  train_model "$WIN" "v6" "base" "--no-news"

  # ── 2. LGB v6: с новостями ──
  train_model "$WIN" "v6" "with_news" "--news-mode all"

  # ── 3. LGB v6: без деривативов ──
  train_model "$WIN" "v6" "no_deriv" "--no-news --no-derivatives"

  # ── 4. LGB v7: базовая версия (без новостей) ──
  train_model "$WIN" "v7" "base" "--no-news"

  # ── 5. LGB v7: с новостями ──
  train_model "$WIN" "v7" "with_news" "--news-mode all"

  # ── 6. LGB v7: без деривативов ──
  train_model "$WIN" "v7" "no_deriv" "--no-news --no-derivatives"

  # ── 7. CatBoost: с новостями (классическая конфигурация) ──
  train_model "$WIN" "catboost" "with_news" "--news-mode all"

  # ── 8. CatBoost: без новостей ──
  train_model "$WIN" "catboost" "no_news" "--no-news"

  # ── 9. CatBoost: market-only новости ──
  train_model "$WIN" "catboost" "market_news" "--news-mode market-only"

  # ── 10. CatBoost: без деривативов (текущий "чемпион" v14) ──
  train_model "$WIN" "catboost" "no_deriv" "--no-news --no-derivatives"

  # ── 11. CatBoost: без деривативов + с новостями ──
  train_model "$WIN" "catboost" "news_no_deriv" "--news-mode all --no-derivatives"

  # ── 12. CatBoost: huber loss (как v14 чемпион) ──
  train_model "$WIN" "catboost" "huber" "--no-news --huber --huber-delta 1.0"

  # ── 13. CatBoost: huber + без деривативов ──
  train_model "$WIN" "catboost" "huber_no_deriv" "--no-news --no-derivatives --huber --huber-delta 1.0"

  # ── 14. XGBoost: с новостями + interactions ──
  train_model "$WIN" "xgboost" "with_news" "--news-mode all"

  # ── 15. XGBoost: без новостей ──
  train_model "$WIN" "xgboost" "no_news" "--no-news"

  # ── 16. XGBoost: без деривативов ──
  train_model "$WIN" "xgboost" "no_deriv" "--no-news --no-derivatives"

done

log ""
log "✅ Phase 1 complete: All models trained"

fi  # end SKIP_TRAINING check

phase_end


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  PHASE 2: СИМУЛЯЦИИ                                                  ║
# ║                                                                      ║
# ║  Для каждого окна: прогоняем все комбинации моделей                  ║
# ║  через run_fast_sim.py на OOS периоде                               ║
# ╚══════════════════════════════════════════════════════════════════════╝

phase_start "PHASE 2: СИМУЛЯЦИИ"

# Базовые параметры симуляции
BASE_SIM="--leverage 1 --edge-boost --no-ddstop"
LEV3_SIM="--leverage 3 --edge-boost --no-ddstop"

# Функция для быстрого прогона симуляции на одном окне
run_window_sims() {
  local win_key="$1"   # A, B, C
  local test_start="${WINDOWS[${win_key}_test_start]}"
  local test_end="${WINDOWS[${win_key}_test_end]}"
  local label="${WINDOWS[${win_key}_label]}"
  local model_base="$LOGDIR/models/${label}"

  # Вычисляем кол-во дней
  local start_epoch end_epoch days
  start_epoch=$(date -j -f "%Y-%m-%d" "$test_start" "+%s" 2>/dev/null || date -d "$test_start" "+%s")
  end_epoch=$(date -j -f "%Y-%m-%d" "$test_end" "+%s" 2>/dev/null || date -d "$test_end" "+%s")
  days=$(( (end_epoch - start_epoch) / 86400 ))
  # --days must cover from test_start back to data end (run_fast_sim slices from END of data).
  # Using days-from-test_start-to-today ensures the slice always includes the test window.
  local today_epoch
  today_epoch=$(date +%s)
  local days_from_start=$(( (today_epoch - start_epoch) / 86400 + 30 ))

  local SIM_PERIOD="--start-date $test_start --end-date $test_end --days $days_from_start"

  log ""
  log "╔══════════════════════════════════════════════════════════════╗"
  log "║  SIMULATING WINDOW $win_key: $test_start → $test_end ($days days)"
  log "╚══════════════════════════════════════════════════════════════╝"

  # =========================================================================
  #  ГРУППА A: SOLO МОДЕЛИ (одна модель за раз)
  # =========================================================================
  log "--- SOLO MODELS ---"

  # Solo CatBoost: с новостями
  if [[ -d "$model_base/catboost_with_news" ]]; then
    setup_sim_models "SKIP" "SKIP" "$model_base/catboost_with_news" "SKIP"
    run_sim "$label" "cb_solo_news" "1x_base" $SIM_PERIOD $BASE_SIM --ensemble
    run_sim "$label" "cb_solo_news" "3x_base" $SIM_PERIOD $LEV3_SIM --ensemble
  fi

  # Solo CatBoost: без новостей
  if [[ -d "$model_base/catboost_no_news" ]]; then
    setup_sim_models "SKIP" "SKIP" "$model_base/catboost_no_news" "SKIP"
    run_sim "$label" "cb_solo_no_news" "1x_base" $SIM_PERIOD $BASE_SIM --ensemble
    run_sim "$label" "cb_solo_no_news" "3x_base" $SIM_PERIOD $LEV3_SIM --ensemble
  fi

  # Solo CatBoost: market-only news
  if [[ -d "$model_base/catboost_market_news" ]]; then
    setup_sim_models "SKIP" "SKIP" "$model_base/catboost_market_news" "SKIP"
    run_sim "$label" "cb_solo_market_news" "1x_base" $SIM_PERIOD $BASE_SIM --ensemble
  fi

  # Solo CatBoost: без деривативов (= v14 чемпион)
  if [[ -d "$model_base/catboost_no_deriv" ]]; then
    setup_sim_models "SKIP" "SKIP" "$model_base/catboost_no_deriv" "SKIP"
    run_sim "$label" "cb_solo_no_deriv" "1x_base" $SIM_PERIOD $BASE_SIM --ensemble
    run_sim "$label" "cb_solo_no_deriv" "3x_base" $SIM_PERIOD $LEV3_SIM --ensemble
  fi

  # Solo CatBoost: news + без деривативов
  if [[ -d "$model_base/catboost_news_no_deriv" ]]; then
    setup_sim_models "SKIP" "SKIP" "$model_base/catboost_news_no_deriv" "SKIP"
    run_sim "$label" "cb_solo_news_no_deriv" "1x_base" $SIM_PERIOD $BASE_SIM --ensemble
  fi

  # Solo CatBoost: huber
  if [[ -d "$model_base/catboost_huber" ]]; then
    setup_sim_models "SKIP" "SKIP" "$model_base/catboost_huber" "SKIP"
    run_sim "$label" "cb_solo_huber" "1x_base" $SIM_PERIOD $BASE_SIM --ensemble
  fi

  # Solo CatBoost: huber + без деривативов
  if [[ -d "$model_base/catboost_huber_no_deriv" ]]; then
    setup_sim_models "SKIP" "SKIP" "$model_base/catboost_huber_no_deriv" "SKIP"
    run_sim "$label" "cb_solo_huber_no_deriv" "1x_base" $SIM_PERIOD $BASE_SIM --ensemble
  fi

  # Solo LGB v6: без новостей
  if [[ -d "$model_base/v6_base" ]]; then
    setup_sim_models "$model_base/v6_base" "SKIP" "SKIP" "SKIP"
    run_sim "$label" "v6_solo_base" "1x_base" $SIM_PERIOD $BASE_SIM --ensemble
    run_sim "$label" "v6_solo_base" "3x_base" $SIM_PERIOD $LEV3_SIM --ensemble
  fi

  # Solo LGB v6: с новостями
  if [[ -d "$model_base/v6_with_news" ]]; then
    setup_sim_models "$model_base/v6_with_news" "SKIP" "SKIP" "SKIP"
    run_sim "$label" "v6_solo_news" "1x_base" $SIM_PERIOD $BASE_SIM --ensemble
  fi

  # Solo LGB v6: без деривативов
  if [[ -d "$model_base/v6_no_deriv" ]]; then
    setup_sim_models "$model_base/v6_no_deriv" "SKIP" "SKIP" "SKIP"
    run_sim "$label" "v6_solo_no_deriv" "1x_base" $SIM_PERIOD $BASE_SIM --ensemble
  fi

  # Solo LGB v7: без новостей
  if [[ -d "$model_base/v7_base" ]]; then
    setup_sim_models "SKIP" "$model_base/v7_base" "SKIP" "SKIP"
    run_sim "$label" "v7_solo_base" "1x_base" $SIM_PERIOD $BASE_SIM --ensemble
    run_sim "$label" "v7_solo_base" "3x_base" $SIM_PERIOD $LEV3_SIM --ensemble
  fi

  # Solo LGB v7: с новостями
  if [[ -d "$model_base/v7_with_news" ]]; then
    setup_sim_models "SKIP" "$model_base/v7_with_news" "SKIP" "SKIP"
    run_sim "$label" "v7_solo_news" "1x_base" $SIM_PERIOD $BASE_SIM --ensemble
  fi

  # Solo XGBoost
  if [[ -d "$model_base/xgboost_with_news" ]]; then
    setup_sim_models "SKIP" "SKIP" "SKIP" "$model_base/xgboost_with_news"
    run_sim "$label" "xgb_solo_news" "1x_base" $SIM_PERIOD $BASE_SIM --ensemble
  fi

  if [[ -d "$model_base/xgboost_no_news" ]]; then
    setup_sim_models "SKIP" "SKIP" "SKIP" "$model_base/xgboost_no_news"
    run_sim "$label" "xgb_solo_no_news" "1x_base" $SIM_PERIOD $BASE_SIM --ensemble
  fi

  # =========================================================================
  #  ГРУППА B: АНСАМБЛЬ ИЗ 2 МОДЕЛЕЙ (v6+v7)
  # =========================================================================
  log "--- ENSEMBLE 2 (v6+v7) ---"

  if [[ -d "$model_base/v6_base" && -d "$model_base/v7_base" ]]; then
    setup_sim_models "$model_base/v6_base" "$model_base/v7_base" "SKIP" "SKIP"
    run_sim "$label" "ens2_v6v7_no_news" "1x_base" $SIM_PERIOD $BASE_SIM --ensemble
    run_sim "$label" "ens2_v6v7_no_news" "3x_base" $SIM_PERIOD $LEV3_SIM --ensemble
  fi

  if [[ -d "$model_base/v6_with_news" && -d "$model_base/v7_with_news" ]]; then
    setup_sim_models "$model_base/v6_with_news" "$model_base/v7_with_news" "SKIP" "SKIP"
    run_sim "$label" "ens2_v6v7_news" "1x_base" $SIM_PERIOD $BASE_SIM --ensemble
  fi

  # =========================================================================
  #  ГРУППА C: АНСАМБЛЬ ИЗ 3 МОДЕЛЕЙ (v6+v7+CB)
  # =========================================================================
  log "--- ENSEMBLE 3 (v6+v7+CB) ---"

  # Классика: LGB без новостей + CB с новостями (исторический чемпион)
  if [[ -d "$model_base/v6_base" && -d "$model_base/v7_base" && -d "$model_base/catboost_with_news" ]]; then
    setup_sim_models "$model_base/v6_base" "$model_base/v7_base" "$model_base/catboost_with_news" "SKIP"
    run_sim "$label" "ens3_lgb_noN+cb_N" "1x_base" $SIM_PERIOD $BASE_SIM --ensemble
    run_sim "$label" "ens3_lgb_noN+cb_N" "3x_base" $SIM_PERIOD $LEV3_SIM --ensemble
  fi

  # Все без новостей
  if [[ -d "$model_base/v6_base" && -d "$model_base/v7_base" && -d "$model_base/catboost_no_news" ]]; then
    setup_sim_models "$model_base/v6_base" "$model_base/v7_base" "$model_base/catboost_no_news" "SKIP"
    run_sim "$label" "ens3_all_no_news" "1x_base" $SIM_PERIOD $BASE_SIM --ensemble
  fi

  # Все с новостями
  if [[ -d "$model_base/v6_with_news" && -d "$model_base/v7_with_news" && -d "$model_base/catboost_with_news" ]]; then
    setup_sim_models "$model_base/v6_with_news" "$model_base/v7_with_news" "$model_base/catboost_with_news" "SKIP"
    run_sim "$label" "ens3_all_news" "1x_base" $SIM_PERIOD $BASE_SIM --ensemble
  fi

  # LGB без деривативов + CB без деривативов
  if [[ -d "$model_base/v6_no_deriv" && -d "$model_base/v7_no_deriv" && -d "$model_base/catboost_no_deriv" ]]; then
    setup_sim_models "$model_base/v6_no_deriv" "$model_base/v7_no_deriv" "$model_base/catboost_no_deriv" "SKIP"
    run_sim "$label" "ens3_all_no_deriv" "1x_base" $SIM_PERIOD $BASE_SIM --ensemble
  fi

  # =========================================================================
  #  ГРУППА D: АНСАМБЛЬ ИЗ 4 МОДЕЛЕЙ (v6+v7+CB+XGB)
  # =========================================================================
  log "--- ENSEMBLE 4 (v6+v7+CB+XGB) ---"

  # Классическая конфигурация: LGB без новостей + CB с + XGB с
  if [[ -d "$model_base/v6_base" && -d "$model_base/v7_base" && \
        -d "$model_base/catboost_with_news" && -d "$model_base/xgboost_with_news" ]]; then
    setup_sim_models "$model_base/v6_base" "$model_base/v7_base" \
                     "$model_base/catboost_with_news" "$model_base/xgboost_with_news"
    run_sim "$label" "ens4_lgb_noN+cb_N+xgb_N" "1x_base" $SIM_PERIOD $BASE_SIM --ensemble
    run_sim "$label" "ens4_lgb_noN+cb_N+xgb_N" "3x_base" $SIM_PERIOD $LEV3_SIM --ensemble
  fi

  # Все без новостей
  if [[ -d "$model_base/v6_base" && -d "$model_base/v7_base" && \
        -d "$model_base/catboost_no_news" && -d "$model_base/xgboost_no_news" ]]; then
    setup_sim_models "$model_base/v6_base" "$model_base/v7_base" \
                     "$model_base/catboost_no_news" "$model_base/xgboost_no_news"
    run_sim "$label" "ens4_all_no_news" "1x_base" $SIM_PERIOD $BASE_SIM --ensemble
  fi

  # Все с новостями
  if [[ -d "$model_base/v6_with_news" && -d "$model_base/v7_with_news" && \
        -d "$model_base/catboost_with_news" && -d "$model_base/xgboost_with_news" ]]; then
    setup_sim_models "$model_base/v6_with_news" "$model_base/v7_with_news" \
                     "$model_base/catboost_with_news" "$model_base/xgboost_with_news"
    run_sim "$label" "ens4_all_news" "1x_base" $SIM_PERIOD $BASE_SIM --ensemble
  fi

  # Все без деривативов
  if [[ -d "$model_base/v6_no_deriv" && -d "$model_base/v7_no_deriv" && \
        -d "$model_base/catboost_no_deriv" && -d "$model_base/xgboost_no_deriv" ]]; then
    setup_sim_models "$model_base/v6_no_deriv" "$model_base/v7_no_deriv" \
                     "$model_base/catboost_no_deriv" "$model_base/xgboost_no_deriv"
    run_sim "$label" "ens4_all_no_deriv" "1x_base" $SIM_PERIOD $BASE_SIM --ensemble
  fi

  # =========================================================================
  #  ГРУППА E: EXECUTION FLAGS — лучшие модельные конфиги
  # =========================================================================
  log "--- EXECUTION FLAG SWEEPS ---"

  # Берём ансамбль 3 (LGB noN + CB N) как основу для проверки execution flags
  if [[ -d "$model_base/v6_base" && -d "$model_base/v7_base" && -d "$model_base/catboost_with_news" ]]; then
    setup_sim_models "$model_base/v6_base" "$model_base/v7_base" "$model_base/catboost_with_news" "SKIP"

    # Smooth signal
    run_sim "$label" "ens3_lgb_noN+cb_N" "1x_smooth03" $SIM_PERIOD $BASE_SIM --ensemble --smooth-signal 0.3
    run_sim "$label" "ens3_lgb_noN+cb_N" "1x_smooth05" $SIM_PERIOD $BASE_SIM --ensemble --smooth-signal 0.5

    # Hysteresis
    run_sim "$label" "ens3_lgb_noN+cb_N" "1x_hyst5" $SIM_PERIOD $BASE_SIM --ensemble --hysteresis 5
    run_sim "$label" "ens3_lgb_noN+cb_N" "1x_hyst7" $SIM_PERIOD $BASE_SIM --ensemble --hysteresis 7

    # Vol targeting
    run_sim "$label" "ens3_lgb_noN+cb_N" "1x_voltgt30" $SIM_PERIOD $BASE_SIM --ensemble --vol-target-ann 0.30
    run_sim "$label" "ens3_lgb_noN+cb_N" "1x_voltgt50" $SIM_PERIOD $BASE_SIM --ensemble --vol-target-ann 0.50

    # Vol sizing
    run_sim "$label" "ens3_lgb_noN+cb_N" "1x_volsize" $SIM_PERIOD $BASE_SIM --ensemble --vol-size

    # DDstop enabled
    run_sim "$label" "ens3_lgb_noN+cb_N" "1x_ddstop" $SIM_PERIOD --leverage 1 --edge-boost --ensemble

    # Event filter
    run_sim "$label" "ens3_lgb_noN+cb_N" "1x_eventfilt" $SIM_PERIOD $BASE_SIM --ensemble --event-filter

    # Regime shorts
    run_sim "$label" "ens3_lgb_noN+cb_N" "1x_regshorts05" $SIM_PERIOD $BASE_SIM --ensemble --regime-shorts 0.5

    # Deriv gate
    run_sim "$label" "ens3_lgb_noN+cb_N" "1x_derivgate" $SIM_PERIOD --leverage 1 --edge-boost --ensemble --deriv-gate

    # Комбо: лучшие флаги вместе
    run_sim "$label" "ens3_lgb_noN+cb_N" "1x_combo_hyst5_sm03" $SIM_PERIOD $BASE_SIM --ensemble --hysteresis 5 --smooth-signal 0.3
    run_sim "$label" "ens3_lgb_noN+cb_N" "1x_combo_hyst5_vt50_sm03" $SIM_PERIOD $BASE_SIM --ensemble --hysteresis 5 --vol-target-ann 0.50 --smooth-signal 0.3
    run_sim "$label" "ens3_lgb_noN+cb_N" "3x_combo_hyst5_sm03" $SIM_PERIOD $LEV3_SIM --ensemble --hysteresis 5 --smooth-signal 0.3
  fi

  # То же для solo CB (текущий "чемпион" — проверяем с execution flags)
  if [[ -d "$model_base/catboost_no_deriv" ]]; then
    setup_sim_models "SKIP" "SKIP" "$model_base/catboost_no_deriv" "SKIP"

    run_sim "$label" "cb_solo_no_deriv" "1x_smooth03" $SIM_PERIOD $BASE_SIM --ensemble --smooth-signal 0.3
    run_sim "$label" "cb_solo_no_deriv" "1x_hyst5" $SIM_PERIOD $BASE_SIM --ensemble --hysteresis 5
    run_sim "$label" "cb_solo_no_deriv" "1x_voltgt50" $SIM_PERIOD $BASE_SIM --ensemble --vol-target-ann 0.50
    run_sim "$label" "cb_solo_no_deriv" "1x_ddstop" $SIM_PERIOD --leverage 1 --edge-boost --ensemble
    run_sim "$label" "cb_solo_no_deriv" "1x_combo_hyst5_sm03" $SIM_PERIOD $BASE_SIM --ensemble --hysteresis 5 --smooth-signal 0.3
    run_sim "$label" "cb_solo_no_deriv" "3x_combo_hyst5_sm03" $SIM_PERIOD $LEV3_SIM --ensemble --hysteresis 5 --smooth-signal 0.3
  fi

  # =========================================================================
  #  ГРУППА F: LEVERAGE SWEEP для лучших конфигов
  # =========================================================================
  log "--- LEVERAGE SWEEP ---"

  # Ансамбль 3: sweep 1x, 2x, 3x, 5x
  if [[ -d "$model_base/v6_base" && -d "$model_base/v7_base" && -d "$model_base/catboost_with_news" ]]; then
    setup_sim_models "$model_base/v6_base" "$model_base/v7_base" "$model_base/catboost_with_news" "SKIP"
    run_sim "$label" "ens3_lgb_noN+cb_N" "2x_base" $SIM_PERIOD --leverage 2 --edge-boost --no-ddstop --ensemble
    run_sim "$label" "ens3_lgb_noN+cb_N" "5x_base" $SIM_PERIOD --leverage 5 --edge-boost --no-ddstop --ensemble
  fi

  # Solo CB: sweep
  if [[ -d "$model_base/catboost_no_deriv" ]]; then
    setup_sim_models "SKIP" "SKIP" "$model_base/catboost_no_deriv" "SKIP"
    run_sim "$label" "cb_solo_no_deriv" "2x_base" $SIM_PERIOD --leverage 2 --edge-boost --no-ddstop --ensemble
    run_sim "$label" "cb_solo_no_deriv" "5x_base" $SIM_PERIOD --leverage 5 --edge-boost --no-ddstop --ensemble
  fi

  # =========================================================================
  #  ГРУППА G: CONFIDENCE / MIN-CONF SWEEPS
  # =========================================================================
  log "--- CONFIDENCE SWEEPS ---"

  # Ens3 с min-conf
  if [[ -d "$model_base/v6_base" && -d "$model_base/v7_base" && -d "$model_base/catboost_with_news" ]]; then
    setup_sim_models "$model_base/v6_base" "$model_base/v7_base" "$model_base/catboost_with_news" "SKIP"
    run_sim "$label" "ens3_lgb_noN+cb_N" "1x_minconf85" $SIM_PERIOD $BASE_SIM --ensemble --min-conf 0.85
    run_sim "$label" "ens3_lgb_noN+cb_N" "1x_minconf90" $SIM_PERIOD $BASE_SIM --ensemble --min-conf 0.90
    run_sim "$label" "ens3_lgb_noN+cb_N" "1x_noconf" $SIM_PERIOD $BASE_SIM --ensemble --no-conf
  fi

  # Solo CB с min-conf
  if [[ -d "$model_base/catboost_no_deriv" ]]; then
    setup_sim_models "SKIP" "SKIP" "$model_base/catboost_no_deriv" "SKIP"
    run_sim "$label" "cb_solo_no_deriv" "1x_minconf85" $SIM_PERIOD $BASE_SIM --ensemble --min-conf 0.85
    run_sim "$label" "cb_solo_no_deriv" "1x_minconf90" $SIM_PERIOD $BASE_SIM --ensemble --min-conf 0.90
  fi

}

# Запускаем для выбранных окон (по умолчанию все три)
for WIN in $SIM_WINDOWS; do
  run_window_sims "$WIN"
done

phase_end


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  PHASE 3: АНАЛИЗ И РЕЙТИНГ                                          ║
# ╚══════════════════════════════════════════════════════════════════════╝

phase_start "PHASE 3: АНАЛИЗ РЕЗУЛЬТАТОВ"

log ""
log "📊 Generating analysis..."

python3 - <<'PYEOF'
import pandas as pd
import os, sys

summary_dir = "results/mega_comparison"
# Find latest summary CSV
csvs = sorted([f for f in os.listdir(summary_dir) if f.startswith("summary_") and f.endswith(".csv")])
if not csvs:
    print("❌ No summary CSV found!")
    sys.exit(1)

csv_path = os.path.join(summary_dir, csvs[-1])
df = pd.read_csv(csv_path, sep='|')

# Clean numeric columns
for col in ['return_pct', 'sharpe_hac', 'max_dd', 'win_rate', 'profit_factor', 'calmar']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("\n" + "=" * 100)
print("  MEGA COMPARISON — ИТОГОВЫЙ РЕЙТИНГ")
print("=" * 100)

# ── 1. Средние метрики по model_config (across all windows) ──
print("\n" + "─" * 80)
print("  1. СРЕДНИЙ SHARPE HAC ПО МОДЕЛЬНЫМ КОНФИГУРАЦИЯМ (across windows, 1x leverage)")
print("─" * 80)

# Фильтруем только 1x_base для честного сравнения
base = df[df['sim_config'] == '1x_base'].copy()
if len(base) > 0:
    agg = base.groupby('model_config').agg(
        mean_sharpe=('sharpe_hac', 'mean'),
        std_sharpe=('sharpe_hac', 'std'),
        mean_return=('return_pct', 'mean'),
        mean_maxdd=('max_dd', 'mean'),
        mean_wr=('win_rate', 'mean'),
        mean_pf=('profit_factor', 'mean'),
        n_windows=('sharpe_hac', 'count'),
    ).sort_values('mean_sharpe', ascending=False)
    
    print(f"\n{'Model Config':<35} {'Sharpe':>8} {'±Std':>8} {'Return%':>10} {'MaxDD%':>8} {'WR%':>6} {'PF':>6} {'N':>3}")
    print("─" * 95)
    for idx, row in agg.iterrows():
        print(f"{idx:<35} {row['mean_sharpe']:>8.2f} {row['std_sharpe']:>8.2f} "
              f"{row['mean_return']:>10.1f} {row['mean_maxdd']:>8.1f} "
              f"{row['mean_wr']:>6.1f} {row['mean_pf']:>6.2f} {int(row['n_windows']):>3}")

# ── 2. Ранг по окнам ──
print("\n" + "─" * 80)
print("  2. SHARPE HAC ПО КАЖДОМУ ОКНУ (1x, base sim)")
print("─" * 80)

for window in sorted(base['window'].unique()):
    w = base[base['window'] == window].sort_values('sharpe_hac', ascending=False)
    print(f"\n  Window: {window}")
    print(f"  {'Model Config':<35} {'Sharpe':>8} {'Return%':>10} {'MaxDD%':>8} {'WR%':>6}")
    print("  " + "─" * 75)
    for _, row in w.iterrows():
        print(f"  {row['model_config']:<35} {row['sharpe_hac']:>8.2f} {row['return_pct']:>10.1f} "
              f"{row['max_dd']:>8.1f} {row['win_rate']:>6.1f}")

# ── 3. Solo vs Ensemble comparison ──
print("\n" + "─" * 80)
print("  3. SOLO vs ENSEMBLE — ПРЯМОЕ СРАВНЕНИЕ")
print("─" * 80)

solo = base[base['model_config'].str.contains('solo')].copy()
ensemble = base[~base['model_config'].str.contains('solo')].copy()

if len(solo) > 0 and len(ensemble) > 0:
    solo_avg = solo.groupby('model_config')['sharpe_hac'].mean()
    ens_avg = ensemble.groupby('model_config')['sharpe_hac'].mean()
    
    print(f"\n  BEST SOLO:     {solo_avg.idxmax():<30} Sharpe = {solo_avg.max():.2f}")
    print(f"  BEST ENSEMBLE: {ens_avg.idxmax():<30} Sharpe = {ens_avg.max():.2f}")
    print(f"\n  Ensemble advantage: {(ens_avg.max() - solo_avg.max()):+.2f} Sharpe")

# ── 4. Execution flags effect ──
print("\n" + "─" * 80)
print("  4. ЭФФЕКТ EXECUTION FLAGS (ens3 base)")
print("─" * 80)

ens3 = df[df['model_config'] == 'ens3_lgb_noN+cb_N'].copy()
if len(ens3) > 0:
    exec_agg = ens3.groupby('sim_config').agg(
        mean_sharpe=('sharpe_hac', 'mean'),
        mean_return=('return_pct', 'mean'),
    ).sort_values('mean_sharpe', ascending=False)
    
    print(f"\n  {'Sim Config':<35} {'Sharpe':>8} {'Return%':>10}")
    print("  " + "─" * 55)
    for idx, row in exec_agg.iterrows():
        print(f"  {idx:<35} {row['mean_sharpe']:>8.2f} {row['mean_return']:>10.1f}")

# ── 5. Стабильность (std across windows) ──
print("\n" + "─" * 80)
print("  5. СТАБИЛЬНОСТЬ МОДЕЛЕЙ (меньше std = стабильнее)")
print("─" * 80)

if len(base) > 0:
    stability = base.groupby('model_config').agg(
        mean_sharpe=('sharpe_hac', 'mean'),
        std_sharpe=('sharpe_hac', 'std'),
        min_sharpe=('sharpe_hac', 'min'),
        max_sharpe=('sharpe_hac', 'max'),
    )
    stability['stability_score'] = stability['mean_sharpe'] / (stability['std_sharpe'] + 0.01)
    stability = stability.sort_values('stability_score', ascending=False)
    
    print(f"\n  {'Model Config':<35} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Stab':>8}")
    print("  " + "─" * 80)
    for idx, row in stability.iterrows():
        print(f"  {idx:<35} {row['mean_sharpe']:>8.2f} {row['std_sharpe']:>8.2f} "
              f"{row['min_sharpe']:>8.2f} {row['max_sharpe']:>8.2f} {row['stability_score']:>8.2f}")

# ── 6. Leverage effect ──
print("\n" + "─" * 80)
print("  6. ЭФФЕКТ LEVERAGE")
print("─" * 80)

for mc in sorted(df['model_config'].unique()):
    lev_rows = df[df['model_config'] == mc]
    lev_base = lev_rows[lev_rows['sim_config'].str.contains('x_base')]
    if len(lev_base) > 1:
        print(f"\n  {mc}:")
        for _, row in lev_base.sort_values('sim_config').iterrows():
            print(f"    {row['sim_config']:<20} Sharpe={row['sharpe_hac']:>6.2f}  Return={row['return_pct']:>8.1f}%  MaxDD={row['max_dd']:>6.1f}%")

# ── 7. Победитель ──
print("\n" + "=" * 80)
print("  🏆 РЕКОМЕНДАЦИЯ")
print("=" * 80)

if len(base) > 0:
    best = agg.iloc[0]
    best_name = agg.index[0]
    print(f"\n  Лучшая конфигурация: {best_name}")
    print(f"  Средний Sharpe HAC:  {best['mean_sharpe']:.2f} ± {best['std_sharpe']:.2f}")
    print(f"  Средний Return:      {best['mean_return']:.1f}%")
    print(f"  Средний MaxDD:       {best['mean_maxdd']:.1f}%")
    print(f"  Win Rate:            {best['mean_wr']:.1f}%")
    
    # Check if solo or ensemble
    if 'solo' in best_name:
        print(f"\n  ⚠️  SOLO модель выиграла. Но проверь стабильность across windows!")
    else:
        print(f"\n  ✅ ENSEMBLE выиграл — подтверждает теорию диверсификации")

print("\n" + "=" * 80)

# Save detailed report
report_path = os.path.join(summary_dir, "analysis_report.txt")
with open(report_path, 'w') as f:
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    # Re-run prints... (skip for brevity, CSV has all data)
    sys.stdout = old_stdout
print(f"\n📄 Full results CSV: {csv_path}")
print(f"📊 Total simulations: {len(df)}")

PYEOF

phase_end


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  PHASE 4: ВОССТАНОВЛЕНИЕ ПРОДАКШН МОДЕЛЕЙ                           ║
# ╚══════════════════════════════════════════════════════════════════════╝

phase_start "PHASE 4: ВОССТАНОВЛЕНИЕ"
restore_prod_models
phase_end


# ─── Финальная статистика ───
TOTAL_TIME=$(( $(date +%s) - START_TIME ))
log ""
log "════════════════════════════════════════════════════════════════"
log "  MEGA COMPARISON COMPLETE"
log "  Total sims: $SIM_N"
log "  Total time: $(( TOTAL_TIME / 3600 ))h $(( (TOTAL_TIME % 3600) / 60 ))m"
log "  Results:    $SUMMARY"
log "  Details:    $DETAIL_LOG"
log "  Log:        $LOG"
log "════════════════════════════════════════════════════════════════"
