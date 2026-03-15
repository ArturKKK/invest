#!/bin/bash
set -uo pipefail

# ============================================================
# OVERNIGHT RESEARCH v10 — Full Retrain + Comprehensive Sim Grid
# ============================================================
#
# Goal: retrain ALL models from scratch on equal conditions,
#       then test every meaningful combination of:
#       - Ensemble vs solo
#       - Horizons (4h / 12h / 24h)
#       - News (all / none for 24h)
#       - Z-score thresholds
#       - Rebalance frequency (12h / 24h)
#       - Mini-ensembles (v6-only multi-horizon)
#
# Phase 0: Reference sims with CURRENT production models  (~5 min)
# Phase 1: Retrain all models (7 pipelines)               (~2.5 h)
# Phase 2: Ensemble sims with fresh v10 models             (~20 min)
# Phase 3: Solo sims                                       (~15 min)
# Phase 4: Mini-ensembles                                  (~5 min)
# Phase 5: Z-score sweep                                   (~15 min)
# Phase 6: Summary                                         (~0 min)
#
# Expected total runtime: ~3.5-4 hours
#
# v9 findings to validate:
#   - 24h solo crushed ensembles (HAC 8.79 vs 7.55)
#   - MLP was contaminating v9 sims (auto-loaded, 5 groups)
#   - This run: clean retrain, controlled MLP inclusion
# ============================================================

TRAIN_END="2026-02-01"
VAL_END="2026-03-07"

LOGDIR="results/overnight_v10"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG="$LOGDIR/run_${TIMESTAMP}.log"
RESULTS="$LOGDIR/summary_${TIMESTAMP}.txt"

# Sim config (same period as v8/v9 for comparability)
SIM_ARGS="--data data/features/crypto_features_1h.parquet \
  --days 120 --start-date 2026-02-09 --end-date 2026-03-07 \
  --leverage 3 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop"

export SKIP_CALENDAR=1

BAK="_bak_v10"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ─── Safety: restore production models on exit/crash ─────────
cleanup() {
  log "CLEANUP: restoring production models..."
  # Restore Phase 0 hidden dirs (if crash during Phase 0)
  for d in results_mlp_prod results_v6_4h_prod results_v6_24h_prod; do
    if [[ -d "${d}_ref_hidden" ]]; then
      rm -rf "$d" 2>/dev/null || true
      mv "${d}_ref_hidden" "$d"
      log "  restored $d (from ref_hidden)"
    fi
  done
  # Restore Phase 2+ backups
  for d in results_v6_prod results_v7_prod results_catboost_prod results_xgboost_prod; do
    if [[ -d "${d}${BAK}" ]]; then
      rm -rf "$d" 2>/dev/null || true
      mv "${d}${BAK}" "$d"
      log "  restored $d"
    fi
  done
  for d in results_mlp_prod results_v6_4h_prod results_v6_24h_prod; do
    if [[ -d "${d}${BAK}" ]]; then
      rm -rf "$d" 2>/dev/null || true
      mv "${d}${BAK}" "$d"
      log "  restored $d"
    fi
  done
  log "Production models restored."
}
trap cleanup EXIT

# ─── Sim helper ──────────────────────────────────────────────
SIM_N=0
run_sim() {
  local label="$1"
  shift
  SIM_N=$((SIM_N + 1))
  log "────────────────────────────────────"
  log "SIM #${SIM_N}: $label"
  log "CMD: python run_fast_sim.py $SIM_ARGS $*"

  local output
  output=$(python run_fast_sim.py $SIM_ARGS "$@" 2>&1) || true
  echo "$output" >> "$LOG"

  # Save structured summary
  echo "=== #${SIM_N}: $label ===" >> "$RESULTS"
  echo "$output" | grep -E "Return:|Max DD:|Sharpe|Calmar:|Win Rate:" | head -6 >> "$RESULTS"
  echo "" >> "$RESULTS"

  # Print compact result to console
  local ret hac calmar
  ret=$(echo "$output" | grep "Return:" | head -1 | awk '{print $2}')
  hac=$(echo "$output" | grep "Sharpe HAC:" | awk '{print $3}')
  calmar=$(echo "$output" | grep "Calmar:" | awk '{print $2}')
  log "  => Return=$ret  HAC=$hac  Calmar=$calmar"
  log ""
}

# ─── Swap helpers ────────────────────────────────────────────
swap_12h_v10() {
  for pair in v6:results_v6_v10 v7:results_v7_v10 catboost:results_catboost_v10 xgboost:results_xgboost_v10; do
    IFS=: read -r suffix src <<< "$pair"
    local dst="results_${suffix}_prod"
    rm -rf "$dst"
    if [[ -d "$src" ]]; then
      cp -r "$src" "$dst"
    else
      log "  WARNING: $src not found, skipping swap for $dst"
    fi
  done
}

show_24h() {
  local variant="$1"  # "news" or "nonews"
  if [[ "$variant" == "news" ]]; then
    cp -r results_v6_24h_news_v10 results_v6_24h_prod 2>/dev/null || true
  else
    cp -r results_v6_24h_nonews_v10 results_v6_24h_prod 2>/dev/null || true
  fi
}

show_4h() {
  cp -r results_v6_4h_v10 results_v6_4h_prod 2>/dev/null || true
}

hide_multihorizon() {
  rm -rf results_v6_24h_prod results_v6_4h_prod 2>/dev/null || true
}

show_mlp() {
  if [[ -d "results_mlp_prod${BAK}" ]]; then
    cp -r "results_mlp_prod${BAK}" results_mlp_prod
  fi
}

hide_mlp() {
  rm -rf results_mlp_prod 2>/dev/null || true
}

# ============================================================
#  PHASE 0: REFERENCE SIMS (current production, pre-retrain)
# ============================================================
log "════════════════════════════════════════════════════════"
log "  PHASE 0: REFERENCE SIMS (current production models)"
log "════════════════════════════════════════════════════════"
echo "=== PHASE 0: REFERENCE (current production) ===" > "$RESULTS"
echo "" >> "$RESULTS"

# 0a: Clean 4-model ensemble (hide MLP + multi-horizon)
for d in results_mlp_prod results_v6_4h_prod results_v6_24h_prod; do
  [[ -d "$d" ]] && mv "$d" "${d}_ref_hidden"
done

run_sim "REF-A: current prod 4-model clean (no MLP/horizon)" \
  --ensemble --min-zscore 0.5

# Restore hidden dirs
for d in results_mlp_prod results_v6_4h_prod results_v6_24h_prod; do
  [[ -d "${d}_ref_hidden" ]] && mv "${d}_ref_hidden" "$d"
done

# 0b: Current prod as deployed (everything included)
run_sim "REF-B: current prod as deployed (all groups)" \
  --ensemble --min-zscore 0.5

# ============================================================
#  PHASE 1: RETRAIN ALL MODELS FROM SCRATCH
# ============================================================
log ""
log "════════════════════════════════════════════════════════"
log "  PHASE 1: RETRAIN ALL MODELS"
log "════════════════════════════════════════════════════════"
echo "" >> "$RESULTS"
echo "=== PHASE 1: TRAINING ===" >> "$RESULTS"

TRAIN_COMMON="--production --skip-hpo --train-end $TRAIN_END --val-end $VAL_END"
PHASE1_START=$SECONDS

train_step() {
  local n="$1" label="$2"
  shift 2
  log ">>> [$n/7] Training: $label"
  local t0=$SECONDS
  if "$@" 2>&1 | tee -a "$LOG"; then
    local elapsed=$(( (SECONDS - t0) / 60 ))
    log ">>> [$n/7] $label — done (${elapsed}m)"
    echo "  [$n/7] $label — OK (${elapsed}m)" >> "$RESULTS"
  else
    log ">>> [$n/7] $label — FAILED!"
    echo "  [$n/7] $label — FAILED" >> "$RESULTS"
  fi
}

# 1) LightGBM v6 – 12h, news=all, Huber
train_step 1 "v6 12h (news=all, huber)" \
  python run_pipeline_v6.py $TRAIN_COMMON \
  --news-mode all --huber --results results_v6_v10

# 2) LightGBM v7 – 12h, news=none, Huber
train_step 2 "v7 12h (news=none, huber)" \
  python run_pipeline_v7.py $TRAIN_COMMON \
  --news-mode none --huber --results results_v7_v10

# 3) CatBoost – 12h, news=all, Huber, GPU
train_step 3 "CatBoost 12h (news=all, huber, gpu)" \
  python run_pipeline_catboost.py $TRAIN_COMMON \
  --news-mode all --huber --gpu --results results_catboost_v10

# 4) XGBoost – 12h, news=all, Huber slope=1.0, GPU
train_step 4 "XGBoost 12h (news=all, huber, gpu)" \
  python run_pipeline_xgboost.py $TRAIN_COMMON \
  --news-mode all --huber --huber-slope 1.0 --gpu --results results_xgboost_v10

# 5) LightGBM v6 – 24h, news=all, Huber
train_step 5 "v6 24h (news=all, huber)" \
  python run_pipeline_v6.py $TRAIN_COMMON \
  --horizon 24 --news-mode all --huber --results results_v6_24h_news_v10

# 6) LightGBM v6 – 24h, news=none, Huber
train_step 6 "v6 24h (news=none, huber)" \
  python run_pipeline_v6.py $TRAIN_COMMON \
  --horizon 24 --news-mode none --huber --results results_v6_24h_nonews_v10

# 7) LightGBM v6 – 4h, news=all, Huber
train_step 7 "v6 4h (news=all, huber)" \
  python run_pipeline_v6.py $TRAIN_COMMON \
  --horizon 4 --news-mode all --huber --results results_v6_4h_v10

PHASE1_ELAPSED=$(( (SECONDS - PHASE1_START) / 60 ))
log ">>> Phase 1 complete — all 7 models trained in ${PHASE1_ELAPSED}m"
echo "  Total training time: ${PHASE1_ELAPSED}m" >> "$RESULTS"
echo "" >> "$RESULTS"

# ============================================================
#  PHASE 2: ENSEMBLE SIMS (4-model base + multi-horizon combos)
# ============================================================
log ""
log "════════════════════════════════════════════════════════"
log "  PHASE 2: ENSEMBLE SIMS"
log "════════════════════════════════════════════════════════"
echo "=== PHASE 2: ENSEMBLE SIMS ===" >> "$RESULTS"
echo "" >> "$RESULTS"

# Backup production models
log "Backing up production models..."
for d in results_v6_prod results_v7_prod results_catboost_prod results_xgboost_prod; do
  if [[ -d "$d" ]]; then
    rm -rf "${d}${BAK}" 2>/dev/null || true
    cp -r "$d" "${d}${BAK}"
    log "  backed up $d"
  fi
done
for d in results_mlp_prod results_v6_4h_prod results_v6_24h_prod; do
  if [[ -d "$d" ]]; then
    rm -rf "${d}${BAK}" 2>/dev/null || true
    mv "$d" "${d}${BAK}"
    log "  hidden $d"
  fi
done

# Swap fresh v10 12h models into _prod dirs
swap_12h_v10
log "Swapped v10 12h models into prod dirs"

# A: 4-model ensemble baseline (v6+v7+CB+XGB only)
run_sim "A: 4-model ensemble (v10, no MLP/horizon) mz=0.5" \
  --ensemble --min-zscore 0.5

# B: + 24h with news
show_24h news
run_sim "B: ensemble + 24h(news) mz=0.5" \
  --ensemble --min-zscore 0.5
hide_multihorizon

# C: + 24h without news
show_24h nonews
run_sim "C: ensemble + 24h(nonews) mz=0.5" \
  --ensemble --min-zscore 0.5
hide_multihorizon

# D: + 24h(news) + 4h
show_24h news
show_4h
run_sim "D: ensemble + 24h(news) + 4h mz=0.5" \
  --ensemble --min-zscore 0.5
hide_multihorizon

# E: + 24h(nonews) + 4h
show_24h nonews
show_4h
run_sim "E: ensemble + 24h(nonews) + 4h mz=0.5" \
  --ensemble --min-zscore 0.5
hide_multihorizon

# F: Full kitchen sink (+ MLP + 24h + 4h)
show_mlp
show_24h news
show_4h
run_sim "F: full (4-model + MLP + 24h + 4h) mz=0.5" \
  --ensemble --min-zscore 0.5
hide_mlp
hide_multihorizon

# ============================================================
#  PHASE 3: SOLO MODEL SIMS
# ============================================================
log ""
log "════════════════════════════════════════════════════════"
log "  PHASE 3: SOLO MODEL SIMS"
log "════════════════════════════════════════════════════════"
echo "" >> "$RESULTS"
echo "=== PHASE 3: SOLO SIMS ===" >> "$RESULTS"
echo "" >> "$RESULTS"

# G: Solo 24h with news (rebal=12)
run_sim "G: solo 24h news, rebal=12" \
  --model-dir results_v6_24h_news_v10 --min-zscore 0.5

# H: Solo 24h no news (rebal=12)
run_sim "H: solo 24h nonews, rebal=12" \
  --model-dir results_v6_24h_nonews_v10 --min-zscore 0.5

# I: Solo 24h with news (rebal=24)
run_sim "I: solo 24h news, rebal=24" \
  --model-dir results_v6_24h_news_v10 --min-zscore 0.5 --rebal 24

# J: Solo 24h no news (rebal=24)
run_sim "J: solo 24h nonews, rebal=24" \
  --model-dir results_v6_24h_nonews_v10 --min-zscore 0.5 --rebal 24

# K: Solo v6 12h
run_sim "K: solo v6 12h" \
  --model-dir results_v6_v10 --min-zscore 0.5

# L: Solo 4h
run_sim "L: solo v6 4h" \
  --model-dir results_v6_4h_v10 --min-zscore 0.5

# ============================================================
#  PHASE 4: MINI-ENSEMBLES (v6-only, multi-horizon)
# ============================================================
log ""
log "════════════════════════════════════════════════════════"
log "  PHASE 4: MINI-ENSEMBLES"
log "════════════════════════════════════════════════════════"
echo "" >> "$RESULTS"
echo "=== PHASE 4: MINI-ENSEMBLES ===" >> "$RESULTS"
echo "" >> "$RESULTS"

# Strip down to v6-only: remove v7/CB/XGB from prod
rm -rf results_v7_prod results_catboost_prod results_xgboost_prod

# M: v6_12h + v6_24h(news) — 2-group ensemble
show_24h news
run_sim "M: mini v6_12h + v6_24h(news) mz=0.5" \
  --ensemble --min-zscore 0.5
hide_multihorizon

# N: v6 all horizons (4h + 12h + 24h) — 3-group ensemble
show_24h news
show_4h
run_sim "N: mini v6 all horizons (4h+12h+24h) mz=0.5" \
  --ensemble --min-zscore 0.5
hide_multihorizon

# Restore full v10 12h models for remaining sims
swap_12h_v10

# ============================================================
#  PHASE 5: Z-SCORE SWEEP
# ============================================================
log ""
log "════════════════════════════════════════════════════════"
log "  PHASE 5: Z-SCORE SWEEP"
log "════════════════════════════════════════════════════════"
echo "" >> "$RESULTS"
echo "=== PHASE 5: Z-SCORE SWEEP ===" >> "$RESULTS"
echo "" >> "$RESULTS"

# Ensemble z-score sweep
run_sim "O: ensemble mz=0.0" --ensemble --min-zscore 0.0
run_sim "P: ensemble mz=0.3" --ensemble --min-zscore 0.3
run_sim "Q: ensemble mz=0.7" --ensemble --min-zscore 0.7

# Solo 24h(news) z-score sweep
run_sim "R: solo 24h news mz=0.0" --model-dir results_v6_24h_news_v10 --min-zscore 0.0
run_sim "S: solo 24h news mz=0.3" --model-dir results_v6_24h_news_v10 --min-zscore 0.3
run_sim "T: solo 24h news mz=0.7" --model-dir results_v6_24h_news_v10 --min-zscore 0.7

# ============================================================
#  PHASE 6: FINAL SUMMARY
# ============================================================
log ""
log "════════════════════════════════════════════════════════"
log "  DONE — $(date)"
log "════════════════════════════════════════════════════════"

TOTAL_MIN=$(( SECONDS / 60 ))
log "Total wall time: ${TOTAL_MIN}m"

echo "" >> "$RESULTS"
echo "════════════════════════════════════════" >> "$RESULTS"
echo "Total wall time: ${TOTAL_MIN}m" >> "$RESULTS"

log ""
log "RESULTS SUMMARY:"
log ""
cat "$RESULTS" | tee -a "$LOG"

log ""
log "Full log:    $LOG"
log "Summary:     $RESULTS"
log ""
log "Cleanup will restore production models via EXIT trap..."
