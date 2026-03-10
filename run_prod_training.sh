#!/bin/bash
# =============================================================================
#  Production Retrain — ALL models with maximum data
#  
#  Данные до 2026-03-07. Используем расширенное окно:
#    train_end  = 2025-12-01  (максимум данных для train)
#    val_end    = 2026-03-07  (последние ~3 мес для val)
#
#  Модели:
#    1. LGB v7 (лидер exp12, DDStop 2.12)
#    2. LGB v6 (стабильный, DDStop 1.61) 
#    3. CatBoost with news (DDStop 1.76)
#    4. CatBoost no news (DDStop 1.64 — может быть лучше)
#    5. Derivatives-only mini-model (risk gate, Rank_IC 0.025)
#
#  XGBoost убран из ансамбля — дублирует LGB/CB, не даёт доп. alpha.
#
#  Запуск на кластере:
#    git pull && bash run_prod_training.sh 2>&1 | tee prod_training.log
# =============================================================================

set -euo pipefail

TRAIN_END="2025-12-01"
VAL_END="2026-03-07"

LOGS="results/prod_training_logs"
mkdir -p "$LOGS"

TS=$(date +%Y%m%d_%H%M%S)

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

run_step() {
    local name="$1"; shift
    local logfile="${LOGS}/${name}_${TS}.log"
    echo -e "${YELLOW}▶ [${name}]${NC} $@"
    local start_time=$(date +%s)
    if "$@" > "$logfile" 2>&1; then
        local end_time=$(date +%s)
        local elapsed=$(( end_time - start_time ))
        echo -e "${GREEN}✅ [${name}] done in ${elapsed}s${NC}"
    else
        echo -e "${RED}❌ [${name}] FAILED — see ${logfile}${NC}"
    fi
}

echo "============================================================"
echo "  PRODUCTION RETRAIN — $(date)"
echo "  train_end=${TRAIN_END}, val_end=${VAL_END}"
echo "============================================================"
echo ""

# ─── Phase 1: LightGBM ────────────────────────────────────────
echo -e "${YELLOW}═══ Phase 1: LightGBM ═══${NC}"

run_step "lgb_v7" \
    python run_pipeline_v7.py --production --skip-hpo \
    --train-end "$TRAIN_END" --val-end "$VAL_END" \
    --results results/production/lgb_v7_no_news

run_step "lgb_v6" \
    python run_pipeline_v6.py --production --skip-hpo --news-mode none \
    --train-end "$TRAIN_END" --val-end "$VAL_END" \
    --results results/production/lgb_v6_no_news

# ─── Phase 2: CatBoost ────────────────────────────────────────
echo ""
echo -e "${YELLOW}═══ Phase 2: CatBoost ═══${NC}"

run_step "cb_with_news" \
    python run_pipeline_catboost.py --production --skip-hpo --gpu \
    --train-end "$TRAIN_END" --val-end "$VAL_END" \
    --results results/production/catboost_with_news

run_step "cb_no_news" \
    python run_pipeline_catboost.py --production --skip-hpo --gpu --no-news \
    --train-end "$TRAIN_END" --val-end "$VAL_END" \
    --results results/production/catboost_no_news

# ─── Phase 3: Derivatives-only (risk gate model) ──────────────
echo ""
echo -e "${YELLOW}═══ Phase 3: Derivatives-Only ═══${NC}"

run_step "deriv_only" \
    python run_pipeline_derivatives.py --production --skip-hpo \
    --results results/production/deriv_only

# ─── Done ──────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  DONE — $(date)"
echo "  Production models saved to: results/production/"
echo "  Logs: ${LOGS}/"
echo "============================================================"
echo ""
echo "  Models trained:"
echo "    results/production/lgb_v7_no_news/    (ensemble member)"
echo "    results/production/lgb_v6_no_news/    (ensemble member)"
echo "    results/production/catboost_with_news/ (ensemble member)"
echo "    results/production/catboost_no_news/   (ensemble member)"
echo "    results/production/deriv_only/         (risk gate)"
echo ""
echo "  Architecture: mean(v6, v7, CB) + deriv_only risk gate"
echo "  Next: скачать models → прогнать инференс → залить в прод"
