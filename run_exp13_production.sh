#!/bin/bash
# =============================================================================
#  exp13: Production Retrain
#  
#  Обучает лучшие модели из exp12 на PRODUCTION окне (данные до 2025-09).
#  Результаты идут в results/production/ — для деплоя на боевой сервер.
#
#  Что обучаем:
#    1. LGB v7 baseline (DDStop 2.12, лучший в exp12)
#    2. LGB v6 baseline (DDStop 1.61, стабильный)
#    3. CatBoost baseline (DDStop 1.76, с news)
#    4. CatBoost baseline без news (DDStop 1.64, для сравнения)
#    5. Derivatives-only mini-model (починён баг evaluate_model)
#
#  Запуск: bash run_exp13_production.sh
# =============================================================================

set -euo pipefail

EXP_NAME="exp13_production"
EXP_DIR="results/${EXP_NAME}"
LOGS="${EXP_DIR}/logs"
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
    if "$@" > "$logfile" 2>&1; then
        local last_line=$(tail -1 "$logfile")
        echo -e "${GREEN}✅ [${name}] done${NC} (${last_line})"
    else
        echo -e "${RED}❌ [${name}] FAILED — see ${logfile}${NC}"
    fi
}

echo "============================================================"
echo "  exp13: Production Retrain — $(date)"
echo "  Best models from exp12 → production window"
echo "============================================================"
echo ""

# ============================================================
# ФАЗА 1: Production модели (данные до 2025-09)
# ============================================================
echo -e "${YELLOW}═══ ФАЗА 1: LightGBM Production ═══${NC}"

# v7 baseline — лидер exp12 (DDStop 2.12)
run_step "v7_prod" \
    python run_pipeline_v7.py --production --skip-hpo \
    --results results/production/lgb_v7_no_news

# v6 baseline — стабильный (DDStop 1.61)
run_step "v6_prod" \
    python run_pipeline_v6.py --production --skip-hpo --news-mode none \
    --results results/production/lgb_v6_no_news

echo ""
echo -e "${YELLOW}═══ ФАЗА 2: CatBoost Production ═══${NC}"

# CatBoost baseline с news — 4й в exp12 (DDStop 1.76)
run_step "cb_prod" \
    python run_pipeline_catboost.py --production --skip-hpo --gpu \
    --results results/production/catboost_with_news

# CatBoost без news — для сравнения (DDStop 1.64)
run_step "cb_no_news_prod" \
    python run_pipeline_catboost.py --production --skip-hpo --gpu --no-news \
    --results results/production/catboost_no_news

echo ""
echo -e "${YELLOW}═══ ФАЗА 3: Derivatives-Only Mini-Model ═══${NC}"

# Derivatives expert — починено (was: evaluate_model bug)
run_step "deriv_prod" \
    python run_pipeline_derivatives.py --production --skip-hpo \
    --results results/production/deriv_only

echo ""

# ============================================================
# ФАЗА 4: Walk-Forward validation (для сравнения с exp12)
# ============================================================
echo -e "${YELLOW}═══ ФАЗА 4: Walk-Forward (для валидации) ═══${NC}"

# Перегнать лучшие baselines через walk-forward, чтобы убедиться что починка deriv_only работает
run_step "deriv_wf" \
    python run_pipeline_derivatives.py --skip-hpo \
    --results "${EXP_DIR}/deriv_only_wf"

echo ""
echo "============================================================"
echo "  DONE — $(date)"
echo "  Production models: results/production/"
echo "  WF validation:     ${EXP_DIR}/"
echo "  Logs:              ${LOGS}/"
echo "============================================================"
echo ""
echo "  Следующий шаг: прогнать инференс"
echo "    python run_fast_sim.py --data data/features/crypto_features_1h.parquet \\"
echo "      --ensemble --edge-boost --meta-risk --days 60 --capital 5000"
