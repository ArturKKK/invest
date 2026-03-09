#!/bin/bash
# =============================================================================
#  Полный цикл обучения всех моделей
#  Запуск: bash run_train_all.sh [EXP_NAME]
#
#  Пример: bash run_train_all.sh exp11_with_derivatives
#
#  Структура результатов:
#    results/<EXP_NAME>/v6_baseline/
#    results/<EXP_NAME>/v6_hybrid/
#    results/<EXP_NAME>/catboost_baseline/
#    results/<EXP_NAME>/logs/
#
#  Логика:
#    Фаза 1 — A/B тесты v6/v7 (skip-hpo, быстро)
#    Фаза 2 — CatBoost + XGBoost
#    Фаза 3 — HPO на лучших комбинациях (раскомментировать)
#    Фаза 4 — Production модели (раскомментировать после проверки)
# =============================================================================

set -euo pipefail

# ── Experiment naming ──
EXP_NAME="${1:-exp_$(date +%Y%m%d_%H%M%S)}"
EXP_DIR="results/${EXP_NAME}"
LOGS="${EXP_DIR}/logs"
mkdir -p "$LOGS"

TS=$(date +%Y%m%d_%H%M%S)

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

run_step() {
    local name="$1"
    shift
    local logfile="${LOGS}/${name}_${TS}.log"
    echo -e "${YELLOW}▶ [$name]${NC} $*"
    echo "=== START: $(date) ===" >> "$logfile"
    echo "CMD: $*" >> "$logfile"
    if "$@" >> "$logfile" 2>&1; then
        echo -e "${GREEN}✅ [$name]${NC} done ($(tail -1 "$logfile"))"
    else
        echo -e "${RED}❌ [$name]${NC} FAILED — see $logfile"
    fi
    echo "=== END: $(date) ===" >> "$logfile"
    echo ""
}

echo "============================================================"
echo "  FULL TRAINING PIPELINE — $(date)"
echo "  Experiment: ${EXP_NAME}"
echo "  Results  → ${EXP_DIR}/"
echo "  Logs     → ${LOGS}/"
echo "============================================================"
echo ""

# ============================================================
# ФАЗА 1: A/B тесты v6 (skip-hpo)
# ============================================================
echo -e "${YELLOW}═══ ФАЗА 1: A/B тесты v6 (skip-hpo) ═══${NC}"

run_step "v6_baseline" \
    python run_pipeline_v6.py --skip-hpo \
    --results "${EXP_DIR}/v6_baseline"

run_step "v6_residual" \
    python run_pipeline_v6.py --skip-hpo --residual-target \
    --results "${EXP_DIR}/v6_residual"

run_step "v6_hybrid" \
    python run_pipeline_v6.py --skip-hpo --hybrid-norm \
    --results "${EXP_DIR}/v6_hybrid"

run_step "v6_res_hyb" \
    python run_pipeline_v6.py --skip-hpo --residual-target --hybrid-norm \
    --results "${EXP_DIR}/v6_res_hyb"

run_step "v6_res_hyb_null" \
    python run_pipeline_v6.py --skip-hpo --residual-target --hybrid-norm --null-importance \
    --results "${EXP_DIR}/v6_res_hyb_null"

# LambdaRank — disabled (needs integer labels, see exp10 logs)
# run_step "v6_lambdarank" \
#     python run_pipeline_v6.py --skip-hpo --residual-target --lambdarank \
#     --results "${EXP_DIR}/v6_lambdarank"

# ============================================================
# ФАЗА 1b: A/B тесты v7 (skip-hpo)
# ============================================================
echo -e "${YELLOW}═══ ФАЗА 1b: A/B тесты v7 (skip-hpo) ═══${NC}"

run_step "v7_baseline" \
    python run_pipeline_v7.py --skip-hpo \
    --results "${EXP_DIR}/v7_baseline"

run_step "v7_res_hyb" \
    python run_pipeline_v7.py --skip-hpo --residual-target --hybrid-norm \
    --results "${EXP_DIR}/v7_res_hyb"

run_step "v7_res_hyb_null" \
    python run_pipeline_v7.py --skip-hpo --residual-target --hybrid-norm --null-importance \
    --results "${EXP_DIR}/v7_res_hyb_null"

# ============================================================
# ФАЗА 2: CatBoost
# ============================================================
echo -e "${YELLOW}═══ ФАЗА 2: CatBoost (skip-hpo) ═══${NC}"

run_step "cb_baseline" \
    python run_pipeline_catboost.py --skip-hpo --gpu \
    --results "${EXP_DIR}/catboost_baseline"

run_step "cb_res_hyb" \
    python run_pipeline_catboost.py --skip-hpo --gpu --residual-target --hybrid-norm \
    --results "${EXP_DIR}/catboost_res_hyb"

# ============================================================
# ФАЗА 3: XGBoost
# ============================================================
echo -e "${YELLOW}═══ ФАЗА 3: XGBoost (skip-hpo) ═══${NC}"

run_step "xgb_res_hyb" \
    python run_pipeline_xgboost.py --skip-hpo --gpu --residual-target --hybrid-norm \
    --results "${EXP_DIR}/xgboost_res_hyb"

# ============================================================
# ФАЗА 4: HPO (раскомментировать лучшую комбо)
# ============================================================
echo -e "${YELLOW}═══ ФАЗА 4: HPO ═══${NC}"
echo "  (раскомментируй лучшую комбинацию)"

# run_step "v6_res_hyb_hpo" \
#     python run_pipeline_v6.py --hpo-trials 50 --residual-target --hybrid-norm \
#     --results "${EXP_DIR}/v6_res_hyb_hpo"

# run_step "v7_res_hyb_hpo" \
#     python run_pipeline_v7.py --hpo-trials 50 --residual-target --hybrid-norm \
#     --results "${EXP_DIR}/v7_res_hyb_hpo"

# run_step "cb_res_hyb_hpo" \
#     python run_pipeline_catboost.py --hpo-trials 50 --gpu --residual-target --hybrid-norm \
#     --results "${EXP_DIR}/cb_res_hyb_hpo"

# ============================================================
# ФАЗА 5: Production (ПОСЛЕ проверки!)
# ============================================================
echo "  Фаза 5 (production) — раскомментируй после проверки"

# run_step "v6_prod" \
#     python run_pipeline_v6.py --production --residual-target --hybrid-norm \
#     --results results/production/lgb_v6_no_news

# run_step "v7_prod" \
#     python run_pipeline_v7.py --production --residual-target --hybrid-norm \
#     --results results/production/lgb_v7_no_news

# run_step "cb_prod" \
#     python run_pipeline_catboost.py --production --gpu --residual-target --hybrid-norm \
#     --results results/production/catboost_with_news

# ============================================================
echo ""
echo "============================================================"
echo "  DONE — $(date)"
echo "  Experiment: ${EXP_NAME}"
echo "  Results:    ${EXP_DIR}/"
echo "  Logs:       ${LOGS}/"
echo "============================================================"
