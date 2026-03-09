#!/bin/bash
# =============================================================================
#  Полный цикл обучения всех моделей с новыми флагами
#  Запуск: bash run_train_all.sh
#
#  Логика:
#    Фаза 1 — A/B тесты (skip-hpo, быстро): каждый флаг отдельно
#    Фаза 2 — Лучшая комбинация с HPO (долго, но один раз)
#    Фаза 3 — CatBoost + XGBoost с теми же флагами
#    Фаза 4 — Production модели (после проверки результатов)
#
#  skip-hpo: HPO = 50 Optuna trials × 5000 деревьев = часы на каждый пайплайн.
#  Сначала находим лучшую комбинацию фичей на default params,
#  потом один раз делаем HPO на ней.
# =============================================================================

set -euo pipefail

LOGS="logs"
mkdir -p "$LOGS"

# Timestamp для уникальных имён логов
TS=$(date +%Y%m%d_%H%M%S)

# Цвета
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
echo "  Logs → $LOGS/"
echo "============================================================"
echo ""

# ============================================================
# ФАЗА 1: A/B тесты — каждый флаг отдельно (skip-hpo)
# ============================================================
echo -e "${YELLOW}═══ ФАЗА 1: A/B тесты v6 (skip-hpo) ═══${NC}"

# 1a. Baseline v6 (как было, для сравнения)
run_step "v6_baseline" \
    python run_pipeline_v6.py --skip-hpo \
    --results results_v6_baseline

# 1b. Residual target
run_step "v6_residual" \
    python run_pipeline_v6.py --skip-hpo --residual-target \
    --results results_v6_residual

# 1c. Hybrid normalization
run_step "v6_hybrid" \
    python run_pipeline_v6.py --skip-hpo --hybrid-norm \
    --results results_v6_hybrid

# 1d. Residual + hybrid (combo)
run_step "v6_res_hyb" \
    python run_pipeline_v6.py --skip-hpo --residual-target --hybrid-norm \
    --results results_v6_res_hyb

# 1e. Residual + hybrid + null importance FS
run_step "v6_res_hyb_null" \
    python run_pipeline_v6.py --skip-hpo --residual-target --hybrid-norm --null-importance \
    --results results_v6_res_hyb_null

# 1f. LambdaRank (отдельно — другой objective)
run_step "v6_lambdarank" \
    python run_pipeline_v6.py --skip-hpo --residual-target --lambdarank \
    --results results_v6_lambdarank

# ============================================================
# ФАЗА 1b: A/B тесты v7 (skip-hpo)
# ============================================================
echo -e "${YELLOW}═══ ФАЗА 1b: A/B тесты v7 (skip-hpo) ═══${NC}"

# v7 baseline
run_step "v7_baseline" \
    python run_pipeline_v7.py --skip-hpo \
    --results results_v7_baseline

# v7 с лучшей комбинацией (residual + hybrid)
run_step "v7_res_hyb" \
    python run_pipeline_v7.py --skip-hpo --residual-target --hybrid-norm \
    --results results_v7_res_hyb

# v7 residual + hybrid + null importance
run_step "v7_res_hyb_null" \
    python run_pipeline_v7.py --skip-hpo --residual-target --hybrid-norm --null-importance \
    --results results_v7_res_hyb_null

# ============================================================
# ФАЗА 2: CatBoost — те же комбинации
# ============================================================
echo -e "${YELLOW}═══ ФАЗА 2: CatBoost (skip-hpo) ═══${NC}"

# CatBoost baseline
run_step "cb_baseline" \
    python run_pipeline_catboost.py --skip-hpo --gpu \
    --results results_catboost_baseline

# CatBoost residual + hybrid
run_step "cb_res_hyb" \
    python run_pipeline_catboost.py --skip-hpo --gpu --residual-target --hybrid-norm \
    --results results_catboost_res_hyb

# ============================================================
# ФАЗА 3: XGBoost (disabled в ансамбле, но обучим для сравнения)
# ============================================================
echo -e "${YELLOW}═══ ФАЗА 3: XGBoost (skip-hpo, для сравнения) ═══${NC}"

run_step "xgb_res_hyb" \
    python run_pipeline_xgboost.py --skip-hpo --gpu --residual-target --hybrid-norm \
    --results results_xgboost_res_hyb

# ============================================================
# ФАЗА 4: HPO на лучших комбинациях (долго!)
# Раскомментируй после проверки результатов фазы 1-3
# ============================================================
echo -e "${YELLOW}═══ ФАЗА 4: HPO на лучших комбинациях ═══${NC}"
echo "  (раскомментируй в скрипте когда увидишь какая комбинация лучше)"

# run_step "v6_res_hyb_hpo" \
#     python run_pipeline_v6.py --hpo-trials 50 --residual-target --hybrid-norm \
#     --results results_v6_res_hyb_hpo

# run_step "v7_res_hyb_hpo" \
#     python run_pipeline_v7.py --hpo-trials 50 --residual-target --hybrid-norm \
#     --results results_v7_res_hyb_hpo

# run_step "cb_res_hyb_hpo" \
#     python run_pipeline_catboost.py --hpo-trials 50 --gpu --residual-target --hybrid-norm \
#     --results results_catboost_res_hyb_hpo

# ============================================================
# ФАЗА 5: Production (ТОЛЬКО после проверки!)
# ============================================================
echo "  Фаза 5 (production) — раскомментируй после проверки"

# run_step "v6_prod" \
#     python run_pipeline_v6.py --production --residual-target --hybrid-norm \
#     --results results_v6_prod

# run_step "v7_prod" \
#     python run_pipeline_v7.py --production --residual-target --hybrid-norm \
#     --results results_v7_prod

# run_step "cb_prod" \
#     python run_pipeline_catboost.py --production --gpu --residual-target --hybrid-norm \
#     --results results_catboost_prod

# ============================================================
echo ""
echo "============================================================"
echo "  DONE — $(date)"
echo "  Логи: $LOGS/"
echo ""
echo "  Сравнить результаты:"
echo "    grep -h 'LS_DDStop_Sharpe\|Rank_IC\|window' results_v6_*/all_results_v6.json"
echo "============================================================"
