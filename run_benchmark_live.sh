#!/bin/bash
# Benchmark: compare fast_sim configurations with LIVE data fetch
# Uses production models (cluster-trained 2026-03-12)

set -e
cd "$(dirname "$0")"
source .venv/bin/activate

DAYS=30
LOG_DIR="trading_logs/benchmark_prod_$(date +%Y%m%d_%H%M)"
mkdir -p "$LOG_DIR"

COMMON="--days $DAYS --capital 5000 --rebal 12"

echo "========================================"
echo "  BENCHMARK SUITE — LIVE FETCH — ${DAYS}d"
echo "  Models: production (cluster 2026-03-12)"
echo "  Output: $LOG_DIR"
echo "========================================"

# 1) Single v6
echo -e "\n\n▶▶▶ [1/7] Single v6 ◀◀◀"
python run_fast_sim.py $COMMON --model-dir results_v6 --edge-boost --no-deriv-gate 2>&1 | tee "$LOG_DIR/01_single_v6.log"
cp trading_logs/fast_sim_equity.csv "$LOG_DIR/01_single_v6_equity.csv"

# 2) Single v7
echo -e "\n\n▶▶▶ [2/7] Single v7 ◀◀◀"
python run_fast_sim.py $COMMON --model-dir results_v7 --edge-boost --no-deriv-gate 2>&1 | tee "$LOG_DIR/02_single_v7.log"
cp trading_logs/fast_sim_equity.csv "$LOG_DIR/02_single_v7_equity.csv"

# 3) Ensemble v6+v7+CB — no deriv gate
echo -e "\n\n▶▶▶ [3/7] Ensemble (no deriv gate) ◀◀◀"
python run_fast_sim.py $COMMON --ensemble --edge-boost --no-deriv-gate 2>&1 | tee "$LOG_DIR/03_ensemble_no_deriv.log"
cp trading_logs/fast_sim_equity.csv "$LOG_DIR/03_ensemble_no_deriv_equity.csv"

# 4) Ensemble v6+v7+CB + deriv gate
echo -e "\n\n▶▶▶ [4/7] Ensemble + deriv gate ◀◀◀"
python run_fast_sim.py $COMMON --ensemble --edge-boost 2>&1 | tee "$LOG_DIR/04_ensemble_deriv.log"
cp trading_logs/fast_sim_equity.csv "$LOG_DIR/04_ensemble_deriv_equity.csv"

# 5) Ensemble + meta lgb_minimal
echo -e "\n\n▶▶▶ [5/7] Ensemble + meta lgb_minimal ◀◀◀"
python run_fast_sim.py $COMMON --ensemble --edge-boost --meta-model auto --meta-variant lgb_minimal 2>&1 | tee "$LOG_DIR/05_ensemble_meta_lgb_min.log"
cp trading_logs/fast_sim_equity.csv "$LOG_DIR/05_ensemble_meta_lgb_min_equity.csv"

# 6) Ensemble + meta ridge
echo -e "\n\n▶▶▶ [6/7] Ensemble + meta ridge ◀◀◀"
python run_fast_sim.py $COMMON --ensemble --edge-boost --meta-model auto --meta-variant ridge 2>&1 | tee "$LOG_DIR/06_ensemble_meta_ridge.log"
cp trading_logs/fast_sim_equity.csv "$LOG_DIR/06_ensemble_meta_ridge_equity.csv"

# 7) Ensemble + deriv gate + 3x leverage
echo -e "\n\n▶▶▶ [7/7] Ensemble + deriv gate + 3x leverage ◀◀◀"
python run_fast_sim.py $COMMON --ensemble --edge-boost --leverage 3 2>&1 | tee "$LOG_DIR/07_ensemble_deriv_3x.log"
cp trading_logs/fast_sim_equity.csv "$LOG_DIR/07_ensemble_deriv_3x_equity.csv"

echo -e "\n\n========================================"
echo "  BENCHMARK COMPLETE"
echo "  Logs: $LOG_DIR/"
echo "========================================"

# Extract summary metrics from logs
echo -e "\n  SUMMARY TABLE:"
echo "  Config                        | Return  | MaxDD   | Sharpe | Sharpe_HAC | Calmar"
echo "  ------------------------------|---------|---------|--------|------------|-------"
for f in "$LOG_DIR"/*.log; do
    name=$(basename "$f" .log | sed 's/^[0-9]*_//')
    ret=$(grep "Return:" "$f" | tail -1 | awk '{print $2}')
    dd=$(grep "Max DD:" "$f" | tail -1 | awk '{print $3}')
    sh=$(grep "Sharpe:" "$f" | grep -v HAC | tail -1 | awk '{print $2}')
    sh_hac=$(grep "Sharpe HAC:" "$f" | tail -1 | awk '{print $3}')
    cal=$(grep "Calmar:" "$f" | tail -1 | awk '{print $2}')
    printf "  %-30s | %7s | %7s | %6s | %10s | %s\n" "$name" "$ret" "$dd" "$sh" "$sh_hac" "$cal"
done | tee "$LOG_DIR/summary.txt"

echo ""
