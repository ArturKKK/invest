#!/usr/bin/env bash
# Overnight experiment runner for R60, R63, R61
# Run on MLC: bash /workdir/invest/_run_overnight.sh
set -euo pipefail

LOG_DIR="/data/datasets"
WORKDIR="/workdir/invest"
VENV="$WORKDIR/.venv"

cd "$WORKDIR"
source "$VENV/bin/activate"

echo "====================================================="
echo "  OVERNIGHT EXPERIMENTS: $(date)"
echo "====================================================="

run_exp() {
    local name="$1"
    local script="$2"
    local log="$LOG_DIR/${name}.log"
    echo ""
    echo ">>> Starting $name at $(date)"
    echo ">>> Log: $log"
    python "$script" > "$log" 2>&1
    local status=$?
    if [ $status -eq 0 ]; then
        echo ">>> $name DONE at $(date) — SUCCESS"
    else
        echo ">>> $name FAILED (exit $status) at $(date)"
    fi
    return $status
}

# Run R60 and R63 in parallel
python _research_r60_portfolio_opt.py > "$LOG_DIR/run_r60.log" 2>&1 &
PID_R60=$!
echo ">>> R60 started PID=$PID_R60"

python _research_r63_uncertainty.py > "$LOG_DIR/run_r63.log" 2>&1 &
PID_R63=$!
echo ">>> R63 started PID=$PID_R63"

# Wait for both
echo ">>> Waiting for R60 (PID=$PID_R60) and R63 (PID=$PID_R63)..."
wait $PID_R60 && echo ">>> R60 DONE" || echo ">>> R60 FAILED"
wait $PID_R63 && echo ">>> R63 DONE" || echo ">>> R63 FAILED"

echo ""
echo ">>> Starting R61 at $(date)"
python _research_r61_temporal.py > "$LOG_DIR/run_r61.log" 2>&1
echo ">>> R61 DONE at $(date)"

echo ""
echo "====================================================="
echo "  ALL EXPERIMENTS COMPLETE: $(date)"
echo "====================================================="

# Print summary tails
echo ""
echo "=== R60 summary ==="
tail -20 "$LOG_DIR/run_r60.log"
echo ""
echo "=== R63 summary ==="
tail -20 "$LOG_DIR/run_r63.log"
echo ""
echo "=== R61 summary ==="
tail -20 "$LOG_DIR/run_r61.log"
