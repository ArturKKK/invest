#!/usr/bin/env bash
# run_r48_overnight.sh — Sequential R48 overnight research run
# All output saved to individual phase log files + master log

MASTER="results_r48_master.log"
PYTHON="./venv/bin/python3.10"

to_master() { echo "$1" | tee -a "$MASTER"; }

to_master "=========================================="
to_master "R48 OVERNIGHT RUN START: $(date)"
to_master "=========================================="

run_phase() {
    local phase="$1"
    local script="$2"
    local logfile="$3"
    to_master ""
    to_master "=== PHASE ${phase} START: $(date) ==="
    if $PYTHON "${script}" 2>&1 | tee "${logfile}"; then
        to_master "=== PHASE ${phase} COMPLETE (OK): $(date) ==="
    else
        to_master "=== PHASE ${phase} FAILED (rc=$?): $(date) ==="
    fi
}

run_phase "0"  "_research_r48_validation.py" "results_r48_phase0.log"
run_phase "12" "_research_r48_features.py"   "results_r48_phase12.log"
run_phase "3"  "_research_r48_cost.py"       "results_r48_phase3.log"
run_phase "4"  "_research_r48_combo.py"      "results_r48_phase4.log"

to_master ""
to_master "=========================================="
to_master "ALL PHASES DONE: $(date)"
to_master "=========================================="
