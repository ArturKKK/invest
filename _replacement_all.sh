#!/bin/bash
# Sweep all replacement experiments. Assumes baseline F10_F20 (NO FIXES)
# is the state we compare against (Net Sharpe 4L/2S cont = 3.777).
#
# IMPORTANT: caller must ensure the codebase is at F10_F20 baseline,
# i.e. FIX1=0 FIX2=0 applied via _ablation_harness.py BEFORE starting.
set -u
cd /workdir/invest

EXPS=(BASELINE A_seasonal_x_symbol B_hod_vol_rank C_relative_breadth D_session_regime E_regime_x_beta F_drop_only)

# sanity: make sure ablation harness left us at F10_F20
FIX1=0 FIX2=0 python _ablation_harness.py

MASTER=/data/datasets/replacement_master.log
: > "$MASTER"

SUMMARY=/data/datasets/replacement_summary.csv
echo "exp,mode,GrossSh,NetSh,Ret%,DD%,WR%,N" > "$SUMMARY"

for EXP in "${EXPS[@]}"; do
  LOG="/data/datasets/replacement_${EXP}.log"
  OUT="/data/datasets/replacement_${EXP}.csv"
  echo "=== [$(date -Iseconds)] EXP=${EXP} START ===" | tee -a "$MASTER"
  rm -f /data/datasets/results_r68_continuous_wf.csv
  EXP=$EXP python _run_replacement.py 2>&1 | tee "$LOG" | tail -5 | tee -a "$MASTER"
  echo "=== [$(date -Iseconds)] EXP=${EXP} DONE ===" | tee -a "$MASTER"
  if [ -f "$OUT" ]; then
    awk -v e="$EXP" 'NR>1 {print e","$0}' "$OUT" >> "$SUMMARY"
  else
    echo "${EXP},MISSING_CSV,,,,,," >> "$SUMMARY"
  fi
done

echo "=== ALL DONE ===" | tee -a "$MASTER"
echo "--- SUMMARY ---"
cat "$SUMMARY"
