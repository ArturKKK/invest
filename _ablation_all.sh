#!/bin/bash
# Launches all 4 ablation configs sequentially in background on the VM.
# Each config: FIX1 × FIX2 ∈ {0,1}² → saves ablation_F1{a}_F2{b}.{log,csv}
# Usage on VM: nohup bash /workdir/invest/_ablation_all.sh > /data/datasets/ablation_all.nohup 2>&1 &
set -u
cd /workdir/invest
MASTER_LOG=/data/datasets/ablation_master.log
echo "=== [$(date -Iseconds)] MASTER START ===" | tee -a "$MASTER_LOG"
for F1 in 0 1; do
  for F2 in 0 1; do
    TAG="F1${F1}_F2${F2}"
    echo "--- [$(date -Iseconds)] BEGIN $TAG ---" | tee -a "$MASTER_LOG"
    FIX1=$F1 FIX2=$F2 bash /workdir/invest/_run_ablation.sh
    echo "--- [$(date -Iseconds)] END   $TAG ---" | tee -a "$MASTER_LOG"
  done
done
echo "=== [$(date -Iseconds)] MASTER DONE ===" | tee -a "$MASTER_LOG"
echo "--- SUMMARY ---" | tee -a "$MASTER_LOG"
for F1 in 0 1; do for F2 in 0 1; do
  TAG="F1${F1}_F2${F2}"
  CSV="/data/datasets/ablation_${TAG}.csv"
  echo "== $TAG ==" | tee -a "$MASTER_LOG"
  [ -f "$CSV" ] && cat "$CSV" | tee -a "$MASTER_LOG"
done; done
