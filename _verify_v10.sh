#!/bin/bash
set -euo pipefail
cd /Users/a.s.tabakov/Developer/invest
source .venv/bin/activate

cleanup() {
  echo "=== CLEANUP: Restoring prod ==="
  for d in results_v6_prod results_v7_prod results_catboost_prod results_xgboost_prod; do
    if [[ -d "${d}_bak_verify" ]]; then
      rm -rf "$d" 2>/dev/null || true
      mv "${d}_bak_verify" "$d"
      echo "  restored $d"
    fi
  done
  for d in results_mlp_prod results_v6_4h_prod results_v6_24h_prod; do
    if [[ -d "${d}_bak_verify" ]]; then
      rm -rf "$d" 2>/dev/null || true
      mv "${d}_bak_verify" "$d"
      echo "  restored $d"
    fi
  done
}
trap cleanup EXIT

echo "=== Step 1: Backup current prod ==="
for d in results_v6_prod results_v7_prod results_catboost_prod results_xgboost_prod; do
  if [[ -d "${d}_bak_verify" ]]; then
    rm -rf "${d}_bak_verify"
  fi
  cp -r "$d" "${d}_bak_verify"
  echo "  backed up $d"
done

echo ""
echo "=== Step 2: Hide MLP + multi-horizon ==="
for d in results_mlp_prod results_v6_4h_prod results_v6_24h_prod; do
  if [[ -d "$d" ]]; then
    rm -rf "${d}_bak_verify" 2>/dev/null || true
    mv "$d" "${d}_bak_verify"
    echo "  hidden $d"
  fi
done

echo ""
echo "=== Step 3: Swap v10 into prod ==="
for pair in v6:results_v6_v10 v7:results_v7_v10 catboost:results_catboost_v10 xgboost:results_xgboost_v10; do
  IFS=: read -r suffix src <<< "$pair"
  dst="results_${suffix}_prod"
  rm -rf "$dst"
  cp -r "$src" "$dst"
  echo "  $src -> $dst"
done

echo ""
echo "=== Step 4: Run sim ==="
python3 run_fast_sim.py \
  --data data/features/crypto_features_1h.parquet \
  --days 120 --start-date 2026-02-09 --end-date 2026-03-07 \
  --leverage 3 --kelly 0.8 --ensemble --edge-boost \
  --no-deriv-gate --no-ddstop --min-zscore 0.5

echo ""
echo "=== DONE — cleanup trap will restore prod ==="
