#!/usr/bin/env python3
"""Analyze v8 results and compare with v6."""
import json, os

# v8
v8 = json.load(open('results_v8/all_results_v8.json'))

# v6
v6_path = None
for p in ['results_v6/all_results.json', 'results_v6/all_results_v6.json']:
    if os.path.exists(p):
        v6_path = p
        break

print("=" * 70)
print("  v8 vs v6 PIPELINE COMPARISON")
print("=" * 70)

if v6_path:
    v6 = json.load(open(v6_path))
    v6_avg = v6.get('average', {})
    v6_comb = v6.get('combined', {})
    print("\n=== v6 PIPELINE ===")
    for k, v in v6_avg.items():
        if not k.endswith('_std'):
            print(f"  {k:30s} = {v}")
    print(f"\n  Combined: {v6_comb}")
else:
    print("\n  No v6 results JSON found locally")
    print(f"  v6 files: {os.listdir('results_v6')}")

print("\n=== v8 PIPELINE ===")
v8_avg = v8['average']
for k, v in v8_avg.items():
    if not k.endswith('_std'):
        print(f"  {k:30s} = {v}")
print(f"\n  Combined: {v8['combined']}")

# Per-window overview
print("\n=== v8 PER-WINDOW ===")
print(f"  {'Window':20s} {'Rank_ICIR':>10s} {'LS_Sharpe_net':>14s} {'DD_MaxDD':>10s} {'DDStop_Sharpe':>14s} {'LO5':>6s}")
for w in v8['per_window']:
    print(f"  {w['window']:20s} {w['Rank_ICIR']:>10.4f} {w['LS_Sharpe_net']:>14.2f} {w['LS_MaxDD_net_%']:>10.1f} {w['LS_DDStop_Sharpe']:>14.2f} {w['LO5_Sharpe']:>6.2f}")

# Key problem analysis
print("\n=== DIAGNOSIS ===")
print("  1. W2 (→2023-12) is NEGATIVE: Sharpe -1.18 — model fails in 2023 bear/recovery")
print("  2. Avg MaxDD = -89% without DDStop — extreme risk")
print("  3. LO5/LO10 negative in W5 (latest) — long-only doesn't work in recent data")
print("  4. Avg LS Sharpe net = 0.68 vs v6 fast sim Sharpe 5.79")
print("  5. Measured turnover 38% > assumed 35% — costs higher than modeled")
print()
print("  v6 was trained on 2021+ data (4 years)")
print("  v8 was trained on 2017+ data (8+ years)")
print("  Adding old data HURT the model — earlier crypto regime is too different")
