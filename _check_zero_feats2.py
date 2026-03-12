#!/usr/bin/env python3
"""Check which features are all-zero in the ACTUAL latest signals."""
import json, glob, os
import numpy as np

logs = sorted(glob.glob('trading_logs/trade_20260312_*.json'))
if not logs:
    logs = sorted(glob.glob('trading_logs/trade_202603*.json'))
print(f"Latest log: {logs[-1]}")

with open(logs[-1]) as f:
    trade = json.load(f)

top5 = trade.get('signals_top5', [])
bot5 = trade.get('signals_bot5', [])
all_signals = top5 + bot5
print(f"Signals available: {len(all_signals)} (top5 + bot5)")

if not all_signals:
    print("No signals in trade log!")
    exit()

s = all_signals[0]
feature_keys = [k for k, v in s.items() if isinstance(v, (int, float)) and k not in ['score']]
print(f"Features in signal: {len(feature_keys)}")

zero_features = []
nan_features = []
deriv_keywords = ['oi_', 'taker_', 'ls_ratio', 'funding_bi', 'liquidation', 'basis_', 'global_ls', 'deriv', 'binance_fund']

for k in sorted(feature_keys):
    vals = [sig.get(k, 0) for sig in all_signals]
    if all(v is None or (isinstance(v, float) and np.isnan(v)) for v in vals):
        nan_features.append(k)
    elif all(v == 0 or v is None for v in vals):
        zero_features.append(k)

print(f"\n=== ALL-ZERO features ({len(zero_features)}) ===")
for c in zero_features:
    tag = " [DERIV]" if any(x in c for x in deriv_keywords) else ""
    print(f"  {c}{tag}")

print(f"\n=== ALL-NaN features ({len(nan_features)}) ===")
for c in nan_features:
    tag = " [DERIV]" if any(x in c for x in deriv_keywords) else ""
    print(f"  {c}{tag}")

print(f"\n=== Derivatives feature check ===")
for k in sorted(feature_keys):
    if any(x in k for x in deriv_keywords):
        vals = [sig.get(k, 0) for sig in all_signals]
        nz = sum(1 for v in vals if v and v != 0)
        mn = np.nanmean([v for v in vals if v is not None])
        status = "OK" if nz > 0 else "ZERO"
        print(f"  {status:4s} {k:<45} non-zero={nz}/{len(vals)} mean={mn:.6f}")
