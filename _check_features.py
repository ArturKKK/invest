#!/usr/bin/env python3
"""Deep feature health check — especially derivatives."""
import pandas as pd
import numpy as np
import json
import os

# Load the latest features the same way run_trading.py does
data_dir = 'data/sentiment'
raw_dir = 'data/raw'

# Check binance futures metrics
bfm_path = os.path.join(data_dir, 'binance_futures_metrics.parquet')
if os.path.exists(bfm_path):
    df = pd.read_parquet(bfm_path)
    print(f"binance_futures_metrics.parquet:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Date range: {df.timestamp.min()} → {df.timestamp.max()}")
    print(f"  Symbols: {df.symbol.nunique()}")
    # Check for recent data
    recent = df[df.timestamp >= '2026-03-10']
    print(f"  Rows since Mar 10: {len(recent)}")
    print(f"  Non-null counts in recent data:")
    for col in df.columns:
        if col not in ['timestamp', 'symbol']:
            nz = recent[col].notna().sum()
            total = len(recent)
            print(f"    {col}: {nz}/{total} ({nz/total*100:.0f}%)")
else:
    print("NO binance_futures_metrics.parquet!")

# Check what features end up as zero in the latest snapshot
# Load the latest trade log to see feature names
print("\n--- Checking trade log for zero features ---")
import glob
logs = sorted(glob.glob('trading_logs/trade_20260312_*.json'))
if logs:
    with open(logs[-1]) as f:
        trade = json.load(f)
    signals = trade.get('signals_top5', [])
    if signals:
        s = signals[0]
        zero_feats = [k for k, v in s.items() if isinstance(v, (int, float)) and v == 0 and k not in ['target_ret', 'target_cls']]
        print(f"  Zero features in top signal: {len(zero_feats)}")
        deriv_zeros = [f for f in zero_feats if any(x in f for x in ['oi_', 'taker_', 'ls_ratio', 'funding_bi', 'liquidation', 'basis_', 'global_ls'])]
        print(f"  Deriv-related zeros: {len(deriv_zeros)}")
        for f in deriv_zeros:
            print(f"    {f}")
        other_zeros = [f for f in zero_feats if f not in deriv_zeros]
        if other_zeros:
            print(f"  Other zeros: {len(other_zeros)}")
            for f in other_zeros:
                print(f"    {f}")
else:
    print("  No trade logs for today!")

# Check deriv_only model state
print("\n--- Deriv-only model state ---")
for p in ['results/production/deriv_only', 'results_deriv']:
    if os.path.isdir(p):
        files = os.listdir(p)
        print(f"  {p}: {len(files)} files")
        for f in files:
            print(f"    {f}")
    else:
        print(f"  {p}: NOT FOUND")

# Check XGBoost prod
print("\n--- XGBoost model state ---")
for p in ['results_xgboost_prod', 'results/production/xgboost']:
    if os.path.isdir(p):
        files = os.listdir(p)
        print(f"  {p}: {len(files)} files")
    else:
        print(f"  {p}: NOT FOUND")
