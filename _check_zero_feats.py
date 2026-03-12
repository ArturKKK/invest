#!/usr/bin/env python3
"""Identify exactly which features are all-zero in the latest snapshot."""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import os, json, glob

# Use the build path from run_trading
from run_trading import SYMBOLS, EXCLUDE_COLS, fetch_ohlcv, build_features

root = '.'
print("Fetching data...")
raw_df = fetch_ohlcv(SYMBOLS, hours=800)
print(f"Raw: {raw_df.shape}")

print("Building features...")
features_df = build_features(raw_df)
print(f"Features: {features_df.shape}")

# Get latest snapshot
latest = features_df.groupby('symbol').last().reset_index()

# Check which columns are all-zero
feature_cols = [c for c in latest.columns if c not in EXCLUDE_COLS and c != 'symbol']
print(f"Total feature cols: {len(feature_cols)}")

all_zero = []
all_nan = []
deriv_feats = []
for c in sorted(feature_cols):
    vals = latest[c].values
    is_deriv = any(x in c for x in ['oi_', 'taker_', 'ls_ratio', 'funding_bi', 'liquidation', 'basis_', 'global_ls', 'deriv', 'binance_fund'])
    if is_deriv:
        deriv_feats.append(c)
    if np.all(np.isnan(vals)):
        all_nan.append(c)
        continue
    if np.all(vals == 0) or np.all(np.nan_to_num(vals) == 0):
        all_zero.append(c)

print(f"\n=== ALL-ZERO features ({len(all_zero)}) ===")
for c in all_zero:
    tag = " [DERIV]" if any(x in c for x in ['oi_', 'taker_', 'ls_ratio', 'funding_bi', 'liquidation', 'basis_', 'global_ls', 'deriv', 'binance_fund']) else ""
    print(f"  {c}{tag}")

print(f"\n=== ALL-NaN features ({len(all_nan)}) ===")
for c in all_nan:
    print(f"  {c}")

print(f"\n=== Derivatives features status ({len(deriv_feats)}) ===")
for c in sorted(deriv_feats):
    vals = latest[c].values
    nz = np.count_nonzero(np.nan_to_num(vals))
    mn = np.nanmean(vals) if not np.all(np.isnan(vals)) else float('nan')
    status = "OK" if nz > 0 else "ZERO"
    print(f"  {status:4s} {c:<40} non-zero={nz}/{len(vals)}  mean={mn:.6f}")
