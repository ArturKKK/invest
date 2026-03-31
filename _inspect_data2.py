#!/usr/bin/env python3
"""Inspect available data files to understand what we have."""
import pandas as pd
import os

DATA = "data/sentiment"

for fname in sorted(os.listdir(DATA)):
    fpath = os.path.join(DATA, fname)
    if not fname.endswith(".parquet"):
        continue
    print(f"\n{'='*60}")
    print(f"FILE: {fname}")
    print(f"{'='*60}")
    df = pd.read_parquet(fpath)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Index: {df.index.name} dtype={df.index.dtype}")
    if hasattr(df.index, 'min') and len(df) > 0:
        print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string())
    print(f"\nDtypes:\n{df.dtypes}")

# Also check raw OHLCV
print(f"\n{'='*60}")
print("RAW OHLCV files:")
print(f"{'='*60}")
raw_dir = "data/raw"
for fname in sorted(os.listdir(raw_dir))[:3]:
    fpath = os.path.join(raw_dir, fname)
    if fname.endswith(".parquet"):
        df = pd.read_parquet(fpath)
        print(f"{fname}: shape={df.shape}, cols={list(df.columns)}, range={df.index.min()} to {df.index.max()}")
