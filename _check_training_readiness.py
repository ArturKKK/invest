#!/usr/bin/env python3
"""Check if we have everything locally to train deriv_only model."""
import pandas as pd
import os
import glob
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=== LOCAL DATA CHECK ===\n")

# 1. Features cache
ff = pd.read_parquet('data/features/crypto_features_1h.parquet')
print(f"Features cache: {ff.shape}")
print(f"  Range: {ff['timestamp'].min()} -> {ff['timestamp'].max()}")
print(f"  Symbols: {ff['symbol'].nunique()}")
print(f"  Size: {os.path.getsize('data/features/crypto_features_1h.parquet')/1e9:.2f} GB")

# 2. OHLCV
raw_files = sorted(glob.glob('data/raw/*_1h.parquet'))
print(f"\nRaw OHLCV files: {len(raw_files)}")
btc = pd.read_parquet('data/raw/BTC_USDT_1h.parquet')
print(f"  BTC range: {btc['timestamp'].min()} -> {btc['timestamp'].max()}")

# 3. Binance derivatives
m = pd.read_parquet('data/sentiment/binance_futures_metrics.parquet')
print(f"\nBinance derivatives metrics:")
print(f"  Shape: {m.shape}")
print(f"  Range: {m['timestamp'].min()} -> {m['timestamp'].max()}")
print(f"  Symbols: {m['symbol'].nunique()}")

# 4. Overlap
overlap_start = max(m['timestamp'].min(), ff['timestamp'].min())
overlap_end = min(m['timestamp'].max(), ff['timestamp'].max())
overlap = ff[(ff['timestamp'] >= overlap_start) & (ff['timestamp'] <= overlap_end)]
print(f"\n  Features-Deriv overlap: {overlap_start.date()} -> {overlap_end.date()}")
print(f"  Overlap rows: {len(overlap):,}")

# 5. PRODUCTION_WINDOW
from run_pipeline_v6 import PRODUCTION_WINDOW, WALK_FORWARD_WINDOWS
print(f"\nPRODUCTION_WINDOW:")
for k,v in PRODUCTION_WINDOW.items():
    print(f"  {k}: {v}")

# 6. Check OHLCV freshness issue
print(f"\n=== FRESHNESS ISSUE ===")
print(f"OHLCV ends: {btc['timestamp'].max()}")
print(f"Features cache ends: {ff['timestamp'].max()}")
print(f"Deriv metrics ends: {m['timestamp'].max()}")
gap = pd.Timestamp.now('UTC') - ff['timestamp'].max()
print(f"Features cache age: {gap.days} days old")
if gap.days > 1:
    print(f"  -> Need to update OHLCV and rebuild features cache!")

# 7. Deps check
print(f"\n=== DEPENDENCIES ===")
try:
    import lightgbm as lgb
    print(f"  lightgbm: {lgb.__version__}")
except ImportError:
    print(f"  lightgbm: NOT INSTALLED")
try:
    import catboost
    print(f"  catboost: {catboost.__version__}")
except ImportError:
    print(f"  catboost: NOT INSTALLED")
try:
    import ta
    print(f"  ta: {ta.__version__ if hasattr(ta, '__version__') else 'OK'}")
except ImportError:
    print(f"  ta: NOT INSTALLED")
try:
    import scipy
    print(f"  scipy: {scipy.__version__}")
except ImportError:
    print(f"  scipy: NOT INSTALLED")
