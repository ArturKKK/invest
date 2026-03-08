#!/usr/bin/env python3
"""
Test inference on VPS — runs one full signal generation cycle
and measures time + memory usage.
"""
import os
import sys
import time
import resource
import psutil

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("  INFERENCE TEST — measuring resources")
print("=" * 60)

proc = psutil.Process()
mem_before = proc.memory_info().rss / 1024 / 1024  # MB

# System-wide
sys_mem = psutil.virtual_memory()
print(f"\nSystem RAM: {sys_mem.total / 1024**3:.1f} GB total, "
      f"{sys_mem.available / 1024**3:.1f} GB available")
print(f"Process RAM before: {mem_before:.0f} MB")

# Step 1: Import + load models
print(f"\n--- Step 1: Import libraries + load models ---")
t0 = time.time()

import numpy as np
import pandas as pd
import lightgbm as lgb

from run_trading import (
    fetch_ohlcv, build_features, generate_signal,
    cross_sectional_rank,
    load_lgb_models, load_catboost_models,
    SYMBOLS, EXCLUDE_COLS, DEFAULT_RISK
)

mem_after_import = proc.memory_info().rss / 1024 / 1024
print(f"  Import time: {time.time() - t0:.1f}s")
print(f"  RAM after imports: {mem_after_import:.0f} MB (+{mem_after_import - mem_before:.0f} MB)")

# Step 2: Fetch OHLCV
print(f"\n--- Step 2: Fetch OHLCV data ({len(SYMBOLS)} coins, 800h) ---")
t1 = time.time()
raw = fetch_ohlcv(SYMBOLS, hours=800)
mem_after_fetch = proc.memory_info().rss / 1024 / 1024
print(f"  Fetch time: {time.time() - t1:.1f}s")
print(f"  Rows: {len(raw):,}")
print(f"  RAM: {mem_after_fetch:.0f} MB (+{mem_after_fetch - mem_after_import:.0f} MB)")

# Step 3: Build features
print(f"\n--- Step 3: Build features ---")
t2 = time.time()
df = build_features(raw)
feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS
             and not c.startswith('target_')
             and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
df = cross_sectional_rank(df, feat_cols)
mem_after_features = proc.memory_info().rss / 1024 / 1024
print(f"  Feature time: {time.time() - t2:.1f}s")
print(f"  Rows: {len(df):,}, Features: {len(feat_cols)}")
print(f"  RAM: {mem_after_features:.0f} MB (+{mem_after_features - mem_after_fetch:.0f} MB)")

# Step 4: Inference (generate signal)
print(f"\n--- Step 4: ML Inference (15 models) ---")
t3 = time.time()
root = os.path.dirname(os.path.abspath(__file__))
signals = generate_signal(df, feat_cols, root)
mem_after_inference = proc.memory_info().rss / 1024 / 1024
print(f"  Inference time: {time.time() - t3:.1f}s")
if signals is not None:
    print(f"  Signals shape: {signals.shape}")
    print(f"  Top 5:")
    for _, row in signals.head(5).iterrows():
        print(f"    {row['symbol']:15s} score={row['score']:+.4f}")
    print(f"  Bottom 5:")
    for _, row in signals.tail(5).iterrows():
        print(f"    {row['symbol']:15s} score={row['score']:+.4f}")
else:
    print(f"  ⚠️ No signals generated!")
print(f"  RAM: {mem_after_inference:.0f} MB (+{mem_after_inference - mem_after_features:.0f} MB)")

# Summary
total_time = time.time() - t0
peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB -> MB on Linux
# On Linux ru_maxrss is in KB
final_ram = proc.memory_info().rss / 1024 / 1024
sys_mem_after = psutil.virtual_memory()

print(f"\n{'=' * 60}")
print(f"  SUMMARY")
print(f"{'=' * 60}")
print(f"  Total time:      {total_time:.1f}s")
print(f"  Process RAM:     {final_ram:.0f} MB (peak ~{peak_mem:.0f} MB)")
print(f"  RAM increase:    +{final_ram - mem_before:.0f} MB")
print(f"  System RAM used: {sys_mem_after.percent:.1f}% "
      f"({sys_mem_after.used / 1024**3:.1f} / {sys_mem_after.total / 1024**3:.1f} GB)")
print(f"  System RAM free: {sys_mem_after.available / 1024**3:.1f} GB")

if sys_mem_after.percent > 85:
    print(f"\n  ⚠️  WARNING: RAM usage high! Consider upgrading VPS")
elif sys_mem_after.percent > 70:
    print(f"\n  ⚡ RAM usage moderate — should be fine for production")
else:
    print(f"\n  ✅ RAM usage low — VPS has plenty of headroom")

if total_time > 300:
    print(f"  ⚠️  WARNING: Inference took > 5min! May need more CPU")
elif total_time > 60:
    print(f"  ⚡ Inference OK but not fast — acceptable for 24h cycles")
else:
    print(f"  ✅ Inference fast enough — no issues")
