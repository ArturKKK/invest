#!/usr/bin/env python3
"""VM setup check: pkg versions, fresh-archive data freshness, canonical cache presence."""
import os
import pandas as pd

print("=== VERSIONS ===")
for m in ['numpy', 'pandas', 'scipy', 'lightgbm', 'xgboost', 'sklearn']:
    try:
        mod = __import__(m)
        print(f"  {m:10s} {mod.__version__}")
    except Exception as e:
        print(f"  {m:10s} ERR {e}")

print("=== DATA FRESHNESS (data/ -> fresh S3 archive) ===")
files = {
    'OHLCV BTC': 'data/raw/BTC_USDT_1h.parquet',
    'futures_metrics': 'data/sentiment/binance_futures_metrics.parquet',
    'funding': 'data/sentiment/binance_funding_rates.parquet',
    'premium': 'data/sentiment/binance_premium_index.parquet',
    'dvol': 'data/sentiment/deribit_dvol.parquet',
    'fear_greed': 'data/sentiment/fear_greed.parquet',
    'macro': 'data/sentiment/macro_daily.parquet',
    'cg_taker': 'data/raw/coinglass/taker.parquet',
}
for n, p in files.items():
    try:
        df = pd.read_parquet(p)
        tc = next((c for c in ['timestamp', 'date'] if c in df.columns), None)
        print(f"  {n:16s} max={pd.to_datetime(df[tc]).max()}  rows={len(df):,}")
    except Exception as e:
        print(f"  {n:16s} ERR {e}")

print("=== CANONICAL CACHE / SCRIPTS ===")
for p in ['cache/r128_canonical_preds.parquet', 'cache/r128_canonical_regime.parquet',
          '_r136_s6_retest.py', 'src/costs.py', '_tmp_verify_2831.py']:
    print(f"  {'OK ' if os.path.exists(p) else 'MISSING'} {p}")
