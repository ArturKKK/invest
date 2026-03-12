#!/usr/bin/env python3
"""
Real-time derivatives data fetcher via Binance REST API.

Fetches the latest OI, taker ratio, and top-trader L/S from /futures/data/
endpoints (free, no API key). Appends fresh rows to the metrics parquet
so that add_derivatives_features() picks them up naturally via merge_asof.

This fixes the "10 zero features" problem caused by data.binance.vision
lagging ~24h behind real-time.

Endpoints used:
  - /futures/data/openInterestHist       (OI history, 1h, last 30 periods)
  - /futures/data/takerlongshortRatio     (taker buy/sell ratio, 1h)
  - /futures/data/topLongShortAccountRatio (top trader L/S, 1h)
  - /futures/data/globalLongShortAccountRatio (global L/S, 1h)

Usage:
  from src.data.fetch_realtime_derivatives import patch_metrics_realtime
  patch_metrics_realtime(project_root)  # patches parquet in-place
"""

import os
import time
import warnings
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import requests

warnings.filterwarnings('ignore')

FAPI_DATA = "https://fapi.binance.com/futures/data"

# Same 50 symbols as download_binance_futures.py
SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
    'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT',
    'MATICUSDT', 'UNIUSDT', 'ATOMUSDT', 'LTCUSDT', 'ETCUSDT',
    'FILUSDT', 'APTUSDT', 'ARBUSDT', 'OPUSDT', 'NEARUSDT',
    'AAVEUSDT', 'MKRUSDT', 'GRTUSDT', 'INJUSDT', 'FTMUSDT',
    'ALGOUSDT', 'SANDUSDT', 'MANAUSDT', 'AXSUSDT', 'THETAUSDT',
    'RUNEUSDT', 'EGLDUSDT', 'XTZUSDT', 'FLOWUSDT', 'CHZUSDT',
    'CRVUSDT', 'LDOUSDT', 'SNXUSDT', 'COMPUSDT', 'YFIUSDT',
    'SUSHIUSDT', 'ENJUSDT', 'BATUSDT', 'ZILUSDT', 'ONEUSDT',
    'IOTAUSDT', 'ICXUSDT', 'ENSUSDT', 'IMXUSDT', 'GALAUSDT',
]


def _to_our_symbol(binance_sym: str) -> str:
    """BTCUSDT → BTC/USDT"""
    return binance_sym.replace('USDT', '/USDT')


def _fetch_json(url: str, params: dict, retries: int = 2) -> list | None:
    """Fetch JSON from Binance with retry."""
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            # Some symbols may not be available
            if resp.status_code in (400, 404):
                return None
        except Exception:
            time.sleep(1)
    return None


def _fetch_symbol_data(symbol: str) -> pd.DataFrame | None:
    """Fetch the last 30h of OI, taker, top L/S, global L/S for one symbol.

    Returns a DataFrame with columns matching binance_futures_metrics.parquet:
      timestamp, symbol, oi_value_usd, top_ls_ratio, top_long_pct,
      global_ls_ratio, global_long_pct, taker_buy_sell_ratio
    """
    frames = {}

    # 1. OI history (30 × 1h)
    data = _fetch_json(f"{FAPI_DATA}/openInterestHist", {
        'symbol': symbol, 'period': '1h', 'limit': 30
    })
    if data:
        df_oi = pd.DataFrame(data)
        df_oi['timestamp'] = pd.to_datetime(df_oi['timestamp'], unit='ms', utc=True)
        df_oi['oi_value_usd'] = pd.to_numeric(df_oi['sumOpenInterestValue'], errors='coerce')
        frames['oi'] = df_oi[['timestamp', 'oi_value_usd']].set_index('timestamp')

    # 2. Taker buy/sell ratio (30 × 1h)
    data = _fetch_json(f"{FAPI_DATA}/takerlongshortRatio", {
        'symbol': symbol, 'period': '1h', 'limit': 30
    })
    if data:
        df_tk = pd.DataFrame(data)
        df_tk['timestamp'] = pd.to_datetime(df_tk['timestamp'], unit='ms', utc=True)
        df_tk['taker_buy_sell_ratio'] = pd.to_numeric(df_tk['buySellRatio'], errors='coerce')
        frames['taker'] = df_tk[['timestamp', 'taker_buy_sell_ratio']].set_index('timestamp')

    # 3. Top trader L/S account ratio (30 × 1h)
    data = _fetch_json(f"{FAPI_DATA}/topLongShortAccountRatio", {
        'symbol': symbol, 'period': '1h', 'limit': 30
    })
    if data:
        df_ls = pd.DataFrame(data)
        df_ls['timestamp'] = pd.to_datetime(df_ls['timestamp'], unit='ms', utc=True)
        df_ls['top_ls_ratio'] = pd.to_numeric(df_ls['longShortRatio'], errors='coerce')
        df_ls['top_long_pct'] = pd.to_numeric(df_ls['longAccount'], errors='coerce')
        frames['top_ls'] = df_ls[['timestamp', 'top_ls_ratio', 'top_long_pct']].set_index('timestamp')

    # 4. Global L/S ratio (30 × 1h)
    data = _fetch_json(f"{FAPI_DATA}/globalLongShortAccountRatio", {
        'symbol': symbol, 'period': '1h', 'limit': 30
    })
    if data:
        df_gl = pd.DataFrame(data)
        df_gl['timestamp'] = pd.to_datetime(df_gl['timestamp'], unit='ms', utc=True)
        df_gl['global_ls_ratio'] = pd.to_numeric(df_gl['longShortRatio'], errors='coerce')
        df_gl['global_long_pct'] = pd.to_numeric(df_gl['longAccount'], errors='coerce')
        frames['global_ls'] = df_gl[['timestamp', 'global_ls_ratio', 'global_long_pct']].set_index('timestamp')

    if not frames:
        return None

    # Combine all frames on timestamp (outer join → fill gaps)
    combined = None
    for key, frame in frames.items():
        if combined is None:
            combined = frame
        else:
            combined = combined.join(frame, how='outer')

    combined = combined.reset_index()
    combined['symbol'] = _to_our_symbol(symbol)

    return combined


def fetch_realtime_metrics(symbols: list = None, max_workers: int = 8) -> pd.DataFrame:
    """Fetch real-time derivatives metrics for all symbols in parallel.

    Returns DataFrame with same schema as binance_futures_metrics.parquet.
    """
    if symbols is None:
        symbols = SYMBOLS

    all_dfs = []
    errors = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_symbol_data, sym): sym for sym in symbols}
        for future in as_completed(futures):
            sym = futures[future]
            try:
                result = future.result()
                if result is not None and len(result) > 0:
                    all_dfs.append(result)
            except Exception:
                errors += 1

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)

    # Ensure column order matches parquet schema
    expected_cols = ['timestamp', 'symbol', 'oi_value_usd', 'top_ls_ratio',
                     'top_long_pct', 'global_ls_ratio', 'global_long_pct',
                     'taker_buy_sell_ratio']
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[expected_cols].sort_values(['symbol', 'timestamp'])
    return df


def patch_metrics_realtime(project_root: str, verbose: bool = True) -> int:
    """Fetch real-time data and append to binance_futures_metrics.parquet.

    Only adds rows with timestamps newer than what's already in the parquet.
    Returns the number of new rows added.
    """
    sent_dir = os.path.join(project_root, 'data', 'sentiment')
    metrics_path = os.path.join(sent_dir, 'binance_futures_metrics.parquet')

    # Load existing data
    if os.path.exists(metrics_path):
        existing = pd.read_parquet(metrics_path)
        existing['timestamp'] = pd.to_datetime(existing['timestamp'], utc=True)
        max_ts = existing['timestamp'].max()
    else:
        existing = pd.DataFrame()
        max_ts = pd.Timestamp('2020-01-01', tz='UTC')

    if verbose:
        print(f"   🔄 Fetching real-time derivatives (last 30h, 50 symbols)...")
        stale_hours = (pd.Timestamp.now(tz='UTC') - max_ts).total_seconds() / 3600
        print(f"      Parquet last ts: {max_ts} ({stale_hours:.0f}h stale)")

    # Fetch fresh data
    fresh = fetch_realtime_metrics()
    if fresh.empty:
        if verbose:
            print("      ⚠️  No real-time data fetched")
        return 0

    fresh['timestamp'] = pd.to_datetime(fresh['timestamp'], utc=True)

    # Only keep rows newer than existing
    new_rows = fresh[fresh['timestamp'] > max_ts]
    if new_rows.empty:
        if verbose:
            print(f"      ✅ Already up to date (fresh max: {fresh['timestamp'].max()})")
        return 0

    # Append and save
    if len(existing) > 0:
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows

    # Deduplicate (in case of overlapping timestamps)
    combined = combined.drop_duplicates(['timestamp', 'symbol'], keep='last')
    combined = combined.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

    combined.to_parquet(metrics_path, index=False)

    n_new = len(new_rows)
    fresh_max = new_rows['timestamp'].max()
    n_syms = new_rows['symbol'].nunique()
    if verbose:
        print(f"      ✅ Added {n_new} rows ({n_syms} symbols) up to {fresh_max}")

    return n_new


# ── CLI entrypoint ──────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="Fetch real-time Binance derivatives data")
    ap.add_argument('--root', type=str, default=None,
                    help="Project root (default: auto-detect)")
    ap.add_argument('--dry-run', action='store_true',
                    help="Fetch and print but don't save to parquet")
    args = ap.parse_args()

    root = args.root or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')

    if args.dry_run:
        print("Fetching real-time derivatives (dry run)...")
        df = fetch_realtime_metrics()
        print(f"\nFetched: {len(df)} rows, {df['symbol'].nunique()} symbols")
        print(f"Time range: {df['timestamp'].min()} → {df['timestamp'].max()}")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nSample (BTC):")
        btc = df[df['symbol'] == 'BTC/USDT'].tail(5)
        print(btc.to_string(index=False))
        print(f"\nOI non-null: {df['oi_value_usd'].notna().sum()}/{len(df)}")
        print(f"Taker non-null: {df['taker_buy_sell_ratio'].notna().sum()}/{len(df)}")
    else:
        n = patch_metrics_realtime(root)
        print(f"\nDone. {n} new rows added.")
