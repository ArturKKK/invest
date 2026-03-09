#!/usr/bin/env python3
"""
Download derivatives data from Binance Futures (public endpoints, no API key).

Data sources:
1. Open Interest history — hourly, paginated (500 per request, ~months back)
2. Taker Buy/Sell Volume — hourly, same pagination
3. Top Trader Long/Short Ratio (accounts) — hourly
4. Top Trader Long/Short Ratio (positions) — hourly

Saves to data/sentiment/ alongside existing OKX data.
Runs incrementally: loads existing parquet, fetches only new data.

Usage:
  python src/data/download_binance_futures.py
  python src/data/download_binance_futures.py --days 365  # fetch last 365 days
  python src/data/download_binance_futures.py --symbol BTCUSDT  # single symbol

Note: These are /futures/data/* endpoints (analytics), NOT /fapi/v1/* (trading).
      They work from most geos including Russia.
"""

import os
import sys
import time
import json
import argparse
import warnings
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np
import requests

warnings.filterwarnings('ignore')
import urllib3
urllib3.disable_warnings()

# ── config ────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', 'data', 'sentiment')

BASE_URL = "https://fapi.binance.com"

# All 50 symbols (futures use no slash: BTCUSDT)
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

# Map Binance symbol → our format (for merging with OHLCV data)
def to_our_symbol(binance_sym: str) -> str:
    """BTCUSDT → BTC/USDT"""
    return binance_sym.replace('USDT', '/USDT')

RATE_LIMIT_SLEEP = 0.12   # ~8 req/s (Binance limit ~10/s for public)
MAX_RETRIES = 3


# ── generic paginated fetcher ─────────────────────────────────
def fetch_paginated(endpoint: str, symbol: str, period: str = '1h',
                    limit: int = 500, start_ms: int = None,
                    end_ms: int = None) -> list[dict]:
    """Fetch paginated data from Binance Futures analytics endpoint.

    Paginates backward from end_ms (or now) to start_ms.
    Returns list of dicts with raw JSON data.
    """
    url = BASE_URL + endpoint
    all_records = []
    current_end = end_ms or int(datetime.now(timezone.utc).timestamp() * 1000)

    while True:
        params = {
            'symbol': symbol,
            'period': period,
            'limit': limit,
            'endTime': current_end,
        }
        if start_ms:
            params['startTime'] = start_ms

        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(url, params=params, timeout=20, verify=False)
                if resp.status_code == 429:
                    # Rate limited — back off
                    time.sleep(5)
                    continue
                if resp.status_code == 418:
                    # IP banned — long backoff
                    print(f"\n   ⚠️  IP temp-banned, sleeping 60s...")
                    time.sleep(60)
                    continue
                if resp.status_code != 200:
                    # Some symbols may not have futures (404/400)
                    return all_records
                data = resp.json()
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    return all_records
                time.sleep(2 ** attempt)
        else:
            break

        if not data:
            break

        all_records.extend(data)

        # Binance returns oldest→newest; paginate backward
        oldest_ts = min(int(r.get('timestamp', r.get('createTime', 0))) for r in data)
        if start_ms and oldest_ts <= start_ms:
            break
        if len(data) < limit:
            break  # no more data

        current_end = oldest_ts - 1
        time.sleep(RATE_LIMIT_SLEEP)

    return all_records


# ── 1. Open Interest History ──────────────────────────────────
def download_oi_history(symbols: list, start_ms: int, end_ms: int) -> pd.DataFrame | None:
    """
    Endpoint: /futures/data/openInterestHist
    Returns: symbol, sumOpenInterest, sumOpenInterestValue, timestamp
    Period: 1h
    """
    print(f"\n📊 Downloading Open Interest history ({len(symbols)} symbols)...")

    all_dfs = []
    for i, sym in enumerate(symbols):
        records = fetch_paginated(
            '/futures/data/openInterestHist',
            symbol=sym, period='1h', limit=500,
            start_ms=start_ms, end_ms=end_ms,
        )
        if records:
            df = pd.DataFrame(records)
            df['symbol'] = to_our_symbol(sym)
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms', utc=True)
            df['open_interest'] = df['sumOpenInterest'].astype(float)
            df['oi_value_usd'] = df['sumOpenInterestValue'].astype(float)
            df = df[['timestamp', 'symbol', 'open_interest', 'oi_value_usd']]
            df = df.drop_duplicates(['timestamp', 'symbol']).sort_values('timestamp')
            all_dfs.append(df)

        sys.stdout.write(f"\r   [{i+1}/{len(symbols)}] {sym}: {len(records)} records   ")
        sys.stdout.flush()
        time.sleep(RATE_LIMIT_SLEEP)

    print()
    if not all_dfs:
        print("   ❌ No OI data downloaded")
        return None

    result = pd.concat(all_dfs, ignore_index=True).sort_values(['symbol', 'timestamp'])
    print(f"   ✅ {len(result):,} rows, {result['symbol'].nunique()} symbols")
    print(f"   Range: {result['timestamp'].min()} → {result['timestamp'].max()}")
    return result


# ── 2. Taker Buy/Sell Volume ─────────────────────────────────
def download_taker_volume(symbols: list, start_ms: int, end_ms: int) -> pd.DataFrame | None:
    """
    Endpoint: /futures/data/takerlongshortRatio
    Returns: buySellRatio, sellVol, buyVol, timestamp
    Period: 1h

    buySellRatio = buyVol / sellVol
    We store both raw volumes + ratio.
    """
    print(f"\n📊 Downloading Taker Buy/Sell Volume ({len(symbols)} symbols)...")

    all_dfs = []
    for i, sym in enumerate(symbols):
        records = fetch_paginated(
            '/futures/data/takerlongshortRatio',
            symbol=sym, period='1h', limit=500,
            start_ms=start_ms, end_ms=end_ms,
        )
        if records:
            df = pd.DataFrame(records)
            df['symbol'] = to_our_symbol(sym)
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms', utc=True)
            df['taker_buy_sell_ratio'] = df['buySellRatio'].astype(float)
            df['taker_buy_vol'] = df['buyVol'].astype(float)
            df['taker_sell_vol'] = df['sellVol'].astype(float)
            df = df[['timestamp', 'symbol', 'taker_buy_sell_ratio',
                     'taker_buy_vol', 'taker_sell_vol']]
            df = df.drop_duplicates(['timestamp', 'symbol']).sort_values('timestamp')
            all_dfs.append(df)

        sys.stdout.write(f"\r   [{i+1}/{len(symbols)}] {sym}: {len(records)} records   ")
        sys.stdout.flush()
        time.sleep(RATE_LIMIT_SLEEP)

    print()
    if not all_dfs:
        print("   ❌ No taker volume data downloaded")
        return None

    result = pd.concat(all_dfs, ignore_index=True).sort_values(['symbol', 'timestamp'])
    print(f"   ✅ {len(result):,} rows, {result['symbol'].nunique()} symbols")
    print(f"   Range: {result['timestamp'].min()} → {result['timestamp'].max()}")
    return result


# ── 3. Top Trader Long/Short Ratio (accounts) ────────────────
def download_top_ls_account(symbols: list, start_ms: int, end_ms: int) -> pd.DataFrame | None:
    """
    Endpoint: /futures/data/topLongShortAccountRatio
    Returns: longShortRatio, longAccount, shortAccount, timestamp
    Period: 1h
    """
    print(f"\n📊 Downloading Top Trader L/S Ratio (accounts) ({len(symbols)} symbols)...")

    all_dfs = []
    for i, sym in enumerate(symbols):
        records = fetch_paginated(
            '/futures/data/topLongShortAccountRatio',
            symbol=sym, period='1h', limit=500,
            start_ms=start_ms, end_ms=end_ms,
        )
        if records:
            df = pd.DataFrame(records)
            df['symbol'] = to_our_symbol(sym)
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms', utc=True)
            df['top_ls_ratio'] = df['longShortRatio'].astype(float)
            df['top_long_pct'] = df['longAccount'].astype(float)
            df['top_short_pct'] = df['shortAccount'].astype(float)
            df = df[['timestamp', 'symbol', 'top_ls_ratio', 'top_long_pct', 'top_short_pct']]
            df = df.drop_duplicates(['timestamp', 'symbol']).sort_values('timestamp')
            all_dfs.append(df)

        sys.stdout.write(f"\r   [{i+1}/{len(symbols)}] {sym}: {len(records)} records   ")
        sys.stdout.flush()
        time.sleep(RATE_LIMIT_SLEEP)

    print()
    if not all_dfs:
        print("   ❌ No top trader L/S data downloaded")
        return None

    result = pd.concat(all_dfs, ignore_index=True).sort_values(['symbol', 'timestamp'])
    print(f"   ✅ {len(result):,} rows, {result['symbol'].nunique()} symbols")
    print(f"   Range: {result['timestamp'].min()} → {result['timestamp'].max()}")
    return result


# ── 4. Global Long/Short Ratio ───────────────────────────────
def download_global_ls(symbols: list, start_ms: int, end_ms: int) -> pd.DataFrame | None:
    """
    Endpoint: /futures/data/globalLongShortAccountRatio
    Returns: longShortRatio, longAccount, shortAccount, timestamp
    Period: 1h

    This is the GLOBAL (all traders) ratio, vs top-trader above.
    """
    print(f"\n📊 Downloading Global L/S Ratio ({len(symbols)} symbols)...")

    all_dfs = []
    for i, sym in enumerate(symbols):
        records = fetch_paginated(
            '/futures/data/globalLongShortAccountRatio',
            symbol=sym, period='1h', limit=500,
            start_ms=start_ms, end_ms=end_ms,
        )
        if records:
            df = pd.DataFrame(records)
            df['symbol'] = to_our_symbol(sym)
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms', utc=True)
            df['global_ls_ratio'] = df['longShortRatio'].astype(float)
            df['global_long_pct'] = df['longAccount'].astype(float)
            df['global_short_pct'] = df['shortAccount'].astype(float)
            df = df[['timestamp', 'symbol', 'global_ls_ratio',
                     'global_long_pct', 'global_short_pct']]
            df = df.drop_duplicates(['timestamp', 'symbol']).sort_values('timestamp')
            all_dfs.append(df)

        sys.stdout.write(f"\r   [{i+1}/{len(symbols)}] {sym}: {len(records)} records   ")
        sys.stdout.flush()
        time.sleep(RATE_LIMIT_SLEEP)

    print()
    if not all_dfs:
        print("   ❌ No global L/S data downloaded")
        return None

    result = pd.concat(all_dfs, ignore_index=True).sort_values(['symbol', 'timestamp'])
    print(f"   ✅ {len(result):,} rows, {result['symbol'].nunique()} symbols")
    print(f"   Range: {result['timestamp'].min()} → {result['timestamp'].max()}")
    return result


# ── incremental save ──────────────────────────────────────────
def save_incremental(new_df: pd.DataFrame, filename: str, key_cols: list):
    """Merge new data with existing parquet, dedup by key_cols, save."""
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        existing = pd.read_parquet(path)
        existing['timestamp'] = pd.to_datetime(existing['timestamp'], utc=True)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(key_cols, keep='last')
        n_new = len(combined) - len(existing)
        print(f"   💾 {filename}: {len(existing):,} existing + {n_new:,} new = {len(combined):,}")
    else:
        combined = new_df
        print(f"   💾 {filename}: {len(combined):,} rows (new file)")

    combined = combined.sort_values(key_cols).reset_index(drop=True)
    combined.to_parquet(path, index=False)
    return combined


# ── main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Download Binance Futures data")
    parser.add_argument('--days', type=int, default=180,
                        help="Days of history to fetch (default: 180)")
    parser.add_argument('--symbol', type=str, default=None,
                        help="Single symbol to fetch (e.g. BTCUSDT)")
    parser.add_argument('--skip-oi', action='store_true')
    parser.add_argument('--skip-taker', action='store_true')
    parser.add_argument('--skip-ls', action='store_true')
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    symbols = [args.symbol] if args.symbol else SYMBOLS
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=args.days)).timestamp() * 1000)

    print("=" * 70)
    print("  BINANCE FUTURES DATA DOWNLOADER")
    print(f"  {len(symbols)} symbols, last {args.days} days")
    print(f"  {datetime.utcfromtimestamp(start_ms/1000):%Y-%m-%d} → "
          f"{datetime.utcfromtimestamp(end_ms/1000):%Y-%m-%d}")
    print("=" * 70)

    # 1. Open Interest
    if not args.skip_oi:
        oi = download_oi_history(symbols, start_ms, end_ms)
        if oi is not None:
            save_incremental(oi, 'binance_open_interest.parquet',
                           ['timestamp', 'symbol'])

    # 2. Taker volume
    if not args.skip_taker:
        taker = download_taker_volume(symbols, start_ms, end_ms)
        if taker is not None:
            save_incremental(taker, 'binance_taker_volume.parquet',
                           ['timestamp', 'symbol'])

    # 3. Top trader L/S ratio
    if not args.skip_ls:
        top_ls = download_top_ls_account(symbols, start_ms, end_ms)
        if top_ls is not None:
            save_incremental(top_ls, 'binance_top_ls_ratio.parquet',
                           ['timestamp', 'symbol'])

    # 4. Global L/S ratio
    if not args.skip_ls:
        global_ls = download_global_ls(symbols, start_ms, end_ms)
        if global_ls is not None:
            save_incremental(global_ls, 'binance_global_ls_ratio.parquet',
                           ['timestamp', 'symbol'])

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  ✅ Download complete — {DATA_DIR}")
    print(f"{'='*70}")
    for f in sorted(os.listdir(DATA_DIR)):
        if f.startswith('binance_') and f.endswith('.parquet'):
            p = os.path.join(DATA_DIR, f)
            size = os.path.getsize(p) / 1024
            df_tmp = pd.read_parquet(p)
            nsym = df_tmp['symbol'].nunique() if 'symbol' in df_tmp.columns else 0
            ts_min = pd.to_datetime(df_tmp['timestamp']).min()
            ts_max = pd.to_datetime(df_tmp['timestamp']).max()
            print(f"   {f}: {len(df_tmp):,} rows, {nsym} symbols, "
                  f"{ts_min:%Y-%m-%d} → {ts_max:%Y-%m-%d} ({size:.0f} KB)")


if __name__ == '__main__':
    main()
