#!/usr/bin/env python3
"""Download macro / cross-market data for regime features.

Sources (via Alpha Vantage free API):
  - VIX — equity implied volatility / fear gauge
  - DXY — US Dollar Index (via UUP ETF proxy)
  - SPX — S&P 500 (via SPY ETF)
  - Gold — Gold spot price
  - 10Y Yield — US 10-Year Treasury yield

Alpha Vantage: free key, 25 req/day (standard), 75 req/min (premium).
  Get key: https://www.alphavantage.co/support/#api-key

Daily frequency — forward-filled to hourly in pipeline.

Output: data/sentiment/macro_daily.parquet
Columns: date, vix_close, vix_high, dxy_close, spx_close, gold_close, yield_10y_close

Incremental: loads existing, appends new dates, dedup.
"""

import os
import sys
import time
import argparse
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'sentiment')
OUTPUT_FILE = 'macro_daily.parquet'

DEFAULT_START = '2020-01-01'

# ── Alpha Vantage endpoints ──────────────────────────────────────

AV_BASE = 'https://www.alphavantage.co/query'

# Each ticker: AV function + params + column mapping
AV_TICKERS = {
    'vix': {
        'desc': 'VIX (CBOE Volatility Index)',
        'function': 'TIME_SERIES_DAILY',
        'symbol': 'VIX',       # CBOE VIX on AV
        'outputsize': 'full',
        'cols': {'4. close': 'vix_close', '2. high': 'vix_high'},
    },
    'spx': {
        'desc': 'S&P 500 (SPY ETF)',
        'function': 'TIME_SERIES_DAILY',
        'symbol': 'SPY',       # SPY ETF as S&P proxy
        'outputsize': 'full',
        'cols': {'4. close': 'spx_close'},
    },
    'dxy': {
        'desc': 'US Dollar Index (UUP ETF)',
        'function': 'TIME_SERIES_DAILY',
        'symbol': 'UUP',       # Invesco DB USD Index Bullish Fund
        'outputsize': 'full',
        'cols': {'4. close': 'dxy_close'},
    },
    'gold': {
        'desc': 'Gold spot price',
        'function': 'GOLD_SILVER_HISTORY',
        'symbol': 'GOLD',
        'interval': 'daily',
        'cols': {'close': 'gold_close'},
    },
    'yield10y': {
        'desc': '10Y US Treasury Yield',
        'function': 'TREASURY_YIELD',
        'interval': 'daily',
        'maturity': '10year',
        'cols': {'value': 'yield_10y_close'},
    },
}


def get_api_key(key_arg: str = None) -> str:
    """Get Alpha Vantage API key from arg, env, or .env file."""
    key = key_arg or os.environ.get('ALPHAVANTAGE_API_KEY') or os.environ.get('AV_API_KEY')
    if key:
        return key

    # Try .env file in project root
    env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith('ALPHAVANTAGE_API_KEY=') or line.startswith('AV_API_KEY='):
                    return line.split('=', 1)[1].strip().strip('"').strip("'")

    print("   ❌ No Alpha Vantage API key found!")
    print("      Get a FREE key: https://www.alphavantage.co/support/#api-key")
    print("      Then either:")
    print("        --api-key YOUR_KEY")
    print("        export ALPHAVANTAGE_API_KEY=YOUR_KEY")
    print("        echo 'ALPHAVANTAGE_API_KEY=YOUR_KEY' >> .env")
    sys.exit(1)


def download_av_time_series(api_key: str, ticker_cfg: dict, start_date: str) -> pd.DataFrame:
    """Download daily data from Alpha Vantage."""
    params = {
        'function': ticker_cfg['function'],
        'apikey': api_key,
    }

    # Add optional params
    if 'symbol' in ticker_cfg:
        params['symbol'] = ticker_cfg['symbol']
    if 'outputsize' in ticker_cfg:
        params['outputsize'] = ticker_cfg['outputsize']
    if 'interval' in ticker_cfg:
        params['interval'] = ticker_cfg['interval']
    if 'maturity' in ticker_cfg:
        params['maturity'] = ticker_cfg['maturity']

    for attempt in range(3):
        try:
            resp = requests.get(AV_BASE, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            # Check for rate limit / error messages
            if 'Note' in data or 'Information' in data:
                msg = data.get('Note', data.get('Information', ''))
                if 'call frequency' in msg.lower() or 'rate limit' in msg.lower():
                    wait = 65
                    print(f"\n      Rate limited, waiting {wait}s...", end=' ')
                    time.sleep(wait)
                    continue
                print(f"\n      ⚠️ API note: {msg[:80]}...")

            if 'Error Message' in data:
                print(f"\n      ❌ API error: {data['Error Message'][:80]}")
                return pd.DataFrame()

            # Parse based on function type
            func = ticker_cfg['function']

            if func == 'TIME_SERIES_DAILY':
                ts_key = 'Time Series (Daily)'
                if ts_key not in data:
                    print(f"\n      ❌ No '{ts_key}' in response. Keys: {list(data.keys())[:5]}")
                    return pd.DataFrame()
                ts = data[ts_key]
                df = pd.DataFrame.from_dict(ts, orient='index')
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            elif func == 'TREASURY_YIELD':
                if 'data' not in data:
                    print(f"\n      ❌ No 'data' in response. Keys: {list(data.keys())[:5]}")
                    return pd.DataFrame()
                records = data['data']
                df = pd.DataFrame(records)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
                df['value'] = pd.to_numeric(df['value'], errors='coerce')

            elif func == 'GOLD_SILVER_HISTORY':
                if 'data' not in data:
                    print(f"\n      ❌ No 'data' in response. Keys: {list(data.keys())[:5]}")
                    return pd.DataFrame()
                records = data['data']
                df = pd.DataFrame(records)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
                df['close'] = pd.to_numeric(df['close'], errors='coerce')

            else:
                print(f"\n      ❌ Unknown function: {func}")
                return pd.DataFrame()

            # Filter by start date
            start_ts = pd.Timestamp(start_date)
            df = df[df.index >= start_ts]

            # Rename columns
            col_map = ticker_cfg['cols']
            rename = {}
            for src, dst in col_map.items():
                if src in df.columns:
                    rename[src] = dst
            df = df.rename(columns=rename)

            # Keep only mapped columns
            keep = [c for c in col_map.values() if c in df.columns]
            if not keep:
                print(f"\n      ❌ No matching columns. Available: {list(df.columns)[:10]}")
                return pd.DataFrame()

            df = df[keep]
            df.index.name = 'date'
            return df.reset_index()

        except Exception as e:
            if attempt < 2:
                wait = 10 * (attempt + 1)
                print(f"\n      Retry {attempt+1}/3: {e}, waiting {wait}s...", end=' ')
                time.sleep(wait)
            else:
                print(f"\n      ❌ Failed after 3 attempts: {e}")

    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description='Download macro/cross-market data (Alpha Vantage)')
    parser.add_argument('--start', type=str, default=None,
                       help=f'Start date (default: {DEFAULT_START} or resume)')
    parser.add_argument('--full', action='store_true',
                       help='Force full re-download')
    parser.add_argument('--api-key', type=str, default=None,
                       help='Alpha Vantage API key (or set ALPHAVANTAGE_API_KEY env)')
    args = parser.parse_args()

    print("=" * 60)
    print("  Macro / Cross-Market Data Downloader (Alpha Vantage)")
    print("=" * 60)

    api_key = get_api_key(args.api_key)
    print(f"   API key: {api_key[:4]}...{api_key[-4:]}")

    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, OUTPUT_FILE)

    # Determine start date
    if args.full or args.start:
        start_date = args.start or DEFAULT_START
        print(f"   Start: {start_date} (manual)")
    elif os.path.exists(path):
        existing = pd.read_parquet(path)
        last_date = existing['date'].max()
        start_date = (pd.Timestamp(last_date) - timedelta(days=5)).strftime('%Y-%m-%d')
        print(f"   Resuming from: {start_date} (5d overlap for corrections)")
    else:
        start_date = DEFAULT_START
        print(f"   Start: {DEFAULT_START} (first run)")

    end_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    print(f"   End:   {end_date}")
    print(f"\n   ⚠️  Free tier: 25 req/day. We need {len(AV_TICKERS)} requests.")
    print()

    # Download each ticker
    all_data = {}
    for i, (prefix, cfg) in enumerate(AV_TICKERS.items()):
        if i > 0:
            time.sleep(13)  # ~5 req/min on free tier = 12s between calls
        desc = cfg['desc']
        print(f"   📊 [{i+1}/{len(AV_TICKERS)}] {desc}...", end=' ')

        df = download_av_time_series(api_key, cfg, start_date)
        if df.empty:
            print("❌ no data")
            continue

        all_data[prefix] = df
        cols = [c for c in df.columns if c != 'date']
        print(f"✅ {len(df):,} days ({df['date'].iloc[0].strftime('%Y-%m-%d')} → "
              f"{df['date'].iloc[-1].strftime('%Y-%m-%d')}) cols={cols}")

    if not all_data:
        print("\n   ❌ No data downloaded!")
        return

    # Merge all tickers on date
    merged = None
    for prefix, df in all_data.items():
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on='date', how='outer')

    # Sort and forward-fill missing days (weekends/holidays)
    merged = merged.sort_values('date').reset_index(drop=True)
    merged = merged.ffill()

    # Ensure date is date type
    merged['date'] = pd.to_datetime(merged['date']).dt.date

    # Merge with existing
    if os.path.exists(path) and not args.full:
        existing = pd.read_parquet(path)
        existing['date'] = pd.to_datetime(existing['date']).dt.date
        existing = existing.set_index('date')
        merged_idx = merged.set_index('date')
        combined = merged_idx.combine_first(existing).reset_index()
        combined = combined.sort_values('date').reset_index(drop=True)
    else:
        combined = merged.sort_values('date').reset_index(drop=True)

    # Save
    combined.to_parquet(path, index=False)
    size_kb = os.path.getsize(path) / 1024
    print(f"\n   💾 Saved: {OUTPUT_FILE}")
    print(f"      Total: {len(combined):,} days ({size_kb:.0f} KB)")
    print(f"      Columns: {list(combined.columns)}")
    print(f"      Date range: {combined['date'].iloc[0]} → {combined['date'].iloc[-1]}")

    # Check for NaN columns
    nan_pct = combined.drop(columns=['date']).isna().mean()
    bad = nan_pct[nan_pct > 0.1]
    if len(bad) > 0:
        print(f"\n   ⚠️ High NaN columns (>10%):")
        for col, pct in bad.items():
            print(f"      {col}: {pct:.1%}")

    print(f"\n{'=' * 60}")


if __name__ == '__main__':
    main()
