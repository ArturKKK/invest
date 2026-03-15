#!/usr/bin/env python3
"""Download macro / cross-market data for regime features.

Sources (via FRED API — Federal Reserve Economic Data):
  - VIX  (VIXCLS)          — CBOE Volatility Index, daily close
  - SPX  (SP500)            — S&P 500 index
  - DXY  (DTWEXBGS)         — Trade-Weighted USD Index (Broad, Goods & Services)
  - Gold (GOLDAMGBD228NLBM) — Gold fixing price, London Bullion Market (USD/troy oz)
  - 10Y  (DGS10)            — 10-Year Treasury Constant Maturity Rate

FRED API: completely free, 120 req/min, no daily limit.
  Get key: https://fred.stlouisfed.org/docs/api/api_key.html

Daily frequency — forward-filled to hourly in pipeline.

Output: data/sentiment/macro_daily.parquet
Columns: date, vix_close, spx_close, dxy_close, gold_close, yield_10y_close

Incremental: loads existing, appends new dates, dedup.
"""

import os
import sys
import argparse
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'sentiment')
OUTPUT_FILE = 'macro_daily.parquet'

DEFAULT_START = '2020-01-01'

# ── FRED series ──────────────────────────────────────────────────

FRED_BASE = 'https://api.stlouisfed.org/fred/series/observations'

FRED_SERIES = {
    'vix': {
        'series_id': 'VIXCLS',
        'desc': 'VIX (CBOE Volatility Index)',
        'col': 'vix_close',
    },
    'spx': {
        'series_id': 'SP500',
        'desc': 'S&P 500',
        'col': 'spx_close',
    },
    'dxy': {
        'series_id': 'DTWEXBGS',
        'desc': 'US Dollar Index (Trade-Weighted)',
        'col': 'dxy_close',
    },
    'gold': {
        'series_id': 'NASDAQQGLDI',
        'desc': 'Gold (NASDAQ Gold FLOWS Index)',
        'col': 'gold_close',
    },
    'yield10y': {
        'series_id': 'DGS10',
        'desc': '10-Year Treasury Yield',
        'col': 'yield_10y_close',
    },
    'hy_spread': {
        'series_id': 'BAMLH0A0HYM2',
        'desc': 'High Yield Credit Spread (ICE BofA)',
        'col': 'hy_spread',
    },
    'breakeven10y': {
        'series_id': 'T10YIE',
        'desc': '10Y Breakeven Inflation Rate',
        'col': 'breakeven_10y',
    },
    'yield_curve': {
        'series_id': 'T10Y2Y',
        'desc': 'Yield Curve (10Y minus 2Y)',
        'col': 'yield_curve_10y2y',
    },
    'fed_rate': {
        'series_id': 'DFF',
        'desc': 'Federal Funds Effective Rate',
        'col': 'fed_funds_rate',
    },
}


def get_api_key(key_arg: str = None) -> str:
    """Get FRED API key from arg, env, or .env file."""
    key = key_arg or os.environ.get('FRED_API_KEY')
    if key:
        return key

    # Try .env file in project root
    env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith('FRED_API_KEY='):
                    return line.split('=', 1)[1].strip().strip('"').strip("'")

    print("   ❌ No FRED API key found!")
    print("      Get a FREE key: https://fred.stlouisfed.org/docs/api/api_key.html")
    print("      Then either:")
    print("        --api-key YOUR_KEY")
    print("        export FRED_API_KEY=YOUR_KEY")
    print("        echo 'FRED_API_KEY=YOUR_KEY' >> .env")
    sys.exit(1)


def download_fred_series(api_key: str, series_id: str, col_name: str,
                         start_date: str, end_date: str) -> pd.DataFrame:
    """Download a single FRED series as DataFrame with columns [date, col_name]."""
    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': start_date,
        'observation_end': end_date,
        'sort_order': 'asc',
    }

    for attempt in range(3):
        try:
            resp = requests.get(FRED_BASE, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if 'error_code' in data:
                print(f"\n      ❌ FRED error: {data.get('error_message', 'unknown')[:80]}")
                return pd.DataFrame()

            observations = data.get('observations', [])
            if not observations:
                print(f"\n      ❌ No observations returned")
                return pd.DataFrame()

            df = pd.DataFrame(observations)
            df['date'] = pd.to_datetime(df['date'])
            # FRED uses '.' for missing values
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df[['date', 'value']].rename(columns={'value': col_name})
            df = df.dropna(subset=[col_name])
            df = df.sort_values('date').reset_index(drop=True)
            return df

        except Exception as e:
            if attempt < 2:
                import time
                wait = 5 * (attempt + 1)
                print(f"\n      Retry {attempt+1}/3: {e}, waiting {wait}s...", end=' ')
                time.sleep(wait)
            else:
                print(f"\n      ❌ Failed after 3 attempts: {e}")

    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description='Download macro/cross-market data (FRED API)')
    parser.add_argument('--start', type=str, default=None,
                       help=f'Start date (default: {DEFAULT_START} or resume)')
    parser.add_argument('--full', action='store_true',
                       help='Force full re-download')
    parser.add_argument('--api-key', type=str, default=None,
                       help='FRED API key (or set FRED_API_KEY env)')
    args = parser.parse_args()

    print("=" * 60)
    print("  Macro / Cross-Market Data Downloader (FRED)")
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
    print(f"\n   ℹ️  FRED API: free, 120 req/min, full history.")
    print()

    # Download each series
    all_data = {}
    for i, (prefix, cfg) in enumerate(FRED_SERIES.items()):
        series_id = cfg['series_id']
        desc = cfg['desc']
        col = cfg['col']
        print(f"   📊 [{i+1}/{len(FRED_SERIES)}] {desc} ({series_id})...", end=' ')

        df = download_fred_series(api_key, series_id, col, start_date, end_date)
        if df.empty:
            print("❌ no data")
            continue

        all_data[prefix] = df
        print(f"✅ {len(df):,} days ({df['date'].iloc[0].strftime('%Y-%m-%d')} → "
              f"{df['date'].iloc[-1].strftime('%Y-%m-%d')})")

    if not all_data:
        print("\n   ❌ No data downloaded!")
        return

    # Merge all series on date
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
