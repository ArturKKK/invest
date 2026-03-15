#!/usr/bin/env python3
"""Download macro / cross-market data for regime features.

Sources:
  - VIX (^VIX) — equity implied volatility / fear gauge
  - DXY (DX-Y.NYB) — US Dollar Index
  - SPX (^GSPC) — S&P 500
  - Gold (GC=F) — Gold futures (safe haven)
  - 10Y Yield (^TNX) — US 10-Year Treasury yield

Uses Yahoo Finance chart API directly (no yfinance dependency).
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
import urllib3
import pandas as pd
import numpy as np

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'sentiment')
OUTPUT_FILE = 'macro_daily.parquet'

DEFAULT_START = '2020-01-01'

YAHOO_CHART_URL = 'https://query2.finance.yahoo.com/v8/finance/chart/{symbol}'

TICKERS = {
    '^VIX':      {'prefix': 'vix',      'desc': 'VIX (equity fear gauge)'},
    'DX-Y.NYB':  {'prefix': 'dxy',      'desc': 'US Dollar Index'},
    '^GSPC':     {'prefix': 'spx',      'desc': 'S&P 500'},
    'GC=F':      {'prefix': 'gold',     'desc': 'Gold Futures'},
    '^TNX':      {'prefix': 'yield10y', 'desc': '10Y Treasury Yield'},
}

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}


def get_yahoo_session():
    """Create authenticated Yahoo Finance session with cookie + crumb.

    Uses fc.yahoo.com trick: requesting this endpoint returns a 404 but sets
    the required consent cookie, which then allows fetching a valid crumb.
    """
    session = requests.Session()
    session.verify = False
    session.headers.update(HEADERS)

    # Step 1: Get consent cookie via fc.yahoo.com
    try:
        session.get('https://fc.yahoo.com', timeout=10, allow_redirects=True)
    except Exception:
        pass  # 404 is expected, we just need the cookie

    # Step 2: Get crumb
    for attempt in range(5):
        try:
            r = session.get('https://query2.finance.yahoo.com/v1/test/getcrumb', timeout=10)
            if r.status_code == 200 and 'Too Many' not in r.text:
                crumb = r.text
                print(f"   Session ready (crumb: {crumb[:8]}...)")
                return session, crumb
            elif r.status_code == 429:
                wait = 60 * (attempt + 1)  # 60, 120, 180, 240, 300
                print(f"   Rate limited getting crumb, waiting {wait}s...")
                time.sleep(wait)
        except Exception as e:
            print(f"   Crumb attempt {attempt+1} failed: {e}")
            time.sleep(10)

    print("   ⚠️ Could not get crumb, will try without it")
    return session, None


def download_ticker_raw(session, crumb: str, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download daily OHLCV from Yahoo Finance chart API."""
    period1 = int(datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())
    period2 = int(datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())

    params = {
        'period1': period1,
        'period2': period2,
        'interval': '1d',
        'includePrePost': 'false',
    }
    if crumb:
        params['crumb'] = crumb

    for attempt in range(5):
        try:
            resp = session.get(
                YAHOO_CHART_URL.format(symbol=symbol),
                params=params,
                timeout=30,
            )
            if resp.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f"\n      Rate limited, waiting {wait}s...", end=' ')
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()

            result = data['chart']['result']
            if not result:
                return pd.DataFrame()

            result = result[0]
            timestamps = result.get('timestamp', [])
            quote = result.get('indicators', {}).get('quote', [{}])[0]

            if not timestamps:
                return pd.DataFrame()

            df = pd.DataFrame({
                'date': pd.to_datetime(timestamps, unit='s', utc=True).normalize(),
                'Close': quote.get('close'),
                'High': quote.get('high'),
                'Low': quote.get('low'),
            })
            return df

        except Exception as e:
            if attempt < 4:
                wait = 5 * (attempt + 1)
                print(f"\n      Retry {attempt+1}/5: {e}, waiting {wait}s...", end=' ')
                time.sleep(wait)
            else:
                print(f"\n      ❌ Failed: {symbol} — {e}")
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description='Download macro/cross-market data')
    parser.add_argument('--start', type=str, default=None,
                       help=f'Start date (default: {DEFAULT_START} or resume)')
    parser.add_argument('--full', action='store_true',
                       help='Force full re-download')
    args = parser.parse_args()

    print("=" * 60)
    print("  Macro / Cross-Market Data Downloader")
    print("=" * 60)

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

    end_date = (datetime.now(timezone.utc) + timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"   End:   {end_date}")
    print()

    # Create authenticated session
    session, crumb = get_yahoo_session()
    print()

    # Download each ticker
    all_data = {}
    for i, (ticker, info) in enumerate(TICKERS.items()):
        if i > 0:
            time.sleep(3)  # Polite delay between tickers
        prefix = info['prefix']
        desc = info['desc']
        print(f"   📊 {desc} ({ticker})...", end=' ')

        df = download_ticker_raw(session, crumb, ticker, start_date, end_date)
        if df.empty:
            print("❌ no data")
            continue

        # Keep Close and High (for VIX intraday spike)
        rename_map = {'Close': f'{prefix}_close'}
        if 'High' in df.columns:
            rename_map['High'] = f'{prefix}_high'
        if 'Low' in df.columns:
            rename_map['Low'] = f'{prefix}_low'

        df = df.rename(columns=rename_map)
        keep_cols = ['date'] + [c for c in rename_map.values() if c in df.columns]
        df = df[keep_cols]

        all_data[prefix] = df
        print(f"✅ {len(df):,} days ({df['date'].iloc[0].strftime('%Y-%m-%d')} → "
              f"{df['date'].iloc[-1].strftime('%Y-%m-%d')})")

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
        combined = pd.concat([existing, merged], ignore_index=True)
        combined = combined.drop_duplicates('date', keep='last')
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
    print(f"\n{'=' * 60}")


if __name__ == '__main__':
    main()
