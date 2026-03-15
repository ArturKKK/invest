#!/usr/bin/env python3
"""Download Deribit DVOL (implied volatility index) for BTC and ETH.

Deribit public API — no authentication required.
Hourly OHLC candles of the DVOL index.

Output: data/sentiment/deribit_dvol.parquet
Columns: timestamp, currency, dvol_open, dvol_high, dvol_low, dvol_close

Incremental: reads existing parquet, resumes from last timestamp.
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime, timezone

import requests
import urllib3
import pandas as pd
import numpy as np

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'sentiment')
OUTPUT_FILE = 'deribit_dvol.parquet'

DERIBIT_URL = 'https://www.deribit.com/api/v2/public/get_volatility_index_data'

# DVOL available from ~April 2021
DEFAULT_START = '2021-04-01'

CURRENCIES = ['BTC', 'ETH']

# Deribit returns max 1000 records per request; uses continuation token
MAX_PER_REQUEST = 1000


def ts_to_ms(dt_str: str) -> int:
    """Convert date string to millisecond timestamp."""
    dt = datetime.strptime(dt_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def ms_to_dt(ms: int) -> datetime:
    """Convert millisecond timestamp to datetime."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def fetch_dvol(currency: str, start_ms: int, end_ms: int) -> list:
    """Fetch all DVOL data for a currency using backward pagination.

    Deribit API returns most recent 1000 records first, and continuation
    token points backward in time. Use continuation as end_timestamp for
    next page to walk backward through history.
    """
    all_records = []
    current_end = end_ms
    page = 0

    while current_end > start_ms:
        params = {
            'currency': currency,
            'start_timestamp': start_ms,
            'end_timestamp': current_end,
            'resolution': 3600,  # 1 hour
        }

        for attempt in range(5):
            try:
                resp = requests.get(DERIBIT_URL, params=params, timeout=30, verify=False)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                if attempt < 4:
                    wait = 2 ** attempt
                    print(f"      Retry {attempt+1}/5 ({e}), waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"      ❌ Failed after 5 attempts: {e}")
                    return all_records

        result = data.get('result', {})
        records = result.get('data', [])
        continuation = result.get('continuation')

        if records:
            all_records.extend(records)
            page += 1
            earliest = ms_to_dt(records[0][0]).strftime('%Y-%m-%d %H:%M')
            sys.stdout.write(f"\r   {currency}: {len(all_records):,} records "
                           f"(page {page}, earliest so far: {earliest})")
            sys.stdout.flush()

        # No more pages: fewer than MAX records or no continuation
        if len(records) < MAX_PER_REQUEST or not continuation:
            break

        # Walk backward: use continuation as new end_timestamp
        current_end = continuation
        time.sleep(0.2)  # Rate limit: be polite

    print()
    return all_records


def save_incremental(new_df: pd.DataFrame):
    """Merge new data with existing parquet, dedup, save."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, OUTPUT_FILE)

    if os.path.exists(path):
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(['timestamp', 'currency'], keep='last')
    else:
        combined = new_df

    combined = combined.sort_values(['timestamp', 'currency']).reset_index(drop=True)
    combined.to_parquet(path, index=False)
    return combined


def main():
    parser = argparse.ArgumentParser(description='Download Deribit DVOL data')
    parser.add_argument('--start', type=str, default=None,
                       help=f'Start date (default: {DEFAULT_START} or resume from last)')
    parser.add_argument('--full', action='store_true',
                       help='Force full re-download from DEFAULT_START')
    args = parser.parse_args()

    print("=" * 60)
    print("  Deribit DVOL Downloader")
    print("=" * 60)

    # Determine start date
    path = os.path.join(DATA_DIR, OUTPUT_FILE)
    if args.full or args.start:
        start_date = args.start or DEFAULT_START
        start_ms = ts_to_ms(start_date)
        print(f"   Start: {start_date} (manual)")
    elif os.path.exists(path):
        existing = pd.read_parquet(path)
        last_ts = existing['timestamp'].max()
        start_ms = int(last_ts.timestamp() * 1000)
        print(f"   Resuming from: {last_ts}")
    else:
        start_ms = ts_to_ms(DEFAULT_START)
        print(f"   Start: {DEFAULT_START} (first run)")

    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    print(f"   End:   {ms_to_dt(end_ms).strftime('%Y-%m-%d %H:%M')} UTC")
    print()

    all_dfs = []
    for currency in CURRENCIES:
        print(f"   📊 Downloading {currency} DVOL...")
        records = fetch_dvol(currency, start_ms, end_ms)

        if not records:
            print(f"      No new data for {currency}")
            continue

        df = pd.DataFrame(records, columns=['timestamp_ms', 'dvol_open', 'dvol_high',
                                             'dvol_low', 'dvol_close'])
        df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms', utc=True)
        df['currency'] = currency
        df = df.drop(columns=['timestamp_ms'])
        all_dfs.append(df)
        print(f"      ✅ {currency}: {len(df):,} records "
              f"({df['timestamp'].min().strftime('%Y-%m-%d')} → "
              f"{df['timestamp'].max().strftime('%Y-%m-%d')})")

    if all_dfs:
        combined_new = pd.concat(all_dfs, ignore_index=True)
        final = save_incremental(combined_new)
        size_mb = os.path.getsize(os.path.join(DATA_DIR, OUTPUT_FILE)) / 1024 / 1024
        print(f"\n   💾 Saved: {OUTPUT_FILE}")
        print(f"      Total: {len(final):,} records ({size_mb:.1f} MB)")
        print(f"      BTC: {len(final[final['currency']=='BTC']):,} | "
              f"ETH: {len(final[final['currency']=='ETH']):,}")
    else:
        print("\n   No new data to save.")

    print(f"\n{'=' * 60}")


if __name__ == '__main__':
    main()
