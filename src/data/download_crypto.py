"""
Download historical 1h OHLCV data from Binance via CCXT.
Binance public API — no API key needed for historical data.
"""

import ccxt
import pandas as pd
import os
import time
import urllib3
from datetime import datetime, timezone
from tqdm import tqdm

urllib3.disable_warnings()

# === CONFIG ===
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
TIMEFRAME = '1h'
SINCE = '2017-01-01T00:00:00Z'  # Start date — go as far back as possible
LIMIT = 1000  # Binance max per request

# Top coins by liquidity (USDT pairs on Binance)
# Many coins only have data from 2019-2023 — that's fine, download will get what's available
SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
    'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'DOT/USDT', 'LINK/USDT',
    'MATIC/USDT', 'UNI/USDT', 'ATOM/USDT', 'LTC/USDT', 'ETC/USDT',
    'FIL/USDT', 'APT/USDT', 'ARB/USDT', 'OP/USDT', 'NEAR/USDT',
    'AAVE/USDT', 'MKR/USDT', 'GRT/USDT', 'INJ/USDT', 'FTM/USDT',
    'ALGO/USDT', 'SAND/USDT', 'MANA/USDT', 'AXS/USDT', 'THETA/USDT',
    'RUNE/USDT', 'EGLD/USDT', 'XTZ/USDT', 'FLOW/USDT', 'CHZ/USDT',
    'CRV/USDT', 'LDO/USDT', 'SNX/USDT', 'COMP/USDT', 'YFI/USDT',
    'SUSHI/USDT', 'ENJ/USDT', 'BAT/USDT', 'ZIL/USDT', 'ONE/USDT',
    'IOTA/USDT', 'ICX/USDT', 'ENS/USDT', 'IMX/USDT', 'GALA/USDT',
]


def fetch_all_ohlcv(exchange, symbol: str, timeframe: str, since_ts: int) -> pd.DataFrame:
    """Fetch all OHLCV data from since_ts to now, paginating through API limits."""
    all_data = []
    current_since = since_ts

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=LIMIT)
        except ccxt.BadSymbol:
            print(f"  ⚠ Symbol {symbol} not found on Binance, skipping")
            return pd.DataFrame()
        except Exception as e:
            print(f"  ⚠ Error fetching {symbol}: {e}, retrying in 5s...")
            time.sleep(5)
            continue

        if not ohlcv:
            break

        all_data.extend(ohlcv)

        # Move to next page
        last_ts = ohlcv[-1][0]
        if last_ts == current_since:
            break  # No new data
        current_since = last_ts + 1  # +1ms to avoid duplicates

        # Rate limiting (Binance: 1200 req/min)
        time.sleep(0.1)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    return df


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Initialize Binance (public API, no key needed)
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'},
    })
    exchange.session.verify = False  # SSL workaround

    since_ts = exchange.parse8601(SINCE)
    total_rows = 0
    downloaded = 0

    print(f"📊 Downloading {TIMEFRAME} OHLCV data for {len(SYMBOLS)} symbols")
    print(f"📅 From: {SINCE}")
    print(f"📁 Saving to: {os.path.abspath(DATA_DIR)}\n")

    for symbol in tqdm(SYMBOLS, desc="Downloading"):
        safe_name = symbol.replace('/', '_')
        filepath = os.path.join(DATA_DIR, f'{safe_name}_{TIMEFRAME}.parquet')

        # If file exists, check if we need to extend backward
        if os.path.exists(filepath):
            existing = pd.read_parquet(filepath)
            existing_start = existing['timestamp'].min()
            target_start = pd.Timestamp(SINCE, tz='UTC')
            if existing_start <= target_start + pd.Timedelta(hours=24):
                # Already have data from target start date, skip
                print(f"  ✓ {symbol}: already complete ({len(existing)} rows, from {existing_start})")
                total_rows += len(existing)
                downloaded += 1
                continue
            else:
                # Need to fetch older data and merge
                print(f"  ↻ {symbol}: extending backward ({existing_start} → {SINCE})")
                old_df = fetch_all_ohlcv(exchange, symbol, TIMEFRAME, since_ts)
                if not old_df.empty:
                    # Merge old and existing, dedup
                    combined = pd.concat([old_df, existing], ignore_index=True)
                    combined = combined.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
                    combined.to_parquet(filepath, index=False)
                    total_rows += len(combined)
                    downloaded += 1
                    print(f"  ✓ {symbol}: extended to {len(combined)} rows ({combined['timestamp'].min()} → {combined['timestamp'].max()})")
                    continue
                else:
                    total_rows += len(existing)
                    downloaded += 1
                    continue

        df = fetch_all_ohlcv(exchange, symbol, TIMEFRAME, since_ts)

        if df.empty:
            print(f"  ✗ {symbol}: no data")
            continue

        df.to_parquet(filepath, index=False)
        total_rows += len(df)
        downloaded += 1
        print(f"  ✓ {symbol}: {len(df)} rows ({df['timestamp'].min()} → {df['timestamp'].max()})")

    print(f"\n✅ Done! Downloaded {downloaded}/{len(SYMBOLS)} symbols, {total_rows:,} total rows")
    print(f"📁 Data saved to: {os.path.abspath(DATA_DIR)}")


if __name__ == '__main__':
    main()
