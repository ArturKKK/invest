#!/usr/bin/env python3
"""
Download derivatives data from Binance Futures (public, no API key needed).

Two data sources combined:
  A) data.binance.vision — bulk CSV zips, 5min granularity, history from Dec 2021
     Contains: OI, top-trader L/S, global L/S, taker buy/sell ratio per symbol
  B) fapi.binance.com/fapi/v1/fundingRate — full funding rate history from Jan 2020

Strategy:
  1. Download daily ZIP files from data.binance.vision for each symbol × each date
  2. Resample 5min → 1h (OHLC for OI, mean for ratios)
  3. Fetch Binance funding rates (8h frequency, since 2020)
  4. Save everything to data/sentiment/binance_futures_metrics.parquet
  5. Runs incrementally: skips dates that already exist in the parquet

Output columns (per row = 1h × symbol):
  - timestamp, symbol
  - oi_value_usd (sum open interest value)
  - top_ls_ratio, top_long_pct (top trader account long/short)
  - global_ls_ratio, global_long_pct (all accounts)
  - taker_buy_sell_ratio (taker volume ratio)
  - funding_rate_binance (8h funding, forward-filled to 1h)

Usage:
  python src/data/download_binance_futures.py                    # full history
  python src/data/download_binance_futures.py --start 2023-01-01 # from date
  python src/data/download_binance_futures.py --symbol BTCUSDT   # single symbol
  python src/data/download_binance_futures.py --skip-funding     # skip funding rates

Runtime: ~2-3 hours for full history (50 symbols × 1500+ days), ~1 min for daily update.
"""

import os
import sys
import time
import io
import zipfile
import argparse
import warnings
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import requests

warnings.filterwarnings('ignore')
import urllib3
urllib3.disable_warnings()

# ── config ────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', 'data', 'sentiment')

VISION_BASE = "https://data.binance.vision/data/futures/um/daily/metrics"
FAPI_BASE = "https://fapi.binance.com"

OUTPUT_FILE = 'binance_futures_metrics.parquet'
FUNDING_FILE = 'binance_funding_rates.parquet'
PREMIUM_FILE = 'binance_premium_index.parquet'
LIQUIDATION_FILE = 'binance_liquidations.parquet'

# Earliest date on data.binance.vision for metrics
VISION_START = datetime(2021, 12, 1, tzinfo=timezone.utc)

# All 50 symbols (futures format: BTCUSDT)
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


def to_our_symbol(binance_sym: str) -> str:
    """BTCUSDT → BTC/USDT"""
    return binance_sym.replace('USDT', '/USDT')


# ── Part A: data.binance.vision bulk downloads ────────────────

def download_day_zip(symbol: str, date: datetime) -> pd.DataFrame | None:
    """Download one day's metrics zip for a symbol. Returns DataFrame or None."""
    date_str = date.strftime('%Y-%m-%d')
    url = f"{VISION_BASE}/{symbol}/{symbol}-metrics-{date_str}.zip"

    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=20, verify=False)
            if resp.status_code == 404:
                return None  # no data for this symbol/date (e.g. futures not yet listed)
            if resp.status_code == 200:
                zf = zipfile.ZipFile(io.BytesIO(resp.content))
                csv_name = zf.namelist()[0]
                df = pd.read_csv(io.BytesIO(zf.read(csv_name)))
                return df
            if resp.status_code == 429:
                time.sleep(5)
                continue
        except Exception:
            time.sleep(2 ** attempt)

    return None


def process_day_csv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Parse raw 5min CSV into 1h resampled metrics.

    Columns in raw CSV:
      create_time, symbol,
      sum_open_interest, sum_open_interest_value,
      count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio,
      count_long_short_ratio, sum_taker_long_short_vol_ratio
    """
    df['timestamp'] = pd.to_datetime(df['create_time'], utc=True)
    df = df.sort_values('timestamp')

    # Parse numeric columns (some may be strings)
    num_cols = {
        'sum_open_interest_value': 'oi_value_usd',
        'sum_toptrader_long_short_ratio': 'top_ls_ratio',
        'count_toptrader_long_short_ratio': 'top_long_pct',
        'count_long_short_ratio': 'global_ls_ratio',
        'sum_taker_long_short_vol_ratio': 'taker_buy_sell_ratio',
    }

    result = pd.DataFrame()
    result['timestamp'] = df['timestamp']

    for src, dst in num_cols.items():
        if src in df.columns:
            result[dst] = pd.to_numeric(df[src], errors='coerce')

    result = result.set_index('timestamp')

    # Resample 5min → 1h
    # OI: take last value in the hour (snapshot)
    # Ratios: take mean over the hour
    agg_rules = {}
    if 'oi_value_usd' in result.columns:
        agg_rules['oi_value_usd'] = 'last'
    for col in ['top_ls_ratio', 'top_long_pct', 'global_ls_ratio', 'taker_buy_sell_ratio']:
        if col in result.columns:
            agg_rules[col] = 'mean'

    hourly = result.resample('1h').agg(agg_rules).dropna(how='all')
    hourly = hourly.reset_index()
    hourly['symbol'] = to_our_symbol(symbol)

    # Derive top_long_pct from the count ratio
    # count_toptrader_long_short_ratio = longAccount / shortAccount count
    # top_long_pct = long_count / (long_count + short_count) = ratio / (1 + ratio)
    if 'top_long_pct' in hourly.columns:
        r = hourly['top_long_pct']
        hourly['top_long_pct'] = r / (1 + r)  # convert ratio to percentage

    # Similarly for global
    if 'global_ls_ratio' in hourly.columns:
        hourly['global_long_pct'] = hourly['global_ls_ratio'] / (1 + hourly['global_ls_ratio'])

    return hourly


def download_metrics_bulk(symbols: list, start_date: datetime, end_date: datetime,
                          existing_dates_per_symbol: dict = None,
                          max_workers: int = 8) -> pd.DataFrame:
    """Download all metric zips from data.binance.vision.

    Uses ThreadPoolExecutor for parallel downloads per-symbol.
    Skips dates already in existing_dates_per_symbol.
    """
    all_dates = []
    d = max(start_date, VISION_START)
    while d <= end_date:
        all_dates.append(d)
        d += timedelta(days=1)

    print(f"\n📊 Downloading metrics from data.binance.vision")
    print(f"   {len(symbols)} symbols × {len(all_dates)} days = "
          f"{len(symbols) * len(all_dates):,} potential downloads")

    if existing_dates_per_symbol:
        total_skip = sum(len(v) for v in existing_dates_per_symbol.values())
        print(f"   Skipping ~{total_skip:,} already-downloaded symbol-days")

    all_dfs = []
    total = len(symbols)
    errors = 0

    for sym_idx, sym in enumerate(symbols):
        # Determine which dates to fetch for this symbol
        existing = existing_dates_per_symbol.get(to_our_symbol(sym), set()) if existing_dates_per_symbol else set()
        dates_to_fetch = [d for d in all_dates if d.strftime('%Y-%m-%d') not in existing]

        if not dates_to_fetch:
            sys.stdout.write(f"\r   [{sym_idx+1}/{total}] {sym}: all {len(all_dates)} days cached")
            sys.stdout.flush()
            continue

        sym_dfs = []

        def _fetch(date):
            return date, download_day_zip(sym, date)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_fetch, d): d for d in dates_to_fetch}
            for future in as_completed(futures):
                try:
                    date, raw_df = future.result()
                    if raw_df is not None:
                        hourly = process_day_csv(raw_df, sym)
                        sym_dfs.append(hourly)
                except Exception:
                    errors += 1

        if sym_dfs:
            sym_combined = pd.concat(sym_dfs, ignore_index=True)
            all_dfs.append(sym_combined)
            n_rows = len(sym_combined)
        else:
            n_rows = 0

        sys.stdout.write(
            f"\r   [{sym_idx+1}/{total}] {sym}: {len(dates_to_fetch)} days → "
            f"{n_rows} hourly rows ({len(all_dates)-len(dates_to_fetch)} cached)   "
        )
        sys.stdout.flush()

    print(f"\n   Download complete. Errors: {errors}")

    if not all_dfs:
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    result = result.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    print(f"   ✅ {len(result):,} new rows, {result['symbol'].nunique()} symbols")
    return result


# ── Part B: Binance funding rates ─────────────────────────────

def download_funding_rates(symbols: list, start_date: datetime) -> pd.DataFrame:
    """Download funding rate history from fapi/v1/fundingRate.

    This endpoint supports startTime and has full history back to Jan 2020.
    Funding is every 8h; we keep the raw 8h frequency (pipeline will merge).
    """
    print(f"\n📊 Downloading Binance funding rates ({len(symbols)} symbols)...")
    start_ms = int(start_date.timestamp() * 1000)

    all_dfs = []
    for i, sym in enumerate(symbols):
        url = f"{FAPI_BASE}/fapi/v1/fundingRate"
        records = []
        current_start = start_ms

        while True:
            try:
                resp = requests.get(url, params={
                    'symbol': sym, 'startTime': current_start, 'limit': 1000
                }, timeout=15, verify=False)
                if resp.status_code != 200:
                    break
                data = resp.json()
                if not data:
                    break
                records.extend(data)
                # Continue from after the last funding time
                last_ts = max(int(r['fundingTime']) for r in data)
                if last_ts <= current_start:
                    break
                current_start = last_ts + 1
                time.sleep(0.1)
            except Exception:
                break

        if records:
            df = pd.DataFrame(records)
            df['timestamp'] = pd.to_datetime(df['fundingTime'].astype(int), unit='ms', utc=True)
            df['symbol'] = to_our_symbol(sym)
            df['funding_rate_binance'] = df['fundingRate'].astype(float)
            df = df[['timestamp', 'symbol', 'funding_rate_binance']]
            df = df.drop_duplicates(['timestamp', 'symbol'])
            all_dfs.append(df)

        sys.stdout.write(f"\r   [{i+1}/{len(symbols)}] {sym}: {len(records)} funding records   ")
        sys.stdout.flush()
        time.sleep(0.05)

    print()
    if not all_dfs:
        print("   ❌ No funding data")
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True).sort_values(['symbol', 'timestamp'])
    print(f"   ✅ {len(result):,} rows, {result['symbol'].nunique()} symbols")
    print(f"   Range: {result['timestamp'].min()} → {result['timestamp'].max()}")
    return result


# ── Part C: Binance Premium Index (basis = perp - spot) ───────

def download_premium_index(symbols: list, start_date: datetime) -> pd.DataFrame:
    """Download premium index klines from fapi/v1/premiumIndexKlines.

    Returns 1h candles with markPrice, indexPrice → basis_pct computed.
    Premium index = (markPrice - indexPrice) / indexPrice ≈ perp premium.
    Free, no API key needed. Limit 1500 candles/request (~62 days).
    """
    print(f"\n📊 Downloading Premium Index klines ({len(symbols)} symbols)...")
    start_ms = int(start_date.timestamp() * 1000)

    all_dfs = []
    for i, sym in enumerate(symbols):
        url = f"{FAPI_BASE}/fapi/v1/premiumIndexKlines"
        records = []
        current_start = start_ms

        while True:
            try:
                resp = requests.get(url, params={
                    'pair': sym.replace('USDT', ''),  # pair format
                    'contractType': 'PERPETUAL',
                    'interval': '1h',
                    'startTime': current_start,
                    'limit': 1500,
                }, timeout=15, verify=False)

                # Fall back to symbol-based endpoint if pair fails
                if resp.status_code != 200:
                    resp = requests.get(url, params={
                        'symbol': sym,
                        'interval': '1h',
                        'startTime': current_start,
                        'limit': 1500,
                    }, timeout=15, verify=False)

                if resp.status_code != 200:
                    break
                data = resp.json()
                if not data:
                    break
                records.extend(data)
                last_ts = int(data[-1][0])
                if last_ts <= current_start or len(data) < 2:
                    break
                current_start = last_ts + 1
                time.sleep(0.15)
            except Exception:
                break

        if records:
            # Kline format: [openTime, open, high, low, close, ?, closeTime, ...]
            # For premium index klines: values are the premium index itself
            df = pd.DataFrame(records)
            df['timestamp'] = pd.to_datetime(df[0].astype(int), unit='ms', utc=True)
            df['symbol'] = to_our_symbol(sym)
            # Premium index open/high/low/close — use close as representative
            df['premium_index'] = pd.to_numeric(df[4], errors='coerce')
            df = df[['timestamp', 'symbol', 'premium_index']].drop_duplicates(
                ['timestamp', 'symbol'])
            all_dfs.append(df)

        sys.stdout.write(f"\r   [{i+1}/{len(symbols)}] {sym}: {len(records)} klines   ")
        sys.stdout.flush()
        time.sleep(0.05)

    print()
    if not all_dfs:
        print("   ❌ No premium index data")
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True).sort_values(['symbol', 'timestamp'])
    result = result.drop_duplicates(['timestamp', 'symbol'], keep='last')
    print(f"   ✅ {len(result):,} rows, {result['symbol'].nunique()} symbols")
    print(f"   Range: {result['timestamp'].min()} → {result['timestamp'].max()}")
    return result


# ── Part D: Binance Liquidation snapshots ─────────────────────

def download_liquidation_snapshot(symbols: list, start_date: datetime) -> pd.DataFrame:
    """Download liquidation data from Binance data.binance.vision.

    Binance provides liquidation snapshot CSVs at data.binance.vision:
      /data/futures/um/daily/liquidationSnapshot/{SYMBOL}/{SYMBOL}-liquidationSnapshot-{date}.zip
    Each zip has rows: time, symbol, side, price, qty, ...
    We aggregate to 1h: long_liq_usd, short_liq_usd per symbol.
    """
    print(f"\n📊 Downloading liquidation snapshots ({len(symbols)} symbols)...")
    LIQ_BASE = "https://data.binance.vision/data/futures/um/daily/liquidationSnapshot"

    end_date = datetime.now(timezone.utc) - timedelta(days=1)
    all_dates = []
    d = max(start_date, VISION_START)
    while d <= end_date:
        all_dates.append(d)
        d += timedelta(days=1)

    # Check existing data to skip already-downloaded dates
    existing = get_existing_dates(LIQUIDATION_FILE)

    all_dfs = []
    errors = 0

    for sym_idx, sym in enumerate(symbols):
        sym_our = to_our_symbol(sym)
        existing_dates = existing.get(sym_our, set())
        dates_to_fetch = [d for d in all_dates if d.strftime('%Y-%m-%d') not in existing_dates]

        if not dates_to_fetch:
            sys.stdout.write(f"\r   [{sym_idx+1}/{len(symbols)}] {sym}: all cached")
            sys.stdout.flush()
            continue

        sym_dfs = []

        def _fetch_liq(date):
            date_str = date.strftime('%Y-%m-%d')
            url = f"{LIQ_BASE}/{sym}/{sym}-liquidationSnapshot-{date_str}.zip"
            for attempt in range(3):
                try:
                    resp = requests.get(url, timeout=20, verify=False)
                    if resp.status_code == 404:
                        return None
                    if resp.status_code == 200:
                        zf = zipfile.ZipFile(io.BytesIO(resp.content))
                        csv_name = zf.namelist()[0]
                        liq_df = pd.read_csv(io.BytesIO(zf.read(csv_name)))
                        return liq_df
                    if resp.status_code == 429:
                        time.sleep(5)
                except Exception:
                    time.sleep(2 ** attempt)
            return None

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_fetch_liq, d): d for d in dates_to_fetch}
            for future in as_completed(futures):
                try:
                    raw = future.result()
                    if raw is not None and len(raw) > 0:
                        # Parse liquidation CSV
                        # Columns vary but typically: time, symbol, side, order_type,
                        # time_in_force, original_quantity, price, average_price,
                        # order_status, last_fill_quantity
                        raw['timestamp'] = pd.to_datetime(
                            raw.iloc[:, 0], unit='ms', utc=True, errors='coerce')
                        raw['side'] = raw.iloc[:, 2].astype(str).str.upper()

                        # Calculate USD value
                        # CSV columns: time(0), symbol(1), side(2), order_type(3),
                        # time_in_force(4), original_quantity(5), price(6), average_price(7), ...
                        qty_col = raw.columns[5] if len(raw.columns) > 5 else None
                        price_col = raw.columns[6] if len(raw.columns) > 6 else raw.columns[5]
                        if qty_col and price_col:
                            raw['usd_value'] = (
                                pd.to_numeric(raw[qty_col], errors='coerce') *
                                pd.to_numeric(raw[price_col], errors='coerce')
                            )
                        else:
                            raw['usd_value'] = 0

                        raw['is_long_liq'] = raw['side'].isin(['SELL', 'S'])  # forced sell = long liquidated
                        raw = raw.set_index('timestamp')

                        # Aggregate to 1h
                        hourly_long = raw[raw['is_long_liq']].resample('1h')['usd_value'].sum()
                        hourly_short = raw[~raw['is_long_liq']].resample('1h')['usd_value'].sum()

                        hourly = pd.DataFrame({
                            'liq_long_usd': hourly_long,
                            'liq_short_usd': hourly_short,
                        }).fillna(0)
                        hourly['symbol'] = sym_our
                        hourly = hourly.reset_index()
                        sym_dfs.append(hourly)
                except Exception:
                    errors += 1

        if sym_dfs:
            all_dfs.append(pd.concat(sym_dfs, ignore_index=True))

        n_h = sum(len(d) for d in sym_dfs) if sym_dfs else 0
        sys.stdout.write(
            f"\r   [{sym_idx+1}/{len(symbols)}] {sym}: {len(dates_to_fetch)} days → "
            f"{n_h} hourly rows ({len(all_dates)-len(dates_to_fetch)} cached)   "
        )
        sys.stdout.flush()

    print(f"\n   Download complete. Errors: {errors}")

    if not all_dfs:
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    result = result.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    print(f"   ✅ {len(result):,} rows, {result['symbol'].nunique()} symbols")
    return result


# ── incremental save ──────────────────────────────────────────

def save_incremental(new_df: pd.DataFrame, filename: str):
    """Merge new data with existing parquet, dedup, save."""
    path = os.path.join(DATA_DIR, filename)
    key_cols = ['timestamp', 'symbol']

    if os.path.exists(path):
        existing = pd.read_parquet(path)
        existing['timestamp'] = pd.to_datetime(existing['timestamp'], utc=True)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(key_cols, keep='last')
        n_new = len(combined) - len(existing)
        print(f"   💾 {filename}: {len(existing):,} + {n_new:,} new = {len(combined):,}")
    else:
        combined = new_df
        print(f"   💾 {filename}: {len(combined):,} rows (new)")

    combined = combined.sort_values(key_cols).reset_index(drop=True)
    combined.to_parquet(path, index=False)
    return combined


def get_existing_dates(filename: str) -> dict[str, set]:
    """Get set of date strings per symbol already in the parquet."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return {}

    df = pd.read_parquet(path, columns=['timestamp', 'symbol'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['date_str'] = df['timestamp'].dt.strftime('%Y-%m-%d')

    result = {}
    for sym, group in df.groupby('symbol'):
        result[sym] = set(group['date_str'].unique())

    return result


# ── main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download Binance Futures derivatives data")
    parser.add_argument('--start', type=str, default='2021-12-01',
                        help="Start date YYYY-MM-DD (default: 2021-12-01, earliest available)")
    parser.add_argument('--symbol', type=str, default=None,
                        help="Single symbol (e.g. BTCUSDT)")
    parser.add_argument('--skip-funding', action='store_true',
                        help="Skip funding rate download")
    parser.add_argument('--skip-metrics', action='store_true',
                        help="Skip OI/LS/taker metrics download")
    parser.add_argument('--skip-premium', action='store_true',
                        help="Skip premium index (basis) download")
    parser.add_argument('--skip-liquidations', action='store_true',
                        help="Skip liquidation snapshot download")
    parser.add_argument('--workers', type=int, default=8,
                        help="Parallel download threads per symbol (default: 8)")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    symbols = [args.symbol] if args.symbol else SYMBOLS
    start_date = datetime.strptime(args.start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    end_date = datetime.now(timezone.utc) - timedelta(days=1)  # yesterday (today's zip not ready yet)

    print("=" * 70)
    print("  BINANCE FUTURES DATA DOWNLOADER (data.binance.vision + API)")
    print(f"  {len(symbols)} symbols")
    print(f"  Metrics: {start_date:%Y-%m-%d} → {end_date:%Y-%m-%d} ({(end_date-start_date).days} days)")
    print(f"  Funding: {'skip' if args.skip_funding else f'{start_date:%Y-%m-%d} → now'}")
    print(f"  Premium: {'skip' if args.skip_premium else f'{start_date:%Y-%m-%d} → now'}")
    print(f"  Liquidations: {'skip' if args.skip_liquidations else f'{start_date:%Y-%m-%d} → now'}")
    print("=" * 70)

    # ── 1. Metrics (OI, L/S, taker) from data.binance.vision ──
    if not args.skip_metrics:
        existing = get_existing_dates(OUTPUT_FILE)
        new_metrics = download_metrics_bulk(
            symbols, start_date, end_date,
            existing_dates_per_symbol=existing,
            max_workers=args.workers,
        )
        if len(new_metrics) > 0:
            save_incremental(new_metrics, OUTPUT_FILE)

    # ── 2. Funding rates from API ─────────────────────────────
    if not args.skip_funding:
        # Always start from 2020 for funding (full history available)
        funding_start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        # Check existing funding data to start from where we left off
        funding_path = os.path.join(DATA_DIR, FUNDING_FILE)
        if os.path.exists(funding_path):
            existing_funding = pd.read_parquet(funding_path)
            last_ts = pd.to_datetime(existing_funding['timestamp']).max()
            if pd.notna(last_ts):
                funding_start = last_ts.to_pydatetime().replace(tzinfo=timezone.utc)
                print(f"\n   Funding: resuming from {funding_start:%Y-%m-%d %H:%M}")

        funding = download_funding_rates(symbols, funding_start)
        if len(funding) > 0:
            save_incremental(funding, FUNDING_FILE)

    # ── 3. Premium index (basis) from API ─────────────────────
    if not args.skip_premium:
        premium_start = start_date
        premium_path = os.path.join(DATA_DIR, PREMIUM_FILE)
        if os.path.exists(premium_path):
            existing_premium = pd.read_parquet(premium_path)
            last_ts = pd.to_datetime(existing_premium['timestamp']).max()
            if pd.notna(last_ts):
                premium_start = last_ts.to_pydatetime().replace(tzinfo=timezone.utc)
                print(f"\n   Premium: resuming from {premium_start:%Y-%m-%d %H:%M}")

        premium = download_premium_index(symbols, premium_start)
        if len(premium) > 0:
            save_incremental(premium, PREMIUM_FILE)

    # ── 4. Liquidation snapshots from data.binance.vision ─────
    if not args.skip_liquidations:
        liquidations = download_liquidation_snapshot(symbols, start_date)
        if len(liquidations) > 0:
            save_incremental(liquidations, LIQUIDATION_FILE)

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  ✅ Download complete — {DATA_DIR}")
    print(f"{'='*70}")
    for f in sorted(os.listdir(DATA_DIR)):
        if f.startswith('binance_') and f.endswith('.parquet'):
            p = os.path.join(DATA_DIR, f)
            size_mb = os.path.getsize(p) / 1024 / 1024
            df_tmp = pd.read_parquet(p)
            nsym = df_tmp['symbol'].nunique() if 'symbol' in df_tmp.columns else 0
            ts_min = pd.to_datetime(df_tmp['timestamp']).min()
            ts_max = pd.to_datetime(df_tmp['timestamp']).max()
            print(f"   {f}")
            print(f"      {len(df_tmp):,} rows, {nsym} syms, "
                  f"{ts_min:%Y-%m-%d} → {ts_max:%Y-%m-%d} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
