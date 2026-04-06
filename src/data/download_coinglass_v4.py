#!/usr/bin/env python3
"""
Download historical derivatives data from CoinGlass API V4.

Endpoints (5 per coin, 35 coins = 175 download tasks):
  1. Liquidations     — aggregated long/short liquidation USD across exchanges
  2. OI OHLC          — aggregated open interest OHLC across exchanges
  3. Taker Buy/Sell   — aggregated taker buy/sell volume USD
  4. Funding Rate     — OHLC funding rate per exchange-pair
  5. L/S Ratio        — top account long/short ratio per exchange-pair

Output: data/raw/coinglass/{endpoint}_{symbol}.parquet
Interval: 1d (gives 4+ years history on Hobbyist) or 12h (360 days)
History: paginated from 2022-01-01

Rate limit: Hobbyist = 30 req/min → 2.2s delay between requests.
Estimated runtime: ~5 min per endpoint × 35 coins = ~25 min total.

Usage:
  python src/data/download_coinglass_v4.py              # all endpoints, all symbols
  python src/data/download_coinglass_v4.py --only liq   # single endpoint
  python src/data/download_coinglass_v4.py --symbol BTC # single symbol test
"""

import os
import sys
import time
import argparse
import warnings
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# ── config ────────────────────────────────────────────────────

BASE_URL = "https://open-api-v4.coinglass.com"

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'coinglass')

# Hobbyist: 30 req/min → 2.2s delay (with margin)
RATE_LIMIT_DELAY = 2.2

# Interval: 1d for max history (4+ years), 12h for intraday (360 days)
INTERVAL = '1d'

# Start from 2022-01-01
DEFAULT_START_MS = int(datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

# Research universe (35 coins)
SYM_35 = [
    'BTC', 'ETH', 'SOL', 'BNB', 'XRP',
    'ADA', 'DOGE', 'AVAX', 'DOT', 'LINK',
    'MATIC', 'UNI', 'ATOM', 'LTC', 'NEAR',
    'FIL', 'APT', 'ARB', 'OP', 'AAVE',
    'INJ', 'FTM', 'ALGO', 'SAND', 'MANA',
    'AXS', 'THETA', 'RUNE', 'EGLD', 'XTZ',
    'FLOW', 'CHZ', 'CRV', 'LDO', 'SNX',
]

EXCHANGES = "Binance,OKX,Bybit"


# ── API client ────────────────────────────────────────────────

class CoinGlassV4:
    def __init__(self, api_key: str):
        self.session = requests.Session()
        self.session.headers.update({
            'CG-API-KEY': api_key,
            'Accept': 'application/json',
        })
        self.session.verify = False  # CG cert chain issue on some systems
        self._last_req = 0
        self.total_requests = 0

    def _rate_limit(self):
        elapsed = time.time() - self._last_req
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self._last_req = time.time()

    def get(self, endpoint: str, params: dict, retries: int = 3):
        url = f"{BASE_URL}{endpoint}"
        for attempt in range(retries):
            self._rate_limit()
            self.total_requests += 1
            try:
                resp = self.session.get(url, params=params, timeout=30)

                if resp.status_code == 429:
                    wait = min(60, 10 * (attempt + 1))
                    print(f"\n   ⚠ Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue

                if resp.status_code in (401, 403):
                    print(f"\n   ✗ Auth error {resp.status_code}: {resp.text[:200]}")
                    return None

                if resp.status_code != 200:
                    if attempt < retries - 1:
                        time.sleep(3 * (attempt + 1))
                        continue
                    print(f"\n   ✗ HTTP {resp.status_code}: {resp.text[:200]}")
                    return None

                body = resp.json()
                if body.get('code') == '0' and body.get('data') is not None:
                    return body['data']

                if attempt < retries - 1:
                    time.sleep(2)
                    continue
                print(f"\n   ✗ API error: {body.get('msg', body)}")
                return None

            except requests.exceptions.Timeout:
                if attempt < retries - 1:
                    time.sleep(3 * (attempt + 1))
                    continue
                return None
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(3)
                    continue
                print(f"\n   ✗ Exception: {e}")
                return None
        return None


# ── Download helpers ──────────────────────────────────────────

def paginate(api: CoinGlassV4, endpoint: str, params_fn, symbol: str,
             start_ms: int, end_ms: int) -> list:
    """Paginate through history in 1000-row chunks."""
    all_rows = []
    current_start = start_ms

    while current_start < end_ms:
        params = params_fn(symbol, current_start, end_ms)
        data = api.get(endpoint, params)

        if not data or not isinstance(data, list) or len(data) == 0:
            break

        all_rows.extend(data)

        # Advance past last timestamp
        last_time = max(row.get('time', 0) for row in data)
        if last_time <= current_start:
            break
        current_start = last_time + 1

        # If we got fewer than limit, we've reached the end
        if len(data) < 1000:
            break

    return all_rows


# ── Endpoint definitions ──────────────────────────────────────

ENDPOINTS = {
    'liq': {
        'name': 'Liquidations',
        'path': '/api/futures/liquidation/aggregated-history',
        'params': lambda sym, start, end: {
            'exchange_list': EXCHANGES,
            'symbol': sym,
            'interval': INTERVAL,
            'limit': 1000,
            'start_time': start,
            'end_time': end,
        },
        'columns': {
            'time': 'timestamp',
            'aggregated_long_liquidation_usd': 'liq_long_usd',
            'aggregated_short_liquidation_usd': 'liq_short_usd',
        },
    },
    'oi': {
        'name': 'Open Interest',
        'path': '/api/futures/open-interest/aggregated-history',
        'params': lambda sym, start, end: {
            'symbol': sym,
            'interval': INTERVAL,
            'limit': 1000,
            'start_time': start,
            'end_time': end,
            'unit': 'usd',
        },
        'columns': {
            'time': 'timestamp',
            'open': 'oi_open',
            'high': 'oi_high',
            'low': 'oi_low',
            'close': 'oi_close',
        },
    },
    'taker': {
        'name': 'Taker Buy/Sell',
        'path': '/api/futures/aggregated-taker-buy-sell-volume/history',
        'params': lambda sym, start, end: {
            'exchange_list': EXCHANGES,
            'symbol': sym,
            'interval': INTERVAL,
            'limit': 1000,
            'start_time': start,
            'end_time': end,
            'unit': 'usd',
        },
        'columns': {
            'time': 'timestamp',
            'aggregated_buy_volume_usd': 'taker_buy_usd',
            'aggregated_sell_volume_usd': 'taker_sell_usd',
        },
    },
    'funding': {
        'name': 'Funding Rate',
        'path': '/api/futures/funding-rate/history',
        'params': lambda sym, start, end: {
            'exchange': 'Binance',
            'symbol': f'{sym}USDT',  # pair format for this endpoint
            'interval': INTERVAL,
            'limit': 1000,
            'start_time': start,
            'end_time': end,
        },
        'columns': {
            'time': 'timestamp',
            'open': 'fr_open',
            'high': 'fr_high',
            'low': 'fr_low',
            'close': 'fr_close',
        },
    },
    'ls_ratio': {
        'name': 'L/S Ratio',
        'path': '/api/futures/top-long-short-account-ratio/history',
        'params': lambda sym, start, end: {
            'exchange': 'Binance',
            'symbol': f'{sym}USDT',  # pair format
            'interval': INTERVAL,
            'limit': 1000,
            'start_time': start,
            'end_time': end,
        },
        'columns': {
            'time': 'timestamp',
            'top_account_long_percent': 'ls_long_pct',
            'top_account_short_percent': 'ls_short_pct',
            'top_account_long_short_ratio': 'ls_ratio',
        },
    },
    'basis': {
        'name': 'Futures Basis',
        'path': '/api/futures/basis/history',
        'params': lambda sym, start, end: {
            'exchange': 'Binance',
            'symbol': f'{sym}USDT',
            'interval': INTERVAL,
            'limit': 1000,
            'start_time': start,
            'end_time': end,
        },
        'columns': {
            'time': 'timestamp',
            'open_basis': 'basis_open',
            'close_basis': 'basis_close',
            'open_change': 'basis_open_chg',
            'close_change': 'basis_close_chg',
        },
    },
    'pos_ratio': {
        'name': 'Top Position Ratio',
        'path': '/api/futures/top-long-short-position-ratio/history',
        'params': lambda sym, start, end: {
            'exchange': 'Binance',
            'symbol': f'{sym}USDT',
            'interval': INTERVAL,
            'limit': 1000,
            'start_time': start,
            'end_time': end,
        },
        'columns': {
            'time': 'timestamp',
            'top_position_long_percent': 'pos_long_pct',
            'top_position_short_percent': 'pos_short_pct',
            'top_position_long_short_ratio': 'pos_ls_ratio',
        },
    },
}


# ── Main download logic ──────────────────────────────────────

def download_endpoint(api: CoinGlassV4, ep_key: str, symbols: list,
                      start_ms: int, end_ms: int) -> pd.DataFrame:
    """Download one endpoint for all symbols."""
    ep = ENDPOINTS[ep_key]
    print(f"\n{'─'*60}")
    print(f"  📊 {ep['name'].upper()} — {len(symbols)} symbols, {INTERVAL} interval")
    print(f"{'─'*60}")

    all_dfs = []

    for i, sym in enumerate(symbols):
        rows = paginate(api, ep['path'], ep['params'], sym, start_ms, end_ms)

        if rows:
            df = pd.DataFrame(rows)
            # Rename columns
            rename_map = {}
            for src_col, dst_col in ep['columns'].items():
                if src_col in df.columns:
                    rename_map[src_col] = dst_col
            df = df.rename(columns=rename_map)

            # Keep only our columns + convert types
            keep = [c for c in ep['columns'].values() if c in df.columns]
            df = df[keep].copy()

            # Convert timestamp from ms → datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

            # Convert numeric columns
            for col in df.columns:
                if col != 'timestamp':
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            df['symbol'] = f"{sym}/USDT"
            all_dfs.append(df)

        n = len(rows)
        sys.stdout.write(f"\r   [{i+1}/{len(symbols)}] {sym}: {n} rows   ")
        sys.stdout.flush()

    print()

    if not all_dfs:
        print(f"   ✗ No data for {ep['name']}")
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    result = result.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    print(f"   ✓ {len(result):,} rows, {result['symbol'].nunique()} symbols, "
          f"{result['timestamp'].min():%Y-%m-%d} → {result['timestamp'].max():%Y-%m-%d}")
    return result


def save_parquet(df: pd.DataFrame, name: str):
    """Save DataFrame to parquet in data/raw/coinglass/."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{name}.parquet")

    # Incremental: merge with existing
    if os.path.exists(path):
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset=['symbol', 'timestamp'], keep='last')
        df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

    df.to_parquet(path, index=False)
    print(f"   💾 Saved {path} ({len(df):,} rows)")
    return path


def main():
    load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

    parser = argparse.ArgumentParser(description='Download CoinGlass V4 data')
    parser.add_argument('--api-key', default=os.getenv('COINGLASS_API_KEY'))
    parser.add_argument('--only', choices=list(ENDPOINTS.keys()),
                        help='Download only this endpoint')
    parser.add_argument('--symbol', help='Download only this symbol (e.g. BTC)')
    parser.add_argument('--start', default='2022-01-01',
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--interval', default=None,
                        help='Override interval (e.g. 12h, 4h, 1d)')
    parser.add_argument('--outdir', default=None,
                        help='Override output directory (default: data/raw/coinglass/)')
    args = parser.parse_args()

    # Apply CLI overrides to module globals (lambdas capture by name at call time)
    if args.interval:
        global INTERVAL
        INTERVAL = args.interval
    if args.outdir:
        global DATA_DIR
        DATA_DIR = os.path.join(PROJECT_ROOT, args.outdir) if not os.path.isabs(args.outdir) else args.outdir

    if not args.api_key:
        print("✗ No API key. Set COINGLASS_API_KEY in .env or pass --api-key")
        sys.exit(1)

    api = CoinGlassV4(args.api_key)

    # Test auth
    print("Testing API key...")
    test = api.get('/api/futures/supported-coins', {})
    if test is None:
        print("✗ API key test failed. Check your key and plan.")
        sys.exit(1)
    print(f"✓ API key valid. {len(test) if isinstance(test, list) else '?'} coins available.\n")

    # Determine symbols
    symbols = [args.symbol.upper()] if args.symbol else SYM_35

    # Time range
    start_ms = int(datetime.strptime(args.start, '%Y-%m-%d').replace(
        tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    # Determine endpoints
    endpoints = [args.only] if args.only else list(ENDPOINTS.keys())

    t0 = time.time()
    results = {}

    for ep_key in endpoints:
        df = download_endpoint(api, ep_key, symbols, start_ms, end_ms)
        if len(df) > 0:
            save_parquet(df, ep_key)
            results[ep_key] = len(df)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  DONE in {elapsed/60:.1f} min | {api.total_requests} API requests")
    for k, n in results.items():
        print(f"  {ENDPOINTS[k]['name']:20s}: {n:>8,} rows")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
