#!/usr/bin/env python3
"""
Download historical derivatives data from CoinGlass API (v3).

Requires a paid CoinGlass API key (Hobbyist tier: $29/month, 100 req/min).
Plan: download 2+ years of history, then cancel subscription.

Data downloaded:
  1. Liquidations — hourly long/short liquidation volumes (PRIORITY #1)
  2. OI OHLC History — open interest with OHLC candles
  3. Aggregated Funding Rates — weighted across exchanges
  4. Long/Short Ratio — top trader long/short positions
  5. Taker Buy/Sell Volume — aggregated net taker flow
  6. Exchange Netflow — BTC/ETH exchange in/outflows
  7. Coinbase Premium Index — US institutional demand proxy

Output: data/raw/coinglass/*.parquet (one file per data type)

Usage:
  # Full download (all endpoints, all symbols):
  python src/data/download_coinglass.py --api-key YOUR_KEY

  # Or set env / .env:
  echo 'COINGLASS_API_KEY=your_key_here' >> .env
  python src/data/download_coinglass.py

  # Single endpoint:
  python src/data/download_coinglass.py --only liquidations

  # Single symbol:
  python src/data/download_coinglass.py --symbol BTC

  # Resume (incremental — skips existing data):
  python src/data/download_coinglass.py

Rate limits: Hobbyist = 100 req/min → we use 0.65s delay between requests.
Runtime: ~4 hours for full download (50 symbols × all endpoints).
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

# ── config ────────────────────────────────────────────────────

BASE_URL = "https://open-api-v3.coinglass.com"

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'coinglass')

# Rate limit: 100 req/min → 0.65s between requests (with safety margin)
RATE_LIMIT_DELAY = 0.65

# Default history start — aligned with binance_futures_metrics (earliest deriv data)
DEFAULT_START = "2021-12-01"

# CoinGlass symbols (without USDT suffix — their API uses base coin names)
SYMBOLS = [
    'BTC', 'ETH', 'BNB', 'SOL', 'XRP',
    'ADA', 'DOGE', 'AVAX', 'DOT', 'LINK',
    'MATIC', 'UNI', 'ATOM', 'LTC', 'ETC',
    'FIL', 'APT', 'ARB', 'OP', 'NEAR',
    'AAVE', 'MKR', 'GRT', 'INJ', 'FTM',
    'ALGO', 'SAND', 'MANA', 'AXS', 'THETA',
    'RUNE', 'EGLD', 'XTZ', 'FLOW', 'CHZ',
    'CRV', 'LDO', 'SNX', 'COMP', 'YFI',
    'SUSHI', 'ENJ', 'BAT', 'ZIL', 'ONE',
    'IOTA', 'ICX', 'ENS', 'IMX', 'GALA',
]

# Map CoinGlass symbol → our format
def to_our_symbol(cg_sym: str) -> str:
    """BTC → BTC/USDT"""
    return f"{cg_sym}/USDT"


# ── API helpers ───────────────────────────────────────────────

class CoinGlassAPI:
    """Thin wrapper around CoinGlass v3 API with rate limiting and retries."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'CG-API-KEY': api_key,
            'Accept': 'application/json',
        })
        self._last_request_time = 0
        self.total_requests = 0

    def _rate_limit(self):
        """Enforce rate limit between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    def get(self, endpoint: str, params: dict = None, retries: int = 3) -> dict | None:
        """Make a GET request with rate limiting and retries."""
        url = f"{BASE_URL}{endpoint}"

        for attempt in range(retries):
            self._rate_limit()
            self.total_requests += 1
            try:
                resp = self.session.get(url, params=params, timeout=30)

                if resp.status_code == 429:
                    # Rate limited — back off
                    wait = min(30, 5 * (attempt + 1))
                    print(f"\n   ⚠️  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue

                if resp.status_code == 401:
                    print(f"\n   ❌ API key invalid or expired (401)")
                    return None

                if resp.status_code == 403:
                    print(f"\n   ❌ Access denied (403) — check your plan tier")
                    return None

                if resp.status_code != 200:
                    if attempt < retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return None

                data = resp.json()

                # CoinGlass v3 wraps responses: {"code": "0", "msg": "success", "data": ...}
                if data.get('code') == '0' or data.get('success'):
                    return data.get('data')

                # Some endpoints return data directly
                if 'data' in data:
                    return data['data']

                # If it's just a list/dict without wrapper
                if isinstance(data, (list, dict)) and 'code' not in data:
                    return data

                if attempt < retries - 1:
                    time.sleep(2)
                    continue
                return None

            except requests.exceptions.Timeout:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                print(f"\n   ❌ Error: {e}")
                return None

        return None


# ── Download functions ────────────────────────────────────────

def download_liquidations(api: CoinGlassAPI, symbols: list, start_date: datetime,
                          end_date: datetime) -> pd.DataFrame:
    """
    Download hourly liquidation history.
    Endpoint: /api/futures/liquidation/v2/history

    Returns: timestamp, symbol, liq_long_usd, liq_short_usd
    """
    print(f"\n{'─'*60}")
    print(f"  📊 [1/7] LIQUIDATIONS — {len(symbols)} symbols")
    print(f"  {start_date:%Y-%m-%d} → {end_date:%Y-%m-%d}")
    print(f"{'─'*60}")

    all_dfs = []

    for i, sym in enumerate(symbols):
        sym_rows = []
        # CoinGlass returns data in chunks; paginate by time
        current_start = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())

        page = 0
        while current_start < end_ts:
            data = api.get('/api/futures/liquidation/v2/history', params={
                'symbol': sym,
                'interval': 'h1',
                'startTime': current_start,
                'endTime': end_ts,
            })

            if not data:
                break

            # data is typically a list of records or dict with time series
            records = _parse_time_series(data, sym)
            if not records:
                break

            sym_rows.extend(records)

            # Find the latest timestamp and continue from there
            last_ts = max(r['timestamp'] for r in records)
            if last_ts <= current_start:
                break
            current_start = last_ts + 1
            page += 1

            # Safety: don't loop forever
            if page > 1000:
                break

        if sym_rows:
            df = pd.DataFrame(sym_rows)
            df['symbol'] = to_our_symbol(sym)
            all_dfs.append(df)

        n = len(sym_rows)
        sys.stdout.write(f"\r   [{i+1}/{len(symbols)}] {sym}: {n} rows   ")
        sys.stdout.flush()

    print()
    if not all_dfs:
        print("   ❌ No liquidation data received")
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    result['timestamp'] = pd.to_datetime(result['timestamp'], unit='s', utc=True)
    result = result.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    print(f"   ✅ {len(result):,} rows, {result['symbol'].nunique()} symbols")
    return result


def download_oi_history(api: CoinGlassAPI, symbols: list, start_date: datetime,
                        end_date: datetime) -> pd.DataFrame:
    """
    Download OI OHLC history.
    Endpoint: /api/futures/openInterest/ohlc-history

    Returns: timestamp, symbol, oi_open, oi_high, oi_low, oi_close
    """
    print(f"\n{'─'*60}")
    print(f"  📊 [2/7] OI OHLC HISTORY — {len(symbols)} symbols")
    print(f"{'─'*60}")

    all_dfs = []

    for i, sym in enumerate(symbols):
        sym_rows = []
        current_start = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())
        page = 0

        while current_start < end_ts:
            data = api.get('/api/futures/openInterest/ohlc-history', params={
                'symbol': sym,
                'interval': 'h1',
                'startTime': current_start,
                'endTime': end_ts,
            })

            if not data:
                break

            records = _parse_ohlc_data(data, sym, prefix='oi')
            if not records:
                break

            sym_rows.extend(records)
            last_ts = max(r['timestamp'] for r in records)
            if last_ts <= current_start:
                break
            current_start = last_ts + 1
            page += 1
            if page > 1000:
                break

        if sym_rows:
            df = pd.DataFrame(sym_rows)
            df['symbol'] = to_our_symbol(sym)
            all_dfs.append(df)

        sys.stdout.write(f"\r   [{i+1}/{len(symbols)}] {sym}: {len(sym_rows)} rows   ")
        sys.stdout.flush()

    print()
    if not all_dfs:
        print("   ❌ No OI history data")
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    result['timestamp'] = pd.to_datetime(result['timestamp'], unit='s', utc=True)
    result = result.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    print(f"   ✅ {len(result):,} rows, {result['symbol'].nunique()} symbols")
    return result


def download_funding_rates(api: CoinGlassAPI, symbols: list, start_date: datetime,
                           end_date: datetime) -> pd.DataFrame:
    """
    Download aggregated funding rates (weighted across exchanges).
    Endpoint: /api/futures/funding-rates-history
    """
    print(f"\n{'─'*60}")
    print(f"  📊 [3/7] AGGREGATED FUNDING RATES — {len(symbols)} symbols")
    print(f"{'─'*60}")

    all_dfs = []

    for i, sym in enumerate(symbols):
        sym_rows = []
        current_start = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())
        page = 0

        while current_start < end_ts:
            data = api.get('/api/futures/funding-rates-history', params={
                'symbol': sym,
                'startTime': current_start,
                'endTime': end_ts,
            })

            if not data:
                break

            records = _parse_funding_data(data, sym)
            if not records:
                break

            sym_rows.extend(records)
            last_ts = max(r['timestamp'] for r in records)
            if last_ts <= current_start:
                break
            current_start = last_ts + 1
            page += 1
            if page > 1000:
                break

        if sym_rows:
            df = pd.DataFrame(sym_rows)
            df['symbol'] = to_our_symbol(sym)
            all_dfs.append(df)

        sys.stdout.write(f"\r   [{i+1}/{len(symbols)}] {sym}: {len(sym_rows)} rows   ")
        sys.stdout.flush()

    print()
    if not all_dfs:
        print("   ❌ No funding rate data")
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    result['timestamp'] = pd.to_datetime(result['timestamp'], unit='s', utc=True)
    result = result.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    print(f"   ✅ {len(result):,} rows, {result['symbol'].nunique()} symbols")
    return result


def download_long_short_ratio(api: CoinGlassAPI, symbols: list, start_date: datetime,
                               end_date: datetime) -> pd.DataFrame:
    """
    Download top trader long/short ratio.
    Endpoint: /api/futures/longShort/chart
    """
    print(f"\n{'─'*60}")
    print(f"  📊 [4/7] LONG/SHORT RATIO — {len(symbols)} symbols")
    print(f"{'─'*60}")

    all_dfs = []

    for i, sym in enumerate(symbols):
        sym_rows = []
        current_start = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())
        page = 0

        while current_start < end_ts:
            data = api.get('/api/futures/longShort/chart', params={
                'symbol': sym,
                'interval': 'h1',
                'startTime': current_start,
                'endTime': end_ts,
            })

            if not data:
                break

            records = _parse_long_short_data(data, sym)
            if not records:
                break

            sym_rows.extend(records)
            last_ts = max(r['timestamp'] for r in records)
            if last_ts <= current_start:
                break
            current_start = last_ts + 1
            page += 1
            if page > 1000:
                break

        if sym_rows:
            df = pd.DataFrame(sym_rows)
            df['symbol'] = to_our_symbol(sym)
            all_dfs.append(df)

        sys.stdout.write(f"\r   [{i+1}/{len(symbols)}] {sym}: {len(sym_rows)} rows   ")
        sys.stdout.flush()

    print()
    if not all_dfs:
        print("   ❌ No long/short ratio data")
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    result['timestamp'] = pd.to_datetime(result['timestamp'], unit='s', utc=True)
    result = result.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    print(f"   ✅ {len(result):,} rows, {result['symbol'].nunique()} symbols")
    return result


def download_taker_volume(api: CoinGlassAPI, symbols: list, start_date: datetime,
                          end_date: datetime) -> pd.DataFrame:
    """
    Download aggregated taker buy/sell volumes.
    Endpoint: /api/futures/aggregated-taker-buy-sell-volume/history
    """
    print(f"\n{'─'*60}")
    print(f"  📊 [5/7] TAKER BUY/SELL VOLUME — {len(symbols)} symbols")
    print(f"{'─'*60}")

    all_dfs = []

    for i, sym in enumerate(symbols):
        sym_rows = []
        current_start = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())
        page = 0

        while current_start < end_ts:
            data = api.get('/api/futures/aggregated-taker-buy-sell-volume/history', params={
                'symbol': sym,
                'interval': 'h1',
                'startTime': current_start,
                'endTime': end_ts,
            })

            if not data:
                break

            records = _parse_taker_volume_data(data, sym)
            if not records:
                break

            sym_rows.extend(records)
            last_ts = max(r['timestamp'] for r in records)
            if last_ts <= current_start:
                break
            current_start = last_ts + 1
            page += 1
            if page > 1000:
                break

        if sym_rows:
            df = pd.DataFrame(sym_rows)
            df['symbol'] = to_our_symbol(sym)
            all_dfs.append(df)

        sys.stdout.write(f"\r   [{i+1}/{len(symbols)}] {sym}: {len(sym_rows)} rows   ")
        sys.stdout.flush()

    print()
    if not all_dfs:
        print("   ❌ No taker volume data")
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    result['timestamp'] = pd.to_datetime(result['timestamp'], unit='s', utc=True)
    result = result.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    print(f"   ✅ {len(result):,} rows, {result['symbol'].nunique()} symbols")
    return result


def download_exchange_netflow(api: CoinGlassAPI, start_date: datetime,
                              end_date: datetime) -> pd.DataFrame:
    """
    Download BTC/ETH exchange netflow (inflow/outflow).
    Endpoint: /api/bitcoin/exchange-balance-list (BTC)
              /api/ethereum/exchange-balance-list (ETH)

    Only for BTC and ETH (on-chain data, not per-altcoin).
    """
    print(f"\n{'─'*60}")
    print(f"  📊 [6/7] EXCHANGE NETFLOW (BTC + ETH)")
    print(f"{'─'*60}")

    all_dfs = []

    for coin, endpoint in [('BTC', '/api/bitcoin/exchange-balance-list'),
                           ('ETH', '/api/ethereum/exchange-balance-list')]:
        data = api.get(endpoint, params={
            'startTime': int(start_date.timestamp()),
            'endTime': int(end_date.timestamp()),
        })

        if not data:
            print(f"   ⚠️  No netflow data for {coin}")
            continue

        records = _parse_netflow_data(data, coin)
        if records:
            df = pd.DataFrame(records)
            df['symbol'] = to_our_symbol(coin)
            all_dfs.append(df)
            print(f"   {coin}: {len(records)} rows")

    if not all_dfs:
        print("   ❌ No exchange netflow data")
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    result['timestamp'] = pd.to_datetime(result['timestamp'], unit='s', utc=True)
    result = result.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    print(f"   ✅ {len(result):,} rows")
    return result


def download_coinbase_premium(api: CoinGlassAPI, start_date: datetime,
                              end_date: datetime) -> pd.DataFrame:
    """
    Download Coinbase Premium Index.
    Endpoint: /api/index/coinbase-premium-index
    """
    print(f"\n{'─'*60}")
    print(f"  📊 [7/7] COINBASE PREMIUM INDEX")
    print(f"{'─'*60}")

    sym_rows = []
    current_start = int(start_date.timestamp())
    end_ts = int(end_date.timestamp())
    page = 0

    while current_start < end_ts:
        data = api.get('/api/index/coinbase-premium-index', params={
            'interval': 'h1',
            'startTime': current_start,
            'endTime': end_ts,
        })

        if not data:
            break

        records = _parse_premium_data(data)
        if not records:
            break

        sym_rows.extend(records)
        last_ts = max(r['timestamp'] for r in records)
        if last_ts <= current_start:
            break
        current_start = last_ts + 1
        page += 1
        if page > 1000:
            break

    if not sym_rows:
        print("   ❌ No Coinbase Premium data")
        return pd.DataFrame()

    result = pd.DataFrame(sym_rows)
    result['symbol'] = 'BTC/USDT'  # Premium is BTC-specific
    result['timestamp'] = pd.to_datetime(result['timestamp'], unit='s', utc=True)
    result = result.sort_values('timestamp').reset_index(drop=True)
    print(f"   ✅ {len(result):,} rows")
    return result


# ── Parsers (CoinGlass response formats vary by endpoint) ────

def _parse_time_series(data, symbol: str) -> list[dict]:
    """Generic parser for CoinGlass time-series responses.

    CoinGlass v3 typically returns:
      - List of dicts with 't' (timestamp), and value fields
      - Or dict with 'dataMap'/'list' containing arrays
    """
    records = []

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                ts = item.get('t') or item.get('time') or item.get('createTime') or item.get('timestamp')
                if ts is None:
                    continue
                ts = _normalize_timestamp(ts)
                record = {'timestamp': ts}

                # Liquidation fields
                for key in ['longLiquidationUsd', 'longVolUsd', 'buyVolUsd', 'longLiqUsd']:
                    if key in item:
                        record['liq_long_usd'] = float(item[key])
                        break

                for key in ['shortLiquidationUsd', 'shortVolUsd', 'sellVolUsd', 'shortLiqUsd']:
                    if key in item:
                        record['liq_short_usd'] = float(item[key])
                        break

                # Total liquidation
                for key in ['volUsd', 'totalVolUsd', 'liquidationUsd']:
                    if key in item and 'liq_long_usd' not in record:
                        record['liq_total_usd'] = float(item[key])
                        break

                records.append(record)

    elif isinstance(data, dict):
        # Try nested formats
        for list_key in ['dataMap', 'list', 'data', 'items']:
            if list_key in data:
                sub = data[list_key]
                if isinstance(sub, list):
                    return _parse_time_series(sub, symbol)
                elif isinstance(sub, dict):
                    # dataMap format: {"timestamps": [...], "values": [...]}
                    if 'timestamps' in sub or 't' in sub:
                        timestamps = sub.get('timestamps') or sub.get('t', [])
                        for j, ts in enumerate(timestamps):
                            record = {'timestamp': _normalize_timestamp(ts)}
                            for vk in ['longLiquidationUsd', 'shortLiquidationUsd',
                                       'longVolUsd', 'shortVolUsd']:
                                if vk in sub and j < len(sub[vk]):
                                    clean_key = vk.replace('Liquidation', '_liq_').replace('Vol', '_vol_')
                                    if 'long' in vk.lower():
                                        record['liq_long_usd'] = float(sub[vk][j])
                                    elif 'short' in vk.lower():
                                        record['liq_short_usd'] = float(sub[vk][j])
                            records.append(record)

        # Some endpoints: {"dateList": [...], "dataMap": {"exchange": [...]}}
        if 'dateList' in data:
            dates = data['dateList']
            dmap = data.get('dataMap', {})
            for j, ts in enumerate(dates):
                record = {'timestamp': _normalize_timestamp(ts)}
                for k, vals in dmap.items():
                    if isinstance(vals, list) and j < len(vals):
                        record[k] = vals[j]
                records.append(record)

    return records


def _parse_ohlc_data(data, symbol: str, prefix: str = 'oi') -> list[dict]:
    """Parse OI OHLC or similar candle data."""
    records = []

    items = data if isinstance(data, list) else data.get('list', data.get('data', []))
    if not isinstance(items, list):
        return records

    for item in items:
        if not isinstance(item, dict):
            continue
        ts = item.get('t') or item.get('time') or item.get('timestamp')
        if ts is None:
            continue
        record = {
            'timestamp': _normalize_timestamp(ts),
            f'{prefix}_open': _safe_float(item.get('o') or item.get('open')),
            f'{prefix}_high': _safe_float(item.get('h') or item.get('high')),
            f'{prefix}_low': _safe_float(item.get('l') or item.get('low')),
            f'{prefix}_close': _safe_float(item.get('c') or item.get('close')),
        }
        records.append(record)

    return records


def _parse_funding_data(data, symbol: str) -> list[dict]:
    """Parse funding rate response."""
    records = []

    items = data if isinstance(data, list) else data.get('list', data.get('data', []))
    if not isinstance(items, list):
        return records

    for item in items:
        if not isinstance(item, dict):
            continue
        ts = item.get('t') or item.get('time') or item.get('timestamp') or item.get('calcTime')
        if ts is None:
            continue
        record = {'timestamp': _normalize_timestamp(ts)}

        # Weighted/average funding rate
        for key in ['fundingRate', 'rate', 'avgRate', 'weightedRate', 'averageFundingRate']:
            if key in item and item[key] is not None:
                record['cg_funding_rate'] = _safe_float(item[key])
                break

        records.append(record)

    return records


def _parse_long_short_data(data, symbol: str) -> list[dict]:
    """Parse long/short ratio response."""
    records = []

    items = data if isinstance(data, list) else data.get('list', data.get('data', []))
    if not isinstance(items, list):
        return records

    for item in items:
        if not isinstance(item, dict):
            continue
        ts = item.get('t') or item.get('time') or item.get('timestamp')
        if ts is None:
            continue
        record = {'timestamp': _normalize_timestamp(ts)}

        for key in ['longRate', 'longPercent', 'longRatio', 'longAccount']:
            if key in item and item[key] is not None:
                record['cg_long_rate'] = _safe_float(item[key])
                break

        for key in ['shortRate', 'shortPercent', 'shortRatio', 'shortAccount']:
            if key in item and item[key] is not None:
                record['cg_short_rate'] = _safe_float(item[key])
                break

        for key in ['longShortRatio', 'ratio', 'lsRatio']:
            if key in item and item[key] is not None:
                record['cg_ls_ratio'] = _safe_float(item[key])
                break

        records.append(record)

    return records


def _parse_taker_volume_data(data, symbol: str) -> list[dict]:
    """Parse taker buy/sell volume response."""
    records = []

    items = data if isinstance(data, list) else data.get('list', data.get('data', []))
    if not isinstance(items, list):
        return records

    for item in items:
        if not isinstance(item, dict):
            continue
        ts = item.get('t') or item.get('time') or item.get('timestamp')
        if ts is None:
            continue
        record = {'timestamp': _normalize_timestamp(ts)}

        for key in ['buyVol', 'buyVolume', 'takerBuyVolume', 'buy']:
            if key in item and item[key] is not None:
                record['taker_buy_vol'] = _safe_float(item[key])
                break

        for key in ['sellVol', 'sellVolume', 'takerSellVolume', 'sell']:
            if key in item and item[key] is not None:
                record['taker_sell_vol'] = _safe_float(item[key])
                break

        for key in ['buyVolUsd', 'takerBuyVolumeUsd']:
            if key in item and item[key] is not None:
                record['taker_buy_vol_usd'] = _safe_float(item[key])
                break

        for key in ['sellVolUsd', 'takerSellVolumeUsd']:
            if key in item and item[key] is not None:
                record['taker_sell_vol_usd'] = _safe_float(item[key])
                break

        records.append(record)

    return records


def _parse_netflow_data(data, coin: str) -> list[dict]:
    """Parse exchange netflow response."""
    records = []

    items = data if isinstance(data, list) else data.get('list', data.get('data', []))
    if not isinstance(items, list):
        return records

    for item in items:
        if not isinstance(item, dict):
            continue
        ts = item.get('t') or item.get('time') or item.get('timestamp') or item.get('date')
        if ts is None:
            continue
        record = {'timestamp': _normalize_timestamp(ts)}

        for key in ['netflow', 'netFlow', 'flowTotal']:
            if key in item and item[key] is not None:
                record['exchange_netflow'] = _safe_float(item[key])
                break

        for key in ['inflow', 'inflowTotal']:
            if key in item and item[key] is not None:
                record['exchange_inflow'] = _safe_float(item[key])
                break

        for key in ['outflow', 'outflowTotal']:
            if key in item and item[key] is not None:
                record['exchange_outflow'] = _safe_float(item[key])
                break

        for key in ['balance', 'balanceTotal']:
            if key in item and item[key] is not None:
                record['exchange_balance'] = _safe_float(item[key])
                break

        records.append(record)

    return records


def _parse_premium_data(data) -> list[dict]:
    """Parse Coinbase premium index response."""
    records = []

    items = data if isinstance(data, list) else data.get('list', data.get('data', []))
    if not isinstance(items, list):
        return records

    for item in items:
        if not isinstance(item, dict):
            continue
        ts = item.get('t') or item.get('time') or item.get('timestamp')
        if ts is None:
            continue
        record = {'timestamp': _normalize_timestamp(ts)}

        for key in ['premium', 'premiumIndex', 'value', 'coinbasePremium']:
            if key in item and item[key] is not None:
                record['coinbase_premium'] = _safe_float(item[key])
                break

        records.append(record)

    return records


# ── Utilities ─────────────────────────────────────────────────

def _normalize_timestamp(ts) -> int:
    """Convert various timestamp formats to unix seconds."""
    if isinstance(ts, (int, float)):
        # If timestamp is in milliseconds (> year 2100 in seconds), convert
        if ts > 1e12:
            return int(ts / 1000)
        return int(ts)
    if isinstance(ts, str):
        try:
            dt = pd.to_datetime(ts, utc=True)
            return int(dt.timestamp())
        except Exception:
            return 0
    return 0


def _safe_float(val) -> float:
    """Safely convert to float."""
    if val is None:
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def get_api_key(key_arg: str = None) -> str:
    """Get CoinGlass API key from arg, env, or .env file."""
    key = key_arg or os.environ.get('COINGLASS_API_KEY')
    if key:
        return key

    # Try .env file in project root
    env_path = os.path.join(PROJECT_ROOT, '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith('COINGLASS_API_KEY='):
                    return line.split('=', 1)[1].strip().strip('"').strip("'")

    print("   ❌ No CoinGlass API key found!")
    print("      1. Get a key: https://www.coinglass.com/pricing (Hobbyist $29/mo)")
    print("      2. Then either:")
    print("         --api-key YOUR_KEY")
    print("         export COINGLASS_API_KEY=YOUR_KEY")
    print("         echo 'COINGLASS_API_KEY=YOUR_KEY' >> .env")
    sys.exit(1)


def save_parquet(df: pd.DataFrame, filename: str):
    """Save DataFrame to parquet, merging with existing data."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)
    key_cols = ['timestamp', 'symbol'] if 'symbol' in df.columns else ['timestamp']

    if os.path.exists(path):
        existing = pd.read_parquet(path)
        existing['timestamp'] = pd.to_datetime(existing['timestamp'], utc=True)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(key_cols, keep='last')
        n_new = len(combined) - len(existing)
        print(f"   💾 {filename}: {len(existing):,} + {n_new:,} new = {len(combined):,}")
    else:
        combined = df
        print(f"   💾 {filename}: {len(combined):,} rows (new)")

    combined = combined.sort_values(key_cols).reset_index(drop=True)
    combined.to_parquet(path, index=False)
    return combined


# ── Probe: test API key and discover response formats ─────────

def probe_api(api: CoinGlassAPI) -> bool:
    """Test API key with a small request, print response format for debugging."""
    print("\n🔍 Probing API key...")

    # Test with a simple endpoint
    data = api.get('/api/futures/liquidation/v2/history', params={
        'symbol': 'BTC',
        'interval': 'h1',
    })

    if data is None:
        print("   ❌ API probe failed — check your key and internet connection")
        return False

    print(f"   ✅ API key valid! Response type: {type(data).__name__}")

    # Log sample for debugging
    sample_path = os.path.join(DATA_DIR, '_api_probe_sample.json')
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        sample = data[:3] if isinstance(data, list) else (
            {k: (v[:3] if isinstance(v, list) else v) for k, v in data.items()}
            if isinstance(data, dict) else data
        )
        with open(sample_path, 'w') as f:
            json.dump(sample, f, indent=2, default=str)
        print(f"   📝 Sample response saved to {sample_path}")
        print(f"      (inspect this file if parsing looks wrong)")
    except Exception:
        pass

    return True


# ── Main ──────────────────────────────────────────────────────

ENDPOINTS = {
    'liquidations': 'cg_liquidations.parquet',
    'oi': 'cg_oi_history.parquet',
    'funding': 'cg_funding_rates.parquet',
    'longshort': 'cg_long_short_ratio.parquet',
    'taker': 'cg_taker_volume.parquet',
    'netflow': 'cg_exchange_netflow.parquet',
    'premium': 'cg_coinbase_premium.parquet',
}


def main():
    parser = argparse.ArgumentParser(
        description="Download derivatives data from CoinGlass API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/data/download_coinglass.py --api-key KEY123
  python src/data/download_coinglass.py --only liquidations
  python src/data/download_coinglass.py --only liquidations,oi,funding
  python src/data/download_coinglass.py --symbol BTC --only liquidations
  python src/data/download_coinglass.py --probe    # test API key only
        """)
    parser.add_argument('--api-key', type=str, default=None,
                        help="CoinGlass API key (or set COINGLASS_API_KEY env)")
    parser.add_argument('--start', type=str, default=DEFAULT_START,
                        help=f"Start date YYYY-MM-DD (default: {DEFAULT_START})")
    parser.add_argument('--symbol', type=str, default=None,
                        help="Single symbol (e.g. BTC)")
    parser.add_argument('--only', type=str, default=None,
                        help="Comma-separated endpoints: liquidations,oi,funding,longshort,taker,netflow,premium")
    parser.add_argument('--probe', action='store_true',
                        help="Only test API key, don't download")
    args = parser.parse_args()

    # Get API key
    api_key = get_api_key(args.api_key)
    api = CoinGlassAPI(api_key)

    os.makedirs(DATA_DIR, exist_ok=True)

    # Probe
    if not probe_api(api):
        sys.exit(1)

    if args.probe:
        print("\n✅ Probe complete — API key works!")
        return

    # Setup
    symbols = [args.symbol.upper()] if args.symbol else SYMBOLS
    start_date = datetime.strptime(args.start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    end_date = datetime.now(timezone.utc)

    only = set(args.only.split(',')) if args.only else set(ENDPOINTS.keys())

    print(f"\n{'='*60}")
    print(f"  COINGLASS DATA DOWNLOADER")
    print(f"  Symbols: {len(symbols)}")
    print(f"  Period: {start_date:%Y-%m-%d} → {end_date:%Y-%m-%d}")
    print(f"  Endpoints: {', '.join(sorted(only))}")
    print(f"  Output: {DATA_DIR}")
    print(f"  Rate limit: {RATE_LIMIT_DELAY}s/req (~{int(60/RATE_LIMIT_DELAY)} req/min)")
    est_requests = len(symbols) * len(only) * 5  # rough estimate
    est_minutes = est_requests * RATE_LIMIT_DELAY / 60
    print(f"  Estimated: ~{est_requests} requests, ~{est_minutes:.0f} min")
    print(f"{'='*60}")

    results = {}

    # 1. Liquidations (PRIORITY #1)
    if 'liquidations' in only:
        df = download_liquidations(api, symbols, start_date, end_date)
        if len(df) > 0:
            save_parquet(df, ENDPOINTS['liquidations'])
            results['liquidations'] = len(df)

    # 2. OI History
    if 'oi' in only:
        df = download_oi_history(api, symbols, start_date, end_date)
        if len(df) > 0:
            save_parquet(df, ENDPOINTS['oi'])
            results['oi'] = len(df)

    # 3. Funding Rates
    if 'funding' in only:
        df = download_funding_rates(api, symbols, start_date, end_date)
        if len(df) > 0:
            save_parquet(df, ENDPOINTS['funding'])
            results['funding'] = len(df)

    # 4. Long/Short Ratio
    if 'longshort' in only:
        df = download_long_short_ratio(api, symbols, start_date, end_date)
        if len(df) > 0:
            save_parquet(df, ENDPOINTS['longshort'])
            results['longshort'] = len(df)

    # 5. Taker Volume
    if 'taker' in only:
        df = download_taker_volume(api, symbols, start_date, end_date)
        if len(df) > 0:
            save_parquet(df, ENDPOINTS['taker'])
            results['taker'] = len(df)

    # 6. Exchange Netflow (BTC + ETH only)
    if 'netflow' in only:
        df = download_exchange_netflow(api, start_date, end_date)
        if len(df) > 0:
            save_parquet(df, ENDPOINTS['netflow'])
            results['netflow'] = len(df)

    # 7. Coinbase Premium
    if 'premium' in only:
        df = download_coinbase_premium(api, start_date, end_date)
        if len(df) > 0:
            save_parquet(df, ENDPOINTS['premium'])
            results['premium'] = len(df)

    # Summary
    print(f"\n{'='*60}")
    print(f"  ✅ DOWNLOAD COMPLETE")
    print(f"  Total API requests: {api.total_requests}")
    print(f"{'='*60}")

    for name, count in results.items():
        print(f"   {name:15s}: {count:>10,} rows → {ENDPOINTS[name]}")

    if not results:
        print("   ⚠️  No data downloaded. Check:")
        print("      - API key is valid (--probe)")
        print("      - Endpoints are correct (--only)")
        print("      - CoinGlass API response format may have changed")
        print(f"      - Check {DATA_DIR}/_api_probe_sample.json")

    print(f"\n  Files saved to: {DATA_DIR}/")
    for f in sorted(os.listdir(DATA_DIR)):
        if f.endswith('.parquet'):
            p = os.path.join(DATA_DIR, f)
            size_mb = os.path.getsize(p) / 1024 / 1024
            print(f"   {f} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
