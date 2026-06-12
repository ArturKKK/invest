#!/usr/bin/env python3
"""
Real-time venue data fetcher (R183 / deploy phase 2A).

Appends fresh hourly bars to the THREE venue parquets the specialist leg's
features are built from, mirroring fetch_realtime_derivatives' patch pattern:

  data/raw/okx/okx_candles_1h.parquet          OKX swap candles
      GET https://www.okx.com/api/v5/market/candles  (public, no key)
  data/raw/coinbase/coinbase_candles_1h.parquet Coinbase spot candles
      GET https://api.exchange.coinbase.com/products/{p}/candles (public)
  data/raw/basis/premium_index_klines_1h.parquet Binance premium index klines
      GET https://fapi.binance.com/fapi/v1/premiumIndexKlines (public)

Universe = whatever instruments already exist in each parquet (kept in sync
with the backfill; no separate hardcoded lists). Only CLOSED hourly bars are
appended (current partial hour excluded — z168 features must see the same
bar the backtest saw). Each venue is independently fault-isolated: an outage
degrades that venue's features to stale, never kills the cycle.

Usage:
  from src.data.fetch_realtime_venue import patch_venue_realtime
  patch_venue_realtime(project_root)
"""

import os
import time
import warnings
from datetime import datetime, timezone

import pandas as pd
import requests

warnings.filterwarnings('ignore')

OKX_URL = "https://www.okx.com/api/v5/market/candles"
CB_URL = "https://api.exchange.coinbase.com/products/{product}/candles"
PREMIUM_URL = "https://fapi.binance.com/fapi/v1/premiumIndexKlines"


def _get(url, params=None, retries=2, timeout=15):
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            if resp.status_code in (400, 404):
                return None
        except Exception:
            time.sleep(1)
    return None


def _atomic_save(df, path):
    """tmp + os.replace: a kill mid-write must never truncate the parquet
    (a corrupt venue file silently disables the specialist leg)."""
    tmp = path + '.tmp'
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _warn_gap(name, last_ts_utc):
    """Single-page fetchers can't heal gaps older than ~300h — say so."""
    age_h = (datetime.now(timezone.utc) - last_ts_utc).total_seconds() / 3600
    if age_h > 290:
        print(f"      🚨 {name}: parquet is {age_h:.0f}h stale — beyond one fetch "
              f"page; run the backfill script, live patching cannot heal this gap")


def _floor_hour_utc():
    now = datetime.now(timezone.utc)
    return now.replace(minute=0, second=0, microsecond=0)


def _patch_okx(root, verbose):
    path = os.path.join(root, 'data', 'raw', 'okx', 'okx_candles_1h.parquet')
    if not os.path.exists(path):
        return 0
    existing = pd.read_parquet(path)
    inst_ids = sorted(existing['instId'].unique())
    ts_num = pd.to_numeric(existing['ts'], errors='coerce')
    _warn_gap('okx', pd.Timestamp(int(ts_num.max()), unit='ms', tz='UTC'))
    cutoff_ms = int(_floor_hour_utc().timestamp() * 1000)  # exclude current hour
    new_rows = []
    for inst in inst_ids:
        last_ms = int(ts_num[existing['instId'] == inst].max())
        data = _get(OKX_URL, {'instId': inst, 'bar': '1H', 'limit': '300'})
        if not data or data.get('code') != '0':
            continue
        for row in data.get('data', []):
            ts = int(row[0])
            confirmed = (len(row) < 9) or (str(row[8]) == '1')
            if ts > last_ms and ts < cutoff_ms and confirmed:
                new_rows.append({'instId': inst, 'ts': str(ts),
                                 'open': float(row[1]), 'high': float(row[2]),
                                 'low': float(row[3]), 'close': float(row[4]),
                                 'vol': float(row[5])})
        time.sleep(0.12)  # OKX public rate limit headroom
    if not new_rows:
        return 0
    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    combined['_tsn'] = pd.to_numeric(combined['ts'], errors='coerce')
    combined = (combined.drop_duplicates(['instId', '_tsn'], keep='last')
                .sort_values(['instId', '_tsn']).drop(columns=['_tsn'])
                .reset_index(drop=True))
    _atomic_save(combined, path)
    if verbose:
        print(f"      ✅ okx: +{len(new_rows)} bars ({len(inst_ids)} inst)")
    return len(new_rows)


def _patch_coinbase(root, verbose):
    path = os.path.join(root, 'data', 'raw', 'coinbase', 'coinbase_candles_1h.parquet')
    if not os.path.exists(path):
        return 0
    existing = pd.read_parquet(path)
    products = sorted(existing['product'].unique())
    _warn_gap('coinbase', pd.Timestamp(int(existing['ts'].max()), unit='s', tz='UTC'))
    cutoff_s = int(_floor_hour_utc().timestamp())
    new_rows = []
    for product in products:
        last_s = int(existing.loc[existing['product'] == product, 'ts'].max())
        data = _get(CB_URL.format(product=product), {'granularity': 3600})
        if not isinstance(data, list):
            continue
        for row in data:  # [time, low, high, open, close, volume], newest first
            ts = int(row[0])
            if ts > last_s and ts < cutoff_s:
                new_rows.append({'product': product, 'ts': ts,
                                 'low': float(row[1]), 'high': float(row[2]),
                                 'open': float(row[3]), 'close': float(row[4]),
                                 'volume': float(row[5]),
                                 'datetime': str(pd.Timestamp(ts, unit='s', tz='UTC'))})
        time.sleep(0.15)
    if not new_rows:
        return 0
    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    combined = (combined.drop_duplicates(['product', 'ts'], keep='last')
                .sort_values(['product', 'ts']).reset_index(drop=True))
    _atomic_save(combined, path)
    if verbose:
        print(f"      ✅ coinbase: +{len(new_rows)} bars ({len(products)} prod)")
    return len(new_rows)


def _patch_premium(root, verbose):
    path = os.path.join(root, 'data', 'raw', 'basis', 'premium_index_klines_1h.parquet')
    if not os.path.exists(path):
        return 0
    existing = pd.read_parquet(path)
    # stored tz-naive UTC (matches backfill; research applies utc=True on read)
    ex_ts = pd.to_datetime(existing['timestamp'])
    symbols = sorted(existing['symbol'].unique())
    _warn_gap('premium', ex_ts.max().tz_localize('UTC'))
    cutoff = _floor_hour_utc().replace(tzinfo=None)
    new_rows = []
    for sym in symbols:
        last = ex_ts[existing['symbol'] == sym].max()
        data = _get(PREMIUM_URL, {'symbol': sym, 'interval': '1h', 'limit': 1000})
        if not isinstance(data, list):
            continue
        for k in data:  # klines: [openTime, open, high, low, close, ...]
            ts = pd.Timestamp(int(k[0]), unit='ms')
            if ts > last and ts < cutoff:
                new_rows.append({'timestamp': ts, 'symbol': sym,
                                 'open': float(k[1]), 'high': float(k[2]),
                                 'low': float(k[3]), 'close': float(k[4])})
        time.sleep(0.1)
    if not new_rows:
        return 0
    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    combined = (combined.drop_duplicates(['symbol', 'timestamp'], keep='last')
                .sort_values(['symbol', 'timestamp']).reset_index(drop=True))
    _atomic_save(combined, path)
    if verbose:
        print(f"      ✅ premium: +{len(new_rows)} bars ({len(symbols)} syms)")
    return len(new_rows)


def patch_venue_realtime(project_root: str, verbose: bool = True) -> dict:
    """Top up all three venue parquets to the last CLOSED hour.

    Returns {'okx': n, 'coinbase': n, 'premium': n}; each venue is isolated —
    a failing venue reports 0 and leaves its parquet untouched.
    """
    if verbose:
        print("   🔄 Patching venue data (okx/coinbase/premium)...")
    out = {}
    for name, fn in (('okx', _patch_okx), ('coinbase', _patch_coinbase),
                     ('premium', _patch_premium)):
        try:
            out[name] = fn(project_root, verbose)
        except Exception as e:
            print(f"      ⚠️  venue patch {name} failed: {str(e)[:100]}")
            out[name] = 0
    return out


if __name__ == '__main__':
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    print(patch_venue_realtime(root))
