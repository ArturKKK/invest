#!/usr/bin/env python3
"""
Download alternative alpha data from OKX + Alternative.me:
1. Funding rates from OKX (free, per-coin, 8h) — most predictive
2. Fear & Greed Index from Alternative.me (free, daily)
3. Open Interest from OKX (free, per-coin, 1h)
4. Long/Short ratio from OKX (free, per-coin, 1h)

Note: Binance Futures API is geo-blocked from Russia.
      OKX is used instead (works for RF residents).

Usage:
  python src/data/download_sentiment.py
"""

import os
import sys
import time
import json
import warnings
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np
import requests

warnings.filterwarnings('ignore')
import urllib3
urllib3.disable_warnings()

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'sentiment')

# OKX instrument IDs for perpetual swaps
# Maps OKX instId → our symbol format
OKX_INSTRUMENTS = {
    'BTC-USDT-SWAP': 'BTC/USDT',
    'ETH-USDT-SWAP': 'ETH/USDT',
    'SOL-USDT-SWAP': 'SOL/USDT',
    'XRP-USDT-SWAP': 'XRP/USDT',
    'DOGE-USDT-SWAP': 'DOGE/USDT',
    'ADA-USDT-SWAP': 'ADA/USDT',
    'AVAX-USDT-SWAP': 'AVAX/USDT',
    'DOT-USDT-SWAP': 'DOT/USDT',
    'LINK-USDT-SWAP': 'LINK/USDT',
    'UNI-USDT-SWAP': 'UNI/USDT',
    'ATOM-USDT-SWAP': 'ATOM/USDT',
    'LTC-USDT-SWAP': 'LTC/USDT',
    'ETC-USDT-SWAP': 'ETC/USDT',
    'FIL-USDT-SWAP': 'FIL/USDT',
    'APT-USDT-SWAP': 'APT/USDT',
    'ARB-USDT-SWAP': 'ARB/USDT',
    'OP-USDT-SWAP': 'OP/USDT',
    'NEAR-USDT-SWAP': 'NEAR/USDT',
    'AAVE-USDT-SWAP': 'AAVE/USDT',
    'MKR-USDT-SWAP': 'MKR/USDT',
    'GRT-USDT-SWAP': 'GRT/USDT',
    'INJ-USDT-SWAP': 'INJ/USDT',
    'FTM-USDT-SWAP': 'FTM/USDT',
    'SAND-USDT-SWAP': 'SAND/USDT',
    'MANA-USDT-SWAP': 'MANA/USDT',
    'AXS-USDT-SWAP': 'AXS/USDT',
    'THETA-USDT-SWAP': 'THETA/USDT',
    'RUNE-USDT-SWAP': 'RUNE/USDT',
    'CRV-USDT-SWAP': 'CRV/USDT',
    'LDO-USDT-SWAP': 'LDO/USDT',
    'SNX-USDT-SWAP': 'SNX/USDT',
    'COMP-USDT-SWAP': 'COMP/USDT',
    'YFI-USDT-SWAP': 'YFI/USDT',
    'SUSHI-USDT-SWAP': 'SUSHI/USDT',
    'ENJ-USDT-SWAP': 'ENJ/USDT',
    'BAT-USDT-SWAP': 'BAT/USDT',
    'ZIL-USDT-SWAP': 'ZIL/USDT',
    'IOTA-USDT-SWAP': 'IOTA/USDT',
    'ENS-USDT-SWAP': 'ENS/USDT',
    'IMX-USDT-SWAP': 'IMX/USDT',
    'GALA-USDT-SWAP': 'GALA/USDT',
    'BNB-USDT-SWAP': 'BNB/USDT',
    'MATIC-USDT-SWAP': 'MATIC/USDT',
    'ALGO-USDT-SWAP': 'ALGO/USDT',
    'EGLD-USDT-SWAP': 'EGLD/USDT',
    'XTZ-USDT-SWAP': 'XTZ/USDT',
    'FLOW-USDT-SWAP': 'FLOW/USDT',
    'CHZ-USDT-SWAP': 'CHZ/USDT',
    'ONE-USDT-SWAP': 'ONE/USDT',
    'ICX-USDT-SWAP': 'ICX/USDT',
}


def download_fear_greed():
    """
    Download Crypto Fear & Greed Index from Alternative.me.
    Daily data, free API, no key needed.
    """
    print("\n📊 Downloading Fear & Greed Index...")
    
    url = "https://api.alternative.me/fng/?limit=0&format=json"
    try:
        resp = requests.get(url, timeout=30, verify=False)
        data = resp.json()
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return None

    if 'data' not in data:
        print(f"   ❌ Unexpected response format")
        return None

    rows = []
    for entry in data['data']:
        rows.append({
            'timestamp': pd.to_datetime(int(entry['timestamp']), unit='s', utc=True),
            'fng_value': int(entry['value']),
            'fng_class': entry['value_classification'],
        })

    df = pd.DataFrame(rows).sort_values('timestamp').reset_index(drop=True)
    print(f"   ✅ {len(df)} days: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print(f"   Mean: {df['fng_value'].mean():.1f}, Current: {df['fng_value'].iloc[-1]}")
    
    return df


def download_okx_funding_rates(instruments=None):
    """
    Download historical funding rates from OKX.
    OKX funding = every 8h (00:00, 08:00, 16:00 UTC).
    
    OKX public API: /api/v5/public/funding-rate-history
    Max 100 records per call, paginate with 'after' param.
    """
    if instruments is None:
        instruments = list(OKX_INSTRUMENTS.keys())
    
    print(f"\n📊 Downloading OKX funding rates for {len(instruments)} instruments...")
    
    base_url = "https://www.okx.com/api/v5/public/funding-rate-history"
    all_dfs = []
    
    for inst_id in instruments:
        sym = OKX_INSTRUMENTS.get(inst_id, inst_id)
        sym_data = []
        after = ''  # pagination cursor
        retries = 0
        
        while True:
            params = {'instId': inst_id, 'limit': '100'}
            if after:
                params['after'] = after
            
            try:
                resp = requests.get(base_url, params=params, timeout=15, verify=False)
                data = resp.json()
            except Exception as e:
                retries += 1
                if retries > 3:
                    break
                time.sleep(2)
                continue
            
            if data.get('code') != '0' or not data.get('data'):
                break
            
            records = data['data']
            sym_data.extend(records)
            
            # OKX pagination: use last item's fundingTime as 'after'
            last_ts = records[-1]['fundingTime']
            if after == last_ts:
                break
            after = last_ts
            
            time.sleep(0.15)  # Rate limit: ~10 req/s
        
        if sym_data:
            df = pd.DataFrame(sym_data)
            df['timestamp'] = pd.to_datetime(df['fundingTime'].astype(int), unit='ms', utc=True)
            df['symbol'] = sym
            df['funding_rate'] = df['fundingRate'].astype(float)
            df['realized_rate'] = df['realizedRate'].astype(float)
            df = df[['timestamp', 'symbol', 'funding_rate', 'realized_rate']]
            df = df.drop_duplicates('timestamp').sort_values('timestamp')
            all_dfs.append(df)
            
        sys.stdout.write(f"\r   {sym}: {len(sym_data)} records   ")
        sys.stdout.flush()
    
    print()
    
    if not all_dfs:
        print("   ❌ No funding rate data downloaded")
        return None
    
    result = pd.concat(all_dfs, ignore_index=True)
    result = result.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    
    n_symbols = result['symbol'].nunique()
    print(f"   ✅ {len(result):,} rows, {n_symbols} symbols")
    print(f"   Period: {result['timestamp'].min()} → {result['timestamp'].max()}")
    print(f"   Mean funding rate: {result['funding_rate'].mean():.6f}")
    
    return result


def download_okx_open_interest(instruments=None):
    """
    Download OI history from OKX.
    /api/v5/rubik/stat/contracts/open-interest-history
    Returns [ts, oi, oiCcy, oiUsd]
    Note: Very limited history (last ~100h only).
    """
    if instruments is None:
        instruments = list(OKX_INSTRUMENTS.keys())[:20]
    
    print(f"\n📊 Downloading OKX open interest for {len(instruments)} instruments...")
    
    base_url = "https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-history"
    all_dfs = []
    
    for inst_id in instruments:
        sym = OKX_INSTRUMENTS.get(inst_id, inst_id)
        
        params = {
            'instId': inst_id,
            'period': '1H',
            'limit': '100',
        }
        
        try:
            resp = requests.get(base_url, params=params, timeout=15, verify=False)
            data = resp.json()
        except Exception as e:
            continue
        
        if data.get('code') != '0' or not data.get('data'):
            continue
        
        records = data['data']
        rows = []
        for r in records:
            rows.append({
                'timestamp': pd.to_datetime(int(r[0]), unit='ms', utc=True),
                'symbol': sym,
                'open_interest': float(r[1]),
                'oi_usd': float(r[3]) if len(r) > 3 else np.nan,
            })
        
        if rows:
            df = pd.DataFrame(rows).sort_values('timestamp').drop_duplicates('timestamp')
            all_dfs.append(df)
        
        sys.stdout.write(f"\r   {sym}: {len(rows)} records   ")
        sys.stdout.flush()
        time.sleep(0.15)
    
    print()
    
    if not all_dfs:
        print("   ❌ No OI data downloaded")
        return None
    
    result = pd.concat(all_dfs, ignore_index=True)
    result = result.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    n_symbols = result['symbol'].nunique()
    print(f"   ✅ {len(result):,} rows, {n_symbols} symbols")
    
    return result


def download_okx_long_short_ratio(instruments=None):
    """
    Download top trader long/short account ratio from OKX.
    /api/v5/rubik/stat/contracts/long-short-account-ratio-contract-top-trader
    """
    if instruments is None:
        instruments = list(OKX_INSTRUMENTS.keys())[:20]
    
    print(f"\n📊 Downloading OKX long/short ratio for {len(instruments)} instruments...")
    
    base_url = "https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio-contract-top-trader"
    all_dfs = []
    
    for inst_id in instruments:
        sym = OKX_INSTRUMENTS.get(inst_id, inst_id)
        
        params = {
            'instId': inst_id,
            'period': '1H',
            'limit': '100',
        }
        
        try:
            resp = requests.get(base_url, params=params, timeout=15, verify=False)
            data = resp.json()
        except Exception:
            continue
        
        if data.get('code') != '0' or not data.get('data'):
            continue
        
        records = data['data']
        rows = []
        for r in records:
            rows.append({
                'timestamp': pd.to_datetime(int(r[0]), unit='ms', utc=True),
                'symbol': sym,
                'long_short_ratio': float(r[1]),
            })
        
        if rows:
            df = pd.DataFrame(rows).sort_values('timestamp').drop_duplicates('timestamp')
            all_dfs.append(df)
        
        sys.stdout.write(f"\r   {sym}: {len(rows)} records   ")
        sys.stdout.flush()
        time.sleep(0.15)
    
    print()
    
    if not all_dfs:
        print("   ❌ No LS ratio data downloaded")
        return None
    
    result = pd.concat(all_dfs, ignore_index=True)
    n_symbols = result['symbol'].nunique()
    print(f"   ✅ {len(result):,} rows, {n_symbols} symbols")
    
    return result


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("=" * 70)
    print("  SENTIMENT & ALTERNATIVE DATA DOWNLOADER (OKX)")
    print("=" * 70)
    
    # 1. Fear & Greed
    fng = download_fear_greed()
    if fng is not None:
        fng.to_parquet(os.path.join(DATA_DIR, 'fear_greed.parquet'), index=False)
        print(f"   💾 Saved fear_greed.parquet")
    
    # 2. Funding Rates (all symbols)
    funding = download_okx_funding_rates()
    if funding is not None:
        funding.to_parquet(os.path.join(DATA_DIR, 'funding_rates.parquet'), index=False)
        print(f"   💾 Saved funding_rates.parquet")
    
    # 3. Open Interest (top 20)
    oi = download_okx_open_interest()
    if oi is not None:
        oi.to_parquet(os.path.join(DATA_DIR, 'open_interest.parquet'), index=False)
        print(f"   💾 Saved open_interest.parquet")
    
    # 4. Long/Short Ratio (top 20)
    lsr = download_okx_long_short_ratio()
    if lsr is not None:
        lsr.to_parquet(os.path.join(DATA_DIR, 'long_short_ratio.parquet'), index=False)
        print(f"   💾 Saved long_short_ratio.parquet")
    
    print(f"\n{'='*70}")
    print(f"  ✅ All data saved to {os.path.abspath(DATA_DIR)}")
    print(f"{'='*70}")
    
    total_size = 0
    for f in sorted(os.listdir(DATA_DIR)):
        if f.endswith('.parquet'):
            size = os.path.getsize(os.path.join(DATA_DIR, f))
            total_size += size
            print(f"   {f}: {size / 1024:.1f} KB")
    print(f"   Total: {total_size / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    main()
