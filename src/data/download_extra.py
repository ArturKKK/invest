#!/usr/bin/env python3
"""
Download additional features from free APIs:
1. Binance Spot OHLCV → spot/futures volume ratio per coin (CS feature)
2. Stablecoin supply from CoinMetrics (USDT, USDC, DAI) — market-level regime
3. Blockchain.com BTC on-chain charts — market-level regime

Output:
  data/sentiment/spot_futures_volume.parquet  (per-coin, daily)
  data/sentiment/stablecoin_supply.parquet    (market-level, daily)
  data/sentiment/btc_onchain.parquet          (market-level, daily)
"""
import os, sys, time, json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "sentiment"
DATA_DIR.mkdir(parents=True, exist_ok=True)

START_TS = int(datetime(2020, 1, 1).timestamp() * 1000)  # 2020-01-01 in ms

SYM_35 = [
    'BTC/USDT','ETH/USDT','SOL/USDT','BNB/USDT','XRP/USDT',
    'ADA/USDT','DOGE/USDT','AVAX/USDT','DOT/USDT','LINK/USDT',
    'MATIC/USDT','UNI/USDT','ATOM/USDT','LTC/USDT','NEAR/USDT',
    'FIL/USDT','APT/USDT','ARB/USDT','OP/USDT','AAVE/USDT',
    'INJ/USDT','FTM/USDT','ALGO/USDT','SAND/USDT','MANA/USDT',
    'AXS/USDT','THETA/USDT','RUNE/USDT','EGLD/USDT','XTZ/USDT',
    'FLOW/USDT','CHZ/USDT','CRV/USDT','LDO/USDT','SNX/USDT',
]


# ─── 1. Binance Spot + Futures Volume ──────────────────────────────────────────
def fetch_binance_klines(symbol_binance, interval='1d', start_ms=None, futures=False):
    """Fetch all klines from Binance (spot or futures), paginating by 1000."""
    base = 'https://fapi.binance.com/fapi/v1/klines' if futures else 'https://api.binance.com/api/v3/klines'
    all_klines = []
    current_start = start_ms or START_TS

    while True:
        params = {
            'symbol': symbol_binance,
            'interval': interval,
            'startTime': current_start,
            'limit': 1000,
        }
        try:
            r = requests.get(base, params=params, timeout=30)
            if r.status_code == 400:
                break  # symbol doesn't exist on this exchange
            r.raise_for_status()
            data = r.json()
            if not data:
                break
            all_klines.extend(data)
            last_close_time = data[-1][6]  # closeTime in ms
            current_start = last_close_time + 1
            if len(data) < 1000:
                break
            time.sleep(0.1)
        except requests.exceptions.RequestException as e:
            print(f"    Error: {e}")
            break

    return all_klines


def download_spot_futures_volume():
    """Download spot and futures daily volume for all 35 coins."""
    print("=" * 60)
    print("Binance Spot vs Futures Volume")
    print("=" * 60)

    all_rows = []

    for sym in SYM_35:
        binance_sym = sym.replace('/', '')  # BTC/USDT -> BTCUSDT
        print(f"  {sym}...", end=" ", flush=True)

        # Spot
        spot = fetch_binance_klines(binance_sym, futures=False)
        # Futures
        fut = fetch_binance_klines(binance_sym, futures=True)

        if not spot and not fut:
            print("NO DATA")
            continue

        # Parse into DataFrames
        # Kline: [openTime, open, high, low, close, volume, closeTime, quoteVolume, trades, takerBuyBase, takerBuyQuote, ignore]
        def parse_klines(klines, prefix):
            rows = []
            for k in klines:
                rows.append({
                    'date': pd.Timestamp(k[0], unit='ms').normalize(),
                    f'{prefix}_volume_base': float(k[5]),
                    f'{prefix}_volume_quote': float(k[7]),
                    f'{prefix}_trades': int(k[8]),
                    f'{prefix}_taker_buy_ratio': float(k[10]) / float(k[7]) if float(k[7]) > 0 else 0.5,
                })
            return pd.DataFrame(rows)

        spot_df = parse_klines(spot, 'spot') if spot else pd.DataFrame()
        fut_df = parse_klines(fut, 'fut') if fut else pd.DataFrame()

        if not spot_df.empty and not fut_df.empty:
            merged = spot_df.merge(fut_df, on='date', how='outer')
        elif not spot_df.empty:
            merged = spot_df
        else:
            merged = fut_df

        merged['symbol'] = sym
        all_rows.append(merged)

        n_spot = len(spot_df) if not spot_df.empty else 0
        n_fut = len(fut_df) if not fut_df.empty else 0
        print(f"spot={n_spot}d, fut={n_fut}d")
        time.sleep(0.2)

    if not all_rows:
        print("No data!")
        return pd.DataFrame()

    df = pd.concat(all_rows, ignore_index=True)
    df = df.sort_values(['symbol', 'date'])

    # Compute derived features
    # Spot/futures volume ratio
    df['vol_spot_fut_ratio'] = np.where(
        df['fut_volume_quote'] > 0,
        df['spot_volume_quote'] / df['fut_volume_quote'],
        np.nan
    )

    # Spot taker buy ratio (already computed per source)
    # Futures-to-spot volume dominance
    total = df['spot_volume_quote'].fillna(0) + df['fut_volume_quote'].fillna(0)
    df['fut_vol_dominance'] = np.where(total > 0, df['fut_volume_quote'].fillna(0) / total, np.nan)

    # Rolling changes
    for col in ['spot_volume_quote', 'fut_volume_quote', 'vol_spot_fut_ratio', 'fut_vol_dominance']:
        if col in df.columns:
            df[f'{col}_chg7d'] = df.groupby('symbol')[col].pct_change(7)

    # Spot trades intensity
    df['spot_avg_trade_size'] = np.where(
        df['spot_trades'] > 0,
        df['spot_volume_quote'] / df['spot_trades'],
        np.nan
    )

    out_path = DATA_DIR / "spot_futures_volume.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved: {df.shape} to {out_path}")
    print(f"  Symbols: {df['symbol'].nunique()}, Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"  Columns: {list(df.columns)}")
    return df


# ─── 2. Stablecoin Supply from CoinMetrics ────────────────────────────────────
def download_stablecoin_supply():
    """Download USDT, USDC, DAI supply from CoinMetrics."""
    print("\n" + "=" * 60)
    print("Stablecoin Supply (CoinMetrics)")
    print("=" * 60)

    stables = {'usdt': 'USDT', 'usdc': 'USDC', 'dai': 'DAI'}
    all_rows = []

    for cm_id, name in stables.items():
        print(f"  {name}...", end=" ", flush=True)
        url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
        params = {
            'assets': cm_id,
            'metrics': 'SplyCur,CapMrktCurUSD',
            'frequency': '1d',
            'start_time': '2020-01-01',
            'end_time': (datetime.utcnow() + timedelta(days=1)).strftime('%Y-%m-%d'),
            'page_size': 10000,
        }

        rows = []
        page_url = url
        pages = 0
        while page_url:
            try:
                r = requests.get(page_url, params=params if pages == 0 else None, timeout=30)
                r.raise_for_status()
                data = r.json()
                if 'error' in data:
                    print(f"Error: {data['error']['message'][:60]}")
                    break
                batch = data.get('data', [])
                rows.extend(batch)
                page_url = data.get('next_page_url')
                params = None
                pages += 1
            except requests.exceptions.RequestException as e:
                print(f"Error: {e}")
                break
            time.sleep(0.3)

        for row in rows:
            row['stablecoin'] = name
        all_rows.extend(rows)
        print(f"{len(rows)} rows")

    if not all_rows:
        print("No data!")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df['date'] = pd.to_datetime(df['time']).dt.tz_localize(None)
    df['supply'] = pd.to_numeric(df['SplyCur'], errors='coerce')
    df['mcap'] = pd.to_numeric(df['CapMrktCurUSD'], errors='coerce')

    # Pivot: one row per date with USDT_supply, USDC_supply, etc.
    result = pd.DataFrame()
    for name in stables.values():
        sub = df[df['stablecoin'] == name][['date', 'supply', 'mcap']].copy()
        sub = sub.rename(columns={'supply': f'{name}_supply', 'mcap': f'{name}_mcap'})
        if result.empty:
            result = sub
        else:
            result = result.merge(sub, on='date', how='outer')

    result = result.sort_values('date')

    # Total stablecoin supply
    supply_cols = [c for c in result.columns if c.endswith('_supply')]
    result['total_stable_supply'] = result[supply_cols].sum(axis=1)

    # Changes
    for col in supply_cols + ['total_stable_supply']:
        result[f'{col}_chg7d'] = result[col].pct_change(7)
        result[f'{col}_chg30d'] = result[col].pct_change(30)

    out_path = DATA_DIR / "stablecoin_supply.parquet"
    result.to_parquet(out_path, index=False)
    print(f"\nSaved: {result.shape} to {out_path}")
    print(f"  Date range: {result['date'].min()} → {result['date'].max()}")
    print(f"  Columns: {list(result.columns)}")
    return result


# ─── 3. Blockchain.com BTC On-Chain ───────────────────────────────────────────
def download_btc_onchain():
    """Download BTC on-chain metrics from Blockchain.com."""
    print("\n" + "=" * 60)
    print("Blockchain.com BTC On-Chain")
    print("=" * 60)

    charts = {
        'mempool-size': 'btc_mempool_bytes',
        'n-transactions': 'btc_n_transactions',
        'estimated-transaction-volume-usd': 'btc_tx_volume_usd',
        'miners-revenue': 'btc_miners_revenue_usd',
        'hash-rate': 'btc_hashrate',
        'n-unique-addresses': 'btc_unique_addresses',
        'difficulty': 'btc_difficulty',
    }

    result = pd.DataFrame()

    for chart_name, col_name in charts.items():
        print(f"  {chart_name}...", end=" ", flush=True)
        try:
            r = requests.get(
                f'https://api.blockchain.info/charts/{chart_name}',
                params={'timespan': '6years', 'rollingAverage': '24hours', 'format': 'json'},
                timeout=30
            )
            r.raise_for_status()
            data = r.json()
            values = data.get('values', [])

            if values:
                df_chart = pd.DataFrame(values)
                df_chart['date'] = pd.to_datetime(df_chart['x'], unit='s').dt.normalize()
                df_chart = df_chart.rename(columns={'y': col_name})[['date', col_name]]
                df_chart = df_chart.drop_duplicates('date')

                if result.empty:
                    result = df_chart
                else:
                    result = result.merge(df_chart, on='date', how='outer')

                print(f"{len(values)} points")
            else:
                print("empty")

        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")

        time.sleep(0.5)

    if result.empty:
        print("No data!")
        return pd.DataFrame()

    result = result.sort_values('date')

    # Derived features
    for col in ['btc_mempool_bytes', 'btc_n_transactions', 'btc_tx_volume_usd',
                'btc_miners_revenue_usd', 'btc_unique_addresses']:
        if col in result.columns:
            result[f'{col}_chg7d'] = result[col].pct_change(7)

    out_path = DATA_DIR / "btc_onchain.parquet"
    result.to_parquet(out_path, index=False)
    print(f"\nSaved: {result.shape} to {out_path}")
    print(f"  Date range: {result['date'].min()} → {result['date'].max()}")
    print(f"  Columns: {list(result.columns)}")
    return result


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    vol_df = download_spot_futures_volume()
    stable_df = download_stablecoin_supply()
    btc_df = download_btc_onchain()

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"DONE in {elapsed:.0f}s")
    print(f"{'=' * 60}")
    if not vol_df.empty:
        print(f"  Spot/Futures Volume: {vol_df.shape[0]} rows, {vol_df['symbol'].nunique()} symbols")
    if not stable_df.empty:
        print(f"  Stablecoin Supply:   {stable_df.shape[0]} rows")
    if not btc_df.empty:
        print(f"  BTC On-Chain:        {btc_df.shape[0]} rows")


if __name__ == "__main__":
    main()
