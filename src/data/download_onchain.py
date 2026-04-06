#!/usr/bin/env python3
"""
Download on-chain + DeFi features from free APIs:
1. CoinMetrics Community — on-chain metrics for 9 coins (daily, 2020+)
2. DeFi Llama — TVL per chain and per protocol (daily)

Output: data/sentiment/onchain_daily.parquet
        data/sentiment/defi_tvl_daily.parquet

These are daily-resolution features that will be forward-filled to hourly in the pipeline.
"""
import os
import sys
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# ─── Config ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "sentiment"
DATA_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2020-01-01"
END_DATE = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")

# CoinMetrics free metrics (verified available for community tier)
CM_FREE_METRICS = [
    "AdrActCnt",       # Active addresses count
    "TxCnt",           # Transaction count
    "HashRate",        # Hash rate (PoW only: BTC, DOGE, LTC)
    "CapMrktCurUSD",   # Market cap USD
    "AdrBalCnt",       # Addresses with non-zero balance
    "BlkCnt",          # Block count
    "FlowInExUSD",     # Exchange inflow USD (BTC, ETH only)
    "FlowOutExUSD",    # Exchange outflow USD (BTC, ETH only)
    "SplyCur",         # Current supply
]

# Coins available in CoinMetrics community tier (verified)
CM_COINS = {
    "btc":  "BTC/USDT",
    "eth":  "ETH/USDT",
    "xrp":  "XRP/USDT",
    "ada":  "ADA/USDT",
    "doge": "DOGE/USDT",
    "link": "LINK/USDT",
    "uni":  "UNI/USDT",
    "ltc":  "LTC/USDT",
    "aave": "AAVE/USDT",
    "bnb":  "BNB/USDT",
    "dot":  "DOT/USDT",
}

# DeFi Llama: chain name → symbol
CHAIN_MAP = {
    "Ethereum":  "ETH/USDT",
    "Solana":    "SOL/USDT",
    "BSC":       "BNB/USDT",
    "Avalanche": "AVAX/USDT",
    "Polygon":   "MATIC/USDT",
    "Arbitrum":  "ARB/USDT",
    "Optimism":  "OP/USDT",
    "Near":      "NEAR/USDT",
    "Fantom":    "FTM/USDT",
    "Algorand":  "ALGO/USDT",
    "Cosmos":    "ATOM/USDT",
    "Polkadot":  "DOT/USDT",
    "Cardano":   "ADA/USDT",
    "Bitcoin":   "BTC/USDT",
}

# DeFi Llama: protocol slug → token symbol
PROTOCOL_MAP = {
    "lido":              "LDO/USDT",
    "aave":              "AAVE/USDT",
    "uniswap":           "UNI/USDT",
    "curve-dex":         "CRV/USDT",
    "synthetix":         "SNX/USDT",
    "makerdao":          "MKR/USDT",
    "sushiswap":         "SUSHI/USDT",
    "yearn-finance":     "YFI/USDT",
    "compound-finance":  "COMP/USDT",
    "injective":         "INJ/USDT",
    "thorchain":         "RUNE/USDT",
}

RATE_LIMIT_SLEEP = 0.5  # seconds between API calls


# ─── CoinMetrics Download ─────────────────────────────────────────────────────
def download_coinmetrics():
    """Download on-chain metrics from CoinMetrics community API."""
    print("=" * 60)
    print("CoinMetrics Community API — On-Chain Data")
    print("=" * 60)

    base_url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    all_rows = []

    for cm_ticker, our_symbol in CM_COINS.items():
        print(f"\n  {our_symbol} ({cm_ticker})...")
        
        for metric in CM_FREE_METRICS:
            page_url = base_url
            params = {
                "assets": cm_ticker,
                "metrics": metric,
                "frequency": "1d",
                "start_time": START_DATE,
                "end_time": END_DATE,
                "page_size": 10000,
            }
            
            metric_rows = []
            pages = 0
            while page_url:
                try:
                    r = requests.get(page_url, params=params if pages == 0 else None, timeout=30)
                    r.raise_for_status()
                    data = r.json()
                    
                    if "error" in data:
                        # Metric not available for this asset (expected for some combos)
                        break
                    
                    rows = data.get("data", [])
                    metric_rows.extend(rows)
                    
                    # Pagination
                    page_url = data.get("next_page_url")
                    params = None  # next_page_url has params embedded
                    pages += 1
                    
                except requests.exceptions.RequestException as e:
                    print(f"    Error fetching {metric} for {cm_ticker}: {e}")
                    break
                
                time.sleep(RATE_LIMIT_SLEEP)
            
            if metric_rows:
                for row in metric_rows:
                    row["symbol"] = our_symbol
                    row["metric_name"] = metric
                all_rows.extend(metric_rows)
                print(f"    {metric}: {len(metric_rows)} rows", end="")
            
        print()

    if not all_rows:
        print("No CoinMetrics data downloaded!")
        return pd.DataFrame()

    # Parse into wide format
    df = pd.DataFrame(all_rows)
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df = df.rename(columns={"time": "date"})
    
    # Pivot metrics into columns
    records = []
    for (date, symbol), group in df.groupby(["date", "symbol"]):
        row = {"date": date, "symbol": symbol}
        for _, r in group.iterrows():
            metric = r["metric_name"]
            val = r.get(metric)
            if val is not None:
                try:
                    row[metric] = float(val)
                except (ValueError, TypeError):
                    pass
        records.append(row)
    
    result = pd.DataFrame(records)
    
    # Compute derived features
    result = result.sort_values(["symbol", "date"])
    for col in ["AdrActCnt", "TxCnt", "CapMrktCurUSD", "AdrBalCnt"]:
        if col in result.columns:
            # 7-day change
            result[f"{col}_chg7d"] = result.groupby("symbol")[col].pct_change(7)
            # 30-day change
            result[f"{col}_chg30d"] = result.groupby("symbol")[col].pct_change(30)
    
    # Exchange flow metrics (BTC/ETH only)
    if "FlowInExUSD" in result.columns and "FlowOutExUSD" in result.columns:
        result["FlowNetExUSD"] = result["FlowOutExUSD"] - result["FlowInExUSD"]
        # 7-day rolling sum
        result["FlowNetEx7d"] = result.groupby("symbol")["FlowNetExUSD"].transform(
            lambda x: x.rolling(7, min_periods=1).sum()
        )
    
    out_path = DATA_DIR / "onchain_daily.parquet"
    result.to_parquet(out_path, index=False)
    print(f"\nSaved CoinMetrics: {result.shape} to {out_path}")
    print(f"  Symbols: {sorted(result['symbol'].unique())}")
    print(f"  Date range: {result['date'].min()} → {result['date'].max()}")
    print(f"  Columns: {list(result.columns)}")
    
    return result


# ─── DeFi Llama Download ──────────────────────────────────────────────────────
def download_defi_llama():
    """Download TVL data from DeFi Llama (free, no key needed)."""
    print("\n" + "=" * 60)
    print("DeFi Llama — TVL Data")
    print("=" * 60)

    all_data = []

    # 1. Chain-level TVL
    print("\n  Chain-level TVL:")
    for chain, symbol in CHAIN_MAP.items():
        try:
            r = requests.get(f"https://api.llama.fi/v2/historicalChainTvl/{chain}", timeout=30)
            r.raise_for_status()
            data = r.json()
            
            if isinstance(data, list) and len(data) > 0:
                for row in data:
                    ts = row.get("date")
                    tvl = row.get("tvl")
                    if ts and tvl:
                        all_data.append({
                            "date": pd.Timestamp.utcfromtimestamp(ts).normalize(),
                            "symbol": symbol,
                            "chain_tvl_usd": float(tvl),
                            "source": "chain",
                        })
                print(f"    {chain:12s} → {symbol:12s}: {len(data)} days")
            
        except requests.exceptions.RequestException as e:
            print(f"    {chain}: Error - {e}")
        
        time.sleep(RATE_LIMIT_SLEEP)

    # 2. Protocol-level TVL
    print("\n  Protocol-level TVL:")
    for slug, symbol in PROTOCOL_MAP.items():
        try:
            r = requests.get(f"https://api.llama.fi/protocol/{slug}", timeout=30)
            r.raise_for_status()
            data = r.json()
            
            tvl_hist = data.get("tvl", [])
            if tvl_hist:
                for row in tvl_hist:
                    ts = row.get("date")
                    tvl = row.get("totalLiquidityUSD")
                    if ts and tvl:
                        all_data.append({
                            "date": pd.Timestamp.utcfromtimestamp(ts).normalize(),
                            "symbol": symbol,
                            "protocol_tvl_usd": float(tvl),
                            "source": "protocol",
                        })
                print(f"    {slug:25s} → {symbol:12s}: {len(tvl_hist)} days")
            
        except requests.exceptions.RequestException as e:
            print(f"    {slug}: Error - {e}")
        
        time.sleep(RATE_LIMIT_SLEEP)

    if not all_data:
        print("No DeFi Llama data downloaded!")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    
    # Aggregate: for symbols with both chain + protocol TVL, keep both
    # Group by date + symbol, aggregate
    chain_df = df[df["source"] == "chain"][["date", "symbol", "chain_tvl_usd"]].copy()
    proto_df = df[df["source"] == "protocol"][["date", "symbol", "protocol_tvl_usd"]].copy()
    
    # Merge chain and protocol
    result = chain_df.merge(proto_df, on=["date", "symbol"], how="outer")
    result = result.sort_values(["symbol", "date"])
    
    # Compute derived features
    for col in ["chain_tvl_usd", "protocol_tvl_usd"]:
        if col in result.columns:
            result[f"{col}_chg7d"] = result.groupby("symbol")[col].pct_change(7)
            result[f"{col}_chg30d"] = result.groupby("symbol")[col].pct_change(30)
    
    # Filter to our date range
    result = result[result["date"] >= START_DATE]
    
    out_path = DATA_DIR / "defi_tvl_daily.parquet"
    result.to_parquet(out_path, index=False)
    print(f"\nSaved DeFi Llama: {result.shape} to {out_path}")
    print(f"  Symbols: {sorted(result['symbol'].unique())}")
    print(f"  Date range: {result['date'].min()} → {result['date'].max()}")
    print(f"  Columns: {list(result.columns)}")
    
    return result


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"Download period: {START_DATE} → {END_DATE}")
    print(f"Output directory: {DATA_DIR}")
    
    t0 = time.time()
    
    cm_df = download_coinmetrics()
    tvl_df = download_defi_llama()
    
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"DONE in {elapsed:.0f}s")
    print(f"{'=' * 60}")
    
    if not cm_df.empty:
        print(f"  CoinMetrics: {cm_df.shape[0]} rows, {cm_df['symbol'].nunique()} symbols")
    if not tvl_df.empty:
        print(f"  DeFi Llama:  {tvl_df.shape[0]} rows, {tvl_df['symbol'].nunique()} symbols")


if __name__ == "__main__":
    main()
