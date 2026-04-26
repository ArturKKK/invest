"""Audit data freshness on VM — what's the max timestamp of every input feed."""
import os
import pandas as pd
from pathlib import Path

CHECKS = [
    "data/raw/BTC_USDT_1h.parquet",
    "data/sentiment/binance_funding_rates.parquet",
    "data/sentiment/binance_futures_metrics.parquet",
    "data/sentiment/binance_premium_index.parquet",
    "data/sentiment/btc_onchain.parquet",
    "data/sentiment/fear_greed.parquet",
    "data/sentiment/funding_rates.parquet",
    "data/sentiment/deribit_dvol.parquet",
    "data/sentiment/defi_tvl_daily.parquet",
    "data/sentiment/llama_stablecoin_chains.parquet",
    "data/features/crypto_features_1h.parquet",
    "data/features/spot_hourly_taker.parquet",
    "data/features/cc_social_daily.parquet",
    "data/features/google_trends.parquet",
    "data/features/binance_orderbook_depth_features.parquet",
]


def time_col(df):
    for c in ("timestamp", "time", "date", "datetime", "ts"):
        if c in df.columns:
            return c
    return None


def main():
    print(f"{'PATH':<60s}  {'TIME_COL':<12s}  {'MAX':<32s}  {'ROWS':>10s}")
    for p in CHECKS:
        if not os.path.exists(p):
            print(f"{p:<60s}  {'MISSING':<12s}")
            continue
        try:
            df = pd.read_parquet(p)
            tc = time_col(df)
            if tc is None:
                print(f"{p:<60s}  {'?':<12s}  cols={list(df.columns)[:5]}")
                continue
            maxv = df[tc].max()
            print(f"{p:<60s}  {tc:<12s}  {str(maxv):<32s}  {len(df):>10d}")
        except Exception as e:
            print(f"{p}  ERR {e}")


if __name__ == "__main__":
    main()
