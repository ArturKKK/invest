#!/usr/bin/env python3
"""
Build hourly orderbook depth features from raw Binance snapshots.

Inputs:
  - data/raw/orderbook_depth/binance_orderbook_depth_snapshots.parquet

Outputs:
  - data/features/binance_orderbook_depth_features.parquet

Primary features for D6:
  - bid_depth_top10
  - ask_depth_top10
  - imbalance_ratio
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
RAW_FILE = os.path.join(ROOT, "data", "raw", "orderbook_depth", "binance_orderbook_depth_snapshots.parquet")
OUT_DIR = os.path.join(ROOT, "data", "features")
OUT_FILE = os.path.join(OUT_DIR, "binance_orderbook_depth_features.parquet")


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(6, window // 4)).mean()
    std = series.rolling(window, min_periods=max(6, window // 4)).std()
    return (series - mean) / (std + 1e-10)


def build_feature_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    frame = raw_df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["timestamp_hour"] = pd.to_datetime(frame["timestamp_hour"], utc=True)

    latest = (
        frame.sort_values(["symbol", "timestamp"])
        .groupby(["symbol", "timestamp_hour"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    latest["timestamp"] = latest["timestamp_hour"]
    latest = latest.drop(columns=["timestamp_hour"], errors="ignore")

    latest = latest[[
        "timestamp",
        "symbol",
        "bid_depth_top10",
        "ask_depth_top10",
        "imbalance_ratio",
        "bid_depth_top5",
        "ask_depth_top5",
        "spread_bps",
        "mid_price",
    ]].copy()

    latest = latest.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    latest["bid_ask_depth_ratio"] = latest["bid_depth_top10"] / (latest["ask_depth_top10"] + 1e-10)
    latest["depth_total_top10"] = latest["bid_depth_top10"] + latest["ask_depth_top10"]

    for col in ["bid_depth_top10", "ask_depth_top10", "imbalance_ratio", "depth_total_top10"]:
        latest[f"{col}_z24"] = latest.groupby("symbol")[col].transform(lambda s: rolling_zscore(s, 24))

    for col in ["bid_depth_top10", "ask_depth_top10", "depth_total_top10"]:
        latest[f"{col}_chg24h"] = latest.groupby("symbol")[col].pct_change(24)

    return latest


def main() -> None:
    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError(f"Missing raw orderbook snapshot file: {RAW_FILE}")

    os.makedirs(OUT_DIR, exist_ok=True)
    raw_df = pd.read_parquet(RAW_FILE)
    feature_df = build_feature_frame(raw_df)
    feature_df.to_parquet(OUT_FILE, index=False)

    print(f"Built orderbook depth features: {len(feature_df):,} rows")
    print(f"Symbols: {feature_df['symbol'].nunique()}")
    print(f"Range: {feature_df['timestamp'].min()} -> {feature_df['timestamp'].max()}")
    print(f"Saved to: {OUT_FILE}")


if __name__ == "__main__":
    main()