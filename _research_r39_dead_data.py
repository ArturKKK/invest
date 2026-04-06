#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R39.1 — inspect downloaded-but-unused data files.

Goal:
  - inspect stablecoin_supply.parquet
  - inspect defi_tvl_daily.parquet
  - inspect onchain_daily.parquet
  - produce a durable summary for follow-up feature engineering
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "sentiment"

FILES = [
    DATA_DIR / "stablecoin_supply.parquet",
    DATA_DIR / "defi_tvl_daily.parquet",
    DATA_DIR / "onchain_daily.parquet",
]


def detect_time_column(df: pd.DataFrame) -> str | None:
    for column in ("timestamp", "date", "time", "datetime"):
        if column in df.columns:
            return column
    return None


def detect_resolution_hours(df: pd.DataFrame, time_col: str) -> float | None:
    series = pd.to_datetime(df[time_col], errors="coerce").dropna().sort_values().drop_duplicates()
    if len(series) < 3:
        return None
    deltas = series.diff().dropna().dt.total_seconds() / 3600
    if len(deltas) == 0:
        return None
    return float(deltas.median())


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def summarize_file(path: Path) -> dict:
    if not path.exists():
        print(f"\n[missing] {path.name}")
        return {"file": path.name, "status": "missing"}

    df = pd.read_parquet(path)
    time_col = detect_time_column(df)
    symbol_count = int(df["symbol"].nunique()) if "symbol" in df.columns else 0

    print_section(path.name)
    print(f"shape: {df.shape}")
    print(f"columns ({len(df.columns)}): {list(df.columns)}")

    if time_col is not None:
        time_series = pd.to_datetime(df[time_col], errors="coerce")
        print(f"time column: {time_col}")
        print(f"date range: {time_series.min()} -> {time_series.max()}")
        resolution = detect_resolution_hours(df, time_col)
        if resolution is not None:
            print(f"median resolution: {resolution:.1f}h")
    else:
        resolution = None
        print("time column: not found")

    if "symbol" in df.columns:
        symbols = sorted(df["symbol"].dropna().astype(str).unique().tolist())
        preview = symbols[:15]
        suffix = " ..." if len(symbols) > len(preview) else ""
        print(f"symbols ({len(symbols)}): {preview}{suffix}")

    null_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    print("top null ratios:")
    for column, value in null_pct.head(12).items():
        print(f"  {column:<28} {value:>6.2f}%")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        print("top numeric columns by non-null count:")
        non_null = df[numeric_cols].notna().sum().sort_values(ascending=False)
        for column, value in non_null.head(10).items():
            print(f"  {column:<28} {int(value):>8}")

    candidate_lines = []
    if path.name == "stablecoin_supply.parquet":
        candidate_lines = [
            "total_stable_supply_chg7d / chg30d",
            "USDT_supply_chg7d, USDC_supply_chg7d",
            "stablecoin mcap breadth / dominance if total crypto cap available",
        ]
    elif path.name == "defi_tvl_daily.parquet":
        candidate_lines = [
            "chain_tvl_usd_chg7d / chg30d",
            "protocol_tvl_usd_chg7d / chg30d",
            "cross-chain TVL breadth by timestamp",
        ]
    elif path.name == "onchain_daily.parquet":
        candidate_lines = [
            "AdrActCnt_chg7d / chg30d",
            "TxCnt_chg7d / chg30d",
            "FlowNetExUSD / FlowNetEx7d for BTC, ETH",
        ]

    if candidate_lines:
        print("candidate features for next stage:")
        for line in candidate_lines:
            print(f"  - {line}")

    return {
        "file": path.name,
        "status": "ok",
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "time_col": time_col or "",
        "resolution_h": resolution if resolution is not None else np.nan,
        "symbols": symbol_count,
    }


def main() -> None:
    print_section("R39.1 — DEAD DATA INSPECTION")
    results = [summarize_file(path) for path in FILES]

    print_section("SUMMARY")
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))

    print("\nVerdict:")
    print("- stablecoin_supply.parquet: market-level regime candidate, cheap to test")
    print("- onchain_daily.parquet: likely usable only for subset of symbols; verify coverage before production use")
    print("- defi_tvl_daily.parquet: mostly regime/breadth use-case, likely not direct per-coin alpha")


if __name__ == "__main__":
    main()