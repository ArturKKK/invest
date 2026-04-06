#!/usr/bin/env python3
"""
Download hourly Binance spot orderbook depth snapshots.

Public endpoint, no API key required:
  GET /api/v3/depth

This collector stores one snapshot row per symbol per fetch with enough raw
summary data to derive orderbook features later.

Outputs:
  - data/raw/orderbook_depth/binance_orderbook_depth_snapshots.parquet

Usage:
  python src/data/download_binance_depth.py
  python src/data/download_binance_depth.py --symbol BTCUSDT --dry-run
"""

from __future__ import annotations

import json
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

API_BASE = "https://data-api.binance.vision/api/v3"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
RAW_DIR = os.path.join(ROOT, "data", "raw", "orderbook_depth")
RAW_FILE = os.path.join(RAW_DIR, "binance_orderbook_depth_snapshots.parquet")

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "MATICUSDT", "UNIUSDT", "ATOMUSDT", "LTCUSDT", "ETCUSDT",
    "FILUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "NEARUSDT",
    "AAVEUSDT", "MKRUSDT", "GRTUSDT", "INJUSDT", "FTMUSDT",
    "ALGOUSDT", "SANDUSDT", "MANAUSDT", "AXSUSDT", "THETAUSDT",
    "RUNEUSDT", "EGLDUSDT", "XTZUSDT", "FLOWUSDT", "CHZUSDT",
    "CRVUSDT", "LDOUSDT", "SNXUSDT", "COMPUSDT", "YFIUSDT",
    "SUSHIUSDT", "ENJUSDT", "BATUSDT", "ZILUSDT", "ONEUSDT",
    "IOTAUSDT", "ICXUSDT", "ENSUSDT", "IMXUSDT", "GALAUSDT",
]


def to_our_symbol(binance_symbol: str) -> str:
    return binance_symbol.replace("USDT", "/USDT")


def fetch_depth(symbol: str, limit: int = 20, retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            response = requests.get(
                f"{API_BASE}/depth",
                params={"symbol": symbol, "limit": limit},
                timeout=10,
                verify=False,
            )
            if response.status_code == 200:
                return response.json()
            if response.status_code in (400, 404):
                return None
            if response.status_code == 429:
                time.sleep(1 + attempt)
                continue
        except Exception:
            time.sleep(1 + attempt)
    return None


def summarize_side(levels: list[list[str]], depth_levels: int = 10) -> tuple[float, float, float]:
    selected = levels[:depth_levels]
    if not selected:
        return 0.0, 0.0, 0.0
    prices = np.array([float(level[0]) for level in selected], dtype=float)
    qtys = np.array([float(level[1]) for level in selected], dtype=float)
    notional = float(np.sum(prices * qtys))
    quantity = float(np.sum(qtys))
    best_price = float(prices[0])
    return notional, quantity, best_price


def summarize_book(symbol: str, book: dict, snapshot_ts: pd.Timestamp) -> dict[str, object]:
    bids = book.get("bids", [])
    asks = book.get("asks", [])

    bid_depth_top10, bid_qty_top10, best_bid = summarize_side(bids, depth_levels=10)
    ask_depth_top10, ask_qty_top10, best_ask = summarize_side(asks, depth_levels=10)
    bid_depth_top5, _, _ = summarize_side(bids, depth_levels=5)
    ask_depth_top5, _, _ = summarize_side(asks, depth_levels=5)

    total_depth = bid_depth_top10 + ask_depth_top10
    imbalance_ratio = (bid_depth_top10 - ask_depth_top10) / (total_depth + 1e-10)
    mid_price = 0.5 * (best_bid + best_ask) if best_bid > 0 and best_ask > 0 else np.nan
    spread_bps = ((best_ask - best_bid) / (mid_price + 1e-10) * 10000) if pd.notna(mid_price) else np.nan

    return {
        "timestamp": snapshot_ts,
        "timestamp_hour": snapshot_ts.floor("h"),
        "symbol": to_our_symbol(symbol),
        "exchange_symbol": symbol,
        "last_update_id": int(book.get("lastUpdateId", 0)),
        "bid_depth_top10": bid_depth_top10,
        "ask_depth_top10": ask_depth_top10,
        "bid_qty_top10": bid_qty_top10,
        "ask_qty_top10": ask_qty_top10,
        "bid_depth_top5": bid_depth_top5,
        "ask_depth_top5": ask_depth_top5,
        "imbalance_ratio": imbalance_ratio,
        "mid_price": mid_price,
        "spread_bps": spread_bps,
        "bid_levels_json": json.dumps(bids[:10]),
        "ask_levels_json": json.dumps(asks[:10]),
    }


def fetch_symbol_snapshot(symbol: str, limit: int, snapshot_ts: pd.Timestamp) -> dict[str, object] | None:
    book = fetch_depth(symbol, limit=limit)
    if not book:
        return None
    return summarize_book(symbol, book, snapshot_ts)


def fetch_all_snapshots(symbols: list[str], limit: int = 20, max_workers: int = 8) -> pd.DataFrame:
    snapshot_ts = pd.Timestamp(datetime.now(timezone.utc)).floor("s")
    rows: list[dict[str, object]] = []
    errors = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_symbol_snapshot, symbol, limit, snapshot_ts): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                row = future.result()
                if row is not None:
                    rows.append(row)
            except Exception:
                errors += 1
                print(f"  Warning: failed to fetch {symbol}")

    if errors:
        print(f"  Fetch errors: {errors}")
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def save_snapshots(df: pd.DataFrame) -> int:
    os.makedirs(RAW_DIR, exist_ok=True)
    if os.path.exists(RAW_FILE):
        existing = pd.read_parquet(RAW_FILE)
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df.copy()

    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
    combined["timestamp_hour"] = pd.to_datetime(combined["timestamp_hour"], utc=True)
    combined = combined.drop_duplicates(["symbol", "timestamp"], keep="last")
    combined = combined.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    combined.to_parquet(RAW_FILE, index=False)
    return len(df)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Download Binance orderbook depth snapshots")
    parser.add_argument("--symbol", action="append", default=None, help="Single Binance symbol, e.g. BTCUSDT")
    parser.add_argument("--limit", type=int, default=20, help="Depth levels to fetch from Binance")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    symbols = args.symbol or SYMBOLS
    print(f"Downloading Binance depth snapshots for {len(symbols)} symbols...")
    snapshots = fetch_all_snapshots(symbols, limit=args.limit, max_workers=args.max_workers)
    if snapshots.empty:
        raise RuntimeError("No orderbook snapshots fetched")

    print(
        "Fetched: "
        f"{len(snapshots)} rows, range={snapshots['timestamp'].min()} -> {snapshots['timestamp'].max()}"
    )
    print(
        "Median depth: "
        f"bid_top10=${snapshots['bid_depth_top10'].median():,.0f}, "
        f"ask_top10=${snapshots['ask_depth_top10'].median():,.0f}"
    )

    if not args.dry_run:
        n_saved = save_snapshots(snapshots)
        print(f"Saved {n_saved} new snapshot rows to {RAW_FILE}")
    else:
        print("Dry run: not saving snapshots")


if __name__ == "__main__":
    main()