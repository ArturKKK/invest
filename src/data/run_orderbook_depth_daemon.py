#!/usr/bin/env python3
"""
Run a lightweight local daemon for hourly orderbook depth collection.

This is a convenience runner for unattended local accumulation while the user
is away. It reuses the one-shot D6 collector and feature builder.

Usage:
  python src/data/run_orderbook_depth_daemon.py
  python src/data/run_orderbook_depth_daemon.py --max-cycles 3
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
ROOT = os.path.abspath(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.download_binance_depth import RAW_FILE, SYMBOLS, fetch_all_snapshots, save_snapshots
from src.features.build_orderbook_depth_features import OUT_FILE, build_feature_frame


def rebuild_features() -> int:
    raw_df = pd.read_parquet(RAW_FILE)
    feature_df = build_feature_frame(raw_df)
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    feature_df.to_parquet(OUT_FILE, index=False)
    return len(feature_df)


def seconds_until_next_hour() -> float:
    now = pd.Timestamp.utcnow()
    next_hour = now.ceil("h")
    return max(1.0, (next_hour - now).total_seconds())


def run_cycle(cycle_num: int) -> None:
    started = pd.Timestamp.utcnow()
    print(f"[{started}] D6 daemon cycle {cycle_num} — fetching depth snapshots...", flush=True)
    snapshots = fetch_all_snapshots(SYMBOLS)
    if snapshots.empty:
        print(f"[{pd.Timestamp.utcnow()}] D6 daemon cycle {cycle_num} — no snapshots fetched", flush=True)
        return

    n_saved = save_snapshots(snapshots)
    n_feat = rebuild_features()
    ended = pd.Timestamp.utcnow()
    print(
        f"[{ended}] D6 daemon cycle {cycle_num} — saved {n_saved} snapshots, "
        f"feature rows now {n_feat}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hourly D6 orderbook collection locally")
    parser.add_argument("--max-cycles", type=int, default=0, help="Stop after N cycles; 0 means forever")
    parser.add_argument("--run-now", action="store_true", help="Run one cycle immediately before waiting for the next hour")
    args = parser.parse_args()

    cycle_num = 0
    if args.run_now:
        cycle_num += 1
        run_cycle(cycle_num)
        if args.max_cycles and cycle_num >= args.max_cycles:
            print(f"[{pd.Timestamp.utcnow()}] D6 daemon reached max cycles={args.max_cycles}", flush=True)
            return

    while True:
        if args.max_cycles and cycle_num >= args.max_cycles:
            print(f"[{pd.Timestamp.utcnow()}] D6 daemon reached max cycles={args.max_cycles}", flush=True)
            break
        sleep_seconds = seconds_until_next_hour()
        wake_ts = pd.Timestamp.utcnow() + pd.Timedelta(seconds=sleep_seconds)
        print(f"[{pd.Timestamp.utcnow()}] D6 daemon sleeping until {wake_ts}", flush=True)
        time.sleep(sleep_seconds)
        cycle_num += 1
        run_cycle(cycle_num)


if __name__ == "__main__":
    main()