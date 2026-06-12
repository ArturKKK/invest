"""Incremental OHLCV updater — appends fresh 1h candles to existing parquet files.

For each parquet in data/raw/*_1h.parquet:
  - read last timestamp
  - fetch new candles from (last_ts + 1h) to now via Binance public API
  - dedupe + append + save
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

DATA_DIR = Path("data/raw")
TIMEFRAME = "1h"
LIMIT = 1000


def fetch_incremental(exchange, symbol: str, since_ts: int) -> pd.DataFrame:
    """Fetch all candles from since_ts to now, paginating. Retries are BOUNDED
    (the unbounded loop once parked the updater on one symbol for hours)."""
    import ccxt
    rows = []
    cur = since_ts
    errors = 0
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=cur, limit=LIMIT)
            errors = 0
        except ccxt.BadSymbol:
            print(f"  ⚠ {symbol}: BadSymbol on Binance, skip")
            return pd.DataFrame()
        except Exception as e:
            errors += 1
            if errors >= 5:
                print(f"  ⚠ {symbol}: 5 consecutive errors, keeping what we have ({len(rows)} bars)")
                break
            print(f"  ⚠ {symbol}: error {e}, retry {errors}/5 in 5s")
            time.sleep(5)
            continue
        if not ohlcv:
            break
        rows.extend(ohlcv)
        last_ts = ohlcv[-1][0]
        if last_ts == cur:
            break
        cur = last_ts + 1
        time.sleep(0.1)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


def main():
    import ccxt
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA_DIR.glob("*_1h.parquet"))
    if not files:
        print("  No existing _1h.parquet files in data/raw/")
        sys.exit(1)

    exchange = ccxt.binance({"enableRateLimit": True, "timeout": 90000,
                             "options": {"defaultType": "spot"}})
    # ccxt 4.x ignores env HTTP(S)_PROXY — must set explicitly (one attr only)
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy:
        exchange.httpsProxy = proxy
    exchange.session.verify = False

    print(f"  Updating {len(files)} OHLCV files...")
    total_added = 0
    summary = []
    for fp in files:
        symbol = fp.stem.replace("_1h", "").replace("_", "/")
        existing = pd.read_parquet(fp)
        if existing.empty:
            print(f"  ⚠ {fp.name}: empty file, skip")
            continue
        existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
        last_ts = existing["timestamp"].max()
        # Resume from one hour after last bar
        since_ms = int((last_ts + pd.Timedelta(hours=1)).timestamp() * 1000)
        new_df = fetch_incremental(exchange, symbol, since_ms)
        if new_df.empty:
            summary.append((symbol, last_ts, last_ts, 0))
            print(f"  · {symbol}: up-to-date ({last_ts})")
            continue
        # Filter strictly newer
        new_df = new_df[new_df["timestamp"] > last_ts]
        if new_df.empty:
            summary.append((symbol, last_ts, last_ts, 0))
            continue
        # Backup once
        backup = fp.with_suffix(".parquet.bak")
        if not backup.exists():
            existing.to_parquet(backup, index=False)
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged = merged.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        merged.to_parquet(fp, index=False)
        n_added = len(merged) - len(existing)
        total_added += n_added
        new_last = merged["timestamp"].max()
        summary.append((symbol, last_ts, new_last, n_added))
        print(f"  ✓ {symbol}: +{n_added} bars  ({last_ts} → {new_last})")

    print(f"\n  Done. Total added: {total_added:,} bars across {len(files)} symbols.")
    print(f"\n  Summary:")
    for sym, old, new, n in summary[:5]:
        print(f"    {sym:<14s}  was: {old}  →  now: {new}  (+{n})")
    if len(summary) > 5:
        print(f"    ... ({len(summary)-5} more)")


if __name__ == "__main__":
    main()
