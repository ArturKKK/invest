"""Incremental OHLCV updater using data.binance.vision (geoblock-safe).

For each parquet in data/raw/*_1h.parquet:
  - read last timestamp
  - download monthly + daily klines zips covering (last_ts, now)
  - parse CSVs, dedupe, append, save

Endpoints (no auth, no rate-limit, public):
  monthly: https://data.binance.vision/data/spot/monthly/klines/{SYM}/{INT}/{SYM}-{INT}-{YYYY-MM}.zip
  daily:   https://data.binance.vision/data/spot/daily/klines/{SYM}/{INT}/{SYM}-{INT}-{YYYY-MM-DD}.zip

CSV columns: open_time, open, high, low, close, volume, close_time, quote_volume,
             count, taker_buy_volume, taker_buy_quote_volume, ignore
open_time is in ms (newer files) OR microseconds (some older files >1e15).
"""
from __future__ import annotations

import io
import sys
import time
import zipfile
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

DATA_DIR = Path("data/raw")
TIMEFRAME = "1h"
BASE = "https://data.binance.vision/data/spot"
KCOLS = ["open_time","open","high","low","close","volume",
        "close_time","quote_volume","count","taker_buy_volume",
        "taker_buy_quote_volume","ignore"]
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "ohlcv-updater/1.0"})


def fetch_zip_csv(url: str) -> pd.DataFrame:
    r = SESSION.get(url, timeout=30)
    if r.status_code == 404:
        return pd.DataFrame()
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            df = pd.read_csv(f, header=None, names=KCOLS)
    if df.empty:
        return df
    # Some 2025+ files have header row as first data row
    if str(df.iloc[0, 0]).lower() == "open_time":
        df = df.iloc[1:].reset_index(drop=True)
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df = df.dropna(subset=["open_time"]).copy()
    # Normalize ms vs us
    if df["open_time"].iloc[0] > 1e14:
        df["timestamp"] = pd.to_datetime(df["open_time"].astype("int64"), unit="us", utc=True)
    else:
        df["timestamp"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    for c in ("open","high","low","close","volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["timestamp","open","high","low","close","volume"]]


def vision_symbol(local_sym: str) -> str:
    # local "BTC_USDT" -> vision "BTCUSDT"
    return local_sym.replace("_", "").upper()


def month_iter(start: datetime, end: datetime):
    cur = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    end_m = datetime(end.year, end.month, 1, tzinfo=timezone.utc)
    while cur <= end_m:
        yield cur
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)


def day_iter(start: datetime, end: datetime):
    cur = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
    end_d = datetime(end.year, end.month, end.day, tzinfo=timezone.utc)
    while cur <= end_d:
        yield cur
        cur += timedelta(days=1)


def fetch_range(local_sym: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    sym = vision_symbol(local_sym)
    parts: list[pd.DataFrame] = []
    # Use monthly for any complete past month (i.e. month strictly before end's month)
    # otherwise daily. Simpler: fetch monthly for any month <= (end-30d), daily for the rest.
    today_utc = datetime.now(timezone.utc)
    # Fetch full monthly archives for completed months
    for m in month_iter(start.to_pydatetime(), end.to_pydatetime()):
        # if month is current (not yet finished), skip — use daily
        if m.year == today_utc.year and m.month == today_utc.month:
            continue
        if m.year == end.year and m.month == end.month and end.day < 28:
            # might be incomplete, fall back to daily
            continue
        url = f"{BASE}/monthly/klines/{sym}/{TIMEFRAME}/{sym}-{TIMEFRAME}-{m:%Y-%m}.zip"
        df = fetch_zip_csv(url)
        if not df.empty:
            parts.append(df)
    # Fetch daily for days NOT covered by monthly we just downloaded
    covered_months = set()
    for p in parts:
        for ts in p["timestamp"].dt.to_period("M").unique():
            covered_months.add((ts.year, ts.month))
    for d in day_iter(start.to_pydatetime(), end.to_pydatetime()):
        if (d.year, d.month) in covered_months:
            continue
        url = f"{BASE}/daily/klines/{sym}/{TIMEFRAME}/{sym}-{TIMEFRAME}-{d:%Y-%m-%d}.zip"
        df = fetch_zip_csv(url)
        if not df.empty:
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return out


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA_DIR.glob("*_1h.parquet"))
    if not files:
        print("No existing _1h.parquet files in data/raw/")
        sys.exit(1)

    now_utc = pd.Timestamp.now(tz="UTC")
    print(f"Updating {len(files)} OHLCV files (target = {now_utc})")
    total_added = 0
    summary = []
    for fp in files:
        local_sym = fp.stem.replace("_1h", "")
        existing = pd.read_parquet(fp)
        if existing.empty:
            print(f"  ⚠ {fp.name}: empty file, skip")
            continue
        existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
        last_ts = existing["timestamp"].max()
        start = last_ts + pd.Timedelta(hours=1)
        if start >= now_utc:
            print(f"  · {local_sym}: up-to-date ({last_ts})")
            continue
        try:
            new_df = fetch_range(local_sym, start, now_utc)
        except Exception as e:
            print(f"  ⚠ {local_sym}: fetch error {e!r} — skip")
            continue
        if new_df.empty:
            print(f"  · {local_sym}: no new bars (vision)")
            continue
        new_df = new_df[new_df["timestamp"] > last_ts]
        if new_df.empty:
            continue
        backup = fp.with_suffix(".parquet.bak")
        if not backup.exists():
            existing.to_parquet(backup, index=False)
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged = merged.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        merged.to_parquet(fp, index=False)
        n_added = len(merged) - len(existing)
        total_added += n_added
        new_last = merged["timestamp"].max()
        summary.append((local_sym, last_ts, new_last, n_added))
        print(f"  ✓ {local_sym}: +{n_added} bars  ({last_ts} → {new_last})")
        time.sleep(0.05)

    print()
    print(f"Done. Total bars added: {total_added}")
    if summary:
        print(f"{'symbol':<14} {'old_max':<32} {'new_max':<32} added")
        for s, a, b, n in summary:
            print(f"{s:<14} {str(a):<32} {str(b):<32} {n}")


if __name__ == "__main__":
    main()
