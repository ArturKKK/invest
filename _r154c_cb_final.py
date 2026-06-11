#!/usr/bin/env python3
"""R154c — Coinbase backfill, bulletproof edition.
1. Download ALL rows -> save RAW immediately (data/raw/coinbase/_backfill_raw.parquet).
2. Merge as a SEPARATE step (re-runnable without re-download).
Mixed ts units handled per-value; datetime recomputed with errors='coerce'.
"""
import os, socket, sys, time
socket.setdefaulttimeout(30)  # covers proxy CONNECT hangs that requests timeout misses
import pandas as pd
import requests
sys.path.insert(0, ".")

PATH = "data/raw/coinbase/coinbase_candles_1h.parquet"
RAW = "data/raw/coinbase/_backfill_raw.parquet"
TARGET = pd.Timestamp("2021-01-01", tz="UTC")

def norm_ts(series):
    t = pd.to_numeric(series, errors="coerce")
    if t.notna().mean() < 0.9:  # ISO strings
        t = pd.to_datetime(series, utc=True, errors="coerce").astype("int64") // 10**9
    t = t.where(t < 1e11, t // 1000)  # ms -> s per-value
    return t

def download():
    old = pd.read_parquet(PATH)
    old_ts = norm_ts(old["ts"])
    allrows = []
    for prod in sorted(old["product"].unique()):
        oldest = pd.to_datetime(old_ts[old["product"] == prod].min(), unit="s", utc=True)
        if pd.isna(oldest) or oldest <= TARGET:
            continue
        end, fails = oldest, 0
        n0 = len(allrows)
        while end > TARGET and fails <= 5:
            start = max(end - pd.Timedelta(hours=300), TARGET)
            try:
                r = requests.get(f"https://api.exchange.coinbase.com/products/{prod}/candles",
                                 params={"granularity": 3600, "start": start.isoformat(), "end": end.isoformat()},
                                 timeout=30)
                data = r.json() if r.status_code == 200 else []
                fails = 0
            except Exception:
                fails += 1; time.sleep(2 * fails); continue
            if not isinstance(data, list) or not data:
                break
            for row in data:
                allrows.append({"product": prod, "ts": int(row[0]), "low": float(row[1]),
                                "high": float(row[2]), "open": float(row[3]),
                                "close": float(row[4]), "volume": float(row[5])})
            end = start
            time.sleep(0.12)
        print(f"  {prod}: +{len(allrows)-n0}", flush=True)
    raw = pd.DataFrame(allrows)
    raw.to_parquet(RAW, index=False)
    print(f"RAW SAVED: {len(raw):,} rows -> {RAW}", flush=True)

def merge():
    old = pd.read_parquet(PATH)
    old2 = pd.DataFrame({
        "product": old["product"],
        "ts": norm_ts(old["ts"]).astype("int64"),
        "low": pd.to_numeric(old["low"], errors="coerce"),
        "high": pd.to_numeric(old["high"], errors="coerce"),
        "open": pd.to_numeric(old["open"], errors="coerce"),
        "close": pd.to_numeric(old["close"], errors="coerce"),
        "volume": pd.to_numeric(old["volume"], errors="coerce"),
    })
    raw = pd.read_parquet(RAW)
    new = pd.concat([old2, raw], ignore_index=True)
    new = new.drop_duplicates(subset=["product", "ts"]).sort_values(["product", "ts"])
    new["datetime"] = pd.to_datetime(new["ts"], unit="s", utc=True, errors="coerce").astype(str)
    new.to_parquet(PATH, index=False)
    mn = pd.to_datetime(new['ts'].min(), unit='s', utc=True)
    mx = pd.to_datetime(new['ts'].max(), unit='s', utc=True)
    print(f"MERGED: {len(new):,} rows, {new['product'].nunique()} products, {mn.date()} -> {mx.date()}")

if __name__ == "__main__":
    if not os.path.exists(RAW) or "--redownload" in sys.argv:
        download()
    merge()
    print("R154c done.")
