#!/usr/bin/env python3
"""R151 — download Binance futures bookDepth history (data.binance.vision).

The D6 orderbook thesis (depth/liquidity features) but WITH history: daily
bookDepth zips exist from 2023-01-01, ~460KB/day/symbol, snapshots every ~30s
at levels -5..-1,+1..+5 percent (columns: timestamp,percentage,depth,notional).

Downloads SYM_35 (Binance naming), resamples to HOURLY per (symbol, level):
mean depth + mean notional, pivots levels to columns, writes one parquet per
symbol to data/raw/bookdepth/{SYM}.parquet. Incremental: skips dates already
present. 404 dates (delisted symbols / gaps) are logged and skipped.

VM ONLY (network + disk). Runtime ~2-3h full backfill with 8 workers.
"""
import io
import json
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import pandas as pd
import requests

sys.path.insert(0, ".")
from _research_round7 import SYM_35

BASE = "https://data.binance.vision/data/futures/um/daily/bookDepth"
OUT_DIR = "data/raw/bookdepth"
START = date(2023, 1, 1)
END = date(2026, 6, 8)
WORKERS = 8

os.makedirs(OUT_DIR, exist_ok=True)
SYMS = [s.replace("/", "") for s in SYM_35]


def fetch_day(sym, d):
    url = f"{BASE}/{sym}/{sym}-bookDepth-{d.isoformat()}.zip"
    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            return sym, d, None, r.status_code
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f)
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.floor("h")
        agg = (df.groupby(["timestamp", "percentage"])
                 .agg(depth=("depth", "mean"), notional=("notional", "mean"))
                 .reset_index())
        return sym, d, agg, 200
    except Exception as e:
        return sym, d, None, f"ERR:{e}"


def existing_dates(sym):
    p = f"{OUT_DIR}/{sym}.parquet"
    if not os.path.exists(p):
        return set()
    try:
        ts = pd.read_parquet(p, columns=["timestamp"])["timestamp"]
        return set(pd.to_datetime(ts).dt.date.unique())
    except Exception:
        return set()


def pivot_hourly(agg):
    out = agg.pivot_table(index="timestamp", columns="percentage",
                          values=["depth", "notional"], aggfunc="first")
    out.columns = [f"{a}_{'m' if b < 0 else 'p'}{abs(int(b))}" for a, b in out.columns]
    return out.reset_index()


t0 = time.time()
all_dates = [START + timedelta(days=i) for i in range((END - START).days + 1)]
skip_log = {}

for si, sym in enumerate(SYMS, 1):
    have = existing_dates(sym)
    todo = [d for d in all_dates if d not in have]
    if not todo:
        print(f"[{si:2d}/35] {sym}: complete ({len(have)} days)", flush=True)
        continue
    rows, missing = [], 0
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(fetch_day, sym, d) for d in todo]
        for fu in as_completed(futs):
            s, d, agg, code = fu.result()
            if agg is None:
                missing += 1
                continue
            rows.append(pivot_hourly(agg))
    if rows:
        new = pd.concat(rows, ignore_index=True)
        new["symbol"] = sym
        p = f"{OUT_DIR}/{sym}.parquet"
        if os.path.exists(p):
            old = pd.read_parquet(p)
            new = (pd.concat([old, new], ignore_index=True)
                     .drop_duplicates(subset=["timestamp"])
                     .sort_values("timestamp"))
        new.to_parquet(p, index=False)
        print(f"[{si:2d}/35] {sym}: +{len(rows)} days (missing {missing}), "
              f"total rows {len(new):,}  ({time.time()-t0:.0f}s)", flush=True)
    else:
        print(f"[{si:2d}/35] {sym}: NO DATA ({missing} missing/404)", flush=True)
    skip_log[sym] = missing
    with open(f"{OUT_DIR}/_download_log.json", "w") as f:
        json.dump({"missing_per_sym": skip_log, "elapsed_s": time.time() - t0}, f, indent=2)

print(f"\nR151 done in {(time.time()-t0)/60:.0f} min")
