#!/usr/bin/env python3
"""R153 — download Binance futures basis klines (data.binance.vision).

Three synthetic-kline datasets (futures/um, 1h, MONTHLY zips + daily top-up):
  - premiumIndexKlines : OHLC of (mark-index)/index per hour -> basis level +
    intra-hour basis range (new vs existing close-only premium_index) +
    2020-2021 history extension
  - markPriceKlines    : mark-price OHLC -> mark-vs-last gap, clean RV
  - indexPriceKlines   : index (spot composite) OHLC -> spot leg, perp-vs-spot

Universe: SYM_35 (Binance naming = strip '/'). History 2020-01 .. 2026-05
monthly + daily files 2026-06-01..2026-06-08. ~120 MB compressed total.

DATA VALIDITY TRAP (verified by scout): delisted perps keep publishing frozen
klines. MATICUSDT delisted ~2024-09, FTMUSDT ~2025-01 (aggTrades end months)
-> months after cutoff are not fetched and rows after cutoff are trimmed.

Incremental: per (dataset, symbol) part parquets in data/raw/basis/_parts/;
months already present are skipped. Final output: ONE parquet per dataset in
data/raw/basis/ (columns: timestamp, symbol, open, high, low, close).
4 download threads (laptop rule: bookDepth job already uses 8).
"""
import io
import json
import os
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import pandas as pd
import requests

sys.path.insert(0, ".")
from _research_round7 import SYM_35

for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.setdefault(var, "http://192.168.1.1:8888")

BASE = "https://data.binance.vision/data/futures/um"
DATASETS = ["premiumIndexKlines", "markPriceKlines", "indexPriceKlines"]
OUT_NAME = {
    "premiumIndexKlines": "premium_index_klines_1h.parquet",
    "markPriceKlines": "mark_price_klines_1h.parquet",
    "indexPriceKlines": "index_price_klines_1h.parquet",
}
OUT_DIR = "data/raw/basis"
PARTS_DIR = f"{OUT_DIR}/_parts"
MONTH_START = (2020, 1)
MONTH_END = (2026, 5)          # last complete month with monthly zip
DAILY_DATES = [date(2026, 6, 1) + timedelta(days=i) for i in range(8)]  # ..06-08
WORKERS = 4

# delisted perps publish frozen fake klines afterwards -> hard cutoffs (UTC)
DELIST_CUTOFF = {
    "MATICUSDT": pd.Timestamp("2024-09-30 23:59:59"),
    "FTMUSDT": pd.Timestamp("2025-01-31 23:59:59"),
}

SYMS = [s.replace("/", "") for s in SYM_35]
os.makedirs(OUT_DIR, exist_ok=True)
for ds in DATASETS:
    os.makedirs(f"{PARTS_DIR}/{ds}", exist_ok=True)

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume",
              "close_time", "quote_volume", "count", "taker_buy_volume",
              "taker_buy_quote_volume", "ignore"]

_tls = threading.local()


def _session():
    if not hasattr(_tls, "s"):
        s = requests.Session()
        _tls.s = s
    return _tls.s


def months_range():
    y, m = MONTH_START
    out = []
    while (y, m) <= MONTH_END:
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


ALL_MONTHS = months_range()


def parse_kline_zip(content):
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        with z.open(z.namelist()[0]) as f:
            head = f.read(64)
        has_header = head.lstrip()[:9].lower() == b"open_time"
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, header=0 if has_header else None,
                             names=None if has_header else KLINE_COLS,
                             usecols=range(12))
    df.columns = KLINE_COLS  # normalize header spelling variants
    ot = df["open_time"].astype("int64")
    # guard: spot switched to microseconds in 2025; futures should be ms
    unit = "us" if ot.iloc[0] > 10**14 else "ms"
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(ot, unit=unit),
        "open": pd.to_numeric(df["open"], errors="coerce"),
        "high": pd.to_numeric(df["high"], errors="coerce"),
        "low": pd.to_numeric(df["low"], errors="coerce"),
        "close": pd.to_numeric(df["close"], errors="coerce"),
    })
    return out


def fetch(url, retries=3):
    """Return (df|None, status). 404 -> (None, 404). Retries on errors."""
    for attempt in range(retries):
        try:
            r = _session().get(url, timeout=90)
            if r.status_code == 404:
                return None, 404
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            return parse_kline_zip(r.content), 200
        except Exception as e:
            if attempt == retries - 1:
                return None, f"ERR:{e}"
            time.sleep(2 * (attempt + 1))
    return None, "ERR:unreachable"


def fetch_month(ds, sym, ym):
    url = f"{BASE}/monthly/{ds}/{sym}/1h/{sym}-1h-{ym}.zip"
    df, code = fetch(url)
    return ("M", ym, df, code)

def fetch_daily(ds, sym, d):
    url = f"{BASE}/daily/{ds}/{sym}/1h/{sym}-1h-{d.isoformat()}.zip"
    df, code = fetch(url)
    return ("D", d.isoformat(), df, code)


def part_path(ds, sym):
    return f"{PARTS_DIR}/{ds}/{sym}.parquet"


def existing_keys(ds, sym):
    """Set of 'YYYY-MM' months and 'YYYY-MM-DD' days already in the part file."""
    p = part_path(ds, sym)
    if not os.path.exists(p):
        return set(), set()
    try:
        ts = pd.read_parquet(p, columns=["timestamp"])["timestamp"]
        months = set(ts.dt.strftime("%Y-%m").unique())
        days = set(ts.dt.strftime("%Y-%m-%d").unique())
        return months, days
    except Exception:
        return set(), set()


def main():
    t0 = time.time()
    log = {}
    units = [(ds, sym) for ds in DATASETS for sym in SYMS]
    for ui, (ds, sym) in enumerate(units, 1):
        cutoff = DELIST_CUTOFF.get(sym)
        have_m, have_d = existing_keys(ds, sym)
        todo_m = [ym for ym in ALL_MONTHS if ym not in have_m]
        if cutoff is not None:
            todo_m = [ym for ym in todo_m
                      if pd.Timestamp(ym + "-01") <= cutoff]
        todo_d = [] if cutoff is not None else \
                 [d for d in DAILY_DATES if d.isoformat() not in have_d]
        if not todo_m and not todo_d:
            print(f"[{ui:3d}/{len(units)}] {ds}/{sym}: complete", flush=True)
            continue
        frames, n404, nerr = [], 0, 0
        with ThreadPoolExecutor(WORKERS) as ex:
            futs = ([ex.submit(fetch_month, ds, sym, ym) for ym in todo_m] +
                    [ex.submit(fetch_daily, ds, sym, d) for d in todo_d])
            for fu in as_completed(futs):
                kind, key, df, code = fu.result()
                if df is None:
                    if code == 404:
                        n404 += 1
                    else:
                        nerr += 1
                        print(f"    {ds}/{sym} {key}: {code}", flush=True)
                    continue
                frames.append(df)
        if frames:
            new = pd.concat(frames, ignore_index=True)
            p = part_path(ds, sym)
            if os.path.exists(p):
                old = pd.read_parquet(p)
                new = pd.concat([old, new], ignore_index=True)
            new = (new.drop_duplicates(subset=["timestamp"])
                      .sort_values("timestamp").reset_index(drop=True))
            if cutoff is not None:
                new = new[new["timestamp"] <= cutoff]
            new.to_parquet(p, index=False)
            print(f"[{ui:3d}/{len(units)}] {ds}/{sym}: {len(new):,} rows "
                  f"({new['timestamp'].min()} -> {new['timestamp'].max()}), "
                  f"404s={n404} errs={nerr}  ({time.time()-t0:.0f}s)", flush=True)
        else:
            print(f"[{ui:3d}/{len(units)}] {ds}/{sym}: NO NEW DATA "
                  f"(404s={n404} errs={nerr})", flush=True)
        log[f"{ds}/{sym}"] = {"n404": n404, "nerr": nerr}
        with open(f"{OUT_DIR}/_download_log.json", "w") as f:
            json.dump({"log": log, "elapsed_s": time.time() - t0}, f, indent=2)

    # ---- assemble one parquet per dataset ----
    for ds in DATASETS:
        parts = []
        for sym in SYMS:
            p = part_path(ds, sym)
            if not os.path.exists(p):
                continue
            df = pd.read_parquet(p)
            df["symbol"] = sym
            parts.append(df)
        if not parts:
            print(f"{ds}: NO PARTS", flush=True)
            continue
        full = (pd.concat(parts, ignore_index=True)
                  .sort_values(["symbol", "timestamp"]).reset_index(drop=True))
        full = full[["timestamp", "symbol", "open", "high", "low", "close"]]
        out = f"{OUT_DIR}/{OUT_NAME[ds]}"
        full.to_parquet(out, index=False)
        print(f"WROTE {out}: {len(full):,} rows, {full['symbol'].nunique()} syms, "
              f"{full['timestamp'].min()} -> {full['timestamp'].max()}", flush=True)

    print(f"\nR153 done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
