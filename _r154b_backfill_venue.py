#!/usr/bin/env python3
"""R154b — extend okx_candles_1h + coinbase_candles_1h BACKWARDS to 2021-01-01.
Model training needs feature history from 2022 (W1 train); the first pull
started 2024-06. Incremental prepend: paginate backwards from each dataset's
current oldest ts. LOCAL via proxy.
"""
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, ".")
from _research_round7 import SYM_35

TARGET = pd.Timestamp("2021-01-01", tz="UTC")
BASES = [s.split("/")[0] for s in SYM_35]


def okx_backfill():
    path = "data/raw/okx/okx_candles_1h.parquet"
    df = pd.read_parquet(path)
    out = [df]
    for base in BASES + ["POL", "S"]:
        inst = f"{base}-USDT-SWAP"
        sub = df[df["instId"] == inst]
        if len(sub) == 0:
            continue
        oldest = int(pd.to_numeric(sub["ts"]).min())
        if pd.Timestamp(oldest, unit="ms", tz="UTC") <= TARGET:
            continue
        rows, after, fails = [], oldest, 0
        while True:
            try:
                r = requests.get("https://www.okx.com/api/v5/market/history-candles",
                                 params={"instId": inst, "bar": "1H", "after": str(after), "limit": "300"},
                                 timeout=30)
                data = r.json().get("data", [])
                fails = 0
            except Exception:
                fails += 1
                if fails > 5:
                    print(f"  okx {inst}: giving up after 5 fails at {after}", flush=True)
                    break
                time.sleep(2 * fails)
                continue
            if not data:
                break
            for row in data:
                rows.append({"instId": inst, "ts": row[0], "open": row[1], "high": row[2],
                             "low": row[3], "close": row[4], "vol": row[5],
                             "vol_ccy": row[6] if len(row) > 6 else None})
            after = int(data[-1][0])
            if pd.Timestamp(after, unit="ms", tz="UTC") <= TARGET:
                break
            time.sleep(0.12)
        if rows:
            out.append(pd.DataFrame(rows))
            print(f"  okx {inst}: +{len(rows)} rows back to "
                  f"{pd.Timestamp(int(rows[-1]['ts']), unit='ms', tz='UTC')}", flush=True)
    new = pd.concat(out, ignore_index=True)
    new["ts"] = new["ts"].astype(str)
    for c in ["open", "high", "low", "close", "vol", "vol_ccy"]:
        if c in new.columns:
            new[c] = pd.to_numeric(new[c], errors="coerce")
    new = new.drop_duplicates(subset=["instId", "ts"])
    new.to_parquet(path, index=False)
    print(f"okx candles total: {len(new):,}")


def cb_backfill():
    path = "data/raw/coinbase/coinbase_candles_1h.parquet"
    df = pd.read_parquet(path)
    out = [df]
    for prod in sorted(df["product"].unique()):
        sub = df[df["product"] == prod]
        ts_num = pd.to_numeric(sub["ts"], errors="coerce")
        unit = "s" if ts_num.dropna().lt(1e12).all() else "ms"
        oldest = pd.to_datetime(ts_num.min(), unit=unit, utc=True)
        if oldest <= TARGET:
            continue
        rows = []
        end = oldest
        fails = 0
        while end > TARGET:
            start = max(end - pd.Timedelta(hours=300), TARGET)
            try:
                r = requests.get(f"https://api.exchange.coinbase.com/products/{prod}/candles",
                                 params={"granularity": 3600,
                                         "start": start.isoformat(), "end": end.isoformat()},
                                 timeout=30)
                data = r.json() if r.status_code == 200 else []
                fails = 0
            except Exception:
                fails += 1
                if fails > 5:
                    print(f"  cb {prod}: giving up after 5 fails at {end}", flush=True)
                    break
                time.sleep(2 * fails)
                continue
            if not isinstance(data, list) or not data:
                break
            for row in data:
                rows.append({"product": prod, "ts": row[0], "low": row[1], "high": row[2],
                             "open": row[3], "close": row[4], "volume": row[5],
                             "datetime": pd.Timestamp(row[0], unit="s", tz="UTC").isoformat()})
            end = start
            time.sleep(0.15)
        if rows:
            out.append(pd.DataFrame(rows))
            print(f"  cb {prod}: +{len(rows)} rows back to "
                  f"{pd.Timestamp(min(r2['ts'] for r2 in rows), unit='s', tz='UTC')}", flush=True)
    # Normalize BOTH old and new frames to identical dtypes before concat
    norm = []
    for fr in out:
        fr = fr.copy()
        tsn = pd.to_numeric(fr["ts"], errors="coerce")
        if tsn.notna().mean() < 0.9:  # ISO strings
            tsn = pd.to_datetime(fr["ts"], utc=True, errors="coerce").astype("int64") // 10**9
        fr["ts"] = tsn.astype("int64")
        for c in ["low", "high", "open", "close", "volume"]:
            fr[c] = pd.to_numeric(fr[c], errors="coerce")
        fr = fr[["product", "ts", "low", "high", "open", "close", "volume"]]
        norm.append(fr)
    new = pd.concat(norm, ignore_index=True)
    new = new.drop_duplicates(subset=["product", "ts"]).sort_values(["product", "ts"])
    new["datetime"] = pd.to_datetime(new["ts"], unit="s", utc=True).astype(str)
    new.to_parquet(path, index=False)
    print(f"coinbase candles total: {len(new):,}")


t0 = time.time()
okx_backfill()
cb_backfill()
print(f"R154b done in {(time.time()-t0)/60:.1f} min")
