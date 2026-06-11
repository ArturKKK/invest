#!/usr/bin/env python
"""R154 Acquirer 2: OKX + Coinbase venue data download.

HIGH-priority API picks from the R154 scout report:
  1. OKX perp 1H candles ({BASE}-USDT-SWAP), 2024-06-01 -> now.
     -> data/raw/okx/candles_1h/{instId}.parquet  + combined okx_candles_1h.parquet
  2. OKX rubik contract-level 1D stats (5 endpoints: taker volume, OI history,
     LS account ratio, top-trader account & position LS ratios), 2024-01-01 -> now.
     -> data/raw/okx/rubik_1d/{key}/{instId}.parquet + combined okx_rubik_1d_{key}.parquet
     NOTE: rubik daily bars are aligned to UTC+8 midnight (= 16:00 UTC prior day).
  3. Coinbase Exchange spot 1h candles ({BASE}-USD), 2024-06-01 -> now.
     -> data/raw/coinbase/candles_1h/{product}.parquet + combined coinbase_candles_1h.parquet
  4. (bonus, tiny) OKX funding-rate-history backfill — rolling ~93 days only,
     does NOT cover screening period; forward-collection value.
     -> data/raw/okx/funding_history/{instId}.parquet + combined okx_funding_history.parquet

Universe: SYM_35 bases. OKX delisted MATIC/FTM/RUNE entirely (no history retrievable);
POL-USDT-SWAP and S-USDT-SWAP added as marked replacements. Coinbase never listed
FTM/THETA/RUNE; MATIC-USD partial (ends ~2025-02); BNB-USD starts 2025-10-22.

Rate limits honoured (empirical, from scout probes):
  - rubik: 5 req/2s per endpoint -> 0.45s sleep between same-endpoint requests
  - history-candles: 20 req/2s -> 2 threads x 0.35s sleep (~5 rps combined max)
  - coinbase public: 10 rps -> 0.15s sleep single thread
Threads: 4 total (2 OKX candles, 1 rubik+funding, 1 coinbase). Resume-safe:
per-symbol parquet files are skipped when they already exist.
"""
import json
import os
import sys
import time
import threading
import traceback
from datetime import datetime, timezone

import pandas as pd
import requests

# ---------------------------------------------------------------- config
PROXY = "http://192.168.1.1:8888"
for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.setdefault(var, PROXY)

ROOT = "/Users/a.s.tabakov/Developer/invest"
OKX_DIR = os.path.join(ROOT, "data/raw/okx")
CB_DIR = os.path.join(ROOT, "data/raw/coinbase")

SYM_35_BASES = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "DOT", "LINK",
    "MATIC", "UNI", "ATOM", "LTC", "NEAR", "FIL", "APT", "ARB", "OP", "AAVE",
    "INJ", "FTM", "ALGO", "SAND", "MANA", "AXS", "THETA", "RUNE", "EGLD",
    "XTZ", "FLOW", "CHZ", "CRV", "LDO", "SNX",
]
OKX_DELISTED = {"MATIC", "FTM", "RUNE"}          # 51001/51012, no history at all
OKX_EXTRA = ["POL", "S"]                          # replacement listings (marked extra)
CB_NEVER_LISTED = {"FTM", "THETA", "RUNE"}

OKX_BASES = [b for b in SYM_35_BASES if b not in OKX_DELISTED] + OKX_EXTRA  # 34
CB_BASES = [b for b in SYM_35_BASES if b not in CB_NEVER_LISTED]            # 32

CANDLE_START_MS = int(datetime(2024, 6, 1, tzinfo=timezone.utc).timestamp() * 1000)
# rubik fixed start 2024-01-01 UTC+8 == 2023-12-31 16:00 UTC; begin a day early
RUBIK_BEGIN_MS = int(datetime(2023, 12, 31, tzinfo=timezone.utc).timestamp() * 1000)
NOW_MS = int(time.time() * 1000)

OKX_BASE_URL = "https://www.okx.com"
CB_BASE_URL = "https://api.exchange.coinbase.com"

RUBIK_ENDPOINTS = {
    # key -> (path, column names after ts)
    "taker_volume": ("/api/v5/rubik/stat/taker-volume-contract",
                     ["sell_vol", "buy_vol"]),
    "oi_history": ("/api/v5/rubik/stat/contracts/open-interest-history",
                   ["oi", "oi_ccy", "oi_usd"]),
    "ls_account_ratio": ("/api/v5/rubik/stat/contracts/long-short-account-ratio-contract",
                         ["ls_acct_ratio"]),
    "ls_account_ratio_top": ("/api/v5/rubik/stat/contracts/long-short-account-ratio-contract-top-trader",
                             ["ls_acct_ratio_top"]),
    "ls_position_ratio_top": ("/api/v5/rubik/stat/contracts/long-short-position-ratio-contract-top-trader",
                              ["ls_pos_ratio_top"]),
}

LOG_LOCK = threading.Lock()
ERRORS = []


def log(msg):
    with LOG_LOCK:
        print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def http_get(session, url, params, tag, max_tries=6):
    """GET with retries on 429/5xx/connection errors. Returns Response or None."""
    for attempt in range(max_tries):
        try:
            r = session.get(url, params=params, timeout=30)
        except Exception as e:
            log(f"WARN {tag}: conn error try{attempt + 1}: {type(e).__name__}: {e}")
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 429:
            log(f"WARN {tag}: HTTP 429, backing off")
            time.sleep(2.5 * (attempt + 1))
            continue
        if r.status_code >= 500:
            log(f"WARN {tag}: HTTP {r.status_code}, retrying")
            time.sleep(2 * (attempt + 1))
            continue
        return r
    log(f"ERROR {tag}: gave up after {max_tries} tries")
    ERRORS.append(tag)
    return None


def okx_get_data(session, path, params, tag):
    """OKX wrapper: returns (data_list, code). Handles 50011 rate-limit code."""
    for attempt in range(6):
        r = http_get(session, OKX_BASE_URL + path, params, tag)
        if r is None:
            return None, "http_fail"
        try:
            j = r.json()
        except Exception:
            log(f"WARN {tag}: bad json, retrying")
            time.sleep(2)
            continue
        code = j.get("code")
        if code == "0":
            return j.get("data", []), code
        if code == "50011":  # rate limit
            time.sleep(2.5 * (attempt + 1))
            continue
        return None, code  # 51001/51012/50030/etc -> caller decides
    return None, "rate_limited"


def save_parquet(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------- OKX candles
def fetch_okx_candles(bases, thread_name):
    session = requests.Session()
    cols = ["ts", "open", "high", "low", "close", "vol", "vol_ccy",
            "vol_ccy_quote", "confirm"]
    for base in bases:
        inst = f"{base}-USDT-SWAP"
        out = os.path.join(OKX_DIR, "candles_1h", f"{inst}.parquet")
        if os.path.exists(out):
            log(f"{thread_name}: {inst} exists, skip")
            continue
        rows = []
        # recent candles first (history-candles can lag the live edge)
        data, code = okx_get_data(session, "/api/v5/market/candles",
                                  {"instId": inst, "bar": "1H", "limit": "300"},
                                  f"candles:{inst}:recent")
        if code in ("51001", "51012"):
            log(f"{thread_name}: {inst} delisted ({code}), skip")
            continue
        if data:
            rows.extend(data)
        time.sleep(0.35)
        after = str(NOW_MS)
        prev_oldest = None
        while True:
            data, code = okx_get_data(session, "/api/v5/market/history-candles",
                                      {"instId": inst, "bar": "1H",
                                       "after": after, "limit": "300"},
                                      f"hist-candles:{inst}")
            if data is None:
                log(f"{thread_name}: {inst} stop on code={code}")
                break
            if not data:
                break
            rows.extend(data)
            oldest = min(int(d[0]) for d in data)
            if prev_oldest is not None and oldest >= prev_oldest:
                break  # no progress safeguard
            prev_oldest = oldest
            if oldest <= CANDLE_START_MS:
                break
            after = str(oldest)
            time.sleep(0.35)
        if not rows:
            log(f"{thread_name}: {inst} NO DATA")
            continue
        df = pd.DataFrame(rows, columns=cols[:len(rows[0])])
        df["ts"] = df["ts"].astype("int64")
        for c in df.columns[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df[df["ts"] >= CANDLE_START_MS]
        df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
        df.insert(0, "instId", inst)
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        save_parquet(df, out)
        log(f"{thread_name}: {inst} candles saved {len(df)} rows "
            f"[{df['datetime'].iloc[0]} .. {df['datetime'].iloc[-1]}]")


# ---------------------------------------------------------------- OKX rubik 1D
def fetch_okx_rubik():
    session = requests.Session()
    for key, (path, val_cols) in RUBIK_ENDPOINTS.items():
        for base in OKX_BASES:
            inst = f"{base}-USDT-SWAP"
            out = os.path.join(OKX_DIR, "rubik_1d", key, f"{inst}.parquet")
            if os.path.exists(out):
                log(f"rubik:{key}: {inst} exists, skip")
                continue
            rows = []
            end = str(NOW_MS)
            prev_oldest = None
            while True:
                data, code = okx_get_data(
                    session, path,
                    {"instId": inst, "period": "1D", "end": end, "limit": "100"},
                    f"rubik:{key}:{inst}")
                time.sleep(0.45)  # 5 req/2s per endpoint
                if data is None:
                    if code not in ("50030",):
                        log(f"rubik:{key}: {inst} stop on code={code}")
                    break
                if not data:
                    break
                rows.extend(data)
                oldest = min(int(d[0]) for d in data)
                if prev_oldest is not None and oldest >= prev_oldest:
                    break
                prev_oldest = oldest
                if oldest <= RUBIK_BEGIN_MS:
                    break
                end = str(oldest - 1)
            if not rows:
                log(f"rubik:{key}: {inst} NO DATA")
                continue
            width = max(len(r) for r in rows)
            names = ["ts"] + val_cols
            names = names + [f"extra_{i}" for i in range(width - len(names))]
            df = pd.DataFrame(rows, columns=names[:width])
            df["ts"] = df["ts"].astype("int64")
            for c in df.columns[1:]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
            df.insert(0, "instId", inst)
            df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            save_parquet(df, out)
            log(f"rubik:{key}: {inst} saved {len(df)} rows "
                f"[{df['datetime'].iloc[0].date()} .. {df['datetime'].iloc[-1].date()}]")
    # bonus: funding-rate-history backfill (rolling ~93d, forward-collection only)
    for base in OKX_BASES:
        inst = f"{base}-USDT-SWAP"
        out = os.path.join(OKX_DIR, "funding_history", f"{inst}.parquet")
        if os.path.exists(out):
            continue
        recs = []
        after = str(NOW_MS)
        prev_oldest = None
        while True:
            data, code = okx_get_data(
                session, "/api/v5/public/funding-rate-history",
                {"instId": inst, "after": after, "limit": "100"},
                f"funding:{inst}")
            time.sleep(0.45)
            if data is None or not data:
                break
            recs.extend(data)
            oldest = min(int(d["fundingTime"]) for d in data)
            if prev_oldest is not None and oldest >= prev_oldest:
                break
            prev_oldest = oldest
            after = str(oldest)
        if not recs:
            log(f"funding: {inst} NO DATA")
            continue
        df = pd.json_normalize(recs)
        df["fundingTime"] = df["fundingTime"].astype("int64")
        df = df.drop_duplicates("fundingTime").sort_values("fundingTime").reset_index(drop=True)
        df["datetime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        save_parquet(df, out)
        log(f"funding: {inst} saved {len(df)} rows "
            f"[{df['datetime'].iloc[0].date()} .. {df['datetime'].iloc[-1].date()}]")


# ---------------------------------------------------------------- Coinbase
def fetch_coinbase():
    session = requests.Session()
    step_s = 300 * 3600  # 300 hourly candles per request window
    start_s = CANDLE_START_MS // 1000
    now_s = int(time.time())
    for base in CB_BASES:
        product = f"{base}-USD"
        out = os.path.join(CB_DIR, "candles_1h", f"{product}.parquet")
        if os.path.exists(out):
            log(f"coinbase: {product} exists, skip")
            continue
        rows = []
        missing = False
        t0 = start_s
        while t0 < now_s:
            t1 = min(t0 + step_s, now_s)
            params = {
                "granularity": 3600,
                "start": datetime.fromtimestamp(t0, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end": datetime.fromtimestamp(t1, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            r = http_get(session, f"{CB_BASE_URL}/products/{product}/candles",
                         params, f"cb:{product}:{params['start'][:10]}")
            time.sleep(0.15)
            if r is None:
                t0 = t1
                continue
            if r.status_code == 404:
                log(f"coinbase: {product} 404 (not listed), skip")
                missing = True
                break
            if r.status_code != 200:
                log(f"WARN coinbase: {product} HTTP {r.status_code}: {r.text[:120]}")
                t0 = t1
                continue
            try:
                batch = r.json()
            except Exception:
                batch = []
            if isinstance(batch, list):
                rows.extend(batch)
            t0 = t1
        if missing or not rows:
            if not missing:
                log(f"coinbase: {product} NO DATA")
            continue
        df = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "volume"])
        df["ts"] = df["ts"].astype("int64") * 1000  # s -> ms
        df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
        df.insert(0, "product", product)
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        save_parquet(df, out)
        log(f"coinbase: {product} saved {len(df)} rows "
            f"[{df['datetime'].iloc[0]} .. {df['datetime'].iloc[-1]}]")


# ---------------------------------------------------------------- combine + report
def combine(per_symbol_dir, combined_path):
    if not os.path.isdir(per_symbol_dir):
        return None
    files = sorted(f for f in os.listdir(per_symbol_dir) if f.endswith(".parquet"))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(os.path.join(per_symbol_dir, f)) for f in files],
                   ignore_index=True)
    save_parquet(df, combined_path)
    return df


def report(df, name, sym_col):
    if df is None:
        log(f"REPORT {name}: EMPTY")
        return
    g = df.groupby(sym_col)["datetime"].agg(["count", "min", "max"])
    log(f"REPORT {name}: {len(g)} symbols, {len(df)} rows total")
    for sym, row in g.iterrows():
        print(f"    {sym:<22} {int(row['count']):>7}  "
              f"{row['min'].strftime('%Y-%m-%d %H:%M')} .. {row['max'].strftime('%Y-%m-%d %H:%M')}",
              flush=True)


def main():
    log(f"START universe: OKX={len(OKX_BASES)} insts (incl extras {OKX_EXTRA}), "
        f"Coinbase={len(CB_BASES)} products")
    threads = []

    def wrap(fn, *args):
        def run():
            try:
                fn(*args)
            except Exception:
                log(f"FATAL in {fn.__name__}: {traceback.format_exc()}")
                ERRORS.append(fn.__name__)
        return run

    threads.append(threading.Thread(target=wrap(fetch_okx_candles, OKX_BASES[0::2], "okx-c1"),
                                    name="okx-c1"))
    threads.append(threading.Thread(target=wrap(fetch_okx_candles, OKX_BASES[1::2], "okx-c2"),
                                    name="okx-c2"))
    threads.append(threading.Thread(target=wrap(fetch_okx_rubik), name="okx-rubik"))
    threads.append(threading.Thread(target=wrap(fetch_coinbase), name="coinbase"))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    log("Downloads done; building combined parquets")
    df = combine(os.path.join(OKX_DIR, "candles_1h"),
                 os.path.join(OKX_DIR, "okx_candles_1h.parquet"))
    report(df, "okx_candles_1h", "instId")
    for key in RUBIK_ENDPOINTS:
        df = combine(os.path.join(OKX_DIR, "rubik_1d", key),
                     os.path.join(OKX_DIR, f"okx_rubik_1d_{key}.parquet"))
        report(df, f"okx_rubik_1d_{key}", "instId")
    df = combine(os.path.join(OKX_DIR, "funding_history"),
                 os.path.join(OKX_DIR, "okx_funding_history.parquet"))
    report(df, "okx_funding_history", "instId")
    df = combine(os.path.join(CB_DIR, "candles_1h"),
                 os.path.join(CB_DIR, "coinbase_candles_1h.parquet"))
    report(df, "coinbase_candles_1h", "product")

    if ERRORS:
        log(f"FINISHED WITH {len(ERRORS)} ERROR TAGS: {ERRORS[:20]}")
    else:
        log("FINISHED CLEAN")


if __name__ == "__main__":
    main()
