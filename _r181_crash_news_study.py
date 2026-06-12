#!/usr/bin/env python3
"""R181 — descriptive event study: market indicators BEFORE the worst 12h
periods of the s30 stack. LOCAL, instant.

HONEST FRAMING (pre-declared): with ~10 events this is DESCRIPTIVE, not a
tradable pattern test. We report where each indicator stood (percentile of
its own full-sample distribution) 24h and 2h before each crash period.
Indicators: market news volume z (sum of per-coin news_count_24h, 30d z),
market news sentiment (mean news_sentiment_24h), BTC DVOL (level z168),
Fear&Greed value, and realized strategy vol (the VT input, trailing 30p).
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

port = pd.read_parquet("cache/r178_s30_port.parquet").sort_values("timestamp").reset_index(drop=True)
port["timestamp"] = pd.to_datetime(port["timestamp"], utc=True)
worst = port.nsmallest(10, "net_ret")[["timestamp", "net_ret"]]

nw = pd.read_parquet("data/sentiment/crypto_news.parquet",
                     columns=["timestamp", "symbol", "news_count_24h", "news_sentiment_24h"])
nw["timestamp"] = pd.to_datetime(nw["timestamp"], utc=True)
agg = nw.groupby("timestamp").agg(cnt=("news_count_24h", "sum"),
                                  sent=("news_sentiment_24h", "mean")).sort_index()
agg["cnt_z30d"] = (agg["cnt"] - agg["cnt"].rolling(720, min_periods=360).mean()) / (agg["cnt"].rolling(720, min_periods=360).std() + 1e-9)
del nw

dv = pd.read_parquet("data/sentiment/deribit_dvol.parquet")
dv["timestamp"] = pd.to_datetime(dv["timestamp"], utc=True)
dv = dv[dv["currency"].str.upper().str.contains("BTC")] if "currency" in dv.columns else dv
dvol = dv.set_index("timestamp")["dvol_close"].sort_index()
dvol_z = (dvol - dvol.rolling(168, min_periods=84).mean()) / (dvol.rolling(168, min_periods=84).std() + 1e-9)

fg = pd.read_parquet("data/sentiment/fear_greed.parquet")
tcol = [c for c in fg.columns if "time" in c.lower() or "date" in c.lower()][0]
vcol = [c for c in fg.columns if "value" in c.lower()][0]
fg[tcol] = pd.to_datetime(fg[tcol], utc=True)
fng = fg.set_index(tcol)[vcol].astype(float).sort_index()

svol = port.set_index("timestamp")["net_ret"].rolling(30, min_periods=30).std()

SERIES = {"news_volume_z": agg["cnt_z30d"], "news_sentiment": agg["sent"],
          "dvol_z168": dvol_z, "fear_greed": fng, "strategy_vol30": svol}


def pctile(series, ts):
    s = series.dropna()
    s = s[s.index <= ts]
    if len(s) < 50:
        return np.nan, np.nan
    val = s.iloc[-1]
    return float(val), float((s < val).mean() * 100)


print("Worst 10 periods of the s30 stack + indicators BEFORE each (value | percentile):")
header = f"{'event':17s} {'ret':>7s}"
for k in SERIES:
    header += f" | {k[:14]:>16s}"
print(header)
rows = []
for _, ev in worst.iterrows():
    ts = ev["timestamp"]
    line = f"{str(ts)[:16]:17s} {ev['net_ret']*100:+6.2f}%"
    rec = {"ts": str(ts), "ret": round(ev["net_ret"] * 100, 2)}
    for k, s in SERIES.items():
        v, p = pctile(s, ts - pd.Timedelta(hours=2))
        rec[k] = None if np.isnan(p) else round(p)
        line += f" | {'—':>16s}" if np.isnan(p) else f" | {v:7.2f} (p{p:3.0f})"
    rows.append(rec)
    print(line)

print("\nMedian percentile across events (100 = highest ever, 50 = typical):")
for k in SERIES:
    ps = [r[k] for r in rows if r[k] is not None]
    if ps:
        print(f"  {k:16s}: p{int(np.median(ps))} (n={len(ps)})")
print("\nNB: 10 events = descriptive only; high percentiles on vol-type indicators")
print("mean VT already 'sees' these regimes; a NEW edge needs indicators that are")
print("elevated before crashes but NOT already captured by vol/regime gates.")
print("R181 done.")
