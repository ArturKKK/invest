#!/usr/bin/env python3
"""R157 — bookDepth feature IC screen (LOCAL, light). 35 syms, 2023-01+.

Columns per symbol parquet: timestamp, depth_{m5..m1,p1..p5},
notional_{m5..m1,p1..p5}, symbol. m=bids below, p=asks above (% levels).
D6 lesson: static depth LEVELS = frozen size-factor (rank autocorr 0.96) —
screen DYNAMIC forms (z-scores per symbol, changes, imbalances).
Same protocol as R155: CS Spearman IC vs fwd12, NW(12) t, thirds,
window <= 2026-04-25.
"""
import gc
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import sys
sys.path.insert(0, ".")
from _research_round7 import SYM_35

SCREEN_END = pd.Timestamp("2026-04-25", tz="UTC")
SYMS = [s.replace("/", "") for s in SYM_35]
OUT = "results_r157_bookdepth_ic.json"


def _nw_tstat(x, lags=12):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 50:
        return np.nan
    d = x - x.mean()
    var = d @ d / n
    for k in range(1, min(lags, n - 1) + 1):
        var += 2.0 * (1.0 - k / (lags + 1.0)) * (d[:-k] @ d[k:]) / n
    return x.mean() / (np.sqrt(max(var, 1e-18) / n) + 1e-18)


print("Close panel...")
closes = {}
for s in SYMS:
    try:
        df = pd.read_parquet(f"data/raw/{s.replace('USDT','_USDT')}_1h.parquet",
                             columns=["timestamp", "close"])
    except Exception:
        continue
    ser = df.set_index(pd.to_datetime(df["timestamp"], utc=True))["close"].astype("float32")
    closes[s] = ser[~ser.index.duplicated()]
close = pd.DataFrame(closes).sort_index()
fwd12 = close.shift(-12) / close - 1

print("Loading bookDepth panels...")
panels = {}   # raw building blocks
bidn, askn, bidn5, askn5 = {}, {}, {}, {}
for s in SYMS:
    try:
        d = pd.read_parquet(f"data/raw/bookdepth/{s}.parquet")
    except Exception:
        continue
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    d = d.set_index("timestamp").sort_index()
    d = d[~d.index.duplicated()]
    bidn[s] = d["notional_m1"].astype("float32")
    askn[s] = d["notional_p1"].astype("float32")
    bidn5[s] = (d[[f"notional_m{i}" for i in range(1, 6)]].sum(axis=1)).astype("float32")
    askn5[s] = (d[[f"notional_p{i}" for i in range(1, 6)]].sum(axis=1)).astype("float32")
B1 = pd.DataFrame(bidn).reindex(close.index)
A1 = pd.DataFrame(askn).reindex(close.index)
B5 = pd.DataFrame(bidn5).reindex(close.index)
A5 = pd.DataFrame(askn5).reindex(close.index)
del bidn, askn, bidn5, askn5
gc.collect()
print(f"  panels {B1.shape}")


def zscore(p, w):
    return (p - p.rolling(w, min_periods=w // 2).mean()) / (p.rolling(w, min_periods=w // 2).std() + 1e-12)


def screen(name, f):
    common = [c for c in f.columns if c in fwd12.columns]
    ff = f[common].reindex(close.index)
    rr = fwd12[common]
    mask = close.index <= SCREEN_END
    ff, rr = ff[mask], rr[mask]
    ics = []
    for ts in ff.index:
        x, y = ff.loc[ts].values, rr.loc[ts].values
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() >= 10 and len(np.unique(x[ok])) > 2:
            ic = spearmanr(x[ok], y[ok]).correlation
            if not np.isnan(ic):
                ics.append((ts, ic))
    if len(ics) < 500:
        return {"feature": name, "ic": np.nan, "t_nw12": np.nan, "n": len(ics),
                "thirds": "0/3", "verdict": "NO_DATA"}
    s = pd.Series(dict(ics)).sort_index()
    t = _nw_tstat(s.values)
    third = len(s) // 3
    signs = [np.sign(s.iloc[i*third:(i+1)*third].mean()) for i in range(3)]
    agree = sum(1 for x in signs if x == np.sign(s.mean()))
    a = abs(t)
    verdict = ("STRONG" if (a >= 4 and agree == 3) else
               "PASS" if (a >= 3 and agree == 3) else
               "WEAK" if a >= 2 else "DEAD")
    print(f"  {name:26s} IC={s.mean():+.4f} t={t:+6.2f} n={len(s)} {agree}/3 -> {verdict}", flush=True)
    return {"feature": name, "ic": round(float(s.mean()), 4), "t_nw12": round(float(t), 2),
            "n": len(s), "thirds": f"{agree}/3", "verdict": verdict}


total1 = B1 + A1
total5 = B5 + A5
imb1 = (B1 - A1) / (total1 + 1e-9)
imb5 = (B5 - A5) / (total5 + 1e-9)
shape = total1 / (total5 + 1e-9)   # near-book concentration

results = []
results.append(screen("bd_imb1_level", imb1))
results.append(screen("bd_imb1_z168", zscore(imb1, 168)))
results.append(screen("bd_imb1_chg24", imb1 - imb1.shift(24)))
results.append(screen("bd_imb5_z168", zscore(imb5, 168)))
results.append(screen("bd_total5_z168", zscore(total5, 168)))
results.append(screen("bd_total5_chg24", total5 / total5.shift(24) - 1))
results.append(screen("bd_shape_z168", zscore(shape, 168)))
results.append(screen("bd_shape_chg24", shape - shape.shift(24)))
results.append(screen("bd_bid5_z168", zscore(B5, 168)))
results.append(screen("bd_ask5_z168", zscore(A5, 168)))

print("\nSTRONG/PASS:", [r["feature"] for r in results if r["verdict"] in ("STRONG", "PASS")])
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print("R157 done.")
