#!/usr/bin/env python3
"""R155 — IC screen of the NEWLY acquired datasets (LOCAL, laptop-light).

Data: data/raw/basis/{um_premium,um_mark,um_index,cm_premium}.parquet,
      data/raw/okx/okx_*.parquet, data/raw/coinbase/coinbase_candles_1h.parquet.
Returns: close-only panel from data/raw/{SYM}_USDT_1h.parquet (light).
Method: per-timestamp cross-sectional Spearman rank-IC vs fwd_ret_12h,
Newey-West(12) t, 3 sub-window same-sign thirds. Screen window <= 2026-04-25
(pristine 2026-04-26..06-08 stays untouched). Daily OKX rubik stats are
shifted +1 day before use (no lookahead).
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
OUT = "results_r155_newdata_ic.json"


def _nw_tstat(x, lags=12):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 50:
        return np.nan
    d = x - x.mean()
    var = d @ d / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        var += 2.0 * w * (d[:-k] @ d[k:]) / n
    return x.mean() / (np.sqrt(max(var, 1e-18) / n) + 1e-18)


print("Building close panel + fwd returns...")
closes = {}
for s in SYMS:
    try:
        df = pd.read_parquet(f"data/raw/{s.replace('USDT','_USDT')}_1h.parquet",
                             columns=["timestamp", "close"])
    except Exception:
        try:
            df = pd.read_parquet(f"data/raw/{s.replace('USDT','_USDT')}_1h.parquet")
            df = df.reset_index()[["timestamp", "close"]]
        except Exception:
            continue
    ser = df.set_index(pd.to_datetime(df["timestamp"], utc=True))["close"].astype("float32")
    closes[s] = ser[~ser.index.duplicated()]
close = pd.DataFrame(closes).sort_index()
fwd12 = close.shift(-12) / close - 1
binance_vol = None
print(f"  panel: {close.shape}, {close.index.min()} -> {close.index.max()}")


def screen(name, feat_panel):
    """feat_panel: DataFrame indexed by hourly UTC ts, columns = symbols."""
    common = [c for c in feat_panel.columns if c in fwd12.columns]
    f = feat_panel[common].reindex(close.index)
    r = fwd12[common]
    mask = close.index <= SCREEN_END
    f, r = f[mask], r[mask]
    ics = []
    for ts in f.index:
        x, y = f.loc[ts].values, r.loc[ts].values
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
    return {"feature": name, "ic": round(float(s.mean()), 4),
            "t_nw12": round(float(t), 2), "n": len(s),
            "thirds": f"{agree}/3", "verdict": verdict}


def zscore(p, w):
    return (p - p.rolling(w, min_periods=w // 2).mean()) / (p.rolling(w, min_periods=w // 2).std() + 1e-12)


results = []

# ── A. UM premium-index basis ────────────────────────────────────────────
try:
    pr = pd.read_parquet("data/raw/basis/um_premium.parquet")
    pr["timestamp"] = pd.to_datetime(pr["timestamp"], utc=True)
    lvl = pr.pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="first")
    rng = (pr.pivot_table(index="timestamp", columns="symbol", values="high", aggfunc="first")
           - pr.pivot_table(index="timestamp", columns="symbol", values="low", aggfunc="first"))
    results.append(screen("basis_level", lvl))
    results.append(screen("basis_z168", zscore(lvl, 168)))
    results.append(screen("basis_mom24", lvl - lvl.shift(24)))
    results.append(screen("basis_range_z168", zscore(rng, 168)))
    del pr, lvl, rng; gc.collect()
except Exception as e:
    print(f"A basis: {e}")

# ── B. mark vs index gap ─────────────────────────────────────────────────
try:
    mk = pd.read_parquet("data/raw/basis/um_mark.parquet")
    ix = pd.read_parquet("data/raw/basis/um_index.parquet")
    for d in (mk, ix):
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    mkp = mk.pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="first")
    ixp = ix.pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="first")
    gap = mkp / ixp - 1
    results.append(screen("markindex_gap_z168", zscore(gap, 168)))
    results.append(screen("markindex_gap_mom24", gap - gap.shift(24)))
    del mk, ix, mkp, ixp, gap; gc.collect()
except Exception as e:
    print(f"B markindex: {e}")

# ── C. OKX rubik 1D (shift +1d, ffill to hourly) ─────────────────────────
def daily_to_hourly(piv):
    piv = piv.copy()
    piv.index = pd.to_datetime(piv.index, utc=True) + pd.Timedelta(days=1)  # no lookahead
    return piv.reindex(close.index, method="ffill", limit=48)

try:
    tv = pd.read_parquet("data/raw/okx/okx_rubik_1d_taker_volume.parquet")
    cols = tv.columns.tolist()
    # expect columns like instId/ts/buyVol/sellVol — detect
    tcol = next(c for c in cols if c.lower() in ("ts", "timestamp"))
    icol = next(c for c in cols if "inst" in c.lower() or c.lower() == "symbol")
    bcol = next(c for c in cols if "buy" in c.lower())
    scol = next(c for c in cols if "sell" in c.lower())
    tv["sym"] = tv[icol].str.replace("-USDT-SWAP", "", regex=False).str.replace("-", "") + "USDT"
    tv["ts"] = pd.to_datetime(pd.to_numeric(tv[tcol]), unit="ms", utc=True)
    tv["imb"] = (pd.to_numeric(tv[bcol]) - pd.to_numeric(tv[scol])) / \
                (pd.to_numeric(tv[bcol]) + pd.to_numeric(tv[scol]) + 1e-12)
    piv = tv.pivot_table(index="ts", columns="sym", values="imb", aggfunc="first")
    h = daily_to_hourly(piv)
    results.append(screen("okx_taker_imb_1d", h))
    results.append(screen("okx_taker_imb_z30d", daily_to_hourly(zscore(piv, 30))))
    del tv, piv, h; gc.collect()
except Exception as e:
    print(f"C okx taker: {e}")

for fname, label in [("okx_rubik_1d_ls_account_ratio.parquet", "okx_ls_acct"),
                     ("okx_rubik_1d_ls_position_ratio_top.parquet", "okx_ls_pos_top"),
                     ("okx_rubik_1d_oi_history.parquet", "okx_oi")]:
    try:
        d = pd.read_parquet(f"data/raw/okx/{fname}")
        cols = d.columns.tolist()
        tcol = next(c for c in cols if c.lower() in ("ts", "timestamp"))
        icol = next(c for c in cols if "inst" in c.lower() or c.lower() == "symbol")
        vcols = [c for c in cols if c not in (tcol, icol) and pd.api.types.is_numeric_dtype(d[c]) or
                 (c not in (tcol, icol) and d[c].dtype == object)]
        vcol = vcols[0]
        d["sym"] = d[icol].str.replace("-USDT-SWAP", "", regex=False).str.replace("-", "") + "USDT"
        d["ts"] = pd.to_datetime(pd.to_numeric(d[tcol]), unit="ms", utc=True)
        d["val"] = pd.to_numeric(d[vcol], errors="coerce")
        piv = d.pivot_table(index="ts", columns="sym", values="val", aggfunc="first")
        results.append(screen(f"{label}_z30d", daily_to_hourly(zscore(piv, 30))))
        if label == "okx_oi":
            results.append(screen("okx_oi_chg5d", daily_to_hourly(piv / piv.shift(5) - 1)))
        del d, piv; gc.collect()
    except Exception as e:
        print(f"C {label}: {e}")

# ── D. OKX vs Binance price basis + volume share ─────────────────────────
try:
    oc = pd.read_parquet("data/raw/okx/okx_candles_1h.parquet")
    cols = oc.columns.tolist()
    tcol = next(c for c in cols if c.lower() in ("ts", "timestamp"))
    icol = next(c for c in cols if "inst" in c.lower() or c.lower() == "symbol")
    ccol = next(c for c in cols if c.lower() in ("close", "c"))
    oc["sym"] = oc[icol].str.replace("-USDT-SWAP", "", regex=False).str.replace("-", "") + "USDT"
    ts_num = pd.to_numeric(oc[tcol], errors="coerce")
    oc["ts"] = (pd.to_datetime(ts_num, unit="ms", utc=True) if ts_num.notna().mean() > 0.9
                else pd.to_datetime(oc[tcol], utc=True))
    okx_close = oc.pivot_table(index="ts", columns="sym",
                               values=ccol, aggfunc="first").astype("float32")
    okx_close = okx_close.reindex(close.index)
    vb = (okx_close / close[okx_close.columns.intersection(close.columns)] - 1)
    results.append(screen("okx_binance_basis_z168", zscore(vb, 168)))
    results.append(screen("okx_binance_basis_mom24", vb - vb.shift(24)))
    del oc, okx_close, vb; gc.collect()
except Exception as e:
    print(f"D okx basis: {e}")

# ── E. Coinbase premium ──────────────────────────────────────────────────
try:
    cb = pd.read_parquet("data/raw/coinbase/coinbase_candles_1h.parquet")
    cols = cb.columns.tolist()
    tcol = next(c for c in cols if c.lower() in ("ts", "timestamp", "time"))
    pcol = next(c for c in cols if "product" in c.lower() or "symbol" in c.lower() or "inst" in c.lower())
    ccol = next(c for c in cols if c.lower() in ("close", "c"))
    cb["sym"] = cb[pcol].str.replace("-USD", "", regex=False) + "USDT"
    ts_num = pd.to_numeric(cb[tcol], errors="coerce")
    unit = "s" if ts_num.dropna().lt(1e12).all() else "ms"
    cb["ts"] = (pd.to_datetime(ts_num, unit=unit, utc=True) if ts_num.notna().mean() > 0.9
                else pd.to_datetime(cb[tcol], utc=True))
    cbp = cb.pivot_table(index="ts", columns="sym", values=ccol, aggfunc="first").astype("float32")
    cbp = cbp.reindex(close.index)
    prem = cbp / close[cbp.columns.intersection(close.columns)] - 1
    results.append(screen("coinbase_premium_z168", zscore(prem, 168)))
    results.append(screen("coinbase_premium_mom24", prem - prem.shift(24)))
    # market-level: median premium as TS regime signal
    med = prem.median(axis=1)
    csr = fwd12.mean(axis=1)
    both = pd.concat([med.rename("x"), csr.rename("y")], axis=1).dropna()
    both = both[both.index <= SCREEN_END]
    if len(both) > 500:
        t = _nw_tstat((both["x"].rank(pct=True) - 0.5).values * np.sign(1) *
                      np.sign(both["y"].values) * np.abs(both["y"].rank(pct=True) - 0.5).values * 0 +
                      0)  # placeholder not used
    rho = both["x"].corr(both["y"], method="spearman")
    print(f"  [TS] coinbase_median_premium vs next CS mean ret: rho={rho:+.3f} (n={len(both)})")
    del cb, cbp, prem; gc.collect()
except Exception as e:
    print(f"E coinbase: {e}")

# ── F. CM vs UM premium divergence ───────────────────────────────────────
try:
    cm = pd.read_parquet("data/raw/basis/cm_premium.parquet")
    um = pd.read_parquet("data/raw/basis/um_premium.parquet")
    for d in (cm, um):
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    cm["symu"] = cm["symbol"].str.replace("USD_PERP", "USDT", regex=False)
    cmp_ = cm.pivot_table(index="timestamp", columns="symu", values="close", aggfunc="first")
    ump = um.pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="first")
    inter = [c for c in cmp_.columns if c in ump.columns]
    div = cmp_[inter] - ump[inter]
    results.append(screen("cm_um_premium_div_z168", zscore(div, 168)))
    del cm, um, cmp_, ump, div; gc.collect()
except Exception as e:
    print(f"F cm_um: {e}")

print("\n" + "=" * 86)
print(f"  R155 — NEW DATA IC SCREEN (window <= {SCREEN_END.date()}, NW12, thirds)")
print("=" * 86)
res = sorted([r for r in results], key=lambda r: -abs(r.get("t_nw12") or 0))
for r in res:
    print(f"  {r['feature']:28s} IC={r['ic']!s:>8} t_NW12={r['t_nw12']!s:>7} "
          f"n={r['n']:>6} thirds={r['thirds']}  -> {r['verdict']}")
with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
strong = [r["feature"] for r in res if r["verdict"] in ("STRONG", "PASS")]
print(f"\n  STRONG/PASS: {strong if strong else 'none'}")
print("R155 done.")
