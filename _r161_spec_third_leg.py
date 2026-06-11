#!/usr/bin/env python3
"""R161 (plan P1) — specialist third leg. VM ONLY.

Specialist LGB+XGB trained ONLY on new features, ONLY on the covered slice
(2023-07+), W2/W3 windows. Its per-timestamp centered rank is ADDED to the
frozen champion rank with fixed pre-registered k in {0.1, 0.25, 0.4}; missing
spec coverage contributes exactly 0 (neutral). Paired vs champion-alone,
std + alt seeds. PRE-GATE: spec standalone rank-IC t_NW12 >= 2 on W2+W3 test.
Arms: PRIMARY venue-5f; SECONDARY all-8f.
"""
from _preflight_check import check_versions
check_versions()

import json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import _research_r68_continuous_wf as r68
from _research_r68_continuous_wf import CHAMPION_FEAT_31, CONTINUOUS_WINDOWS, sharpe, train_ensemble
from _research_r35_new_features import MARKET_LEVEL_FEATURES
from _research_r121_realistic_costs import R114B_CFG
from src.costs import cost_prod_blended
from _r136_s6_retest import simulate_r136

SEEDS_STD = [0, 7, 13, 42, 99]
SEEDS_ALT = [1, 8, 14, 43, 100]
VENUE = ["okx_binance_basis_z168", "okx_binance_basis_mom24",
         "coinbase_premium_z168", "coinbase_premium_mom24", "basis_range_z168"]
BD = ["bd_imb1_z168", "bd_imb1_chg24", "bd_shape_z168"]
ARMS = {"venue5": VENUE, "all8": VENUE + BD}
KGRID = [0.1, 0.25, 0.4]
SPEC_START = pd.Timestamp("2023-07-01", tz="UTC")
W23 = CONTINUOUS_WINDOWS[1:]  # W2, W3


def zscore(p, w):
    return (p - p.rolling(w, min_periods=w // 2).mean()) / (p.rolling(w, min_periods=w // 2).std() + 1e-12)


def _nw_tstat(x, lags=12):
    x = np.asarray(x, dtype=float); n = len(x)
    if n < 50: return np.nan
    d = x - x.mean(); var = d @ d / n
    for k in range(1, min(lags, n - 1) + 1):
        var += 2.0 * (1.0 - k / (lags + 1.0)) * (d[:-k] @ d[k:]) / n
    return x.mean() / (np.sqrt(max(var, 1e-18) / n) + 1e-18)


# ── frame + new features (verbatim construction from _r156) ──────────────
print("Loading frame + building features...")
df, regime_df = r68.load_data()
if "timestamp" in regime_df.columns:
    regime_df = regime_df.set_index("timestamp")
bclose = df.pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="first")
bclose.columns = [c.replace("/", "") for c in bclose.columns]
grid = bclose.index
panels = {}
oc = pd.read_parquet("data/raw/okx/okx_candles_1h.parquet")
oc["sym"] = oc["instId"].str.replace("-USDT-SWAP", "", regex=False) + "USDT"
oc["ts"] = pd.to_datetime(pd.to_numeric(oc["ts"]), unit="ms", utc=True)
okxp = oc.pivot_table(index="ts", columns="sym", values="close", aggfunc="first").astype(float).reindex(grid)
com = [c for c in okxp.columns if c in bclose.columns]
vb = okxp[com] / bclose[com] - 1
panels["okx_binance_basis_z168"] = zscore(vb, 168)
panels["okx_binance_basis_mom24"] = vb - vb.shift(24)
del oc, okxp, vb
cb = pd.read_parquet("data/raw/coinbase/coinbase_candles_1h.parquet")
cb["sym"] = cb["product"].str.replace("-USD", "", regex=False) + "USDT"
cb["tsx"] = pd.to_datetime(pd.to_numeric(cb["ts"], errors="coerce"), unit="s", utc=True)
cbp = cb.pivot_table(index="tsx", columns="sym", values="close", aggfunc="first").astype(float).reindex(grid)
com = [c for c in cbp.columns if c in bclose.columns]
prem = cbp[com] / bclose[com] - 1
panels["coinbase_premium_z168"] = zscore(prem, 168)
panels["coinbase_premium_mom24"] = prem - prem.shift(24)
del cb, cbp, prem
pr = pd.read_parquet("data/raw/basis/premium_index_klines_1h.parquet")
pr["timestamp"] = pd.to_datetime(pr["timestamp"], utc=True)
rng = (pr.pivot_table(index="timestamp", columns="symbol", values="high", aggfunc="first")
       - pr.pivot_table(index="timestamp", columns="symbol", values="low", aggfunc="first")).reindex(grid)
panels["basis_range_z168"] = zscore(rng, 168)
del pr, rng
import glob as _g
b1, a1, b5, a5 = {}, {}, {}, {}
for p in _g.glob("data/raw/bookdepth/*.parquet"):
    s = p.split("/")[-1].replace(".parquet", "")
    d2 = pd.read_parquet(p)
    d2["timestamp"] = pd.to_datetime(d2["timestamp"], utc=True)
    d2 = d2.set_index("timestamp").sort_index()
    d2 = d2[~d2.index.duplicated()]
    b1[s] = d2["notional_m1"].astype(float); a1[s] = d2["notional_p1"].astype(float)
    b5[s] = d2[[f"notional_m{i}" for i in range(1, 6)]].sum(axis=1)
    a5[s] = d2[[f"notional_p{i}" for i in range(1, 6)]].sum(axis=1)
B1 = pd.DataFrame(b1).reindex(grid); A1 = pd.DataFrame(a1).reindex(grid)
B5 = pd.DataFrame(b5).reindex(grid); A5 = pd.DataFrame(a5).reindex(grid)
imb1 = (B1 - A1) / (B1 + A1 + 1e-9)
shape = (B1 + A1) / (B5 + A5 + 1e-9)
panels["bd_imb1_z168"] = zscore(imb1, 168)
panels["bd_imb1_chg24"] = imb1 - imb1.shift(24)
panels["bd_shape_z168"] = zscore(shape, 168)
del b1, a1, b5, a5, B1, A1, B5, A5, imb1, shape

df["bsym"] = df["symbol"].str.replace("/", "", regex=False)
for name, p in panels.items():
    out = p.astype("float32").reset_index()
    idc = out.columns[0]
    out = out.melt(id_vars=idc, var_name="bsym", value_name=name).rename(columns={idc: "timestamp"})
    df = df.merge(out, on=["timestamp", "bsym"], how="left")
panels.clear()

feats30 = [f for f in CHAMPION_FEAT_31 if f in df.columns and f != "cg_taker_imb"]
df_spec = df[df["timestamp"] >= SPEC_START].copy()
print(f"spec slice: {len(df_spec):,} rows from {df_spec['timestamp'].min()}")


def run_port(preds, label):
    port = simulate_r136(preds, regime_df, 4, 2, dict(R114B_CFG),
                         cutoff_on=0.9, cutoff_off=0.8, min_risk_off_periods=2,
                         cost_fn=cost_prod_blended, funding_per_12h=0.00012,
                         exec_delay_penalty=0.0003)
    ns = sharpe(port["net_ret"])
    print(f"  {label:34s} Net={ns:+.3f}  n={len(port)}", flush=True)
    return ns, port


def boot_paired(a, b, n_boot=1000, block=14, seed=161):
    m = a[["timestamp", "net_ret"]].rename(columns={"net_ret": "x"}).merge(
        b[["timestamp", "net_ret"]].rename(columns={"net_ret": "y"}), on="timestamp")
    x, y = m["x"].values, m["y"].values; n = len(x)
    rng_ = np.random.RandomState(seed); wins = 0
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s, min(s + block, n))
                              for s in rng_.randint(0, n - block, size=n // block + 1)])[:n]
        sx = (x[idx].sum() / (x[idx].std() + 1e-12)) / np.sqrt(len(idx))
        sy = (y[idx].sum() / (y[idx].std() + 1e-12)) / np.sqrt(len(idx))
        wins += (sx > sy)
    return wins / n_boot


results = {}
for seeds, tag in ((SEEDS_STD, "std"), (SEEDS_ALT, "alt")):
    print(f"\n=== {tag.upper()} ===")
    champ = train_ensemble(df, feats30, W23, seeds=seeds,
                           cs_rank_exclude=[f for f in feats30 if f in MARKET_LEVEL_FEATURES])
    ns_c, port_c = run_port(champ, f"champion30 {tag} (W2W3)")
    results[f"champ_{tag}"] = ns_c
    for arm, fl in ARMS.items():
        spec = train_ensemble(df_spec, fl, W23, seeds=seeds, cs_rank_exclude=[])
        spec.to_parquet(f"cache/r161_spec_{arm}_{tag}_preds.parquet", index=False)
        # pre-gate: standalone rank-IC NW t
        ics = [spearmanr(g["pred"], g["fwd_ret"]).correlation
               for _, g in spec.groupby("timestamp") if g["pred"].nunique() > 2]
        ics = pd.Series([i for i in ics if not np.isnan(i)])
        t_nw = _nw_tstat(ics.values)
        print(f"  [{arm} {tag}] standalone IC={ics.mean():+.4f} t_NW12={t_nw:+.2f}")
        results[f"pregate_{arm}_{tag}"] = round(float(t_nw), 2)
        # spec-champ correlation diagnostic
        mg = champ.merge(spec[["timestamp", "symbol", "pred"]].rename(columns={"pred": "spred"}),
                         on=["timestamp", "symbol"], how="left")
        corr = mg[["pred", "spred"]].corr().iloc[0, 1]
        print(f"  [{arm} {tag}] spec-champ rank corr = {corr:+.3f}")
        if t_nw < 2:
            print(f"  [{arm} {tag}] PRE-GATE FAIL — skip blends")
            continue
        mg["spred"] = mg["spred"].fillna(0.0)
        for k in KGRID:
            bl = mg.copy()
            bl["pred"] = bl["pred"] + k * bl["spred"]
            ns_b, port_b = run_port(bl, f"blend {arm} k={k} {tag}")
            pwin = boot_paired(port_b, port_c)
            print(f"     -> delta {ns_b-ns_c:+.3f}, P(blend>champ) = {pwin:.3f}")
            results[f"blend_{arm}_{tag}_k{k}"] = {"ns": round(float(ns_b), 3),
                                                  "delta": round(float(ns_b - ns_c), 3),
                                                  "p": round(float(pwin), 3)}
with open("results_r161_spec_leg.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nGATE (pre-registered): pass = k=0.25 arm with P>=0.85 (std) AND delta>0 (alt) AND pregate t>=2")
print("R161 done.")
