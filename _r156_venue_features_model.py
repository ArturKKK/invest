#!/usr/bin/env python3
"""R156 — model-level test of the 5 STRONG cross-venue features. VM ONLY.

Features (R155 screen, all 3/3 stable, NW t 5.1-10.2):
  okx_binance_basis_z168/mom24  (OKX perp close / Binance close - 1)
  coinbase_premium_z168/mom24   (Coinbase spot close / Binance close - 1)
  basis_range_z168              (premium-index kline high-low intra-hour range)

Protocol: fresh retrain W1-W3, honest S6 sim. PAIRED comparison 30f vs 35f on
SAME seeds (std), then BOTH on alt seeds (seed-lottery guard, R150c lesson:
unpaired deltas < 0.4 are noise). Paired moving-block bootstrap on net_ret.
Needs data/raw/{basis,okx,coinbase} on the VM (ship via S3 archive).
"""
from _preflight_check import check_versions
check_versions()

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from _research_r68_continuous_wf import (
    load_data, train_ensemble, CONTINUOUS_WINDOWS, CHAMPION_FEAT_31, sharpe,
)
from _research_r35_new_features import MARKET_LEVEL_FEATURES
from _research_r22_models import SEEDS
from _research_r121_realistic_costs import R114B_CFG
from src.costs import cost_prod_blended
from _r136_s6_retest import simulate_r136

SEEDS_STD = [0, 7, 13, 42, 99]
SEEDS_ALT = [1, 8, 14, 43, 100]
VENUE = ["okx_binance_basis_z168", "okx_binance_basis_mom24",
         "coinbase_premium_z168", "coinbase_premium_mom24", "basis_range_z168"]
BD = ["bd_imb1_z168", "bd_imb1_chg24", "bd_shape_z168"]
NEWF = VENUE + BD


def zscore(p, w):
    return (p - p.rolling(w, min_periods=w // 2).mean()) / (p.rolling(w, min_periods=w // 2).std() + 1e-12)


def to_long(panel, name):
    out = panel.reset_index()
    idcol = out.columns[0]
    out = out.melt(id_vars=idcol, var_name="bsym", value_name=name)
    return out.rename(columns={idcol: "timestamp"})


print("Loading frame...")
df, regime_df = load_data()
if "timestamp" in regime_df.columns:
    regime_df = regime_df.set_index("timestamp")

# Binance close panel from the frame itself (hourly grid)
bclose = df.pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="first")
bclose.columns = [c.replace("/", "") for c in bclose.columns]
grid = bclose.index

print("Building venue features...")
# OKX basis
oc = pd.read_parquet("data/raw/okx/okx_candles_1h.parquet")
oc["sym"] = oc["instId"].str.replace("-USDT-SWAP", "", regex=False) + "USDT"
oc["ts"] = pd.to_datetime(pd.to_numeric(oc["ts"]), unit="ms", utc=True)
okx_close = oc.pivot_table(index="ts", columns="sym", values="close", aggfunc="first").astype(float)
okx_close = okx_close.reindex(grid)
common = [c for c in okx_close.columns if c in bclose.columns]
vb = okx_close[common] / bclose[common] - 1
panels = {
    "okx_binance_basis_z168": zscore(vb, 168),
    "okx_binance_basis_mom24": vb - vb.shift(24),
}
del oc, okx_close, vb

# Coinbase premium
cb = pd.read_parquet("data/raw/coinbase/coinbase_candles_1h.parquet")
cb["sym"] = cb["product"].str.replace("-USD", "", regex=False) + "USDT"
tsn = pd.to_numeric(cb["ts"], errors="coerce")
unit = "s" if tsn.dropna().lt(1e12).all() else "ms"
cb["tsx"] = pd.to_datetime(tsn, unit=unit, utc=True)
cbp = cb.pivot_table(index="tsx", columns="sym", values="close", aggfunc="first").astype(float)
cbp = cbp.reindex(grid)
common = [c for c in cbp.columns if c in bclose.columns]
prem = cbp[common] / bclose[common] - 1
panels["coinbase_premium_z168"] = zscore(prem, 168)
panels["coinbase_premium_mom24"] = prem - prem.shift(24)
del cb, cbp, prem

# Basis range
pr = pd.read_parquet("data/raw/basis/premium_index_klines_1h.parquet")
pr["timestamp"] = pd.to_datetime(pr["timestamp"], utc=True)
rng = (pr.pivot_table(index="timestamp", columns="symbol", values="high", aggfunc="first")
       - pr.pivot_table(index="timestamp", columns="symbol", values="low", aggfunc="first"))
rng = rng.reindex(grid)
panels["basis_range_z168"] = zscore(rng, 168)
del pr, rng

# bookDepth imbalance + shape
import glob as _glob
bidn, askn, bidn5, askn5 = {}, {}, {}, {}
for p in _glob.glob("data/raw/bookdepth/*.parquet"):
    s = p.split("/")[-1].replace(".parquet", "")
    d2 = pd.read_parquet(p)
    d2["timestamp"] = pd.to_datetime(d2["timestamp"], utc=True)
    d2 = d2.set_index("timestamp").sort_index()
    d2 = d2[~d2.index.duplicated()]
    bidn[s] = d2["notional_m1"].astype(float)
    askn[s] = d2["notional_p1"].astype(float)
    bidn5[s] = d2[[f"notional_m{i}" for i in range(1, 6)]].sum(axis=1)
    askn5[s] = d2[[f"notional_p{i}" for i in range(1, 6)]].sum(axis=1)
B1 = pd.DataFrame(bidn).reindex(grid); A1 = pd.DataFrame(askn).reindex(grid)
B5 = pd.DataFrame(bidn5).reindex(grid); A5 = pd.DataFrame(askn5).reindex(grid)
imb1 = (B1 - A1) / (B1 + A1 + 1e-9)
shape = (B1 + A1) / (B5 + A5 + 1e-9)
panels["bd_imb1_z168"] = zscore(imb1, 168)
panels["bd_imb1_chg24"] = imb1 - imb1.shift(24)
panels["bd_shape_z168"] = zscore(shape, 168)
del bidn, askn, bidn5, askn5, B1, A1, B5, A5, imb1, shape

# Merge into long frame
df["bsym"] = df["symbol"].str.replace("/", "", regex=False)
for name, p in panels.items():
    long = to_long(p.astype("float32"), name)
    df = df.merge(long, on=["timestamp", "bsym"], how="left")
    cov = df[name].notna().mean() * 100
    print(f"  {name}: merged, coverage {cov:.1f}%", flush=True)
panels.clear()

feats30 = [f for f in CHAMPION_FEAT_31 if f in df.columns and f != "cg_taker_imb"]
feats35 = feats30 + NEWF


def run(feats, seeds, label):
    no_rank = [f for f in feats if f in MARKET_LEVEL_FEATURES]
    preds = train_ensemble(df, feats, CONTINUOUS_WINDOWS, seeds=seeds, cs_rank_exclude=no_rank)
    port = simulate_r136(
        preds, regime_df, 4, 2, dict(R114B_CFG),
        cutoff_on=0.9, cutoff_off=0.8, min_risk_off_periods=2,
        cost_fn=cost_prod_blended, funding_per_12h=0.00012, exec_delay_penalty=0.0003,
    )
    ns = sharpe(port["net_ret"])
    ret = ((1 + port["net_ret"]).prod() - 1) * 100
    print(f"  {label:24s} Net={ns:+.3f}  Ret={ret:+.1f}%  n={len(port)}", flush=True)
    return ns, port


def boot_paired(a, b, n_boot=1000, block=14, seed=156):
    m = a[["timestamp", "net_ret"]].rename(columns={"net_ret": "x"}).merge(
        b[["timestamp", "net_ret"]].rename(columns={"net_ret": "y"}), on="timestamp")
    x, y = m["x"].values, m["y"].values
    n = len(x)
    rng_ = np.random.RandomState(seed)
    wins = 0
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s, min(s + block, n))
                              for s in rng_.randint(0, n - block, size=n // block + 1)])[:n]
        sx = (x[idx].sum() / (x[idx].std() + 1e-12)) / np.sqrt(len(idx))
        sy = (y[idx].sum() / (y[idx].std() + 1e-12)) / np.sqrt(len(idx))
        wins += (sx > sy)
    return wins / n_boot


GROUPS = {"30f_base": feats30, "venue": feats30 + VENUE, "bookdepth": feats30 + BD,
          "all": feats30 + VENUE + BD}
print("\n=== STD SEEDS (paired groups) ===")
ports_s, ns_s = {}, {}
for g, fl in GROUPS.items():
    ns_s[g], ports_s[g] = run(fl, SEEDS_STD, f"{g} std ({len(fl)}f)")
for g in ("venue", "bookdepth", "all"):
    p = boot_paired(ports_s[g], ports_s["30f_base"])
    print(f"  {g}: delta {ns_s[g]-ns_s['30f_base']:+.3f}, P(>base) = {p:.3f}")

print("\n=== ALT SEEDS (confirmation) ===")
ports_a, ns_a = {}, {}
for g, fl in GROUPS.items():
    ns_a[g], ports_a[g] = run(fl, SEEDS_ALT, f"{g} alt ({len(fl)}f)")
for g in ("venue", "bookdepth", "all"):
    p = boot_paired(ports_a[g], ports_a["30f_base"])
    print(f"  {g}: delta {ns_a[g]-ns_a['30f_base']:+.3f}, P(>base) = {p:.3f}")

print("\n" + "=" * 70)
for g in ("venue", "bookdepth", "all"):
    ds, da = ns_s[g]-ns_s["30f_base"], ns_a[g]-ns_a["30f_base"]
    ok = ds > 0 and da > 0
    print(f"  {g:10s}: std {ds:+.3f} | alt {da:+.3f} -> {'CANDIDATE' if ok else 'not confirmed'}")
print("R156 done.")
