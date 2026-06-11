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
NEWF = ["okx_binance_basis_z168", "okx_binance_basis_mom24",
        "coinbase_premium_z168", "coinbase_premium_mom24", "basis_range_z168"]


def zscore(p, w):
    return (p - p.rolling(w, min_periods=w // 2).mean()) / (p.rolling(w, min_periods=w // 2).std() + 1e-12)


def to_long(panel, name):
    out = panel.reset_index().melt(id_vars="index", var_name="bsym", value_name=name)
    out = out.rename(columns={"index": "timestamp"})
    return out


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


print("\n=== STD SEEDS (paired) ===")
ns30s, p30s = run(feats30, SEEDS_STD, "30f std")
ns35s, p35s = run(feats35, SEEDS_STD, "35f std (+5 venue)")
p_imp_s = boot_paired(p35s, p30s)
print(f"  -> delta {ns35s-ns30s:+.3f}, P(35f>30f) = {p_imp_s:.3f}")

print("\n=== ALT SEEDS (confirmation) ===")
ns30a, p30a = run(feats30, SEEDS_ALT, "30f alt")
ns35a, p35a = run(feats35, SEEDS_ALT, "35f alt (+5 venue)")
p_imp_a = boot_paired(p35a, p30a)
print(f"  -> delta {ns35a-ns30a:+.3f}, P(35f>30f) = {p_imp_a:.3f}")

print("\n" + "=" * 70)
print(f"  VERDICT: std delta {ns35s-ns30s:+.3f} (P={p_imp_s:.2f}) | "
      f"alt delta {ns35a-ns30a:+.3f} (P={p_imp_a:.2f})")
ok = (ns35s - ns30s > 0) and (ns35a - ns30a > 0) and min(p_imp_s, p_imp_a) >= 0.6
print(f"  {'PROMOTE to pristine-OOS check' if ok else 'NOT confirmed (seed-robust gate failed)'}")
print("R156 done.")
