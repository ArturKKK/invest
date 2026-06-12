#!/usr/bin/env python3
"""R172 — 20 seeds both legs (final variance-reduction booster). VM ONLY.

R166 showed 5→10 seeds bought +0.28-0.34 (pure variance reduction of the seed
lottery). This trains 10 NEW seeds per leg and combines with the cached 10
(equal-weight average of the centered-rank 'pred' columns) → 20-seed stack.
Seed count is not a tunable signal — adopt iff delta > 0 (paired bootstrap
reported for context, no P threshold needed for a pure variance move).

Needs caches: r167_champ30_s10_w23_preds.parquet, r166_spec_venue5_s10_preds.parquet.
Saves: r172_champ30_s10b_w23_preds.parquet, r172_spec_venue5_s10b_preds.parquet.
"""
from _preflight_check import check_versions
check_versions()

import json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import _research_r68_continuous_wf as r68
from _research_r68_continuous_wf import CHAMPION_FEAT_31, CONTINUOUS_WINDOWS, sharpe, train_ensemble
from _research_r35_new_features import MARKET_LEVEL_FEATURES
from _research_r121_realistic_costs import R114B_CFG
from src.costs import cost_prod_blended
from _r136_s6_retest import simulate_r136, A1_FROZEN, L_FROZEN, Q_FROZEN
import _r129_persistence_gate as r129

SEEDS_NEW = [2, 9, 15, 44, 101, 3, 10, 16, 45, 102]
VENUE = ["okx_binance_basis_z168", "okx_binance_basis_mom24",
         "coinbase_premium_z168", "coinbase_premium_mom24", "basis_range_z168"]
SPEC_START = pd.Timestamp("2023-07-01", tz="UTC")
W23 = CONTINUOUS_WINDOWS[1:]


def zscore(p, w):
    return (p - p.rolling(w, min_periods=w // 2).mean()) / (p.rolling(w, min_periods=w // 2).std() + 1e-12)


def boot_paired(a, b, n_boot=1000, block=14, seed=172):
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


# ── frame + venue features (verbatim from _r166) ─────────────────────────
print("Loading frame + building venue features...")
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

df["bsym"] = df["symbol"].str.replace("/", "", regex=False)
for name, p in panels.items():
    out = p.astype("float32").reset_index()
    idc = out.columns[0]
    out = out.melt(id_vars=idc, var_name="bsym", value_name=name).rename(columns={idc: "timestamp"})
    df = df.merge(out, on=["timestamp", "bsym"], how="left")
panels.clear()

feats30 = [f for f in CHAMPION_FEAT_31 if f in df.columns and f != "cg_taker_imb"]
df_spec = df[df["timestamp"] >= SPEC_START].copy()

regime_aug = r129.add_persistence(regime_df, lookback=L_FROZEN)
thr = r129.expanding_quantile_threshold(regime_aug[f"td_persist_{L_FROZEN}h"], Q_FROZEN, min_periods=720)
gate = (regime_aug[f"td_persist_{L_FROZEN}h"] < thr)


def run_gated(preds, label):
    port = simulate_r136(preds, regime_aug, 4, 2, dict(R114B_CFG),
                         cutoff_on=0.9, cutoff_off=0.8, min_risk_off_periods=2,
                         cost_fn=cost_prod_blended, funding_per_12h=0.00012,
                         exec_delay_penalty=0.0003, a1_cfg=A1_FROZEN, gate_series=gate)
    ns = sharpe(port["net_ret"])
    ret = ((1 + port["net_ret"]).prod() - 1) * 100
    dd = ((1 + port["net_ret"]).cumprod() / (1 + port["net_ret"]).cumprod().cummax() - 1).min() * 100
    print(f"  {label:38s} Net={ns:+.3f}  Ret={ret:+.1f}%  DD={dd:+.1f}%  n={len(port)}", flush=True)
    return ns, port


# ── train the NEW 10 seeds, cache immediately ─────────────────────────────
print("\n=== TRAIN champion30 NEW s10 (seeds batch B) ===", flush=True)
champ_b = train_ensemble(df, feats30, W23, seeds=SEEDS_NEW,
                         cs_rank_exclude=[f for f in feats30 if f in MARKET_LEVEL_FEATURES])
champ_b.to_parquet("cache/r172_champ30_s10b_w23_preds.parquet", index=False)
print("cached champ batch B", flush=True)
print("\n=== TRAIN specialist NEW s10 (seeds batch B) ===", flush=True)
spec_b = train_ensemble(df_spec, VENUE, W23, seeds=SEEDS_NEW, cs_rank_exclude=[])
spec_b.to_parquet("cache/r172_spec_venue5_s10b_preds.parquet", index=False)
print("cached spec batch B", flush=True)

champ_a = pd.read_parquet("cache/r167_champ30_s10_w23_preds.parquet")
spec_a = pd.read_parquet("cache/r166_spec_venue5_s10_preds.parquet")


def avg_preds(a, b):
    m = a.merge(b[["timestamp", "symbol", "pred"]].rename(columns={"pred": "pred_b"}),
                on=["timestamp", "symbol"], how="inner")
    m["pred"] = 0.5 * (m["pred"] + m["pred_b"])
    return m.drop(columns=["pred_b"])


champ20 = avg_preds(champ_a, champ_b)
spec20 = avg_preds(spec_a, spec_b)
print(f"champ20 rows {len(champ20):,} (a {len(champ_a):,}); spec20 rows {len(spec20):,}")


def stack(champ, spec):
    mg = champ.merge(spec[["timestamp", "symbol", "pred"]].rename(columns={"pred": "spred"}),
                     on=["timestamp", "symbol"], how="left")
    mg["spred"] = mg["spred"].fillna(0.0)
    mg["pred"] = mg["pred"] + 0.5 * mg["spred"]
    return mg


results = {}
print("\n=== STACKS ===")
ns10, p10 = run_gated(stack(champ_a, spec_a), "STACK s10 (R166 baseline)")
ns20, p20 = run_gated(stack(champ20, spec20), "STACK s20 (10+10 seeds)")
pwin = boot_paired(p20, p10)
print(f"   delta {ns20 - ns10:+.3f}, P(s20>s10) = {pwin:.3f}")
results = {"stack_s10": round(float(ns10), 3), "stack_s20": round(float(ns20), 3),
           "delta": round(float(ns20 - ns10), 3), "p": round(float(pwin), 3)}
# diagnostic: batch-B-only stack (independent 10-seed draw of the same architecture)
nsb, pb = run_gated(stack(champ_b, spec_b), "STACK s10-B (independent draw)")
results["stack_s10b"] = round(float(nsb), 3)

with open("results_r172_seeds20.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nRULE: seed count is not a signal — adopt s20 iff delta > 0.")
print("R172 done.")
