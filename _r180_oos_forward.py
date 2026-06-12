#!/usr/bin/env python3
"""R180 — TRUE forward test of the s30 deploy artifact: 2026-03-18 → 2026-06-08. VM.

The s30 stack's sims end at W3 test_end 2026-03-17. Everything after is
out-of-sample for THIS artifact. Caveats (pre-declared):
  1) CG-derived champion features (oi_*, taker_cvd_*, ls_divergence,
     cum_funding) go stale after ~2026-04-05 (subscription lapse) → results
     segmented pre/post 04-05; post-04-05 is a degraded-features run.
  2) W3 train cutoff is 2025-07-01 — ~11 months stale by June 2026. The
     deploy model will be retrained fresher, so this is a LOWER BOUND.
  3) n≈165 periods — wide CI; rank-IC t-test carries more power than Sharpe.

Trains W3ext (same cutoff/val as W3, test 2026-03-18..06-08) with ALL 30
seeds both legs (prob-avg by construction), stacks k=0.5 + GATED_A1, reports
1x and 3.5x+VT (VT warmed up on the cached W2W3 series).
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
from _research_r68_continuous_wf import CHAMPION_FEAT_31, sharpe, train_ensemble
from _research_r35_new_features import MARKET_LEVEL_FEATURES
from _research_r121_realistic_costs import R114B_CFG
from src.costs import cost_prod_blended
from _r136_s6_retest import simulate_r136, A1_FROZEN, L_FROZEN, Q_FROZEN
import _r129_persistence_gate as r129

SEEDS30 = [0, 7, 13, 42, 99, 1, 8, 14, 43, 100,
           2, 9, 15, 44, 101, 3, 10, 16, 45, 102,
           4, 11, 17, 46, 103, 5, 12, 18, 47, 104]
VENUE = ["okx_binance_basis_z168", "okx_binance_basis_mom24",
         "coinbase_premium_z168", "coinbase_premium_mom24", "basis_range_z168"]
SPEC_START = pd.Timestamp("2023-07-01", tz="UTC")
W3EXT = [{"name": "W3ext", "train_end": "2025-07-01", "val_start": "2025-07-01",
          "val_end": "2025-10-31", "test_start": "2026-03-18", "test_end": "2026-06-08"}]
CG_STALE = pd.Timestamp("2026-04-05", tz="UTC")


def zscore(p, w):
    return (p - p.rolling(w, min_periods=w // 2).mean()) / (p.rolling(w, min_periods=w // 2).std() + 1e-12)


def _nw_tstat(x, lags=12):
    x = np.asarray(x, dtype=float); n = len(x)
    if n < 30: return np.nan
    d = x - x.mean(); var = d @ d / n
    for k in range(1, min(lags, n - 1) + 1):
        var += 2.0 * (1.0 - k / (lags + 1.0)) * (d[:-k] @ d[k:]) / n
    return x.mean() / (np.sqrt(max(var, 1e-18) / n) + 1e-18)


print("Loading frame + venue features...")
df, regime_df = r68.load_data()
if "timestamp" in regime_df.columns:
    regime_df = regime_df.set_index("timestamp")
bclose = df.pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="first")
bclose.columns = [c.replace("/", "") for c in bclose.columns]
grid = bclose.index
print(f"frame end: {grid.max()}")
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

print("\n=== TRAIN W3ext 30 seeds both legs (slow) ===", flush=True)
champ = train_ensemble(df, feats30, W3EXT, seeds=SEEDS30,
                       cs_rank_exclude=[f for f in feats30 if f in MARKET_LEVEL_FEATURES])
champ.to_parquet("cache/r180_champ30_s30_w3ext_preds.parquet", index=False)
spec = train_ensemble(df_spec, VENUE, W3EXT, seeds=SEEDS30, cs_rank_exclude=[])
spec.to_parquet("cache/r180_spec_s30_w3ext_preds.parquet", index=False)
print("preds cached", flush=True)

mg = champ.merge(spec[["timestamp", "symbol", "pred"]].rename(columns={"pred": "spred"}),
                 on=["timestamp", "symbol"], how="left")
mg["spred"] = mg["spred"].fillna(0.0)
mg["pred"] = mg["pred"] + 0.5 * mg["spred"]

port = simulate_r136(mg, regime_aug, 4, 2, dict(R114B_CFG),
                     cutoff_on=0.9, cutoff_off=0.8, min_risk_off_periods=2,
                     cost_fn=cost_prod_blended, funding_per_12h=0.00012,
                     exec_delay_penalty=0.0003, a1_cfg=A1_FROZEN, gate_series=gate)
port["timestamp"] = pd.to_datetime(port["timestamp"], utc=True)
port = port.sort_values("timestamp").reset_index(drop=True)
port[["timestamp", "net_ret"]].to_parquet("cache/r180_oos_port.parquet", index=False)

results = {}
print("\n=== FORWARD OOS (artifact never saw this window) ===")
for tag, mask in (("full_0318_0608", port["timestamp"] >= pd.Timestamp("2026-03-18", tz="UTC")),
                  ("clean_0318_0405", port["timestamp"] < CG_STALE),
                  ("stale_0405_0608", port["timestamp"] >= CG_STALE)):
    seg = port[mask] if tag != "full_0318_0608" else port
    if len(seg) < 10:
        continue
    ns = sharpe(seg["net_ret"])
    ret = ((1 + seg["net_ret"]).prod() - 1) * 100
    dd = ((1 + seg["net_ret"]).cumprod() / (1 + seg["net_ret"]).cumprod().cummax() - 1).min() * 100
    results[tag] = {"sharpe": round(float(ns), 3), "ret": round(float(ret), 1),
                    "dd": round(float(dd), 1), "n": int(len(seg))}
    print(f"  {tag:18s} Sharpe={ns:+.3f}  Ret={ret:+.1f}%  DD={dd:+.1f}%  n={len(seg)}")

# rank-IC with NW t (more power than Sharpe at this n)
ev = mg.dropna(subset=["fwd_ret"])
ics_all = [(ts, spearmanr(g["pred"], g["fwd_ret"]).correlation)
           for ts, g in ev.groupby("timestamp") if g["pred"].nunique() > 2]
s = pd.Series({ts: ic for ts, ic in ics_all if not np.isnan(ic)}).sort_index()
for tag, ss in (("ic_full", s), ("ic_clean", s[s.index < CG_STALE]), ("ic_stale", s[s.index >= CG_STALE])):
    if len(ss) < 30:
        continue
    results[tag] = {"ic": round(float(ss.mean()), 4), "t_nw12": round(float(_nw_tstat(ss.values)), 2),
                    "n": int(len(ss))}
    print(f"  {tag:18s} IC={ss.mean():+.4f}  t_NW12={_nw_tstat(ss.values):+.2f}  n={len(ss)}")

# 3.5x + VT (warmed on cached W2W3 series)
hist = pd.read_parquet("cache/r178_s30_port.parquet")
hist["timestamp"] = pd.to_datetime(hist["timestamp"], utc=True)
full = pd.concat([hist, port[["timestamp", "net_ret"]]], ignore_index=True).sort_values("timestamp")
full = full.drop_duplicates("timestamp").reset_index(drop=True)
r = full["net_ret"].astype(float)
vol = r.rolling(30, min_periods=30).std()
ref = vol.expanding(min_periods=60).median()
sc = (ref / vol).clip(0.5, 1.0).fillna(1.0)
oos_mask = full["timestamp"] >= pd.Timestamp("2026-03-18", tz="UTC")
rv = (r * sc)[oos_mask]
for L in (1, 3.5):
    eq = np.cumprod(1 + L * np.asarray(rv, float))
    total = (eq[-1] - 1) * 100
    dd = ((eq / np.maximum.accumulate(eq)) - 1).min() * 100
    results[f"vt_{L}x_oos"] = {"ret": round(float(total), 1), "dd": round(float(dd), 1)}
    print(f"  VT {L}x OOS: Ret={total:+.1f}%  DD={dd:+.1f}%")

with open("results_r180_oos_forward.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("R180 done.")
