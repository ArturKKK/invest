#!/usr/bin/env python3
"""R182 — cutoff freshness comparison on the COMMON fresh window. VM, sims only.

User question: does training fresher matter? Prior evidence (R132/R134): yes,
direction favors fresh. This re-checks at CHAMPION level on the cached V-grid
(R143): V1 train<2025-07, V2 train<2026-01, V3 train<2026-02-25 — same 30
features, 5 seeds each, predictions cached through ~2026-06-08.
Common window = intersection of all three (clean of every variant's train/val).
Also V3+spec stack (r162 cache) as the fresh-stack reference.
Segmented at 2026-04-05 (cg staleness hits ALL variants equally — fair).
HONEST: n is small (weeks); IC t-stats carry the signal, Sharpe is directional.
PRE-DECLARED USE: sanity check of the 'retrain fresh' deploy step — NOT a
knob to pick the cutoff by maximizing this window's Sharpe.
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
from _research_r68_continuous_wf import sharpe
from _research_r121_realistic_costs import R114B_CFG
from src.costs import cost_prod_blended
from _r136_s6_retest import simulate_r136, A1_FROZEN, L_FROZEN, Q_FROZEN
import _r129_persistence_gate as r129

CG_STALE = pd.Timestamp("2026-04-05", tz="UTC")

print("Loading regime...")
df, regime_df = r68.load_data()
if "timestamp" in regime_df.columns:
    regime_df = regime_df.set_index("timestamp")
regime_aug = r129.add_persistence(regime_df, lookback=L_FROZEN)
thr = r129.expanding_quantile_threshold(regime_aug[f"td_persist_{L_FROZEN}h"], Q_FROZEN, min_periods=720)
gate = (regime_aug[f"td_persist_{L_FROZEN}h"] < thr)
del df

V = {}
for tag, path in (("V1_jul25", "cache/r143_V1_stale_2025-07_30f_preds.parquet"),
                  ("V2_jan26", "cache/r143_V2_2026-01_R132_30f_preds.parquet"),
                  ("V3_feb26", "cache/r143_V3_fresh_2026-02-25_30f_preds.parquet")):
    d = pd.read_parquet(path)
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    V[tag] = d
    print(f"  {tag}: {len(d):,} rows, {d['timestamp'].min()} -> {d['timestamp'].max()}")
spec3 = pd.read_parquet("cache/r162_spec_v3_preds.parquet")
spec3["timestamp"] = pd.to_datetime(spec3["timestamp"], utc=True)
print(f"  spec_v3: {len(spec3):,} rows, {spec3['timestamp'].min()} -> {spec3['timestamp'].max()}")

common = set(V["V1_jul25"]["timestamp"]) & set(V["V2_jan26"]["timestamp"]) & set(V["V3_feb26"]["timestamp"])
lo, hi = min(common), max(common)
print(f"\ncommon window: {lo} -> {hi} ({len(common)} stamps)")


def _nw_tstat(x, lags=12):
    x = np.asarray(x, dtype=float); n = len(x)
    if n < 30: return np.nan
    d = x - x.mean(); var = d @ d / n
    for k in range(1, min(lags, n - 1) + 1):
        var += 2.0 * (1.0 - k / (lags + 1.0)) * (d[:-k] @ d[k:]) / n
    return x.mean() / (np.sqrt(max(var, 1e-18) / n) + 1e-18)


def run(preds, label):
    p = preds[preds["timestamp"].isin(common)].copy()
    port = simulate_r136(p, regime_aug, 4, 2, dict(R114B_CFG),
                         cutoff_on=0.9, cutoff_off=0.8, min_risk_off_periods=2,
                         cost_fn=cost_prod_blended, funding_per_12h=0.00012,
                         exec_delay_penalty=0.0003, a1_cfg=A1_FROZEN, gate_series=gate)
    port["timestamp"] = pd.to_datetime(port["timestamp"], utc=True)
    res = {}
    for seg, m in (("full", port["timestamp"].notna()),
                   ("clean", port["timestamp"] < CG_STALE),
                   ("stale", port["timestamp"] >= CG_STALE)):
        sp = port[m]
        if len(sp) < 8: continue
        res[seg] = {"sharpe": round(float(sharpe(sp["net_ret"])), 2),
                    "ret": round(float(((1 + sp["net_ret"]).prod() - 1) * 100), 1),
                    "n": int(len(sp))}
    ev = p.dropna(subset=["fwd_ret"])
    ics = pd.Series({ts: spearmanr(g["pred"], g["fwd_ret"]).correlation
                     for ts, g in ev.groupby("timestamp") if g["pred"].nunique() > 2}).dropna().sort_index()
    res["ic"] = {"ic": round(float(ics.mean()), 4), "t_nw12": round(float(_nw_tstat(ics.values)), 2)}
    parts = "  ".join(f"{k}: NS={v['sharpe']:+.2f} R={v['ret']:+.1f}% n={v['n']}"
                      for k, v in res.items() if k != "ic")
    print(f"  {label:16s} {parts} | IC={res['ic']['ic']:+.4f} t={res['ic']['t_nw12']:+.2f}", flush=True)
    return res


results = {}
print("\n=== CHAMPION-ONLY by cutoff (common window) ===")
for tag, d in V.items():
    results[tag] = run(d, tag)

print("\n=== V3 + specialist stack (fresh-stack reference) ===")
mg = V["V3_feb26"].merge(spec3[["timestamp", "symbol", "pred"]].rename(columns={"pred": "spred"}),
                         on=["timestamp", "symbol"], how="left")
mg["spred"] = mg["spred"].fillna(0.0)
mg["pred"] = mg["pred"] + 0.5 * mg["spred"]
results["V3_stack"] = run(mg, "V3+spec stack")

with open("results_r182_cutoff_compare.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("R182 done.")
