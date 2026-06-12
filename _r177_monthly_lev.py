#!/usr/bin/env python3
"""R177 — monthly returns + leverage table for the deploy artifact. VM, sims only.

Artifact: s30 prob-avg stack (champ A+B+C raw_prob mean -> re-rank; spec same;
+0.5 blend; GATED_A1), W2W3. Leverage L scales each 12h net return by L
(costs/funding scale linearly with notional on perps). Liquidation risk and
intraperiod spikes are NOT modeled — flagged in the output.
"""
from _preflight_check import check_versions
check_versions()

import json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import _research_r68_continuous_wf as r68
from _research_r68_continuous_wf import sharpe
from _research_r121_realistic_costs import R114B_CFG
from src.costs import cost_prod_blended
from _r136_s6_retest import simulate_r136, A1_FROZEN, L_FROZEN, Q_FROZEN
import _r129_persistence_gate as r129

print("Loading regime + caches...")
df, regime_df = r68.load_data()
if "timestamp" in regime_df.columns:
    regime_df = regime_df.set_index("timestamp")
regime_aug = r129.add_persistence(regime_df, lookback=L_FROZEN)
thr = r129.expanding_quantile_threshold(regime_aug[f"td_persist_{L_FROZEN}h"], Q_FROZEN, min_periods=720)
gate = (regime_aug[f"td_persist_{L_FROZEN}h"] < thr)
del df

CH = [pd.read_parquet(p) for p in ("cache/r167_champ30_s10_w23_preds.parquet",
                                   "cache/r172_champ30_s10b_w23_preds.parquet",
                                   "cache/r175_champ30_s10c_w23_preds.parquet")]
SP = [pd.read_parquet(p) for p in ("cache/r166_spec_venue5_s10_preds.parquet",
                                   "cache/r172_spec_venue5_s10b_preds.parquet",
                                   "cache/r175_spec_venue5_s10c_preds.parquet")]


def prob_avg(frames):
    m = frames[0][["timestamp", "symbol", "raw_prob", "fwd_ret"]].rename(columns={"raw_prob": "v0"})
    for i, f in enumerate(frames[1:], 1):
        m = m.merge(f[["timestamp", "symbol", "raw_prob"]].rename(columns={"raw_prob": f"v{i}"}),
                    on=["timestamp", "symbol"], how="inner")
    vc = [c for c in m.columns if c.startswith("v")]
    m["agg"] = m[vc].mean(axis=1)
    m["pred"] = m.groupby("timestamp")["agg"].rank(pct=True) - 0.5
    return m[["timestamp", "symbol", "pred", "fwd_ret"]]


champ = prob_avg(CH)
spec = prob_avg(SP)
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
print(f"s30 prob-avg stack: Net={sharpe(port['net_ret']):+.3f}, n={len(port)}")

out = {"sharpe_1x": round(float(sharpe(port["net_ret"])), 3), "monthly": {}, "summary": {}}
print(f"\n{'Месяц':8s} | {'1x':>8s} | {'3x':>8s} | {'5x':>8s}")
print("-" * 44)
port["month"] = port["timestamp"].dt.strftime("%Y-%m")
for month, g in port.groupby("month"):
    row = {}
    for L in (1, 3, 5):
        r = (1 + L * g["net_ret"]).prod() - 1
        row[f"{L}x"] = round(float(r) * 100, 1)
    out["monthly"][month] = row
    print(f"{month:8s} | {row['1x']:+7.1f}% | {row['3x']:+7.1f}% | {row['5x']:+7.1f}%")

print("-" * 44)
for L in (1, 3, 5):
    eq = (1 + L * port["net_ret"]).cumprod()
    total = float(eq.iloc[-1] - 1) * 100
    dd = float((eq / eq.cummax() - 1).min()) * 100
    months = len(out["monthly"])
    ann = ((1 + total / 100) ** (12.0 / months) - 1) * 100
    wiped = bool((eq <= 0).any() or (L * port["net_ret"] <= -1).any())
    out["summary"][f"{L}x"] = {"total_pct": round(total, 1), "ann_pct": round(ann, 1),
                               "maxdd_pct": round(dd, 1), "wiped": wiped}
    print(f"{L}x: total {total:+.1f}% | annualized {ann:+.1f}% | maxDD {dd:+.1f}%"
          f"{' | WIPED OUT' if wiped else ''}")
worst_period = float(port["net_ret"].min()) * 100
print(f"\nworst single 12h period at 1x: {worst_period:+.2f}% "
      f"(at 5x: {worst_period*5:+.1f}%) — liquidation/intraperiod risk NOT modeled")
out["worst_period_1x_pct"] = round(worst_period, 2)

with open("results_r177_monthly_lev.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print("R177 done.")
