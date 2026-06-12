#!/usr/bin/env python3
"""R178 — tail-period counterfactuals on the s30 deploy artifact. VM, sims only.

User question: what if we sat out the 3 WORST 12h periods (hindsight)?
Symmetric control: what if we missed the 3 BEST 12h periods instead?
Scenarios x leverage {1,3,5}: total / annualized / maxDD.
Also caches the portfolio return series (cache/r178_s30_port.parquet) so any
future what-if runs in seconds instead of re-simulating.
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


champ, spec = prob_avg(CH), prob_avg(SP)
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
port[["timestamp", "net_ret"]].to_parquet("cache/r178_s30_port.parquet", index=False)
print(f"s30 stack: Net={sharpe(port['net_ret']):+.3f}, n={len(port)} (series CACHED)")

r = port["net_ret"].values.copy()
ts = port["timestamp"]
worst_idx = np.argsort(r)[:3]
best_idx = np.argsort(r)[-3:][::-1]
print("\n3 WORST 12h periods:")
for i in worst_idx:
    print(f"  {ts.iloc[i]}  {r[i]*100:+.2f}%")
print("3 BEST 12h periods:")
for i in best_idx:
    print(f"  {ts.iloc[i]}  {r[i]*100:+.2f}%")

months = port["timestamp"].dt.to_period("M").nunique()
SCEN = {
    "full": r,
    "no_worst3": np.where(np.isin(np.arange(len(r)), worst_idx), 0.0, r),
    "no_best3": np.where(np.isin(np.arange(len(r)), best_idx), 0.0, r),
    "no_both": np.where(np.isin(np.arange(len(r)), np.concatenate([worst_idx, best_idx])), 0.0, r),
}
out = {"worst": [[str(ts.iloc[i]), round(float(r[i]) * 100, 2)] for i in worst_idx],
       "best": [[str(ts.iloc[i]), round(float(r[i]) * 100, 2)] for i in best_idx],
       "table": {}}
print(f"\n{'Сценарий':12s} {'L':>2s} | {'итог':>9s} | {'годовых':>9s} | {'maxDD':>7s}")
print("-" * 52)
for name, rr in SCEN.items():
    for L in (1, 3, 5):
        eq = np.cumprod(1 + L * rr)
        total = (eq[-1] - 1) * 100
        ann = ((1 + total / 100) ** (12.0 / months) - 1) * 100
        dd = ((eq / np.maximum.accumulate(eq)) - 1).min() * 100
        out["table"][f"{name}_{L}x"] = {"total": round(float(total), 1),
                                        "ann": round(float(ann), 1),
                                        "dd": round(float(dd), 1)}
        print(f"{name:12s} {L}x | {total:+8.1f}% | {ann:+8.1f}% | {dd:+6.1f}%")
    print("-" * 52)

with open("results_r178_tail_analysis.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print("R178 done.")
