#!/usr/bin/env python3
"""R192 — L/S mix × regime test on cached stack preds. VM, sims only (cheap).

Tests the user's two ideas WITHOUT retraining, on the validated W2W3 cached
preds (champ s10 + 0.5*venue5 spec s10):
  - dollar-neutral (3L/3S) vs baseline (4L/2S)
  - short-tilt (2L/4S, 1L/5S, 0L/6S) and long-tilt (6L/0S)
Each variant is split BULL (2025, <2026-01-01) vs BEAR (2026, >=2026-01-01) so
we see whether a short-tilt actually rescues the bear WITHOUT wrecking the bull
(a variant that only helps the cherry-picked bear window is overfitting).

Honest framing: our 'pred' is a CENTERED cross-sectional rank (relative, not
directional). Short-tilt bets on the weak absolute-direction component, so the
prior is that it helps bear modestly at best and hurts bull. This quantifies it.
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

SPLIT = pd.Timestamp("2026-01-01", tz="UTC")
MIXES = [(4, 2), (3, 3), (2, 4), (1, 5), (0, 6), (6, 0)]

print("Loading regime + cached preds...")
df, regime_df = r68.load_data()
if "timestamp" in regime_df.columns:
    regime_df = regime_df.set_index("timestamp")
regime_aug = r129.add_persistence(regime_df, lookback=L_FROZEN)
thr = r129.expanding_quantile_threshold(regime_aug[f"td_persist_{L_FROZEN}h"], Q_FROZEN, min_periods=720)
gate = (regime_aug[f"td_persist_{L_FROZEN}h"] < thr)
del df

champ = pd.read_parquet("cache/r167_champ30_s10_w23_preds.parquet")
spec = pd.read_parquet("cache/r166_spec_venue5_s10_preds.parquet")
mg = champ.merge(spec[["timestamp", "symbol", "pred"]].rename(columns={"pred": "spred"}),
                 on=["timestamp", "symbol"], how="left")
mg["spred"] = mg["spred"].fillna(0.0)
mg["pred"] = mg["pred"] + 0.5 * mg["spred"]
mg["timestamp"] = pd.to_datetime(mg["timestamp"], utc=True)


def seg(port, lo, hi):
    p = port[(port["timestamp"] >= lo) & (port["timestamp"] < hi)] if hi else \
        port[port["timestamp"] >= lo]
    if len(p) < 10:
        return None
    return (round(float(sharpe(p["net_ret"])), 2),
            round(float(((1 + p["net_ret"]).prod() - 1) * 100), 1), len(p))


results = {}
print(f"\n{'mix':8s} | {'FULL':>16s} | {'BULL 2025':>16s} | {'BEAR 2026':>16s}")
print("-" * 70)
for nl, ns in MIXES:
    port = simulate_r136(mg, regime_aug, nl, ns, dict(R114B_CFG),
                         cutoff_on=0.9, cutoff_off=0.8, min_risk_off_periods=2,
                         cost_fn=cost_prod_blended, funding_per_12h=0.00012,
                         exec_delay_penalty=0.0003, a1_cfg=A1_FROZEN, gate_series=gate)
    port["timestamp"] = pd.to_datetime(port["timestamp"], utc=True)
    full = seg(port, port["timestamp"].min(), None)
    bull = seg(port, port["timestamp"].min(), SPLIT)
    bear = seg(port, SPLIT, None)
    results[f"{nl}L{ns}S"] = {"full": full, "bull2025": bull, "bear2026": bear}

    def fmt(x):
        return f"Sh{x[0]:+.2f} {x[1]:+.0f}% n{x[2]}" if x else "—"
    print(f"{nl}L/{ns}S    | {fmt(full):>16s} | {fmt(bull):>16s} | {fmt(bear):>16s}")

with open("results_r192_ls_mix.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nRead: if a short-tilt beats 4L/2S in BEAR but tanks in BULL → regime-")
print("conditional sizing is worth building. If it tanks both → relative signal,")
print("directional bet doesn't work.")
print("R192 done.")
