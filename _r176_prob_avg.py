#!/usr/bin/env python3
"""R176 — rank-avg vs prob-avg for multi-batch seed ensembles. SIMS ONLY, VM.

R175 anomaly: s30 built by averaging per-batch centered RANKS = 2.065, BELOW
all three component draws (3.080/2.397/2.458). Within a batch train_ensemble
averages PROBS across seeds before ranking — so the correct 30-seed artifact
is prob-avg → re-rank, not rank-avg. This grid measures both methods on every
batch pair + the triple, plus per-timestamp IC (variance reduction must show
up at the IC level regardless of sim nonlinearity).
Needs caches: r167/r166 (A), r172 (B), r175 (C), champ+spec each.
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

print("Loading regime + caches...")
df, regime_df = r68.load_data()
if "timestamp" in regime_df.columns:
    regime_df = regime_df.set_index("timestamp")
regime_aug = r129.add_persistence(regime_df, lookback=L_FROZEN)
thr = r129.expanding_quantile_threshold(regime_aug[f"td_persist_{L_FROZEN}h"], Q_FROZEN, min_periods=720)
gate = (regime_aug[f"td_persist_{L_FROZEN}h"] < thr)
del df

CH = {t: pd.read_parquet(p) for t, p in (
    ("A", "cache/r167_champ30_s10_w23_preds.parquet"),
    ("B", "cache/r172_champ30_s10b_w23_preds.parquet"),
    ("C", "cache/r175_champ30_s10c_w23_preds.parquet"))}
SP = {t: pd.read_parquet(p) for t, p in (
    ("A", "cache/r166_spec_venue5_s10_preds.parquet"),
    ("B", "cache/r172_spec_venue5_s10b_preds.parquet"),
    ("C", "cache/r175_spec_venue5_s10c_preds.parquet"))}


def run_gated(preds, label):
    port = simulate_r136(preds, regime_aug, 4, 2, dict(R114B_CFG),
                         cutoff_on=0.9, cutoff_off=0.8, min_risk_off_periods=2,
                         cost_fn=cost_prod_blended, funding_per_12h=0.00012,
                         exec_delay_penalty=0.0003, a1_cfg=A1_FROZEN, gate_series=gate)
    ns = sharpe(port["net_ret"])
    print(f"  {label:34s} Net={ns:+.3f}  n={len(port)}", flush=True)
    return ns, port


def combine(frames, method):
    """Combine per-batch pred frames. 'rank': mean of centered ranks.
    'prob': mean raw_prob -> per-timestamp re-rank (matches in-batch logic)."""
    col = "pred" if method == "rank" else "raw_prob"
    m = frames[0][["timestamp", "symbol", col, "fwd_ret"]].rename(columns={col: "v0"})
    for i, f in enumerate(frames[1:], 1):
        m = m.merge(f[["timestamp", "symbol", col]].rename(columns={col: f"v{i}"}),
                    on=["timestamp", "symbol"], how="inner")
    vc = [c for c in m.columns if c.startswith("v")]
    m["agg"] = m[vc].mean(axis=1)
    if method == "rank":
        m["pred"] = m["agg"]
    else:
        m["pred"] = m.groupby("timestamp")["agg"].rank(pct=True) - 0.5
    return m[["timestamp", "symbol", "pred", "fwd_ret"]]


def stack(champ, spec):
    mg = champ.merge(spec[["timestamp", "symbol", "pred"]].rename(columns={"pred": "spred"}),
                     on=["timestamp", "symbol"], how="left")
    mg["spred"] = mg["spred"].fillna(0.0)
    mg["pred"] = mg["pred"] + 0.5 * mg["spred"]
    return mg


def mean_ic(preds):
    ics = [spearmanr(g["pred"], g["fwd_ret"]).correlation
           for _, g in preds.groupby("timestamp") if g["pred"].nunique() > 2]
    ics = [i for i in ics if not np.isnan(i)]
    return float(np.mean(ics))


results = {}
COMBOS = [("AB", ["A", "B"]), ("AC", ["A", "C"]), ("BC", ["B", "C"]),
          ("ABC", ["A", "B", "C"])]
for method in ("rank", "prob"):
    print(f"\n=== method: {method}-avg ===")
    for name, tags in COMBOS:
        ch = combine([CH[t] for t in tags], method)
        sp = combine([SP[t] for t in tags], method)
        st = stack(ch, sp)
        ns, _ = run_gated(st, f"s{10*len(tags)} {name} ({method})")
        results[f"{method}_{name}"] = {"ns": round(float(ns), 3),
                                       "stack_ic": round(mean_ic(st), 4)}
# single-batch reference ICs
for t in ("A", "B", "C"):
    st = stack(CH[t][["timestamp", "symbol", "pred", "fwd_ret"]], SP[t])
    results[f"ic_s10_{t}"] = round(mean_ic(st), 4)
print("\nIC reference:", {k: v for k, v in results.items() if k.startswith("ic_")})

with open("results_r176_prob_avg.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("R176 done.")
