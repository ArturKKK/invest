#!/usr/bin/env python3
"""R149 — confirmation + first model-level interaction test. VM ONLY.

1. BASE_30f with ALTERNATE seeds -> measures pure retrain noise scale.
2. 29f (drop residual_12h, the only R145 harmful candidate) std + alt seeds.
3. 30f + top-3 STRONG interactions from R147 screen.
4. 30f + all-8 STRONG interactions.
All: fresh retrain W1-W3, honest S6 simulate, vs BASE_30f std seeds (2.2605).
"""
from _preflight_check import check_versions
check_versions()

import json
import time
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

from _research_r68_continuous_wf import (
    load_data, train_ensemble, CONTINUOUS_WINDOWS, CHAMPION_FEAT_31, sharpe,
)
from _research_r35_new_features import MARKET_LEVEL_FEATURES
from _research_r121_realistic_costs import R114B_CFG
from src.costs import cost_prod_blended
from _r136_s6_retest import simulate_r136
from _r147_interaction_ic import CANDIDATES

SEEDS_STD = [0, 7, 13, 42, 99]
SEEDS_ALT = [1, 8, 14, 43, 100]
TOP3 = ["ret24_extremity", "upvol_share", "mom168_lowvol"]
STRONG8 = TOP3 + ["illiq_reversal", "breakout_volume", "idio_momentum",
                  "funding_extremity", "oi_conf_momentum"]
OUT = "results_r149.json"


def _attr(c, k):
    return getattr(c, k) if hasattr(c, k) else c[k]


df, regime_df = load_data()
if "timestamp" in regime_df.columns:
    regime_df = regime_df.set_index("timestamp")

# Compute interaction candidates as frame columns (rank-product features).
rank_cache = {}
def R(col):
    if col not in rank_cache:
        rank_cache[col] = df.groupby("timestamp")[col].rank(pct=True) - 0.5
    return rank_cache[col]

for c in CANDIDATES:
    nm = _attr(c, "name")
    if nm in STRONG8:
        df[nm] = _attr(c, "fn")(df, R).astype("float32")
        print(f"  computed {nm}: non-null {df[nm].notna().mean()*100:.1f}%", flush=True)
rank_cache.clear()

feat_all = [f for f in CHAMPION_FEAT_31 if f in df.columns]
BASE30 = [f for f in feat_all if f != "cg_taker_imb"]


def run(feats, label, seeds):
    t0 = time.time()
    no_rank = [f for f in feats if f in MARKET_LEVEL_FEATURES]
    preds = train_ensemble(df, feats, CONTINUOUS_WINDOWS, seeds=seeds,
                           cs_rank_exclude=no_rank)
    port = simulate_r136(
        preds, regime_df, 4, 2, dict(R114B_CFG),
        cutoff_on=0.9, cutoff_off=0.8, min_risk_off_periods=2,
        cost_fn=cost_prod_blended, funding_per_12h=0.00012,
        exec_delay_penalty=0.0003,
    )
    ns = sharpe(port["net_ret"])
    ret = ((1 + port["net_ret"]).prod() - 1) * 100
    print(f"  {label:34s} Net={ns:+.3f}  Ret={ret:+.1f}%  ({time.time()-t0:.0f}s)", flush=True)
    return round(float(ns), 4)


results = {}
print("=" * 80)
print("  R149 — noise scale + residual_12h confirmation + interaction features")
print("=" * 80)
EXPS = [
    ("BASE_30f_altseeds", BASE30, SEEDS_ALT),
    ("29f_no_residual12_std", [f for f in BASE30 if f != "residual_12h"], SEEDS_STD),
    ("29f_no_residual12_alt", [f for f in BASE30 if f != "residual_12h"], SEEDS_ALT),
    ("33f_top3_interactions", BASE30 + TOP3, SEEDS_STD),
    ("38f_strong8_interactions", BASE30 + STRONG8, SEEDS_STD),
]
for label, feats, seeds in EXPS:
    results[label] = run(feats, label, seeds)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

print("\n" + "=" * 80)
print("  SUMMARY (reference BASE_30f std seeds = 2.2605 from R145)")
print("=" * 80)
for k, v in results.items():
    print(f"  {k:34s} {v:+.3f}  (vs 30f-std: {v - 2.2605:+.3f})")
noise = abs(results.get("BASE_30f_altseeds", 2.2605) - 2.2605)
print(f"\n  Seed-noise scale (|alt - std| of identical 30f): {noise:.3f}")
print("  Interpretation rule: any |delta| < 2x noise scale is NOT actionable.")
print("R149 done.")
