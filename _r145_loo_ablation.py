#!/usr/bin/env python3
"""R145 — leave-one-out feature ablation on the fresh-retrain protocol.

Base = 30f (champion 31 minus cg_taker_imb, per R142). For each remaining
feature: drop it, retrain the full W1-W3 x 5-seed ensemble, honest S6 simulate,
record delta vs the 30f base. Positive delta when dropped = feature HURTS.

This is a SCREEN: single (deterministic) run per feature; |delta| < ~0.25 is
noise — candidates beyond that get a confirmation run with alternate seeds
before any feature-set change.

Writes incremental results to results_r145_loo.json after every run.
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
from _research_r22_models import SEEDS
from _research_r121_realistic_costs import R114B_CFG
from src.costs import cost_prod_blended
from _r136_s6_retest import simulate_r136

OUT = "results_r145_loo.json"

df, regime_df = load_data()
if "timestamp" in regime_df.columns:
    regime_df = regime_df.set_index("timestamp")

feat_all = [f for f in CHAMPION_FEAT_31 if f in df.columns]
BASE30 = [f for f in feat_all if f != "cg_taker_imb"]
assert len(BASE30) == 30, f"expected 30 base feats, got {len(BASE30)}"


def run(feats, label):
    t0 = time.time()
    no_rank = [f for f in feats if f in MARKET_LEVEL_FEATURES]
    preds = train_ensemble(df, feats, CONTINUOUS_WINDOWS, seeds=SEEDS,
                           cs_rank_exclude=no_rank)
    port = simulate_r136(
        preds, regime_df, 4, 2, dict(R114B_CFG),
        cutoff_on=0.9, cutoff_off=0.8, min_risk_off_periods=2,
        cost_fn=cost_prod_blended, funding_per_12h=0.00012,
        exec_delay_penalty=0.0003,
    )
    ns = sharpe(port["net_ret"])
    ret = ((1 + port["net_ret"]).prod() - 1) * 100
    print(f"  {label:34s} Net={ns:+.3f}  Ret={ret:+.1f}%  ({time.time()-t0:.0f}s)",
          flush=True)
    return round(float(ns), 4)


results = {}
print("=" * 80)
print(f"  R145 — LOO ablation | base=30f | {len(BASE30)} features to test")
print("=" * 80)

results["BASE_30f"] = run(BASE30, "BASE_30f")
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)

for i, feat in enumerate(BASE30, 1):
    feats = [x for x in BASE30 if x != feat]
    ns = run(feats, f"[{i:2d}/30] drop {feat}")
    results[feat] = ns
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

base = results["BASE_30f"]
print("\n" + "=" * 80)
print("  SUMMARY (delta = drop-it Sharpe - base; POSITIVE = feature HURTS)")
print("=" * 80)
rows = sorted(((v - base, k) for k, v in results.items() if k != "BASE_30f"),
              reverse=True)
for d, k in rows:
    flag = " <-- CANDIDATE HARMFUL" if d > 0.25 else (" <-- LOAD-BEARING" if d < -0.25 else "")
    print(f"  {k:28s} drop->{results[k]:+.3f}  delta={d:+.3f}{flag}")
print(f"\n  BASE_30f = {base:+.3f}")
print("R145 done.")
