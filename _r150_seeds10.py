#!/usr/bin/env python3
"""R150 — ensemble variance reduction: 10 seeds vs 5. VM ONLY.

Hypothesis: part of the canonical-vs-fresh-retrain gap (2.831 vs ~2.26) is
ensemble seed noise; doubling seeds tightens averaging and may lift the mean.
Protocol identical to R145 BASE_30f (fresh retrain W1-W3, honest S6 sim).
Reference points: 5-seed std = 2.2605, 5-seed alt = 2.305 (noise +-0.045).
"""
from _preflight_check import check_versions
check_versions()

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

import sys
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
ALL = [0, 7, 13, 42, 99, 1, 8, 14, 43, 100, 2, 9, 15, 44, 101, 3, 10, 16, 45, 102]
SEEDS10 = ALL[:N]
print(f"SEEDS({N}): {SEEDS10}")

df, regime_df = load_data()
if "timestamp" in regime_df.columns:
    regime_df = regime_df.set_index("timestamp")
feats = [f for f in CHAMPION_FEAT_31 if f in df.columns and f != "cg_taker_imb"]
no_rank = [f for f in feats if f in MARKET_LEVEL_FEATURES]

preds = train_ensemble(df, feats, CONTINUOUS_WINDOWS, seeds=SEEDS10,
                       cs_rank_exclude=no_rank)
port = simulate_r136(
    preds, regime_df, 4, 2, dict(R114B_CFG),
    cutoff_on=0.9, cutoff_off=0.8, min_risk_off_periods=2,
    cost_fn=cost_prod_blended, funding_per_12h=0.00012, exec_delay_penalty=0.0003,
)
ns = sharpe(port["net_ret"])
ret = ((1 + port["net_ret"]).prod() - 1) * 100
dd = ((1 + port["net_ret"]).cumprod() / (1 + port["net_ret"]).cumprod().cummax() - 1).min() * 100
print(f"\nR150 {len(SEEDS10)}-SEED 30f: Net={ns:+.3f}  Ret={ret:+.1f}%  DD={dd:+.1f}%  n={len(port)}")
print(f"  vs 5-seed std 2.2605: {ns-2.2605:+.3f} | vs 5-seed alt 2.305: {ns-2.305:+.3f}")
preds.to_parquet(f"cache/r150_seeds{len(SEEDS10)}_preds.parquet", index=False)
print("R150 done.")
