#!/usr/bin/env python3
"""R142 — cg_taker_imb ablation.

How much Net Sharpe does the ONLY CoinGlass feature (cg_taker_imb) contribute?
Fresh retrain of the champion ensemble on 31 features vs 30 features (drop
cg_taker_imb), same CONTINUOUS_WINDOWS / seeds / S6 prod_blended costs / honest
simulate_r136 accounting. The DELTA is what matters (fresh retrain absolute
differs slightly from the canonical 2.831 cache, by design).

If delta is tiny (~0.05) the dead CoinGlass subscription is not worth renewing.
"""
from _preflight_check import check_versions
check_versions()

import warnings
warnings.filterwarnings("ignore")
import pandas as pd

from _research_r68_continuous_wf import (
    load_data, train_ensemble, CONTINUOUS_WINDOWS, CHAMPION_FEAT_31,
)
from _research_r35_new_features import MARKET_LEVEL_FEATURES
from _research_r22_models import SEEDS
from _research_r121_realistic_costs import R114B_CFG
from _research_r113_trend_cutoff_reopt import analyze_config
from src.costs import cost_prod_blended
from _r136_s6_retest import simulate_r136

df, regime_df = load_data()
if "timestamp" in regime_df.columns:
    regime_df = regime_df.set_index("timestamp")


def run(feats, label):
    no_rank = [f for f in feats if f in MARKET_LEVEL_FEATURES]
    preds = train_ensemble(df, feats, CONTINUOUS_WINDOWS, seeds=SEEDS,
                           cs_rank_exclude=no_rank)
    port = simulate_r136(
        preds, regime_df, 4, 2, dict(R114B_CFG),
        cutoff_on=0.9, cutoff_off=0.8, min_risk_off_periods=2,
        cost_fn=cost_prod_blended, funding_per_12h=0.00012,
        exec_delay_penalty=0.0003,
    )
    m = analyze_config(port, label)
    print(f"  {label:32s} Net={m['net_sharpe']:+.3f}  Gross={m.get('gross_sharpe', float('nan')):+.3f}  "
          f"Ret={m.get('total_ret_pct', float('nan')):.1f}%  DD={m.get('max_dd_pct', float('nan')):.1f}%  "
          f"periods={len(port)}", flush=True)
    return m["net_sharpe"], len(port)


feat31 = [f for f in CHAMPION_FEAT_31 if f in df.columns]
feat30 = [f for f in feat31 if f != "cg_taker_imb"]

print("=" * 78)
print("  R142 — cg_taker_imb ABLATION (fresh retrain, S6 prod_blended costs)")
print("=" * 78)
print(f"  feat31={len(feat31)} (has cg_taker_imb: {'cg_taker_imb' in feat31})")
print(f"  feat30={len(feat30)} (cg_taker_imb dropped: {'cg_taker_imb' not in feat30})")
print("-" * 78)

ns31, n31 = run(feat31, "31f (full, with cg_taker_imb)")
ns30, n30 = run(feat30, "30f (NO cg_taker_imb)")

print("-" * 78)
print(f"  DELTA (cg_taker_imb contribution) = {ns31 - ns30:+.3f} Net Sharpe")
print(f"  31f fresh retrain = {ns31:.3f}  (canonical cache baseline = 2.831)")
print(f"  30f (no CoinGlass) = {ns30:.3f}")
print("=" * 78)
if abs(ns31 - ns30) < 0.10:
    print("  VERDICT: cg_taker_imb contributes < 0.10 Sharpe -> CoinGlass renewal NOT worth it.")
else:
    print(f"  VERDICT: cg_taker_imb worth {ns31 - ns30:+.3f} Sharpe -> consider CoinGlass renewal.")
