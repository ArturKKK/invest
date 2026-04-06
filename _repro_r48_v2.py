#!/usr/bin/env python3
"""Reproduce R48=1.66: NO cs_rank_exclude (all 31 feats ranked)."""
import warnings; warnings.filterwarnings("ignore")
from _research_r35_new_features import load_research_frame, add_r35_features
from _research_r47_coinglass import (
    CANONICAL_EXEC_CFG, add_cg_features, compute_cg_features,
    load_cg_daily, make_feature_set,
)
from _research_round7 import WINDOWS
from _research_r30b_fixed import train_ensemble, eval_with_costs, simulate_with_costs
from _research_r48_cost import simulate_with_hybrid_costs

print("=" * 60)
print("  R48 REPRODUCTION v2: WITHOUT cs_rank_exclude")
print("=" * 60)

df, regime_df = load_research_frame()
df, _ = add_r35_features(df)
cg = load_cg_daily()
cg_daily = compute_cg_features(cg)
df, per_sym_cols, mkt_cols = add_cg_features(df, cg_daily)
feats, no_rank = make_feature_set(["cg_taker_imb"], mkt_cols)
print(f"Features ({len(feats)})")
print(f"no_rank (NOT USED): {no_rank}")

# Train WITHOUT cs_rank_exclude — all 31 feats get ranked
preds = train_ensemble(df, feats, WINDOWS, l2=1.0, rolling=False,
                       label="R48_no_excl")

for wn in ["W1", "W2", "W3"]:
    sub = preds[preds["window"] == wn]
    # uniform cost
    port_u = simulate_with_costs(sub, regime_df, CANONICAL_EXEC_CFG)
    r_u = eval_with_costs(port_u, f"uni_{wn}")
    # hybrid cost
    port_h = simulate_with_hybrid_costs(sub, regime_df, CANONICAL_EXEC_CFG)
    r_h = eval_with_costs(port_h, f"hyb_{wn}")
    print(f"  {wn}: uniform={r_u['sharpe']:+.2f}  hybrid={r_h['sharpe']:+.2f}")

# ALL
port_u = simulate_with_costs(preds, regime_df, CANONICAL_EXEC_CFG)
r_u = eval_with_costs(port_u, "uni_ALL")
port_h = simulate_with_hybrid_costs(preds, regime_df, CANONICAL_EXEC_CFG)
r_h = eval_with_costs(port_h, "hyb_ALL")
print(f"  ALL: uniform={r_u['sharpe']:+.2f}  hybrid={r_h['sharpe']:+.2f}")

print()
print(f"  Expected: uniform~1.31, hybrid~1.66")
print(f"  Got:      uniform={r_u['sharpe']:+.2f}, hybrid={r_h['sharpe']:+.2f}")
if abs(r_h["sharpe"] - 1.66) < 0.1:
    print("  ✅ MATCH!")
else:
    print(f"  ❌ Gap: {r_h['sharpe'] - 1.66:+.2f}")
