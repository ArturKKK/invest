#!/usr/bin/env python3
"""Reproduce R48=1.66 with EXACT same pipeline (no R55 merge)."""
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
print("  R48 EXACT REPRODUCTION (5 seeds, no R55)")
print("=" * 60)

df, regime_df = load_research_frame()
df, _ = add_r35_features(df)
cg = load_cg_daily()
cg_daily = compute_cg_features(cg)
df, per_sym_cols, mkt_cols = add_cg_features(df, cg_daily)
feats, no_rank = make_feature_set(["cg_taker_imb"], mkt_cols)
print(f"Features ({len(feats)}): {feats}")
print(f"no_rank: {no_rank}")
print(f"Rows: {len(df):,}")

# Train EXACTLY like R48 phase34
preds = train_ensemble(df, feats, WINDOWS, l2=1.0, rolling=False,
                       label="R48_exact", cs_rank_exclude=no_rank)

# Eval with hybrid cost
for wn in ["W1", "W2", "W3"]:
    sub = preds[preds["window"] == wn]
    port = simulate_with_hybrid_costs(sub, regime_df, CANONICAL_EXEC_CFG)
    r = eval_with_costs(port, f"R48_{wn}")
    print(f"  {wn}: Sh={r['sharpe']:+.2f} (gross={r['sharpe_gross']:+.2f})"
          f"  cost={r.get('total_cost_pct', 0):.1f}%"
          f"  DD={r.get('max_dd_pct', 0):.1f}%")

port_all = simulate_with_hybrid_costs(preds, regime_df, CANONICAL_EXEC_CFG)
r_all = eval_with_costs(port_all, "R48_ALL")
print(f"  ALL: Sh={r_all['sharpe']:+.2f} (gross={r_all['sharpe_gross']:+.2f})"
      f"  cost={r_all.get('total_cost_pct', 0):.1f}%"
      f"  DD={r_all.get('max_dd_pct', 0):.1f}%")

# Also uniform for comparison
port_u = simulate_with_costs(preds, regime_df, CANONICAL_EXEC_CFG)
r_u = eval_with_costs(port_u, "R48_uniform")
print(f"  ALL (uniform 7bp): Sh={r_u['sharpe']:+.2f}")

print()
if abs(r_all["sharpe"] - 1.66) < 0.05:
    print("  ✅ REPRODUCED: R48=1.66 matches")
else:
    print(f"  ❌ NOT REPRODUCED: got {r_all['sharpe']:+.2f}, expected ~1.66")
    print(f"     Gap: {r_all['sharpe'] - 1.66:+.2f}")
