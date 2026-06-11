"""Minimal verification runner: reproduce canonical S6 Net Sharpe 2.831
from the cached canonical preds, using the EXISTING simulate_r121 (imported,
not reimplemented) with the documented S6_prod_blended methodology.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from _research_r121_realistic_costs import (
    simulate_r121, R114B_CFG, cost_prod_blended, COST_MODELS,
)
from _research_r113_trend_cutoff_reopt import analyze_config, print_result

PREDS = "cache/r128_canonical_preds.parquet"
REGIME = "cache/r128_canonical_regime.parquet"

preds = pd.read_parquet(PREDS)
regime_df = pd.read_parquet(REGIME).set_index("timestamp")

print(f"preds rows={len(preds):,}  syms={preds['symbol'].nunique()}  "
      f"range {preds['timestamp'].min()} .. {preds['timestamp'].max()}")
print(f"regime rows={len(regime_df):,} cols={list(regime_df.columns)}")

cost_fn, funding = COST_MODELS["prod_blended"]
assert cost_fn is cost_prod_blended and funding == 0.00012

port = simulate_r121(
    preds, regime_df, 4, 2, dict(R114B_CFG),
    cutoff_on=0.9, cutoff_off=0.8,
    min_risk_off_periods=2, min_risk_on_periods=0,
    cost_fn=cost_fn, funding_per_12h=funding,
    exec_delay_penalty=0.0003,
)
m = analyze_config(port, "S6_prod_blended_CANONICAL")
print_result(m)
print("\nRAW METRICS DICT:")
for k, v in m.items():
    print(f"  {k}: {v}")

# Honesty audit numbers
n = len(port)
n_flat = int(port["risk_off"].sum())
active = port[~port["risk_off"]]
print(f"\nperiods total={n}  risk_off/flat={n_flat}  active={n - n_flat}")
ts = port["timestamp"]
span_days = (ts.max() - ts.min()).total_seconds() / 86400
expected_periods = span_days * 2 + 1
print(f"calendar span: {ts.min()} .. {ts.max()}  = {span_days:.1f} days "
      f"-> expected ~{expected_periods:.0f} periods at 12h; recorded {n} "
      f"({n / expected_periods * 100:.1f}% coverage)")
print(f"total cost (sum of per-period cost): {port['cost'].sum()*100:.2f}%")
print(f"avg turnover (positions opened+closed per period): {port['turnover'].mean():.2f}")

# Sharpe excluding flat periods (the inflated alternative) for comparison
from _research_r68_continuous_wf import sharpe
print(f"\nNet Sharpe ALL periods (canonical):     {sharpe(port['net_ret']):.3f}")
print(f"Net Sharpe ACTIVE-only (would inflate): {sharpe(active['net_ret']):.3f}")
print(f"Gross Sharpe ALL periods:               {sharpe(port['gross_ret']):.3f}")

# Re-run without exec-delay noise to isolate its effect
port_nonoise = simulate_r121(
    preds, regime_df, 4, 2, dict(R114B_CFG),
    cutoff_on=0.9, cutoff_off=0.8,
    min_risk_off_periods=2, min_risk_on_periods=0,
    cost_fn=cost_fn, funding_per_12h=funding,
    exec_delay_penalty=0.0,
)
print(f"Net Sharpe without 3bp exec noise:      {sharpe(port_nonoise['net_ret']):.3f}")

# Determinism check: run again, must be identical
port2 = simulate_r121(
    preds, regime_df, 4, 2, dict(R114B_CFG),
    cutoff_on=0.9, cutoff_off=0.8,
    min_risk_off_periods=2, min_risk_on_periods=0,
    cost_fn=cost_fn, funding_per_12h=funding,
    exec_delay_penalty=0.0003,
)
print(f"deterministic rerun identical: {port['net_ret'].equals(port2['net_ret'])}")
