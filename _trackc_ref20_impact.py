"""TRACK C — REF20 cost impact on canonical cache (honest simulate_r136).

Baseline cell (cost_prod_blended) must reproduce 2.831; then one sim with
cost_prod_blended_ref20 (20% OKX referral cashback on fee components).
Funding unchanged 0.00012/12h. No training, no load_data — cache only.
"""
import numpy as np
import pandas as pd

from src.costs import cost_prod_blended, cost_prod_blended_ref20
from _r136_s6_retest import simulate_r136
from _research_r121_realistic_costs import R114B_CFG
from _research_r113_trend_cutoff_reopt import analyze_config

PREDS = "cache/r128_canonical_preds.parquet"
REGIME = "cache/r128_canonical_regime.parquet"


def main():
    preds = pd.read_parquet(PREDS)
    regime_df = pd.read_parquet(REGIME).set_index("timestamp")
    print(f"preds {len(preds):,} rows / {preds['symbol'].nunique()} syms; "
          f"regime {len(regime_df):,} rows")

    # sanity: tier numbers
    for s in ["BTC/USDT", "ADA/USDT", "SAND/USDT"]:
        print(f"  {s:10s} S6={cost_prod_blended(s)*1e4:.2f}bp  "
              f"ref20={cost_prod_blended_ref20(s)*1e4:.2f}bp")

    out = {}
    for label, fn in [("S6 prod_blended", cost_prod_blended),
                      ("REF20", cost_prod_blended_ref20)]:
        port = simulate_r136(
            preds, regime_df, 4, 2, dict(R114B_CFG),
            cutoff_on=0.9, cutoff_off=0.8,
            min_risk_off_periods=2, min_risk_on_periods=0,
            cost_fn=fn, funding_per_12h=0.00012,
            exec_delay_penalty=0.0003)
        assert len(port) == 1013, f"{label}: {len(port)} != 1013"
        m = analyze_config(port, label)
        out[label] = m
        print(f"{label:18s} Net={m['net_sharpe']:+.3f}  "
              f"Gross={m['gross_sharpe']:+.3f}  Ret={m['total_ret_pct']:.1f}%  "
              f"DD={m['max_dd_pct']:.1f}%  Cost={m['total_cost_pct']:.2f}%  "
              f"n={m['n_periods']}")

    base = out["S6 prod_blended"]["net_sharpe"]
    ref = out["REF20"]["net_sharpe"]
    assert round(base, 3) == 2.831, f"baseline {base} != 2.831"
    print(f"\nDELTA REF20 vs S6: {ref - base:+.3f}  "
          f"(Net {base:.3f} -> {ref:.3f})")
    dcost = (out['S6 prod_blended']['total_cost_pct']
             - out['REF20']['total_cost_pct'])
    print(f"cost saved: {dcost:.2f}pp of total cost over 1013 periods")


if __name__ == "__main__":
    main()
