"""R135 — reproduce canonical baseline locally on cached r128 preds.

Memory says local d9019ea simulate = 1.887 @ 1013 periods (or 2.179 honest).
Run baseline simulate on cached r128_preds_cont.parquet and report.
"""
import sys
import pandas as pd

sys.path.insert(0, ".")
from _preflight_check import check_versions
check_versions()

import _research_r68_continuous_wf as r68


def main():
    print("=" * 70)
    print("  R135 — local baseline reproduction")
    print("=" * 70)

    preds = pd.read_parquet("cache/r128_preds_cont.parquet")
    regime = pd.read_parquet("cache/r128_regime.parquet")
    if "timestamp" in regime.columns and regime.index.name != "timestamp":
        regime = regime.set_index("timestamp")

    print(f"  preds: {len(preds):,} rows, {preds['symbol'].nunique()} symbols")
    print(f"  range: {preds['timestamp'].min()} → {preds['timestamp'].max()}")
    print(f"  windows: {sorted(preds['window'].unique())}")

    for n_long, n_short, label in [(4, 2, "4L/2S"), (6, 3, "6L/3S")]:
        port = r68.simulate(preds, regime, n_long, n_short)
        r = r68.analyze(port, label)
        if r:
            print(f"\n  {label}:")
            print(f"    Net Sharpe   = {r['net_sharpe']:.3f}")
            print(f"    Gross Sharpe = {r['gross_sharpe']:.3f}")
            print(f"    n_periods    = {r['n_periods']}")
            print(f"    Total ret    = {r['total_ret_pct']:.1f}%")
            print(f"    MaxDD        = {r['max_dd_pct']:.1f}%")

    print("\n  Memory targets:")
    print("    d9019ea local (record risk-off): 1.887 @ 1013 periods")
    print("    cef6e2f VM    (skip risk-off):   3.777 @ 688  periods")


if __name__ == "__main__":
    main()
