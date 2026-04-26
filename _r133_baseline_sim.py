"""R133 — baseline-only simulate on R128-style preds (fresh OOS)."""
from __future__ import annotations

import sys
import pandas as pd

sys.path.insert(0, ".")
from _preflight_check import check_versions
check_versions()

import _r130_validate_r129 as r130
import _r131_prod_sim_validate as r131

OOS_START = pd.Timestamp("2026-03-18", tz="UTC")
OOS_END = pd.Timestamp("2026-04-26", tz="UTC")


def main():
    print("=" * 80)
    print("  R133 — R128-style baseline on fresh OOS (39 days)")
    print(f"  train_end=2025-07-01  test=2026-03-18→2026-04-25")
    print("=" * 80)

    preds = pd.read_parquet("cache/r133_r128style_preds.parquet")
    regime_df = pd.read_parquet("cache/r132_regime_oos.parquet")
    if "timestamp" in regime_df.columns and regime_df.index.name != "timestamp":
        regime_df = regime_df.set_index("timestamp")

    print(f"  preds: {len(preds):,} rows, {preds['symbol'].nunique()} symbols")
    print(f"  range: {preds['timestamp'].min()} → {preds['timestamp'].max()}")

    port_full = r131.simulate_prod(preds, regime_df, n_long=4, n_short=2)
    port = port_full[(port_full["timestamp"] >= OOS_START) & (port_full["timestamp"] < OOS_END)].copy()

    if port.empty:
        print("  ❌ empty OOS slice")
        sys.exit(1)

    r = port["net_ret"].values
    s = r130.sharpe(r)
    so = r130.sortino(r)
    dd = r130.max_drawdown(r)
    cvar = r131.cvar_5pct(r)
    n = len(port)
    n_active = int((~port["risk_off"]).sum())

    print("\n" + "─" * 80)
    print(f"  OOS baseline (R128 cutoff 2025-07-01):")
    print(f"    Sharpe   = {s:+.3f}")
    print(f"    Sortino  = {so:+.3f}")
    print(f"    maxDD    = {dd*100:+.2f}%")
    print(f"    CVaR5%   = {cvar*1e4:+.1f}bp")
    print(f"    n        = {n} periods, {n_active} active")
    print("─" * 80)

    print("\n  Comparison:")
    print(f"    R128-style (train_end=2025-07-01) on OOS: Sharpe = {s:+.3f}")
    print(f"    R132       (train_end=2026-01-01) on OOS: Sharpe = +1.926")
    print(f"    Delta (newer cutoff vs older): {1.926 - s:+.3f}")


if __name__ == "__main__":
    main()
