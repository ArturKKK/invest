"""R134 — simulate baseline on freshest-cutoff preds."""
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


def sim(path, label):
    preds = pd.read_parquet(path)
    regime = pd.read_parquet("cache/r132_regime_oos.parquet")
    if "timestamp" in regime.columns and regime.index.name != "timestamp":
        regime = regime.set_index("timestamp")
    port_full = r131.simulate_prod(preds, regime, n_long=4, n_short=2)
    port = port_full[(port_full["timestamp"] >= OOS_START) & (port_full["timestamp"] < OOS_END)]
    r = port["net_ret"].values
    return {
        "label": label,
        "sharpe": r130.sharpe(r),
        "sortino": r130.sortino(r),
        "ret": float(r.sum()),
        "maxDD": r130.max_drawdown(r),
        "cvar5": r131.cvar_5pct(r),
        "n": len(port),
        "n_act": int((~port["risk_off"]).sum()),
    }


def main():
    print("=" * 80)
    print("  Apples-to-apples: same architecture, same OOS, different train_end")
    print("  OOS window: 2026-03-18 → 2026-04-25 (39 days, 77 12h periods)")
    print("=" * 80)

    rows = [
        sim("cache/r133_r128style_preds.parquet", "R128 cutoff 2025-07-01 (9 mo old)"),
        sim("cache/r132_oos_preds.parquet",       "R132 cutoff 2026-01-01 (3 mo old)"),
        sim("cache/r134_fresh_preds.parquet",     "R134 cutoff 2026-03-15 (3 d old) "),
    ]

    print(f"\n  {'Model':<38s}  {'Sharpe':>8s} {'Sortino':>9s} {'Ret%':>8s} {'maxDD':>8s} {'CVaR5%':>9s} {'n_act':>6s}")
    print("  " + "─" * 90)
    for r in rows:
        print(f"  {r['label']:<38s}  {r['sharpe']:+8.3f} {r['sortino']:+9.3f} "
              f"{r['ret']*100:+7.2f}% {r['maxDD']*100:+7.2f}% {r['cvar5']*1e4:+8.1f}bp {r['n_act']:>6d}")
    print()

    # Deltas vs R128 baseline
    base = rows[0]["sharpe"]
    print("  Delta vs R128 (oldest):")
    for r in rows[1:]:
        print(f"    {r['label']}: ΔS = {r['sharpe'] - base:+.3f}")

    if all(r["sharpe"] <= 0 or r["ret"] <= 0 for r in rows):
        print("\n  ABSOLUTE GUARD: all variants have negative OOS Sharpe/return. Treat deltas as relative damage control, not a deploy signal.")


if __name__ == "__main__":
    main()
