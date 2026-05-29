"""R132 — OOS forward-test validation (Apr 26, 2026).

Replicates R131 prod-simulate analysis on the OOS window W4_OOS
(2026-03-18 → 2026-04-25), using fresh predictions from `_r132_oos_train.py`.

Compares:
  baseline (no overlay)
  A1 always-on (frozen 0.25/0.60)
  Gated A1 (L=720, q=0.20, expanding-quantile threshold)

Critic gates:
    Absolute OOS Sharpe <= 0 or Ret <= 0 -> DO NOT DEPLOY
    Otherwise, use relative OOS ΔSharpe as a secondary tie-breaker.
"""
from __future__ import annotations

import time
import sys

import numpy as np
import pandas as pd

from _preflight_check import check_versions

check_versions()

import _r129_persistence_gate as r129
import _r130_validate_r129 as r130
import _r131_prod_sim_validate as r131

L = 720
Q = 0.20
A1 = {"trend_thr": 0.25, "weak_scale": 0.60}

OOS_START = pd.Timestamp("2026-03-18", tz="UTC")
OOS_END = pd.Timestamp("2026-04-26", tz="UTC")


def slice_oos(port: pd.DataFrame) -> pd.DataFrame:
    return port[(port["timestamp"] >= OOS_START) & (port["timestamp"] < OOS_END)].copy()


def report(label: str, port: pd.DataFrame, base: pd.DataFrame | None = None):
    if port.empty:
        print(f"  {label:<24s} EMPTY")
        return None
    r = port["net_ret"].values
    s = r130.sharpe(r)
    so = r130.sortino(r)
    dd = r130.max_drawdown(r)
    cvar = r131.cvar_5pct(r)
    ret = float(r.sum())
    n = len(port)
    n_active = int((~port["risk_off"]).sum())
    n_gate = int(port["gate_on"].sum())
    delta = ""
    if base is not None and not base.empty:
        delta = f"  ΔS={s - r130.sharpe(base['net_ret'].values):+.3f}"
        print(f"  {label:<24s} S={s:+.3f}{delta}  Sortino={so:+.3f}  Ret={ret*100:+.2f}%  maxDD={dd*100:+.2f}%  "
          f"CVaR5%={cvar*1e4:+.1f}bp  n={n} act={n_active} gate={n_gate}")
    return {"sharpe": s, "sortino": so, "maxDD": dd, "cvar5": cvar,
            "ret": ret, "n": n, "n_active": n_active, "n_gate": n_gate}


def main():
    t0 = time.time()
    print("=" * 80)
    print("  R132 — OOS forward-test validation")
    print(f"  OOS window: {OOS_START.date()} → {OOS_END.date()}")
    print(f"  Frozen params: L={L}, q={Q}, A1={A1}")
    print("=" * 80)

    preds = pd.read_parquet("cache/r132_oos_preds.parquet")
    regime_df = pd.read_parquet("cache/r132_regime_oos.parquet")
    if "timestamp" in regime_df.columns and regime_df.index.name != "timestamp":
        regime_df = regime_df.set_index("timestamp")

    print(f"  preds rows: {len(preds):,}  range: {preds['timestamp'].min()} → {preds['timestamp'].max()}")
    print(f"  regime range: {regime_df.index.min()} → {regime_df.index.max()}")

    # Add persistence (over full regime history, so threshold is well-warmed at OOS)
    regime_aug = r129.add_persistence(regime_df, lookback=L)
    persist_col = f"td_persist_{L}h"
    persist_ts = regime_aug[persist_col]
    thr_series = r129.expanding_quantile_threshold(persist_ts, Q, min_periods=720)

    # ── Three sims on OOS-only preds ──
    print("\n  1) Baseline (prod sim, no overlay)")
    base_full = r131.simulate_prod(preds, regime_aug, n_long=4, n_short=2)
    print("  2) A1 always-on")
    a1_full = r131.simulate_prod(preds, regime_aug, n_long=4, n_short=2, a1_cfg=A1)
    print("  3) Gated A1 (FROZEN)")
    gated_full = r131.simulate_prod(preds, regime_aug, n_long=4, n_short=2, a1_cfg=A1,
                                     gate_persist_col=persist_col, gate_threshold_series=thr_series)

    # OOS slices
    base = slice_oos(base_full)
    a1 = slice_oos(a1_full)
    gated = slice_oos(gated_full)

    print("\n" + "─" * 80)
    print("  OOS SLICE (2026-03-18 → 2026-04-26)")
    print("─" * 80)
    base_m = report("baseline", base)
    a1_m = report("A1 always", a1, base)
    gated_m = report("Gated A1 (FROZEN)", gated, base)

    # Bootstrap on OOS only
    print("\n  OOS bootstrap (gated vs baseline)")
    m = (
        gated[["timestamp", "net_ret"]].rename(columns={"net_ret": "alt"}).merge(
            base[["timestamp", "net_ret"]].rename(columns={"net_ret": "b"}),
            on="timestamp", how="inner"
        ).sort_values("timestamp").reset_index(drop=True)
    )
    if len(m) >= 14:
        bs_iid = r129.boot_p_improvement(m["b"].values, m["alt"].values, n_boot=5000)
        bs_blk = r130.block_bootstrap_diff(m["b"].values, m["alt"].values,
                                            block_len=14, n_boot=5000)
        print(f"    iid    P(Δ>0)={bs_iid['p_pos']:.3f}  CI95=[{bs_iid['ci_low']:+.3f},{bs_iid['ci_high']:+.3f}]")
        print(f"    block  P(Δ>0)={bs_blk['p_pos']:.3f}  CI95=[{bs_blk['ci_low']:+.3f},{bs_blk['ci_high']:+.3f}]")
    else:
        print(f"    too few periods for bootstrap (n={len(m)})")
        bs_blk = None

    # Verdict per critic
    print("\n" + "=" * 80)
    print("  CRITIC GATE (R130)")
    print("=" * 80)
    if base_m is None or gated_m is None:
        print("  ❌ Cannot evaluate — empty slice.")
        return
    delta = gated_m["sharpe"] - base_m["sharpe"]
    a1_delta = (a1_m["sharpe"] - base_m["sharpe"]) if a1_m else float("nan")
    print(f"  OOS ΔSharpe (gated  vs base): {delta:+.3f}")
    print(f"  OOS ΔSharpe (always vs base): {a1_delta:+.3f}")
    print(f"  OOS maxDD  (gated/base): {gated_m['maxDD']*100:+.2f}%  /  {base_m['maxDD']*100:+.2f}%")
    print(f"  OOS Sharpe (gated/base): {gated_m['sharpe']:+.3f}  /  {base_m['sharpe']:+.3f}")
    print(f"  OOS Ret    (gated/base): {gated_m['ret']*100:+.2f}%  /  {base_m['ret']*100:+.2f}%")
    print(f"  Active periods OOS: {gated_m['n_active']}/{gated_m['n']}  gate fires: {gated_m['n_gate']}")

    if gated_m["sharpe"] <= 0 or gated_m["ret"] <= 0:
        verdict = "❌ DO NOT DEPLOY  (absolute OOS Sharpe/Ret <= 0; positive Δ only means less bad)"
    elif delta > 0:
        verdict = "✅ DEPLOY  (ΔSharpe > 0)"
    elif delta >= -0.3:
        verdict = "⚠ DEPLOY CAREFULLY  (-0.3 ≤ ΔSharpe ≤ 0)"
    else:
        verdict = "❌ DO NOT DEPLOY  (ΔSharpe < -0.3)"
    print(f"\n  VERDICT: {verdict}")

    print(f"\n  Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
