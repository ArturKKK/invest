"""R131 — Prod-simulate (record-zero) parity check for R129 gated A1.

Goals:
  1. Reproduce r68-style simulate (record-zero on risk-off, with closing cost)
     baseline ~ 2.831 to confirm we're comparing apples to apples.
  2. Run gated A1 (L=720, q=0.20) on prod simulate.
  3. Report:
     - full-period Sharpe (base, A1-always, gated)
     - per-window Sharpe + DD
     - gate_on% overall and per window
     - P(gate_on | risk_off) and P(gate_on | risk_on)  ← critical: don't double-skip
     - block-bootstrap p>0
     - CVaR 5% / worst-week metrics

Parameters FROZEN BEFORE THIS RUN:
  L = 720
  q = 0.20
  A1 thr = 0.25, scale = 0.60

OOS forward-test is BLOCKED: preds cache ends 2026-03-07 (last W3 day).
Pending: regenerate preds on post-2026-03-07 data.
"""
from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from _preflight_check import check_versions
check_versions()

import _r128_all_overlays_canonical as r128
import _r129_persistence_gate as r129
import _r130_validate_r129 as r130


# Frozen params
L_FROZEN = 720
Q_FROZEN = 0.20
A1_FROZEN = {"trend_thr": 0.25, "weak_scale": 0.60}

PERIODS_PER_YEAR = 2 * 365
FUNDING_PER_12H = 0.00008


# ─────────────────────────────────────────────────────────────────────
# PROD-STYLE SIMULATE (mirrors _research_r68_continuous_wf.simulate exactly,
# extended with optional A1 + gate). NEVER mutates r68 source file.
# ─────────────────────────────────────────────────────────────────────

def simulate_prod(
    merged: pd.DataFrame,
    regime_df: pd.DataFrame,
    n_long: int,
    n_short: int,
    cfg: Optional[Dict] = None,
    *,
    a1_cfg: Optional[Dict] = None,
    gate_persist_col: Optional[str] = None,
    gate_threshold_series: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Mirror r68.simulate (record-zero on risk-off + closing cost) with optional gated A1."""
    cfg = cfg or r128.PROD_CFG
    trend_cutoff = cfg["trend_cutoff"]
    rebal_hours = cfg["rebal_hours"]
    ema_alpha = cfg.get("ema_alpha", None)
    hysteresis = cfg.get("hysteresis", 0)

    all_rets: List[Dict] = []
    prev_longs: Set[str] = set()
    prev_shorts: Set[str] = set()
    prev_preds: Dict[str, float] = {}

    timestamps_sorted = sorted(merged["timestamp"].unique())
    grouped = {ts: grp for ts, grp in merged.groupby("timestamp")}
    rebal_timestamps = timestamps_sorted[::rebal_hours]

    for ts in rebal_timestamps:
        if ts not in regime_df.index or ts not in grouped:
            continue
        row = regime_df.loc[ts]
        trend_str = row.get("trend_strength", 0)
        trend_dir = row.get("trend_direction", 0) if "trend_direction" in row else 0

        # ── Decide gate state for this ts (only applies when not risk-off) ──
        gate_on = False
        if a1_cfg is not None and gate_persist_col is not None and gate_threshold_series is not None:
            persist_val = regime_df.loc[ts].get(gate_persist_col, np.nan) if gate_persist_col in regime_df.columns else np.nan
            thr_val = gate_threshold_series.get(ts, np.nan) if hasattr(gate_threshold_series, "get") else np.nan
            if pd.notna(persist_val) and pd.notna(thr_val):
                gate_on = bool(persist_val < thr_val)
        elif a1_cfg is not None:
            gate_on = True  # always-on A1

        # ── RISK-OFF: record zero with closing cost (record-zero behavior) ──
        if trend_str > trend_cutoff:
            if prev_longs or prev_shorts:
                n_prev = len(prev_longs) + len(prev_shorts)
                avg_weight = 1.0 / n_prev if n_prev > 0 else 0
                close_cost = sum(r128._cost_for_sym(s) * avg_weight for s in prev_longs | prev_shorts)
                all_rets.append({
                    "timestamp": ts, "gross_ret": 0.0, "net_ret": -close_cost,
                    "cost": close_cost, "n_long": 0, "n_short": 0, "turnover": n_prev,
                    "risk_off": True, "gate_on": gate_on,
                })
            else:
                all_rets.append({
                    "timestamp": ts, "gross_ret": 0.0, "net_ret": 0.0,
                    "cost": 0.0, "n_long": 0, "n_short": 0, "turnover": 0,
                    "risk_off": True, "gate_on": gate_on,
                })
            prev_longs, prev_shorts = set(), set()
            continue

        grp = grouped[ts].copy()
        n = len(grp)
        nl = min(n_long, n // 3)
        ns = min(n_short, n // 3)
        if nl == 0 and ns == 0:
            continue

        exposure = 1.0
        dyn_threshold = cfg.get("dyn_threshold", 0.5)
        if dyn_threshold is not None and trend_str > dyn_threshold:
            exposure = max(0.1, 1.0 - (trend_str - dyn_threshold) /
                           (trend_cutoff - dyn_threshold + 1e-10) * 0.5)

        if ema_alpha is not None and ema_alpha < 1.0:
            for idx, r in grp.iterrows():
                sym = r["symbol"]
                raw_pred = r["pred"]
                smoothed = ema_alpha * raw_pred + (1 - ema_alpha) * prev_preds.get(sym, raw_pred)
                prev_preds[sym] = smoothed
                grp.at[idx, "pred"] = smoothed

        grp["pred_rank"] = grp["pred"].rank(ascending=False)

        if hysteresis > 0 and (prev_longs or prev_shorts):
            new_longs: Set[str] = set()
            new_shorts: Set[str] = set()
            for idx, r in grp.iterrows():
                sym, rank = r["symbol"], r["pred_rank"]
                if sym in prev_longs and rank <= nl + hysteresis:
                    new_longs.add(sym)
                elif sym in prev_shorts and rank > (n - ns - hysteresis):
                    new_shorts.add(sym)
            for _, r in grp[~grp["symbol"].isin(new_longs | new_shorts)].sort_values("pred_rank").head(nl - len(new_longs)).iterrows():
                new_longs.add(r["symbol"])
            for _, r in grp[~grp["symbol"].isin(new_longs | new_shorts)].sort_values("pred_rank", ascending=False).head(ns - len(new_shorts)).iterrows():
                new_shorts.add(r["symbol"])
        else:
            new_longs = set(grp[grp["pred_rank"] <= nl]["symbol"].tolist()) if nl > 0 else set()
            new_shorts = set(grp[grp["pred_rank"] > (n - ns)]["symbol"].tolist()) if ns > 0 else set()

        new_opened = (new_longs - prev_longs) | (new_shorts - prev_shorts)
        closed = (prev_longs - new_longs) | (prev_shorts - new_shorts)
        total_positions = len(new_longs) + len(new_shorts)

        longs = grp[grp["symbol"].isin(new_longs)]
        shorts = grp[grp["symbol"].isin(new_shorts)]
        long_ret = longs["fwd_ret"].mean() if len(longs) > 0 else 0
        short_ret = shorts["fwd_ret"].mean() if len(shorts) > 0 else 0

        nl_act, ns_act = len(new_longs), len(new_shorts)

        # ── A1 SIDE WEIGHTS (apply only if gate_on and both sides active) ──
        w_l_side, w_s_side = 0.5, 0.5
        if a1_cfg is not None and gate_on and nl_act > 0 and ns_act > 0:
            trend_thr = a1_cfg.get("trend_thr", 0.25)
            scale = a1_cfg.get("weak_scale", 0.60)
            if trend_dir > trend_thr:
                w_s_side *= scale
            elif trend_dir < -trend_thr:
                w_l_side *= scale
            tot = w_l_side + w_s_side
            w_l_side /= tot
            w_s_side /= tot

        if nl_act > 0 and ns_act > 0:
            gross_ret = w_l_side * long_ret - w_s_side * short_ret
        elif ns_act > 0:
            gross_ret = -short_ret
        else:
            gross_ret = long_ret
        gross_ret *= exposure

        if total_positions > 0:
            avg_weight = 1.0 / total_positions
            turnover_cost = sum(r128._cost_for_sym(sym) * avg_weight for sym in new_opened)
            turnover_cost += sum(r128._cost_for_sym(sym) * avg_weight for sym in closed)
            holding_cost = FUNDING_PER_12H * (rebal_hours / 12)
            total_cost = turnover_cost + holding_cost
        else:
            total_cost = 0.0

        net_ret = gross_ret - total_cost
        prev_longs, prev_shorts = new_longs, new_shorts

        all_rets.append({
            "timestamp": ts, "gross_ret": gross_ret, "net_ret": net_ret,
            "cost": total_cost, "n_long": nl_act, "n_short": ns_act,
            "turnover": len(new_opened) + len(closed),
            "risk_off": False, "gate_on": gate_on,
        })

    return pd.DataFrame(all_rets) if all_rets else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────

def cvar_5pct(rets: np.ndarray) -> float:
    if len(rets) < 5:
        return 0.0
    thr = np.percentile(rets, 5)
    tail = rets[rets <= thr]
    return float(tail.mean()) if len(tail) > 0 else 0.0


def worst_week_ret(port: pd.DataFrame, week_periods: int = 14) -> float:
    """Worst rolling 14-period (=7d at 12h cadence) cumulative return."""
    r = port["net_ret"].values
    if len(r) < week_periods:
        return 0.0
    rolling_sum = pd.Series(r).rolling(week_periods).sum().dropna()
    return float(rolling_sum.min())


def report_block(label: str, port: pd.DataFrame, base_port: Optional[pd.DataFrame] = None):
    if port.empty:
        print(f"  {label}: EMPTY")
        return None
    r = port["net_ret"].values
    s = r130.sharpe(r)
    so = r130.sortino(r)
    dd = r130.max_drawdown(r)
    cvar = cvar_5pct(r)
    ww = worst_week_ret(port)
    n = len(port)
    n_active = int((~port["risk_off"]).sum())
    n_gate = int(port["gate_on"].sum())

    delta_str = ""
    if base_port is not None and not base_port.empty:
        delta_str = f"  ΔS={s - r130.sharpe(base_port['net_ret'].values):+.3f}"
    print(f"  {label:<22s} S={s:+.3f}{delta_str}  Sortino={so:+.3f}  maxDD={dd*100:+.2f}%  "
          f"CVaR5%={cvar*1e4:+.1f}bp  worst_7d={ww*1e4:+.1f}bp  n={n} active={n_active} gate={n_gate}")
    return {"sharpe": s, "sortino": so, "maxDD": dd, "cvar5": cvar, "worst_7d": ww,
            "n": n, "n_active": n_active, "n_gate": n_gate}


def per_window_report(label: str, port: pd.DataFrame):
    print(f"\n  {label} per-window:")
    for i, win in enumerate(r128.r68.CONTINUOUS_WINDOWS, 1):
        ts_s = pd.Timestamp(win["test_start"], tz="UTC")
        ts_e = pd.Timestamp(win["test_end"], tz="UTC")
        sub = port[(port["timestamp"] >= ts_s) & (port["timestamp"] < ts_e)]
        if len(sub) < 2:
            print(f"    W{i}: empty")
            continue
        r = sub["net_ret"].values
        s = r130.sharpe(r)
        dd = r130.max_drawdown(r)
        n_total = len(sub)
        n_active = int((~sub["risk_off"]).sum())
        n_gate = int(sub["gate_on"].sum())
        gate_pct_total = n_gate / n_total * 100
        gate_pct_active = (n_gate / n_active * 100) if n_active > 0 else 0.0
        print(f"    W{i}: S={s:+.3f}  DD={dd*100:+.2f}%  "
              f"n={n_total} (active={n_active}, risk_off={n_total-n_active})  "
              f"gate_on: {n_gate} ({gate_pct_total:.1f}% of total, {gate_pct_active:.1f}% of active)")


def gate_vs_riskoff_intersection(port: pd.DataFrame, label: str = ""):
    n = len(port)
    n_off = int(port["risk_off"].sum())
    n_on = n - n_off
    n_gate = int(port["gate_on"].sum())
    n_gate_when_off = int((port["gate_on"] & port["risk_off"]).sum())
    n_gate_when_on = int((port["gate_on"] & ~port["risk_off"]).sum())

    print(f"\n  GATE × RISK_OFF intersection ({label}):")
    print(f"    Total periods: {n}")
    print(f"    risk_off frac: {n_off/n*100:.1f}%  ({n_off}/{n})")
    print(f"    gate_on overall: {n_gate/n*100:.1f}%  ({n_gate}/{n})")
    if n_off > 0:
        print(f"    P(gate_on | risk_off) = {n_gate_when_off/n_off*100:.1f}%  "
              f"← gating during risk-off is wasted (no positions)")
    if n_on > 0:
        print(f"    P(gate_on | risk_on)  = {n_gate_when_on/n_on*100:.1f}%  "
              f"← effective gating fraction (matters for KPI)")
    if n_gate > 0:
        print(f"    P(risk_off | gate_on) = {n_gate_when_off/n_gate*100:.1f}%  "
              f"← share of gate that is wasted")


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--block_len", type=int, default=14)
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 100)
    print(f"  R131 — Prod-simulate (record-zero) parity for R129 gated A1")
    print(f"  Frozen params: L={L_FROZEN}, q={Q_FROZEN}, A1={A1_FROZEN}")
    print("=" * 100)

    preds, regime_df = r128.build_or_load_cache()
    print(f"  Preds range: {preds['timestamp'].min()} → {preds['timestamp'].max()}")
    print(f"  Note: OOS forward-test BLOCKED (preds cache ends before W3 end + post-W3 OOS).")

    # Add persistence column to regime_df
    regime_aug = r129.add_persistence(regime_df, lookback=L_FROZEN)
    persist_col = f"td_persist_{L_FROZEN}h"
    persist_ts = regime_aug[persist_col]
    thr_series = r129.expanding_quantile_threshold(persist_ts, Q_FROZEN, min_periods=720)

    # ─── 1) Baseline parity ───
    print("\n" + "=" * 100)
    print("  1) BASELINE PARITY CHECK (should be ~2.831 to match verified prod baseline)")
    print("=" * 100)
    base_port = simulate_prod(preds, regime_aug, n_long=4, n_short=2)
    base_metrics = report_block("baseline (prod)", base_port)

    # ─── 2) A1 always-on (prod) ───
    print("\n  2) A1 always-on (prod simulate)")
    a1_port = simulate_prod(preds, regime_aug, n_long=4, n_short=2,
                              a1_cfg=A1_FROZEN)
    report_block("A1 always (prod)", a1_port, base_port)

    # ─── 3) Gated A1 (prod) ───
    print("\n  3) Gated A1 (prod simulate, frozen params)")
    gated_port = simulate_prod(preds, regime_aug, n_long=4, n_short=2,
                                 a1_cfg=A1_FROZEN,
                                 gate_persist_col=persist_col,
                                 gate_threshold_series=thr_series)
    # Strip pre-warmup periods (where threshold is NaN -> gate_on always False; no effect)
    # but we report on full set since thresh NaN just means gate=off
    gated_metrics = report_block("Gated A1 (prod)", gated_port, base_port)

    # ─── Per-window breakdown ───
    per_window_report("baseline", base_port)
    per_window_report("A1 always", a1_port)
    per_window_report("Gated A1 (FROZEN)", gated_port)

    # ─── Gate vs risk_off intersection ───
    gate_vs_riskoff_intersection(gated_port, "gated A1 prod")

    # ─── Block bootstrap (gated vs baseline, aligned timestamps) ───
    print("\n" + "=" * 100)
    print(f"  BLOCK BOOTSTRAP (block={args.block_len}p={args.block_len*12}h, n_boot={args.n_boot})")
    print("=" * 100)
    m = gated_port[["timestamp", "net_ret"]].rename(columns={"net_ret": "alt"}).merge(
        base_port[["timestamp", "net_ret"]].rename(columns={"net_ret": "base"}),
        on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    bs_iid = r129.boot_p_improvement(m["base"].values, m["alt"].values, n_boot=args.n_boot)
    bs_blk = r130.block_bootstrap_diff(m["base"].values, m["alt"].values,
                                         block_len=args.block_len, n_boot=args.n_boot)
    print(f"  iid    bootstrap: mean Δ={bs_iid['mean']:+.3f}  P(Δ>0)={bs_iid['p_pos']:.3f}  "
          f"CI95=[{bs_iid['ci_low']:+.3f},{bs_iid['ci_high']:+.3f}]")
    print(f"  block  bootstrap: mean Δ={bs_blk['mean']:+.3f}  P(Δ>0)={bs_blk['p_pos']:.3f}  "
          f"CI95=[{bs_blk['ci_low']:+.3f},{bs_blk['ci_high']:+.3f}]")

    # ─── Verdict ───
    print("\n" + "=" * 100)
    print("  VERDICT")
    print("=" * 100)
    base_S = base_metrics["sharpe"]
    gated_S = gated_metrics["sharpe"]
    delta = gated_S - base_S

    parity_ok = abs(base_S - 2.831) < 0.20
    print(f"  Baseline parity: prod-sim base S={base_S:+.3f}, verified=2.831 → "
          f"{'✅ PARITY OK' if parity_ok else '⚠ MISMATCH (>0.20 from 2.831)'}")
    print(f"  Δ Sharpe (gated - base): {delta:+.3f}")
    print(f"  block bootstrap P(Δ>0): {bs_blk['p_pos']:.3f}  ({'✅ ≥0.85' if bs_blk['p_pos'] >= 0.85 else '⚠ <0.85'})")
    print(f"  maxDD vs baseline: gated={gated_metrics['maxDD']*100:+.2f}%  vs  "
          f"base={base_metrics['maxDD']*100:+.2f}%  "
          f"({'✅ not worse' if gated_metrics['maxDD'] >= base_metrics['maxDD']*1.05 else '⚠ worse'})")
    print(f"\n  ⏳ OOS forward-test: BLOCKED — preds cache ends 2026-03-07.")
    print(f"     Need to regenerate preds on post-2026-03-07 data before final go/no-go.")

    print(f"\n  Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
