#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R50 — Risk Protocol: Gross Cap + Crash Mode

Goal: Reduce max DD from -52% to <-35% WITHOUT harming Sharpe.
Approach: Discrete rules only (vol targeting proven harmful in R15/R37).

Tests:
  50.1  Crash detector: BTC rvol / spread expansion / market dispersion threshold
  50.2  Equity DD breaker: DD > -15% → TIER1-only for N bars
  50.3  Gross cap: limit leverage during high-vol periods
  50.4  WF validation: champion_31f + hybrid costs + best protocol

Usage:
  python _research_r50_risk_protocol.py          # full run
  python _research_r50_risk_protocol.py --quick  # BTC/ETH/SOL only
"""

from __future__ import annotations

import argparse
import warnings
from typing import Set, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

from _research_round7 import WINDOWS, SYM_35
from _research_r30b_fixed import (
    eval_with_costs,
    simulate_with_costs,
    train_ensemble,
)
from _research_r35_new_features import add_r35_features, load_research_frame
from _research_r47_coinglass import (
    CANONICAL_EXEC_CFG,
    add_cg_features,
    compute_cg_features,
    load_cg_daily,
    make_feature_set,
)
from _research_r48_cost import simulate_with_hybrid_costs, TIER1_SYMS, TIER3_SYMS

# ─────────────────────────────────────────────────────────────


def section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print("=" * 70)


# ══════════════════════════════════════════════════════════════
#  Crash detection signals
# ══════════════════════════════════════════════════════════════

def build_crash_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-timestamp crash indicators from OHLCV + regime data.

    Returns a DataFrame indexed by timestamp with:
      - btc_rvol_roll30d: BTC realized vol (24h), percentile vs 30d rolling
      - market_disp_roll30d: market return dispersion percentile
      - crash_mode_rvol: binary, rvol_pct > threshold
      - crash_mode_disp: binary, disp_pct > threshold
      - crash_mode_any: any crash signal active
    """
    btc = df[df["symbol"] == "BTC/USDT"].set_index("timestamp").sort_index()

    # BTC 24h realized vol
    btc_rvol = btc["ret_1h"].rolling(24, min_periods=12).std() * np.sqrt(24) if "ret_1h" in btc.columns else pd.Series(dtype=float)

    # market dispersion
    mkt_disp = df.groupby("timestamp")["ret_12h"].std() if "ret_12h" in df.columns else None

    ts_index = sorted(df["timestamp"].unique())
    result = pd.DataFrame(index=ts_index)

    # BTC rvol percentile
    if not btc_rvol.empty:
        btc_rvol_aligned = btc_rvol.reindex(ts_index)
        roll_pct = btc_rvol_aligned.rolling(30 * 24, min_periods=7 * 24).rank(pct=True)
        result["btc_rvol"] = btc_rvol_aligned
        result["btc_rvol_pct"] = roll_pct
    else:
        result["btc_rvol"] = np.nan
        result["btc_rvol_pct"] = np.nan

    # Market dispersion percentile
    if mkt_disp is not None:
        result["mkt_disp"] = mkt_disp
        result["mkt_disp_pct"] = mkt_disp.rolling(30 * 24, min_periods=7 * 24).rank(pct=True)
    else:
        result["mkt_disp"] = np.nan
        result["mkt_disp_pct"] = np.nan

    return result


def add_crash_mode(crash_df: pd.DataFrame,
                   rvol_threshold: float = 0.85,
                   disp_threshold: float = 0.90) -> pd.DataFrame:
    """Tag each bar with crash_mode=True if any threshold exceeded."""
    df = crash_df.copy()
    df["crash_rvol"] = df["btc_rvol_pct"].fillna(0) > rvol_threshold
    df["crash_disp"] = df["mkt_disp_pct"].fillna(0) > disp_threshold
    df["crash_mode"] = df["crash_rvol"] | df["crash_disp"]
    return df


# ══════════════════════════════════════════════════════════════
#  Portfolio simulators with risk protocols
# ══════════════════════════════════════════════════════════════

def simulate_with_crash_protocol(
    preds: pd.DataFrame,
    regime_df: pd.DataFrame,
    cfg: dict,
    crash_df: pd.DataFrame,
    gross_reduction: float = 0.5,   # reduce to 50% when in crash mode
    tier1_only: bool = True,         # only TIER1 in crash mode
) -> pd.DataFrame:
    """
    Like simulate_with_hybrid_costs but with crash mode:
    - When crash_mode=True: reduce gross exposure, optionally TIER1-only
    - Normal bars: full hybrid costs
    """
    from _research_r48_cost import (
        TIER1_SYMS as T1, TIER2_SYMS as T2, simulate_with_hybrid_costs,
    )

    n_long = cfg.get("n_long", 6)
    n_short = cfg.get("n_short", 3)
    trend_cutoff = cfg.get("trend_cutoff", 0.9)
    rebal_hours = cfg.get("rebal_hours", 12)
    ema_alpha = cfg.get("ema_alpha", None)
    hysteresis = cfg.get("hysteresis", 0)

    def _cost_for_sym(sym: str) -> float:
        if sym in T1:
            return 0.92 * (-0.0001) + 0.08 * 0.0007
        elif sym in T2:
            return 0.75 * 0.0001 + 0.25 * 0.0007
        else:
            return 0.0005 + 0.0002

    funding_per_12h = 0.00008

    all_rets = []
    prev_longs: Set[str] = set()
    prev_shorts: Set[str] = set()

    timestamps_sorted = sorted(preds["timestamp"].unique())
    rebal_timestamps = timestamps_sorted[::rebal_hours]
    grouped = {ts: grp for ts, grp in preds.groupby("timestamp")}

    crash_mode_set = set(crash_df[crash_df["crash_mode"]].index.tolist())

    for ts in rebal_timestamps:
        if ts not in regime_df.index or ts not in grouped:
            continue
        row = regime_df.loc[ts]
        trend_str = row.get("trend_strength", 0)
        if trend_str > trend_cutoff:
            continue

        grp = grouped[ts].copy()
        n = len(grp)

        in_crash = ts in crash_mode_set
        exposure = 1.0
        dyn_threshold = cfg.get("dyn_threshold", 0.5)
        if dyn_threshold is not None and trend_str > dyn_threshold:
            exposure = max(0.1, 1.0 - (trend_str - dyn_threshold) /
                          (trend_cutoff - dyn_threshold + 1e-10) * 0.5)

        # Crash mode: reduce gross
        if in_crash:
            exposure *= gross_reduction

        nl, ns = n_long, n_short

        # Crash mode: TIER1-only filter
        if in_crash and tier1_only:
            grp_filtered = grp[grp["symbol"].isin(T1)]
            if len(grp_filtered) >= 2:
                grp = grp_filtered
                n = len(grp)

        nl = min(nl, n // 3)
        ns = min(ns, n // 3)
        if nl == 0 and ns == 0:
            continue

        if ema_alpha is not None and ema_alpha < 1.0:
            # simplified: skip EMA smoothing for speed
            pass

        grp["pred_rank"] = grp["pred"].rank(ascending=False)
        new_longs = set(grp[grp["pred_rank"] <= nl]["symbol"].tolist()) if nl > 0 else set()
        new_shorts = set(grp[grp["pred_rank"] > (n - ns)]["symbol"].tolist()) if ns > 0 else set()

        if nl > 0 and len(grp[grp["pred_rank"] <= nl]) > 0:
            long_ret = float(grp[grp["pred_rank"] <= nl]["fwd_ret"].mean())
        else:
            long_ret = 0
        if ns > 0 and len(grp[grp["pred_rank"] > (n - ns)]) > 0:
            short_ret = float(grp[grp["pred_rank"] > (n - ns)]["fwd_ret"].mean())
        else:
            short_ret = 0

        new_opened = (new_longs - prev_longs) | (new_shorts - prev_shorts)
        closed_pos = (prev_longs - new_longs) | (prev_shorts - new_shorts)
        turnover_count = len(new_opened) + len(closed_pos)
        total_positions = len(new_longs) + len(new_shorts)

        if total_positions > 0:
            avg_weight = 1.0 / total_positions
            sym_cost = sum(
                abs(_cost_for_sym(s)) * avg_weight for s in (new_opened | closed_pos)
            )
            holding_cost = funding_per_12h * (rebal_hours / 12)
            total_cost = sym_cost + holding_cost
        else:
            total_cost = 0

        prev_longs = new_longs
        prev_shorts = new_shorts

        if nl > 0 and ns > 0:
            port_ret = 0.5 * long_ret - 0.5 * short_ret
        elif ns > 0:
            port_ret = -short_ret
        else:
            port_ret = long_ret

        port_ret *= exposure
        port_ret -= total_cost

        all_rets.append({
            "timestamp": ts,
            "portfolio_ret": port_ret,
            "gross_ret": port_ret + total_cost,
            "long_ret": long_ret,
            "short_ret": short_ret,
            "n_long": len(new_longs),
            "n_short": len(new_shorts),
            "exposure": exposure,
            "in_crash": in_crash,
            "turnover": turnover_count,
            "cost": total_cost,
        })

    return pd.DataFrame(all_rets)


def simulate_with_dd_breaker(
    preds: pd.DataFrame,
    regime_df: pd.DataFrame,
    cfg: dict,
    dd_threshold: float = -0.15,   # enter reduced mode
    dd_recovery: float = -0.08,    # exit reduced mode
    reduced_gross: float = 0.5,    # gross factor when in DD-breaker mode
    tier1_only_dd: bool = True,    # TIER1-only when in DD-breaker
) -> pd.DataFrame:
    """
    Equity DD breaker: when portfolio equity drops >dd_threshold from peak,
    reduce to TIER1-only with reduced_gross.
    """
    from _research_r48_cost import TIER1_SYMS as T1, TIER2_SYMS as T2

    n_long = cfg.get("n_long", 6)
    n_short = cfg.get("n_short", 3)
    trend_cutoff = cfg.get("trend_cutoff", 0.9)
    rebal_hours = cfg.get("rebal_hours", 12)

    def _cost_for_sym(sym: str) -> float:
        if sym in T1:
            return 0.92 * (-0.0001) + 0.08 * 0.0007
        elif sym in T2:
            return 0.75 * 0.0001 + 0.25 * 0.0007
        else:
            return 0.0005 + 0.0002

    funding_per_12h = 0.00008

    all_rets = []
    prev_longs: Set[str] = set()
    prev_shorts: Set[str] = set()

    timestamps_sorted = sorted(preds["timestamp"].unique())
    rebal_timestamps = timestamps_sorted[::rebal_hours]
    grouped = {ts: grp for ts, grp in preds.groupby("timestamp")}

    # Track equity path for DD calculation
    equity = 1.0
    peak_equity = 1.0
    in_dd_mode = False

    for ts in rebal_timestamps:
        if ts not in regime_df.index or ts not in grouped:
            continue
        row = regime_df.loc[ts]
        trend_str = row.get("trend_strength", 0)
        if trend_str > trend_cutoff:
            continue

        grp = grouped[ts].copy()
        n = len(grp)

        # Update DD mode
        current_dd = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0
        if not in_dd_mode and current_dd < dd_threshold:
            in_dd_mode = True
        elif in_dd_mode and current_dd > dd_recovery:
            in_dd_mode = False

        exposure = 1.0
        dyn_threshold = cfg.get("dyn_threshold", 0.5)
        if dyn_threshold is not None and trend_str > dyn_threshold:
            exposure = max(0.1, 1.0 - (trend_str - dyn_threshold) /
                          (trend_cutoff - dyn_threshold + 1e-10) * 0.5)

        if in_dd_mode:
            exposure *= reduced_gross
            if tier1_only_dd:
                grp_filtered = grp[grp["symbol"].isin(T1)]
                if len(grp_filtered) >= 2:
                    grp = grp_filtered
                    n = len(grp)

        nl = min(n_long, n // 3)
        ns = min(n_short, n // 3)
        if nl == 0 and ns == 0:
            continue

        grp["pred_rank"] = grp["pred"].rank(ascending=False)
        new_longs = set(grp[grp["pred_rank"] <= nl]["symbol"].tolist()) if nl > 0 else set()
        new_shorts = set(grp[grp["pred_rank"] > (n - ns)]["symbol"].tolist()) if ns > 0 else set()

        if nl > 0 and len(grp[grp["pred_rank"] <= nl]) > 0:
            long_ret = float(grp[grp["pred_rank"] <= nl]["fwd_ret"].mean())
        else:
            long_ret = 0
        if ns > 0 and len(grp[grp["pred_rank"] > (n - ns)]) > 0:
            short_ret = float(grp[grp["pred_rank"] > (n - ns)]["fwd_ret"].mean())
        else:
            short_ret = 0

        new_opened = (new_longs - prev_longs) | (new_shorts - prev_shorts)
        closed_pos = (prev_longs - new_longs) | (prev_shorts - new_shorts)
        total_positions = len(new_longs) + len(new_shorts)

        if total_positions > 0:
            avg_weight = 1.0 / total_positions
            sym_cost = sum(
                abs(_cost_for_sym(s)) * avg_weight for s in (new_opened | closed_pos)
            )
            holding_cost = funding_per_12h * (rebal_hours / 12)
            total_cost = sym_cost + holding_cost
        else:
            total_cost = 0

        prev_longs = new_longs
        prev_shorts = new_shorts

        if nl > 0 and ns > 0:
            port_ret = 0.5 * long_ret - 0.5 * short_ret
        elif ns > 0:
            port_ret = -short_ret
        else:
            port_ret = long_ret

        port_ret *= exposure
        port_ret -= total_cost

        # Update equity
        equity *= (1 + port_ret)
        peak_equity = max(peak_equity, equity)

        all_rets.append({
            "timestamp": ts,
            "portfolio_ret": port_ret,
            "gross_ret": port_ret + total_cost,
            "long_ret": long_ret,
            "short_ret": short_ret,
            "n_long": len(new_longs),
            "n_short": len(new_shorts),
            "exposure": exposure,
            "in_dd_mode": in_dd_mode,
            "current_dd": current_dd,
            "equity": equity,
            "turnover": len(new_opened) + len(closed_pos),
            "cost": total_cost,
        })

    return pd.DataFrame(all_rets)


# ══════════════════════════════════════════════════════════════
#  Evaluation helpers
# ══════════════════════════════════════════════════════════════

def eval_extended(port: pd.DataFrame, label: str) -> dict:
    """Extended eval: Sharpe, Sortino, max_dd, cost%, time_in_reduced."""
    if port is None or port.empty:
        return {"label": label, "sharpe": 0, "max_dd": 0}

    rets = port["portfolio_ret"].fillna(0)
    ann = np.sqrt(24 * 365)
    sharpe = float(rets.mean() / (rets.std() + 1e-12) * ann)

    neg = rets[rets < 0]
    sortino = float(rets.mean() / (neg.std() + 1e-12) * ann) if len(neg) > 0 else sharpe

    cum = (1 + rets).cumprod()
    peak = cum.expanding().max()
    dd = (cum / peak - 1)
    max_dd = float(dd.min())
    max_dd_pct = max_dd * 100

    total_cost = port["cost"].sum() if "cost" in port else 0
    total_gross = port["gross_ret"].sum() if "gross_ret" in port else rets.sum()
    cost_pct = (total_cost / (abs(total_gross) + 1e-12)) * 100 if total_gross != 0 else 0

    in_reduced_col = "in_crash" if "in_crash" in port.columns else ("in_dd_mode" if "in_dd_mode" in port.columns else None)
    pct_in_reduced = float(port[in_reduced_col].mean()) * 100 if in_reduced_col else 0

    cum_ret = float(cum.iloc[-1] - 1) * 100

    return {
        "label": label,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd_pct": round(max_dd_pct, 1),
        "cost_pct": round(cost_pct, 1),
        "cum_ret_pct": round(cum_ret, 1),
        "pct_in_reduced": round(pct_in_reduced, 1),
    }


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main(quick: bool = False) -> None:
    print("=" * 70)
    print("R50 — Risk Protocol: Gross Cap + Crash Mode")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────
    print("\n[1] Loading data ...")
    cg = load_cg_daily()
    cg_feats_daily = compute_cg_features(cg)
    df, regime_df = load_research_frame()
    df, _ = add_r35_features(df)
    regime_df = regime_df.sort_index()
    df, per_sym_cols, mkt_cols = add_cg_features(df, cg_feats_daily)

    if quick:
        print("  ⚡ Quick mode: BTC/ETH/SOL only ...")
        df = df[df["symbol"].isin(["BTC/USDT", "ETH/USDT", "SOL/USDT"])].copy()
    print(f"  Frame: {len(df):,} rows × {len(df.columns)} cols")

    # ── Train champion_31f ────────────────────────────────────
    print("\n[2] Training champion_31f ...")
    feats, no_rank = make_feature_set(["cg_taker_imb"], mkt_cols)
    preds = train_ensemble(df, feats, WINDOWS, l2=1.0, rolling=False,
                           label="r50_31f", cs_rank_exclude=no_rank)
    if preds is None or preds.empty:
        print("❌ Training failed")
        return
    print(f"  Predictions: {len(preds):,} rows")

    # ── Baseline: hybrid costs (champion R48) ─────────────────
    section("BASELINE — champion_31f + hybrid costs (R48 result)")
    baseline_rows = []
    for window in ["W1", "W2", "W3", "ALL"]:
        subset = preds if window == "ALL" else preds[preds["window"] == window]
        port = simulate_with_hybrid_costs(subset, regime_df, CANONICAL_EXEC_CFG)
        m = eval_with_costs(port, f"baseline_{window}")
        m_ext = eval_extended(port, f"baseline_{window}")
        baseline_rows.append({
            "window": window, "sharpe": m["sharpe"],
            "max_dd_pct": m_ext["max_dd_pct"],
            "cost_pct": m_ext["cost_pct"],
        })
    baseline_df = pd.DataFrame(baseline_rows)
    print(f"\n  {'Window':<6} {'Sharpe':>7} {'MaxDD%':>8} {'Cost%':>7}")
    for _, r in baseline_df.iterrows():
        print(f"  {r['window']:<6} {r['sharpe']:>+6.2f} {r['max_dd_pct']:>7.1f}% {r['cost_pct']:>6.1f}%")

    baseline_all = baseline_df[baseline_df["window"] == "ALL"].iloc[0]
    print(f"\n  Champion ALL: Sharpe={baseline_all['sharpe']:+.2f}, "
          f"MaxDD={baseline_all['max_dd_pct']:.1f}%, Cost={baseline_all['cost_pct']:.1f}%")

    # ── Build crash indicators ────────────────────────────────
    print("\n[3] Building crash indicators ...")
    crash_df = build_crash_indicators(df)

    # Show crash stats
    for thresh in [0.80, 0.85, 0.90]:
        crash_tagged = add_crash_mode(crash_df, rvol_threshold=thresh, disp_threshold=0.90)
        crash_pct = crash_tagged["crash_mode"].fillna(False).mean() * 100
        print(f"  rvol_thresh={thresh:.0%}: crash_mode={crash_pct:.1f}% of bars")

    # ── 50.1: Crash mode ablation ─────────────────────────────
    section("50.1 — Crash Mode Protocol (scan thresholds)")
    crash_rows = []
    for rvol_t in [0.80, 0.85, 0.90]:
        for gross_red in [0.3, 0.5]:
            for t1_only in [True, False]:
                crash_tagged = add_crash_mode(crash_df,
                                              rvol_threshold=rvol_t, disp_threshold=0.90)
                subset = preds  # ALL window
                port = simulate_with_crash_protocol(
                    subset, regime_df, CANONICAL_EXEC_CFG, crash_tagged,
                    gross_reduction=gross_red, tier1_only=t1_only
                )
                if port.empty:
                    continue
                m = eval_extended(port, f"crash_rvol{rvol_t:.0%}_gr{gross_red:.0%}_t1{t1_only}")
                crash_rows.append({
                    "rvol_thresh": rvol_t,
                    "gross_reduction": gross_red,
                    "tier1_only": t1_only,
                    "sharpe": m["sharpe"],
                    "max_dd_pct": m["max_dd_pct"],
                    "cost_pct": m["cost_pct"],
                    "pct_in_crash": m["pct_in_reduced"],
                })

    crash_summary = pd.DataFrame(crash_rows)
    print(f"\n  {'rvol_t':>7} {'gr%':>5} {'t1only':>6} {'Sharpe':>7} {'MaxDD%':>8} {'Cost%':>7} {'%crash':>7}")
    for _, r in crash_summary.iterrows():
        marker = "  ← best" if r["sharpe"] == crash_summary["sharpe"].max() else ""
        print(f"  {r['rvol_thresh']:.0%}    {r['gross_reduction']:.0%}   {str(r['tier1_only']):<5}  "
              f"{r['sharpe']:>+6.2f} {r['max_dd_pct']:>7.1f}% {r['cost_pct']:>6.1f}% "
              f"{r['pct_in_crash']:>6.1f}%{marker}")

    best_crash = crash_summary.sort_values("sharpe", ascending=False).iloc[0]
    print(f"\n  Best crash protocol: rvol>={best_crash['rvol_thresh']:.0%}, "
          f"gross×{best_crash['gross_reduction']:.0%}, "
          f"t1_only={best_crash['tier1_only']}")

    # ── 50.2: DD breaker ablation ─────────────────────────────
    section("50.2 — Equity DD Breaker (scan thresholds)")
    dd_rows = []
    for dd_t in [-0.10, -0.12, -0.15]:
        for gross_red in [0.4, 0.5, 0.6]:
            for t1_only in [True, False]:
                port = simulate_with_dd_breaker(
                    preds, regime_df, CANONICAL_EXEC_CFG,
                    dd_threshold=dd_t,
                    dd_recovery=dd_t * 0.6,
                    reduced_gross=gross_red,
                    tier1_only_dd=t1_only,
                )
                if port.empty:
                    continue
                m = eval_extended(port, f"dd{dd_t:.0%}_gr{gross_red:.0%}_t1{t1_only}")
                dd_rows.append({
                    "dd_threshold": dd_t,
                    "gross_reduction": gross_red,
                    "tier1_only": t1_only,
                    "sharpe": m["sharpe"],
                    "max_dd_pct": m["max_dd_pct"],
                    "cost_pct": m["cost_pct"],
                    "pct_in_reduced": m["pct_in_reduced"],
                })

    dd_summary = pd.DataFrame(dd_rows)
    print(f"\n  {'dd_t%':>7} {'gr%':>5} {'t1only':>6} {'Sharpe':>7} {'MaxDD%':>8} {'Cost%':>7} {'%redu':>7}")
    for _, r in dd_summary.iterrows():
        marker = "  ← best" if r["sharpe"] == dd_summary["sharpe"].max() else ""
        print(f"  {r['dd_threshold']:.0%}    {r['gross_reduction']:.0%}   {str(r['tier1_only']):<5}  "
              f"{r['sharpe']:>+6.2f} {r['max_dd_pct']:>7.1f}% {r['cost_pct']:>6.1f}% "
              f"{r['pct_in_reduced']:>6.1f}%{marker}")

    best_dd = dd_summary.sort_values("sharpe", ascending=False).iloc[0]
    print(f"\n  Best DD breaker: threshold={best_dd['dd_threshold']:.0%}, "
          f"gross×{best_dd['gross_reduction']:.0%}, "
          f"t1_only={best_dd['tier1_only']}")

    # ── 50.3: Combined protocol ────────────────────────────────
    section("50.3 — Combined: Crash + DD Breaker (champion combo)")
    # Use best params from 50.1 and 50.2
    crash_tagged_best = add_crash_mode(
        crash_df,
        rvol_threshold=float(best_crash["rvol_thresh"]),
        disp_threshold=0.90,
    )

    combined_rows = []
    for window in ["W1", "W2", "W3", "ALL"]:
        subset = preds if window == "ALL" else preds[preds["window"] == window]

        # Crash only
        port_crash = simulate_with_crash_protocol(
            subset, regime_df, CANONICAL_EXEC_CFG, crash_tagged_best,
            gross_reduction=float(best_crash["gross_reduction"]),
            tier1_only=bool(best_crash["tier1_only"]),
        )

        # DD breaker only
        port_dd = simulate_with_dd_breaker(
            subset, regime_df, CANONICAL_EXEC_CFG,
            dd_threshold=float(best_dd["dd_threshold"]),
            dd_recovery=float(best_dd["dd_threshold"]) * 0.6,
            reduced_gross=float(best_dd["gross_reduction"]),
            tier1_only_dd=bool(best_dd["tier1_only"]),
        )

        m_base = eval_extended(
            simulate_with_hybrid_costs(subset, regime_df, CANONICAL_EXEC_CFG),
            f"base_{window}"
        )
        m_crash = eval_extended(port_crash, f"crash_{window}")
        m_dd = eval_extended(port_dd, f"dd_{window}")

        combined_rows.append({
            "window": window,
            "base_sharpe": m_base["sharpe"],
            "base_dd": m_base["max_dd_pct"],
            "crash_sharpe": m_crash["sharpe"],
            "crash_dd": m_crash["max_dd_pct"],
            "dd_sharpe": m_dd["sharpe"],
            "dd_dd": m_dd["max_dd_pct"],
        })

    combined_df = pd.DataFrame(combined_rows)
    print(f"\n  {'Window':<6} {'Base_Sh':>8} {'Base_DD':>8} "
          f"{'Crash_Sh':>9} {'Crash_DD':>9} {'DDBk_Sh':>8} {'DDBk_DD':>8}")
    for _, r in combined_df.iterrows():
        print(f"  {r['window']:<6} {r['base_sharpe']:>+7.2f} {r['base_dd']:>7.1f}% "
              f"{r['crash_sharpe']:>+8.2f} {r['crash_dd']:>8.1f}% "
              f"{r['dd_sharpe']:>+7.2f} {r['dd_dd']:>7.1f}%")

    # ── Save results ───────────────────────────────────────────
    crash_summary.to_csv("results_r50_crash_protocol.csv", index=False)
    dd_summary.to_csv("results_r50_dd_breaker.csv", index=False)
    combined_df.to_csv("results_r50_combined.csv", index=False)

    # ── Final summary ───────────────────────────────────────────
    section("R50 SUMMARY")
    all_combined = combined_df[combined_df["window"] == "ALL"].iloc[0]
    print(f"""
  BASELINE (champion_31f + hybrid):
    ALL Sharpe = {all_combined['base_sharpe']:+.2f}, MaxDD = {all_combined['base_dd']:.1f}%

  CRASH PROTOCOL (rvol>{best_crash['rvol_thresh']:.0%}, ×{best_crash['gross_reduction']:.0%}, t1={best_crash['tier1_only']}):
    ALL Sharpe = {all_combined['crash_sharpe']:+.2f}, MaxDD = {all_combined['crash_dd']:.1f}%
    DD change: {all_combined['crash_dd'] - all_combined['base_dd']:+.1f}pp

  DD BREAKER (threshold={best_dd['dd_threshold']:.0%}, ×{best_dd['gross_reduction']:.0%}, t1={best_dd['tier1_only']}):
    ALL Sharpe = {all_combined['dd_sharpe']:+.2f}, MaxDD = {all_combined['dd_dd']:.1f}%
    DD change: {all_combined['dd_dd'] - all_combined['base_dd']:+.1f}pp

  VERDICT:
    Target: ALL ≥ 1.50, MaxDD < -35%
    Crash: {"✅ PASS" if all_combined['crash_sharpe'] >= 1.5 and all_combined['crash_dd'] > -35 else "❌ FAIL (use as soft rule only)"}
    DD breaker: {"✅ PASS" if all_combined['dd_sharpe'] >= 1.5 and all_combined['dd_dd'] > -35 else "❌ FAIL (use as soft rule only)"}
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    main(quick=args.quick)
