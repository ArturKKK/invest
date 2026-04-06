#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R48 Phase 3 — Hybrid Cost Model + Liquidity-Weighted Portfolio

Tests:
  3.1  Tiered cost function (TIER1/2/3 based on liquidity)
  3.2  WF with hybrid costs vs uniform 7bps
  3.3  Liquidity-weighted portfolio (soft scaling)

Usage:
  python _research_r48_cost.py          # full Phase 3
  python _research_r48_cost.py --quick  # BTC/ETH/SOL only
"""

from __future__ import annotations

import argparse
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── project imports ───────────────────────────────────────────

from _research_round7 import WINDOWS, SYM_35
from _research_r30b_fixed import (
    compute_regime_extended,
    eval_with_costs,
    simulate_with_costs,
    train_ensemble,
)
from _research_r33_creative_features import FEAT_28
from _research_r35_new_features import (
    MARKET_LEVEL_FEATURES,
    add_r35_features,
    load_research_frame,
)
from _research_r47_coinglass import (
    CANONICAL_EXEC_CFG,
    CHAMPION_FEAT_30,
    add_cg_features,
    compute_cg_features,
    load_cg_daily,
    make_feature_set,
)

# ── config ─────────────────────────────────────────────────────

CHAMPION_FEAT_31 = CHAMPION_FEAT_30 + ["cg_taker_imb"]

TIER1_SYMS = {"BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"}
# TIER2 = mid-cap with decent liquidity (from SYM_35, excluding TIER1 and small)
TIER3_SYMS = {
    "SAND/USDT", "LDO/USDT", "INJ/USDT", "APT/USDT", "ARB/USDT",
    "GALA/USDT", "FTM/USDT", "MATIC/USDT",
}
TIER2_SYMS = set(SYM_35) - TIER1_SYMS - TIER3_SYMS


# ═══════════════════════════════════════════════════════════════
#  3.1 — Hybrid cost simulator
# ═══════════════════════════════════════════════════════════════

def simulate_with_hybrid_costs(merged, regime_df, cfg):
    """
    Like simulate_with_costs but with tiered cost model:
      TIER1: fill_prob=0.92, maker=-1bp, taker=7bp → expected ~0.4bp one-way
      TIER2: fill_prob=0.75, maker=1bp, taker=7bp  → expected ~2.5bp one-way
      TIER3: taker only → 7bp one-way
    """
    n_long = cfg.get("n_long", 6)
    n_short = cfg.get("n_short", 3)
    trend_cutoff = cfg.get("trend_cutoff", 0.9)
    rebal_hours = cfg.get("rebal_hours", 12)
    ema_alpha = cfg.get("ema_alpha", None)
    hysteresis = cfg.get("hysteresis", 0)

    # Tiered costs (one-way, in notional fraction)
    def _cost_for_sym(sym: str) -> float:
        if sym in TIER1_SYMS:
            # 92% maker fill at -1bp + 8% taker at 7bp
            return 0.92 * (-0.0001) + 0.08 * 0.0007  # ≈ -0.000036 (rebate!)
        elif sym in TIER2_SYMS:
            # 75% maker at 1bp + 25% taker at 7bp
            return 0.75 * 0.0001 + 0.25 * 0.0007  # ≈ 0.00025
        else:
            return 0.0005 + 0.0002  # taker + slippage = 7bp

    funding_per_12h = 0.00008

    all_rets = []
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
        if trend_str > trend_cutoff:
            continue
        grp = grouped[ts].copy()
        n = len(grp)

        nl, ns = n_long, n_short
        nl = min(nl, n // 3)
        ns = min(ns, n // 3)
        if nl == 0 and ns == 0:
            continue

        exposure = 1.0
        dyn_threshold = cfg.get("dyn_threshold", 0.5)
        if dyn_threshold is not None and trend_str > dyn_threshold:
            exposure = max(0.1, 1.0 - (trend_str - dyn_threshold) /
                          (trend_cutoff - dyn_threshold + 1e-10) * 0.5)

        if ema_alpha is not None and ema_alpha < 1.0:
            for _, r in grp.iterrows():
                sym = r["symbol"]
                raw_pred = r["pred"]
                if sym in prev_preds:
                    smoothed = ema_alpha * raw_pred + (1 - ema_alpha) * prev_preds[sym]
                else:
                    smoothed = raw_pred
                prev_preds[sym] = smoothed
                grp.loc[grp["symbol"] == sym, "pred"] = smoothed

        grp["pred_rank"] = grp["pred"].rank(ascending=False)

        if hysteresis > 0 and (prev_longs or prev_shorts):
            new_longs: Set[str] = set()
            new_shorts: Set[str] = set()
            for _, r in grp.iterrows():
                sym = r["symbol"]
                rank = r["pred_rank"]
                if sym in prev_longs and rank <= nl + hysteresis:
                    new_longs.add(sym)
                elif sym in prev_shorts and rank > (n - ns - hysteresis):
                    new_shorts.add(sym)
            remaining_long = nl - len(new_longs)
            remaining_short = ns - len(new_shorts)
            if remaining_long > 0:
                candidates = grp[~grp["symbol"].isin(new_longs | new_shorts)]
                candidates = candidates.sort_values("pred_rank")
                for _, r in candidates.head(remaining_long).iterrows():
                    new_longs.add(r["symbol"])
            if remaining_short > 0:
                candidates = grp[~grp["symbol"].isin(new_longs | new_shorts)]
                candidates = candidates.sort_values("pred_rank", ascending=False)
                for _, r in candidates.head(remaining_short).iterrows():
                    new_shorts.add(r["symbol"])
        else:
            new_longs = set(grp[grp["pred_rank"] <= nl]["symbol"].tolist()) if nl > 0 else set()
            new_shorts = set(grp[grp["pred_rank"] > (n - ns)]["symbol"].tolist()) if ns > 0 else set()

        # ── Hybrid cost calculation ──────────────────────────
        new_opened = (new_longs - prev_longs) | (new_shorts - prev_shorts)
        closed = (prev_longs - new_longs) | (prev_shorts - new_shorts)
        total_positions = len(new_longs) + len(new_shorts)

        if total_positions > 0:
            avg_weight = 1.0 / total_positions
            # Per-symbol cost based on tier
            turnover_cost = 0.0
            for sym in new_opened:
                turnover_cost += _cost_for_sym(sym) * avg_weight
            for sym in closed:
                turnover_cost += _cost_for_sym(sym) * avg_weight
            holding_cost = funding_per_12h * (rebal_hours / 12)
            total_cost = turnover_cost + holding_cost
        else:
            total_cost = 0.0

        turnover_count = len(new_opened) + len(closed)
        prev_longs = new_longs
        prev_shorts = new_shorts

        longs = grp[grp["symbol"].isin(new_longs)]
        shorts = grp[grp["symbol"].isin(new_shorts)]
        long_ret = longs["fwd_ret"].mean() if len(longs) > 0 else 0
        short_ret = shorts["fwd_ret"].mean() if len(shorts) > 0 else 0

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
            "turnover": turnover_count,
            "cost": total_cost,
        })

    if not all_rets:
        return pd.DataFrame()
    return pd.DataFrame(all_rets)


# ═══════════════════════════════════════════════════════════════
#  3.3 — Liquidity-weighted portfolio simulator
# ═══════════════════════════════════════════════════════════════

def simulate_liq_weighted(merged, regime_df, cfg, volume_df):
    """
    Liquidity-weighted L/S portfolio.
    weight_i = signal_rank_i × clip(log(volume_i) / median_log_volume, 0.5, 2.0)
    Uses uniform 7bp costs.
    """
    n_long = cfg.get("n_long", 6)
    n_short = cfg.get("n_short", 3)
    trend_cutoff = cfg.get("trend_cutoff", 0.9)
    rebal_hours = cfg.get("rebal_hours", 12)
    ema_alpha = cfg.get("ema_alpha", None)
    hysteresis = cfg.get("hysteresis", 0)

    taker_fee = 0.0005
    slippage = 0.0002
    cost_one_way = taker_fee + slippage
    funding_per_12h = 0.00008

    all_rets = []
    prev_longs: Set[str] = set()
    prev_shorts: Set[str] = set()
    prev_preds: Dict[str, float] = {}

    timestamps_sorted = sorted(merged["timestamp"].unique())
    grouped = {ts: grp for ts, grp in merged.groupby("timestamp")}
    vol_grouped = {ts: grp for ts, grp in volume_df.groupby("timestamp")} if volume_df is not None else {}
    rebal_timestamps = timestamps_sorted[::rebal_hours]

    for ts in rebal_timestamps:
        if ts not in regime_df.index or ts not in grouped:
            continue
        row = regime_df.loc[ts]
        trend_str = row.get("trend_strength", 0)
        if trend_str > trend_cutoff:
            continue
        grp = grouped[ts].copy()
        n = len(grp)

        nl, ns = n_long, n_short
        nl = min(nl, n // 3)
        ns = min(ns, n // 3)
        if nl == 0 and ns == 0:
            continue

        exposure = 1.0
        dyn_threshold = cfg.get("dyn_threshold", 0.5)
        if dyn_threshold is not None and trend_str > dyn_threshold:
            exposure = max(0.1, 1.0 - (trend_str - dyn_threshold) /
                          (trend_cutoff - dyn_threshold + 1e-10) * 0.5)

        if ema_alpha is not None and ema_alpha < 1.0:
            for _, r in grp.iterrows():
                sym = r["symbol"]
                raw_pred = r["pred"]
                if sym in prev_preds:
                    smoothed = ema_alpha * raw_pred + (1 - ema_alpha) * prev_preds[sym]
                else:
                    smoothed = raw_pred
                prev_preds[sym] = smoothed
                grp.loc[grp["symbol"] == sym, "pred"] = smoothed

        grp["pred_rank"] = grp["pred"].rank(ascending=False)

        new_longs = set(grp[grp["pred_rank"] <= nl]["symbol"].tolist()) if nl > 0 else set()
        new_shorts = set(grp[grp["pred_rank"] > (n - ns)]["symbol"].tolist()) if ns > 0 else set()

        # ── Liquidity weighting ──────────────────────────────
        # Get volume data for this timestamp
        vol_grp = vol_grouped.get(ts)
        if vol_grp is not None:
            sym_vol = dict(zip(vol_grp["symbol"], vol_grp["volume"]))
        else:
            sym_vol = {}

        def _get_liq_weight(sym: str) -> float:
            v = sym_vol.get(sym, 0)
            if v <= 0:
                return 1.0
            log_v = np.log1p(v)
            # Median across active symbols
            all_vols = [np.log1p(sym_vol.get(s, 1)) for s in (new_longs | new_shorts)]
            med = np.median(all_vols) if all_vols else log_v
            return float(np.clip(log_v / (med + 1e-10), 0.5, 2.0))

        longs = grp[grp["symbol"].isin(new_longs)]
        shorts = grp[grp["symbol"].isin(new_shorts)]

        # Weighted returns
        if len(longs) > 0:
            w_l = np.array([_get_liq_weight(s) for s in longs["symbol"]])
            w_l = w_l / w_l.sum()
            long_ret = float((longs["fwd_ret"].values * w_l).sum())
        else:
            long_ret = 0
        if len(shorts) > 0:
            w_s = np.array([_get_liq_weight(s) for s in shorts["symbol"]])
            w_s = w_s / w_s.sum()
            short_ret = float((shorts["fwd_ret"].values * w_s).sum())
        else:
            short_ret = 0

        new_opened = (new_longs - prev_longs) | (new_shorts - prev_shorts)
        closed = (prev_longs - new_longs) | (prev_shorts - new_shorts)
        turnover_count = len(new_opened) + len(closed)
        total_positions = len(new_longs) + len(new_shorts)

        if total_positions > 0:
            avg_weight = 1.0 / total_positions
            turnover_cost = turnover_count * cost_one_way * avg_weight
            holding_cost = funding_per_12h * (rebal_hours / 12)
            total_cost = turnover_cost + holding_cost
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
            "turnover": turnover_count,
            "cost": total_cost,
        })

    if not all_rets:
        return pd.DataFrame()
    return pd.DataFrame(all_rets)


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main(quick: bool = False) -> None:
    print("=" * 80)
    print("R48 Phase 3 — HYBRID COST MODEL")
    print("=" * 80)

    # ── Load CG daily features ──────────────────────────────────
    print("\n[1] Loading CoinGlass daily data ...")
    cg = load_cg_daily()
    cg_feats_daily = compute_cg_features(cg)
    if cg_feats_daily.empty:
        print("❌ No CG features — aborting")
        return

    # ── Load research frame ──────────────────────────────────────
    print("\n[2] Loading research frame ...")
    df, regime_df = load_research_frame()
    df, _ = add_r35_features(df)
    regime_df = regime_df.sort_index()
    print(f"  Base frame: {len(df):,} rows × {len(df.columns)} cols")

    if quick:
        print("  ⚡ Quick mode: BTC/ETH/SOL only ...")
        df = df[df["symbol"].isin(["BTC/USDT", "ETH/USDT", "SOL/USDT"])].copy()

    # ── Merge CG features ───────────────────────────────────────
    print("\n[3] Merging CG features ...")
    df, per_sym_cols, mkt_cols = add_cg_features(df, cg_feats_daily)

    # ── Train predictions once (31f = champion + cg_taker_imb) ──
    print("\n[4] Training champion_31f predictions ...")
    feats, no_rank = make_feature_set(["cg_taker_imb"], mkt_cols)
    preds = train_ensemble(df, feats, WINDOWS, l2=1.0, rolling=False,
                           label="cost_31f", cs_rank_exclude=no_rank)
    if preds is None or preds.empty:
        print("❌ Training failed — aborting")
        return

    # ── 3.1 + 3.2: Uniform vs Hybrid costs ─────────────────────
    print("\n" + "=" * 70)
    print("  3.1/3.2 — Uniform 7bps vs Hybrid Tiered Costs")
    print("=" * 70)

    rows = []
    for window in ["W1", "W2", "W3", "ALL"]:
        subset = preds if window == "ALL" else preds[preds["window"] == window]

        # Uniform costs
        port_uniform = simulate_with_costs(subset, regime_df, CANONICAL_EXEC_CFG)
        m_uniform = eval_with_costs(port_uniform, f"uniform_{window}")

        # Hybrid costs
        port_hybrid = simulate_with_hybrid_costs(subset, regime_df, CANONICAL_EXEC_CFG)
        m_hybrid = eval_with_costs(port_hybrid, f"hybrid_{window}")

        rows.append({
            "window": window,
            "uniform_sharpe": m_uniform["sharpe"],
            "hybrid_sharpe": m_hybrid["sharpe"],
            "uniform_cost": m_uniform.get("total_cost_pct", 0),
            "hybrid_cost": m_hybrid.get("total_cost_pct", 0),
            "uniform_dd": m_uniform.get("max_dd_pct", 0),
            "hybrid_dd": m_hybrid.get("max_dd_pct", 0),
        })

    cost_summary = pd.DataFrame(rows)
    print(f"\n  {'Window':<6} {'Unif_Sh':>8} {'Hyb_Sh':>8} {'Unif_Cost%':>10} {'Hyb_Cost%':>10} {'Δ_Cost':>7}")
    print(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*10} {'─'*10} {'─'*7}")
    for _, r in cost_summary.iterrows():
        delta_cost = r["hybrid_cost"] - r["uniform_cost"]
        print(f"  {r['window']:<6} {r['uniform_sharpe']:>+7.2f} {r['hybrid_sharpe']:>+7.2f} "
              f"{r['uniform_cost']:>9.1f}% {r['hybrid_cost']:>9.1f}% {delta_cost:>+6.1f}%")

    cost_summary.to_csv("results_r48_phase3_cost_comparison.csv", index=False)
    print("  → Saved results_r48_phase3_cost_comparison.csv")

    # ── 3.3: Liquidity-weighted portfolio ───────────────────────
    print("\n" + "=" * 70)
    print("  3.3 — Liquidity-Weighted Portfolio")
    print("=" * 70)

    # Prepare volume data
    volume_df = df[["timestamp", "symbol", "volume"]].copy()

    lw_rows = []
    for window in ["W1", "W2", "W3", "ALL"]:
        subset = preds if window == "ALL" else preds[preds["window"] == window]

        # Equal-weight (baseline)
        port_eq = simulate_with_costs(subset, regime_df, CANONICAL_EXEC_CFG)
        m_eq = eval_with_costs(port_eq, f"equal_{window}")

        # Liquidity-weighted
        port_lw = simulate_liq_weighted(subset, regime_df, CANONICAL_EXEC_CFG, volume_df)
        m_lw = eval_with_costs(port_lw, f"liqwt_{window}")

        # Hybrid costs + liquidity-weighted (best case)
        port_best = simulate_with_hybrid_costs(subset, regime_df, CANONICAL_EXEC_CFG)
        m_best = eval_with_costs(port_best, f"hybrid_{window}")

        lw_rows.append({
            "window": window,
            "equal_sharpe": m_eq["sharpe"],
            "liqwt_sharpe": m_lw["sharpe"],
            "hybrid_sharpe": m_best["sharpe"],
            "equal_cost": m_eq.get("total_cost_pct", 0),
            "liqwt_cost": m_lw.get("total_cost_pct", 0),
        })

    lw_summary = pd.DataFrame(lw_rows)
    print(f"\n  {'Window':<6} {'EqWt_Sh':>8} {'LiqWt_Sh':>9} {'Hyb_Sh':>7} {'EqWt_Cost%':>11} {'LiqWt_Cost%':>11}")
    print(f"  {'─'*6} {'─'*8} {'─'*9} {'─'*7} {'─'*11} {'─'*11}")
    for _, r in lw_summary.iterrows():
        print(f"  {r['window']:<6} {r['equal_sharpe']:>+7.2f} {r['liqwt_sharpe']:>+8.2f} "
              f"{r['hybrid_sharpe']:>+6.2f} {r['equal_cost']:>10.1f}% {r['liqwt_cost']:>10.1f}%")

    lw_summary.to_csv("results_r48_phase3_liqwt.csv", index=False)
    print("  → Saved results_r48_phase3_liqwt.csv")

    # ── Final summary ───────────────────────────────────────────
    print("\n" + "=" * 80)
    print("R48 Phase 3 — COMPLETE")
    print("=" * 80)

    # Best cost model
    all_row = cost_summary[cost_summary["window"] == "ALL"].iloc[0]
    print(f"\n  Uniform costs:  ALL Sharpe = {all_row['uniform_sharpe']:+.2f}, Cost = {all_row['uniform_cost']:.1f}%")
    print(f"  Hybrid costs:   ALL Sharpe = {all_row['hybrid_sharpe']:+.2f}, Cost = {all_row['hybrid_cost']:.1f}%")

    lw_all = lw_summary[lw_summary["window"] == "ALL"].iloc[0]
    print(f"  Liq-weighted:   ALL Sharpe = {lw_all['liqwt_sharpe']:+.2f}, Cost = {lw_all['liqwt_cost']:.1f}%")

    # Save best config for combo script
    import json
    best_cost = {
        "hybrid_all_sharpe": float(all_row["hybrid_sharpe"]),
        "hybrid_all_cost": float(all_row["hybrid_cost"]),
        "liqwt_all_sharpe": float(lw_all["liqwt_sharpe"]),
        "improved_cost": float(all_row["hybrid_cost"] < all_row["uniform_cost"]),
        "improved_sharpe": float(all_row["hybrid_sharpe"] > all_row["uniform_sharpe"]),
    }
    with open("results_r48_phase3_best.json", "w") as f:
        json.dump(best_cost, f, indent=2)
    print(f"\n  → Saved results_r48_phase3_best.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    main(quick=args.quick)
