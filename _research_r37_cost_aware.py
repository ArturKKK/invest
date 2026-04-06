#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R37 — cost-aware execution rules.

Outputs:
  - results_r37.log
  - results_r37_execution_summary.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from _research_round7 import SYM_35, WINDOWS
from _ic_scanner import build_features_minimal, load_derivatives, load_ohlcv
from _research_r22_models import add_new_features, build_r19_features
from _research_r30b_fixed import add_extra_features_clean, compute_regime_extended, eval_with_costs, train_ensemble
from _research_r33_creative_features import FEAT_28, add_r33_features


BASE_DIR = Path(__file__).resolve().parent

TAKER_FEE = 0.0005
SLIPPAGE = 0.0002
FUNDING_PER_12H = 0.00008
COST_ONE_WAY = TAKER_FEE + SLIPPAGE

BASE_CFG = {
    "n_long": 6,
    "n_short": 3,
    "rebal_hours": 12,
    "trend_cutoff": 0.9,
    "dyn_threshold": 0.7,
    "ema_alpha": 0.5,
    "hysteresis": 3,
}


def load_research_frame() -> Tuple[pd.DataFrame, pd.DataFrame]:
    ohlcv = load_ohlcv()
    ohlcv = ohlcv[ohlcv["symbol"].isin(SYM_35)]
    derivs = load_derivatives()
    df = build_features_minimal(ohlcv, derivs)
    df = build_r19_features(df)
    df, _ = add_new_features(df)
    df = df[df["symbol"].isin(SYM_35)].copy()
    df = add_extra_features_clean(df)
    df = add_r33_features(df)
    regime_df = compute_regime_extended(df)
    return df, regime_df


def add_liquidity_rank(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "timestamp"]).copy()
    df["dollar_volume_24h"] = (
        (df["close"] * df["volume"]).groupby(df["symbol"]).transform(
            lambda series: series.rolling(24, min_periods=12).sum()
        )
    )
    df["liquidity_rank"] = df.groupby("timestamp")["dollar_volume_24h"].rank(pct=True)
    return df[["timestamp", "symbol", "liquidity_rank"]]


def estimate_edge_threshold(preds: pd.DataFrame, multiplier: float) -> float:
    pred_std = float(preds["pred"].std())
    return pred_std * 0.1 * multiplier


def simulate_cost_aware(
    merged: pd.DataFrame,
    regime_df: pd.DataFrame,
    cfg: dict,
    liquidity_floor: Optional[float] = None,
    edge_multiplier: Optional[float] = None,
) -> pd.DataFrame:
    n_long = cfg.get("n_long", 6)
    n_short = cfg.get("n_short", 3)
    trend_cutoff = cfg.get("trend_cutoff", 0.9)
    rebal_hours = cfg.get("rebal_hours", 12)
    ema_alpha = cfg.get("ema_alpha")
    hysteresis = cfg.get("hysteresis", 0)
    dyn_threshold = cfg.get("dyn_threshold", 0.7)

    all_rets = []
    prev_longs: set[str] = set()
    prev_shorts: set[str] = set()
    prev_preds: dict[str, float] = {}
    grouped = {timestamp: group.copy() for timestamp, group in merged.groupby("timestamp")}
    rebal_timestamps = sorted(merged["timestamp"].unique())[::rebal_hours]
    edge_threshold = estimate_edge_threshold(merged, edge_multiplier or 0.0)

    for timestamp in rebal_timestamps:
        if timestamp not in grouped or timestamp not in regime_df.index:
            continue
        trend_strength = regime_df.loc[timestamp].get("trend_strength", 0.0)
        if trend_strength > trend_cutoff:
            continue

        group = grouped[timestamp].copy()
        if liquidity_floor is not None and "liquidity_rank" in group.columns:
            group = group[group["liquidity_rank"] >= liquidity_floor].copy()
        if len(group) == 0:
            continue

        if ema_alpha is not None and ema_alpha < 1.0:
            for index, row in group.iterrows():
                symbol = row["symbol"]
                raw_pred = row["pred"]
                smoothed = ema_alpha * raw_pred + (1 - ema_alpha) * prev_preds.get(symbol, raw_pred)
                prev_preds[symbol] = smoothed
                group.at[index, "pred"] = smoothed

        group["pred_rank"] = group["pred"].rank(ascending=False, method="first")
        n = len(group)
        nl = min(n_long, n // 3)
        ns = min(n_short, n // 3)
        if nl == 0 and ns == 0:
            continue

        exposure = 1.0
        if dyn_threshold is not None and trend_strength > dyn_threshold:
            exposure = max(0.1, 1.0 - (trend_strength - dyn_threshold) /
                           (trend_cutoff - dyn_threshold + 1e-10) * 0.5)

        long_cutoff = group.sort_values("pred", ascending=False).iloc[max(nl - 1, 0)]["pred"] if nl > 0 else None
        short_cutoff = group.sort_values("pred", ascending=True).iloc[max(ns - 1, 0)]["pred"] if ns > 0 else None

        new_longs = set()
        new_shorts = set()
        if hysteresis > 0 and (prev_longs or prev_shorts):
            for _, row in group.iterrows():
                symbol = row["symbol"]
                rank = row["pred_rank"]
                if symbol in prev_longs and rank <= nl + hysteresis:
                    new_longs.add(symbol)
                elif symbol in prev_shorts and rank > (n - ns - hysteresis):
                    new_shorts.add(symbol)

        remaining_long = nl - len(new_longs)
        remaining_short = ns - len(new_shorts)
        if remaining_long > 0:
            candidates = group[~group["symbol"].isin(new_longs | new_shorts)].sort_values("pred", ascending=False)
            for _, row in candidates.iterrows():
                if remaining_long <= 0:
                    break
                if edge_multiplier and long_cutoff is not None and row["symbol"] not in prev_longs:
                    if float(row["pred"] - long_cutoff) < edge_threshold:
                        continue
                new_longs.add(row["symbol"])
                remaining_long -= 1
        if remaining_short > 0:
            candidates = group[~group["symbol"].isin(new_longs | new_shorts)].sort_values("pred", ascending=True)
            for _, row in candidates.iterrows():
                if remaining_short <= 0:
                    break
                if edge_multiplier and short_cutoff is not None and row["symbol"] not in prev_shorts:
                    if float(short_cutoff - row["pred"]) < edge_threshold:
                        continue
                new_shorts.add(row["symbol"])
                remaining_short -= 1

        new_opened = (new_longs - prev_longs) | (new_shorts - prev_shorts)
        closed = (prev_longs - new_longs) | (prev_shorts - new_shorts)
        turnover_count = len(new_opened) + len(closed)
        total_positions = len(new_longs) + len(new_shorts)
        avg_weight = (1.0 / total_positions) if total_positions > 0 else 0.0
        turnover_cost = turnover_count * COST_ONE_WAY * avg_weight if total_positions > 0 else 0.0
        holding_cost = FUNDING_PER_12H * (rebal_hours / 12)
        total_cost = turnover_cost + holding_cost

        longs = group[group["symbol"].isin(new_longs)].copy()
        shorts = group[group["symbol"].isin(new_shorts)].copy()
        long_ret = float(longs["fwd_ret"].mean()) if len(longs) else 0.0
        short_ret = float(shorts["fwd_ret"].mean()) if len(shorts) else 0.0

        if nl > 0 and ns > 0:
            portfolio_ret = 0.5 * long_ret - 0.5 * short_ret
        elif ns > 0:
            portfolio_ret = -short_ret
        else:
            portfolio_ret = long_ret
        portfolio_ret *= exposure
        portfolio_ret -= total_cost

        all_rets.append({
            "timestamp": timestamp,
            "portfolio_ret": portfolio_ret,
            "gross_ret": portfolio_ret + total_cost,
            "turnover": turnover_count,
            "cost": total_cost,
            "n_long": len(new_longs),
            "n_short": len(new_shorts),
            "exposure": exposure,
        })

        prev_longs = new_longs
        prev_shorts = new_shorts

    return pd.DataFrame(all_rets)


def main() -> None:
    summary_path = BASE_DIR / "results_r37_execution_summary.csv"

    print("=" * 80)
    print("R37 — COST-AWARE EXECUTION")
    print("=" * 80)

    print("\n[1] Loading research frame...")
    df, regime_df = load_research_frame()
    liquidity = add_liquidity_rank(df)
    regime_df = regime_df.sort_index()
    print(f"  Data: {len(df):,} rows, {len(df.columns)} cols")

    print("\n[2] Training baseline predictions...")
    preds = train_ensemble(df, FEAT_28, WINDOWS, l2=1.0, rolling=False, label="A_28f")
    if preds is None or preds.empty:
        raise RuntimeError("No predictions produced")
    preds = preds.merge(liquidity, on=["timestamp", "symbol"], how="left")

    configs = [
        ("baseline_ema05_h3", {**BASE_CFG}, None, None),
        ("band_k1", {**BASE_CFG, "hysteresis": 1}, None, None),
        ("band_k2", {**BASE_CFG, "hysteresis": 2}, None, None),
        ("band_k3", {**BASE_CFG, "hysteresis": 3}, None, None),
        ("edge_x1", {**BASE_CFG, "hysteresis": 3}, None, 1.0),
        ("edge_x2", {**BASE_CFG, "hysteresis": 3}, None, 2.0),
        ("liq60", {**BASE_CFG, "hysteresis": 3}, 0.60, None),
        ("liq70", {**BASE_CFG, "hysteresis": 3}, 0.70, None),
        ("combo_liq60_edge2", {**BASE_CFG, "hysteresis": 3}, 0.60, 2.0),
    ]

    print("\n[3] Execution sweeps...")
    rows = []
    for name, cfg, liquidity_floor, edge_multiplier in configs:
        print(f"  {name}...")
        for window in ["W1", "W2", "W3", "ALL"]:
            subset = preds if window == "ALL" else preds[preds["window"] == window].copy()
            port = simulate_cost_aware(
                subset,
                regime_df,
                cfg,
                liquidity_floor=liquidity_floor,
                edge_multiplier=edge_multiplier,
            )
            metric = eval_with_costs(port, f"{name}_{window}")
            rows.append({
                "config": name,
                "window": window,
                "liquidity_floor": liquidity_floor,
                "edge_multiplier": edge_multiplier,
                "hysteresis": cfg.get("hysteresis", 0),
                "sharpe": metric.get("sharpe", np.nan),
                "sharpe_gross": metric.get("sharpe_gross", np.nan),
                "equity": metric.get("equity", np.nan),
                "cost_pct": metric.get("total_cost_pct", np.nan),
                "avg_turnover": metric.get("avg_turnover", np.nan),
                "max_dd_pct": metric.get("max_dd_pct", np.nan),
            })

    result_df = pd.DataFrame(rows).sort_values(["window", "sharpe"], ascending=[True, False])
    result_df.to_csv(summary_path, index=False)

    print("\n[4] Best execution variants")
    for window in ["W2", "W3", "ALL"]:
        top = result_df[result_df["window"] == window].head(5)
        if top.empty:
            continue
        print(f"  {window}:")
        print(top[["config", "sharpe", "cost_pct", "avg_turnover"]].to_string(index=False))

    print("\n[5] Saved artifacts")
    print(f"  Summary CSV: {summary_path.name}")


if __name__ == "__main__":
    main()