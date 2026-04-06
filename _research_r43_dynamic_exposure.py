#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R43 — dynamic net exposure.

Goal:
  reduce the weak long-leg bias in bad regimes by dynamically adjusting
  the long/short book sizes under the same canonical training setup.

Feature set options:
  - baseline: FEAT_28
  - r42_candidate: FEAT_28 + {ret_dispersion_12h, cs_rank_ma_5}
  - r35a_full: FEAT_28 + full R35a bundle

Outputs:
  - results_r43.log
  - results_r43_dynamic_exposure_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from _research_round7 import WINDOWS
from _research_r30b_fixed import compute_regime_extended, eval_with_costs, train_ensemble
from _research_r33_creative_features import FEAT_28
from _research_r35_new_features import GROUPS, MARKET_LEVEL_FEATURES, add_r35_features, load_research_frame


BASE_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = BASE_DIR / "results_r43_dynamic_exposure_summary.csv"

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

R35A_FEATURES = GROUPS["r35a_cs_second_order"]
R42_CANDIDATE = ["ret_dispersion_12h", "cs_rank_ma_5"]

FEATURE_SETS = {
    "baseline": FEAT_28,
    "r42_candidate": FEAT_28 + R42_CANDIDATE,
    "r35a_full": FEAT_28 + R35A_FEATURES,
}


def build_feature_set(features: Sequence[str]) -> Tuple[List[str], List[str]]:
    feats = list(dict.fromkeys(features))
    no_rank = [feature for feature in feats if feature in MARKET_LEVEL_FEATURES]
    return feats, no_rank


def build_market_context(df: pd.DataFrame) -> pd.DataFrame:
    context = (
        df.groupby("timestamp")
        .agg(
            breadth_12h=("pct_coins_up_12h", "mean"),
            dispersion_12h=("ret_12h", "std"),
        )
        .sort_index()
    )
    btc = df[df["symbol"] == "BTC/USDT"].copy().set_index("timestamp")
    btc_context = btc[["rvol_24h"]].rename(columns={"rvol_24h": "btc_rvol_24h"})
    context = context.join(btc_context, how="left")
    context["dispersion_pct"] = context["dispersion_12h"].rank(pct=True)
    context["btc_rvol_pct"] = context["btc_rvol_24h"].rank(pct=True)
    return context.reset_index()


def choose_book_sizes(rule_name: str, row: pd.Series, base_long: int = 6, base_short: int = 3) -> Tuple[int, int, str]:
    breadth = float(row.get("breadth_12h", np.nan))
    dispersion_pct = float(row.get("dispersion_pct", np.nan))
    rvol_pct = float(row.get("btc_rvol_pct", np.nan))

    if rule_name == "baseline":
        return base_long, base_short, "base"
    if rule_name == "breadth_5L4S":
        if pd.notna(breadth) and breadth < 0.45:
            return 5, 4, "breadth_low"
        return base_long, base_short, "base"
    if rule_name == "breadth_4L4S":
        if pd.notna(breadth) and breadth < 0.45:
            return 4, 4, "breadth_low"
        return base_long, base_short, "base"
    if rule_name == "dispersion_4L4S":
        if pd.notna(dispersion_pct) and dispersion_pct < 0.35:
            return 4, 4, "disp_low"
        return base_long, base_short, "base"
    if rule_name == "breadth_or_disp_4L4S":
        if (pd.notna(breadth) and breadth < 0.45) or (pd.notna(dispersion_pct) and dispersion_pct < 0.35):
            return 4, 4, "breadth_or_disp"
        return base_long, base_short, "base"
    if rule_name == "stress_4L5S":
        if (pd.notna(breadth) and breadth < 0.45) and (pd.notna(rvol_pct) and rvol_pct >= 0.70):
            return 4, 5, "stress"
        if (pd.notna(breadth) and breadth < 0.45) or (pd.notna(dispersion_pct) and dispersion_pct < 0.35):
            return 5, 4, "soft_stress"
        return base_long, base_short, "base"
    raise ValueError(f"Unknown rule: {rule_name}")


def simulate_dynamic_exposure(preds: pd.DataFrame, regime_df: pd.DataFrame, context_df: pd.DataFrame, cfg: dict, rule_name: str) -> pd.DataFrame:
    trend_cutoff = cfg.get("trend_cutoff", 0.9)
    rebal_hours = cfg.get("rebal_hours", 12)
    ema_alpha = cfg.get("ema_alpha")
    hysteresis = cfg.get("hysteresis", 0)
    dyn_threshold = cfg.get("dyn_threshold", 0.7)

    all_rets = []
    prev_longs: set[str] = set()
    prev_shorts: set[str] = set()
    prev_preds: dict[str, float] = {}

    grouped = {timestamp: group.copy() for timestamp, group in preds.groupby("timestamp")}
    context = context_df.set_index("timestamp")
    rebal_timestamps = sorted(preds["timestamp"].unique())[::rebal_hours]

    for timestamp in rebal_timestamps:
        if timestamp not in grouped or timestamp not in regime_df.index or timestamp not in context.index:
            continue

        trend_strength = regime_df.loc[timestamp].get("trend_strength", 0.0)
        if trend_strength > trend_cutoff:
            continue

        group = grouped[timestamp].copy()
        n = len(group)
        if n == 0:
            continue

        book_n_long, book_n_short, state = choose_book_sizes(rule_name, context.loc[timestamp], cfg["n_long"], cfg["n_short"])
        book_n_long = min(book_n_long, n // 3)
        book_n_short = min(book_n_short, n // 3)
        if book_n_long == 0 and book_n_short == 0:
            continue

        if ema_alpha is not None and ema_alpha < 1.0:
            for index, row in group.iterrows():
                symbol = row["symbol"]
                raw_pred = row["pred"]
                smoothed = ema_alpha * raw_pred + (1 - ema_alpha) * prev_preds.get(symbol, raw_pred)
                prev_preds[symbol] = smoothed
                group.at[index, "pred"] = smoothed

        group["pred_rank"] = group["pred"].rank(ascending=False, method="first")

        exposure = 1.0
        if dyn_threshold is not None and trend_strength > dyn_threshold:
            exposure = max(0.1, 1.0 - (trend_strength - dyn_threshold) /
                           (trend_cutoff - dyn_threshold + 1e-10) * 0.5)

        if hysteresis > 0 and (prev_longs or prev_shorts):
            new_longs = set()
            new_shorts = set()
            for _, row in group.iterrows():
                symbol = row["symbol"]
                rank = row["pred_rank"]
                if symbol in prev_longs and rank <= book_n_long + hysteresis:
                    new_longs.add(symbol)
                elif symbol in prev_shorts and rank > (n - book_n_short - hysteresis):
                    new_shorts.add(symbol)

            remain_long = book_n_long - len(new_longs)
            remain_short = book_n_short - len(new_shorts)
            if remain_long > 0:
                candidates = group[~group["symbol"].isin(new_longs | new_shorts)].sort_values("pred", ascending=False)
                new_longs |= set(candidates.head(remain_long)["symbol"].tolist())
            if remain_short > 0:
                candidates = group[~group["symbol"].isin(new_longs | new_shorts)].sort_values("pred", ascending=True)
                new_shorts |= set(candidates.head(remain_short)["symbol"].tolist())
        else:
            new_longs = set(group.nlargest(book_n_long, "pred")["symbol"].tolist()) if book_n_long > 0 else set()
            new_shorts = set(group.nsmallest(book_n_short, "pred")["symbol"].tolist()) if book_n_short > 0 else set()

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

        if book_n_long > 0 and book_n_short > 0:
            portfolio_ret = 0.5 * long_ret - 0.5 * short_ret
        elif book_n_short > 0:
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
            "state": state,
        })

        prev_longs = new_longs
        prev_shorts = new_shorts

    return pd.DataFrame(all_rets)


def evaluate_rule(preds: pd.DataFrame, regime_df: pd.DataFrame, context_df: pd.DataFrame, rule_name: str) -> Dict[str, object]:
    row: Dict[str, object] = {"rule": rule_name}
    all_states: List[pd.Series] = []
    for window in ["W1", "W2", "W3", "ALL"]:
        subset = preds if window == "ALL" else preds[preds["window"] == window].copy()
        port = simulate_dynamic_exposure(subset, regime_df, context_df, BASE_CFG, rule_name)
        metric = eval_with_costs(port, f"{rule_name}_{window}")
        row[f"{window}_sh"] = metric.get("sharpe", 0.0)
        row[f"{window}_sh_gross"] = metric.get("sharpe_gross", 0.0)
        row[f"{window}_eq"] = metric.get("equity", 0.0)
        row[f"{window}_cost"] = metric.get("total_cost_pct", 0.0)
        row[f"{window}_turn"] = metric.get("avg_turnover", 0.0)
        row[f"{window}_dd"] = metric.get("max_dd_pct", 0.0)
        if not port.empty:
            all_states.append(port["state"])
    if all_states:
        states = pd.concat(all_states)
        row["non_base_share"] = float((states != "base").mean())
    else:
        row["non_base_share"] = 0.0
    row["passes_target"] = bool(row["W2_sh"] > 0 and row["W3_sh"] > 2.0)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="r42_candidate")
    args = parser.parse_args()

    print("=" * 80)
    print("R43 — DYNAMIC NET EXPOSURE")
    print("=" * 80)
    print(f"Feature set: {args.feature_set}")
    print(BASE_CFG)

    print("\n[1] Loading research frame...")
    df, regime_df = load_research_frame()
    df, _ = add_r35_features(df)
    regime_df = compute_regime_extended(df).sort_index()
    context_df = build_market_context(df)
    print(f"  Data: {len(df):,} rows, {len(df.columns)} cols")

    print("\n[2] Training base predictions...")
    feats, no_rank = build_feature_set(FEATURE_SETS[args.feature_set])
    preds = train_ensemble(
        df,
        feats,
        WINDOWS,
        l2=1.0,
        rolling=False,
        label=args.feature_set,
        cs_rank_exclude=no_rank,
    )
    if preds is None or preds.empty:
        raise RuntimeError("No predictions produced for R43")

    rules = [
        "baseline",
        "breadth_5L4S",
        "breadth_4L4S",
        "dispersion_4L4S",
        "breadth_or_disp_4L4S",
        "stress_4L5S",
    ]

    print("\n[3] Evaluating dynamic exposure rules...")
    rows = []
    for rule in rules:
        print(f"  {rule}...")
        row = evaluate_rule(preds, regime_df, context_df, rule)
        rows.append(row)
        print(
            f"    -> W2={row['W2_sh']:.2f}, W3={row['W3_sh']:.2f}, ALL={row['ALL_sh']:.2f}, "
            f"non_base_share={row['non_base_share']:.2%}"
        )

    summary_df = pd.DataFrame(rows).sort_values(["ALL_sh", "W2_sh", "W3_sh"], ascending=False).reset_index(drop=True)
    baseline_all = float(summary_df.loc[summary_df["rule"] == "baseline", "ALL_sh"].iloc[0])
    summary_df["delta_all_vs_baseline"] = summary_df["ALL_sh"] - baseline_all
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print("\n[4] Best rules")
    print(summary_df[["rule", "W2_sh", "W3_sh", "ALL_sh", "ALL_cost", "ALL_turn", "non_base_share", "delta_all_vs_baseline"]].to_string(index=False))

    print("\n[5] Saved artifacts")
    print(f"  Summary CSV: {SUMMARY_PATH.name}")


if __name__ == "__main__":
    main()