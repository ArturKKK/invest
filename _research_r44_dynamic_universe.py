#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R44 — dynamic universe / coin quality filter.

Goal:
  remove toxic symbols from the candidate universe without leaking test data.

Approach:
  - base feature set = R42 winner: FEAT_28 + {ret_dispersion_12h, cs_rank_ma_5}
  - liquidity filters from rolling 30d/60d dollar volume + OI ranks
  - long-quality filter from validation-period selected-long performance

Outputs:
  - results_r44.log
  - results_r44_dynamic_universe_summary.csv
  - results_r44_quality_filter.csv
  - results_r44_toxic_coin_check.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from _research_round7 import WINDOWS
from _research_r30b_fixed import compute_regime_extended, eval_with_costs, train_ensemble
from _research_r33_creative_features import FEAT_28
from _research_r35_new_features import MARKET_LEVEL_FEATURES, add_r35_features, load_research_frame


BASE_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = BASE_DIR / "results_r44_dynamic_universe_summary.csv"
QUALITY_PATH = BASE_DIR / "results_r44_quality_filter.csv"
TOXIC_CHECK_PATH = BASE_DIR / "results_r44_toxic_coin_check.csv"

BASE_CFG = {
    "n_long": 6,
    "n_short": 3,
    "rebal_hours": 12,
    "trend_cutoff": 0.9,
    "dyn_threshold": 0.7,
    "ema_alpha": 0.5,
    "hysteresis": 3,
}

TAKER_FEE = 0.0005
SLIPPAGE = 0.0002
FUNDING_PER_12H = 0.00008
COST_ONE_WAY = TAKER_FEE + SLIPPAGE

R42_CANDIDATE = ["ret_dispersion_12h", "cs_rank_ma_5"]
TOXIC_LONGS_W2 = ["XRP/USDT", "ADA/USDT", "SAND/USDT", "APT/USDT"]

FEATURE_SETS = {
    "baseline": FEAT_28,
    "r42_candidate": FEAT_28 + R42_CANDIDATE,
}


def build_feature_set(features: Sequence[str]) -> Tuple[List[str], List[str]]:
    feats = list(dict.fromkeys(features))
    no_rank = [feature for feature in feats if feature in MARKET_LEVEL_FEATURES]
    return feats, no_rank


def build_validation_windows() -> List[Dict[str, str]]:
    return [
        {
            "name": window["name"],
            "train_end": window["train_end"],
            "val_start": window["val_start"],
            "val_end": window["val_end"],
            "test_start": window["val_start"],
            "test_end": window["val_end"],
        }
        for window in WINDOWS
    ]


def build_universe_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "timestamp"]).copy()
    df["dollar_volume"] = df["close"] * df["volume"]

    windows = [(30, 24 * 30), (60, 24 * 60)]
    for days, hours in windows:
        min_periods = max(hours // 2, 24 * 10)
        df[f"dollar_vol_{days}d"] = df.groupby("symbol")["dollar_volume"].transform(
            lambda series: series.rolling(hours, min_periods=min_periods).mean()
        )
        df[f"oi_value_{days}d"] = df.groupby("symbol")["oi_value_usd"].transform(
            lambda series: series.rolling(hours, min_periods=min_periods).mean()
        )
        df[f"dv_rank_{days}d"] = df.groupby("timestamp")[f"dollar_vol_{days}d"].rank(pct=True)
        df[f"oi_rank_{days}d"] = df.groupby("timestamp")[f"oi_value_{days}d"].rank(pct=True)
        df[f"liq_combo_{days}d"] = 0.5 * (df[f"dv_rank_{days}d"] + df[f"oi_rank_{days}d"])

    return df[[
        "timestamp",
        "symbol",
        "liq_combo_30d",
        "liq_combo_60d",
        "dv_rank_30d",
        "oi_rank_30d",
        "dv_rank_60d",
        "oi_rank_60d",
    ]].copy()


def derive_long_quality(val_preds: pd.DataFrame, n_long: int, n_short: int, rebal_hours: int) -> Tuple[pd.DataFrame, Dict[str, set[str]]]:
    rows: List[Dict[str, object]] = []
    blocked: Dict[str, set[str]] = {}

    for window in ["W1", "W2", "W3"]:
        sub = val_preds[val_preds["window"] == window].copy()
        if sub.empty:
            blocked[window] = set()
            continue

        timestamps = sorted(sub["timestamp"].unique())[::rebal_hours]
        for timestamp in timestamps:
            grp = sub[sub["timestamp"] == timestamp].copy()
            n = len(grp)
            nl = min(n_long, n // 3)
            ns = min(n_short, n // 3)
            if nl > 0:
                longs = grp.nlargest(nl, "pred")[["symbol", "fwd_ret"]].copy()
                for _, row in longs.iterrows():
                    rows.append({
                        "window": window,
                        "timestamp": timestamp,
                        "symbol": row["symbol"],
                        "side": "long",
                        "selected_ret": float(row["fwd_ret"]),
                    })
            if ns > 0:
                shorts = grp.nsmallest(ns, "pred")[["symbol", "fwd_ret"]].copy()
                for _, row in shorts.iterrows():
                    rows.append({
                        "window": window,
                        "timestamp": timestamp,
                        "symbol": row["symbol"],
                        "side": "short",
                        "selected_ret": float(-row["fwd_ret"]),
                    })

        blocked[window] = set()

    quality_df = pd.DataFrame(rows)
    quality_summary = (
        quality_df.groupby(["window", "symbol", "side"], as_index=False)
        .agg(mean_selected_ret=("selected_ret", "mean"), appearances=("timestamp", "count"))
    )
    quality_summary.to_csv(QUALITY_PATH, index=False)

    long_quality = quality_summary[quality_summary["side"] == "long"].copy()
    for window in ["W1", "W2", "W3"]:
        sub = long_quality[long_quality["window"] == window].copy()
        blocked[window] = set(
            sub[(sub["appearances"] >= 6) & (sub["mean_selected_ret"] < 0)]["symbol"].tolist()
        )

    return quality_summary, blocked


def filter_candidates(group: pd.DataFrame, blocked_longs: set[str], rule_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base_universe = group.copy()

    if rule_name == "baseline":
        return base_universe, base_universe
    if rule_name == "liq30_combo35":
        filtered = base_universe[base_universe["liq_combo_30d"] >= 0.35].copy()
        return filtered, filtered
    if rule_name == "liq60_combo35":
        filtered = base_universe[base_universe["liq_combo_60d"] >= 0.35].copy()
        return filtered, filtered
    if rule_name == "quality_long_only":
        longs = base_universe[~base_universe["symbol"].isin(blocked_longs)].copy()
        return longs, base_universe
    if rule_name == "liq30_plus_quality":
        filtered = base_universe[base_universe["liq_combo_30d"] >= 0.35].copy()
        longs = filtered[~filtered["symbol"].isin(blocked_longs)].copy()
        return longs, filtered
    raise ValueError(f"Unknown rule: {rule_name}")


def simulate_dynamic_universe(preds: pd.DataFrame, regime_df: pd.DataFrame, metrics_df: pd.DataFrame, blocked_longs_by_window: Dict[str, set[str]], rule_name: str) -> pd.DataFrame:
    merged = preds.merge(metrics_df, on=["timestamp", "symbol"], how="left")
    grouped = {timestamp: group.copy() for timestamp, group in merged.groupby("timestamp")}
    rebal_timestamps = sorted(merged["timestamp"].unique())[::BASE_CFG["rebal_hours"]]

    prev_longs: set[str] = set()
    prev_shorts: set[str] = set()
    prev_preds: dict[str, float] = {}
    rows: List[Dict[str, object]] = []

    for timestamp in rebal_timestamps:
        if timestamp not in grouped or timestamp not in regime_df.index:
            continue
        trend_strength = regime_df.loc[timestamp].get("trend_strength", 0.0)
        if trend_strength > BASE_CFG["trend_cutoff"]:
            continue

        group = grouped[timestamp].copy()
        if group.empty:
            continue
        window = str(group["window"].iloc[0])
        blocked_longs = blocked_longs_by_window.get(window, set())

        long_candidates, short_candidates = filter_candidates(group, blocked_longs, rule_name)
        long_candidates = long_candidates.copy()
        short_candidates = short_candidates.copy()

        if BASE_CFG["ema_alpha"] is not None and BASE_CFG["ema_alpha"] < 1.0:
            for index, row in group.iterrows():
                symbol = row["symbol"]
                raw_pred = row["pred"]
                smoothed = BASE_CFG["ema_alpha"] * raw_pred + (1 - BASE_CFG["ema_alpha"]) * prev_preds.get(symbol, raw_pred)
                prev_preds[symbol] = smoothed
                group.at[index, "pred"] = smoothed
            long_candidates = group[group["symbol"].isin(long_candidates["symbol"])].copy()
            short_candidates = group[group["symbol"].isin(short_candidates["symbol"])].copy()

        n_total = len(group)
        n_long = min(BASE_CFG["n_long"], n_total // 3, len(long_candidates))
        n_short = min(BASE_CFG["n_short"], n_total // 3, len(short_candidates))
        if n_long == 0 and n_short == 0:
            continue

        exposure = 1.0
        if BASE_CFG["dyn_threshold"] is not None and trend_strength > BASE_CFG["dyn_threshold"]:
            exposure = max(0.1, 1.0 - (trend_strength - BASE_CFG["dyn_threshold"]) /
                           (BASE_CFG["trend_cutoff"] - BASE_CFG["dyn_threshold"] + 1e-10) * 0.5)

        if BASE_CFG["hysteresis"] > 0 and (prev_longs or prev_shorts):
            new_longs = set()
            new_shorts = set()

            long_rank = long_candidates.sort_values("pred", ascending=False).reset_index(drop=True)
            long_rank["candidate_rank"] = long_rank.index + 1
            short_rank = short_candidates.sort_values("pred", ascending=True).reset_index(drop=True)
            short_rank["candidate_rank"] = short_rank.index + 1

            for _, row in long_rank.iterrows():
                if row["symbol"] in prev_longs and row["candidate_rank"] <= n_long + BASE_CFG["hysteresis"]:
                    new_longs.add(row["symbol"])
            for _, row in short_rank.iterrows():
                if row["symbol"] in prev_shorts and row["candidate_rank"] <= n_short + BASE_CFG["hysteresis"]:
                    new_shorts.add(row["symbol"])

            remain_long = n_long - len(new_longs)
            remain_short = n_short - len(new_shorts)
            if remain_long > 0:
                candidates = long_candidates[~long_candidates["symbol"].isin(new_longs | new_shorts)].sort_values("pred", ascending=False)
                new_longs |= set(candidates.head(remain_long)["symbol"].tolist())
            if remain_short > 0:
                candidates = short_candidates[~short_candidates["symbol"].isin(new_longs | new_shorts)].sort_values("pred", ascending=True)
                new_shorts |= set(candidates.head(remain_short)["symbol"].tolist())
        else:
            new_longs = set(long_candidates.nlargest(n_long, "pred")["symbol"].tolist()) if n_long > 0 else set()
            new_shorts = set(short_candidates.nsmallest(n_short, "pred")["symbol"].tolist()) if n_short > 0 else set()

        new_opened = (new_longs - prev_longs) | (new_shorts - prev_shorts)
        closed = (prev_longs - new_longs) | (prev_shorts - new_shorts)
        turnover_count = len(new_opened) + len(closed)
        total_positions = len(new_longs) + len(new_shorts)
        avg_weight = (1.0 / total_positions) if total_positions > 0 else 0.0
        turnover_cost = turnover_count * COST_ONE_WAY * avg_weight if total_positions > 0 else 0.0
        holding_cost = FUNDING_PER_12H * (BASE_CFG["rebal_hours"] / 12)
        total_cost = turnover_cost + holding_cost

        longs = group[group["symbol"].isin(new_longs)].copy()
        shorts = group[group["symbol"].isin(new_shorts)].copy()
        long_ret = float(longs["fwd_ret"].mean()) if len(longs) else 0.0
        short_ret = float(shorts["fwd_ret"].mean()) if len(shorts) else 0.0

        if n_long > 0 and n_short > 0:
            portfolio_ret = 0.5 * long_ret - 0.5 * short_ret
        elif n_short > 0:
            portfolio_ret = -short_ret
        else:
            portfolio_ret = long_ret
        portfolio_ret *= exposure
        portfolio_ret -= total_cost

        rows.append({
            "timestamp": timestamp,
            "window": window,
            "portfolio_ret": portfolio_ret,
            "gross_ret": portfolio_ret + total_cost,
            "turnover": turnover_count,
            "cost": total_cost,
            "n_long": len(new_longs),
            "n_short": len(new_shorts),
            "filtered_longs": len(blocked_longs),
            "long_pool": len(long_candidates),
            "short_pool": len(short_candidates),
        })

        prev_longs = new_longs
        prev_shorts = new_shorts

    return pd.DataFrame(rows)


def evaluate_rule(preds: pd.DataFrame, regime_df: pd.DataFrame, metrics_df: pd.DataFrame, blocked_longs_by_window: Dict[str, set[str]], rule_name: str) -> Dict[str, object]:
    row: Dict[str, object] = {"rule": rule_name}
    all_ports = []
    for window in ["W1", "W2", "W3", "ALL"]:
        subset = preds if window == "ALL" else preds[preds["window"] == window].copy()
        port = simulate_dynamic_universe(subset, regime_df, metrics_df, blocked_longs_by_window, rule_name)
        all_ports.append(port)
        metric = eval_with_costs(port, f"{rule_name}_{window}")
        row[f"{window}_sh"] = metric.get("sharpe", 0.0)
        row[f"{window}_sh_gross"] = metric.get("sharpe_gross", 0.0)
        row[f"{window}_eq"] = metric.get("equity", 0.0)
        row[f"{window}_cost"] = metric.get("total_cost_pct", 0.0)
        row[f"{window}_turn"] = metric.get("avg_turnover", 0.0)
        row[f"{window}_dd"] = metric.get("max_dd_pct", 0.0)

    port_all = pd.concat([port for port in all_ports if not port.empty], ignore_index=True) if all_ports else pd.DataFrame()
    row["avg_long_pool"] = float(port_all["long_pool"].mean()) if not port_all.empty else 0.0
    row["avg_short_pool"] = float(port_all["short_pool"].mean()) if not port_all.empty else 0.0
    row["avg_filtered_longs"] = float(port_all["filtered_longs"].mean()) if not port_all.empty else 0.0
    return row


def build_toxic_coin_report(metrics_df: pd.DataFrame, blocked_longs_by_window: Dict[str, set[str]]) -> pd.DataFrame:
    rows = []
    for window in ["W1", "W2", "W3"]:
        blocked = blocked_longs_by_window.get(window, set())
        for symbol in TOXIC_LONGS_W2:
            rows.append({
                "window": window,
                "symbol": symbol,
                "blocked_by_quality": symbol in blocked,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="r42_candidate")
    args = parser.parse_args()

    print("=" * 80)
    print("R44 — DYNAMIC UNIVERSE / COIN QUALITY FILTER")
    print("=" * 80)
    print(f"Feature set: {args.feature_set}")
    print(BASE_CFG)

    print("\n[1] Loading research frame...")
    df, regime_df = load_research_frame()
    df, _ = add_r35_features(df)
    regime_df = compute_regime_extended(df).sort_index()
    metrics_df = build_universe_metrics(df)
    print(f"  Data: {len(df):,} rows, {len(df.columns)} cols")

    feats, no_rank = build_feature_set(FEATURE_SETS[args.feature_set])

    print("\n[2] Training test predictions...")
    test_preds = train_ensemble(
        df,
        feats,
        WINDOWS,
        l2=1.0,
        rolling=False,
        label=f"r44_{args.feature_set}_test",
        cs_rank_exclude=no_rank,
    )
    if test_preds is None or test_preds.empty:
        raise RuntimeError("No test predictions for R44")

    print("\n[3] Training validation proxy predictions for quality filter...")
    val_preds = train_ensemble(
        df,
        feats,
        build_validation_windows(),
        l2=1.0,
        rolling=False,
        label=f"r44_{args.feature_set}_val",
        cs_rank_exclude=no_rank,
    )
    if val_preds is None or val_preds.empty:
        raise RuntimeError("No validation predictions for R44")

    quality_df, blocked_longs_by_window = derive_long_quality(
        val_preds,
        n_long=BASE_CFG["n_long"],
        n_short=BASE_CFG["n_short"],
        rebal_hours=BASE_CFG["rebal_hours"],
    )
    toxic_report = build_toxic_coin_report(metrics_df, blocked_longs_by_window)
    toxic_report.to_csv(TOXIC_CHECK_PATH, index=False)

    print("\n[4] Quality filter snapshot:")
    for window in ["W1", "W2", "W3"]:
        blocked = sorted(blocked_longs_by_window.get(window, set()))
        print(f"  {window}: blocked_longs={blocked}")

    rules = [
        "baseline",
        "liq30_combo35",
        "liq60_combo35",
        "quality_long_only",
        "liq30_plus_quality",
    ]

    print("\n[5] Evaluating dynamic universe rules...")
    rows = []
    for rule in rules:
        print(f"  {rule}...")
        row = evaluate_rule(test_preds, regime_df, metrics_df, blocked_longs_by_window, rule)
        rows.append(row)
        print(
            f"    -> W2={row['W2_sh']:.2f}, W3={row['W3_sh']:.2f}, ALL={row['ALL_sh']:.2f}, "
            f"avg_long_pool={row['avg_long_pool']:.1f}"
        )

    summary_df = pd.DataFrame(rows).sort_values(["ALL_sh", "W2_sh", "W3_sh"], ascending=False).reset_index(drop=True)
    baseline_all = float(summary_df.loc[summary_df["rule"] == "baseline", "ALL_sh"].iloc[0])
    summary_df["delta_all_vs_baseline"] = summary_df["ALL_sh"] - baseline_all
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print("\n[6] Best rules")
    print(summary_df[["rule", "W2_sh", "W3_sh", "ALL_sh", "ALL_cost", "ALL_turn", "avg_long_pool", "avg_filtered_longs", "delta_all_vs_baseline"]].to_string(index=False))

    print("\n[7] Toxic coin check")
    print(toxic_report.to_string(index=False))

    print("\n[8] Saved artifacts")
    print(f"  Summary CSV: {SUMMARY_PATH.name}")
    print(f"  Quality CSV: {QUALITY_PATH.name}")
    print(f"  Toxic check CSV: {TOXIC_CHECK_PATH.name}")


if __name__ == "__main__":
    main()