#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R39.2 / R39.5 pilot — stablecoin supply regime features.

Goal:
  - engineer conservative stablecoin market features from stablecoin_supply.parquet
  - keep them as market-level signals (no cross-sectional ranking)
  - run pilot walk-forward uplift tests on top of A_28f

Outputs:
  - results_r39_stablecoin.log
  - results_r39_stablecoin_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from _research_round7 import SYM_35, WINDOWS
from _ic_scanner import build_features_minimal, load_derivatives, load_ohlcv
from _research_r22_models import add_new_features, build_r19_features
from _research_r30b_fixed import (
    add_extra_features_clean,
    compute_regime_extended,
    eval_per_window,
    train_ensemble,
)
from _research_r33_creative_features import FEAT_28, add_r33_features


BASE_DIR = Path(__file__).resolve().parent
SENT_DIR = BASE_DIR / "data" / "sentiment"

CFG = {
    "n_long": 6,
    "n_short": 3,
    "rebal_hours": 12,
    "trend_cutoff": 0.9,
    "dyn_threshold": 0.7,
    "ema_alpha": 0.5,
    "hysteresis": 3,
}

BASELINE_SHARPES = {
    "W1": -0.69,
    "W2": -0.98,
    "W3": 2.88,
    "ALL": 0.47,
}

FEATURE_BUNDLES = {
    "stable_flow4": [
        "stable_total_supply_chg7d",
        "stable_total_supply_chg30d",
        "stable_supply_accel",
        "stable_usdt_vs_usdc_chg7d",
    ],
    "stable_regime6": [
        "stable_total_supply_chg7d",
        "stable_total_supply_chg30d",
        "stable_supply_accel",
        "stable_usdt_vs_usdc_chg7d",
        "stable_total_supply_chg7d_z",
        "stable_usdt_dom_z",
    ],
}


def rolling_zscore(series: pd.Series, window: int = 90, min_periods: int = 30) -> pd.Series:
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return (series - mean) / (std + 1e-10)


def load_research_frame() -> pd.DataFrame:
    ohlcv = load_ohlcv()
    ohlcv = ohlcv[ohlcv["symbol"].isin(SYM_35)]
    derivs = load_derivatives()
    df = build_features_minimal(ohlcv, derivs)
    df = build_r19_features(df)
    df, _ = add_new_features(df)
    df = df[df["symbol"].isin(SYM_35)].copy()
    df = add_extra_features_clean(df)
    df = add_r33_features(df)
    return df


def add_stablecoin_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    stable_path = SENT_DIR / "stablecoin_supply.parquet"
    stable = pd.read_parquet(stable_path).copy()
    stable["date"] = pd.to_datetime(stable["date"], utc=True)
    stable = stable.sort_values("date").drop_duplicates("date")

    stable["stable_total_supply_chg7d"] = stable["total_stable_supply_chg7d"]
    stable["stable_total_supply_chg30d"] = stable["total_stable_supply_chg30d"]
    stable["stable_supply_accel"] = (
        stable["stable_total_supply_chg7d"] - stable["stable_total_supply_chg30d"] / 4.0
    )
    stable["stable_usdt_dom"] = stable["USDT_supply"] / (stable["total_stable_supply"] + 1e-10)
    stable["stable_usdt_vs_usdc_chg7d"] = stable["USDT_supply_chg7d"] - stable["USDC_supply_chg7d"]
    stable["stable_total_supply_chg7d_z"] = rolling_zscore(stable["stable_total_supply_chg7d"])
    stable["stable_total_supply_chg30d_z"] = rolling_zscore(stable["stable_total_supply_chg30d"])
    stable["stable_usdt_dom_z"] = rolling_zscore(stable["stable_usdt_dom"])

    feature_cols = sorted({feature for bundle in FEATURE_BUNDLES.values() for feature in bundle})
    stable["timestamp"] = stable["date"] + pd.Timedelta(days=1)
    stable = stable[["timestamp"] + feature_cols].replace([np.inf, -np.inf], np.nan)
    stable = stable.set_index("timestamp").resample("1h").ffill().reset_index()

    merged = df.merge(stable, on="timestamp", how="left").sort_values(["timestamp", "symbol"]).copy()
    for feature in feature_cols:
        merged[feature] = merged[feature].ffill().fillna(0.0)

    return merged, feature_cols


def summarize_results(label: str, bundle_name: str, bundle_feats: List[str], results: Dict) -> pd.DataFrame:
    rows = []
    for window in ["W1", "W2", "W3", "ALL"]:
        metric = results[window]
        baseline = BASELINE_SHARPES[window]
        rows.append({
            "config": label,
            "bundle": bundle_name,
            "window": window,
            "n_new_features": len(bundle_feats),
            "sharpe": metric.get("sharpe", np.nan),
            "baseline_sharpe": baseline,
            "delta_vs_baseline": metric.get("sharpe", np.nan) - baseline,
            "sharpe_gross": metric.get("sharpe_gross", np.nan),
            "equity": metric.get("equity", np.nan),
            "cost_pct": metric.get("total_cost_pct", np.nan),
            "avg_turnover": metric.get("avg_turnover", np.nan),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", choices=["all", *sorted(FEATURE_BUNDLES)], default="all")
    args = parser.parse_args()

    summary_path = BASE_DIR / "results_r39_stablecoin_summary.csv"
    bundles = FEATURE_BUNDLES if args.bundle == "all" else {args.bundle: FEATURE_BUNDLES[args.bundle]}

    print("=" * 80)
    print("R39.2 / R39.5 PILOT — STABLECOIN REGIME FEATURES")
    print("=" * 80)
    print("Conservative assumptions:")
    print("- stablecoin daily features are shifted by +1 day before merge")
    print("- market-level features are excluded from CS ranking so they survive model input")

    print("\n[1] Loading research frame...")
    df = load_research_frame()
    df, stable_features = add_stablecoin_features(df)
    regime_df = compute_regime_extended(df).sort_index()
    print(f"  Data: {len(df):,} rows, {len(df.columns)} cols")
    print(f"  Stable features: {stable_features}")

    all_results = []
    for bundle_name, bundle_feats in bundles.items():
        label = f"A_28f+{bundle_name}"
        print(f"\n[2] Training {label}...")
        preds = train_ensemble(
            df,
            FEAT_28 + bundle_feats,
            WINDOWS,
            l2=1.0,
            rolling=False,
            label=label,
            cs_rank_exclude=bundle_feats,
        )
        if preds is None or preds.empty:
            print("  No predictions produced; skipping")
            continue

        print(f"\n[3] Evaluating {label}...")
        results = eval_per_window(preds, regime_df, CFG, label=label)
        summary = summarize_results(label, bundle_name, bundle_feats, results)
        all_results.append(summary)
        print(summary[[
            "window",
            "sharpe",
            "baseline_sharpe",
            "delta_vs_baseline",
            "cost_pct",
            "avg_turnover",
        ]].to_string(index=False))

    if not all_results:
        raise RuntimeError("No successful stablecoin experiments")

    result_df = pd.concat(all_results, ignore_index=True)
    result_df.to_csv(summary_path, index=False)

    print("\n[4] Best uplift candidates")
    w2_best = result_df[result_df["window"] == "W2"].sort_values(
        ["sharpe", "delta_vs_baseline"], ascending=False
    )
    all_best = result_df[result_df["window"] == "ALL"].sort_values(
        ["sharpe", "delta_vs_baseline"], ascending=False
    )
    if not w2_best.empty:
        print("  W2 best:")
        print(w2_best.head(1).to_string(index=False))
    if not all_best.empty:
        print("\n  ALL best:")
        print(all_best.head(1).to_string(index=False))

    print("\n[5] Saved artifacts")
    print(f"  Summary CSV: {summary_path.name}")


if __name__ == "__main__":
    main()