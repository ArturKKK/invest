#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R42 — ablation inside the R35a bundle.

Goal:
  find the minimal subset of R35a features that repairs W2 without
  breaking W3 under the same canonical execution setup used in R41.

Evaluated:
  - baseline A_28f
  - every single feature from R35a
  - every pair from R35a
  - every triple from R35a
  - full R35a bundle as reference

Outputs:
  - results_r42.log
  - results_r42_summary.csv
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from _research_round7 import WINDOWS
from _research_r30b_fixed import compute_regime_extended, eval_with_costs, simulate_with_costs, train_ensemble
from _research_r33_creative_features import FEAT_28
from _research_r35_new_features import GROUPS, MARKET_LEVEL_FEATURES, add_r35_features, load_research_frame


BASE_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = BASE_DIR / "results_r42_summary.csv"

CANONICAL_EXEC_CFG = {
    "n_long": 6,
    "n_short": 3,
    "rebal_hours": 12,
    "trend_cutoff": 0.9,
    "dyn_threshold": 0.7,
    "ema_alpha": 0.5,
    "hysteresis": 3,
}

R35A_FEATURES = GROUPS["r35a_cs_second_order"]

SHORT_NAMES = {
    "ret_dispersion_12h": "dispersion",
    "cs_rank_ma_5": "rankma",
    "oi_chg_12h_cs": "oi_cs",
    "taker_cvd_12h_cs": "taker_cs",
    "cum_funding_24h_cs": "funding_cs",
}


def build_feature_set(base_features: Sequence[str], extra_features: Sequence[str]) -> Tuple[List[str], List[str]]:
    feats = list(base_features)
    for feature in extra_features:
        if feature not in feats:
            feats.append(feature)
    no_rank = [feature for feature in extra_features if feature in MARKET_LEVEL_FEATURES]
    return feats, no_rank


def subset_label(features: Sequence[str]) -> str:
    if not features:
        return "A_28f"
    names = [SHORT_NAMES[feature] for feature in features]
    return "A_28f+" + "+".join(names)


def iter_feature_subsets(features: Sequence[str]) -> Iterable[Tuple[int, Tuple[str, ...]]]:
    yield 0, tuple()
    for size in [1, 2, 3]:
        for subset in combinations(features, size):
            yield size, subset
    yield len(features), tuple(features)


def evaluate_predictions(preds: pd.DataFrame, regime_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for window in ["W1", "W2", "W3", "ALL"]:
        subset = preds if window == "ALL" else preds[preds["window"] == window].copy()
        port = simulate_with_costs(subset, regime_df, CANONICAL_EXEC_CFG)
        out[window] = eval_with_costs(port, window)
    return out


def summarize_result(label: str, subset_size: int, subset: Sequence[str], results: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    row: Dict[str, object] = {
        "config": label,
        "subset_size": subset_size,
        "features": "|".join(subset),
        "feature_count": len(FEAT_28) + subset_size,
        "contains_dispersion": "ret_dispersion_12h" in subset,
    }
    for window in ["W1", "W2", "W3", "ALL"]:
        metric = results[window]
        row[f"{window}_sh"] = metric.get("sharpe", 0.0)
        row[f"{window}_sh_gross"] = metric.get("sharpe_gross", 0.0)
        row[f"{window}_eq"] = metric.get("equity", 0.0)
        row[f"{window}_cost"] = metric.get("total_cost_pct", 0.0)
        row[f"{window}_turn"] = metric.get("avg_turnover", 0.0)
        row[f"{window}_dd"] = metric.get("max_dd_pct", 0.0)

    row["passes_target"] = bool(row["W2_sh"] > 0 and row["W3_sh"] >= 2.5)
    row["passes_strict"] = bool(row["W2_sh"] > 1.0 and row["W3_sh"] >= 2.5 and row["ALL_sh"] >= 0.74)
    return row


def print_bucket(title: str, frame: pd.DataFrame, limit: int = 5) -> None:
    if frame.empty:
        return
    print(f"  {title}:")
    cols = ["config", "W2_sh", "W3_sh", "ALL_sh", "ALL_cost", "ALL_turn"]
    print(frame.head(limit)[cols].to_string(index=False))


def main() -> None:
    print("=" * 80)
    print("R42 — R35A ABLATION")
    print("=" * 80)
    print("Canonical execution config:")
    print(CANONICAL_EXEC_CFG)
    print(f"R35a features: {R35A_FEATURES}")

    print("\n[1] Loading research frame...")
    df, regime_df = load_research_frame()
    df, _ = add_r35_features(df)
    regime_df = compute_regime_extended(df).sort_index()
    print(f"  Data: {len(df):,} rows, {len(df.columns)} cols")

    rows: List[Dict[str, object]] = []

    print("\n[2] Training ablation matrix sequentially...")
    for subset_size, subset in iter_feature_subsets(R35A_FEATURES):
        label = subset_label(subset)
        feats, no_rank = build_feature_set(FEAT_28, subset)
        print(f"\n  Training {label} ({len(feats)}f)...")
        preds = train_ensemble(
            df,
            feats,
            WINDOWS,
            l2=1.0,
            rolling=False,
            label=label,
            cs_rank_exclude=no_rank,
        )
        if preds is None or preds.empty:
            raise RuntimeError(f"No predictions for {label}")

        results = evaluate_predictions(preds, regime_df)
        row = summarize_result(label, subset_size, subset, results)
        rows.append(row)

        print(
            "    -> "
            f"W2={row['W2_sh']:.2f}, W3={row['W3_sh']:.2f}, ALL={row['ALL_sh']:.2f}, "
            f"cost={row['ALL_cost']:.2f}%, turn={row['ALL_turn']:.1f}"
        )

    summary_df = pd.DataFrame(rows)
    baseline_all = float(summary_df.loc[summary_df["config"] == "A_28f", "ALL_sh"].iloc[0])
    summary_df["delta_all_vs_baseline"] = summary_df["ALL_sh"] - baseline_all

    summary_df = summary_df.sort_values(
        ["passes_target", "passes_strict", "ALL_sh", "W2_sh", "W3_sh"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print("\n[3] Best buckets")
    singles = summary_df[summary_df["subset_size"] == 1]
    pairs = summary_df[summary_df["subset_size"] == 2]
    triples = summary_df[summary_df["subset_size"] == 3]
    strict = summary_df[summary_df["passes_strict"]]
    target = summary_df[summary_df["passes_target"]]
    dispersion = summary_df[(summary_df["contains_dispersion"]) & (summary_df["subset_size"].isin([2, 3]))]

    print_bucket("Best singles", singles)
    print_bucket("Best pairs", pairs)
    print_bucket("Best triples", triples)
    print_bucket("Target passers", target)
    print_bucket("Strict passers", strict)
    print_bucket("Dispersion-led combos", dispersion)

    print("\n[4] Saved artifacts")
    print(f"  Summary CSV: {SUMMARY_PATH.name}")


if __name__ == "__main__":
    main()