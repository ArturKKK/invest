#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R41 — consolidation run.

Combine the post-R39 winners under one canonical execution setup:
  - A_28f
  - A_28f + R35a
  - D_30f
  - D_30f + R35a
  - 50/50 blend(A_28f+R35a, D_30f)

Each config is evaluated with and without the R37 liquidity floor (liq70)
using the canonical simulator from _research_r30b_fixed.py.

Outputs:
  - results_r41.log
  - results_r41_summary.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from _research_round7 import WINDOWS
from _research_r30b_fixed import (
    compute_regime_extended,
    eval_with_costs,
    simulate_with_costs,
    train_ensemble,
)
from _research_r33_creative_features import FEAT_28, FEAT_30
from _research_r35_new_features import GROUPS, MARKET_LEVEL_FEATURES, add_r35_features, load_research_frame
from _research_r37_cost_aware import add_liquidity_rank


BASE_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = BASE_DIR / "results_r41_summary.csv"

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
LIQ70 = 0.70


def build_feature_set(base_features: List[str], extra_features: List[str]) -> Tuple[List[str], List[str]]:
    feats = list(base_features)
    for feature in extra_features:
        if feature not in feats:
            feats.append(feature)
    no_rank = [feature for feature in extra_features if feature in MARKET_LEVEL_FEATURES]
    return feats, no_rank


def average_blend(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    merged = left.rename(columns={"pred": "pred_left"}).merge(
        right[["timestamp", "symbol", "window", "pred"]].rename(columns={"pred": "pred_right"}),
        on=["timestamp", "symbol", "window"],
        how="inner",
    )
    merged["pred"] = 0.5 * (merged["pred_left"] + merged["pred_right"])
    return merged[["timestamp", "symbol", "pred", "fwd_ret", "window"]].copy()


def evaluate_matrix(
    preds: pd.DataFrame,
    regime_df: pd.DataFrame,
    liquidity_df: pd.DataFrame,
    label: str,
    liquidity_floor: float | None,
) -> List[Dict[str, object]]:
    merged = preds.merge(liquidity_df, on=["timestamp", "symbol"], how="left")
    rows: List[Dict[str, object]] = []
    for window in ["W1", "W2", "W3", "ALL"]:
        subset = merged if window == "ALL" else merged[merged["window"] == window].copy()
        if liquidity_floor is not None:
            subset = subset[subset["liquidity_rank"] >= liquidity_floor].copy()
        port = simulate_with_costs(subset, regime_df, CANONICAL_EXEC_CFG)
        metric = eval_with_costs(port, f"{label}_{window}")
        rows.append({
            "config": label,
            "window": window,
            "liquidity_floor": liquidity_floor,
            "sharpe": metric.get("sharpe", np.nan),
            "sharpe_gross": metric.get("sharpe_gross", np.nan),
            "equity": metric.get("equity", np.nan),
            "cost_pct": metric.get("total_cost_pct", np.nan),
            "avg_turnover": metric.get("avg_turnover", np.nan),
            "max_dd_pct": metric.get("max_dd_pct", np.nan),
        })
    return rows


def add_baseline_deltas(summary_df: pd.DataFrame, baseline_name: str) -> pd.DataFrame:
    baseline = (
        summary_df[summary_df["config"] == baseline_name][["window", "sharpe"]]
        .rename(columns={"sharpe": "baseline_sharpe"})
    )
    out = summary_df.merge(baseline, on="window", how="left")
    out["delta_vs_baseline"] = out["sharpe"] - out["baseline_sharpe"]
    return out


def main() -> None:
    print("=" * 80)
    print("R41 — CONSOLIDATION RUN")
    print("=" * 80)
    print("Canonical execution config:")
    print(CANONICAL_EXEC_CFG)
    print(f"R35a features: {R35A_FEATURES}")

    print("\n[1] Loading research frame...")
    df, regime_df = load_research_frame()
    df, _ = add_r35_features(df)
    regime_df = compute_regime_extended(df).sort_index()
    liquidity_df = add_liquidity_rank(df)
    print(f"  Data: {len(df):,} rows, {len(df.columns)} cols")

    model_specs = []
    model_specs.append(("A_28f", FEAT_28, []))
    feats_r35a, no_rank_r35a = build_feature_set(FEAT_28, R35A_FEATURES)
    model_specs.append(("A_28f+r35a", feats_r35a, no_rank_r35a))
    model_specs.append(("D_30f", FEAT_30, []))
    feats_d30_r35a, no_rank_d30_r35a = build_feature_set(FEAT_30, R35A_FEATURES)
    model_specs.append(("D_30f+r35a", feats_d30_r35a, no_rank_d30_r35a))

    preds_map: Dict[str, pd.DataFrame] = {}
    print("\n[2] Training model matrix sequentially...")
    for label, feats, no_rank in model_specs:
        n_new = len([feature for feature in feats if feature not in FEAT_28])
        print(f"\n  Training {label} ({len(feats)}f, new_vs_28f={n_new})...")
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
        preds_map[label] = preds

    print("\n[3] Building simple blend...")
    preds_map["blend_r35a_d30f"] = average_blend(preds_map["A_28f+r35a"], preds_map["D_30f"])

    print("\n[4] Evaluating canonical matrix...")
    configs = [
        ("A_28f", preds_map["A_28f"], None),
        ("A_28f+liq70", preds_map["A_28f"], LIQ70),
        ("A_28f+r35a", preds_map["A_28f+r35a"], None),
        ("A_28f+r35a+liq70", preds_map["A_28f+r35a"], LIQ70),
        ("D_30f", preds_map["D_30f"], None),
        ("D_30f+liq70", preds_map["D_30f"], LIQ70),
        ("D_30f+r35a", preds_map["D_30f+r35a"], None),
        ("D_30f+r35a+liq70", preds_map["D_30f+r35a"], LIQ70),
        ("blend_r35a_d30f", preds_map["blend_r35a_d30f"], None),
        ("blend_r35a_d30f+liq70", preds_map["blend_r35a_d30f"], LIQ70),
    ]

    rows: List[Dict[str, object]] = []
    for label, preds, liquidity_floor in configs:
        print(f"  {label}...")
        rows.extend(evaluate_matrix(preds, regime_df, liquidity_df, label, liquidity_floor))

    summary_df = pd.DataFrame(rows)
    summary_df = add_baseline_deltas(summary_df, baseline_name="A_28f")
    summary_df = summary_df.sort_values(["window", "sharpe"], ascending=[True, False]).reset_index(drop=True)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print("\n[5] Best configs")
    for window in ["W2", "W3", "ALL"]:
        top = summary_df[summary_df["window"] == window].head(6)
        if top.empty:
            continue
        print(f"  {window}:")
        print(top[["config", "sharpe", "delta_vs_baseline", "cost_pct", "avg_turnover"]].to_string(index=False))

    print("\n[6] Saved artifacts")
    print(f"  Summary CSV: {SUMMARY_PATH.name}")


if __name__ == "__main__":
    main()