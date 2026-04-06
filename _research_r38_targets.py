#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R38 — target engineering rerun on current baseline.

Outputs:
  - results_r38.log
  - results_r38_target_summary.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from _research_round7 import SYM_35
from _ic_scanner import build_features_minimal, load_derivatives, load_ohlcv
from _research_r22_models import add_new_features, build_r19_features
from _research_r27_horizons import add_relative_returns, blend_predictions, make_temporal_weight_fn, train_lgb_horizon, train_xgb_horizon
from _research_r30b_fixed import add_extra_features_clean, compute_regime_extended, eval_per_window
from _research_r33_creative_features import FEAT_28, add_r33_features


BASE_DIR = Path(__file__).resolve().parent

CFG = {
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
    df = add_relative_returns(df)
    regime_df = compute_regime_extended(df)
    return df, regime_df


def summarize(label: str, results: Dict) -> List[Dict]:
    rows = []
    for window in ["W1", "W2", "W3", "ALL"]:
        metric = results.get(window, {})
        rows.append({
            "config": label,
            "window": window,
            "sharpe": metric.get("sharpe", np.nan),
            "sharpe_gross": metric.get("sharpe_gross", np.nan),
            "equity": metric.get("equity", np.nan),
            "cost_pct": metric.get("total_cost_pct", np.nan),
            "avg_turnover": metric.get("avg_turnover", np.nan),
        })
    return rows


def main() -> None:
    summary_path = BASE_DIR / "results_r38_target_summary.csv"

    print("=" * 80)
    print("R38 — TARGET ENGINEERING")
    print("=" * 80)

    print("\n[1] Loading research frame...")
    df, regime_df = load_research_frame()
    regime_df = regime_df.sort_index()
    print(f"  Data: {len(df):,} rows, {len(df.columns)} cols")

    print("\n[2] Building target variants...")
    thresholds = [0.0, 0.005, 0.01, 0.015, 0.02]
    for threshold in thresholds[1:]:
        name = f"fwd_ret_12h_thr_{int(threshold * 10000):03d}bps"
        df[name] = df["fwd_ret_12h"] - threshold

    variants = [
        ("baseline_p0", "fwd_ret_12h", None),
        ("p_gt_005", "fwd_ret_12h_thr_050bps", None),
        ("p_gt_010", "fwd_ret_12h_thr_100bps", None),
        ("p_gt_015", "fwd_ret_12h_thr_150bps", None),
        ("p_gt_020", "fwd_ret_12h_thr_200bps", None),
        ("excess_vs_btc", "fwd_ret_12h_vs_btc", None),
        ("decay_90d", "fwd_ret_12h", make_temporal_weight_fn(90)),
        ("decay_180d", "fwd_ret_12h", make_temporal_weight_fn(180)),
    ]

    print("\n[3] Training target variants...")
    rows = []
    for label, target_col, weight_fn in variants:
        print(f"  {label}...")
        preds_lgb = train_lgb_horizon(df, FEAT_28, target_col=target_col, eval_col="fwd_ret_12h", sample_weight_fn=weight_fn)
        preds_xgb = train_xgb_horizon(df, FEAT_28, target_col=target_col, eval_col="fwd_ret_12h", sample_weight_fn=weight_fn)
        if preds_lgb is None or preds_xgb is None:
            continue
        preds = blend_predictions([preds_lgb, preds_xgb])
        results = eval_per_window(preds, regime_df, CFG, label=label)
        rows.extend(summarize(label, results))

    result_df = pd.DataFrame(rows).sort_values(["window", "sharpe"], ascending=[True, False])
    result_df.to_csv(summary_path, index=False)

    print("\n[4] Best target variants")
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