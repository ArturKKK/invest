#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R45 — calibrated soft gate / expert blending.

Goal:
  test whether a soft, validation-calibrated blend can keep the new R42
  candidate's ALL improvement while borrowing regime-specific uplift from
  stability and stable-flow experts.

Experts:
  - expert_r42 = FEAT_28 + {ret_dispersion_12h, cs_rank_ma_5}
  - expert_stability = FEAT_30
  - expert_stable_flow = FEAT_28 + stable_flow4

Calibration:
  - for each WF window, use ONLY that window's validation period
  - search soft-weight schedules on regime percentile ranks

Outputs:
  - results_r45.log
  - results_r45_soft_gate_summary.csv
  - results_r45_soft_gate_calibration.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from _research_round7 import WINDOWS
from _research_r30b_fixed import compute_regime_extended, eval_with_costs, simulate_with_costs, train_ensemble
from _research_r33_creative_features import FEAT_28, FEAT_30
from _research_r35_new_features import MARKET_LEVEL_FEATURES, add_r35_features, load_research_frame
from _research_r39_stablecoin_features import FEATURE_BUNDLES, add_stablecoin_features


BASE_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = BASE_DIR / "results_r45_soft_gate_summary.csv"
CAL_PATH = BASE_DIR / "results_r45_soft_gate_calibration.csv"

CFG = {
    "n_long": 6,
    "n_short": 3,
    "rebal_hours": 12,
    "trend_cutoff": 0.9,
    "dyn_threshold": 0.7,
    "ema_alpha": 0.5,
    "hysteresis": 3,
}

R42_CANDIDATE = ["ret_dispersion_12h", "cs_rank_ma_5"]
R42_NO_RANK = [feature for feature in R42_CANDIDATE if feature in MARKET_LEVEL_FEATURES]
STABLE_FLOW4 = FEATURE_BUNDLES["stable_flow4"]


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


def build_market_context(df: pd.DataFrame) -> pd.DataFrame:
    context = (
        df.groupby("timestamp")
        .agg(
            breadth_12h=("pct_coins_up_12h", "mean"),
            dispersion_12h=("ret_12h", "std"),
            market_funding_24h=("cum_funding_24h", "mean"),
            stable_supply_accel=("stable_supply_accel", "first"),
            stable_total_supply_chg7d_z=("stable_total_supply_chg7d_z", "first"),
        )
        .reset_index()
    )
    btc = df[df["symbol"] == "BTC/USDT"][["timestamp", "rvol_24h"]].drop_duplicates("timestamp")
    btc = btc.rename(columns={"rvol_24h": "btc_rvol_24h"})
    context = context.merge(btc, on="timestamp", how="left")
    return context.sort_values("timestamp").reset_index(drop=True)


def attach_alt_pred(base: pd.DataFrame, alt: pd.DataFrame, name: str) -> pd.DataFrame:
    return base.merge(
        alt[["timestamp", "symbol", "pred"]].rename(columns={"pred": name}),
        on=["timestamp", "symbol"],
        how="inner",
    )


def make_soft_blend(two_expert_df: pd.DataFrame, weight_col: str, alt_col: str) -> pd.DataFrame:
    out = two_expert_df.copy()
    out["pred"] = (1.0 - out[weight_col]) * out["pred"] + out[weight_col] * out[alt_col]
    return out[["timestamp", "symbol", "pred", "fwd_ret", "window"]]


def make_tri_blend(merged: pd.DataFrame, w_stability: pd.Series, w_flow: pd.Series) -> pd.DataFrame:
    out = merged.copy()
    w_stability = w_stability.clip(0.0, 1.0)
    w_flow = w_flow.clip(0.0, 1.0)
    total = (w_stability + w_flow).clip(upper=1.0)
    base_weight = 1.0 - total
    out["pred"] = (
        base_weight * out["pred"]
        + w_stability * out["pred_stability"]
        + w_flow * out["pred_flow"]
    )
    return out[["timestamp", "symbol", "pred", "fwd_ret", "window"]]


def percentile_score(history: pd.Series, values: pd.Series) -> pd.Series:
    hist = history.dropna()
    if hist.empty:
        return pd.Series(0.5, index=values.index)
    return values.apply(lambda x: float((hist <= x).mean()) if pd.notna(x) else 0.5)


def evaluate_preds(preds: pd.DataFrame, regime_df: pd.DataFrame, label: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for window in ["W1", "W2", "W3", "ALL"]:
        subset = preds if window == "ALL" else preds[preds["window"] == window].copy()
        port = simulate_with_costs(subset, regime_df, CFG)
        out[window] = eval_with_costs(port, f"{label}_{window}")
    return out


def summarize(label: str, results: Dict[str, Dict[str, float]]) -> List[Dict[str, object]]:
    rows = []
    for window in ["W1", "W2", "W3", "ALL"]:
        metric = results[window]
        rows.append({
            "config": label,
            "window": window,
            "sharpe": metric.get("sharpe", np.nan),
            "sharpe_gross": metric.get("sharpe_gross", np.nan),
            "equity": metric.get("equity", np.nan),
            "cost_pct": metric.get("total_cost_pct", np.nan),
            "avg_turnover": metric.get("avg_turnover", np.nan),
            "max_dd_pct": metric.get("max_dd_pct", np.nan),
        })
    return rows


def calibrate_two_expert(
    base_val: pd.DataFrame,
    alt_val: pd.DataFrame,
    context: pd.DataFrame,
    regime_df: pd.DataFrame,
    window: Dict[str, str],
    regime_col: str,
    alt_col_name: str,
    label: str,
) -> Tuple[Tuple[float, float], Dict[str, object]]:
    merged = attach_alt_pred(base_val, alt_val, alt_col_name).merge(context, on="timestamp", how="left")
    merged = merged[merged["window"] == window["name"]].copy()
    if merged.empty:
        return (0.0, 0.0), {"label": label, "window": window["name"], "val_sharpe": np.nan}

    test_start = pd.Timestamp(window["test_start"], tz=merged["timestamp"].dt.tz)
    history = context[context["timestamp"] < test_start][regime_col]
    merged["score"] = percentile_score(history, merged[regime_col])

    best_params = (0.0, 0.0)
    best_metric = {"sharpe": -np.inf}
    low_grid = [0.0, 0.1, 0.2, 0.3]
    high_grid = [0.5, 0.7, 0.9, 1.0]
    for w_low in low_grid:
        for w_high in high_grid:
            if w_high < w_low:
                continue
            merged["soft_w"] = w_low + (w_high - w_low) * merged["score"]
            preds = make_soft_blend(merged, "soft_w", alt_col_name)
            metric = evaluate_preds(preds, regime_df, f"{label}_val_{window['name']}")["ALL"]
            if metric.get("sharpe", -np.inf) > best_metric.get("sharpe", -np.inf):
                best_metric = metric
                best_params = (w_low, w_high)

    return best_params, {
        "label": label,
        "window": window["name"],
        "regime_col": regime_col,
        "w_low": best_params[0],
        "w_high": best_params[1],
        "val_sharpe": best_metric.get("sharpe", np.nan),
        "val_cost_pct": best_metric.get("total_cost_pct", np.nan),
    }


def build_two_expert_test_blend(
    base_test: pd.DataFrame,
    alt_test: pd.DataFrame,
    context: pd.DataFrame,
    window: Dict[str, str],
    regime_col: str,
    alt_col_name: str,
    params: Tuple[float, float],
) -> pd.DataFrame:
    merged = attach_alt_pred(base_test, alt_test, alt_col_name).merge(context, on="timestamp", how="left")
    merged = merged[merged["window"] == window["name"]].copy()
    test_start = pd.Timestamp(window["test_start"], tz=merged["timestamp"].dt.tz)
    history = context[context["timestamp"] < test_start][regime_col]
    merged["score"] = percentile_score(history, merged[regime_col])
    w_low, w_high = params
    merged["soft_w"] = w_low + (w_high - w_low) * merged["score"]
    return make_soft_blend(merged, "soft_w", alt_col_name)


def calibrate_tri_expert(
    base_val: pd.DataFrame,
    stability_val: pd.DataFrame,
    flow_val: pd.DataFrame,
    context: pd.DataFrame,
    regime_df: pd.DataFrame,
    window: Dict[str, str],
) -> Tuple[Tuple[float, float], Dict[str, object]]:
    merged = attach_alt_pred(base_val, stability_val, "pred_stability")
    merged = attach_alt_pred(merged, flow_val, "pred_flow")
    merged = merged.merge(context, on="timestamp", how="left")
    merged = merged[merged["window"] == window["name"]].copy()
    if merged.empty:
        return (0.0, 0.0), {"label": "soft_tri", "window": window["name"], "val_sharpe": np.nan}

    test_start = pd.Timestamp(window["test_start"], tz=merged["timestamp"].dt.tz)
    hist_rvol = context[context["timestamp"] < test_start]["btc_rvol_24h"]
    hist_stable = context[context["timestamp"] < test_start]["stable_supply_accel"]
    merged["rvol_score"] = percentile_score(hist_rvol, merged["btc_rvol_24h"])
    merged["stable_score"] = percentile_score(hist_stable, merged["stable_supply_accel"])

    best_params = (0.0, 0.0)
    best_metric = {"sharpe": -np.inf}
    grids = [0.3, 0.5, 0.7, 0.9]
    for a in grids:
        for b in grids:
            w_stability = a * merged["rvol_score"]
            w_flow = b * merged["stable_score"] * (1.0 - w_stability.clip(0.0, 1.0))
            preds = make_tri_blend(merged, w_stability, w_flow)
            metric = evaluate_preds(preds, regime_df, f"soft_tri_val_{window['name']}")["ALL"]
            if metric.get("sharpe", -np.inf) > best_metric.get("sharpe", -np.inf):
                best_metric = metric
                best_params = (a, b)

    return best_params, {
        "label": "soft_tri",
        "window": window["name"],
        "regime_col": "btc_rvol_24h+stable_supply_accel",
        "w_low": best_params[0],
        "w_high": best_params[1],
        "val_sharpe": best_metric.get("sharpe", np.nan),
        "val_cost_pct": best_metric.get("total_cost_pct", np.nan),
    }


def build_tri_test_blend(
    base_test: pd.DataFrame,
    stability_test: pd.DataFrame,
    flow_test: pd.DataFrame,
    context: pd.DataFrame,
    window: Dict[str, str],
    params: Tuple[float, float],
) -> pd.DataFrame:
    merged = attach_alt_pred(base_test, stability_test, "pred_stability")
    merged = attach_alt_pred(merged, flow_test, "pred_flow")
    merged = merged.merge(context, on="timestamp", how="left")
    merged = merged[merged["window"] == window["name"]].copy()
    test_start = pd.Timestamp(window["test_start"], tz=merged["timestamp"].dt.tz)
    hist_rvol = context[context["timestamp"] < test_start]["btc_rvol_24h"]
    hist_stable = context[context["timestamp"] < test_start]["stable_supply_accel"]
    merged["rvol_score"] = percentile_score(hist_rvol, merged["btc_rvol_24h"])
    merged["stable_score"] = percentile_score(hist_stable, merged["stable_supply_accel"])
    a, b = params
    w_stability = a * merged["rvol_score"]
    w_flow = b * merged["stable_score"] * (1.0 - w_stability.clip(0.0, 1.0))
    return make_tri_blend(merged, w_stability, w_flow)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", choices=["r42_candidate"], default="r42_candidate")
    args = parser.parse_args()

    print("=" * 80)
    print("R45 — CALIBRATED SOFT GATE")
    print("=" * 80)
    print(f"Base expert: {args.base}")
    print(CFG)

    print("\n[1] Loading research frame...")
    df, _ = load_research_frame()
    df, _ = add_r35_features(df)
    df, _ = add_stablecoin_features(df)
    regime_df = compute_regime_extended(df).sort_index()
    context = build_market_context(df)
    print(f"  Data: {len(df):,} rows, {len(df.columns)} cols")

    print("\n[2] Training test experts...")
    base_test = train_ensemble(
        df,
        FEAT_28 + R42_CANDIDATE,
        WINDOWS,
        l2=1.0,
        rolling=False,
        label="expert_r42",
        cs_rank_exclude=R42_NO_RANK,
    )
    stability_test = train_ensemble(df, FEAT_30, WINDOWS, l2=1.0, rolling=False, label="expert_stability")
    flow_test = train_ensemble(
        df,
        FEAT_28 + STABLE_FLOW4,
        WINDOWS,
        l2=1.0,
        rolling=False,
        label="expert_stable_flow",
        cs_rank_exclude=STABLE_FLOW4,
    )

    print("\n[3] Training validation experts for calibration...")
    val_windows = build_validation_windows()
    base_val = train_ensemble(
        df,
        FEAT_28 + R42_CANDIDATE,
        val_windows,
        l2=1.0,
        rolling=False,
        label="expert_r42_val",
        cs_rank_exclude=R42_NO_RANK,
    )
    stability_val = train_ensemble(df, FEAT_30, val_windows, l2=1.0, rolling=False, label="expert_stability_val")
    flow_val = train_ensemble(
        df,
        FEAT_28 + STABLE_FLOW4,
        val_windows,
        l2=1.0,
        rolling=False,
        label="expert_stable_flow_val",
        cs_rank_exclude=STABLE_FLOW4,
    )

    for name, preds in {
        "base_test": base_test,
        "stability_test": stability_test,
        "flow_test": flow_test,
        "base_val": base_val,
        "stability_val": stability_val,
        "flow_val": flow_val,
    }.items():
        if preds is None or preds.empty:
            raise RuntimeError(f"Missing predictions: {name}")

    print("\n[4] Calibrating soft blends on validation slices...")
    calibration_rows: List[Dict[str, object]] = []
    soft_rvol_frames = []
    soft_stable_frames = []
    soft_tri_frames = []

    for window in WINDOWS:
        params_rvol, calib_rvol = calibrate_two_expert(
            base_val,
            stability_val,
            context,
            regime_df,
            window,
            regime_col="btc_rvol_24h",
            alt_col_name="pred_alt",
            label="soft_r42_vs_stability",
        )
        params_stable, calib_stable = calibrate_two_expert(
            base_val,
            flow_val,
            context,
            regime_df,
            window,
            regime_col="stable_supply_accel",
            alt_col_name="pred_alt",
            label="soft_r42_vs_flow",
        )
        params_tri, calib_tri = calibrate_tri_expert(
            base_val,
            stability_val,
            flow_val,
            context,
            regime_df,
            window,
        )

        calibration_rows.extend([calib_rvol, calib_stable, calib_tri])

        soft_rvol_frames.append(
            build_two_expert_test_blend(base_test, stability_test, context, window, "btc_rvol_24h", "pred_alt", params_rvol)
        )
        soft_stable_frames.append(
            build_two_expert_test_blend(base_test, flow_test, context, window, "stable_supply_accel", "pred_alt", params_stable)
        )
        soft_tri_frames.append(
            build_tri_test_blend(base_test, stability_test, flow_test, context, window, params_tri)
        )

    calibration_df = pd.DataFrame(calibration_rows)
    calibration_df.to_csv(CAL_PATH, index=False)

    soft_rvol = pd.concat(soft_rvol_frames, ignore_index=True)
    soft_stable = pd.concat(soft_stable_frames, ignore_index=True)
    soft_tri = pd.concat(soft_tri_frames, ignore_index=True)

    print("\n[5] Evaluating experts and soft blends...")
    rows: List[Dict[str, object]] = []
    configs = [
        ("expert_r42", base_test),
        ("expert_stability", stability_test),
        ("expert_stable_flow", flow_test),
        ("soft_r42_vs_stability", soft_rvol),
        ("soft_r42_vs_flow", soft_stable),
        ("soft_tri", soft_tri),
    ]
    for label, preds in configs:
        print(f"  {label}...")
        results = evaluate_preds(preds, regime_df, label)
        rows.extend(summarize(label, results))

    result_df = pd.DataFrame(rows).sort_values(["window", "sharpe"], ascending=[True, False])
    result_df.to_csv(SUMMARY_PATH, index=False)

    print("\n[6] Best configs")
    for window in ["W2", "W3", "ALL"]:
        top = result_df[result_df["window"] == window].head(6)
        print(f"  {window}:")
        print(top[["config", "sharpe", "cost_pct", "avg_turnover"]].to_string(index=False))

    print("\n[7] Saved artifacts")
    print(f"  Summary CSV: {SUMMARY_PATH.name}")
    print(f"  Calibration CSV: {CAL_PATH.name}")


if __name__ == "__main__":
    main()