#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R36 — regime gating / two-expert blend.

Outputs:
  - results_r36.log
  - results_r36_gating_summary.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from _research_round7 import WINDOWS
from _research_r30b_fixed import compute_regime_extended, eval_per_window, train_ensemble
from _research_r33_creative_features import FEAT_28, FEAT_30
from _research_r39_stablecoin_features import FEATURE_BUNDLES, add_stablecoin_features, load_research_frame


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

STABLE_FLOW4 = FEATURE_BUNDLES["stable_flow4"]


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
    btc = df[df["symbol"] == "BTC/USDT"][ ["timestamp", "rvol_24h"] ].drop_duplicates("timestamp")
    btc = btc.rename(columns={"rvol_24h": "btc_rvol_24h"})
    context = context.merge(btc, on="timestamp", how="left")
    return context


def blend_experts(base: pd.DataFrame, alt: pd.DataFrame, choose_alt: pd.DataFrame) -> pd.DataFrame:
    merged = base.merge(
        alt[["timestamp", "symbol", "pred"]].rename(columns={"pred": "pred_alt"}),
        on=["timestamp", "symbol"],
        how="inner",
    )
    merged = merged.merge(choose_alt, on="timestamp", how="left")
    merged["pred"] = np.where(merged["choose_alt"].fillna(False), merged["pred_alt"], merged["pred"])
    return merged[["timestamp", "symbol", "pred", "fwd_ret", "window"]]


def tri_gate(base: pd.DataFrame, stability: pd.DataFrame, stable_flow: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    merged = (
        base.rename(columns={"pred": "pred_base"})
        .merge(stability[["timestamp", "symbol", "pred"]].rename(columns={"pred": "pred_stability"}), on=["timestamp", "symbol"], how="inner")
        .merge(stable_flow[["timestamp", "symbol", "pred"]].rename(columns={"pred": "pred_stable_flow"}), on=["timestamp", "symbol"], how="inner")
        .merge(context, on="timestamp", how="left")
    )

    out_frames = []
    tz = merged["timestamp"].dt.tz
    for window in WINDOWS:
        test_start = pd.Timestamp(window["test_start"], tz=tz)
        hist = context[context["timestamp"] < test_start].copy()
        if hist.empty:
            continue
        rvol_cut = float(hist["btc_rvol_24h"].quantile(0.67))
        stable_cut = float(hist["stable_total_supply_chg7d_z"].quantile(0.67))
        mask = merged["window"] == window["name"]
        sub = merged[mask].copy()
        use_stability = sub["btc_rvol_24h"] >= rvol_cut
        use_stable_flow = sub["stable_total_supply_chg7d_z"] >= stable_cut
        sub["pred"] = np.select(
            [use_stable_flow, use_stability],
            [sub["pred_stable_flow"], sub["pred_stability"]],
            default=sub["pred_base"],
        )
        out_frames.append(sub[["timestamp", "symbol", "pred", "fwd_ret", "window"]])

    return pd.concat(out_frames, ignore_index=True)


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
    summary_path = BASE_DIR / "results_r36_gating_summary.csv"

    print("=" * 80)
    print("R36 — REGIME GATING")
    print("=" * 80)

    print("\n[1] Loading research frame...")
    df = load_research_frame()
    df, _ = add_stablecoin_features(df)
    regime_df = compute_regime_extended(df).sort_index()
    context = build_market_context(df)
    print(f"  Data: {len(df):,} rows, {len(df.columns)} cols")

    print("\n[2] Training experts...")
    base_preds = train_ensemble(df, FEAT_28, WINDOWS, l2=1.0, rolling=False, label="A_28f")
    stability_preds = train_ensemble(df, FEAT_30, WINDOWS, l2=1.0, rolling=False, label="D_30f")
    stable_preds = train_ensemble(
        df,
        FEAT_28 + STABLE_FLOW4,
        WINDOWS,
        l2=1.0,
        rolling=False,
        label="A_28f+stable_flow4",
        cs_rank_exclude=STABLE_FLOW4,
    )
    if base_preds is None or stability_preds is None or stable_preds is None:
        raise RuntimeError("One or more expert predictions are missing")

    rows = []

    print("\n[3] Evaluating standalone experts...")
    for label, preds in [
        ("expert_base", base_preds),
        ("expert_stability", stability_preds),
        ("expert_stable_flow", stable_preds),
    ]:
        results = eval_per_window(preds, regime_df, CFG, label=label)
        rows.extend(summarize(label, results))

    print("\n[4] Evaluating gated blends...")
    tz = context["timestamp"].dt.tz
    base_vs_stability_flags = []
    base_vs_stable_flags = []
    for window in WINDOWS:
        test_start = pd.Timestamp(window["test_start"], tz=tz)
        hist = context[context["timestamp"] < test_start].copy()
        if hist.empty:
            continue
        rvol_cut = float(hist["btc_rvol_24h"].quantile(0.67))
        stable_cut = float(hist["stable_supply_accel"].quantile(0.67))
        sub = context.copy()
        sub["window"] = window["name"]
        sub = sub[(sub["timestamp"] >= pd.Timestamp(window["test_start"], tz=tz)) & (sub["timestamp"] <= pd.Timestamp(window["test_end"], tz=tz))]
        base_vs_stability_flags.append(sub[["timestamp"]].assign(choose_alt=sub["btc_rvol_24h"] >= rvol_cut))
        base_vs_stable_flags.append(sub[["timestamp"]].assign(choose_alt=sub["stable_supply_accel"] >= stable_cut))

    gate_rvol = pd.concat(base_vs_stability_flags, ignore_index=True)
    gate_stable = pd.concat(base_vs_stable_flags, ignore_index=True)

    blended_rvol = blend_experts(base_preds, stability_preds, gate_rvol)
    blended_stable = blend_experts(base_preds, stable_preds, gate_stable)
    blended_tri = tri_gate(base_preds, stability_preds, stable_preds, context)

    for label, preds in [
        ("gate_rvol_base_vs_stability", blended_rvol),
        ("gate_stable_base_vs_flow", blended_stable),
        ("gate_tri_regime", blended_tri),
    ]:
        results = eval_per_window(preds, regime_df, CFG, label=label)
        rows.extend(summarize(label, results))

    result_df = pd.DataFrame(rows).sort_values(["window", "sharpe"], ascending=[True, False])
    result_df.to_csv(summary_path, index=False)

    print("\n[5] Best gating configs")
    for window in ["W2", "W3", "ALL"]:
        top = result_df[result_df["window"] == window].head(5)
        if top.empty:
            continue
        print(f"  {window}:")
        print(top[["config", "sharpe", "cost_pct", "avg_turnover"]].to_string(index=False))

    print("\n[6] Saved artifacts")
    print(f"  Summary CSV: {summary_path.name}")


if __name__ == "__main__":
    main()