#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R35 — new features from existing data.

Outputs:
  - results_r35.log
  - results_r35_feature_ic.csv
  - results_r35_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from _research_round7 import SYM_35, WINDOWS
from _ic_scanner import build_features_minimal, load_derivatives, load_ohlcv
from _research_r22_models import add_new_features, build_r19_features
from _research_r30b_fixed import add_extra_features_clean, compute_regime_extended, eval_per_window, train_ensemble
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

GROUPS = {
    "r35a_cs_second_order": [
        "ret_dispersion_12h",
        "cs_rank_ma_5",
        "oi_chg_12h_cs",
        "taker_cvd_12h_cs",
        "cum_funding_24h_cs",
    ],
    "r35b_interactions": [
        "oi_ret_divergence",
        "funding_ret_cross",
        "vol_ret_confirm",
        "ret_168h_x_disp",
    ],
    "r35c_temporal": [
        "ret_autocorr_24h",
        "ret_accel_12h",
        "funding_momentum",
        "vol_concentration_4of12",
    ],
    "r35d_market": [
        "mkt_funding_mean",
        "mkt_funding_pct_pos",
        "mkt_funding_dispersion",
        "mkt_oi_chg_sum",
        "mkt_oi_extreme_pct",
    ],
}

MARKET_LEVEL_FEATURES = {
    "ret_dispersion_12h",
    "mkt_funding_mean",
    "mkt_funding_pct_pos",
    "mkt_funding_dispersion",
    "mkt_oi_chg_sum",
    "mkt_oi_extreme_pct",
    # ── Same value for all symbols at each timestamp ──
    # Without this, cs_rank makes them constant 0.0 (all tied → rank 0.5 - 0.5 = 0)
    "pct_coins_up_12h",
    "pct_coins_up_1h",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
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


def rolling_autocorr(series: pd.Series, lag: int, window: int) -> pd.Series:
    return series.rolling(window, min_periods=max(window // 2, lag + 2)).corr(series.shift(lag))


def add_r35_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.sort_values(["symbol", "timestamp"]).copy()

    df["ret_dispersion_12h"] = df.groupby("timestamp")["ret_12h"].transform("std")
    if "ret_12h" in df.columns:
        cs_rank = df.groupby("timestamp")["ret_12h"].rank(pct=True) - 0.5
        df["cs_rank_ma_5"] = cs_rank.groupby(df["symbol"]).transform(
            lambda series: series.rolling(5, min_periods=3).mean()
        )
    if "oi_chg_12h" in df.columns:
        df["oi_chg_12h_cs"] = df.groupby("timestamp")["oi_chg_12h"].rank(pct=True) - 0.5
    if "taker_cvd_12h" in df.columns:
        df["taker_cvd_12h_cs"] = df.groupby("timestamp")["taker_cvd_12h"].rank(pct=True) - 0.5
    if "cum_funding_24h" in df.columns:
        df["cum_funding_24h_cs"] = df.groupby("timestamp")["cum_funding_24h"].rank(pct=True) - 0.5

    if {"oi_chg_12h", "ret_12h"}.issubset(df.columns):
        df["oi_ret_divergence"] = df["oi_chg_12h"] * (-df["ret_12h"])
    if {"cum_funding_24h", "ret_24h"}.issubset(df.columns):
        df["funding_ret_cross"] = df["cum_funding_24h"] * df["ret_24h"]
    if {"rel_volume_cs", "ret_12h"}.issubset(df.columns):
        df["vol_ret_confirm"] = df["rel_volume_cs"] * np.sign(df["ret_12h"].fillna(0.0))
    if {"ret_168h", "ret_dispersion_12h"}.issubset(df.columns):
        df["ret_168h_x_disp"] = df["ret_168h"] * df["ret_dispersion_12h"]

    def per_symbol_features(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("timestamp").copy()
        ret_1h = group["ret_1h"] if "ret_1h" in group.columns else group["close"].pct_change()
        group["ret_autocorr_24h"] = rolling_autocorr(ret_1h, lag=1, window=24)
        if "ret_12h" in group.columns:
            group["ret_accel_12h"] = group["ret_12h"] - group["ret_12h"].shift(12)
        if "cum_funding_24h" in group.columns:
            group["funding_momentum"] = group["cum_funding_24h"] - group["cum_funding_24h"].shift(12)
        vol = group["volume"]
        group["vol_concentration_4of12"] = (
            vol.rolling(4, min_periods=2).sum() / (vol.rolling(12, min_periods=6).sum() + 1e-10)
        )
        return group

    # pandas 3.0: groupby.apply excludes grouping column; save & restore
    _sym_save = df["symbol"].copy()
    df = df.groupby("symbol", group_keys=False).apply(per_symbol_features)
    if "symbol" not in df.columns:
        df["symbol"] = _sym_save

    if "cum_funding_24h" in df.columns:
        df["mkt_funding_mean"] = df.groupby("timestamp")["cum_funding_24h"].transform("mean")
        df["mkt_funding_pct_pos"] = df.groupby("timestamp")["cum_funding_24h"].transform(
            lambda series: float((series > 0).mean())
        )
        df["mkt_funding_dispersion"] = df.groupby("timestamp")["cum_funding_24h"].transform("std")
    if "oi_chg_12h" in df.columns:
        df["mkt_oi_chg_sum"] = df.groupby("timestamp")["oi_chg_12h"].transform("sum")
    if "oi_zscore" in df.columns:
        df["mkt_oi_extreme_pct"] = df.groupby("timestamp")["oi_zscore"].transform(
            lambda series: float((series.abs() >= 1.0).mean())
        )

    added = []
    for feature_list in GROUPS.values():
        for feature in feature_list:
            if feature in df.columns:
                df[feature] = df[feature].replace([np.inf, -np.inf], np.nan)
                added.append(feature)
    return df, sorted(set(added))


def mean_timestamp_ic(frame: pd.DataFrame, feature: str) -> float:
    ics = []
    for _, group in frame.groupby("timestamp"):
        sample = group[[feature, "fwd_ret_12h"]].dropna()
        if len(sample) < 10 or sample[feature].nunique() < 3:
            continue
        ic = stats.spearmanr(sample[feature], sample["fwd_ret_12h"])[0]
        if pd.notna(ic):
            ics.append(float(ic))
    return float(np.mean(ics)) if ics else np.nan


def pooled_ic(frame: pd.DataFrame, feature: str) -> float:
    sample = frame[[feature, "fwd_ret_12h"]].dropna()
    if len(sample) < 50 or sample[feature].nunique() < 5:
        return np.nan
    return float(stats.spearmanr(sample[feature], sample["fwd_ret_12h"])[0])


def market_level_ts_corr(frame: pd.DataFrame, feature: str) -> float:
    agg = (
        frame.groupby("timestamp")
        .agg(feature_value=(feature, "first"), market_ret=("fwd_ret_12h", "mean"))
        .dropna()
    )
    if len(agg) < 30 or agg["feature_value"].nunique() < 5:
        return np.nan
    return float(stats.spearmanr(agg["feature_value"], agg["market_ret"])[0])


def ic_scan(df: pd.DataFrame, features: List[str], output_csv: Path) -> pd.DataFrame:
    rows = []
    tz = df["timestamp"].dt.tz
    for window in WINDOWS:
        train_end = pd.Timestamp(window["train_end"], tz=tz)
        train_df = df[df["timestamp"] < train_end].copy()
        for feature in features:
            if feature not in train_df.columns:
                continue
            is_market_level = feature in MARKET_LEVEL_FEATURES
            rows.append({
                "window": window["name"],
                "feature": feature,
                "group": next(group for group, cols in GROUPS.items() if feature in cols),
                "is_market_level": is_market_level,
                "train_mean_ts_ic": np.nan if is_market_level else mean_timestamp_ic(train_df, feature),
                "train_pooled_ic": np.nan if is_market_level else pooled_ic(train_df, feature),
                "train_market_ts_corr": market_level_ts_corr(train_df, feature) if is_market_level else np.nan,
                "rows": int(len(train_df)),
            })
    result = pd.DataFrame(rows)
    result["scan_score"] = np.where(
        result["is_market_level"],
        result["train_market_ts_corr"].abs(),
        result["train_mean_ts_ic"].abs(),
    )
    result = result.sort_values(["group", "window", "scan_score"], ascending=[True, True, False])
    result.to_csv(output_csv, index=False)
    return result


def select_top_features(ic_df: pd.DataFrame) -> List[str]:
    summary = (
        ic_df.groupby(["feature", "group", "is_market_level"], as_index=False)
        .agg(
            scan_score=("scan_score", "mean"),
            train_mean_ts_ic=("train_mean_ts_ic", "mean"),
            train_market_ts_corr=("train_market_ts_corr", "mean"),
        )
        .sort_values("scan_score", ascending=False)
    )
    top_non_market = summary[~summary["is_market_level"]].head(3)["feature"].tolist()
    top_market = summary[summary["is_market_level"]].head(2)["feature"].tolist()
    return list(dict.fromkeys(top_non_market + top_market))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-bundles", type=int, default=6)
    args = parser.parse_args()

    ic_path = BASE_DIR / "results_r35_feature_ic.csv"
    summary_path = BASE_DIR / "results_r35_summary.csv"

    print("=" * 80)
    print("R35 — NEW FEATURES FROM EXISTING DATA")
    print("=" * 80)

    print("\n[1] Loading research frame...")
    df, regime_df = load_research_frame()
    df, added = add_r35_features(df)
    regime_df = compute_regime_extended(df).sort_index()
    print(f"  Data: {len(df):,} rows, {len(df.columns)} cols")
    print(f"  Added features: {added}")

    print("\n[2] Train-only IC scan...")
    ic_df = ic_scan(df, added, ic_path)
    for group_name in GROUPS:
        top = (
            ic_df[ic_df["group"] == group_name]
            .groupby("feature", as_index=False)
            .agg(scan_score=("scan_score", "mean"), is_market_level=("is_market_level", "first"))
            .sort_values("scan_score", ascending=False)
            .head(3)
        )
        if top.empty:
            continue
        print(f"  {group_name}:")
        print(top.to_string(index=False))

    top_features = select_top_features(ic_df)
    bundle_map = {
        "top3_plus_market": top_features,
        **{name: [feature for feature in features if feature in added] for name, features in GROUPS.items()},
    }
    ordered_bundles = [(name, feats) for name, feats in bundle_map.items() if feats][: args.max_bundles]

    print("\n[3] Walk-forward bundle tests...")
    summaries = []
    for bundle_name, bundle_feats in ordered_bundles:
        feats = FEAT_28 + [feature for feature in bundle_feats if feature not in FEAT_28]
        no_rank = [feature for feature in bundle_feats if feature in MARKET_LEVEL_FEATURES]
        label = f"A_28f+{bundle_name}"
        print(f"\n  Training {label} ({len(bundle_feats)} new features)...")
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
            continue
        results = eval_per_window(preds, regime_df, CFG, label=label)
        for window in ["W1", "W2", "W3", "ALL"]:
            metric = results.get(window, {})
            summaries.append({
                "bundle": bundle_name,
                "window": window,
                "n_new_features": len(bundle_feats),
                "market_level_features": ",".join(no_rank),
                "features": ",".join(bundle_feats),
                "sharpe": metric.get("sharpe", np.nan),
                "sharpe_gross": metric.get("sharpe_gross", np.nan),
                "equity": metric.get("equity", np.nan),
                "cost_pct": metric.get("total_cost_pct", np.nan),
                "avg_turnover": metric.get("avg_turnover", np.nan),
            })

    summary_df = pd.DataFrame(summaries).sort_values(["window", "sharpe"], ascending=[True, False])
    summary_df.to_csv(summary_path, index=False)

    print("\n[4] Best bundles")
    for window in ["W2", "W3", "ALL"]:
        top = summary_df[summary_df["window"] == window].head(5)
        if top.empty:
            continue
        print(f"  {window}:")
        print(top[["bundle", "sharpe", "cost_pct", "avg_turnover"]].to_string(index=False))

    print("\n[5] Saved artifacts")
    print(f"  IC CSV: {ic_path.name}")
    print(f"  Summary CSV: {summary_path.name}")


if __name__ == "__main__":
    main()