#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D7.2 — inspect / feature-engineer / IC-scan google trends + cc_social.

Conservative assumptions:
  - all daily data is shifted by +1 day before merge to avoid same-day lookahead
  - scan is done only on the symbol overlap universe
  - train-only IC metrics per WF window

Outputs:
  - results_d7.log
  - results_d7_social_search_ic.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from _research_round7 import SYM_35, WINDOWS
from _ic_scanner import build_features_minimal, load_derivatives, load_ohlcv


BASE_DIR = Path(__file__).resolve().parent
FEATURE_DIR = BASE_DIR / "data" / "features"
OUT_PATH = BASE_DIR / "results_d7_social_search_ic.csv"


def rolling_zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return (series - mean) / (std + 1e-10)


def load_base_frame(symbols: List[str]) -> pd.DataFrame:
    ohlcv = load_ohlcv()
    ohlcv = ohlcv[ohlcv["symbol"].isin(symbols)]
    derivs = load_derivatives()
    df = build_features_minimal(ohlcv, derivs)
    return df[["timestamp", "symbol", "fwd_ret_12h"]].copy()


def load_google_trends() -> pd.DataFrame:
    df = pd.read_parquet(FEATURE_DIR / "google_trends.parquet").copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values(["symbol", "date"])
    df["gtrends_7d_delta"] = df.groupby("symbol")["gtrends"].diff(7)
    df["gtrends_30d_delta"] = df.groupby("symbol")["gtrends"].diff(30)
    df["gtrends_z_30d"] = df.groupby("symbol")["gtrends"].transform(lambda s: rolling_zscore(s, 30, 10))
    df["timestamp"] = df["date"] + pd.Timedelta(days=1)
    df = df[[
        "timestamp",
        "symbol",
        "gtrends",
        "gtrends_chg4w",
        "gtrends_z",
        "gtrends_7d_delta",
        "gtrends_30d_delta",
        "gtrends_z_30d",
    ]]
    return df


def load_cc_social() -> pd.DataFrame:
    df = pd.read_parquet(FEATURE_DIR / "cc_social_daily.parquet").copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values(["symbol", "date"])

    df["social_reddit_activity"] = df["reddit_posts_per_day"].fillna(0) + df["reddit_comments_per_day"].fillna(0)
    df["social_dev_activity"] = (
        df["code_repo_stars"].fillna(0)
        + df["code_repo_forks"].fillna(0)
        + df["code_repo_contributors"].fillna(0)
    )
    df["twitter_followers_chg7d"] = df.groupby("symbol")["twitter_followers"].diff(7)
    df["reddit_users_chg7d"] = df.groupby("symbol")["reddit_active_users"].diff(7)
    df["social_dev_chg30d"] = df.groupby("symbol")["social_dev_activity"].diff(30)
    df["social_activity_z_30d"] = df.groupby("symbol")["social_reddit_activity"].transform(
        lambda s: rolling_zscore(s, 30, 10)
    )
    df["timestamp"] = df["date"] + pd.Timedelta(days=1)
    df = df[[
        "timestamp",
        "symbol",
        "reddit_subscribers",
        "reddit_active_users",
        "twitter_followers",
        "social_reddit_activity",
        "social_dev_activity",
        "twitter_followers_chg7d",
        "reddit_users_chg7d",
        "social_dev_chg30d",
        "social_activity_z_30d",
    ]]
    return df


def hourly_merge(base: pd.DataFrame, daily_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    merged_frames = []
    for symbol, group in daily_df.groupby("symbol"):
        g = group.sort_values("timestamp").set_index("timestamp").resample("1h").ffill().reset_index()
        g["symbol"] = symbol
        merged_frames.append(g)
    hourly = pd.concat(merged_frames, ignore_index=True)
    merged = base.merge(hourly[["timestamp", "symbol"] + feature_cols], on=["timestamp", "symbol"], how="left")
    return merged


def mean_timestamp_ic(frame: pd.DataFrame, feature: str) -> float:
    ics = []
    for _, group in frame.groupby("timestamp"):
        sample = group[[feature, "fwd_ret_12h"]].dropna()
        if len(sample) < 5 or sample[feature].nunique() < 3:
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


def coverage(frame: pd.DataFrame, feature: str) -> float:
    return float(1.0 - frame[feature].isna().mean())


def main() -> None:
    print("=" * 80)
    print("D7.2 — SOCIAL / SEARCH DATA INSPECTION + IC SCAN")
    print("=" * 80)
    print("Conservative merge: all daily data shifted by +1 day before hourly forward-fill")

    gtrends = load_google_trends()
    social = load_cc_social()
    symbols = sorted(set(gtrends["symbol"]).union(set(social["symbol"])).intersection(set(SYM_35)))

    print("\n[1] Base overlap universe...")
    print(f"  Overlap symbols ({len(symbols)}): {symbols}")
    base = load_base_frame(symbols)
    print(f"  Base rows: {len(base):,}")

    g_features = [c for c in gtrends.columns if c not in ["timestamp", "symbol"]]
    s_features = [c for c in social.columns if c not in ["timestamp", "symbol"]]
    merged = hourly_merge(base, gtrends, g_features)
    merged = hourly_merge(merged, social, s_features)

    print("\n[2] Coverage snapshot...")
    for feature in g_features + s_features:
        print(f"  {feature:<24} coverage={coverage(merged, feature) * 100:5.1f}%")

    print("\n[3] Train-only IC scan...")
    rows: List[Dict[str, object]] = []
    tz = merged["timestamp"].dt.tz
    for window in WINDOWS:
        train_end = pd.Timestamp(window["train_end"], tz=tz)
        train_df = merged[merged["timestamp"] < train_end].copy()
        for feature in g_features + s_features:
            rows.append({
                "window": window["name"],
                "feature": feature,
                "mean_ts_ic": mean_timestamp_ic(train_df, feature),
                "pooled_ic": pooled_ic(train_df, feature),
                "coverage": coverage(train_df, feature),
                "symbols": int(train_df.loc[train_df[feature].notna(), "symbol"].nunique()),
                "rows": int(train_df[feature].notna().sum()),
            })

    result = pd.DataFrame(rows)
    result["score"] = result["mean_ts_ic"].abs() * result["coverage"]
    result = result.sort_values(["window", "score"], ascending=[True, False])
    result.to_csv(OUT_PATH, index=False)

    print("\n[4] Top features by window")
    for window in ["W1", "W2", "W3"]:
        top = result[result["window"] == window].head(8)
        if top.empty:
            continue
        print(f"  {window}:")
        print(top[["feature", "mean_ts_ic", "pooled_ic", "coverage", "symbols"]].to_string(index=False))

    print("\n[5] Saved artifacts")
    print(f"  IC CSV: {OUT_PATH.name}")


if __name__ == "__main__":
    main()