#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R34 — W2 attribution for the current champion.

Outputs:
  - results_r34.log
  - results_r34_w2_conditional_ic.csv
  - results_r34_w2_coin_contrib.csv
    - results_r34_feature_importance.csv

Primary goal: explain why W2 breaks for the FEAT_28 baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

from _research_round7 import SYM_35, WINDOWS
from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal
from _research_r22_models import SEEDS, add_new_features, build_r19_features, cs_rank_cols, log
from _research_r30b_fixed import (
    add_extra_features_clean,
    compute_regime_extended,
    eval_with_costs,
    simulate_with_costs,
    train_ensemble,
)
from _research_r33_creative_features import FEAT_28, FEAT_30, add_r33_features


BASE_DIR = Path(__file__).resolve().parent

FEATURE_SETS = {
    "baseline": ("A_28f", FEAT_28),
    "stability": ("D_30f", FEAT_30),
}

CFG = {
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


def load_research_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
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


def build_market_context(df: pd.DataFrame) -> pd.DataFrame:
    context = (
        df.groupby("timestamp")
        .agg(
            breadth_12h=("pct_coins_up_12h", "mean"),
            breadth_1h=("pct_coins_up_1h", "mean"),
            dispersion_12h=("ret_12h", "std"),
            market_funding_24h=("cum_funding_24h", "mean"),
        )
        .sort_index()
    )
    btc = df[df["symbol"] == "BTC/USDT"].copy().set_index("timestamp")
    btc_context = btc[["rvol_24h", "ret_24h", "ret_168h"]].rename(columns={
        "rvol_24h": "btc_rvol_24h",
        "ret_24h": "btc_ret_24h_feature",
        "ret_168h": "btc_ret_168h_feature",
    })
    context = context.join(btc_context, how="left")
    return context.reset_index()


def add_bins(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    labels = ["low", "mid", "high"]
    for column in columns:
        valid = out[column].replace([np.inf, -np.inf], np.nan)
        try:
            out[f"{column}_bin"] = pd.qcut(valid, q=3, labels=labels, duplicates="drop")
        except ValueError:
            out[f"{column}_bin"] = pd.Series(index=out.index, dtype="object")
    return out


def pooled_ic(frame: pd.DataFrame, feature: str) -> float:
    sub = frame[[feature, "fwd_ret_12h"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 50 or sub[feature].nunique() < 5:
        return np.nan
    return float(stats.spearmanr(sub[feature], sub["fwd_ret_12h"])[0])


def mean_timestamp_ic(frame: pd.DataFrame, feature: str) -> float:
    ics = []
    for _, grp in frame.groupby("timestamp"):
        sub = grp[[feature, "fwd_ret_12h"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < 10 or sub[feature].nunique() < 3:
            continue
        ic = stats.spearmanr(sub[feature], sub["fwd_ret_12h"])[0]
        if pd.notna(ic):
            ics.append(float(ic))
    if not ics:
        return np.nan
    return float(np.mean(ics))


def conditional_ic_report(df: pd.DataFrame, feature_cols: list[str], output_csv: Path) -> pd.DataFrame:
    ranked = cs_rank_cols(df.copy(), feature_cols)
    context_cols = [
        "btc_rvol_24h",
        "breadth_12h",
        "dispersion_12h",
        "market_funding_24h",
    ]
    ranked = add_bins(ranked, context_cols)
    records = []
    for regime_col in context_cols:
        bin_col = f"{regime_col}_bin"
        for bin_name, regime_df in ranked.groupby(bin_col, dropna=True):
            for feature in feature_cols:
                records.append({
                    "regime": regime_col,
                    "bin": str(bin_name),
                    "feature": feature,
                    "pooled_ic": pooled_ic(regime_df, feature),
                    "mean_ts_ic": mean_timestamp_ic(regime_df, feature),
                    "rows": int(len(regime_df)),
                })
    result = pd.DataFrame(records).sort_values(["regime", "bin", "mean_ts_ic"], ascending=[True, True, False])
    result.to_csv(output_csv, index=False)
    return result


def per_window_feature_importance(df: pd.DataFrame, feature_cols: list[str], output_csv: Path) -> pd.DataFrame:
    avail = [feature for feature in feature_cols if feature in df.columns]
    tz = df["timestamp"].dt.tz
    rows = []

    for window in WINDOWS:
        train_end = pd.Timestamp(window["train_end"], tz=tz)
        val_start = pd.Timestamp(window["val_start"], tz=tz)
        val_end = pd.Timestamp(window["val_end"], tz=tz)
        test_start = pd.Timestamp(window["test_start"], tz=tz)
        test_end = pd.Timestamp(window["test_end"], tz=tz)

        train_df = df[df["timestamp"] < train_end].copy()
        val_df = df[(df["timestamp"] >= val_start) & (df["timestamp"] < val_end)].copy()
        test_df = df[(df["timestamp"] >= test_start) & (df["timestamp"] <= test_end)].copy()
        if len(train_df) < 5000 or len(test_df) < 200:
            continue

        ranked_frames = {
            "train": cs_rank_cols(train_df, avail),
            "val": cs_rank_cols(val_df, avail),
            "test": cs_rank_cols(test_df, avail),
        }
        for split_df in ranked_frames.values():
            split_df["target_binary"] = (split_df["fwd_ret_12h"] > 0).astype(int)
            for feature in avail:
                if split_df[feature].isna().any():
                    split_df[feature] = split_df[feature].fillna(0)
            split_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        train_clean = ranked_frames["train"][avail + ["target_binary"]].dropna()
        val_clean = ranked_frames["val"][avail + ["target_binary"]].dropna()
        test_ranked = ranked_frames["test"].dropna(subset=["fwd_ret_12h"]).copy()
        if train_clean.empty or val_clean.empty or test_ranked.empty:
            continue

        gain_sum = {feature: 0.0 for feature in avail}
        n_models = 0
        for seed in SEEDS:
            params_lgb = {
                "objective": "binary",
                "metric": "auc",
                "learning_rate": 0.03,
                "num_leaves": 63,
                "min_child_samples": 100,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "lambda_l2": 1.0,
                "verbose": -1,
                "n_jobs": -1,
                "seed": seed,
            }
            dtrain = lgb.Dataset(train_clean[avail], label=train_clean["target_binary"])
            dval = lgb.Dataset(val_clean[avail], label=val_clean["target_binary"])
            model = lgb.train(
                params_lgb,
                dtrain,
                num_boost_round=600,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)],
            )
            for feature, value in zip(avail, model.feature_importance(importance_type="gain")):
                gain_sum[feature] += float(value)
            n_models += 1

        if n_models == 0:
            continue

        avg_gain = {feature: gain_sum[feature] / n_models for feature in avail}
        total_gain = sum(avg_gain.values())
        for feature in avail:
            rows.append({
                "window": window["name"],
                "feature": feature,
                "avg_gain": avg_gain[feature],
                "gain_pct": avg_gain[feature] / (total_gain + 1e-10),
                "test_mean_ts_ic": mean_timestamp_ic(test_ranked, feature),
                "test_pooled_ic": pooled_ic(test_ranked, feature),
                "test_rows": int(len(test_ranked)),
            })

    result = pd.DataFrame(rows).sort_values(["window", "avg_gain"], ascending=[True, False])
    result.to_csv(output_csv, index=False)
    return result


def _apply_prediction_ema(grp: pd.DataFrame, prev_preds: dict[str, float], ema_alpha: float | None) -> pd.DataFrame:
    if ema_alpha is None or ema_alpha >= 1.0:
        return grp
    grp = grp.copy()
    for idx, row in grp.iterrows():
        sym = row["symbol"]
        raw_pred = row["pred"]
        smoothed = ema_alpha * raw_pred + (1 - ema_alpha) * prev_preds.get(sym, raw_pred)
        prev_preds[sym] = smoothed
        grp.at[idx, "pred"] = smoothed
    return grp


def build_position_trace(preds: pd.DataFrame, regime_df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_long = cfg["n_long"]
    n_short = cfg["n_short"]
    trend_cutoff = cfg["trend_cutoff"]
    rebal_hours = cfg["rebal_hours"]
    dyn_threshold = cfg["dyn_threshold"]
    ema_alpha = cfg.get("ema_alpha")
    hysteresis = cfg.get("hysteresis", 0)

    records = []
    contrib_records = []
    for window in sorted(preds["window"].unique()):
        prev_preds: dict[str, float] = {}
        prev_longs: set[str] = set()
        prev_shorts: set[str] = set()
        wsub = preds[preds["window"] == window].copy()
        timestamps_sorted = sorted(wsub["timestamp"].unique())
        rebal_timestamps = timestamps_sorted[::rebal_hours]
        grouped = {ts: grp.copy() for ts, grp in wsub.groupby("timestamp")}

        for ts in rebal_timestamps:
            if ts not in grouped or ts not in regime_df.index:
                continue

            trend_strength = regime_df.loc[ts].get("trend_strength", 0.0)
            if trend_strength > trend_cutoff:
                continue

            grp = _apply_prediction_ema(grouped[ts].copy(), prev_preds, ema_alpha)
            grp["pred_rank"] = grp["pred"].rank(ascending=False, method="first")
            n = len(grp)
            nl = min(n_long, n // 3)
            ns = min(n_short, n // 3)
            if nl == 0 and ns == 0:
                continue

            exposure = 1.0
            if dyn_threshold is not None and trend_strength > dyn_threshold:
                exposure = max(0.1, 1.0 - (trend_strength - dyn_threshold) /
                               (trend_cutoff - dyn_threshold + 1e-10) * 0.5)

            if hysteresis > 0 and (prev_longs or prev_shorts):
                new_longs = set()
                new_shorts = set()
                for _, row in grp.iterrows():
                    symbol = row["symbol"]
                    rank = row["pred_rank"]
                    if symbol in prev_longs and rank <= nl + hysteresis:
                        new_longs.add(symbol)
                    elif symbol in prev_shorts and rank > (n - ns - hysteresis):
                        new_shorts.add(symbol)

                remain_long = nl - len(new_longs)
                remain_short = ns - len(new_shorts)
                if remain_long > 0:
                    candidates = grp[~grp["symbol"].isin(new_longs | new_shorts)].sort_values("pred_rank")
                    new_longs |= set(candidates.head(remain_long)["symbol"].tolist())
                if remain_short > 0:
                    candidates = grp[~grp["symbol"].isin(new_longs | new_shorts)].sort_values("pred_rank", ascending=False)
                    new_shorts |= set(candidates.head(remain_short)["symbol"].tolist())
            else:
                new_longs = set(grp[grp["pred_rank"] <= nl]["symbol"].tolist()) if nl > 0 else set()
                new_shorts = set(grp[grp["pred_rank"] > (n - ns)]["symbol"].tolist()) if ns > 0 else set()

            new_opened = (new_longs - prev_longs) | (new_shorts - prev_shorts)
            closed = (prev_longs - new_longs) | (prev_shorts - new_shorts)
            turnover_count = len(new_opened) + len(closed)
            total_positions = len(new_longs) + len(new_shorts)
            avg_weight = (1.0 / total_positions) if total_positions > 0 else 0.0
            turnover_cost = turnover_count * COST_ONE_WAY * avg_weight if total_positions > 0 else 0.0
            holding_cost = FUNDING_PER_12H * (rebal_hours / 12)
            total_cost = turnover_cost + holding_cost

            longs = grp[grp["symbol"].isin(new_longs)].copy()
            shorts = grp[grp["symbol"].isin(new_shorts)].copy()
            long_ret = float(longs["fwd_ret"].mean()) if len(longs) else 0.0
            short_ret = float(shorts["fwd_ret"].mean()) if len(shorts) else 0.0

            if nl > 0 and ns > 0:
                port_ret = 0.5 * long_ret - 0.5 * short_ret
            elif ns > 0:
                port_ret = -short_ret
            else:
                port_ret = long_ret
            port_ret *= exposure
            port_ret -= total_cost

            for _, row in longs.iterrows():
                contrib_records.append({
                    "window": window,
                    "timestamp": ts,
                    "symbol": row["symbol"],
                    "side": "long",
                    "fwd_ret": float(row["fwd_ret"]),
                    "gross_contribution": (0.5 / max(len(longs), 1)) * float(row["fwd_ret"]) * exposure,
                    "is_new": row["symbol"] in new_opened,
                })
            for _, row in shorts.iterrows():
                contrib_records.append({
                    "window": window,
                    "timestamp": ts,
                    "symbol": row["symbol"],
                    "side": "short",
                    "fwd_ret": float(row["fwd_ret"]),
                    "gross_contribution": (-0.5 / max(len(shorts), 1)) * float(row["fwd_ret"]) * exposure,
                    "is_new": row["symbol"] in new_opened,
                })

            records.append({
                "window": window,
                "timestamp": ts,
                "portfolio_ret": port_ret,
                "gross_ret": port_ret + total_cost,
                "long_ret": long_ret,
                "short_ret": short_ret,
                "long_leg_ret": 0.5 * long_ret * exposure if len(longs) else 0.0,
                "short_leg_ret": -0.5 * short_ret * exposure if len(shorts) else 0.0,
                "turnover": turnover_count,
                "cost": total_cost,
                "exposure": exposure,
                "n_long": len(longs),
                "n_short": len(shorts),
                "longs": ",".join(sorted(new_longs)),
                "shorts": ",".join(sorted(new_shorts)),
            })

            prev_longs = new_longs
            prev_shorts = new_shorts

    return pd.DataFrame(records), pd.DataFrame(contrib_records)


def annualized_sharpe(series: pd.Series, timestamps: pd.Series) -> float:
    values = series.dropna().values
    if len(values) < 3:
        return np.nan
    mean_r = float(np.mean(values))
    std_r = float(np.std(values)) + 1e-10
    total_hours = (timestamps.max() - timestamps.min()).total_seconds() / 3600
    years = max(total_hours / 8760, 0.01)
    ppy = len(values) / years
    return mean_r / std_r * np.sqrt(ppy)


def leg_sharpe_report(trace: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window in ["W1", "W2", "W3"]:
        sub = trace.loc[trace["window"] == window].copy()
        if sub.empty:
            continue
        if ((sub["n_long"] > 0) & (sub["n_short"] > 0)).any():
            half_cost = 0.5 * sub["cost"]
        else:
            half_cost = sub["cost"]
        rows.append({
            "window": window,
            "long_leg_sharpe": annualized_sharpe(sub["long_leg_ret"] - half_cost, sub["timestamp"]),
            "short_leg_sharpe": annualized_sharpe(sub["short_leg_ret"] - half_cost, sub["timestamp"]),
            "portfolio_sharpe": annualized_sharpe(sub["portfolio_ret"], sub["timestamp"]),
            "avg_turnover": float(sub["turnover"].mean()),
            "avg_cost_pct": float(sub["cost"].mean() * 100),
        })
    return pd.DataFrame(rows)


def rank_diagnostics(preds: pd.DataFrame, windows: list[str]) -> pd.DataFrame:
    rows = []
    for window in windows:
        wsub = preds[preds["window"] == window].copy()
        timestamps = sorted(wsub["timestamp"].unique())[::CFG["rebal_hours"]]
        prev_rank = None
        prev_top = None
        prev_bottom = None
        corrs = []
        top_changes = []
        bottom_changes = []
        for ts in timestamps:
            grp = wsub[wsub["timestamp"] == ts].copy()
            grp["rank"] = grp["pred"].rank(ascending=False, method="average")
            current_rank = grp.set_index("symbol")["rank"]
            top_set = set(grp.nsmallest(CFG["n_long"], "rank")["symbol"].tolist())
            bottom_set = set(grp.nlargest(CFG["n_short"], "rank")["symbol"].tolist())
            if prev_rank is not None:
                common = current_rank.index.intersection(prev_rank.index)
                corr = stats.spearmanr(current_rank.loc[common], prev_rank.loc[common])[0]
                if pd.notna(corr):
                    corrs.append(float(corr))
                top_overlap = len(top_set & prev_top)
                bottom_overlap = len(bottom_set & prev_bottom)
                top_changes.append(1.0 - top_overlap / max(len(top_set), 1))
                bottom_changes.append(1.0 - bottom_overlap / max(len(bottom_set), 1))
            prev_rank = current_rank
            prev_top = top_set
            prev_bottom = bottom_set
        rows.append({
            "window": window,
            "mean_rank_corr": np.mean(corrs) if corrs else np.nan,
            "mean_top_change_ratio": np.mean(top_changes) if top_changes else np.nan,
            "mean_bottom_change_ratio": np.mean(bottom_changes) if bottom_changes else np.nan,
            "rebalances": len(timestamps),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="baseline")
    args = parser.parse_args()

    label, feature_cols = FEATURE_SETS[args.feature_set]
    conditional_ic_path = BASE_DIR / "results_r34_w2_conditional_ic.csv"
    coin_contrib_path = BASE_DIR / "results_r34_w2_coin_contrib.csv"
    feature_importance_path = BASE_DIR / "results_r34_feature_importance.csv"

    print("=" * 80)
    print(f"R34 — W2 ATTRIBUTION ({args.feature_set})")
    print("=" * 80)

    print("\n[1] Loading research frame...")
    df, regime_df = load_research_frame()
    regime_df = regime_df.sort_index()
    print(f"  Data: {len(df):,} rows, {len(df.columns)} cols")

    print("\n[2] Training ensemble predictions...")
    preds = train_ensemble(df, feature_cols, WINDOWS, l2=1.0, rolling=False, label=label)
    if preds is None or preds.empty:
        raise RuntimeError("No predictions produced")

    print("\n[3] Building W2 frame...")
    context = build_market_context(df)
    w2 = next(w for w in WINDOWS if w["name"] == "W2")
    w2_start = pd.Timestamp(w2["test_start"], tz=df["timestamp"].dt.tz)
    w2_end = pd.Timestamp(w2["test_end"], tz=df["timestamp"].dt.tz)
    w2_df = df[(df["timestamp"] >= w2_start) & (df["timestamp"] <= w2_end)].copy()
    w2_df = w2_df.merge(context, on="timestamp", how="left")
    print(f"  W2 rows: {len(w2_df):,}")

    print("\n[4] Conditional IC report...")
    conditional_ic = conditional_ic_report(w2_df, feature_cols, conditional_ic_path)
    for regime in ["btc_rvol_24h", "breadth_12h", "dispersion_12h", "market_funding_24h"]:
        print(f"\n  Top mean timestamp IC by {regime} bin:")
        sub = conditional_ic[conditional_ic["regime"] == regime]
        for bin_name in ["low", "mid", "high"]:
            top = sub[sub["bin"] == bin_name].dropna(subset=["mean_ts_ic"]).head(5)
            if top.empty:
                continue
            print(f"    {bin_name}:")
            for _, row in top.iterrows():
                print(f"      {row['feature']:<22} mean_ts_ic={row['mean_ts_ic']:+.4f} pooled_ic={row['pooled_ic']:+.4f}")

    print("\n[5] Position trace + leg decomposition...")
    trace, contrib = build_position_trace(preds, regime_df, CFG)
    leg_report = leg_sharpe_report(trace)
    print(leg_report.to_string(index=False))

    print("\n[5b] Official simulation cross-check...")
    official_rows = []
    for window in ["W1", "W2", "W3"]:
        port = simulate_with_costs(preds[preds["window"] == window].copy(), regime_df, CFG)
        metric = eval_with_costs(port, f"{label}_{window}")
        trace_metric = leg_report.loc[leg_report["window"] == window, "portfolio_sharpe"]
        official_rows.append({
            "window": window,
            "official_sharpe": metric["sharpe"],
            "trace_sharpe": float(trace_metric.iloc[0]) if not trace_metric.empty else np.nan,
            "official_turnover": metric.get("avg_turnover", np.nan),
            "official_cost_pct": metric.get("total_cost_pct", np.nan),
        })
    official_df = pd.DataFrame(official_rows)
    official_df["delta_trace_vs_official"] = official_df["trace_sharpe"] - official_df["official_sharpe"]
    print(official_df.to_string(index=False))

    print("\n[6] Coin contribution report (W2 gross contribution)...")
    w2_contrib = contrib[contrib["window"] == "W2"].copy()
    coin_contrib = (
        w2_contrib.groupby(["symbol", "side"])
        .agg(
            gross_contribution=("gross_contribution", "sum"),
            mean_fwd_ret=("fwd_ret", "mean"),
            appearances=("timestamp", "count"),
            new_entries=("is_new", "sum"),
        )
        .reset_index()
        .sort_values("gross_contribution")
    )
    coin_contrib.to_csv(coin_contrib_path, index=False)
    print("  Worst contributors:")
    print(coin_contrib.head(10).to_string(index=False))
    print("\n  Best contributors:")
    print(coin_contrib.tail(10).sort_values("gross_contribution", ascending=False).to_string(index=False))

    print("\n[7] Rank stability diagnostics...")
    rank_diag = rank_diagnostics(preds, ["W2", "W3"])
    print(rank_diag.to_string(index=False))

    print("\n[8] Per-window feature importance...")
    feature_importance = per_window_feature_importance(df, feature_cols, feature_importance_path)
    for window in ["W1", "W2", "W3"]:
        top = feature_importance[feature_importance["window"] == window].head(8)
        if top.empty:
            continue
        print(f"  {window}:")
        print(
            top[["feature", "avg_gain", "gain_pct", "test_mean_ts_ic", "test_pooled_ic"]].to_string(
                index=False,
                formatters={
                    "avg_gain": lambda value: f"{value:,.1f}",
                    "gain_pct": lambda value: f"{value * 100:.1f}%",
                    "test_mean_ts_ic": lambda value: f"{value:+.4f}",
                    "test_pooled_ic": lambda value: f"{value:+.4f}",
                },
            )
        )

    print("\n[9] Saved artifacts")
    print(f"  Conditional IC CSV: {conditional_ic_path.name}")
    print(f"  Coin contribution CSV: {coin_contrib_path.name}")
    print(f"  Feature importance CSV: {feature_importance_path.name}")
    print("\nVerdict focus:")
    print("- Compare W2 long_leg_sharpe vs short_leg_sharpe")
    print("- Compare W2 rank churn vs W3 rank churn")
    print("- Inspect whether specific coins dominate the negative contribution tail")
    print("- Check whether W2 high-gain features also keep positive post-hoc test IC")


if __name__ == "__main__":
    main()