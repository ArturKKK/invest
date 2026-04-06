#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R46 — separate long/short models.

Hypothesis:
  signals for long selection and short selection are asymmetric, so a
  single unified ranker is leaving alpha on the table.

Setup:
  - unified baseline: standard train_ensemble on the same feature set
  - long model target: fwd_ret_12h > cross-sectional median at timestamp
  - short model target: fwd_ret_12h < cross-sectional 25th percentile at timestamp
  - long book from model_long, short book from model_short

Outputs:
  - results_r46.log
  - results_r46_asymmetric_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats

from _research_round7 import WINDOWS
from _research_r22_models import SEEDS, cs_rank_cols
from _research_r30b_fixed import compute_regime_extended, eval_with_costs, simulate_with_costs
from _research_r33_creative_features import FEAT_28
from _research_r35_new_features import MARKET_LEVEL_FEATURES, add_r35_features, load_research_frame
from _research_r30b_fixed import train_ensemble


BASE_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = BASE_DIR / "results_r46_asymmetric_summary.csv"
IMPORTANCE_PATH = BASE_DIR / "results_r46_feature_importance.csv"

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

FEATURE_SETS = {
    "baseline": FEAT_28,
    "r42_candidate": FEAT_28 + R42_CANDIDATE,
}


def build_feature_set(features: Sequence[str]) -> Tuple[List[str], List[str]]:
    feats = list(dict.fromkeys(features))
    no_rank = [feature for feature in feats if feature in MARKET_LEVEL_FEATURES]
    return feats, no_rank


def add_asymmetric_targets(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["cs_median"] = out.groupby("timestamp")["fwd_ret_12h"].transform("median")
    out["cs_p25"] = out.groupby("timestamp")["fwd_ret_12h"].transform(lambda s: s.quantile(0.25))
    out["target_long"] = (out["fwd_ret_12h"] > out["cs_median"]).astype(int)
    out["target_short"] = (out["fwd_ret_12h"] < out["cs_p25"]).astype(int)
    return out


def train_binary_target(
    df: pd.DataFrame,
    feats: Sequence[str],
    windows: Sequence[Dict[str, str]],
    target_col: str,
    label: str,
    seeds: Sequence[int],
    cs_rank_exclude: Sequence[str] | None = None,
) -> Tuple[pd.DataFrame | None, pd.DataFrame]:
    avail = [feature for feature in feats if feature in df.columns]
    rank_exclude = set(cs_rank_exclude or [])
    rank_feats = [feature for feature in avail if feature not in rank_exclude]
    tz = df["timestamp"].dt.tz
    all_lgb = []
    all_xgb = []
    importance_rows = []

    print(f"  Training {label}: {len(avail)}f, {len(seeds)} seeds × {len(windows)} windows")

    for seed in seeds:
        params_lgb = {
            "objective": "binary", "metric": "auc",
            "learning_rate": 0.03, "num_leaves": 63,
            "min_child_samples": 100, "subsample": 0.8,
            "colsample_bytree": 0.8, "lambda_l2": 1.0,
            "verbose": -1, "n_jobs": -1, "seed": seed,
        }
        params_xgb = {
            "objective": "binary:logistic", "eval_metric": "auc",
            "learning_rate": 0.03, "max_depth": 6,
            "min_child_weight": 100, "subsample": 0.8,
            "colsample_bytree": 0.8, "reg_lambda": 1.0,
            "seed": seed, "n_jobs": -1, "verbosity": 0,
        }

        for window in windows:
            te_end = pd.Timestamp(window["test_end"], tz=tz)
            te_start = pd.Timestamp(window["test_start"], tz=tz)
            tr_end = pd.Timestamp(window["train_end"], tz=tz)
            va_start = pd.Timestamp(window["val_start"], tz=tz)
            va_end = pd.Timestamp(window["val_end"], tz=tz)

            train_ = df[df["timestamp"] < tr_end].copy()
            val_ = df[(df["timestamp"] >= va_start) & (df["timestamp"] < va_end)].copy()
            test_ = df[(df["timestamp"] >= te_start) & (df["timestamp"] <= te_end)].copy()

            if len(train_) < 5000 or len(test_) < 200:
                continue

            if rank_feats:
                train_ = cs_rank_cols(train_, rank_feats)
                val_ = cs_rank_cols(val_, rank_feats)
                test_ = cs_rank_cols(test_, rank_feats)

            for col in avail:
                for d in [train_, val_, test_]:
                    if d[col].isna().any():
                        d[col] = d[col].fillna(0)

            tr = train_[avail + [target_col]].replace([np.inf, -np.inf], np.nan).dropna()
            va = val_[avail + [target_col]].replace([np.inf, -np.inf], np.nan).dropna()
            te = test_[avail + [target_col, "timestamp", "symbol"]].replace([np.inf, -np.inf], np.nan).dropna()
            fwd = test_[["timestamp", "symbol", "fwd_ret_12h"]].rename(columns={"fwd_ret_12h": "fwd_ret"}).dropna()
            if len(tr) == 0 or len(va) == 0 or len(te) == 0:
                continue

            dt = lgb.Dataset(tr[avail], label=tr[target_col])
            dv = lgb.Dataset(va[avail], label=va[target_col])
            model_lgb = lgb.train(
                params_lgb,
                dt,
                num_boost_round=600,
                valid_sets=[dv],
                callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)],
            )
            for feature, gain in zip(avail, model_lgb.feature_importance(importance_type="gain")):
                importance_rows.append({
                    "target": label,
                    "model": "lgb",
                    "window": window["name"],
                    "seed": seed,
                    "feature": feature,
                    "importance": float(gain),
                })
            pred_lgb = model_lgb.predict(te[avail])
            rec = te[["timestamp", "symbol"]].copy()
            rec["pred_lgb"] = pred_lgb
            rec = rec.merge(fwd, on=["timestamp", "symbol"], how="inner")
            rec["window"] = window["name"]
            rec["seed"] = seed
            all_lgb.append(rec)

            dt_x = xgb.DMatrix(tr[avail], label=tr[target_col])
            dv_x = xgb.DMatrix(va[avail], label=va[target_col])
            model_xgb = xgb.train(
                params_xgb,
                dt_x,
                num_boost_round=600,
                evals=[(dv_x, "val")],
                early_stopping_rounds=40,
                verbose_eval=False,
            )
            xgb_gain = model_xgb.get_score(importance_type="gain")
            for feature in avail:
                importance_rows.append({
                    "target": label,
                    "model": "xgb",
                    "window": window["name"],
                    "seed": seed,
                    "feature": feature,
                    "importance": float(xgb_gain.get(feature, 0.0)),
                })
            pred_xgb = model_xgb.predict(xgb.DMatrix(te[avail]))
            rec2 = te[["timestamp", "symbol"]].copy()
            rec2["pred_xgb"] = pred_xgb
            rec2 = rec2.merge(fwd, on=["timestamp", "symbol"], how="inner")
            rec2["window"] = window["name"]
            rec2["seed"] = seed
            all_xgb.append(rec2)

            if seed == seeds[0]:
                ic = stats.spearmanr(rec["pred_lgb"], rec["fwd_ret"])[0]
                print(f"    {window['name']}/s{seed}: train={len(tr):,}, test={len(te):,}, IC={ic:.4f}")

    if not all_lgb:
        return None, pd.DataFrame(importance_rows)

    lgb_df = pd.concat(all_lgb)
    xgb_df = pd.concat(all_xgb)
    lgb_avg = lgb_df.groupby(["timestamp", "symbol"]).agg(
        pred_lgb=("pred_lgb", "mean"), fwd_ret=("fwd_ret", "first"), window=("window", "first")
    ).reset_index()
    xgb_avg = xgb_df.groupby(["timestamp", "symbol"]).agg(pred_xgb=("pred_xgb", "mean")).reset_index()
    merged = lgb_avg.merge(xgb_avg, on=["timestamp", "symbol"], how="inner")
    merged["rank_lgb"] = merged.groupby("timestamp")["pred_lgb"].rank(pct=True) - 0.5
    merged["rank_xgb"] = merged.groupby("timestamp")["pred_xgb"].rank(pct=True) - 0.5
    merged["pred"] = 0.5 * merged["rank_lgb"] + 0.5 * merged["rank_xgb"]
    importance_df = pd.DataFrame(importance_rows)
    return merged[["timestamp", "symbol", "pred", "fwd_ret", "window"]], importance_df


def summarize_importance(importance_df: pd.DataFrame) -> pd.DataFrame:
    if importance_df.empty:
        return importance_df

    summary = (
        importance_df.groupby(["target", "feature"], as_index=False)
        .agg(mean_importance=("importance", "mean"))
    )
    summary["target_total"] = summary.groupby("target")["mean_importance"].transform("sum")
    summary["importance_share"] = np.where(
        summary["target_total"] > 0,
        summary["mean_importance"] / summary["target_total"],
        0.0,
    )
    pivot = summary.pivot(index="feature", columns="target", values="importance_share").fillna(0.0)
    target_cols = [col for col in pivot.columns]
    if len(target_cols) >= 2:
        pivot["importance_gap"] = pivot[target_cols[0]] - pivot[target_cols[1]]
    else:
        pivot["importance_gap"] = 0.0
    out = pivot.reset_index()
    rename_map = {target: f"importance_share_{target}" for target in target_cols}
    out = out.rename(columns=rename_map)
    return out.sort_values("importance_gap", ascending=False)


def combine_long_short(long_preds: pd.DataFrame, short_preds: pd.DataFrame) -> pd.DataFrame:
    merged = long_preds.rename(columns={"pred": "pred_long"}).merge(
        short_preds[["timestamp", "symbol", "window", "pred"]].rename(columns={"pred": "pred_short"}),
        on=["timestamp", "symbol", "window"],
        how="inner",
    )
    return merged[["timestamp", "symbol", "window", "pred_long", "pred_short", "fwd_ret"]].copy()


def simulate_dual_model(preds: pd.DataFrame, regime_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if preds.empty:
        return pd.DataFrame(columns=[
            "timestamp", "portfolio_ret", "gross_ret", "turnover", "cost",
            "n_long", "n_short", "exposure",
        ])

    trend_cutoff = cfg.get("trend_cutoff", 0.9)
    rebal_hours = cfg.get("rebal_hours", 12)
    ema_alpha = cfg.get("ema_alpha")
    hysteresis = cfg.get("hysteresis", 0)
    dyn_threshold = cfg.get("dyn_threshold", 0.7)
    n_long = cfg.get("n_long", 6)
    n_short = cfg.get("n_short", 3)

    all_rets = []
    prev_longs: set[str] = set()
    prev_shorts: set[str] = set()
    prev_long_scores: dict[str, float] = {}
    prev_short_scores: dict[str, float] = {}
    grouped = {timestamp: group.copy() for timestamp, group in preds.groupby("timestamp")}
    rebal_timestamps = sorted(preds["timestamp"].unique())[::rebal_hours]

    for timestamp in rebal_timestamps:
        if timestamp not in grouped or timestamp not in regime_df.index:
            continue
        trend_strength = regime_df.loc[timestamp].get("trend_strength", 0.0)
        if trend_strength > trend_cutoff:
            continue

        group = grouped[timestamp].copy()
        n = len(group)
        if n == 0:
            continue

        if ema_alpha is not None and ema_alpha < 1.0:
            for index, row in group.iterrows():
                symbol = row["symbol"]
                raw_long = row["pred_long"]
                raw_short = row["pred_short"]
                smooth_long = ema_alpha * raw_long + (1 - ema_alpha) * prev_long_scores.get(symbol, raw_long)
                smooth_short = ema_alpha * raw_short + (1 - ema_alpha) * prev_short_scores.get(symbol, raw_short)
                prev_long_scores[symbol] = smooth_long
                prev_short_scores[symbol] = smooth_short
                group.at[index, "pred_long"] = smooth_long
                group.at[index, "pred_short"] = smooth_short

        nl = min(n_long, n // 3)
        ns = min(n_short, n // 3)
        if nl == 0 and ns == 0:
            continue

        exposure = 1.0
        if dyn_threshold is not None and trend_strength > dyn_threshold:
            exposure = max(0.1, 1.0 - (trend_strength - dyn_threshold) /
                           (trend_cutoff - dyn_threshold + 1e-10) * 0.5)

        long_rank = group.sort_values("pred_long", ascending=False).reset_index(drop=True)
        long_rank["rank_long"] = long_rank.index + 1
        short_rank = group.sort_values("pred_short", ascending=False).reset_index(drop=True)
        short_rank["rank_short"] = short_rank.index + 1

        if hysteresis > 0 and (prev_longs or prev_shorts):
            new_longs = set(long_rank[(long_rank["symbol"].isin(prev_longs)) & (long_rank["rank_long"] <= nl + hysteresis)]["symbol"].tolist())
            new_shorts = set(short_rank[(short_rank["symbol"].isin(prev_shorts)) & (short_rank["rank_short"] <= ns + hysteresis)]["symbol"].tolist())
            remain_long = nl - len(new_longs)
            remain_short = ns - len(new_shorts)
            if remain_long > 0:
                long_candidates = long_rank[~long_rank["symbol"].isin(new_longs | new_shorts)]
                new_longs |= set(long_candidates.head(remain_long)["symbol"].tolist())
            if remain_short > 0:
                short_candidates = short_rank[~short_rank["symbol"].isin(new_longs | new_shorts)]
                new_shorts |= set(short_candidates.head(remain_short)["symbol"].tolist())
        else:
            new_longs = set(long_rank.head(nl)["symbol"].tolist()) if nl > 0 else set()
            short_candidates = short_rank[~short_rank["symbol"].isin(new_longs)]
            new_shorts = set(short_candidates.head(ns)["symbol"].tolist()) if ns > 0 else set()

        new_opened = (new_longs - prev_longs) | (new_shorts - prev_shorts)
        closed = (prev_longs - new_longs) | (prev_shorts - new_shorts)
        turnover_count = len(new_opened) + len(closed)
        total_positions = len(new_longs) + len(new_shorts)
        avg_weight = (1.0 / total_positions) if total_positions > 0 else 0.0
        turnover_cost = turnover_count * (0.0005 + 0.0002) * avg_weight if total_positions > 0 else 0.0
        holding_cost = 0.00008 * (rebal_hours / 12)
        total_cost = turnover_cost + holding_cost

        longs = group[group["symbol"].isin(new_longs)].copy()
        shorts = group[group["symbol"].isin(new_shorts)].copy()
        long_ret = float(longs["fwd_ret"].mean()) if len(longs) else 0.0
        short_ret = float(shorts["fwd_ret"].mean()) if len(shorts) else 0.0

        if nl > 0 and ns > 0:
            portfolio_ret = 0.5 * long_ret - 0.5 * short_ret
        elif ns > 0:
            portfolio_ret = -short_ret
        else:
            portfolio_ret = long_ret
        portfolio_ret *= exposure
        portfolio_ret -= total_cost

        all_rets.append({
            "timestamp": timestamp,
            "portfolio_ret": portfolio_ret,
            "gross_ret": portfolio_ret + total_cost,
            "turnover": turnover_count,
            "cost": total_cost,
            "n_long": len(new_longs),
            "n_short": len(new_shorts),
            "exposure": exposure,
        })

        prev_longs = new_longs
        prev_shorts = new_shorts

    return pd.DataFrame(all_rets)


def evaluate_dual_model(
    preds: pd.DataFrame,
    regime_df: pd.DataFrame,
    label: str,
    window_names: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for window in [*window_names, "ALL"]:
        subset = preds if window == "ALL" else preds[preds["window"] == window].copy()
        port = simulate_dual_model(subset, regime_df, CFG)
        out[window] = eval_with_costs(port, f"{label}_{window}")
    return out


def summarize(label: str, results: Dict[str, Dict[str, float]], window_names: Sequence[str]) -> List[Dict[str, object]]:
    rows = []
    for window in [*window_names, "ALL"]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="r42_candidate")
    parser.add_argument("--max-seeds", type=int, default=None)
    parser.add_argument("--windows", nargs="*", default=None)
    args = parser.parse_args()

    selected_windows = WINDOWS
    if args.windows:
        window_names = set(args.windows)
        selected_windows = [window for window in WINDOWS if window["name"] in window_names]
        if not selected_windows:
            raise ValueError(f"No matching windows for {sorted(window_names)}")

    selected_seeds = SEEDS[: args.max_seeds] if args.max_seeds else SEEDS
    if not selected_seeds:
        raise ValueError("No seeds selected")
    window_names = [window["name"] for window in selected_windows]

    print("=" * 80)
    print("R46 — SEPARATE LONG/SHORT MODELS")
    print("=" * 80)
    print(f"Feature set: {args.feature_set}")
    print(f"Windows: {window_names}")
    print(f"Seeds: {selected_seeds}")
    print(CFG)

    print("\n[1] Loading research frame...")
    df, regime_df = load_research_frame()
    df, _ = add_r35_features(df)
    df = add_asymmetric_targets(df)
    regime_df = compute_regime_extended(df).sort_index()
    print(f"  Data: {len(df):,} rows, {len(df.columns)} cols")

    feats, no_rank = build_feature_set(FEATURE_SETS[args.feature_set])

    print("\n[2] Training unified baseline...")
    unified = train_ensemble(
        df,
        feats,
        selected_windows,
        seeds=selected_seeds,
        l2=1.0,
        rolling=False,
        label=f"unified_{args.feature_set}",
        cs_rank_exclude=no_rank,
    )
    if unified is None or unified.empty:
        raise RuntimeError("No unified predictions")

    print("\n[3] Training asymmetric experts...")
    long_preds, long_importance = train_binary_target(
        df,
        feats,
        selected_windows,
        target_col="target_long",
        label=f"long_{args.feature_set}",
        seeds=selected_seeds,
        cs_rank_exclude=no_rank,
    )
    short_preds, short_importance = train_binary_target(
        df,
        feats,
        selected_windows,
        target_col="target_short",
        label=f"short_{args.feature_set}",
        seeds=selected_seeds,
        cs_rank_exclude=no_rank,
    )
    if long_preds is None or short_preds is None:
        raise RuntimeError("Missing asymmetric predictions")
    dual_preds = combine_long_short(long_preds, short_preds)

    print("\n[4] Evaluating unified vs asymmetric...")
    unified_results = {}
    for window in [*window_names, "ALL"]:
        subset = unified if window == "ALL" else unified[unified["window"] == window].copy()
        port = simulate_with_costs(subset, regime_df, CFG)
        unified_results[window] = eval_with_costs(port, f"unified_{window}")
    dual_results = evaluate_dual_model(dual_preds, regime_df, "asymmetric", window_names)

    rows = []
    rows.extend(summarize("unified", unified_results, window_names))
    rows.extend(summarize("asymmetric", dual_results, window_names))
    result_df = pd.DataFrame(rows).sort_values(["window", "sharpe"], ascending=[True, False])
    result_df.to_csv(SUMMARY_PATH, index=False)

    importance_df = summarize_importance(pd.concat([long_importance, short_importance], ignore_index=True))
    importance_df.to_csv(IMPORTANCE_PATH, index=False)

    print("\n[5] Comparison")
    report_windows = [window for window in ["W2", "W3"] if window in window_names] + ["ALL"]
    for window in report_windows:
        top = result_df[result_df["window"] == window]
        print(f"  {window}:")
        print(top[["config", "sharpe", "cost_pct", "avg_turnover"]].to_string(index=False))

    if not importance_df.empty:
        print("\n  Top long-skew features:")
        print(importance_df.head(8).to_string(index=False))

    print("\n[6] Saved artifacts")
    print(f"  Summary CSV: {SUMMARY_PATH.name}")
    print(f"  Importance CSV: {IMPORTANCE_PATH.name}")


if __name__ == "__main__":
    main()