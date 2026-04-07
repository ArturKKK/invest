#!/usr/bin/env python3
"""
R70 — LambdaRank / Ranking Objective

Current system trains on binary classification (fwd_ret > 0).
But we care about **rank** of top-4/bottom-2 coins.
LambdaRank optimizes NDCG@K — directly what we need.

LGB: objective="lambdarank"
XGB: objective="rank:ndcg"
Group: timestamp (each group = one cross-section of coins)
Label: cross-sectional rank of fwd_ret_12h (scaled to 0..4 relevance grades)

Compare with binary baseline on 4L/2S and 6L/3S.
"""

import sys, warnings, time
from typing import Dict, Set

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats

warnings.filterwarnings("ignore")

from _research_round7 import SYM_35
from _research_r22_models import SEEDS, LEVERAGE, CAPITAL, log, cs_rank_cols
from _research_r30b_fixed import compute_regime_extended
from _research_r35_new_features import (
    add_r35_features, load_research_frame, MARKET_LEVEL_FEATURES,
)
from _research_r47_coinglass import (
    CHAMPION_FEAT_30, add_cg_features, compute_cg_features, load_cg_daily,
)

CHAMPION_FEAT_31 = CHAMPION_FEAT_30 + ["cg_taker_imb"]

TIER1_SYMS = {"BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"}
TIER3_SYMS = {
    "SAND/USDT", "LDO/USDT", "INJ/USDT", "APT/USDT", "ARB/USDT",
    "GALA/USDT", "FTM/USDT", "MATIC/USDT",
}
TIER2_SYMS = set(SYM_35) - TIER1_SYMS - TIER3_SYMS

def _cost_for_sym(sym):
    if sym in TIER1_SYMS: return 0.92 * (-0.0001) + 0.08 * 0.0007
    elif sym in TIER2_SYMS: return 0.75 * 0.0001 + 0.25 * 0.0007
    else: return 0.0005 + 0.0002

ORIGINAL_WINDOWS = [
    {"name": "W1", "train_end": "2024-06-01",
     "val_start": "2024-06-01", "val_end": "2024-09-30",
     "test_start": "2024-10-15", "test_end": "2025-01-31"},
    {"name": "W2", "train_end": "2025-01-01",
     "val_start": "2025-01-01", "val_end": "2025-04-30",
     "test_start": "2025-05-15", "test_end": "2025-08-31"},
    {"name": "W3", "train_end": "2025-07-01",
     "val_start": "2025-07-01", "val_end": "2025-10-31",
     "test_start": "2025-11-15", "test_end": "2026-03-17"},
]

PROD_CFG = {
    "n_long": 4, "n_short": 2, "rebal_hours": 12,
    "trend_cutoff": 0.9, "dyn_threshold": 0.7,
    "ema_alpha": 0.5, "hysteresis": 3,
}

# Binary baseline params (same as R65)
LGB_BINARY = {
    "objective": "binary", "metric": "auc",
    "learning_rate": 0.03, "num_leaves": 63,
    "min_child_samples": 100, "subsample": 0.8,
    "colsample_bytree": 0.8, "lambda_l2": 1.0,
    "verbose": -1, "n_jobs": -1,
}
XGB_BINARY = {
    "objective": "binary:logistic", "eval_metric": "auc",
    "learning_rate": 0.03, "max_depth": 6,
    "min_child_weight": 100, "subsample": 0.8,
    "colsample_bytree": 0.8, "reg_lambda": 1.0,
    "n_jobs": -1, "verbosity": 0,
}

# LambdaRank params
LGB_RANK = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [2, 4, 6],
    "learning_rate": 0.03, "num_leaves": 63,
    "min_child_samples": 100, "subsample": 0.8,
    "colsample_bytree": 0.8, "lambda_l2": 1.0,
    "verbose": -1, "n_jobs": -1,
}
XGB_RANK = {
    "objective": "rank:ndcg",
    "eval_metric": "ndcg@4",
    "learning_rate": 0.03, "max_depth": 6,
    "min_child_weight": 100, "subsample": 0.8,
    "colsample_bytree": 0.8, "reg_lambda": 1.0,
    "n_jobs": -1, "verbosity": 0,
}

N_ROUNDS = 600
EARLY_STOP = 40


def load_data():
    print("=" * 70)
    print("  LOADING DATA")
    print("=" * 70)
    df, regime_df = load_research_frame()
    df, _ = add_r35_features(df)
    cg = load_cg_daily()
    cg_feats = compute_cg_features(cg)
    df, _, _ = add_cg_features(df, cg_feats)
    present = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    missing = [f for f in CHAMPION_FEAT_31 if f not in df.columns]
    if missing:
        print(f"  WARNING: Missing features: {missing}")
        CHAMPION_FEAT_31[:] = present
    print(f"  Frame: {len(df):,} rows, {df['symbol'].nunique()} symbols")
    return df, regime_df


def _make_relevance_labels(group_fwd_ret: np.ndarray, n_grades=5) -> np.ndarray:
    """Convert fwd_ret values within a group to relevance grades 0..n_grades-1.
    Higher fwd_ret → higher relevance grade.
    """
    n = len(group_fwd_ret)
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    # Rank from 0 to n-1 (ascending)
    order = np.argsort(np.argsort(group_fwd_ret))  # double argsort = rank
    # Scale to 0..n_grades-1 (must be int for LambdaRank)
    grades = (order * n_grades // n).astype(np.int32)
    return np.clip(grades, 0, n_grades - 1)


def train_ensemble_ranking(df, feats, windows, seeds, objective="rank"):
    """Train ensemble with ranking or binary objective.
    objective: "rank" for LambdaRank, "binary" for baseline.
    """
    avail = [f for f in feats if f in df.columns]
    rank_exclude = set(f for f in avail if f in MARKET_LEVEL_FEATURES)
    rank_feats = [f for f in avail if f not in rank_exclude]
    tz = df["timestamp"].dt.tz
    all_preds = []

    for seed in seeds:
        if objective == "rank":
            lgb_p = {**LGB_RANK, "seed": seed}
            xgb_p = {**XGB_RANK, "seed": seed}
        else:
            lgb_p = {**LGB_BINARY, "seed": seed}
            xgb_p = {**XGB_BINARY, "seed": seed}

        for w in windows:
            te_end = pd.Timestamp(w["test_end"], tz=tz)
            te_start = pd.Timestamp(w["test_start"], tz=tz)
            tr_end = pd.Timestamp(w["train_end"], tz=tz)
            va_start = pd.Timestamp(w["val_start"], tz=tz)
            va_end = pd.Timestamp(w["val_end"], tz=tz)

            train_ = df[df["timestamp"] < tr_end].copy()
            val_ = df[(df["timestamp"] >= va_start) & (df["timestamp"] < va_end)].copy()
            test_ = df[(df["timestamp"] >= te_start) & (df["timestamp"] <= te_end)].copy()
            if len(train_) < 5000 or len(test_) < 200:
                continue

            if rank_feats:
                train_ = cs_rank_cols(train_, rank_feats)
                val_ = cs_rank_cols(val_, rank_feats)
                test_ = cs_rank_cols(test_, rank_feats)

            # Binary target (for binary baseline)
            for d in [train_, val_, test_]:
                d["target_binary"] = (d["fwd_ret_12h"] > 0).astype(int)

            for col in avail:
                for d in [train_, val_, test_]:
                    if d[col].isna().any():
                        d[col] = d[col].fillna(0)

            tr = train_[avail + ["target_binary", "fwd_ret_12h", "timestamp"]].dropna()
            va = val_[avail + ["target_binary", "fwd_ret_12h", "timestamp"]].dropna()
            te = test_[avail + ["target_binary", "timestamp", "symbol", "fwd_ret_12h"]].dropna()
            fwd = test_[["timestamp", "symbol", "fwd_ret_12h"]].rename(
                columns={"fwd_ret_12h": "fwd_ret"}).dropna()

            for d in [tr, va, te]:
                d.replace([np.inf, -np.inf], np.nan, inplace=True)
            tr, va, te = tr.dropna(), va.dropna(), te.dropna()
            if len(te) == 0:
                continue

            if objective == "rank":
                # Create relevance labels per timestamp group
                tr_labels = np.concatenate([
                    _make_relevance_labels(g["fwd_ret_12h"].values)
                    for _, g in tr.groupby("timestamp")
                ])
                va_labels = np.concatenate([
                    _make_relevance_labels(g["fwd_ret_12h"].values)
                    for _, g in va.groupby("timestamp")
                ])
                # Group sizes
                tr_groups = tr.groupby("timestamp").size().values
                va_groups = va.groupby("timestamp").size().values

                # LightGBM ranking
                dt = lgb.Dataset(tr[avail], label=tr_labels, group=tr_groups)
                dv = lgb.Dataset(va[avail], label=va_labels, group=va_groups)
                m = lgb.train(lgb_p, dt, num_boost_round=N_ROUNDS,
                              valid_sets=[dv],
                              callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                                         lgb.log_evaluation(-1)])
                p_lgb = m.predict(te[avail])

                # XGBoost ranking
                te_groups = te.groupby("timestamp").size().values
                te_labels_xgb = np.concatenate([
                    _make_relevance_labels(g["fwd_ret_12h"].values)
                    for _, g in te.groupby("timestamp")
                ])

                # For XGB train, need sorted by group
                tr_sorted = tr.sort_values("timestamp")
                va_sorted = va.sort_values("timestamp")
                tr_labels_s = np.concatenate([
                    _make_relevance_labels(g["fwd_ret_12h"].values)
                    for _, g in tr_sorted.groupby("timestamp")
                ])
                va_labels_s = np.concatenate([
                    _make_relevance_labels(g["fwd_ret_12h"].values)
                    for _, g in va_sorted.groupby("timestamp")
                ])
                tr_groups_s = tr_sorted.groupby("timestamp").size().values
                va_groups_s = va_sorted.groupby("timestamp").size().values

                dt_x = xgb.DMatrix(tr_sorted[avail], label=tr_labels_s)
                dt_x.set_group(tr_groups_s)
                dv_x = xgb.DMatrix(va_sorted[avail], label=va_labels_s)
                dv_x.set_group(va_groups_s)

                te_sorted = te.sort_values("timestamp")
                te_groups_s = te_sorted.groupby("timestamp").size().values
                dte_x = xgb.DMatrix(te_sorted[avail])
                dte_x.set_group(te_groups_s)
                m_x = xgb.train(xgb_p, dt_x, num_boost_round=N_ROUNDS,
                                evals=[(dv_x, "val")],
                                early_stopping_rounds=EARLY_STOP, verbose_eval=False)
                p_xgb_sorted = m_x.predict(dte_x)

                # Map back to original te order
                te_sorted_idx = te_sorted.index
                p_xgb = pd.Series(p_xgb_sorted, index=te_sorted_idx).reindex(te.index).values

            else:
                # Binary classification
                dt = lgb.Dataset(tr[avail], label=tr["target_binary"])
                dv = lgb.Dataset(va[avail], label=va["target_binary"])
                m = lgb.train(lgb_p, dt, num_boost_round=N_ROUNDS,
                              valid_sets=[dv],
                              callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                                         lgb.log_evaluation(-1)])
                p_lgb = m.predict(te[avail])

                dt_x = xgb.DMatrix(tr[avail], label=tr["target_binary"])
                dv_x = xgb.DMatrix(va[avail], label=va["target_binary"])
                m_x = xgb.train(xgb_p, dt_x, num_boost_round=N_ROUNDS,
                                evals=[(dv_x, "val")],
                                early_stopping_rounds=EARLY_STOP, verbose_eval=False)
                p_xgb = m_x.predict(xgb.DMatrix(te[avail]))

            rec = te[["timestamp", "symbol"]].copy()
            rec["pred_lgb"] = p_lgb
            rec["pred_xgb"] = p_xgb
            rec = rec.merge(fwd, on=["timestamp", "symbol"], how="inner")
            rec["window"] = w["name"]
            rec["seed"] = seed
            all_preds.append(rec)

            if seed == seeds[0]:
                n_groups_train = tr.groupby("timestamp").ngroups
                n_groups_test = te.groupby("timestamp").ngroups
                log(f"  {w['name']}/s{seed}: train={len(tr):,} ({n_groups_train} groups) "
                    f"test={len(te):,} ({n_groups_test} groups)")

    if not all_preds:
        return None

    merged = pd.concat(all_preds)
    # Average across seeds
    agg = merged.groupby(["timestamp", "symbol"]).agg(
        pred_lgb=("pred_lgb", "mean"),
        pred_xgb=("pred_xgb", "mean"),
        fwd_ret=("fwd_ret", "first"),
        window=("window", "first"),
    ).reset_index()

    # Rank normalization
    agg["rank_lgb"] = agg.groupby("timestamp")["pred_lgb"].rank(pct=True) - 0.5
    agg["rank_xgb"] = agg.groupby("timestamp")["pred_xgb"].rank(pct=True) - 0.5
    agg["pred"] = 0.5 * agg["rank_lgb"] + 0.5 * agg["rank_xgb"]
    agg["raw_prob"] = 0.5 * agg["pred_lgb"] + 0.5 * agg["pred_xgb"]

    return agg[["timestamp", "symbol", "pred", "raw_prob", "fwd_ret", "window"]]


def compute_ndcg(preds_df, k=4):
    """Compute NDCG@K across all timestamps."""
    ndcgs = []
    for ts, grp in preds_df.groupby("timestamp"):
        if len(grp) < k:
            continue
        # Relevance = actual fwd_ret rank (higher = better)
        grp = grp.copy()
        grp["relevance"] = grp["fwd_ret"].rank(ascending=True)
        # Predicted ranking by pred score
        grp = grp.sort_values("pred", ascending=False)
        dcg = sum(grp["relevance"].iloc[i] / np.log2(i + 2) for i in range(min(k, len(grp))))
        # Ideal: sort by actual relevance
        ideal = grp.sort_values("relevance", ascending=False)
        idcg = sum(ideal["relevance"].iloc[i] / np.log2(i + 2) for i in range(min(k, len(grp))))
        if idcg > 0:
            ndcgs.append(dcg / idcg)
    return np.mean(ndcgs) if ndcgs else 0.0


def simulate(merged, regime_df, n_long=4, n_short=2, cfg=PROD_CFG):
    """Standard simulation (same as R65)."""
    trend_cutoff = cfg["trend_cutoff"]
    rebal_hours = cfg["rebal_hours"]
    ema_alpha = cfg.get("ema_alpha", None)
    hysteresis = cfg.get("hysteresis", 0)
    funding_per_12h = 0.00008

    all_rets = []
    prev_longs: Set[str] = set()
    prev_shorts: Set[str] = set()
    prev_preds: Dict[str, float] = {}

    timestamps_sorted = sorted(merged["timestamp"].unique())
    grouped = {ts: grp for ts, grp in merged.groupby("timestamp")}
    rebal_timestamps = timestamps_sorted[::rebal_hours]

    for ts in rebal_timestamps:
        if ts not in regime_df.index or ts not in grouped:
            continue
        row = regime_df.loc[ts]
        trend_str = row.get("trend_strength", 0)
        if trend_str > trend_cutoff:
            continue
        grp = grouped[ts].copy()
        n = len(grp)
        nl = min(n_long, n // 3)
        ns = min(n_short, n // 3)
        if nl == 0 and ns == 0:
            continue

        exposure = 1.0
        dyn_threshold = cfg.get("dyn_threshold", 0.5)
        if dyn_threshold is not None and trend_str > dyn_threshold:
            exposure = max(0.1, 1.0 - (trend_str - dyn_threshold) /
                           (trend_cutoff - dyn_threshold + 1e-10) * 0.5)

        if ema_alpha is not None and ema_alpha < 1.0:
            for idx, r in grp.iterrows():
                sym = r["symbol"]
                raw_pred = r["pred"]
                smoothed = ema_alpha * raw_pred + (1 - ema_alpha) * prev_preds.get(sym, raw_pred)
                prev_preds[sym] = smoothed
                grp.at[idx, "pred"] = smoothed

        grp["pred_rank"] = grp["pred"].rank(ascending=False)

        if hysteresis > 0 and (prev_longs or prev_shorts):
            new_longs: Set[str] = set()
            new_shorts: Set[str] = set()
            for idx, r in grp.iterrows():
                sym, rank = r["symbol"], r["pred_rank"]
                if sym in prev_longs and rank <= nl + hysteresis:
                    new_longs.add(sym)
                elif sym in prev_shorts and rank > (n - ns - hysteresis):
                    new_shorts.add(sym)
            remaining = grp[~grp["symbol"].isin(new_longs | new_shorts)]
            for _, r in remaining.sort_values("pred_rank").head(nl - len(new_longs)).iterrows():
                new_longs.add(r["symbol"])
            for _, r in remaining.sort_values("pred_rank", ascending=False).head(ns - len(new_shorts)).iterrows():
                new_shorts.add(r["symbol"])
        else:
            new_longs = set(grp[grp["pred_rank"] <= nl]["symbol"].tolist()) if nl > 0 else set()
            new_shorts = set(grp[grp["pred_rank"] > (n - ns)]["symbol"].tolist()) if ns > 0 else set()

        new_opened = (new_longs - prev_longs) | (new_shorts - prev_shorts)
        closed = (prev_longs - new_longs) | (prev_shorts - new_shorts)
        total_positions = len(new_longs) + len(new_shorts)

        longs = grp[grp["symbol"].isin(new_longs)]
        shorts = grp[grp["symbol"].isin(new_shorts)]
        long_ret = longs["fwd_ret"].mean() if len(longs) > 0 else 0
        short_ret = shorts["fwd_ret"].mean() if len(shorts) > 0 else 0

        nl_act, ns_act = len(new_longs), len(new_shorts)
        if nl_act > 0 and ns_act > 0:
            gross_ret = 0.5 * long_ret - 0.5 * short_ret
        elif ns_act > 0:
            gross_ret = -short_ret
        else:
            gross_ret = long_ret
        gross_ret *= exposure

        if total_positions > 0:
            avg_weight = 1.0 / total_positions
            turnover_cost = sum(_cost_for_sym(sym) * avg_weight for sym in new_opened)
            turnover_cost += sum(_cost_for_sym(sym) * avg_weight for sym in closed)
            holding_cost = funding_per_12h * (rebal_hours / 12)
            total_cost = turnover_cost + holding_cost
        else:
            total_cost = 0.0

        net_ret = gross_ret - total_cost
        prev_longs, prev_shorts = new_longs, new_shorts

        all_rets.append({
            "timestamp": ts, "gross_ret": gross_ret, "net_ret": net_ret,
            "cost": total_cost, "n_long": nl_act, "n_short": ns_act,
            "turnover": len(new_opened) + len(closed),
        })

    return pd.DataFrame(all_rets) if all_rets else pd.DataFrame()


def sharpe(rets_series, periods_per_year=2 * 365):
    if len(rets_series) < 2:
        return 0.0
    eq = (1 + rets_series).cumprod()
    r = eq.pct_change().dropna()
    return r.mean() / (r.std() + 1e-10) * np.sqrt(periods_per_year)


def analyze_portfolio(port, label):
    """Print full analysis for one portfolio configuration."""
    gs = sharpe(port["gross_ret"])
    ns = sharpe(port["net_ret"])
    eq = (1 + port["net_ret"]).cumprod() * 100
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1
    maxdd = (eq / eq.cummax() - 1).min()
    wr = (port["net_ret"] > 0).mean() * 100
    avg_cost = port["cost"].mean() * 10000

    print(f"  {label}:")
    print(f"    Gross Sharpe: {gs:.3f}  Net Sharpe: {ns:.3f}")
    print(f"    Ret: {total_ret*100:.1f}%  DD: {maxdd*100:.1f}%  WR: {wr:.1f}%")
    print(f"    Avg cost: {avg_cost:.2f} bps  Periods: {len(port)}")

    # Quarterly
    port_c = port.copy()
    port_c["quarter"] = port_c["timestamp"].dt.to_period("Q").astype(str)
    for q in sorted(port_c["quarter"].unique()):
        qdf = port_c[port_c["quarter"] == q]
        qns = sharpe(qdf["net_ret"])
        qr = ((1 + qdf["net_ret"]).cumprod().iloc[-1] - 1) * 100
        print(f"    {q}: Sharpe={qns:.2f} Ret={qr:.1f}%")

    return {"label": label, "gross_sharpe": round(gs, 3), "net_sharpe": round(ns, 3),
            "total_ret_pct": round(total_ret * 100, 1), "max_dd_pct": round(maxdd * 100, 1),
            "win_rate": round(wr, 1), "avg_cost_bps": round(avg_cost, 2)}


def main():
    t0 = time.time()
    print("=" * 70)
    print("  R70 — LAMBDARANK vs BINARY CLASSIFICATION")
    print("=" * 70)

    df, regime_df = load_data()
    feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    print(f"  Features: {len(feats)}")

    results = []

    # --- 1) Binary baseline (R65 reproduction) ---
    print("\n" + "=" * 70)
    print("  1) BINARY CLASSIFICATION (baseline)")
    print("=" * 70)
    t1 = time.time()
    preds_bin = train_ensemble_ranking(df, feats, ORIGINAL_WINDOWS, seeds=SEEDS, objective="binary")
    if preds_bin is None:
        print("  FAILED"); return
    print(f"  Trained in {time.time()-t1:.0f}s, {len(preds_bin):,} predictions")

    for nl, ns in [(4, 2), (6, 3)]:
        port = simulate(preds_bin, regime_df, n_long=nl, n_short=ns)
        if port.empty:
            continue
        ndcg4 = compute_ndcg(preds_bin, k=4)
        ndcg2 = compute_ndcg(preds_bin, k=2)
        r = analyze_portfolio(port, f"binary_{nl}L{ns}S")
        r["ndcg@4"] = round(ndcg4, 4)
        r["ndcg@2"] = round(ndcg2, 4)
        r["objective"] = "binary"
        results.append(r)

    # --- 2) LambdaRank ---
    print("\n" + "=" * 70)
    print("  2) LAMBDARANK OBJECTIVE")
    print("=" * 70)
    t2 = time.time()
    preds_rank = train_ensemble_ranking(df, feats, ORIGINAL_WINDOWS, seeds=SEEDS, objective="rank")
    if preds_rank is None:
        print("  FAILED"); return
    print(f"  Trained in {time.time()-t2:.0f}s, {len(preds_rank):,} predictions")

    for nl, ns in [(4, 2), (6, 3)]:
        port = simulate(preds_rank, regime_df, n_long=nl, n_short=ns)
        if port.empty:
            continue
        ndcg4 = compute_ndcg(preds_rank, k=4)
        ndcg2 = compute_ndcg(preds_rank, k=2)
        r = analyze_portfolio(port, f"rank_{nl}L{ns}S")
        r["ndcg@4"] = round(ndcg4, 4)
        r["ndcg@2"] = round(ndcg2, 4)
        r["objective"] = "lambdarank"
        results.append(r)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  SUMMARY: BINARY vs LAMBDARANK")
    print("=" * 70)
    print(f"  {'Config':<20} {'Gross Sh':>10} {'Net Sh':>10} {'NDCG@4':>8} {'NDCG@2':>8} {'Ret%':>8} {'DD%':>8}")
    print(f"  {'-'*74}")
    for r in results:
        print(f"  {r['label']:<20} {r['gross_sharpe']:>10.3f} {r['net_sharpe']:>10.3f} "
              f"{r['ndcg@4']:>8.4f} {r['ndcg@2']:>8.4f} "
              f"{r['total_ret_pct']:>7.1f}% {r['max_dd_pct']:>7.1f}%")

    pd.DataFrame(results).to_csv("/data/datasets/results_r70_lambdarank.csv", index=False)
    print(f"\n  Saved: /data/datasets/results_r70_lambdarank.csv")
    print(f"  Total: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)")


if __name__ == "__main__":
    main()
