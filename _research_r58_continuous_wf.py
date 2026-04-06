#!/usr/bin/env python3
"""
R58 — Continuous Walk-Forward (no gap months)

Purpose: Original W1/W2/W3 windows have gaps (purge periods) where the strategy
"doesn't trade". In production, the last model keeps trading through gaps.
This test reveals whether gap months were hiding bad performance.

Design:
  - Same train_end dates as original (2024-06-01, 2025-01-01, 2025-07-01)
  - Same expanding training approach
  - Test windows are CONTINUOUS: each window trades until the next train_end
  - The purge gap (train_end → test_start) still applies for leakage safety,
    but the model keeps trading in "gap" months too
  - Additional "bridge" predictions for gap months using the previous window's model

Windows:
  W1: train_end=2024-06-01, test= 2024-10-15 → 2025-01-01 (original)
  W1b: same model, test= 2025-01-01 → 2025-05-15  (gap fill)
  W2: train_end=2025-01-01, test= 2025-05-15 → 2025-07-01 (original)
  W2b: same model, test= 2025-07-01 → 2025-11-15  (gap fill only to purge start of W3)
  W3: train_end=2025-07-01, test= 2025-11-15 → 2026-03-17 (original)

Result: Full continuous equity curve from Oct 2024 → Mar 2026 with NO gaps.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats

warnings.filterwarnings("ignore")

# ── project imports ───────────────────────────────────────
from _research_round7 import SYM_35
from _research_r22_models import SEEDS, LEVERAGE, CAPITAL, log, cs_rank_cols
from _research_r30b_fixed import eval_with_costs, compute_regime_extended
from _research_r35_new_features import (
    add_r35_features, load_research_frame, MARKET_LEVEL_FEATURES,
)
from _research_r47_coinglass import (
    CANONICAL_EXEC_CFG, CHAMPION_FEAT_30,
    add_cg_features, compute_cg_features, load_cg_daily,
)
from _research_r48_cost import simulate_with_hybrid_costs

CHAMPION_FEAT_31 = CHAMPION_FEAT_30 + ["cg_taker_imb"]

# ── Original windows (with gaps) ─────────────────────────
ORIGINAL_WINDOWS = [
    {"name": "W1",
     "train_end": "2024-06-01",
     "val_start": "2024-06-01", "val_end": "2024-09-30",
     "test_start": "2024-10-15", "test_end": "2025-01-31"},
    {"name": "W2",
     "train_end": "2025-01-01",
     "val_start": "2025-01-01", "val_end": "2025-04-30",
     "test_start": "2025-05-15", "test_end": "2025-08-31"},
    {"name": "W3",
     "train_end": "2025-07-01",
     "val_start": "2025-07-01", "val_end": "2025-10-31",
     "test_start": "2025-11-15", "test_end": "2026-03-17"},
]

# ── Continuous windows (no gaps: model keeps trading through purge periods) ──
# W1 model trades from 2024-10-15 until 2025-05-15 (when W2 model takes over)
# W2 model trades from 2025-05-15 until 2025-11-15 (when W3 model takes over)
# W3 model trades from 2025-11-15 until data end
CONTINUOUS_WINDOWS = [
    {"name": "W1",
     "train_end": "2024-06-01",
     "val_start": "2024-06-01", "val_end": "2024-09-30",
     "test_start": "2024-10-15", "test_end": "2025-05-14"},
    {"name": "W2",
     "train_end": "2025-01-01",
     "val_start": "2025-01-01", "val_end": "2025-04-30",
     "test_start": "2025-05-15", "test_end": "2025-11-14"},
    {"name": "W3",
     "train_end": "2025-07-01",
     "val_start": "2025-07-01", "val_end": "2025-10-31",
     "test_start": "2025-11-15", "test_end": "2026-03-17"},
]

PROD_CFG = {
    "n_long": 6, "n_short": 3, "rebal_hours": 12,
    "trend_cutoff": 0.9, "dyn_threshold": 0.7,
    "ema_alpha": 0.5, "hysteresis": 3,
}

LGB_PARAMS = {
    "objective": "binary", "metric": "auc",
    "learning_rate": 0.03, "num_leaves": 63,
    "min_child_samples": 100, "subsample": 0.8,
    "colsample_bytree": 0.8, "lambda_l2": 1.0,
    "verbose": -1, "n_jobs": -1,
}
XGB_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "auc",
    "learning_rate": 0.03, "max_depth": 6,
    "min_child_weight": 100, "subsample": 0.8,
    "colsample_bytree": 0.8, "reg_lambda": 1.0,
    "n_jobs": -1, "verbosity": 0,
}
N_ROUNDS = 600
EARLY_STOP = 40


def load_data():
    """Load full model frame with all features."""
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
        raise ValueError(f"Missing champion features: {missing}")
    print(f"  Model frame: {len(df):,} rows, {df['symbol'].nunique()} symbols")
    print(f"  Date range: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print(f"  Champion features: {len(present)} ✅")
    return df, regime_df


def train_ensemble_continuous(df, feats, windows, seeds=SEEDS,
                               cs_rank_exclude=None):
    """Train LGB+XGB ensemble with continuous test windows (no gaps)."""
    avail = [f for f in feats if f in df.columns]
    rank_exclude = set(cs_rank_exclude or [])
    rank_feats = [f for f in avail if f not in rank_exclude]
    tz = df["timestamp"].dt.tz

    all_lgb, all_xgb = [], []

    for seed in seeds:
        p_lgb = {**LGB_PARAMS, "seed": seed}
        p_xgb = {**XGB_PARAMS, "seed": seed}

        for w in windows:
            te_end = pd.Timestamp(w["test_end"], tz=tz)
            te_start = pd.Timestamp(w["test_start"], tz=tz)
            tr_end = pd.Timestamp(w["train_end"], tz=tz)
            va_start = pd.Timestamp(w["val_start"], tz=tz)
            va_end = pd.Timestamp(w["val_end"], tz=tz)

            # Expanding train
            train_ = df[df["timestamp"] < tr_end].copy()
            val_ = df[(df["timestamp"] >= va_start) &
                      (df["timestamp"] < va_end)].copy()
            test_ = df[(df["timestamp"] >= te_start) &
                       (df["timestamp"] <= te_end)].copy()

            if len(train_) < 5000 or len(test_) < 200:
                log(f"  {w['name']}/s{seed}: skip (train={len(train_)}, test={len(test_)})")
                continue

            # CS rank
            if rank_feats:
                train_ = cs_rank_cols(train_, rank_feats)
                val_ = cs_rank_cols(val_, rank_feats)
                test_ = cs_rank_cols(test_, rank_feats)

            for d in [train_, val_, test_]:
                d["target_binary"] = (d["fwd_ret_12h"] > 0).astype(int)

            # Fill NaN
            for col in avail:
                for d in [train_, val_, test_]:
                    if d[col].isna().any():
                        d[col] = d[col].fillna(0)

            tr = train_[avail + ["target_binary"]].dropna()
            va = val_[avail + ["target_binary"]].dropna()
            te = test_[avail + ["target_binary", "timestamp", "symbol"]].dropna()
            fwd = test_[["timestamp", "symbol", "fwd_ret_12h"]].rename(
                columns={"fwd_ret_12h": "fwd_ret"}).dropna()

            for d in [tr, va, te]:
                d.replace([np.inf, -np.inf], np.nan, inplace=True)
            tr, va, te = tr.dropna(), va.dropna(), te.dropna()
            if len(te) == 0:
                continue

            # LGB
            dt = lgb.Dataset(tr[avail], label=tr["target_binary"])
            dv = lgb.Dataset(va[avail], label=va["target_binary"])
            m = lgb.train(p_lgb, dt, num_boost_round=N_ROUNDS,
                          valid_sets=[dv],
                          callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                                     lgb.log_evaluation(-1)])
            p = m.predict(te[avail])
            rec = te[["timestamp", "symbol"]].copy()
            rec["pred_lgb"] = p
            rec = rec.merge(fwd, on=["timestamp", "symbol"], how="inner")
            rec["window"] = w["name"]
            rec["seed"] = seed
            all_lgb.append(rec)

            # XGB
            dt_x = xgb.DMatrix(tr[avail], label=tr["target_binary"])
            dv_x = xgb.DMatrix(va[avail], label=va["target_binary"])
            m_x = xgb.train(p_xgb, dt_x, num_boost_round=N_ROUNDS,
                             evals=[(dv_x, "val")],
                             early_stopping_rounds=EARLY_STOP, verbose_eval=False)
            p_x = m_x.predict(xgb.DMatrix(te[avail]))
            rec2 = te[["timestamp", "symbol"]].copy()
            rec2["pred_xgb"] = p_x
            rec2 = rec2.merge(fwd, on=["timestamp", "symbol"], how="inner")
            rec2["window"] = w["name"]
            rec2["seed"] = seed
            all_xgb.append(rec2)

            if seed == seeds[0]:
                ic = stats.spearmanr(rec["pred_lgb"], rec["fwd_ret"])[0]
                log(f"  {w['name']}/s{seed}: train={len(tr):,}, test={len(te):,}, "
                    f"IC={ic:.4f}, test_dates={te['timestamp'].min().date()}→{te['timestamp'].max().date()}")

    if not all_lgb:
        return None

    lgb_df = pd.concat(all_lgb)
    xgb_df = pd.concat(all_xgb)
    lgb_avg = lgb_df.groupby(["timestamp", "symbol"]).agg(
        pred_lgb=("pred_lgb", "mean"), fwd_ret=("fwd_ret", "first"),
        window=("window", "first")).reset_index()
    xgb_avg = xgb_df.groupby(["timestamp", "symbol"]).agg(
        pred_xgb=("pred_xgb", "mean")).reset_index()
    merged = lgb_avg.merge(xgb_avg, on=["timestamp", "symbol"], how="inner")
    merged["rank_lgb"] = merged.groupby("timestamp")["pred_lgb"].rank(pct=True) - 0.5
    merged["rank_xgb"] = merged.groupby("timestamp")["pred_xgb"].rank(pct=True) - 0.5
    merged["pred"] = 0.5 * merged["rank_lgb"] + 0.5 * merged["rank_xgb"]
    return merged[["timestamp", "symbol", "pred", "fwd_ret", "window"]]


def analyze(port_all, preds, label):
    """Detailed analysis of a simulation run."""
    eq = port_all["cumret"]
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1
    rets = eq.pct_change().dropna()
    sharpe = rets.mean() / (rets.std() + 1e-10) * np.sqrt(2 * 365)  # 12h periods
    maxdd = (eq / eq.cummax() - 1).min()

    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    print(f"  Sharpe (net):    {sharpe:.2f}")
    print(f"  Total return:    {total_ret * 100:+.1f}%")
    print(f"  Final equity:    ${eq.iloc[-1]:.0f} (from ${eq.iloc[0]:.0f})")
    print(f"  Max drawdown:    {maxdd * 100:.1f}%")
    print(f"  Test period:     {port_all['timestamp'].min().date()} → {port_all['timestamp'].max().date()}")

    # Win rate
    n_win = (rets > 0).sum()
    n_total = len(rets)
    print(f"  Win rate (12h):  {100 * n_win / n_total:.1f}%")

    # Monthly returns
    port_all = port_all.copy()
    port_all["month"] = port_all["timestamp"].dt.to_period("M")
    months = port_all.groupby("month").agg(
        start=("cumret", "first"), end=("cumret", "last")).reset_index()
    months["ret"] = months["end"] / months["start"] - 1

    print(f"\n  📅 Monthly Returns")
    print(f"  {'Month':<12} {'Return%':>10}")
    print(f"  {'─' * 25}")
    for _, row in months.iterrows():
        flag = "✅" if row["ret"] > 0 else "❌"
        print(f"  {str(row['month']):<12} {row['ret'] * 100:>+8.1f}%  {flag}")

    win_months = (months["ret"] > 0).sum()
    print(f"\n  Win months: {win_months}/{len(months)} ({100 * win_months / len(months):.0f}%)")
    print(f"  Avg month:  {months['ret'].mean() * 100:+.1f}%")
    print(f"  Best month: {months['ret'].max() * 100:+.1f}%")
    print(f"  Worst month:{months['ret'].min() * 100:+.1f}%")

    # Quarterly
    port_all["quarter"] = port_all["timestamp"].dt.to_period("Q")
    quarters = port_all.groupby("quarter").agg(
        start=("cumret", "first"), end=("cumret", "last")).reset_index()
    quarters["ret"] = quarters["end"] / quarters["start"] - 1
    print(f"\n  📅 Quarterly Returns")
    for _, row in quarters.iterrows():
        flag = "✅" if row["ret"] > 0 else "❌"
        print(f"  {str(row['quarter']):<12} {row['ret'] * 100:>+8.1f}%  {flag}")
    win_q = (quarters["ret"] > 0).sum()
    print(f"  Win quarters: {win_q}/{len(quarters)} ({100 * win_q / len(quarters):.0f}%)")

    # Per-window breakdown
    print(f"\n  📊 Per-Window Breakdown")
    print(f"  {'Window':<8} {'Period':<28} {'Sharpe':>8} {'Return%':>10} {'WinRate':>8}")
    print(f"  {'─' * 70}")
    for wname in preds["window"].unique():
        wp = preds[preds["window"] == wname]
        ts_min, ts_max = wp["timestamp"].min(), wp["timestamp"].max()
        w_port = port_all[(port_all["timestamp"] >= ts_min) &
                          (port_all["timestamp"] <= ts_max)]
        if len(w_port) < 2:
            continue
        w_eq = w_port["cumret"]
        w_rets = w_eq.pct_change().dropna()
        w_sh = w_rets.mean() / (w_rets.std() + 1e-10) * np.sqrt(2 * 365)
        w_ret = w_eq.iloc[-1] / w_eq.iloc[0] - 1
        w_wr = 100 * (w_rets > 0).sum() / len(w_rets) if len(w_rets) > 0 else 0
        n_months = (ts_max - ts_min).days / 30
        print(f"  {wname:<8} {str(ts_min.date())}→{str(ts_max.date())}  "
              f"{n_months:>4.1f}m  {w_sh:>+6.2f}  {w_ret * 100:>+8.1f}%  {w_wr:>6.1f}%")

    # Drawdown
    dd = eq / eq.cummax() - 1
    worst_idx = dd.idxmin()
    peak_idx = eq[:worst_idx].idxmax()
    dd_start = port_all.loc[peak_idx, "timestamp"]
    dd_end = port_all.loc[worst_idx, "timestamp"]
    dd_days = (dd_end - dd_start).days
    print(f"\n  Worst DD: {maxdd * 100:.1f}%  ({dd_start.date()} → {dd_end.date()}, {dd_days}d)")

    return {
        "sharpe": round(sharpe, 2),
        "total_return_pct": round(total_ret * 100, 1),
        "max_dd_pct": round(maxdd * 100, 1),
        "win_rate_12h": round(100 * n_win / n_total, 1),
        "win_months": f"{win_months}/{len(months)}",
        "win_quarters": f"{win_q}/{len(quarters)}",
    }


def main():
    parser = argparse.ArgumentParser(description="R58 Continuous Walk-Forward")
    parser.add_argument("--original", action="store_true",
                        help="Also run original (gapped) windows for comparison")
    args = parser.parse_args()

    print("=" * 70)
    print("  R58 — CONTINUOUS WALK-FORWARD (no gap months)")
    print("  Champion: 31f + hybrid tiered costs")
    print("=" * 70)

    df, regime_df = load_data()

    feats = CHAMPION_FEAT_31
    no_rank = [f for f in feats if f in MARKET_LEVEL_FEATURES]

    # ── Run continuous WF ──────────────────────────────────
    print("\n" + "=" * 70)
    print("  CONTINUOUS WALK-FORWARD (gap months traded by prior model)")
    print("=" * 70)

    preds_cont = train_ensemble_continuous(
        df, feats, CONTINUOUS_WINDOWS, seeds=SEEDS,
        cs_rank_exclude=no_rank)

    if preds_cont is None:
        print("  ❌ Training failed")
        return

    n_ts = preds_cont["timestamp"].nunique()
    print(f"\n  Predictions: {len(preds_cont):,} rows, {n_ts:,} timestamps")
    print(f"  Date range: {preds_cont['timestamp'].min().date()} → {preds_cont['timestamp'].max().date()}")
    print(f"  Windows: {preds_cont['window'].value_counts().to_dict()}")

    port_cont = simulate_with_hybrid_costs(preds_cont, regime_df, PROD_CFG)
    port_cont["cumret"] = (1 + port_cont["portfolio_ret"]).cumprod() * CAPITAL
    summary_cont = analyze(port_cont, preds_cont, "CONTINUOUS WF (no gaps)")

    # ── Optionally run original (gapped) for comparison ───
    if args.original:
        print("\n" + "=" * 70)
        print("  ORIGINAL WALK-FORWARD (with gaps, for comparison)")
        print("=" * 70)

        preds_orig = train_ensemble_continuous(
            df, feats, ORIGINAL_WINDOWS, seeds=SEEDS,
            cs_rank_exclude=no_rank)

        if preds_orig is not None:
            port_orig = simulate_with_hybrid_costs(preds_orig, regime_df, PROD_CFG)
            port_orig["cumret"] = (1 + port_orig["portfolio_ret"]).cumprod() * CAPITAL
            summary_orig = analyze(port_orig, preds_orig, "ORIGINAL WF (with gaps)")

            # Comparison
            print("\n" + "=" * 70)
            print("  COMPARISON: CONTINUOUS vs ORIGINAL")
            print("=" * 70)
            for key in ["sharpe", "total_return_pct", "max_dd_pct", "win_rate_12h",
                         "win_months", "win_quarters"]:
                v1 = summary_cont.get(key, "?")
                v2 = summary_orig.get(key, "?")
                print(f"  {key:<20} Continuous={v1:<12} Original={v2}")

    print("\n" + "=" * 70)
    print("  ✅ R58 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
