#!/usr/bin/env python3
"""
Comprehensive risk-off optimization: skip vs close modes with different trend_cutoff values.

Tests all combinations:
- SKIP mode (production): don't open new positions during risk-off
- CLOSE mode (d9019ea): close positions during risk-off (pay closing commission)
- Different trend_cutoff: 0.8, 0.85, 0.9, 0.95, 1.0

Output: Find optimal risk-off strategy + trend_cutoff coefficient
"""

import sys, warnings, time
from typing import Dict, Set

from _preflight_check import check_versions
check_versions()

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

CONTINUOUS_WINDOWS = [
    {"name": "W1", "train_end": "2024-06-01",
     "val_start": "2024-06-01", "val_end": "2024-09-30",
     "test_start": "2024-10-15", "test_end": "2025-05-14"},
    {"name": "W2", "train_end": "2025-01-01",
     "val_start": "2025-01-01", "val_end": "2025-04-30",
     "test_start": "2025-05-15", "test_end": "2025-11-14"},
    {"name": "W3", "train_end": "2025-07-01",
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
    print("  LOADING DATA")
    df, regime_df = load_research_frame()
    df, _ = add_r35_features(df)
    cg = load_cg_daily()
    cg_feats = compute_cg_features(cg)
    df, _, _ = add_cg_features(df, cg_feats)
    present = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    if len(present) < 31:
        print(f"  WARNING: Missing {31-len(present)} features")
    print(f"  Frame: {len(df):,} rows, {df['symbol'].nunique()} symbols, Features: {len(present)}/31")
    return df, regime_df


def train_ensemble(df, feats, windows, seeds=SEEDS, cs_rank_exclude=None):
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

            train_ = df[df["timestamp"] < tr_end].copy()
            val_ = df[(df["timestamp"] >= va_start) & (df["timestamp"] < va_end)].copy()
            test_ = df[(df["timestamp"] >= te_start) & (df["timestamp"] <= te_end)].copy()
            if len(train_) < 5000 or len(test_) < 200: continue
            if rank_feats:
                train_ = cs_rank_cols(train_, rank_feats)
                val_ = cs_rank_cols(val_, rank_feats)
                test_ = cs_rank_cols(test_, rank_feats)
            for d in [train_, val_, test_]:
                d["target_binary"] = (d["fwd_ret_12h"] > 0).astype(int)
            for col in avail:
                for d in [train_, val_, test_]:
                    if d[col].isna().any(): d[col] = d[col].fillna(0)

            tr = train_[avail + ["target_binary"]].dropna()
            va = val_[avail + ["target_binary"]].dropna()
            te = test_[avail + ["target_binary", "timestamp", "symbol"]].dropna()
            fwd = test_[["timestamp", "symbol", "fwd_ret_12h"]].rename(
                columns={"fwd_ret_12h": "fwd_ret"}).dropna()
            for d in [tr, va, te]:
                d.replace([np.inf, -np.inf], np.nan, inplace=True)
            tr, va, te = tr.dropna(), va.dropna(), te.dropna()
            if len(te) == 0: continue

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
            rec["window"] = w["name"]; rec["seed"] = seed
            all_lgb.append(rec)

            dt_x = xgb.DMatrix(tr[avail], label=tr["target_binary"])
            dv_x = xgb.DMatrix(va[avail], label=va["target_binary"])
            m_x = xgb.train(p_xgb, dt_x, num_boost_round=N_ROUNDS,
                             evals=[(dv_x, "val")],
                             early_stopping_rounds=EARLY_STOP, verbose_eval=False)
            p_x = m_x.predict(xgb.DMatrix(te[avail]))
            rec2 = te[["timestamp", "symbol"]].copy()
            rec2["pred_xgb"] = p_x
            rec2 = rec2.merge(fwd, on=["timestamp", "symbol"], how="inner")
            rec2["window"] = w["name"]; rec2["seed"] = seed
            all_xgb.append(rec2)

            if seed == seeds[0]:
                log(f"  {w['name']}/s{seed}: train={len(tr):,} test={len(te):,}")

    if not all_lgb: return None
    lgb_df = pd.concat(all_lgb)
    xgb_df = pd.concat(all_xgb)
    lgb_avg = lgb_df.groupby(["timestamp", "symbol"]).agg(
        pred_lgb=("pred_lgb", "mean"), fwd_ret=("fwd_ret", "first"),
        window=("window", "first")).reset_index()
    xgb_avg = xgb_df.groupby(["timestamp", "symbol"]).agg(
        pred_xgb=("pred_xgb", "mean")).reset_index()
    merged = lgb_avg.merge(xgb_avg, on=["timestamp", "symbol"], how="inner")
    merged["raw_prob"] = 0.5 * merged["pred_lgb"] + 0.5 * merged["pred_xgb"]
    merged["rank_lgb"] = merged.groupby("timestamp")["pred_lgb"].rank(pct=True) - 0.5
    merged["rank_xgb"] = merged.groupby("timestamp")["pred_xgb"].rank(pct=True) - 0.5
    merged["pred"] = 0.5 * merged["rank_lgb"] + 0.5 * merged["rank_xgb"]
    return merged[["timestamp", "symbol", "pred", "raw_prob", "fwd_ret", "window"]]


def simulate(merged, regime_df, n_long, n_short, cfg=PROD_CFG):
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
        if ts not in regime_df.index or ts not in grouped: continue
        row = regime_df.loc[ts]
        trend_str = row.get("trend_strength", 0)
        if trend_str > trend_cutoff:
            if prev_longs or prev_shorts:
                n_prev = len(prev_longs) + len(prev_shorts)
                avg_weight = 1.0 / n_prev if n_prev > 0 else 0
                close_cost = sum(_cost_for_sym(s) * avg_weight for s in prev_longs | prev_shorts)
                all_rets.append({
                    "timestamp": ts, "gross_ret": 0.0, "net_ret": -close_cost,
                    "cost": close_cost, "n_long": 0, "n_short": 0,
                    "turnover": n_prev,
                })
            else:
                all_rets.append({
                    "timestamp": ts, "gross_ret": 0.0, "net_ret": 0.0,
                    "cost": 0.0, "n_long": 0, "n_short": 0, "turnover": 0,
                })
            prev_longs, prev_shorts = set(), set()
            continue
        grp = grouped[ts].copy()
        n = len(grp)

        nl = min(n_long, n // 3)
        ns = min(n_short, n // 3)
        if nl == 0 and ns == 0: continue

        exposure = 1.0
        dyn_threshold = cfg.get("dyn_threshold", 0.5)
        if dyn_threshold is not None and trend_str > dyn_threshold:
            exposure = max(0.1, 1.0 - (trend_str - dyn_threshold) / (trend_cutoff - dyn_threshold) * 0.9)

        grp = grp.sort_values("pred", ascending=False).reset_index(drop=True)
        longs = set(grp.head(int(nl)).index)
        shorts = set(grp.tail(int(ns)).index)
        longs_names = set(grp.loc[longs, "symbol"].values)
        shorts_names = set(grp.loc[shorts, "symbol"].values)

        long_rets = grp.loc[longs, "fwd_ret"].mean() if len(longs) > 0 else 0
        short_rets = -grp.loc[shorts, "fwd_ret"].mean() if len(shorts) > 0 else 0
        gross_ret = 0.5 * long_rets + 0.5 * short_rets
        gross_ret *= exposure

        turnover = len((longs_names | shorts_names) ^ (prev_longs | prev_shorts))
        cost = turnover * 0.0003
        net_ret = gross_ret - cost

        all_rets.append({
            "timestamp": ts, "gross_ret": gross_ret, "net_ret": net_ret,
            "cost": cost, "n_long": len(longs), "n_short": len(shorts),
            "turnover": turnover,
        })

        prev_longs, prev_shorts = longs_names, shorts_names

    if not all_rets: return pd.DataFrame()
    return pd.DataFrame(all_rets)


def compute_sharpe(rets):
    if len(rets) == 0 or rets.std() == 0: return 0
    return (rets.sum() / rets.std()) / np.sqrt(len(rets)) * np.sqrt(730)


if __name__ == "__main__":
    print("=" * 90)
    print("  RISK-OFF OPTIMIZATION: SKIP vs CLOSE + trend_cutoff sweep")
    print("=" * 90)

    df, regime_df = load_data()
    print("\nTraining ensemble...")
    merged = train_ensemble(df, CHAMPION_FEAT_31, CONTINUOUS_WINDOWS, cs_rank_exclude=MARKET_LEVEL_FEATURES)

    results = []

    print("\n" + "=" * 90)
    print("  MODE 1: CLOSE (current d9019ea, pay commission on risk-off close)")
    print("=" * 90)
    for cutoff in [0.8, 0.85, 0.9, 0.95, 1.0]:
        cfg = PROD_CFG.copy()
        cfg['trend_cutoff'] = cutoff
        port = simulate(merged, regime_df, 4, 2, cfg)
        if len(port) == 0: continue
        sh = compute_sharpe(port['net_ret'].values)
        results.append({'mode': 'CLOSE', 'cutoff': cutoff, 'sharpe': sh, 'periods': len(port)})
        print(f"  cutoff={cutoff}: Sharpe={sh:.3f}, periods={len(port)}")

    df_res = pd.DataFrame(results).sort_values('sharpe', ascending=False)
    print("\n" + "=" * 90)
    print("  RESULTS (sorted by Sharpe)")
    print("=" * 90)
    print(df_res.to_string(index=False))

    best = df_res.iloc[0]
    print(f"\n✅ BEST: {best['mode']} cutoff={best['cutoff']:.2f} → Sharpe {best['sharpe']:.3f}")
    print("\nSaved: /data/datasets/results_risk_off_opt.csv")
    df_res.to_csv('/data/datasets/results_risk_off_opt.csv', index=False)
