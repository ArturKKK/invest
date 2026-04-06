#!/usr/bin/env python3
"""
R61 — Temporal Features (Lags + Rolling Statistics)

Adds ~12 temporal features to the champion 31-feature set:
  - ret_12h: lag2 (=ret_36h), lag4 (=ret_60h), rolling_std_5
  - cg_taker_imb: lag1, lag2, rolling_mean_5, rolling_std_5
  - oi_chg_12h: lag1, lag2, rolling_mean_5
  - sign_persistence_10 = rolling_mean(sign(ret_12h_t) == sign(ret_12h_{t-1}), 10)
  - reversal_count_10 = rolling_sum(sign(ret_12h_t) != sign(ret_12h_{t-1}), 10)

Key constraints:
  - All temporal features computed per-symbol (no cross-sectional)
  - NOT added to cs_rank_exclude? No, they ARE excluded from CS ranking
    (per-symbol lags should not be ranked cross-sectionally)
  - cg_taker_imb is daily data - lags in terms of 12h bars (lag1 = 12h ago, etc.)

Uses ORIGINAL_WINDOWS for comparability with baseline Sharpe 1.66.
"""

import sys
import warnings
from typing import Dict, List, Set
import time

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
    CANONICAL_EXEC_CFG, CHAMPION_FEAT_30,
    add_cg_features, compute_cg_features, load_cg_daily,
)

CHAMPION_FEAT_31 = CHAMPION_FEAT_30 + ["cg_taker_imb"]

TIER1_SYMS = {"BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"}
TIER3_SYMS = {
    "SAND/USDT", "LDO/USDT", "INJ/USDT", "APT/USDT", "ARB/USDT",
    "GALA/USDT", "FTM/USDT", "MATIC/USDT",
}
TIER2_SYMS = set(SYM_35) - TIER1_SYMS - TIER3_SYMS


def _cost_for_sym(sym: str) -> float:
    if sym in TIER1_SYMS:
        return 0.92 * (-0.0001) + 0.08 * 0.0007
    elif sym in TIER2_SYMS:
        return 0.75 * 0.0001 + 0.25 * 0.0007
    else:
        return 0.0005 + 0.0002


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


# ══════════════════════════════════════════════════════════
#  TEMPORAL FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════

# New temporal features (all excluded from CS ranking)
TEMPORAL_FEATURES = [
    "ret_12h_lag2",        # ret_12h 2 bars ago (=ret 24-36h window)
    "ret_12h_lag4",        # ret_12h 4 bars ago (=ret 48-60h window)
    "ret_12h_roll_std5",   # rolling 5-bar std of ret_12h
    "cg_taker_imb_lag1",   # cg_taker_imb 1 bar ago
    "cg_taker_imb_lag2",   # cg_taker_imb 2 bars ago
    "cg_taker_imb_roll5_mean",  # rolling 5-bar mean
    "cg_taker_imb_roll5_std",   # rolling 5-bar std
    "oi_chg_12h_lag1",     # oi_chg_12h 1 bar ago
    "oi_chg_12h_lag2",     # oi_chg_12h 2 bars ago
    "oi_chg_12h_roll5_mean",    # rolling 5-bar mean
    "sign_persistence_10",  # fraction of time ret_12h keeps same sign (10 bars)
    "reversal_count_10",    # number of sign reversals in last 10 bars
]

# Features that exist in baseline 31f and have temporal variants
BASE_WITH_LAGS = {"ret_12h", "cg_taker_imb", "oi_chg_12h"}

# All features for "full" model
FEAT_43 = CHAMPION_FEAT_31 + TEMPORAL_FEATURES


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 12 temporal features to dataframe. Computed per-symbol."""
    df = df.copy()

    def per_symbol(grp):
        grp = grp.sort_values("timestamp").copy()

        # ret_12h lags and rolling
        if "ret_12h" in grp.columns:
            grp["ret_12h_lag2"] = grp["ret_12h"].shift(2)
            grp["ret_12h_lag4"] = grp["ret_12h"].shift(4)
            grp["ret_12h_roll_std5"] = grp["ret_12h"].rolling(5, min_periods=3).std()

            # Sign persistence and reversal
            sign_now = np.sign(grp["ret_12h"].fillna(0))
            sign_prev = np.sign(grp["ret_12h"].shift(1).fillna(0))
            same_sign = (sign_now == sign_prev).astype(float)
            diff_sign = (sign_now != sign_prev).astype(float)
            grp["sign_persistence_10"] = same_sign.rolling(10, min_periods=5).mean()
            grp["reversal_count_10"] = diff_sign.rolling(10, min_periods=5).sum()

        # cg_taker_imb lags and rolling (daily data, but indexed by 12h bars)
        if "cg_taker_imb" in grp.columns:
            grp["cg_taker_imb_lag1"] = grp["cg_taker_imb"].shift(1)
            grp["cg_taker_imb_lag2"] = grp["cg_taker_imb"].shift(2)
            grp["cg_taker_imb_roll5_mean"] = grp["cg_taker_imb"].rolling(5, min_periods=3).mean()
            grp["cg_taker_imb_roll5_std"] = grp["cg_taker_imb"].rolling(5, min_periods=3).std()

        # oi_chg_12h lags and rolling
        if "oi_chg_12h" in grp.columns:
            grp["oi_chg_12h_lag1"] = grp["oi_chg_12h"].shift(1)
            grp["oi_chg_12h_lag2"] = grp["oi_chg_12h"].shift(2)
            grp["oi_chg_12h_roll5_mean"] = grp["oi_chg_12h"].rolling(5, min_periods=3).mean()

        return grp

    df = df.groupby("symbol", group_keys=False).apply(per_symbol)
    return df


def get_ic_report(df, feats, n_sample=50000):
    """Quick IC scan for new temporal features."""
    sample = df.dropna(subset=["fwd_ret_12h"]).sample(min(n_sample, len(df)), random_state=42)
    print(f"\n  IC scan on {len(sample):,} samples:")
    print(f"  {'Feature':<35} {'IC':>8} {'|IC|':>8}")
    print(f"  {'─' * 55}")
    ics = {}
    for feat in feats:
        if feat not in sample.columns:
            continue
        vals = sample[feat].fillna(0)
        if vals.std() < 1e-10:
            continue
        ic = stats.spearmanr(vals, sample["fwd_ret_12h"], nan_policy='omit')[0]
        ics[feat] = ic
    for feat, ic in sorted(ics.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feat:<35} {ic:>+8.4f} {abs(ic):>8.4f}")
    return ics


def load_data():
    print("=" * 70)
    print("  LOADING DATA")
    print("=" * 70)
    df, regime_df = load_research_frame()
    df, _ = add_r35_features(df)
    cg = load_cg_daily()
    cg_feats = compute_cg_features(cg)
    df, _, _ = add_cg_features(df, cg_feats)

    present_31 = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    missing_31 = [f for f in CHAMPION_FEAT_31 if f not in df.columns]
    if missing_31:
        print(f"  WARNING: Missing from 31f: {missing_31}")

    # Add temporal features
    print("  Adding temporal features...")
    df = add_temporal_features(df)

    present_temp = [f for f in TEMPORAL_FEATURES if f in df.columns]
    print(f"  Temporal features added: {len(present_temp)}/{len(TEMPORAL_FEATURES)}")
    print(f"  Frame: {len(df):,} rows, {df['symbol'].nunique()} symbols")
    print(f"  Dates: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    return df, regime_df


def train_ensemble(df, feats, windows, seeds=SEEDS, cs_rank_exclude=None):
    """Standard LGB+XGB ensemble training."""
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

            if len(train_) < 5000 or len(test_) < 200:
                continue

            if rank_feats:
                train_ = cs_rank_cols(train_, rank_feats)
                val_ = cs_rank_cols(val_, rank_feats)
                test_ = cs_rank_cols(test_, rank_feats)

            for d in [train_, val_, test_]:
                d["target_binary"] = (d["fwd_ret_12h"] > 0).astype(int)

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
                ic = stats.spearmanr(p, fwd.set_index(["timestamp", "symbol"]).reindex(
                    te.set_index(["timestamp", "symbol"]).index)["fwd_ret"].values)[0]
                log(f"  {w['name']}/s{seed}: train={len(tr):,} test={len(te):,} IC={ic:.4f}")

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


def simulate_with_hybrid_costs(merged, regime_df, cfg):
    """Standard hybrid cost simulation (from r48)."""
    n_long = cfg.get("n_long", 6)
    n_short = cfg.get("n_short", 3)
    trend_cutoff = cfg.get("trend_cutoff", 0.9)
    rebal_hours = cfg.get("rebal_hours", 12)
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

        nl, ns = min(n_long, n // 3), min(n_short, n // 3)
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
            for _, r in grp[~grp["symbol"].isin(new_longs | new_shorts)].sort_values("pred_rank").head(nl - len(new_longs)).iterrows():
                new_longs.add(r["symbol"])
            for _, r in grp[~grp["symbol"].isin(new_longs | new_shorts)].sort_values("pred_rank", ascending=False).head(ns - len(new_shorts)).iterrows():
                new_shorts.add(r["symbol"])
        else:
            new_longs = set(grp[grp["pred_rank"] <= nl]["symbol"].tolist()) if nl > 0 else set()
            new_shorts = set(grp[grp["pred_rank"] > (n - ns)]["symbol"].tolist()) if ns > 0 else set()

        longs = grp[grp["symbol"].isin(new_longs)]
        shorts = grp[grp["symbol"].isin(new_shorts)]
        long_ret = longs["fwd_ret"].mean() if len(longs) > 0 else 0
        short_ret = shorts["fwd_ret"].mean() if len(shorts) > 0 else 0

        new_opened = (new_longs - prev_longs) | (new_shorts - prev_shorts)
        closed = (prev_longs - new_longs) | (prev_shorts - new_shorts)
        total_positions = len(new_longs) + len(new_shorts)

        if total_positions > 0:
            avg_weight = 1.0 / total_positions
            turnover_cost = sum(_cost_for_sym(sym) * avg_weight for sym in new_opened)
            turnover_cost += sum(_cost_for_sym(sym) * avg_weight for sym in closed)
            holding_cost = funding_per_12h * (rebal_hours / 12)
            total_cost = turnover_cost + holding_cost
        else:
            total_cost = 0.0

        nl_act, ns_act = len(new_longs), len(new_shorts)
        if nl_act > 0 and ns_act > 0:
            port_ret = 0.5 * long_ret - 0.5 * short_ret
        elif ns_act > 0:
            port_ret = -short_ret
        else:
            port_ret = long_ret

        port_ret *= exposure
        port_ret -= total_cost

        prev_longs = new_longs
        prev_shorts = new_shorts

        all_rets.append({
            "timestamp": ts,
            "portfolio_ret": port_ret,
            "gross_ret": port_ret + total_cost,
            "long_ret": long_ret,
            "short_ret": short_ret,
            "n_long": nl_act,
            "n_short": ns_act,
            "exposure": exposure,
            "turnover": len(new_opened) + len(closed),
            "cost": total_cost,
        })

    return pd.DataFrame(all_rets) if all_rets else pd.DataFrame()


def compute_metrics(port_df, capital=100):
    if port_df.empty:
        return {"sharpe": 0, "total_return_pct": 0, "max_dd_pct": 0, "win_rate": 0}
    eq = (1 + port_df["portfolio_ret"]).cumprod() * capital
    rets = eq.pct_change().dropna()
    sharpe = rets.mean() / (rets.std() + 1e-10) * np.sqrt(2 * 365)
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1
    maxdd = (eq / eq.cummax() - 1).min()
    win_rate = (rets > 0).sum() / len(rets) * 100
    return {
        "sharpe": round(sharpe, 3),
        "total_return_pct": round(total_ret * 100, 1),
        "max_dd_pct": round(maxdd * 100, 1),
        "win_rate": round(win_rate, 1),
    }


def compute_window_metrics(port_df, preds, capital=100):
    results = {}
    for wname in preds["window"].unique():
        wp = preds[preds["window"] == wname]
        ts_min, ts_max = wp["timestamp"].min(), wp["timestamp"].max()
        w_port = port_df[(port_df["timestamp"] >= ts_min) & (port_df["timestamp"] <= ts_max)]
        if len(w_port) < 2:
            results[wname] = 0
            continue
        eq = (1 + w_port["portfolio_ret"]).cumprod()
        rets = eq.pct_change().dropna()
        sh = rets.mean() / (rets.std() + 1e-10) * np.sqrt(2 * 365)
        results[wname] = round(sh, 2)
    return results


def main():
    t0 = time.time()
    print("=" * 70)
    print("  R61 — TEMPORAL FEATURES (LAGS + ROLLING)")
    print("  Extending champion 31f → 43f with temporal lags")
    print("=" * 70)

    df, regime_df = load_data()

    # CS rank exclude list: MARKET_LEVEL_FEATURES + all temporal features
    base_no_rank = [f for f in CHAMPION_FEAT_31 if f in MARKET_LEVEL_FEATURES]
    temporal_no_rank = [f for f in TEMPORAL_FEATURES if f in df.columns]
    # Also exclude cg_taker_imb lags from cs_rank (they're already daily/non-cross-sectional)
    all_no_rank = list(set(base_no_rank + temporal_no_rank))

    # --- IC scan on temporal features ---
    print("\n  Running IC scan on temporal features vs baseline...")
    temp_feats_present = [f for f in TEMPORAL_FEATURES if f in df.columns]
    get_ic_report(df, temp_feats_present[:12])

    experiments = []

    # 1. Baseline: 31 features (for reference in this run)
    baseline_feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    baseline_no_rank = [f for f in baseline_feats if f in MARKET_LEVEL_FEATURES]
    experiments.append(("baseline_31f", baseline_feats, baseline_no_rank))

    # 2. +All temporal (43 features)
    feat_43 = [f for f in FEAT_43 if f in df.columns]
    experiments.append(("all_temporal_43f", feat_43, all_no_rank))

    # 3. Ablation: +ret_12h temporal only
    ret_temp = [f for f in feat_43 if f in baseline_feats or
                f in ["ret_12h_lag2", "ret_12h_lag4", "ret_12h_roll_std5",
                       "sign_persistence_10", "reversal_count_10"]]
    experiments.append(("+ret_temporal", ret_temp,
                        [f for f in ret_temp if f in all_no_rank]))

    # 4. Ablation: +cg_taker_imb temporal only
    cg_temp = [f for f in feat_43 if f in baseline_feats or
               f in ["cg_taker_imb_lag1", "cg_taker_imb_lag2",
                     "cg_taker_imb_roll5_mean", "cg_taker_imb_roll5_std"]]
    experiments.append(("+cg_temporal", cg_temp,
                        [f for f in cg_temp if f in all_no_rank]))

    # 5. Ablation: +oi_chg_12h temporal only
    oi_temp = [f for f in feat_43 if f in baseline_feats or
               f in ["oi_chg_12h_lag1", "oi_chg_12h_lag2", "oi_chg_12h_roll5_mean"]]
    experiments.append(("+oi_temporal", oi_temp,
                        [f for f in oi_temp if f in all_no_rank]))

    results = []

    print(f"\n  Running {len(experiments)} feature set experiments...\n")

    for exp_name, feats, no_rank in experiments:
        print(f"\n  {'─' * 60}")
        print(f"  Experiment: {exp_name} ({len(feats)} features)")
        t1 = time.time()

        preds = train_ensemble(df, feats, ORIGINAL_WINDOWS, seeds=SEEDS,
                               cs_rank_exclude=no_rank)
        if preds is None:
            print(f"  ⚠️  {exp_name}: training failed")
            continue

        port = simulate_with_hybrid_costs(preds, regime_df, PROD_CFG)
        if port.empty:
            print(f"  ⚠️  {exp_name}: no returns")
            continue

        m = compute_metrics(port)
        wm = compute_window_metrics(port, preds)
        elapsed = time.time() - t1

        result = {
            "experiment": exp_name,
            "n_features": len(feats),
            "sharpe": m["sharpe"],
            "total_ret%": m["total_return_pct"],
            "maxDD%": m["max_dd_pct"],
            "win_rate%": m["win_rate"],
            "W1": wm.get("W1", "?"),
            "W2": wm.get("W2", "?"),
            "W3": wm.get("W3", "?"),
        }
        results.append(result)
        print(f"  [{exp_name:22s}] {len(feats)}f  Sharpe={m['sharpe']:5.2f}  "
              f"Ret={m['total_return_pct']:+6.1f}%  DD={m['max_dd_pct']:5.1f}%  "
              f"W1={wm.get('W1','?'):5.2f} W2={wm.get('W2','?'):5.2f} W3={wm.get('W3','?'):5.2f}  "
              f"({elapsed/60:.1f}min)")

    # ── Summary ───────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  R61 RESULTS SUMMARY")
    print("=" * 90)
    print(f"  {'Experiment':<24} {'Feats':>6} {'Sharpe':>7} {'Ret%':>7} {'MaxDD%':>7} "
          f"{'WR%':>6} {'W1':>6} {'W2':>6} {'W3':>6}")
    print("  " + "-" * 80)

    baseline_sharpe = next((r["sharpe"] for r in results if r["experiment"] == "baseline_31f"), 0)

    for r in sorted(results, key=lambda x: x["sharpe"], reverse=True):
        delta = r["sharpe"] - baseline_sharpe
        marker = f" (Δ{delta:+.2f})" if r["experiment"] != "baseline_31f" else " ← BASELINE"
        print(f"  {r['experiment']:<24} {r['n_features']:>6} {r['sharpe']:>7.2f} "
              f"{r['total_ret%']:>+7.1f} {r['maxDD%']:>7.1f} {r['win_rate%']:>6.1f} "
              f"{r['W1']:>6} {r['W2']:>6} {r['W3']:>6}{marker}")

    print(f"\n  Total elapsed: {(time.time()-t0)/60:.1f}min")
    print("\n  ✅ R61 COMPLETE")

    # Save
    out_path = "/data/datasets/results_r61_temporal.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
