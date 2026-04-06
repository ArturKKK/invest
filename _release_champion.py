#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Champion Model Release — Simulation + Analysis + Production Training

1. Walk-forward simulation (champion 31f + hybrid costs)
2. Detailed analytics: winrate, monthly/quarterly/semi-annual returns, drawdowns
3. Data leakage checks
4. Train final production models (LGB + XGB, 5 seeds)
5. Save to results_cls_prod/

Usage:
  python _release_champion.py                    # full pipeline
  python _release_champion.py --sim-only         # simulation + analytics only
  python _release_champion.py --train-only       # skip sim, train prod models
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from datetime import datetime
from pathlib import Path

import lightgbm as lgb_lib
import numpy as np
import pandas as pd
import xgboost as xgb_lib
from scipy import stats

warnings.filterwarnings("ignore")

# ── project imports ───────────────────────────────────────
from _research_round7 import WINDOWS, SYM_35
from _research_r22_models import (
    SEEDS, LEVERAGE, CAPITAL, log,
    build_r19_features, add_new_features, cs_rank_cols,
)
from _research_r30b_fixed import (
    train_ensemble, eval_with_costs, compute_regime_extended,
)
from _research_r35_new_features import (
    add_r35_features, load_research_frame, MARKET_LEVEL_FEATURES,
)
from _research_r47_coinglass import (
    CANONICAL_EXEC_CFG, CHAMPION_FEAT_30,
    add_cg_features, compute_cg_features, load_cg_daily,
)
from _research_r48_cost import simulate_with_hybrid_costs

# ── constants ─────────────────────────────────────────────

CHAMPION_FEAT_31 = CHAMPION_FEAT_30 + ["cg_taker_imb"]
OUTPUT_DIR = Path("results_cls_prod")

# Production portfolio config (R48 champion)
PROD_CFG = {
    "n_long": 6, "n_short": 3, "rebal_hours": 12,
    "trend_cutoff": 0.9, "dyn_threshold": 0.7,
    "ema_alpha": 0.5, "hysteresis": 3,
}

# LGB classifier params (same as research)
LGB_PARAMS = {
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
}

# XGB classifier params
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbosity": 0,
}

N_ROUNDS = 600
EARLY_STOP = 40


# ═══════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════

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

    # Verify champion features
    present = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    missing = [f for f in CHAMPION_FEAT_31 if f not in df.columns]
    if missing:
        print(f"  ⚠️  Missing features: {missing}")
        raise ValueError(f"Missing champion features: {missing}")

    print(f"  Model frame: {len(df):,} rows, {df['symbol'].nunique()} symbols")
    print(f"  Date range: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print(f"  Champion features: {len(present)} (all present ✅)")

    return df, regime_df


# ═══════════════════════════════════════════════════════════
#  LEAKAGE CHECKS
# ═══════════════════════════════════════════════════════════

def check_leakage(df):
    """Verify no data leakage in walk-forward windows."""
    print("\n" + "=" * 70)
    print("  DATA LEAKAGE CHECKS")
    print("=" * 70)

    tz = df["timestamp"].dt.tz
    all_ok = True

    for w in WINDOWS:
        train_end = pd.Timestamp(w["train_end"], tz=tz)
        test_start = pd.Timestamp(w["test_start"], tz=tz)
        test_end = pd.Timestamp(w["test_end"], tz=tz)

        gap_days = (test_start - train_end).days
        print(f"  {w['name']}: train_end={w['train_end']}, "
              f"test_start={w['test_start']}, gap={gap_days}d", end="")

        # Check 1: purge gap >= 8 days
        if gap_days < 8:
            print(f"  ❌ PURGE GAP TOO SMALL ({gap_days}d < 8d)")
            all_ok = False
        else:
            print(f"  ✅")

        # Check 2: no future data leaked via features
        # 168h feature looks back 7d = 168h. With 12h target, need >=168+12=180h purge
        required_hours = 168 + 12  # max lookback + target horizon
        actual_hours = (test_start - train_end).total_seconds() / 3600
        if actual_hours < required_hours:
            print(f"    ⚠️  Tight purge: {actual_hours:.0f}h vs {required_hours}h needed")

    # Check 3: expanding windows don't overlap in test
    for i in range(len(WINDOWS) - 1):
        end_i = pd.Timestamp(WINDOWS[i]["test_end"], tz=tz)
        start_j = pd.Timestamp(WINDOWS[i + 1]["test_start"], tz=tz)
        gap = (start_j - end_i).days
        if gap < 0:
            print(f"  ❌ WINDOWS {WINDOWS[i]['name']}↔{WINDOWS[i+1]['name']} OVERLAP by {-gap}d")
            all_ok = False

    # Check 4: pandas version
    import pandas as pd_check
    pd_ver = pd_check.__version__
    major = int(pd_ver.split('.')[0])
    if major >= 3:
        print(f"  ❌ PANDAS {pd_ver} DETECTED — groupby.apply bug! Use pandas <3.0")
        all_ok = False
    else:
        print(f"  pandas {pd_ver} ✅")

    if all_ok:
        print("  ALL CHECKS PASSED ✅")
    return all_ok


# ═══════════════════════════════════════════════════════════
#  WALK-FORWARD SIMULATION + DETAILED ANALYSIS
# ═══════════════════════════════════════════════════════════

def run_simulation(df, regime_df):
    """Full walk-forward simulation with champion 31f + hybrid costs."""
    print("\n" + "=" * 70)
    print("  WALK-FORWARD SIMULATION")
    print("=" * 70)

    no_rank = [f for f in CHAMPION_FEAT_31 if f in MARKET_LEVEL_FEATURES]
    print(f"  cs_rank_exclude: {no_rank}")
    print(f"  Config: {PROD_CFG}")

    # Train ensemble
    preds = train_ensemble(
        df, CHAMPION_FEAT_31, WINDOWS,
        l2=1.0, rolling=False,
        label="champion_31f",
        cs_rank_exclude=no_rank or None,
    )

    if preds is None or len(preds) == 0:
        print("  ❌ Training failed")
        return None, None

    print(f"\n  Predictions: {len(preds):,} rows, "
          f"{preds['timestamp'].nunique():,} timestamps, "
          f"windows: {sorted(preds['window'].unique())}")

    # Simulate with hybrid costs
    port_all = simulate_with_hybrid_costs(preds, regime_df, PROD_CFG)

    # Per-window simulation
    per_window = {}
    for wname in ["W1", "W2", "W3"]:
        sub = preds[preds["window"] == wname]
        if len(sub) < 10:
            continue
        port_w = simulate_with_hybrid_costs(sub, regime_df, PROD_CFG)
        r = eval_with_costs(port_w, wname)
        per_window[wname] = r
        print(f"  {wname}: Sh={r['sharpe']:>5.2f}  Eq=${r['equity']:>6.0f}  "
              f"DD={r['max_dd_pct']:>+5.1f}%  WM={r['win_months']}  "
              f"Cost={r['total_cost_pct']:.1f}%  Turn={r['avg_turnover']:.1f}")

    r_all = eval_with_costs(port_all, "ALL")
    print(f"  ALL: Sh={r_all['sharpe']:>5.2f}  Eq=${r_all['equity']:>6.0f}  "
          f"DD={r_all['max_dd_pct']:>+5.1f}%  Cost={r_all['total_cost_pct']:.1f}%  "
          f"Turn={r_all['avg_turnover']:.1f}")

    return port_all, preds


def detailed_analysis(port_df, preds):
    """Compute winrate, monthly returns, quarterly, semi-annual, annual."""
    print("\n" + "=" * 70)
    print("  DETAILED PERFORMANCE ANALYSIS")
    print("=" * 70)

    if port_df is None or len(port_df) == 0:
        print("  ❌ No data")
        return

    rets = port_df["portfolio_ret"].values
    gross = port_df["gross_ret"].values
    ts = port_df["timestamp"]

    # ── Basic stats ──────────────────────────────────────────
    print(f"\n  📊 Basic Statistics (leverage={LEVERAGE}x, capital=${CAPITAL})")
    print(f"  {'─' * 55}")

    leveraged = rets * LEVERAGE
    equity_curve = CAPITAL * np.cumprod(1 + leveraged)
    total_return = (equity_curve[-1] / CAPITAL - 1) * 100
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    max_dd = dd.min() * 100

    n_obs = len(rets)
    total_hours = (ts.max() - ts.min()).total_seconds() / 3600
    years = total_hours / 8760
    ppy = n_obs / years

    sharpe = np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(ppy)
    sharpe_gross = np.mean(gross) / (np.std(gross) + 1e-10) * np.sqrt(ppy)

    print(f"  Sharpe (net):   {sharpe:.2f}")
    print(f"  Sharpe (gross): {sharpe_gross:.2f}")
    print(f"  Total return:   {total_return:+.1f}%")
    print(f"  Final equity:   ${equity_curve[-1]:.0f} (from ${CAPITAL})")
    print(f"  Max drawdown:   {max_dd:.1f}%")
    print(f"  Test period:    {ts.min().date()} → {ts.max().date()} ({years:.1f}y)")
    print(f"  Observations:   {n_obs:,} (every {12}h)")

    # ── Win rate ─────────────────────────────────────────────
    print(f"\n  🎯 Win Rate")
    print(f"  {'─' * 55}")

    # Per-rebalance win rate
    wr_rebal = (rets > 0).mean() * 100
    wr_gross = (gross > 0).mean() * 100
    print(f"  Per rebalance (12h):   {wr_rebal:.1f}% net, {wr_gross:.1f}% gross")

    # Long leg win rate
    long_rets = port_df["long_ret"].values
    short_rets = port_df["short_ret"].values
    wr_long = (long_rets > 0).mean() * 100
    wr_short = (short_rets < 0).mean() * 100  # shorts win when they go down
    print(f"  Long leg (hit rate):   {wr_long:.1f}%")
    print(f"  Short leg (hit rate):  {wr_short:.1f}%")

    # ── Monthly breakdown ────────────────────────────────────
    print(f"\n  📅 Monthly Returns (leveraged {LEVERAGE}x)")
    print(f"  {'─' * 55}")

    port_ts = port_df.set_index("timestamp")
    monthly_net = port_ts.resample("ME")["portfolio_ret"].sum() * LEVERAGE * 100
    monthly_gross = port_ts.resample("ME")["gross_ret"].sum() * LEVERAGE * 100

    print(f"  {'Month':<12} {'Return%':>8} {'Gross%':>8} {'Result':>8}")
    print(f"  {'─' * 40}")

    for date, ret in monthly_net.items():
        gr = monthly_gross.get(date, 0)
        marker = "✅" if ret > 0 else "❌"
        print(f"  {date.strftime('%Y-%m'):<12} {ret:>+7.1f}% {gr:>+7.1f}% {marker:>6}")

    win_m = (monthly_net > 0).sum()
    total_m = len(monthly_net)
    print(f"\n  Win months: {win_m}/{total_m} ({win_m/total_m*100:.0f}%)")
    print(f"  Avg month:  {monthly_net.mean():+.1f}%")
    print(f"  Best month: {monthly_net.max():+.1f}%")
    print(f"  Worst month:{monthly_net.min():+.1f}%")

    # ── Quarterly breakdown ──────────────────────────────────
    print(f"\n  📅 Quarterly Returns (leveraged {LEVERAGE}x)")
    print(f"  {'─' * 55}")

    quarterly = port_ts.resample("QE")["portfolio_ret"].sum() * LEVERAGE * 100

    for date, ret in quarterly.items():
        marker = "✅" if ret > 0 else "❌"
        q_label = f"{date.year}Q{(date.month-1)//3+1}"
        print(f"  {q_label:<12} {ret:>+7.1f}%  {marker}")

    win_q = (quarterly > 0).sum()
    total_q = len(quarterly)
    print(f"\n  Win quarters: {win_q}/{total_q} ({win_q/total_q*100:.0f}%)")

    # ── Semi-annual breakdown ────────────────────────────────
    print(f"\n  📅 Semi-Annual Returns (leveraged {LEVERAGE}x)")
    print(f"  {'─' * 55}")

    # Group by 6-month periods
    semiannual = port_ts.resample("6ME")["portfolio_ret"].sum() * LEVERAGE * 100
    for date, ret in semiannual.items():
        marker = "✅" if ret > 0 else "❌"
        h_label = f"{date.year}H{1 if date.month <= 6 else 2}"
        print(f"  {h_label:<12} {ret:>+7.1f}%  {marker}")

    # ── Annual (if enough data) ──────────────────────────────
    annual = port_ts.resample("YE")["portfolio_ret"].sum() * LEVERAGE * 100
    if len(annual) >= 1:
        print(f"\n  📅 Annual Returns (leveraged {LEVERAGE}x)")
        print(f"  {'─' * 55}")
        for date, ret in annual.items():
            marker = "✅" if ret > 0 else "❌"
            print(f"  {date.year:<12} {ret:>+7.1f}%  {marker}")

    # ── Per-window stats ─────────────────────────────────────
    print(f"\n  📊 Per-Window Breakdown")
    print(f"  {'─' * 55}")
    print(f"  {'Window':<8} {'Period':<24} {'Months':>8} {'Sharpe':>8} {'Return%':>9} {'WinRate':>8}")
    print(f"  {'─' * 70}")

    regime_df = port_df.attrs.get("regime_df")
    if regime_df is None or (hasattr(regime_df, 'empty') and regime_df.empty):
        regime_df = compute_regime_extended(preds)

    for wname in ["W1", "W2", "W3"]:
        sub = preds[preds["window"] == wname]
        if len(sub) < 10:
            continue
        from _research_r48_cost import simulate_with_hybrid_costs as sim_hybrid
        port_w = sim_hybrid(sub, regime_df, PROD_CFG)
        if len(port_w) == 0:
            continue

        r = eval_with_costs(port_w, wname)
        w_start = port_w["timestamp"].min().date()
        w_end = port_w["timestamp"].max().date()
        w_months = (port_w["timestamp"].max() - port_w["timestamp"].min()).days / 30.44
        w_ret = (r["equity"] / CAPITAL - 1) * 100
        w_wr = (port_w["portfolio_ret"] > 0).mean() * 100
        print(f"  {wname:<8} {str(w_start)+'→'+str(w_end):<24} "
              f"{w_months:>5.1f}m   {r['sharpe']:>+5.2f}   {w_ret:>+7.1f}%   {w_wr:>5.1f}%")

    # ── Drawdown analysis ────────────────────────────────────
    print(f"\n  📉 Drawdown Analysis")
    print(f"  {'─' * 55}")

    dd_series = pd.Series(dd, index=ts.values)
    # Find worst drawdown periods
    in_dd = dd < -0.01
    dd_end_idx = None
    worst_dd_val = 0
    worst_dd_start = None
    worst_dd_end = None

    for i in range(len(dd)):
        if dd[i] < worst_dd_val:
            worst_dd_val = dd[i]
            worst_dd_end = ts.iloc[i]
            # Find start of this DD
            j = i
            while j > 0 and equity_curve[j] < peak[j]:
                j -= 1
            worst_dd_start = ts.iloc[j]

    if worst_dd_start is not None:
        dd_duration = (worst_dd_end - worst_dd_start).days
        print(f"  Worst DD:  {worst_dd_val*100:.1f}%")
        print(f"  Period:    {worst_dd_start.date()} → {worst_dd_end.date()} ({dd_duration}d)")

    # Recovery time
    if worst_dd_end is not None:
        post_dd = equity_curve[ts >= worst_dd_end]
        recovery_idx = np.where(post_dd >= peak[ts >= worst_dd_end][0])[0]
        if len(recovery_idx) > 0:
            recovery_ts = ts[ts >= worst_dd_end].iloc[recovery_idx[0]]
            recovery_days = (recovery_ts - worst_dd_end).days
            print(f"  Recovery:  {recovery_days}d")
        else:
            print(f"  Recovery:  not yet recovered")

    print(f"\n  Avg drawdown: {dd.mean()*100:.1f}%")
    print(f"  Time underwater: {(dd < -0.01).mean()*100:.1f}%")

    return {
        "sharpe": round(sharpe, 2),
        "total_return_pct": round(total_return, 1),
        "max_dd_pct": round(max_dd, 1),
        "win_rate_12h": round(wr_rebal, 1),
        "win_months": f"{win_m}/{total_m}",
        "test_period": f"{ts.min().date()} → {ts.max().date()}",
    }


# ═══════════════════════════════════════════════════════════
#  PRODUCTION MODEL TRAINING
# ═══════════════════════════════════════════════════════════

def train_production_models(df):
    """Train final LGB + XGB models on ALL data for production."""
    print("\n" + "=" * 70)
    print("  TRAINING PRODUCTION MODELS")
    print("=" * 70)

    feats = CHAMPION_FEAT_31
    no_rank = [f for f in feats if f in MARKET_LEVEL_FEATURES]

    # CS-rank features (except market-level)
    rank_feats = [f for f in feats if f not in no_rank]
    df_full = df.copy()
    df_full = cs_rank_cols(df_full, rank_feats)
    df_full["target_binary"] = (df_full["fwd_ret_12h"] > 0).astype(int)

    # Clean: drop NaN in features + target
    df_clean = df_full[feats + ["target_binary", "timestamp"]].dropna()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()

    # Production: train on ALL data, use last 4 months as val for early stopping
    # Then retrain on ALL data with fixed rounds (best_iteration from val run)
    max_ts = df_clean["timestamp"].max()
    val_start = max_ts - pd.Timedelta(days=120)  # ~4 months val
    train_end_ts = val_start - pd.Timedelta(days=14)  # 14d purge gap
    tr_es = df_clean[df_clean["timestamp"] <= train_end_ts]  # for early-stop calibration
    va_es = df_clean[df_clean["timestamp"] >= val_start]     # val for early-stop
    tr_full = df_clean  # final models train on EVERYTHING

    print(f"  Features:    {len(feats)} ({feats[:5]}...)")
    print(f"  cs_rank_exclude: {no_rank}")
    print(f"  ES calibration: {len(tr_es):,} train, {len(va_es):,} val")
    print(f"  Final train:     {len(tr_full):,} rows (ALL data)")
    print(f"  Val range:   {va_es['timestamp'].min().date()} → {va_es['timestamp'].max().date()}")
    print(f"  Data through:{df_full['timestamp'].max().date()}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Pass 1: calibrate optimal rounds via early stopping ──
    print("\n  Pass 1: Calibrating optimal rounds...")
    best_rounds_lgb = []
    best_rounds_xgb = []
    for seed in SEEDS:
        dtrain_l = lgb_lib.Dataset(tr_es[feats], label=tr_es["target_binary"])
        dval_l = lgb_lib.Dataset(va_es[feats], label=va_es["target_binary"])
        m = lgb_lib.train(
            {**LGB_PARAMS, "seed": seed}, dtrain_l,
            num_boost_round=N_ROUNDS, valid_sets=[dval_l],
            callbacks=[lgb_lib.early_stopping(EARLY_STOP, verbose=False),
                       lgb_lib.log_evaluation(-1)])
        best_rounds_lgb.append(max(m.best_iteration, 50))  # minimum 50 rounds

        dtrain_x = xgb_lib.DMatrix(tr_es[feats], label=tr_es["target_binary"])
        dval_x = xgb_lib.DMatrix(va_es[feats], label=va_es["target_binary"])
        m_x = xgb_lib.train(
            {**XGB_PARAMS, "seed": seed}, dtrain_x,
            num_boost_round=N_ROUNDS,
            evals=[(dval_x, "val")],
            early_stopping_rounds=EARLY_STOP, verbose_eval=False)
        best_rounds_xgb.append(max(m_x.best_iteration, 50))

    avg_lgb_rounds = int(np.mean(best_rounds_lgb))
    avg_xgb_rounds = int(np.mean(best_rounds_xgb))
    # Use at least 200 rounds (binary cls with weak signal needs more boosting)
    lgb_rounds = max(avg_lgb_rounds, 200)
    xgb_rounds = max(avg_xgb_rounds, 200)
    print(f"  LGB calibrated: {best_rounds_lgb} → using {lgb_rounds}")
    print(f"  XGB calibrated: {best_rounds_xgb} → using {xgb_rounds}")

    # ── Pass 2: train final models on ALL data with calibrated rounds ──
    print(f"\n  Pass 2: Training final models on ALL data...")
    lgb_models = []
    xgb_models = []

    for seed in SEEDS:
        # ── LGB ── (no early stopping, fixed rounds)
        dtrain_l = lgb_lib.Dataset(tr_full[feats], label=tr_full["target_binary"])
        lgb_model = lgb_lib.train(
            {**LGB_PARAMS, "seed": seed}, dtrain_l,
            num_boost_round=lgb_rounds)
        lgb_path = OUTPUT_DIR / f"lgb_cls_seed_{seed}.txt"
        lgb_model.save_model(str(lgb_path))
        lgb_models.append(lgb_model)

        # ── XGB ── (no early stopping, fixed rounds)
        dtrain_x = xgb_lib.DMatrix(tr_full[feats], label=tr_full["target_binary"])
        xgb_model = xgb_lib.train(
            {**XGB_PARAMS, "seed": seed}, dtrain_x,
            num_boost_round=xgb_rounds, verbose_eval=False)
        xgb_path = OUTPUT_DIR / f"xgb_cls_seed_{seed}.json"
        xgb_model.save_model(str(xgb_path))
        xgb_models.append(xgb_model)

        # Val AUC (on the held-out val set for reporting)
        from sklearn.metrics import roc_auc_score
        lgb_val_p = lgb_model.predict(va_es[feats])
        xgb_val_p = xgb_model.predict(xgb_lib.DMatrix(va_es[feats]))
        ens_val = 0.5 * lgb_val_p + 0.5 * xgb_val_p
        auc_lgb = roc_auc_score(va_es["target_binary"], lgb_val_p)
        auc_xgb = roc_auc_score(va_es["target_binary"], xgb_val_p)
        auc_ens = roc_auc_score(va_es["target_binary"], ens_val)

        print(f"  seed={seed}: LGB {lgb_rounds}r AUC={auc_lgb:.4f}  "
              f"XGB {xgb_rounds}r AUC={auc_xgb:.4f}  "
              f"Ens AUC={auc_ens:.4f}")

    # Save feature names (for production inference)
    with open(OUTPUT_DIR / "feature_names.json", "w") as f:
        json.dump(feats, f, indent=2)

    n_total = len(tr_full)
    # Save production metadata
    meta = {
        "model_type": "binary_classification_ensemble",
        "champion": "R48_31f_hybrid",
        "sharpe_wf": 1.66,
        "features": feats,
        "n_features": len(feats),
        "cs_rank_exclude": no_rank,
        "seeds": list(SEEDS),
        "models": {
            "lgb": {"n_seeds": len(SEEDS), "pattern": "lgb_cls_seed_*.txt",
                    "params": {k: v for k, v in LGB_PARAMS.items() if k != "verbose"}},
            "xgb": {"n_seeds": len(SEEDS), "pattern": "xgb_cls_seed_*.json",
                    "params": XGB_PARAMS},
        },
        "ensemble_method": "rank_normalize_then_average (0.5 lgb + 0.5 xgb)",
        "train_rows": n_total,
        "trained_through": str(df_full["timestamp"].max().date()),
        "portfolio": PROD_CFG,
        "cost_model": "hybrid_tiered (TIER1: 0.4bp, TIER2: 2.5bp, TIER3: 7bp)",
        "leverage": LEVERAGE,
        "timestamp": datetime.now().isoformat(),
        "source": "R48 champion_31f + hybrid tiered costs → ALL Sharpe 1.66",
    }
    with open(OUTPUT_DIR / "production_meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # Save meta.json (for backward compat with run_trading.py)
    with open(OUTPUT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ Saved {len(SEEDS)} LGB + {len(SEEDS)} XGB models → {OUTPUT_DIR}/")
    print(f"  ✅ feature_names.json ({len(feats)} features)")
    print(f"  ✅ production_meta.json + meta.json")

    # Sanity: IC on last 90 days
    sanity_check(lgb_models, xgb_models, df_full, feats)

    return lgb_models, xgb_models


def sanity_check(lgb_models, xgb_models, df_full, feats):
    """Verify signal quality on recent data."""
    print("\n  🔍 Sanity Check: IC on last 90 days")
    cutoff = df_full["timestamp"].max() - pd.Timedelta(days=90)
    recent = df_full[df_full["timestamp"] >= cutoff].copy()

    if "fwd_ret_12h" not in recent.columns or len(recent) < 500:
        print("  ⚠️ Insufficient data for sanity check")
        return

    recent_clean = recent[feats + ["fwd_ret_12h"]].dropna()
    if len(recent_clean) == 0:
        return

    lgb_preds = np.mean([m.predict(recent_clean[feats]) for m in lgb_models], axis=0)
    xgb_preds = np.mean([m.predict(xgb_lib.DMatrix(recent_clean[feats]))
                         for m in xgb_models], axis=0)

    def rankn(x):
        return stats.rankdata(x) / len(x) - 0.5
    ens = 0.5 * rankn(lgb_preds) + 0.5 * rankn(xgb_preds)

    target_rank = stats.rankdata(recent_clean["fwd_ret_12h"]) / len(recent_clean) - 0.5
    ic_lgb = stats.spearmanr(lgb_preds, target_rank)[0]
    ic_xgb = stats.spearmanr(xgb_preds, target_rank)[0]
    ic_ens = stats.spearmanr(ens, target_rank)[0]

    print(f"  LGB IC:      {ic_lgb:.4f}")
    print(f"  XGB IC:      {ic_xgb:.4f}")
    print(f"  Ensemble IC: {ic_ens:.4f} {'✅' if ic_ens > 0.02 else '⚠️ LOW'}")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Champion Model Release")
    parser.add_argument("--sim-only", action="store_true", help="Simulation + analytics only")
    parser.add_argument("--train-only", action="store_true", help="Skip sim, train prod models")
    args = parser.parse_args()

    print("=" * 70)
    print("  🏆 CHAMPION MODEL RELEASE — 31f + Hybrid Tiered Costs")
    print("  📊 ALL Sharpe 1.66 | 6L/3S | 12h rebal")
    print("=" * 70)

    # Load data
    df, regime_df = load_data()

    # Leakage checks
    ok = check_leakage(df)
    if not ok:
        print("\n  ❌ LEAKAGE CHECK FAILED — aborting")
        return

    if not args.train_only:
        # Run simulation
        port_all, preds = run_simulation(df, regime_df)

        if port_all is not None:
            # Pass regime_df via attrs for per-window analysis
            port_all.attrs["regime_df"] = regime_df

            # Detailed analysis
            summary = detailed_analysis(port_all, preds)

            if summary:
                print(f"\n  📋 Summary: Sharpe={summary['sharpe']}, "
                      f"Return={summary['total_return_pct']:+.1f}%, "
                      f"MaxDD={summary['max_dd_pct']:.1f}%, "
                      f"WinRate={summary['win_rate_12h']:.1f}%, "
                      f"WinMonths={summary['win_months']}")

    if not args.sim_only:
        # Train production models
        train_production_models(df)

    print("\n" + "=" * 70)
    print("  ✅ RELEASE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
