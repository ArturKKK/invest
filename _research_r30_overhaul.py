#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R30 — Model Overhaul: Fix what's actually broken.

Current model: Sh=4.52 on W1 (Oct24-Jan25) but Sh=1.10 on W3 (Nov25-Mar26).
Live: lost $14 in 2 days. Model doesn't cover transaction costs on recent data.

This R30 tests COMBINATIONS of the following improvements:
  1. Transaction cost model in simulation (realistic)
  2. Rolling window training (12mo) vs expanding window
  3. New features: macro, volume dynamics, TA (CLEAN pipeline)
  4. Regime-aware portfolio: short-only in crashes, skip in chop
  5. BTC crash gate: hard skip when BTC ret_24h < -5%
  6. Fewer positions (3L/2S) to cut commissions
  7. 24h rebalance instead of 12h

Metrics: focus on W3 Sharpe (most recent = most relevant).
If W3 Sharpe after costs < 1.0, the config is NOT deployable.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from scipy import stats
import warnings, time, sys, os

warnings.filterwarnings("ignore")

from _research_round7 import (
    SYM_35, WINDOWS, compute_regime,
)
from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal
from _research_r22_models import (
    FEATURES_23, SEEDS, LEVERAGE, CAPITAL,
    log, build_r19_features, add_new_features, cs_rank_cols,
)

# ══════════════════════════════════════════════════════════════
# FEATURE SETS
# ══════════════════════════════════════════════════════════════

# Production set (baseline)
FEAT_23 = FEATURES_23[:]

# Extended: add features that R29 found useful + volume dynamics
FEAT_EXTENDED = FEAT_23 + [
    # Derivatives (top R29 forward winners)
    "global_ls_ratio", "taker_buy_sell_ratio", "buy_pressure",
    "funding_rate_binance", "funding_zscore",
    # Volume dynamics
    "ret_std_24h", "ret_skew_24h", "ret_kurt_24h",
    "vol_mom_12h", "vol_mom_24h",
    # TA
    "rsi_14", "cci_14",
    # Macro-regime
    "fng_value", "fng_momentum",
]

# Rolling window training boundaries
WINDOWS_ROLLING = [
    {
        "name": "W1", "train_months": 12,
        "train_end": "2024-06-01", "val_start": "2024-06-01", "val_end": "2024-09-30",
        "test_start": "2024-10-15", "test_end": "2025-01-31",
    },
    {
        "name": "W2", "train_months": 12,
        "train_end": "2025-01-01", "val_start": "2025-01-01", "val_end": "2025-04-30",
        "test_start": "2025-05-15", "test_end": "2025-08-31",
    },
    {
        "name": "W3", "train_months": 12,
        "train_end": "2025-07-01", "val_start": "2025-07-01", "val_end": "2025-10-31",
        "test_start": "2025-11-15", "test_end": "2026-03-17",
    },
]


# ══════════════════════════════════════════════════════════════
# TRANSACTION COST MODEL
# ══════════════════════════════════════════════════════════════

def simulate_with_costs(merged, regime_df, cfg):
    """
    Simulate L/S portfolio WITH realistic transaction costs.
    Costs: taker_fee=0.05% per side × leverage + slippage + funding.
    """
    n_long = cfg.get("n_long", 6)
    n_short = cfg.get("n_short", 3)
    trend_cutoff = cfg.get("trend_cutoff", 0.9)
    rebal_hours = cfg.get("rebal_hours", 12)
    leverage = cfg.get("leverage", LEVERAGE)
    crash_gate = cfg.get("crash_gate", False)  # skip longs when BTC crashing
    short_only_crash = cfg.get("short_only_crash", False)  # only short when BTC crashing
    crash_threshold = cfg.get("crash_threshold", -0.05)  # BTC ret_24h threshold

    # Cost params (OKX taker fees through proxy)
    taker_fee = 0.0005  # 5bps per side
    slippage = 0.0002   # 2bps slippage
    funding_per_12h = 0.00008  # ~1bp per 12h avg

    cost_per_trade = (taker_fee + slippage) * leverage  # cost per $notional

    all_rets = []
    prev_longs = set()
    prev_shorts = set()

    timestamps_sorted = sorted(merged["timestamp"].unique())
    grouped = {ts: grp for ts, grp in merged.groupby("timestamp")}
    rebal_timestamps = timestamps_sorted[::rebal_hours]

    for ts in rebal_timestamps:
        if ts not in regime_df.index or ts not in grouped:
            continue
        row = regime_df.loc[ts]
        trend_str = row.get("trend_strength", 0)
        trend_dir = row.get("trend_direction", 0)

        if trend_str > trend_cutoff:
            continue

        grp = grouped[ts].copy()
        n = len(grp)

        # BTC crash gate: check BTC 24h return
        btc_rows = grp[grp["symbol"].str.startswith("BTC")]
        btc_ret_24h = None
        if len(btc_rows) > 0 and "btc_ret_24h" in regime_df.columns:
            btc_ret_24h = regime_df.loc[ts, "btc_ret_24h"] if ts in regime_df.index else None
        elif "btc_ret_7d" in regime_df.columns:
            btc_ret_24h = regime_df.loc[ts].get("btc_ret_7d", 0) / 7 if ts in regime_df.index else None

        # Adjust portfolio based on crash detection
        nl, ns = n_long, n_short
        if crash_gate and btc_ret_24h is not None and btc_ret_24h < crash_threshold:
            if short_only_crash:
                nl = 0  # no longs during crash
                ns = min(ns + 2, n // 3)  # more shorts
            else:
                continue  # skip entirely

        nl = min(nl, n // 3)
        ns = min(ns, n // 3)
        if nl == 0 and ns == 0:
            continue

        # Dynamic exposure
        exposure = 1.0
        dyn_threshold = cfg.get("dyn_threshold", 0.5)
        if dyn_threshold is not None and trend_str > dyn_threshold:
            exposure = max(0.1, 1.0 - (trend_str - dyn_threshold) /
                          (trend_cutoff - dyn_threshold + 1e-10) * 0.5)

        grp["pred_rank"] = grp["pred"].rank(ascending=False)
        new_longs = set(grp[grp["pred_rank"] <= nl]["symbol"].tolist()) if nl > 0 else set()
        new_shorts = set(grp[grp["pred_rank"] > (n - ns)]["symbol"].tolist()) if ns > 0 else set()

        # Count turnover (new positions)
        new_opened = (new_longs - prev_longs) | (new_shorts - prev_shorts)
        closed = (prev_longs - new_longs) | (prev_shorts - new_shorts)
        turnover_count = len(new_opened) + len(closed)
        total_positions = len(new_longs) + len(new_shorts)

        # Cost: turnover × cost_per_trade + funding on all positions
        if total_positions > 0:
            # Each turned-over position costs 2× (close old + open new)
            turnover_cost = turnover_count * cost_per_trade / total_positions
            holding_cost = funding_per_12h * leverage * rebal_hours / 12
            total_cost = turnover_cost + holding_cost
        else:
            total_cost = 0

        prev_longs = new_longs
        prev_shorts = new_shorts

        longs = grp[grp["symbol"].isin(new_longs)]
        shorts = grp[grp["symbol"].isin(new_shorts)]

        long_ret = longs["fwd_ret"].mean() if len(longs) > 0 else 0
        short_ret = shorts["fwd_ret"].mean() if len(shorts) > 0 else 0

        if nl > 0 and ns > 0:
            port_ret = 0.5 * long_ret - 0.5 * short_ret
        elif ns > 0:
            # Short-only mode during crash
            port_ret = -short_ret  # full weight on shorts
        else:
            port_ret = long_ret

        port_ret *= exposure
        port_ret -= total_cost  # subtract costs

        all_rets.append({
            "timestamp": ts,
            "portfolio_ret": port_ret,
            "turnover": turnover_count,
            "cost": total_cost,
        })

    if not all_rets:
        return pd.DataFrame(columns=["timestamp", "portfolio_ret", "turnover", "cost"])
    return pd.DataFrame(all_rets)


# ══════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════

def eval_with_costs(port_df, name, leverage=LEVERAGE, capital=CAPITAL):
    """Evaluate portfolio with cost breakdown."""
    if len(port_df) == 0:
        return {"name": name, "sharpe": 0, "equity": capital, "worst_m": 0,
                "total_cost": 0, "n_periods": 0}

    rets = port_df["portfolio_ret"].values
    n_obs = len(rets)

    ts = port_df["timestamp"]
    total_hours = (ts.max() - ts.min()).total_seconds() / 3600
    years = max(total_hours / 8760, 0.01)
    ppy = n_obs / years

    mean_r = np.mean(rets)
    std_r = np.std(rets) + 1e-10
    sharpe = mean_r / std_r * np.sqrt(ppy)
    leveraged = rets * leverage
    equity_curve = capital * np.cumprod(1 + leveraged)
    final_eq = equity_curve[-1]

    # Monthly breakdown
    monthly = port_df.set_index("timestamp").resample("ME")["portfolio_ret"].sum()
    worst_m = monthly.min() if len(monthly) > 0 else 0
    win_months = (monthly > 0).sum()
    total_months = len(monthly)

    total_cost = port_df["cost"].sum() if "cost" in port_df.columns else 0
    avg_turnover = port_df["turnover"].mean() if "turnover" in port_df.columns else 0

    r = {
        "name": name, "sharpe": round(sharpe, 2),
        "equity": round(final_eq, 0),
        "worst_m": round(worst_m * 100, 1),
        "win_months": f"{win_months}/{total_months}",
        "total_cost_pct": round(total_cost * 100, 2),
        "avg_turnover": round(avg_turnover, 1),
        "n_periods": n_obs,
    }
    return r


def eval_per_window(preds, regime_df, cfg, label=""):
    """Per-window evaluation with costs."""
    results = {}
    for wname in ["W1", "W2", "W3"]:
        sub = preds[preds["window"] == wname]
        if len(sub) < 10:
            results[wname] = {"sharpe": 0}
            continue
        port = simulate_with_costs(sub, regime_df, cfg)
        r = eval_with_costs(port, f"{label}_{wname}")
        results[wname] = r
        log(f"  {wname}: Sh={r['sharpe']:>5.2f}  Eq=${r['equity']:>6.0f}  "
            f"Wr={r['worst_m']:>+5.1f}%  WM={r['win_months']}  "
            f"Cost={r['total_cost_pct']:.1f}%  Turn={r['avg_turnover']:.1f}")

    # Combined
    port_all = simulate_with_costs(preds, regime_df, cfg)
    r_all = eval_with_costs(port_all, label)
    log(f"  ALL: Sh={r_all['sharpe']:>5.2f}  Eq=${r_all['equity']:>6.0f}  "
        f"Wr={r_all['worst_m']:>+5.1f}%  Cost={r_all['total_cost_pct']:.1f}%")

    results["ALL"] = r_all
    return results


# ══════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════

def train_ensemble(df, feats, windows, seeds=SEEDS, l2=1.0,
                   rolling=False, use_cs_rank=True, label=""):
    """Train LGB+XGB ensemble. rolling=True uses 12mo train window."""
    avail = [f for f in feats if f in df.columns]
    missing = [f for f in feats if f not in df.columns]
    if missing:
        log(f"  ⚠️  Missing features: {missing}")
    log(f"  Training: {label}, {len(avail)}f, L2={l2}, {'rolling' if rolling else 'expanding'}, "
        f"{len(seeds)} seeds × {len(windows)} windows")

    all_lgb, all_xgb = [], []
    tz = df["timestamp"].dt.tz

    for seed in seeds:
        params_lgb = {
            "objective": "binary", "metric": "auc",
            "learning_rate": 0.03, "num_leaves": 63,
            "min_child_samples": 100, "subsample": 0.8,
            "colsample_bytree": 0.8, "lambda_l2": l2,
            "verbose": -1, "n_jobs": -1, "seed": seed,
        }
        params_xgb = {
            "objective": "binary:logistic", "eval_metric": "auc",
            "learning_rate": 0.03, "max_depth": 6,
            "min_child_weight": 100, "subsample": 0.8,
            "colsample_bytree": 0.8, "reg_lambda": l2,
            "seed": seed, "n_jobs": -1, "verbosity": 0,
        }

        for w in windows:
            te_end = pd.Timestamp(w["test_end"], tz=tz)
            te_start = pd.Timestamp(w["test_start"], tz=tz)
            tr_end = pd.Timestamp(w["train_end"], tz=tz)
            va_start = pd.Timestamp(w["val_start"], tz=tz)
            va_end = pd.Timestamp(w["val_end"], tz=tz)

            if rolling:
                # Rolling: only use last N months for training
                train_months = w.get("train_months", 12)
                tr_start = tr_end - pd.DateOffset(months=train_months)
                train_ = df[(df["timestamp"] >= tr_start) &
                            (df["timestamp"] < tr_end)].copy()
            else:
                train_ = df[df["timestamp"] < tr_end].copy()

            val_ = df[(df["timestamp"] >= va_start) &
                      (df["timestamp"] < va_end)].copy()
            test_ = df[(df["timestamp"] >= te_start) &
                       (df["timestamp"] <= te_end)].copy()

            if len(train_) < 5000 or len(test_) < 200:
                log(f"    {w['name']}/s{seed}: skip (train={len(train_)}, test={len(test_)})")
                continue

            if use_cs_rank:
                train_ = cs_rank_cols(train_, avail)
                val_ = cs_rank_cols(val_, avail)
                test_ = cs_rank_cols(test_, avail)

            for d in [train_, val_, test_]:
                d["target_binary"] = (d["fwd_ret_12h"] > 0).astype(int)

            tr = train_[avail + ["target_binary"]].dropna()
            va = val_[avail + ["target_binary"]].dropna()
            te = test_[avail + ["target_binary", "timestamp", "symbol"]].dropna()
            fwd = test_[["timestamp", "symbol", "fwd_ret_12h"]].rename(
                columns={"fwd_ret_12h": "fwd_ret"}).dropna()

            for d in [tr, va, te]:
                d.replace([np.inf, -np.inf], np.nan, inplace=True)
            tr = tr.dropna()
            va = va.dropna()
            te = te.dropna()
            if len(te) == 0:
                continue

            # LGB
            dt = lgb.Dataset(tr[avail], label=tr["target_binary"])
            dv = lgb.Dataset(va[avail], label=va["target_binary"])
            m = lgb.train(params_lgb, dt, num_boost_round=600,
                          valid_sets=[dv],
                          callbacks=[lgb.early_stopping(40, verbose=False),
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
            m_x = xgb.train(params_xgb, dt_x, num_boost_round=600,
                             evals=[(dv_x, "val")],
                             early_stopping_rounds=40, verbose_eval=False)
            p_x = m_x.predict(xgb.DMatrix(te[avail]))
            rec2 = te[["timestamp", "symbol"]].copy()
            rec2["pred_xgb"] = p_x
            rec2 = rec2.merge(fwd, on=["timestamp", "symbol"], how="inner")
            rec2["window"] = w["name"]
            rec2["seed"] = seed
            all_xgb.append(rec2)

            if seed == seeds[0]:
                ic = stats.spearmanr(rec["pred_lgb"], rec["fwd_ret"])[0]
                log(f"    {w['name']}/s{seed}: train={len(tr):,}, test={len(te):,}, IC={ic:.4f}")

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


# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (CLEAN — no NaN contamination)
# ══════════════════════════════════════════════════════════════

def add_extra_features_clean(df):
    """Add extra features WITHOUT contaminating existing columns.
    Only creates NEW columns, never modifies existing NaN patterns."""
    n_before = len(df.columns)
    new_cols = []

    for sym, gdf in df.groupby("symbol"):
        idx = gdf.index
        close = gdf["close"]
        volume = gdf["volume"]
        high = gdf["high"]
        low = gdf["low"]

        # Volume dynamics
        if "buy_pressure" not in df.columns:
            bp = (close - low) / (high - low + 1e-10)
            df.loc[idx, "buy_pressure"] = bp

        # Return distribution features
        ret_1h = close.pct_change(1)
        for w in [24, 48]:
            col_std = f"ret_std_{w}h"
            col_skew = f"ret_skew_{w}h"
            col_kurt = f"ret_kurt_{w}h"
            if col_std not in df.columns:
                df.loc[idx, col_std] = ret_1h.rolling(w, min_periods=w//2).std()
            if col_skew not in df.columns:
                df.loc[idx, col_skew] = ret_1h.rolling(w, min_periods=w//2).skew()
            if col_kurt not in df.columns:
                df.loc[idx, col_kurt] = ret_1h.rolling(w, min_periods=w//2).kurt()

        # Volume momentum
        for w in [12, 24]:
            col = f"vol_mom_{w}h"
            if col not in df.columns:
                vol_ma = volume.rolling(w, min_periods=w//2).mean()
                vol_ma_long = volume.rolling(w*4, min_periods=w*2).mean()
                df.loc[idx, col] = vol_ma / (vol_ma_long + 1e-10) - 1

        # RSI (simple, no ta library needed)
        for period in [14]:
            col = f"rsi_{period}"
            if col not in df.columns:
                delta = close.diff()
                gain = delta.clip(lower=0).rolling(period).mean()
                loss = (-delta.clip(upper=0)).rolling(period).mean()
                rs = gain / (loss + 1e-10)
                df.loc[idx, col] = 100 - 100 / (1 + rs)

        # CCI
        for period in [14]:
            col = f"cci_{period}"
            if col not in df.columns:
                tp = (high + low + close) / 3
                tp_ma = tp.rolling(period).mean()
                tp_std = tp.rolling(period).std()
                df.loc[idx, col] = (tp - tp_ma) / (0.015 * tp_std + 1e-10)

    # FNG features (already should be present from add_new_features)
    if "fng_value" in df.columns and "fng_momentum" not in df.columns:
        df["fng_momentum"] = df.groupby("symbol")["fng_value"].transform(
            lambda x: x.diff(24))

    n_after = len(df.columns)
    new_cols = n_after - n_before
    log(f"  Added {new_cols} extra features (clean)")

    # Fill NaN in NEW columns only with 0 (safe — doesn't affect existing columns)
    for col in df.columns:
        if col not in ["timestamp", "symbol", "open", "high", "low", "close", "volume"]:
            pass  # don't auto-fill — let dropna handle it per feature set
    return df


def compute_regime_extended(df):
    """Compute regime with BTC 24h return for crash gate."""
    regime_df = compute_regime(df)

    # Add BTC 24h return
    btc = df[df["symbol"].str.startswith("BTC")].copy()
    if len(btc) > 0:
        btc = btc.sort_values("timestamp").set_index("timestamp")
        btc["btc_ret_24h"] = btc["close"].pct_change(24)
        regime_df = regime_df.join(btc[["btc_ret_24h"]], how="left")
        regime_df["btc_ret_24h"] = regime_df["btc_ret_24h"].ffill()

    return regime_df


# ══════════════════════════════════════════════════════════════
# MAIN EXPERIMENTS
# ══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log("=" * 80)
    log("  R30 — MODEL OVERHAUL: Transaction costs, regime, features, rolling")
    log("=" * 80)

    # ── 1. Load data (CLEAN pipeline only) ──
    log("\n[1] Loading data (CLEAN pipeline)...")
    ohlcv = load_ohlcv()
    ohlcv = ohlcv[ohlcv["symbol"].isin(SYM_35)]
    derivs = load_derivatives()
    df = build_features_minimal(ohlcv, derivs)
    df = build_r19_features(df)
    df, _ = add_new_features(df)
    df = df[df["symbol"].isin(SYM_35)].copy()
    log(f"  Base: {len(df):,} rows, {len(df.columns)} cols")

    # Add extra features (clean)
    df = add_extra_features_clean(df)
    log(f"  After extras: {len(df):,} rows, {len(df.columns)} cols")

    # Compute regime
    regime_df = compute_regime_extended(df)

    # ── 2. Train models ──

    # ── EXP-A: Baseline 23f expanding (= current prod) WITH costs ──
    log("\n" + "=" * 60)
    log("[EXP-A] Baseline: 23f, expanding, L2=1 (current prod, WITH costs)")
    log("=" * 60)
    preds_A = train_ensemble(df, FEAT_23, WINDOWS, l2=1.0, rolling=False, label="23f-expand")

    # ── EXP-B: Extended features, expanding ──
    log("\n" + "=" * 60)
    log("[EXP-B] Extended features: ~37f, expanding, L2=1")
    log("=" * 60)
    preds_B = train_ensemble(df, FEAT_EXTENDED, WINDOWS, l2=1.0, rolling=False,
                              label="extended-expand")

    # ── EXP-C: 23f rolling 12 months ──
    log("\n" + "=" * 60)
    log("[EXP-C] Rolling 12mo: 23f, L2=1")
    log("=" * 60)
    preds_C = train_ensemble(df, FEAT_23, WINDOWS_ROLLING, l2=1.0, rolling=True,
                              label="23f-rolling12m")

    # ── EXP-D: Extended features + rolling 12mo ──
    log("\n" + "=" * 60)
    log("[EXP-D] Extended + rolling 12mo: ~37f, L2=1")
    log("=" * 60)
    preds_D = train_ensemble(df, FEAT_EXTENDED, WINDOWS_ROLLING, l2=1.0, rolling=True,
                              label="ext-rolling12m")

    # ── EXP-E: Extended + rolling + higher regularization ──
    log("\n" + "=" * 60)
    log("[EXP-E] Extended + rolling + L2=5")
    log("=" * 60)
    preds_E = train_ensemble(df, FEAT_EXTENDED, WINDOWS_ROLLING, l2=5.0, rolling=True,
                              label="ext-rolling-L2=5")

    # ── 3. Evaluate all configs ──
    configs_to_test = [
        # (name, cfg_dict)
        ("6L3S_12h", {"n_long": 6, "n_short": 3, "rebal_hours": 12,
                       "trend_cutoff": 0.9, "dyn_threshold": 0.7}),
        ("3L2S_12h", {"n_long": 3, "n_short": 2, "rebal_hours": 12,
                       "trend_cutoff": 0.9, "dyn_threshold": 0.7}),
        ("3L2S_24h", {"n_long": 3, "n_short": 2, "rebal_hours": 24,
                       "trend_cutoff": 0.9, "dyn_threshold": 0.7}),
        ("6L3S_crash_skip", {"n_long": 6, "n_short": 3, "rebal_hours": 12,
                             "trend_cutoff": 0.9, "dyn_threshold": 0.7,
                             "crash_gate": True, "crash_threshold": -0.05}),
        ("6L3S_crash_short", {"n_long": 6, "n_short": 3, "rebal_hours": 12,
                              "trend_cutoff": 0.9, "dyn_threshold": 0.7,
                              "crash_gate": True, "short_only_crash": True,
                              "crash_threshold": -0.05}),
    ]

    experiments = [
        ("EXP-A (baseline 23f)", preds_A),
        ("EXP-B (extended feats)", preds_B),
        ("EXP-C (23f rolling)", preds_C),
        ("EXP-D (ext rolling)", preds_D),
        ("EXP-E (ext roll L2=5)", preds_E),
    ]

    log("\n\n" + "=" * 80)
    log("  RESULTS MATRIX: Model × Portfolio Config (WITH transaction costs)")
    log("=" * 80)

    # Collect all results for summary table
    all_results = []

    for exp_name, preds in experiments:
        if preds is None:
            log(f"\n{exp_name}: FAILED (no predictions)")
            continue

        log(f"\n{'─' * 60}")
        log(f"  {exp_name}")
        log(f"{'─' * 60}")

        for cfg_name, cfg in configs_to_test:
            log(f"\n  [{cfg_name}]")
            results = eval_per_window(preds, regime_df, cfg, f"{exp_name}_{cfg_name}")

            all_results.append({
                "experiment": exp_name,
                "config": cfg_name,
                "W1_sh": results.get("W1", {}).get("sharpe", 0),
                "W2_sh": results.get("W2", {}).get("sharpe", 0),
                "W3_sh": results.get("W3", {}).get("sharpe", 0),
                "ALL_sh": results.get("ALL", {}).get("sharpe", 0),
                "W3_eq": results.get("W3", {}).get("equity", 0),
                "W3_worst": results.get("W3", {}).get("worst_m", 0),
                "total_cost": results.get("ALL", {}).get("total_cost_pct", 0),
            })

    # ── 4. Summary table ──
    log("\n\n" + "=" * 80)
    log("  SUMMARY TABLE (sorted by W3 Sharpe — most relevant for deployment)")
    log("=" * 80)
    log(f"\n{'Experiment':<28} {'Config':<18} {'W1':>5} {'W2':>5} {'W3':>5} "
        f"{'ALL':>5} {'W3 Eq':>7} {'W3 Wr':>6} {'Cost%':>6}")
    log("─" * 100)

    all_results.sort(key=lambda x: -x["W3_sh"])
    for r in all_results:
        log(f"{r['experiment']:<28} {r['config']:<18} "
            f"{r['W1_sh']:>5.2f} {r['W2_sh']:>5.2f} {r['W3_sh']:>5.2f} "
            f"{r['ALL_sh']:>5.2f} ${r['W3_eq']:>6.0f} {r['W3_worst']:>+5.1f}% "
            f"{r['total_cost']:>5.1f}%")

    # ── 5. IC analysis per window ──
    log("\n\n" + "=" * 80)
    log("  IC ANALYSIS (per month, best model)")
    log("=" * 80)

    best_exp_name = all_results[0]["experiment"] if all_results else None
    best_preds = None
    for exp_name, preds in experiments:
        if exp_name == best_exp_name:
            best_preds = preds
            break

    if best_preds is not None:
        monthly_ic = []
        for ts, grp in best_preds.groupby(best_preds["timestamp"].dt.to_period("M")):
            if len(grp) >= 50:
                ic = stats.spearmanr(grp["pred"], grp["fwd_ret"])[0]
                monthly_ic.append({"month": str(ts), "ic": ic, "n": len(grp)})
        if monthly_ic:
            log(f"\n{'Month':<10} {'IC':>8} {'N':>6}")
            log("─" * 28)
            for m in monthly_ic:
                marker = " ⚠️" if m["ic"] < 0 else ""
                log(f"{m['month']:<10} {m['ic']:>+8.4f} {m['n']:>6}{marker}")
            ics = [m["ic"] for m in monthly_ic]
            log(f"\nMean IC: {np.mean(ics):.4f}, IC>0: {sum(1 for x in ics if x > 0)}/{len(ics)}")

    elapsed = time.time() - t0
    log(f"\n\n✅ R30 complete in {elapsed/60:.1f} min")

    # ── 6. Recommendation ──
    if all_results:
        best = all_results[0]
        log(f"\n🏆 Best config for W3: {best['experiment']} × {best['config']}")
        log(f"   W3 Sharpe: {best['W3_sh']:.2f} (after costs)")
        if best["W3_sh"] < 1.0:
            log(f"   ⚠️  W3 Sharpe < 1.0 — NOT profitable after costs on recent data")
            log(f"   → Need more fundamental changes or stop trading")
        elif best["W3_sh"] < 2.0:
            log(f"   ⚠️  W3 Sharpe 1-2 — marginal after costs, risky to deploy")
        else:
            log(f"   ✅ W3 Sharpe > 2.0 — deployable")


if __name__ == "__main__":
    # Redirect to log
    log_path = "results_r30.log"
    original_log = log

    class Tee:
        def __init__(self, filepath):
            self.file = open(filepath, "w")
            self.stdout = sys.stdout
        def write(self, data):
            self.stdout.write(data)
            self.file.write(data)
        def flush(self):
            self.stdout.flush()
            self.file.flush()

    tee = Tee(log_path)
    sys.stdout = tee
    try:
        main()
    finally:
        sys.stdout = tee.stdout
        tee.file.close()
        print(f"\nResults saved to {log_path}")
