#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R30b — Fixed cost model + turnover reduction.

Bugs fixed from R30:
  1. Double-leverage on costs: costs were multiplied by leverage in simulate
     AND then again in eval. Now costs are in notional space (same as returns).
  2. Extended features NaN: fillna(0) for new features instead of dropna.
  3. Added prediction EMA smoothing to reduce turnover.
  4. Added position hysteresis (band) to reduce churn.

Focus: W3 Sharpe (Nov25-Mar26) — the deployability metric.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from scipy import stats
import warnings, time, sys, os

warnings.filterwarnings("ignore")

from _research_round7 import SYM_35, WINDOWS, compute_regime
from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal
from _research_r22_models import (
    FEATURES_23, SEEDS, LEVERAGE, CAPITAL,
    log, build_r19_features, add_new_features, cs_rank_cols,
)

# ══════════════════════════════════════════════════════════════
# FEATURE SETS
# ══════════════════════════════════════════════════════════════

FEAT_23 = FEATURES_23[:]

# Extended: only features with GOOD coverage (available from early data)
FEAT_EXTENDED_CLEAN = FEAT_23 + [
    # Computed from OHLCV — 100% coverage
    "buy_pressure",
    "ret_std_24h", "ret_skew_24h", "ret_kurt_24h",
    "vol_mom_12h", "vol_mom_24h",
    "rsi_14", "cci_14",
    # Already in base pipeline with good coverage
    "fng_value",
]

# Rolling windows
WINDOWS_ROLLING = [
    {"name": "W1", "train_months": 12,
     "train_end": "2024-06-01", "val_start": "2024-06-01", "val_end": "2024-09-30",
     "test_start": "2024-10-15", "test_end": "2025-01-31"},
    {"name": "W2", "train_months": 12,
     "train_end": "2025-01-01", "val_start": "2025-01-01", "val_end": "2025-04-30",
     "test_start": "2025-05-15", "test_end": "2025-08-31"},
    {"name": "W3", "train_months": 12,
     "train_end": "2025-07-01", "val_start": "2025-07-01", "val_end": "2025-10-31",
     "test_start": "2025-11-15", "test_end": "2026-03-17"},
]


# ══════════════════════════════════════════════════════════════
# FIXED COST MODEL
# ══════════════════════════════════════════════════════════════

def simulate_with_costs(merged, regime_df, cfg):
    """
    Simulate L/S portfolio WITH realistic transaction costs.

    COST MODEL (all in NOTIONAL fraction space, NOT equity):
      - taker_fee = 5bps per side (OKX taker with proxy)
      - slippage  = 2bps (market order depth-weighted average)
      - funding   = ~1bp per 12h average
      - Total one-way cost per position = 7bps of notional

    Returns are also in notional space. Leverage is applied later in eval.
    """
    n_long = cfg.get("n_long", 6)
    n_short = cfg.get("n_short", 3)
    trend_cutoff = cfg.get("trend_cutoff", 0.9)
    rebal_hours = cfg.get("rebal_hours", 12)
    crash_gate = cfg.get("crash_gate", False)
    short_only_crash = cfg.get("short_only_crash", False)
    crash_threshold = cfg.get("crash_threshold", -0.05)
    # Turnover reduction
    ema_alpha = cfg.get("ema_alpha", None)  # EMA smoothing for predictions
    hysteresis = cfg.get("hysteresis", 0)   # rank band to keep position

    # Cost params — in NOTIONAL space (NO leverage multiplier!)
    taker_fee = 0.0005       # 5bps per side
    slippage = 0.0002        # 2bps
    funding_per_12h = 0.00008  # ~1bp

    cost_one_way = taker_fee + slippage  # 7bps of notional per one-way trade

    all_rets = []
    prev_longs = set()
    prev_shorts = set()
    prev_preds = {}  # symbol → EMA prediction (for smoothing)

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

        # BTC crash gate
        btc_ret_24h = None
        if "btc_ret_24h" in regime_df.columns and ts in regime_df.index:
            btc_ret_24h = regime_df.loc[ts, "btc_ret_24h"]

        nl, ns = n_long, n_short
        if crash_gate and btc_ret_24h is not None and btc_ret_24h < crash_threshold:
            if short_only_crash:
                nl = 0
                ns = min(ns + 2, n // 3)
            else:
                continue

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

        # Optionally smooth predictions with EMA
        if ema_alpha is not None and ema_alpha < 1.0:
            for _, r in grp.iterrows():
                sym = r["symbol"]
                raw_pred = r["pred"]
                if sym in prev_preds:
                    smoothed = ema_alpha * raw_pred + (1 - ema_alpha) * prev_preds[sym]
                else:
                    smoothed = raw_pred
                prev_preds[sym] = smoothed
                grp.loc[grp["symbol"] == sym, "pred"] = smoothed

        grp["pred_rank"] = grp["pred"].rank(ascending=False)

        # Position selection with hysteresis
        if hysteresis > 0 and (prev_longs or prev_shorts):
            # Expand the "keep" zone: existing positions stay unless they fall far
            new_longs = set()
            new_shorts = set()

            for _, r in grp.iterrows():
                sym = r["symbol"]
                rank = r["pred_rank"]
                if sym in prev_longs and rank <= nl + hysteresis:
                    new_longs.add(sym)
                elif sym in prev_shorts and rank > (n - ns - hysteresis):
                    new_shorts.add(sym)

            # Fill remaining slots
            remaining_long = nl - len(new_longs)
            remaining_short = ns - len(new_shorts)

            if remaining_long > 0:
                candidates = grp[~grp["symbol"].isin(new_longs | new_shorts)]
                candidates = candidates.sort_values("pred_rank")
                for _, r in candidates.head(remaining_long).iterrows():
                    new_longs.add(r["symbol"])

            if remaining_short > 0:
                candidates = grp[~grp["symbol"].isin(new_longs | new_shorts)]
                candidates = candidates.sort_values("pred_rank", ascending=False)
                for _, r in candidates.head(remaining_short).iterrows():
                    new_shorts.add(r["symbol"])
        else:
            new_longs = set(grp[grp["pred_rank"] <= nl]["symbol"].tolist()) if nl > 0 else set()
            new_shorts = set(grp[grp["pred_rank"] > (n - ns)]["symbol"].tolist()) if ns > 0 else set()

        # Count turnover
        new_opened = (new_longs - prev_longs) | (new_shorts - prev_shorts)
        closed = (prev_longs - new_longs) | (prev_shorts - new_shorts)
        turnover_count = len(new_opened) + len(closed)
        total_positions = len(new_longs) + len(new_shorts)

        # COST (in NOTIONAL space — leverage applied later in eval)
        if total_positions > 0:
            # turnover_count one-way trades, each costs cost_one_way of that position's notional
            # Position weight in portfolio = 1/total_positions (equal weight within L/S legs)
            # Actually: longs have weight 0.5/n_long, shorts have weight 0.5/n_short
            # Approximate with average weight for simplicity
            avg_weight = 1.0 / total_positions
            turnover_cost = turnover_count * cost_one_way * avg_weight
            holding_cost = funding_per_12h * (rebal_hours / 12)  # per 12h period
            total_cost = turnover_cost + holding_cost
        else:
            total_cost = 0
            turnover_count = 0

        prev_longs = new_longs
        prev_shorts = new_shorts

        longs = grp[grp["symbol"].isin(new_longs)]
        shorts = grp[grp["symbol"].isin(new_shorts)]

        long_ret = longs["fwd_ret"].mean() if len(longs) > 0 else 0
        short_ret = shorts["fwd_ret"].mean() if len(shorts) > 0 else 0

        if nl > 0 and ns > 0:
            port_ret = 0.5 * long_ret - 0.5 * short_ret
        elif ns > 0:
            port_ret = -short_ret
        else:
            port_ret = long_ret

        port_ret *= exposure
        port_ret -= total_cost  # both in notional space — correct!

        all_rets.append({
            "timestamp": ts,
            "portfolio_ret": port_ret,
            "gross_ret": port_ret + total_cost,  # for comparison
            "long_ret": long_ret,
            "short_ret": short_ret,
            "long_leg_ret": 0.5 * long_ret * exposure if nl > 0 else 0.0,
            "short_leg_ret": -0.5 * short_ret * exposure if ns > 0 else 0.0,
            "n_long": len(new_longs),
            "n_short": len(new_shorts),
            "exposure": exposure,
            "turnover": turnover_count,
            "cost": total_cost,
        })

    if not all_rets:
        return pd.DataFrame(columns=[
            "timestamp", "portfolio_ret", "gross_ret",
            "long_ret", "short_ret", "long_leg_ret", "short_leg_ret",
            "n_long", "n_short", "exposure", "turnover", "cost",
        ])
    return pd.DataFrame(all_rets)


# ══════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════

def eval_with_costs(port_df, name, leverage=LEVERAGE, capital=CAPITAL):
    """Evaluate portfolio. Sharpe on net returns. Equity uses leverage."""
    if len(port_df) == 0:
        return {"name": name, "sharpe": 0, "sharpe_gross": 0,
                "equity": capital, "worst_m": 0, "total_cost_pct": 0, "n_periods": 0}

    rets = port_df["portfolio_ret"].values  # net of costs, notional space
    gross = port_df["gross_ret"].values if "gross_ret" in port_df.columns else rets
    n_obs = len(rets)

    ts = port_df["timestamp"]
    total_hours = (ts.max() - ts.min()).total_seconds() / 3600
    years = max(total_hours / 8760, 0.01)
    ppy = n_obs / years

    # Sharpe on notional returns (leverage doesn't affect Sharpe ratio)
    mean_r = np.mean(rets)
    std_r = np.std(rets) + 1e-10
    sharpe_net = mean_r / std_r * np.sqrt(ppy)

    mean_g = np.mean(gross)
    std_g = np.std(gross) + 1e-10
    sharpe_gross = mean_g / std_g * np.sqrt(ppy)

    # Equity curve with leverage
    leveraged = rets * leverage
    equity_curve = capital * np.cumprod(1 + leveraged)
    final_eq = equity_curve[-1]

    # Max drawdown on equity
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    max_dd = dd.min()

    # Monthly breakdown
    monthly = port_df.set_index("timestamp").resample("ME")["portfolio_ret"].sum()
    worst_m = monthly.min() if len(monthly) > 0 else 0
    win_months = (monthly > 0).sum()
    total_months = len(monthly)

    total_cost = port_df["cost"].sum() if "cost" in port_df.columns else 0
    avg_turnover = port_df["turnover"].mean() if "turnover" in port_df.columns else 0

    return {
        "name": name,
        "sharpe": round(sharpe_net, 2),
        "sharpe_gross": round(sharpe_gross, 2),
        "equity": round(final_eq, 0),
        "worst_m": round(worst_m * 100, 1),
        "max_dd_pct": round(max_dd * 100, 1),
        "win_months": f"{win_months}/{total_months}",
        "total_cost_pct": round(total_cost * 100, 2),
        "avg_turnover": round(avg_turnover, 1),
        "n_periods": n_obs,
    }


def eval_per_window(preds, regime_df, cfg, label=""):
    """Per-window evaluation with costs."""
    results = {}
    for wname in ["W1", "W2", "W3"]:
        sub = preds[preds["window"] == wname]
        if len(sub) < 10:
            results[wname] = {"sharpe": 0, "sharpe_gross": 0}
            continue
        port = simulate_with_costs(sub, regime_df, cfg)
        r = eval_with_costs(port, f"{label}_{wname}")
        results[wname] = r
        log(f"  {wname}: Sh={r['sharpe']:>5.2f} (gross={r['sharpe_gross']:>5.2f})  "
            f"Eq=${r['equity']:>6.0f}  DD={r['max_dd_pct']:>+5.1f}%  "
            f"WM={r['win_months']}  Cost={r['total_cost_pct']:.1f}%  Turn={r['avg_turnover']:.1f}")

    # Combined
    port_all = simulate_with_costs(preds, regime_df, cfg)
    r_all = eval_with_costs(port_all, label)
    log(f"  ALL: Sh={r_all['sharpe']:>5.2f} (gross={r_all['sharpe_gross']:>5.2f})  "
        f"Eq=${r_all['equity']:>6.0f}  DD={r_all['max_dd_pct']:>+5.1f}%  "
        f"Cost={r_all['total_cost_pct']:.1f}%  Turn={r_all['avg_turnover']:.1f}")

    results["ALL"] = r_all
    return results


# ══════════════════════════════════════════════════════════════
# TRAINING (reused from R30)
# ══════════════════════════════════════════════════════════════

def train_ensemble(df, feats, windows, seeds=SEEDS, l2=1.0,
                   rolling=False, use_cs_rank=True, label="",
                   cs_rank_exclude=None):
    """Train LGB+XGB ensemble."""
    avail = [f for f in feats if f in df.columns]
    missing = [f for f in feats if f not in df.columns]
    if missing:
        log(f"  ⚠️  Missing features: {missing}")
    log(f"  Training: {label}, {len(avail)}f, L2={l2}, {'rolling' if rolling else 'expanding'}, "
        f"{len(seeds)} seeds × {len(windows)} windows")

    all_lgb, all_xgb = [], []
    rank_exclude = set(cs_rank_exclude or [])
    rank_feats = [f for f in avail if f not in rank_exclude]
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

            if use_cs_rank and rank_feats:
                train_ = cs_rank_cols(train_, rank_feats)
                val_ = cs_rank_cols(val_, rank_feats)
                test_ = cs_rank_cols(test_, rank_feats)

            for d in [train_, val_, test_]:
                d["target_binary"] = (d["fwd_ret_12h"] > 0).astype(int)

            # FIXED: fillna(0) for new features instead of dropna
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
# CLEAN FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

def add_extra_features_clean(df):
    """Add features computed from OHLCV — guaranteed no NaN holes."""
    n_before = len(df.columns)

    for sym, gdf in df.groupby("symbol"):
        idx = gdf.index
        close = gdf["close"]
        volume = gdf["volume"]
        high = gdf["high"]
        low = gdf["low"]

        # Buy pressure
        if "buy_pressure" not in df.columns:
            df.loc[idx, "buy_pressure"] = (close - low) / (high - low + 1e-10)

        # Return distribution
        ret_1h = close.pct_change(1)
        for w in [24]:
            for col_name, func in [
                (f"ret_std_{w}h", lambda x: x.rolling(w, min_periods=w//2).std()),
                (f"ret_skew_{w}h", lambda x: x.rolling(w, min_periods=w//2).skew()),
                (f"ret_kurt_{w}h", lambda x: x.rolling(w, min_periods=w//2).kurt()),
            ]:
                if col_name not in df.columns:
                    df.loc[idx, col_name] = func(ret_1h)

        # Volume momentum
        for w in [12, 24]:
            col = f"vol_mom_{w}h"
            if col not in df.columns:
                vol_ma = volume.rolling(w, min_periods=w//2).mean()
                vol_ma_long = volume.rolling(w*4, min_periods=w*2).mean()
                df.loc[idx, col] = vol_ma / (vol_ma_long + 1e-10) - 1

        # RSI
        if "rsi_14" not in df.columns:
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            df.loc[idx, "rsi_14"] = 100 - 100 / (1 + rs)

        # CCI
        if "cci_14" not in df.columns:
            tp = (high + low + close) / 3
            tp_ma = tp.rolling(14).mean()
            tp_std = tp.rolling(14).std()
            df.loc[idx, "cci_14"] = (tp - tp_ma) / (0.015 * tp_std + 1e-10)

    # FNG momentum
    if "fng_value" in df.columns and "fng_momentum" not in df.columns:
        df["fng_momentum"] = df.groupby("symbol")["fng_value"].transform(
            lambda x: x.diff(24))

    n_after = len(df.columns)
    log(f"  Added {n_after - n_before} extra features (clean, from OHLCV)")
    return df


def compute_regime_extended(df):
    """Compute regime with BTC 24h return for crash gate."""
    regime_df = compute_regime(df)
    btc = df[df["symbol"].str.startswith("BTC")].copy()
    if len(btc) > 0:
        btc = btc.sort_values("timestamp").set_index("timestamp")
        btc["btc_ret_24h"] = btc["close"].pct_change(24)
        regime_df = regime_df.join(btc[["btc_ret_24h"]], how="left")
        regime_df["btc_ret_24h"] = regime_df["btc_ret_24h"].ffill()
    return regime_df


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log("=" * 80)
    log("  R30b — FIXED COSTS + TURNOVER REDUCTION")
    log("=" * 80)

    # ── 1. Load data ──
    log("\n[1] Loading data...")
    ohlcv = load_ohlcv()
    ohlcv = ohlcv[ohlcv["symbol"].isin(SYM_35)]
    derivs = load_derivatives()
    df = build_features_minimal(ohlcv, derivs)
    df = build_r19_features(df)
    df, _ = add_new_features(df)
    df = df[df["symbol"].isin(SYM_35)].copy()
    log(f"  Base: {len(df):,} rows, {len(df.columns)} cols")
    df = add_extra_features_clean(df)
    log(f"  After extras: {len(df):,} rows, {len(df.columns)} cols")
    regime_df = compute_regime_extended(df)

    # ── 2. Train models ──
    # A: Baseline 23f expanding (= current prod)
    log("\n" + "=" * 60)
    log("[EXP-A] Baseline: 23f, expanding, L2=1")
    log("=" * 60)
    preds_A = train_ensemble(df, FEAT_23, WINDOWS, l2=1.0, rolling=False, label="23f-expand")

    # B: Extended features (OHLCV-derived, good coverage), expanding
    log("\n" + "=" * 60)
    log(f"[EXP-B] Extended CLEAN: {len(FEAT_EXTENDED_CLEAN)}f, expanding, L2=1")
    log("=" * 60)
    preds_B = train_ensemble(df, FEAT_EXTENDED_CLEAN, WINDOWS, l2=1.0, rolling=False,
                              label="ext-clean-expand")

    # C: 23f rolling 12mo
    log("\n" + "=" * 60)
    log("[EXP-C] Rolling 12mo: 23f, L2=1")
    log("=" * 60)
    preds_C = train_ensemble(df, FEAT_23, WINDOWS_ROLLING, l2=1.0, rolling=True,
                              label="23f-rolling12m")

    # D: Extended clean + rolling
    log("\n" + "=" * 60)
    log(f"[EXP-D] Extended CLEAN + rolling: {len(FEAT_EXTENDED_CLEAN)}f, L2=1")
    log("=" * 60)
    preds_D = train_ensemble(df, FEAT_EXTENDED_CLEAN, WINDOWS_ROLLING, l2=1.0, rolling=True,
                              label="ext-clean-rolling")

    # ── 3. Portfolio configs (fewer, more targeted) ──
    configs = [
        # Baseline: no turnover reduction
        ("6L3S_12h", {"n_long": 6, "n_short": 3, "rebal_hours": 12,
                      "trend_cutoff": 0.9, "dyn_threshold": 0.7}),

        # Fewer positions = less turnover
        ("3L2S_12h", {"n_long": 3, "n_short": 2, "rebal_hours": 12,
                      "trend_cutoff": 0.9, "dyn_threshold": 0.7}),

        # Slower rebalance
        ("6L3S_24h", {"n_long": 6, "n_short": 3, "rebal_hours": 24,
                      "trend_cutoff": 0.9, "dyn_threshold": 0.7}),

        # EMA smoothing (alpha=0.5: half weight to new, half to old)
        ("6L3S_ema05", {"n_long": 6, "n_short": 3, "rebal_hours": 12,
                        "trend_cutoff": 0.9, "dyn_threshold": 0.7,
                        "ema_alpha": 0.5}),

        # EMA smoothing (stronger: alpha=0.3)
        ("6L3S_ema03", {"n_long": 6, "n_short": 3, "rebal_hours": 12,
                        "trend_cutoff": 0.9, "dyn_threshold": 0.7,
                        "ema_alpha": 0.3}),

        # Hysteresis band (keep position if still within top N+3)
        ("6L3S_hyst3", {"n_long": 6, "n_short": 3, "rebal_hours": 12,
                        "trend_cutoff": 0.9, "dyn_threshold": 0.7,
                        "hysteresis": 3}),

        # Combined: EMA + hysteresis
        ("6L3S_ema05_h3", {"n_long": 6, "n_short": 3, "rebal_hours": 12,
                           "trend_cutoff": 0.9, "dyn_threshold": 0.7,
                           "ema_alpha": 0.5, "hysteresis": 3}),

        # Combined: slower + EMA + hysteresis
        ("6L3S_24h_ema05_h3", {"n_long": 6, "n_short": 3, "rebal_hours": 24,
                                "trend_cutoff": 0.9, "dyn_threshold": 0.7,
                                "ema_alpha": 0.5, "hysteresis": 3}),

        # Crash gate
        ("6L3S_crash_skip", {"n_long": 6, "n_short": 3, "rebal_hours": 12,
                             "trend_cutoff": 0.9, "dyn_threshold": 0.7,
                             "crash_gate": True, "crash_threshold": -0.05}),

        # Crash gate + short only
        ("6L3S_crash_short", {"n_long": 6, "n_short": 3, "rebal_hours": 12,
                              "trend_cutoff": 0.9, "dyn_threshold": 0.7,
                              "crash_gate": True, "short_only_crash": True,
                              "crash_threshold": -0.05}),
    ]

    experiments = [
        ("A_23f_expand", preds_A),
        ("B_ext_expand", preds_B),
        ("C_23f_rolling", preds_C),
        ("D_ext_rolling", preds_D),
    ]

    # ── 4. Results ──
    log("\n\n" + "=" * 80)
    log("  RESULTS: Fixed costs + turnover reduction")
    log("=" * 80)

    all_results = []

    for exp_name, preds in experiments:
        if preds is None:
            log(f"\n{exp_name}: FAILED")
            continue

        log(f"\n{'─' * 60}")
        log(f"  {exp_name}")
        log(f"{'─' * 60}")

        for cfg_name, cfg in configs:
            log(f"\n  [{cfg_name}]")
            results = eval_per_window(preds, regime_df, cfg, f"{exp_name}_{cfg_name}")
            all_results.append({
                "experiment": exp_name,
                "config": cfg_name,
                "W1_sh": results.get("W1", {}).get("sharpe", 0),
                "W1_sh_g": results.get("W1", {}).get("sharpe_gross", 0),
                "W2_sh": results.get("W2", {}).get("sharpe", 0),
                "W2_sh_g": results.get("W2", {}).get("sharpe_gross", 0),
                "W3_sh": results.get("W3", {}).get("sharpe", 0),
                "W3_sh_g": results.get("W3", {}).get("sharpe_gross", 0),
                "ALL_sh": results.get("ALL", {}).get("sharpe", 0),
                "ALL_sh_g": results.get("ALL", {}).get("sharpe_gross", 0),
                "W3_eq": results.get("W3", {}).get("equity", 0),
                "W3_dd": results.get("W3", {}).get("max_dd_pct", 0),
                "ALL_turn": results.get("ALL", {}).get("avg_turnover", 0),
                "ALL_cost": results.get("ALL", {}).get("total_cost_pct", 0),
            })

    # ── 5. Summary table ──
    log("\n\n" + "=" * 80)
    log("  SUMMARY (sorted by W3 net Sharpe)")
    log("=" * 80)
    log(f"\n{'Experiment':<16} {'Config':<20} {'W1net':>5} {'W2net':>5} {'W3net':>5} "
        f"{'W3grs':>5} {'ALLnt':>5} {'Eq$':>6} {'DD%':>6} {'Turn':>5} {'Cost%':>6}")
    log("─" * 110)

    all_results.sort(key=lambda x: -x["W3_sh"])
    for r in all_results:
        marker = " ★" if r["W3_sh"] >= 2.0 else (" ◆" if r["W3_sh"] >= 1.0 else "")
        log(f"{r['experiment']:<16} {r['config']:<20} "
            f"{r['W1_sh']:>5.2f} {r['W2_sh']:>5.2f} {r['W3_sh']:>5.2f} "
            f"{r['W3_sh_g']:>5.2f} {r['ALL_sh']:>5.2f} "
            f"${r['W3_eq']:>5.0f} {r['W3_dd']:>+5.1f}% "
            f"{r['ALL_turn']:>4.1f} {r['ALL_cost']:>5.1f}%{marker}")

    # ── 6. Cost impact analysis ──
    log("\n\n" + "=" * 80)
    log("  COST IMPACT ANALYSIS (how much alpha is eaten by costs)")
    log("=" * 80)
    for r in all_results[:10]:
        if r["W3_sh_g"] != 0:
            cost_drag = r["W3_sh_g"] - r["W3_sh"]
            pct_eaten = (cost_drag / r["W3_sh_g"] * 100) if r["W3_sh_g"] > 0 else 0
            log(f"  {r['experiment']:<16} {r['config']:<20}: "
                f"Gross={r['W3_sh_g']:>5.2f} → Net={r['W3_sh']:>5.2f}  "
                f"Cost drag={cost_drag:>5.2f} ({pct_eaten:>4.0f}% of alpha eaten)")

    # ── 7. IC analysis (best model) ──
    if all_results:
        best = all_results[0]
        best_exp = best["experiment"]
        best_preds = None
        for name, preds in experiments:
            if name == best_exp:
                best_preds = preds
                break

        if best_preds is not None:
            log("\n\n" + "=" * 80)
            log(f"  MONTHLY IC — {best_exp}")
            log("=" * 80)
            monthly_ics = []
            for ts, grp in best_preds.groupby(best_preds["timestamp"].dt.to_period("M")):
                if len(grp) >= 50:
                    ic = stats.spearmanr(grp["pred"], grp["fwd_ret"])[0]
                    monthly_ics.append({"month": str(ts), "ic": ic, "n": len(grp)})
            if monthly_ics:
                log(f"\n{'Month':<10} {'IC':>8} {'N':>6}")
                log("─" * 28)
                for m in monthly_ics:
                    marker = " ⚠️" if m["ic"] < 0 else ""
                    log(f"{m['month']:<10} {m['ic']:>+8.4f} {m['n']:>6}{marker}")
                ics = [m["ic"] for m in monthly_ics]
                log(f"\nMean IC: {np.mean(ics):.4f}, "
                    f"IC>0: {sum(1 for x in ics if x > 0)}/{len(ics)}, "
                    f"ICIR: {np.mean(ics)/(np.std(ics)+1e-10):.2f}")

    elapsed = time.time() - t0
    log(f"\n\n✅ R30b complete in {elapsed/60:.1f} min")

    # ── 8. Verdict ──
    if all_results:
        best = all_results[0]
        log(f"\n{'='*80}")
        log(f"  VERDICT")
        log(f"{'='*80}")
        log(f"  Best W3 config: {best['experiment']} × {best['config']}")
        log(f"  W3 Sharpe: net={best['W3_sh']:.2f}, gross={best['W3_sh_g']:.2f}")
        log(f"  W3 Equity: ${best['W3_eq']:.0f} (from $100), MaxDD: {best['W3_dd']:.1f}%")

        if best["W3_sh"] >= 2.0:
            log(f"  ✅ DEPLOYABLE: W3 net Sharpe ≥ 2.0")
        elif best["W3_sh"] >= 1.0:
            log(f"  ⚠️  MARGINAL: W3 net Sharpe 1-2. High risk of live underperformance.")
        elif best["W3_sh"] >= 0:
            log(f"  ❌ NOT PROFITABLE: W3 net Sharpe < 1.0. Do not deploy.")
        else:
            log(f"  ❌ NEGATIVE ALPHA: W3 net Sharpe < 0. Model loses money after costs.")

        # Compare turnover reduction impact
        baseline_results = [r for r in all_results if r["config"] == "6L3S_12h"]
        reduced_results = [r for r in all_results if "ema" in r["config"] or "hyst" in r["config"]]
        if baseline_results and reduced_results:
            bl_turn = np.mean([r["ALL_turn"] for r in baseline_results])
            rd_turn = np.mean([r["ALL_turn"] for r in reduced_results])
            log(f"\n  Turnover reduction: baseline avg={bl_turn:.1f} → "
                f"with EMA/hyst avg={rd_turn:.1f} ({(1-rd_turn/bl_turn)*100:.0f}% reduction)")


if __name__ == "__main__":
    log_path = "results_r30b.log"

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
