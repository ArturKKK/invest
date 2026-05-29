#!/usr/bin/env python3
"""
R120 — Portfolio Construction Experiments

Two experiments on top of R114b champion:
  A) Inverse-volatility position sizing (vs equal-weight baseline)
  B) Per-coin time-series momentum filter (exclude counter-trend picks)

Train models ONCE (same as R68), then run simulate() variants.
Compare Sharpe, MaxDD, Calmar, Sortino across W1/W2/W3.
"""

import sys, warnings, time
from typing import Dict, Set, Optional, List, Tuple

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


# ═══════════════════════════════════════════════════════════
# DATA & MODEL (same as R68)
# ═══════════════════════════════════════════════════════════

def load_data():
    log("=" * 70)
    log("  LOADING DATA")
    log("=" * 70)
    df, regime_df = load_research_frame()
    df, _ = add_r35_features(df)
    cg = load_cg_daily()
    cg_feats = compute_cg_features(cg)
    df, _, _ = add_cg_features(df, cg_feats)
    present = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    missing = [f for f in CHAMPION_FEAT_31 if f not in df.columns]
    if missing:
        log(f"  WARNING: Missing features: {missing}")
        CHAMPION_FEAT_31[:] = present
    log(f"  Frame: {len(df):,} rows, {df['symbol'].nunique()} symbols")
    log(f"  Features: {len(present)}/31")
    return df, regime_df


def train_ensemble(df, feats, windows, seeds=SEEDS, cs_rank_exclude=None):
    """Train LGB+XGB ensemble, return merged predictions (same as R68)."""
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
    merged["raw_prob"] = 0.5 * merged["pred_lgb"] + 0.5 * merged["pred_xgb"]
    merged["rank_lgb"] = merged.groupby("timestamp")["pred_lgb"].rank(pct=True) - 0.5
    merged["rank_xgb"] = merged.groupby("timestamp")["pred_xgb"].rank(pct=True) - 0.5
    merged["pred"] = 0.5 * merged["rank_lgb"] + 0.5 * merged["rank_xgb"]
    return merged[["timestamp", "symbol", "pred", "raw_prob", "fwd_ret", "window"]]


# ═══════════════════════════════════════════════════════════
# PRECOMPUTE: per-coin rolling volatility & trailing return
# ═══════════════════════════════════════════════════════════

def precompute_coin_stats(df):
    """
    For each (timestamp, symbol), compute:
      - rolling realized vol over various lookback windows
      - trailing cumulative return over various lookback windows
    Returns a DataFrame indexed by (timestamp, symbol).
    """
    log("  Precomputing per-coin vol & trailing returns...")
    t0 = time.time()

    vol_lookbacks = [14, 28, 56, 120]    # 12h periods → 7d, 14d, 28d, 60d
    ret_lookbacks = [14, 28, 60]          # 12h periods → 7d, 14d, 30d

    all_lookbacks = sorted(set(vol_lookbacks + ret_lookbacks))

    results = []
    for sym, grp in df.groupby("symbol"):
        grp = grp.sort_values("timestamp").copy()
        # Use fwd_ret_12h shifted back by 1 as the realized return for this period
        # Actually we need the PAST return of this coin, not fwd_ret
        # close-to-close return over 12h
        grp["ret_12h_raw"] = grp["close"].pct_change()

        rec = grp[["timestamp", "symbol"]].copy()
        for lb in all_lookbacks:
            rolling_ret = grp["ret_12h_raw"].rolling(lb, min_periods=max(lb // 2, 2))
            if lb in vol_lookbacks:
                rec[f"vol_{lb}"] = rolling_ret.std().values
            if lb in ret_lookbacks:
                # Trailing cumulative return: product of (1+r) - 1
                rec[f"trail_ret_{lb}"] = grp["close"].pct_change(lb).values

        results.append(rec)

    coin_stats = pd.concat(results, ignore_index=True)
    log(f"  Done: {len(coin_stats):,} rows, {time.time() - t0:.1f}s")
    return coin_stats


# ═══════════════════════════════════════════════════════════
# SIMULATE VARIANTS
# ═══════════════════════════════════════════════════════════

def simulate_baseline(merged, regime_df, n_long=4, n_short=2):
    """R114b baseline: equal-weight, trend cutoff, hysteresis, EMA smoothing."""
    return _simulate_core(
        merged, regime_df, n_long, n_short,
        vol_weight=False, vol_lookback=None,
        ts_filter=False, ts_lookback=None, ts_mode=None,
        coin_stats=None,
    )


def simulate_vol_scaled(merged, regime_df, coin_stats, vol_lookback,
                        n_long=4, n_short=2):
    """Experiment A: inverse-vol position sizing."""
    return _simulate_core(
        merged, regime_df, n_long, n_short,
        vol_weight=True, vol_lookback=vol_lookback,
        ts_filter=False, ts_lookback=None, ts_mode=None,
        coin_stats=coin_stats,
    )


def simulate_ts_filter(merged, regime_df, coin_stats, ts_lookback,
                       ts_mode="hard", n_long=4, n_short=2):
    """Experiment B: per-coin TS momentum filter."""
    return _simulate_core(
        merged, regime_df, n_long, n_short,
        vol_weight=False, vol_lookback=None,
        ts_filter=True, ts_lookback=ts_lookback, ts_mode=ts_mode,
        coin_stats=coin_stats,
    )


def simulate_combined(merged, regime_df, coin_stats, vol_lookback,
                      ts_lookback, ts_mode="hard", n_long=4, n_short=2):
    """Combined: vol scaling + TS filter."""
    return _simulate_core(
        merged, regime_df, n_long, n_short,
        vol_weight=True, vol_lookback=vol_lookback,
        ts_filter=True, ts_lookback=ts_lookback, ts_mode=ts_mode,
        coin_stats=coin_stats,
    )


def _simulate_core(
    merged, regime_df, n_long, n_short,
    vol_weight, vol_lookback,
    ts_filter, ts_lookback, ts_mode,
    coin_stats,
    trend_cutoff=0.9, rebal_hours=12,
    ema_alpha=0.5, hysteresis=3, dyn_threshold=0.7,
):
    """
    Core simulation with optional vol weighting and TS filter.

    vol_weight: if True, weight positions by 1/vol (using vol_{vol_lookback})
    ts_filter: if True, filter positions by trailing return alignment
    ts_mode: "hard" = exclude misaligned, "soft" = down-weight by alignment
    """
    funding_per_12h = 0.00008
    all_rets = []
    prev_longs: Set[str] = set()
    prev_shorts: Set[str] = set()
    prev_preds: Dict[str, float] = {}

    # Pre-index coin_stats for fast lookup
    cs_indexed = None
    if coin_stats is not None:
        cs_indexed = coin_stats.set_index(["timestamp", "symbol"])

    timestamps_sorted = sorted(merged["timestamp"].unique())
    grouped = {ts: grp for ts, grp in merged.groupby("timestamp")}
    rebal_timestamps = timestamps_sorted[::rebal_hours]

    for ts in rebal_timestamps:
        if ts not in regime_df.index or ts not in grouped:
            continue
        row = regime_df.loc[ts]
        trend_str = row.get("trend_strength", 0)

        # ── Trend risk-off (same as R114b) ──
        if trend_str > trend_cutoff:
            if prev_longs or prev_shorts:
                n_prev = len(prev_longs) + len(prev_shorts)
                avg_weight = 1.0 / n_prev if n_prev > 0 else 0
                close_cost = sum(_cost_for_sym(s) * avg_weight
                                 for s in prev_longs | prev_shorts)
                all_rets.append({
                    "timestamp": ts, "gross_ret": 0.0, "net_ret": -close_cost,
                    "cost": close_cost, "n_long": 0, "n_short": 0, "turnover": n_prev,
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
        if nl == 0 and ns == 0:
            continue

        # ── Dynamic exposure (same as R114b) ──
        exposure = 1.0
        if dyn_threshold is not None and trend_str > dyn_threshold:
            exposure = max(0.1, 1.0 - (trend_str - dyn_threshold) /
                           (trend_cutoff - dyn_threshold + 1e-10) * 0.5)

        # ── EMA smoothing (same as R114b) ──
        if ema_alpha is not None and ema_alpha < 1.0:
            for idx, r in grp.iterrows():
                sym = r["symbol"]
                raw_pred = r["pred"]
                smoothed = (ema_alpha * raw_pred +
                            (1 - ema_alpha) * prev_preds.get(sym, raw_pred))
                prev_preds[sym] = smoothed
                grp.at[idx, "pred"] = smoothed

        grp["pred_rank"] = grp["pred"].rank(ascending=False)

        # ── Hysteresis (same as R114b) ──
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
            for _, r in remaining.sort_values("pred_rank").head(
                    nl - len(new_longs)).iterrows():
                new_longs.add(r["symbol"])
            for _, r in remaining.sort_values("pred_rank", ascending=False).head(
                    ns - len(new_shorts)).iterrows():
                new_shorts.add(r["symbol"])
        else:
            new_longs = set(grp[grp["pred_rank"] <= nl]["symbol"].tolist()) if nl > 0 else set()
            new_shorts = set(grp[grp["pred_rank"] > (n - ns)]["symbol"].tolist()) if ns > 0 else set()

        # ─────────────────────────────────────────────────
        # EXPERIMENT B: Per-coin TS momentum filter
        # ─────────────────────────────────────────────────
        if ts_filter and cs_indexed is not None:
            col = f"trail_ret_{ts_lookback}"
            filtered_longs = set()
            filtered_shorts = set()

            for sym in new_longs:
                try:
                    tr = cs_indexed.loc[(ts, sym), col]
                except (KeyError, TypeError):
                    tr = np.nan
                if ts_mode == "hard":
                    # Keep long only if trailing return >= 0
                    if not np.isnan(tr) and tr < 0:
                        continue
                filtered_longs.add(sym)

            for sym in new_shorts:
                try:
                    tr = cs_indexed.loc[(ts, sym), col]
                except (KeyError, TypeError):
                    tr = np.nan
                if ts_mode == "hard":
                    # Keep short only if trailing return <= 0
                    if not np.isnan(tr) and tr > 0:
                        continue
                filtered_shorts.add(sym)

            # Backfill: if we lost positions, try to add next-ranked candidates
            remaining = grp[~grp["symbol"].isin(filtered_longs | filtered_shorts)]
            if len(filtered_longs) < nl:
                for _, r in remaining.sort_values("pred_rank").iterrows():
                    if len(filtered_longs) >= nl:
                        break
                    sym = r["symbol"]
                    try:
                        tr = cs_indexed.loc[(ts, sym), col]
                    except (KeyError, TypeError):
                        tr = np.nan
                    if ts_mode == "hard" and not np.isnan(tr) and tr < 0:
                        continue
                    filtered_longs.add(sym)

            if len(filtered_shorts) < ns:
                for _, r in remaining.sort_values("pred_rank", ascending=False).iterrows():
                    if len(filtered_shorts) >= ns:
                        break
                    sym = r["symbol"]
                    if sym in filtered_longs:
                        continue
                    try:
                        tr = cs_indexed.loc[(ts, sym), col]
                    except (KeyError, TypeError):
                        tr = np.nan
                    if ts_mode == "hard" and not np.isnan(tr) and tr > 0:
                        continue
                    filtered_shorts.add(sym)

            new_longs = filtered_longs
            new_shorts = filtered_shorts

        # ─────────────────────────────────────────────────
        # Compute portfolio return
        # ─────────────────────────────────────────────────
        new_opened = (new_longs - prev_longs) | (new_shorts - prev_shorts)
        closed = (prev_longs - new_longs) | (prev_shorts - new_shorts)
        total_positions = len(new_longs) + len(new_shorts)

        longs_df = grp[grp["symbol"].isin(new_longs)]
        shorts_df = grp[grp["symbol"].isin(new_shorts)]

        # ─────────────────────────────────────────────────
        # EXPERIMENT A: Inverse-vol weighting
        # ─────────────────────────────────────────────────
        if vol_weight and cs_indexed is not None:
            vol_col = f"vol_{vol_lookback}"
            # Long side
            long_ret = 0.0
            if len(longs_df) > 0:
                weights_l = []
                rets_l = []
                for _, r in longs_df.iterrows():
                    sym = r["symbol"]
                    try:
                        v = cs_indexed.loc[(ts, sym), vol_col]
                    except (KeyError, TypeError):
                        v = np.nan
                    if np.isnan(v) or v < 0.001:
                        v = 0.05  # default vol
                    weights_l.append(1.0 / v)
                    rets_l.append(r["fwd_ret"])
                w_arr = np.array(weights_l)
                w_arr = w_arr / w_arr.sum()
                # Cap at 40% per position to avoid concentration
                w_arr = np.minimum(w_arr, 0.40)
                w_arr = w_arr / w_arr.sum()
                long_ret = float(np.dot(w_arr, rets_l))

            # Short side
            short_ret = 0.0
            if len(shorts_df) > 0:
                weights_s = []
                rets_s = []
                for _, r in shorts_df.iterrows():
                    sym = r["symbol"]
                    try:
                        v = cs_indexed.loc[(ts, sym), vol_col]
                    except (KeyError, TypeError):
                        v = np.nan
                    if np.isnan(v) or v < 0.001:
                        v = 0.05
                    weights_s.append(1.0 / v)
                    rets_s.append(r["fwd_ret"])
                w_arr = np.array(weights_s)
                w_arr = w_arr / w_arr.sum()
                w_arr = np.minimum(w_arr, 0.40)
                w_arr = w_arr / w_arr.sum()
                short_ret = float(np.dot(w_arr, rets_s))

            # TS soft mode: multiply weights by alignment factor
            if ts_filter and ts_mode == "soft" and cs_indexed is not None:
                ret_col = f"trail_ret_{ts_lookback}"
                # Re-weight: positive trailing → boost long, negative → boost short
                # Not implemented in hard mode (already filtered above)
                pass  # Soft mode only adjusts via the filter above in hard mode

        else:
            # Equal-weight (baseline)
            long_ret = longs_df["fwd_ret"].mean() if len(longs_df) > 0 else 0
            short_ret = shorts_df["fwd_ret"].mean() if len(shorts_df) > 0 else 0

        nl_act, ns_act = len(new_longs), len(new_shorts)
        if nl_act > 0 and ns_act > 0:
            gross_ret = 0.5 * long_ret - 0.5 * short_ret
        elif ns_act > 0:
            gross_ret = -short_ret
        else:
            gross_ret = long_ret
        gross_ret *= exposure

        # Cost model (same as R68)
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


# ═══════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════

def compute_metrics(port, label):
    """Compute Sharpe, Sortino, Calmar, MaxDD, total return."""
    if port.empty:
        return {"label": label, "net_sharpe": 0, "gross_sharpe": 0,
                "sortino": 0, "calmar": 0, "max_dd_pct": 0, "total_ret_pct": 0,
                "win_rate": 0, "n_periods": 0, "avg_turnover": 0}

    periods_per_year = 2 * 365

    def _sharpe(rets):
        if len(rets) < 2:
            return 0.0
        eq = (1 + rets).cumprod()
        r = eq.pct_change().dropna()
        return r.mean() / (r.std() + 1e-10) * np.sqrt(periods_per_year)

    def _sortino(rets):
        if len(rets) < 2:
            return 0.0
        eq = (1 + rets).cumprod()
        r = eq.pct_change().dropna()
        neg = r[r < 0]
        downside_std = neg.std() if len(neg) > 1 else r.std()
        return r.mean() / (downside_std + 1e-10) * np.sqrt(periods_per_year)

    gs = _sharpe(port["gross_ret"])
    ns = _sharpe(port["net_ret"])
    sortino = _sortino(port["net_ret"])

    eq = (1 + port["net_ret"]).cumprod() * 100
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1
    maxdd = (eq / eq.cummax() - 1).min()
    calmar = (total_ret / abs(maxdd)) if maxdd < -0.001 else 99.0
    wr = (port["net_ret"] > 0).mean() * 100
    avg_to = port["turnover"].mean()

    return {
        "label": label,
        "gross_sharpe": round(gs, 3),
        "net_sharpe": round(ns, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 2),
        "max_dd_pct": round(maxdd * 100, 1),
        "total_ret_pct": round(total_ret * 100, 1),
        "win_rate": round(wr, 1),
        "n_periods": len(port),
        "avg_turnover": round(avg_to, 2),
    }


def print_metrics(m):
    log(f"  {m['label']}:")
    log(f"    Net Sharpe: {m['net_sharpe']:.3f}  Gross Sharpe: {m['gross_sharpe']:.3f}")
    log(f"    Sortino: {m['sortino']:.3f}  Calmar: {m['calmar']:.2f}")
    log(f"    Ret: {m['total_ret_pct']:.1f}%  DD: {m['max_dd_pct']:.1f}%  "
        f"WR: {m['win_rate']:.1f}%  Periods: {m['n_periods']}")
    log(f"    Avg turnover/period: {m['avg_turnover']:.2f}")


def bootstrap_delta_sharpe(port_new, port_base, n_boot=2000, alpha=0.05):
    """Bootstrap test: is ΔSharpe significantly positive?"""
    if port_new.empty or port_base.empty:
        return 0.0, 1.0
    r_new = port_new["net_ret"].values
    r_base = port_base["net_ret"].values
    n = min(len(r_new), len(r_base))
    r_new, r_base = r_new[:n], r_base[:n]
    diff = r_new - r_base

    rng = np.random.RandomState(42)
    deltas = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        s_new = np.mean(r_new[idx]) / (np.std(r_new[idx]) + 1e-10)
        s_base = np.mean(r_base[idx]) / (np.std(r_base[idx]) + 1e-10)
        deltas.append(s_new - s_base)
    deltas = np.array(deltas)
    p_value = (deltas <= 0).mean()
    mean_delta = np.mean(deltas)
    return mean_delta, p_value


# ═══════════════════════════════════════════════════════════
# PER-WINDOW ANALYSIS
# ═══════════════════════════════════════════════════════════

def per_window_analysis(port, merged, label):
    """Show metrics per walk-forward window."""
    if port.empty:
        return
    port_c = port.copy()
    port_c = port_c.merge(
        merged[["timestamp", "window"]].drop_duplicates(),
        on="timestamp", how="left"
    )
    log(f"\n  {label} — Per-Window:")
    log(f"    {'Window':<6} {'NetSh':>8} {'Sortino':>8} {'Ret%':>8} {'DD%':>8} {'Per':>6}")
    for w in sorted(port_c["window"].dropna().unique()):
        wdf = port_c[port_c["window"] == w]
        m = compute_metrics(wdf, w)
        log(f"    {w:<6} {m['net_sharpe']:>8.3f} {m['sortino']:>8.3f} "
            f"{m['total_ret_pct']:>7.1f}% {m['max_dd_pct']:>7.1f}% {m['n_periods']:>6}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log("=" * 70)
    log("  R120 — PORTFOLIO CONSTRUCTION EXPERIMENTS")
    log("  A) Inverse-vol position sizing")
    log("  B) Per-coin TS momentum filter")
    log("=" * 70)

    # ── 1. Load data ──
    df, regime_df = load_data()
    feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    no_rank = [f for f in feats if f in MARKET_LEVEL_FEATURES]

    # ── 2. Train ensemble (ONCE) ──
    log(f"\n  Training ensemble...")
    t1 = time.time()
    preds = train_ensemble(df, feats, CONTINUOUS_WINDOWS, seeds=SEEDS,
                           cs_rank_exclude=no_rank)
    log(f"  Done in {time.time()-t1:.0f}s, {len(preds):,} predictions")

    if preds is None or preds.empty:
        log("  ERROR: No predictions generated!")
        return

    # ── 3. Precompute coin stats ──
    coin_stats = precompute_coin_stats(df)

    # ── 4. Run experiments ──
    all_results = []

    # ── BASELINE (R114b: 4L/2S equal-weight) ──
    log("\n" + "=" * 70)
    log("  BASELINE: 4L/2S equal-weight (R114b)")
    log("=" * 70)
    port_base = simulate_baseline(preds, regime_df, 4, 2)
    m = compute_metrics(port_base, "BASELINE_4L2S")
    print_metrics(m)
    per_window_analysis(port_base, preds, "BASELINE_4L2S")
    all_results.append(m)

    # ── EXP-A: Inverse-vol sizing ──
    log("\n" + "=" * 70)
    log("  EXP-A: INVERSE-VOL POSITION SIZING")
    log("=" * 70)
    vol_lookbacks = [14, 28, 56, 120]
    best_vol_sharpe = -999
    best_vol_lb = None
    for vlb in vol_lookbacks:
        label = f"VOL_SCALE_lb{vlb}"
        port = simulate_vol_scaled(preds, regime_df, coin_stats, vlb, 4, 2)
        m = compute_metrics(port, label)
        print_metrics(m)
        all_results.append(m)
        delta, pval = bootstrap_delta_sharpe(port, port_base)
        log(f"    vs baseline: ΔSharpe={delta:.3f}, p={pval:.3f}")
        if m["net_sharpe"] > best_vol_sharpe:
            best_vol_sharpe = m["net_sharpe"]
            best_vol_lb = vlb
            best_vol_port = port

    log(f"\n  Best vol lookback: {best_vol_lb} (Sharpe={best_vol_sharpe:.3f})")
    per_window_analysis(best_vol_port, preds, f"BEST_VOL_lb{best_vol_lb}")

    # ── EXP-B: Per-coin TS momentum filter ──
    log("\n" + "=" * 70)
    log("  EXP-B: PER-COIN TS MOMENTUM FILTER")
    log("=" * 70)
    ts_lookbacks = [14, 28, 60]
    best_ts_sharpe = -999
    best_ts_lb = None
    for tlb in ts_lookbacks:
        label = f"TS_FILTER_lb{tlb}"
        port = simulate_ts_filter(preds, regime_df, coin_stats, tlb, "hard", 4, 2)
        m = compute_metrics(port, label)
        print_metrics(m)
        all_results.append(m)
        delta, pval = bootstrap_delta_sharpe(port, port_base)
        log(f"    vs baseline: ΔSharpe={delta:.3f}, p={pval:.3f}")
        if m["net_sharpe"] > best_ts_sharpe:
            best_ts_sharpe = m["net_sharpe"]
            best_ts_lb = tlb
            best_ts_port = port

    log(f"\n  Best TS lookback: {best_ts_lb} (Sharpe={best_ts_sharpe:.3f})")
    per_window_analysis(best_ts_port, preds, f"BEST_TS_lb{best_ts_lb}")

    # ── EXP-C: COMBINED (best vol + best TS) ──
    log("\n" + "=" * 70)
    log("  EXP-C: COMBINED (vol scaling + TS filter)")
    log("=" * 70)
    if best_vol_lb and best_ts_lb:
        label = f"COMBINED_vol{best_vol_lb}_ts{best_ts_lb}"
        port_comb = simulate_combined(preds, regime_df, coin_stats,
                                      best_vol_lb, best_ts_lb, "hard", 4, 2)
        m = compute_metrics(port_comb, label)
        print_metrics(m)
        per_window_analysis(port_comb, preds, label)
        all_results.append(m)
        delta, pval = bootstrap_delta_sharpe(port_comb, port_base)
        log(f"    vs baseline: ΔSharpe={delta:.3f}, p={pval:.3f}")

    # ── SUMMARY ──
    log("\n" + "=" * 70)
    log("  SUMMARY: ALL EXPERIMENTS")
    log("=" * 70)
    log(f"  {'Config':<30} {'NetSh':>8} {'GrSh':>8} {'Sort':>8} {'Cal':>7} "
        f"{'Ret%':>8} {'DD%':>8} {'WR%':>6} {'TO':>5}")
    log(f"  {'-' * 93}")
    for r in all_results:
        marker = " ***" if r["net_sharpe"] == max(x["net_sharpe"] for x in all_results) else ""
        log(f"  {r['label']:<30} {r['net_sharpe']:>8.3f} {r['gross_sharpe']:>8.3f} "
            f"{r['sortino']:>8.3f} {r['calmar']:>7.2f} "
            f"{r['total_ret_pct']:>7.1f}% {r['max_dd_pct']:>7.1f}% "
            f"{r['win_rate']:>5.1f}% {r['avg_turnover']:>5.2f}{marker}")

    # ── VALIDATION: per-window consistency ──
    log("\n" + "=" * 70)
    log("  VALIDATION: Per-window Sharpe consistency")
    log("=" * 70)
    baseline_sharpe = all_results[0]["net_sharpe"]
    for r in all_results[1:]:
        if r["net_sharpe"] > baseline_sharpe:
            log(f"  {r['label']}: +{r['net_sharpe'] - baseline_sharpe:.3f} Sharpe "
                f"(POTENTIAL IMPROVEMENT)")
        else:
            log(f"  {r['label']}: {r['net_sharpe'] - baseline_sharpe:+.3f} Sharpe (no improvement)")

    # Save results
    results_df = pd.DataFrame(all_results)
    out_path = "/data/datasets/results_r120_portfolio_experiments.csv"
    results_df.to_csv(out_path, index=False)
    log(f"\n  Saved: {out_path}")
    log(f"  Total: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)")


if __name__ == "__main__":
    main()
