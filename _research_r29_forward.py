#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R29 — Exhaustive Forward Feature Selection

Tests ALL per-coin features from both research and production pipelines
that were NOT tested in R28c. Builds production-level features within
the research walk-forward framework.

Groups:
  BATCH_A: 16 features from research pipeline (untested in R28c)
  BATCH_B: ~30 best production-only features (price shape, liquidity,
           basis, extended derivatives, 12h-specific, etc.)

Each candidate is tested as FEATURES_23 + [candidate] → ensemble Sharpe.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from pathlib import Path
import warnings, time, sys
warnings.filterwarnings("ignore")

try:
    import ta
except ImportError:
    print("pip install ta")
    sys.exit(1)

from _research_round7 import (
    SYM_35, WINDOWS, compute_regime, simulate, eval_config, show,
)
from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal
from _research_r22_models import (
    FEATURES_23, SEEDS, LEVERAGE, CAPITAL, DATA_DIR, SENT_DIR,
    log, build_r19_features, add_new_features, cs_rank_cols,
)

CFG_BEST = {
    "n_long": 6, "n_short": 3, "trend_cutoff": 0.9,
    "dyn_threshold": 0.5625, "rebal_hours": 12,
    "kelly_sizing": False, "vol_scaling": False, "regime_asym": False,
}
CFG_6L3S = {**CFG_BEST, "n_long": 6, "n_short": 3, "dyn_threshold": 0.7}


# ═══════════════════════════════════════════════════════════════════════════════
#  PRODUCTION FEATURE BUILDER (adapted from run_trading.py build_features)
# ═══════════════════════════════════════════════════════════════════════════════

def build_production_features(df):
    """
    Build production-level features that are NOT in the research pipeline.
    Input df should already have build_features_minimal + build_r19_features
    + add_new_features applied.
    """
    log("  [R29] Building production-level features...")
    n_before = len(df.columns)

    result_dfs = []
    syms = sorted(df["symbol"].unique())
    for si, (sym, gdf) in enumerate(df.groupby("symbol")):
        if si % 10 == 0:
            log(f"    sym {si+1}/{len(syms)}: {sym}")
        g = gdf.sort_values("timestamp").copy()
        c = g["close"]
        h = g["high"]
        l = g["low"]
        o = g["open"]
        v = g["volume"]

        # --- Price shape features ---
        g["close_open_ratio"] = c / (o + 1e-10) - 1
        g["high_low_ratio"] = h / (l + 1e-10) - 1
        g["upper_shadow"] = (h - np.maximum(c, o)) / (h - l + 1e-10)
        g["lower_shadow"] = (np.minimum(c, o) - l) / (h - l + 1e-10)
        g["body"] = np.abs(c - o) / (h - l + 1e-10)

        # --- Close-to-MA ratios ---
        for w in [6, 12, 24, 48, 168]:
            ma = c.rolling(w, min_periods=max(w // 2, 1)).mean()
            g[f"close_ma{w}_ratio"] = c / (ma + 1e-10) - 1

        # --- Volume-to-MA ratios ---
        for w in [6, 12, 24, 48]:
            g[f"vol_ma{w}_ratio"] = v / (v.rolling(w, min_periods=max(w // 2, 1)).mean() + 1e-10) - 1

        # --- Return distribution ---
        r1h = c.pct_change()
        for w in [24, 48, 168]:
            g[f"ret_std_{w}h"] = r1h.rolling(w).std()
            g[f"ret_sharpe_{w}h"] = r1h.rolling(w).mean() / (r1h.rolling(w).std() + 1e-10)

        # --- Volume momentum ---
        for w in [12, 24]:
            g[f"vol_mom_{w}h"] = v / (v.shift(w) + 1e-10) - 1

        # --- Volume-price correlation ---
        for w in [24, 168]:
            g[f"vol_price_corr_{w}h"] = c.pct_change().rolling(w).corr(v.pct_change())

        # --- Buy pressure ---
        g["buy_pressure"] = (c - l) / (h - l + 1e-10)

        # --- TA extras not in research ---
        macd_ind = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
        g["macd_diff"] = macd_ind.macd_diff()

        stoch = ta.momentum.StochasticOscillator(h, l, c, window=14, smooth_window=3)
        g["stoch_k"] = stoch.stoch()

        g["cci_14"] = ta.trend.CCIIndicator(h, l, c, window=14).cci()
        g["willr_14"] = ta.momentum.WilliamsRIndicator(h, l, c, lbp=14).williams_r()

        adx_ind = ta.trend.ADXIndicator(h, l, c, window=14)
        g["adx_pos"] = adx_ind.adx_pos()
        g["adx_neg"] = adx_ind.adx_neg()

        # --- Extended GK vol ---
        log_hl = np.log(h / (l + 1e-10) + 1e-10) ** 2
        log_co = np.log(c / (o + 1e-10) + 1e-10) ** 2
        g["gk_vol_48h"] = np.sqrt((0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(48).mean().abs())
        g["gk_vol_168h"] = np.sqrt((0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(168).mean().abs())

        # --- Liquidity ---
        dv24 = (c * v).rolling(24, min_periods=12).sum()
        g["dollar_volume_24h"] = dv24
        g["amihud_illiq_24h"] = c.pct_change().abs() / (dv24 + 1e-10) * 1e9
        g["range_per_dv_24h"] = (h.rolling(24).max() - l.rolling(24).min()) / (dv24 + 1e-10) * 1e9

        # --- 12h holding features ---
        g["mom_accel_12h"] = g.get("ret_12h", c.pct_change(12)) - c.pct_change(12).shift(12)
        g["range_expansion_12h"] = (h.rolling(12).max() - l.rolling(12).min()) / (c + 1e-10)
        g["range_position_12h"] = (c - l.rolling(12).min()) / (h.rolling(12).max() - l.rolling(12).min() + 1e-10)
        vwap_12 = (c * v).rolling(12).sum() / (v.rolling(12).sum() + 1e-10)
        g["vwap_12h_dist"] = c / (vwap_12 + 1e-10) - 1
        # Vectorized direction quality: fraction of last 12 bars with same sign as current
        sign_r = np.sign(r1h)
        same_sign = (sign_r.rolling(12).apply(lambda x: np.mean(x == x[-1]), raw=True))
        g["direction_quality_12h"] = same_sign

        # --- Basis/Premium features ---
        if "premium_index" in g.columns:
            basis = g["premium_index"]
            g["basis_pct"] = basis
            g["basis_change_12h"] = basis - basis.shift(12)
            g["basis_change_24h"] = basis - basis.shift(24)
            g["basis_zscore_7d"] = (basis - basis.rolling(168, min_periods=84).mean()) / (basis.rolling(168, min_periods=84).std() + 1e-10)

        # --- Reversal features ---
        for fast, slow in [(4, 24), (24, 168)]:
            fr = f"ret_{fast}h" if f"ret_{fast}h" in g.columns else None
            sr = f"ret_{slow}h" if f"ret_{slow}h" in g.columns else None
            if fr and sr:
                g[f"reversal_{fast}v{slow}"] = -g[fr] * g[sr].abs()

        # --- Volume surge ---
        for w in [12, 24]:
            g[f"vol_surge_{w}h"] = v / (v.rolling(w).mean() + 1e-10) - 1

        result_dfs.append(g)

    df = pd.concat(result_dfs, ignore_index=True)

    # --- Cross-sectional liquidity ranks ---
    for col in ["dollar_volume_24h", "amihud_illiq_24h"]:
        if col in df.columns:
            df[f"{col}_cs"] = df.groupby("timestamp")[col].rank(pct=True)

    # --- BTC beta 48h (research only has 168h) ---
    btc_rets = df[df["symbol"] == "BTC/USDT"][["timestamp", "ret_1h"]].rename(
        columns={"ret_1h": "_btc_r"}).drop_duplicates("timestamp")
    df = df.merge(btc_rets, on="timestamp", how="left")
    df["btc_beta_48h"] = df.groupby("symbol").apply(
        lambda g: g["ret_1h"].rolling(48, min_periods=24).corr(g["_btc_r"]) *
                  (g["ret_1h"].rolling(48, min_periods=24).std() /
                   (g["_btc_r"].rolling(48, min_periods=24).std() + 1e-10))
    ).reset_index(level=0, drop=True)
    df.drop(columns=["_btc_r"], inplace=True, errors="ignore")

    # --- Extended OI derivatives ---
    if "oi_chg_1h" in df.columns and "ret_1h" in df.columns:
        df["oi_ret_interaction"] = df["oi_chg_1h"] * df["ret_1h"]
    if "oi_chg_12h" in df.columns and "ret_12h" in df.columns:
        df["oi_ret_interaction_12h"] = df["oi_chg_12h"] * df["ret_12h"]

    # --- Per-coin news features (if available) ---
    news_path = DATA_DIR / "sentiment" / "news_hourly.parquet"
    if news_path.exists():
        try:
            news = pd.read_parquet(news_path)
            news["timestamp"] = pd.to_datetime(news["timestamp"], utc=True)
            # Only per-coin news
            coin_news = news.groupby(["timestamp", "symbol"]).agg(
                news_count_1h=("title", "count"),
                news_sentiment_1h=("sentiment", "mean"),
            ).reset_index()
            df = df.merge(coin_news, on=["timestamp", "symbol"], how="left")
            df["news_count_1h"] = df["news_count_1h"].fillna(0)
            df["news_sentiment_1h"] = df["news_sentiment_1h"].fillna(0)
            for w in [24, 168]:
                df[f"news_count_{w}h"] = df.groupby("symbol")["news_count_1h"].transform(
                    lambda x: x.rolling(w, min_periods=1).sum())
                df[f"news_sentiment_{w}h"] = df.groupby("symbol")["news_sentiment_1h"].transform(
                    lambda x: x.rolling(w, min_periods=1).mean())
            df["news_volume_zscore"] = df.groupby("symbol")["news_count_24h"].transform(
                lambda x: (x - x.rolling(168, min_periods=24).mean()) / (x.rolling(168, min_periods=24).std() + 1e-10))
            log(f"  [R29] News features added: news_count/sentiment 1h/24h/168h + zscore")
        except Exception as e:
            log(f"  [R29] News load failed: {e}")

    # --- Funding per-coin features (if in data) ---
    if "funding_rate_binance" in df.columns:
        # Funding surprise: deviation from rolling mean (per-coin specific)
        df["funding_surprise"] = df.groupby("symbol")["funding_rate_binance"].transform(
            lambda x: x - x.rolling(168, min_periods=24).mean())

    # Clean inf
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    n_after = len(df.columns)
    log(f"  [R29] Production features added: {n_after - n_before} new columns ({n_after} total)")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  ALL CANDIDATES — exhaustive list of untested per-coin features
# ═══════════════════════════════════════════════════════════════════════════════

# Batch A: 16 from research pipeline (never tested in R28c)
BATCH_A = [
    "btc_beta_168h",     # per-coin rolling beta vs BTC
    "btc_outperform",    # ret_24h - btc_ret_24h
    "funding_x_mom_12h", # funding × momentum interaction
    "funding_x_mom_24h", # funding × momentum interaction
    "global_ls_ratio",   # raw ratio (not z-scored)
    "mom_z_12h",         # momentum z 12h (we use 24h)
    "oi_chg_1h",         # OI change 1h (we use 12h/24h)
    "oi_chg_4h",         # OI change 4h
    "range_24h",         # high/low range
    "reversal_12v48",    # short vs long momentum
    "taker_buy_sell_ratio",  # raw ratio
    "taker_cvd_4h",      # CVD 4h (we use 12h/24h)
    "taker_imbalance",   # raw imbalance
    "top_ls_ratio",      # raw ratio
    "vol_crush",         # rvol_12h / rvol_168h
    "vol_ratio_12h",     # volume ratio 12h
]

# Batch B: production-only per-coin features (best candidates)
BATCH_B = [
    # Price shape
    "close_open_ratio",
    "upper_shadow",
    "lower_shadow",
    "body",
    "buy_pressure",
    # Close-to-MA (mean reversion signals)
    "close_ma6_ratio",
    "close_ma12_ratio",
    "close_ma24_ratio",
    "close_ma48_ratio",
    "close_ma168_ratio",
    # Volume MA
    "vol_ma6_ratio",
    "vol_ma12_ratio",
    "vol_ma24_ratio",
    # Return distribution
    "ret_std_24h",
    "ret_sharpe_24h",
    "ret_std_168h",
    "ret_sharpe_168h",
    # Volume dynamics
    "vol_mom_12h",
    "vol_mom_24h",
    "vol_price_corr_24h",
    # TA extras
    "macd_diff",
    "stoch_k",
    "cci_14",
    "willr_14",
    "adx_pos",
    "adx_neg",
    # Extended vol
    "gk_vol_48h",
    "gk_vol_168h",
    # Liquidity
    "dollar_volume_24h_cs",   # CS-ranked dollar volume
    "amihud_illiq_24h_cs",    # CS-ranked illiquidity
    "range_per_dv_24h",
    # 12h holding features
    "mom_accel_12h",
    "range_expansion_12h",
    "range_position_12h",
    "vwap_12h_dist",
    "direction_quality_12h",
    # Basis/premium
    "basis_pct",
    "basis_change_12h",
    "basis_change_24h",
    "basis_zscore_7d",
    # Reversal & volume surge
    "reversal_4v24",
    "reversal_24v168",
    "vol_surge_12h",
    "vol_surge_24h",
    # Beta
    "btc_beta_48h",
    # Extended derivatives
    "oi_ret_interaction",
    "oi_ret_interaction_12h",
    # Funding
    "funding_surprise",
    # News per-coin
    "news_count_24h",
    "news_sentiment_24h",
    "news_volume_zscore",
]


ALL_CANDIDATES = BATCH_A + BATCH_B


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAIN + EVAL (copied from R28c — minimal, fast)
# ═══════════════════════════════════════════════════════════════════════════════

def train_lgb_cls(df, feats, seeds=SEEDS):
    avail = [f for f in feats if f in df.columns]
    all_preds = []
    tz = df["timestamp"].dt.tz
    for seed in seeds:
        params = {
            "objective": "binary", "metric": "auc",
            "learning_rate": 0.03, "num_leaves": 63,
            "min_child_samples": 100, "subsample": 0.8,
            "colsample_bytree": 0.8, "lambda_l2": 1.0,
            "verbose": -1, "n_jobs": -1, "seed": seed,
        }
        seed_preds = []
        for w in WINDOWS:
            train = df[df["timestamp"] < pd.Timestamp(w["train_end"], tz=tz)].copy()
            val = df[(df["timestamp"] >= pd.Timestamp(w["val_start"], tz=tz)) &
                     (df["timestamp"] < pd.Timestamp(w["val_end"], tz=tz))].copy()
            test = df[(df["timestamp"] >= pd.Timestamp(w["test_start"], tz=tz)) &
                      (df["timestamp"] <= pd.Timestamp(w["test_end"], tz=tz))].copy()
            if len(train) < 5000 or len(test) < 200:
                continue
            train = cs_rank_cols(train, avail)
            val = cs_rank_cols(val, avail)
            test = cs_rank_cols(test, avail)
            for d in [train, val, test]:
                d["target_binary"] = (d["fwd_ret_12h"] > 0).astype(int)
            train_c = train[avail + ["target_binary"]].dropna()
            val_c = val[avail + ["target_binary"]].dropna()
            dtrain = lgb.Dataset(train_c[avail], label=train_c["target_binary"])
            dval = lgb.Dataset(val_c[avail], label=val_c["target_binary"])
            model = lgb.train(params, dtrain, num_boost_round=600,
                              valid_sets=[dval],
                              callbacks=[lgb.early_stopping(40, verbose=False),
                                         lgb.log_evaluation(-1)])
            test_c = test[avail + ["target_binary", "timestamp", "symbol"]].dropna()
            if len(test_c) == 0:
                continue
            preds = model.predict(test_c[avail])
            fwd = test[["timestamp", "symbol", "fwd_ret_12h"]].rename(
                columns={"fwd_ret_12h": "fwd_ret"}).dropna()
            merged = test_c[["timestamp", "symbol"]].copy()
            merged["pred"] = preds
            merged = merged.merge(fwd, on=["timestamp", "symbol"], how="inner")
            merged["window"] = w["name"]
            seed_preds.append(merged)
        if seed_preds:
            all_preds.append(pd.concat(seed_preds, ignore_index=True))
    if not all_preds:
        return None
    combined = pd.concat(all_preds, ignore_index=True)
    return (combined.groupby(["timestamp", "symbol"])
            .agg(pred=("pred", "mean"), fwd_ret=("fwd_ret", "first"),
                 window=("window", "first"))
            .reset_index())


def train_xgb_cls(df, feats, seeds=SEEDS):
    avail = [f for f in feats if f in df.columns]
    all_preds = []
    tz = df["timestamp"].dt.tz
    for seed in seeds:
        seed_preds = []
        for w in WINDOWS:
            train = df[df["timestamp"] < pd.Timestamp(w["train_end"], tz=tz)].copy()
            val = df[(df["timestamp"] >= pd.Timestamp(w["val_start"], tz=tz)) &
                     (df["timestamp"] < pd.Timestamp(w["val_end"], tz=tz))].copy()
            test = df[(df["timestamp"] >= pd.Timestamp(w["test_start"], tz=tz)) &
                      (df["timestamp"] <= pd.Timestamp(w["test_end"], tz=tz))].copy()
            if len(train) < 5000 or len(test) < 200:
                continue
            train = cs_rank_cols(train, avail)
            val = cs_rank_cols(val, avail)
            test = cs_rank_cols(test, avail)
            for d in [train, val, test]:
                d["target_binary"] = (d["fwd_ret_12h"] > 0).astype(int)
            train_c = train[avail + ["target_binary"]].dropna()
            val_c = val[avail + ["target_binary"]].dropna()
            dtrain = xgb.DMatrix(train_c[avail], label=train_c["target_binary"])
            dval = xgb.DMatrix(val_c[avail], label=val_c["target_binary"])
            model = xgb.train(
                {"objective": "binary:logistic", "eval_metric": "auc",
                 "learning_rate": 0.03, "max_depth": 6,
                 "min_child_weight": 100, "subsample": 0.8,
                 "colsample_bytree": 0.8, "reg_lambda": 1.0,
                 "seed": seed, "n_jobs": -1, "verbosity": 0},
                dtrain, num_boost_round=600,
                evals=[(dval, "val")],
                early_stopping_rounds=40, verbose_eval=False)
            test_c = test[avail + ["target_binary", "timestamp", "symbol"]].dropna()
            if len(test_c) == 0:
                continue
            dtest = xgb.DMatrix(test_c[avail])
            preds = model.predict(dtest)
            fwd = test[["timestamp", "symbol", "fwd_ret_12h"]].rename(
                columns={"fwd_ret_12h": "fwd_ret"}).dropna()
            merged = test_c[["timestamp", "symbol"]].copy()
            merged["pred"] = preds
            merged = merged.merge(fwd, on=["timestamp", "symbol"], how="inner")
            merged["window"] = w["name"]
            seed_preds.append(merged)
        if seed_preds:
            all_preds.append(pd.concat(seed_preds, ignore_index=True))
    if not all_preds:
        return None
    combined = pd.concat(all_preds, ignore_index=True)
    return (combined.groupby(["timestamp", "symbol"])
            .agg(pred=("pred", "mean"), fwd_ret=("fwd_ret", "first"),
                 window=("window", "first"))
            .reset_index())


def ensemble_preds(lgb_preds, xgb_preds):
    if lgb_preds is None or xgb_preds is None:
        return lgb_preds if lgb_preds is not None else xgb_preds
    merged = lgb_preds.rename(columns={"pred": "pred_lgb"}).merge(
        xgb_preds[["timestamp", "symbol", "pred"]].rename(columns={"pred": "pred_xgb"}),
        on=["timestamp", "symbol"], how="inner")
    merged["rank_lgb"] = merged.groupby("timestamp")["pred_lgb"].rank(pct=True) - 0.5
    merged["rank_xgb"] = merged.groupby("timestamp")["pred_xgb"].rank(pct=True) - 0.5
    merged["pred"] = 0.5 * merged["rank_lgb"] + 0.5 * merged["rank_xgb"]
    return merged[["timestamp", "symbol", "pred", "fwd_ret", "window"]]


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log("=" * 80)
    log("  R29 — Exhaustive Forward Feature Selection")
    log(f"  Date: {pd.Timestamp.now()}")
    log(f"  Candidates: {len(ALL_CANDIDATES)} features")
    log("=" * 80)

    t_start = time.time()

    # ── Load & build features ────────────────────────────────────
    log("\n  Loading data...")
    ohlcv = load_ohlcv()
    ohlcv = ohlcv[ohlcv["symbol"].isin(SYM_35)]
    derivs = load_derivatives()
    log(f"  OHLCV: {len(ohlcv):,} rows")

    log("  Building research features...")
    df = build_features_minimal(ohlcv, derivs)
    df = build_r19_features(df)
    df, _ = add_new_features(df)

    log("  Building production features...")
    df = build_production_features(df)
    df = df[df["symbol"].isin(SYM_35)].copy()
    log(f"  Final: {len(df):,} rows, {len(df.columns)} cols")

    # Check availability
    avail = [f for f in ALL_CANDIDATES if f in df.columns]
    miss = [f for f in ALL_CANDIDATES if f not in df.columns]
    log(f"  Available: {len(avail)}/{len(ALL_CANDIDATES)}")
    if miss:
        log(f"  Missing: {miss}")

    regime_df = compute_regime(df)
    load_time = time.time() - t_start
    log(f"  Load/build time: {load_time:.0f}s\n")

    # ── Baseline ─────────────────────────────────────────────────
    log("  [BASELINE] FEATURES_23")
    p_lgb = train_lgb_cls(df, FEATURES_23)
    p_xgb = train_xgb_cls(df, FEATURES_23)
    p_ens = ensemble_preds(p_lgb, p_xgb)
    port = simulate(p_ens, regime_df, 12, CFG_6L3S)
    r_base = eval_config(port, 12, "baseline-23f", LEVERAGE, CAPITAL)
    if r_base:
        show(r_base)
    baseline_sh = r_base["sharpe"] if r_base else 0
    log(f"  Baseline Sharpe: {baseline_sh:.2f}\n")

    # ── Forward selection ────────────────────────────────────────
    results = [("BASELINE (23f)", baseline_sh, r_base.get("equity", 0) if r_base else 0)]

    for i, feat in enumerate(avail):
        feats_24 = FEATURES_23 + [feat]
        log(f"\n  [{i+1}/{len(avail)}] +{feat} ({len(feats_24)}f)")
        t0 = time.time()

        p_lgb = train_lgb_cls(df, feats_24)
        p_xgb = train_xgb_cls(df, feats_24)
        p_ens = ensemble_preds(p_lgb, p_xgb)
        port = simulate(p_ens, regime_df, 12, CFG_6L3S)
        r = eval_config(port, 12, f"+{feat}", LEVERAGE, CAPITAL)
        if r:
            show(r)
            delta = r["sharpe"] - baseline_sh
            log(f"  Delta: {delta:+.2f}  ({time.time()-t0:.0f}s)")
            results.append((f"+{feat}", r["sharpe"], r["equity"]))
        else:
            log(f"  FAILED ({time.time()-t0:.0f}s)")
            results.append((f"+{feat}", None, None))

    # ── Summary ──────────────────────────────────────────────────
    log("\n" + "=" * 80)
    log("  R29 FORWARD SELECTION RESULTS")
    log("=" * 80)
    for name, sh, eq in results:
        if sh is not None:
            delta = sh - baseline_sh if name != "BASELINE (23f)" else 0
            marker = " ★★★" if delta > 0.1 else (" ★" if delta > 0 else "")
            log(f"  {name:35s}  Sh={sh:.2f}  Eq=${eq:.0f}  delta={delta:+.2f}{marker}")
        else:
            log(f"  {name:35s}  SKIP/FAIL")

    # Top features
    additions = [(n, s, e) for n, s, e in results if s is not None and n != "BASELINE (23f)"]
    if additions:
        additions.sort(key=lambda x: x[1], reverse=True)
        log("\n  TOP 10 features by Sharpe:")
        for n, s, e in additions[:10]:
            log(f"    {n:35s}  Sh={s:.2f}  delta={s - baseline_sh:+.2f}")
        
        # Features that beat baseline
        winners = [(n, s) for n, s, e in additions if s > baseline_sh]
        if winners:
            log(f"\n  🏆 WINNERS (beat Sh={baseline_sh:.2f}):")
            for n, s in winners:
                log(f"    {n:35s}  Sh={s:.2f}  delta={s - baseline_sh:+.2f}")
        else:
            log(f"\n  ❌ NO feature improved Sh={baseline_sh:.2f}")

    total = time.time() - t_start
    log(f"\n  Total runtime: {total/60:.1f} min")
    log("  Done.")


if __name__ == "__main__":
    main()
