#!/usr/bin/env python3
"""
R18 — Major Model Improvement Round.

Strategy: expand features, better targets, multi-model ensemble.

New feature groups:
  A) News sentiment (crypto_news.parquet — 2.4M rows, untapped)
  B) TA features from crypto_features_1h.parquet (RSI, MACD, BB, ATR, etc.)
  C) Fear & Greed index
  D) DVOL (BTC implied volatility)
  E) Macro (VIX, DXY)
  F) Unused derivative fields (top_long_pct, global_long_pct)
  G) Interaction features & time-series z-scores
  
Target engineering:
  H) Winsorized target (clip extreme returns)
  I) Residual target (remove BTC component)
  
Model improvements:
  J) XGBoost ensemble
  K) CatBoost ensemble
  L) LGB + XGB + CatBoost meta-ensemble
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from sklearn.linear_model import Ridge
from pathlib import Path
import warnings, time, sys
warnings.filterwarnings("ignore")

from _research_round7 import (
    SYM_35, WINDOWS, cs_rank,
    compute_regime, simulate, eval_config, show,
)
from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal

PROJECT = Path(__file__).parent
DATA_DIR = PROJECT / "data"
SENT_DIR = DATA_DIR / "sentiment"

FEATURES_12 = [
    "ret_12h", "ret_24h", "ret_48h",
    "residual_12h", "residual_24h",
    "mom_z_24h",
    "oi_chg_12h", "oi_chg_24h", "oi_zscore",
    "taker_cvd_12h", "taker_cvd_24h",
    "ls_divergence",
]

SEEDS = [0, 7, 13, 42, 99]
LEVERAGE = 5
CAPITAL = 100

CFG_BARE = {
    "n_long": 6, "n_short": 3, "trend_cutoff": 999,
    "dyn_threshold": None, "kelly_sizing": False,
    "vol_scaling": False, "regime_asym": False, "rebal_hours": 12,
}
CFG_REGIME = {
    "n_long": 6, "n_short": 3, "trend_cutoff": 0.8,
    "dyn_threshold": 0.5, "kelly_sizing": False,
    "vol_scaling": False, "regime_asym": False, "rebal_hours": 12,
}


def log(msg):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════

def add_news_features(df):
    """Add news sentiment features from crypto_news.parquet."""
    try:
        news = pd.read_parquet(SENT_DIR / "crypto_news.parquet")
        news["timestamp"] = pd.to_datetime(news["timestamp"], utc=True)
        # Key features: sentiment, volume zscore, momentum
        news_cols = ["news_sentiment_24h", "news_sentiment_7d",
                     "news_sentiment_momentum", "news_volume_zscore",
                     "market_news_sentiment_24h"]
        avail = [c for c in news_cols if c in news.columns]
        if not avail:
            log("  [NEWS] No usable columns found")
            return df
        df = df.merge(news[["timestamp", "symbol"] + avail],
                      on=["timestamp", "symbol"], how="left")
        for c in avail:
            df[c] = df.groupby("symbol")[c].ffill()
        log(f"  [NEWS] Added {len(avail)} features: {avail}")
        return df
    except Exception as e:
        log(f"  [NEWS] Error: {e}")
        return df


def add_ta_features(df):
    """Add select TA features from crypto_features_1h.parquet."""
    try:
        ta_path = DATA_DIR / "features" / "crypto_features_1h.parquet"
        ta = pd.read_parquet(ta_path)
        ta["timestamp"] = pd.to_datetime(ta["timestamp"], utc=True)
        # Select most promising TA features
        ta_cols = []
        for c in ["rsi_14", "macd_diff", "bb_width_20", "bb_pband_20",
                   "atr_14", "adx", "stoch_k", "cci_14", "willr_14",
                   "obv_ma_ratio_24", "mfi_14", "gk_vol_24h",
                   "ret_skew_24h", "ret_kurt_24h", "ret_sharpe_24h",
                   "vwap_dev_24h", "buy_pressure", "vol_price_corr_24h",
                   "vol_mom_24h", "bb_width_48"]:
            if c in ta.columns:
                ta_cols.append(c)
        if not ta_cols:
            log(f"  [TA] No matching columns. Available: {ta.columns.tolist()[:20]}")
            return df
        df = df.merge(ta[["timestamp", "symbol"] + ta_cols],
                      on=["timestamp", "symbol"], how="left")
        log(f"  [TA] Added {len(ta_cols)} features: {ta_cols}")
        return df
    except Exception as e:
        log(f"  [TA] Error: {e}")
        return df


def add_fng_features(df):
    """Add Fear & Greed index."""
    try:
        fng = pd.read_parquet(SENT_DIR / "fear_greed.parquet")
        fng["timestamp"] = pd.to_datetime(fng["timestamp"], utc=True)
        fng["date"] = fng["timestamp"].dt.date
        df["date"] = df["timestamp"].dt.date
        df = df.merge(fng[["date", "fng_value"]], on="date", how="left")
        df["fng_value"] = df["fng_value"].ffill()
        # Z-score
        df["fng_zscore"] = (df["fng_value"] - df["fng_value"].rolling(168*7, min_periods=168).mean()) / \
                           (df["fng_value"].rolling(168*7, min_periods=168).std() + 1e-10)
        # Extreme flags
        df["fng_extreme_fear"] = (df["fng_value"] < 20).astype(float)
        df["fng_extreme_greed"] = (df["fng_value"] > 80).astype(float)
        df.drop(columns=["date"], inplace=True)
        log(f"  [FNG] Added: fng_value, fng_zscore, fng_extreme_fear, fng_extreme_greed")
        return df
    except Exception as e:
        log(f"  [FNG] Error: {e}")
        return df


def add_dvol_features(df):
    """Add BTC DVOL (implied volatility) features."""
    try:
        dv = pd.read_parquet(SENT_DIR / "deribit_dvol.parquet")
        dv["timestamp"] = pd.to_datetime(dv["timestamp"], utc=True)
        btc_dv = dv[dv["currency"] == "BTC"][["timestamp", "dvol_close"]].copy()
        btc_dv = btc_dv.sort_values("timestamp").rename(columns={"dvol_close": "btc_dvol"})
        btc_dv = btc_dv.set_index("timestamp").resample("1h").ffill().reset_index()
        df = df.merge(btc_dv, on="timestamp", how="left")
        df["btc_dvol"] = df["btc_dvol"].ffill()
        # DVOL z-score
        df["dvol_zscore"] = (df["btc_dvol"] - df["btc_dvol"].rolling(168*4, min_periods=168).mean()) / \
                            (df["btc_dvol"].rolling(168*4, min_periods=168).std() + 1e-10)
        # DVOL change
        df["dvol_chg_24h"] = df["btc_dvol"].pct_change(24)
        # IV-RV spread (implied vs realized)
        if "rvol_24h" in df.columns:
            df["iv_rv_spread"] = df["btc_dvol"] / 100 - df.groupby("symbol")["rvol_24h"].transform(
                lambda x: x * np.sqrt(24 * 365))
        log(f"  [DVOL] Added: btc_dvol, dvol_zscore, dvol_chg_24h, iv_rv_spread")
        return df
    except Exception as e:
        log(f"  [DVOL] Error: {e}")
        return df


def add_macro_features(df):
    """Add macro features (VIX, DXY)."""
    try:
        macro = pd.read_parquet(SENT_DIR / "macro_daily.parquet")
        macro["date"] = pd.to_datetime(macro["date"]).dt.date
        df["date"] = df["timestamp"].dt.date
        macro_cols = []
        if "vix_close" in macro.columns:
            macro["vix_ret_5d"] = macro["vix_close"].pct_change(5)
            macro["vix_zscore"] = (macro["vix_close"] - macro["vix_close"].rolling(60).mean()) / \
                                  (macro["vix_close"].rolling(60).std() + 1e-10)
            macro_cols.extend(["vix_close", "vix_ret_5d", "vix_zscore"])
        if "dxy_close" in macro.columns:
            macro["dxy_ret_5d"] = macro["dxy_close"].pct_change(5)
            macro_cols.append("dxy_ret_5d")
        if "yield_curve_10y2y" in macro.columns:
            macro_cols.append("yield_curve_10y2y")

        df = df.merge(macro[["date"] + macro_cols], on="date", how="left")
        for c in macro_cols:
            df[c] = df[c].ffill()
        df.drop(columns=["date"], inplace=True)
        log(f"  [MACRO] Added: {macro_cols}")
        return df
    except Exception as e:
        log(f"  [MACRO] Error: {e}")
        return df


def add_extra_deriv_features(df):
    """Add unused derivative features (long_pct fields) and interaction features."""
    try:
        fm = pd.read_parquet(SENT_DIR / "binance_futures_metrics.parquet")
        fm["timestamp"] = pd.to_datetime(fm["timestamp"], utc=True)
        extra_cols = []
        for c in ["top_long_pct", "global_long_pct"]:
            if c in fm.columns:
                extra_cols.append(c)
        if extra_cols:
            df = df.merge(fm[["timestamp", "symbol"] + extra_cols],
                          on=["timestamp", "symbol"], how="left")
            # Long pct z-scores
            for c in extra_cols:
                mean = df.groupby("symbol")[c].transform(lambda x: x.rolling(168, min_periods=84).mean())
                std = df.groupby("symbol")[c].transform(lambda x: x.rolling(168, min_periods=84).std()) + 1e-10
                df[f"{c}_zscore"] = (df[c] - mean) / std
            # Crowd divergence: top traders vs global
            if "top_long_pct" in df.columns and "global_long_pct" in df.columns:
                df["smart_money_diverge"] = df["top_long_pct"] - df["global_long_pct"]
        log(f"  [DERIV+] Added: {extra_cols + [c + '_zscore' for c in extra_cols] + ['smart_money_diverge']}")
    except Exception as e:
        log(f"  [DERIV+] Error: {e}")

    # Interaction features (from existing data)
    if "oi_chg_12h" in df.columns and "ret_12h" in df.columns:
        df["oi_contra_price"] = df["oi_chg_12h"] * (-df["ret_12h"])  # OI up + price down
    if "funding_rate_binance" in df.columns and "oi_chg_12h" in df.columns:
        df["oi_funding_interact"] = df["oi_chg_12h"] * df["funding_rate_binance"]
    if "taker_imbalance" in df.columns:
        df["taker_accel"] = df.groupby("symbol")["taker_imbalance"].diff(12)

    # Time-series z-scores (normalize features by own history)
    for feat in ["ret_12h", "oi_chg_12h", "taker_cvd_12h"]:
        if feat in df.columns:
            ts_mean = df.groupby("symbol")[feat].transform(lambda x: x.rolling(168*4, min_periods=168).mean())
            ts_std = df.groupby("symbol")[feat].transform(lambda x: x.rolling(168*4, min_periods=168).std()) + 1e-10
            df[f"ts_z_{feat}"] = (df[feat] - ts_mean) / ts_std

    log(f"  [INTERACT] Added interaction + TS z-scores")
    return df


# ═══════════════════════════════════════════════════════════════
#  MODEL TRAINING
# ═══════════════════════════════════════════════════════════════

def cs_rank_inplace(df, feats):
    df = df.copy()
    for feat in feats:
        if feat in df.columns:
            df[feat] = df.groupby("timestamp")[feat].rank(pct=True) - 0.5
    return df


def train_lgb_ensemble(df, feats, seeds=SEEDS, params_override=None, target_col="fwd_ret_12h"):
    """Train LGB 5-seed ensemble."""
    all_preds = []
    tz = df["timestamp"].dt.tz

    for seed in seeds:
        seed_preds = []
        for w in WINDOWS:
            train = df[df["timestamp"] < pd.Timestamp(w["train_end"], tz=tz)].copy()
            val   = df[(df["timestamp"] >= pd.Timestamp(w["val_start"], tz=tz)) &
                       (df["timestamp"] <  pd.Timestamp(w["val_end"],   tz=tz))].copy()
            test  = df[(df["timestamp"] >= pd.Timestamp(w["test_start"], tz=tz)) &
                       (df["timestamp"] <= pd.Timestamp(w["test_end"],   tz=tz))].copy()
            if len(train) < 5000 or len(test) < 200:
                continue

            train = cs_rank_inplace(train, feats)
            val   = cs_rank_inplace(val, feats)
            test  = cs_rank_inplace(test, feats)

            for d in [train, val, test]:
                d["target_rank"] = d.groupby("timestamp")[target_col].rank(pct=True) - 0.5

            train_c = train[feats + ["target_rank"]].dropna()
            val_c   = val[feats + ["target_rank"]].dropna()

            params = {
                "objective": "regression", "metric": "mse",
                "learning_rate": 0.03, "num_leaves": 63,
                "min_child_samples": 100,
                "subsample": 0.8, "colsample_bytree": 0.8,
                "lambda_l2": 1.0,
                "seed": seed,
                "verbose": -1, "n_jobs": -1,
            }
            if params_override:
                params.update(params_override)

            dtrain = lgb.Dataset(train_c[feats], label=train_c["target_rank"])
            dval   = lgb.Dataset(val_c[feats],   label=val_c["target_rank"])
            model = lgb.train(params, dtrain, num_boost_round=500,
                              valid_sets=[dval],
                              callbacks=[lgb.early_stopping(30, verbose=False),
                                         lgb.log_evaluation(-1)])

            test_c = test[feats + ["target_rank", "timestamp", "symbol"]].dropna()
            preds = model.predict(test_c[feats])

            fwd_data = test[["timestamp", "symbol", "fwd_ret_12h"]].rename(
                columns={"fwd_ret_12h": "fwd_ret"}).dropna()
            merged = test_c[["timestamp", "symbol"]].copy()
            merged["pred"] = preds
            merged = merged.merge(fwd_data, on=["timestamp", "symbol"], how="inner")
            merged["window"] = w["name"]
            seed_preds.append(merged)

        if seed_preds:
            all_preds.append(pd.concat(seed_preds, ignore_index=True))

    if not all_preds:
        return None
    combined = pd.concat(all_preds, ignore_index=True)
    ens = (combined.groupby(["timestamp", "symbol"])
           .agg(pred=("pred", "mean"), fwd_ret=("fwd_ret", "first"),
                window=("window", "first"))
           .reset_index())
    return ens


def train_xgb_ensemble(df, feats, seeds=SEEDS, target_col="fwd_ret_12h"):
    """Train XGBoost 5-seed ensemble."""
    import xgboost as xgb

    all_preds = []
    tz = df["timestamp"].dt.tz

    for seed in seeds:
        seed_preds = []
        for w in WINDOWS:
            train = df[df["timestamp"] < pd.Timestamp(w["train_end"], tz=tz)].copy()
            val   = df[(df["timestamp"] >= pd.Timestamp(w["val_start"], tz=tz)) &
                       (df["timestamp"] <  pd.Timestamp(w["val_end"],   tz=tz))].copy()
            test  = df[(df["timestamp"] >= pd.Timestamp(w["test_start"], tz=tz)) &
                       (df["timestamp"] <= pd.Timestamp(w["test_end"],   tz=tz))].copy()
            if len(train) < 5000 or len(test) < 200:
                continue

            train = cs_rank_inplace(train, feats)
            val   = cs_rank_inplace(val, feats)
            test  = cs_rank_inplace(test, feats)

            for d in [train, val, test]:
                d["target_rank"] = d.groupby("timestamp")[target_col].rank(pct=True) - 0.5

            train_c = train[feats + ["target_rank"]].dropna()
            val_c   = val[feats + ["target_rank"]].dropna()

            dtrain = xgb.DMatrix(train_c[feats], label=train_c["target_rank"])
            dval   = xgb.DMatrix(val_c[feats],   label=val_c["target_rank"])

            params = {
                "objective": "reg:squarederror",
                "max_depth": 6, "learning_rate": 0.03,
                "min_child_weight": 100, "subsample": 0.8,
                "colsample_bytree": 0.8, "lambda": 1.0,
                "seed": seed, "nthread": -1, "verbosity": 0,
            }
            model = xgb.train(params, dtrain, num_boost_round=500,
                              evals=[(dval, "val")],
                              early_stopping_rounds=30, verbose_eval=False)

            test_c = test[feats + ["target_rank", "timestamp", "symbol"]].dropna()
            dtest = xgb.DMatrix(test_c[feats])
            preds = model.predict(dtest)

            fwd_data = test[["timestamp", "symbol", "fwd_ret_12h"]].rename(
                columns={"fwd_ret_12h": "fwd_ret"}).dropna()
            merged = test_c[["timestamp", "symbol"]].copy()
            merged["pred"] = preds
            merged = merged.merge(fwd_data, on=["timestamp", "symbol"], how="inner")
            merged["window"] = w["name"]
            seed_preds.append(merged)

        if seed_preds:
            all_preds.append(pd.concat(seed_preds, ignore_index=True))

    if not all_preds:
        return None
    combined = pd.concat(all_preds, ignore_index=True)
    ens = (combined.groupby(["timestamp", "symbol"])
           .agg(pred=("pred", "mean"), fwd_ret=("fwd_ret", "first"),
                window=("window", "first"))
           .reset_index())
    return ens


def train_catboost_ensemble(df, feats, seeds=SEEDS, target_col="fwd_ret_12h"):
    """Train CatBoost 5-seed ensemble."""
    from catboost import CatBoostRegressor, Pool

    all_preds = []
    tz = df["timestamp"].dt.tz

    for seed in seeds:
        seed_preds = []
        for w in WINDOWS:
            train = df[df["timestamp"] < pd.Timestamp(w["train_end"], tz=tz)].copy()
            val   = df[(df["timestamp"] >= pd.Timestamp(w["val_start"], tz=tz)) &
                       (df["timestamp"] <  pd.Timestamp(w["val_end"],   tz=tz))].copy()
            test  = df[(df["timestamp"] >= pd.Timestamp(w["test_start"], tz=tz)) &
                       (df["timestamp"] <= pd.Timestamp(w["test_end"],   tz=tz))].copy()
            if len(train) < 5000 or len(test) < 200:
                continue

            train = cs_rank_inplace(train, feats)
            val   = cs_rank_inplace(val, feats)
            test  = cs_rank_inplace(test, feats)

            for d in [train, val, test]:
                d["target_rank"] = d.groupby("timestamp")[target_col].rank(pct=True) - 0.5

            train_c = train[feats + ["target_rank"]].dropna()
            val_c   = val[feats + ["target_rank"]].dropna()

            model = CatBoostRegressor(
                iterations=500, learning_rate=0.03,
                depth=6, l2_leaf_reg=1.0,
                subsample=0.8, colsample_bylevel=0.8,
                random_seed=seed, verbose=0,
                early_stopping_rounds=30,
            )
            model.fit(train_c[feats], train_c["target_rank"],
                      eval_set=(val_c[feats], val_c["target_rank"]),
                      verbose=False)

            test_c = test[feats + ["target_rank", "timestamp", "symbol"]].dropna()
            preds = model.predict(test_c[feats])

            fwd_data = test[["timestamp", "symbol", "fwd_ret_12h"]].rename(
                columns={"fwd_ret_12h": "fwd_ret"}).dropna()
            merged = test_c[["timestamp", "symbol"]].copy()
            merged["pred"] = preds
            merged = merged.merge(fwd_data, on=["timestamp", "symbol"], how="inner")
            merged["window"] = w["name"]
            seed_preds.append(merged)

        if seed_preds:
            all_preds.append(pd.concat(seed_preds, ignore_index=True))

    if not all_preds:
        return None
    combined = pd.concat(all_preds, ignore_index=True)
    ens = (combined.groupby(["timestamp", "symbol"])
           .agg(pred=("pred", "mean"), fwd_ret=("fwd_ret", "first"),
                window=("window", "first"))
           .reset_index())
    return ens


# ═══════════════════════════════════════════════════════════════
#  IC ANALYSIS
# ═══════════════════════════════════════════════════════════════

def ic_analysis(preds, label):
    """Per-window IC analysis."""
    log(f"\n  IC: {label}")
    all_ics = []
    for wname in ["W1", "W2", "W3"]:
        sub = preds[preds["window"] == wname] if "window" in preds.columns else preds
        ics = []
        for ts, grp in sub.groupby("timestamp"):
            if len(grp) >= 10:
                ic = stats.spearmanr(grp["pred"], grp["fwd_ret"])[0]
                ics.append(ic)
        if ics:
            ic_arr = np.array(ics)
            all_ics.extend(ics)
            log(f"    {wname}: IC={ic_arr.mean():.4f} +/-{ic_arr.std():.4f} "
                f"IC>0={100*(ic_arr>0).mean():.0f}%")
    if all_ics:
        a = np.array(all_ics)
        log(f"    ALL: IC={a.mean():.4f} +/-{a.std():.4f}")
    return np.mean(all_ics) if all_ics else 0


def evaluate(preds, regime_df, label, cfgs=None):
    """Evaluate predictions with multiple portfolio configs."""
    if preds is None or len(preds) < 100:
        log(f"  {label}: no predictions")
        return []
    results = []
    if cfgs is None:
        cfgs = [("bare", CFG_BARE), ("regime", CFG_REGIME)]
    for cfg_name, cfg in cfgs:
        sub = simulate(preds, regime_df, 12, cfg)
        r = eval_config(sub, 12, f"{label} [{cfg_name}]", LEVERAGE, CAPITAL)
        if r:
            results.append(r)
            show(r)
    return results


def per_window_sharpe(preds, regime_df, cfg):
    """Return dict of per-window Sharpe."""
    result = {}
    for wname in ["W1", "W2", "W3"]:
        sub = preds[preds["window"] == wname] if "window" in preds.columns else preds
        if len(sub) == 0:
            continue
        port = simulate(sub, regime_df, 12, cfg)
        r = eval_config(port, 12, wname)
        if r:
            result[wname] = r["sharpe"]
    return result


# ═══════════════════════════════════════════════════════════════
#  FEATURE IC SCAN
# ═══════════════════════════════════════════════════════════════

def scan_feature_ics(df, candidate_feats, target="fwd_ret_12h"):
    """Quick IC scan for candidate features to pick the best ones."""
    log(f"\n  Feature IC scan ({len(candidate_feats)} candidates):")
    results = []
    tz = df["timestamp"].dt.tz

    for feat in candidate_feats:
        if feat not in df.columns:
            continue
        ics_oos = []
        for w in WINDOWS:
            test = df[(df["timestamp"] >= pd.Timestamp(w["test_start"], tz=tz)) &
                      (df["timestamp"] <= pd.Timestamp(w["test_end"],   tz=tz))].copy()
            if len(test) < 200:
                continue
            # CS-rank the feature and compute IC per timestamp
            test["feat_r"] = test.groupby("timestamp")[feat].rank(pct=True) - 0.5
            test["tgt_r"] = test.groupby("timestamp")[target].rank(pct=True) - 0.5
            for ts, grp in test.groupby("timestamp"):
                valid = grp[["feat_r", "tgt_r"]].dropna()
                if len(valid) >= 10:
                    ic = stats.spearmanr(valid["feat_r"], valid["tgt_r"])[0]
                    if not np.isnan(ic):
                        ics_oos.append(ic)

        if ics_oos:
            arr = np.array(ics_oos)
            mean_ic = arr.mean()
            icir = mean_ic / (arr.std() + 1e-10)
            results.append({"feature": feat, "mean_ic": mean_ic, "icir": icir,
                           "ic_pos_pct": (arr > 0).mean(), "n": len(arr)})

    results.sort(key=lambda x: abs(x["mean_ic"]), reverse=True)
    for r in results[:30]:
        sign = "+" if r["mean_ic"] > 0 else "-"
        log(f"    {sign} {r['feature']:<35s} IC={r['mean_ic']:+.4f} "
            f"ICIR={r['icir']:+.3f} IC>0={r['ic_pos_pct']*100:.0f}%")
    return results


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log("=" * 80)
    log("  R18 — MAJOR MODEL IMPROVEMENT ROUND")
    log("=" * 80)

    # ── Load base data ──
    log("\n  Loading base data...")
    ohlcv  = load_ohlcv()
    ohlcv  = ohlcv[ohlcv["symbol"].isin(SYM_35)]
    derivs = load_derivatives()
    df     = build_features_minimal(ohlcv, derivs)
    regime_df = compute_regime(df)
    log(f"  Base: {len(df):,} rows, {df['symbol'].nunique()} symbols")

    # ── Enrich with new features ──
    log(f"\n{'═' * 80}")
    log(f"  FEATURE ENRICHMENT")
    log(f"{'═' * 80}")

    df = add_news_features(df)
    df = add_ta_features(df)
    df = add_fng_features(df)
    df = add_dvol_features(df)
    df = add_macro_features(df)
    df = add_extra_deriv_features(df)

    total_feats = [c for c in df.columns if c not in
                   ["timestamp", "symbol", "open", "high", "low", "close", "volume",
                    "btc_close", "coin_ret", "btc_ret", "ret_1h_sq", "date",
                    "oi_value_usd", "taker_buy_sell_ratio", "top_ls_ratio",
                    "global_ls_ratio", "premium_index", "funding_rate_binance",
                    "top_long_pct", "global_long_pct"] and
                   not c.startswith("fwd_ret")]
    log(f"\n  Total available features: {len(total_feats)}")

    # ── Target engineering ──
    log(f"\n  Target engineering...")
    # Winsorized target: clip extreme returns at 1st/99th percentile
    for col in ["fwd_ret_12h"]:
        q01 = df[col].quantile(0.01)
        q99 = df[col].quantile(0.99)
        df[f"{col}_wins"] = df[col].clip(q01, q99)
        log(f"  Winsorized {col}: [{q01:.4f}, {q99:.4f}]")

    # Residual target: remove BTC component
    if "btc_beta_168h" in df.columns and "btc_ret_12h" in df.columns:
        df["fwd_ret_12h_resid"] = df["fwd_ret_12h"] - df["btc_beta_168h"] * df.groupby("symbol")["btc_ret_12h"].shift(-12)
        log(f"  Created residual target: fwd_ret_12h_resid")

    results_all = []

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: Feature IC Scan
    # ═══════════════════════════════════════════════════════════
    log(f"\n{'═' * 80}")
    log(f"  PHASE 1: Feature IC Scan")
    log(f"{'═' * 80}")

    # New candidate features to scan
    new_candidates = [c for c in total_feats if c not in FEATURES_12]
    ic_results = scan_feature_ics(df, new_candidates)

    # Pick features with |IC| > 0.005 and IC>0 > 52%
    good_new_feats = [r["feature"] for r in ic_results
                      if abs(r["mean_ic"]) > 0.005 and r["ic_pos_pct"] > 0.52]
    log(f"\n  Good new features (|IC|>0.005, IC>0>52%): {len(good_new_feats)}")
    for f in good_new_feats[:15]:
        log(f"    {f}")

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: LGB with expanded features
    # ═══════════════════════════════════════════════════════════
    log(f"\n{'═' * 80}")
    log(f"  PHASE 2: LGB with expanded feature sets")
    log(f"{'═' * 80}")

    # 2a: Baseline (12 features)
    feats_12 = [f for f in FEATURES_12 if f in df.columns]
    log(f"\n  2a: LGB baseline (12f)...")
    p_base = train_lgb_ensemble(df, feats_12)
    ic_base = ic_analysis(p_base, "LGB-12f baseline")
    results_all.extend(evaluate(p_base, regime_df, "LGB-12f"))

    # 2b: 12f + all good new features
    if good_new_feats:
        feats_expanded = feats_12 + [f for f in good_new_feats if f in df.columns]
        feats_expanded = list(dict.fromkeys(feats_expanded))  # deduplicate
        log(f"\n  2b: LGB expanded ({len(feats_expanded)}f)...")
        p_exp = train_lgb_ensemble(df, feats_expanded)
        ic_exp = ic_analysis(p_exp, f"LGB-{len(feats_expanded)}f expanded")
        results_all.extend(evaluate(p_exp, regime_df, f"LGB-{len(feats_expanded)}f"))

    # 2c: 12f + top-5 new features only
    top5_new = [r["feature"] for r in ic_results[:5] if r["feature"] in df.columns]
    if top5_new:
        feats_top5 = feats_12 + top5_new
        log(f"\n  2c: LGB 12f + top5 ({len(feats_top5)}f): {top5_new}...")
        p_top5 = train_lgb_ensemble(df, feats_top5)
        ic_top5 = ic_analysis(p_top5, f"LGB-{len(feats_top5)}f top5")
        results_all.extend(evaluate(p_top5, regime_df, f"LGB-{len(feats_top5)}f-top5"))

    # 2d: Kitchen sink (all 42+ original features)
    all_orig_feats = [f for f in [
        "ret_12h", "ret_24h", "ret_48h", "ret_168h",
        "rvol_12h", "rvol_24h", "rvol_168h",
        "vol_ratio_12h", "vol_ratio_24h",
        "mom_z_12h", "mom_z_24h",
        "range_24h", "dist_from_high_24h",
        "residual_12h", "residual_24h",
        "cum_funding_24h", "cum_funding_72h", "cum_funding_168h",
        "funding_zscore", "funding_x_mom_12h", "funding_x_mom_24h",
        "oi_chg_1h", "oi_chg_4h", "oi_chg_12h", "oi_chg_24h", "oi_zscore",
        "oi_ret_diverge",
        "taker_cvd_4h", "taker_cvd_12h", "taker_cvd_24h", "taker_zscore",
        "top_ls_ratio_zscore", "global_ls_ratio_zscore", "ls_divergence",
        "premium_zscore",
    ] if f in df.columns]
    log(f"\n  2d: LGB kitchen-sink ({len(all_orig_feats)}f)...")
    p_ks = train_lgb_ensemble(df, all_orig_feats)
    ic_ks = ic_analysis(p_ks, f"LGB-{len(all_orig_feats)}f kitchen-sink")
    results_all.extend(evaluate(p_ks, regime_df, f"LGB-{len(all_orig_feats)}f-ks"))

    # ═══════════════════════════════════════════════════════════
    # PHASE 3: Target engineering
    # ═══════════════════════════════════════════════════════════
    log(f"\n{'═' * 80}")
    log(f"  PHASE 3: Target engineering")
    log(f"{'═' * 80}")

    # Pick the best feature set so far
    best_feats = feats_12  # default
    if good_new_feats:
        best_feats = feats_12 + [f for f in good_new_feats if f in df.columns][:10]
        best_feats = list(dict.fromkeys(best_feats))

    # 3a: Winsorized target
    log(f"\n  3a: LGB with winsorized target...")
    p_wins = train_lgb_ensemble(df, best_feats, target_col="fwd_ret_12h_wins")
    if p_wins is not None:
        ic_wins = ic_analysis(p_wins, "LGB winsorized target")
        results_all.extend(evaluate(p_wins, regime_df, "LGB-wins-target"))

    # 3b: Residual target
    if "fwd_ret_12h_resid" in df.columns:
        log(f"\n  3b: LGB with residual target...")
        p_resid = train_lgb_ensemble(df, best_feats, target_col="fwd_ret_12h_resid")
        if p_resid is not None:
            ic_resid = ic_analysis(p_resid, "LGB residual target")
            results_all.extend(evaluate(p_resid, regime_df, "LGB-resid-target"))

    # ═══════════════════════════════════════════════════════════
    # PHASE 4: XGBoost
    # ═══════════════════════════════════════════════════════════
    log(f"\n{'═' * 80}")
    log(f"  PHASE 4: XGBoost ensemble")
    log(f"{'═' * 80}")

    try:
        log(f"\n  4a: XGB with 12f...")
        p_xgb12 = train_xgb_ensemble(df, feats_12)
        ic_xgb = ic_analysis(p_xgb12, "XGB-12f")
        results_all.extend(evaluate(p_xgb12, regime_df, "XGB-12f"))

        if best_feats != feats_12:
            log(f"\n  4b: XGB with expanded features...")
            p_xgb_exp = train_xgb_ensemble(df, best_feats)
            ic_xgb_exp = ic_analysis(p_xgb_exp, f"XGB-{len(best_feats)}f")
            results_all.extend(evaluate(p_xgb_exp, regime_df, f"XGB-{len(best_feats)}f"))
    except ImportError:
        log("  XGBoost not installed, skipping")
    except Exception as e:
        log(f"  XGBoost error: {e}")

    # ═══════════════════════════════════════════════════════════
    # PHASE 5: CatBoost
    # ═══════════════════════════════════════════════════════════
    log(f"\n{'═' * 80}")
    log(f"  PHASE 5: CatBoost ensemble")
    log(f"{'═' * 80}")

    try:
        log(f"\n  5a: CatBoost with 12f...")
        p_cb12 = train_catboost_ensemble(df, feats_12)
        ic_cb = ic_analysis(p_cb12, "CatBoost-12f")
        results_all.extend(evaluate(p_cb12, regime_df, "CB-12f"))

        if best_feats != feats_12:
            log(f"\n  5b: CatBoost with expanded features...")
            p_cb_exp = train_catboost_ensemble(df, best_feats)
            ic_cb_exp = ic_analysis(p_cb_exp, f"CatBoost-{len(best_feats)}f")
            results_all.extend(evaluate(p_cb_exp, regime_df, f"CB-{len(best_feats)}f"))
    except ImportError:
        log("  CatBoost not installed, skipping")
    except Exception as e:
        log(f"  CatBoost error: {e}")

    # ═══════════════════════════════════════════════════════════
    # PHASE 6: Multi-model meta-ensemble
    # ═══════════════════════════════════════════════════════════
    log(f"\n{'═' * 80}")
    log(f"  PHASE 6: Multi-model meta-ensemble")
    log(f"{'═' * 80}")

    # Collect all available predictions and average them
    model_preds = {}
    if p_base is not None: model_preds["lgb"] = p_base
    try:
        if p_xgb12 is not None: model_preds["xgb"] = p_xgb12
    except NameError:
        pass
    try:
        if p_cb12 is not None: model_preds["cb"] = p_cb12
    except NameError:
        pass

    if len(model_preds) >= 2:
        log(f"\n  Models available for ensemble: {list(model_preds.keys())}")

        # Simple average ensemble
        base_key = list(model_preds.keys())[0]
        meta = model_preds[base_key][["timestamp", "symbol", "fwd_ret", "window"]].copy()
        meta[f"pred_{base_key}"] = model_preds[base_key]["pred"]
        for k in list(model_preds.keys())[1:]:
            other = model_preds[k][["timestamp", "symbol", "pred"]].rename(
                columns={"pred": f"pred_{k}"})
            meta = meta.merge(other, on=["timestamp", "symbol"], how="inner")

        pred_cols = [f"pred_{k}" for k in model_preds.keys()]
        meta["pred"] = meta[pred_cols].mean(axis=1)
        log(f"  Meta-ensemble ({'+'.join(model_preds.keys())}): {len(meta):,} obs")
        ic_meta = ic_analysis(meta, f"Meta-{'+'.join(model_preds.keys())}")
        results_all.extend(evaluate(meta, regime_df,
                                    f"META-{'+'.join(model_preds.keys())}"))

        # Weighted ensemble (if we have enough models)
        if len(model_preds) >= 3:
            # Weight by IC
            weights = {}
            for k, preds in model_preds.items():
                ics = []
                for ts, grp in preds.groupby("timestamp"):
                    if len(grp) >= 10:
                        ic = stats.spearmanr(grp["pred"], grp["fwd_ret"])[0]
                        ics.append(ic)
                weights[k] = np.mean(ics) if ics else 0
            total_w = sum(max(0, w) for w in weights.values()) + 1e-10
            weights = {k: max(0, w) / total_w for k, w in weights.items()}
            log(f"  IC-weighted: {weights}")

            meta["pred_icw"] = sum(meta[f"pred_{k}"] * w for k, w in weights.items())
            meta["pred"] = meta["pred_icw"]
            results_all.extend(evaluate(meta, regime_df,
                                        f"META-ICW-{'+'.join(model_preds.keys())}"))

    # ═══════════════════════════════════════════════════════════
    # PHASE 7: LGB hyperparameter refinement on best feature set
    # ═══════════════════════════════════════════════════════════
    log(f"\n{'═' * 80}")
    log(f"  PHASE 7: LGB hyperparameter refinement ({len(best_feats)}f)")
    log(f"{'═' * 80}")

    hpo_configs = [
        ("nl=127,lr=0.02", {"num_leaves": 127, "learning_rate": 0.02}),
        ("nl=127,lr=0.03", {"num_leaves": 127, "learning_rate": 0.03}),
        ("nl=63,lr=0.02,L2=2", {"num_leaves": 63, "learning_rate": 0.02, "lambda_l2": 2.0}),
        ("nl=31,lr=0.05", {"num_leaves": 31, "learning_rate": 0.05}),
        ("nl=63,mc=50", {"min_child_samples": 50}),
        ("nl=63,bf=0.6,ff=0.6", {"bagging_fraction": 0.6, "feature_fraction": 0.6,
                                   "bagging_freq": 1}),
    ]

    for hpo_name, hpo_params in hpo_configs:
        log(f"\n  HPO: {hpo_name}...")
        p = train_lgb_ensemble(df, best_feats, params_override=hpo_params)
        if p is not None:
            ic_analysis(p, f"LGB-HPO-{hpo_name}")
            results_all.extend(evaluate(p, regime_df, f"LGB-HPO-{hpo_name}"))

    # ═══════════════════════════════════════════════════════════
    # RANKINGS
    # ═══════════════════════════════════════════════════════════
    log(f"\n{'═' * 80}")
    log(f"  FINAL RANKINGS ({len(results_all)} configs)")
    log(f"{'═' * 80}")

    if not results_all:
        log("  No results!")
        return

    results_all.sort(key=lambda x: x["sharpe"], reverse=True)

    log(f"\n  By Sharpe:")
    for i, r in enumerate(results_all[:25]):
        wm = f"{r['win_months']}/{r['total_months']}"
        flag = "OK" if r["worst_m"] > -0.15 else ("WARN" if r["worst_m"] > -0.25 else "BAD")
        log(f"    #{i+1:2d} [{flag:4s}] {r['name']:<55s} Sh={r['sharpe']:+.2f} "
            f"WM={wm} Wr={r['worst_m']*100:+.1f}% Eq=${r['equity']:.0f}")

    # Per-window for top-3
    log(f"\n  Per-window detail (top-3):")
    for r in results_all[:3]:
        log(f"    {r['name']}: Sh={r['sharpe']:.2f}")
        if hasattr(r.get("monthly", None), "__iter__"):
            for md in r.get("month_data", []):
                log(f"      {md['month']:>10s}  {md['ret']*100:>+7.1f}%  eq=${md['equity']:>7.0f}")

    elapsed = time.time() - t0
    log(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
