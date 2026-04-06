#!/usr/bin/env python3
"""
R19 — Model Improvement Round 2.

Changes vs R18:
  1) LEAKAGE FIX: IC scan now uses TRAIN data only (not test).
     R18 scanned IC on OOS test windows → selection bias.

  2) Regime features in model: include trend_strength + trend_direction
     as direct LGB features → no hard cutoff → model learns best threshold.
     Compare vs hard regime filter.

  3) Market breadth: per-timestamp % coins up / % beating BTC.
     Cross-sectional signal, no per-symbol leakage.

  4) Seasonality: hour_sin, hour_cos (daily crypto cycle signals).

  5) Funding carry: cum. funding z-score as "crowded longs" indicator.

  6) LGB + CatBoost 2-model ensemble (best two models from R18).

Confirmed R18 facts (no re-run needed):
  - Baseline (12f + regime): Sh=1.84
  - Best R18: LGB-17f-top5 + regime: Sh=2.23, WM=10/13, Wr=-22.4%
  - Top features by IC: atr_14, rvol_12h, gk_vol_24h, rvol_24h, iv_rv_spread
  - All top features are BACKWARD-LOOKING vol features — no TA leakage confirmed.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from scipy import stats
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

# ─── R18 winner feature set (confirmed good, backward-looking only) ──────────
FEATURES_12 = [
    "ret_12h", "ret_24h", "ret_48h",
    "residual_12h", "residual_24h",
    "mom_z_24h",
    "oi_chg_12h", "oi_chg_24h", "oi_zscore",
    "taker_cvd_12h", "taker_cvd_24h",
    "ls_divergence",
]
FEATURES_17 = FEATURES_12 + ["atr_14", "rvol_12h", "gk_vol_24h", "rvol_24h", "iv_rv_spread"]

SEEDS = [0, 7, 13, 42, 99]
LEVERAGE = 5
CAPITAL  = 100

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


# ═══════════════════════════════════════════════════════════════════════════════
#  NEW FEATURE BUILDERS (R19)
# ═══════════════════════════════════════════════════════════════════════════════

def add_vol_features(df):
    """
    Add TA vol features already confirmed good in R18.
    SOURCE: crypto_features_1h.parquet — correctly aligned (backward-looking).
    """
    try:
        ta_path = DATA_DIR / "features" / "crypto_features_1h.parquet"
        ta = pd.read_parquet(ta_path, columns=["timestamp", "symbol",
                                                "atr_14", "gk_vol_24h",
                                                "rsi_14", "bb_pband_20"])
        ta["timestamp"] = pd.to_datetime(ta["timestamp"], utc=True)
        df = df.merge(ta, on=["timestamp", "symbol"], how="left")
        log("  [TA-VOL] Added: atr_14, gk_vol_24h, rsi_14, bb_pband_20")
    except Exception as e:
        log(f"  [TA-VOL] Error: {e}")
    return df


def add_iv_rv_spread(df):
    """IV-RV spread: DVOL (implied) minus realized vol."""
    try:
        dv = pd.read_parquet(SENT_DIR / "deribit_dvol.parquet")
        dv["timestamp"] = pd.to_datetime(dv["timestamp"], utc=True)
        btc_dv = (dv[dv["currency"] == "BTC"][["timestamp", "dvol_close"]]
                  .sort_values("timestamp")
                  .rename(columns={"dvol_close": "btc_dvol"}))
        btc_dv = btc_dv.set_index("timestamp").resample("1h").ffill().reset_index()
        df = df.merge(btc_dv, on="timestamp", how="left")
        df["btc_dvol"] = df["btc_dvol"].ffill()
        if "rvol_24h" in df.columns:
            df["iv_rv_spread"] = (df["btc_dvol"] / 100
                                  - df.groupby("symbol")["rvol_24h"]
                                  .transform(lambda x: x * np.sqrt(24 * 365)))
        # Also: dvol z-score
        df["dvol_zscore"] = ((df["btc_dvol"] - df["btc_dvol"].rolling(168 * 4, min_periods=168).mean())
                             / (df["btc_dvol"].rolling(168 * 4, min_periods=168).std() + 1e-10))
        log("  [DVOL] Added: btc_dvol, iv_rv_spread, dvol_zscore")
    except Exception as e:
        log(f"  [DVOL] Error: {e}")
    return df


def add_market_breadth(df):
    """
    Market breadth: cross-sectional signals from ALL coins at each timestamp.
    - pct_coins_up_12h: fraction with ret_12h > 0 (market direction)
    - pct_coins_up_1h:  fraction with ret_1h > 0  (short-term breadth)
    - btc_outperform:   ret_12h - BTC ret_12h (relative strength)
    NO leakage: all computed purely within each timestamp group.
    """
    if "ret_12h" not in df.columns:
        log("  [BREADTH] ret_12h missing, skip")
        return df

    # Market breadth per timestamp
    breadth = (df.groupby("timestamp")[["ret_12h", "ret_1h"]]
               .agg(pct_coins_up_12h=("ret_12h", lambda x: (x > 0).mean()),
                    pct_coins_up_1h =("ret_1h",  lambda x: (x > 0).mean()))
               .reset_index())
    df = df.merge(breadth, on="timestamp", how="left")

    # BTC relative strength per timestamp
    btc_mask = df["symbol"] == "BTC/USDT"
    btc_ret  = df.loc[btc_mask, ["timestamp", "ret_12h"]].rename(
        columns={"ret_12h": "btc_ret12_ts"})
    df = df.merge(btc_ret, on="timestamp", how="left")
    df["btc_outperform"] = df["ret_12h"] - df["btc_ret12_ts"]
    df.drop(columns=["btc_ret12_ts"], inplace=True, errors="ignore")

    log("  [BREADTH] Added: pct_coins_up_12h, pct_coins_up_1h, btc_outperform")
    return df


def add_seasonality(df):
    """
    Hour-of-day seasonality. Crypto has consistent intraday patterns.
    Source: timestamp (pure calendar, zero leakage).
    """
    df["hour"] = df["timestamp"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow"]      = df["timestamp"].dt.dayofweek
    df["dow_sin"]  = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["dow"] / 7)
    df.drop(columns=["hour", "dow"], inplace=True)
    log("  [SEASON] Added: hour_sin, hour_cos, dow_sin, dow_cos")
    return df


def add_funding_carry(df):
    """
    Funding rate carry: cumulative funding as 'cost of crowded longs'.
    High positive funding → longs being squeezed → potential mean-reversion.
    All backward-looking rolling sums. No leakage.
    """
    try:
        fm = pd.read_parquet(SENT_DIR / "binance_funding_rates.parquet")
        fm["timestamp"] = pd.to_datetime(fm["timestamp"], utc=True)
        if "funding_rate_binance" not in fm.columns:
            log("  [FUNDING] funding_rate_binance not found")
            return df
        fund = fm[["timestamp", "symbol", "funding_rate_binance"]].copy()
        # 8h funding → map to hourly; forward-fill within symbol
        df = df.merge(fund, on=["timestamp", "symbol"], how="left")
        df["funding_rate_binance"] = df.groupby("symbol")["funding_rate_binance"].ffill()

        # Cumulative funding over past 24h and 168h (carrying cost)
        df["fund_cum_24h"] = df.groupby("symbol")["funding_rate_binance"].transform(
            lambda x: x.rolling(24, min_periods=12).sum())
        df["fund_cum_168h"] = df.groupby("symbol")["funding_rate_binance"].transform(
            lambda x: x.rolling(168, min_periods=84).sum())

        # Funding z-score vs own history
        f_mean = df.groupby("symbol")["fund_cum_24h"].transform(
            lambda x: x.rolling(168 * 4, min_periods=168).mean())
        f_std  = df.groupby("symbol")["fund_cum_24h"].transform(
            lambda x: x.rolling(168 * 4, min_periods=168).std()) + 1e-10
        df["fund_zscore_24h"] = (df["fund_cum_24h"] - f_mean) / f_std

        log("  [FUNDING] Added: fund_cum_24h, fund_cum_168h, fund_zscore_24h")
    except Exception as e:
        log(f"  [FUNDING] Error: {e}")
    return df


def add_regime_features(df, regime_df):
    """
    Add BTC regime metrics as direct model features (not hard cutoff).
    trend_strength, trend_direction, vol_regime.
    These are BTC-derived, backward-looking. No leakage.
    """
    reg = regime_df[["trend_strength", "trend_direction", "vol_regime"]].copy()
    reg.index.name = "timestamp"
    reg = reg.reset_index()
    df = df.merge(reg, on="timestamp", how="left")
    # Fill any NaN in regime features
    for c in ["trend_strength", "trend_direction", "vol_regime"]:
        if c in df.columns:
            df[c] = df[c].ffill().fillna(0)
    log("  [REGIME-FEAT] Added: trend_strength, trend_direction, vol_regime")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  IC SCAN — TRAIN SET ONLY (no leakage)
# ═══════════════════════════════════════════════════════════════════════════════

def scan_ic_train_only(df, candidate_feats, target="fwd_ret_12h"):
    """
    Compute IC on TRAIN data only (before val/test period).
    This avoids the R18 bug of computing IC on OOS test data.
    Returns sorted list of features by mean |IC| across windows.
    """
    log(f"\n  Feature IC scan (TRAIN ONLY, {len(candidate_feats)} candidates):")
    tz = df["timestamp"].dt.tz
    results = []

    for feat in candidate_feats:
        if feat not in df.columns:
            continue
        ics_train = []
        for w in WINDOWS:
            # Use ONLY training data for IC scan
            train = df[df["timestamp"] < pd.Timestamp(w["train_end"], tz=tz)].copy()
            # Further restrict to last 12 months of train to get recent IC
            cutoff = pd.Timestamp(w["train_end"], tz=tz) - pd.Timedelta(days=365)
            train  = train[train["timestamp"] >= cutoff]
            if len(train) < 500:
                continue
            train["feat_r"] = train.groupby("timestamp")[feat].rank(pct=True) - 0.5
            train["tgt_r"]  = train.groupby("timestamp")[target].rank(pct=True) - 0.5
            for ts, grp in train.groupby("timestamp"):
                valid = grp[["feat_r", "tgt_r"]].dropna()
                if len(valid) >= 8:
                    ic = stats.spearmanr(valid["feat_r"], valid["tgt_r"])[0]
                    if not np.isnan(ic):
                        ics_train.append(ic)

        if ics_train:
            arr = np.array(ics_train)
            mean_ic = arr.mean()
            icir    = mean_ic / (arr.std() + 1e-10)
            results.append({"feature": feat, "mean_ic": mean_ic, "icir": icir,
                            "ic_pos_pct": (arr > 0).mean(), "n": len(arr)})

    results.sort(key=lambda x: abs(x["mean_ic"]), reverse=True)
    for r in results[:20]:
        sign = "+" if r["mean_ic"] > 0 else "-"
        log(f"    {sign} {r['feature']:<38s} IC={r['mean_ic']:+.4f} "
            f"ICIR={r['icir']:+.3f} IC>0={r['ic_pos_pct']*100:.0f}%")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def cs_rank_inplace(df, feats):
    df = df.copy()
    for feat in feats:
        if feat in df.columns:
            df[feat] = df.groupby("timestamp")[feat].rank(pct=True) - 0.5
    return df


def train_lgb(df, feats, seeds=SEEDS, target_col="fwd_ret_12h", params_override=None):
    """LGB 5-seed ensemble with walk-forward splits."""
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
            avail = [f for f in feats if f in df.columns]
            train = cs_rank_inplace(train, avail)
            val   = cs_rank_inplace(val,   avail)
            test  = cs_rank_inplace(test,  avail)
            for d in [train, val, test]:
                d["target_rank"] = d.groupby("timestamp")[target_col].rank(pct=True) - 0.5
            train_c = train[avail + ["target_rank"]].dropna()
            val_c   = val[avail + ["target_rank"]].dropna()
            params = {
                "objective": "regression", "metric": "mse",
                "learning_rate": 0.03, "num_leaves": 63,
                "min_child_samples": 100,
                "subsample": 0.8, "colsample_bytree": 0.8,
                "lambda_l2": 1.0, "seed": seed,
                "verbose": -1, "n_jobs": -1,
            }
            if params_override:
                params.update(params_override)
            dtrain = lgb.Dataset(train_c[avail], label=train_c["target_rank"])
            dval   = lgb.Dataset(val_c[avail],   label=val_c["target_rank"])
            model  = lgb.train(params, dtrain, num_boost_round=600,
                               valid_sets=[dval],
                               callbacks=[lgb.early_stopping(40, verbose=False),
                                          lgb.log_evaluation(-1)])
            test_c = test[avail + ["target_rank", "timestamp", "symbol"]].dropna()
            preds  = model.predict(test_c[avail])
            fwd    = test[["timestamp", "symbol", "fwd_ret_12h"]].rename(
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


def train_catboost(df, feats, seeds=SEEDS, target_col="fwd_ret_12h"):
    """CatBoost 5-seed ensemble."""
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
            avail = [f for f in feats if f in df.columns]
            train = cs_rank_inplace(train, avail)
            val   = cs_rank_inplace(val,   avail)
            test  = cs_rank_inplace(test,  avail)
            for d in [train, val, test]:
                d["target_rank"] = d.groupby("timestamp")[target_col].rank(pct=True) - 0.5
            train_c = train[avail + ["target_rank"]].dropna()
            val_c   = val[avail + ["target_rank"]].dropna()
            model   = CatBoostRegressor(
                iterations=600, learning_rate=0.03,
                depth=6, l2_leaf_reg=1.0,
                subsample=0.8, colsample_bylevel=0.8,
                random_seed=seed, verbose=0,
                early_stopping_rounds=40,
            )
            model.fit(train_c[avail], train_c["target_rank"],
                      eval_set=(val_c[avail], val_c["target_rank"]),
                      verbose=False)
            test_c = test[avail + ["target_rank", "timestamp", "symbol"]].dropna()
            preds  = model.predict(test_c[avail])
            fwd    = test[["timestamp", "symbol", "fwd_ret_12h"]].rename(
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


def ic_analysis(preds, label):
    """Per-window IC."""
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
                f"IC>0={( ic_arr>0).mean()*100:.0f}%")
    if all_ics:
        a = np.array(all_ics)
        log(f"    ALL: IC={a.mean():.4f} +/-{a.std():.4f}")


def run_eval(preds, regime_df, label, cfgs=None):
    """Evaluate a set of predictions under different configs."""
    if preds is None:
        log(f"  ⚠  {label}: no predictions")
        return []
    if cfgs is None:
        cfgs = [("bare", CFG_BARE), ("regime", CFG_REGIME)]

    ic_analysis(preds, label)
    results = []
    for cfg_name, cfg in cfgs:
        port = simulate(preds, regime_df, 12, cfg)
        r    = eval_config(port, 12, f"{label} [{cfg_name}]", LEVERAGE, CAPITAL)
        if r:
            show(r)
            results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log("=" * 80)
    log("  R19 — MODEL IMPROVEMENT ROUND v2 (Leakage-Fixed + New Signals)")
    log("=" * 80)

    # ── Load data ──────────────────────────────────────────────────────────────
    log("\n  Loading base data...")
    ohlcv   = load_ohlcv()
    ohlcv   = ohlcv[ohlcv["symbol"].isin(SYM_35)]
    derivs  = load_derivatives()
    df      = build_features_minimal(ohlcv, derivs)
    regime_df = compute_regime(df)
    log(f"  Base: {len(df):,} rows, {df['symbol'].nunique()} symbols")

    # ── Feature enrichment ────────────────────────────────────────────────────
    log("\n" + "═" * 80)
    log("  FEATURE ENRICHMENT (R19)")
    log("═" * 80)
    df = add_vol_features(df)        # ATR, gk_vol, RSI, bb_pband from TA file
    df = add_iv_rv_spread(df)        # DVOL-derived IV-RV spread + dvol_zscore
    df = add_market_breadth(df)      # pct_coins_up_12h/1h, btc_outperform
    df = add_seasonality(df)         # hour_sin/cos, dow_sin/cos
    df = add_funding_carry(df)       # fund_cum_24h/168h, fund_zscore_24h
    df = add_regime_features(df, regime_df)  # trend_strength/direction/vol_regime

    available = [c for c in df.columns
                 if c not in ("timestamp", "symbol", "close", "open", "high", "low",
                              "volume", "fwd_ret_12h", "fwd_ret_4h", "fwd_ret_24h",
                              "fwd_cls_12h", "target_rank")]
    log(f"\n  Total available features: {len(available)}")

    # ── Phase 1: IC scan (TRAIN ONLY — no leakage) ───────────────────────────
    log("\n" + "═" * 80)
    log("  PHASE 1: In-Sample IC Scan (train data only)")
    log("═" * 80)

    new_candidates = [
        # New R19 features
        "pct_coins_up_12h", "pct_coins_up_1h", "btc_outperform",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "fund_cum_24h", "fund_cum_168h", "fund_zscore_24h",
        "trend_strength", "trend_direction", "vol_regime",
        # TA vol (confirmed in R18 — now checking with train-only IC)
        "atr_14", "rvol_12h", "rvol_24h", "gk_vol_24h", "iv_rv_spread",
        "dvol_zscore", "rsi_14", "bb_pband_20",
        # Existing features (sanity check)
        "ret_12h", "ret_24h", "ret_48h", "residual_12h",
        "mom_z_24h", "oi_chg_12h", "taker_cvd_12h", "ls_divergence",
        "oi_zscore", "taker_cvd_24h", "dist_from_high_24h",
    ]
    ic_results = scan_ic_train_only(df, new_candidates)

    # Features with |IC| > 0.005 and IC>0 in right direction > 52% (or < 48%)
    good_new = [r["feature"] for r in ic_results
                if abs(r["mean_ic"]) > 0.005 and r["n"] >= 50
                and r["feature"] not in FEATURES_17]
    log(f"\n  Good new features (clean train IC): {len(good_new)}")
    for f in good_new[:10]:
        log(f"    {f}")

    all_results = []

    # ── Phase 2: R18 winner verified (no IC scan leakage) ────────────────────
    log("\n" + "═" * 80)
    log("  PHASE 2: R18 Winner Verification (17f-top5, leakage-free)")
    log("═" * 80)
    log("  Note: R18 feature selection was done on test data (bias). Verifying true OOS perf.")

    log("\n  2a: LGB 12f baseline (control)...")
    preds_12f = train_lgb(df, FEATURES_12)
    all_results += run_eval(preds_12f, regime_df, "LGB-12f (control)")

    log("\n  2b: LGB 17f-top5 (R18 winner rerun)...")
    preds_17f = train_lgb(df, FEATURES_17)
    all_results += run_eval(preds_17f, regime_df, "LGB-17f-top5")

    # ── Phase 3: Regime features in model (replaces hard filter) ─────────────
    log("\n" + "═" * 80)
    log("  PHASE 3: Regime Features IN Model (no hard cutoff)")
    log("═" * 80)

    feats_regime_aware = FEATURES_17 + ["trend_strength", "trend_direction", "vol_regime"]
    feats_regime_aware = [f for f in feats_regime_aware if f in df.columns]
    log(f"  3a: LGB 17f + regime features as inputs ({len(feats_regime_aware)}f), BARE-BONES config...")
    preds_regime_feat = train_lgb(df, feats_regime_aware)
    # Only test bare-bones (regime cutoff is now IN model, not post-filter)
    all_results += run_eval(preds_regime_feat, regime_df, "LGB-regime-aware",
                            cfgs=[("bare", CFG_BARE), ("regime", CFG_REGIME)])

    # ── Phase 4: Market breadth + seasonality ────────────────────────────────
    log("\n" + "═" * 80)
    log("  PHASE 4: Market Breadth + Seasonality")
    log("═" * 80)

    breadth_season = ["pct_coins_up_12h", "pct_coins_up_1h",
                      "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    breadth_season = [f for f in breadth_season if f in df.columns]

    log(f"  4a: LGB 17f + breadth + seasonality ({len(FEATURES_17) + len(breadth_season)}f)...")
    feats_bs = FEATURES_17 + breadth_season
    preds_bs = train_lgb(df, feats_bs)
    all_results += run_eval(preds_bs, regime_df, "LGB-17f+breadth+season")

    # ── Phase 5: Funding carry ────────────────────────────────────────────────
    log("\n" + "═" * 80)
    log("  PHASE 5: Funding Carry Signal")
    log("═" * 80)

    funding_feats = [f for f in ["fund_cum_24h", "fund_zscore_24h", "fund_cum_168h"]
                     if f in df.columns]
    if funding_feats:
        log(f"  5a: LGB 17f + funding carry ({len(FEATURES_17 + funding_feats)}f)...")
        feats_fund = FEATURES_17 + funding_feats
        preds_fund = train_lgb(df, feats_fund)
        all_results += run_eval(preds_fund, regime_df, "LGB-17f+funding")
    else:
        log("  5a: No funding features available, skip")

    # ── Phase 6: Best combo (top new signals) ────────────────────────────────
    log("\n" + "═" * 80)
    log("  PHASE 6: Best Combination")
    log("═" * 80)

    # Build best feature set: 17f + top new signals from Phase 1 IC scan
    # Limit to features with confirmed train IC and not too many (overfitting risk)
    top_new = [r["feature"] for r in ic_results
               if abs(r["mean_ic"]) > 0.008 and r["n"] >= 50
               and r["feature"] not in FEATURES_17][:5]
    log(f"  Top new: {top_new}")

    if top_new:
        feats_combo = FEATURES_17 + top_new + \
                      [f for f in funding_feats if f not in FEATURES_17][:1]
        feats_combo = list(dict.fromkeys(feats_combo))  # dedupe
        feats_combo = [f for f in feats_combo if f in df.columns]
        log(f"\n  6a: LGB best combo ({len(feats_combo)}f)...")
        preds_combo = train_lgb(df, feats_combo)
        all_results += run_eval(preds_combo, regime_df, "LGB-combo")

    # Always test 17f + regime_feats + top breadth/season
    key_adds = [f for f in ["trend_strength", "pct_coins_up_12h", "hour_sin", "hour_cos",
                             "iv_rv_spread", "fund_zscore_24h"]
                if f in df.columns and f not in FEATURES_17]
    feats_full = FEATURES_17 + key_adds
    feats_full = list(dict.fromkeys(feats_full))
    log(f"\n  6b: LGB full-enhanced ({len(feats_full)}f)...")
    preds_full = train_lgb(df, feats_full)
    all_results += run_eval(preds_full, regime_df, "LGB-full-enhanced")

    # ── Phase 7: LGB + CatBoost ensemble (17f) ───────────────────────────────
    log("\n" + "═" * 80)
    log("  PHASE 7: LGB + CatBoost Ensemble (2-model, 17f)")
    log("═" * 80)

    log("\n  7a: CatBoost 17f...")
    preds_cb17 = train_catboost(df, FEATURES_17)
    all_results += run_eval(preds_cb17, regime_df, "CB-17f")

    # Average ensemble
    if preds_17f is not None and preds_cb17 is not None:
        log("\n  7b: LGB+CB ensemble (avg)...")
        merged_ens = preds_17f.merge(
            preds_cb17[["timestamp", "symbol", "pred"]].rename(columns={"pred": "pred_cb"}),
            on=["timestamp", "symbol"], how="inner")
        merged_ens["pred"] = 0.5 * merged_ens["pred"] + 0.5 * merged_ens["pred_cb"]
        all_results += run_eval(merged_ens, regime_df, "LGB+CB-17f-avg")

    # ── Phase 8: Best combo with CatBoost ────────────────────────────────────
    if top_new:
        log("\n" + "═" * 80)
        log("  PHASE 8: Best Combo with CatBoost")
        log("═" * 80)
        log(f"\n  8a: CB best combo ({len(feats_combo)}f)...")
        preds_cb_combo = train_catboost(df, feats_combo)
        all_results += run_eval(preds_cb_combo, regime_df, "CB-combo")

        if preds_combo is not None and preds_cb_combo is not None:
            log("\n  8b: LGB+CB combo ensemble...")
            ens2 = preds_combo.merge(
                preds_cb_combo[["timestamp", "symbol", "pred"]].rename(columns={"pred": "pred_cb"}),
                on=["timestamp", "symbol"], how="inner")
            ens2["pred"] = 0.5 * ens2["pred"] + 0.5 * ens2["pred_cb"]
            all_results += run_eval(ens2, regime_df, "LGB+CB-combo-avg")

    # ── Final ranking ─────────────────────────────────────────────────────────
    log("\n" + "═" * 80)
    log("  FINAL RANKINGS")
    log("═" * 80)
    if all_results:
        ranked = sorted(all_results, key=lambda r: -r["sharpe"])
        log(f"\n  By Sharpe ({len(ranked)} configs):")
        for i, r in enumerate(ranked[:20], 1):
            flag = "✅" if r["sharpe"] >= 2.5 else ("⚠️ " if r["sharpe"] >= 1.8 else "❌")
            log(f"    #{i:2d} {flag} {r['name']:<60s} "
                f"Sh={r['sharpe']:+.2f} WM={r['win_months']}/{r['total_months']} "
                f"Wr={r['worst_m']*100:.1f}% Eq=${r['equity']:.0f}")

        # Per-window detail of top-3
        log("\n  Per-window detail (top-3):")
        for r in ranked[:3]:
            log(f"    {r['name']}: Sh={r['sharpe']:.2f}")
            for m in r.get("month_data", []):
                log(f"         {m['month']}    {m['ret']*100:+.1f}%  eq=${m['equity']:>8.0f}")

    elapsed = time.time() - t0
    log(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
