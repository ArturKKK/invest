#!/usr/bin/env python3
"""
R22 — Deep Model Improvement Round (overnight pipeline).

Base: R20-C winner — LGB-23f, cutoff=0.9, 12h rebal, 6L/3S → Sh=2.80, Eq=$2096

Six experiments:
  K: LGB Hyperparameter Optimization (Optuna, 40 trials)
  L: Feature importance + pruned model (drop worst features)
  M: XGBoost baseline (same 23f)
  N: CatBoost baseline (same 23f)
  O: Stacked ensemble (LGB + XGB + CB → Ridge meta-learner)
  P: New features from untapped sources (macro, premium, TA, fear&greed)

All with 12h rebal (no overlap bug), cutoff=0.9.
Monthly equity breakdown for every config.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None
from sklearn.linear_model import Ridge
from scipy import stats
from pathlib import Path
import warnings, time, sys, json
warnings.filterwarnings("ignore")

from _research_round7 import (
    SYM_35, WINDOWS, cs_rank,
    compute_regime, simulate, eval_config, show,
)
from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal

PROJECT = Path(__file__).parent
DATA_DIR = PROJECT / "data"
SENT_DIR = DATA_DIR / "sentiment"

FEATURES_23 = [
    "ret_12h", "ret_24h", "ret_48h",
    "residual_12h", "residual_24h",
    "mom_z_24h",
    "oi_chg_12h", "oi_chg_24h", "oi_zscore",
    "taker_cvd_12h", "taker_cvd_24h",
    "ls_divergence",
    "atr_14", "rvol_12h", "gk_vol_24h", "rvol_24h", "iv_rv_spread",
    "pct_coins_up_12h", "pct_coins_up_1h",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]

SEEDS = [0, 7, 13, 42, 99]
LEVERAGE = 5
CAPITAL  = 100

CFG_BEST = {
    "n_long": 6, "n_short": 3, "trend_cutoff": 0.9,
    "dyn_threshold": 0.5625, "rebal_hours": 12,
    "kelly_sizing": False, "vol_scaling": False, "regime_asym": False,
}


def log(msg):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_r19_features(df):
    try:
        ta = pd.read_parquet(DATA_DIR / "features" / "crypto_features_1h.parquet",
                             columns=["timestamp", "symbol", "atr_14", "gk_vol_24h",
                                      "rsi_14", "bb_pband_20"])
        ta["timestamp"] = pd.to_datetime(ta["timestamp"], utc=True)
        df = df.merge(ta, on=["timestamp", "symbol"], how="left")
        log("  [TA] atr_14, gk_vol_24h")
    except Exception as e:
        log(f"  [TA] Error: {e}")

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
        df["dvol_zscore"] = ((df["btc_dvol"]
                              - df["btc_dvol"].rolling(168 * 4, min_periods=168).mean())
                             / (df["btc_dvol"].rolling(168 * 4, min_periods=168).std() + 1e-10))
        log("  [DVOL] btc_dvol, iv_rv_spread, dvol_zscore")
    except Exception as e:
        log(f"  [DVOL] Error: {e}")

    if "ret_12h" in df.columns:
        breadth = (df.groupby("timestamp")[["ret_12h", "ret_1h"]]
                   .agg(pct_coins_up_12h=("ret_12h", lambda x: (x > 0).mean()),
                        pct_coins_up_1h =("ret_1h",  lambda x: (x > 0).mean()))
                   .reset_index())
        df = df.merge(breadth, on="timestamp", how="left")
        btc_r = df.loc[df["symbol"] == "BTC/USDT", ["timestamp", "ret_12h"]].rename(
            columns={"ret_12h": "btc_ret12_ts"})
        df = df.merge(btc_r, on="timestamp", how="left")
        df["btc_outperform"] = df["ret_12h"] - df["btc_ret12_ts"]
        df.drop(columns=["btc_ret12_ts"], inplace=True, errors="ignore")
        log("  [BREADTH] pct_coins_up_12h/1h, btc_outperform")

    df["hour_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.hour / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["timestamp"].dt.dayofweek / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["timestamp"].dt.dayofweek / 7)
    log("  [SEASON] hour_sin/cos, dow_sin/cos")
    return df


def add_new_features(df):
    """EXP-P: add features from untapped sources."""
    added = []

    # --- Premium ---
    if "premium_index" in df.columns:
        df["premium_zscore_12h"] = (df.groupby("symbol")["premium_index"]
            .transform(lambda x: (x - x.rolling(12, min_periods=6).mean())
                       / (x.rolling(12, min_periods=6).std() + 1e-10)))
        added.append("premium_zscore_12h")

    # --- OI velocity (2nd derivative) ---
    if "oi_chg_12h" in df.columns:
        df["oi_velocity"] = df.groupby("symbol")["oi_chg_12h"].diff()
        added.append("oi_velocity")

    # --- Taker imbalance zscore ---
    if "taker_imbalance" in df.columns:
        df["taker_imb_z"] = (df.groupby("symbol")["taker_imbalance"]
            .transform(lambda x: (x - x.rolling(48, min_periods=12).mean())
                       / (x.rolling(48, min_periods=12).std() + 1e-10)))
        added.append("taker_imb_z")

    # --- Vol-of-vol ---
    if "rvol_24h" in df.columns:
        df["vol_of_vol"] = df.groupby("symbol")["rvol_24h"].transform(
            lambda x: x.rolling(48, min_periods=12).std())
        added.append("vol_of_vol")

    # --- Dist from high ---
    if "dist_from_high_24h" in df.columns:
        added.append("dist_from_high_24h")

    # --- Vol ratio ---
    if "vol_ratio_24h" in df.columns:
        added.append("vol_ratio_24h")

    # --- ret_168h (weekly momentum) ---
    if "ret_168h" in df.columns:
        added.append("ret_168h")

    # --- Fear & Greed ---
    try:
        fng = pd.read_parquet(SENT_DIR / "fear_greed.parquet")
        fng["timestamp"] = pd.to_datetime(fng["timestamp"], utc=True)
        fng = fng[["timestamp", "fng_value"]].drop_duplicates("timestamp")
        fng = fng.set_index("timestamp").resample("1h").ffill().reset_index()
        df = df.merge(fng, on="timestamp", how="left")
        df["fng_value"] = df["fng_value"].ffill()
        df["fng_zscore"] = ((df["fng_value"]
                             - df["fng_value"].rolling(720, min_periods=168).mean())
                            / (df["fng_value"].rolling(720, min_periods=168).std() + 1e-10))
        added.extend(["fng_value", "fng_zscore"])
        log(f"  [FNG] fear_greed → fng_value, fng_zscore")
    except Exception as e:
        log(f"  [FNG] Error: {e}")

    # --- Macro (VIX, DXY) ---
    try:
        macro = pd.read_parquet(SENT_DIR / "macro_daily.parquet")
        date_col = "date" if "date" in macro.columns else "timestamp"
        macro["timestamp"] = pd.to_datetime(macro[date_col], utc=True)
        mcols = []
        if "vix_close" in macro.columns:
            mcols.append("vix_close")
        if "dxy_close" in macro.columns:
            mcols.append("dxy_close")
        if mcols:
            macro = macro[["timestamp"] + mcols].drop_duplicates("timestamp")
            macro = macro.set_index("timestamp").resample("1h").ffill().reset_index()
            df = df.merge(macro, on="timestamp", how="left")
            for c in mcols:
                df[c] = df[c].ffill()
            if "vix_close" in df.columns:
                df["vix_zscore"] = ((df["vix_close"]
                                     - df["vix_close"].rolling(720, min_periods=168).mean())
                                    / (df["vix_close"].rolling(720, min_periods=168).std() + 1e-10))
                added.extend(["vix_close", "vix_zscore"])
            if "dxy_close" in df.columns:
                df["dxy_ret_7d"] = df["dxy_close"].pct_change(168)
                added.extend(["dxy_ret_7d"])
            log(f"  [MACRO] {mcols + ['vix_zscore', 'dxy_ret_7d']}")
    except Exception as e:
        log(f"  [MACRO] Error: {e}")

    # --- Extra TA from crypto_features_1h ---
    try:
        extra_ta_cols = ["rsi_14", "bb_pband_20", "adx", "mfi_14",
                         "ret_skew_24h", "ret_kurt_24h",
                         "vwap_dev_24h", "obv_ma_ratio_24"]
        ta = pd.read_parquet(DATA_DIR / "features" / "crypto_features_1h.parquet",
                             columns=["timestamp", "symbol"] + extra_ta_cols)
        ta["timestamp"] = pd.to_datetime(ta["timestamp"], utc=True)
        existing = [c for c in extra_ta_cols if c in df.columns]
        new_ta = [c for c in extra_ta_cols if c not in df.columns]
        if new_ta:
            df = df.merge(ta[["timestamp", "symbol"] + new_ta], on=["timestamp", "symbol"], how="left")
            added.extend(new_ta)
            log(f"  [TA-extra] {new_ta}")
    except Exception as e:
        log(f"  [TA-extra] Error: {e}")

    added = [f for f in added if f in df.columns]
    log(f"  New features added: {len(added)}: {added}")
    return df, added


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def cs_rank_cols(df, feats):
    df = df.copy()
    for f in feats:
        if f in df.columns:
            df[f] = df.groupby("timestamp")[f].rank(pct=True) - 0.5
    return df


def train_lgb(df, feats, seeds=SEEDS, target_col="fwd_ret_12h", params_override=None):
    avail = [f for f in feats if f in df.columns]
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
            train = cs_rank_cols(train, avail)
            val   = cs_rank_cols(val,   avail)
            test  = cs_rank_cols(test,  avail)
            for d in [train, val, test]:
                if target_col in d.columns:
                    d["target_rank"] = d.groupby("timestamp")[target_col].rank(pct=True) - 0.5
            train_c = train[avail + ["target_rank"]].dropna()
            val_c   = val[avail + ["target_rank"]].dropna()
            params = {
                "objective": "regression", "metric": "mse",
                "learning_rate": 0.03, "num_leaves": 63,
                "min_child_samples": 100, "subsample": 0.8,
                "colsample_bytree": 0.8, "lambda_l2": 1.0,
                "seed": seed, "verbose": -1, "n_jobs": -1,
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
            if len(test_c) == 0:
                continue
            preds  = model.predict(test_c[avail])
            fwd    = test[["timestamp", "symbol", target_col]].rename(
                     columns={target_col: "fwd_ret"}).dropna()
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


def train_xgb(df, feats, seeds=SEEDS, target_col="fwd_ret_12h", params_override=None):
    avail = [f for f in feats if f in df.columns]
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
            train = cs_rank_cols(train, avail)
            val   = cs_rank_cols(val,   avail)
            test  = cs_rank_cols(test,  avail)
            for d in [train, val, test]:
                if target_col in d.columns:
                    d["target_rank"] = d.groupby("timestamp")[target_col].rank(pct=True) - 0.5
            train_c = train[avail + ["target_rank"]].dropna()
            val_c   = val[avail + ["target_rank"]].dropna()
            params = {
                "objective": "reg:squarederror",
                "learning_rate": 0.03, "max_depth": 6,
                "min_child_weight": 100, "subsample": 0.8,
                "colsample_bytree": 0.8, "reg_lambda": 1.0,
                "seed": seed, "n_jobs": -1, "verbosity": 0,
            }
            if params_override:
                params.update(params_override)
            dtrain = xgb.DMatrix(train_c[avail], label=train_c["target_rank"])
            dval   = xgb.DMatrix(val_c[avail],   label=val_c["target_rank"])
            model  = xgb.train(params, dtrain, num_boost_round=600,
                               evals=[(dval, "val")],
                               early_stopping_rounds=40, verbose_eval=False)
            test_c = test[avail + ["target_rank", "timestamp", "symbol"]].dropna()
            if len(test_c) == 0:
                continue
            dtest = xgb.DMatrix(test_c[avail])
            preds  = model.predict(dtest)
            fwd    = test[["timestamp", "symbol", target_col]].rename(
                     columns={target_col: "fwd_ret"}).dropna()
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


def train_catboost(df, feats, seeds=SEEDS, target_col="fwd_ret_12h", params_override=None):
    avail = [f for f in feats if f in df.columns]
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
            train = cs_rank_cols(train, avail)
            val   = cs_rank_cols(val,   avail)
            test  = cs_rank_cols(test,  avail)
            for d in [train, val, test]:
                if target_col in d.columns:
                    d["target_rank"] = d.groupby("timestamp")[target_col].rank(pct=True) - 0.5
            train_c = train[avail + ["target_rank"]].dropna()
            val_c   = val[avail + ["target_rank"]].dropna()
            params = {
                "loss_function": "RMSE",
                "learning_rate": 0.03, "depth": 6,
                "l2_leaf_reg": 3.0, "subsample": 0.8,
                "random_seed": seed, "verbose": 0,
                "iterations": 600, "early_stopping_rounds": 40,
            }
            if params_override:
                params.update(params_override)
            model = cb.CatBoostRegressor(**params)
            model.fit(train_c[avail], train_c["target_rank"],
                      eval_set=(val_c[avail], val_c["target_rank"]),
                      verbose=0)
            test_c = test[avail + ["target_rank", "timestamp", "symbol"]].dropna()
            if len(test_c) == 0:
                continue
            preds  = model.predict(test_c[avail])
            fwd    = test[["timestamp", "symbol", target_col]].rename(
                     columns={target_col: "fwd_ret"}).dropna()
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


def ic_quick(preds, label=""):
    ics = []
    for wname in ["W1", "W2", "W3"]:
        sub = preds[preds["window"] == wname]
        w_ics = []
        for ts, grp in sub.groupby("timestamp"):
            if len(grp) >= 10:
                ic = stats.spearmanr(grp["pred"], grp["fwd_ret"])[0]
                if not np.isnan(ic):
                    w_ics.append(ic)
        if w_ics:
            a = np.array(w_ics)
            ics.extend(w_ics)
            log(f"    {wname}: IC={a.mean():.4f} IC>0={(a>0).mean()*100:.0f}%")
    if ics:
        a = np.array(ics)
        log(f"    ALL: IC={a.mean():.4f} +/-{a.std():.4f}")
    return np.mean(ics) if ics else 0


def run_eval(preds, regime_df, label, cfg=None, verbose_months=True):
    if preds is None:
        log(f"  ⚠  {label}: no predictions")
        return None
    if cfg is None:
        cfg = CFG_BEST
    port = simulate(preds, regime_df, 12, cfg)
    if port is None:
        log(f"  ⚠  {label}: simulate returned None")
        return None
    r = eval_config(port, 12, label, LEVERAGE, CAPITAL)
    if r:
        show(r)
        if verbose_months:
            for m in r.get("month_data", []):
                log(f"       {m['month']}   {m['ret']*100:+.1f}%  eq=${m['equity']:>8.0f}")
    return r


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-K: Optuna HPO for LGB
# ═══════════════════════════════════════════════════════════════════════════════

def exp_k(df, regime_df, n_trials=40):
    log("\n" + "=" * 80)
    log("  EXP-K: LGB Hyperparameter Optimization (Optuna, n=%d)" % n_trials)
    log("=" * 80)

    avail = [f for f in FEATURES_23 if f in df.columns]

    # Quick evaluation: train with 2 seeds (faster) per trial
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("lr", 0.005, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 30, 300),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.01, 10.0, log=True),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
            "max_depth": trial.suggest_int("max_depth", -1, 10),
        }
        t_trial = time.time()
        preds = train_lgb(df, avail, seeds=[0, 42], params_override=params)
        if preds is None:
            log(f"    trial {trial.number}: FAILED (no preds), {time.time()-t_trial:.0f}s")
            return -10
        port = simulate(preds, regime_df, 12, CFG_BEST)
        r = eval_config(port, 12, "hpo", LEVERAGE, CAPITAL)
        if r is None:
            log(f"    trial {trial.number}: FAILED (no eval), {time.time()-t_trial:.0f}s")
            return -10
        log(f"    trial {trial.number}: Sh={r['sharpe']:.2f} Eq=${r['equity']:.0f} ({time.time()-t_trial:.0f}s)")
        return r["sharpe"]

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)

    log(f"\n  Best trial: Sh={study.best_value:.2f}")
    log(f"  Best params: {json.dumps(study.best_params, indent=4)}")

    # Retrain best with all 5 seeds
    log(f"\n  Retraining best HPO with 5 seeds...")
    best = study.best_params
    best_params = {
        "learning_rate": best["lr"],
        "num_leaves": best["num_leaves"],
        "min_child_samples": best["min_child_samples"],
        "subsample": best["subsample"],
        "colsample_bytree": best["colsample_bytree"],
        "lambda_l2": best["lambda_l2"],
        "lambda_l1": best["lambda_l1"],
        "max_depth": best["max_depth"],
    }
    preds_hpo = train_lgb(df, avail, seeds=SEEDS, params_override=best_params)
    ic = ic_quick(preds_hpo, "LGB-HPO")
    r = run_eval(preds_hpo, regime_df, "K-LGB-HPO-23f")

    # Compare with default
    log("\n  Control (default params):")
    preds_ctrl = train_lgb(df, avail)
    r_ctrl = run_eval(preds_ctrl, regime_df, "K-LGB-default-23f")

    return preds_hpo, best_params, r


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-L: Feature importance + pruning
# ═══════════════════════════════════════════════════════════════════════════════

def exp_l(df, regime_df):
    log("\n" + "=" * 80)
    log("  EXP-L: Feature Importance & Pruning")
    log("=" * 80)

    avail = [f for f in FEATURES_23 if f in df.columns]
    tz = df["timestamp"].dt.tz

    # Collect feature importances across all models
    importances = {f: 0 for f in avail}
    n_models = 0
    for seed in SEEDS:
        for w in WINDOWS:
            train = df[df["timestamp"] < pd.Timestamp(w["train_end"], tz=tz)].copy()
            val   = df[(df["timestamp"] >= pd.Timestamp(w["val_start"], tz=tz)) &
                       (df["timestamp"] <  pd.Timestamp(w["val_end"],   tz=tz))].copy()
            if len(train) < 5000:
                continue
            train = cs_rank_cols(train, avail)
            val   = cs_rank_cols(val, avail)
            for d in [train, val]:
                d["target_rank"] = d.groupby("timestamp")["fwd_ret_12h"].rank(pct=True) - 0.5
            train_c = train[avail + ["target_rank"]].dropna()
            val_c   = val[avail + ["target_rank"]].dropna()
            dtrain = lgb.Dataset(train_c[avail], label=train_c["target_rank"])
            dval   = lgb.Dataset(val_c[avail],   label=val_c["target_rank"])
            model  = lgb.train(
                {"objective": "regression", "metric": "mse",
                 "learning_rate": 0.03, "num_leaves": 63,
                 "min_child_samples": 100, "subsample": 0.8,
                 "colsample_bytree": 0.8, "lambda_l2": 1.0,
                 "seed": seed, "verbose": -1, "n_jobs": -1},
                dtrain, num_boost_round=600, valid_sets=[dval],
                callbacks=[lgb.early_stopping(40, verbose=False),
                           lgb.log_evaluation(-1)])
            imp = model.feature_importance(importance_type="gain")
            for f, v in zip(avail, imp):
                importances[f] += v
            n_models += 1

    # Normalize
    for f in importances:
        importances[f] /= max(n_models, 1)

    ranked = sorted(importances.items(), key=lambda x: -x[1])
    log("\n  Feature importance (avg gain across 15 models):")
    for i, (f, v) in enumerate(ranked, 1):
        bar = "█" * max(1, int(v / ranked[0][1] * 30))
        log(f"    {i:2d}. {f:<22s} {v:>8.1f}  {bar}")

    # Try dropping bottom N features
    results = []
    for drop_n in [0, 3, 5, 7, 10]:
        if drop_n == 0:
            feats_pruned = avail
        else:
            feats_pruned = [f for f, _ in ranked[:-drop_n]]
        preds = train_lgb(df, feats_pruned)
        r = run_eval(preds, regime_df, f"L-prune-{len(feats_pruned)}f", verbose_months=(drop_n == 0))
        if r:
            results.append((len(feats_pruned), r))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-M: XGBoost baseline
# ═══════════════════════════════════════════════════════════════════════════════

def exp_m(df, regime_df):
    log("\n" + "=" * 80)
    log("  EXP-M: XGBoost Baseline (23f)")
    log("=" * 80)

    avail = [f for f in FEATURES_23 if f in df.columns]
    preds = train_xgb(df, avail)
    ic = ic_quick(preds, "XGB-23f")
    r = run_eval(preds, regime_df, "M-XGB-23f")
    return preds, r


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-N: CatBoost baseline
# ═══════════════════════════════════════════════════════════════════════════════

def exp_n(df, regime_df):
    log("\n" + "=" * 80)
    log("  EXP-N: CatBoost Baseline (23f)")
    log("=" * 80)

    avail = [f for f in FEATURES_23 if f in df.columns]
    preds = train_catboost(df, avail)
    ic = ic_quick(preds, "CB-23f")
    r = run_eval(preds, regime_df, "N-CB-23f")
    return preds, r


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-O: Stacked Ensemble (LGB + XGB + CB → Ridge)
# ═══════════════════════════════════════════════════════════════════════════════

def exp_o(preds_lgb, preds_xgb, preds_cb, regime_df):
    log("\n" + "=" * 80)
    log("  EXP-O: Stacked Ensemble (LGB + XGB + CB → Ridge meta)")
    log("=" * 80)

    if preds_lgb is None or preds_xgb is None or preds_cb is None:
        log("  ⚠  Missing predictions, skipping")
        return None

    # Rank-average ensemble (simple)
    ens = preds_lgb[["timestamp", "symbol", "fwd_ret", "window"]].copy()
    ens = ens.merge(preds_lgb[["timestamp", "symbol", "pred"]].rename(
        columns={"pred": "pred_lgb"}), on=["timestamp", "symbol"])
    ens = ens.merge(preds_xgb[["timestamp", "symbol", "pred"]].rename(
        columns={"pred": "pred_xgb"}), on=["timestamp", "symbol"], how="inner")
    ens = ens.merge(preds_cb[["timestamp", "symbol", "pred"]].rename(
        columns={"pred": "pred_cb"}), on=["timestamp", "symbol"], how="inner")

    log(f"  Ensemble size: {len(ens)} rows ({ens['timestamp'].nunique()} timestamps)")

    # 1. Simple average
    ens["pred"] = (ens["pred_lgb"] + ens["pred_xgb"] + ens["pred_cb"]) / 3
    r_avg = run_eval(ens, regime_df, "O-avg-ensemble", verbose_months=True)

    # 2. Rank-then-average
    ens_rank = ens.copy()
    for col in ["pred_lgb", "pred_xgb", "pred_cb"]:
        ens_rank[col] = ens_rank[col].astype(np.float64)
    for col in ["pred_lgb", "pred_xgb", "pred_cb"]:
        ens_rank[col] = ens_rank.groupby("timestamp")[col].rank(pct=True) - 0.5
    ens_rank["pred"] = (ens_rank["pred_lgb"] + ens_rank["pred_xgb"] + ens_rank["pred_cb"]) / 3
    r_rank = run_eval(ens_rank, regime_df, "O-rank-ensemble", verbose_months=True)

    # 3. Ridge stacking (OOS: train on W1 test W2, train on W1+W2 test W3)
    log("\n  Ridge stacking (walk-forward)...")
    stacked = ens.copy()
    for col in ["pred_lgb", "pred_xgb", "pred_cb", "fwd_ret"]:
        stacked[col] = stacked[col].astype(np.float64)
    stacked["pred_meta"] = np.nan

    # W1 → train meta → predict W2
    w1 = stacked[stacked["window"] == "W1"]
    w2 = stacked[stacked["window"] == "W2"]
    w3 = stacked[stacked["window"] == "W3"]

    if len(w1) > 100 and len(w2) > 100:
        ridge1 = Ridge(alpha=1.0)
        X1 = w1[["pred_lgb", "pred_xgb", "pred_cb"]].values
        y1 = w1["fwd_ret"].values
        ridge1.fit(X1, y1)
        X2 = w2[["pred_lgb", "pred_xgb", "pred_cb"]].values
        stacked.loc[w2.index, "pred_meta"] = ridge1.predict(X2)
        log(f"    Ridge W1→W2: coef={ridge1.coef_}")

    if len(w1) > 100 and len(w2) > 100 and len(w3) > 100:
        w12 = pd.concat([w1, w2])
        ridge2 = Ridge(alpha=1.0)
        X12 = w12[["pred_lgb", "pred_xgb", "pred_cb"]].values
        y12 = w12["fwd_ret"].values
        ridge2.fit(X12, y12)
        X3 = w3[["pred_lgb", "pred_xgb", "pred_cb"]].values
        stacked.loc[w3.index, "pred_meta"] = ridge2.predict(X3)
        log(f"    Ridge W1+W2→W3: coef={ridge2.coef_}")

    stacked_valid = stacked.dropna(subset=["pred_meta"])
    if len(stacked_valid) > 100:
        stacked_valid = stacked_valid.rename(columns={"pred_meta": "pred_tmp"})
        stacked_valid["pred"] = stacked_valid["pred_tmp"]
        r_ridge = run_eval(stacked_valid, regime_df, "O-ridge-stack", verbose_months=True)
    else:
        r_ridge = None
        log("  ⚠  Ridge stacking: not enough OOS data")

    return r_avg, r_rank, r_ridge


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-P: New features from untapped sources
# ═══════════════════════════════════════════════════════════════════════════════

def exp_p(df, regime_df, new_feats):
    log("\n" + "=" * 80)
    log("  EXP-P: New Features from Untapped Sources")
    log("=" * 80)

    if not new_feats:
        log("  No new features available, skipping")
        return None

    avail_23 = [f for f in FEATURES_23 if f in df.columns]
    avail_new = [f for f in new_feats if f in df.columns]
    coverage = {f: df[f].notna().mean() for f in avail_new}
    log(f"  New feature coverage:")
    for f, cov in sorted(coverage.items(), key=lambda x: -x[1]):
        log(f"    {f:<25s} {cov*100:.1f}%")

    # Filter good coverage (>50%)
    good_new = [f for f in avail_new if coverage[f] > 0.5]
    log(f"\n  Good coverage (>50%): {len(good_new)} features")

    # Test incrementally
    results = []

    # Base 23f control
    log("\n  P0: Control (23f baseline)...")
    preds0 = train_lgb(df, avail_23)
    r0 = run_eval(preds0, regime_df, "P-ctrl-23f")
    if r0:
        results.append(("23f", r0))

    # 23f + all new
    if good_new:
        feats_all = avail_23 + good_new
        log(f"\n  P1: 23f + {len(good_new)} new features = {len(feats_all)}f...")
        preds1 = train_lgb(df, feats_all)
        ic1 = ic_quick(preds1, f"LGB-{len(feats_all)}f")
        r1 = run_eval(preds1, regime_df, f"P-all-{len(feats_all)}f")
        if r1:
            results.append((f"{len(feats_all)}f", r1))

    # Try feature groups individually
    groups = {
        "macro": [f for f in good_new if f in ["vix_close", "vix_zscore", "dxy_ret_7d"]],
        "fng": [f for f in good_new if f in ["fng_value", "fng_zscore"]],
        "deriv": [f for f in good_new if f in ["premium_zscore_12h", "oi_velocity", "taker_imb_z"]],
        "vol": [f for f in good_new if f in ["vol_of_vol", "vol_ratio_24h"]],
        "ta": [f for f in good_new if f in ["rsi_14", "bb_pband_20", "adx", "mfi_14",
                                              "ret_skew_24h", "ret_kurt_24h",
                                              "vwap_dev_24h", "obv_ma_ratio_24"]],
        "mom": [f for f in good_new if f in ["dist_from_high_24h", "ret_168h"]],
    }
    for gname, gfeats in groups.items():
        if not gfeats:
            continue
        feats_g = avail_23 + gfeats
        log(f"\n  P-{gname}: 23f + {gfeats} = {len(feats_g)}f...")
        preds_g = train_lgb(df, feats_g)
        r_g = run_eval(preds_g, regime_df, f"P-{gname}-{len(feats_g)}f")
        if r_g:
            results.append((gname, r_g))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log("=" * 80)
    log("  R22 — DEEP MODEL IMPROVEMENT (overnight pipeline)")
    log("=" * 80)
    log("  Base: R20-C winner — LGB-23f, cutoff=0.9, 12h rebal → Sh=2.80")
    log("  Experiments: K(HPO) L(prune) M(XGB) N(CB) O(stack) P(new feats)")

    # ── Load data + build features ───────────────────────────────────────────
    log("\n  Loading data...")
    ohlcv    = load_ohlcv()
    ohlcv    = ohlcv[ohlcv["symbol"].isin(SYM_35)]
    derivs   = load_derivatives()
    df       = build_features_minimal(ohlcv, derivs)
    regime_df = compute_regime(df)
    log(f"  Base: {len(df):,} rows, {df['symbol'].nunique()} symbols")

    log("\n  Building R19 features...")
    df = build_r19_features(df)
    avail_23 = [f for f in FEATURES_23 if f in df.columns]
    log(f"  FEATURES_23 availability: {len(avail_23)}/{len(FEATURES_23)}")

    # ── Add new features for EXP-P ──────────────────────────────────────────
    log("\n  Adding new features for EXP-P...")
    df, new_feats = add_new_features(df)

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-K: HPO
    # ══════════════════════════════════════════════════════════════════════════
    preds_hpo, best_params, r_k = exp_k(df, regime_df, n_trials=20)

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-L: Feature pruning
    # ══════════════════════════════════════════════════════════════════════════
    results_l = exp_l(df, regime_df)

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-M: XGBoost
    # ══════════════════════════════════════════════════════════════════════════
    preds_xgb, r_m = exp_m(df, regime_df)

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-N: CatBoost
    # ══════════════════════════════════════════════════════════════════════════
    preds_cb, r_n = exp_n(df, regime_df)

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-O: Stacked ensemble (uses LGB default + XGB + CB)
    # ══════════════════════════════════════════════════════════════════════════
    preds_lgb_ctrl = train_lgb(df, avail_23)
    r_o = exp_o(preds_lgb_ctrl, preds_xgb, preds_cb, regime_df)

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-P: New features
    # ══════════════════════════════════════════════════════════════════════════
    results_p = exp_p(df, regime_df, new_feats)

    # ══════════════════════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 80)
    log("  FINAL RANKINGS — R22 ALL EXPERIMENTS")
    log("=" * 80)

    all_results = []
    if r_k:
        all_results.append(r_k)
    if results_l:
        for _, r in results_l:
            if r:
                all_results.append(r)
    if r_m:
        all_results.append(r_m)
    if r_n:
        all_results.append(r_n)
    if r_o:
        for r in r_o:
            if r:
                all_results.append(r)
    if results_p:
        for _, r in results_p:
            if r:
                all_results.append(r)

    if all_results:
        ranked = sorted(all_results, key=lambda r: -r["sharpe"])
        for i, r in enumerate(ranked, 1):
            flag = "✅" if r["sharpe"] >= 2.80 else ("⚠️" if r["sharpe"] >= 2.50 else "❌")
            log(f"  #{i:2d} {flag} {r['name']:<50s} "
                f"Sh={r['sharpe']:+.2f} WM={r['win_months']}/{r['total_months']} "
                f"Wr={r['worst_m']*100:+.1f}% Eq=${r['equity']:.0f}")

    log(f"\n  R20-C baseline: Sh=2.80, Eq=$2096")

    if best_params:
        log(f"\n  Best LGB params from HPO:")
        for k, v in best_params.items():
            log(f"    {k}: {v}")

    elapsed = time.time() - t0
    log(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
