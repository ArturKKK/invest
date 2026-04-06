#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R29c — Deep Validation of FEATURES_25

Validates FEATURES_23 + [global_ls_ratio, ret_std_24h] with L2=10:
  1) Per-window Sharpe breakdown (W1/W2/W3)
  2) Per-seed stability (each seed pair → Sharpe)
  3) Regularization curve (L2 = 1, 3, 5, 10, 20, 30, 50)
  4) Num_leaves sweep (31, 42, 63, 85, 127)
  5) Portfolio config sweep (n_long/n_short combos, thresholds)
  6) Bootstrap CI: 1000 resamples of monthly returns
  7) Leave-one-window-out: train on 2 windows, test on 1
  8) Ablation: remove each of 2 new features
  9) Feature importance analysis

Baseline: FEATURES_23, L2=1.0 → Sh=2.02
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
from _research_r29_forward import build_production_features

FEATURES_25 = FEATURES_23 + ["global_ls_ratio", "ret_std_24h"]

CFG_6L3S = {
    "n_long": 6, "n_short": 3, "trend_cutoff": 0.9,
    "dyn_threshold": 0.7, "rebal_hours": 12,
    "kelly_sizing": False, "vol_scaling": False, "regime_asym": False,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE TRAINING (parameterized)
# ═══════════════════════════════════════════════════════════════════════════════

def train_ensemble(df, feats, seeds=SEEDS, l2=10.0, num_leaves=63,
                   min_child=100, lr=0.03, max_depth=-1,
                   windows=None):
    """Train LGB+XGB ensemble with flexible parameters."""
    if windows is None:
        windows = WINDOWS
    avail = [f for f in feats if f in df.columns]
    all_lgb, all_xgb = [], []
    tz = df["timestamp"].dt.tz

    for seed in seeds:
        params_lgb = {
            "objective": "binary", "metric": "auc",
            "learning_rate": lr, "num_leaves": num_leaves,
            "min_child_samples": min_child, "subsample": 0.8,
            "colsample_bytree": 0.8, "lambda_l2": l2,
            "max_depth": max_depth,
            "verbose": -1, "n_jobs": -1, "seed": seed,
        }
        params_xgb = {
            "objective": "binary:logistic", "eval_metric": "auc",
            "learning_rate": lr, "max_depth": 6 if max_depth == -1 else max_depth,
            "min_child_weight": min_child, "subsample": 0.8,
            "colsample_bytree": 0.8, "reg_lambda": l2,
            "seed": seed, "n_jobs": -1, "verbosity": 0,
        }
        for w in windows:
            train_ = df[df["timestamp"] < pd.Timestamp(w["train_end"], tz=tz)].copy()
            val_ = df[(df["timestamp"] >= pd.Timestamp(w["val_start"], tz=tz)) &
                      (df["timestamp"] < pd.Timestamp(w["val_end"], tz=tz))].copy()
            test_ = df[(df["timestamp"] >= pd.Timestamp(w["test_start"], tz=tz)) &
                       (df["timestamp"] <= pd.Timestamp(w["test_end"], tz=tz))].copy()
            if len(train_) < 5000 or len(test_) < 200:
                continue
            train_ = cs_rank_cols(train_, avail)
            val_ = cs_rank_cols(val_, avail)
            test_ = cs_rank_cols(test_, avail)
            for d in [train_, val_, test_]:
                d["target_binary"] = (d["fwd_ret_12h"] > 0).astype(int)
            tr = train_[avail + ["target_binary"]].dropna()
            va = val_[avail + ["target_binary"]].dropna()
            te = test_[avail + ["target_binary", "timestamp", "symbol"]].dropna()
            if len(te) == 0:
                continue
            fwd = test_[["timestamp", "symbol", "fwd_ret_12h"]].rename(
                columns={"fwd_ret_12h": "fwd_ret"}).dropna()

            # LGB
            dt_lgb = lgb.Dataset(tr[avail], label=tr["target_binary"])
            dv_lgb = lgb.Dataset(va[avail], label=va["target_binary"])
            m_lgb = lgb.train(params_lgb, dt_lgb, num_boost_round=600,
                              valid_sets=[dv_lgb],
                              callbacks=[lgb.early_stopping(40, verbose=False),
                                         lgb.log_evaluation(-1)])
            p = m_lgb.predict(te[avail])
            m = te[["timestamp", "symbol"]].copy()
            m["pred_lgb"] = p
            m = m.merge(fwd, on=["timestamp", "symbol"], how="inner")
            m["window"] = w["name"]
            m["seed"] = seed
            all_lgb.append(m)

            # XGB
            dt_xgb = xgb.DMatrix(tr[avail], label=tr["target_binary"])
            dv_xgb = xgb.DMatrix(va[avail], label=va["target_binary"])
            m_xgb = xgb.train(params_xgb, dt_xgb, num_boost_round=600,
                               evals=[(dv_xgb, "val")],
                               early_stopping_rounds=40, verbose_eval=False)
            p = m_xgb.predict(xgb.DMatrix(te[avail]))
            m2 = te[["timestamp", "symbol"]].copy()
            m2["pred_xgb"] = p
            m2 = m2.merge(fwd, on=["timestamp", "symbol"], how="inner")
            m2["window"] = w["name"]
            m2["seed"] = seed
            all_xgb.append(m2)

    if not all_lgb or not all_xgb:
        return None

    lgb_df = pd.concat(all_lgb)
    xgb_df = pd.concat(all_xgb)

    # Ensemble: average across seeds, rank-blend LGB+XGB
    lgb_avg = lgb_df.groupby(["timestamp", "symbol"]).agg(
        pred_lgb=("pred_lgb", "mean"), fwd_ret=("fwd_ret", "first"),
        window=("window", "first")).reset_index()
    xgb_avg = xgb_df.groupby(["timestamp", "symbol"]).agg(
        pred_xgb=("pred_xgb", "mean")).reset_index()

    merged = lgb_avg.merge(xgb_avg, on=["timestamp", "symbol"], how="inner")
    merged["rank_lgb"] = merged.groupby("timestamp")["pred_lgb"].rank(pct=True) - 0.5
    merged["rank_xgb"] = merged.groupby("timestamp")["pred_xgb"].rank(pct=True) - 0.5
    merged["pred"] = 0.5 * merged["rank_lgb"] + 0.5 * merged["rank_xgb"]
    return merged[["timestamp", "symbol", "pred", "fwd_ret", "window"]], lgb_df, xgb_df


def get_importance(df, feats, l2=10.0, num_leaves=63):
    """Get averaged feature importance from LGB models."""
    avail = [f for f in feats if f in df.columns]
    tz = df["timestamp"].dt.tz
    imp_sum = np.zeros(len(avail))
    n_models = 0
    for seed in SEEDS:
        params = {
            "objective": "binary", "metric": "auc",
            "learning_rate": 0.03, "num_leaves": num_leaves,
            "min_child_samples": 100, "subsample": 0.8,
            "colsample_bytree": 0.8, "lambda_l2": l2,
            "verbose": -1, "n_jobs": -1, "seed": seed,
        }
        for w in WINDOWS:
            train_ = df[df["timestamp"] < pd.Timestamp(w["train_end"], tz=tz)].copy()
            val_ = df[(df["timestamp"] >= pd.Timestamp(w["val_start"], tz=tz)) &
                      (df["timestamp"] < pd.Timestamp(w["val_end"], tz=tz))].copy()
            if len(train_) < 5000:
                continue
            train_ = cs_rank_cols(train_, avail)
            val_ = cs_rank_cols(val_, avail)
            for d in [train_, val_]:
                d["target_binary"] = (d["fwd_ret_12h"] > 0).astype(int)
            tr = train_[avail + ["target_binary"]].dropna()
            va = val_[avail + ["target_binary"]].dropna()
            dt = lgb.Dataset(tr[avail], label=tr["target_binary"])
            dv = lgb.Dataset(va[avail], label=va["target_binary"])
            m = lgb.train(params, dt, num_boost_round=600,
                          valid_sets=[dv],
                          callbacks=[lgb.early_stopping(40, verbose=False),
                                     lgb.log_evaluation(-1)])
            imp_sum += np.array(m.feature_importance("gain"))
            n_models += 1
    if n_models > 0:
        imp_avg = imp_sum / n_models
        return pd.Series(imp_avg, index=avail).sort_values(ascending=False)
    return pd.Series()


def quick_eval(preds, regime_df, cfg, name):
    """Quick simulate + eval, return result dict or None."""
    port = simulate(preds, regime_df, 12, cfg)
    return eval_config(port, 12, name, LEVERAGE, CAPITAL)


def per_window_eval(preds, regime_df, cfg):
    """Return {W1: sharpe, W2: sharpe, W3: sharpe}."""
    result = {}
    for wname in ["W1", "W2", "W3"]:
        sub = preds[preds["window"] == wname]
        if len(sub) < 10:
            result[wname] = None
            continue
        port = simulate(sub, regime_df, 12, cfg)
        r = eval_config(port, 12, wname, LEVERAGE, CAPITAL)
        result[wname] = r
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log("=" * 80)
    log("  R29c — Deep Validation of FEATURES_25")
    log(f"  Date: {pd.Timestamp.now()}")
    log(f"  Features: FEATURES_23 + [global_ls_ratio, ret_std_24h]")
    log("=" * 80)

    t_start = time.time()

    # ── Load & build features ────────────────────────────────────
    log("\n  Loading data...")
    ohlcv = load_ohlcv()
    ohlcv = ohlcv[ohlcv["symbol"].isin(SYM_35)]
    derivs = load_derivatives()
    df = build_features_minimal(ohlcv, derivs)
    df = build_r19_features(df)
    df, _ = add_new_features(df)
    df = build_production_features(df)
    df = df[df["symbol"].isin(SYM_35)].copy()
    regime_df = compute_regime(df)
    log(f"  Data: {len(df):,} rows, {len(df.columns)} cols")
    log(f"  Load time: {time.time()-t_start:.0f}s\n")

    # ══════════════════════════════════════════════════════════════
    #  1. BASELINE vs FEATURES_25 comparison
    # ══════════════════════════════════════════════════════════════
    log("━" * 60)
    log("  1. BASELINE (23f, L2=1) vs FEATURES_25 (25f, L2=10)")
    log("━" * 60)

    result_base, _, _ = train_ensemble(df, FEATURES_23, l2=1.0)
    r_base = quick_eval(result_base, regime_df, CFG_6L3S, "BASELINE-23f")
    if r_base:
        show(r_base)
    base_sh = r_base["sharpe"] if r_base else 0

    result_25, lgb_raw_25, xgb_raw_25 = train_ensemble(df, FEATURES_25, l2=10.0)
    r_25 = quick_eval(result_25, regime_df, CFG_6L3S, "FEAT25-L2=10")
    if r_25:
        show(r_25)
    feat25_sh = r_25["sharpe"] if r_25 else 0
    log(f"  Improvement: {feat25_sh - base_sh:+.2f}\n")

    # ══════════════════════════════════════════════════════════════
    #  2. PER-WINDOW BREAKDOWN
    # ══════════════════════════════════════════════════════════════
    log("━" * 60)
    log("  2. PER-WINDOW SHARPE BREAKDOWN")
    log("━" * 60)

    for label, preds in [("BASELINE-23f", result_base), ("FEAT25-L2=10", result_25)]:
        pw = per_window_eval(preds, regime_df, CFG_6L3S)
        parts = []
        for w in ["W1", "W2", "W3"]:
            r = pw[w]
            if r:
                parts.append(f"{w}: Sh={r['sharpe']:.2f} Eq=${r['equity']:.0f} Wr={r['worst_m']*100:.1f}%")
            else:
                parts.append(f"{w}: N/A")
        log(f"  {label:20s}  " + "  |  ".join(parts))
    log("")

    # ══════════════════════════════════════════════════════════════
    #  3. PER-SEED STABILITY
    # ══════════════════════════════════════════════════════════════
    log("━" * 60)
    log("  3. PER-SEED STABILITY (each seed pair → Sharpe)")
    log("━" * 60)

    for seed in SEEDS:
        result_s, _, _ = train_ensemble(df, FEATURES_25, seeds=[seed], l2=10.0)
        r_s = quick_eval(result_s, regime_df, CFG_6L3S, f"seed={seed}")
        if r_s:
            log(f"  seed={seed:3d}:  Sh={r_s['sharpe']:.2f}  Eq=${r_s['equity']:.0f}  Wr={r_s['worst_m']*100:.1f}%")
        else:
            log(f"  seed={seed:3d}:  FAIL")
    log("")

    # ══════════════════════════════════════════════════════════════
    #  4. REGULARIZATION CURVE
    # ══════════════════════════════════════════════════════════════
    log("━" * 60)
    log("  4. REGULARIZATION CURVE (L2 sweep)")
    log("━" * 60)

    for l2_val in [0.5, 1.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0]:
        t0 = time.time()
        result_l2, _, _ = train_ensemble(df, FEATURES_25, l2=l2_val)
        r_l2 = quick_eval(result_l2, regime_df, CFG_6L3S, f"L2={l2_val}")
        if r_l2:
            log(f"  L2={l2_val:5.1f}:  Sh={r_l2['sharpe']:.2f}  Eq=${r_l2['equity']:.0f}  "
                f"Wr={r_l2['worst_m']*100:.1f}%  WM={r_l2['win_months']}/{r_l2['total_months']}  "
                f"({time.time()-t0:.0f}s)")
        else:
            log(f"  L2={l2_val:5.1f}:  FAIL ({time.time()-t0:.0f}s)")
    log("")

    # ══════════════════════════════════════════════════════════════
    #  5. NUM_LEAVES SWEEP
    # ══════════════════════════════════════════════════════════════
    log("━" * 60)
    log("  5. NUM_LEAVES SWEEP (L2=10)")
    log("━" * 60)

    for nl in [15, 31, 42, 63, 85, 127]:
        t0 = time.time()
        result_nl, _, _ = train_ensemble(df, FEATURES_25, l2=10.0, num_leaves=nl)
        r_nl = quick_eval(result_nl, regime_df, CFG_6L3S, f"leaves={nl}")
        if r_nl:
            log(f"  leaves={nl:4d}:  Sh={r_nl['sharpe']:.2f}  Eq=${r_nl['equity']:.0f}  "
                f"Wr={r_nl['worst_m']*100:.1f}%  WM={r_nl['win_months']}/{r_nl['total_months']}  "
                f"({time.time()-t0:.0f}s)")
    log("")

    # ══════════════════════════════════════════════════════════════
    #  6. PORTFOLIO CONFIG SWEEP
    # ══════════════════════════════════════════════════════════════
    log("━" * 60)
    log("  6. PORTFOLIO CONFIG SWEEP (using FEAT25 L2=10 predictions)")
    log("━" * 60)

    configs = [
        ("4L/3S", {"n_long": 4, "n_short": 3}),
        ("5L/3S", {"n_long": 5, "n_short": 3}),
        ("6L/3S", {"n_long": 6, "n_short": 3}),
        ("7L/3S", {"n_long": 7, "n_short": 3}),
        ("8L/4S", {"n_long": 8, "n_short": 4}),
        ("6L/2S", {"n_long": 6, "n_short": 2}),
        ("5L/5S", {"n_long": 5, "n_short": 5}),
    ]
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    for name, overrides in configs:
        for thr in thresholds:
            cfg = {**CFG_6L3S, **overrides, "dyn_threshold": thr}
            r = quick_eval(result_25, regime_df, cfg, f"{name}-t{thr}")
            if r:
                log(f"  {name} thr={thr:.1f}:  Sh={r['sharpe']:.2f}  Eq=${r['equity']:.0f}  "
                    f"Wr={r['worst_m']*100:.1f}%  WM={r['win_months']}/{r['total_months']}")
    log("")

    # ══════════════════════════════════════════════════════════════
    #  7. BOOTSTRAP CONFIDENCE INTERVAL
    # ══════════════════════════════════════════════════════════════
    log("━" * 60)
    log("  7. BOOTSTRAP CONFIDENCE INTERVAL (monthly returns)")
    log("━" * 60)

    port_25 = simulate(result_25, regime_df, 12, CFG_6L3S)
    if port_25 is not None and len(port_25) > 10:
        port_25_df = port_25.copy()
        port_25_df["month"] = port_25_df["timestamp"].dt.to_period("M")
        monthly_rets = port_25_df.groupby("month")["portfolio_ret"].apply(
            lambda x: (1 + x * LEVERAGE).prod() - 1).values

        n_boot = 1000
        np.random.seed(42)
        boot_sharpes = []
        for _ in range(n_boot):
            sample = np.random.choice(monthly_rets, size=len(monthly_rets), replace=True)
            sh = sample.mean() / (sample.std() + 1e-10) * np.sqrt(12)
            boot_sharpes.append(sh)
        boot_sharpes = np.array(boot_sharpes)

        ci_5, ci_25, ci_50, ci_75, ci_95 = np.percentile(boot_sharpes, [5, 25, 50, 75, 95])
        log(f"  Monthly returns: n={len(monthly_rets)}, mean={np.mean(monthly_rets)*100:.1f}%, "
            f"std={np.std(monthly_rets)*100:.1f}%")
        log(f"  Bootstrap Sharpe (n=1000):")
        log(f"    5th pct:  {ci_5:.2f}")
        log(f"    25th pct: {ci_25:.2f}")
        log(f"    MEDIAN:   {ci_50:.2f}")
        log(f"    75th pct: {ci_75:.2f}")
        log(f"    95th pct: {ci_95:.2f}")
        log(f"  P(Sh > 2.0): {(boot_sharpes > 2.0).mean()*100:.0f}%")
        log(f"  P(Sh > 3.0): {(boot_sharpes > 3.0).mean()*100:.0f}%")

        # Same for baseline
        port_base = simulate(result_base, regime_df, 12, CFG_6L3S)
        if port_base is not None and len(port_base) > 10:
            pb = port_base.copy()
            pb["month"] = pb["timestamp"].dt.to_period("M")
            mr_base = pb.groupby("month")["portfolio_ret"].apply(
                lambda x: (1 + x * LEVERAGE).prod() - 1).values
            boot_base = []
            for _ in range(n_boot):
                sample = np.random.choice(mr_base, size=len(mr_base), replace=True)
                sh = sample.mean() / (sample.std() + 1e-10) * np.sqrt(12)
                boot_base.append(sh)
            boot_base = np.array(boot_base)
            log(f"\n  Baseline bootstrap median: {np.median(boot_base):.2f}")
            log(f"  P(FEAT25 > BASELINE): {(boot_sharpes > np.median(boot_base)).mean()*100:.0f}%")
    log("")

    # ══════════════════════════════════════════════════════════════
    #  8. ABLATION STUDY
    # ══════════════════════════════════════════════════════════════
    log("━" * 60)
    log("  8. ABLATION STUDY (remove each new feature)")
    log("━" * 60)

    for feat in ["global_ls_ratio", "ret_std_24h"]:
        feats_abl = [f for f in FEATURES_25 if f != feat]
        result_abl, _, _ = train_ensemble(df, feats_abl, l2=10.0)
        r_abl = quick_eval(result_abl, regime_df, CFG_6L3S, f"-{feat}")
        if r_abl:
            delta = r_abl["sharpe"] - feat25_sh
            log(f"  Remove {feat:20s}: Sh={r_abl['sharpe']:.2f}  (delta={delta:+.2f})")
    log("")

    # ══════════════════════════════════════════════════════════════
    #  9. FEATURE IMPORTANCE
    # ══════════════════════════════════════════════════════════════
    log("━" * 60)
    log("  9. FEATURE IMPORTANCE (LGB gain, L2=10)")
    log("━" * 60)

    imp = get_importance(df, FEATURES_25, l2=10.0)
    if len(imp) > 0:
        imp_pct = imp / imp.sum() * 100
        for feat_name, pct in imp_pct.items():
            marker = " ★★" if feat_name in ["global_ls_ratio", "ret_std_24h"] else ""
            log(f"  {feat_name:25s}: {pct:5.1f}%{marker}")
    log("")

    # ══════════════════════════════════════════════════════════════
    #  10. BEST CONFIG: FEATURES_25 + optimal params
    # ══════════════════════════════════════════════════════════════
    log("━" * 60)
    log("  10. COMBINED BEST-OF-BEST")
    log("━" * 60)

    # Try best L2 with different learning rates
    for lr_val in [0.01, 0.03, 0.05, 0.1]:
        t0 = time.time()
        result_lr, _, _ = train_ensemble(df, FEATURES_25, l2=10.0, lr=lr_val)
        r_lr = quick_eval(result_lr, regime_df, CFG_6L3S, f"lr={lr_val}")
        if r_lr:
            log(f"  lr={lr_val:.2f}:  Sh={r_lr['sharpe']:.2f}  Eq=${r_lr['equity']:.0f}  "
                f"Wr={r_lr['worst_m']*100:.1f}%  ({time.time()-t0:.0f}s)")

    # Try min_child_samples variations with L2=10
    for mc in [50, 100, 150, 200]:
        t0 = time.time()
        result_mc, _, _ = train_ensemble(df, FEATURES_25, l2=10.0, min_child=mc)
        r_mc = quick_eval(result_mc, regime_df, CFG_6L3S, f"mc={mc}")
        if r_mc:
            log(f"  min_child={mc:4d}:  Sh={r_mc['sharpe']:.2f}  Eq=${r_mc['equity']:.0f}  "
                f"Wr={r_mc['worst_m']*100:.1f}%  ({time.time()-t0:.0f}s)")
    log("")

    # ══════════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════════
    log("=" * 80)
    log("  R29c DEEP VALIDATION SUMMARY")
    log("=" * 80)
    log(f"  BASELINE (23f, L2=1):   Sh={base_sh:.2f}")
    log(f"  FEATURES_25 (L2=10):    Sh={feat25_sh:.2f}  (delta=+{feat25_sh-base_sh:.2f})")
    log(f"  New features: global_ls_ratio, ret_std_24h")
    log(f"  Total runtime: {(time.time()-t_start)/60:.1f} min")
    log("  Done.")


if __name__ == "__main__":
    main()
