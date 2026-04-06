#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R29b — Combo Testing & Greedy Forward Selection

Uses R29 results to build optimal feature combos:
  EXP-A: Greedy forward from top-12 individual winners
  EXP-B: Predefined combos (top3, top5, diverse, by-category)
  EXP-C: Kitchen sink (all winners δ>+0.2)

Baseline: FEATURES_23 → Sh=2.02
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from pathlib import Path
import warnings, time, sys, itertools
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

CFG_BEST = {
    "n_long": 6, "n_short": 3, "trend_cutoff": 0.9,
    "dyn_threshold": 0.5625, "rebal_hours": 12,
    "kelly_sizing": False, "vol_scaling": False, "regime_asym": False,
}
CFG_6L3S = {**CFG_BEST, "n_long": 6, "n_short": 3, "dyn_threshold": 0.7}

# ═══════════════════════════════════════════════════════════════════════════════
#  R29 RESULTS — top individual features (deduplicated, sorted by delta)
# ═══════════════════════════════════════════════════════════════════════════════

# Top 12 winners by delta (taker_imbalance dropped — identical to taker_buy_sell_ratio)
TOP_12 = [
    "global_ls_ratio",      # +0.76  deriv
    "taker_buy_sell_ratio",  # +0.63  deriv
    "buy_pressure",          # +0.61  price shape
    "vol_ratio_12h",         # +0.54  volume
    "btc_beta_48h",          # +0.52  cross-asset
    "gk_vol_168h",           # +0.41  volatility
    "oi_chg_4h",             # +0.40  deriv
    "stoch_k",               # +0.37  TA
    "close_open_ratio",      # +0.36  price shape
    "ret_std_24h",           # +0.35  return stats
    "upper_shadow",          # +0.33  price shape
    "oi_chg_1h",             # +0.30  deriv
]

# Features by category (diverse — one per category)
DIVERSE_7 = [
    "global_ls_ratio",      # deriv
    "buy_pressure",          # price shape
    "vol_ratio_12h",         # volume
    "btc_beta_48h",          # cross-asset
    "gk_vol_168h",           # volatility
    "stoch_k",               # TA
    "ret_std_24h",           # return stats
]

# All features with delta > +0.2
ALL_WINNERS_02 = [
    "global_ls_ratio",       # +0.76
    "taker_buy_sell_ratio",  # +0.63
    "buy_pressure",          # +0.61
    "vol_ratio_12h",         # +0.54
    "btc_beta_48h",          # +0.52
    "gk_vol_168h",           # +0.41
    "oi_chg_4h",             # +0.40
    "stoch_k",               # +0.37
    "close_open_ratio",      # +0.36
    "ret_std_24h",           # +0.35
    "upper_shadow",          # +0.33
    "oi_chg_1h",             # +0.30
    "vol_mom_24h",           # +0.29
    "mom_accel_12h",         # +0.25
    "cci_14",                # +0.24
    "funding_x_mom_12h",     # +0.22
    "close_ma24_ratio",      # +0.22
    "btc_outperform",        # +0.22
    "vol_ma6_ratio",         # +0.21
    "close_ma6_ratio",       # +0.20
]


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAIN + EVAL (same as R29)
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


def run_experiment(df, regime_df, feats, name, baseline_sh):
    """Train LGB+XGB ensemble, simulate, return (name, sharpe, equity)."""
    avail = [f for f in feats if f in df.columns]
    log(f"  [{name}] {len(avail)}f (avail {len(avail)}/{len(feats)})")
    t0 = time.time()
    p_lgb = train_lgb_cls(df, avail)
    p_xgb = train_xgb_cls(df, avail)
    p_ens = ensemble_preds(p_lgb, p_xgb)
    port = simulate(p_ens, regime_df, 12, CFG_6L3S)
    r = eval_config(port, 12, name, LEVERAGE, CAPITAL)
    if r:
        show(r)
        delta = r["sharpe"] - baseline_sh
        log(f"  Delta: {delta:+.2f}  ({time.time()-t0:.0f}s)")
        return (name, r["sharpe"], r["equity"])
    else:
        log(f"  FAILED ({time.time()-t0:.0f}s)")
        return (name, None, None)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log("=" * 80)
    log("  R29b — Combo Testing & Greedy Forward Selection")
    log(f"  Date: {pd.Timestamp.now()}")
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

    regime_df = compute_regime(df)
    load_time = time.time() - t_start
    log(f"  Load/build time: {load_time:.0f}s\n")

    results = []

    # ══════════════════════════════════════════════════════════════
    #  EXP-A: BASELINE
    # ══════════════════════════════════════════════════════════════
    log("━" * 60)
    log("  EXP-A: BASELINE")
    log("━" * 60)
    _, baseline_sh, baseline_eq = run_experiment(
        df, regime_df, FEATURES_23, "BASELINE-23f", 0)
    if baseline_sh is None:
        baseline_sh = 0
    results.append(("BASELINE-23f", baseline_sh, baseline_eq))

    # ══════════════════════════════════════════════════════════════
    #  EXP-B: PREDEFINED COMBOS
    # ══════════════════════════════════════════════════════════════
    log("\n" + "━" * 60)
    log("  EXP-B: PREDEFINED COMBOS")
    log("━" * 60)

    combos = {
        "TOP3": TOP_12[:3],
        "TOP5": TOP_12[:5],
        "TOP7": TOP_12[:7],
        "TOP10": TOP_12[:10],
        "TOP12": TOP_12,
        "DIVERSE7": DIVERSE_7,
        "ALL_d02": ALL_WINNERS_02,
    }

    for name, extra in combos.items():
        feats = FEATURES_23 + extra
        r = run_experiment(df, regime_df, feats, name, baseline_sh)
        results.append(r)

    # ══════════════════════════════════════════════════════════════
    #  EXP-C: GREEDY FORWARD SELECTION
    # ══════════════════════════════════════════════════════════════
    log("\n" + "━" * 60)
    log("  EXP-C: GREEDY FORWARD from top-12")
    log("━" * 60)

    current_feats = list(FEATURES_23)
    current_sh = baseline_sh
    selected = []

    for step in range(len(TOP_12)):
        log(f"\n  --- Greedy step {step+1} (current Sh={current_sh:.2f}, {len(current_feats)}f) ---")
        best_feat = None
        best_sh = current_sh
        best_eq = None

        remaining = [f for f in TOP_12 if f not in selected]
        if not remaining:
            break

        for feat in remaining:
            trial_feats = current_feats + [feat]
            name = f"G{step+1}+{feat}"
            _, sh, eq = run_experiment(df, regime_df, trial_feats, name, current_sh)
            if sh is not None and sh > best_sh:
                best_feat = feat
                best_sh = sh
                best_eq = eq

        if best_feat:
            current_feats.append(best_feat)
            selected.append(best_feat)
            current_sh = best_sh
            log(f"  ✅ Selected: {best_feat} → Sh={best_sh:.2f} ({len(current_feats)}f)")
            results.append((f"GREEDY-{len(selected)}: +{best_feat}", best_sh, best_eq))
        else:
            log(f"  ❌ No improvement at step {step+1}. Stopping.")
            break

    # ══════════════════════════════════════════════════════════════
    #  EXP-D: GREEDY FINAL SET — test with stronger regularization
    # ══════════════════════════════════════════════════════════════
    if selected:
        log("\n" + "━" * 60)
        log(f"  EXP-D: GREEDY FINAL ({len(selected)} new features) — reg sweep")
        log("━" * 60)
        greedy_feats = FEATURES_23 + selected
        log(f"  Final feature set: FEATURES_23 + {selected}")

        # Test with higher regularization to confirm not overfitting
        for l2 in [3.0, 10.0]:
            name = f"GREEDY-FINAL-L2={l2}"
            log(f"\n  [{name}] {len(greedy_feats)}f, lambda_l2={l2}")
            t0 = time.time()

            # Custom train with stronger regularization
            avail = [f for f in greedy_feats if f in df.columns]
            tz = df["timestamp"].dt.tz
            all_lgb, all_xgb = [], []

            for seed in SEEDS:
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
                for w in WINDOWS:
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
                    m["pred"] = p
                    m = m.merge(fwd, on=["timestamp", "symbol"], how="inner")
                    m["window"] = w["name"]
                    all_lgb.append(m)

                    # XGB
                    dt_xgb = xgb.DMatrix(tr[avail], label=tr["target_binary"])
                    dv_xgb = xgb.DMatrix(va[avail], label=va["target_binary"])
                    m_xgb = xgb.train(params_xgb, dt_xgb, num_boost_round=600,
                                       evals=[(dv_xgb, "val")],
                                       early_stopping_rounds=40, verbose_eval=False)
                    p = m_xgb.predict(xgb.DMatrix(te[avail]))
                    m2 = te[["timestamp", "symbol"]].copy()
                    m2["pred"] = p
                    m2 = m2.merge(fwd, on=["timestamp", "symbol"], how="inner")
                    m2["window"] = w["name"]
                    all_xgb.append(m2)

            if all_lgb and all_xgb:
                lgb_df = pd.concat(all_lgb).groupby(["timestamp", "symbol"]).agg(
                    pred=("pred", "mean"), fwd_ret=("fwd_ret", "first"),
                    window=("window", "first")).reset_index()
                xgb_df = pd.concat(all_xgb).groupby(["timestamp", "symbol"]).agg(
                    pred=("pred", "mean"), fwd_ret=("fwd_ret", "first"),
                    window=("window", "first")).reset_index()
                p_ens = ensemble_preds(lgb_df, xgb_df)
                port = simulate(p_ens, regime_df, 12, CFG_6L3S)
                r = eval_config(port, 12, name, LEVERAGE, CAPITAL)
                if r:
                    show(r)
                    delta = r["sharpe"] - baseline_sh
                    log(f"  Delta vs baseline: {delta:+.2f}  ({time.time()-t0:.0f}s)")
                    results.append((name, r["sharpe"], r["equity"]))

    # ══════════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════════
    log("\n" + "=" * 80)
    log("  R29b COMBO RESULTS")
    log("=" * 80)
    for name, sh, eq in results:
        if sh is not None:
            delta = sh - baseline_sh
            marker = " ★★★" if delta > 0.5 else (" ★★" if delta > 0.3 else (" ★" if delta > 0 else ""))
            log(f"  {name:40s}  Sh={sh:.2f}  Eq=${eq:.0f}  delta={delta:+.2f}{marker}")
        else:
            log(f"  {name:40s}  FAIL")

    # Best result
    valid = [(n, s, e) for n, s, e in results if s is not None]
    if valid:
        best = max(valid, key=lambda x: x[1])
        log(f"\n  🏆 BEST: {best[0]} → Sh={best[1]:.2f}  (delta={best[1] - baseline_sh:+.2f})")

    total = time.time() - t_start
    log(f"\n  Total runtime: {total/60:.1f} min")
    log("  Done.")


if __name__ == "__main__":
    main()
