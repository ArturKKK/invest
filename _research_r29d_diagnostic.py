#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R29d -- Diagnostic: WHY Sh=2.02 vs Sh=3.36?

Tests hypotheses:
  H1: build_production_features() contaminates data
  H2: R29c's train_ensemble differs from R25/R26's
  H3: Data has changed since R25/R26
  H4: cs_rank_cols on these features hurts

Also: W2 deep dive — what happens in May-Aug 2025?
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import warnings, time, sys
warnings.filterwarnings("ignore")

try:
    import ta
except ImportError:
    print("pip install ta"); sys.exit(1)

from _research_round7 import (
    SYM_35, WINDOWS, compute_regime, simulate, eval_config, show,
)
from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal
from _research_r22_models import (
    FEATURES_23, SEEDS, LEVERAGE, CAPITAL,
    log, build_r19_features, add_new_features, cs_rank_cols,
)
from _research_r29_forward import build_production_features

FEATURES_25 = FEATURES_23 + ["global_ls_ratio", "ret_std_24h"]

CFG_6L3S = {
    "n_long": 6, "n_short": 3, "trend_cutoff": 0.9,
    "dyn_threshold": 0.7, "rebal_hours": 12,
    "kelly_sizing": False, "vol_scaling": False, "regime_asym": False,
}
CFG_5L3S = {
    "n_long": 5, "n_short": 3, "trend_cutoff": 0.9,
    "dyn_threshold": 0.5625, "rebal_hours": 12,
    "kelly_sizing": False, "vol_scaling": False, "regime_asym": False,
}


def train_ensemble_diag(df, feats, seeds=SEEDS, l2=1.0, num_leaves=63,
                        min_child=100, use_cs_rank=True, label=""):
    """Minimal train, returns preds + diagnostic info."""
    avail = [f for f in feats if f in df.columns]
    missing = [f for f in feats if f not in df.columns]
    if missing:
        log(f"  WARNING: Missing features for {label}: {missing}")
    all_lgb, all_xgb = [], []
    tz = df["timestamp"].dt.tz
    diag = {"rows_train": [], "rows_test": [], "n_features": len(avail)}

    for seed in seeds:
        params_lgb = {
            "objective": "binary", "metric": "auc",
            "learning_rate": 0.03, "num_leaves": num_leaves,
            "min_child_samples": min_child, "subsample": 0.8,
            "colsample_bytree": 0.8, "lambda_l2": l2,
            "verbose": -1, "n_jobs": -1, "seed": seed,
        }
        params_xgb = {
            "objective": "binary:logistic", "eval_metric": "auc",
            "learning_rate": 0.03, "max_depth": 6,
            "min_child_weight": min_child, "subsample": 0.8,
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

            if use_cs_rank:
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

            # Replace inf with NaN then drop
            for d in [tr, va, te]:
                d.replace([np.inf, -np.inf], np.nan, inplace=True)
            tr = tr.dropna()
            va = va.dropna()
            te = te.dropna()
            if len(te) == 0:
                continue

            diag["rows_train"].append(len(tr))
            diag["rows_test"].append(len(te))

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

    if not all_lgb:
        return None, diag

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
    return merged[["timestamp", "symbol", "pred", "fwd_ret", "window"]], diag


def eval_and_show(preds, regime_df, cfg, name):
    port = simulate(preds, regime_df, 12, cfg)
    r = eval_config(port, 12, name, LEVERAGE, CAPITAL)
    show(r)
    return r


def per_window_sharpe(preds, regime_df, cfg):
    for wname in ["W1", "W2", "W3"]:
        sub = preds[preds["window"] == wname]
        if len(sub) < 10:
            log(f"  {wname}: NO DATA")
            continue
        port = simulate(sub, regime_df, 12, cfg)
        r = eval_config(port, 12, wname, LEVERAGE, CAPITAL)
        log(f"  {wname}: Sh={r['sharpe']:.2f}  Eq=${r['equity']:.0f}  Wr={r['worst_m']*100:.1f}%  WM={r['win_months']}/{r['total_months']}")


def main():
    t0 = time.time()
    log("=" * 80)
    log("  R29d — DIAGNOSTIC: Sharpe discrepancy + W2 deep dive")
    log("=" * 80)

    # ── Load data TWO ways ────────────────────────────────────
    log("\n[1] Loading data WITHOUT build_production_features (R25/R26 way)...")
    ohlcv = load_ohlcv()
    ohlcv = ohlcv[ohlcv["symbol"].isin(SYM_35)]
    derivs = load_derivatives()
    df_clean = build_features_minimal(ohlcv, derivs)
    df_clean = build_r19_features(df_clean)
    df_clean, _ = add_new_features(df_clean)
    df_clean = df_clean[df_clean["symbol"].isin(SYM_35)].copy()
    regime_clean = compute_regime(df_clean)
    log(f"  Clean: {len(df_clean):,} rows, {len(df_clean.columns)} cols")

    log("\n[2] Loading data WITH build_production_features (R29c way)...")
    df_prod = build_features_minimal(ohlcv, derivs)
    df_prod = build_r19_features(df_prod)
    df_prod, _ = add_new_features(df_prod)
    df_prod = build_production_features(df_prod)
    df_prod = df_prod[df_prod["symbol"].isin(SYM_35)].copy()
    regime_prod = compute_regime(df_prod)
    log(f"  Prod:  {len(df_prod):,} rows, {len(df_prod.columns)} cols")

    # Check NaN counts in FEATURES_23
    log("\n[3] NaN counts for FEATURES_23 columns:")
    for f in FEATURES_23:
        if f in df_clean.columns and f in df_prod.columns:
            n_clean = df_clean[f].isna().sum()
            n_prod = df_prod[f].isna().sum()
            if n_clean != n_prod:
                log(f"  DIFF {f}: clean={n_clean} vs prod={n_prod}")
    log("  (only showing mismatches)")

    # ══════════════════════════════════════════════════════════════
    #  H1: build_production_features contaminates baseline?
    # ══════════════════════════════════════════════════════════════
    log("\n" + "━" * 60)
    log("  H1: BASELINE 23f/L2=1 — CLEAN vs PROD data")
    log("━" * 60)

    log("\n  Training on CLEAN data (R25/R26 way)...")
    t1 = time.time()
    preds_clean, diag_clean = train_ensemble_diag(
        df_clean, FEATURES_23, l2=1.0, label="CLEAN-23f")
    r_clean = eval_and_show(preds_clean, regime_clean, CFG_6L3S, "CLEAN-23f-6L3S")
    log(f"  Rows train: {np.mean(diag_clean['rows_train']):.0f} avg")
    log(f"  Rows test: {np.mean(diag_clean['rows_test']):.0f} avg")
    log(f"  Time: {time.time()-t1:.0f}s")

    log("\n  Per-window (CLEAN):")
    per_window_sharpe(preds_clean, regime_clean, CFG_6L3S)

    log("\n  Training on PROD data (R29c way)...")
    t1 = time.time()
    preds_prod, diag_prod = train_ensemble_diag(
        df_prod, FEATURES_23, l2=1.0, label="PROD-23f")
    r_prod = eval_and_show(preds_prod, regime_prod, CFG_6L3S, "PROD-23f-6L3S")
    log(f"  Rows train: {np.mean(diag_prod['rows_train']):.0f} avg")
    log(f"  Rows test: {np.mean(diag_prod['rows_test']):.0f} avg")
    log(f"  Time: {time.time()-t1:.0f}s")

    log("\n  Per-window (PROD):")
    per_window_sharpe(preds_prod, regime_prod, CFG_6L3S)

    # Also try 5L/3S config (what R25 originally used)
    log("\n  Also: CLEAN-23f with R25's 5L/3S config:")
    r_5l3s = eval_and_show(preds_clean, regime_clean, CFG_5L3S, "CLEAN-23f-5L3S")

    # ══════════════════════════════════════════════════════════════
    #  H2: cs_rank_cols effect
    # ══════════════════════════════════════════════════════════════
    log("\n" + "━" * 60)
    log("  H2: EFFECT OF cs_rank_cols")
    log("━" * 60)

    log("\n  Training WITHOUT cs_rank_cols on CLEAN data...")
    t1 = time.time()
    preds_norank, diag_norank = train_ensemble_diag(
        df_clean, FEATURES_23, l2=1.0, use_cs_rank=False, label="NO-RANK-23f")
    r_norank = eval_and_show(preds_norank, regime_clean, CFG_6L3S, "NO-RANK-23f")
    log(f"  Time: {time.time()-t1:.0f}s")

    # ══════════════════════════════════════════════════════════════
    #  FEAT25 L2=10 on CLEAN data (fair comparison)
    # ══════════════════════════════════════════════════════════════
    log("\n" + "━" * 60)
    log("  FEAT25 L2=10 on PROD data (R29c result) — for reference")
    log("━" * 60)

    log("\n  Training FEAT25 on PROD data, L2=10...")
    t1 = time.time()
    preds_f25, diag_f25 = train_ensemble_diag(
        df_prod, FEATURES_25, l2=10.0, label="FEAT25-L2=10")
    r_f25 = eval_and_show(preds_f25, regime_prod, CFG_6L3S, "FEAT25-L2=10")
    log(f"  Time: {time.time()-t1:.0f}s")

    log("\n  Per-window (FEAT25):")
    per_window_sharpe(preds_f25, regime_prod, CFG_6L3S)

    # ══════════════════════════════════════════════════════════════
    #  Also: 23f with L2=10 on CLEAN (isolate feature effect)
    # ══════════════════════════════════════════════════════════════
    log("\n" + "━" * 60)
    log("  ISOLATED: 23f + L2=10 on CLEAN data")
    log("━" * 60)

    log("\n  Training 23f/L2=10 on CLEAN data...")
    t1 = time.time()
    preds_23l10, _ = train_ensemble_diag(
        df_clean, FEATURES_23, l2=10.0, label="CLEAN-23f-L2=10")
    r_23l10 = eval_and_show(preds_23l10, regime_clean, CFG_6L3S, "CLEAN-23f-L2=10")
    log(f"  Time: {time.time()-t1:.0f}s")

    log("\n  Per-window (23f L2=10 CLEAN):")
    per_window_sharpe(preds_23l10, regime_clean, CFG_6L3S)

    # ══════════════════════════════════════════════════════════════
    #  W2 DEEP DIVE
    # ══════════════════════════════════════════════════════════════
    log("\n" + "━" * 60)
    log("  W2 DEEP DIVE (May-Aug 2025)")
    log("━" * 60)

    # Compare W2 predictions quality
    for label_name, preds in [("CLEAN-23f", preds_clean), ("PROD-23f", preds_prod),
                               ("FEAT25-L2=10", preds_f25)]:
        w2 = preds[preds["window"] == "W2"]
        if len(w2) == 0:
            continue
        # Prediction-return correlation
        corr = w2["pred"].corr(w2["fwd_ret"])
        icir = w2.groupby("timestamp").apply(
            lambda g: g["pred"].corr(g["fwd_ret"])).dropna()
        mean_ic = icir.mean()
        std_ic = icir.std()
        ic_ir = mean_ic / (std_ic + 1e-10)

        # Monthly breakdown
        w2_copy = w2.copy()
        w2_copy["month"] = w2_copy["timestamp"].dt.to_period("M")
        months = sorted(w2_copy["month"].unique())
        m_ics = []
        for mo in months:
            sub = w2_copy[w2_copy["month"] == mo]
            if len(sub) > 0:
                mo_ic = sub.groupby("timestamp").apply(
                    lambda g: g["pred"].corr(g["fwd_ret"])).dropna()
                m_ics.append(f"{mo}: IC={mo_ic.mean():.4f}")

        log(f"\n  {label_name} W2:")
        log(f"    Rows: {len(w2)}, periods: {w2['timestamp'].nunique()}")
        log(f"    Overall corr: {corr:.4f}")
        log(f"    Mean IC: {mean_ic:.4f}, Std IC: {std_ic:.4f}, ICIR: {ic_ir:.4f}")
        log(f"    Monthly: {', '.join(m_ics)}")

    # ══════════════════════════════════════════════════════════════
    #  LEAKAGE CHECK
    # ══════════════════════════════════════════════════════════════
    log("\n" + "━" * 60)
    log("  LEAKAGE CHECK")
    log("━" * 60)

    if preds_f25 is not None:
        # Check if features built from production use future data
        # Simple check: shuffle labels within each window/timestamp
        log("\n  Shuffled labels test (FEAT25 L2=10)...")
        # Train with shuffled target — should get ~0 Sharpe
        df_shuffled = df_prod.copy()
        np.random.seed(42)
        for w in WINDOWS:
            tz = df_shuffled["timestamp"].dt.tz
            mask = ((df_shuffled["timestamp"] >= pd.Timestamp(w["test_start"], tz=tz)) &
                    (df_shuffled["timestamp"] <= pd.Timestamp(w["test_end"], tz=tz)))
            idx = df_shuffled[mask].index
            df_shuffled.loc[idx, "fwd_ret_12h"] = np.random.permutation(
                df_shuffled.loc[idx, "fwd_ret_12h"].values)

        preds_shuf, _ = train_ensemble_diag(
            df_shuffled, FEATURES_25, l2=10.0, label="SHUFFLED")
        if preds_shuf is not None:
            r_shuf = eval_and_show(preds_shuf, regime_prod, CFG_6L3S, "SHUFFLED-FEAT25")
            log(f"  Shuffled Sharpe: {r_shuf['sharpe']:.2f} (should be ~0 if no leakage)")
        else:
            log("  Shuffled: no preds produced")

    # ══════════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════════
    log("\n" + "=" * 80)
    log("  DIAGNOSTIC SUMMARY")
    log("=" * 80)

    results = {
        "CLEAN-23f-6L3S": r_clean["sharpe"] if r_clean else None,
        "PROD-23f-6L3S": r_prod["sharpe"] if r_prod else None,
        "NO-RANK-23f": r_norank["sharpe"] if r_norank else None,
        "CLEAN-23f-5L3S": r_5l3s["sharpe"] if r_5l3s else None,
        "CLEAN-23f-L2=10": r_23l10["sharpe"] if r_23l10 else None,
        "FEAT25-L2=10": r_f25["sharpe"] if r_f25 else None,
    }
    for name, sh in results.items():
        if sh is not None:
            log(f"  {name:25s}: Sh={sh:.2f}")

    log(f"\n  R25/R26 reported: Sh=3.36-3.39 (same pipeline, unknown diff)")
    log(f"  Total runtime: {(time.time()-t0)/60:.1f} min")
    log("  Done.")


if __name__ == "__main__":
    main()
