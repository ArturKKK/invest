#!/usr/bin/env python3
"""
Train LGB + XGB binary classification ensemble for production.

R25 winner: A-lgb+xgb ensemble — Sh=3.36, Eq=$3371, Worst=-5.7%
  - LGB binary classifier (P(ret_12h > 0)), 23 features, 5 seeds
  - XGB binary classifier (same), 5 seeds
  - Ensemble: average probabilities → rank → 5L/3S

Pipeline:
  1. Validate on walk-forward (3 windows) → backtest Sharpe
  2. Train FINAL models on ALL data (LGB × 5 seeds + XGB × 5 seeds)
  3. Save to results_cls_prod/

Usage:
  python train_cls_prod.py
  python train_cls_prod.py --validate-first
  python train_cls_prod.py --seeds 0 7 42
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from _research_round7 import (
    SYM_35, WINDOWS, cs_rank,
    compute_regime, simulate, eval_config, show,
)
from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal
from _research_r22_models import (
    FEATURES_23, SEEDS as DEFAULT_SEEDS,
    build_r19_features, add_new_features, cs_rank_cols,
    log, DATA_DIR, SENT_DIR,
)

OUTPUT_DIR = "results_cls_prod"
SEEDS = list(DEFAULT_SEEDS)

# ── LGB classifier params (R25 defaults, same as train_cls) ──
LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda_l2": 1.0,
    "verbose": -1,
    "n_jobs": -1,
}

# ── XGB classifier params (R25 EXP-A) ──
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbosity": 0,
}

N_ROUNDS = 600
EARLY_STOP = 40
LEVERAGE = 5
CAPITAL = 100

CFG_5L3S = {
    "n_long": 5, "n_short": 3, "trend_cutoff": 0.9,
    "dyn_threshold": 0.5625, "rebal_hours": 12,
    "kelly_sizing": False, "vol_scaling": False, "regime_asym": False,
}


def validate_on_walkforward(df, feats):
    """Run walk-forward: LGB+XGB cls ensemble → backtest."""
    print("\n" + "=" * 60)
    print("  Walk-Forward Validation (LGB+XGB cls ensemble)")
    print("=" * 60)

    tz = df["timestamp"].dt.tz
    all_lgb_preds = []
    all_xgb_preds = []

    for seed in SEEDS:
        print(f"\n  Seed {seed}:")
        lgb_seed, xgb_seed = [], []

        for w in WINDOWS:
            train = df[df["timestamp"] < pd.Timestamp(w["train_end"], tz=tz)].copy()
            val = df[(df["timestamp"] >= pd.Timestamp(w["val_start"], tz=tz)) &
                     (df["timestamp"] < pd.Timestamp(w["val_end"], tz=tz))].copy()
            test = df[(df["timestamp"] >= pd.Timestamp(w["test_start"], tz=tz)) &
                      (df["timestamp"] <= pd.Timestamp(w["test_end"], tz=tz))].copy()
            if len(train) < 5000 or len(test) < 200:
                continue

            train = cs_rank_cols(train, feats)
            val = cs_rank_cols(val, feats)
            test = cs_rank_cols(test, feats)

            for d in [train, val, test]:
                d["target_binary"] = (d["fwd_ret_12h"] > 0).astype(int)

            train_c = train[feats + ["target_binary"]].dropna()
            val_c = val[feats + ["target_binary"]].dropna()
            test_c = test[feats + ["target_binary", "timestamp", "symbol"]].dropna()
            if len(test_c) == 0:
                continue

            fwd = test[["timestamp", "symbol", "fwd_ret_12h"]].rename(
                columns={"fwd_ret_12h": "fwd_ret"}).dropna()

            # ── LGB ──
            dtrain_l = lgb.Dataset(train_c[feats], label=train_c["target_binary"])
            dval_l = lgb.Dataset(val_c[feats], label=val_c["target_binary"])
            lgb_model = lgb.train(
                {**LGB_PARAMS, "seed": seed}, dtrain_l,
                num_boost_round=N_ROUNDS, valid_sets=[dval_l],
                callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                           lgb.log_evaluation(-1)])
            lgb_p = lgb_model.predict(test_c[feats])
            m = test_c[["timestamp", "symbol"]].copy()
            m["pred"] = lgb_p
            m = m.merge(fwd, on=["timestamp", "symbol"], how="inner")
            lgb_seed.append(m)

            # ── XGB ──
            dtrain_x = xgb.DMatrix(train_c[feats], label=train_c["target_binary"])
            dval_x = xgb.DMatrix(val_c[feats], label=val_c["target_binary"])
            xgb_model = xgb.train(
                {**XGB_PARAMS, "seed": seed}, dtrain_x,
                num_boost_round=N_ROUNDS,
                evals=[(dval_x, "val")],
                early_stopping_rounds=EARLY_STOP, verbose_eval=False)
            dtest_x = xgb.DMatrix(test_c[feats])
            xgb_p = xgb_model.predict(dtest_x)
            m2 = test_c[["timestamp", "symbol"]].copy()
            m2["pred"] = xgb_p
            m2 = m2.merge(fwd, on=["timestamp", "symbol"], how="inner")
            xgb_seed.append(m2)

            print(f"    {w['name']}: LGB trees={lgb_model.best_iteration:3d}  "
                  f"XGB trees={xgb_model.best_iteration:3d}")

        if lgb_seed:
            all_lgb_preds.append(pd.concat(lgb_seed, ignore_index=True))
        if xgb_seed:
            all_xgb_preds.append(pd.concat(xgb_seed, ignore_index=True))

    if not all_lgb_preds or not all_xgb_preds:
        print("  ❌ Not enough predictions")
        return None

    # Average across seeds per model
    lgb_ens = (pd.concat(all_lgb_preds).groupby(["timestamp", "symbol"])
               .agg(pred_lgb=("pred", "mean"), fwd_ret=("fwd_ret", "first"))
               .reset_index())
    xgb_ens = (pd.concat(all_xgb_preds).groupby(["timestamp", "symbol"])
               .agg(pred_xgb=("pred", "mean"))
               .reset_index())

    merged = lgb_ens.merge(xgb_ens, on=["timestamp", "symbol"], how="inner")

    # Rank-normalize each model, then average
    for col in ["pred_lgb", "pred_xgb"]:
        merged[col] = merged.groupby("timestamp")[col].rank(pct=True) - 0.5
    merged["pred"] = 0.5 * merged["pred_lgb"] + 0.5 * merged["pred_xgb"]

    print(f"\n  Ensemble: {len(merged):,} rows, {merged['timestamp'].nunique():,} timestamps")

    regime_df = compute_regime(df)
    r = eval_config(simulate(merged, regime_df, 12, CFG_5L3S),
                    12, "LGB+XGB cls ensemble (walk-forward)", LEVERAGE, CAPITAL)
    if r:
        show(r)
    return r


def train_final_models(df, feats, output_dir):
    """Train LGB + XGB on ALL data, save to output_dir/."""
    print("\n" + "=" * 60)
    print(f"  Training FINAL models → {output_dir}/")
    print("=" * 60)

    df_full = cs_rank_cols(df.copy(), feats)
    for d in [df_full]:
        d["target_binary"] = (d["fwd_ret_12h"] > 0).astype(int)

    df_train = df_full[feats + ["target_binary"]].dropna()
    n_total = len(df_train)
    split = int(n_total * 0.9)
    tr = df_train.iloc[:split]
    va = df_train.iloc[split:]

    print(f"  Train rows: {len(tr):,}  Val rows: {len(va):,}")
    print(f"  Data through: {df_full['timestamp'].max().date()}")

    os.makedirs(output_dir, exist_ok=True)

    lgb_models = []
    xgb_models = []

    for seed in SEEDS:
        # ── LGB ──
        dtrain_l = lgb.Dataset(tr[feats], label=tr["target_binary"])
        dval_l = lgb.Dataset(va[feats], label=va["target_binary"])
        lgb_model = lgb.train(
            {**LGB_PARAMS, "seed": seed}, dtrain_l,
            num_boost_round=N_ROUNDS, valid_sets=[dval_l],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                       lgb.log_evaluation(-1)])
        lgb_path = os.path.join(output_dir, f"lgb_cls_seed_{seed}.txt")
        lgb_model.save_model(lgb_path)
        lgb_models.append(lgb_model)

        # ── XGB ──
        dtrain_x = xgb.DMatrix(tr[feats], label=tr["target_binary"])
        dval_x = xgb.DMatrix(va[feats], label=va["target_binary"])
        xgb_model = xgb.train(
            {**XGB_PARAMS, "seed": seed}, dtrain_x,
            num_boost_round=N_ROUNDS,
            evals=[(dval_x, "val")],
            early_stopping_rounds=EARLY_STOP, verbose_eval=False)
        xgb_path = os.path.join(output_dir, f"xgb_cls_seed_{seed}.json")
        xgb_model.save_model(xgb_path)
        xgb_models.append(xgb_model)

        # Quick AUC on val
        lgb_val_p = lgb_model.predict(va[feats])
        xgb_val_p = xgb_model.predict(xgb.DMatrix(va[feats]))
        ens_val = 0.5 * lgb_val_p + 0.5 * xgb_val_p

        from sklearn.metrics import roc_auc_score
        auc_lgb = roc_auc_score(va["target_binary"], lgb_val_p)
        auc_xgb = roc_auc_score(va["target_binary"], xgb_val_p)
        auc_ens = roc_auc_score(va["target_binary"], ens_val)
        print(f"  seed={seed}: LGB trees={lgb_model.best_iteration:3d} AUC={auc_lgb:.4f}  "
              f"XGB trees={xgb_model.best_iteration:3d} AUC={auc_xgb:.4f}  "
              f"Ensemble AUC={auc_ens:.4f}")

    # Save metadata
    meta = {
        "model_type": "binary_classification_ensemble",
        "models": {
            "lgb": {"n_seeds": len(SEEDS), "pattern": "lgb_cls_seed_*.txt",
                    "params": {k: v for k, v in LGB_PARAMS.items() if k != "verbose"}},
            "xgb": {"n_seeds": len(SEEDS), "pattern": "xgb_cls_seed_*.json",
                    "params": XGB_PARAMS},
        },
        "features": feats,
        "n_features": len(feats),
        "seeds": SEEDS,
        "ensemble_method": "rank_normalize_then_average",
        "train_rows": n_total,
        "trained_through": str(df_full["timestamp"].max().date()),
        "portfolio": {"n_long": 5, "n_short": 3, "rebal_hours": 12},
        "source": "R25 EXP-A (LGB+XGB cls ensemble, Sh=3.36, Worst=-5.7%)",
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  ✅ Saved {len(SEEDS)} LGB + {len(SEEDS)} XGB models + meta.json → {output_dir}/")
    return lgb_models, xgb_models


def sanity_check(lgb_models, xgb_models, df, feats, output_dir):
    """IC on last 90 days to verify signal quality."""
    print("\n  Sanity check: IC on last 90 days")
    cutoff = df["timestamp"].max() - pd.Timedelta(days=90)
    recent = cs_rank_cols(df[df["timestamp"] >= cutoff].copy(), feats)
    recent["target_rank"] = recent.groupby("timestamp")["fwd_ret_12h"].rank(pct=True) - 0.5
    recent_c = recent[feats + ["target_rank"]].dropna()
    if len(recent_c) < 100:
        print("  ⚠️  Not enough data")
        return

    lgb_preds = np.mean([m.predict(recent_c[feats]) for m in lgb_models], axis=0)
    xgb_preds = np.mean([m.predict(xgb.DMatrix(recent_c[feats])) for m in xgb_models], axis=0)

    # Rank-normalize + average (same as production)
    def rankn(x):
        return stats.rankdata(x) / len(x) - 0.5
    ens_preds = 0.5 * rankn(lgb_preds) + 0.5 * rankn(xgb_preds)

    ic_lgb = stats.spearmanr(lgb_preds, recent_c["target_rank"])[0]
    ic_xgb = stats.spearmanr(xgb_preds, recent_c["target_rank"])[0]
    ic_ens = stats.spearmanr(ens_preds, recent_c["target_rank"])[0]

    print(f"  LGB IC (90d): {ic_lgb:.4f}")
    print(f"  XGB IC (90d): {ic_xgb:.4f}")
    print(f"  Ensemble IC:  {ic_ens:.4f}  "
          f"({'✅ OK' if ic_ens > 0.03 else '⚠️ LOW'})")


def main():
    global SEEDS, OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Train LGB+XGB cls ensemble for production (R25)")
    parser.add_argument("--validate-first", action="store_true",
                        help="Run walk-forward validation before training")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    SEEDS = args.seeds
    OUTPUT_DIR = args.output_dir

    print("=" * 60)
    print("  TRAIN CLS PRODUCTION (R25: LGB+XGB ensemble)")
    print("=" * 60)
    print(f"  Seeds: {SEEDS}")
    print(f"  Output: {OUTPUT_DIR}/")
    print(f"  Features: FEATURES_23 ({len(FEATURES_23)} features)")

    # Load data
    print("\n📊 Loading data...")
    ohlcv = load_ohlcv()
    ohlcv = ohlcv[ohlcv["symbol"].isin(SYM_35)]
    derivs = load_derivatives()
    df = build_features_minimal(ohlcv, derivs)
    print(f"  Base: {len(df):,} rows, {df['symbol'].nunique()} symbols")

    print("\n  Building features...")
    df = build_r19_features(df)
    df, _ = add_new_features(df)

    feats = [f for f in FEATURES_23 if f in df.columns]
    print(f"  Available: {len(feats)}/{len(FEATURES_23)} features")
    print(f"  Date range: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

    missing = [f for f in FEATURES_23 if f not in df.columns]
    if missing:
        print(f"  ⚠️  Missing features: {missing}")

    # Optional validation
    if args.validate_first:
        r = validate_on_walkforward(df, feats)
        if r is None or r["sharpe"] < 2.5:
            sh_str = 'N/A' if r is None else f'{r["sharpe"]:.2f}'
            print(f"\n  ⚠️  Validation Sharpe {sh_str} — check before deploying!")
        else:
            print(f"\n  ✅ Validation OK (Sh={r['sharpe']:.2f})")

    # Train final models
    lgb_models, xgb_models = train_final_models(df, feats, OUTPUT_DIR)

    # Sanity check
    sanity_check(lgb_models, xgb_models, df, feats, OUTPUT_DIR)

    print(f"""
  ══════════════════════════════════════════════════════
  DONE. Deploy:
  1. rsync -av {OUTPUT_DIR}/ root@185.42.163.63:~/invest/{OUTPUT_DIR}/
  2. python run_trading.py --mode live --loop --capital 100 \\
       --leverage 3 --cls --vol-size
  3. Monitor first cycles in logs.
  ══════════════════════════════════════════════════════
""")


if __name__ == "__main__":
    main()
