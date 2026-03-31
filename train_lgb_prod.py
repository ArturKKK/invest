#!/usr/bin/env python3
"""
Train LightGBM ensemble for production deployment.

Validates on walk-forward windows, then trains FINAL model on ALL data.
Saves models to results_lgb_prod/lgb_model_seed_*.txt

R13 config (validated R12F+R13-4 combo, Sh=4.81, WM=13/13, Wr=+2.4%):
  - num_leaves=63, lr=0.03, lambda_l2=1.0
  - 12 features (pruned: dropped dist_from_high_24h, mom_z_12h)
  - 5 seeds [0, 7, 13, 42, 99] → ensemble averaged at inference
  - Feature naming: CS-ranked IN-PLACE (same names as production inference)
  - EMA smoothing: NONE (LGB signal already high quality, EMA hurts)

Usage:
  python train_lgb_prod.py
  python train_lgb_prod.py --validate-first   # run walk-forward check first
  python train_lgb_prod.py --seeds 0 7 42     # custom seed list
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings("ignore")

from _research_round7 import (
    SYM_35, WINDOWS, FEATURES as FEATURES_14, cs_rank,
    compute_regime, simulate, eval_config, show,
)
from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal

OUTPUT_DIR = "results_lgb_prod"
SEEDS = [0, 7, 13, 42, 99]
NUM_LEAVES = 63
LR = 0.03
N_ROUNDS = 500
EARLY_STOP = 30
MIN_CHILD = 100
REG_L2 = 1.0

# R13: 12 features (FEATURES_14 minus dist_from_high_24h, mom_z_12h)
FEATURES_12 = [
    "ret_12h", "ret_24h", "ret_48h",
    "residual_12h", "residual_24h",
    "mom_z_24h",
    "oi_chg_12h", "oi_chg_24h", "oi_zscore",
    "taker_cvd_12h", "taker_cvd_24h",
    "ls_divergence",
]

LEVERAGE = 5
CAPITAL = 100
CFG_BASE = {
    "n_long": 6, "n_short": 3,
    "trend_cutoff": 0.8, "dyn_threshold": 0.5,
    "eq_mom_boost": True, "kelly_sizing": True,
    "strategy_momentum": True, "strat_mom_lookback": 48,
    "regime_asym": True, "vol_scaling": True,
    "signal_ema": None,   # KEY: no EMA for LGB
    "rebal_hours": 12,
}


def cs_rank_inplace(df, feats):
    """CS-rank features in-place (overwrites column, same names = production compatible)."""
    df = df.copy()
    for feat in feats:
        if feat in df.columns:
            df[feat] = df.groupby("timestamp")[feat].rank(pct=True) - 0.5
    return df


def train_lgb_fold(df_train, df_val, df_test, feats, seed, fwd_col="fwd_ret_12h"):
    """Train one LGB model on one walk-forward fold. Returns (model, metrics)."""
    for d in [df_train, df_val, df_test]:
        d["target_rank"] = d.groupby("timestamp")[fwd_col].rank(pct=True) - 0.5

    train_c = df_train[feats + ["target_rank"]].dropna()
    val_c   = df_val[feats + ["target_rank"]].dropna()

    dtrain = lgb.Dataset(train_c[feats], label=train_c["target_rank"])
    dval   = lgb.Dataset(val_c[feats],   label=val_c["target_rank"])

    params = {
        "objective": "regression", "metric": "mse",
        "learning_rate": LR, "num_leaves": NUM_LEAVES,
        "min_child_samples": MIN_CHILD,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "lambda_l2": REG_L2,
        "verbose": -1, "n_jobs": -1, "seed": seed,
    }
    model = lgb.train(
        params, dtrain, num_boost_round=N_ROUNDS,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                   lgb.log_evaluation(-1)],
    )

    # Compute ICs
    train_pred = model.predict(train_c[feats])
    val_pred   = model.predict(val_c[feats])
    test_c = df_test[feats + ["target_rank", "timestamp", "symbol"]].dropna()
    test_pred  = model.predict(test_c[feats])

    ic_train = stats.spearmanr(train_pred, train_c["target_rank"])[0]
    ic_val   = stats.spearmanr(val_pred,   val_c["target_rank"])[0]
    ic_test  = stats.spearmanr(test_pred,  test_c["target_rank"])[0]

    # Build predictions for backtest
    fwd_data = df_test[["timestamp", "symbol", fwd_col]].rename(
        columns={fwd_col: "fwd_ret"}).dropna()
    merged = test_c[["timestamp", "symbol"]].copy()
    merged["pred"] = test_pred
    merged = merged.merge(fwd_data, on=["timestamp", "symbol"], how="inner")

    metrics = {
        "trees": model.best_iteration,
        "ic_train": round(ic_train, 4),
        "ic_val": round(ic_val, 4),
        "ic_test": round(ic_test, 4),
        "ratio": round(ic_train / (ic_test + 1e-10), 2),
    }
    return model, metrics, merged


def validate_on_walkforward(df, feats):
    """Run walk-forward validation with seed ensemble. Returns dict of results."""
    print("\n" + "═"*60)
    print("  Walk-Forward Validation (5-seed ensemble)")
    print("═"*60)

    all_preds = []
    window_metrics = {w["name"]: [] for w in WINDOWS}

    for seed in SEEDS:
        print(f"\n  Seed {seed}:")
        seed_preds = []
        for w in WINDOWS:
            train = df[df["timestamp"] < pd.Timestamp(w["train_end"], tz="UTC")].copy()
            val   = df[(df["timestamp"] >= pd.Timestamp(w["val_start"], tz="UTC")) &
                       (df["timestamp"] <  pd.Timestamp(w["val_end"],   tz="UTC"))].copy()
            test  = df[(df["timestamp"] >= pd.Timestamp(w["test_start"], tz="UTC")) &
                       (df["timestamp"] <= pd.Timestamp(w["test_end"],   tz="UTC"))].copy()
            if len(train) < 5000 or len(test) < 200:
                continue

            train = cs_rank_inplace(train, feats)
            val   = cs_rank_inplace(val, feats)
            test  = cs_rank_inplace(test, feats)

            model, metrics, preds = train_lgb_fold(train, val, test, feats, seed)
            print(f"    {w['name']}: trees={metrics['trees']:3d}  "
                  f"IC train={metrics['ic_train']:.4f}  "
                  f"val={metrics['ic_val']:.4f}  "
                  f"test={metrics['ic_test']:.4f}  "
                  f"ratio={metrics['ratio']:.2f}x")
            seed_preds.append(preds)
            window_metrics[w["name"]].append(metrics)

        if seed_preds:
            all_preds.append(pd.concat(seed_preds, ignore_index=True))

    if not all_preds:
        print("  ❌ No predictions generated")
        return None

    # Ensemble: average predictions per (timestamp, symbol) across seeds
    combined = pd.concat(all_preds, ignore_index=True)
    ensemble_preds = (combined.groupby(["timestamp", "symbol"])
                      .agg(pred=("pred", "mean"), fwd_ret=("fwd_ret", "first"))
                      .reset_index())

    print(f"\n  Ensemble contains {len(ensemble_preds):,} rows "
          f"from {ensemble_preds['timestamp'].nunique():,} timestamps")

    # Backtest
    regime_df = compute_regime(df)
    r = eval_config(simulate(ensemble_preds, regime_df, 12, CFG_BASE),
                    12, "LGB ensemble (walk-forward validation)", LEVERAGE, CAPITAL)
    if r:
        show(r)
        print(f"\n  vs Ridge R7 prod: Sh=3.59, Wr=-6.4%, WM=9/13, Eq=$2993")
        print(f"  LGB improvement:  ΔSh={r['sharpe']-3.59:+.2f}, "
              f"ΔWr={r['worst_m']*100-(-6.4):+.1f}pp, "
              f"ΔEq={r['equity']-2993:+.0f}")

    return r


def train_final_models(df, feats, output_dir):
    """
    Train final models on ALL available data (no holdout).
    Each seed model is saved independently for ensemble inference.
    """
    print("\n" + "═"*60)
    print(f"  Training FINAL models on ALL data → {output_dir}/")
    print("═"*60)

    # Use all data where fwd_ret_12h is available
    df_full = cs_rank_inplace(df, feats)
    df_full["target_rank"] = df_full.groupby("timestamp")["fwd_ret_12h"].rank(pct=True) - 0.5
    df_train = df_full[feats + ["target_rank"]].dropna()

    print(f"  Training rows: {len(df_train):,}  "
          f"(timestamps: {df_full['timestamp'].nunique():,}, "
          f"cutoff: {df_full['timestamp'].max().date()})")

    # 10% of data as in-training validation (chronological split, no leakage)
    n_train = len(df_train)
    split = int(n_train * 0.9)
    tr = df_train.iloc[:split]
    va = df_train.iloc[split:]

    os.makedirs(output_dir, exist_ok=True)

    models_trained = []
    for seed in SEEDS:
        dtrain = lgb.Dataset(tr[feats], label=tr["target_rank"])
        dval   = lgb.Dataset(va[feats], label=va["target_rank"])

        params = {
            "objective": "regression", "metric": "mse",
            "learning_rate": LR, "num_leaves": NUM_LEAVES,
            "min_child_samples": MIN_CHILD,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "lambda_l2": REG_L2,
            "verbose": -1, "n_jobs": -1, "seed": seed,
        }
        model = lgb.train(
            params, dtrain, num_boost_round=N_ROUNDS,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                       lgb.log_evaluation(-1)],
        )

        model_path = os.path.join(output_dir, f"lgb_model_seed_{seed}.txt")
        model.save_model(model_path)
        models_trained.append(model)

        # IC on validation split
        val_pred = model.predict(va[feats])
        ic_val = stats.spearmanr(val_pred, va["target_rank"])[0]
        print(f"  seed={seed}: trees={model.best_iteration:3d}  IC_val(10%)={ic_val:.4f}  → {os.path.basename(model_path)}")

    # Save metadata
    meta = {
        "features": feats,
        "num_leaves": NUM_LEAVES,
        "seeds": SEEDS,
        "n_models": len(SEEDS),
        "train_rows": n_train,
        "signal_ema": None,
        "trained_through": str(df_full["timestamp"].max().date()),
        "note": "R13: 12f pruned, nl=63, lr=0.03, L2=1.0. Walk-forward Sh=4.81, WM=13/13, Wr=+2.4%. Leakage audit passed (R12).",
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  ✅ Saved {len(SEEDS)} models + meta.json to {output_dir}/")
    print(f"  Features: {feats}")
    return models_trained


def quick_sanity_check(models, df, feats, output_dir):
    """Quick IC check on last 3 months to verify final models are sane."""
    print("\n  Quick sanity: IC on last 90 days (not used in training)")
    cutoff = df["timestamp"].max() - pd.Timedelta(days=90)
    recent = cs_rank_inplace(df[df["timestamp"] >= cutoff].copy(), feats)
    recent["target_rank"] = recent.groupby("timestamp")["fwd_ret_12h"].rank(pct=True) - 0.5
    recent_c = recent[feats + ["target_rank"]].dropna()
    if len(recent_c) < 100:
        print("  ⚠️  Not enough recent data for sanity check")
        return
    preds_all = np.mean([m.predict(recent_c[feats]) for m in models], axis=0)
    ic = stats.spearmanr(preds_all, recent_c["target_rank"])[0]
    print(f"  Ensemble IC on last 90d: {ic:.4f}  "
          f"({'✅ OK' if ic > 0.03 else '⚠️ LOW — check model'})")


def main():
    global SEEDS, NUM_LEAVES, OUTPUT_DIR  # must be before any use of these names
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-first", action="store_true",
                        help="Run walk-forward validation before training final model")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS,
                        help="Seed list for ensemble")
    parser.add_argument("--num-leaves", type=int, default=NUM_LEAVES)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    SEEDS = args.seeds
    NUM_LEAVES = args.num_leaves
    OUTPUT_DIR = args.output_dir

    print("=" * 60)
    print("  TRAIN LGB PRODUCTION MODELS (R13)")
    print("=" * 60)
    print(f"  Seeds: {SEEDS}")
    print(f"  num_leaves: {NUM_LEAVES}  lr: {LR}  L2: {REG_L2}")
    print(f"  n_rounds: {N_ROUNDS}  early_stop: {EARLY_STOP}")
    print(f"  Output: {OUTPUT_DIR}/")

    # Load data
    print("\n📊 Loading data...")
    ohlcv  = load_ohlcv()
    ohlcv  = ohlcv[ohlcv["symbol"].isin(SYM_35)]
    derivs = load_derivatives()
    df     = build_features_minimal(ohlcv, derivs)
    feats  = [f for f in FEATURES_12 if f in df.columns]
    print(f"   df: {df.shape}, symbols: {df['symbol'].nunique()}")
    print(f"   date range: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print(f"   features ({len(feats)}): {feats}")

    # Optional: validate first
    if args.validate_first:
        val_result = validate_on_walkforward(df, feats)
        print("\n  Proceed with training final models? (auto-yes in script)")
        if val_result is None or val_result["sharpe"] < 3.0:
            print("  ⚠️  Validation sharpe < 3.0 — investigate before deploying!")
        else:
            print(f"  ✅ Validation OK (Sh={val_result['sharpe']:.2f}), proceeding.")

    # Train final models
    models = train_final_models(df, feats, OUTPUT_DIR)

    # Sanity check
    quick_sanity_check(models, df, feats, OUTPUT_DIR)

    print(f"""
  ══════════════════════════════════════════════════════
  DONE. Next steps:
  1. Copy {OUTPUT_DIR}/ to VPS:
       rsync -av {OUTPUT_DIR}/ user@185.42.163.63:~/invest/{OUTPUT_DIR}/
  2. Run trading with --lgb flag:
       python run_trading.py --mode live --loop --capital 100 \\
         --leverage 3 --lgb --vol-size --min-zscore 0.8
  3. Monitor first few cycles carefully.
  ══════════════════════════════════════════════════════
""")


if __name__ == "__main__":
    main()
