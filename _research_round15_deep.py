#!/usr/bin/env python3
"""
Research Round 15 — Deep Optimization.

Baseline: R13 prod (12f, nl=63, lr=0.03, L2=1.0) → Sh=4.81, WM=13/13, Wr=+2.4%

R15A: Fine-grained hyperparameter grid around winner
R15B: Engineered feature interactions
R15C: DART mode (dropout regularization)
R15D: Target winsorization (clip extreme returns)
R15E: Time-weighted training (recent data weighted higher)
R15F: Extra Trees mode (less greedy splitting)
R15G: Bagging variations (subsample/colsample)
R15H: Heterogeneous ensemble (different configs per seed)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
import warnings
import time
warnings.filterwarnings("ignore")

from _research_round7 import (
    SYM_35, WINDOWS, FEATURES as FEATURES_14, cs_rank,
    compute_regime, simulate, eval_config, show,
)
from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal

FEATURES_12 = [
    "ret_12h", "ret_24h", "ret_48h",
    "residual_12h", "residual_24h",
    "mom_z_24h",
    "oi_chg_12h", "oi_chg_24h", "oi_zscore",
    "taker_cvd_12h", "taker_cvd_24h",
    "ls_divergence",
]

SEEDS = [0, 7, 13, 42, 99]
N_ROUNDS = 500
EARLY_STOP = 30
LEVERAGE = 5
CAPITAL = 100

CFG_BASE = {
    "n_long": 6, "n_short": 3,
    "trend_cutoff": 0.8, "dyn_threshold": 0.5,
    "eq_mom_boost": True, "kelly_sizing": True,
    "strategy_momentum": True, "strat_mom_lookback": 48,
    "regime_asym": True, "vol_scaling": True,
    "signal_ema": None,
    "rebal_hours": 12,
}


def cs_rank_inplace(df, feats):
    df = df.copy()
    for feat in feats:
        if feat in df.columns:
            df[feat] = df.groupby("timestamp")[feat].rank(pct=True) - 0.5
    return df


def run_wf(df, feats, name, regime_df, lgb_params=None, cs_feats=None,
           target_transform=None, sample_weight_fn=None):
    """Full walk-forward eval with flexible params."""
    if lgb_params is None:
        lgb_params = {
            "objective": "regression", "metric": "mse",
            "learning_rate": 0.03, "num_leaves": 63,
            "min_child_samples": 100,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "lambda_l2": 1.0,
            "verbose": -1, "n_jobs": -1,
        }
    if cs_feats is None:
        cs_feats = feats

    all_preds = []
    all_ics = []

    for seed in SEEDS:
        seed_preds = []
        for w in WINDOWS:
            train = df[df["timestamp"] < pd.Timestamp(w["train_end"], tz="UTC")].copy()
            val   = df[(df["timestamp"] >= pd.Timestamp(w["val_start"], tz="UTC")) &
                       (df["timestamp"] <  pd.Timestamp(w["val_end"],   tz="UTC"))].copy()
            test  = df[(df["timestamp"] >= pd.Timestamp(w["test_start"], tz="UTC")) &
                       (df["timestamp"] <= pd.Timestamp(w["test_end"],   tz="UTC"))].copy()
            if len(train) < 5000 or len(test) < 200:
                continue

            train = cs_rank_inplace(train, cs_feats)
            val   = cs_rank_inplace(val, cs_feats)
            test  = cs_rank_inplace(test, cs_feats)

            for d in [train, val, test]:
                d["target_rank"] = d.groupby("timestamp")["fwd_ret_12h"].rank(pct=True) - 0.5

            if target_transform is not None:
                train = target_transform(train)
                val = target_transform(val)

            train_c = train[feats + ["target_rank"]].dropna()
            val_c   = val[feats + ["target_rank"]].dropna()

            dtrain = lgb.Dataset(train_c[feats], label=train_c["target_rank"])
            dval   = lgb.Dataset(val_c[feats],   label=val_c["target_rank"])

            if sample_weight_fn is not None:
                weights = sample_weight_fn(train, train_c)
                dtrain = lgb.Dataset(train_c[feats], label=train_c["target_rank"],
                                     weight=weights)

            p = {**lgb_params, "seed": seed}
            callbacks = [lgb.early_stopping(EARLY_STOP, verbose=False),
                         lgb.log_evaluation(-1)]

            # DART doesn't support early stopping well
            if p.get("boosting_type") == "dart":
                model = lgb.train(p, dtrain, num_boost_round=200,
                                  valid_sets=[dval], callbacks=[lgb.log_evaluation(-1)])
            else:
                model = lgb.train(p, dtrain, num_boost_round=N_ROUNDS,
                                  valid_sets=[dval], callbacks=callbacks)

            test_c = test[feats + ["target_rank", "timestamp", "symbol"]].dropna()
            test_pred = model.predict(test_c[feats])
            ic_test = stats.spearmanr(test_pred, test_c["target_rank"])[0]
            all_ics.append(ic_test)

            fwd_data = test[["timestamp", "symbol", "fwd_ret_12h"]].rename(
                columns={"fwd_ret_12h": "fwd_ret"}).dropna()
            merged = test_c[["timestamp", "symbol"]].copy()
            merged["pred"] = test_pred
            merged = merged.merge(fwd_data, on=["timestamp", "symbol"], how="inner")
            seed_preds.append(merged)

        if seed_preds:
            all_preds.append(pd.concat(seed_preds, ignore_index=True))

    if not all_preds:
        return None

    combined = pd.concat(all_preds, ignore_index=True)
    ens = (combined.groupby(["timestamp", "symbol"])
           .agg(pred=("pred", "mean"), fwd_ret=("fwd_ret", "first"))
           .reset_index())

    r = eval_config(simulate(ens, regime_df, 12, CFG_BASE), 12, name, LEVERAGE, CAPITAL)
    if r:
        r["mean_ic_test"] = round(np.mean(all_ics), 4)
    return r


def run_het_ensemble(df, feats, configs, name, regime_df, cs_feats=None):
    """Heterogeneous ensemble: each seed uses a different config."""
    if cs_feats is None:
        cs_feats = feats

    all_preds = []
    all_ics = []

    for i, (seed, params) in enumerate(zip(SEEDS, configs)):
        seed_preds = []
        for w in WINDOWS:
            train = df[df["timestamp"] < pd.Timestamp(w["train_end"], tz="UTC")].copy()
            val   = df[(df["timestamp"] >= pd.Timestamp(w["val_start"], tz="UTC")) &
                       (df["timestamp"] <  pd.Timestamp(w["val_end"],   tz="UTC"))].copy()
            test  = df[(df["timestamp"] >= pd.Timestamp(w["test_start"], tz="UTC")) &
                       (df["timestamp"] <= pd.Timestamp(w["test_end"],   tz="UTC"))].copy()
            if len(train) < 5000 or len(test) < 200:
                continue

            train = cs_rank_inplace(train, cs_feats)
            val   = cs_rank_inplace(val, cs_feats)
            test  = cs_rank_inplace(test, cs_feats)

            for d in [train, val, test]:
                d["target_rank"] = d.groupby("timestamp")["fwd_ret_12h"].rank(pct=True) - 0.5

            train_c = train[feats + ["target_rank"]].dropna()
            val_c   = val[feats + ["target_rank"]].dropna()

            p = {**params, "seed": seed}
            dtrain = lgb.Dataset(train_c[feats], label=train_c["target_rank"])
            dval   = lgb.Dataset(val_c[feats],   label=val_c["target_rank"])
            model = lgb.train(p, dtrain, num_boost_round=N_ROUNDS,
                              valid_sets=[dval],
                              callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                                         lgb.log_evaluation(-1)])

            test_c = test[feats + ["target_rank", "timestamp", "symbol"]].dropna()
            test_pred = model.predict(test_c[feats])
            ic_test = stats.spearmanr(test_pred, test_c["target_rank"])[0]
            all_ics.append(ic_test)

            fwd_data = test[["timestamp", "symbol", "fwd_ret_12h"]].rename(
                columns={"fwd_ret_12h": "fwd_ret"}).dropna()
            merged = test_c[["timestamp", "symbol"]].copy()
            merged["pred"] = test_pred
            merged = merged.merge(fwd_data, on=["timestamp", "symbol"], how="inner")
            seed_preds.append(merged)

        if seed_preds:
            all_preds.append(pd.concat(seed_preds, ignore_index=True))

    if not all_preds:
        return None

    combined = pd.concat(all_preds, ignore_index=True)
    ens = (combined.groupby(["timestamp", "symbol"])
           .agg(pred=("pred", "mean"), fwd_ret=("fwd_ret", "first"))
           .reset_index())

    r = eval_config(simulate(ens, regime_df, 12, CFG_BASE), 12, name, LEVERAGE, CAPITAL)
    if r:
        r["mean_ic_test"] = round(np.mean(all_ics), 4)
    return r


def add_interaction_features(df):
    """Add engineered feature interactions."""
    # OI momentum change
    df["oi_accel"] = df["oi_chg_24h"] - df["oi_chg_12h"]
    # Return acceleration
    df["ret_accel"] = df["ret_24h"] - 2 * df["ret_12h"]
    # Taker momentum
    df["taker_accel"] = df["taker_cvd_24h"] - df["taker_cvd_12h"]
    # OI-return divergence
    df["oi_ret_div"] = df["oi_chg_12h"] - df["ret_12h"]
    # LS divergence × OI zscore interaction
    df["ls_oi_interact"] = df["ls_divergence"] * df["oi_zscore"]
    return df


def main():
    t0 = time.time()
    print("=" * 70)
    print("  RESEARCH ROUND 15 — Deep Optimization")
    print("  Baseline: R13 (12f, nl=63, lr=0.03, L2=1.0) → Sh=4.81")
    print("=" * 70)

    print("\n📊 Loading data...")
    ohlcv  = load_ohlcv()
    ohlcv  = ohlcv[ohlcv["symbol"].isin(SYM_35)]
    derivs = load_derivatives()
    df     = build_features_minimal(ohlcv, derivs)
    feats  = [f for f in FEATURES_12 if f in df.columns]
    print(f"   df: ({len(df):,}, {len(df.columns)})")

    regime_df = compute_regime(df)

    results = []

    # ═══════════════════════════════════════════════
    # BASELINE
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  BASELINE")
    print("═" * 70)
    r = run_wf(df, feats, "Baseline: R13 prod", regime_df)
    if r:
        show(r)
        results.append(r)

    # ═══════════════════════════════════════════════
    # R15A: Fine-grained hyperparameter grid
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  R15A — Fine-Grained Hyperparameter Grid")
    print("═" * 70)

    grid = [
        # (lr, nl, mc, l2, name_suffix)
        (0.02, 63,  100, 1.0, "lr=0.02"),
        (0.04, 63,  100, 1.0, "lr=0.04"),
        (0.03, 63,  100, 0.5, "L2=0.5"),
        (0.03, 63,  100, 2.0, "L2=2.0"),
        (0.03, 63,  100, 3.0, "L2=3.0"),
        (0.03, 63,  75,  1.0, "mc=75"),
        (0.03, 63,  50,  1.0, "mc=50"),
        (0.03, 47,  100, 1.0, "nl=47"),
        (0.03, 95,  100, 1.0, "nl=95"),
        (0.02, 63,  100, 2.0, "lr=0.02+L2=2"),
        (0.03, 63,  100, 1.0, "baseline_check"),  # verify reproducibility
    ]

    for lr, nl, mc, l2, suffix in grid:
        params = {
            "objective": "regression", "metric": "mse",
            "learning_rate": lr, "num_leaves": nl,
            "min_child_samples": mc,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "lambda_l2": l2,
            "verbose": -1, "n_jobs": -1,
        }
        r = run_wf(df, feats, f"R15A: {suffix}", regime_df, lgb_params=params)
        if r:
            show(r)
            results.append(r)

    # ═══════════════════════════════════════════════
    # R15B: Feature Interactions
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  R15B — Feature Interactions")
    print("═" * 70)

    df = add_interaction_features(df)

    interact_feats_1 = feats + ["oi_accel", "ret_accel"]
    r = run_wf(df, interact_feats_1, "R15B: +oi_accel +ret_accel (14f)", regime_df,
               cs_feats=feats)  # only CS-rank original feats
    if r:
        show(r)
        results.append(r)

    interact_feats_2 = feats + ["oi_accel", "ret_accel", "taker_accel"]
    r = run_wf(df, interact_feats_2, "R15B: +accel trio (15f)", regime_df,
               cs_feats=feats)
    if r:
        show(r)
        results.append(r)

    interact_feats_3 = feats + ["oi_ret_div", "ls_oi_interact"]
    r = run_wf(df, interact_feats_3, "R15B: +divergence feats (14f)", regime_df,
               cs_feats=feats)
    if r:
        show(r)
        results.append(r)

    interact_feats_all = feats + ["oi_accel", "ret_accel", "taker_accel",
                                  "oi_ret_div", "ls_oi_interact"]
    r = run_wf(df, interact_feats_all, "R15B: all interactions (17f)", regime_df,
               cs_feats=feats)
    if r:
        show(r)
        results.append(r)

    # ═══════════════════════════════════════════════
    # R15C: DART (dropout in boosting)
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  R15C — DART Mode (Dropout Regularization)")
    print("═" * 70)

    for drop_rate in [0.1, 0.2]:
        params = {
            "objective": "regression", "metric": "mse",
            "boosting_type": "dart",
            "learning_rate": 0.03, "num_leaves": 63,
            "min_child_samples": 100,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "lambda_l2": 1.0,
            "drop_rate": drop_rate,
            "verbose": -1, "n_jobs": -1,
        }
        r = run_wf(df, feats, f"R15C: DART drop={drop_rate}", regime_df, lgb_params=params)
        if r:
            show(r)
            results.append(r)

    # ═══════════════════════════════════════════════
    # R15D: Target Winsorization
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  R15D — Target Winsorization")
    print("═" * 70)

    for clip_pct in [0.01, 0.05]:
        def winsorize_target(d, pct=clip_pct):
            low = d["target_rank"].quantile(pct)
            high = d["target_rank"].quantile(1 - pct)
            d = d.copy()
            d["target_rank"] = d["target_rank"].clip(low, high)
            return d
        r = run_wf(df, feats, f"R15D: winsorize {clip_pct*100:.0f}%", regime_df,
                   target_transform=winsorize_target)
        if r:
            show(r)
            results.append(r)

    # ═══════════════════════════════════════════════
    # R15E: Time-Weighted Training
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  R15E — Time-Weighted Training")
    print("═" * 70)

    for decay in [0.5, 0.8]:
        def time_weight_fn(train_full, train_clean, decay_rate=decay):
            """Exponential time decay: recent data gets higher weight."""
            # Get timestamps from the full training data that match clean indices
            ts = train_full.loc[train_clean.index, "timestamp"] if "timestamp" in train_full.columns else None
            if ts is None:
                return np.ones(len(train_clean))
            ts_min = ts.min()
            ts_max = ts.max()
            total_range = (ts_max - ts_min).total_seconds() + 1
            time_frac = (ts - ts_min).dt.total_seconds() / total_range
            weights = decay_rate + (1 - decay_rate) * time_frac
            return weights.values

        r = run_wf(df, feats, f"R15E: time-weight decay={decay}", regime_df,
                   sample_weight_fn=time_weight_fn)
        if r:
            show(r)
            results.append(r)

    # ═══════════════════════════════════════════════
    # R15F: Extra Trees
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  R15F — Extra Trees Mode")
    print("═" * 70)

    params_et = {
        "objective": "regression", "metric": "mse",
        "learning_rate": 0.03, "num_leaves": 63,
        "min_child_samples": 100,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "lambda_l2": 1.0,
        "extra_trees": True,
        "verbose": -1, "n_jobs": -1,
    }
    r = run_wf(df, feats, "R15F: Extra Trees", regime_df, lgb_params=params_et)
    if r:
        show(r)
        results.append(r)

    # ═══════════════════════════════════════════════
    # R15G: Bagging Variations
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  R15G — Bagging Variations")
    print("═" * 70)

    bagging_configs = [
        (0.7, 0.7, "sub=0.7 col=0.7"),
        (0.9, 0.9, "sub=0.9 col=0.9"),
        (0.6, 0.8, "sub=0.6 col=0.8"),
        (0.8, 0.6, "sub=0.8 col=0.6"),
        (1.0, 0.8, "sub=1.0 col=0.8"),
    ]
    for sub, col, suffix in bagging_configs:
        params = {
            "objective": "regression", "metric": "mse",
            "learning_rate": 0.03, "num_leaves": 63,
            "min_child_samples": 100,
            "subsample": sub, "colsample_bytree": col,
            "lambda_l2": 1.0,
            "verbose": -1, "n_jobs": -1,
        }
        r = run_wf(df, feats, f"R15G: {suffix}", regime_df, lgb_params=params)
        if r:
            show(r)
            results.append(r)

    # ═══════════════════════════════════════════════
    # R15H: Heterogeneous Ensemble
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  R15H — Heterogeneous Ensemble (diverse configs)")
    print("═" * 70)

    # 5 different configs for 5 seeds — diversity in the ensemble
    het_configs = [
        {"objective": "regression", "metric": "mse", "learning_rate": 0.03,
         "num_leaves": 63, "min_child_samples": 100, "subsample": 0.8,
         "colsample_bytree": 0.8, "lambda_l2": 1.0, "verbose": -1, "n_jobs": -1},
        {"objective": "regression", "metric": "mse", "learning_rate": 0.02,
         "num_leaves": 63, "min_child_samples": 100, "subsample": 0.8,
         "colsample_bytree": 0.8, "lambda_l2": 2.0, "verbose": -1, "n_jobs": -1},
        {"objective": "regression", "metric": "mse", "learning_rate": 0.04,
         "num_leaves": 47, "min_child_samples": 100, "subsample": 0.7,
         "colsample_bytree": 0.7, "lambda_l2": 1.0, "verbose": -1, "n_jobs": -1},
        {"objective": "regression", "metric": "mse", "learning_rate": 0.03,
         "num_leaves": 95, "min_child_samples": 75, "subsample": 0.9,
         "colsample_bytree": 0.9, "lambda_l2": 0.5, "verbose": -1, "n_jobs": -1},
        {"objective": "regression", "metric": "mse", "learning_rate": 0.03,
         "num_leaves": 63, "min_child_samples": 100, "subsample": 0.8,
         "colsample_bytree": 0.6, "lambda_l2": 3.0, "verbose": -1, "n_jobs": -1},
    ]
    r = run_het_ensemble(df, feats, het_configs, "R15H: het ensemble (5 configs)", regime_df)
    if r:
        show(r)
        results.append(r)

    # ═══════════════════════════════════════════════
    # R15I: BONUS — best interaction feats + best hyperparams combo
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  R15I — Combo: Best Interactions + Best Hyperparams")
    print("═" * 70)

    # Try oi_accel + ret_accel with hyperparams that did well in R15A
    # We'll test after R15A results are known, but run all combos now
    for suffix, lr, l2 in [("prod", 0.03, 1.0), ("lr02_l22", 0.02, 2.0),
                           ("lr04", 0.04, 1.0)]:
        params = {
            "objective": "regression", "metric": "mse",
            "learning_rate": lr, "num_leaves": 63,
            "min_child_samples": 100,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "lambda_l2": l2,
            "verbose": -1, "n_jobs": -1,
        }
        for feat_name, feat_list in [
            ("12f+accel", feats + ["oi_accel", "ret_accel"]),
            ("12f+div", feats + ["oi_ret_div", "ls_oi_interact"]),
        ]:
            r = run_wf(df, feat_list, f"R15I: {feat_name} {suffix}", regime_df,
                       lgb_params=params, cs_feats=feats)
            if r:
                show(r)
                results.append(r)

    # ═══════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  R15 — FINAL SUMMARY (sorted by Sharpe)")
    print("═" * 70)

    results.sort(key=lambda x: -x["sharpe"])
    print(f"\n  {'Config':<50s} {'Sh':>6s} {'Wr%':>7s} {'WM':>6s} {'Eq':>7s} {'IC':>6s}")
    print(f"  {'-'*50} {'-'*6} {'-'*7} {'-'*6} {'-'*7} {'-'*6}")

    for r in results:
        ic_str = f"{r.get('mean_ic_test', 0):.3f}" if r.get("mean_ic_test") else "—"
        wm_str = f"{r['win_months']}/{r['total_months']}"
        print(f"  {r['name']:<50s} {r['sharpe']:>+6.2f} {r['worst_m']*100:>+7.1f} "
              f"{wm_str:>6s} ${r['equity']:>6.0f} {ic_str:>6s}")

    elapsed = time.time() - t0
    print(f"\n  ⏱  Total time: {elapsed/60:.1f} min")

    if results:
        top = results[0]
        wm = f"{top['win_months']}/{top['total_months']}"
        print(f"\n  🏆 WINNER: {top['name']}")
        print(f"     Sh={top['sharpe']:.2f}, Wr={top['worst_m']*100:.1f}%, WM={wm}, Eq=${top['equity']:.0f}")

        if top["sharpe"] > 4.85:
            print(f"     ✅ BEATS R13 prod (Sh=4.81) — consider deployment!")
        else:
            print(f"     📊 R13 prod (Sh=4.81) remains optimal")


if __name__ == "__main__":
    main()
