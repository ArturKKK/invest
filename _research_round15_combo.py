#!/usr/bin/env python3
"""
Research Round 15.5 — Combo Tests of R15 Winners.

Top 3 from R15:
  - Extra Trees: Sh=4.93 (but Wr=-10.1%, WM=11/13 — risky)
  - Winsorize 1%: Sh=4.87 (Wr=+1.2%, WM=13/13 — safe)
  - lr=0.02: Sh=4.85 (Wr=+2.7%, WM=13/13 — safe)

Now testing combos + deeper exploration around winners.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
import warnings
import time
warnings.filterwarnings("ignore")

from _research_round7 import (
    SYM_35, WINDOWS, cs_rank,
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

            if p.get("boosting_type") == "dart":
                model = lgb.train(p, dtrain, num_boost_round=200,
                                  valid_sets=[dval], callbacks=[lgb.log_evaluation(-1)])
            elif p.get("extra_trees"):
                model = lgb.train(p, dtrain, num_boost_round=N_ROUNDS,
                                  valid_sets=[dval], callbacks=callbacks)
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


def winsorize_target_1pct(d):
    low = d["target_rank"].quantile(0.01)
    high = d["target_rank"].quantile(0.99)
    d = d.copy()
    d["target_rank"] = d["target_rank"].clip(low, high)
    return d


def winsorize_target_2pct(d):
    low = d["target_rank"].quantile(0.02)
    high = d["target_rank"].quantile(0.98)
    d = d.copy()
    d["target_rank"] = d["target_rank"].clip(low, high)
    return d


def main():
    t0 = time.time()
    print("=" * 70)
    print("  RESEARCH ROUND 15.5 — Combo Tests of R15 Winners")
    print("  Baseline: R13 prod → Sh=4.81, WM=13/13")
    print("  R15 winners: ExtraTrees(4.93), Winsorize1%(4.87), lr=0.02(4.85)")
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
    # BASELINE check
    # ═══════════════════════════════════════════════
    r = run_wf(df, feats, "Baseline: R13 prod", regime_df)
    if r: show(r); results.append(r)

    # ═══════════════════════════════════════════════
    # COMBO 1: lr=0.02 + winsorize 1%
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  COMBO 1: lr=0.02 + winsorize 1%")
    print("═" * 70)
    params = {
        "objective": "regression", "metric": "mse",
        "learning_rate": 0.02, "num_leaves": 63,
        "min_child_samples": 100,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "lambda_l2": 1.0,
        "verbose": -1, "n_jobs": -1,
    }
    r = run_wf(df, feats, "Combo1: lr=0.02 + winsor1%", regime_df,
               lgb_params=params, target_transform=winsorize_target_1pct)
    if r: show(r); results.append(r)

    # ═══════════════════════════════════════════════
    # COMBO 2: Extra Trees + winsorize 1%
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  COMBO 2: Extra Trees + winsorize 1%")
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
    r = run_wf(df, feats, "Combo2: ExtraTrees + winsor1%", regime_df,
               lgb_params=params_et, target_transform=winsorize_target_1pct)
    if r: show(r); results.append(r)

    # ═══════════════════════════════════════════════
    # COMBO 3: Extra Trees + lr=0.02
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  COMBO 3: Extra Trees + lr=0.02")
    print("═" * 70)
    params_et_lr = {
        "objective": "regression", "metric": "mse",
        "learning_rate": 0.02, "num_leaves": 63,
        "min_child_samples": 100,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "lambda_l2": 1.0,
        "extra_trees": True,
        "verbose": -1, "n_jobs": -1,
    }
    r = run_wf(df, feats, "Combo3: ExtraTrees + lr=0.02", regime_df,
               lgb_params=params_et_lr)
    if r: show(r); results.append(r)

    # ═══════════════════════════════════════════════
    # COMBO 4: All 3 winners combined
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  COMBO 4: Extra Trees + lr=0.02 + winsorize 1%")
    print("═" * 70)
    r = run_wf(df, feats, "Combo4: ET + lr=0.02 + winsor1%", regime_df,
               lgb_params=params_et_lr, target_transform=winsorize_target_1pct)
    if r: show(r); results.append(r)

    # ═══════════════════════════════════════════════
    # COMBO 5: Extra Trees + L2 tuning (fix drawdown)
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  COMBO 5: Extra Trees + L2 variations")
    print("═" * 70)
    for l2 in [2.0, 3.0, 5.0]:
        params = {
            "objective": "regression", "metric": "mse",
            "learning_rate": 0.03, "num_leaves": 63,
            "min_child_samples": 100,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "lambda_l2": l2,
            "extra_trees": True,
            "verbose": -1, "n_jobs": -1,
        }
        r = run_wf(df, feats, f"Combo5: ET + L2={l2}", regime_df, lgb_params=params)
        if r: show(r); results.append(r)

    # ═══════════════════════════════════════════════
    # COMBO 6: Extra Trees + mc tuning (fix drawdown)
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  COMBO 6: Extra Trees + min_child tuning")
    print("═" * 70)
    for mc in [150, 200, 300]:
        params = {
            "objective": "regression", "metric": "mse",
            "learning_rate": 0.03, "num_leaves": 63,
            "min_child_samples": mc,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "lambda_l2": 1.0,
            "extra_trees": True,
            "verbose": -1, "n_jobs": -1,
        }
        r = run_wf(df, feats, f"Combo6: ET + mc={mc}", regime_df, lgb_params=params)
        if r: show(r); results.append(r)

    # ═══════════════════════════════════════════════
    # COMBO 7: Deep lr exploration (0.015, 0.025)
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  COMBO 7: Fine lr sweep near 0.02")
    print("═" * 70)
    for lr in [0.015, 0.025]:
        params = {
            "objective": "regression", "metric": "mse",
            "learning_rate": lr, "num_leaves": 63,
            "min_child_samples": 100,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "lambda_l2": 1.0,
            "verbose": -1, "n_jobs": -1,
        }
        r = run_wf(df, feats, f"Combo7: lr={lr}", regime_df, lgb_params=params)
        if r: show(r); results.append(r)

    # ═══════════════════════════════════════════════
    # COMBO 8: Winsorize 2% (between 1% and 5%)
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  COMBO 8: Winsorize 2%")
    print("═" * 70)
    r = run_wf(df, feats, "Combo8: winsor2%", regime_df,
               target_transform=winsorize_target_2pct)
    if r: show(r); results.append(r)

    r = run_wf(df, feats, "Combo8b: lr=0.02 + winsor2%", regime_df,
               lgb_params={
                   "objective": "regression", "metric": "mse",
                   "learning_rate": 0.02, "num_leaves": 63,
                   "min_child_samples": 100,
                   "subsample": 0.8, "colsample_bytree": 0.8,
                   "lambda_l2": 1.0,
                   "verbose": -1, "n_jobs": -1,
               }, target_transform=winsorize_target_2pct)
    if r: show(r); results.append(r)

    # ═══════════════════════════════════════════════
    # COMBO 9: ExtraTrees + lr=0.02 + higher regularization
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  COMBO 9: ExtraTrees + lr=0.02 + L2=2.0")
    print("═" * 70)
    params = {
        "objective": "regression", "metric": "mse",
        "learning_rate": 0.02, "num_leaves": 63,
        "min_child_samples": 100,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "lambda_l2": 2.0,
        "extra_trees": True,
        "verbose": -1, "n_jobs": -1,
    }
    r = run_wf(df, feats, "Combo9: ET + lr=0.02 + L2=2", regime_df, lgb_params=params)
    if r: show(r); results.append(r)

    params = {
        "objective": "regression", "metric": "mse",
        "learning_rate": 0.02, "num_leaves": 63,
        "min_child_samples": 150,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "lambda_l2": 2.0,
        "extra_trees": True,
        "verbose": -1, "n_jobs": -1,
    }
    r = run_wf(df, feats, "Combo9b: ET + lr=0.02 + L2=2 + mc=150", regime_df,
               lgb_params=params)
    if r: show(r); results.append(r)

    # ═══════════════════════════════════════════════
    # COMBO 10: The Ultimate — all best techniques
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  COMBO 10: ExtraTrees + lr=0.02 + L2=2 + winsor1%")
    print("═" * 70)
    params = {
        "objective": "regression", "metric": "mse",
        "learning_rate": 0.02, "num_leaves": 63,
        "min_child_samples": 100,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "lambda_l2": 2.0,
        "extra_trees": True,
        "verbose": -1, "n_jobs": -1,
    }
    r = run_wf(df, feats, "Combo10: ET+lr02+L22+winsor1", regime_df,
               lgb_params=params, target_transform=winsorize_target_1pct)
    if r: show(r); results.append(r)

    # ═══════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  R15.5 — FINAL SUMMARY (sorted by Sharpe)")
    print("═" * 70)

    results.sort(key=lambda x: -x["sharpe"])
    print(f"\n  {'Config':<50s} {'Sh':>6s} {'Wr%':>7s} {'WM':>6s} {'Eq':>7s} {'IC':>6s}")
    print(f"  {'-'*50} {'-'*6} {'-'*7} {'-'*6} {'-'*7} {'-'*6}")

    for r in results:
        ic_str = f"{r.get('mean_ic_test', 0):.3f}" if r.get("mean_ic_test") else "—"
        wm_str = f"{r['win_months']}/{r['total_months']}"
        print(f"  {r['name']:<50s} {r['sharpe']:>+6.2f} {r['worst_m']*100:>+7.1f} "
              f"{wm_str:>6s} ${r['equity']:>6.0f} {ic_str:>6s}")

    # Mark configs that are strictly better (higher Sharpe + WM>=13/13)
    print(f"\n  🎯 Production-safe winners (WM=13/13 + Sh>4.81):")
    prod_safe = [r for r in results if r["win_months"] == 13 and r["sharpe"] > 4.81]
    if prod_safe:
        for r in prod_safe:
            print(f"     {r['name']}: Sh={r['sharpe']:.2f}, Wr={r['worst_m']*100:.1f}%, Eq=${r['equity']:.0f}")
    else:
        print(f"     None — R13 prod (Sh=4.81) remains optimal with WM=13/13")

    print(f"\n  📈 High-risk high-reward (Sh>4.81, any WM):")
    high_sh = [r for r in results if r["sharpe"] > 4.81]
    if high_sh:
        for r in high_sh:
            wm_str = f"{r['win_months']}/{r['total_months']}"
            print(f"     {r['name']}: Sh={r['sharpe']:.2f}, WM={wm_str}, Wr={r['worst_m']*100:.1f}%")

    elapsed = time.time() - t0
    print(f"\n  ⏱  Total time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
