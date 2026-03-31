#!/usr/bin/env python3
"""
R13 — Quick combo test: merge R12's best hyperparams + feature pruning.

Tests:
  1. 12f pruned + min_child=200
  2. 12f pruned + min_child=200 + lr=0.03 + L2=1
  3. 12f pruned + L1=0.1
  4. 12f pruned + min_child=200 + L1=0.1
  5. 10f pruned + min_child=200
  6. 12f pruned + min_child=300

Baseline: R12F 12f pruned nl=63 (Sh=4.77, Wr=-3.6%, Eq=$5280)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
import warnings, time
warnings.filterwarnings("ignore")

from _research_round7 import (
    SYM_35, WINDOWS, FEATURES as FEATURES_14, cs_rank,
    compute_regime, simulate, eval_config, show,
)
from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal

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

FEATS_14 = None  # set in main after loading data
FEATS_12 = None
FEATS_10 = None


def cs_rank_inplace(df, feats):
    df = df.copy()
    for feat in feats:
        if feat in df.columns:
            df[feat] = df.groupby("timestamp")[feat].rank(pct=True) - 0.5
    return df


def run_config(df, feats, num_leaves, lr, min_child, reg_l1, reg_l2, name, regime_df):
    """Full walk-forward eval."""
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

            train = cs_rank_inplace(train, feats)
            val   = cs_rank_inplace(val, feats)
            test  = cs_rank_inplace(test, feats)

            for d in [train, val, test]:
                d["target_rank"] = d.groupby("timestamp")["fwd_ret_12h"].rank(pct=True) - 0.5

            train_c = train[feats + ["target_rank"]].dropna()
            val_c   = val[feats + ["target_rank"]].dropna()

            dtrain = lgb.Dataset(train_c[feats], label=train_c["target_rank"])
            dval   = lgb.Dataset(val_c[feats],   label=val_c["target_rank"])

            params = {
                "objective": "regression", "metric": "mse",
                "learning_rate": lr, "num_leaves": num_leaves,
                "min_child_samples": min_child,
                "subsample": 0.8, "colsample_bytree": 0.8,
                "lambda_l1": reg_l1, "lambda_l2": reg_l2,
                "verbose": -1, "n_jobs": -1, "seed": seed,
            }
            model = lgb.train(
                params, dtrain, num_boost_round=N_ROUNDS,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                           lgb.log_evaluation(-1)],
            )

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
    ensemble_preds = (combined.groupby(["timestamp", "symbol"])
                      .agg(pred=("pred", "mean"), fwd_ret=("fwd_ret", "first"))
                      .reset_index())

    r = eval_config(simulate(ensemble_preds, regime_df, 12, CFG_BASE),
                    12, name, LEVERAGE, CAPITAL)
    if r:
        r["mean_ic_test"] = round(np.mean(all_ics), 4)
    return r


def main():
    global FEATS_14, FEATS_12, FEATS_10
    t0 = time.time()

    print("=" * 70)
    print("  R13 — Combo Test: Best Hyperparams × Feature Pruning")
    print("=" * 70)

    print("\n📊 Loading data...")
    ohlcv  = load_ohlcv()
    ohlcv  = ohlcv[ohlcv["symbol"].isin(SYM_35)]
    derivs = load_derivatives()
    df     = build_features_minimal(ohlcv, derivs)
    print(f"   df: ({len(df):,}, {len(df.columns)})")

    regime_df = compute_regime(df)

    FEATS_14 = [f for f in FEATURES_14 if f in df.columns]
    FEATS_12 = [f for f in FEATS_14 if f not in ["dist_from_high_24h", "mom_z_12h"]]
    FEATS_10 = [f for f in FEATS_14 if f not in
                ["dist_from_high_24h", "mom_z_12h", "residual_24h", "oi_chg_12h"]]

    configs = [
        # (feats, nl, lr, min_child, l1, l2, name)
        # Baselines for comparison
        (FEATS_12, 63, 0.05, 100, 0.0, 0.0, "baseline: 12f nl=63 (R12F)"),
        (FEATS_14, 63, 0.05, 200, 0.0, 0.0, "baseline: 14f nl=63 mc=200 (R12E)"),

        # Combos: 12f pruned + hyperparams
        (FEATS_12, 63, 0.05, 200, 0.0, 0.0, "R13-1: 12f mc=200"),
        (FEATS_12, 63, 0.03, 200, 0.0, 1.0, "R13-2: 12f mc=200 lr=0.03 L2=1"),
        (FEATS_12, 63, 0.05, 200, 0.1, 0.0, "R13-3: 12f mc=200 L1=0.1"),
        (FEATS_12, 63, 0.03, 100, 0.0, 1.0, "R13-4: 12f lr=0.03 L2=1"),
        (FEATS_12, 63, 0.05, 300, 0.0, 0.0, "R13-5: 12f mc=300"),
        (FEATS_10, 63, 0.05, 200, 0.0, 0.0, "R13-6: 10f mc=200"),

        # nl sweep with mc=200
        (FEATS_12, 47, 0.05, 200, 0.0, 0.0, "R13-7: 12f nl=47 mc=200"),
        (FEATS_12, 127, 0.05, 200, 0.0, 0.0, "R13-8: 12f nl=127 mc=200"),
    ]

    results = []
    for feats, nl, lr, mc, l1, l2, name in configs:
        print(f"\n  ▶ {name}  ({len(feats)}f, nl={nl}, lr={lr}, mc={mc}, L1={l1}, L2={l2})")
        r = run_config(df, feats, nl, lr, mc, l1, l2, name, regime_df)
        if r:
            show(r)
            results.append(r)
        else:
            print("    ❌ No result")

    # Summary
    print("\n" + "═" * 70)
    print("  SUMMARY — R13 Combo Tests (sorted by Sharpe)")
    print("═" * 70)

    results.sort(key=lambda x: -x["sharpe"])
    print(f"\n  {'Config':<45s} {'Sh':>6s} {'Wr%':>7s} {'WM':>6s} {'Eq':>7s} {'IC':>6s}")
    print(f"  {'-'*45} {'-'*6} {'-'*7} {'-'*6} {'-'*7} {'-'*6}")

    for r in results:
        ic_str = f"{r.get('mean_ic_test', 0):.3f}" if r.get("mean_ic_test") else "—"
        wm_str = f"{r['win_months']}/{r['total_months']}"
        print(f"  {r['name']:<45s} {r['sharpe']:>+6.2f} {r['worst_m']*100:>+7.1f} "
              f"{wm_str:>6s} ${r['equity']:>6.0f} {ic_str:>6s}")

    elapsed = time.time() - t0
    print(f"\n  ⏱  Total time: {elapsed/60:.1f} min")

    if results:
        top = results[0]
        wm = f"{top['win_months']}/{top['total_months']}"
        print(f"\n  🏆 WINNER: {top['name']}")
        print(f"     Sh={top['sharpe']:.2f}, Wr={top['worst_m']*100:.1f}%, WM={wm}, Eq=${top['equity']:.0f}")


if __name__ == "__main__":
    main()
