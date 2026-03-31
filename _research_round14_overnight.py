#!/usr/bin/env python3
"""
Research Round 14 — Overnight Experiments.

Goal: thorough robustness tests + exploration of new directions.

R14A: Extended walk-forward (5 sliding windows instead of 3)
R14B: Alternative target horizons (6h, 8h, 24h with R13 prod config)
R14C: Rebalance frequency sweep (6h, 8h, 12h, 24h)
R14D: Position count sweep (n_long/n_short combos)
R14E: Temporal decay analysis — IC by quarter
R14F: XGBoost comparison (same 12f, similar hyperparams)
R14G: Bootstrap confidence interval for Sharpe
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
import warnings
import time
import json
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

PROD_PARAMS = {
    "objective": "regression", "metric": "mse",
    "learning_rate": 0.03, "num_leaves": 63,
    "min_child_samples": 100,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "lambda_l2": 1.0,
    "verbose": -1, "n_jobs": -1,
}

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

# Extended windows for R14A
WINDOWS_5 = [
    {"name": "W0",
     "train_end": "2024-01-01",
     "val_start": "2024-01-01", "val_end": "2024-04-30",
     "test_start": "2024-05-15", "test_end": "2024-09-30"},
    {"name": "W1",
     "train_end": "2024-06-01",
     "val_start": "2024-06-01", "val_end": "2024-09-30",
     "test_start": "2024-10-15", "test_end": "2025-01-31"},
    {"name": "W2",
     "train_end": "2025-01-01",
     "val_start": "2025-01-01", "val_end": "2025-04-30",
     "test_start": "2025-05-15", "test_end": "2025-08-31"},
    {"name": "W3",
     "train_end": "2025-07-01",
     "val_start": "2025-07-01", "val_end": "2025-10-31",
     "test_start": "2025-11-15", "test_end": "2026-03-17"},
    {"name": "W4",
     "train_end": "2025-03-01",
     "val_start": "2025-03-01", "val_end": "2025-06-30",
     "test_start": "2025-07-15", "test_end": "2025-10-31"},
]


def cs_rank_inplace(df, feats):
    df = df.copy()
    for feat in feats:
        if feat in df.columns:
            df[feat] = df.groupby("timestamp")[feat].rank(pct=True) - 0.5
    return df


def run_wf(df, feats, windows, cfg, name, regime_df, fwd_col="fwd_ret_12h",
           lgb_params=None, horizon=12):
    """Full walk-forward with given windows and config."""
    params = lgb_params or PROD_PARAMS
    all_preds = []
    all_ics = []

    for seed in SEEDS:
        seed_preds = []
        for w in windows:
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
                d["target_rank"] = d.groupby("timestamp")[fwd_col].rank(pct=True) - 0.5

            train_c = train[feats + ["target_rank"]].dropna()
            val_c   = val[feats + ["target_rank"]].dropna()

            dtrain = lgb.Dataset(train_c[feats], label=train_c["target_rank"])
            dval   = lgb.Dataset(val_c[feats],   label=val_c["target_rank"])

            p = {**params, "seed": seed}
            model = lgb.train(
                p, dtrain, num_boost_round=N_ROUNDS,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                           lgb.log_evaluation(-1)],
            )

            test_c = test[feats + ["target_rank", "timestamp", "symbol"]].dropna()
            test_pred = model.predict(test_c[feats])
            ic_test = stats.spearmanr(test_pred, test_c["target_rank"])[0]
            all_ics.append(ic_test)

            fwd_data = test[["timestamp", "symbol", fwd_col]].rename(
                columns={fwd_col: "fwd_ret"}).dropna()
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

    r = eval_config(simulate(ens, regime_df, horizon, cfg), horizon, name, LEVERAGE, CAPITAL)
    if r:
        r["mean_ic_test"] = round(np.mean(all_ics), 4)
        r["std_ic_test"] = round(np.std(all_ics), 4)
        r["n_windows"] = len(windows)
    return r


# ═══════════════════════════════════════════════════════
# R14A: Extended Walk-Forward (5 windows)
# ═══════════════════════════════════════════════════════

def run_r14a(df, feats, regime_df):
    print("\n" + "═" * 70)
    print("  R14A — Extended Walk-Forward (5 windows vs 3)")
    print("═" * 70)

    r3 = run_wf(df, feats, WINDOWS, CFG_BASE, "R14A: 3 windows (original)", regime_df)
    if r3:
        show(r3)

    r5 = run_wf(df, feats, WINDOWS_5, CFG_BASE, "R14A: 5 windows (extended)", regime_df)
    if r5:
        show(r5)

    # Each window separately
    for w in WINDOWS_5:
        r = run_wf(df, feats, [w], CFG_BASE, f"R14A: {w['name']} only", regime_df)
        if r:
            print(f"    {w['name']}: Sh={r['sharpe']:.2f}, IC={r.get('mean_ic_test', 0):.4f}, "
                  f"Wr={r['worst_m']*100:.1f}%")

    return [r3, r5]


# ═══════════════════════════════════════════════════════
# R14B: Alternative target horizons
# ═══════════════════════════════════════════════════════

def run_r14b(df, feats, regime_df):
    print("\n" + "═" * 70)
    print("  R14B — Alternative Target Horizons (6h, 8h, 24h)")
    print("═" * 70)

    results = []

    # First check which fwd_ret columns exist
    for h in [6, 8, 12, 24]:
        fwd_col = f"fwd_ret_{h}h"
        if fwd_col not in df.columns:
            # Create it
            print(f"    Creating {fwd_col}...")
            for sym in SYM_35:
                mask = df["symbol"] == sym
                closes = df.loc[mask, "close"]
                if h == 6:
                    df.loc[mask, fwd_col] = closes.shift(-6) / closes - 1
                elif h == 8:
                    df.loc[mask, fwd_col] = closes.shift(-8) / closes - 1
                elif h == 24:
                    df.loc[mask, fwd_col] = closes.shift(-24) / closes - 1

        cfg_h = {**CFG_BASE, "rebal_hours": h}
        r = run_wf(df, feats, WINDOWS, cfg_h,
                   f"R14B: target={h}h rebal={h}h", regime_df,
                   fwd_col=fwd_col, horizon=h)
        if r:
            show(r)
            results.append(r)

    return results


# ═══════════════════════════════════════════════════════
# R14C: Rebalance frequency (with 12h target)
# ═══════════════════════════════════════════════════════

def run_r14c(df, feats, regime_df):
    print("\n" + "═" * 70)
    print("  R14C — Rebalance Frequency Sweep (12h target)")
    print("═" * 70)

    results = []
    for rh in [6, 8, 12, 18, 24]:
        cfg_rh = {**CFG_BASE, "rebal_hours": rh}
        r = run_wf(df, feats, WINDOWS, cfg_rh,
                   f"R14C: rebal={rh}h (target=12h)", regime_df, horizon=12)
        if r:
            show(r)
            results.append(r)
    return results


# ═══════════════════════════════════════════════════════
# R14D: Position count sweep
# ═══════════════════════════════════════════════════════

def run_r14d(df, feats, regime_df):
    print("\n" + "═" * 70)
    print("  R14D — Position Count Sweep")
    print("═" * 70)

    results = []
    combos = [
        (4, 2), (6, 3), (8, 4), (10, 5), (6, 6), (8, 3), (4, 4),
    ]
    for nl, ns in combos:
        cfg_pos = {**CFG_BASE, "n_long": nl, "n_short": ns}
        r = run_wf(df, feats, WINDOWS, cfg_pos,
                   f"R14D: n_long={nl} n_short={ns}", regime_df)
        if r:
            show(r)
            results.append(r)
    return results


# ═══════════════════════════════════════════════════════
# R14E: Temporal IC decay analysis
# ═══════════════════════════════════════════════════════

def run_r14e(df, feats):
    print("\n" + "═" * 70)
    print("  R14E — Temporal IC Decay Analysis (by quarter)")
    print("═" * 70)

    # Train on all 3 windows, collect per-timestamp ICs from test periods
    all_ts_ics = []

    for w in WINDOWS:
        train = df[df["timestamp"] < pd.Timestamp(w["train_end"], tz="UTC")].copy()
        val   = df[(df["timestamp"] >= pd.Timestamp(w["val_start"], tz="UTC")) &
                   (df["timestamp"] <  pd.Timestamp(w["val_end"],   tz="UTC"))].copy()
        test  = df[(df["timestamp"] >= pd.Timestamp(w["test_start"], tz="UTC")) &
                   (df["timestamp"] <= pd.Timestamp(w["test_end"],   tz="UTC"))].copy()

        train = cs_rank_inplace(train, feats)
        val   = cs_rank_inplace(val, feats)
        test  = cs_rank_inplace(test, feats)
        for d in [train, val, test]:
            d["target_rank"] = d.groupby("timestamp")["fwd_ret_12h"].rank(pct=True) - 0.5

        train_c = train[feats + ["target_rank"]].dropna()
        val_c   = val[feats + ["target_rank"]].dropna()
        test_c  = test[feats + ["target_rank", "timestamp", "symbol"]].dropna()

        # 5-seed ensemble
        all_preds = []
        for seed in SEEDS:
            p = {**PROD_PARAMS, "seed": seed}
            dtrain = lgb.Dataset(train_c[feats], label=train_c["target_rank"])
            dval   = lgb.Dataset(val_c[feats],   label=val_c["target_rank"])
            model = lgb.train(p, dtrain, num_boost_round=N_ROUNDS,
                              valid_sets=[dval],
                              callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                                         lgb.log_evaluation(-1)])
            all_preds.append(model.predict(test_c[feats]))

        ensemble_pred = np.mean(all_preds, axis=0)
        test_with_pred = test_c.copy()
        test_with_pred["pred"] = ensemble_pred

        for ts, grp in test_with_pred.groupby("timestamp"):
            if len(grp) >= 10:
                ic = stats.spearmanr(grp["pred"], grp["target_rank"])[0]
                all_ts_ics.append({"timestamp": ts, "ic": ic, "window": w["name"]})

    ic_df = pd.DataFrame(all_ts_ics)
    ic_df["quarter"] = ic_df["timestamp"].dt.to_period("Q")
    ic_df["month"] = ic_df["timestamp"].dt.to_period("M")

    # By quarter
    print("\n    IC by Quarter:")
    q_stats = ic_df.groupby("quarter")["ic"].agg(["mean", "median", "std", "count"])
    for q, row in q_stats.iterrows():
        pct_pos = (ic_df[ic_df["quarter"] == q]["ic"] > 0).mean() * 100
        trend = "📈" if row["mean"] > 0.05 else ("📊" if row["mean"] > 0.02 else "📉")
        print(f"    {trend} {q}: mean={row['mean']:.4f} med={row['median']:.4f} "
              f"std={row['std']:.4f} pos={pct_pos:.0f}% (n={row['count']:.0f})")

    # By month
    print("\n    IC by Month:")
    m_stats = ic_df.groupby("month")["ic"].agg(["mean", "count"])
    for m, row in m_stats.iterrows():
        bar = "█" * max(1, int(row["mean"] * 100))
        print(f"    {m}: IC={row['mean']:+.4f} {bar}")

    # Trend detection: is IC getting better or worse?
    ic_df["ts_num"] = (ic_df["timestamp"] - ic_df["timestamp"].min()).dt.total_seconds()
    slope, intercept, r_value, p_value, _ = stats.linregress(ic_df["ts_num"], ic_df["ic"])
    print(f"\n    Temporal trend: slope={slope:.2e}, R²={r_value**2:.4f}, p={p_value:.4f}")
    if slope > 0 and p_value < 0.05:
        print("    📈 IC is IMPROVING over time")
    elif slope < 0 and p_value < 0.05:
        print("    📉 IC is DECLINING over time (⚠️ potential edge decay)")
    else:
        print("    📊 No significant temporal trend (stable)")

    return ic_df


# ═══════════════════════════════════════════════════════
# R14F: XGBoost comparison
# ═══════════════════════════════════════════════════════

def run_r14f(df, feats, regime_df):
    print("\n" + "═" * 70)
    print("  R14F — XGBoost Comparison")
    print("═" * 70)

    try:
        import xgboost as xgb
    except ImportError:
        print("    ⚠️ xgboost not installed, skipping")
        return None

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

            dtrain = xgb.DMatrix(train_c[feats], label=train_c["target_rank"])
            dval   = xgb.DMatrix(val_c[feats], label=val_c["target_rank"])

            params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "learning_rate": 0.03,
                "max_depth": 6,
                "min_child_weight": 100,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "lambda": 1.0,
                "seed": seed,
                "verbosity": 0,
            }
            model = xgb.train(
                params, dtrain, num_boost_round=N_ROUNDS,
                evals=[(dval, "val")],
                early_stopping_rounds=EARLY_STOP,
                verbose_eval=False,
            )

            test_c = test[feats + ["target_rank", "timestamp", "symbol"]].dropna()
            dtest = xgb.DMatrix(test_c[feats])
            test_pred = model.predict(dtest)
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
        print("    ❌ No predictions")
        return None

    combined = pd.concat(all_preds, ignore_index=True)
    ens = (combined.groupby(["timestamp", "symbol"])
           .agg(pred=("pred", "mean"), fwd_ret=("fwd_ret", "first"))
           .reset_index())

    r = eval_config(simulate(ens, regime_df, 12, CFG_BASE), 12,
                    "R14F: XGBoost 12f", LEVERAGE, CAPITAL)
    if r:
        r["mean_ic_test"] = round(np.mean(all_ics), 4)
        show(r)
    return r


# ═══════════════════════════════════════════════════════
# R14G: Bootstrap confidence interval for Sharpe
# ═══════════════════════════════════════════════════════

def run_r14g(df, feats, regime_df, n_bootstrap=1000):
    print("\n" + "═" * 70)
    print("  R14G — Bootstrap Sharpe CI (1000 resamples)")
    print("═" * 70)

    # Get monthly returns from R13 prod config
    r = run_wf(df, feats, WINDOWS, CFG_BASE, "R14G base", regime_df)
    if not r or "monthly" not in r:
        print("    ❌ No monthly data")
        return None

    monthly = r["monthly"].values
    n = len(monthly)
    observed_sharpe = r["sharpe"]
    print(f"    Observed Sharpe: {observed_sharpe:.2f} ({n} months)")

    # Bootstrap monthly returns
    rng = np.random.RandomState(42)
    boot_sharpes = []
    for _ in range(n_bootstrap):
        sample = rng.choice(monthly, size=n, replace=True)
        ppy = 8760 / 12  # approximate
        sh = np.mean(sample) / (np.std(sample) + 1e-10) * np.sqrt(12)  # annualize from monthly
        boot_sharpes.append(sh)

    boot_sharpes = np.array(boot_sharpes)
    ci_5 = np.percentile(boot_sharpes, 5)
    ci_25 = np.percentile(boot_sharpes, 25)
    ci_50 = np.percentile(boot_sharpes, 50)
    ci_75 = np.percentile(boot_sharpes, 75)
    ci_95 = np.percentile(boot_sharpes, 95)

    prob_positive = np.mean(boot_sharpes > 0) * 100
    prob_above_2 = np.mean(boot_sharpes > 2) * 100

    print(f"    Bootstrap Sharpe distribution (N={n_bootstrap}):")
    print(f"      5th pct:  {ci_5:.2f}")
    print(f"      25th pct: {ci_25:.2f}")
    print(f"      Median:   {ci_50:.2f}")
    print(f"      75th pct: {ci_75:.2f}")
    print(f"      95th pct: {ci_95:.2f}")
    print(f"    P(Sharpe > 0) = {prob_positive:.1f}%")
    print(f"    P(Sharpe > 2) = {prob_above_2:.1f}%")
    print(f"    90% CI: [{ci_5:.2f}, {ci_95:.2f}]")

    return {
        "observed": observed_sharpe,
        "ci_5": ci_5, "ci_95": ci_95,
        "median": ci_50,
        "p_positive": prob_positive,
    }


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70)
    print("  RESEARCH ROUND 14 — Overnight Experiments")
    print("  R13 Prod Config: 12f, nl=63, lr=0.03, L2=1.0")
    print("=" * 70)

    print("\n📊 Loading data...")
    ohlcv  = load_ohlcv()
    ohlcv  = ohlcv[ohlcv["symbol"].isin(SYM_35)]
    derivs = load_derivatives()
    df     = build_features_minimal(ohlcv, derivs)
    feats  = [f for f in FEATURES_12 if f in df.columns]
    print(f"   df: ({len(df):,}, {len(df.columns)})")
    print(f"   features ({len(feats)}): {feats}")
    print(f"   date range: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

    regime_df = compute_regime(df)

    all_results = {}

    # R14A: Extended walk-forward
    r14a = run_r14a(df, feats, regime_df)
    all_results["R14A"] = r14a

    # R14B: Alternative targets
    r14b = run_r14b(df, feats, regime_df)
    all_results["R14B"] = r14b

    # R14C: Rebalance frequency
    r14c = run_r14c(df, feats, regime_df)
    all_results["R14C"] = r14c

    # R14D: Position count sweep
    r14d = run_r14d(df, feats, regime_df)
    all_results["R14D"] = r14d

    # R14E: Temporal decay
    r14e = run_r14e(df, feats)
    all_results["R14E"] = "see above"

    # R14F: XGBoost
    r14f = run_r14f(df, feats, regime_df)
    all_results["R14F"] = r14f

    # R14G: Bootstrap CI
    r14g = run_r14g(df, feats, regime_df)
    all_results["R14G"] = r14g

    # ═══════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════
    elapsed = time.time() - t0
    print("\n" + "═" * 70)
    print("  R14 — FINAL SUMMARY")
    print("═" * 70)

    # Collect all show-able results
    all_r = []
    for key in ["R14A", "R14B", "R14C", "R14D"]:
        val = all_results.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            for r in val:
                if r and isinstance(r, dict) and "sharpe" in r:
                    all_r.append(r)
        elif isinstance(val, dict) and "sharpe" in val:
            all_r.append(val)

    if all_r:
        all_r.sort(key=lambda x: -x["sharpe"])
        print(f"\n  {'Config':<50s} {'Sh':>6s} {'Wr%':>7s} {'WM':>6s} {'Eq':>7s} {'IC':>6s}")
        print(f"  {'-'*50} {'-'*6} {'-'*7} {'-'*6} {'-'*7} {'-'*6}")
        for r in all_r:
            ic_str = f"{r.get('mean_ic_test', 0):.3f}" if r.get("mean_ic_test") else "—"
            wm_str = f"{r['win_months']}/{r['total_months']}"
            print(f"  {r['name']:<50s} {r['sharpe']:>+6.2f} {r['worst_m']*100:>+7.1f} "
                  f"{wm_str:>6s} ${r['equity']:>6.0f} {ic_str:>6s}")

    if r14g:
        print(f"\n  Bootstrap 90% CI for Sharpe: [{r14g['ci_5']:.2f}, {r14g['ci_95']:.2f}]")
        print(f"  P(Sharpe > 0) = {r14g['p_positive']:.1f}%")

    print(f"\n  ⏱  Total time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
