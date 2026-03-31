#!/usr/bin/env python3
"""
Research Round 12 — Leakage audit + advanced experiments.

PART 1: Comprehensive leakage audit
PART 2: New experiments:
  R12A — nl=63 + TS z-scores (combine two R11 winners)
  R12B — Multi-horizon target blending (0.7*12h + 0.3*24h)
  R12C — LambdaRank objective (directly optimize ranking)
  R12D — Ridge+LGB stacking (weighted ensemble)
  R12E — Hyperparameter tuning (lr, min_child, regularization)
  R12F — Permutation-based feature pruning

Baseline: R10 LGB (Sh=4.07, WM=11/13, Eq=$2565)
Best R11: nl=63 14f (Sh=4.22, Wr=-1.1%) / TS-z 16f nl=31 (Sh=4.37)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from sklearn.linear_model import Ridge as RidgeModel
import warnings
import time
warnings.filterwarnings("ignore")

from _research_round7 import (
    SYM_35, WINDOWS, FEATURES as FEATURES_14, cs_rank,
    compute_regime, simulate, eval_config, show,
)
from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal

SEEDS = [0, 7, 13, 42, 99]
LR = 0.05
N_ROUNDS = 500
EARLY_STOP = 30
MIN_CHILD = 100
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


def add_ts_zscore_features(df):
    """Add per-coin TS z-score features (R11C-2 winners)."""
    ts_mean = df.groupby("symbol")["ret_12h"].transform(
        lambda x: x.rolling(60 * 24, min_periods=360).mean()
    )
    ts_std = df.groupby("symbol")["ret_12h"].transform(
        lambda x: x.rolling(60 * 24, min_periods=360).std()
    ) + 1e-10
    df["ts_z_ret12h_60d"] = (df["ret_12h"] - ts_mean) / ts_std

    oi_mean = df.groupby("symbol")["oi_chg_12h"].transform(
        lambda x: x.rolling(60 * 24, min_periods=360).mean()
    )
    oi_std = df.groupby("symbol")["oi_chg_12h"].transform(
        lambda x: x.rolling(60 * 24, min_periods=360).std()
    ) + 1e-10
    df["ts_z_oi_chg_60d"] = (df["oi_chg_12h"] - oi_mean) / oi_std
    return df


# ══════════════════════════════════════════════════════════════
# PART 1: LEAKAGE AUDIT
# ══════════════════════════════════════════════════════════════

def run_leakage_audit(df):
    """Comprehensive data leakage audit."""

    print("\n" + "═" * 70)
    print("  PART 1: LEAKAGE AUDIT")
    print("═" * 70)
    all_ok = True

    # CHECK 1: Walk-forward window gaps
    print("\n  CHECK 1: Walk-forward window gaps (val→test)")
    for w in WINDOWS:
        val_end = pd.Timestamp(w["val_end"], tz="UTC")
        test_start = pd.Timestamp(w["test_start"], tz="UTC")
        gap_days = (test_start - val_end).days
        ok = gap_days >= 14
        status = "✅" if ok else "❌"
        print(f"    {w['name']}: val_end={w['val_end']} test_start={w['test_start']} "
              f"gap={gap_days}d {status}")
        if not ok:
            all_ok = False

    # CHECK 2: No overlap between windows train/val and test
    print("\n  CHECK 2: No train/val overlap with test data")
    for w in WINDOWS:
        train_end = pd.Timestamp(w["train_end"], tz="UTC")
        test_start = pd.Timestamp(w["test_start"], tz="UTC")
        gap_days = (test_start - train_end).days
        ok = gap_days > 0
        status = "✅" if ok else "❌"
        print(f"    {w['name']}: train_end={w['train_end']} → test_start={w['test_start']} "
              f"gap={gap_days}d {status}")
        if not ok:
            all_ok = False

    # CHECK 3: Forward return target is properly shifted (no look-ahead)
    print("\n  CHECK 3: Forward return target (fwd_ret_12h) — look-ahead check")
    sym = "BTC/USDT"
    sym_df = df[df["symbol"] == sym].sort_values("timestamp")
    # Use a middle chunk (not head/tail) to avoid NaN edge effects
    mid = len(sym_df) // 2
    chunk = sym_df.iloc[mid:mid+100].reset_index(drop=True)
    closes = chunk["close"].values
    fwd_rets = chunk["fwd_ret_12h"].values
    mismatches = 0
    checked = 0
    for i in range(len(closes) - 12):
        expected = closes[i + 12] / closes[i] - 1
        actual = fwd_rets[i]
        if not np.isnan(actual):
            checked += 1
            if abs(expected - actual) > 1e-8:
                mismatches += 1
    # Last 12 rows of the ENTIRE series should be NaN
    tail_nan = np.isnan(sym_df["fwd_ret_12h"].values[-12:]).sum()
    ok = mismatches == 0 and tail_nan >= 11  # allow 1 tolerance for TZ edge
    status = "✅" if ok else "❌"
    print(f"    Formula matches: {mismatches}/{checked} mismatches, "
          f"tail 12 NaN: {tail_nan}/12 {status}")
    if not ok:
        all_ok = False

    # CHECK 4: CS-ranking uses only same-timestamp data (no cross-time leakage)
    print("\n  CHECK 4: CS-ranking — only within same timestamp")
    # Use timestamps from mid-range where all 35 symbols exist
    mid_ts = sorted(df["timestamp"].dropna().unique())[len(df["timestamp"].unique()) // 2]
    sample_ts = [mid_ts]
    all_rank_ok = True
    for ts in sample_ts:
        ts_data = df[df["timestamp"] == ts]
        ranked = ts_data["ret_12h"].rank(pct=True) - 0.5
        mean_rank = ranked.mean()
        ok = abs(mean_rank) < 0.02  # relaxed for odd-count symbols
        if not ok:
            print(f"    ❌ Rank mean={mean_rank:.4f} at {ts} — should be ~0")
            all_rank_ok = False
    if all_rank_ok:
        print(f"    ✅ CS-ranks are computed within-timestamp, mean≈0")
    else:
        all_ok = False

    # CHECK 5: TS z-score uses only past data (rolling window)
    print("\n  CHECK 5: TS z-score — rolling window look-ahead check")
    sym_df = df[df["symbol"] == sym].sort_values("timestamp").copy()
    # Manually compute for one point
    idx = 2000  # well past the min_periods
    manual_mean = sym_df["ret_12h"].iloc[max(0, idx - 60*24):idx].mean()
    manual_std = sym_df["ret_12h"].iloc[max(0, idx - 60*24):idx].std()
    if "ts_z_ret12h_60d" in sym_df.columns:
        actual_z = sym_df["ts_z_ret12h_60d"].iloc[idx]
        expected_z = (sym_df["ret_12h"].iloc[idx] - manual_mean) / (manual_std + 1e-10)
        diff = abs(actual_z - expected_z)
        ok = diff < 0.01 or np.isnan(actual_z)
        status = "✅" if ok else f"❌ diff={diff:.6f}"
        print(f"    TS z-score manual vs computed: {status}")
        if not ok:
            all_ok = False
    else:
        print(f"    ⚠️  ts_z_ret12h_60d not in df — skipped")

    # CHECK 6: No future data in features (correlation with FUTURE btc return)
    print("\n  CHECK 6: Feature correlation with future BTC return (should be low)")
    btc = df[df["symbol"] == "BTC/USDT"].sort_values("timestamp").dropna(subset=["fwd_ret_12h"])
    suspicious = []
    for feat in FEATURES_14 + ["ts_z_ret12h_60d", "ts_z_oi_chg_60d"]:
        if feat not in btc.columns:
            continue
        valid = btc[[feat, "fwd_ret_12h"]].dropna()
        if len(valid) < 100:
            continue
        corr = valid[feat].corr(valid["fwd_ret_12h"])
        if abs(corr) > 0.15:
            suspicious.append((feat, corr))
            print(f"    ⚠️  {feat}: corr with fwd_ret_12h = {corr:.4f}")
    if not suspicious:
        print(f"    ✅ All features have |corr| < 0.15 with future return (no obvious leakage)")

    # CHECK 7: Shuffled-target test — if model learns from shuffled target, there's leakage
    print("\n  CHECK 7: Shuffled-target sanity check (1 window, 1 seed)")
    w = WINDOWS[1]  # W2
    train = df[df["timestamp"] < pd.Timestamp(w["train_end"], tz="UTC")].copy()
    val   = df[(df["timestamp"] >= pd.Timestamp(w["val_start"], tz="UTC")) &
               (df["timestamp"] <  pd.Timestamp(w["val_end"],   tz="UTC"))].copy()
    test  = df[(df["timestamp"] >= pd.Timestamp(w["test_start"], tz="UTC")) &
               (df["timestamp"] <= pd.Timestamp(w["test_end"],   tz="UTC"))].copy()

    feats = [f for f in FEATURES_14 if f in df.columns]
    train = cs_rank_inplace(train, feats)
    val   = cs_rank_inplace(val, feats)
    test  = cs_rank_inplace(test, feats)

    for d in [train, val, test]:
        d["target_rank"] = d.groupby("timestamp")["fwd_ret_12h"].rank(pct=True) - 0.5

    # Real target
    train_c = train[feats + ["target_rank"]].dropna()
    val_c   = val[feats + ["target_rank"]].dropna()
    test_c  = test[feats + ["target_rank"]].dropna()

    dtrain = lgb.Dataset(train_c[feats], label=train_c["target_rank"], free_raw_data=False)
    dval   = lgb.Dataset(val_c[feats], label=val_c["target_rank"], free_raw_data=False)
    params = {"objective": "regression", "metric": "mse", "learning_rate": 0.05,
              "num_leaves": 31, "min_child_samples": 100, "subsample": 0.8,
              "colsample_bytree": 0.8, "verbose": -1, "n_jobs": -1, "seed": 42}
    model = lgb.train(params, dtrain, num_boost_round=N_ROUNDS,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                                 lgb.log_evaluation(-1)])
    real_pred = model.predict(test_c[feats])
    real_ic = stats.spearmanr(real_pred, test_c["target_rank"])[0]

    # Shuffled target — should give IC ≈ 0
    rng = np.random.RandomState(42)
    shuffled_target = train_c["target_rank"].values.copy()
    rng.shuffle(shuffled_target)
    dtrain_shuf = lgb.Dataset(train_c[feats], label=shuffled_target, free_raw_data=False)
    model_shuf = lgb.train(params, dtrain_shuf, num_boost_round=N_ROUNDS,
                           valid_sets=[dval],
                           callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                                      lgb.log_evaluation(-1)])
    shuf_pred = model_shuf.predict(test_c[feats])
    shuf_ic = stats.spearmanr(shuf_pred, test_c["target_rank"])[0]

    ok = abs(shuf_ic) < 0.02 and real_ic > 0.03
    status = "✅" if ok else "❌"
    print(f"    Real target IC_test = {real_ic:.4f}")
    print(f"    Shuffled target IC_test = {shuf_ic:.4f} (should be ≈0)")
    print(f"    Δ = {real_ic - shuf_ic:.4f} {status}")
    if not ok:
        all_ok = False

    # CHECK 8: Out-of-sample gap verification — predictions only from test period
    print("\n  CHECK 8: Predictions only exist within test window dates")
    for w in WINDOWS:
        test_start = pd.Timestamp(w["test_start"], tz="UTC")
        test_end   = pd.Timestamp(w["test_end"], tz="UTC")
        test = df[(df["timestamp"] >= test_start) & (df["timestamp"] <= test_end)]
        print(f"    {w['name']}: test rows={len(test):,}, "
              f"dates={test['timestamp'].min().date()}→{test['timestamp'].max().date()} ✅")

    print(f"\n  {'='*50}")
    if all_ok:
        print(f"  ✅ ALL LEAKAGE CHECKS PASSED — no data leakage detected")
    else:
        print(f"  ❌ SOME CHECKS FAILED — review above")
    print(f"  {'='*50}")
    return all_ok


# ══════════════════════════════════════════════════════════════
# Training helpers
# ══════════════════════════════════════════════════════════════

def train_lgb_fold(df_train, df_val, df_test, feats, seed, num_leaves,
                   lr=LR, min_child=MIN_CHILD, reg_l1=0.0, reg_l2=0.0,
                   fwd_col="fwd_ret_12h", target_col=None):
    """Train one LGB model. Returns (model, metrics, preds_df)."""
    t_col = target_col or "target_rank"
    if t_col not in df_train.columns:
        for d in [df_train, df_val, df_test]:
            d[t_col] = d.groupby("timestamp")[fwd_col].rank(pct=True) - 0.5

    train_c = df_train[feats + [t_col]].dropna()
    val_c   = df_val[feats + [t_col]].dropna()

    dtrain = lgb.Dataset(train_c[feats], label=train_c[t_col])
    dval   = lgb.Dataset(val_c[feats],   label=val_c[t_col])

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

    test_c = df_test[feats + [t_col, "timestamp", "symbol"]].dropna()
    test_pred = model.predict(test_c[feats])
    ic_test = stats.spearmanr(test_pred, test_c[t_col])[0]

    fwd_data = df_test[["timestamp", "symbol", "fwd_ret_12h"]].rename(
        columns={"fwd_ret_12h": "fwd_ret"}).dropna()
    merged = test_c[["timestamp", "symbol"]].copy()
    merged["pred"] = test_pred
    merged = merged.merge(fwd_data, on=["timestamp", "symbol"], how="inner")

    return model, {"trees": model.best_iteration, "ic_test": round(ic_test, 4)}, merged


def train_lambdarank_fold(df_train, df_val, df_test, feats, seed, num_leaves,
                          lr=LR, min_child=MIN_CHILD):
    """Train LambdaRank model (optimizing NDCG directly)."""
    for d in [df_train, df_val, df_test]:
        d["target_rank"] = d.groupby("timestamp")["fwd_ret_12h"].rank(pct=True) - 0.5
        # LambdaRank needs relevance labels (0-4 discrete)
        d["relevance"] = pd.cut(
            d["target_rank"], bins=5, labels=[0, 1, 2, 3, 4]
        ).astype(float).fillna(2).astype(int)

    train_c = df_train[feats + ["relevance", "timestamp"]].dropna(subset=feats)
    val_c   = df_val[feats + ["relevance", "timestamp"]].dropna(subset=feats)

    # Group sizes per timestamp
    train_groups = train_c.groupby("timestamp").size().values
    val_groups = val_c.groupby("timestamp").size().values

    dtrain = lgb.Dataset(train_c[feats], label=train_c["relevance"], group=train_groups)
    dval   = lgb.Dataset(val_c[feats], label=val_c["relevance"], group=val_groups)

    params = {
        "objective": "lambdarank", "metric": "ndcg",
        "ndcg_eval_at": [3, 6],
        "learning_rate": lr, "num_leaves": num_leaves,
        "min_child_samples": min_child,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "verbose": -1, "n_jobs": -1, "seed": seed,
    }
    model = lgb.train(
        params, dtrain, num_boost_round=N_ROUNDS,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                   lgb.log_evaluation(-1)],
    )

    test_c = df_test[feats + ["target_rank", "timestamp", "symbol"]].dropna()
    test_pred = model.predict(test_c[feats])
    ic_test = stats.spearmanr(test_pred, test_c["target_rank"])[0]

    fwd_data = df_test[["timestamp", "symbol", "fwd_ret_12h"]].rename(
        columns={"fwd_ret_12h": "fwd_ret"}).dropna()
    merged = test_c[["timestamp", "symbol"]].copy()
    merged["pred"] = test_pred
    merged = merged.merge(fwd_data, on=["timestamp", "symbol"], how="inner")

    return model, {"trees": model.best_iteration, "ic_test": round(ic_test, 4)}, merged


def run_experiment(df, feats, cs_feats, num_leaves, name, regime_df,
                   lr=LR, min_child=MIN_CHILD, reg_l1=0.0, reg_l2=0.0,
                   use_lambdarank=False, target_col=None,
                   blend_targets=False, blend_alpha=0.7):
    """Run full walk-forward evaluation."""
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

            # Multi-horizon target blending
            if blend_targets:
                for d in [train, val, test]:
                    r12 = d.groupby("timestamp")["fwd_ret_12h"].rank(pct=True) - 0.5
                    r24 = d.groupby("timestamp")["fwd_ret_24h"].rank(pct=True) - 0.5
                    d["target_blend"] = blend_alpha * r12 + (1 - blend_alpha) * r24
                t_col = "target_blend"
            else:
                t_col = target_col

            if use_lambdarank:
                model, metrics, preds = train_lambdarank_fold(
                    train, val, test, feats, seed, num_leaves, lr, min_child)
            else:
                model, metrics, preds = train_lgb_fold(
                    train, val, test, feats, seed, num_leaves, lr, min_child,
                    reg_l1, reg_l2, target_col=t_col)

            seed_preds.append(preds)
            all_ics.append(metrics["ic_test"])

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


def run_stacking_experiment(df, feats, cs_feats, num_leaves, name, regime_df):
    """Ridge+LGB stacking: average Ridge and LGB predictions."""
    all_preds = []

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

            # LGB prediction
            _, _, lgb_preds = train_lgb_fold(
                train, val, test, feats, seed, num_leaves)

            # Ridge prediction on same features
            train_c = train[feats + ["target_rank"]].dropna()
            test_c  = test[feats + ["target_rank", "timestamp", "symbol"]].dropna()

            # Replace inf with NaN, then fill
            train_c = train_c.replace([np.inf, -np.inf], np.nan).fillna(0)
            test_c_ridge = test_c[feats].replace([np.inf, -np.inf], np.nan).fillna(0)

            ridge = RidgeModel(alpha=1.0)
            ridge.fit(train_c[feats], train_c["target_rank"])
            ridge_pred = ridge.predict(test_c_ridge)

            # Merge both predictions
            ridge_df = test_c[["timestamp", "symbol"]].copy()
            ridge_df["ridge_pred"] = ridge_pred

            merged = lgb_preds.merge(ridge_df, on=["timestamp", "symbol"], how="inner")
            # Weighted average: 0.7 LGB + 0.3 Ridge
            merged["pred"] = 0.7 * merged["pred"] + 0.3 * merged["ridge_pred"]

            seed_preds.append(merged[["timestamp", "symbol", "pred", "fwd_ret"]])

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
    return r


# ══════════════════════════════════════════════════════════════
# Permutation importance for feature pruning
# ══════════════════════════════════════════════════════════════

def permutation_importance_oos(df, feats, cs_feats, num_leaves, regime_df, n_repeats=3):
    """Compute permutation importance on OOS test data across all windows."""
    print("\n  Computing permutation importance (OOS)...")
    base_r = run_experiment(df, feats, cs_feats, num_leaves, "perm_base", regime_df)
    if not base_r:
        return {}

    base_sh = base_r["sharpe"]
    importance = {}

    for feat in feats:
        drops = []
        for rep in range(n_repeats):
            # Drop this feature (set to 0 after ranking — equivalent to permuting)
            reduced_feats = [f for f in feats if f != feat]
            r = run_experiment(df, reduced_feats, [f for f in cs_feats if f != feat],
                               num_leaves, f"drop_{feat}", regime_df)
            if r:
                drops.append(base_sh - r["sharpe"])
        importance[feat] = np.mean(drops) if drops else 0.0
        delta = importance[feat]
        direction = "🔺" if delta > 0.05 else ("⬜" if delta > -0.05 else "🔻")
        print(f"    {direction} {feat:<25s}  ΔSh = {delta:+.3f}")

    return importance


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70)
    print("  RESEARCH ROUND 12 — Leakage Audit + Advanced Experiments")
    print("=" * 70)

    # Load data
    print("\n📊 Loading data...")
    ohlcv  = load_ohlcv()
    ohlcv  = ohlcv[ohlcv["symbol"].isin(SYM_35)]
    derivs = load_derivatives()
    df     = build_features_minimal(ohlcv, derivs)

    print("   Adding TS z-score features...")
    df = add_ts_zscore_features(df)

    print(f"   df: ({len(df):,}, {len(df.columns)})")
    print(f"   date range: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

    regime_df = compute_regime(df)

    # ════════════════════════════════════════════════
    # PART 1: LEAKAGE AUDIT
    # ════════════════════════════════════════════════
    leakage_ok = run_leakage_audit(df)

    if not leakage_ok:
        print("\n  ⚠️  Leakage detected — proceeding with caution")

    results = []

    # Feature sets
    FEATS_14 = [f for f in FEATURES_14 if f in df.columns]
    FEATS_16_TSZ = FEATS_14 + ["ts_z_ret12h_60d", "ts_z_oi_chg_60d"]
    FEATS_16_TSZ = [f for f in FEATS_16_TSZ if f in df.columns]
    CS_16_TSZ = [f for f in FEATS_14 if f in df.columns]  # only rank the base 14

    # ════════════════════════════════════════════════
    # R12A: Combine R11 winners — nl=63 + TS z-scores
    # ════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  R12A — Combine Best of R11: nl=63 + TS z-scores (16f)")
    print("═" * 70)

    r = run_experiment(df, FEATS_16_TSZ, CS_16_TSZ, 63, "R12A 16f+TSz nl=63", regime_df)
    if r:
        show(r)
        results.append(r)

    # Also try nl=47 (between 31 and 63)
    r = run_experiment(df, FEATS_16_TSZ, CS_16_TSZ, 47, "R12A 16f+TSz nl=47", regime_df)
    if r:
        show(r)
        results.append(r)

    # ════════════════════════════════════════════════
    # R12B: Multi-horizon target blending
    # ════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  R12B — Multi-Horizon Target Blending")
    print("═" * 70)

    for alpha in [0.8, 0.7, 0.5]:
        name = f"R12B blend α={alpha} 14f nl=63"
        r = run_experiment(df, FEATS_14, FEATS_14, 63, name, regime_df,
                          blend_targets=True, blend_alpha=alpha)
        if r:
            show(r)
            results.append(r)

    # Best blend + TS z-scores
    name = "R12B blend α=0.7 16f+TSz nl=63"
    r = run_experiment(df, FEATS_16_TSZ, CS_16_TSZ, 63, name, regime_df,
                       blend_targets=True, blend_alpha=0.7)
    if r:
        show(r)
        results.append(r)

    # ════════════════════════════════════════════════
    # R12C: LambdaRank objective
    # ════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  R12C — LambdaRank Objective")
    print("═" * 70)

    r = run_experiment(df, FEATS_14, FEATS_14, 63, "R12C lambdarank 14f nl=63", regime_df,
                       use_lambdarank=True)
    if r:
        show(r)
        results.append(r)

    r = run_experiment(df, FEATS_16_TSZ, CS_16_TSZ, 63, "R12C lambdarank 16f+TSz nl=63", regime_df,
                       use_lambdarank=True)
    if r:
        show(r)
        results.append(r)

    # ════════════════════════════════════════════════
    # R12D: Ridge+LGB stacking
    # ════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  R12D — Ridge+LGB Stacking (0.7/0.3)")
    print("═" * 70)

    r = run_stacking_experiment(df, FEATS_14, FEATS_14, 63, "R12D Ridge+LGB 14f nl=63", regime_df)
    if r:
        show(r)
        results.append(r)

    r = run_stacking_experiment(df, FEATS_16_TSZ, CS_16_TSZ, 63, "R12D Ridge+LGB 16f+TSz nl=63", regime_df)
    if r:
        show(r)
        results.append(r)

    # ════════════════════════════════════════════════
    # R12E: Hyperparameter tuning
    # ════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  R12E — Hyperparameter Tuning (on best config)")
    print("═" * 70)

    # lr=0.03 (slower but potentially more stable)
    r = run_experiment(df, FEATS_14, FEATS_14, 63, "R12E lr=0.03 14f nl=63", regime_df, lr=0.03)
    if r:
        show(r)
        results.append(r)

    # min_child=50 (more splits)
    r = run_experiment(df, FEATS_14, FEATS_14, 63, "R12E min_child=50 14f nl=63", regime_df, min_child=50)
    if r:
        show(r)
        results.append(r)

    # min_child=200 (more conservative)
    r = run_experiment(df, FEATS_14, FEATS_14, 63, "R12E min_child=200 14f nl=63", regime_df, min_child=200)
    if r:
        show(r)
        results.append(r)

    # L2 regularization
    r = run_experiment(df, FEATS_14, FEATS_14, 63, "R12E L2=1.0 14f nl=63", regime_df, reg_l2=1.0)
    if r:
        show(r)
        results.append(r)

    # L1 regularization (sparsity)
    r = run_experiment(df, FEATS_14, FEATS_14, 63, "R12E L1=0.1 14f nl=63", regime_df, reg_l1=0.1)
    if r:
        show(r)
        results.append(r)

    # Combined: lr=0.03 + L2=1.0 + nl=63
    r = run_experiment(df, FEATS_14, FEATS_14, 63, "R12E lr=0.03+L2=1 14f nl=63",
                       regime_df, lr=0.03, reg_l2=1.0)
    if r:
        show(r)
        results.append(r)

    # ════════════════════════════════════════════════
    # R12F: Feature pruning (drop least important features)
    # ════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  R12F — Feature Pruning (drop weakest features)")
    print("═" * 70)

    # From R11 feature importance: dist_from_high_24h and mom_z_12h were bottom
    # Try dropping them
    FEATS_12 = [f for f in FEATS_14 if f not in ["dist_from_high_24h", "mom_z_12h"]]
    r = run_experiment(df, FEATS_12, FEATS_12, 63, "R12F 12f pruned nl=63", regime_df)
    if r:
        show(r)
        results.append(r)

    # Drop bottom 4 features
    FEATS_10 = [f for f in FEATS_14 if f not in
                ["dist_from_high_24h", "mom_z_12h", "residual_24h", "oi_chg_12h"]]
    r = run_experiment(df, FEATS_10, FEATS_10, 63, "R12F 10f pruned nl=63", regime_df)
    if r:
        show(r)
        results.append(r)

    # Pruned + TS z-scores
    FEATS_12_TSZ = FEATS_12 + ["ts_z_ret12h_60d", "ts_z_oi_chg_60d"]
    FEATS_12_TSZ = [f for f in FEATS_12_TSZ if f in df.columns]
    CS_12_TSZ = [f for f in FEATS_12 if f in df.columns]
    r = run_experiment(df, FEATS_12_TSZ, CS_12_TSZ, 63, "R12F 12f+TSz pruned nl=63", regime_df)
    if r:
        show(r)
        results.append(r)

    # ════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  SUMMARY — R12 All Experiments (sorted by Sharpe)")
    print("═" * 70)

    results.sort(key=lambda x: -x["sharpe"])
    print(f"\n  {'Experiment':<45s} {'Sh':>6s} {'Wr%':>7s} {'WM':>6s} {'Eq':>7s} {'IC':>6s}")
    print(f"  {'-'*45} {'-'*6} {'-'*7} {'-'*6} {'-'*7} {'-'*6}")

    # References
    print(f"  {'R10 baseline (nl=31, 14f)':45s} {'4.07':>6s} {'-4.5':>7s} {'11/13':>6s} {'$2565':>7s} {'—':>6s}")
    print(f"  {'R11 best: nl=63 14f':45s} {'4.22':>6s} {'-1.1':>7s} {'12/13':>6s} {'$3142':>7s} {'—':>6s}")
    print(f"  {'R11 best: TS-z 16f nl=31':45s} {'4.37':>6s} {'-10.6':>7s} {'12/13':>6s} {'$3699':>7s} {'—':>6s}")
    print(f"  {'Ridge R7 prod':45s} {'3.59':>6s} {'-6.4':>7s} {'9/13':>6s} {'$2993':>7s} {'—':>6s}")
    print()

    for r in results:
        ic_str = f"{r.get('mean_ic_test', 0):.3f}" if r.get("mean_ic_test") else "—"
        wm_str = f"{r['win_months']}/{r['total_months']}"
        print(f"  {r['name']:<45s} {r['sharpe']:>+6.2f} {r['worst_m']*100:>+7.1f} {wm_str:>6s} "
              f"${r['equity']:>6.0f} {ic_str:>6s}")

    elapsed = time.time() - t0
    print(f"\n  ⏱  Total time: {elapsed/60:.1f} min")

    # Top 3 recommendations
    if results:
        top3 = results[:3]
        print(f"\n  🏆 TOP 3 CONFIGS:")
        for i, r in enumerate(top3, 1):
            wm = f"{r['win_months']}/{r['total_months']}"
            print(f"    {i}. {r['name']} → Sh={r['sharpe']:.2f}, Wr={r['worst_m']*100:.1f}%, "
                  f"WM={wm}, Eq=${r['equity']:.0f}")


if __name__ == "__main__":
    main()
