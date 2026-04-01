#!/usr/bin/env python3
"""
R23 — Deep Training & Signal Experiments

Base: R20-C — LGB-23f, cutoff=0.9, 12h rebal, 6L/3S → Sh=2.80, Eq=$2096

R22 tried breadth (models, features, ensembles) — nothing beat baseline.
R23 tries depth — HOW the model trains and HOW predictions are used.

Experiments:
  A: Individual feature addition (1 at a time, not groups)
  B: Time-weighted training (recent data weighted higher)
  C: Risk-adjusted target (ret / vol)
  D: Signal confidence filter (skip low-spread timestamps)
  E: Rolling window training (drop old data)
  F: Signal EMA + prediction shrinkage combos
  G: Classification target (predict sign, not rank)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from pathlib import Path
import warnings, time, sys
warnings.filterwarnings("ignore")

from _research_round7 import (
    SYM_35, WINDOWS, compute_regime, simulate, eval_config, show,
)
from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal
from _research_r22_models import (
    FEATURES_23, SEEDS, LEVERAGE, CAPITAL, CFG_BEST, DATA_DIR, SENT_DIR,
    log, build_r19_features, add_new_features, cs_rank_cols,
    train_lgb, run_eval,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-A: Individual feature addition (1 at a time)
# ═══════════════════════════════════════════════════════════════════════════════

def exp_a(df, regime_df, new_feats):
    log("\n" + "=" * 80)
    log("  EXP-A: Individual Feature Addition (1 at a time)")
    log("=" * 80)

    avail_23 = [f for f in FEATURES_23 if f in df.columns]
    results = []

    # Control
    log("\n  A0: Control (23f)...")
    preds0 = train_lgb(df, avail_23)
    r0 = run_eval(preds0, regime_df, "A-ctrl-23f", verbose_months=False)
    if r0:
        results.append(("baseline", r0))

    # Test each new feature individually
    good_new = [f for f in new_feats if f in df.columns and df[f].notna().mean() > 0.5]
    for feat in good_new:
        feats = avail_23 + [feat]
        log(f"\n  A-{feat}: 23f + {feat} = {len(feats)}f...")
        preds = train_lgb(df, feats)
        r = run_eval(preds, regime_df, f"A-+{feat}", verbose_months=False)
        if r:
            delta = r["sharpe"] - r0["sharpe"] if r0 else 0
            flag = "✅" if delta > 0 else "❌"
            log(f"    {flag} Δ={delta:+.2f}")
            results.append((feat, r))

    # Sort by sharpe
    log("\n  Individual feature ranking:")
    for feat, r in sorted(results, key=lambda x: -x[1]["sharpe"]):
        delta = r["sharpe"] - r0["sharpe"] if r0 else 0
        flag = "+" if delta > 0 else " "
        log(f"    {flag} {feat:<30s} Sh={r['sharpe']:.2f} Eq=${r['equity']:.0f} Δ={delta:+.2f}")

    # Try combining top-3 improving features
    improving = [(f, r) for f, r in results if f != "baseline" and r0 and r["sharpe"] > r0["sharpe"]]
    if len(improving) >= 2:
        improving.sort(key=lambda x: -x[1]["sharpe"])
        top_feats = [f for f, _ in improving[:3]]
        feats_combo = avail_23 + top_feats
        log(f"\n  A-combo: 23f + top-{len(top_feats)} = {len(feats_combo)}f ({top_feats})...")
        preds_combo = train_lgb(df, feats_combo)
        r_combo = run_eval(preds_combo, regime_df, f"A-combo-{len(feats_combo)}f")
        if r_combo:
            results.append(("combo", r_combo))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-B: Time-weighted training
# ═══════════════════════════════════════════════════════════════════════════════

def train_lgb_weighted(df, feats, decay_half_life_days=180, seeds=SEEDS):
    """Train LGB with exponential time decay sample weights."""
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
                d["target_rank"] = d.groupby("timestamp")["fwd_ret_12h"].rank(pct=True) - 0.5

            train_c = train[avail + ["target_rank", "timestamp"]].dropna()
            val_c = val[avail + ["target_rank"]].dropna()

            # Compute time weights: more recent = higher weight
            max_ts = train_c["timestamp"].max()
            days_ago = (max_ts - train_c["timestamp"]).dt.total_seconds() / 86400
            half_life = decay_half_life_days
            weights = np.exp(-np.log(2) * days_ago / half_life)
            weights = weights / weights.mean()  # normalize so mean=1

            dtrain = lgb.Dataset(train_c[avail], label=train_c["target_rank"],
                                 weight=weights.values)
            dval = lgb.Dataset(val_c[avail], label=val_c["target_rank"])
            model = lgb.train(
                {"objective": "regression", "metric": "mse",
                 "learning_rate": 0.03, "num_leaves": 63,
                 "min_child_samples": 100, "subsample": 0.8,
                 "colsample_bytree": 0.8, "lambda_l2": 1.0,
                 "seed": seed, "verbose": -1, "n_jobs": -1},
                dtrain, num_boost_round=600, valid_sets=[dval],
                callbacks=[lgb.early_stopping(40, verbose=False),
                           lgb.log_evaluation(-1)])

            test_c = test[avail + ["target_rank", "timestamp", "symbol"]].dropna()
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


def exp_b(df, regime_df):
    log("\n" + "=" * 80)
    log("  EXP-B: Time-Weighted Training")
    log("=" * 80)

    avail_23 = [f for f in FEATURES_23 if f in df.columns]
    results = []

    for half_life in [90, 180, 365, 730]:
        log(f"\n  B-hl{half_life}d: half_life={half_life} days...")
        preds = train_lgb_weighted(df, avail_23, decay_half_life_days=half_life)
        r = run_eval(preds, regime_df, f"B-hl{half_life}d", verbose_months=(half_life == 180))
        if r:
            results.append((half_life, r))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-C: Risk-adjusted target
# ═══════════════════════════════════════════════════════════════════════════════

def exp_c(df, regime_df):
    log("\n" + "=" * 80)
    log("  EXP-C: Risk-Adjusted Target (ret/vol)")
    log("=" * 80)

    avail_23 = [f for f in FEATURES_23 if f in df.columns]

    # Create risk-adjusted target: fwd_ret / trailing_vol
    df = df.copy()
    df["trailing_vol_12h"] = df.groupby("symbol")["fwd_ret_12h"].transform(
        lambda x: x.shift(1).rolling(48, min_periods=12).std())
    df["fwd_ret_riskadjusted"] = df["fwd_ret_12h"] / (df["trailing_vol_12h"] + 1e-6)

    # Train with risk-adjusted target
    preds_ra = train_lgb_custom_target(df, avail_23, target_col="fwd_ret_riskadjusted")
    r_ra = run_eval(preds_ra, regime_df, "C-riskadjusted")

    # Train with winsorized target (clip outlier returns)
    df["fwd_ret_12h_winsor"] = df.groupby("timestamp")["fwd_ret_12h"].transform(
        lambda x: x.clip(x.quantile(0.05), x.quantile(0.95)))
    preds_w = train_lgb_custom_target(df, avail_23, target_col="fwd_ret_12h_winsor")
    r_w = run_eval(preds_w, regime_df, "C-winsorized")

    return r_ra, r_w


def train_lgb_custom_target(df, feats, target_col, seeds=SEEDS):
    """Train LGB with a custom target column (still using rank transformation)."""
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
                if target_col in d.columns:
                    d["target_rank"] = d.groupby("timestamp")[target_col].rank(pct=True) - 0.5

            train_c = train[avail + ["target_rank"]].dropna()
            val_c = val[avail + ["target_rank"]].dropna()

            dtrain = lgb.Dataset(train_c[avail], label=train_c["target_rank"])
            dval = lgb.Dataset(val_c[avail], label=val_c["target_rank"])
            model = lgb.train(
                {"objective": "regression", "metric": "mse",
                 "learning_rate": 0.03, "num_leaves": 63,
                 "min_child_samples": 100, "subsample": 0.8,
                 "colsample_bytree": 0.8, "lambda_l2": 1.0,
                 "seed": seed, "verbose": -1, "n_jobs": -1},
                dtrain, num_boost_round=600, valid_sets=[dval],
                callbacks=[lgb.early_stopping(40, verbose=False),
                           lgb.log_evaluation(-1)])

            test_c = test[avail + ["target_rank", "timestamp", "symbol"]].dropna()
            if len(test_c) == 0:
                continue
            preds = model.predict(test_c[avail])
            # NOTE: fwd_ret is ALWAYS raw return for simulation fairness
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


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-D: Signal confidence filter
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_with_confidence(merged, regime_df, cfg, min_spread_pct=0.3):
    """Modified simulate: skip timestamps where pred spread is too narrow."""
    merged = merged.copy()

    # Compute prediction spread per timestamp
    spreads = merged.groupby("timestamp")["pred"].agg(
        lambda x: x.quantile(0.9) - x.quantile(0.1))
    median_spread = spreads.median()
    threshold = median_spread * min_spread_pct

    # Filter out low-confidence timestamps
    good_ts = spreads[spreads >= threshold].index
    merged_filtered = merged[merged["timestamp"].isin(good_ts)]

    log(f"    Confidence filter: {len(good_ts)}/{len(spreads)} timestamps kept "
        f"(spread threshold={threshold:.4f})")

    return simulate(merged_filtered, regime_df, 12, cfg)


def exp_d(df, regime_df):
    log("\n" + "=" * 80)
    log("  EXP-D: Signal Confidence Filter")
    log("=" * 80)

    avail_23 = [f for f in FEATURES_23 if f in df.columns]
    preds = train_lgb(df, avail_23)
    results = []

    for min_spread in [0.0, 0.3, 0.5, 0.7, 0.9]:
        log(f"\n  D-spread{min_spread:.1f}:")
        port = simulate_with_confidence(preds, regime_df, CFG_BEST,
                                         min_spread_pct=min_spread)
        r = eval_config(port, 12, f"D-conf{min_spread:.1f}", LEVERAGE, CAPITAL)
        if r:
            show(r)
            results.append((min_spread, r))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-E: Rolling window training (limit training data recency)
# ═══════════════════════════════════════════════════════════════════════════════

def train_lgb_rolling(df, feats, max_train_days=365, seeds=SEEDS):
    """Train LGB with rolling window: only use last N days of training data."""
    avail = [f for f in feats if f in df.columns]
    all_preds = []
    tz = df["timestamp"].dt.tz

    for seed in seeds:
        seed_preds = []
        for w in WINDOWS:
            train_end_ts = pd.Timestamp(w["train_end"], tz=tz)
            train_start_ts = train_end_ts - pd.Timedelta(days=max_train_days)

            train = df[(df["timestamp"] >= train_start_ts) &
                       (df["timestamp"] < train_end_ts)].copy()
            val = df[(df["timestamp"] >= pd.Timestamp(w["val_start"], tz=tz)) &
                     (df["timestamp"] < pd.Timestamp(w["val_end"], tz=tz))].copy()
            test = df[(df["timestamp"] >= pd.Timestamp(w["test_start"], tz=tz)) &
                      (df["timestamp"] <= pd.Timestamp(w["test_end"], tz=tz))].copy()

            if len(train) < 5000 or len(test) < 200:
                log(f"      {w['name']} skip: train={len(train)} (need 5000+)")
                continue

            train = cs_rank_cols(train, avail)
            val = cs_rank_cols(val, avail)
            test = cs_rank_cols(test, avail)

            for d in [train, val, test]:
                d["target_rank"] = d.groupby("timestamp")["fwd_ret_12h"].rank(pct=True) - 0.5

            train_c = train[avail + ["target_rank"]].dropna()
            val_c = val[avail + ["target_rank"]].dropna()

            dtrain = lgb.Dataset(train_c[avail], label=train_c["target_rank"])
            dval = lgb.Dataset(val_c[avail], label=val_c["target_rank"])
            model = lgb.train(
                {"objective": "regression", "metric": "mse",
                 "learning_rate": 0.03, "num_leaves": 63,
                 "min_child_samples": 100, "subsample": 0.8,
                 "colsample_bytree": 0.8, "lambda_l2": 1.0,
                 "seed": seed, "verbose": -1, "n_jobs": -1},
                dtrain, num_boost_round=600, valid_sets=[dval],
                callbacks=[lgb.early_stopping(40, verbose=False),
                           lgb.log_evaluation(-1)])

            test_c = test[avail + ["target_rank", "timestamp", "symbol"]].dropna()
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


def exp_e(df, regime_df):
    log("\n" + "=" * 80)
    log("  EXP-E: Rolling Window Training")
    log("=" * 80)

    avail_23 = [f for f in FEATURES_23 if f in df.columns]
    results = []

    for max_days in [180, 365, 540, 730]:
        log(f"\n  E-roll{max_days}d: last {max_days} days of training data...")
        preds = train_lgb_rolling(df, avail_23, max_train_days=max_days)
        r = run_eval(preds, regime_df, f"E-roll{max_days}d",
                     verbose_months=(max_days == 365))
        if r:
            results.append((max_days, r))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-F: Signal EMA + Prediction Shrinkage combos
# ═══════════════════════════════════════════════════════════════════════════════

def exp_f(df, regime_df):
    log("\n" + "=" * 80)
    log("  EXP-F: Signal EMA + Prediction Shrinkage")
    log("=" * 80)

    avail_23 = [f for f in FEATURES_23 if f in df.columns]
    preds = train_lgb(df, avail_23)
    results = []

    # Signal EMA (built into simulate)
    for ema in [None, 2, 3, 5]:
        for shrink in [None, 0.1, 0.2, 0.3]:
            cfg = {**CFG_BEST}
            if ema is not None:
                cfg["signal_ema"] = ema
            if shrink is not None:
                cfg["pred_shrinkage"] = shrink
            label = f"F-ema{ema or 0}-shrink{shrink or 0}"
            port = simulate(preds, regime_df, 12, cfg)
            r = eval_config(port, 12, label, LEVERAGE, CAPITAL)
            if r:
                show(r)
                results.append((label, r))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXP-G: Classification target (predict direction, not magnitude)
# ═══════════════════════════════════════════════════════════════════════════════

def train_lgb_classification(df, feats, seeds=SEEDS):
    """Train LGB binary classifier: predict P(positive return)."""
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

            dtrain = lgb.Dataset(train_c[avail], label=train_c["target_binary"])
            dval = lgb.Dataset(val_c[avail], label=val_c["target_binary"])
            model = lgb.train(
                {"objective": "binary", "metric": "auc",
                 "learning_rate": 0.03, "num_leaves": 63,
                 "min_child_samples": 100, "subsample": 0.8,
                 "colsample_bytree": 0.8, "lambda_l2": 1.0,
                 "seed": seed, "verbose": -1, "n_jobs": -1},
                dtrain, num_boost_round=600, valid_sets=[dval],
                callbacks=[lgb.early_stopping(40, verbose=False),
                           lgb.log_evaluation(-1)])

            test_c = test[avail + ["target_binary", "timestamp", "symbol"]].dropna()
            if len(test_c) == 0:
                continue
            preds = model.predict(test_c[avail])  # probabilities
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


def exp_g(df, regime_df):
    log("\n" + "=" * 80)
    log("  EXP-G: Classification Target (predict direction)")
    log("=" * 80)

    avail_23 = [f for f in FEATURES_23 if f in df.columns]
    preds_cls = train_lgb_classification(df, avail_23)
    r_cls = run_eval(preds_cls, regime_df, "G-classification")
    return r_cls


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log("=" * 80)
    log("  R23 — DEEP TRAINING & SIGNAL EXPERIMENTS")
    log("=" * 80)
    log("  Base: R20-C — LGB-23f, cutoff=0.9, 12h rebal → Sh=2.80, Eq=$2096")
    log("  Experiments: A(individual feats) B(time-weight) C(riskadjust)")
    log("               D(confidence) E(rolling) F(ema/shrink) G(classification)")

    # ── Load data ─────────────────────────────────────────────────────────────
    log("\n  Loading data...")
    ohlcv = load_ohlcv()
    ohlcv = ohlcv[ohlcv["symbol"].isin(SYM_35)]
    derivs = load_derivatives()
    df = build_features_minimal(ohlcv, derivs)
    regime_df = compute_regime(df)
    log(f"  Base: {len(df):,} rows, {df['symbol'].nunique()} symbols")

    log("\n  Building features...")
    df = build_r19_features(df)
    df, new_feats = add_new_features(df)
    avail_23 = [f for f in FEATURES_23 if f in df.columns]
    log(f"  FEATURES_23: {len(avail_23)}/23, new feats: {len(new_feats)}")

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-F: Signal EMA + Shrinkage (fast — no retraining needed)
    # ══════════════════════════════════════════════════════════════════════════
    results_f = exp_f(df, regime_df)

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-D: Confidence filter (fast — no retraining needed)
    # ══════════════════════════════════════════════════════════════════════════
    results_d = exp_d(df, regime_df)

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-G: Classification target
    # ══════════════════════════════════════════════════════════════════════════
    r_g = exp_g(df, regime_df)

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-C: Risk-adjusted target
    # ══════════════════════════════════════════════════════════════════════════
    r_c_ra, r_c_w = exp_c(df, regime_df)

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-B: Time-weighted training
    # ══════════════════════════════════════════════════════════════════════════
    results_b = exp_b(df, regime_df)

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-E: Rolling window
    # ══════════════════════════════════════════════════════════════════════════
    results_e = exp_e(df, regime_df)

    # ══════════════════════════════════════════════════════════════════════════
    # EXP-A: Individual features (heaviest — multiple retrains)
    # ══════════════════════════════════════════════════════════════════════════
    results_a = exp_a(df, regime_df, new_feats)

    # ══════════════════════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 80)
    log("  FINAL RANKINGS — R23 ALL EXPERIMENTS")
    log("=" * 80)

    all_results = []
    if results_f:
        for _, r in results_f:
            all_results.append(r)
    if results_d:
        for _, r in results_d:
            all_results.append(r)
    if r_g:
        all_results.append(r_g)
    if r_c_ra:
        all_results.append(r_c_ra)
    if r_c_w:
        all_results.append(r_c_w)
    if results_b:
        for _, r in results_b:
            all_results.append(r)
    if results_e:
        for _, r in results_e:
            all_results.append(r)
    if results_a:
        for _, r in results_a:
            all_results.append(r)

    if all_results:
        ranked = sorted(all_results, key=lambda r: -r["sharpe"])
        for i, r in enumerate(ranked, 1):
            delta = r["sharpe"] - 2.80
            flag = "✅" if delta > 0 else ("⚠️" if delta > -0.10 else "❌")
            log(f"  #{i:2d} {flag} {r['name']:<50s} "
                f"Sh={r['sharpe']:+.2f} Eq=${r['equity']:.0f} "
                f"WM={r['win_months']}/{r['total_months']} "
                f"Wr={r['worst_m']*100:+.1f}% Δ={delta:+.2f}")

    log(f"\n  R20-C baseline: Sh=2.80, Eq=$2096")

    elapsed = time.time() - t0
    log(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
