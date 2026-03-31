#!/usr/bin/env python3
"""
Research Round 11 — Improve LGB model beyond R10 baseline (Sh=4.07).

Three independent experiments:
  R11A — num_leaves sweep (15, 31, 63, 127) on 5-seed ensemble
  R11B — Expanded feature set (+6 new features from existing data)
  R11C — Interaction features + TS z-scores

Baseline: R10 LGB 5-seed nl=31, 14 feats → walk-forward Sh=4.07, WM=11/13, Eq=$2565

All experiments use IDENTICAL walk-forward windows and evaluation as R10.
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

# ── Constants ──
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


# ══════════════════════════════════════════════════════════════════
# Feature engineering helpers
# ══════════════════════════════════════════════════════════════════

def add_r11b_features(df):
    """Add R11B expanded features (from existing data, no new downloads)."""

    # 1. ret_sharpe_24h = rolling sharpe of hourly returns over last 24h
    df["ret_sharpe_24h"] = df.groupby("symbol")["ret_1h"].transform(
        lambda x: x.rolling(24, min_periods=12).mean() / (x.rolling(24, min_periods=12).std() + 1e-10)
    )

    # 2. ret_sharpe_48h
    df["ret_sharpe_48h"] = df.groupby("symbol")["ret_1h"].transform(
        lambda x: x.rolling(48, min_periods=24).mean() / (x.rolling(48, min_periods=24).std() + 1e-10)
    )

    # 3. ret_skew_48h = rolling skewness of returns, 48h window
    df["ret_skew_48h"] = df.groupby("symbol")["ret_1h"].transform(
        lambda x: x.rolling(48, min_periods=24).skew()
    )

    # 4. vol_ratio_24h already exists in build_features_minimal → use vol_ratio_24h

    # 5. top_ls_ratio change 12h
    df["top_ls_change_12h"] = df.groupby("symbol")["top_ls_ratio"].pct_change(12)

    # 6. DVOL BTC z-score 30d (unranked — same for all coins per timestamp)
    #    dvol data already loaded and available in derivs → need to compute externally
    #    For now, placeholder — will be injected from merge

    return df


def add_dvol_features(df, derivs):
    """Add DVOL-based features from Deribit data."""
    dvol = derivs["dvol"].copy()
    btc_dvol = dvol[dvol["currency"] == "BTC"][["timestamp", "dvol_close"]].copy()
    btc_dvol = btc_dvol.sort_values("timestamp").drop_duplicates("timestamp")
    btc_dvol = btc_dvol.rename(columns={"dvol_close": "dvol_btc"})

    # DVOL z-score over 30d (720h)
    btc_dvol["dvol_btc_z30d"] = (
        btc_dvol["dvol_btc"] - btc_dvol["dvol_btc"].rolling(720, min_periods=360).mean()
    ) / (btc_dvol["dvol_btc"].rolling(720, min_periods=360).std() + 1e-10)

    # DVOL 24h change
    btc_dvol["dvol_btc_chg24h"] = btc_dvol["dvol_btc"].pct_change(24)

    df = df.merge(
        btc_dvol[["timestamp", "dvol_btc_z30d", "dvol_btc_chg24h"]],
        on="timestamp", how="left"
    )
    # Forward fill for missing hours
    df["dvol_btc_z30d"] = df["dvol_btc_z30d"].ffill()
    df["dvol_btc_chg24h"] = df["dvol_btc_chg24h"].ffill()

    return df


def add_r11c_features(df):
    """Add R11C interaction + TS z-score features."""

    # 1. OI contra price: OI rising while price falling = squeeze setup
    df["oi_contra_price"] = df["oi_chg_12h"] * (-df["ret_12h"])

    # 2. funding × momentum divergence
    df["funding_mom_div"] = df["funding_zscore"] * (-df["ret_24h"])

    # 3. TS z-score of ret_12h over 60d rolling window (per coin)
    #    Different coordinate system from CS-rank: "is THIS coin extreme for ITSELF"
    ts_mean = df.groupby("symbol")["ret_12h"].transform(
        lambda x: x.rolling(60 * 24, min_periods=360).mean()
    )
    ts_std = df.groupby("symbol")["ret_12h"].transform(
        lambda x: x.rolling(60 * 24, min_periods=360).std()
    ) + 1e-10
    df["ts_z_ret12h_60d"] = (df["ret_12h"] - ts_mean) / ts_std

    # 4. TS z-score of OI changes
    oi_mean = df.groupby("symbol")["oi_chg_12h"].transform(
        lambda x: x.rolling(60 * 24, min_periods=360).mean()
    )
    oi_std = df.groupby("symbol")["oi_chg_12h"].transform(
        lambda x: x.rolling(60 * 24, min_periods=360).std()
    ) + 1e-10
    df["ts_z_oi_chg_60d"] = (df["oi_chg_12h"] - oi_mean) / oi_std

    return df


# ══════════════════════════════════════════════════════════════════
# Core walk-forward evaluation
# ══════════════════════════════════════════════════════════════════

def cs_rank_inplace(df, feats):
    """CS-rank features in-place."""
    df = df.copy()
    for feat in feats:
        if feat in df.columns:
            df[feat] = df.groupby("timestamp")[feat].rank(pct=True) - 0.5
    return df


def train_lgb_fold(df_train, df_val, df_test, feats, seed, num_leaves,
                   fwd_col="fwd_ret_12h"):
    """Train one LGB model on one WF fold."""
    for d in [df_train, df_val, df_test]:
        d["target_rank"] = d.groupby("timestamp")[fwd_col].rank(pct=True) - 0.5

    train_c = df_train[feats + ["target_rank"]].dropna()
    val_c   = df_val[feats + ["target_rank"]].dropna()

    dtrain = lgb.Dataset(train_c[feats], label=train_c["target_rank"])
    dval   = lgb.Dataset(val_c[feats],   label=val_c["target_rank"])

    params = {
        "objective": "regression", "metric": "mse",
        "learning_rate": LR, "num_leaves": num_leaves,
        "min_child_samples": MIN_CHILD,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "verbose": -1, "n_jobs": -1, "seed": seed,
    }
    model = lgb.train(
        params, dtrain, num_boost_round=N_ROUNDS,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                   lgb.log_evaluation(-1)],
    )

    train_pred = model.predict(train_c[feats])
    val_pred   = model.predict(val_c[feats])
    test_c = df_test[feats + ["target_rank", "timestamp", "symbol"]].dropna()
    test_pred  = model.predict(test_c[feats])

    ic_train = stats.spearmanr(train_pred, train_c["target_rank"])[0]
    ic_val   = stats.spearmanr(val_pred,   val_c["target_rank"])[0]
    ic_test  = stats.spearmanr(test_pred,  test_c["target_rank"])[0]

    fwd_data = df_test[["timestamp", "symbol", fwd_col]].rename(
        columns={fwd_col: "fwd_ret"}).dropna()
    merged = test_c[["timestamp", "symbol"]].copy()
    merged["pred"] = test_pred
    merged = merged.merge(fwd_data, on=["timestamp", "symbol"], how="inner")

    return model, {
        "trees": model.best_iteration,
        "ic_train": round(ic_train, 4),
        "ic_val": round(ic_val, 4),
        "ic_test": round(ic_test, 4),
        "ratio": round(ic_train / (ic_test + 1e-10), 2),
    }, merged


def run_experiment(df, feats, cs_feats, num_leaves, name, regime_df):
    """
    Run full walk-forward evaluation for a given feature set + num_leaves.

    Args:
        feats: list of feature column names to use
        cs_feats: subset of feats that should be CS-ranked (others stay raw/unranked)
        num_leaves: LGB num_leaves parameter
        name: experiment name string
    """
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

            model, metrics, preds = train_lgb_fold(
                train, val, test, feats, seed, num_leaves)
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
        r["std_ic_test"] = round(np.std(all_ics), 4)
    return r


# ══════════════════════════════════════════════════════════════════
# Feature importance analysis
# ══════════════════════════════════════════════════════════════════

def get_feature_importance(df, feats, cs_feats, num_leaves, top_n=20):
    """Train on biggest window, return feature importance."""
    w = WINDOWS[-1]  # W3 — most recent
    train = df[df["timestamp"] < pd.Timestamp(w["train_end"], tz="UTC")].copy()
    val   = df[(df["timestamp"] >= pd.Timestamp(w["val_start"], tz="UTC")) &
               (df["timestamp"] <  pd.Timestamp(w["val_end"],   tz="UTC"))].copy()

    train = cs_rank_inplace(train, cs_feats)
    val   = cs_rank_inplace(val, cs_feats)

    for d in [train, val]:
        d["target_rank"] = d.groupby("timestamp")["fwd_ret_12h"].rank(pct=True) - 0.5

    train_c = train[feats + ["target_rank"]].dropna()
    val_c   = val[feats + ["target_rank"]].dropna()

    dtrain = lgb.Dataset(train_c[feats], label=train_c["target_rank"])
    dval   = lgb.Dataset(val_c[feats], label=val_c["target_rank"])

    params = {
        "objective": "regression", "metric": "mse",
        "learning_rate": LR, "num_leaves": num_leaves,
        "min_child_samples": MIN_CHILD,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "verbose": -1, "n_jobs": -1, "seed": 42,
    }
    model = lgb.train(
        params, dtrain, num_boost_round=N_ROUNDS,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                   lgb.log_evaluation(-1)],
    )

    importance = dict(zip(feats, model.feature_importance("gain")))
    sorted_imp = sorted(importance.items(), key=lambda x: -x[1])
    return sorted_imp[:top_n]


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70)
    print("  RESEARCH ROUND 11 — Push LGB Beyond R10 Baseline (Sh=4.07)")
    print("=" * 70)

    # ── Load data ──
    print("\n📊 Loading data...")
    ohlcv  = load_ohlcv()
    ohlcv  = ohlcv[ohlcv["symbol"].isin(SYM_35)]
    derivs = load_derivatives()
    df     = build_features_minimal(ohlcv, derivs)

    # Add all R11 features at once
    print("   Adding R11B features...")
    df = add_r11b_features(df)
    print("   Adding DVOL features...")
    df = add_dvol_features(df, derivs)
    print("   Adding R11C features...")
    df = add_r11c_features(df)

    avail_feats = [f for f in df.columns if f not in
                   ["timestamp", "symbol", "open", "high", "low", "close", "volume",
                    "btc_close", "coin_ret", "btc_ret"] and "fwd_ret" not in f
                   and "target" not in f]

    print(f"   df: ({len(df):,}, {len(df.columns)}), symbols: {df['symbol'].nunique()}")
    print(f"   date range: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print(f"   available features: {len(avail_feats)}")

    regime_df = compute_regime(df)

    results = []

    # ══════════════════════════════════════════════════════════
    # R11A — num_leaves sweep
    # ══════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  R11A — num_leaves Sweep (14 feats baseline)")
    print("═" * 70)

    for nl in [15, 31, 63, 127]:
        name = f"R11A nl={nl:3d} 14f"
        print(f"\n  ▶ {name}")
        r = run_experiment(df, FEATURES_14, FEATURES_14, nl, name, regime_df)
        if r:
            show(r)
            results.append(r)

    # ══════════════════════════════════════════════════════════
    # R11B — Expanded feature set
    # ══════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  R11B — Expanded Feature Sets")
    print("═" * 70)

    # R11B-1: 14 baseline + 4 new CS-ranked features
    FEATS_18 = FEATURES_14 + [
        "ret_sharpe_24h", "ret_sharpe_48h",
        "ret_skew_48h", "top_ls_change_12h",
    ]
    FEATS_18 = [f for f in FEATS_18 if f in df.columns]

    print(f"\n  ▶ R11B-1: {len(FEATS_18)} feats (14 base + sharpe/skew/ls_chg)")
    r = run_experiment(df, FEATS_18, FEATS_18, 31, f"R11B-1 {len(FEATS_18)}f nl=31", regime_df)
    if r:
        show(r)
        results.append(r)

    # R11B-2: 18 + DVOL (unranked)
    FEATS_20 = FEATS_18 + ["dvol_btc_z30d", "dvol_btc_chg24h"]
    FEATS_20 = [f for f in FEATS_20 if f in df.columns]
    CS_FEATS_20 = [f for f in FEATS_18 if f in df.columns]  # only CS-rank the non-dvol feats

    print(f"\n  ▶ R11B-2: {len(FEATS_20)} feats (+DVOL, dvol unranked)")
    r = run_experiment(df, FEATS_20, CS_FEATS_20, 31, f"R11B-2 {len(FEATS_20)}f+dvol nl=31", regime_df)
    if r:
        show(r)
        results.append(r)

    # R11B-3: Adding more derivative features
    FEATS_DERIV = FEATS_20 + [
        "funding_zscore", "premium_zscore", "taker_zscore",
        "oi_ret_diverge", "vol_ratio_24h",
    ]
    FEATS_DERIV = [f for f in FEATS_DERIV if f in df.columns]
    CS_FEATS_DERIV = [f for f in FEATS_DERIV
                      if f not in ["dvol_btc_z30d", "dvol_btc_chg24h"]]

    print(f"\n  ▶ R11B-3: {len(FEATS_DERIV)} feats (full deriv suite)")
    r = run_experiment(df, FEATS_DERIV, CS_FEATS_DERIV, 31, f"R11B-3 {len(FEATS_DERIV)}f full-deriv nl=31", regime_df)
    if r:
        show(r)
        results.append(r)

    # ══════════════════════════════════════════════════════════
    # R11C — Interaction + TS z-score features
    # ══════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  R11C — Interaction + TS Z-Score Features")
    print("═" * 70)

    # R11C-1: 14 base + interactions (CS-ranked)
    FEATS_INTERACT = FEATURES_14 + [
        "oi_contra_price", "funding_mom_div",
    ]
    FEATS_INTERACT = [f for f in FEATS_INTERACT if f in df.columns]

    print(f"\n  ▶ R11C-1: {len(FEATS_INTERACT)} feats (14 + interactions)")
    r = run_experiment(df, FEATS_INTERACT, FEATS_INTERACT, 31,
                       f"R11C-1 {len(FEATS_INTERACT)}f interactions nl=31", regime_df)
    if r:
        show(r)
        results.append(r)

    # R11C-2: 14 base + TS z-scores (NOT CS-ranked — already per-coin normalized)
    FEATS_TSZ = FEATURES_14 + [
        "ts_z_ret12h_60d", "ts_z_oi_chg_60d",
    ]
    FEATS_TSZ = [f for f in FEATS_TSZ if f in df.columns]
    CS_TSZ = [f for f in FEATURES_14 if f in df.columns]  # only rank base 14

    print(f"\n  ▶ R11C-2: {len(FEATS_TSZ)} feats (14 + TS z-scores, unranked)")
    r = run_experiment(df, FEATS_TSZ, CS_TSZ, 31,
                       f"R11C-2 {len(FEATS_TSZ)}f TS-z nl=31", regime_df)
    if r:
        show(r)
        results.append(r)

    # R11C-3: Kitchen sink = all R11B + all R11C features
    FEATS_ALL = list(set(FEATS_DERIV + FEATS_INTERACT + FEATS_TSZ))
    FEATS_ALL = sorted([f for f in FEATS_ALL if f in df.columns])
    CS_ALL = [f for f in FEATS_ALL
              if f not in ["dvol_btc_z30d", "dvol_btc_chg24h",
                           "ts_z_ret12h_60d", "ts_z_oi_chg_60d"]]

    print(f"\n  ▶ R11C-3: {len(FEATS_ALL)} feats (KITCHEN SINK)")
    r = run_experiment(df, FEATS_ALL, CS_ALL, 31,
                       f"R11C-3 {len(FEATS_ALL)}f kitchen-sink nl=31", regime_df)
    if r:
        show(r)
        results.append(r)

    # ══════════════════════════════════════════════════════════
    # Best config with different num_leaves
    # ══════════════════════════════════════════════════════════
    if results:
        best = max(results, key=lambda x: x["sharpe"])
        best_name = best["name"]
        print(f"\n" + "═" * 70)
        print(f"  Best so far: {best_name} → Sh={best['sharpe']:.2f}")
        print(f"  Testing best config with nl=63...")
        print("═" * 70)

        # Figure out feature set from name
        if "kitchen-sink" in best_name:
            best_feats, best_cs = FEATS_ALL, CS_ALL
        elif "full-deriv" in best_name:
            best_feats, best_cs = FEATS_DERIV, CS_FEATS_DERIV
        elif "dvol" in best_name:
            best_feats, best_cs = FEATS_20, CS_FEATS_20
        elif "B-1" in best_name:
            best_feats, best_cs = FEATS_18, FEATS_18
        elif "interactions" in best_name:
            best_feats, best_cs = FEATS_INTERACT, FEATS_INTERACT
        elif "TS-z" in best_name:
            best_feats, best_cs = FEATS_TSZ, CS_TSZ
        else:
            best_feats, best_cs = FEATURES_14, FEATURES_14

        for nl in [63, 127]:
            name = f"BEST+nl={nl} ({len(best_feats)}f)"
            print(f"\n  ▶ {name}")
            r = run_experiment(df, best_feats, best_cs, nl, name, regime_df)
            if r:
                show(r)
                results.append(r)

    # ══════════════════════════════════════════════════════════
    # Feature importance for best config
    # ══════════════════════════════════════════════════════════
    if results:
        overall_best = max(results, key=lambda x: x["sharpe"])
        print(f"\n" + "═" * 70)
        print(f"  Feature Importance for best config")
        print("═" * 70)

        if "kitchen-sink" in overall_best["name"]:
            fi_feats, fi_cs = FEATS_ALL, CS_ALL
        elif "full-deriv" in overall_best["name"]:
            fi_feats, fi_cs = FEATS_DERIV, CS_FEATS_DERIV
        elif "dvol" in overall_best["name"]:
            fi_feats, fi_cs = FEATS_20, CS_FEATS_20
        elif "B-1" in overall_best["name"]:
            fi_feats, fi_cs = FEATS_18, FEATS_18
        elif "interactions" in overall_best["name"]:
            fi_feats, fi_cs = FEATS_INTERACT, FEATS_INTERACT
        elif "TS-z" in overall_best["name"]:
            fi_feats, fi_cs = FEATS_TSZ, CS_TSZ
        else:
            fi_feats, fi_cs = FEATURES_14, FEATURES_14

        # Extract num_leaves from name
        fi_nl = 31
        if "nl=" in overall_best["name"]:
            nl_str = overall_best["name"].split("nl=")[-1].split(" ")[0].split(")")[0]
            try:
                fi_nl = int(nl_str)
            except ValueError:
                fi_nl = 31

        imp = get_feature_importance(df, fi_feats, fi_cs, fi_nl)
        total_gain = sum(v for _, v in imp)
        print(f"\n  {'Feature':<25s} {'Gain':>10s} {'%':>6s}")
        print(f"  {'-'*25} {'-'*10} {'-'*6}")
        for feat, gain in imp:
            pct = gain / (total_gain + 1e-10) * 100
            print(f"  {feat:<25s} {gain:>10.0f} {pct:>5.1f}%")

    # ══════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  SUMMARY — All Experiments (sorted by Sharpe)")
    print("═" * 70)

    results.sort(key=lambda x: -x["sharpe"])
    print(f"\n  {'Experiment':<45s} {'Sh':>6s} {'Wr%':>7s} {'WM':>6s} {'Eq':>7s} {'IC':>6s}")
    print(f"  {'-'*45} {'-'*6} {'-'*7} {'-'*6} {'-'*7} {'-'*6}")

    # Reference baseline
    print(f"  {'R10 baseline (nl=31, 14f)':45s} {'4.07':>6s} {'-4.5':>7s} {'11/13':>6s} {'$2565':>7s} {'—':>6s}")
    print(f"  {'Ridge R7 prod':45s} {'3.59':>6s} {'-6.4':>7s} {'9/13':>6s} {'$2993':>7s} {'—':>6s}")
    print()

    for r in results:
        ic_str = f"{r.get('mean_ic_test', 0):.3f}" if r.get("mean_ic_test") else "—"
        wm_str = f"{r['win_months']}/{r['total_months']}"
        print(f"  {r['name']:<45s} {r['sharpe']:>+6.2f} {r['worst_m']*100:>+7.1f} {wm_str:>6s} "
              f"${r['equity']:>6.0f} {ic_str:>6s}")

    elapsed = time.time() - t0
    print(f"\n  ⏱  Total time: {elapsed/60:.1f} min")

    # Save results JSON
    results_out = []
    for r in results:
        results_out.append({
            "name": r["name"],
            "sharpe": round(r["sharpe"], 3),
            "worst_m": round(r["worst_m"] * 100, 2),
            "equity": round(r["equity"], 0),
            "win_months": r["win_months"],
            "total_months": r["total_months"],
            "mean_ic_test": r.get("mean_ic_test"),
        })
    with open("_r11_results.json", "w") as f:
        json.dump(results_out, f, indent=2)
    print(f"\n  💾 Results saved to _r11_results.json")


if __name__ == "__main__":
    main()
