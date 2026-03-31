#!/usr/bin/env python3
"""
R17 — Comprehensive retest with FIXED simulation.

Uses the corrected _research_round7.py:
  - simulate() processes only rebalance timestamps (no overlapping returns)
  - eval_config() uses actual observation frequency for Sharpe annualization
  - eq_mom_boost and strategy_momentum removed (caused look-ahead)

Tests:
  1. LGB R13 config (5-seed ensemble) — various portfolio configs
  2. Ridge baseline — compare model complexity
  3. Feature set ablation (12f vs 14f vs minimal)
  4. Per-window decomposition
  5. Permutation test (final verification)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from sklearn.linear_model import Ridge
import warnings, time
warnings.filterwarnings("ignore")

from _research_round7 import (
    SYM_35, WINDOWS, FEATURES, cs_rank,
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

FEATURES_MINIMAL = [
    "ret_12h", "ret_24h", "ret_48h",
    "residual_12h", "residual_24h",
    "oi_chg_12h", "oi_chg_24h",
    "taker_cvd_12h", "taker_cvd_24h",
]

SEEDS = [0, 7, 13, 42, 99]
LEVERAGE = 5
CAPITAL = 100


def cs_rank_inplace(df, feats):
    df = df.copy()
    for feat in feats:
        if feat in df.columns:
            df[feat] = df.groupby("timestamp")[feat].rank(pct=True) - 0.5
    return df


# ═══════════════════════════════════════════
# LGB ENSEMBLE (R13 config)
# ═══════════════════════════════════════════
def train_lgb_ensemble(df, feats, seeds=SEEDS, params_override=None):
    """Train LGB 5-seed ensemble, return merged predictions."""
    all_preds = []
    for seed in seeds:
        seed_preds = []
        for w in WINDOWS:
            tz = df["timestamp"].dt.tz
            train = df[df["timestamp"] < pd.Timestamp(w["train_end"], tz=tz)].copy()
            val   = df[(df["timestamp"] >= pd.Timestamp(w["val_start"], tz=tz)) &
                       (df["timestamp"] <  pd.Timestamp(w["val_end"],   tz=tz))].copy()
            test  = df[(df["timestamp"] >= pd.Timestamp(w["test_start"], tz=tz)) &
                       (df["timestamp"] <= pd.Timestamp(w["test_end"],   tz=tz))].copy()
            if len(train) < 5000 or len(test) < 200:
                continue

            train = cs_rank_inplace(train, feats)
            val   = cs_rank_inplace(val, feats)
            test  = cs_rank_inplace(test, feats)

            for d in [train, val, test]:
                d["target_rank"] = d.groupby("timestamp")["fwd_ret_12h"].rank(pct=True) - 0.5

            train_c = train[feats + ["target_rank"]].dropna()
            val_c   = val[feats + ["target_rank"]].dropna()

            params = {
                "objective": "regression", "metric": "mse",
                "learning_rate": 0.03, "num_leaves": 63,
                "min_child_samples": 100,
                "subsample": 0.8, "colsample_bytree": 0.8,
                "lambda_l2": 1.0,
                "seed": seed,
                "verbose": -1, "n_jobs": -1,
            }
            if params_override:
                params.update(params_override)

            dtrain = lgb.Dataset(train_c[feats], label=train_c["target_rank"])
            dval   = lgb.Dataset(val_c[feats],   label=val_c["target_rank"])
            model = lgb.train(params, dtrain, num_boost_round=500,
                              valid_sets=[dval],
                              callbacks=[lgb.early_stopping(30, verbose=False),
                                         lgb.log_evaluation(-1)])

            test_c = test[feats + ["target_rank", "timestamp", "symbol"]].dropna()
            test_pred = model.predict(test_c[feats])

            fwd_data = test[["timestamp", "symbol", "fwd_ret_12h"]].rename(
                columns={"fwd_ret_12h": "fwd_ret"}).dropna()
            merged = test_c[["timestamp", "symbol"]].copy()
            merged["pred"] = test_pred
            merged = merged.merge(fwd_data, on=["timestamp", "symbol"], how="inner")
            merged["window"] = w["name"]
            seed_preds.append(merged)

        if seed_preds:
            all_preds.append(pd.concat(seed_preds, ignore_index=True))

    if not all_preds:
        return None
    combined = pd.concat(all_preds, ignore_index=True)
    ens = (combined.groupby(["timestamp", "symbol"])
           .agg(pred=("pred", "mean"), fwd_ret=("fwd_ret", "first"),
                window=("window", "first"))
           .reset_index())
    return ens


# ═══════════════════════════════════════════
# RIDGE MODEL
# ═══════════════════════════════════════════
def train_ridge(df, feats):
    """Train Ridge model (simpler baseline)."""
    feat_r = [f"{f}_r" for f in feats]
    results = []

    for w in WINDOWS:
        tz = df["timestamp"].dt.tz
        train = df[df["timestamp"] < pd.Timestamp(w["train_end"], tz=tz)].copy()
        val   = df[(df["timestamp"] >= pd.Timestamp(w["val_start"], tz=tz)) &
                   (df["timestamp"] <  pd.Timestamp(w["val_end"],   tz=tz))].copy()
        test  = df[(df["timestamp"] >= pd.Timestamp(w["test_start"], tz=tz)) &
                   (df["timestamp"] <= pd.Timestamp(w["test_end"],   tz=tz))].copy()
        if len(train) < 5000 or len(test) < 200:
            continue

        for d in [train, val, test]:
            for feat in feats:
                d[f"{feat}_r"] = cs_rank(d, feat)
            d["target_rank"] = d.groupby("timestamp")["fwd_ret_12h"].rank(pct=True) - 0.5

        train_c = train[feat_r + ["target_rank"]].dropna()
        val_c   = val[feat_r + ["target_rank"]].dropna()

        # HPO on val
        best_alpha, best_ic = 1.0, -999
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
            m = Ridge(alpha=alpha)
            m.fit(train_c[feat_r], train_c["target_rank"])
            pred_v = m.predict(val_c[feat_r])
            ic = stats.spearmanr(pred_v, val_c["target_rank"])[0]
            if ic > best_ic:
                best_ic = ic
                best_alpha = alpha

        # Retrain on train+val
        m = Ridge(alpha=best_alpha)
        m.fit(pd.concat([train_c[feat_r], val_c[feat_r]]),
              pd.concat([train_c["target_rank"], val_c["target_rank"]]))

        test_c = test[feat_r + ["target_rank", "timestamp", "symbol"]].dropna()
        test_c = test_c.copy()
        test_c["pred"] = m.predict(test_c[feat_r])

        fwd_data = test[["timestamp", "symbol", "fwd_ret_12h"]].rename(
            columns={"fwd_ret_12h": "fwd_ret"}).dropna()
        merged = test_c[["timestamp", "symbol", "pred"]].merge(
            fwd_data, on=["timestamp", "symbol"], how="inner")
        merged["window"] = w["name"]
        results.append(merged)
        print(f"    Ridge {w['name']}: α={best_alpha:.0f} val_IC={best_ic:.3f}")

    if not results:
        return None
    return pd.concat(results, ignore_index=True)


# ═══════════════════════════════════════════
# IC & SIGNAL DIAGNOSTICS
# ═══════════════════════════════════════════
def ic_analysis(preds, label):
    """Per-window IC analysis."""
    print(f"\n  IC analysis: {label}")
    for wname in ["W1", "W2", "W3"]:
        sub = preds[preds["window"] == wname] if "window" in preds.columns else preds
        if len(sub) == 0:
            continue
        # Per-timestamp IC
        ics = []
        for ts, grp in sub.groupby("timestamp"):
            if len(grp) >= 10:
                ic = stats.spearmanr(grp["pred"], grp["fwd_ret"])[0]
                ics.append(ic)
        if ics:
            ic_arr = np.array(ics)
            print(f"    {wname}: meanIC={ic_arr.mean():.4f} ±{ic_arr.std():.4f} "
                  f"IC>0={100*(ic_arr>0).mean():.0f}% n={len(ics)}")


# ═══════════════════════════════════════════
# PERMUTATION TEST
# ═══════════════════════════════════════════
def permutation_test(preds, regime_df, cfg, n_perm=100, label=""):
    """Shuffle predictions across symbols within each timestamp."""
    real_sub = simulate(preds, regime_df, 12, cfg)
    real_r = eval_config(real_sub, 12, "real")
    if real_r is None:
        print(f"  Permutation test ({label}): no result")
        return
    real_sh = real_r["sharpe"]

    perm_sharpes = []
    for i in range(n_perm):
        perm = preds.copy()
        perm["pred"] = perm.groupby("timestamp")["pred"].transform(
            lambda x: np.random.permutation(x.values))
        sub = simulate(perm, regime_df, 12, cfg)
        r = eval_config(sub, 12, "perm")
        if r:
            perm_sharpes.append(r["sharpe"])

    perm_arr = np.array(perm_sharpes)
    z = (real_sh - perm_arr.mean()) / (perm_arr.std() + 1e-10)
    p = (perm_arr >= real_sh).mean()
    print(f"  Permutation test ({label}): real_Sh={real_sh:.2f} "
          f"perm_mean={perm_arr.mean():.2f} ±{perm_arr.std():.2f} "
          f"z={z:.2f} p={p:.4f}")
    return real_sh, perm_arr


# ═══════════════════════════════════════════
# PER-WINDOW DECOMPOSITION
# ═══════════════════════════════════════════
def per_window_analysis(preds, regime_df, cfg, label=""):
    """Evaluate each window separately."""
    print(f"\n  Per-window analysis: {label}")
    for wname in ["W1", "W2", "W3"]:
        sub_preds = preds[preds["window"] == wname] if "window" in preds.columns else preds
        if len(sub_preds) == 0:
            print(f"    {wname}: no data")
            continue
        port = simulate(sub_preds, regime_df, 12, cfg)
        r = eval_config(port, 12, f"{wname}")
        if r:
            wm = f"{r['win_months']}/{r['total_months']}"
            print(f"    {wname}: Sh={r['sharpe']:+.2f} WM={wm} "
                  f"Wr={r['worst_m']*100:+.1f}% Eq=${r['equity']:.0f}")
        else:
            print(f"    {wname}: insufficient data")


def main():
    t0 = time.time()
    print("=" * 80)
    print("  R17 — COMPREHENSIVE RETEST (FIXED SIMULATION)")
    print("  - simulate() processes only rebalance timestamps")
    print("  - eval_config() uses actual observation frequency")
    print("  - eq_mom_boost / strategy_momentum REMOVED")
    print("=" * 80)

    # ── Load data ──
    print("\n  Loading data...")
    ohlcv  = load_ohlcv()
    ohlcv  = ohlcv[ohlcv["symbol"].isin(SYM_35)]
    derivs = load_derivatives()
    df     = build_features_minimal(ohlcv, derivs)
    regime_df = compute_regime(df)
    print(f"  Data: {len(df):,} rows, {df['symbol'].nunique()} symbols, "
          f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

    results_all = []

    # ═══════════════════════════════════════════════
    # SECTION 1: LGB R13 (production config)
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 80)
    print("  SECTION 1: LightGBM R13 config (5-seed ensemble, 12 features)")
    print("═" * 80)

    feats_12 = [f for f in FEATURES_12 if f in df.columns]
    print(f"\n  Training LGB ensemble ({len(feats_12)} features, {len(SEEDS)} seeds)...")
    lgb_preds = train_lgb_ensemble(df, feats_12)
    if lgb_preds is None:
        print("  ERROR: no predictions")
        return

    print(f"  Total predictions: {len(lgb_preds):,}")
    ic_analysis(lgb_preds, "LGB-R13")

    # Test portfolio configs
    print(f"\n{'─' * 80}")
    print(f"  LGB R13 — Portfolio config sweep")
    print(f"{'─' * 80}")

    configs = [
        ("Bare-bones 6L3S (no overlays)",
         {"n_long": 6, "n_short": 3, "trend_cutoff": 999,
          "dyn_threshold": None, "kelly_sizing": False,
          "vol_scaling": False, "regime_asym": False, "rebal_hours": 12}),

        ("Bare-bones 5L5S",
         {"n_long": 5, "n_short": 5, "trend_cutoff": 999,
          "dyn_threshold": None, "kelly_sizing": False,
          "vol_scaling": False, "regime_asym": False, "rebal_hours": 12}),

        ("Bare-bones 4L4S",
         {"n_long": 4, "n_short": 4, "trend_cutoff": 999,
          "dyn_threshold": None, "kelly_sizing": False,
          "vol_scaling": False, "regime_asym": False, "rebal_hours": 12}),

        ("Bare-bones 8L4S",
         {"n_long": 8, "n_short": 4, "trend_cutoff": 999,
          "dyn_threshold": None, "kelly_sizing": False,
          "vol_scaling": False, "regime_asym": False, "rebal_hours": 12}),

        ("6L3S + kelly",
         {"n_long": 6, "n_short": 3, "trend_cutoff": 999,
          "dyn_threshold": None, "kelly_sizing": True,
          "vol_scaling": False, "regime_asym": False, "rebal_hours": 12}),

        ("6L3S + regime filter",
         {"n_long": 6, "n_short": 3, "trend_cutoff": 0.8,
          "dyn_threshold": 0.5, "kelly_sizing": False,
          "vol_scaling": False, "regime_asym": False, "rebal_hours": 12}),

        ("6L3S + kelly + regime",
         {"n_long": 6, "n_short": 3, "trend_cutoff": 0.8,
          "dyn_threshold": 0.5, "kelly_sizing": True,
          "vol_scaling": False, "regime_asym": False, "rebal_hours": 12}),

        ("6L3S + kelly + regime + vol_scale",
         {"n_long": 6, "n_short": 3, "trend_cutoff": 0.8,
          "dyn_threshold": 0.5, "kelly_sizing": True,
          "vol_scaling": True, "regime_asym": False, "rebal_hours": 12}),

        ("6L3S + kelly + regime + regime_asym",
         {"n_long": 6, "n_short": 3, "trend_cutoff": 0.8,
          "dyn_threshold": 0.5, "kelly_sizing": True,
          "vol_scaling": False, "regime_asym": True, "rebal_hours": 12}),

        ("6L3S + all overlays (no eq_mom/strat_mom)",
         {"n_long": 6, "n_short": 3, "trend_cutoff": 0.8,
          "dyn_threshold": 0.5, "kelly_sizing": True,
          "vol_scaling": True, "regime_asym": True, "rebal_hours": 12}),
    ]

    for name, cfg in configs:
        sub = simulate(lgb_preds, regime_df, 12, cfg)
        r = eval_config(sub, 12, f"LGB {name}", LEVERAGE, CAPITAL)
        if r:
            results_all.append(r)
            show(r)

    # ═══════════════════════════════════════════════
    # SECTION 2: LGB hyperparameter variants
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 80)
    print("  SECTION 2: LGB hyperparameter variants")
    print("═" * 80)

    # Use bare-bones config for fair comparison
    cfg_bare = {"n_long": 6, "n_short": 3, "trend_cutoff": 999,
                "dyn_threshold": None, "kelly_sizing": False,
                "vol_scaling": False, "regime_asym": False, "rebal_hours": 12}

    lgb_variants = [
        ("lr=0.02", {"learning_rate": 0.02}),
        ("lr=0.05", {"learning_rate": 0.05}),
        ("nl=31", {"num_leaves": 31}),
        ("nl=127", {"num_leaves": 127}),
        ("L2=0.1", {"lambda_l2": 0.1}),
        ("L2=2.0", {"lambda_l2": 2.0}),
        ("L2=5.0", {"lambda_l2": 5.0}),
        ("ExtraTrees", {"extra_trees": True}),
    ]

    for vname, vparams in lgb_variants:
        print(f"\n  Training LGB variant: {vname}...")
        preds = train_lgb_ensemble(df, feats_12, params_override=vparams)
        if preds is None:
            print(f"    SKIP: no predictions")
            continue
        sub = simulate(preds, regime_df, 12, cfg_bare)
        r = eval_config(sub, 12, f"LGB {vname} bare-bones", LEVERAGE, CAPITAL)
        if r:
            results_all.append(r)
            show(r)

    # ═══════════════════════════════════════════════
    # SECTION 3: Ridge model
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 80)
    print("  SECTION 3: Ridge model (simpler baseline)")
    print("═" * 80)

    for feat_set_name, feat_set in [("12f", FEATURES_12), ("14f", FEATURES), ("9f", FEATURES_MINIMAL)]:
        feats = [f for f in feat_set if f in df.columns]
        print(f"\n  Training Ridge ({feat_set_name}: {len(feats)} features)...")
        ridge_preds = train_ridge(df, feats)
        if ridge_preds is None:
            print(f"    SKIP: no predictions")
            continue

        ic_analysis(ridge_preds, f"Ridge-{feat_set_name}")

        for cfg_name, cfg in [("bare-bones", cfg_bare),
                               ("+ kelly", {**cfg_bare, "kelly_sizing": True})]:
            sub = simulate(ridge_preds, regime_df, 12, cfg)
            r = eval_config(sub, 12, f"Ridge {feat_set_name} {cfg_name}", LEVERAGE, CAPITAL)
            if r:
                results_all.append(r)
                show(r)

    # ═══════════════════════════════════════════════
    # SECTION 4: Feature set ablation (LGB)
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 80)
    print("  SECTION 4: Feature set ablation (LGB)")
    print("═" * 80)

    for feat_set_name, feat_set in [("14f", FEATURES), ("9f_minimal", FEATURES_MINIMAL)]:
        feats = [f for f in feat_set if f in df.columns]
        print(f"\n  Training LGB {feat_set_name} ({len(feats)} features)...")
        preds = train_lgb_ensemble(df, feats)
        if preds is None:
            continue
        sub = simulate(preds, regime_df, 12, cfg_bare)
        r = eval_config(sub, 12, f"LGB {feat_set_name} bare-bones", LEVERAGE, CAPITAL)
        if r:
            results_all.append(r)
            show(r)

    # ═══════════════════════════════════════════════
    # SECTION 5: Per-window decomposition (best config)
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 80)
    print("  SECTION 5: Per-window decomposition")
    print("═" * 80)

    per_window_analysis(lgb_preds, regime_df, cfg_bare, "LGB R13 bare-bones")

    # Also test with kelly
    cfg_kelly = {**cfg_bare, "kelly_sizing": True}
    per_window_analysis(lgb_preds, regime_df, cfg_kelly, "LGB R13 + kelly")

    # ═══════════════════════════════════════════════
    # SECTION 6: Permutation test
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 80)
    print("  SECTION 6: Permutation test (final verification)")
    print("═" * 80)

    permutation_test(lgb_preds, regime_df, cfg_bare, n_perm=100,
                     label="LGB R13 bare-bones")

    # ═══════════════════════════════════════════════
    # SECTION 7: Rankings & Recommendation
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 80)
    print("  RANKINGS — ALL CONFIGS (fixed simulation)")
    print("═" * 80)

    if not results_all:
        print("  No results")
        return

    results_all.sort(key=lambda x: x["sharpe"], reverse=True)

    print(f"\n  By Sharpe:")
    for i, r in enumerate(results_all):
        wm = f"{r['win_months']}/{r['total_months']}"
        flag = "✅" if r["worst_m"] > -0.15 else ("⚠️" if r["worst_m"] > -0.25 else "❌")
        print(f"    #{i+1:2d} {flag} {r['name']:<55s} Sh={r['sharpe']:+.2f} "
              f"WM={wm} Wr={r['worst_m']*100:+.1f}% Eq=${r['equity']:.0f}")

    # Best by equity
    results_all.sort(key=lambda x: x["equity"], reverse=True)
    print(f"\n  By Equity:")
    for i, r in enumerate(results_all[:5]):
        wm = f"{r['win_months']}/{r['total_months']}"
        print(f"    #{i+1} {r['name']:<55s} Eq=${r['equity']:.0f} Sh={r['sharpe']:+.2f} WM={wm}")

    # Monthly detail for top-3
    results_all.sort(key=lambda x: x["sharpe"], reverse=True)
    print(f"\n{'─' * 80}")
    print(f"  Monthly detail — top 3")
    print(f"{'─' * 80}")
    for r in results_all[:3]:
        print(f"\n  {r['name']}: Sh={r['sharpe']:.2f}")
        for md in r.get("month_data", []):
            marker = " ← worst" if md.get("ret") == r["worst_m"] else ""
            print(f"    {md['month']:>10s}  {md['ret']*100:>+7.1f}%  eq=${md['equity']:>7.0f}{marker}")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
