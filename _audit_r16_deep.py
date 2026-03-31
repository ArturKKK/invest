#!/usr/bin/env python3
"""
R16 — Deep Code & Pipeline Audit.

Manual code review found subagent flagged many FALSE ALARMS.
These are VERIFIED CORRECT:
  ✅ fwd_ret = pct_change(h).shift(-h) → correct forward return
  ✅ train/val boundary: < vs >= → no overlap
  ✅ CS-rank per split → within-timestamp, uniform by construction
  ✅ Realized vol includes current bar → standard, not look-ahead
  ✅ Funding rate ffill → last known, not future

THIS SCRIPT tests REAL potential issues:

CHECK A: Verify fwd_ret is actually forward-looking (numeric test)
CHECK B: Overlapping returns in simulate() — does iloc[::12] work correctly?
CHECK C: EQ momentum boost / strategy momentum — do they inflate Sharpe?
CHECK D: Regime filter skip bias — are we cherry-picking calm periods?
CHECK E: Return autocorrelation — does annualization factor overstate Sharpe?
CHECK F: Permutation test on the SIMULATION itself (not just model)
CHECK G: Per-window Sharpe consistency — is one window driving everything?
CHECK H: Alternate Sharpe calculation — independent verification
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
    compute_regime, simulate, eval_config,
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


def get_ensemble_predictions(df, feats, regime_df):
    """Run full walk-forward with R13 config, return (ens_predictions, raw_returns)."""
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
            dtrain = lgb.Dataset(train_c[feats], label=train_c["target_rank"])
            dval   = lgb.Dataset(val_c[feats],   label=val_c["target_rank"])
            model = lgb.train(params, dtrain, num_boost_round=N_ROUNDS,
                              valid_sets=[dval],
                              callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
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

    combined = pd.concat(all_preds, ignore_index=True)
    ens = (combined.groupby(["timestamp", "symbol"])
           .agg(pred=("pred", "mean"), fwd_ret=("fwd_ret", "first"),
                window=("window", "first"))
           .reset_index())
    return ens


def main():
    t0 = time.time()
    print("=" * 70)
    print("  R16 — DEEP CODE & PIPELINE AUDIT")
    print("  Testing for bugs, leakage, and inflated results")
    print("=" * 70)

    print("\n📊 Loading data...")
    ohlcv  = load_ohlcv()
    ohlcv  = ohlcv[ohlcv["symbol"].isin(SYM_35)]
    derivs = load_derivatives()
    df     = build_features_minimal(ohlcv, derivs)
    feats  = [f for f in FEATURES_12 if f in df.columns]
    print(f"   df: ({len(df):,}, {len(df.columns)})")

    regime_df = compute_regime(df)

    # ═══════════════════════════════════════════════
    # CHECK A: Verify fwd_ret is forward-looking
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  CHECK A — Forward Return Verification")
    print("═" * 70)

    btc = df[df["symbol"] == "BTC/USDT"].sort_values("timestamp").reset_index(drop=True)
    # Manually compute forward return at a few random indices
    n_checks = 20
    np.random.seed(42)
    indices = np.random.choice(range(100, len(btc) - 100), n_checks, replace=False)

    errors = 0
    for idx in indices:
        close_now = btc.loc[idx, "close"]
        close_12 = btc.loc[idx + 12, "close"]
        expected_fwd = (close_12 - close_now) / close_now
        actual_fwd = btc.loc[idx, "fwd_ret_12h"]
        ts = btc.loc[idx, "timestamp"]

        if pd.isna(actual_fwd):
            print(f"   ⚠️  {ts}: fwd_ret_12h is NaN")
            continue

        diff = abs(expected_fwd - actual_fwd)
        if diff > 1e-10:
            print(f"   ❌ {ts}: expected={expected_fwd:.8f}, actual={actual_fwd:.8f}, diff={diff:.2e}")
            errors += 1

    if errors == 0:
        print(f"   ✅ PASS: All {n_checks} forward returns verified correct")
        print(f"      fwd_ret_12h[T] = (close[T+12] - close[T]) / close[T]")
    else:
        print(f"   ❌ FAIL: {errors}/{n_checks} mismatches!")

    # Also verify it's NOT backward-looking
    for idx in indices[:5]:
        close_now = btc.loc[idx, "close"]
        close_m12 = btc.loc[idx - 12, "close"]
        backward_ret = (close_now - close_m12) / close_m12
        actual_fwd = btc.loc[idx, "fwd_ret_12h"]
        if pd.notna(actual_fwd) and abs(backward_ret - actual_fwd) < 1e-10:
            print(f"   ❌ ALARM: fwd_ret equals BACKWARD return at idx {idx}!")
            errors += 1

    if errors == 0:
        print(f"   ✅ Confirmed: fwd_ret is NOT backward-looking")

    # ═══════════════════════════════════════════════
    # GET BASELINE PREDICTIONS
    # ═══════════════════════════════════════════════
    print("\n📊 Training baseline models...")
    ens = get_ensemble_predictions(df, feats, regime_df)
    print(f"   Ensemble predictions: {len(ens):,} rows")

    # Baseline result
    sub_base = simulate(ens, regime_df, 12, CFG_BASE)
    r_base = eval_config(sub_base, 12, "BASELINE", LEVERAGE, CAPITAL)
    print(f"   Baseline: Sh={r_base['sharpe']:.2f}, WM={r_base['win_months']}/{r_base['total_months']}")

    # ═══════════════════════════════════════════════
    # CHECK B: Does iloc[::12] subsampling work correctly?
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  CHECK B — Rebalance Subsampling Verification")
    print("═" * 70)

    # Get all returns before subsampling
    sub_all = simulate(ens, regime_df, 12, {**CFG_BASE, "rebal_hours": 1})  # every hour
    sub_12h = simulate(ens, regime_df, 12, CFG_BASE)  # every 12h

    print(f"   All hourly returns: {len(sub_all)}")
    print(f"   Subsampled to 12h:  {len(sub_12h)}")

    # Check timestamp spacing in the 12h subsample
    ts_sorted = sub_12h["timestamp"].sort_values()
    diffs = ts_sorted.diff().dropna()
    diffs_hours = diffs.dt.total_seconds() / 3600

    print(f"   Timestamp gaps (hours): min={diffs_hours.min():.0f}, "
          f"max={diffs_hours.max():.0f}, median={diffs_hours.median():.0f}")

    # Count non-12h gaps
    non_12h = (diffs_hours != 12).sum()
    pct_irregular = non_12h / len(diffs_hours) * 100
    print(f"   Non-12h gaps: {non_12h}/{len(diffs_hours)} ({pct_irregular:.1f}%)")

    if pct_irregular > 5:
        print(f"   ⚠️  WARNING: {pct_irregular:.1f}% of intervals are not exactly 12h")
        print(f"      This means iloc[::12] skips some periods (regime filter?)")
        # What fraction of hourly timestamps were skipped?
        n_expected = len(ens["timestamp"].unique())
        n_got = len(sub_all)
        skip_pct = (1 - n_got / n_expected) * 100
        print(f"      Timestamps in data: {n_expected}, in simulation: {n_got}")
        print(f"      Skipped by regime filter: {skip_pct:.1f}%")
    else:
        print(f"   ✅ PASS: Subsampling is clean")

    # Check if hourly Sharpe inflates vs actual 12h
    rets_1h = sub_all["portfolio_ret"]
    rets_12h = sub_12h["portfolio_ret"]

    sh_1h = rets_1h.mean() / (rets_1h.std() + 1e-10) * np.sqrt(8760)
    sh_12h = rets_12h.mean() / (rets_12h.std() + 1e-10) * np.sqrt(730)
    print(f"   Sharpe (1h subsample, annualized): {sh_1h:.2f}")
    print(f"   Sharpe (12h subsample, annualized): {sh_12h:.2f}")

    # ═══════════════════════════════════════════════
    # CHECK C: Do portfolio bells & whistles inflate Sharpe?
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  CHECK C — Portfolio Enhancements Impact")
    print("═" * 70)

    # Bare-bones config: no EQ momentum, no strategy momentum, no vol scaling,
    # no regime asymmetry, no kelly, no dynamic threshold
    CFG_BARE = {
        "n_long": 6, "n_short": 3,
        "trend_cutoff": 999,       # never filter
        "dyn_threshold": None,     # no dynamic exposure
        "eq_mom_boost": False,
        "kelly_sizing": False,
        "strategy_momentum": False,
        "regime_asym": False,
        "vol_scaling": False,
        "signal_ema": None,
        "rebal_hours": 12,
    }

    sub_bare = simulate(ens, regime_df, 12, CFG_BARE)
    r_bare = eval_config(sub_bare, 12, "BARE-BONES (no enhancements)", LEVERAGE, CAPITAL)

    if r_bare:
        delta = r_base["sharpe"] - r_bare["sharpe"]
        print(f"   BARE-BONES:  Sh={r_bare['sharpe']:.2f}, "
              f"WM={r_bare['win_months']}/{r_bare['total_months']}, "
              f"Wr={r_bare['worst_m']*100:.1f}%")
        print(f"   FULL CONFIG: Sh={r_base['sharpe']:.2f}, "
              f"WM={r_base['win_months']}/{r_base['total_months']}, "
              f"Wr={r_base['worst_m']*100:.1f}%")
        print(f"   Delta: {delta:+.2f} Sharpe from portfolio enhancements")

        if delta > 2.0:
            print(f"   ❌ ALERT: Enhancements add >{delta:.1f} Sharpe — "
                  f"model signal may be weak, simulation tricks inflate results!")
        elif delta > 1.0:
            print(f"   ⚠️  WARNING: Enhancements add {delta:.1f} Sharpe — "
                  f"significant contribution from portfolio logic")
        else:
            print(f"   ✅ PASS: Enhancements add only {delta:.1f} Sharpe — "
                  f"core signal is strong")

    # Test each enhancement individually
    enhancements = [
        ("regime_filter", {**CFG_BARE, "trend_cutoff": 0.8}),
        ("dyn_threshold", {**CFG_BARE, "dyn_threshold": 0.5, "trend_cutoff": 0.8}),
        ("kelly_sizing", {**CFG_BARE, "kelly_sizing": True}),
        ("vol_scaling", {**CFG_BARE, "vol_scaling": True}),
        ("eq_mom_boost", {**CFG_BARE, "eq_mom_boost": True}),
        ("strat_momentum", {**CFG_BARE, "strategy_momentum": True, "strat_mom_lookback": 48}),
        ("regime_asym", {**CFG_BARE, "regime_asym": True}),
    ]

    print(f"\n   Individual enhancement contributions (vs bare-bones Sh={r_bare['sharpe']:.2f}):")
    for name, cfg in enhancements:
        sub = simulate(ens, regime_df, 12, cfg)
        r = eval_config(sub, 12, name, LEVERAGE, CAPITAL)
        if r:
            d = r["sharpe"] - r_bare["sharpe"]
            print(f"     {name:<20s}: Sh={r['sharpe']:.2f} (delta={d:+.2f})")

    # ═══════════════════════════════════════════════
    # CHECK D: Regime filter selection bias
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  CHECK D — Regime Filter Selection Bias")
    print("═" * 70)

    # How many timestamps are skipped by trend_cutoff=0.8?
    ens_ts = sorted(ens["timestamp"].unique())
    n_total = len(ens_ts)
    n_skipped = 0
    for ts in ens_ts:
        if ts in regime_df.index:
            trend_str = regime_df.loc[ts].get("trend_strength", 0)
            if trend_str > 0.8:
                n_skipped += 1

    skip_pct = n_skipped / n_total * 100
    print(f"   Total timestamps: {n_total}")
    print(f"   Skipped (trend>0.8): {n_skipped} ({skip_pct:.1f}%)")

    # Compare returns in skipped vs included periods
    included_fwd = []
    skipped_fwd = []
    for ts in ens_ts:
        if ts in regime_df.index:
            grp = ens[ens["timestamp"] == ts]
            avg_fwd = grp["fwd_ret"].mean()
            trend_str = regime_df.loc[ts].get("trend_strength", 0)
            if trend_str > 0.8:
                skipped_fwd.append(avg_fwd)
            else:
                included_fwd.append(avg_fwd)

    if skipped_fwd:
        print(f"   Avg fwd_ret in included periods: {np.mean(included_fwd)*100:.3f}%")
        print(f"   Avg fwd_ret in skipped periods:  {np.mean(skipped_fwd)*100:.3f}%")
        print(f"   Std fwd_ret in included periods: {np.std(included_fwd)*100:.3f}%")
        print(f"   Std fwd_ret in skipped periods:  {np.std(skipped_fwd)*100:.3f}%")
    else:
        print(f"   ✅ No timestamps skipped by regime filter")

    # ═══════════════════════════════════════════════
    # CHECK E: Return autocorrelation
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  CHECK E — Return Autocorrelation Check")
    print("═" * 70)

    rets_12h = sub_base["portfolio_ret"].values
    # Ljung-Box test for autocorrelation
    n = len(rets_12h)
    max_lag = min(20, n // 5)
    print(f"   N returns: {n}, testing lags 1-{max_lag}")

    autocorrs = []
    for lag in range(1, max_lag + 1):
        ac = pd.Series(rets_12h).autocorr(lag)
        autocorrs.append((lag, ac))

    significant = [(lag, ac) for lag, ac in autocorrs if abs(ac) > 2 / np.sqrt(n)]
    print(f"   Significant autocorrelations (|AC| > {2/np.sqrt(n):.3f}):")
    if significant:
        for lag, ac in significant:
            print(f"     Lag {lag}: AC={ac:.3f}")
        # If AC[1] is positive, standard Sharpe overestimates
        ac1 = autocorrs[0][1]
        if ac1 > 0:
            # Adjusted Sharpe ≈ Sharpe * sqrt((1-ac1)/(1+ac1))
            adj_factor = np.sqrt((1 - ac1) / (1 + ac1))
            adj_sharpe = r_base["sharpe"] * adj_factor
            print(f"   ⚠️  AC(1)={ac1:.3f} > 0: Sharpe may be overestimated")
            print(f"      Adjusted Sharpe ≈ {r_base['sharpe']:.2f} × {adj_factor:.3f} = {adj_sharpe:.2f}")
        else:
            print(f"   ✅ AC(1)={ac1:.3f} ≤ 0: no positive autocorrelation inflation")
    else:
        print(f"   ✅ PASS: No significant autocorrelation at any lag")

    # ═══════════════════════════════════════════════
    # CHECK F: Permutation test on SIMULATION
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  CHECK F — Permutation Test on Full Pipeline")
    print("═" * 70)

    N_PERM = 100
    print(f"   Running {N_PERM} permutations (shuffling predictions within timestamp)...")

    perm_sharpes = []
    for i in range(N_PERM):
        ens_perm = ens.copy()
        # Shuffle predictions WITHIN each timestamp (preserves cross-sectional structure)
        ens_perm["pred"] = ens_perm.groupby("timestamp")["pred"].transform(
            lambda x: x.sample(frac=1.0, random_state=i).values
        )
        sub_perm = simulate(ens_perm, regime_df, 12, CFG_BASE)
        r_perm = eval_config(sub_perm, 12, f"perm_{i}", LEVERAGE, CAPITAL)
        if r_perm:
            perm_sharpes.append(r_perm["sharpe"])
        if (i + 1) % 20 == 0:
            print(f"     ... {i+1}/{N_PERM} done")

    perm_sharpes = np.array(perm_sharpes)
    real_sh = r_base["sharpe"]
    p_value = (perm_sharpes >= real_sh).mean()
    z_score = (real_sh - perm_sharpes.mean()) / (perm_sharpes.std() + 1e-10)

    print(f"\n   Real Sharpe: {real_sh:.2f}")
    print(f"   Permuted: mean={perm_sharpes.mean():.2f}, std={perm_sharpes.std():.2f}")
    print(f"   p-value: {p_value:.4f}")
    print(f"   z-score: {z_score:.2f}")

    if p_value < 0.01:
        print(f"   ✅ PASS: Real Sharpe is {z_score:.1f}σ above random (p={p_value:.4f})")
    elif p_value < 0.05:
        print(f"   ⚠️  WARNING: Marginal significance (p={p_value:.2f})")
    else:
        print(f"   ❌ FAIL: Real Sharpe is NOT significantly better than random (p={p_value:.2f})")

    # ═══════════════════════════════════════════════
    # CHECK G: Per-window Sharpe consistency
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  CHECK G — Per-Window Sharpe Decomposition")
    print("═" * 70)

    for w in WINDOWS:
        ens_w = ens[ens["window"] == w["name"]].copy()
        if len(ens_w) < 100:
            print(f"   {w['name']}: too few rows ({len(ens_w)})")
            continue

        sub_w = simulate(ens_w, regime_df, 12, CFG_BASE)
        if sub_w is None or len(sub_w) < 5:
            print(f"   {w['name']}: simulation returned too few rows")
            continue

        r_w = eval_config(sub_w, 12, w["name"], LEVERAGE, CAPITAL)
        if r_w:
            ic_w = stats.spearmanr(ens_w["pred"], ens_w["fwd_ret"])[0]
            print(f"   {w['name']}: Sh={r_w['sharpe']:+.2f}, "
                  f"WM={r_w['win_months']}/{r_w['total_months']}, "
                  f"Wr={r_w['worst_m']*100:+.1f}%, IC={ic_w:.3f}")

    # ═══════════════════════════════════════════════
    # CHECK H: Independent Sharpe verification
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  CHECK H — Independent Sharpe Calculation")
    print("═" * 70)

    # Method 1: Our eval_config
    sh1 = r_base["sharpe"]

    # Method 2: Manual from raw returns
    rets = sub_base["portfolio_ret"].values
    n_rets = len(rets)
    ts_range = (sub_base["timestamp"].max() - sub_base["timestamp"].min()).total_seconds() / 3600
    actual_obs_per_year = n_rets / (ts_range / 8760)
    sh2 = rets.mean() / (rets.std() + 1e-10) * np.sqrt(actual_obs_per_year)

    # Method 3: Use actual calendar days
    n_days = (sub_base["timestamp"].max() - sub_base["timestamp"].min()).days
    annual_mult = 365.25 / n_days
    cum_ret = np.prod(1 + rets) - 1
    ann_ret = (1 + cum_ret) ** annual_mult - 1
    ann_vol_daily = rets.std() * np.sqrt(2)  # 2 obs per day
    ann_vol = ann_vol_daily * np.sqrt(365.25)
    sh3 = ann_ret / (ann_vol + 1e-10)

    print(f"   Method 1 (eval_config, ppy=730):     Sh={sh1:.2f}")
    print(f"   Method 2 (actual obs frequency):      Sh={sh2:.2f}")
    print(f"   Method 3 (calendar CAGR/vol):         Sh={sh3:.2f}")
    print(f"   Obs: {n_rets}, Days: {n_days}, obs/year: {actual_obs_per_year:.0f}")

    diff_12 = abs(sh1 - sh2)
    diff_13 = abs(sh1 - sh3)

    if diff_12 < 0.5 and diff_13 < 0.5:
        print(f"   ✅ PASS: All methods agree within 0.5 Sharpe")
    else:
        print(f"   ⚠️  WARNING: Sharpe estimates diverge — check annualization")

    # Monthly returns breakdown
    print(f"\n   Monthly returns (leveraged {LEVERAGE}x):")
    monthly = r_base["monthly"]
    for period, ret in monthly.items():
        status = "✅" if ret > 0 else "❌"
        print(f"     {status} {period}: {ret*100:+.1f}%")

    # ═══════════════════════════════════════════════
    # CHECK I: Long-short decomposition
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  CHECK I — Long vs Short Attribution")
    print("═" * 70)

    # Run bare simulation and track long/short separately
    timestamps_sorted = sorted(ens["timestamp"].unique())
    grouped = {ts: grp for ts, grp in ens.groupby("timestamp")}

    long_rets = []
    short_rets = []
    ls_rets = []

    for ts in timestamps_sorted:
        if ts not in regime_df.index or ts not in grouped:
            continue
        grp = grouped[ts].copy()
        n = len(grp)
        if n < 9:
            continue

        nl, ns = 6, 3
        nl = min(nl, n // 3)
        ns = min(ns, n // 3)
        if nl == 0 or ns == 0:
            continue

        grp["pred_rank"] = grp["pred"].rank(ascending=False)
        longs = grp[grp["pred_rank"] <= nl]
        shorts = grp[grp["pred_rank"] > (n - ns)]

        lr = longs["fwd_ret"].mean()
        sr = shorts["fwd_ret"].mean()
        ls = 0.5 * lr - 0.5 * sr

        long_rets.append(lr)
        short_rets.append(sr)
        ls_rets.append(ls)

    long_rets = np.array(long_rets[::12])  # subsample to 12h
    short_rets = np.array(short_rets[::12])
    ls_rets = np.array(ls_rets[::12])

    print(f"   Long  leg: mean={np.mean(long_rets)*100:.3f}%, "
          f"std={np.std(long_rets)*100:.3f}%, "
          f"Sh={np.mean(long_rets)/(np.std(long_rets)+1e-10)*np.sqrt(730):.2f}")
    print(f"   Short leg: mean={np.mean(short_rets)*100:.3f}%, "
          f"std={np.std(short_rets)*100:.3f}%, "
          f"Sh={np.mean(short_rets)/(np.std(short_rets)+1e-10)*np.sqrt(730):.2f}")
    print(f"   L-S combined: mean={np.mean(ls_rets)*100:.3f}%, "
          f"std={np.std(ls_rets)*100:.3f}%, "
          f"Sh={np.mean(ls_rets)/(np.std(ls_rets)+1e-10)*np.sqrt(730):.2f}")

    # Is long or short driving the result?
    long_contrib = np.mean(long_rets) / (np.mean(long_rets) - np.mean(short_rets) + 1e-10)
    print(f"   Long contribution: {long_contrib*100:.0f}%")
    print(f"   Short contribution: {(1-long_contrib)*100:.0f}%")

    # ═══════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  R16 — AUDIT SUMMARY")
    print("═" * 70)

    elapsed = time.time() - t0
    print(f"\n  ⏱  Total time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
