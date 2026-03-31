#!/usr/bin/env python3
"""
R16B — Fixed Simulation & Sharpe.

Fixes found in R16 audit:
1. Sharpe annualization: use actual observation frequency, not assumed 730
2. eq_mom_boost look-ahead: equity_curve uses overlapping hourly returns
3. strategy_momentum: same look-ahead issue

Two solutions tested:
A) FIX the equity_curve to only update at rebal points (every 12h)
B) REMOVE eq_mom_boost and strategy_momentum entirely
C) Fix annualization to use actual observation count
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
    compute_regime, eval_config,
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


def cs_rank_inplace(df, feats):
    df = df.copy()
    for feat in feats:
        if feat in df.columns:
            df[feat] = df.groupby("timestamp")[feat].rank(pct=True) - 0.5
    return df


def simulate_fixed(merged, regime_df, horizon, cfg):
    """
    Fixed simulation that avoids look-ahead in equity momentum.
    Key fix: only update equity/strategy state at REBALANCE points,
    not at every hourly timestamp.
    """
    n_long = cfg.get("n_long", 6)
    n_short = cfg.get("n_short", 3)
    trend_cutoff = cfg.get("trend_cutoff", 0.8)
    dyn_threshold = cfg.get("dyn_threshold", 0.5)
    eq_mom_boost = cfg.get("eq_mom_boost", False)
    kelly_sizing = cfg.get("kelly_sizing", True)
    strategy_momentum = cfg.get("strategy_momentum", False)
    strat_mom_lookback = cfg.get("strat_mom_lookback", 48)
    vol_scaling = cfg.get("vol_scaling", False)
    regime_asym = cfg.get("regime_asym", False)
    rebal_hours = cfg.get("rebal_hours", 12)

    all_rets = []
    equity_curve = [1.0]
    strategy_rets = []
    prev_longs = set()
    prev_shorts = set()

    timestamps_sorted = sorted(merged["timestamp"].unique())
    grouped = {ts: grp for ts, grp in merged.groupby("timestamp")}

    # FIXED: only process every Nth timestamp (rebalance points)
    # instead of processing hourly and subsampling later
    rebal_timestamps = timestamps_sorted[::rebal_hours]

    for ts in rebal_timestamps:
        if ts not in regime_df.index or ts not in grouped:
            continue
        row = regime_df.loc[ts]
        trend_str = row.get("trend_strength", 0)
        trend_dir = row.get("trend_direction", 0)
        vol_regime_val = row.get("vol_regime", 1.0)

        if trend_str > trend_cutoff:
            continue

        grp = grouped[ts].copy()
        n = len(grp)

        # Dynamic exposure
        exposure = 1.0
        if dyn_threshold is not None and trend_str > dyn_threshold:
            exposure = max(0.1, 1.0 - (trend_str - dyn_threshold) /
                          (trend_cutoff - dyn_threshold + 1e-10) * 0.5)

        # Strategy momentum — now uses ONLY rebalance-spaced returns (no overlap)
        if strategy_momentum and len(strategy_rets) >= strat_mom_lookback:
            recent = strategy_rets[-strat_mom_lookback:]
            cum = np.prod([1 + r for r in recent])
            if cum < 0.97:
                exposure *= max(0.3, cum)

        # Vol scaling
        if vol_scaling and not np.isnan(vol_regime_val) and vol_regime_val > 0:
            vol_scale = min(1.5, 1.0 / max(0.5, vol_regime_val))
            exposure *= vol_scale

        # Regime asymmetry
        if regime_asym and not np.isnan(trend_dir):
            nl_base, ns_base = n_long, n_short
            if -0.3 < trend_dir < 0.3:
                nl, ns = nl_base, ns_base
            elif trend_dir >= 0.3:
                nl = min(n // 3, nl_base + 1)
                ns = max(2, ns_base - 1)
            else:
                nl = max(2, nl_base - 1)
                ns = min(n // 3, ns_base + 1)
        else:
            nl, ns = n_long, n_short

        nl = min(nl, n // 3)
        ns = min(ns, n // 3)
        if nl == 0 or ns == 0:
            continue

        grp["pred_rank"] = grp["pred"].rank(ascending=False)
        new_longs = set(grp[grp["pred_rank"] <= nl]["symbol"].tolist())
        new_shorts = set(grp[grp["pred_rank"] > (n - ns)]["symbol"].tolist())

        prev_longs = new_longs
        prev_shorts = new_shorts

        longs = grp[grp["symbol"].isin(new_longs)]
        shorts = grp[grp["symbol"].isin(new_shorts)]

        if len(longs) == 0 or len(shorts) == 0:
            continue

        long_ret = longs["fwd_ret"].mean()
        short_ret = shorts["fwd_ret"].mean()

        # Kelly sizing
        if kelly_sizing:
            pred_spread = longs["pred"].mean() - shorts["pred"].mean()
            long_alloc = np.clip(0.5 + pred_spread * 5, 0.3, 0.7)
            port_ret = long_alloc * long_ret - (1 - long_alloc) * short_ret
        else:
            port_ret = 0.5 * long_ret - 0.5 * short_ret

        port_ret *= exposure

        # EQ momentum — now using ONLY non-overlapping rebalance returns
        if eq_mom_boost and len(equity_curve) > 48:
            recent_eq = equity_curve[-1]
            peak_eq = max(equity_curve[-48:])
            dd = (recent_eq - peak_eq) / (peak_eq + 1e-10)
            if dd < -0.05:
                scale = max(0.3, 1.0 + dd * 3)
                port_ret *= scale
            elif dd > -0.01:
                trough_eq = min(equity_curve[-48:])
                recovery = (recent_eq - trough_eq) / (trough_eq + 1e-10)
                if recovery > 0.05:
                    boost = min(1.5, 1.0 + recovery * 0.5)
                    port_ret *= boost

        all_rets.append({"timestamp": ts, "portfolio_ret": port_ret})
        equity_curve.append(equity_curve[-1] * (1 + port_ret))
        strategy_rets.append(port_ret)

    if not all_rets:
        return None
    return pd.DataFrame(all_rets).sort_values("timestamp")


def eval_config_fixed(sub, name, leverage=5, capital=100):
    """
    Fixed eval_config that uses ACTUAL observation frequency for annualization.
    """
    if sub is None or len(sub) < 10:
        return None

    rets = sub["portfolio_ret"]
    n_obs = len(rets)

    # Actual time span
    ts_range = (sub["timestamp"].max() - sub["timestamp"].min())
    total_hours = ts_range.total_seconds() / 3600
    years = total_hours / 8760

    # Actual observations per year
    if years > 0:
        obs_per_year = n_obs / years
    else:
        obs_per_year = 730  # fallback

    sharpe = rets.mean() / (rets.std() + 1e-10) * np.sqrt(obs_per_year)

    # Monthly returns
    sub_df = sub.copy()
    sub_df["month"] = sub_df["timestamp"].dt.to_period("M")
    monthly = sub_df.groupby("month")["portfolio_ret"].apply(
        lambda x: (1 + x * leverage).prod() - 1)
    worst_m = monthly.min()

    equity = capital
    for month, ret in monthly.items():
        pnl = equity * ret
        equity += pnl

    win_months = (monthly > 0).sum()
    total_months = len(monthly)

    return {
        "name": name, "sharpe": sharpe,
        "worst_m": worst_m, "equity": equity, "monthly": monthly,
        "win_months": win_months, "total_months": total_months,
        "n_obs": n_obs, "obs_per_year": obs_per_year,
    }


def get_ensemble_predictions(df, feats):
    """Run full walk-forward with R13 config."""
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


def show_result(r):
    if r is None:
        print("   (no result)")
        return
    wm = f"{r['win_months']}/{r['total_months']}"
    print(f"   {r['name']:<55s} Sh={r['sharpe']:>+5.2f} "
          f"WM={wm} Wr={r['worst_m']*100:>+5.1f}% Eq=${r['equity']:.0f} "
          f"(n={r['n_obs']}, {r['obs_per_year']:.0f}/yr)")


def main():
    t0 = time.time()
    print("=" * 70)
    print("  R16B — FIXED Simulation & Sharpe Recalculation")
    print("  Fixing: annualization bug, eq_mom_boost look-ahead")
    print("=" * 70)

    print("\n📊 Loading data...")
    ohlcv  = load_ohlcv()
    ohlcv  = ohlcv[ohlcv["symbol"].isin(SYM_35)]
    derivs = load_derivatives()
    df     = build_features_minimal(ohlcv, derivs)
    feats  = [f for f in FEATURES_12 if f in df.columns]
    regime_df = compute_regime(df)

    print("\n📊 Training models...")
    ens = get_ensemble_predictions(df, feats)
    print(f"   {len(ens):,} predictions")

    # ═══════════════════════════════════════════════
    # 1. OLD simulation (buggy) — for reference
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  OLD (buggy) vs FIXED simulation")
    print("═" * 70)

    from _research_round7 import simulate

    CFG_OLD = {
        "n_long": 6, "n_short": 3,
        "trend_cutoff": 0.8, "dyn_threshold": 0.5,
        "eq_mom_boost": True, "kelly_sizing": True,
        "strategy_momentum": True, "strat_mom_lookback": 48,
        "regime_asym": True, "vol_scaling": True,
        "signal_ema": None, "rebal_hours": 12,
    }

    sub_old = simulate(ens, regime_df, 12, CFG_OLD)
    r_old = eval_config(sub_old, 12, "OLD: full config (buggy Sh)", LEVERAGE, CAPITAL)
    r_old_fixed_ann = eval_config_fixed(sub_old, "OLD sim + fixed annualization", LEVERAGE, CAPITAL)
    print("   OLD simulation:")
    print(f"     Buggy annualization:  Sh={r_old['sharpe']:.2f}")
    print(f"     Fixed annualization:  Sh={r_old_fixed_ann['sharpe']:.2f} "
          f"({r_old_fixed_ann['obs_per_year']:.0f} obs/yr)")

    # ═══════════════════════════════════════════════
    # 2. FIXED simulation variations
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  FIXED simulation — various configs")
    print("═" * 70)

    configs = [
        ("FIX-1: bare-bones (no overlays)",
         {"n_long": 6, "n_short": 3, "trend_cutoff": 999,
          "dyn_threshold": None, "eq_mom_boost": False,
          "kelly_sizing": False, "strategy_momentum": False,
          "regime_asym": False, "vol_scaling": False,
          "rebal_hours": 12}),

        ("FIX-2: + kelly sizing",
         {"n_long": 6, "n_short": 3, "trend_cutoff": 999,
          "dyn_threshold": None, "eq_mom_boost": False,
          "kelly_sizing": True, "strategy_momentum": False,
          "regime_asym": False, "vol_scaling": False,
          "rebal_hours": 12}),

        ("FIX-3: + regime filter",
         {"n_long": 6, "n_short": 3, "trend_cutoff": 0.8,
          "dyn_threshold": 0.5, "eq_mom_boost": False,
          "kelly_sizing": True, "strategy_momentum": False,
          "regime_asym": False, "vol_scaling": False,
          "rebal_hours": 12}),

        ("FIX-4: + vol scaling",
         {"n_long": 6, "n_short": 3, "trend_cutoff": 0.8,
          "dyn_threshold": 0.5, "eq_mom_boost": False,
          "kelly_sizing": True, "strategy_momentum": False,
          "regime_asym": False, "vol_scaling": True,
          "rebal_hours": 12}),

        ("FIX-5: + regime asym",
         {"n_long": 6, "n_short": 3, "trend_cutoff": 0.8,
          "dyn_threshold": 0.5, "eq_mom_boost": False,
          "kelly_sizing": True, "strategy_momentum": False,
          "regime_asym": True, "vol_scaling": True,
          "rebal_hours": 12}),

        ("FIX-6: + eq_mom (fixed, no look-ahead)",
         {"n_long": 6, "n_short": 3, "trend_cutoff": 0.8,
          "dyn_threshold": 0.5, "eq_mom_boost": True,
          "kelly_sizing": True, "strategy_momentum": False,
          "regime_asym": True, "vol_scaling": True,
          "rebal_hours": 12}),

        ("FIX-7: + strat_mom (fixed, no look-ahead)",
         {"n_long": 6, "n_short": 3, "trend_cutoff": 0.8,
          "dyn_threshold": 0.5, "eq_mom_boost": True,
          "kelly_sizing": True, "strategy_momentum": True,
          "strat_mom_lookback": 48,
          "regime_asym": True, "vol_scaling": True,
          "rebal_hours": 12}),
    ]

    results = []
    for name, cfg in configs:
        sub = simulate_fixed(ens, regime_df, 12, cfg)
        r = eval_config_fixed(sub, name, LEVERAGE, CAPITAL)
        if r:
            show_result(r)
            results.append(r)

    # ═══════════════════════════════════════════════
    # 3. Per-Window analysis with fixed sim
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  Per-Window Sharpe (fixed simulation, fixed annualization)")
    print("═" * 70)

    # Use FIX-7 (full) config
    best_cfg = configs[-1][1]

    for w in WINDOWS:
        ens_w = ens[ens["window"] == w["name"]].copy()
        if len(ens_w) < 100:
            continue
        sub_w = simulate_fixed(ens_w, regime_df, 12, best_cfg)
        if sub_w is None or len(sub_w) < 5:
            continue
        r_w = eval_config_fixed(sub_w, f"{w['name']}: {w['test_start']}→{w['test_end']}",
                                LEVERAGE, CAPITAL)
        if r_w:
            ic_w = stats.spearmanr(ens_w["pred"], ens_w["fwd_ret"])[0]
            show_result(r_w)
            print(f"         IC = {ic_w:.4f}")

    # ═══════════════════════════════════════════════
    # 4. Permutation test with FIXED sim
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  Permutation test (fixed simulation)")
    print("═" * 70)

    sub_real = simulate_fixed(ens, regime_df, 12, best_cfg)
    r_real = eval_config_fixed(sub_real, "REAL", LEVERAGE, CAPITAL)
    real_sh = r_real["sharpe"]

    N_PERM = 100
    perm_sharpes = []
    for i in range(N_PERM):
        ens_perm = ens.copy()
        ens_perm["pred"] = ens_perm.groupby("timestamp")["pred"].transform(
            lambda x: x.sample(frac=1.0, random_state=i).values
        )
        sub_perm = simulate_fixed(ens_perm, regime_df, 12, best_cfg)
        r_perm = eval_config_fixed(sub_perm, f"perm_{i}", LEVERAGE, CAPITAL)
        if r_perm:
            perm_sharpes.append(r_perm["sharpe"])
        if (i + 1) % 25 == 0:
            print(f"   ... {i+1}/{N_PERM}")

    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= real_sh).mean()
    z_score = (real_sh - perm_sharpes.mean()) / (perm_sharpes.std() + 1e-10)

    print(f"\n   Real (fixed) Sharpe:    {real_sh:.2f}")
    print(f"   Permuted mean Sharpe:   {perm_sharpes.mean():.2f} ± {perm_sharpes.std():.2f}")
    print(f"   Signal contribution:    {real_sh - perm_sharpes.mean():.2f}")
    print(f"   p-value: {p_value:.4f}, z-score: {z_score:.2f}")

    if p_value < 0.01:
        print(f"   ✅ Signal is real (p={p_value:.4f})")
    else:
        print(f"   ⚠️  Signal marginal or weak (p={p_value:.2f})")

    # ═══════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  R16B — SUMMARY")
    print("═" * 70)

    print(f"\n  Before fixes:")
    print(f"    Sh = 4.81 (buggy annualization + look-ahead equity momentum)")

    print(f"\n  After fixes (correct annualization + no look-ahead):")
    for r in results:
        wm = f"{r['win_months']}/{r['total_months']}"
        print(f"    {r['name']:<55s} Sh={r['sharpe']:+.2f} WM={wm}")

    if results:
        best = max(results, key=lambda r: r["sharpe"])
        print(f"\n  🎯 TRUE Sharpe of best config: {best['sharpe']:.2f}")
        print(f"     (was reported as 4.81 → corrected to {best['sharpe']:.2f})")

    elapsed = time.time() - t0
    print(f"\n  ⏱  {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
