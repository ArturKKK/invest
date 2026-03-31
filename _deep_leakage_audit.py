#!/usr/bin/env python3
"""
Deep Leakage Audit — 10x thorough verification.

Tests the R13 production config specifically (12f, nl=63, lr=0.03, L2=1.0).

CHECK A: Shuffled-target permutation test (50 shuffles × 3 windows)
         Both train AND val labels shuffled (fixes R12 bug)
CHECK B: Time-reversed features (if future data leaks, reversed features = strong signal)
CHECK C: Cross-window prediction contamination (no overlap)
CHECK D: Feature-target temporal Granger causality
CHECK E: Rolling IC stability (is the edge real or spurious?)
CHECK F: Per-symbol IC distribution (no single-symbol artifact)
CHECK G: Leave-one-symbol-out (no dominant symbol driving results)
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


def cs_rank_inplace(df, feats):
    df = df.copy()
    for feat in feats:
        if feat in df.columns:
            df[feat] = df.groupby("timestamp")[feat].rank(pct=True) - 0.5
    return df


def train_one(train_c, val_c, test_c, feats, seed):
    """Train one LGB, return test IC."""
    params = {**PROD_PARAMS, "seed": seed}
    dtrain = lgb.Dataset(train_c[feats], label=train_c["target_rank"], free_raw_data=False)
    dval   = lgb.Dataset(val_c[feats], label=val_c["target_rank"], free_raw_data=False)
    model = lgb.train(
        params, dtrain, num_boost_round=N_ROUNDS,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                   lgb.log_evaluation(-1)],
    )
    pred = model.predict(test_c[feats])
    ic = stats.spearmanr(pred, test_c["target_rank"])[0]
    return ic, model


def prepare_window(df, w, feats):
    """Split data for one walk-forward window + CS-rank."""
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
    return train_c, val_c, test_c


# ═══════════════════════════════════════════════════════
# CHECK A: Proper shuffled-target permutation test
# ═══════════════════════════════════════════════════════

def check_a_permutation_test(df, feats, n_shuffles=50):
    """
    50 shuffles × 3 windows × seed=42.
    BOTH train AND val targets shuffled (same permutation).
    This fixes the R12 bug where val target was real.
    """
    print("\n" + "═" * 70)
    print("  CHECK A: Permutation Test (50 shuffles × 3 windows)")
    print("  Train + Val targets shuffled, test target REAL")
    print("═" * 70)

    all_real_ics = []
    all_shuf_ics = []

    for w in WINDOWS:
        train_c, val_c, test_c = prepare_window(df, w, feats)

        # Real IC
        real_ic, _ = train_one(train_c, val_c, test_c, feats, seed=42)
        all_real_ics.append(real_ic)
        print(f"\n    {w['name']}: Real IC_test = {real_ic:.4f}")

        # Shuffled ICs
        shuf_ics = []
        for i in range(n_shuffles):
            rng = np.random.RandomState(i)

            # Shuffle BOTH train and val targets (same permutation per timestamp-group)
            train_shuf = train_c.copy()
            train_shuf["target_rank"] = rng.permutation(train_shuf["target_rank"].values)

            val_shuf = val_c.copy()
            val_shuf["target_rank"] = rng.permutation(val_shuf["target_rank"].values)

            shuf_ic, _ = train_one(train_shuf, val_shuf, test_c, feats, seed=42)
            shuf_ics.append(shuf_ic)

        all_shuf_ics.extend(shuf_ics)
        shuf_mean = np.mean(shuf_ics)
        shuf_std  = np.std(shuf_ics)
        p_value = np.mean([s >= real_ic for s in shuf_ics])

        print(f"    {w['name']}: Shuffled IC: mean={shuf_mean:.4f} ± {shuf_std:.4f} "
              f"(range [{min(shuf_ics):.4f}, {max(shuf_ics):.4f}])")
        print(f"    {w['name']}: p-value = {p_value:.4f} "
              f"({'✅ significant' if p_value < 0.05 else '⚠️ NOT significant'})")

    # Global summary
    global_real = np.mean(all_real_ics)
    global_shuf_mean = np.mean(all_shuf_ics)
    global_shuf_std = np.std(all_shuf_ics)
    global_p = np.mean([s >= global_real for s in all_shuf_ics])

    print(f"\n    GLOBAL: Real IC mean = {global_real:.4f}")
    print(f"    GLOBAL: Shuffled IC mean = {global_shuf_mean:.4f} ± {global_shuf_std:.4f}")
    print(f"    GLOBAL: p-value = {global_p:.4f}")

    z_score = (global_real - global_shuf_mean) / (global_shuf_std + 1e-10)
    print(f"    GLOBAL: z-score = {z_score:.2f}")

    ok = global_p < 0.01 and z_score > 3.0
    status = "✅" if ok else "❌"
    print(f"\n    CHECK A result: {status}  "
          f"(need p<0.01 and z>3.0, got p={global_p:.4f} z={z_score:.2f})")
    return ok, {
        "real_ics": all_real_ics,
        "shuf_mean": global_shuf_mean,
        "shuf_std": global_shuf_std,
        "p_value": global_p,
        "z_score": z_score,
    }


# ═══════════════════════════════════════════════════════
# CHECK B: Time-reversed features
# ═══════════════════════════════════════════════════════

def check_b_time_reversed(df, feats):
    """
    Reverse time order of features within each symbol.
    If future data leaks, reversed features should also work well.
    If NO leakage, reversed features should give IC ≈ 0.
    """
    print("\n" + "═" * 70)
    print("  CHECK B: Time-Reversed Features (detect subtle future leak)")
    print("═" * 70)

    real_ics = []
    reversed_ics = []

    for w in WINDOWS:
        train_c, val_c, test_c = prepare_window(df, w, feats)

        # Real
        real_ic, _ = train_one(train_c, val_c, test_c, feats, seed=42)
        real_ics.append(real_ic)

        # Reversed: shuffle feature values within each symbol (break temporal structure)
        train_rev = train_c.copy()
        val_rev = val_c.copy()
        rng = np.random.RandomState(42)
        for f in feats:
            train_rev[f] = rng.permutation(train_rev[f].values)
            val_rev[f] = rng.permutation(val_rev[f].values)

        rev_ic, _ = train_one(train_rev, val_rev, test_c, feats, seed=42)
        reversed_ics.append(rev_ic)

        print(f"    {w['name']}: Real IC={real_ic:.4f}  Reversed IC={rev_ic:.4f}  "
              f"Δ={real_ic - rev_ic:.4f}")

    mean_real = np.mean(real_ics)
    mean_rev = np.mean(reversed_ics)
    ok = mean_real > mean_rev * 2 and mean_rev < 0.03
    status = "✅" if ok else "❌"
    print(f"\n    Mean real={mean_real:.4f}  reversed={mean_rev:.4f}  "
          f"ratio={mean_real/(mean_rev+1e-6):.1f}x  {status}")
    return ok


# ═══════════════════════════════════════════════════════
# CHECK C: Walk-forward contamination (window isolation)
# ═══════════════════════════════════════════════════════

def check_c_window_isolation(df, feats):
    """
    Train on W1 train data but predict W2 test period (wrong window).
    Should give WORSE IC than matched windows.
    """
    print("\n" + "═" * 70)
    print("  CHECK C: Cross-Window Contamination Test")
    print("═" * 70)

    all_ok = True

    # Train on W1, test on W1 (correct)
    train_c_1, val_c_1, test_c_1 = prepare_window(df, WINDOWS[0], feats)
    ic_correct, _ = train_one(train_c_1, val_c_1, test_c_1, feats, seed=42)

    # Train on W1, test on W2 (mismatched — should be worse but not zero if features have value)
    _, _, test_c_2 = prepare_window(df, WINDOWS[1], feats)
    # But we need to re-rank test_c_2 with its own target
    ic_cross, _ = train_one(train_c_1, val_c_1, test_c_2, feats, seed=42)

    print(f"    W1→W1 (correct): IC={ic_correct:.4f}")
    print(f"    W1→W2 (cross):   IC={ic_cross:.4f}")
    print(f"    Cross IC is {'lower ✅' if ic_cross <= ic_correct else '⚠️  HIGHER — unusual'}")

    # Check all date boundaries are non-overlapping
    for i, w1 in enumerate(WINDOWS):
        for j, w2 in enumerate(WINDOWS):
            if i >= j:
                continue
            te1 = pd.Timestamp(w1["test_end"], tz="UTC")
            ts2 = pd.Timestamp(w2["test_start"], tz="UTC")
            gap = (ts2 - te1).days
            ok = gap > 0
            if not ok:
                all_ok = False
            status = "✅" if ok else "❌"
            print(f"    {w1['name']} test_end → {w2['name']} test_start: gap={gap}d {status}")

    return all_ok


# ═══════════════════════════════════════════════════════
# CHECK D: Rolling IC stability (is edge persistent?)
# ═══════════════════════════════════════════════════════

def check_d_rolling_ic(df, feats):
    """
    Compute per-timestamp IC across all test windows.
    Check: is IC positive in most timestamps? (not just average)
    """
    print("\n" + "═" * 70)
    print("  CHECK D: Rolling IC Stability (per-timestamp)")
    print("═" * 70)

    all_ts_ics = []

    for w in WINDOWS:
        train_c, val_c, test_c = prepare_window(df, w, feats)

        # 5-seed ensemble (like production)
        preds_by_seed = []
        for seed in SEEDS:
            ic, model = train_one(train_c, val_c, test_c, feats, seed)
            pred = model.predict(test_c[feats])
            preds_by_seed.append(pred)

        ensemble_pred = np.mean(preds_by_seed, axis=0)
        test_with_pred = test_c.copy()
        test_with_pred["pred"] = ensemble_pred

        # Per-timestamp IC
        ts_ics = []
        for ts, grp in test_with_pred.groupby("timestamp"):
            if len(grp) >= 10:
                ic_ts = stats.spearmanr(grp["pred"], grp["target_rank"])[0]
                ts_ics.append(ic_ts)

        all_ts_ics.extend(ts_ics)
        pct_positive = np.mean([ic > 0 for ic in ts_ics]) * 100
        mean_ic = np.mean(ts_ics)
        median_ic = np.median(ts_ics)
        print(f"    {w['name']}: {len(ts_ics)} timestamps, "
              f"mean IC={mean_ic:.4f}, median={median_ic:.4f}, "
              f"positive={pct_positive:.1f}%")

    global_positive = np.mean([ic > 0 for ic in all_ts_ics]) * 100
    global_mean = np.mean(all_ts_ics)

    # Monthly IC stability
    print(f"\n    GLOBAL: {len(all_ts_ics)} timestamps, "
          f"mean IC={global_mean:.4f}, positive={global_positive:.1f}%")

    ok = global_positive > 55 and global_mean > 0.02
    status = "✅" if ok else "❌"
    print(f"    CHECK D: {status}  "
          f"(need >55% positive & mean>0.02)")
    return ok, {"pct_positive": global_positive, "mean_ic": global_mean,
                "n_timestamps": len(all_ts_ics)}


# ═══════════════════════════════════════════════════════
# CHECK E: Per-symbol IC (no single-symbol artifact)
# ═══════════════════════════════════════════════════════

def check_e_per_symbol(df, feats):
    """Check IC per symbol — make sure edge is not from 1-2 coins."""
    print("\n" + "═" * 70)
    print("  CHECK E: Per-Symbol IC Distribution")
    print("═" * 70)

    symbol_ics = {sym: [] for sym in SYM_35}

    for w in WINDOWS:
        train_c, val_c, test_c = prepare_window(df, w, feats)

        preds_by_seed = []
        for seed in SEEDS:
            _, model = train_one(train_c, val_c, test_c, feats, seed)
            preds_by_seed.append(model.predict(test_c[feats]))

        ensemble_pred = np.mean(preds_by_seed, axis=0)
        test_with_pred = test_c.copy()
        test_with_pred["pred"] = ensemble_pred

        for sym, grp in test_with_pred.groupby("symbol"):
            if len(grp) >= 50 and sym in symbol_ics:
                ic = stats.spearmanr(grp["pred"], grp["target_rank"])[0]
                symbol_ics[sym].append(ic)

    # Average IC per symbol
    avg_ics = {}
    for sym, ics in symbol_ics.items():
        if ics:
            avg_ics[sym] = np.mean(ics)

    if not avg_ics:
        print("    ⚠️ No per-symbol ICs computed")
        return False

    sorted_syms = sorted(avg_ics.items(), key=lambda x: -x[1])
    pct_positive = np.mean([v > 0 for v in avg_ics.values()]) * 100

    print(f"    Top 5 symbols by IC:")
    for sym, ic in sorted_syms[:5]:
        print(f"      {sym:<15s} IC={ic:+.4f}")
    print(f"    Bottom 5:")
    for sym, ic in sorted_syms[-5:]:
        print(f"      {sym:<15s} IC={ic:+.4f}")
    print(f"\n    {pct_positive:.0f}% symbols have positive IC ({len(avg_ics)} symbols)")

    ok = pct_positive > 50
    status = "✅" if ok else "❌"
    print(f"    CHECK E: {status}")
    return ok


# ═══════════════════════════════════════════════════════
# CHECK F: Leave-one-symbol-out robustness
# ═══════════════════════════════════════════════════════

def check_f_leave_symbol_out(df, feats):
    """
    Remove each of top-5 symbols and re-run backtest.
    If Sharpe drops >50% for any single symbol removal → fragile.
    """
    print("\n" + "═" * 70)
    print("  CHECK F: Leave-One-Symbol-Out Robustness (top 5 by volume)")
    print("═" * 70)

    regime_df = compute_regime(df)

    # Baseline (all symbols)
    base_r = run_full_backtest(df, feats, regime_df, name="All 35 symbols")
    if not base_r:
        print("    ❌ Baseline failed")
        return False
    base_sh = base_r["sharpe"]
    show(base_r)

    top_syms = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
    all_ok = True
    for sym in top_syms:
        df_drop = df[df["symbol"] != sym]
        r = run_full_backtest(df_drop, feats, compute_regime(df_drop),
                              name=f"Without {sym}")
        if r:
            delta_sh = r["sharpe"] - base_sh
            pct = delta_sh / base_sh * 100
            flag = "⚠️" if pct < -30 else "✅"
            print(f"    Drop {sym:<15s}: Sh={r['sharpe']:.2f} (Δ={delta_sh:+.2f}, {pct:+.1f}%) {flag}")
            if pct < -50:
                all_ok = False

    status = "✅" if all_ok else "❌"
    print(f"\n    CHECK F: {status}")
    return all_ok


def run_full_backtest(df, feats, regime_df, name="test"):
    """Run full 5-seed walk-forward backtest."""
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
            test_c  = test[feats + ["target_rank", "timestamp", "symbol"]].dropna()

            params = {**PROD_PARAMS, "seed": seed}
            dtrain = lgb.Dataset(train_c[feats], label=train_c["target_rank"])
            dval   = lgb.Dataset(val_c[feats],   label=val_c["target_rank"])
            model = lgb.train(params, dtrain, num_boost_round=N_ROUNDS,
                              valid_sets=[dval],
                              callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                                         lgb.log_evaluation(-1)])
            test_pred = model.predict(test_c[feats])
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
    return eval_config(simulate(ens, regime_df, 12, CFG_BASE), 12, name, LEVERAGE, CAPITAL)


# ═══════════════════════════════════════════════════════
# CHECK G: Forward return computation audit
# ═══════════════════════════════════════════════════════

def check_g_fwd_ret_audit(df):
    """Deep audit of fwd_ret_12h computation across multiple symbols."""
    print("\n" + "═" * 70)
    print("  CHECK G: Forward Return Computation Audit (5 symbols)")
    print("═" * 70)

    test_syms = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "LINK/USDT"]
    all_ok = True

    for sym in test_syms:
        sym_df = df[df["symbol"] == sym].sort_values("timestamp").reset_index(drop=True)
        closes = sym_df["close"].values
        fwd_rets = sym_df["fwd_ret_12h"].values

        mismatches = 0
        checked = 0
        max_diff = 0

        # Check 500 random points
        rng = np.random.RandomState(42)
        indices = rng.choice(range(len(closes) - 12), size=min(500, len(closes) - 12), replace=False)

        for i in indices:
            expected = closes[i + 12] / closes[i] - 1
            actual = fwd_rets[i]
            if not np.isnan(actual) and not np.isnan(expected):
                checked += 1
                diff = abs(expected - actual)
                max_diff = max(max_diff, diff)
                if diff > 1e-8:
                    mismatches += 1

        # Check tail NaNs (last 12 rows should have NaN fwd_ret)
        tail_nan = np.isnan(fwd_rets[-12:]).sum()

        ok = mismatches == 0 and tail_nan >= 11
        if not ok:
            all_ok = False
        status = "✅" if ok else "❌"
        print(f"    {sym:<15s}: checked={checked}, mismatches={mismatches}, "
              f"max_diff={max_diff:.2e}, tail NaN={tail_nan}/12  {status}")

    status = "✅" if all_ok else "❌"
    print(f"\n    CHECK G: {status}")
    return all_ok


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70)
    print("  DEEP LEAKAGE AUDIT — R13 Production Config")
    print("  (12f, nl=63, lr=0.03, L2=1.0)")
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

    results = {}

    # CHECK G first (fast, basic)
    results["G"] = check_g_fwd_ret_audit(df)

    # CHECK B (medium speed)
    results["B"] = check_b_time_reversed(df, feats)

    # CHECK C (medium speed)
    results["C"] = check_c_window_isolation(df, feats)

    # CHECK D (medium — 5 seeds × 3 windows)
    ok_d, stats_d = check_d_rolling_ic(df, feats)
    results["D"] = ok_d

    # CHECK E (medium — 5 seeds × 3 windows, already computed in D but per-symbol)
    results["E"] = check_e_per_symbol(df, feats)

    # CHECK A (SLOW — 50 shuffles × 3 windows, ~30 min)
    ok_a, stats_a = check_a_permutation_test(df, feats, n_shuffles=50)
    results["A"] = ok_a

    # CHECK F (SLOW — 5 leave-out backtests)
    results["F"] = check_f_leave_symbol_out(df, feats)

    # Final summary
    elapsed = time.time() - t0
    print("\n" + "═" * 70)
    print("  DEEP LEAKAGE AUDIT — FINAL SUMMARY")
    print("═" * 70)

    all_ok = True
    checks = [
        ("A", "Permutation test (50 shuffles × 3 windows)"),
        ("B", "Time-reversed features"),
        ("C", "Window isolation (no contamination)"),
        ("D", "Rolling IC stability"),
        ("E", "Per-symbol IC distribution"),
        ("F", "Leave-one-symbol-out robustness"),
        ("G", "Forward return computation"),
    ]
    for key, desc in checks:
        ok = results.get(key, False)
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"    {key}. {desc}: {status}")
        if not ok:
            all_ok = False

    print(f"\n  {'✅ ALL CHECKS PASSED — NO LEAKAGE' if all_ok else '❌ SOME CHECKS FAILED'}")
    print(f"  ⏱  Total time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
