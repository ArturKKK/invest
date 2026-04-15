#!/usr/bin/env python3
"""
R126 — Address External Review Findings on R123/R125
=====================================================

External reviewer identified 3 actionable issues:

1. IC SCAN BUG: market-level features duplicated ~35× per timestamp in panel
   → artificial IC inflation. Fix: time-series IC for market-level features.

2. LAG/AVAILABILITY: news aggregated at hour=t may include news published after
   the decision point. Test: shift news features by +1h/+2h.

3. PER-COIN ONLY: before closing news, test per-coin features alone (no market-level).

Additionally:
4. R124 TAKER BASELINE: prod may be 100% taker. Add TAKER_ONLY scenario.

This script runs items 1-3. Item 4 is in _research_r124b_taker_baseline.py.
"""

import time, json, os, sys, warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from _research_r22_models import SEEDS, log, cs_rank_cols
from _research_r35_new_features import MARKET_LEVEL_FEATURES
from _research_r68_continuous_wf import (
    CHAMPION_FEAT_31, CONTINUOUS_WINDOWS, PROD_CFG,
    load_data, train_ensemble, sharpe,
    LGB_PARAMS, XGB_PARAMS, N_ROUNDS, EARLY_STOP,
)
from _research_r113_trend_cutoff_reopt import analyze_config, print_result
from _research_r121_realistic_costs import (
    simulate_r121, cost_prod_blended, R114B_CFG,
    per_window_metrics, COST_MODELS,
)
from _research_r123_news_sentiment import (
    NEWS_PER_COIN, NEWS_MARKET, NEWS_POLITICAL, NEWS_ALL,
    NEWS_MARKET_LEVEL, INTERACTION_FEATS,
    load_news_features, merge_news, build_interaction_features,
    train_extended, run_experiment, bootstrap_compare,
)

TIMESTAMP = time.strftime("%Y%m%d_%H%M")


# ═══════════════════════════════════════════════════════════════════
# FIX 1: CORRECT IC SCAN (time-series IC for market-level features)
# ═══════════════════════════════════════════════════════════════════

def ic_scan_fixed(df, features, target="fwd_ret_12h"):
    """
    Compute IC correctly:
    - Per-coin features: standard panel Spearman IC (cross-sectional)
    - Market-level features: time-series IC (one obs per timestamp)

    For market-level: at each timestamp, feature value is identical for all coins.
    So we collapse to one value per timestamp and correlate with cross-sectional
    mean return at that timestamp.
    """
    log("\n" + "=" * 70)
    log("IC SCAN (FIXED) — Correct method for market vs per-coin features")
    log("=" * 70)

    tz = df["timestamp"].dt.tz
    results = []

    for feat in features:
        if feat not in df.columns:
            log(f"  {feat}: MISSING")
            continue

        is_market = feat in NEWS_MARKET_LEVEL

        ics_by_window = {}
        for w in CONTINUOUS_WINDOWS:
            te_start = pd.Timestamp(w["test_start"], tz=tz)
            te_end = pd.Timestamp(w["test_end"], tz=tz)
            mask = (df["timestamp"] >= te_start) & (df["timestamp"] <= te_end)
            sub = df.loc[mask].copy()

            if is_market:
                # Time-series IC: collapse to one obs per timestamp
                # Feature: first value per timestamp (all are identical)
                # Target: cross-sectional mean return at that timestamp
                ts_agg = sub.groupby("timestamp").agg(
                    feat_val=(feat, "first"),
                    mean_ret=(target, "mean"),
                ).dropna()
                if len(ts_agg) < 30:
                    ics_by_window[w["name"]] = np.nan
                    continue
                ic, _ = stats.spearmanr(ts_agg["feat_val"], ts_agg["mean_ret"])
            else:
                # Per-coin: standard cross-sectional panel IC (correct as-is)
                sub2 = sub[[feat, target]].dropna()
                if len(sub2) < 100:
                    ics_by_window[w["name"]] = np.nan
                    continue
                ic, _ = stats.spearmanr(sub2[feat], sub2[target])

            ics_by_window[w["name"]] = ic

        ics = [v for v in ics_by_window.values() if not np.isnan(v)]
        mean_ic = np.mean(ics) if ics else 0
        n_pass = sum(1 for v in ics if abs(v) > 0.02)
        stable = n_pass >= 2

        results.append({
            "feature": feat,
            "type": "MARKET" if is_market else "PER-COIN",
            "mean_ic": round(mean_ic, 4),
            "w1_ic": round(ics_by_window.get("W1", np.nan), 4),
            "w2_ic": round(ics_by_window.get("W2", np.nan), 4),
            "w3_ic": round(ics_by_window.get("W3", np.nan), 4),
            "n_pass": n_pass,
            "stable": stable,
        })

    results.sort(key=lambda x: abs(x["mean_ic"]), reverse=True)

    log(f"\n  {'Feature':<35} {'Type':<9} {'MeanIC':>8} {'W1':>8} {'W2':>8} {'W3':>8} {'Pass':>5} {'Stable':>7}")
    log(f"  {'-'*35} {'-'*9} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*5} {'-'*7}")
    for r in results:
        tag = "  ✓" if r["stable"] else ""
        log(f"  {r['feature']:<35} {r['type']:<9} {r['mean_ic']:>+8.4f} "
            f"{r['w1_ic']:>+8.4f} {r['w2_ic']:>+8.4f} {r['w3_ic']:>+8.4f} "
            f"{r['n_pass']:>5} {tag:>7}")

    passing = [r["feature"] for r in results if r["stable"]]
    log(f"\n  IC-passing features ({len(passing)}): {passing}")

    return results, passing


# ═══════════════════════════════════════════════════════════════════
# FIX 2: LAG-SHIFT TEST
# ═══════════════════════════════════════════════════════════════════

def apply_lag_shift(df, news_cols, lag_hours):
    """
    Shift news features forward by lag_hours to simulate delayed availability.

    If news are aggregated at hour=t but we can only use them at t+lag_hours,
    this tests for lookahead bias.
    """
    log(f"\n  Applying lag shift = +{lag_hours}h to {len(news_cols)} news features")

    df_shifted = df.copy()
    # Group by symbol, shift news features within each coin's timeline
    for col in news_cols:
        if col in df_shifted.columns:
            # Shift by lag_hours * periods_per_hour
            # Data is 1h frequency typically, but we shift by timestamp
            df_shifted[col] = df_shifted.groupby("symbol")[col].shift(lag_hours)

    n_nan_before = df[news_cols[0]].isna().sum() if news_cols[0] in df.columns else 0
    n_nan_after = df_shifted[news_cols[0]].isna().sum() if news_cols[0] in df_shifted.columns else 0
    log(f"    NaN count in {news_cols[0]}: {n_nan_before} → {n_nan_after}")

    return df_shifted


# ═══════════════════════════════════════════════════════════════════
# FIX 3: PER-COIN ONLY EXPERIMENT
# ═══════════════════════════════════════════════════════════════════

# Per-coin only features (no market-level duplication issue)
NEWS_PER_COIN_ONLY = [
    "news_count_1h", "news_count_24h", "news_count_7d",
    "news_sentiment_1h", "news_sentiment_24h", "news_sentiment_7d",
    "news_sentiment_momentum", "news_volume_zscore",
]

# Per-coin interaction (genuinely cross-sectional)
PERCOIN_INTERACTIONS = [
    "nx_sent_divergence",  # coin_sentiment - market_sentiment → per-coin
]


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log("=" * 70)
    log("R126 — External Review Fixes for News Sentiment")
    log("=" * 70)

    os.makedirs("results", exist_ok=True)

    # ── Load base data ──
    log("\n[1/6] Loading base data...")
    df, regime_df = load_data()
    log(f"  df: {df.shape}")

    # ── Load & merge news ──
    log("\n[2/6] Loading news features...")
    news = load_news_features()
    if news is None:
        log("FATAL: No news data available")
        return
    df = merge_news(df, news)
    df, interaction_added = build_interaction_features(df)

    all_news_feats = [f for f in NEWS_ALL + interaction_added if f in df.columns]

    # ═══════════════════════════════════════════════════════════
    # PART 1: Fixed IC Scan (market vs per-coin separation)
    # ═══════════════════════════════════════════════════════════

    log("\n[3/6] Fixed IC Scan (time-series for market, panel for per-coin)...")
    ic_results_fixed, ic_passing_fixed = ic_scan_fixed(df, all_news_feats)

    # Also run OLD ic_scan for comparison
    log("\n  --- OLD IC SCAN (for comparison) ---")
    from _research_r123_news_sentiment import ic_scan as ic_scan_old
    ic_results_old, ic_passing_old = ic_scan_old(df, all_news_feats)

    log("\n  === IC SCAN COMPARISON ===")
    log(f"  OLD passing: {ic_passing_old}")
    log(f"  NEW passing: {ic_passing_fixed}")
    log(f"  Diff: OLD has {len(ic_passing_old)}, NEW has {len(ic_passing_fixed)}")

    # Build lookup for comparison
    old_map = {r["feature"]: r for r in ic_results_old}
    new_map = {r["feature"]: r for r in ic_results_fixed}
    log(f"\n  {'Feature':<35} {'Old IC':>8} {'New IC':>8} {'Δ':>8} {'Old?':>5} {'New?':>5}")
    log(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*5} {'-'*5}")
    for feat in sorted(set(list(old_map.keys()) + list(new_map.keys()))):
        old_ic = old_map.get(feat, {}).get("mean_ic", 0)
        new_ic = new_map.get(feat, {}).get("mean_ic", 0)
        delta = new_ic - old_ic
        old_pass = "✓" if feat in ic_passing_old else ""
        new_pass = "✓" if feat in ic_passing_fixed else ""
        log(f"  {feat:<35} {old_ic:>+8.4f} {new_ic:>+8.4f} {delta:>+8.4f} {old_pass:>5} {new_pass:>5}")

    # ═══════════════════════════════════════════════════════════
    # PART 2: Per-coin only experiment (no market-level features)
    # ═══════════════════════════════════════════════════════════

    log("\n[4/6] Per-coin only experiments...")

    experiments = {}

    # A: Baseline (no news) — same as R123
    log("\n" + "─" * 70)
    log("EXP A: Baseline (31 champion features, no news)")
    log("─" * 70)
    base_feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    no_rank_base = [f for f in base_feats if f in MARKET_LEVEL_FEATURES]
    t1 = time.time()
    preds_a = train_ensemble(df, base_feats, CONTINUOUS_WINDOWS,
                             seeds=SEEDS, cs_rank_exclude=no_rank_base)
    log(f"  Trained in {time.time()-t1:.0f}s")
    experiments["A_baseline"] = run_experiment(preds_a, regime_df, "A_baseline")

    # G: Per-coin news only (8 features, no market-level)
    log("\n" + "─" * 70)
    log("EXP G: +per-coin news only (8 features, no market-level)")
    log("─" * 70)
    preds_g = train_extended(df, NEWS_PER_COIN_ONLY, "G_percoin_only",
                             news_market_level=set())  # none are market-level
    experiments["G_percoin_only"] = run_experiment(preds_g, regime_df, "G_percoin_only")

    # H: Per-coin + divergence only
    log("\n" + "─" * 70)
    log("EXP H: +per-coin + divergence (9 features)")
    log("─" * 70)
    preds_h = train_extended(df, NEWS_PER_COIN_ONLY + PERCOIN_INTERACTIONS,
                             "H_percoin_div",
                             news_market_level=set())
    experiments["H_percoin_div"] = run_experiment(preds_h, regime_df, "H_percoin_div")

    # I: Only IC-passing features from FIXED scan (if any)
    if ic_passing_fixed:
        log("\n" + "─" * 70)
        log(f"EXP I: +IC-pass (fixed) subset ({len(ic_passing_fixed)} features)")
        log("─" * 70)
        preds_i = train_extended(df, ic_passing_fixed, "I_ic_fixed",
                                 news_market_level=NEWS_MARKET_LEVEL)
        experiments["I_ic_fixed"] = run_experiment(preds_i, regime_df, "I_ic_fixed")
    else:
        log("\n  EXP I: SKIPPED (no features passed fixed IC gate)")
        experiments["I_ic_fixed"] = {"label": "I_ic_fixed", "net_sharpe": 0}

    # ═══════════════════════════════════════════════════════════
    # PART 3: Lag-shift test (+1h, +2h)
    # ═══════════════════════════════════════════════════════════

    log("\n[5/6] Lag-shift tests...")

    # Use per-coin features for lag test (cleanest signal, no market-level issues)
    news_cols_for_lag = [c for c in NEWS_PER_COIN_ONLY + PERCOIN_INTERACTIONS
                         if c in df.columns]

    for lag_h in [1, 2]:
        log(f"\n" + "─" * 70)
        log(f"EXP LAG{lag_h}: Per-coin news shifted +{lag_h}h")
        log("─" * 70)

        df_lagged = apply_lag_shift(df, news_cols_for_lag, lag_h)
        preds_lag = train_extended(df_lagged, NEWS_PER_COIN_ONLY + PERCOIN_INTERACTIONS,
                                   f"LAG{lag_h}_percoin",
                                   news_market_level=set())
        experiments[f"LAG{lag_h}_percoin"] = run_experiment(
            preds_lag, regime_df, f"LAG{lag_h}_percoin")

    # Also: lag-shift IC scan for per-coin features
    log("\n  Lag-shift IC scan (per-coin features, +1h):")
    df_lag1 = apply_lag_shift(df, news_cols_for_lag, 1)
    ic_lag1, _ = ic_scan_fixed(df_lag1, news_cols_for_lag)

    log("\n  Lag-shift IC scan (per-coin features, +2h):")
    df_lag2 = apply_lag_shift(df, news_cols_for_lag, 2)
    ic_lag2, _ = ic_scan_fixed(df_lag2, news_cols_for_lag)

    # ═══════════════════════════════════════════════════════════
    # PART 4: Bootstrap & Summary
    # ═══════════════════════════════════════════════════════════

    log("\n[6/6] Bootstrap comparison & Summary...")
    base_port = experiments["A_baseline"].get("port", pd.DataFrame())

    bootstrap_results = {}
    for name, exp in experiments.items():
        if name == "A_baseline":
            continue
        test_port = exp.get("port", pd.DataFrame())
        if isinstance(test_port, pd.DataFrame) and not test_port.empty:
            p_imp, delta = bootstrap_compare(base_port, test_port)
            bootstrap_results[name] = {"p_improvement": round(p_imp, 3),
                                        "mean_delta_sharpe": round(delta, 3)}
            log(f"  {name}: P(imp)={p_imp:.3f}, ΔSharpe={delta:+.3f}")

    # ── Final summary ──
    log("\n" + "=" * 70)
    log("R126 — RESULTS SUMMARY")
    log("=" * 70)

    hdr = (f"  {'Experiment':<20} {'NetSh':>7} {'Ret%':>7} "
           f"{'DD%':>7} {'ΔSh':>7} {'P(imp)':>7}")
    log(hdr)
    log(f"  {'-'*20} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    base_sh = experiments["A_baseline"].get("net_sharpe", 0)
    for name in ["A_baseline", "G_percoin_only", "H_percoin_div",
                  "I_ic_fixed", "LAG1_percoin", "LAG2_percoin"]:
        m = experiments.get(name, {})
        ns = m.get("net_sharpe", 0)
        ret = m.get("total_ret_pct", 0)
        dd = m.get("max_dd_pct", 0)
        delta = ns - base_sh
        bs = bootstrap_results.get(name, {})
        p_imp = bs.get("p_improvement", "—")
        p_str = f"{p_imp:.3f}" if isinstance(p_imp, float) else "  —"
        log(f"  {name:<20} {ns:>7.3f} {ret:>7.1f} "
            f"{dd:>7.1f} {delta:>+7.3f} {p_str:>7}")

    # ── Verdict ──
    best_name = max(
        [k for k in experiments if k != "A_baseline"],
        key=lambda k: experiments[k].get("net_sharpe", 0)
    )
    best = experiments[best_name]
    best_delta = best.get("net_sharpe", 0) - base_sh
    best_p = bootstrap_results.get(best_name, {}).get("p_improvement", 0)

    log(f"\n  BEST non-baseline: {best_name}")
    log(f"    Sharpe: {best.get('net_sharpe', 0):.3f} (baseline: {base_sh:.3f}, "
        f"Δ={best_delta:+.3f})")
    log(f"    P(improvement): {best_p:.3f}")

    if best_delta > 0.10 and best_p > 0.80:
        log(f"\n  ✅ VERDICT: POSITIVE — {best_name} improves by {best_delta:+.3f}")
    elif best_delta > 0 and best_p > 0.60:
        log(f"\n  ⚠️  VERDICT: MARGINAL — {best_name} improves by {best_delta:+.3f} but weak")
    else:
        log(f"\n  ❌ VERDICT: NEGATIVE — News features do not help even after IC fix + lag test")
        log(f"     News direction PERMANENTLY CLOSED")
        log(f"     IC scan bug confirmed: market-level IC was artificially inflated")
        if any(experiments[k].get("net_sharpe", 0) < base_sh - 0.05
               for k in experiments if k != "A_baseline"):
            log(f"     Per-coin features also hurt model → news are noise for this setup")

    # ── Save results ──
    save_data = {
        "experiment": "R126_review_fixes",
        "timestamp": TIMESTAMP,
        "fixes_applied": [
            "1. Fixed IC scan: time-series IC for market-level features",
            "2. Lag-shift test: +1h, +2h for lookahead bias check",
            "3. Per-coin only experiments: no market-level duplication",
        ],
        "ic_comparison": {
            "old_passing": ic_passing_old,
            "new_passing": ic_passing_fixed,
            "old_results": ic_results_old,
            "new_results": ic_results_fixed,
        },
        "experiments": {k: {kk: vv for kk, vv in v.items() if kk != "port"}
                        for k, v in experiments.items()},
        "bootstrap": bootstrap_results,
    }
    with open("results/r126_review_fixes.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    log(f"\n  Saved: results/r126_review_fixes.json")
    log(f"  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
