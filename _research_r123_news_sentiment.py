#!/usr/bin/env python3
"""
R123 — News Sentiment Feature Evaluation
==========================================

Test whether adding pre-computed news sentiment features to the 31 champion
features improves walk-forward Sharpe under realistic (S6) costs.

Existing 15 news features from data/sentiment/crypto_news.parquet:
  Per-coin (8): news_count_1h/24h/7d, news_sentiment_1h/24h/7d,
                news_sentiment_momentum, news_volume_zscore
  Market (2):   market_news_count_24h, market_news_sentiment_24h
  Political (5): political_news_count_24h, political_sentiment_24h/7d,
                 political_sentiment_shock, political_news_volume_zscore

Previous A/B test (from _ab_news_results.json):
  A (no news):           Sharpe 2.57
  B (all news):          Sharpe 2.76   (+7.4%)
  C (LGB old + CB new):  Sharpe 3.77   (market-only CatBoost won)
  D (LGB new + CB old):  Sharpe 2.12   (per-coin hurt LGB)

Approach:
  Step 1: IC scan of all 15 features against fwd_ret_12h (rank IC per window)
  Step 2: Feature addition experiments (A-F) using WF ensemble
  Step 3: Simulate with S6 prod_blended costs
  Step 4: Bootstrap comparison vs R114b baseline

Experiments:
  A: Baseline — 31 champion features (no news)
  B: 31 + market-level (2 feats)
  C: 31 + market + political (7 feats)
  D: 31 + all 15 news features
  E: 31 + IC-pass subset (features with |IC| > 0.02 in ≥2/3 windows)
  F: 31 + market + interaction features (sentiment × return, etc.)

Acceptance:
  - At least 1 experiment Sharpe > 2.831 (S6 baseline) + 0.10
  - Bootstrap P(improvement) > 0.80
  - No DD worse than -15%
"""

import time, json, os, warnings
import numpy as np, pandas as pd
import lightgbm as lgb
import xgboost as xgb
from scipy import stats
warnings.filterwarnings("ignore")

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


# ═══════════════════════════════════════════════════════════════════
# NEWS FEATURE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════

NEWS_PER_COIN = [
    "news_count_1h", "news_count_24h", "news_count_7d",
    "news_sentiment_1h", "news_sentiment_24h", "news_sentiment_7d",
    "news_sentiment_momentum", "news_volume_zscore",
]

NEWS_MARKET = [
    "market_news_count_24h", "market_news_sentiment_24h",
]

NEWS_POLITICAL = [
    "political_news_count_24h", "political_sentiment_24h",
    "political_sentiment_7d", "political_sentiment_shock",
    "political_news_volume_zscore",
]

NEWS_ALL = NEWS_PER_COIN + NEWS_MARKET + NEWS_POLITICAL

# Interaction features we'll create
INTERACTION_FEATS = [
    "nx_mkt_sent_x_ret12",         # market sentiment × 12h return (contrarian)
    "nx_mkt_sent_x_vol",           # market sentiment × realized vol
    "nx_mkt_count_zscore",         # market news count z-score (attention spike)
    "nx_pol_shock_x_vol",          # political shock × vol regime
    "nx_sent_divergence",          # coin sentiment - market sentiment
]

# Which news features are market-level (same for all coins at given timestamp)
NEWS_MARKET_LEVEL = set(NEWS_MARKET + NEWS_POLITICAL + [
    "nx_mkt_sent_x_ret12", "nx_mkt_sent_x_vol",
    "nx_mkt_count_zscore", "nx_pol_shock_x_vol",
])


# ═══════════════════════════════════════════════════════════════════
# LOAD NEWS DATA
# ═══════════════════════════════════════════════════════════════════

def load_news_features():
    """Load pre-computed news features from parquet."""
    path = os.path.join("data", "sentiment", "crypto_news.parquet")
    if not os.path.exists(path):
        log(f"  ERROR: {path} not found!")
        return None

    news = pd.read_parquet(path)
    news["timestamp"] = pd.to_datetime(news["timestamp"], utc=True)
    log(f"  News data: {len(news):,} rows, "
        f"{news['timestamp'].min().date()} → {news['timestamp'].max().date()}")
    log(f"  Columns: {sorted(news.columns.tolist())}")
    log(f"  Symbols: {news['symbol'].nunique()}")

    # Check which features are available
    avail = [c for c in NEWS_ALL if c in news.columns]
    missing = [c for c in NEWS_ALL if c not in news.columns]
    log(f"  Available: {len(avail)}/{len(NEWS_ALL)} features")
    if missing:
        log(f"  Missing: {missing}")

    return news


def merge_news(df, news):
    """Merge news features into main dataframe."""
    avail_cols = [c for c in NEWS_ALL if c in news.columns]
    merge_cols = ["timestamp", "symbol"] + avail_cols
    news_sub = news[merge_cols].drop_duplicates(["timestamp", "symbol"])

    before = len(df)
    df = df.merge(news_sub, on=["timestamp", "symbol"], how="left")
    assert len(df) == before, f"Merge changed row count: {before} → {len(df)}"

    # Log1p transform count features (heavy-tailed)
    for col in ["news_count_1h", "news_count_24h", "news_count_7d",
                 "market_news_count_24h", "political_news_count_24h"]:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    # Two-flag system for NaN handling (LGB handles NaN natively)
    coverage_col = "news_count_24h" if "news_count_24h" in df.columns else \
                   "market_news_count_24h" if "market_news_count_24h" in df.columns else None
    if coverage_col:
        has_coverage = ~df[coverage_col].isna()
        df["news_coverage_ok"] = has_coverage.astype(float)
        n_cov = has_coverage.sum()
        log(f"  News coverage: {n_cov:,}/{len(df):,} ({n_cov/len(df)*100:.1f}%)")

    return df


def build_interaction_features(df):
    """Build interaction features from news + existing features."""
    added = []

    # 1. Market sentiment × 12h return (contrarian signal)
    if "market_news_sentiment_24h" in df.columns and "ret_12h" in df.columns:
        df["nx_mkt_sent_x_ret12"] = df["market_news_sentiment_24h"] * (-df["ret_12h"])
        added.append("nx_mkt_sent_x_ret12")

    # 2. Market sentiment × realized vol
    if "market_news_sentiment_24h" in df.columns and "rvol_24h" in df.columns:
        df["nx_mkt_sent_x_vol"] = df["market_news_sentiment_24h"] * df["rvol_24h"]
        added.append("nx_mkt_sent_x_vol")

    # 3. Market news count z-score (attention spike independent of per-coin)
    if "market_news_count_24h" in df.columns:
        mkt = df.groupby("timestamp")["market_news_count_24h"].first()
        mu = mkt.rolling(720, min_periods=24).mean()
        sigma = mkt.rolling(720, min_periods=24).std().clip(lower=0.01)
        zscore = ((mkt - mu) / sigma).rename("nx_mkt_count_zscore")
        df = df.merge(zscore.reset_index(), on="timestamp", how="left",
                      suffixes=("", "_dup"))
        dup = [c for c in df.columns if c.endswith("_dup")]
        if dup:
            df.drop(columns=dup, inplace=True)
        added.append("nx_mkt_count_zscore")

    # 4. Political shock × vol regime
    if "political_sentiment_shock" in df.columns and "rvol_24h" in df.columns:
        df["nx_pol_shock_x_vol"] = df["political_sentiment_shock"] * df["rvol_24h"]
        added.append("nx_pol_shock_x_vol")

    # 5. Per-coin sentiment divergence from market
    if "news_sentiment_24h" in df.columns and "market_news_sentiment_24h" in df.columns:
        df["nx_sent_divergence"] = df["news_sentiment_24h"] - df["market_news_sentiment_24h"]
        added.append("nx_sent_divergence")

    log(f"  Interaction features: {len(added)} created = {added}")
    return df, added


# ═══════════════════════════════════════════════════════════════════
# STEP 1: IC SCAN
# ═══════════════════════════════════════════════════════════════════

def ic_scan(df, features, target="fwd_ret_12h"):
    """Compute rank IC of each feature against target, per WF window."""
    log("\n" + "=" * 70)
    log("STEP 1: IC SCAN — News Features vs fwd_ret_12h")
    log("=" * 70)

    tz = df["timestamp"].dt.tz
    results = []

    for feat in features:
        if feat not in df.columns:
            log(f"  {feat}: MISSING")
            continue

        ics_by_window = {}
        for w in CONTINUOUS_WINDOWS:
            te_start = pd.Timestamp(w["test_start"], tz=tz)
            te_end = pd.Timestamp(w["test_end"], tz=tz)
            mask = (df["timestamp"] >= te_start) & (df["timestamp"] <= te_end)
            sub = df.loc[mask, [feat, target]].dropna()
            if len(sub) < 100:
                ics_by_window[w["name"]] = np.nan
                continue
            ic, _ = stats.spearmanr(sub[feat], sub[target])
            ics_by_window[w["name"]] = ic

        ics = [v for v in ics_by_window.values() if not np.isnan(v)]
        mean_ic = np.mean(ics) if ics else 0
        n_pass = sum(1 for v in ics if abs(v) > 0.02)
        stable = n_pass >= 2  # ≥2/3 windows

        results.append({
            "feature": feat,
            "mean_ic": round(mean_ic, 4),
            "w1_ic": round(ics_by_window.get("W1", np.nan), 4),
            "w2_ic": round(ics_by_window.get("W2", np.nan), 4),
            "w3_ic": round(ics_by_window.get("W3", np.nan), 4),
            "n_pass": n_pass,
            "stable": stable,
        })

    # Sort by absolute mean IC
    results.sort(key=lambda x: abs(x["mean_ic"]), reverse=True)

    log(f"\n  {'Feature':<35} {'MeanIC':>8} {'W1':>8} {'W2':>8} {'W3':>8} {'Pass':>5} {'Stable':>7}")
    log(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*5} {'-'*7}")
    for r in results:
        tag = "  ✓" if r["stable"] else ""
        log(f"  {r['feature']:<35} {r['mean_ic']:>+8.4f} "
            f"{r['w1_ic']:>+8.4f} {r['w2_ic']:>+8.4f} {r['w3_ic']:>+8.4f} "
            f"{r['n_pass']:>5} {tag:>7}")

    passing = [r["feature"] for r in results if r["stable"]]
    log(f"\n  IC-passing features ({len(passing)}): {passing}")

    return results, passing


# ═══════════════════════════════════════════════════════════════════
# STEP 2: TRAIN ENSEMBLE WITH EXTENDED FEATURES
# ═══════════════════════════════════════════════════════════════════

def train_extended(df, extra_feats, label, news_market_level=None):
    """Train WF ensemble with champion + extra features.

    news_market_level: set of feature names that are market-level
                       (should NOT be cs-ranked)
    """
    base_feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    extra_avail = [f for f in extra_feats if f in df.columns]
    all_feats = base_feats + extra_avail

    # Market-level features should not be cross-sectionally ranked
    existing_market = set(f for f in base_feats if f in MARKET_LEVEL_FEATURES)
    extra_market = set(f for f in extra_avail if f in (news_market_level or set()))
    no_rank = list(existing_market | extra_market)

    log(f"\n  Training [{label}]: {len(base_feats)} base + {len(extra_avail)} news = {len(all_feats)} total")
    log(f"    Market-level (no rank): {len(no_rank)}")

    t1 = time.time()
    preds = train_ensemble(df, all_feats, CONTINUOUS_WINDOWS,
                           seeds=SEEDS, cs_rank_exclude=no_rank)
    log(f"    Trained in {time.time()-t1:.0f}s, {len(preds):,} predictions")
    return preds


# ═══════════════════════════════════════════════════════════════════
# STEP 3: SIMULATE & COMPARE
# ═══════════════════════════════════════════════════════════════════

def run_experiment(preds, regime_df, label):
    """Simulate with S6 costs, return metrics dict."""
    if preds is None or preds.empty:
        log(f"  {label}: NO predictions")
        return {"label": label, "net_sharpe": 0}

    cfg = dict(R114B_CFG)
    cost_fn, funding = COST_MODELS["prod_blended"]

    port = simulate_r121(preds, regime_df, 4, 2, cfg,
                         cutoff_on=0.9, cutoff_off=0.8,
                         min_risk_off_periods=2,
                         cost_fn=cost_fn,
                         funding_per_12h=funding,
                         exec_delay_penalty=0.0003)

    m = analyze_config(port, label)
    print_result(m)

    pw = per_window_metrics(port, preds)
    for w, wm in pw.items():
        log(f"    {w}: Sharpe={wm['sharpe']:.3f}  Ret={wm['ret_pct']:.1f}%")
    m["per_window"] = pw
    m["port"] = port  # Keep for bootstrap
    return m


def bootstrap_compare(port_base, port_test, n_boot=5000, seed=42):
    """Bootstrap test: P(test Sharpe > base Sharpe)."""
    if port_base.empty or port_test.empty:
        return 0.5, 0.0

    rets_base = port_base["net_ret"].values
    rets_test = port_test["net_ret"].values
    n = min(len(rets_base), len(rets_test))
    rets_base = rets_base[:n]
    rets_test = rets_test[:n]

    rng = np.random.RandomState(seed)
    diff_sharpes = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        rb = rets_base[idx]
        rt = rets_test[idx]
        sb = np.mean(rb) / (np.std(rb) + 1e-10) * np.sqrt(2 * 365)
        st = np.mean(rt) / (np.std(rt) + 1e-10) * np.sqrt(2 * 365)
        diff_sharpes.append(st - sb)

    p_improvement = np.mean(np.array(diff_sharpes) > 0)
    mean_delta = np.mean(diff_sharpes)
    return p_improvement, mean_delta


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log("=" * 70)
    log("R123 — News Sentiment Feature Evaluation")
    log("=" * 70)

    os.makedirs("results", exist_ok=True)

    # ── Load base data ──
    log("\n[1/5] Loading base data...")
    df, regime_df = load_data()

    # ── Load & merge news ──
    log("\n[2/5] Loading news features...")
    news = load_news_features()
    if news is None:
        log("FATAL: No news data available")
        return
    df = merge_news(df, news)
    df, interaction_added = build_interaction_features(df)

    # ── IC Scan ──
    log("\n[3/5] IC Scan...")
    all_news_feats = [f for f in NEWS_ALL + interaction_added
                      if f in df.columns]
    ic_results, ic_passing = ic_scan(df, all_news_feats)

    # ── Train experiments ──
    log("\n[4/5] Training experiments...")

    experiments = {}

    # A: Baseline (no news)
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

    # B: + market-level news (2 features)
    log("\n" + "─" * 70)
    log("EXP B: +market news (market_news_count_24h, market_news_sentiment_24h)")
    log("─" * 70)
    preds_b = train_extended(df, NEWS_MARKET, "B_market",
                             news_market_level=NEWS_MARKET_LEVEL)
    experiments["B_market"] = run_experiment(preds_b, regime_df, "B_market")

    # C: + market + political (7 features)
    log("\n" + "─" * 70)
    log("EXP C: +market + political (7 news features)")
    log("─" * 70)
    preds_c = train_extended(df, NEWS_MARKET + NEWS_POLITICAL, "C_mkt_pol",
                             news_market_level=NEWS_MARKET_LEVEL)
    experiments["C_mkt_pol"] = run_experiment(preds_c, regime_df, "C_mkt_pol")

    # D: + all 15 news features
    log("\n" + "─" * 70)
    log("EXP D: +all 15 news features")
    log("─" * 70)
    preds_d = train_extended(df, NEWS_ALL, "D_all_news",
                             news_market_level=NEWS_MARKET_LEVEL)
    experiments["D_all_news"] = run_experiment(preds_d, regime_df, "D_all_news")

    # E: + IC-passing subset only
    if ic_passing:
        log("\n" + "─" * 70)
        log(f"EXP E: +IC-pass subset ({len(ic_passing)} features)")
        log("─" * 70)
        preds_e = train_extended(df, ic_passing, "E_ic_pass",
                                 news_market_level=NEWS_MARKET_LEVEL)
        experiments["E_ic_pass"] = run_experiment(preds_e, regime_df, "E_ic_pass")
    else:
        log("\n  EXP E: SKIPPED (no features passed IC gate)")
        experiments["E_ic_pass"] = {"label": "E_ic_pass", "net_sharpe": 0,
                                     "total_ret_pct": 0, "max_dd_pct": 0}

    # F: + market + interactions
    log("\n" + "─" * 70)
    log(f"EXP F: +market + interactions ({len(interaction_added)} features)")
    log("─" * 70)
    preds_f = train_extended(df, NEWS_MARKET + interaction_added, "F_interact",
                             news_market_level=NEWS_MARKET_LEVEL)
    experiments["F_interact"] = run_experiment(preds_f, regime_df, "F_interact")

    # ── Bootstrap comparison ──
    log("\n[5/5] Bootstrap comparison vs baseline...")
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
            log(f"  {name}: P(improvement)={p_imp:.3f}, mean ΔSharpe={delta:+.3f}")

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("R123 — RESULTS SUMMARY")
    log("=" * 70)

    hdr = (f"  {'Experiment':<20} {'NetSh':>7} {'GrSh':>7} {'Ret%':>7} "
           f"{'DD%':>7} {'Calmar':>7} {'ΔSh':>7} {'P(imp)':>7}")
    log(hdr)
    log(f"  {'-'*20} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    base_sh = experiments["A_baseline"].get("net_sharpe", 0)
    for name, m in experiments.items():
        ns = m.get("net_sharpe", 0)
        gs = m.get("gross_sharpe", 0)
        ret = m.get("total_ret_pct", 0)
        dd = m.get("max_dd_pct", 0)
        cal = m.get("calmar", 0)
        delta = ns - base_sh
        bs = bootstrap_results.get(name, {})
        p_imp = bs.get("p_improvement", "—")
        if isinstance(p_imp, float):
            p_str = f"{p_imp:.3f}"
        else:
            p_str = "  —"
        log(f"  {name:<20} {ns:>7.3f} {gs:>7.3f} {ret:>7.1f} "
            f"{dd:>7.1f} {cal:>7.2f} {delta:>+7.3f} {p_str:>7}")

    # ── Best experiment ──
    best_name = max(experiments, key=lambda k: experiments[k].get("net_sharpe", 0))
    best = experiments[best_name]
    best_p = bootstrap_results.get(best_name, {}).get("p_improvement", 0)

    log(f"\n  BEST: {best_name}")
    log(f"    Sharpe: {best['net_sharpe']:.3f} (baseline: {base_sh:.3f}, "
        f"Δ={best['net_sharpe']-base_sh:+.3f})")
    log(f"    P(improvement): {best_p:.3f}")

    # ── Verdict ──
    delta_best = best["net_sharpe"] - base_sh
    if delta_best > 0.10 and best_p > 0.80:
        verdict = "POSITIVE"
        log(f"\n  ✅ VERDICT: POSITIVE — News features improve Sharpe by {delta_best:+.3f}")
    elif delta_best > 0 and best_p > 0.60:
        verdict = "MARGINAL"
        log(f"\n  ⚠️  VERDICT: MARGINAL — Improvement {delta_best:+.3f} but "
            f"P(imp)={best_p:.3f} < 0.80")
    else:
        verdict = "NEGATIVE"
        log(f"\n  ❌ VERDICT: NEGATIVE — News features do not help. "
            f"ΔSharpe={delta_best:+.3f}, P(imp)={best_p:.3f}")

    # ── IC results summary ──
    log(f"\n  IC Scan: {len(ic_passing)} features passed gate out of {len(all_news_feats)}")
    for r in ic_results[:5]:
        log(f"    {r['feature']}: IC={r['mean_ic']:+.4f} "
            f"{'✓' if r['stable'] else '✗'}")

    # ── Save ──
    save = {
        "verdict": verdict,
        "baseline_sharpe": base_sh,
        "best_experiment": best_name,
        "best_sharpe": best.get("net_sharpe", 0),
        "best_delta": round(delta_best, 3),
        "best_p_improvement": best_p,
        "ic_passing_features": ic_passing,
        "ic_results": ic_results,
        "experiments": {
            name: {k: v for k, v in m.items() if k != "port"}
            for name, m in experiments.items()
        },
        "bootstrap": bootstrap_results,
    }
    with open("results/r123_news_sentiment.json", "w") as f:
        json.dump(save, f, indent=2, default=str)
    log(f"\n  Saved: results/r123_news_sentiment.json")
    log(f"  Total time: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)")


if __name__ == "__main__":
    main()
