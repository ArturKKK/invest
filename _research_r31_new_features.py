#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R31 — New Features: High-IC features missing from production.

Key discovery from IC scan:
  - ret_168h       IC=-0.131 (STRONGEST of all 49 features, NOT in prod)
  - cum_funding_24h IC=-0.071 (stable across W1/W2/W3)
  - cum_funding_72h IC=-0.069
  - funding_rate_binance IC=-0.063
  - dist_from_high_24h IC=+0.056 (mean-reversion)
  - top_ls_ratio_zscore IC=-0.051

All computed from EXISTING data — no external API needed.

Experiments:
  A: Baseline 23f (= current prod, for comparison)
  B: 23f + 3 best new (ret_168h, cum_funding_24h, dist_from_high_24h) = 26f
  C: 23f + 6 new (above + cum_funding_72h, funding_rate_binance, top_ls_ratio_zscore) = 29f
  D: B + TS z-scores (funding_zscore, taker_zscore, premium_zscore) = 29f different
  E: C + D merged (all additions) = 32f

Portfolio configs: focused on best from R30b (3L2S_12h) + baseline (6L3S_12h).
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from scipy import stats
import warnings, time, sys, os

warnings.filterwarnings("ignore")

from _research_round7 import SYM_35, WINDOWS, compute_regime
from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal
from _research_r22_models import (
    FEATURES_23, SEEDS, LEVERAGE, CAPITAL,
    log, build_r19_features, add_new_features, cs_rank_cols,
)

# ══════════════════════════════════════════════════════════════
# FEATURE SETS
# ══════════════════════════════════════════════════════════════

FEAT_23 = FEATURES_23[:]

# B: +3 strongest unused IC features
FEAT_26 = FEAT_23 + [
    "ret_168h",            # IC=-0.131, 7d momentum/reversal
    "cum_funding_24h",     # IC=-0.071, cumulative carry signal
    "dist_from_high_24h",  # IC=+0.056, mean-reversion from 24h high
]

# C: +6 all strong unused
FEAT_29 = FEAT_26 + [
    "cum_funding_72h",        # IC=-0.069, 3d cumulative funding
    "funding_rate_binance",   # IC=-0.063, raw per-coin funding rate
    "top_ls_ratio_zscore",    # IC=-0.051, L/S positioning z-score
]

# D: 23f + 3 best + TS z-scores (already computed in pipeline)
FEAT_29_ZSCORE = FEAT_26 + [
    "funding_zscore",      # IC=-0.039, funding vs own history
    "taker_zscore",        # IC=-0.015, taker vs own history
    "premium_zscore",      # IC=-0.034, premium vs own history
]

# E: everything
FEAT_32 = list(dict.fromkeys(FEAT_29 + FEAT_29_ZSCORE))  # deduplicated, preserves order


# ══════════════════════════════════════════════════════════════
# REUSE INFRASTRUCTURE FROM R30b
# ══════════════════════════════════════════════════════════════

from _research_r30b_fixed import (
    simulate_with_costs,
    eval_with_costs,
    eval_per_window,
    add_extra_features_clean,
    compute_regime_extended,
    train_ensemble,
)


# ══════════════════════════════════════════════════════════════
# MARGINAL IC ANALYSIS
# ══════════════════════════════════════════════════════════════

def marginal_ic_analysis(df, base_feats, new_feats, label=""):
    """Check if new features add IC beyond what base features explain."""
    log(f"\n  [IC ANALYSIS] {label}")
    log(f"  Base: {len(base_feats)}f, New candidates: {len(new_feats)}")

    # Per-feature IC
    for feat in new_feats:
        if feat not in df.columns:
            log(f"    {feat}: MISSING")
            continue
        sub = df[["timestamp", "symbol", feat, "fwd_ret_12h"]].dropna()
        if len(sub) < 1000:
            log(f"    {feat}: too few rows ({len(sub)})")
            continue

        monthly_ics = []
        for ts, grp in sub.groupby(sub["timestamp"].dt.to_period("M")):
            if len(grp) >= 50:
                ic = stats.spearmanr(grp[feat], grp["fwd_ret_12h"])[0]
                monthly_ics.append(ic)

        if monthly_ics:
            mean_ic = np.mean(monthly_ics)
            icir = mean_ic / (np.std(monthly_ics) + 1e-10)
            ic_pos = sum(1 for x in monthly_ics if x > 0)
            log(f"    {feat:<25} IC={mean_ic:>+.4f}  ICIR={icir:>+.2f}  "
                f"IC>0={ic_pos}/{len(monthly_ics)}")

    # Correlation with existing features
    log(f"\n  Correlation of new features with existing (median abs corr):")
    for feat in new_feats:
        if feat not in df.columns:
            continue
        corrs = []
        for bf in base_feats:
            if bf in df.columns:
                c = df[[feat, bf]].dropna().corr().iloc[0, 1]
                corrs.append(abs(c))
        if corrs:
            log(f"    {feat:<25} median|corr|={np.median(corrs):.3f}  "
                f"max|corr|={np.max(corrs):.3f}  "
                f"(with {base_feats[np.argmax(corrs)]})")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log("=" * 80)
    log("  R31 — NEW FEATURES (High-IC, from existing data)")
    log("=" * 80)

    # ── 1. Load data ──
    log("\n[1] Loading data...")
    ohlcv = load_ohlcv()
    ohlcv = ohlcv[ohlcv["symbol"].isin(SYM_35)]
    derivs = load_derivatives()
    df = build_features_minimal(ohlcv, derivs)
    df = build_r19_features(df)
    df, _ = add_new_features(df)
    df = df[df["symbol"].isin(SYM_35)].copy()
    log(f"  Base: {len(df):,} rows, {len(df.columns)} cols")
    df = add_extra_features_clean(df)
    log(f"  After extras: {len(df):,} rows, {len(df.columns)} cols")
    regime_df = compute_regime_extended(df)

    # Check feature availability
    log("\n[1b] Feature availability check:")
    for feat_name, feat_list in [("FEAT_26", FEAT_26), ("FEAT_29", FEAT_29),
                                  ("FEAT_29_ZSCORE", FEAT_29_ZSCORE), ("FEAT_32", FEAT_32)]:
        avail = [f for f in feat_list if f in df.columns]
        missing = [f for f in feat_list if f not in df.columns]
        log(f"  {feat_name}: {len(avail)}/{len(feat_list)} available"
            + (f" (missing: {missing})" if missing else ""))

    # ── 2. Marginal IC analysis ──
    log("\n" + "=" * 60)
    log("[2] MARGINAL IC ANALYSIS")
    log("=" * 60)
    new_feats = [f for f in FEAT_32 if f not in FEAT_23]
    marginal_ic_analysis(df, FEAT_23, new_feats, "New vs FEAT_23")

    # ── 3. Train models ──
    log("\n" + "=" * 60)
    log("[3] TRAINING MODELS")
    log("=" * 60)

    # A: Baseline 23f (= current prod)
    log(f"\n[EXP-A] Baseline: {len(FEAT_23)}f")
    preds_A = train_ensemble(df, FEAT_23, WINDOWS, l2=1.0, rolling=False,
                              label="A_23f")

    # B: 23f + 3 best new
    log(f"\n[EXP-B] +3 new: {len(FEAT_26)}f (ret_168h, cum_funding_24h, dist_from_high_24h)")
    preds_B = train_ensemble(df, FEAT_26, WINDOWS, l2=1.0, rolling=False,
                              label="B_26f")

    # C: 23f + 6 new
    log(f"\n[EXP-C] +6 new: {len(FEAT_29)}f")
    preds_C = train_ensemble(df, FEAT_29, WINDOWS, l2=1.0, rolling=False,
                              label="C_29f")

    # D: 23f + 3 best + z-scores
    log(f"\n[EXP-D] +3 new + z-scores: {len(FEAT_29_ZSCORE)}f")
    preds_D = train_ensemble(df, FEAT_29_ZSCORE, WINDOWS, l2=1.0, rolling=False,
                              label="D_29f_z")

    # E: All features
    log(f"\n[EXP-E] All new: {len(FEAT_32)}f")
    preds_E = train_ensemble(df, FEAT_32, WINDOWS, l2=1.0, rolling=False,
                              label="E_32f")

    experiments = [
        ("A_23f", preds_A),
        ("B_26f", preds_B),
        ("C_29f", preds_C),
        ("D_29f_z", preds_D),
        ("E_32f", preds_E),
    ]

    # ── 4. Portfolio configs ──
    configs = [
        # R30b best: fewer positions
        ("3L2S_12h", {"n_long": 3, "n_short": 2, "rebal_hours": 12,
                      "trend_cutoff": 0.9, "dyn_threshold": 0.7}),
        # Current prod
        ("6L3S_12h", {"n_long": 6, "n_short": 3, "rebal_hours": 12,
                      "trend_cutoff": 0.9, "dyn_threshold": 0.7}),
        # Best turnover-reduced from R30b
        ("3L2S_ema05", {"n_long": 3, "n_short": 2, "rebal_hours": 12,
                        "trend_cutoff": 0.9, "dyn_threshold": 0.7,
                        "ema_alpha": 0.5}),
        ("6L3S_ema05_h3", {"n_long": 6, "n_short": 3, "rebal_hours": 12,
                           "trend_cutoff": 0.9, "dyn_threshold": 0.7,
                           "ema_alpha": 0.5, "hysteresis": 3}),
    ]

    # ── 5. Results ──
    log("\n\n" + "=" * 80)
    log("  RESULTS: R31 New Features")
    log("=" * 80)

    all_results = []

    for exp_name, preds in experiments:
        if preds is None:
            log(f"\n{exp_name}: FAILED")
            continue

        log(f"\n{'─' * 60}")
        log(f"  {exp_name}")
        log(f"{'─' * 60}")

        for cfg_name, cfg in configs:
            log(f"\n  [{cfg_name}]")
            results = eval_per_window(preds, regime_df, cfg, f"{exp_name}_{cfg_name}")
            all_results.append({
                "experiment": exp_name,
                "config": cfg_name,
                "W1_sh": results.get("W1", {}).get("sharpe", 0),
                "W1_sh_g": results.get("W1", {}).get("sharpe_gross", 0),
                "W2_sh": results.get("W2", {}).get("sharpe", 0),
                "W2_sh_g": results.get("W2", {}).get("sharpe_gross", 0),
                "W3_sh": results.get("W3", {}).get("sharpe", 0),
                "W3_sh_g": results.get("W3", {}).get("sharpe_gross", 0),
                "ALL_sh": results.get("ALL", {}).get("sharpe", 0),
                "ALL_sh_g": results.get("ALL", {}).get("sharpe_gross", 0),
                "W3_eq": results.get("W3", {}).get("equity", 0),
                "W3_dd": results.get("W3", {}).get("max_dd_pct", 0),
                "ALL_turn": results.get("ALL", {}).get("avg_turnover", 0),
                "ALL_cost": results.get("ALL", {}).get("total_cost_pct", 0),
            })

    # ── 6. Summary table ──
    log("\n\n" + "=" * 80)
    log("  SUMMARY (sorted by W3 net Sharpe)")
    log("=" * 80)
    log(f"\n{'Experiment':<12} {'Config':<18} {'W1net':>5} {'W2net':>5} {'W3net':>5} "
        f"{'W3grs':>5} {'ALLnt':>5} {'Eq$':>6} {'DD%':>6} {'Turn':>5} {'Cost%':>6}")
    log("─" * 100)

    all_results.sort(key=lambda x: -x["W3_sh"])
    for r in all_results:
        marker = " ★" if r["W3_sh"] >= 2.0 else (" ◆" if r["W3_sh"] >= 1.0 else "")
        log(f"{r['experiment']:<12} {r['config']:<18} "
            f"{r['W1_sh']:>5.2f} {r['W2_sh']:>5.2f} {r['W3_sh']:>5.2f} "
            f"{r['W3_sh_g']:>5.2f} {r['ALL_sh']:>5.2f} "
            f"${r['W3_eq']:>5.0f} {r['W3_dd']:>+5.1f}% "
            f"{r['ALL_turn']:>4.1f} {r['ALL_cost']:>5.1f}%{marker}")

    # ── 7. Head-to-head: A vs B (the key comparison) ──
    log("\n\n" + "=" * 80)
    log("  HEAD-TO-HEAD: A (23f baseline) vs B (26f +3 new)")
    log("=" * 80)
    for cfg_name, _ in configs:
        a_res = [r for r in all_results if r["experiment"] == "A_23f" and r["config"] == cfg_name]
        b_res = [r for r in all_results if r["experiment"] == "B_26f" and r["config"] == cfg_name]
        if a_res and b_res:
            a, b = a_res[0], b_res[0]
            delta_w3 = b["W3_sh"] - a["W3_sh"]
            delta_all = b["ALL_sh"] - a["ALL_sh"]
            verdict = "✅ BETTER" if delta_w3 > 0.1 else ("⚠️ ~SAME" if abs(delta_w3) <= 0.1 else "❌ WORSE")
            log(f"  {cfg_name:<18} A_W3={a['W3_sh']:>5.2f} → B_W3={b['W3_sh']:>5.2f}  "
                f"Δ={delta_w3:>+.2f}  |  A_ALL={a['ALL_sh']:>5.2f} → B_ALL={b['ALL_sh']:>5.2f}  "
                f"Δ={delta_all:>+.2f}  {verdict}")

    # ── 8. Cost impact ──
    log("\n\n" + "=" * 80)
    log("  COST IMPACT (top 10 by W3 net)")
    log("=" * 80)
    for r in all_results[:10]:
        if r["W3_sh_g"] > 0:
            cost_drag = r["W3_sh_g"] - r["W3_sh"]
            pct_eaten = cost_drag / r["W3_sh_g"] * 100
            log(f"  {r['experiment']:<12} {r['config']:<18}: "
                f"Gross={r['W3_sh_g']:>5.2f} → Net={r['W3_sh']:>5.2f}  "
                f"({pct_eaten:>4.0f}% eaten by costs)")

    # ── 9. IC analysis for best model ──
    if all_results:
        best = all_results[0]
        best_exp = best["experiment"]
        best_preds = None
        for name, preds in experiments:
            if name == best_exp:
                best_preds = preds
                break

        if best_preds is not None:
            log("\n\n" + "=" * 80)
            log(f"  MONTHLY IC — {best_exp}")
            log("=" * 80)
            monthly_ics = []
            for ts, grp in best_preds.groupby(best_preds["timestamp"].dt.to_period("M")):
                if len(grp) >= 50:
                    ic = stats.spearmanr(grp["pred"], grp["fwd_ret"])[0]
                    monthly_ics.append({"month": str(ts), "ic": ic, "n": len(grp)})
            if monthly_ics:
                log(f"\n{'Month':<10} {'IC':>8} {'N':>6}")
                log("─" * 28)
                for m in monthly_ics:
                    marker = " ⚠️" if m["ic"] < 0 else ""
                    log(f"{m['month']:<10} {m['ic']:>+8.4f} {m['n']:>6}{marker}")
                ics = [m["ic"] for m in monthly_ics]
                log(f"\nMean IC: {np.mean(ics):.4f}, "
                    f"IC>0: {sum(1 for x in ics if x > 0)}/{len(ics)}, "
                    f"ICIR: {np.mean(ics)/(np.std(ics)+1e-10):.2f}")

    elapsed = time.time() - t0
    log(f"\n\n✅ R31 complete in {elapsed/60:.1f} min")

    # ── 10. Verdict ──
    if all_results:
        best = all_results[0]
        a_best = [r for r in all_results if r["experiment"] == "A_23f"]
        a_best_sh = max(r["W3_sh"] for r in a_best) if a_best else 0

        log(f"\n{'='*80}")
        log(f"  VERDICT")
        log(f"{'='*80}")
        log(f"  Best overall: {best['experiment']} × {best['config']}")
        log(f"  W3 Sharpe: net={best['W3_sh']:.2f}, gross={best['W3_sh_g']:.2f}")
        log(f"  Best baseline (A_23f): W3 net={a_best_sh:.2f}")
        log(f"  Delta: {best['W3_sh'] - a_best_sh:>+.2f}")

        if best["experiment"] != "A_23f" and best["W3_sh"] > a_best_sh + 0.1:
            log(f"  ✅ NEW FEATURES HELP! +{best['W3_sh'] - a_best_sh:.2f} net Sharpe improvement")
        elif best["experiment"] == "A_23f":
            log(f"  ❌ NEW FEATURES DON'T HELP. Baseline 23f is still best.")
        else:
            log(f"  ⚠️  MARGINAL: New features ≈ baseline. Probably not worth the complexity.")


if __name__ == "__main__":
    log_path = "results_r31.log"

    class Tee:
        def __init__(self, fname):
            self.file = open(fname, "w")
            self.stdout = sys.stdout
        def write(self, data):
            self.stdout.write(data)
            self.file.write(data)
        def flush(self):
            self.stdout.flush()
            self.file.flush()

    sys.stdout = Tee(log_path)
    try:
        main()
    finally:
        sys.stdout.file.close()
        sys.stdout = sys.stdout.stdout
    print(f"\nLog saved to {log_path}")
