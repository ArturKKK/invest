#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R32 — Kaggle-inspired features from existing data.

IC scan found 2 candidates with ICIR > 0.10 that are NOT redundant with existing 26f:
  1. rel_volume_cs  (ICIR=-0.106, ORTHOGONAL to all 26 production features)
     = log(volume) - mean(log(volume)) at each timestamp
     = cross-sectional relative volume
  2. ret_skew_168h  (ICIR=-0.103, corr=+0.36 with ret_168h only)
     = rolling 168h skewness of 1h returns

Experiments:
  A: Baseline 26f (= R31 best, for comparison)
  B: 26f + rel_volume_cs = 27f
  C: 26f + ret_skew_168h = 27f_skew
  D: 26f + both = 28f

Key constraint: R31 showed adding noise features HURTS (29f/32f worse than 26f).
Only orthogonal / semi-independent features with ICIR > 0.10 are candidates.
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
from _research_r30b_fixed import (
    simulate_with_costs,
    eval_with_costs,
    eval_per_window,
    add_extra_features_clean,
    compute_regime_extended,
    train_ensemble,
)

# ══════════════════════════════════════════════════════════════
# FEATURE SETS
# ══════════════════════════════════════════════════════════════

FEAT_26 = FEATURES_23 + [
    "ret_168h",
    "cum_funding_24h",
    "dist_from_high_24h",
]

FEAT_27_VOL = FEAT_26 + ["rel_volume_cs"]
FEAT_27_SKEW = FEAT_26 + ["ret_skew_168h"]
FEAT_28 = FEAT_26 + ["rel_volume_cs", "ret_skew_168h"]


# ══════════════════════════════════════════════════════════════
# ADD KAGGLE FEATURES
# ══════════════════════════════════════════════════════════════

def add_kaggle_features(df):
    """Add Kaggle-inspired features computed from existing data."""
    n_before = len(df.columns)

    # 1. rel_volume_cs: cross-sectional relative volume
    #    = log(volume) - mean(log(volume)) at each timestamp
    #    ORTHOGONAL to all 26 prod features (Kaggle insight)
    df["_log_vol"] = np.log(df["volume"].clip(lower=1))
    cs_mean = df.groupby("timestamp")["_log_vol"].transform("mean")
    df["rel_volume_cs"] = df["_log_vol"] - cs_mean
    df.drop(columns=["_log_vol"], inplace=True)

    # 2. ret_skew_168h: rolling 168h skewness of hourly returns
    #    Semi-independent from existing (corr=0.36 with ret_168h)
    if "ret_skew_168h" not in df.columns:
        if "ret_1h" not in df.columns:
            df["ret_1h"] = df.groupby("symbol")["close"].pct_change(1)
        df["ret_skew_168h"] = df.groupby("symbol")["ret_1h"].transform(
            lambda x: x.rolling(168, min_periods=84).skew()
        )

    n_after = len(df.columns)
    log(f"  [KAGGLE] Added {n_after - n_before} features (rel_volume_cs, ret_skew_168h)")
    return df


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log("=" * 80)
    log("  R32 — KAGGLE-INSPIRED FEATURES")
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

    # Add Kaggle features
    df = add_kaggle_features(df)
    log(f"  Final: {len(df):,} rows, {len(df.columns)} cols")

    regime_df = compute_regime_extended(df)

    # Check feature availability
    log("\n[1b] Feature availability:")
    for name, flist in [("FEAT_26", FEAT_26), ("FEAT_27_VOL", FEAT_27_VOL),
                         ("FEAT_27_SKEW", FEAT_27_SKEW), ("FEAT_28", FEAT_28)]:
        avail = [f for f in flist if f in df.columns]
        missing = [f for f in flist if f not in df.columns]
        log(f"  {name}: {len(avail)}/{len(flist)}"
            + (f" MISSING: {missing}" if missing else " ✓"))

    # ── 2. Train models ──
    log("\n" + "=" * 60)
    log("[2] TRAINING MODELS")
    log("=" * 60)

    experiments = []

    # A: Baseline 26f (R31 best)
    log(f"\n[EXP-A] Baseline: {len(FEAT_26)}f")
    preds_A = train_ensemble(df, FEAT_26, WINDOWS, l2=1.0, rolling=False, label="A_26f")
    experiments.append(("A_26f", preds_A))

    # B: +rel_volume_cs
    log(f"\n[EXP-B] +rel_volume_cs: {len(FEAT_27_VOL)}f")
    preds_B = train_ensemble(df, FEAT_27_VOL, WINDOWS, l2=1.0, rolling=False, label="B_27f_vol")
    experiments.append(("B_27f_vol", preds_B))

    # C: +ret_skew_168h
    log(f"\n[EXP-C] +ret_skew_168h: {len(FEAT_27_SKEW)}f")
    preds_C = train_ensemble(df, FEAT_27_SKEW, WINDOWS, l2=1.0, rolling=False, label="C_27f_skew")
    experiments.append(("C_27f_skew", preds_C))

    # D: +both
    log(f"\n[EXP-D] +both: {len(FEAT_28)}f")
    preds_D = train_ensemble(df, FEAT_28, WINDOWS, l2=1.0, rolling=False, label="D_28f")
    experiments.append(("D_28f", preds_D))

    # ── 3. Portfolio configs ──
    configs = [
        ("6L3S_ema05_h3", {"n_long": 6, "n_short": 3, "rebal_hours": 12,
                           "trend_cutoff": 0.9, "dyn_threshold": 0.7,
                           "ema_alpha": 0.5, "hysteresis": 3}),
        ("6L3S_12h", {"n_long": 6, "n_short": 3, "rebal_hours": 12,
                      "trend_cutoff": 0.9, "dyn_threshold": 0.7}),
        ("3L2S_ema05", {"n_long": 3, "n_short": 2, "rebal_hours": 12,
                        "trend_cutoff": 0.9, "dyn_threshold": 0.7,
                        "ema_alpha": 0.5}),
    ]

    # ── 4. Results ──
    log("\n\n" + "=" * 80)
    log("  RESULTS: R32 Kaggle Features")
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

    # ── 5. Summary table ──
    log("\n\n" + "=" * 80)
    log("  SUMMARY (sorted by W3 net Sharpe)")
    log("=" * 80)
    log(f"\n{'Experiment':<14} {'Config':<18} {'W1net':>5} {'W2net':>5} {'W3net':>5} "
        f"{'W3grs':>5} {'ALLnt':>5} {'Eq$':>6} {'DD%':>6} {'Turn':>5} {'Cost%':>6}")
    log("─" * 105)

    all_results.sort(key=lambda x: -x["W3_sh"])
    for r in all_results:
        marker = " ★" if r["W3_sh"] >= 2.0 else (" ◆" if r["W3_sh"] >= 1.0 else "")
        log(f"{r['experiment']:<14} {r['config']:<18} "
            f"{r['W1_sh']:>5.2f} {r['W2_sh']:>5.2f} {r['W3_sh']:>5.2f} "
            f"{r['W3_sh_g']:>5.2f} {r['ALL_sh']:>5.2f} "
            f"${r['W3_eq']:>5.0f} {r['W3_dd']:>+5.1f}% "
            f"{r['ALL_turn']:>4.1f} {r['ALL_cost']:>5.1f}%{marker}")

    # ── 6. Head-to-head vs baseline ──
    log("\n\n" + "=" * 80)
    log("  HEAD-TO-HEAD: A (26f baseline) vs B/C/D")
    log("=" * 80)
    for cfg_name, _ in configs:
        a_res = [r for r in all_results if r["experiment"] == "A_26f" and r["config"] == cfg_name]
        if not a_res:
            continue
        a = a_res[0]
        log(f"\n  [{cfg_name}] Baseline A_26f: W3={a['W3_sh']:.2f}")
        for exp_name in ["B_27f_vol", "C_27f_skew", "D_28f"]:
            b_res = [r for r in all_results if r["experiment"] == exp_name and r["config"] == cfg_name]
            if not b_res:
                continue
            b = b_res[0]
            delta = b["W3_sh"] - a["W3_sh"]
            verdict = "✅ BETTER" if delta > 0.1 else ("⚠️ ~SAME" if abs(delta) <= 0.1 else "❌ WORSE")
            log(f"    {exp_name:<14} W3={b['W3_sh']:.2f}  Δ={delta:>+.2f}  {verdict}")

    # ── 7. Monthly IC for best model ──
    if all_results:
        best = all_results[0]
        best_preds = None
        for name, preds in experiments:
            if name == best["experiment"]:
                best_preds = preds
                break

        if best_preds is not None:
            log(f"\n\n{'=' * 80}")
            log(f"  MONTHLY IC — {best['experiment']}")
            log(f"{'=' * 80}")
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
    log(f"\n\n✅ R32 complete in {elapsed/60:.1f} min")

    # ── 8. Verdict ──
    if all_results:
        best = all_results[0]
        a_best = [r for r in all_results if r["experiment"] == "A_26f"]
        a_best_sh = max(r["W3_sh"] for r in a_best) if a_best else 0

        log(f"\n{'='*80}")
        log(f"  VERDICT")
        log(f"{'='*80}")
        log(f"  Best overall: {best['experiment']} × {best['config']}")
        log(f"  W3 Sharpe: net={best['W3_sh']:.2f}, gross={best['W3_sh_g']:.2f}")
        log(f"  Best baseline (A_26f): W3 net={a_best_sh:.2f}")
        delta = best['W3_sh'] - a_best_sh
        log(f"  Delta: {delta:>+.2f}")

        if best["experiment"] != "A_26f" and delta > 0.1:
            log(f"  ✅ KAGGLE FEATURES HELP! +{delta:.2f} net Sharpe improvement")
        elif best["experiment"] == "A_26f":
            log(f"  ❌ KAGGLE FEATURES DON'T HELP. Baseline 26f is still best.")
        else:
            log(f"  ⚠️  MARGINAL: Kaggle features ≈ baseline.")


if __name__ == "__main__":
    log_path = "results_r32.log"

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
