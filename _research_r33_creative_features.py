#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R33 — Creative features from existing data.

From massive IC scan of 43 candidates, best NEW non-redundant features:
  1. btc_corr_168h  ICIR=+0.172  (rolling 168h correlation with BTC, partially independent)
  2. btc_corr_24h   ICIR=+0.156  (rolling 24h correlation with BTC)

These add a new dimension: HOW CORRELATED a coin is with BTC right now.
High btc_corr → coin moves with market. Low → idiosyncratic move.

Experiments:
  A: Baseline 28f (= R32 best D_28f: 26f + rel_volume_cs + ret_skew_168h)
  B: 28f + btc_corr_168h = 29f
  C: 28f + btc_corr_24h = 29f_24
  D: 28f + both btc_corr = 30f
  E: 28f + btc_corr_168h + upvol_24h = 30f_alt (upvol partially redundant but ICIR=0.163)
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

FEAT_28 = FEATURES_23 + [
    "ret_168h",
    "cum_funding_24h",
    "dist_from_high_24h",
    "rel_volume_cs",
    "ret_skew_168h",
]

FEAT_29_168 = FEAT_28 + ["btc_corr_168h"]
FEAT_29_24 = FEAT_28 + ["btc_corr_24h"]
FEAT_30 = FEAT_28 + ["btc_corr_168h", "btc_corr_24h"]
FEAT_30_ALT = FEAT_28 + ["btc_corr_168h", "upvol_24h"]


# ══════════════════════════════════════════════════════════════
# ADD NEW FEATURES
# ══════════════════════════════════════════════════════════════

def add_r33_features(df):
    """Add R33 creative features from existing data."""
    n_before = len(df.columns)

    # 1. rel_volume_cs: cross-sectional relative volume
    df["_log_vol"] = np.log(df["volume"].clip(lower=1))
    cs_mean = df.groupby("timestamp")["_log_vol"].transform("mean")
    df["rel_volume_cs"] = df["_log_vol"] - cs_mean
    df.drop(columns=["_log_vol"], inplace=True)

    # 2. ret_skew_168h: rolling skewness of hourly returns
    if "ret_skew_168h" not in df.columns:
        if "ret_1h" not in df.columns:
            df["ret_1h"] = df.groupby("symbol")["close"].pct_change(1)
        df["ret_skew_168h"] = df.groupby("symbol")["ret_1h"].transform(
            lambda x: x.rolling(168, min_periods=84).skew()
        )

    # 3. BTC rolling correlation (time-varying beta proxy)
    # btc_ret_1h already exists from build_features_minimal()
    if "ret_1h" not in df.columns:
        df["ret_1h"] = df.groupby("symbol")["close"].pct_change(1)

    if "btc_ret_1h" not in df.columns:
        btc = df[df["symbol"] == "BTC/USDT"][["timestamp", "close"]].copy()
        btc["btc_ret_1h"] = btc["close"].pct_change(1)
        btc = btc[["timestamp", "btc_ret_1h"]]
        df = df.merge(btc, on="timestamp", how="left")

    for n in [24, 168]:
        col = f"btc_corr_{n}h"
        if col not in df.columns:
            df[col] = df.groupby("symbol").apply(
                lambda x: x["ret_1h"].rolling(n, min_periods=n // 2).corr(x["btc_ret_1h"])
            ).droplevel(0)

    # 4. Upside volatility (partially redundant with atr but ICIR=0.163)
    if "upvol_24h" not in df.columns:
        if "ret_1h" not in df.columns:
            df["ret_1h"] = df.groupby("symbol")["close"].pct_change(1)
        ret_pos = df["ret_1h"].clip(lower=0)
        df["upvol_24h"] = ret_pos.groupby(df["symbol"]).transform(
            lambda x: x.rolling(24, min_periods=12).std()
        )

    n_after = len(df.columns)
    log(f"  [R33] Added {n_after - n_before} features (btc_corr, rel_volume_cs, ret_skew, upvol)")
    return df


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log("=" * 80)
    log("  R33 — CREATIVE FEATURES (interactions, BTC correlation, asymmetric vol)")
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
    df = add_r33_features(df)
    log(f"  Final: {len(df):,} rows, {len(df.columns)} cols")

    regime_df = compute_regime_extended(df)

    # Check feature availability
    log("\n[1b] Feature availability:")
    for name, flist in [("FEAT_28", FEAT_28), ("FEAT_29_168", FEAT_29_168),
                         ("FEAT_29_24", FEAT_29_24), ("FEAT_30", FEAT_30),
                         ("FEAT_30_ALT", FEAT_30_ALT)]:
        avail = [f for f in flist if f in df.columns]
        missing = [f for f in flist if f not in df.columns]
        log(f"  {name}: {len(avail)}/{len(flist)}"
            + (f" MISSING: {missing}" if missing else " ✓"))

    # ── 2. Train models ──
    log("\n" + "=" * 60)
    log("[2] TRAINING MODELS")
    log("=" * 60)

    experiments = []

    log(f"\n[EXP-A] Baseline: {len(FEAT_28)}f (R32 best)")
    preds_A = train_ensemble(df, FEAT_28, WINDOWS, l2=1.0, rolling=False, label="A_28f")
    experiments.append(("A_28f", preds_A))

    log(f"\n[EXP-B] +btc_corr_168h: {len(FEAT_29_168)}f")
    preds_B = train_ensemble(df, FEAT_29_168, WINDOWS, l2=1.0, rolling=False, label="B_29f_168")
    experiments.append(("B_29f_168", preds_B))

    log(f"\n[EXP-C] +btc_corr_24h: {len(FEAT_29_24)}f")
    preds_C = train_ensemble(df, FEAT_29_24, WINDOWS, l2=1.0, rolling=False, label="C_29f_24")
    experiments.append(("C_29f_24", preds_C))

    log(f"\n[EXP-D] +both btc_corr: {len(FEAT_30)}f")
    preds_D = train_ensemble(df, FEAT_30, WINDOWS, l2=1.0, rolling=False, label="D_30f")
    experiments.append(("D_30f", preds_D))

    log(f"\n[EXP-E] +btc_corr_168h +upvol_24h: {len(FEAT_30_ALT)}f")
    preds_E = train_ensemble(df, FEAT_30_ALT, WINDOWS, l2=1.0, rolling=False, label="E_30f_alt")
    experiments.append(("E_30f_alt", preds_E))

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
    log("  RESULTS: R33 Creative Features")
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
                "experiment": exp_name, "config": cfg_name,
                "W1_sh": results.get("W1", {}).get("sharpe", 0),
                "W2_sh": results.get("W2", {}).get("sharpe", 0),
                "W3_sh": results.get("W3", {}).get("sharpe", 0),
                "W3_sh_g": results.get("W3", {}).get("sharpe_gross", 0),
                "ALL_sh": results.get("ALL", {}).get("sharpe", 0),
                "W3_eq": results.get("W3", {}).get("equity", 0),
                "W3_dd": results.get("W3", {}).get("max_dd_pct", 0),
                "ALL_turn": results.get("ALL", {}).get("avg_turnover", 0),
                "ALL_cost": results.get("ALL", {}).get("total_cost_pct", 0),
            })

    # ── 5. Summary ──
    log("\n\n" + "=" * 80)
    log("  SUMMARY (sorted by W3 net Sharpe)")
    log("=" * 80)
    log(f"\n{'Experiment':<14} {'Config':<18} {'W1net':>5} {'W2net':>5} {'W3net':>5} "
        f"{'W3grs':>5} {'ALLnt':>5} {'Eq$':>6} {'DD%':>6} {'Turn':>5}")
    log("─" * 95)

    all_results.sort(key=lambda x: -x["W3_sh"])
    for r in all_results:
        marker = " ★" if r["W3_sh"] >= 2.5 else (" ◆" if r["W3_sh"] >= 1.5 else "")
        log(f"{r['experiment']:<14} {r['config']:<18} "
            f"{r['W1_sh']:>5.2f} {r['W2_sh']:>5.2f} {r['W3_sh']:>5.2f} "
            f"{r['W3_sh_g']:>5.2f} {r['ALL_sh']:>5.2f} "
            f"${r['W3_eq']:>5.0f} {r['W3_dd']:>+5.1f}% "
            f"{r['ALL_turn']:>4.1f}{marker}")

    # ── 6. Head-to-head ──
    log("\n\n" + "=" * 80)
    log("  HEAD-TO-HEAD: A (28f baseline) vs B/C/D/E")
    log("=" * 80)
    for cfg_name, _ in configs:
        a_res = [r for r in all_results if r["experiment"] == "A_28f" and r["config"] == cfg_name]
        if not a_res: continue
        a = a_res[0]
        log(f"\n  [{cfg_name}] Baseline A_28f: W3={a['W3_sh']:.2f}")
        for exp in ["B_29f_168", "C_29f_24", "D_30f", "E_30f_alt"]:
            b_res = [r for r in all_results if r["experiment"] == exp and r["config"] == cfg_name]
            if not b_res: continue
            b = b_res[0]
            delta = b["W3_sh"] - a["W3_sh"]
            verdict = "✅ BETTER" if delta > 0.1 else ("⚠️ ~SAME" if abs(delta) <= 0.1 else "❌ WORSE")
            log(f"    {exp:<14} W3={b['W3_sh']:.2f}  Δ={delta:>+.2f}  {verdict}")

    # ── 7. Monthly IC ──
    if all_results:
        best = all_results[0]
        best_preds = None
        for name, preds in experiments:
            if name == best["experiment"]:
                best_preds = preds
                break
        if best_preds is not None:
            log(f"\n\n{'='*80}")
            log(f"  MONTHLY IC — {best['experiment']}")
            log(f"{'='*80}")
            monthly_ics = []
            for ts, grp in best_preds.groupby(best_preds["timestamp"].dt.to_period("M")):
                if len(grp) >= 50:
                    ic = stats.spearmanr(grp["pred"], grp["fwd_ret"])[0]
                    monthly_ics.append({"month": str(ts), "ic": ic})
            if monthly_ics:
                for m in monthly_ics:
                    log(f"  {m['month']:<10} IC={m['ic']:>+.4f}")
                ics = [m["ic"] for m in monthly_ics]
                log(f"\n  Mean IC: {np.mean(ics):.4f}, IC>0: {sum(1 for x in ics if x > 0)}/{len(ics)}, "
                    f"ICIR: {np.mean(ics)/(np.std(ics)+1e-10):.2f}")

    elapsed = time.time() - t0
    log(f"\n\n✅ R33 complete in {elapsed/60:.1f} min")

    # ── 8. Verdict ──
    if all_results:
        best = all_results[0]
        a_best = [r for r in all_results if r["experiment"] == "A_28f"]
        a_best_sh = max(r["W3_sh"] for r in a_best) if a_best else 0
        log(f"\n{'='*80}")
        log(f"  VERDICT")
        log(f"{'='*80}")
        log(f"  Best overall: {best['experiment']} × {best['config']}")
        log(f"  W3 Sharpe: net={best['W3_sh']:.2f}, gross={best['W3_sh_g']:.2f}")
        log(f"  R32 baseline (A_28f): W3 net={a_best_sh:.2f}")
        delta = best['W3_sh'] - a_best_sh
        log(f"  Delta: {delta:>+.2f}")
        if best["experiment"] != "A_28f" and delta > 0.1:
            log(f"  ✅ NEW FEATURES HELP! +{delta:.2f} net Sharpe")
        elif best["experiment"] == "A_28f":
            log(f"  ❌ NEW FEATURES DON'T HELP. 28f is still best.")
        else:
            log(f"  ⚠️  MARGINAL.")


if __name__ == "__main__":
    log_path = "results_r33.log"
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
