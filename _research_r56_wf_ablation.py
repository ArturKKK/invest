#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R56 — WF Ablation: Feature Substitution Experiments

Plan:
  Phase 0: Feature importance (LGB gain) → find weakest features
  Phase 1: Baseline (35-coin) + Coverage stress test (27-coin)
  Phase 2: Basis substitution experiments
  Phase 3: OI/FR substitution experiments (conditional)
  Phase 4: Double substitution (conditional)

Usage:
  python _research_r56_wf_ablation.py                    # full run
  python _research_r56_wf_ablation.py --importance-only   # phase 0 only
  python _research_r56_wf_ablation.py --phase 2           # specific phase
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── project imports ───────────────────────────────────────────

from _research_round7 import WINDOWS, SYM_35, compute_regime
from _research_r22_models import SEEDS, LEVERAGE, CAPITAL, log, cs_rank_cols
from _research_r30b_fixed import (
    train_ensemble,
    eval_with_costs,
)
from _research_r33_creative_features import FEAT_28
from _research_r35_new_features import (
    add_r35_features,
    load_research_frame,
    MARKET_LEVEL_FEATURES,
)
from _research_r47_coinglass import (
    CANONICAL_EXEC_CFG,
    CHAMPION_FEAT_30,
    add_cg_features,
    compute_cg_features,
    load_cg_daily,
)
from _research_r48_cost import simulate_with_hybrid_costs
from _research_r55_cg_features import (
    build_all_r55_features,
    merge_r55_into_model,
)

# True champion = 30 base + cg_taker_imb
CHAMPION_FEAT_31 = CHAMPION_FEAT_30 + ["cg_taker_imb"]

# ── config ─────────────────────────────────────────────────────

# Symbols that have basis data (27 out of 35)
# Determined from R55 download: 8 symbols got server errors
BASIS_AVAILABLE_SYMS = None  # Will be detected at runtime from basis.parquet

# Substitution experiments to run
EXPERIMENTS = {
    # Phase 2: Basis
    "2.1_weakest_to_basis_z": {"old": "__WEAKEST__", "new": "cg_basis_z_60d"},
    "2.2_cumfunding_to_basis_z": {"old": "cum_funding_24h", "new": "cg_basis_z_60d"},
    # Phase 3: OI/FR (conditional)
    "3.1_cumfunding_to_fr_close": {"old": "cum_funding_24h", "new": "cg_fr_close"},
    "3.2_cumfunding_to_fr_disagree": {"old": "cum_funding_24h", "new": "cg_fr_disagreement"},
    "3.3_oizscore_to_oi_chg": {"old": "oi_zscore", "new": "cg_oi_chg_1d"},
}


# ═════════════════════════════════════════════════════════════
#  PHASE 0: Feature Importance
# ═════════════════════════════════════════════════════════════

def train_with_importance(df: pd.DataFrame, feats: List[str],
                           windows: list) -> pd.DataFrame:
    """
    Train single-seed LGB on each window and extract feature importance (gain).
    Returns DataFrame: feature → W1_gain, W2_gain, W3_gain, avg_gain.
    """
    print("\n  Extracting LGB feature importance (gain)...")
    avail = [f for f in feats if f in df.columns]
    rank_feats = avail[:]
    tz = df["timestamp"].dt.tz
    importance_records = []

    params_lgb = {
        "objective": "binary", "metric": "auc",
        "learning_rate": 0.03, "num_leaves": 63,
        "min_child_samples": 100, "subsample": 0.8,
        "colsample_bytree": 0.8, "lambda_l2": 1.0,
        "verbose": -1, "n_jobs": -1, "seed": SEEDS[0],
    }

    for w in windows:
        tr_end = pd.Timestamp(w["train_end"], tz=tz)
        va_start = pd.Timestamp(w["val_start"], tz=tz)
        va_end = pd.Timestamp(w["val_end"], tz=tz)

        train_ = df[df["timestamp"] < tr_end].copy()
        val_ = df[(df["timestamp"] >= va_start) &
                   (df["timestamp"] < va_end)].copy()

        if len(train_) < 5000:
            continue

        train_ = cs_rank_cols(train_, rank_feats)
        val_ = cs_rank_cols(val_, rank_feats)

        for d in [train_, val_]:
            d["target_binary"] = (d["fwd_ret_12h"] > 0).astype(int)

        # Fill NaN with 0 for LGB (same as train_ensemble)
        for col in avail:
            for d in [train_, val_]:
                if d[col].isna().any():
                    d[col] = d[col].fillna(0)

        tr = train_[avail + ["target_binary"]].dropna()
        va = val_[avail + ["target_binary"]].dropna()
        tr.replace([np.inf, -np.inf], np.nan, inplace=True)
        tr = tr.dropna()
        va.replace([np.inf, -np.inf], np.nan, inplace=True)
        va = va.dropna()

        dt = lgb.Dataset(tr[avail], label=tr["target_binary"])
        dv = lgb.Dataset(va[avail], label=va["target_binary"])
        m = lgb.train(params_lgb, dt, num_boost_round=600,
                       valid_sets=[dv],
                       callbacks=[lgb.early_stopping(40, verbose=False),
                                  lgb.log_evaluation(-1)])

        imp = dict(zip(m.feature_name(), m.feature_importance(importance_type="gain")))
        for feat, gain in imp.items():
            importance_records.append({
                "feature": feat, "window": w["name"], "gain": gain
            })

        print(f"    {w['name']}: {len(tr):,} train, {m.best_iteration} rounds")

    if not importance_records:
        return pd.DataFrame()

    imp_df = pd.DataFrame(importance_records)
    pivot = imp_df.pivot_table(index="feature", columns="window",
                                values="gain", aggfunc="first")
    pivot["avg_gain"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("avg_gain", ascending=False)

    return pivot


# ═════════════════════════════════════════════════════════════
#  Hybrid-cost per-window evaluation (replaces eval_per_window)
# ═════════════════════════════════════════════════════════════

def eval_per_window_hybrid(preds, regime_df, cfg, label=""):
    """Per-window evaluation using tiered hybrid costs (R48)."""
    results = {}
    for wname in ["W1", "W2", "W3"]:
        sub = preds[preds["window"] == wname]
        if len(sub) < 10:
            results[wname] = {"sharpe": 0, "sharpe_gross": 0}
            continue
        port = simulate_with_hybrid_costs(sub, regime_df, cfg)
        r = eval_with_costs(port, f"{label}_{wname}")
        results[wname] = r
        log(f"  {wname}: Sh={r['sharpe']:>5.2f} (gross={r['sharpe_gross']:>5.2f})  "
            f"Eq=${r['equity']:>6.0f}  DD={r['max_dd_pct']:>+5.1f}%  "
            f"WM={r['win_months']}  Cost={r['total_cost_pct']:.1f}%  Turn={r['avg_turnover']:.1f}")

    # Combined
    port_all = simulate_with_hybrid_costs(preds, regime_df, cfg)
    r_all = eval_with_costs(port_all, label)
    log(f"  ALL: Sh={r_all['sharpe']:>5.2f} (gross={r_all['sharpe_gross']:>5.2f})  "
        f"Eq=${r_all['equity']:>6.0f}  DD={r_all['max_dd_pct']:>+5.1f}%  "
        f"Cost={r_all['total_cost_pct']:.1f}%  Turn={r_all['avg_turnover']:.1f}")

    results["ALL"] = r_all
    return results


# ═════════════════════════════════════════════════════════════
#  CORE: Run one WF experiment
# ═════════════════════════════════════════════════════════════

def run_wf_experiment(df: pd.DataFrame, regime_df: pd.DataFrame,
                       feats: List[str], label: str,
                       universe_filter: set = None) -> Optional[Dict]:
    """
    Run full walk-forward with given features and optional universe filter.
    Returns dict with per-window and ALL metrics.
    """
    work_df = df.copy()
    if universe_filter:
        work_df = work_df[work_df["symbol"].isin(universe_filter)].copy()
        print(f"  [{label}] Universe: {work_df['symbol'].nunique()} symbols")

    present = [f for f in feats if f in work_df.columns]
    missing = [f for f in feats if f not in work_df.columns]
    if missing:
        print(f"  [{label}] Missing features: {missing}")
    if len(present) < len(feats) * 0.8:
        print(f"  [{label}] Too many missing features, skipping")
        return None

    no_rank = [f for f in present if f in MARKET_LEVEL_FEATURES]
    print(f"  [{label}] Training {len(present)}f ensemble (cs_rank_exclude={no_rank})...")
    preds = train_ensemble(work_df, present, WINDOWS, label=label,
                           cs_rank_exclude=no_rank or None)
    if preds is None:
        print(f"  [{label}] Training failed")
        return None

    print(f"  [{label}] Evaluating...")
    results = eval_per_window_hybrid(preds, regime_df, CANONICAL_EXEC_CFG, label=label)
    return results


def format_result_row(label: str, results: Dict, baseline_sharpe: float = None) -> str:
    """Format one experiment result as a table row."""
    if results is None:
        return f"| {label} | FAIL | | | | | |"

    r = results.get("ALL", {})
    sharpe = r.get("sharpe", 0)
    cost = r.get("total_cost_pct", 0)
    w1 = results.get("W1", {}).get("sharpe", 0)
    w2 = results.get("W2", {}).get("sharpe", 0)
    w3 = results.get("W3", {}).get("sharpe", 0)

    delta = ""
    decision = ""
    if baseline_sharpe is not None:
        d = sharpe - baseline_sharpe
        delta = f"{d:+.2f}"
        if d > 0.05:
            decision = "ACCEPT"
        elif d < -0.05:
            decision = "REJECT"
        else:
            decision = "neutral"
        # W2 veto check
        w2_base = 0  # will be set properly in main
        if results.get("_w2_drop", False):
            decision = "VETO (W2)"

    return (f"| {label} | {sharpe:+.2f} | {w1:+.2f} | {w2:+.2f} | "
            f"{w3:+.2f} | {cost:.1f}% | {delta} | {decision} |")


# ═════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="R56 WF Ablation")
    parser.add_argument("--importance-only", action="store_true",
                        help="Only run Phase 0 (feature importance)")
    parser.add_argument("--phase", type=int, default=None,
                        help="Run specific phase (0,1,2,3,4)")
    args = parser.parse_args()

    print("=" * 70)
    print("  R56 — WF Ablation: Feature Substitution Experiments")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────
    print("\n[LOAD] Building model frame with R55 features...")
    base_df, regime_df = load_research_frame()
    base_df, r35_added = add_r35_features(base_df)

    # Add existing CG features
    cg = load_cg_daily()
    cg_daily = compute_cg_features(cg)
    if not cg_daily.empty:
        base_df, cg_per_sym, cg_mkt = add_cg_features(base_df, cg_daily)

    # Add R55 features (basis, pos_ratio, disagreement, etc.)
    r55_feats = build_all_r55_features()
    df, r55_cols = merge_r55_into_model(base_df, r55_feats)

    print(f"\n  Model frame: {len(df):,} rows, {df['symbol'].nunique()} symbols")
    print(f"  Champion features: {len(CHAMPION_FEAT_31)} (31f = 30 base + cg_taker_imb)")
    print(f"  R55 features available: {r55_cols}")
    # Verify cg_taker_imb is present
    if "cg_taker_imb" not in df.columns:
        print("  ⚠️  cg_taker_imb missing! CG features may not have loaded correctly.")

    # Detect basis-available symbols
    basis_path = Path(__file__).resolve().parent / "data" / "raw" / "coinglass" / "basis.parquet"
    if basis_path.exists():
        basis_df = pd.read_parquet(basis_path, columns=["symbol"])
        basis_syms = set(basis_df["symbol"].unique())
        print(f"  Basis coverage: {len(basis_syms)}/35 symbols")
    else:
        basis_syms = set(SYM_35)
        print("  Basis data not found — using full universe")

    all_results = {}

    # ═════════════════════════════════════════════════════════
    #  PHASE 0: Feature Importance
    # ═════════════════════════════════════════════════════════
    if args.phase is None or args.phase == 0:
        print("\n" + "=" * 70)
        print("  PHASE 0 — Feature Importance (LGB gain)")
        print("=" * 70)

        imp = train_with_importance(df, CHAMPION_FEAT_31, WINDOWS)
        if not imp.empty:
            print("\n  Feature importance (sorted by avg gain):")
            print(imp.to_string(float_format="{:.0f}".format))

            # Identify weakest features
            weakest_3 = imp.tail(3).index.tolist()
            print(f"\n  3 weakest features: {weakest_3}")
            weakest = weakest_3[0]  # absolute weakest
            print(f"  → Will replace: '{weakest}' in experiment 2.1")
        else:
            weakest = "cs_rank_ma_5"  # fallback
            print(f"  Importance extraction failed, using fallback: {weakest}")

        if args.importance_only:
            print("\n  --importance-only flag set, stopping here.")
            return

    else:
        weakest = None  # will be skipped if not phase 0

    # ═════════════════════════════════════════════════════════
    #  PHASE 1: Baselines
    # ═════════════════════════════════════════════════════════
    if args.phase is None or args.phase == 1:
        print("\n" + "=" * 70)
        print("  PHASE 1 — Baselines")
        print("=" * 70)

        # 1.0: Full 35-coin baseline
        print("\n  [1.0] Baseline: Champion 31f on 35 coins (hybrid costs)")
        results_35 = run_wf_experiment(df, regime_df, CHAMPION_FEAT_31,
                                        "baseline_35")
        all_results["baseline_35"] = results_35

        # 1.1: 27-coin baseline (basis-available only)
        print("\n  [1.1] Coverage test: Champion 31f on 27 coins (basis universe)")
        results_27 = run_wf_experiment(df, regime_df, CHAMPION_FEAT_31,
                                        "baseline_27", universe_filter=basis_syms)
        all_results["baseline_27"] = results_27

        # Determine comparison baseline
        if results_35 and results_27:
            s35 = results_35["ALL"]["sharpe"]
            s27 = results_27["ALL"]["sharpe"]
            delta_coverage = abs(s35 - s27)
            print(f"\n  Baseline 35: ALL={s35:+.2f}")
            print(f"  Baseline 27: ALL={s27:+.2f}")
            print(f"  ΔCoverage: {delta_coverage:.2f}")
            if delta_coverage > 0.10:
                print("  ⚠️  Coverage difference > 0.10 → basis experiments "
                      "will compare against baseline_27")
            else:
                print("  ✓ Coverage difference small → compare against baseline_35")

    # ═════════════════════════════════════════════════════════
    #  PHASE 2: Basis Substitution
    # ═════════════════════════════════════════════════════════
    if args.phase is None or args.phase == 2:
        print("\n" + "=" * 70)
        print("  PHASE 2 — Basis Substitution Experiments")
        print("=" * 70)

        # Determine baseline for comparison
        if "baseline_35" in all_results and all_results["baseline_35"]:
            baseline = all_results["baseline_35"]
            baseline_label = "baseline_35"
        else:
            print("  Running baseline first...")
            baseline = run_wf_experiment(df, regime_df, CHAMPION_FEAT_31,
                                          "baseline_35")
            all_results["baseline_35"] = baseline
            baseline_label = "baseline_35"

        if baseline is None:
            print("  ✗ Baseline failed, cannot proceed")
            return

        baseline_sharpe = baseline["ALL"]["sharpe"]
        baseline_w2 = baseline.get("W2", {}).get("sharpe", 0)

        # 2.1: Weakest → cg_basis_z_60d
        if weakest is None:
            # Need to run importance first
            print("  Running importance to find weakest feature...")
            imp = train_with_importance(df, CHAMPION_FEAT_31, WINDOWS)
            weakest = imp.tail(1).index[0] if not imp.empty else "cs_rank_ma_5"
            print(f"  Weakest feature: {weakest}")

        if "cg_basis_z_60d" in df.columns:
            feats_2_1 = [("cg_basis_z_60d" if f == weakest else f)
                         for f in CHAMPION_FEAT_31]
            print(f"\n  [2.1] Replace '{weakest}' → 'cg_basis_z_60d' (31f)")
            results_2_1 = run_wf_experiment(df, regime_df, feats_2_1,
                                             f"2.1_{weakest}→basis_z")
            all_results["2.1"] = results_2_1

            # 2.2: cum_funding_24h → cg_basis_z_60d
            feats_2_2 = [("cg_basis_z_60d" if f == "cum_funding_24h" else f)
                         for f in CHAMPION_FEAT_31]
            print(f"\n  [2.2] Replace 'cum_funding_24h' → 'cg_basis_z_60d' (31f)")
            results_2_2 = run_wf_experiment(df, regime_df, feats_2_2,
                                             "2.2_funding→basis_z")
            all_results["2.2"] = results_2_2
        else:
            print("  ✗ cg_basis_z_60d not in dataframe!")

    # ═════════════════════════════════════════════════════════
    #  PHASE 3: OI/FR Substitution (conditional)
    # ═════════════════════════════════════════════════════════
    if args.phase is None or args.phase == 3:
        print("\n" + "=" * 70)
        print("  PHASE 3 — OI/FR Substitution Experiments (conditional)")
        print("=" * 70)

        # Check if Phase 2 had a winner
        phase2_winner = False
        for key in ["2.1", "2.2"]:
            if key in all_results and all_results[key]:
                r = all_results[key]
                if "baseline_35" in all_results and all_results["baseline_35"]:
                    delta = r["ALL"]["sharpe"] - all_results["baseline_35"]["ALL"]["sharpe"]
                    if delta > 0.05:
                        phase2_winner = True
                        print(f"  Phase 2 winner found ({key}: Δ={delta:+.2f}), "
                              f"Phase 3 runs for double-swap candidates")

        if not phase2_winner and args.phase is None:
            print("  Phase 2 had no winner → running Phase 3 (FR/OI substitutions)")

        # Get baseline
        if "baseline_35" not in all_results:
            baseline = run_wf_experiment(df, regime_df, CHAMPION_FEAT_31, "baseline_35")
            all_results["baseline_35"] = baseline

        # 3.1: cum_funding_24h → cg_fr_close
        if "cg_fr_close" in df.columns:
            feats_3_1 = [("cg_fr_close" if f == "cum_funding_24h" else f)
                         for f in CHAMPION_FEAT_31]
            print(f"\n  [3.1] Replace 'cum_funding_24h' → 'cg_fr_close'")
            results_3_1 = run_wf_experiment(df, regime_df, feats_3_1,
                                             "3.1_funding→fr_close")
            all_results["3.1"] = results_3_1

        # 3.2: cum_funding_24h → cg_fr_disagreement
        if "cg_fr_disagreement" in df.columns:
            feats_3_2 = [("cg_fr_disagreement" if f == "cum_funding_24h" else f)
                         for f in CHAMPION_FEAT_31]
            print(f"\n  [3.2] Replace 'cum_funding_24h' → 'cg_fr_disagreement'")
            results_3_2 = run_wf_experiment(df, regime_df, feats_3_2,
                                             "3.2_funding→fr_disagree")
            all_results["3.2"] = results_3_2

        # 3.3: oi_zscore → cg_oi_chg_1d
        if "cg_oi_chg_1d" in df.columns:
            feats_3_3 = [("cg_oi_chg_1d" if f == "oi_zscore" else f)
                         for f in CHAMPION_FEAT_31]
            print(f"\n  [3.3] Replace 'oi_zscore' → 'cg_oi_chg_1d'")
            results_3_3 = run_wf_experiment(df, regime_df, feats_3_3,
                                             "3.3_oizscore→oi_chg")
            all_results["3.3"] = results_3_3

    # ═════════════════════════════════════════════════════════
    #  PHASE 4: Double Substitution (conditional)
    # ═════════════════════════════════════════════════════════
    if args.phase is None or args.phase == 4:
        print("\n" + "=" * 70)
        print("  PHASE 4 — Double Substitution (conditional)")
        print("=" * 70)

        if "baseline_35" not in all_results or not all_results["baseline_35"]:
            print("  No baseline available, skipping")
        else:
            baseline_sharpe = all_results["baseline_35"]["ALL"]["sharpe"]

            # Find winners from Phase 2/3
            winners = []
            for key, res in all_results.items():
                if key.startswith(("2.", "3.")) and res:
                    delta = res["ALL"]["sharpe"] - baseline_sharpe
                    if delta > 0.05:
                        winners.append((key, delta))

            if len(winners) >= 2:
                print(f"  Found {len(winners)} winners: "
                      f"{[(k, f'{d:+.2f}') for k, d in winners]}")
                print("  Testing double substitution...")

                # Build combined feature set from top 2 winners
                w1_key, w2_key = winners[0][0], winners[1][0]
                # Need to figure out which substitutions the winners represent
                # For now, combine the best basis swap + best FR/OI swap
                # This requires knowing which experiment each winner was
                print(f"  Combining {w1_key} + {w2_key}")
                # Build combined features
                combined_feats = list(CHAMPION_FEAT_31)
                for key in [w1_key, w2_key]:
                    exp = EXPERIMENTS.get(key.replace(".", "_", 1), None)
                    if key == "2.1" and weakest:
                        old_f, new_f = weakest, "cg_basis_z_60d"
                    elif key == "2.2":
                        old_f, new_f = "cum_funding_24h", "cg_basis_z_60d"
                    elif key == "3.1":
                        old_f, new_f = "cum_funding_24h", "cg_fr_close"
                    elif key == "3.2":
                        old_f, new_f = "cum_funding_24h", "cg_fr_disagreement"
                    elif key == "3.3":
                        old_f, new_f = "oi_zscore", "cg_oi_chg_1d"
                    else:
                        continue
                    combined_feats = [(new_f if f == old_f else f)
                                      for f in combined_feats]

                # Check no duplicates
                if len(set(combined_feats)) == len(combined_feats):
                    print(f"\n  [4.1] Double swap: {w1_key}+{w2_key}")
                    results_4 = run_wf_experiment(df, regime_df, combined_feats,
                                                   f"4.1_double")
                    all_results["4.1"] = results_4
                else:
                    print("  ✗ Double swap has duplicate features (same target), skipping")
            elif len(winners) == 1:
                print(f"  Only 1 winner ({winners[0][0]}), no double swap needed")
            else:
                print("  No winners from Phase 2/3, skipping double swap")

    # ═════════════════════════════════════════════════════════
    #  RESULTS TABLE
    # ═════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  R56 RESULTS TABLE")
    print("=" * 70)

    baseline_sharpe = None
    if "baseline_35" in all_results and all_results["baseline_35"]:
        baseline_sharpe = all_results["baseline_35"]["ALL"]["sharpe"]
        baseline_w2 = all_results["baseline_35"].get("W2", {}).get("sharpe", 0)

    header = "| # | Experiment | ALL | W1 | W2 | W3 | Cost% | ΔSharpe | Decision |"
    sep = "|---|-----------|-----|----|----|----|----|---------|----------|"
    print(f"\n{header}")
    print(sep)

    # Print each result
    experiment_labels = [
        ("baseline_35", "Baseline 35-coin"),
        ("baseline_27", "Baseline 27-coin"),
        ("2.1", f"weakest→basis_z"),
        ("2.2", "cum_funding→basis_z"),
        ("3.1", "cum_funding→cg_fr_close"),
        ("3.2", "cum_funding→fr_disagree"),
        ("3.3", "oi_zscore→cg_oi_chg_1d"),
        ("4.1", "double swap"),
    ]

    for key, label in experiment_labels:
        if key not in all_results or all_results[key] is None:
            continue
        r = all_results[key]
        s_all = r["ALL"]["sharpe"]
        s_w1 = r.get("W1", {}).get("sharpe", 0)
        s_w2 = r.get("W2", {}).get("sharpe", 0)
        s_w3 = r.get("W3", {}).get("sharpe", 0)
        cost = r["ALL"].get("total_cost_pct", 0)

        if baseline_sharpe is not None and key not in ("baseline_35", "baseline_27"):
            delta = s_all - baseline_sharpe
            delta_str = f"{delta:+.2f}"
            # Decision logic
            if delta > 0.05:
                decision = "✅ ACCEPT"
            elif delta < -0.05:
                decision = "❌ REJECT"
            else:
                decision = "— neutral"
            # W2 veto
            if baseline_w2 and s_w2 < baseline_w2 - 0.30:
                decision = "🚫 VETO(W2)"
        else:
            delta_str = "—"
            decision = "baseline" if "baseline" in key else ""

        print(f"| {key} | {label} | {s_all:+.2f} | {s_w1:+.2f} | "
              f"{s_w2:+.2f} | {s_w3:+.2f} | {cost:.1f}% | {delta_str} | {decision} |")

    # Summary
    print(f"\n  Baseline ALL Sharpe: {baseline_sharpe}")
    winners = []
    for key, label in experiment_labels:
        if key in all_results and all_results[key] and baseline_sharpe:
            if key not in ("baseline_35", "baseline_27"):
                d = all_results[key]["ALL"]["sharpe"] - baseline_sharpe
                if d > 0.05:
                    winners.append((key, label, d))
    if winners:
        print(f"  Winners: {[(k, f'Δ={d:+.2f}') for k, _, d in winners]}")
    else:
        print("  No winners (all ΔSharpe ≤ 0.05)")

    print("\n" + "=" * 70)
    print("  R56 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
