#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R48 Phase 1+2 — Taker Derivatives + Residualized Liquidations

Phase 1 (Taker imbalance derivatives):
  1.1  cg_taker_imb_ma3     — rolling 3d mean (smoother)
  1.2  cg_taker_imb_delta   — 1d change (acceleration)
  1.3  cg_taker_imb_cs_demean — cross-sectional demean (pure CS component)

Phase 2 (Residualized liquidations):
  2.1  cg_liq_imb_resid_bin    — bin-residual (decile of ret_48h)
  2.2  cg_liq_imb_resid_roll   — rolling regression residual (90d beta)
  2.3  mkt_cg_liq_imb_resid    — market-level residualized

Usage:
  python _research_r48_features.py          # full Phase 1+2
  python _research_r48_features.py --quick  # BTC/ETH/SOL only
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── project imports ───────────────────────────────────────────

from _research_round7 import WINDOWS, SYM_35
from _research_r30b_fixed import (
    compute_regime_extended,
    eval_with_costs,
    simulate_with_costs,
    train_ensemble,
)
from _research_r33_creative_features import FEAT_28
from _research_r35_new_features import (
    MARKET_LEVEL_FEATURES,
    add_r35_features,
    load_research_frame,
)
from _research_r47_coinglass import (
    CANONICAL_EXEC_CFG,
    CHAMPION_FEAT_30,
    CG_DIR,
    add_cg_features,
    compute_cg_features,
    compute_ic_by_period,
    load_cg_daily,
    make_feature_set,
    run_ic_scan,
)

# ── config ─────────────────────────────────────────────────────

# Champion is now 31f = FEAT_30 + cg_taker_imb
CHAMPION_FEAT_31 = CHAMPION_FEAT_30 + ["cg_taker_imb"]


# ═══════════════════════════════════════════════════════════════
#  Phase 1: Taker imbalance derivatives
# ═══════════════════════════════════════════════════════════════

def add_taker_derivatives(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Add 3 derivative features of cg_taker_imb."""
    out = df.copy()
    new_cols = []

    # 1.1 — Rolling 3-day mean
    out["cg_taker_imb_ma3"] = (
        out.groupby("symbol")["cg_taker_imb"]
        .transform(lambda x: x.rolling(3 * 2, min_periods=2).mean())  # 3d × 2 bars/day
    )
    new_cols.append("cg_taker_imb_ma3")

    # 1.2 — 1-day delta (acceleration)
    out["cg_taker_imb_delta"] = (
        out.groupby("symbol")["cg_taker_imb"]
        .transform(lambda x: x.diff(2))  # 2 bars = 1 day
    )
    new_cols.append("cg_taker_imb_delta")

    # 1.3 — Cross-sectional demean
    cs_mean = out.groupby("timestamp")["cg_taker_imb"].transform("mean")
    out["cg_taker_imb_cs_demean"] = out["cg_taker_imb"] - cs_mean
    new_cols.append("cg_taker_imb_cs_demean")

    for c in new_cols:
        nna = out[c].notna().mean() * 100
        print(f"    {c}: {nna:.1f}% non-null")

    return out, new_cols


# ═══════════════════════════════════════════════════════════════
#  Phase 2: Residualized liquidations
# ═══════════════════════════════════════════════════════════════

def add_residualized_liq(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Residualize cg_liq_imbalance against ret_48h (r=-0.57 from R47).
    Three methods: bin-residual, rolling regression, market-level.
    """
    out = df.copy()
    new_cols = []

    if "cg_liq_imbalance" not in out.columns:
        print("    ⚠️  cg_liq_imbalance not in DataFrame — skipping Phase 2")
        return out, new_cols

    # 2.1 — Bin-residual (decile of ret_48h)
    if "ret_48h" in out.columns:
        out["_ret_48h_decile"] = out.groupby("timestamp")["ret_48h"].transform(
            lambda x: pd.qcut(x, 10, labels=False, duplicates="drop")
        )
        out["_liq_imb_bin_mean"] = out.groupby(["symbol", "_ret_48h_decile"])[
            "cg_liq_imbalance"
        ].transform("mean")
        out["cg_liq_imb_resid_bin"] = out["cg_liq_imbalance"] - out["_liq_imb_bin_mean"]
        out.drop(columns=["_ret_48h_decile", "_liq_imb_bin_mean"], inplace=True)
        new_cols.append("cg_liq_imb_resid_bin")
        print(f"    cg_liq_imb_resid_bin: {out['cg_liq_imb_resid_bin'].notna().mean()*100:.1f}% non-null")
    else:
        print("    ⚠️  ret_48h not available — skipping bin-residual")

    # 2.2 — Rolling regression residual (90d beta per symbol)
    def _rolling_resid(grp: pd.DataFrame) -> pd.Series:
        """Rolling 90-day OLS: liq_imb = a + b*ret_48h + eps → return eps."""
        y = grp["cg_liq_imbalance"].values
        x = grp["ret_48h"].values if "ret_48h" in grp.columns else np.zeros(len(y))
        n = len(y)
        window = 180  # 90 days × 2 bars/day = 180 bars
        resid = np.full(n, np.nan)
        for i in range(window, n):
            yy = y[i - window:i + 1]
            xx = x[i - window:i + 1]
            mask = ~(np.isnan(yy) | np.isnan(xx))
            if mask.sum() < 30:
                continue
            yy_ = yy[mask]
            xx_ = xx[mask]
            xx_aug = np.column_stack([np.ones(len(xx_)), xx_])
            try:
                beta = np.linalg.lstsq(xx_aug, yy_, rcond=None)[0]
                pred = xx_aug[-1] @ beta
                resid[i] = yy_[-1] - pred
            except Exception:
                continue
        return pd.Series(resid, index=grp.index)

    if "ret_48h" in out.columns:
        print("    Computing rolling regression residual (slow) ...")
        out["cg_liq_imb_resid_roll"] = (
            out.groupby("symbol", group_keys=False)
            .apply(_rolling_resid)
        )
        new_cols.append("cg_liq_imb_resid_roll")
        print(f"    cg_liq_imb_resid_roll: {out['cg_liq_imb_resid_roll'].notna().mean()*100:.1f}% non-null")

    # 2.3 — Market-level residualized (cross-sectional mean of residuals)
    if "cg_liq_imb_resid_bin" in out.columns:
        out["mkt_cg_liq_imb_resid"] = out.groupby("timestamp")[
            "cg_liq_imb_resid_bin"
        ].transform("mean")
        new_cols.append("mkt_cg_liq_imb_resid")
        print(f"    mkt_cg_liq_imb_resid: {out['mkt_cg_liq_imb_resid'].notna().mean()*100:.1f}% non-null")

    return out, new_cols


# ═══════════════════════════════════════════════════════════════
#  WF ablation engine
# ═══════════════════════════════════════════════════════════════

def run_ablation(df: pd.DataFrame, regime_df: pd.DataFrame,
                 candidate_features: List[str],
                 mkt_cols: List[str],
                 baseline_feats: List[str],
                 baseline_label: str = "champion_31f") -> pd.DataFrame:
    """
    WF ablation: baseline + each candidate feature individually,
    then best pair if any improve.
    """
    rows = []
    no_rank_market = list(MARKET_LEVEL_FEATURES)

    # Baseline
    print(f"\n  [baseline] {baseline_label} ...")
    bl_feats = list(baseline_feats)
    bl_no_rank = [f for f in bl_feats if f in no_rank_market or f in mkt_cols]
    preds = train_ensemble(df, bl_feats, WINDOWS, l2=1.0, rolling=False,
                           label=baseline_label, cs_rank_exclude=bl_no_rank)
    if preds is not None and not preds.empty:
        bl_results = _evaluate(preds, regime_df)
        row = _make_row(baseline_label, [], bl_results)
        rows.append(row)
        _print_row(row)
    else:
        print("  ⚠️  Baseline failed!")
        return pd.DataFrame()

    baseline_all = row["ALL_sh"]

    # Singles
    winners = []
    for feat in candidate_features:
        if feat not in df.columns:
            print(f"  ⚠️  {feat} not in DataFrame — skipping")
            continue
        label = f"+{feat[-25:]}"
        print(f"\n  [{feat}] ...")
        feats = list(baseline_feats)
        if feat not in feats:
            feats.append(feat)
        no_rank = [f for f in feats if f in no_rank_market or f in mkt_cols]
        preds = train_ensemble(df, feats, WINDOWS, l2=1.0, rolling=False,
                               label=label, cs_rank_exclude=no_rank)
        if preds is None or preds.empty:
            print(f"    ⚠️  {feat}: no predictions")
            continue
        results = _evaluate(preds, regime_df)
        row = _make_row(label, [feat], results)
        row["delta_all"] = row["ALL_sh"] - baseline_all
        rows.append(row)
        _print_row(row)
        if row["ALL_sh"] > baseline_all:
            winners.append(feat)

    # Best pair from winners
    if len(winners) >= 2:
        from itertools import combinations
        for a, b in combinations(winners[:4], 2):
            label = f"+{a[-12:]}+{b[-12:]}"
            print(f"\n  [{a} + {b}] ...")
            feats = list(baseline_feats) + [f for f in [a, b] if f not in baseline_feats]
            no_rank = [f for f in feats if f in no_rank_market or f in mkt_cols]
            preds = train_ensemble(df, feats, WINDOWS, l2=1.0, rolling=False,
                                   label=label, cs_rank_exclude=no_rank)
            if preds is None or preds.empty:
                continue
            results = _evaluate(preds, regime_df)
            row = _make_row(label, [a, b], results)
            row["delta_all"] = row["ALL_sh"] - baseline_all
            rows.append(row)
            _print_row(row)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values("ALL_sh", ascending=False).reset_index(drop=True)
    return summary


def _evaluate(preds, regime_df):
    out = {}
    for window in ["W1", "W2", "W3", "ALL"]:
        subset = preds if window == "ALL" else preds[preds["window"] == window]
        port = simulate_with_costs(subset, regime_df, CANONICAL_EXEC_CFG)
        out[window] = eval_with_costs(port, window)
    return out


def _make_row(config, extra_feats, results):
    row = {"config": config, "extra_feats": "|".join(extra_feats)}
    for window in ["W1", "W2", "W3", "ALL"]:
        m = results[window]
        row[f"{window}_sh"] = m.get("sharpe", 0.0)
        row[f"{window}_sh_gr"] = m.get("sharpe_gross", 0.0)
        row[f"{window}_dd"] = m.get("max_dd_pct", 0.0)
        row[f"{window}_cost"] = m.get("total_cost_pct", 0.0)
        row[f"{window}_turn"] = m.get("avg_turnover", 0.0)
    return row


def _print_row(row):
    w = (row.get("W1_sh", 0), row.get("W2_sh", 0),
         row.get("W3_sh", 0), row.get("ALL_sh", 0))
    cost = row.get("ALL_cost", 0)
    delta = row.get("delta_all", 0)
    delta_str = f"Δ{delta:+.2f}" if delta else ""
    print(f"    W1={w[0]:+.2f}  W2={w[1]:+.2f}  W3={w[2]:+.2f}  ALL={w[3]:+.2f}  "
          f"cost={cost:.1f}%  {delta_str}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main(quick: bool = False) -> None:
    print("=" * 80)
    print("R48 Phase 1+2 — TAKER DERIVATIVES + RESIDUALIZED LIQUIDATIONS")
    print("=" * 80)

    # ── Load CG daily features ──────────────────────────────────
    print("\n[1] Loading CoinGlass daily data ...")
    cg = load_cg_daily()
    cg_feats_daily = compute_cg_features(cg)
    if cg_feats_daily.empty:
        print("❌ No CG features — aborting")
        return

    # ── Load research frame ──────────────────────────────────────
    print("\n[2] Loading research frame ...")
    df, regime_df = load_research_frame()
    df, _ = add_r35_features(df)
    regime_df = regime_df.sort_index()
    print(f"  Base frame: {len(df):,} rows × {len(df.columns)} cols")

    if quick:
        print("  ⚡ Quick mode: BTC/ETH/SOL only ...")
        df = df[df["symbol"].isin(["BTC/USDT", "ETH/USDT", "SOL/USDT"])].copy()

    # ── Merge CG features ───────────────────────────────────────
    print("\n[3] Merging CG features ...")
    df, per_sym_cols, mkt_cols = add_cg_features(df, cg_feats_daily)
    print(f"  CG features: {len(per_sym_cols)} per-sym + {len(mkt_cols)} mkt")

    # ── Phase 1: Taker derivatives ──────────────────────────────
    print("\n" + "=" * 70)
    print("  Phase 1 — Taker Imbalance Derivatives")
    print("=" * 70)

    print("\n  Adding derivative features ...")
    df, taker_deriv_cols = add_taker_derivatives(df)

    # IC scan for new features (TRAIN data only)
    print("\n  IC Scan on derivatives ...")
    ic_df = run_ic_scan(df, taker_deriv_cols, WINDOWS)
    if not ic_df.empty:
        print(f"\n  {'Feature':<30} {'mean_IC':>8} {'|IC|':>6} {'ICIR':>7}")
        print(f"  {'─'*30} {'─'*8} {'─'*6} {'─'*7}")
        for _, r in ic_df.iterrows():
            print(f"  {r['feature']:<30} {r['mean_ic']:>+7.3f} {r['abs_ic']:>5.3f} "
                  f"{r['mean_icir']:>+6.2f}")

    # WF ablation: champion_31f + each derivative
    print("\n  WF Ablation (champion_31f + derivative) ...")
    phase1_summary = run_ablation(
        df, regime_df, taker_deriv_cols, mkt_cols,
        baseline_feats=CHAMPION_FEAT_31,
        baseline_label="champion_31f",
    )

    if not phase1_summary.empty:
        print("\n  Phase 1 Summary:")
        print(phase1_summary[["config", "W1_sh", "W2_sh", "W3_sh",
                               "ALL_sh", "ALL_cost"]].to_string(index=False))
        phase1_summary.to_csv("results_r48_phase1_summary.csv", index=False)
        print("  → Saved results_r48_phase1_summary.csv")

    # ── Phase 2: Residualized liquidations ──────────────────────
    print("\n" + "=" * 70)
    print("  Phase 2 — Residualized Liquidations")
    print("=" * 70)

    print("\n  Computing residualized features ...")
    df, resid_cols = add_residualized_liq(df)

    if not resid_cols:
        print("  ⚠️  No residualized features created — skipping Phase 2 WF")
    else:
        # IC scan
        print("\n  IC Scan on residualized features ...")
        ic_df2 = run_ic_scan(df, resid_cols, WINDOWS)
        if not ic_df2.empty:
            print(f"\n  {'Feature':<30} {'mean_IC':>8} {'|IC|':>6} {'ICIR':>7}")
            print(f"  {'─'*30} {'─'*8} {'─'*6} {'─'*7}")
            for _, r in ic_df2.iterrows():
                print(f"  {r['feature']:<30} {r['mean_ic']:>+7.3f} {r['abs_ic']:>5.3f} "
                      f"{r['mean_icir']:>+6.2f}")

        # Determine mkt_cols for ablation
        all_mkt = list(mkt_cols)
        if "mkt_cg_liq_imb_resid" in resid_cols:
            all_mkt.append("mkt_cg_liq_imb_resid")

        # WF ablation: champion_31f + each residualized feature
        print("\n  WF Ablation (champion_31f + residualized) ...")
        phase2_summary = run_ablation(
            df, regime_df, resid_cols, all_mkt,
            baseline_feats=CHAMPION_FEAT_31,
            baseline_label="champion_31f",
        )

        if not phase2_summary.empty:
            print("\n  Phase 2 Summary:")
            print(phase2_summary[["config", "W1_sh", "W2_sh", "W3_sh",
                                   "ALL_sh", "ALL_cost"]].to_string(index=False))
            phase2_summary.to_csv("results_r48_phase2_summary.csv", index=False)
            print("  → Saved results_r48_phase2_summary.csv")

    # ── Combined summary ────────────────────────────────────────
    print("\n" + "=" * 80)
    print("R48 Phase 1+2 — COMPLETE")
    print("=" * 80)

    # Find best features across both phases
    best_p1 = None
    if not phase1_summary.empty:
        p1_winners = phase1_summary[
            (phase1_summary["config"] != "champion_31f") &
            (phase1_summary["ALL_sh"] > phase1_summary.loc[
                phase1_summary["config"] == "champion_31f", "ALL_sh"].iloc[0])
        ]
        if not p1_winners.empty:
            best_p1 = p1_winners.iloc[0]
            print(f"\n  Phase 1 best: {best_p1['config']} → ALL={best_p1['ALL_sh']:.2f}")
        else:
            print("\n  Phase 1: no improvement over baseline")

    best_p2 = None
    if resid_cols and not phase2_summary.empty:
        p2_winners = phase2_summary[
            (phase2_summary["config"] != "champion_31f") &
            (phase2_summary["ALL_sh"] > phase2_summary.loc[
                phase2_summary["config"] == "champion_31f", "ALL_sh"].iloc[0])
        ]
        if not p2_winners.empty:
            best_p2 = p2_winners.iloc[0]
            print(f"  Phase 2 best: {best_p2['config']} → ALL={best_p2['ALL_sh']:.2f}")
        else:
            print("  Phase 2: no improvement over baseline")

    # Save winners info for Phase 4 combo script
    winners = {}
    if best_p1 is not None:
        winners["phase1"] = best_p1["extra_feats"]
    if best_p2 is not None:
        winners["phase2"] = best_p2["extra_feats"]
    if winners:
        import json
        with open("results_r48_phase12_winners.json", "w") as f:
            json.dump(winners, f, indent=2)
        print(f"\n  → Saved results_r48_phase12_winners.json: {winners}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    main(quick=args.quick)
