#!/usr/bin/env python3
"""
R110 — Partial Neutralization Sweep (Numerai-style) over R68 predictions.

Idea: Remove unwanted exposure of R68 scores to risk/regime drivers via
cross-sectional ridge regression per timestamp, then blend.

Grid:
  - 3 exposure sets (SET1/SET2/SET3)
  - 4 lambda values (0, 1e-3, 1e-2, 1e-1)
  - 5 alpha values (0, 0.25, 0.5, 0.75, 1.0)
  = 60 combos + baseline (alpha=0 is always R68)

Pipeline: load_data → train_ensemble → neutralize → simulate → analyze → bootstrap
"""

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
RESULTS   = BASE_DIR / "results"
RESULTS.mkdir(exist_ok=True)

# ── Imports from R68 pipeline ───────────────────────────────────────────────
from _research_round7 import SYM_35
from _research_r22_models import SEEDS, log
from _research_r35_new_features import MARKET_LEVEL_FEATURES
from _research_r68_continuous_wf import (
    CHAMPION_FEAT_31, CONTINUOUS_WINDOWS, PROD_CFG,
    load_data, train_ensemble, simulate, sharpe, analyze,
)

# ── Exposure sets ───────────────────────────────────────────────────────────
EXPOSURE_SETS = {
    "SET1": ["btc_beta_168h", "ret_48h"],
    "SET2": ["btc_beta_168h", "ret_48h", "rel_volume_cs", "rvol_24h"],
    "SET3": ["btc_beta_168h", "ret_48h", "rel_volume_cs", "rvol_24h",
             "cum_funding_24h", "oi_velocity"],
}

LAMBDAS = [0, 1e-3, 1e-2, 1e-1]
ALPHAS  = [0.0, 0.25, 0.5, 0.75, 1.0]


def neutralize_predictions(
    preds: pd.DataFrame,
    exposures_df: pd.DataFrame,
    exposure_cols: List[str],
    lam: float,
) -> pd.DataFrame:
    """
    Cross-sectional ridge neutralization per timestamp.

    For each t:
      p = pred vector (N coins)
      X = exposure matrix (N x K), standardized cross-sectionally
      b = (X'X + λI)^{-1} X' p
      p_neut = p - X b

    Returns preds with new column 'pred_neut'.
    """
    # Merge exposures
    merged = preds.merge(
        exposures_df[["timestamp", "symbol"] + exposure_cols],
        on=["timestamp", "symbol"],
        how="left",
    )

    results = []
    for ts, grp in merged.groupby("timestamp"):
        p = grp["pred"].values.copy()
        X = grp[exposure_cols].values.copy()

        # Drop rows with NaN exposures
        valid = ~np.isnan(X).any(axis=1) & ~np.isnan(p)
        if valid.sum() < 5 or X.shape[1] == 0:
            grp = grp.copy()
            grp["pred_neut"] = p
            results.append(grp)
            continue

        p_v = p[valid]
        X_v = X[valid]

        # Standardize X cross-sectionally
        X_mean = X_v.mean(axis=0)
        X_std = X_v.std(axis=0) + 1e-10
        X_v = (X_v - X_mean) / X_std

        # Ridge regression
        K = X_v.shape[1]
        XtX = X_v.T @ X_v
        Xtp = X_v.T @ p_v
        try:
            b = np.linalg.solve(XtX + lam * np.eye(K), Xtp)
        except np.linalg.LinAlgError:
            b = np.zeros(K)

        p_neut = np.full_like(p, np.nan)
        p_neut[valid] = p_v - X_v @ b
        # For coins with missing exposure, keep original pred
        p_neut[~valid] = p[~valid]

        grp = grp.copy()
        grp["pred_neut"] = p_neut
        results.append(grp)

    out = pd.concat(results, ignore_index=True)
    return out


def blend_predictions(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """
    p_mix = (1 - alpha) * pred + alpha * pred_neut
    Then re-rank for simulate().
    """
    df = df.copy()
    df["pred_orig"] = df["pred"].copy()
    df["pred"] = (1 - alpha) * df["pred_orig"] + alpha * df["pred_neut"]
    return df


def block_bootstrap_sharpe(
    rets_base: pd.Series,
    rets_test: pd.Series,
    n_boot: int = 1000,
    block_size: int = 10,
    seed: int = 42,
) -> dict:
    """Block bootstrap comparison of Sharpe ratios."""
    rng = np.random.RandomState(seed)
    n = min(len(rets_base), len(rets_test))
    if n < 20:
        return {"p_sharpe_better": 0.5, "p_calmar_better": 0.5, "n": n}

    rb = rets_base.values[:n]
    rt = rets_test.values[:n]

    sharpe_diffs = []
    calmar_diffs = []

    for _ in range(n_boot):
        # Generate block bootstrap indices
        indices = []
        while len(indices) < n:
            start = rng.randint(0, n - block_size + 1)
            indices.extend(range(start, start + block_size))
        indices = indices[:n]

        rb_boot = rb[indices]
        rt_boot = rt[indices]

        # Sharpe
        sb = rb_boot.mean() / (rb_boot.std() + 1e-10) * np.sqrt(2 * 365)
        st = rt_boot.mean() / (rt_boot.std() + 1e-10) * np.sqrt(2 * 365)
        sharpe_diffs.append(st - sb)

        # Calmar (return / max dd)
        eq_b = np.cumprod(1 + rb_boot)
        eq_t = np.cumprod(1 + rt_boot)
        dd_b = (eq_b / np.maximum.accumulate(eq_b) - 1).min()
        dd_t = (eq_t / np.maximum.accumulate(eq_t) - 1).min()
        ret_b = eq_b[-1] / eq_b[0] - 1
        ret_t = eq_t[-1] / eq_t[0] - 1
        calmar_b = ret_b / (abs(dd_b) + 1e-10)
        calmar_t = ret_t / (abs(dd_t) + 1e-10)
        calmar_diffs.append(calmar_t - calmar_b)

    return {
        "p_sharpe_better": round(np.mean(np.array(sharpe_diffs) > 0), 3),
        "p_calmar_better": round(np.mean(np.array(calmar_diffs) > 0), 3),
        "sharpe_diff_mean": round(np.mean(sharpe_diffs), 4),
        "calmar_diff_mean": round(np.mean(calmar_diffs), 4),
        "n": n,
    }


def main():
    t0 = time.time()
    log("=" * 70)
    log("R110 — Partial Neutralization Sweep")
    log("=" * 70)

    # ── Load data ───────────────────────────────────────────────────────
    log("\nStep 0: Loading data + training R68 ensemble...")
    df, regime_df = load_data()

    feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    no_rank = [f for f in feats if f in MARKET_LEVEL_FEATURES]

    t1 = time.time()
    preds = train_ensemble(df, feats, CONTINUOUS_WINDOWS, seeds=SEEDS, cs_rank_exclude=no_rank)
    log(f"  Ensemble trained in {time.time()-t1:.0f}s, {len(preds):,} predictions")

    if preds is None or len(preds) == 0:
        log("ERROR: No predictions from train_ensemble")
        return

    # ── Prepare exposures ───────────────────────────────────────────────
    log("\nStep 1: Preparing exposures...")

    # Check which exposure columns exist
    all_exp_cols = set()
    for s, cols in EXPOSURE_SETS.items():
        avail = [c for c in cols if c in df.columns]
        missing = [c for c in cols if c not in df.columns]
        log(f"  {s}: {len(avail)}/{len(cols)} available. Missing: {missing}")
        all_exp_cols.update(avail)

    # Build per-coin per-timestamp exposure lookup from df
    exp_cols_list = list(all_exp_cols)
    exposures_df = df[["timestamp", "symbol"] + exp_cols_list].copy()

    # ── Baseline ────────────────────────────────────────────────────────
    log("\nStep 2: R68 Baseline (alpha=0)...")
    cfg_42 = {**PROD_CFG, "n_long": 4, "n_short": 2}
    port_base = simulate(preds, regime_df, 4, 2, cfg_42)
    m_base = analyze(port_base, "R68_baseline_4L2S")
    base_net_rets = port_base["net_ret"]

    # Calmar for baseline
    eq_base = (1 + port_base["net_ret"]).cumprod()
    dd_base = (eq_base / eq_base.cummax() - 1).min()
    ret_base = eq_base.iloc[-1] / eq_base.iloc[0] - 1
    calmar_base = ret_base / (abs(dd_base) + 1e-10)
    log(f"  Baseline: Sharpe={m_base.get('net_sharpe', 0):.3f}  "
        f"DD={m_base.get('max_dd_pct', 0):.1f}%  "
        f"Calmar={calmar_base:.3f}")

    # ── Grid sweep ──────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("Step 3: Neutralization Grid Sweep")
    log("=" * 70)

    grid_rows = []

    for set_name, exp_cols in EXPOSURE_SETS.items():
        avail_cols = [c for c in exp_cols if c in df.columns]
        if not avail_cols:
            log(f"\n  {set_name}: SKIP (no available exposures)")
            continue

        log(f"\n  === {set_name} ({len(avail_cols)} exposures: {avail_cols}) ===")

        for lam in LAMBDAS:
            # Neutralize once per lambda
            neut_preds = neutralize_predictions(preds, exposures_df, avail_cols, lam)

            for alpha in ALPHAS:
                label = f"{set_name}_l{lam}_a{alpha}"

                if alpha == 0.0:
                    # No blending — same as baseline
                    port = port_base
                    corr_orig = 1.0
                else:
                    blended = blend_predictions(neut_preds, alpha)
                    # Re-rank for simulate
                    sim_input = blended[["timestamp", "symbol", "pred", "fwd_ret", "window"]].copy()
                    port = simulate(sim_input, regime_df, 4, 2, cfg_42)

                    # Correlation with original predictions
                    corr_df = blended[["pred_orig", "pred"]].dropna()
                    corr_orig = corr_df["pred_orig"].corr(corr_df["pred"])

                if port.empty:
                    log(f"    {label}: EMPTY")
                    continue

                # Metrics
                net_s = sharpe(port["net_ret"])
                eq = (1 + port["net_ret"]).cumprod()
                dd = (eq / eq.cummax() - 1).min()
                total_ret = eq.iloc[-1] / eq.iloc[0] - 1
                calmar = total_ret / (abs(dd) + 1e-10)
                turnover = port["turnover"].mean()

                # Bootstrap vs baseline
                if alpha > 0:
                    boot = block_bootstrap_sharpe(base_net_rets, port["net_ret"])
                else:
                    boot = {"p_sharpe_better": 0.5, "p_calmar_better": 0.5,
                            "sharpe_diff_mean": 0, "calmar_diff_mean": 0, "n": len(port)}

                # Acceptance
                base_sharpe = m_base.get("net_sharpe", 0)
                pass_a = (net_s >= base_sharpe + 0.05) and boot["p_sharpe_better"] > 0.80
                pass_b = (net_s >= base_sharpe - 0.05 and calmar >= calmar_base * 1.05
                          and boot["p_calmar_better"] > 0.80)

                row = {
                    "set": set_name,
                    "lambda": lam,
                    "alpha": alpha,
                    "net_sharpe": round(net_s, 3),
                    "total_ret_pct": round(total_ret * 100, 1),
                    "max_dd_pct": round(dd * 100, 1),
                    "calmar": round(calmar, 3),
                    "turnover": round(turnover, 2),
                    "corr_orig": round(corr_orig, 4),
                    "p_sharpe_better": boot["p_sharpe_better"],
                    "p_calmar_better": boot["p_calmar_better"],
                    "sharpe_diff": boot["sharpe_diff_mean"],
                    "pass_a": pass_a,
                    "pass_b": pass_b,
                    "n_periods": len(port),
                }
                grid_rows.append(row)

                flag = ""
                if pass_a: flag = " ✅ PASS-A"
                elif pass_b: flag = " ✅ PASS-B"

                log(f"    {label:>25s}: Sh={net_s:.3f}  DD={dd*100:.1f}%  "
                    f"Calmar={calmar:.2f}  Corr={corr_orig:.3f}  "
                    f"P(Sh↑)={boot['p_sharpe_better']:.2f}  "
                    f"P(Cal↑)={boot['p_calmar_better']:.2f}{flag}")

    # ── Save results ────────────────────────────────────────────────────
    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(RESULTS / "r110_grid.csv", index=False)
    log(f"\n  Saved grid → results/r110_grid.csv ({len(grid_df)} rows)")

    # ── Summary ─────────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)

    log(f"\n  Baseline: Sharpe={m_base.get('net_sharpe', 0):.3f}  "
        f"DD={m_base.get('max_dd_pct', 0):.1f}%  Calmar={calmar_base:.3f}")

    if len(grid_df) > 0:
        non_base = grid_df[grid_df["alpha"] > 0]
        if len(non_base) > 0:
            best_sharpe = non_base.loc[non_base["net_sharpe"].idxmax()]
            best_calmar = non_base.loc[non_base["calmar"].idxmax()]
            log(f"\n  Best Sharpe: {best_sharpe['set']}/λ={best_sharpe['lambda']}/α={best_sharpe['alpha']} "
                f"→ Sh={best_sharpe['net_sharpe']:.3f} (Δ={best_sharpe['net_sharpe'] - m_base.get('net_sharpe', 0):+.3f})")
            log(f"  Best Calmar: {best_calmar['set']}/λ={best_calmar['lambda']}/α={best_calmar['alpha']} "
                f"→ Cal={best_calmar['calmar']:.3f} (baseline={calmar_base:.3f})")

        n_pass_a = grid_df["pass_a"].sum()
        n_pass_b = grid_df["pass_b"].sum()
        log(f"\n  PASS-A (Sharpe uplift): {n_pass_a}/{len(non_base)}")
        log(f"  PASS-B (Risk uplift):   {n_pass_b}/{len(non_base)}")

        if n_pass_a > 0:
            log("  VERDICT: ✅ PASS-A — neutralization improves Sharpe")
        elif n_pass_b > 0:
            log("  VERDICT: ✅ PASS-B — neutralization improves risk-adjusted")
        else:
            log("  VERDICT: ❌ FAIL — neutralization does not improve R68")

    # Summary JSON
    summary = {
        "experiment": "R110",
        "baseline_sharpe": m_base.get("net_sharpe", 0),
        "baseline_calmar": round(calmar_base, 3),
        "n_combos": len(grid_df),
        "n_pass_a": int(grid_df["pass_a"].sum()) if len(grid_df) > 0 else 0,
        "n_pass_b": int(grid_df["pass_b"].sum()) if len(grid_df) > 0 else 0,
    }

    if len(grid_df) > 0 and len(non_base) > 0:
        best = non_base.loc[non_base["net_sharpe"].idxmax()]
        summary["best_config"] = {
            "set": best["set"], "lambda": best["lambda"], "alpha": best["alpha"],
            "net_sharpe": best["net_sharpe"], "calmar": best["calmar"],
        }

    with open(RESULTS / "r110_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t0
    log(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    log("Done.")


if __name__ == "__main__":
    main()
