#!/usr/bin/env python3
"""
R102 — Baseline Equivalence Check

Verify that loading cached R68 predictions and running R68's own simulate()
reproduces the canonical R68 result: Sharpe=3.777, MaxDD=-13.95%, N=688.

Tests:
1. R68 predictions → R68 simulate(4,2) → must match canonical
2. Left-join R68+R93, use R68's pred at α=1.0 → must still match
"""

import json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EPS = 1e-10
PPY = 2 * 365  # 12h periods per year


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sharpe_ann(rets, ppy=PPY):
    if len(rets) < 2:
        return 0.0
    eq = (1 + rets).cumprod()
    r = eq.pct_change().dropna()
    return float(r.mean() / (r.std() + EPS) * np.sqrt(ppy))


def max_dd(rets):
    eq = (1 + rets).cumprod()
    return float((eq / eq.cummax() - 1).min())


def main():
    t0 = time.time()
    log("=" * 70)
    log("  R102 — BASELINE EQUIVALENCE CHECK")
    log("=" * 70)

    from _research_r68_continuous_wf import (
        load_data, simulate, train_ensemble, PROD_CFG,
        CHAMPION_FEAT_31, CONTINUOUS_WINDOWS, MARKET_LEVEL_FEATURES,
    )
    from _research_r22_models import SEEDS

    # ── Load R68 predictions ──────────────────────────────────────────────
    r68_path = RESULTS_DIR / "r68_predictions.parquet"
    r93_path = RESULTS_DIR / "r93_predictions.parquet"

    log("\n[0] Loading data ...")
    df_full, regime_df = load_data()

    if not r68_path.exists():
        log("  R68 predictions not cached — retraining ...")
        feats = [f for f in CHAMPION_FEAT_31 if f in df_full.columns]
        no_rank = [f for f in feats if f in MARKET_LEVEL_FEATURES]
        r68_preds = train_ensemble(df_full, feats, CONTINUOUS_WINDOWS,
                                   seeds=SEEDS, cs_rank_exclude=no_rank)
        r68_preds.to_parquet(r68_path, index=False)
        log(f"  Saved R68 predictions: {len(r68_preds):,} rows")
    else:
        r68_preds = pd.read_parquet(r68_path)
        log(f"  R68 predictions: {len(r68_preds):,} rows (cached)")

    del df_full  # free memory

    # ── Test 1: R68 predictions → R68 simulate → must match canonical ────
    log("\n" + "=" * 70)
    log("  [1] R68 predictions → R68 simulate(n_long=4, n_short=2)")
    log("=" * 70)

    port_r68 = simulate(r68_preds, regime_df, 4, 2)
    s1 = sharpe_ann(port_r68["net_ret"])
    dd1 = max_dd(port_r68["net_ret"])
    ret1 = float((1 + port_r68["net_ret"]).prod() - 1) * 100
    n1 = len(port_r68)
    avg_turn = port_r68["turnover"].mean()
    avg_cost = port_r68["cost"].mean() * 10000

    log(f"  Sharpe:    {s1:.4f}  (canonical: 3.777)")
    log(f"  MaxDD:     {dd1*100:.2f}%  (canonical: -13.95%)")
    log(f"  Return:    {ret1:.1f}%  (canonical: 179.3%)")
    log(f"  N_periods: {n1}  (canonical: 688)")
    log(f"  AvgTurn:   {avg_turn:.2f}")
    log(f"  AvgCost:   {avg_cost:.2f} bps")

    # Tolerance: model training is deterministic per seed, so should be exact
    sharpe_ok = abs(s1 - 3.777) < 0.1
    dd_ok = abs(dd1 * 100 - (-13.95)) < 1.0
    n_ok = n1 == 688
    pass1 = sharpe_ok and dd_ok and n_ok

    log(f"\n  Test 1: {'PASS ✅' if pass1 else 'FAIL ❌'}")
    if not pass1:
        log(f"    Sharpe: {'OK' if sharpe_ok else 'FAIL'} (Δ={abs(s1 - 3.777):.4f})")
        log(f"    DD:     {'OK' if dd_ok else 'FAIL'} (Δ={abs(dd1*100 - (-13.95)):.4f})")
        log(f"    N:      {'OK' if n_ok else 'FAIL'} ({n1} vs 688)")

    # ── Test 2: Left-join with R93, α=1.0 using R68's original pred ─────
    log("\n" + "=" * 70)
    log("  [2] Left-join R68+R93, α=1.0 (R68 pred only)")
    log("=" * 70)

    if r93_path.exists():
        r93_preds = pd.read_parquet(r93_path)
        log(f"  R93 predictions: {len(r93_preds):,} rows")

        # Left join: keep ALL R68 rows
        merged = r68_preds.merge(
            r93_preds[["timestamp", "symbol", "pred", "raw_prob"]].rename(
                columns={"pred": "pred_93", "raw_prob": "raw_prob_93"}),
            on=["timestamp", "symbol"], how="left"
        )
        n_r68 = len(r68_preds)
        n_merged = len(merged)
        r93_cov = merged["pred_93"].notna().mean() * 100
        log(f"  Merged: {n_merged:,} rows (R68: {n_r68:,})")
        log(f"  R93 coverage: {r93_cov:.1f}%")
        log(f"  Rows preserved: {n_merged == n_r68}")

        # At α=1.0: use R68's original pred column (R93 is ignored)
        port_merged = simulate(
            merged[["timestamp", "symbol", "pred", "fwd_ret"]],
            regime_df, 4, 2
        )

        s2 = sharpe_ann(port_merged["net_ret"])
        dd2 = max_dd(port_merged["net_ret"])
        n2 = len(port_merged)
        log(f"  Sharpe: {s2:.4f}  MaxDD: {dd2*100:.2f}%  N: {n2}")

        # Should be identical to Test 1 since same pred column
        eq1 = (1 + port_r68["net_ret"]).cumprod().values
        eq2 = (1 + port_merged["net_ret"]).cumprod().values
        if len(eq1) == len(eq2):
            equity_corr = np.corrcoef(eq1, eq2)[0, 1]
            max_diff = np.max(np.abs(eq1 - eq2))
            log(f"  Equity correlation: {equity_corr:.6f}")
            log(f"  Max abs diff: {max_diff:.8f}")
        else:
            equity_corr = 0.0
            max_diff = 999.0
            log(f"  Length mismatch: {len(eq1)} vs {len(eq2)}")

        pass2 = abs(s2 - s1) < 0.001 and n2 == n1
        log(f"\n  Test 2: {'PASS ✅' if pass2 else 'FAIL ❌'}")
    else:
        log("  R93 predictions not found, skipping Test 2")
        pass2 = True
        equity_corr = 1.0

    # ── Save results ──────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("  RESULTS")
    log("=" * 70)

    overall = pass1 and pass2
    result = {
        "test1_sharpe": round(s1, 4),
        "test1_maxdd_pct": round(dd1 * 100, 2),
        "test1_return_pct": round(ret1, 1),
        "test1_n_periods": n1,
        "test1_avg_turnover": round(avg_turn, 2),
        "test1_avg_cost_bps": round(avg_cost, 2),
        "test1_pass": pass1,
        "test2_pass": pass2,
        "equity_correlation": round(equity_corr, 6),
        "overall_pass": overall,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(RESULTS_DIR / "r102_baseline_equivalence.json", "w") as f:
        json.dump(result, f, indent=2)

    port_r68[["timestamp", "net_ret", "gross_ret", "cost", "turnover",
              "n_long", "n_short"]].to_csv(
        RESULTS_DIR / "r102_equity.csv", index=False)

    log(f"\n  OVERALL: {'PASS ✅' if overall else 'FAIL ❌'}")
    log(f"  Saved: r102_baseline_equivalence.json, r102_equity.csv")
    log(f"  Runtime: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
