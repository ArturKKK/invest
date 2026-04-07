#!/usr/bin/env python3
"""
R103 — Logit Ensemble: combine R68 + R93 in logit-space.

Instead of rank_cs(p) which caused baseline drift in R100,
combine in logit-space:
    s68 = logit(clip(raw_prob_68))
    s93 = logit(clip(raw_prob_93))
    s = α * s68 + (1-α) * s93

Uses R68's actual simulate() function for all configs.
Grid: α ∈ {0.0, 0.25, 0.50, 0.75, 1.0}
Bootstrap: block=10, N=1000
"""

import json, sys, time, warnings
from pathlib import Path
from typing import Dict, Set

import numpy as np
import pandas as pd
from scipy.special import logit as scipy_logit

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EPS = 1e-10
PPY = 2 * 365
CLIP_LO, CLIP_HI = 1e-6, 1 - 1e-6


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


def calmar(rets, ppy=PPY):
    s = sharpe_ann(rets, ppy)
    dd = abs(max_dd(rets))
    return s / (dd + EPS) if dd > 0 else 0.0


def block_bootstrap(rets_base, rets_exp, n_boot=1000, block=10, seed=42):
    """Block bootstrap comparing two return series."""
    rng = np.random.default_rng(seed)
    n = min(len(rets_base), len(rets_exp))
    rb = np.array(rets_base[:n], dtype=float)
    re = np.array(rets_exp[:n], dtype=float)
    n_blocks = max(1, n // block)

    def _sh(r):
        if len(r) < 2:
            return 0.0
        eq = (1 + r).cumprod()
        p = np.diff(eq) / eq[:-1]
        if len(p) < 2 or np.std(p) < EPS:
            return 0.0
        return float(np.mean(p) / np.std(p) * np.sqrt(PPY))

    def _calmar(r):
        s = _sh(r)
        eq = (1 + r).cumprod()
        dd = float(np.min(eq / np.maximum.accumulate(eq) - 1))
        return s / (abs(dd) + EPS) if abs(dd) > 0 else 0.0

    sb, se, cb, ce = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, max(1, n - block + 1), size=n_blocks)
        block_idx = np.concatenate([np.arange(i, min(i + block, n)) for i in idx])[:n]
        sb.append(_sh(rb[block_idx]))
        se.append(_sh(re[block_idx]))
        cb.append(_calmar(rb[block_idx]))
        ce.append(_calmar(re[block_idx]))

    sb, se = np.array(sb), np.array(se)
    cb, ce = np.array(cb), np.array(ce)
    ds = se - sb
    dc = ce - cb

    return {
        "p_sharpe_better": round(float((se > sb).mean()), 3),
        "median_delta_sharpe": round(float(np.median(ds)), 4),
        "p5_delta_sharpe": round(float(np.percentile(ds, 5)), 4),
        "p95_delta_sharpe": round(float(np.percentile(ds, 95)), 4),
        "base_sharpe_med": round(float(np.median(sb)), 4),
        "exp_sharpe_med": round(float(np.median(se)), 4),
        "p_calmar_better": round(float((ce > cb).mean()), 3),
        "median_delta_calmar": round(float(np.median(dc)), 4),
    }


def main():
    t0 = time.time()
    log("=" * 70)
    log("  R103 — LOGIT ENSEMBLE (R68 + R93)")
    log("=" * 70)

    from _research_r68_continuous_wf import (
        load_data, simulate, train_ensemble, PROD_CFG,
        CHAMPION_FEAT_31, CONTINUOUS_WINDOWS, MARKET_LEVEL_FEATURES,
    )
    from _research_r22_models import SEEDS

    # ── Load predictions ──────────────────────────────────────────────────
    log("\n[0] Loading predictions ...")
    r68_path = RESULTS_DIR / "r68_predictions.parquet"
    r93_path = RESULTS_DIR / "r93_predictions.parquet"

    if not r93_path.exists():
        log("  ✗ R93 predictions not found!")
        return

    r93_preds = pd.read_parquet(r93_path)
    log(f"  R93: {len(r93_preds):,} rows")

    if not r68_path.exists():
        log("  R68 predictions not cached — retraining ...")
        df, regime_df = load_data()
        feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
        no_rank = [f for f in feats if f in MARKET_LEVEL_FEATURES]
        r68_preds = train_ensemble(df, feats, CONTINUOUS_WINDOWS,
                                   seeds=SEEDS, cs_rank_exclude=no_rank)
        r68_preds.to_parquet(r68_path, index=False)
        del df
    else:
        r68_preds = pd.read_parquet(r68_path)
        _, regime_df = load_data()

    log(f"  R68: {len(r68_preds):,} rows")

    # ── First: compute R68 canonical baseline ─────────────────────────────
    log("\n[1] Computing R68 canonical baseline ...")
    port_r68_canon = simulate(r68_preds, regime_df, 4, 2)
    s_baseline = sharpe_ann(port_r68_canon["net_ret"])
    dd_baseline = max_dd(port_r68_canon["net_ret"])
    ret_baseline = float((1 + port_r68_canon["net_ret"]).prod() - 1) * 100
    log(f"  R68 baseline: Sharpe={s_baseline:.3f}  MaxDD={dd_baseline*100:.1f}%  "
        f"Ret={ret_baseline:.1f}%  N={len(port_r68_canon)}")

    # ── Merge predictions ─────────────────────────────────────────────────
    log("\n[2] Merging predictions (left join, keep all R68) ...")

    # Left join: keep ALL R68 rows, add R93 where available
    merged = r68_preds.merge(
        r93_preds[["timestamp", "symbol", "raw_prob"]].rename(
            columns={"raw_prob": "raw_prob_93"}),
        on=["timestamp", "symbol"], how="left"
    )

    # Fill missing R93 with 0.5 (neutral → logit=0)
    n_missing = merged["raw_prob_93"].isna().sum()
    merged["raw_prob_93"] = merged["raw_prob_93"].fillna(0.5)
    coverage = (1 - n_missing / len(merged)) * 100

    log(f"  Merged: {len(merged):,} rows")
    log(f"  R93 coverage: {coverage:.1f}% ({n_missing:,} filled with 0.5)")
    log(f"  Timestamps: {merged['timestamp'].nunique()}")

    # Compute logit scores
    merged["logit_68"] = scipy_logit(merged["raw_prob"].clip(CLIP_LO, CLIP_HI))
    merged["logit_93"] = scipy_logit(merged["raw_prob_93"].clip(CLIP_LO, CLIP_HI))

    log(f"  Logit 68: mean={merged['logit_68'].mean():.3f} std={merged['logit_68'].std():.3f}")
    log(f"  Logit 93: mean={merged['logit_93'].mean():.3f} std={merged['logit_93'].std():.3f}")

    # ── Grid search over α ────────────────────────────────────────────────
    log("\n[3] Grid search: α ∈ {0.0, 0.25, 0.50, 0.75, 1.0} ...")

    alphas = [0.0, 0.25, 0.50, 0.75, 1.0]
    results = []
    best_sharpe = -999
    best_port = None
    best_alpha = None

    for alpha in alphas:
        m = merged.copy()
        # Combine in logit space
        m["pred"] = alpha * m["logit_68"] + (1 - alpha) * m["logit_93"]
        m["fwd_ret"] = m["fwd_ret"]  # already fwd_ret_12h from R68

        # Use R68's actual simulate
        port = simulate(
            m[["timestamp", "symbol", "pred", "fwd_ret"]],
            regime_df, 4, 2
        )

        if len(port) == 0:
            log(f"  α={alpha:.2f}: NO DATA")
            continue

        net = port["net_ret"]
        s = sharpe_ann(net)
        dd = max_dd(net)
        total_ret = float((1 + net).prod() - 1) * 100
        wr = float((net > 0).mean())
        cal = calmar(net)

        tag = ""
        if alpha == 1.0:
            tag = " ← pure R68 logit"
        elif alpha == 0.0:
            tag = " ← pure R93 logit"

        log(f"  α={alpha:.2f}: Sharpe={s:>7.3f}  MaxDD={dd*100:>6.1f}%  "
            f"Ret={total_ret:>6.1f}%  Calmar={cal:>6.1f}  N={len(port)}{tag}")

        res = {
            "alpha": alpha,
            "sharpe": round(s, 4),
            "max_dd_pct": round(dd * 100, 2),
            "total_ret_pct": round(total_ret, 1),
            "win_rate": round(wr, 3),
            "n_periods": len(port),
            "calmar": round(cal, 2),
        }
        results.append(res)

        if s > best_sharpe:
            best_sharpe = s
            best_port = port
            best_alpha = alpha

    if best_port is None:
        log("  ✗ No valid configs!")
        return

    log(f"\n  Best: α={best_alpha:.2f}, Sharpe={best_sharpe:.3f}")

    # ── Check acceptance criteria ────────────────────────────────────────
    log("\n[4] Acceptance check ...")
    best_dd = max_dd(best_port["net_ret"])
    dd_improvement = (abs(dd_baseline) - abs(best_dd)) / abs(dd_baseline) * 100

    crit1 = best_sharpe >= s_baseline + 0.05
    crit2 = (dd_improvement >= 15 and best_sharpe >= s_baseline - 0.05)
    log(f"  Criterion 1 (Sharpe >= {s_baseline:.3f} + 0.05 = {s_baseline+0.05:.3f}): "
        f"{'PASS' if crit1 else 'FAIL'} (got {best_sharpe:.3f})")
    log(f"  Criterion 2 (DD↓≥15% + Sharpe≥{s_baseline-0.05:.3f}): "
        f"{'PASS' if crit2 else 'FAIL'} (DD improvement={dd_improvement:.1f}%)")

    # ── Bootstrap: best vs R68 canonical baseline ─────────────────────────
    log("\n[5] Bootstrap: best α={:.2f} vs R68 baseline ...".format(best_alpha))

    n_common = min(len(port_r68_canon), len(best_port))
    boot = block_bootstrap(
        port_r68_canon["net_ret"].values[:n_common],
        best_port["net_ret"].values[:n_common],
        n_boot=1000, block=10
    )

    log(f"  P(Sharpe better) = {boot['p_sharpe_better']:.3f}")
    log(f"  P(Calmar better) = {boot['p_calmar_better']:.3f}")
    log(f"  ΔSharpe: median={boot['median_delta_sharpe']:.4f}  "
        f"[{boot['p5_delta_sharpe']:.4f}, {boot['p95_delta_sharpe']:.4f}]")
    log(f"  Base Sharpe median: {boot['base_sharpe_med']:.4f}")
    log(f"  Best Sharpe median: {boot['exp_sharpe_med']:.4f}")

    accept_sharpe = crit1 and boot["p_sharpe_better"] > 0.8
    accept_calmar = crit2 and boot["p_calmar_better"] > 0.8
    accepted = accept_sharpe or accept_calmar

    log(f"\n  Sharpe acceptance: {'PASS ✅' if accept_sharpe else 'FAIL ❌'}")
    log(f"  Calmar acceptance: {'PASS ✅' if accept_calmar else 'FAIL ❌'}")
    log(f"  OVERALL: {'ACCEPTED ✅' if accepted else 'REJECTED ❌'}")

    # ── Save artifacts ────────────────────────────────────────────────────
    log("\n[6] Saving ...")

    summary = {
        "best_alpha": best_alpha,
        "best_sharpe": round(best_sharpe, 4),
        "best_maxdd_pct": round(best_dd * 100, 2),
        "best_return_pct": round(float((1 + best_port["net_ret"]).prod() - 1) * 100, 1),
        "baseline_sharpe": round(s_baseline, 4),
        "baseline_maxdd_pct": round(dd_baseline * 100, 2),
        "baseline_return_pct": round(ret_baseline, 1),
        "dd_improvement_pct": round(dd_improvement, 1),
        "accepted": accepted,
        "accept_sharpe": accept_sharpe,
        "accept_calmar": accept_calmar,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(RESULTS_DIR / "r103_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    best_port[["timestamp", "net_ret", "gross_ret", "cost", "turnover",
               "n_long", "n_short"]].to_csv(
        RESULTS_DIR / "r103_equity.csv", index=False)

    with open(RESULTS_DIR / "r103_bootstrap.json", "w") as f:
        json.dump(boot, f, indent=2)

    pd.DataFrame(results).to_csv(RESULTS_DIR / "r103_grid.csv", index=False)

    log(f"\n  Saved: r103_summary.json, r103_equity.csv, r103_bootstrap.json, r103_grid.csv")
    log(f"  Runtime: {time.time() - t0:.0f}s ({(time.time()-t0)/60:.1f}min)")


if __name__ == "__main__":
    main()
