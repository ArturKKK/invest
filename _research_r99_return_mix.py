#!/usr/bin/env python3
"""
R99 — Simple Return Mix: R68 + R93 weighted combination.

ret_mix = (1 - w93) * ret_R68 + w93 * ret_R93_scaled

Vol-scaling applied if R97 showed vol mismatch.
Grid: w93 ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50}
Optimisation target: Calmar ratio (return / MaxDD).
Bootstrap best mix vs R68-only.
"""

import json
import sys
import time
import warnings
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
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


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
    return s / (dd + EPS)


# ── Block bootstrap ───────────────────────────────────────────────────────────

def block_bootstrap_calmar(rets_base, rets_exp, n_boot=1000, block=10, seed=42):
    rng = np.random.default_rng(seed)
    n = min(len(rets_base), len(rets_exp))
    rb = np.array(rets_base[:n], dtype=float)
    re = np.array(rets_exp[:n], dtype=float)
    n_blocks = n // block

    def _calmar(r):
        if len(r) < 2 or np.std(r) < EPS:
            return 0.0
        eq = np.cumprod(1 + r)
        dd = np.min(eq / np.maximum.accumulate(eq) - 1)
        sh = float(np.mean(r) / (np.std(r) + EPS) * np.sqrt(PPY))
        return sh / (abs(dd) + EPS)

    def _sh(r):
        if len(r) < 2 or np.std(r) < EPS:
            return 0.0
        return float(np.mean(r) / (np.std(r) + EPS) * np.sqrt(PPY))

    cb_list, ce_list, sb_list, se_list = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n - block + 1, size=n_blocks)
        block_idx = np.concatenate([np.arange(i, i + block) for i in idx])[:n]
        cb_list.append(_calmar(rb[block_idx]))
        ce_list.append(_calmar(re[block_idx]))
        sb_list.append(_sh(rb[block_idx]))
        se_list.append(_sh(re[block_idx]))

    cb, ce = np.array(cb_list), np.array(ce_list)
    sb, se = np.array(sb_list), np.array(se_list)
    delta_c = ce - cb
    delta_s = se - sb
    return {
        "p_calmar_better": round(float((ce > cb).mean()), 3),
        "p_sharpe_better": round(float((se > sb).mean()), 3),
        "median_delta_calmar": round(float(np.median(delta_c)), 3),
        "median_delta_sharpe": round(float(np.median(delta_s)), 4),
        "base_calmar_med": round(float(np.median(cb)), 3),
        "exp_calmar_med": round(float(np.median(ce)), 3),
        "base_sharpe_med": round(float(np.median(sb)), 4),
        "exp_sharpe_med": round(float(np.median(se)), 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log("=" * 70)
    log("  R99 — SIMPLE RETURN MIX (R68 + R93)")
    log("=" * 70)

    # ── Load equity curves ────────────────────────────────────────────────
    log("\n[0] Loading equity curves ...")

    r68_path = RESULTS_DIR / "r86_r84_baseline_equity.csv"
    r93_path = RESULTS_DIR / "r93_best_equity.csv"

    if not r68_path.exists():
        log(f"  ✗ R68 equity not found: {r68_path}")
        return
    if not r93_path.exists():
        log(f"  ✗ R93 equity not found: {r93_path}")
        return

    r68_port = pd.read_csv(r68_path, parse_dates=["timestamp"])
    r93_port = pd.read_csv(r93_path, parse_dates=["timestamp"])
    log(f"  R68: {len(r68_port)} periods")
    log(f"  R93: {len(r93_port)} periods")

    # Align on common timestamps
    r68_ts = r68_port.set_index("timestamp")
    r93_ts = r93_port.set_index("timestamp")
    common = r68_ts.index.intersection(r93_ts.index)
    log(f"  Common timestamps: {len(common)}")

    if len(common) < 50:
        log("  ✗ Not enough common timestamps!")
        return

    r68_rets = r68_ts.loc[common, "net_ret"].values.astype(float)
    r93_rets = r93_ts.loc[common, "net_ret"].values.astype(float)

    # ── Check vol-match from R97 ──────────────────────────────────────────
    log("\n[1] Vol assessment ...")

    r97_path = RESULTS_DIR / "r97_attribution.json"
    vol_match_needed = False
    vol_ratio = 1.0
    if r97_path.exists():
        r97 = json.loads(r97_path.read_text())
        vol_match_needed = r97.get("vol_match_needed", False)
        vol_ratio = r97.get("vol_ratio_r68_over_r93", 1.0)
        log(f"  R97: vol_match_needed={vol_match_needed}, vol_ratio(R68/R93)={vol_ratio:.2f}")
    else:
        vol_r68 = float(np.std(r68_rets))
        vol_r93 = float(np.std(r93_rets))
        vol_ratio = vol_r68 / (vol_r93 + EPS)
        vol_match_needed = abs(vol_ratio - 1.0) > 0.3
        log(f"  Computed: vol_ratio={vol_ratio:.2f}, vol_match_needed={vol_match_needed}")

    # Vol-scale R93 if needed
    if vol_match_needed:
        r93_scaled = r93_rets * vol_ratio
        log(f"  Applied vol-scaling: R93 rets × {vol_ratio:.2f}")
    else:
        r93_scaled = r93_rets
        log(f"  No vol-scaling applied")

    # ── Grid search over w93 ──────────────────────────────────────────────
    log("\n[2] Grid search: w93 ∈ {0.05 ... 0.50} ...")

    weights = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    results = []
    best_calmar = -999
    best_w = None
    best_mix_rets = None

    for w93 in weights:
        mix = (1 - w93) * r68_rets + w93 * r93_scaled
        s = sharpe_ann(pd.Series(mix))
        dd = max_dd(pd.Series(mix))
        cal = s / (abs(dd) + EPS)
        total_ret = float(np.prod(1 + mix) - 1) * 100

        res = {
            "w93": w93,
            "sharpe": round(s, 3),
            "max_dd_pct": round(dd * 100, 1),
            "total_ret_pct": round(total_ret, 1),
            "calmar": round(cal, 2),
            "n_periods": len(mix),
        }
        results.append(res)

        tag = " ← R68 only" if w93 == 0.0 else ""
        log(f"  w93={w93:.2f}: Sharpe={s:>7.3f}  DD={dd*100:>6.1f}%  "
            f"Ret={total_ret:>6.1f}%  Calmar={cal:>6.2f}{tag}")

        if w93 > 0 and cal > best_calmar:
            best_calmar = cal
            best_w = w93
            best_mix_rets = mix

    if best_mix_rets is None:
        log("  ✗ No valid mix!")
        return

    log(f"\n  Best: w93={best_w:.2f}, Calmar={best_calmar:.2f}")

    # Compare with R68-only
    r68_cal = calmar(pd.Series(r68_rets))
    log(f"  R68-only Calmar: {r68_cal:.2f}")
    log(f"  Improvement: {best_calmar - r68_cal:+.2f}")

    # ── Bootstrap: best mix vs R68 ────────────────────────────────────────
    log("\n[3] Bootstrap: best mix vs R68-only ...")

    bs = block_bootstrap_calmar(r68_rets, best_mix_rets, n_boot=1000, block=10)
    log(f"  P(Calmar better) = {bs['p_calmar_better']}")
    log(f"  P(Sharpe better) = {bs['p_sharpe_better']}")
    log(f"  ΔCalmar median: {bs['median_delta_calmar']:.3f}")
    log(f"  ΔSharpe median: {bs['median_delta_sharpe']:.4f}")
    log(f"  Base Calmar: {bs['base_calmar_med']:.2f}  →  Mix Calmar: {bs['exp_calmar_med']:.2f}")

    # ── Per-quarter comparison ────────────────────────────────────────────
    log("\n[4] Per-quarter comparison ...")

    ts_common = sorted(common)
    ts_series = pd.Series(ts_common)
    quarters = ts_series.dt.to_period("Q").astype(str).values

    mix_best = (1 - best_w) * r68_rets + best_w * r93_scaled
    df_q = pd.DataFrame({
        "quarter": quarters,
        "r68_ret": r68_rets,
        "r93_ret": r93_scaled,
        "mix_ret": mix_best,
    })

    log(f"\n  {'Quarter':<10} {'R68 Ret%':>10} {'R93 Ret%':>10} {'Mix Ret%':>10} {'R68 DD%':>10} {'Mix DD%':>10}")
    for q in sorted(df_q["quarter"].unique()):
        qdf = df_q[df_q["quarter"] == q]
        r68_qret = float(np.prod(1 + qdf["r68_ret"]) - 1) * 100
        r93_qret = float(np.prod(1 + qdf["r93_ret"]) - 1) * 100
        mix_qret = float(np.prod(1 + qdf["mix_ret"]) - 1) * 100
        r68_qdd = max_dd(pd.Series(qdf["r68_ret"].values)) * 100
        mix_qdd = max_dd(pd.Series(qdf["mix_ret"].values)) * 100
        log(f"  {q:<10} {r68_qret:>9.1f}% {r93_qret:>9.1f}% {mix_qret:>9.1f}% {r68_qdd:>9.1f}% {mix_qdd:>9.1f}%")

    # ── Save results ──────────────────────────────────────────────────────
    log("\n[5] Saving ...")

    summary = {
        "script": "r99_return_mix",
        "best_w93": best_w,
        "best_calmar": round(best_calmar, 3),
        "r68_calmar": round(r68_cal, 3),
        "vol_scaling_applied": vol_match_needed,
        "vol_ratio": round(vol_ratio, 3),
        "grid_results": results,
        "bootstrap": bs,
        "common_timestamps": len(common),
        "runtime_sec": round(time.time() - t0, 1),
    }
    (RESULTS_DIR / "r99_summary.json").write_text(
        json.dumps(summary, indent=2, default=float))

    # Save mix equity curve
    mix_eq = pd.DataFrame({
        "timestamp": ts_common,
        "net_ret": mix_best,
        "r68_ret": r68_rets,
        "r93_ret": r93_scaled,
    })
    mix_eq.to_csv(RESULTS_DIR / "r99_best_equity.csv", index=False)

    log(f"\n  Saved: r99_summary.json, r99_best_equity.csv")
    log(f"  Runtime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
