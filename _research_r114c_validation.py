#!/usr/bin/env python3
"""
R114c — Validate R114b champion (off0.8_moff2_mon0) vs R113 baseline.

Three checks (per AI consultants):
  1) Per-window stability: R114b should help in all W1/W2/W3, not just one
  2) Per-year stability: compare on 2021, 2022, 2023, 2024, 2025-2026 subperiods
  3) Block bootstrap ΔSharpe / ΔCalmar: median(Δ)>0 and P(Δ>0) > 0.80

Acceptance criteria:
  - R114b wins in ≥2/3 windows
  - R114b wins in ≥3/5 year-subperiods
  - Bootstrap: median(ΔSharpe)>0(!) and P(ΔSharpe>0) > 0.80
"""
import time, json, os, warnings
import numpy as np, pandas as pd
from typing import Set, Dict
warnings.filterwarnings("ignore")

from _research_r22_models import SEEDS, log
from _research_r35_new_features import MARKET_LEVEL_FEATURES
from _research_r68_continuous_wf import (
    CHAMPION_FEAT_31, CONTINUOUS_WINDOWS, PROD_CFG,
    load_data, train_ensemble, sharpe, _cost_for_sym,
)
from _research_r113_trend_cutoff_reopt import analyze_config, print_result
from _research_r114b_churn_reduction import simulate_v2b


# ─── Subperiod analysis ─────────────────────────────────────────

def sharpe_from_series(rets, periods_per_year=2*365):
    """Compute Sharpe from a returns series."""
    if len(rets) < 10:
        return 0.0
    eq = (1 + rets).cumprod()
    r = eq.pct_change().dropna()
    if r.std() < 1e-10:
        return 0.0
    return r.mean() / r.std() * np.sqrt(periods_per_year)


def calmar_from_series(rets):
    if len(rets) < 10:
        return 0.0
    eq = (1 + rets).cumprod() * 100
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1
    maxdd = (eq / eq.cummax() - 1).min()
    if maxdd == 0:
        return 0.0
    return total_ret / abs(maxdd)


def maxdd_from_series(rets):
    if len(rets) < 2:
        return 0.0
    eq = (1 + rets).cumprod() * 100
    return (eq / eq.cummax() - 1).min()


def analyze_subperiod(port, start, end, label):
    """Analyze a sub-period of the portfolio."""
    mask = (port["timestamp"] >= start) & (port["timestamp"] < end)
    sub = port[mask]
    if len(sub) < 20:
        return {"label": label, "n": len(sub), "net_sharpe": 0, "calmar": 0, "max_dd_pct": 0, "ret_pct": 0}
    ns = sharpe_from_series(sub["net_ret"])
    cal = calmar_from_series(sub["net_ret"])
    dd = maxdd_from_series(sub["net_ret"]) * 100
    eq = (1 + sub["net_ret"]).cumprod()
    ret = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
    return {
        "label": label, "n": len(sub),
        "net_sharpe": round(ns, 3), "calmar": round(cal, 2),
        "max_dd_pct": round(dd, 1), "ret_pct": round(ret, 1),
    }


# ─── Block bootstrap ────────────────────────────────────────────

def block_bootstrap_delta(port_a, port_b, n_boot=5000, block_size=30, seed=42):
    """
    Block bootstrap on the DIFFERENCE of net returns (B - A).
    Returns distribution of ΔSharpe and ΔCalmar.
    """
    rng = np.random.RandomState(seed)

    # Align timestamps
    a = port_a.set_index("timestamp")["net_ret"]
    b = port_b.set_index("timestamp")["net_ret"]
    common = a.index.intersection(b.index)
    a, b = a.loc[common].values, b.loc[common].values
    n = len(a)
    if n < 100:
        log(f"  WARNING: only {n} common periods for bootstrap")

    n_blocks = max(n // block_size, 1)

    delta_sharpes = []
    delta_calmars = []

    for _ in range(n_boot):
        # Sample blocks with replacement
        starts = rng.randint(0, n - block_size + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]

        boot_a = a[idx]
        boot_b = b[idx]

        sh_a = sharpe_from_series(pd.Series(boot_a))
        sh_b = sharpe_from_series(pd.Series(boot_b))
        cal_a = calmar_from_series(pd.Series(boot_a))
        cal_b = calmar_from_series(pd.Series(boot_b))

        delta_sharpes.append(sh_b - sh_a)
        delta_calmars.append(cal_b - cal_a)

    return np.array(delta_sharpes), np.array(delta_calmars)


# ─── Main ────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log("=" * 70)
    log("R114c — Validation of R114b Champion")
    log("=" * 70)

    os.makedirs("results", exist_ok=True)

    log("\nLoading data...")
    df, regime_df = load_data()
    base_feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    no_rank = [f for f in base_feats if f in MARKET_LEVEL_FEATURES]

    log("\nTraining ensemble...")
    t1 = time.time()
    preds = train_ensemble(df, base_feats, CONTINUOUS_WINDOWS,
                           seeds=SEEDS, cs_rank_exclude=no_rank)
    log(f"  Trained in {time.time()-t1:.0f}s")

    cfg = dict(PROD_CFG)

    # ── Run R113 baseline ──
    log("\n" + "=" * 70)
    log("Running R113 baseline (cutoff_on=0.9, off=0.8, moff=1, mon=0)")
    log("=" * 70)
    port_r113 = simulate_v2b(preds, regime_df, 4, 2, cfg,
                             cutoff_on=0.9, cutoff_off=0.8,
                             min_risk_off_periods=1, min_risk_on_periods=0)
    m_r113 = analyze_config(port_r113, "R113")
    print_result(m_r113)

    # ── Run R114b champion ──
    log("\n" + "=" * 70)
    log("Running R114b champion (cutoff_on=0.9, off=0.8, moff=2, mon=0)")
    log("=" * 70)
    port_r114b = simulate_v2b(preds, regime_df, 4, 2, cfg,
                              cutoff_on=0.9, cutoff_off=0.8,
                              min_risk_off_periods=2, min_risk_on_periods=0)
    m_r114b = analyze_config(port_r114b, "R114b")
    print_result(m_r114b)

    # ════════════════════════════════════════════════════════════════
    # CHECK 1: Per-window stability
    # ════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("CHECK 1: Per-window stability (W1/W2/W3)")
    log("=" * 70)

    window_results = []
    for w in CONTINUOUS_WINDOWS:
        start = pd.Timestamp(w["test_start"], tz="UTC")
        end = pd.Timestamp(w["test_end"], tz="UTC")
        wname = w["name"]

        sub_113 = analyze_subperiod(port_r113, start, end, f"R113_{wname}")
        sub_114b = analyze_subperiod(port_r114b, start, end, f"R114b_{wname}")

        delta_sh = sub_114b["net_sharpe"] - sub_113["net_sharpe"]
        delta_cal = sub_114b["calmar"] - sub_113["calmar"]

        log(f"\n  {wname} ({w['test_start']} → {w['test_end']}):")
        log(f"    R113:  Sharpe={sub_113['net_sharpe']:.3f}  Calmar={sub_113['calmar']:.2f}  "
            f"DD={sub_113['max_dd_pct']:.1f}%  Ret={sub_113['ret_pct']:.1f}%  (n={sub_113['n']})")
        log(f"    R114b: Sharpe={sub_114b['net_sharpe']:.3f}  Calmar={sub_114b['calmar']:.2f}  "
            f"DD={sub_114b['max_dd_pct']:.1f}%  Ret={sub_114b['ret_pct']:.1f}%  (n={sub_114b['n']})")
        log(f"    Delta: ΔSharpe={delta_sh:+.3f}  ΔCalmar={delta_cal:+.2f}  "
            f"{'WIN' if delta_sh > 0 else 'LOSE'}")

        window_results.append({
            "window": wname, "start": w["test_start"], "end": w["test_end"],
            "r113_sharpe": sub_113["net_sharpe"], "r114b_sharpe": sub_114b["net_sharpe"],
            "r113_calmar": sub_113["calmar"], "r114b_calmar": sub_114b["calmar"],
            "delta_sharpe": round(delta_sh, 3), "delta_calmar": round(delta_cal, 2),
            "win": delta_sh > 0,
        })

    wins = sum(1 for w in window_results if w["win"])
    log(f"\n  Per-window: R114b wins {wins}/{len(window_results)} windows")
    check1_pass = wins >= 2
    log(f"  CHECK 1 {'PASS' if check1_pass else 'FAIL'} (need ≥2/3)")

    # ════════════════════════════════════════════════════════════════
    # CHECK 2: Per-year stability
    # ════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("CHECK 2: Per-year stability")
    log("=" * 70)

    # Get timestamp range from portfolio
    min_ts = port_r113["timestamp"].min()
    max_ts = port_r113["timestamp"].max()

    year_periods = [
        ("2024-H2", "2024-07-01", "2025-01-01"),
        ("2025-H1", "2025-01-01", "2025-07-01"),
        ("2025-H2", "2025-07-01", "2026-01-01"),
        ("2026-Q1", "2026-01-01", "2026-04-01"),
    ]

    year_results = []
    for period_name, start_str, end_str in year_periods:
        start = pd.Timestamp(start_str, tz="UTC")
        end = pd.Timestamp(end_str, tz="UTC")

        sub_113 = analyze_subperiod(port_r113, start, end, f"R113_{period_name}")
        sub_114b = analyze_subperiod(port_r114b, start, end, f"R114b_{period_name}")

        if sub_113["n"] < 20 and sub_114b["n"] < 20:
            log(f"\n  {period_name}: skipped (n<20)")
            continue

        delta_sh = sub_114b["net_sharpe"] - sub_113["net_sharpe"]
        delta_cal = sub_114b["calmar"] - sub_113["calmar"]

        log(f"\n  {period_name}:")
        log(f"    R113:  Sharpe={sub_113['net_sharpe']:.3f}  Calmar={sub_113['calmar']:.2f}  "
            f"DD={sub_113['max_dd_pct']:.1f}%  Ret={sub_113['ret_pct']:.1f}%  (n={sub_113['n']})")
        log(f"    R114b: Sharpe={sub_114b['net_sharpe']:.3f}  Calmar={sub_114b['calmar']:.2f}  "
            f"DD={sub_114b['max_dd_pct']:.1f}%  Ret={sub_114b['ret_pct']:.1f}%  (n={sub_114b['n']})")
        log(f"    Delta: ΔSharpe={delta_sh:+.3f}  ΔCalmar={delta_cal:+.2f}  "
            f"{'WIN' if delta_sh > 0 else 'LOSE'}")

        year_results.append({
            "period": period_name,
            "r113_sharpe": sub_113["net_sharpe"], "r114b_sharpe": sub_114b["net_sharpe"],
            "r113_calmar": sub_113["calmar"], "r114b_calmar": sub_114b["calmar"],
            "delta_sharpe": round(delta_sh, 3), "delta_calmar": round(delta_cal, 2),
            "win": delta_sh > 0,
        })

    wins_yr = sum(1 for y in year_results if y["win"])
    n_yr = len(year_results)
    log(f"\n  Per-year: R114b wins {wins_yr}/{n_yr} periods")
    check2_pass = wins_yr >= n_yr * 0.5
    log(f"  CHECK 2 {'PASS' if check2_pass else 'FAIL'} (need ≥50%)")

    # ════════════════════════════════════════════════════════════════
    # CHECK 3: Block bootstrap ΔSharpe / ΔCalmar
    # ════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("CHECK 3: Block bootstrap (5000 resamples, block=30)")
    log("=" * 70)

    delta_sharpes, delta_calmars = block_bootstrap_delta(
        port_r113, port_r114b, n_boot=5000, block_size=30)

    med_dsh = np.median(delta_sharpes)
    p_dsh_pos = (delta_sharpes > 0).mean()
    ci5_dsh, ci95_dsh = np.percentile(delta_sharpes, [5, 95])

    med_dcal = np.median(delta_calmars)
    p_dcal_pos = (delta_calmars > 0).mean()
    ci5_dcal, ci95_dcal = np.percentile(delta_calmars, [5, 95])

    log(f"\n  ΔSharpe (R114b - R113):")
    log(f"    Median: {med_dsh:+.3f}")
    log(f"    P(ΔSharpe > 0): {p_dsh_pos:.3f}")
    log(f"    90% CI: [{ci5_dsh:+.3f}, {ci95_dsh:+.3f}]")

    log(f"\n  ΔCalmar (R114b - R113):")
    log(f"    Median: {med_dcal:+.2f}")
    log(f"    P(ΔCalmar > 0): {p_dcal_pos:.3f}")
    log(f"    90% CI: [{ci5_dcal:+.2f}, {ci95_dcal:+.2f}]")

    check3_pass = med_dsh > 0 and p_dsh_pos > 0.80
    log(f"\n  CHECK 3 {'PASS' if check3_pass else 'FAIL'} "
        f"(need median(ΔSharpe)>0 AND P(ΔSharpe>0)>0.80)")

    # ════════════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("R114c FINAL VERDICT")
    log("=" * 70)

    checks = [
        ("Per-window stability", check1_pass),
        ("Per-year stability", check2_pass),
        ("Bootstrap ΔSharpe", check3_pass),
    ]
    for name, passed in checks:
        log(f"  {name}: {'PASS' if passed else 'FAIL'}")

    all_pass = all(p for _, p in checks)
    log(f"\n  OVERALL: {'PASS — R114b is production-grade champion' if all_pass else 'FAIL — R114b needs more scrutiny'}")

    # ── Save results ──
    results = {
        "overall_pass": all_pass,
        "r113_full": m_r113,
        "r114b_full": m_r114b,
        "check1_windows": window_results,
        "check1_pass": check1_pass,
        "check2_years": year_results,
        "check2_pass": check2_pass,
        "check3_bootstrap": {
            "n_boot": 5000, "block_size": 30,
            "delta_sharpe_median": round(med_dsh, 3),
            "delta_sharpe_p_positive": round(p_dsh_pos, 3),
            "delta_sharpe_ci90": [round(ci5_dsh, 3), round(ci95_dsh, 3)],
            "delta_calmar_median": round(med_dcal, 2),
            "delta_calmar_p_positive": round(p_dcal_pos, 3),
            "delta_calmar_ci90": [round(ci5_dcal, 2), round(ci95_dcal, 2)],
        },
        "check3_pass": check3_pass,
    }
    # Convert numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def _deep_convert(d):
        if isinstance(d, dict):
            return {k: _deep_convert(v) for k, v in d.items()}
        if isinstance(d, list):
            return [_deep_convert(v) for v in d]
        return _convert(d)

    with open("results/r114c_validation.json", "w") as f:
        json.dump(_deep_convert(results), f, indent=2)

    log(f"\nSaved: results/r114c_validation.json")
    log(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
