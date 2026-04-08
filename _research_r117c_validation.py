#!/usr/bin/env python3
"""
R117c — Validate R117 dynamic K (q40 candidate) vs R114b baseline.

Protocol (same as R114c, applied to R117):
  1) Per-window stability: R117 should help in W1/W2/W3 (≥2/3 wins)
  2) Per-year stability: compare on 2024-H2, 2025-H1, 2025-H2, 2026-Q1 (≥50%)
  3) Block bootstrap ΔSharpe AND ΔCalmar (5000 resamples, block=30)
  4) Sensitivity sweep: q ∈ {0.30, 0.35, 0.40, 0.45} — check stability across thresholds

Acceptance (from both consultants):
  - P(ΔSharpe>0) > 0.80 OR P(ΔCalmar>0) > 0.80 (when Sharpe not worse by >0.05)
  - Wins ≥ 2/3 windows, no catastrophic loss in any window
  - Sensitivity: effect should be stable across q range (not sharp at one point)
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
from _research_r117_dynamic_k import simulate_v2b_dynK


# ─── Helpers (reuse from R114c) ─────────────────────────────────

def sharpe_from_series(rets, periods_per_year=2*365):
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
    mask = (port["timestamp"] >= start) & (port["timestamp"] < end)
    sub = port[mask]
    if len(sub) < 20:
        return {"label": label, "n": len(sub), "net_sharpe": 0, "calmar": 0,
                "max_dd_pct": 0, "ret_pct": 0}
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


def block_bootstrap_delta(port_a, port_b, n_boot=5000, block_size=30, seed=42):
    """Block bootstrap on DIFFERENCE of net returns (B - A)."""
    rng = np.random.RandomState(seed)
    a = port_a.set_index("timestamp")["net_ret"]
    b = port_b.set_index("timestamp")["net_ret"]
    common = a.index.intersection(b.index)
    a, b = a.loc[common].values, b.loc[common].values
    n = len(a)
    if n < 100:
        log(f"  WARNING: only {n} common periods for bootstrap")

    n_blocks = max(n // block_size, 1)
    delta_sharpes, delta_calmars = [], []

    for _ in range(n_boot):
        starts = rng.randint(0, n - block_size + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        boot_a, boot_b = a[idx], b[idx]
        sh_a = sharpe_from_series(pd.Series(boot_a))
        sh_b = sharpe_from_series(pd.Series(boot_b))
        cal_a = calmar_from_series(pd.Series(boot_a))
        cal_b = calmar_from_series(pd.Series(boot_b))
        delta_sharpes.append(sh_b - sh_a)
        delta_calmars.append(cal_b - cal_a)

    return np.array(delta_sharpes), np.array(delta_calmars)


def _deep_convert(d):
    """Recursively convert numpy types for JSON."""
    if isinstance(d, dict):
        return {k: _deep_convert(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_deep_convert(v) for v in d]
    if isinstance(d, (np.bool_,)):
        return bool(d)
    if isinstance(d, (np.integer,)):
        return int(d)
    if isinstance(d, (np.floating,)):
        return float(d)
    if isinstance(d, np.ndarray):
        return d.tolist()
    return d


# ─── Main ────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log("=" * 70)
    log("R117c — Validation of R117 Dynamic K (q40) vs R114b")
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

    # ── R114b baseline ──
    log("\n" + "=" * 70)
    log("R114b baseline (4L/2S fixed, moff=2)")
    log("=" * 70)
    port_base = simulate_v2b(preds, regime_df, 4, 2, cfg,
                             cutoff_on=0.9, cutoff_off=0.8,
                             min_risk_off_periods=2, min_risk_on_periods=0)
    m_base = analyze_config(port_base, "R114b_baseline")
    print_result(m_base)

    # ── R117 q40 candidate ──
    log("\n" + "=" * 70)
    log("R117 candidate: std_q40_4L2S_else2L1S")
    log("=" * 70)
    port_r117 = simulate_v2b_dynK(
        preds, regime_df, cfg,
        cutoff_on=0.9, cutoff_off=0.8,
        min_risk_off_periods=2, min_risk_on_periods=0,
        confidence_method="std_pred",
        high_q=1.0, low_q=0.4,
        high_nl=4, high_ns=2,
        mid_nl=4, mid_ns=2,
        low_nl=2, low_ns=1,
    )
    m_r117 = analyze_config(port_r117, "R117_q40")
    print_result(m_r117)

    # ════════════════════════════════════════════════════════════════
    # CHECK 0: Sensitivity sweep q ∈ {0.30, 0.35, 0.40, 0.45}
    # ════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("CHECK 0: Sensitivity sweep (q threshold)")
    log("=" * 70)

    sweep_results = []
    for q in [0.30, 0.35, 0.40, 0.45]:
        port_q = simulate_v2b_dynK(
            preds, regime_df, cfg,
            cutoff_on=0.9, cutoff_off=0.8,
            min_risk_off_periods=2, min_risk_on_periods=0,
            confidence_method="std_pred",
            high_q=1.0, low_q=q,
            high_nl=4, high_ns=2,
            mid_nl=4, mid_ns=2,
            low_nl=2, low_ns=1,
        )
        m_q = analyze_config(port_q, f"q{int(q*100)}")
        delta_sh = m_q["net_sharpe"] - m_base["net_sharpe"]
        delta_cal = m_q["calmar"] - m_base["calmar"]
        delta_dd = m_q["max_dd_pct"] - m_base["max_dd_pct"]

        log(f"\n  q={q:.2f}: Sharpe={m_q['net_sharpe']:.3f} (Δ={delta_sh:+.3f})  "
            f"Calmar={m_q['calmar']:.2f} (Δ={delta_cal:+.2f})  "
            f"DD={m_q['max_dd_pct']:.1f}% (Δ={delta_dd:+.1f})")

        k_dist = port_q.attrs.get("k_distribution", {}) if hasattr(port_q, 'attrs') else {}
        if k_dist:
            log(f"         K dist: {k_dist}")

        sweep_results.append({
            "q": q,
            "net_sharpe": m_q["net_sharpe"],
            "calmar": m_q["calmar"],
            "max_dd_pct": m_q["max_dd_pct"],
            "total_ret_pct": m_q["total_ret_pct"],
            "total_cost_pct": m_q.get("total_cost_pct", 0),
            "delta_sharpe": round(delta_sh, 3),
            "delta_calmar": round(delta_cal, 2),
            "k_distribution": k_dist,
        })

    # Check: is effect stable? All q values should beat baseline
    sweep_wins = sum(1 for s in sweep_results if s["delta_sharpe"] > 0)
    sweep_stable = sweep_wins >= 3  # at least 3/4 q values improve
    log(f"\n  Sensitivity: {sweep_wins}/4 q values beat baseline")
    log(f"  CHECK 0 {'PASS' if sweep_stable else 'FAIL ⚠️  (effect too sensitive to q)'}"
        f" (need ≥3/4)")

    # ════════════════════════════════════════════════════════════════
    # CHECK 1: Per-window stability (W1/W2/W3)
    # ════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("CHECK 1: Per-window stability (W1/W2/W3)")
    log("=" * 70)

    window_results = []
    for w in CONTINUOUS_WINDOWS:
        start = pd.Timestamp(w["test_start"], tz="UTC")
        end = pd.Timestamp(w["test_end"], tz="UTC")
        wname = w["name"]

        sub_base = analyze_subperiod(port_base, start, end, f"R114b_{wname}")
        sub_r117 = analyze_subperiod(port_r117, start, end, f"R117_{wname}")

        delta_sh = sub_r117["net_sharpe"] - sub_base["net_sharpe"]
        delta_cal = sub_r117["calmar"] - sub_base["calmar"]

        log(f"\n  {wname} ({w['test_start']} → {w['test_end']}):")
        log(f"    R114b: Sharpe={sub_base['net_sharpe']:.3f}  "
            f"Calmar={sub_base['calmar']:.2f}  "
            f"DD={sub_base['max_dd_pct']:.1f}%  Ret={sub_base['ret_pct']:.1f}%")
        log(f"    R117:  Sharpe={sub_r117['net_sharpe']:.3f}  "
            f"Calmar={sub_r117['calmar']:.2f}  "
            f"DD={sub_r117['max_dd_pct']:.1f}%  Ret={sub_r117['ret_pct']:.1f}%")
        log(f"    Delta: ΔSharpe={delta_sh:+.3f}  ΔCalmar={delta_cal:+.2f}  "
            f"{'WIN' if delta_sh > 0 else 'LOSE'}")

        # Check for catastrophic loss
        catastrophic = delta_sh < -0.5 or delta_cal < -5.0
        if catastrophic:
            log(f"    ⚠️  CATASTROPHIC loss in {wname}!")

        window_results.append({
            "window": wname, "start": w["test_start"], "end": w["test_end"],
            "base_sharpe": sub_base["net_sharpe"], "r117_sharpe": sub_r117["net_sharpe"],
            "base_calmar": sub_base["calmar"], "r117_calmar": sub_r117["calmar"],
            "base_dd": sub_base["max_dd_pct"], "r117_dd": sub_r117["max_dd_pct"],
            "delta_sharpe": round(delta_sh, 3), "delta_calmar": round(delta_cal, 2),
            "win": delta_sh > 0, "catastrophic": catastrophic,
        })

    wins = sum(1 for w in window_results if w["win"])
    any_catastrophic = any(w["catastrophic"] for w in window_results)
    check1_pass = wins >= 2 and not any_catastrophic
    log(f"\n  Per-window: R117 wins {wins}/{len(window_results)} windows"
        f"{'  ⚠️  CATASTROPHIC in some window!' if any_catastrophic else ''}")
    log(f"  CHECK 1 {'PASS' if check1_pass else 'FAIL'} (need ≥2/3, no catastrophic)")

    # ════════════════════════════════════════════════════════════════
    # CHECK 2: Per-year stability
    # ════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("CHECK 2: Per-year stability")
    log("=" * 70)

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

        sub_base = analyze_subperiod(port_base, start, end, f"R114b_{period_name}")
        sub_r117 = analyze_subperiod(port_r117, start, end, f"R117_{period_name}")

        if sub_base["n"] < 20 and sub_r117["n"] < 20:
            log(f"\n  {period_name}: skipped (n<20)")
            continue

        delta_sh = sub_r117["net_sharpe"] - sub_base["net_sharpe"]
        delta_cal = sub_r117["calmar"] - sub_base["calmar"]

        log(f"\n  {period_name}:")
        log(f"    R114b: Sharpe={sub_base['net_sharpe']:.3f}  "
            f"Calmar={sub_base['calmar']:.2f}  "
            f"DD={sub_base['max_dd_pct']:.1f}%  Ret={sub_base['ret_pct']:.1f}%")
        log(f"    R117:  Sharpe={sub_r117['net_sharpe']:.3f}  "
            f"Calmar={sub_r117['calmar']:.2f}  "
            f"DD={sub_r117['max_dd_pct']:.1f}%  Ret={sub_r117['ret_pct']:.1f}%")
        log(f"    Delta: ΔSharpe={delta_sh:+.3f}  ΔCalmar={delta_cal:+.2f}  "
            f"{'WIN' if delta_sh > 0 else 'LOSE'}")

        year_results.append({
            "period": period_name,
            "base_sharpe": sub_base["net_sharpe"], "r117_sharpe": sub_r117["net_sharpe"],
            "base_calmar": sub_base["calmar"], "r117_calmar": sub_r117["calmar"],
            "delta_sharpe": round(delta_sh, 3), "delta_calmar": round(delta_cal, 2),
            "win": delta_sh > 0,
        })

    wins_yr = sum(1 for y in year_results if y["win"])
    n_yr = len(year_results)
    check2_pass = wins_yr >= n_yr * 0.5
    log(f"\n  Per-year: R117 wins {wins_yr}/{n_yr} periods")
    log(f"  CHECK 2 {'PASS' if check2_pass else 'FAIL'} (need ≥50%)")

    # ════════════════════════════════════════════════════════════════
    # CHECK 3: Block bootstrap ΔSharpe AND ΔCalmar
    # ════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("CHECK 3: Block bootstrap (5000 resamples, block=30)")
    log("=" * 70)

    delta_sharpes, delta_calmars = block_bootstrap_delta(
        port_base, port_r117, n_boot=5000, block_size=30)

    med_dsh = float(np.median(delta_sharpes))
    p_dsh_pos = float((delta_sharpes > 0).mean())
    ci5_dsh, ci95_dsh = np.percentile(delta_sharpes, [5, 95])

    med_dcal = float(np.median(delta_calmars))
    p_dcal_pos = float((delta_calmars > 0).mean())
    ci5_dcal, ci95_dcal = np.percentile(delta_calmars, [5, 95])

    log(f"\n  ΔSharpe (R117 q40 - R114b):")
    log(f"    Median: {med_dsh:+.3f}")
    log(f"    P(ΔSharpe > 0): {p_dsh_pos:.3f}")
    log(f"    90% CI: [{ci5_dsh:+.3f}, {ci95_dsh:+.3f}]")

    log(f"\n  ΔCalmar (R117 q40 - R114b):")
    log(f"    Median: {med_dcal:+.2f}")
    log(f"    P(ΔCalmar > 0): {p_dcal_pos:.3f}")
    log(f"    90% CI: [{ci5_dcal:+.2f}, {ci95_dcal:+.2f}]")

    # Acceptance: P(ΔSharpe>0) > 0.80 OR P(ΔCalmar>0) > 0.80
    sharpe_pass = p_dsh_pos > 0.80
    calmar_pass = p_dcal_pos > 0.80
    # If Calmar passes but Sharpe doesn't, check Sharpe not catastrophically worse
    if calmar_pass and not sharpe_pass:
        sharpe_not_worse = med_dsh > -0.05
        check3_pass = sharpe_not_worse
        log(f"\n  Sharpe bootstrap FAIL but Calmar PASS → "
            f"check Sharpe not worse by >0.05: {sharpe_not_worse}")
    else:
        check3_pass = sharpe_pass or calmar_pass

    log(f"\n  CHECK 3 {'PASS' if check3_pass else 'FAIL'} "
        f"(P(ΔSh>0)={p_dsh_pos:.3f}, P(ΔCal>0)={p_dcal_pos:.3f}, "
        f"need either > 0.80)")

    # ════════════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("R117c FINAL VERDICT")
    log("=" * 70)

    checks = [
        ("Sensitivity sweep (q stable)", sweep_stable),
        ("Per-window stability", check1_pass),
        ("Per-year stability", check2_pass),
        ("Bootstrap ΔSharpe/ΔCalmar", check3_pass),
    ]
    for name, passed in checks:
        log(f"  {name}: {'PASS' if passed else 'FAIL'}")

    all_pass = all(p for _, p in checks)
    log(f"\n  OVERALL: {'PASS — R117 q40 is new champion' if all_pass else 'FAIL — keep R114b'}")

    # ── Save results ──
    results = {
        "overall_pass": bool(all_pass),
        "baseline_full": m_base,
        "r117_q40_full": m_r117,
        "check0_sensitivity": sweep_results,
        "check0_pass": bool(sweep_stable),
        "check1_windows": window_results,
        "check1_pass": bool(check1_pass),
        "check2_years": year_results,
        "check2_pass": bool(check2_pass),
        "check3_bootstrap": {
            "n_boot": 5000, "block_size": 30,
            "delta_sharpe_median": round(med_dsh, 3),
            "delta_sharpe_p_positive": round(p_dsh_pos, 3),
            "delta_sharpe_ci90": [round(float(ci5_dsh), 3), round(float(ci95_dsh), 3)],
            "delta_calmar_median": round(med_dcal, 2),
            "delta_calmar_p_positive": round(p_dcal_pos, 3),
            "delta_calmar_ci90": [round(float(ci5_dcal), 2), round(float(ci95_dcal), 2)],
        },
        "check3_pass": bool(check3_pass),
    }

    results = _deep_convert(results)

    with open("results/r117c_validation.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nSaved: results/r117c_validation.json")
    log(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
