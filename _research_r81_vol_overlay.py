#!/usr/bin/env python3
"""
R81 — Vol Targeting + Drawdown Overlay on top of R68

Phase 1 of DeepResearch v3: "CG Alpha + Risk Overlay on top of R68".

Takes R68's 4L/2S continuous WF predictions and applies a post-hoc
position-sizing overlay:

  scale_t = clip(vol_target / vol_t, s_min, s_max)
  DD reduce: if dd > 10% → scale×0.7; if dd > 15% → scale×0.5

  scaled_net_t = net_ret_t × scale_t

Grid search (16 combinations):
  L           ∈ {20, 40}          — lookback for rolling vol estimate
  vol_tgt     ∈ {median, p25}     — target vol level
  s_min       ∈ {0.25, 0.35}      — min position scale
  s_max       ∈ {1.25, 1.50}      — max position scale

Acceptance criteria (vs R68 4L/2S baseline):
  - MaxDD ↓ ≥ 20% AND Sharpe ≥ baseline − 0.05  [primary]
  - Calmar ↑ ≥ 20% AND Sharpe ≥ baseline − 0.10  [secondary]

Outputs:
  results/r81_grid.csv           — all 16 combos × metrics
  results/r81_summary.json       — best config + acceptance verdict
  results/r81_best_equity.csv    — equity curve of best config
"""

import json
import sys
import time
import warnings
from itertools import product
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"

# ─── Grid ─────────────────────────────────────────────────────────────────────
LOOKBACKS     = [20, 40]
VOL_TGT_MODES = ["median", "p25"]
S_MINS        = [0.25, 0.35]
S_MAXS        = [1.25, 1.50]

PERIODS_PER_YEAR = 2 * 365   # 12h periods


# ─── Metrics ──────────────────────────────────────────────────────────────────

def sharpe(rets: pd.Series) -> float:
    if len(rets) < 2:
        return 0.0
    r = (1 + rets).cumprod().pct_change().dropna()
    return float(r.mean() / (r.std() + 1e-10) * np.sqrt(PERIODS_PER_YEAR))


def max_dd(rets: pd.Series) -> float:
    eq = (1 + rets).cumprod()
    return float((eq / eq.cummax() - 1).min())


def calmar(rets: pd.Series) -> float:
    s   = sharpe(rets)
    dd  = abs(max_dd(rets))
    return float(s / dd) if dd > 1e-6 else 0.0


def total_ret(rets: pd.Series) -> float:
    return float((1 + rets).prod() - 1)


def win_rate(rets: pd.Series) -> float:
    return float((rets > 0).mean())


def metrics(port: pd.DataFrame, ret_col: str = "net_ret") -> dict:
    rets = port[ret_col]
    return {
        "net_sharpe":    round(sharpe(rets), 4),
        "gross_sharpe":  round(sharpe(port["gross_ret"]), 4) if "gross_ret" in port.columns else None,
        "max_dd_pct":    round(max_dd(rets) * 100, 2),
        "calmar":        round(calmar(rets), 3),
        "total_ret_pct": round(total_ret(rets) * 100, 1),
        "win_rate":      round(win_rate(rets), 3),
        "n_periods":     len(rets),
        "avg_scale":     round(port["scale_t"].mean(), 3) if "scale_t" in port.columns else None,
    }


# ─── Vol Overlay ──────────────────────────────────────────────────────────────

def apply_vol_overlay(
    port: pd.DataFrame,
    L: int,
    vol_tgt_mode: str,
    s_min: float,
    s_max: float,
) -> pd.DataFrame:
    """
    Apply vol targeting + DD overlay sequentially (no lookahead).

    Scale rule:
      vol_t   = rolling std of net_ret over previous L periods (lagged 1)
      vol_tgt = median(vol_series) or percentile(25) of vol_series
      scale_t = clip(vol_tgt / vol_t, s_min, s_max)

    DD overlay (applied after vol scale):
      running equity of the scaled portfolio:
        dd > 10% → scale × 0.7
        dd > 15% → scale × 0.5

    All applied in causal order (only past returns used).
    """
    port = port.sort_values("timestamp").reset_index(drop=True)
    net_ret_arr   = port["net_ret"].values.astype(float)
    n             = len(net_ret_arr)

    # Rolling vol on base net_ret (non-causal for vol_target calculation OK —
    # vol_target is a scalar from the full series and serves as the target level)
    vol_series = (
        pd.Series(net_ret_arr)
        .rolling(L, min_periods=max(L // 2, 5))
        .std()
        .values
    )

    # vol_target: computed from the first half of the series to avoid fwd bias
    half     = n // 2
    half_vol = vol_series[:half]
    valid    = half_vol[~np.isnan(half_vol) & (half_vol > 1e-10)]
    if len(valid) == 0:
        # fallback: full series
        valid_full = vol_series[~np.isnan(vol_series) & (vol_series > 1e-10)]
        valid = valid_full if len(valid_full) > 0 else np.array([1e-4])

    if vol_tgt_mode == "median":
        vol_target = float(np.median(valid))
    else:  # p25
        vol_target = float(np.percentile(valid, 25))

    scale_arr = np.ones(n)
    equity    = 1.0
    peak      = 1.0

    for i in range(n):
        # Use vol estimated from t-1 (causal)
        vol_t = vol_series[i - 1] if i > 0 else np.nan

        if np.isnan(vol_t) or vol_t < 1e-10:
            scale_t = 1.0
        else:
            scale_t = float(np.clip(vol_target / vol_t, s_min, s_max))

        # DD overlay based on running equity of scaled portfolio so far
        dd = equity / peak - 1.0
        if dd < -0.15:
            scale_t *= 0.5
        elif dd < -0.10:
            scale_t *= 0.7

        scale_arr[i] = scale_t

        # Update equity
        equity *= 1.0 + net_ret_arr[i] * scale_t
        peak    = max(peak, equity)

    result                 = port.copy()
    result["scale_t"]      = scale_arr
    result["net_ret"]      = port["net_ret"] * scale_arr   # override for metrics
    result["gross_ret"]    = port["gross_ret"] * scale_arr
    result["cost"]         = port["cost"] * scale_arr
    return result


# ─── Print helpers ────────────────────────────────────────────────────────────

def print_monthly(port: pd.DataFrame, ret_col: str = "net_ret") -> None:
    port = port.copy()
    port["month"] = port["timestamp"].dt.to_period("M").astype(str)
    print(f"    {'Month':<10} {'Net%':>8} {'Gross%':>9} {'Scale':>7}")
    for m in sorted(port["month"].unique()):
        mdf = port[port["month"] == m]
        nr  = ((1 + mdf[ret_col]).cumprod().iloc[-1] - 1) * 100
        gr  = ((1 + mdf["gross_ret"]).cumprod().iloc[-1] - 1) * 100
        sc  = mdf["scale_t"].mean() if "scale_t" in mdf.columns else 1.0
        print(f"    {m:<10} {nr:>7.1f}% {gr:>8.1f}% {sc:>7.2f}x")


def print_quarterly(port: pd.DataFrame, ret_col: str = "net_ret") -> None:
    port = port.copy()
    port["quarter"] = port["timestamp"].dt.to_period("Q").astype(str)
    print(f"    {'Quarter':<10} {'NetSh':>8} {'Ret%':>8} {'MaxDD%':>8} {'Scale':>7}")
    for q in sorted(port["quarter"].unique()):
        qdf = port[port["quarter"] == q]
        ns  = sharpe(qdf[ret_col])
        nr  = total_ret(qdf[ret_col]) * 100
        dd  = max_dd(qdf[ret_col]) * 100
        sc  = qdf["scale_t"].mean() if "scale_t" in qdf.columns else 1.0
        print(f"    {q:<10} {ns:>8.2f} {nr:>7.1f}% {dd:>7.1f}% {sc:>7.2f}x")


def save_equity_csv(port: pd.DataFrame, path: Path, ret_col: str = "net_ret") -> None:
    out = port[["timestamp", "gross_ret", ret_col, "cost",
                "n_long", "n_short", "scale_t"]].copy()
    out = out.rename(columns={ret_col: "net_ret"})
    out["equity"] = (1 + out["net_ret"]).cumprod() * 100
    out.to_csv(path, index=False)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  R81 — VOL TARGETING + DD OVERLAY (Phase 1)")
    print("=" * 70)

    # ── [1/3] Load R68 data + run ensemble ───────────────────────────────────
    print("\n[1/3] Loading R68 data …")

    from _research_r68_continuous_wf import (
        load_data, train_ensemble, simulate, CONTINUOUS_WINDOWS, SEEDS, PROD_CFG,
        CHAMPION_FEAT_31, MARKET_LEVEL_FEATURES, analyze,
    )

    df, regime_df = load_data()
    feats   = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    no_rank = [f for f in feats if f in MARKET_LEVEL_FEATURES]

    print("\n[2/3] Training R68 ensemble (CONTINUOUS windows) …")
    t1 = time.time()
    preds = train_ensemble(df, feats, CONTINUOUS_WINDOWS, seeds=SEEDS,
                           cs_rank_exclude=no_rank)
    print(f"  Done in {time.time()-t1:.0f}s, {len(preds):,} predictions")

    # ── [2/3] Baseline (no overlay) ──────────────────────────────────────────
    print("\n[3/3] Grid search …")

    cfg_4l2s = {**PROD_CFG, "n_long": 4, "n_short": 2}
    base_port = simulate(preds, regime_df, n_long=4, n_short=2, cfg=cfg_4l2s)

    baseline_m = metrics(base_port)
    baseline_m["label"]    = "R68_baseline"
    baseline_m["L"]        = None
    baseline_m["vol_tgt"]  = None
    baseline_m["s_min"]    = None
    baseline_m["s_max"]    = None

    print(f"\n  R68 BASELINE:")
    print(f"    Net Sharpe: {baseline_m['net_sharpe']}  MaxDD: {baseline_m['max_dd_pct']}%  "
          f"Calmar: {baseline_m['calmar']}  Ret: {baseline_m['total_ret_pct']}%")

    # ── Grid search ───────────────────────────────────────────────────────────
    all_metrics: List[dict] = [baseline_m]
    combo_id = 0

    for L, vtm, s_min, s_max in product(LOOKBACKS, VOL_TGT_MODES, S_MINS, S_MAXS):
        combo_id += 1
        label = f"L{L}_{vtm}_smin{str(s_min).replace('.','')}_smax{str(s_max).replace('.','')}"

        scaled_port = apply_vol_overlay(base_port, L=L, vol_tgt_mode=vtm,
                                        s_min=s_min, s_max=s_max)
        m           = metrics(scaled_port)
        m["label"]  = label
        m["L"]      = L
        m["vol_tgt"] = vtm
        m["s_min"]  = s_min
        m["s_max"]  = s_max

        # Acceptance flags
        dd_delta  = abs(m["max_dd_pct"]) - abs(baseline_m["max_dd_pct"])  # negative = improvement
        sh_delta  = m["net_sharpe"] - baseline_m["net_sharpe"]
        cal_delta = m["calmar"] - baseline_m["calmar"]

        dd_improv_pct  = -dd_delta / abs(baseline_m["max_dd_pct"] + 1e-8) * 100
        cal_improv_pct = cal_delta / (baseline_m["calmar"] + 1e-8) * 100

        primary   = (dd_improv_pct >= 20) and (sh_delta >= -0.05)
        secondary = (cal_improv_pct >= 20) and (sh_delta >= -0.10)
        accept    = primary or secondary

        m["dd_improv_pct"]  = round(dd_improv_pct, 1)
        m["sh_delta"]       = round(sh_delta, 4)
        m["cal_improv_pct"] = round(cal_improv_pct, 1)
        m["primary_ok"]     = primary
        m["secondary_ok"]   = secondary
        m["accepted"]       = accept

        verdict = "✅ ACCEPT" if accept else "  "
        print(f"  [{combo_id:02d}] {label:<44}  Sh={m['net_sharpe']:+.3f}/{baseline_m['net_sharpe']:+.3f}"
              f"  DD={m['max_dd_pct']:+.1f}%/{baseline_m['max_dd_pct']:+.1f}%"
              f"  ΔDD={dd_improv_pct:+.1f}%  ΔSh={sh_delta:+.3f}  Sc={m['avg_scale']:.2f}x  {verdict}")
        all_metrics.append(m)

    # ── Save grid CSV ─────────────────────────────────────────────────────────
    grid_df   = pd.DataFrame(all_metrics)
    grid_path = RESULTS_DIR / "r81_grid.csv"
    grid_df.to_csv(grid_path, index=False)
    print(f"\n  Saved grid → {grid_path}")

    # ── Best config ───────────────────────────────────────────────────────────
    accepted = grid_df[grid_df["accepted"] == True]
    if len(accepted) > 0:
        # Primary sort: dd improvement; secondary: sharpe penalty minimized
        best_row = (
            accepted
            .sort_values(["primary_ok", "dd_improv_pct", "net_sharpe"],
                         ascending=[False, False, False])
            .iloc[0]
        )
        best_label = best_row["label"]
        print(f"\n  BEST: {best_label}")
        print(f"    DD ↓ {best_row['dd_improv_pct']:.1f}%  "
              f"ΔSharpe={best_row['sh_delta']:+.4f}  "
              f"Calmar ↑ {best_row['cal_improv_pct']:.1f}%")

        # Rebuild best equity curve
        best_port = apply_vol_overlay(
            base_port,
            L=int(best_row["L"]),
            vol_tgt_mode=best_row["vol_tgt"],
            s_min=float(best_row["s_min"]),
            s_max=float(best_row["s_max"]),
        )

        print(f"\n  BEST CONFIG — Monthly breakdown:")
        print_monthly(best_port)
        print(f"\n  BEST CONFIG — Quarterly breakdown:")
        print_quarterly(best_port)

        equity_path = RESULTS_DIR / "r81_best_equity.csv"
        save_equity_csv(best_port, equity_path)
        print(f"\n  Saved equity → {equity_path}")
    else:
        print("\n  ⚠  No config passed acceptance criteria.")
        best_row  = None
        best_port = None

    # ── Summary JSON ──────────────────────────────────────────────────────────
    accepted_configs = [
        m for m in all_metrics
        if m.get("accepted") and m["label"] != "R68_baseline"
    ]
    summary = {
        "script":        "r81_vol_overlay",
        "baseline":      {k: baseline_m[k] for k in
                          ["net_sharpe", "gross_sharpe", "max_dd_pct", "calmar",
                           "total_ret_pct", "win_rate", "n_periods"]},
        "n_accepted":    len(accepted_configs),
        "best_config":   best_row.to_dict() if best_row is not None else None,
        "all_accepted":  accepted_configs,
        "runtime_s":     round(time.time() - t0, 1),
    }
    summary_path = RESULTS_DIR / "r81_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=lambda x: (
            bool(x) if isinstance(x, (bool, np.bool_))
            else float(x) if isinstance(x, (np.floating, float))
            else int(x) if isinstance(x, (np.integer,))
            else str(x)
        ))
    print(f"  Saved summary → {summary_path}")

    # ── Final comparison ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ACCEPTANCE SUMMARY")
    print("=" * 70)
    print(f"  {'Config':<46} {'Sh':>7} {'DD%':>7} {'ΔDD%':>7} {'ΔSh':>7} {'Cal':>7} {'Accept':>8}")
    print(f"  {'-'*68}")
    for m in all_metrics:
        if m["label"] == "R68_baseline":
            print(f"  {'R68_baseline':<46} {m['net_sharpe']:>7.3f} {m['max_dd_pct']:>6.1f}%  "
                  f"{'—':>6}  {'—':>6} {m['calmar']:>7.3f}  baseline")
            continue
        accept_str = "✅ ACCEPT" if m.get("accepted") else "  ✗"
        print(f"  {m['label']:<46} {m['net_sharpe']:>7.3f} {m['max_dd_pct']:>6.1f}%"
              f" {m['dd_improv_pct']:>6.1f}% {m['sh_delta']:>7.3f} {m['calmar']:>7.3f}  {accept_str}")

    print(f"\n  Runtime: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)")
    print("  DONE.")


if __name__ == "__main__":
    main()
