"""R130 — Validation suite for R129 persistence-gated A1.

Implements critic's checklist:
  (2) gate_on% per window + block bootstrap (7-day blocks)
  (3) sensitivity L ∈ {360, 720, 1440} at q=0.20
  (DD) max drawdown, sortino, tail quantiles per config

Builds on R129 logic but adds:
  - block_bootstrap (7d=14 periods at 12h cadence)
  - per-window gate_on%
  - DD / Sortino / 5%-quantile of returns
  - lookback sweep
"""
from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from _preflight_check import check_versions
check_versions()

import _r128_all_overlays_canonical as r128
import _r129_persistence_gate as r129


A1_BEST = {"trend_thr": 0.25, "weak_scale": 0.60}
PERIODS_PER_YEAR = 2 * 365  # 12h cadence


# ─────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────

def sharpe(rets: pd.Series | np.ndarray) -> float:
    r = np.asarray(rets, dtype=float)
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(PERIODS_PER_YEAR))


def sortino(rets: pd.Series | np.ndarray) -> float:
    r = np.asarray(rets, dtype=float)
    downside = r[r < 0]
    if len(downside) < 2 or downside.std() == 0:
        return 0.0
    return float(r.mean() / downside.std() * np.sqrt(PERIODS_PER_YEAR))


def max_drawdown(rets: pd.Series | np.ndarray) -> float:
    r = np.asarray(rets, dtype=float)
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak  # nonpositive
    return float(dd.min())  # most negative


def tail_metrics(rets: pd.Series) -> Dict[str, float]:
    r = np.asarray(rets, dtype=float)
    return {
        "sharpe": round(sharpe(r), 3),
        "sortino": round(sortino(r), 3),
        "maxDD": round(max_drawdown(r), 4),
        "p05": round(float(np.percentile(r, 5)), 5),
        "p95": round(float(np.percentile(r, 95)), 5),
        "mean_bp": round(float(r.mean()) * 1e4, 2),
        "n": int(len(r)),
    }


# ─────────────────────────────────────────────────────────────────────
# BLOCK BOOTSTRAP
# ─────────────────────────────────────────────────────────────────────

def block_bootstrap_diff(a: np.ndarray, b: np.ndarray,
                          block_len: int = 14,  # 7d at 12h cadence
                          n_boot: int = 5000, seed: int = 0) -> Dict:
    """Stationary block bootstrap of paired (a, b) series.

    Resample fixed-length blocks of consecutive indices, with random start.
    """
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty(n_boot)
    n_blocks = int(np.ceil(n / block_len))
    for k in range(n_boot):
        starts = rng.integers(0, n - block_len + 1, size=n_blocks)
        idx_list = []
        for s in starts:
            idx_list.append(np.arange(s, s + block_len))
        idx = np.concatenate(idx_list)[:n]
        ra = a[idx]; rb = b[idx]
        sa = ra.mean() / (ra.std() + 1e-10) * np.sqrt(PERIODS_PER_YEAR)
        sb = rb.mean() / (rb.std() + 1e-10) * np.sqrt(PERIODS_PER_YEAR)
        diffs[k] = sb - sa
    return {
        "mean": float(diffs.mean()),
        "p_pos": float((diffs > 0).mean()),
        "p_05": float((diffs > 0.05).mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
    }


# ─────────────────────────────────────────────────────────────────────
# PER-WINDOW
# ─────────────────────────────────────────────────────────────────────

def per_window_stats(port: pd.DataFrame, windows, gate_col: Optional[str] = None) -> Dict[str, Dict]:
    out = {}
    for i, win in enumerate(windows, 1):
        ts_s = pd.Timestamp(win["test_start"], tz="UTC")
        ts_e = pd.Timestamp(win["test_end"], tz="UTC")
        sub = port[(port["timestamp"] >= ts_s) & (port["timestamp"] < ts_e)]
        if len(sub) < 2:
            out[f"W{i}"] = {"n": 0}
            continue
        d = tail_metrics(sub["net_ret"])
        if gate_col is not None and gate_col in sub.columns:
            d["gate_on_pct"] = round(float(sub[gate_col].mean()) * 100, 1)
        out[f"W{i}"] = d
    return out


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookbacks", default="360,720,1440")
    ap.add_argument("--quantile", type=float, default=0.20)
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--block_len", type=int, default=14)
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 90)
    print(f"  R130 — R129 validation: gate%/window, block-bootstrap, lookback sensitivity, DD/tail")
    print(f"  q={args.quantile}, lookbacks={args.lookbacks}, n_boot={args.n_boot}, block_len={args.block_len}p (={args.block_len*12}h)")
    print("=" * 90)

    preds, regime_df = r128.build_or_load_cache()
    syms = sorted(preds["symbol"].unique().tolist())
    vol_df = r128.build_or_load_realized_vol(syms)
    vol_lookup = None
    if not vol_df.empty:
        vol_lookup = {sym: g.set_index("timestamp")[["rv_24h", "rv_72h"]].sort_index()
                      for sym, g in vol_df.groupby("symbol")}

    # Baseline once (lookback-independent)
    print("\n  Computing baseline (no overlay)...")
    base_port = r128.simulate_full(preds, regime_df, n_long=4, n_short=2,
                                    overlay=None, vol_lookup=vol_lookup)
    base_tail = tail_metrics(base_port["net_ret"])
    base_pw = per_window_stats(base_port, r128.r68.CONTINUOUS_WINDOWS)
    print(f"  Baseline: Sharpe={base_tail['sharpe']:+.3f}  Sortino={base_tail['sortino']:+.3f}  "
          f"maxDD={base_tail['maxDD']*100:+.2f}%  p05={base_tail['p05']*1e4:+.1f}bp  n={base_tail['n']}")

    # A1 always-on
    print("\n  Computing A1 always-on...")
    a1_port = r128.simulate_full(preds, regime_df, n_long=4, n_short=2,
                                  overlay={"a1": A1_BEST}, vol_lookup=vol_lookup)
    a1_tail = tail_metrics(a1_port["net_ret"])
    a1_pw = per_window_stats(a1_port, r128.r68.CONTINUOUS_WINDOWS)
    print(f"  A1 always: Sharpe={a1_tail['sharpe']:+.3f}  Sortino={a1_tail['sortino']:+.3f}  "
          f"maxDD={a1_tail['maxDD']*100:+.2f}%  p05={a1_tail['p05']*1e4:+.1f}bp  n={a1_tail['n']}")

    # ─── Reference per-window ───
    print("\n  REFERENCE per-window:")
    for label, pw in [("baseline", base_pw), ("A1 always", a1_pw)]:
        cells = []
        for w in ["W1", "W2", "W3"]:
            d = pw[w]
            cells.append(f"{w}: S={d.get('sharpe', 0):+.3f} DD={d.get('maxDD', 0)*100:+.1f}% n={d.get('n', 0)}")
        print(f"    {label:<11s} | " + "  ".join(cells))

    # ─── Lookback sweep ───
    lookbacks = [int(x) for x in args.lookbacks.split(",")]
    q = args.quantile

    print("\n" + "=" * 90)
    print(f"  LOOKBACK SENSITIVITY at q={q}")
    print("=" * 90)
    print(f"  {'L':>5}  {'full_S':>8}{'ΔS':>8}{'Sortino':>9}{'maxDD%':>8}{'p05bp':>8}  "
          f"{'on%':>6}  {'W1_on%':>7}{'W2_on%':>7}{'W3_on%':>7}  "
          f"{'W1_S':>7}{'W2_S':>7}{'W3_S':>7}  "
          f"{'iid p>0':>8}  {'block p>0':>10}  block_CI95")

    summary = []
    for L in lookbacks:
        regime_aug = r129.add_persistence(regime_df, lookback=L)
        persist_col = f"td_persist_{L}h"
        persist_ts = regime_aug[persist_col]
        thr_series = r129.expanding_quantile_threshold(persist_ts, q, min_periods=720)

        gated = r129.simulate_gated(preds, regime_aug, n_long=4, n_short=2,
                                     a1_cfg=A1_BEST, persist_col=persist_col,
                                     gate_thresholds_per_period=thr_series,
                                     vol_lookup=vol_lookup)
        gated_clean = gated.dropna(subset=["thr"]).reset_index(drop=True)

        full = tail_metrics(gated_clean["net_ret"])
        pw = per_window_stats(gated_clean, r128.r68.CONTINUOUS_WINDOWS, gate_col="use_a1")
        on_frac_total = float(gated_clean["use_a1"].mean()) * 100

        # Aligned bootstrap vs baseline
        m = gated_clean[["timestamp", "net_ret"]].rename(columns={"net_ret": "alt"}).merge(
            base_port[["timestamp", "net_ret"]].rename(columns={"net_ret": "base"}),
            on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
        bs_iid = r129.boot_p_improvement(m["base"].values, m["alt"].values, n_boot=args.n_boot)
        bs_blk = block_bootstrap_diff(m["base"].values, m["alt"].values,
                                       block_len=args.block_len, n_boot=args.n_boot)

        delta = full["sharpe"] - base_tail["sharpe"]

        def gw(w): return pw[w].get("gate_on_pct", 0.0)
        def sw(w): return pw[w].get("sharpe", 0.0)

        print(f"  {L:>5d}  {full['sharpe']:>+8.3f}{delta:>+8.3f}{full['sortino']:>+9.3f}"
              f"{full['maxDD']*100:>+8.2f}{full['p05']*1e4:>+8.1f}  "
              f"{on_frac_total:>5.1f}%  {gw('W1'):>6.1f}%{gw('W2'):>6.1f}%{gw('W3'):>6.1f}%  "
              f"{sw('W1'):>+7.3f}{sw('W2'):>+7.3f}{sw('W3'):>+7.3f}  "
              f"{bs_iid['p_pos']:>8.3f}  {bs_blk['p_pos']:>10.3f}  "
              f"[{bs_blk['ci_low']:+.3f},{bs_blk['ci_high']:+.3f}]")

        summary.append({
            "L": L, "full_S": full["sharpe"], "ΔS": delta,
            "sortino": full["sortino"], "maxDD": full["maxDD"], "p05": full["p05"],
            "on_total": on_frac_total,
            "W1_on": gw("W1"), "W2_on": gw("W2"), "W3_on": gw("W3"),
            "W1_S": sw("W1"), "W2_S": sw("W2"), "W3_S": sw("W3"),
            "W1_DD": pw["W1"].get("maxDD", 0) * 100,
            "W2_DD": pw["W2"].get("maxDD", 0) * 100,
            "W3_DD": pw["W3"].get("maxDD", 0) * 100,
            "iid_p_pos": bs_iid["p_pos"], "block_p_pos": bs_blk["p_pos"],
            "block_ci": (bs_blk["ci_low"], bs_blk["ci_high"]),
        })

    # ─── DD detail per window for L=720 ───
    print("\n" + "=" * 90)
    print(f"  DRAWDOWN DETAIL — L=720, q={q}  (compare with baseline / A1-always)")
    print("=" * 90)
    L_show = 720 if 720 in lookbacks else lookbacks[0]
    regime_aug720 = r129.add_persistence(regime_df, lookback=L_show)
    persist_col = f"td_persist_{L_show}h"
    thr_series = r129.expanding_quantile_threshold(regime_aug720[persist_col], q, min_periods=720)
    gated720 = r129.simulate_gated(preds, regime_aug720, n_long=4, n_short=2,
                                     a1_cfg=A1_BEST, persist_col=persist_col,
                                     gate_thresholds_per_period=thr_series,
                                     vol_lookup=vol_lookup).dropna(subset=["thr"]).reset_index(drop=True)
    gated_pw = per_window_stats(gated720, r128.r68.CONTINUOUS_WINDOWS, gate_col="use_a1")

    print(f"  {'window':<10}{'baseline DD%':>14}{'A1-on DD%':>12}{'gated DD%':>12}  "
          f"{'base S':>8}{'A1-on S':>9}{'gated S':>9}  {'gate%':>7}")
    for w in ["W1", "W2", "W3"]:
        print(f"  {w:<10}{base_pw[w].get('maxDD', 0)*100:>+14.2f}"
              f"{a1_pw[w].get('maxDD', 0)*100:>+12.2f}"
              f"{gated_pw[w].get('maxDD', 0)*100:>+12.2f}  "
              f"{base_pw[w].get('sharpe', 0):>+8.3f}"
              f"{a1_pw[w].get('sharpe', 0):>+9.3f}"
              f"{gated_pw[w].get('sharpe', 0):>+9.3f}  "
              f"{gated_pw[w].get('gate_on_pct', 0):>6.1f}%")

    # ─── Verdict ───
    print("\n" + "=" * 90)
    print("  VERDICT")
    print("=" * 90)
    best_L = max(summary, key=lambda x: (x["block_p_pos"], x["full_S"]))
    print(f"  Best by (block_p_pos, full_S): L={best_L['L']}  "
          f"S={best_L['full_S']:+.3f} (Δ{best_L['ΔS']:+.3f})  "
          f"block_p>0={best_L['block_p_pos']:.3f}  iid_p>0={best_L['iid_p_pos']:.3f}")

    sign_changes = sum(1 for s in summary if (s["ΔS"] > 0) != (summary[0]["ΔS"] > 0))
    if sign_changes == 0:
        print("  ✅ ΔS sign STABLE across all lookbacks → no red flag")
    else:
        print(f"  ⚠ ΔS sign FLIPS in {sign_changes}/{len(summary)} lookbacks → red flag")

    all_windows_pass = all(s["W1_S"] > base_pw["W1"]["sharpe"] - 0.01 and
                            s["W2_S"] > base_pw["W2"]["sharpe"] - 0.01 and
                            s["W3_S"] > base_pw["W3"]["sharpe"] - 0.01 for s in summary)
    print(f"  All-windows-pass-baseline: {'✅ YES' if all_windows_pass else '⚠ NO (some L hurts a window)'}")

    # DD comparison
    base_dd = base_tail["maxDD"]
    dd_ok = all(s["maxDD"] >= base_dd * 1.10 for s in summary)  # gated DD not >10% worse
    print(f"  DD not >10% worse than baseline ({base_dd*100:+.2f}%): "
          f"{'✅ YES' if dd_ok else '⚠ NO'}")

    print(f"\n  Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
