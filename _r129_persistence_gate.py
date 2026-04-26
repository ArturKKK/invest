"""R129 — Adaptive A1 gate via directional persistence.

Hypothesis (per AI critic):
  A1 helps in *low-persistence* (choppy / sign-changing) regimes,
  hurts in *high-persistence* (sustained directional drift) regimes.
  W2 (TD mean +0.06) ≈ low persistence; W1 (+0.30), W3 (-0.23) ≈ high.

Plan:
  Step A — ex-post sanity:
    Compute td_persist = |rollmean(td, L)| / (rollstd(td, L) + eps)  past-only.
    Bucket all decision timestamps by td_persist quantile (Q1..Q5).
    Compute Δ Sharpe (A1 - baseline) per bucket. Hypothesis confirmed if
    Q1 (low persist) >> Q5 (high persist).

  Step B — WF gate (run only if Step A confirms):
    For each WF window, fit threshold q ∈ {p20, p30, p40} on prior windows,
    apply gate=on iff td_persist < q on test. Compare full Sharpe + bootstrap.

Run: python _r129_persistence_gate.py [--lookback 720]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from _preflight_check import check_versions
check_versions()

import _r128_all_overlays_canonical as r128


def add_persistence(regime_df: pd.DataFrame, lookback: int = 720) -> pd.DataFrame:
    """Add td_persist column (past-only, shift(1) before rolling)."""
    out = regime_df.copy().sort_index()
    td = out["trend_direction"]
    # Shift(1) ensures the value at t uses only data through t-1
    td_lag = td.shift(1)
    rmean = td_lag.rolling(lookback, min_periods=lookback // 2).mean()
    rstd = td_lag.rolling(lookback, min_periods=lookback // 2).std()
    out[f"td_persist_{lookback}h"] = rmean.abs() / (rstd + 1e-9)
    out[f"td_rmean_{lookback}h"] = rmean
    out[f"td_rstd_{lookback}h"] = rstd
    return out


def run_sim(preds, regime_df, *, overlay, vol_lookup=None) -> pd.DataFrame:
    return r128.simulate_full(preds, regime_df, n_long=4, n_short=2,
                               overlay=overlay, vol_lookup=vol_lookup)


def attach_regime_at_decision(port: pd.DataFrame, regime_df: pd.DataFrame,
                               cols: List[str]) -> pd.DataFrame:
    """Attach regime values at each decision timestamp (past-only by construction)."""
    rd = regime_df[cols].copy()
    return port.merge(rd, left_on="timestamp", right_index=True, how="left")


def bucket_delta_sharpe(base: pd.DataFrame, alt: pd.DataFrame,
                         persist_col: str, n_buckets: int = 5) -> pd.DataFrame:
    """Bucket by persist_col and compute per-bucket Sharpe + delta."""
    merged = base[["timestamp", "net_ret"]].rename(columns={"net_ret": "base"}).merge(
        alt[["timestamp", "net_ret", persist_col]].rename(columns={"net_ret": "alt"}),
        on="timestamp", how="inner"
    )
    merged = merged.dropna(subset=[persist_col])
    qs = pd.qcut(merged[persist_col], n_buckets, labels=[f"Q{i+1}" for i in range(n_buckets)],
                 duplicates="drop")
    merged["bucket"] = qs

    rows = []
    for b, sub in merged.groupby("bucket", observed=True):
        if len(sub) < 5:
            continue
        s_base = r128.sharpe(sub["base"])
        s_alt = r128.sharpe(sub["alt"])
        rows.append({
            "bucket": str(b),
            "n": len(sub),
            "persist_min": float(sub[persist_col].min()),
            "persist_max": float(sub[persist_col].max()),
            "persist_med": float(sub[persist_col].median()),
            "S_base": round(s_base, 3),
            "S_A1": round(s_alt, 3),
            "ΔS": round(s_alt - s_base, 3),
            "mean_base_ret_bp": round(sub["base"].mean() * 1e4, 2),
            "mean_alt_ret_bp": round(sub["alt"].mean() * 1e4, 2),
        })
    return pd.DataFrame(rows)


def per_window_sharpe(port: pd.DataFrame, windows) -> Dict[str, Tuple[float, int]]:
    out = {}
    for i, win in enumerate(windows, 1):
        ts_start = pd.Timestamp(win["test_start"], tz="UTC")
        ts_end = pd.Timestamp(win["test_end"], tz="UTC")
        sub = port[(port["timestamp"] >= ts_start) & (port["timestamp"] < ts_end)]
        s = r128.sharpe(sub["net_ret"]) if len(sub) > 1 else 0.0
        out[f"W{i}"] = (round(float(s), 3), len(sub))
    return out


def boot_p_improvement(a: np.ndarray, b: np.ndarray, n_boot: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ra = a[idx]; rb = b[idx]
        sa = ra.mean() / (ra.std() + 1e-10) * np.sqrt(2*365)
        sb = rb.mean() / (rb.std() + 1e-10) * np.sqrt(2*365)
        diffs[k] = sb - sa
    return {
        "mean": float(diffs.mean()),
        "p_pos": float((diffs > 0).mean()),
        "p_05": float((diffs > 0.05).mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
    }


# ─────────────────────────────────────────────────────────────────────
# STEP B: WF GATE
# ─────────────────────────────────────────────────────────────────────

def simulate_gated(preds, regime_df_aug, n_long=4, n_short=2, *,
                    a1_cfg, persist_col, gate_thresholds_per_period,
                    vol_lookup=None) -> pd.DataFrame:
    """Run sim with A1 gated: A1 active only when persist < threshold(t).

    gate_thresholds_per_period: pd.Series indexed by timestamp, threshold value at each ts.
    """
    # We can't easily inject per-timestamp on/off into r128.simulate_full,
    # so we compute baseline AND A1 separately and stitch by gate decision.
    base = r128.simulate_full(preds, regime_df_aug, n_long, n_short,
                               overlay=None, vol_lookup=vol_lookup)
    a1 = r128.simulate_full(preds, regime_df_aug, n_long, n_short,
                             overlay={"a1": a1_cfg}, vol_lookup=vol_lookup)
    merged = base[["timestamp", "net_ret"]].rename(columns={"net_ret": "base_ret"}).merge(
        a1[["timestamp", "net_ret"]].rename(columns={"net_ret": "a1_ret"}),
        on="timestamp", how="inner"
    )
    # Attach persist at each ts
    persist = regime_df_aug[persist_col].rename("persist")
    merged = merged.merge(persist, left_on="timestamp", right_index=True, how="left")
    # Per-ts threshold
    thr = gate_thresholds_per_period.rename("thr")
    merged = merged.merge(thr, left_on="timestamp", right_index=True, how="left")
    merged["use_a1"] = (merged["persist"] < merged["thr"]) & merged["persist"].notna() & merged["thr"].notna()
    merged["net_ret"] = np.where(merged["use_a1"], merged["a1_ret"], merged["base_ret"])
    return merged[["timestamp", "net_ret", "use_a1", "persist", "thr"]]


def expanding_quantile_threshold(persist_series: pd.Series, q: float,
                                  min_periods: int = 720) -> pd.Series:
    """For each t, return q-th quantile of persist over [start, t-1]. Past-only."""
    return persist_series.shift(1).expanding(min_periods=min_periods).quantile(q)


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=720, help="hours for persistence rolling window")
    ap.add_argument("--quantiles", default="0.20,0.30,0.40", help="grid of gate thresholds")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 78)
    print(f"  R129 — Persistence-gated A1 (lookback={args.lookback}h)")
    print("=" * 78)

    preds, regime_df = r128.build_or_load_cache()
    syms = sorted(preds["symbol"].unique().tolist())
    vol_df = r128.build_or_load_realized_vol(syms)
    vol_lookup = None
    if not vol_df.empty:
        vol_lookup = {sym: g.set_index("timestamp")[["rv_24h", "rv_72h"]].sort_index()
                      for sym, g in vol_df.groupby("symbol")}

    regime_aug = add_persistence(regime_df, lookback=args.lookback)
    persist_col = f"td_persist_{args.lookback}h"
    print(f"\n  Persistence column: {persist_col}")
    print(f"  Non-null persist: {regime_aug[persist_col].notna().sum()} / {len(regime_aug)}")
    print(f"  Persist describe: min={regime_aug[persist_col].min():.3f} "
          f"med={regime_aug[persist_col].median():.3f} "
          f"max={regime_aug[persist_col].max():.3f} "
          f"std={regime_aug[persist_col].std():.3f}")

    # Per-window persistence stats
    print("\n  Per-window persist stats:")
    for win in r128.r68.CONTINUOUS_WINDOWS:
        ts_s = pd.Timestamp(win["test_start"], tz="UTC")
        ts_e = pd.Timestamp(win["test_end"], tz="UTC")
        sub = regime_aug[(regime_aug.index >= ts_s) & (regime_aug.index < ts_e)][persist_col]
        sub = sub.dropna()
        print(f"    {win['name']:<12s}  n={len(sub):>4d}  "
              f"mean={sub.mean():.3f}  med={sub.median():.3f}  "
              f"p20={sub.quantile(0.2):.3f}  p50={sub.quantile(0.5):.3f}  p80={sub.quantile(0.8):.3f}")

    # ─── Run baseline + A1 ───
    A1_BEST = {"trend_thr": 0.25, "weak_scale": 0.60}
    print("\n  Running baseline + A1...")
    base_port = run_sim(preds, regime_aug, overlay=None, vol_lookup=vol_lookup)
    a1_port = run_sim(preds, regime_aug, overlay={"a1": A1_BEST}, vol_lookup=vol_lookup)
    a1_port = attach_regime_at_decision(a1_port, regime_aug, [persist_col])

    base_full = r128.metrics(base_port)
    a1_full = r128.metrics(a1_port)
    print(f"  Baseline full: Net={base_full['net_sharpe']:+.3f}  n={base_full['n']}")
    print(f"  A1 full:       Net={a1_full['net_sharpe']:+.3f}  n={a1_full['n']}")

    # ─── STEP A: Bucketed sanity ───
    print("\n" + "=" * 78)
    print("  STEP A — Δ Sharpe (A1 - baseline) by persistence quantile bucket")
    print("=" * 78)
    bdf = bucket_delta_sharpe(base_port, a1_port, persist_col, n_buckets=5)
    print(bdf.to_string(index=False))

    print("\n  Hypothesis check: Δ should DECREASE from Q1 (low persist) to Q5 (high persist).")
    if not bdf.empty:
        d_q1 = bdf.iloc[0]["ΔS"]
        d_q5 = bdf.iloc[-1]["ΔS"]
        print(f"  Q1 ΔS = {d_q1:+.3f},  Q5 ΔS = {d_q5:+.3f},  drop = {d_q5 - d_q1:+.3f}")
        if d_q1 > 0.05 and d_q5 < d_q1:
            print("  ✅ Hypothesis SUPPORTED — proceeding to Step B (WF gate).")
        else:
            print("  ⚠ Hypothesis WEAK — Step B may not deliver. Running anyway for reference.")
    else:
        print("  ! Empty bucket df, skipping Step B.")
        return

    # ─── STEP B: WF gate ───
    print("\n" + "=" * 78)
    print("  STEP B — WF-gated A1: gate=on iff persist < expanding_quantile(q)")
    print("=" * 78)

    quantiles = [float(q) for q in args.quantiles.split(",")]
    persist_ts = regime_aug[persist_col]

    # Per-window evaluation per quantile
    rows_b = []
    for q in quantiles:
        thr_series = expanding_quantile_threshold(persist_ts, q, min_periods=720)
        gated = simulate_gated(preds, regime_aug, n_long=4, n_short=2,
                                a1_cfg=A1_BEST, persist_col=persist_col,
                                gate_thresholds_per_period=thr_series,
                                vol_lookup=vol_lookup)
        gated_clean = gated.dropna(subset=["thr"])  # drop early period without expanding history
        full_s = r128.sharpe(gated_clean["net_ret"]) if len(gated_clean) > 1 else 0.0
        # Per-window
        pw = per_window_sharpe(gated_clean, r128.r68.CONTINUOUS_WINDOWS)
        # Gate-on fraction overall and per window
        on_frac = float(gated_clean["use_a1"].mean())
        # Bootstrap vs baseline aligned timestamps
        m = gated_clean[["timestamp", "net_ret"]].rename(columns={"net_ret": "alt"}).merge(
            base_port[["timestamp", "net_ret"]].rename(columns={"net_ret": "base"}),
            on="timestamp", how="inner")
        bs = boot_p_improvement(m["base"].values, m["alt"].values, n_boot=5000)
        rows_b.append({
            "q": q, "full_S": round(full_s, 3),
            "ΔS_full": round(full_s - base_full["net_sharpe"], 3),
            "on_frac": round(on_frac, 3),
            "W1": pw.get("W1", (0, 0)), "W2": pw.get("W2", (0, 0)), "W3": pw.get("W3", (0, 0)),
            "boot_mean": round(bs["mean"], 3), "boot_p_pos": round(bs["p_pos"], 3),
            "boot_ci": (round(bs["ci_low"], 3), round(bs["ci_high"], 3)),
        })

    print(f"\n  Reference baseline full Net={base_full['net_sharpe']:+.3f}, A1 always-on full Net={a1_full['net_sharpe']:+.3f}")
    print(f"  {'q':<6}{'full_S':>8}{'ΔS':>8}{'on%':>7}  {'W1':>16}  {'W2':>16}  {'W3':>16}  {'boot_p>0':>10}  CI95")
    for r in rows_b:
        print(f"  {r['q']:<6}{r['full_S']:>+8.3f}{r['ΔS_full']:>+8.3f}{r['on_frac']*100:>6.1f}%  "
              f"{r['W1'][0]:>+8.3f}(n{r['W1'][1]:>4d})  "
              f"{r['W2'][0]:>+8.3f}(n{r['W2'][1]:>4d})  "
              f"{r['W3'][0]:>+8.3f}(n{r['W3'][1]:>4d})  "
              f"{r['boot_p_pos']:>10.3f}  [{r['boot_ci'][0]:+.3f},{r['boot_ci'][1]:+.3f}]")

    # Per-window baseline & A1-always for comparison
    pw_base = per_window_sharpe(base_port, r128.r68.CONTINUOUS_WINDOWS)
    pw_a1 = per_window_sharpe(a1_port, r128.r68.CONTINUOUS_WINDOWS)
    print(f"\n  REFERENCE per-window:")
    print(f"  {'baseline':<14s}  W1={pw_base['W1'][0]:+.3f}(n{pw_base['W1'][1]})  W2={pw_base['W2'][0]:+.3f}(n{pw_base['W2'][1]})  W3={pw_base['W3'][0]:+.3f}(n{pw_base['W3'][1]})")
    print(f"  {'A1 always-on':<14s}  W1={pw_a1['W1'][0]:+.3f}(n{pw_a1['W1'][1]})  W2={pw_a1['W2'][0]:+.3f}(n{pw_a1['W2'][1]})  W3={pw_a1['W3'][0]:+.3f}(n{pw_a1['W3'][1]})")

    print(f"\n  Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
