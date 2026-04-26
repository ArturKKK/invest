"""Bootstrap P(ΔSharpe>0) for R128 winner overlays vs canonical baseline.

Uses paired-period resampling on the per-period net_ret series saved in
results_r128_all_overlays_canonical.csv (which we don't keep — recompute).

Quick approach: re-run baseline + 2 winners only, save full per-period
series, then bootstrap.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

from _preflight_check import check_versions
check_versions()

import _r128_all_overlays_canonical as r128


def boot_p_improvement(rets_a: np.ndarray, rets_b: np.ndarray,
                        n_boot: int = 5000, seed: int = 0) -> dict:
    """Bootstrap P(Sharpe(b) > Sharpe(a))."""
    rng = np.random.default_rng(seed)
    n = len(rets_a)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ra = rets_a[idx]
        rb = rets_b[idx]
        sa = ra.mean() / (ra.std() + 1e-10) * np.sqrt(2*365)
        sb = rb.mean() / (rb.std() + 1e-10) * np.sqrt(2*365)
        diffs.append(sb - sa)
    diffs = np.array(diffs)
    return {
        "mean_delta": float(diffs.mean()),
        "p_improve": float((diffs > 0).mean()),
        "p_improve_05": float((diffs > 0.05).mean()),
        "p_improve_10": float((diffs > 0.10).mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
    }


def per_window_split(port: pd.DataFrame, windows) -> dict:
    """Split per-period series by walk-forward window (W1/W2/W3) and return Sharpes."""
    out = {}
    for i, win in enumerate(windows, 1):
        ts_end = pd.Timestamp(win["test_end"]).tz_localize("UTC") if pd.Timestamp(win["test_end"]).tz is None else pd.Timestamp(win["test_end"])
        ts_start = pd.Timestamp(win["test_start"]).tz_localize("UTC") if pd.Timestamp(win["test_start"]).tz is None else pd.Timestamp(win["test_start"])
        sub = port[(port["timestamp"] >= ts_start) & (port["timestamp"] < ts_end)]
        if len(sub) < 2:
            out[f"W{i}"] = (0.0, 0)
            continue
        s = r128.sharpe(sub["net_ret"])
        out[f"W{i}"] = (round(float(s), 3), len(sub))
    return out


def main():
    t0 = time.time()
    preds, regime_df = r128.build_or_load_cache()
    syms = sorted(preds["symbol"].unique().tolist())
    vol_df = r128.build_or_load_realized_vol(syms)
    vol_lookup = None
    if not vol_df.empty:
        vol_lookup = {sym: g.set_index("timestamp")[["rv_24h", "rv_72h"]].sort_index()
                      for sym, g in vol_df.groupby("symbol")}

    A1_BEST = {"trend_thr": 0.25, "weak_scale": 0.60}
    configs = [
        ("BASELINE", None),
        ("A1.kelly thr0.25 sc0.60", {"a1": A1_BEST}),
        ("A1+G2 72h", {"a1": A1_BEST, "g2": {"window": "72h"}}),
    ]

    print("=" * 78)
    print(f"  BOOTSTRAP & PER-WINDOW VALIDATION (4L/2S only)")
    print("=" * 78)

    series = {}
    for label, ov in configs:
        port = r128.simulate_full(preds, regime_df, n_long=4, n_short=2,
                                   overlay=ov, vol_lookup=vol_lookup)
        m = r128.metrics(port)
        series[label] = port
        print(f"  {label:<28s}  full Net={m['net_sharpe']:+.3f}  n={m['n']}")

    # Per-window
    print("\n  Per-window Net Sharpe:")
    for label, port in series.items():
        ws = per_window_split(port, r128.r68.CONTINUOUS_WINDOWS)
        s_str = "  ".join(f"{k}={v[0]:+.3f}(n={v[1]})" for k, v in ws.items())
        print(f"    {label:<28s}  {s_str}")

    # Bootstrap (paired)
    base = series["BASELINE"]
    print("\n  Bootstrap P(ΔSharpe > x) on PAIRED net_ret periods:")
    for label, port in series.items():
        if label == "BASELINE":
            continue
        # Align by timestamp (cef6e2f baseline has same timestamps as overlays)
        merged = base[["timestamp", "net_ret"]].rename(columns={"net_ret": "a"}).merge(
            port[["timestamp", "net_ret"]].rename(columns={"net_ret": "b"}),
            on="timestamp", how="inner"
        )
        a = merged["a"].values
        b = merged["b"].values
        res = boot_p_improvement(a, b, n_boot=5000)
        print(f"    {label:<28s}")
        print(f"      mean_Δ={res['mean_delta']:+.3f}  CI95=[{res['ci_low']:+.3f}, {res['ci_high']:+.3f}]")
        print(f"      P(Δ>0)={res['p_improve']:.3f}  P(Δ>0.05)={res['p_improve_05']:.3f}  P(Δ>0.10)={res['p_improve_10']:.3f}")

    print(f"\n  Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
