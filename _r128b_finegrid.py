"""R128 round-2 — A1 fine-grid sensitivity + per-window stability + 6L/3S + extra overlays.

Reuses cache from r128_overlay_sweep so no retraining needed.

Sweeps:
  1. A1 fine grid: trend_thr ∈ {0.1,0.15,0.2,0.25,0.3,0.4} × scale ∈ {0.3,0.4,0.5,0.6,0.7}
  2. Best A1 vs baseline split per-window (W1/W2/W3)
  3. Best A1 in 6L/3S mode
  4. F2 confidence-skip (|prob-0.5|<thr)
  5. G2 vol-weighted positions (inverse realized vol)

Output: results_r128b_finegrid.json + per-window CSV
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from _preflight_check import check_versions
check_versions()

import _research_r68_continuous_wf as r68
from _r128_overlay_sweep import (
    PREDS_PATH, REGIME_PATH, load_cache, load_funding,
    simulate_overlay, analyze, _cost_for_sym, PROD_CFG,
    TIER1_SYMS, TIER2_SYMS, TIER3_SYMS,
)

# -------- Confidence-skip overlay (F2) and vol-weighted (G2) need a custom sim
def simulate_extras(
    merged: pd.DataFrame,
    regime_df: pd.DataFrame,
    n_long: int,
    n_short: int,
    cfg: Dict[str, Any] = None,
    *,
    asymm: Optional[Dict[str, Any]] = None,
    confidence_skip_thr: Optional[float] = None,   # F2: skip if |raw_prob-0.5| < thr
    vol_weighted: bool = False,                    # G2: weight by 1/recent_vol per sym
    vol_lookback: int = 28,                        # 14 days at 12h period
) -> pd.DataFrame:
    """Extended simulator supporting confidence-skip and vol-weighting."""
    cfg = cfg or PROD_CFG
    trend_cutoff = cfg["trend_cutoff"]
    rebal_hours = cfg["rebal_hours"]
    ema_alpha = cfg.get("ema_alpha", None)
    hysteresis = cfg.get("hysteresis", 0)
    funding_per_12h = 0.00008

    # Pre-compute per-symbol rolling realized vol of fwd_ret (proxy)
    sym_vol_lookup: Dict[str, pd.Series] = {}
    if vol_weighted:
        for sym, g in merged.groupby("symbol"):
            g = g.sort_values("timestamp")
            roll = g["fwd_ret"].rolling(vol_lookback, min_periods=8).std()
            ts_idx = pd.to_datetime(g["timestamp"].values, utc=True)
            sym_vol_lookup[sym] = pd.Series(roll.values, index=ts_idx)

    all_rets = []
    prev_longs: set = set()
    prev_shorts: set = set()
    prev_preds: Dict[str, float] = {}

    timestamps_sorted = sorted(merged["timestamp"].unique())
    grouped = {ts: grp for ts, grp in merged.groupby("timestamp")}
    rebal_timestamps = timestamps_sorted[::rebal_hours]

    for ts in rebal_timestamps:
        if ts not in regime_df.index or ts not in grouped:
            continue
        row = regime_df.loc[ts]
        trend_str = row.get("trend_strength", 0)
        trend_dir = row.get("trend_direction", 0) if "trend_direction" in row else 0

        if trend_str > trend_cutoff:
            if prev_longs or prev_shorts:
                n_prev = len(prev_longs) + len(prev_shorts)
                avg_w = 1.0 / n_prev
                close_cost = sum(_cost_for_sym(s) * avg_w for s in prev_longs | prev_shorts)
                all_rets.append({"timestamp": ts, "gross_ret": 0.0, "net_ret": -close_cost,
                                 "cost": close_cost, "n_long": 0, "n_short": 0, "turnover": n_prev})
            else:
                all_rets.append({"timestamp": ts, "gross_ret": 0.0, "net_ret": 0.0,
                                 "cost": 0.0, "n_long": 0, "n_short": 0, "turnover": 0})
            prev_longs, prev_shorts = set(), set()
            continue

        grp = grouped[ts].copy()
        n = len(grp)
        nl = min(n_long, n // 3)
        ns = min(n_short, n // 3)
        if nl == 0 and ns == 0:
            continue

        exposure = 1.0
        dyn_threshold = cfg.get("dyn_threshold", 0.5)
        if dyn_threshold is not None and trend_str > dyn_threshold:
            exposure = max(0.1, 1.0 - (trend_str - dyn_threshold) /
                           (trend_cutoff - dyn_threshold + 1e-10) * 0.5)

        if ema_alpha is not None and ema_alpha < 1.0:
            for idx, r in grp.iterrows():
                sym = r["symbol"]
                raw_pred = r["pred"]
                smoothed = ema_alpha * raw_pred + (1 - ema_alpha) * prev_preds.get(sym, raw_pred)
                prev_preds[sym] = smoothed
                grp.at[idx, "pred"] = smoothed

        # Confidence skip (F2): mark low-confidence rows as ineligible
        grp["ok"] = True
        if confidence_skip_thr is not None and "raw_prob" in grp.columns:
            grp.loc[(grp["raw_prob"] - 0.5).abs() < confidence_skip_thr, "ok"] = False

        grp["pred_rank"] = grp["pred"].rank(ascending=False)

        if hysteresis > 0 and (prev_longs or prev_shorts):
            new_longs: set = set()
            new_shorts: set = set()
            for idx, r in grp.iterrows():
                sym, rank = r["symbol"], r["pred_rank"]
                if not r["ok"]:
                    continue
                if sym in prev_longs and rank <= nl + hysteresis:
                    new_longs.add(sym)
                elif sym in prev_shorts and rank > (n - ns - hysteresis):
                    new_shorts.add(sym)
            cand_long = grp[(~grp["symbol"].isin(new_longs | new_shorts)) & grp["ok"]]
            for _, r in cand_long.sort_values("pred_rank").head(nl - len(new_longs)).iterrows():
                new_longs.add(r["symbol"])
            cand_short = grp[(~grp["symbol"].isin(new_longs | new_shorts)) & grp["ok"]]
            for _, r in cand_short.sort_values("pred_rank", ascending=False).head(ns - len(new_shorts)).iterrows():
                new_shorts.add(r["symbol"])
        else:
            elig = grp[grp["ok"]]
            new_longs = set(elig.sort_values("pred_rank").head(nl)["symbol"].tolist()) if nl > 0 else set()
            new_shorts = set(elig.sort_values("pred_rank", ascending=False).head(ns)["symbol"].tolist()) if ns > 0 else set()

        new_opened = (new_longs - prev_longs) | (new_shorts - prev_shorts)
        closed = (prev_longs - new_longs) | (prev_shorts - new_shorts)
        total_positions = len(new_longs) + len(new_shorts)

        longs = grp[grp["symbol"].isin(new_longs)]
        shorts = grp[grp["symbol"].isin(new_shorts)]

        # Per-symbol weights
        if vol_weighted:
            def _w(sym):
                s = sym_vol_lookup.get(sym)
                if s is None or len(s) == 0:
                    return 1.0
                idx = s.index.searchsorted(ts) - 1
                if idx < 0 or pd.isna(s.iloc[idx]):
                    return 1.0
                return 1.0 / max(float(s.iloc[idx]), 1e-4)
            if len(longs) > 0:
                lw = np.array([_w(s) for s in longs["symbol"]])
                lw = lw / lw.sum() if lw.sum() > 0 else np.ones(len(longs)) / len(longs)
                long_ret = float((longs["fwd_ret"].values * lw).sum())
            else:
                long_ret = 0.0
            if len(shorts) > 0:
                sw = np.array([_w(s) for s in shorts["symbol"]])
                sw = sw / sw.sum() if sw.sum() > 0 else np.ones(len(shorts)) / len(shorts)
                short_ret = float((shorts["fwd_ret"].values * sw).sum())
            else:
                short_ret = 0.0
        else:
            long_ret = longs["fwd_ret"].mean() if len(longs) > 0 else 0.0
            short_ret = shorts["fwd_ret"].mean() if len(shorts) > 0 else 0.0

        nl_act, ns_act = len(new_longs), len(new_shorts)
        if asymm is not None and nl_act > 0 and ns_act > 0:
            trend_thr = asymm.get("trend_thr", 0.3)
            scale = asymm.get("weak_side_scale", 0.5)
            if trend_dir > trend_thr:
                w_long, w_short = 0.5, 0.5 * scale
            elif trend_dir < -trend_thr:
                w_long, w_short = 0.5 * scale, 0.5
            else:
                w_long, w_short = 0.5, 0.5
            tot = w_long + w_short
            w_long, w_short = w_long / tot, w_short / tot
            gross_ret = w_long * long_ret - w_short * short_ret
        elif nl_act > 0 and ns_act > 0:
            gross_ret = 0.5 * long_ret - 0.5 * short_ret
        elif ns_act > 0:
            gross_ret = -short_ret
        else:
            gross_ret = long_ret
        gross_ret *= exposure

        if total_positions > 0:
            avg_w = 1.0 / total_positions
            turnover_cost = sum(_cost_for_sym(s) * avg_w for s in new_opened)
            turnover_cost += sum(_cost_for_sym(s) * avg_w for s in closed)
            holding_cost = funding_per_12h * (rebal_hours / 12)
            total_cost = turnover_cost + holding_cost
        else:
            total_cost = 0.0

        net_ret = gross_ret - total_cost
        prev_longs, prev_shorts = new_longs, new_shorts

        all_rets.append({"timestamp": ts, "gross_ret": gross_ret, "net_ret": net_ret,
                         "cost": total_cost, "n_long": nl_act, "n_short": ns_act,
                         "turnover": len(new_opened) + len(closed)})

    return pd.DataFrame(all_rets) if all_rets else pd.DataFrame()


def per_window_breakdown(preds: pd.DataFrame, regime_df: pd.DataFrame,
                         overlay: Optional[Dict], n_long: int, n_short: int) -> pd.DataFrame:
    """Run sim and split returns by window (W1/W2/W3 by timestamp ranges from r68)."""
    port = simulate_overlay(preds, regime_df, n_long, n_short,
                            overlay=overlay, funding_df=None)
    if port.empty:
        return pd.DataFrame()
    rows = []
    for w in r68.CONTINUOUS_WINDOWS:
        ts0 = pd.Timestamp(w["test_start"], tz="UTC")
        ts1 = pd.Timestamp(w["test_end"], tz="UTC")
        sub = port[(port["timestamp"] >= ts0) & (port["timestamp"] <= ts1)]
        rows.append({"window": w["name"], "n": len(sub),
                     "gross_total": (1 + sub["gross_ret"]).prod() - 1,
                     "net_total": (1 + sub["net_ret"]).prod() - 1,
                     "gross_sharpe": r68.sharpe(sub["gross_ret"]),
                     "net_sharpe": r68.sharpe(sub["net_ret"])})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_r128b_finegrid.json")
    args = ap.parse_args()

    print(f"Loading cache from {PREDS_PATH}")
    preds, regime_df = load_cache()
    print(f"  preds: {len(preds):,} rows")

    results: dict = {}

    # 1. Fine grid sweep on A1
    print("\n=== A1 fine grid (4L/2S) ===")
    grid = []
    trends = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    scales = [0.30, 0.40, 0.50, 0.60, 0.70]
    base_port = simulate_overlay(preds, regime_df, 4, 2, overlay=None, funding_df=None)
    base = analyze(base_port, "BASELINE 4L/2S")
    grid_rows = []
    for t in trends:
        for s in scales:
            ov = {"asymm_kelly": {"trend_thr": t, "weak_side_scale": s}}
            port = simulate_overlay(preds, regime_df, 4, 2, overlay=ov, funding_df=None)
            r = analyze(port, f"A1 t={t:.2f} s={s:.2f}")
            r["trend_thr"] = t; r["scale"] = s
            r["delta_sharpe"] = r["net_sharpe"] - base["net_sharpe"]
            r["delta_net"] = r["net_total"] - base["net_total"]
            grid_rows.append(r)
    results["a1_grid_4l2s"] = grid_rows
    results["baseline_4l2s"] = base

    print("\n=== A1 grid heatmap (ΔSharpe) ===")
    df_grid = pd.DataFrame(grid_rows)
    pivot = df_grid.pivot(index="trend_thr", columns="scale", values="delta_sharpe")
    print(pivot.round(3).to_string())
    print(f"\nMax ΔSharpe: {df_grid['delta_sharpe'].max():.3f} at "
          f"trend_thr={df_grid.loc[df_grid['delta_sharpe'].idxmax(),'trend_thr']:.2f}, "
          f"scale={df_grid.loc[df_grid['delta_sharpe'].idxmax(),'scale']:.2f}")

    # 2. Per-window stability for top-3 A1 configs
    print("\n=== Per-window stability (top-3 A1 configs) ===")
    top3 = sorted(grid_rows, key=lambda r: r["delta_sharpe"], reverse=True)[:3]
    pw_rows = []
    base_pw = per_window_breakdown(preds, regime_df, None, 4, 2)
    print("BASELINE per-window:")
    print(base_pw.to_string(index=False, float_format=lambda v: f"{v:+.3f}" if isinstance(v, float) else v))
    for r in top3:
        ov = {"asymm_kelly": {"trend_thr": r["trend_thr"], "weak_side_scale": r["scale"]}}
        pw = per_window_breakdown(preds, regime_df, ov, 4, 2)
        pw["config"] = f"A1 t={r['trend_thr']:.2f} s={r['scale']:.2f}"
        print(f"\n{pw['config'].iloc[0]} per-window:")
        print(pw.drop(columns=["config"]).to_string(index=False, float_format=lambda v: f"{v:+.3f}" if isinstance(v, float) else v))
        # Δ vs baseline
        for _, br in pw.iterrows():
            base_row = base_pw[base_pw["window"] == br["window"]].iloc[0]
            pw_rows.append({"config": br["config"], "window": br["window"],
                            "net_sharpe": br["net_sharpe"],
                            "base_sharpe": base_row["net_sharpe"],
                            "delta_sharpe": br["net_sharpe"] - base_row["net_sharpe"]})
    results["per_window"] = pw_rows

    # 3. Best A1 in 6L/3S
    print("\n=== A1 best vs baseline in 6L/3S ===")
    best = max(grid_rows, key=lambda r: r["delta_sharpe"])
    base63 = analyze(simulate_overlay(preds, regime_df, 6, 3, overlay=None, funding_df=None),
                      "BASELINE 6L/3S")
    ov_best = {"asymm_kelly": {"trend_thr": best["trend_thr"], "weak_side_scale": best["scale"]}}
    a1_63 = analyze(simulate_overlay(preds, regime_df, 6, 3, overlay=ov_best, funding_df=None),
                    f"A1 best 6L/3S (t={best['trend_thr']:.2f} s={best['scale']:.2f})")
    results["63_baseline"] = base63
    results["63_a1_best"] = a1_63

    # 4. F2 — confidence-skip
    print("\n=== F2 confidence-skip (4L/2S) ===")
    f2_rows = []
    for thr in [0.01, 0.02, 0.03, 0.05]:
        port = simulate_extras(preds, regime_df, 4, 2, confidence_skip_thr=thr)
        r = analyze(port, f"F2 conf_skip thr={thr:.3f}")
        r["thr"] = thr
        r["delta_sharpe"] = r["net_sharpe"] - base["net_sharpe"]
        f2_rows.append(r)
    results["f2_grid"] = f2_rows

    # 5. G2 — vol-weighted
    print("\n=== G2 vol-weighted (4L/2S) ===")
    try:
        g2 = analyze(simulate_extras(preds, regime_df, 4, 2, vol_weighted=True),
                     "G2 vol-weighted")
        g2["delta_sharpe"] = g2["net_sharpe"] - base["net_sharpe"]
    except Exception as e:
        print(f"G2 failed: {e}")
        g2 = {"label": "G2 vol-weighted", "error": str(e)}
    results["g2"] = g2

    # 6. F2 + A1 combo
    print("\n=== A1+F2 combo (4L/2S, best A1 + best F2) ===")
    best_f2 = max(f2_rows, key=lambda r: r["delta_sharpe"])
    a1_f2 = analyze(simulate_extras(
        preds, regime_df, 4, 2,
        asymm={"trend_thr": best["trend_thr"], "weak_side_scale": best["scale"]},
        confidence_skip_thr=best_f2["thr"]
    ), f"A1+F2 (t={best['trend_thr']:.2f} s={best['scale']:.2f}, conf={best_f2['thr']:.3f})")
    a1_f2["delta_sharpe"] = a1_f2["net_sharpe"] - base["net_sharpe"]
    results["a1_f2_combo"] = a1_f2

    # 7. A1 + G2
    print("\n=== A1+G2 combo (4L/2S) ===")
    try:
        a1_g2 = analyze(simulate_extras(
            preds, regime_df, 4, 2,
            asymm={"trend_thr": best["trend_thr"], "weak_side_scale": best["scale"]},
            vol_weighted=True
        ), f"A1+G2 (t={best['trend_thr']:.2f} s={best['scale']:.2f}, vol-w)")
        a1_g2["delta_sharpe"] = a1_g2["net_sharpe"] - base["net_sharpe"]
    except Exception as e:
        print(f"A1+G2 failed: {e}")
        a1_g2 = {"label": "A1+G2", "error": str(e)}
    results["a1_g2_combo"] = a1_g2

    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
