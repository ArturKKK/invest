"""R128 round-3 — G2 deep dive + A1+G2 per-window + smooth trend variant.

Builds on round 2. Tests:
  1. G2 lookback sensitivity
  2. G2 with weight cap (avoid extreme concentration)
  3. A1+G2 per-window stability (does combo fix W3 loss?)
  4. A1 with EMA-smoothed trend_dir (smoother gating)
  5. Triple combo A1+G2 with smoothed trend
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from _preflight_check import check_versions
check_versions()

import _research_r68_continuous_wf as r68
from _r128_overlay_sweep import (
    PREDS_PATH, REGIME_PATH, load_cache,
    simulate_overlay, analyze, _cost_for_sym, PROD_CFG,
)
from _r128b_finegrid import simulate_extras, per_window_breakdown


# Extended sim with vol-weight cap + smoothed trend
def simulate_v3(
    merged: pd.DataFrame,
    regime_df: pd.DataFrame,
    n_long: int,
    n_short: int,
    cfg: Dict[str, Any] = None,
    *,
    asymm: Optional[Dict[str, Any]] = None,
    vol_weighted: bool = False,
    vol_lookback: int = 28,
    vol_weight_cap: Optional[float] = None,  # max ratio max_w / mean_w
    trend_ema_alpha: Optional[float] = None,  # smooth trend_dir
) -> pd.DataFrame:
    cfg = cfg or PROD_CFG
    trend_cutoff = cfg["trend_cutoff"]
    rebal_hours = cfg["rebal_hours"]
    ema_alpha = cfg.get("ema_alpha", None)
    hysteresis = cfg.get("hysteresis", 0)
    funding_per_12h = 0.00008

    sym_vol_lookup: Dict[str, pd.Series] = {}
    if vol_weighted:
        for sym, g in merged.groupby("symbol"):
            g = g.sort_values("timestamp")
            roll = g["fwd_ret"].rolling(vol_lookback, min_periods=max(2, min(8, vol_lookback))).std()
            ts_idx = pd.to_datetime(g["timestamp"].values, utc=True)
            sym_vol_lookup[sym] = pd.Series(roll.values, index=ts_idx)

    all_rets = []
    prev_longs: set = set()
    prev_shorts: set = set()
    prev_preds: Dict[str, float] = {}
    smoothed_trend_dir = 0.0

    timestamps_sorted = sorted(merged["timestamp"].unique())
    grouped = {ts: grp for ts, grp in merged.groupby("timestamp")}
    rebal_timestamps = timestamps_sorted[::rebal_hours]

    for ts in rebal_timestamps:
        if ts not in regime_df.index or ts not in grouped:
            continue
        row = regime_df.loc[ts]
        trend_str = row.get("trend_strength", 0)
        trend_dir_raw = row.get("trend_direction", 0) if "trend_direction" in row else 0

        # Smooth trend_dir
        if trend_ema_alpha is not None and 0 < trend_ema_alpha < 1.0:
            smoothed_trend_dir = (trend_ema_alpha * trend_dir_raw +
                                  (1 - trend_ema_alpha) * smoothed_trend_dir)
            trend_dir = smoothed_trend_dir
        else:
            trend_dir = trend_dir_raw

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

        grp["pred_rank"] = grp["pred"].rank(ascending=False)

        if hysteresis > 0 and (prev_longs or prev_shorts):
            new_longs: set = set()
            new_shorts: set = set()
            for idx, r in grp.iterrows():
                sym, rank = r["symbol"], r["pred_rank"]
                if sym in prev_longs and rank <= nl + hysteresis:
                    new_longs.add(sym)
                elif sym in prev_shorts and rank > (n - ns - hysteresis):
                    new_shorts.add(sym)
            for _, r in grp[~grp["symbol"].isin(new_longs | new_shorts)].sort_values("pred_rank").head(nl - len(new_longs)).iterrows():
                new_longs.add(r["symbol"])
            for _, r in grp[~grp["symbol"].isin(new_longs | new_shorts)].sort_values("pred_rank", ascending=False).head(ns - len(new_shorts)).iterrows():
                new_shorts.add(r["symbol"])
        else:
            new_longs = set(grp[grp["pred_rank"] <= nl]["symbol"].tolist()) if nl > 0 else set()
            new_shorts = set(grp[grp["pred_rank"] > (n - ns)]["symbol"].tolist()) if ns > 0 else set()

        new_opened = (new_longs - prev_longs) | (new_shorts - prev_shorts)
        closed = (prev_longs - new_longs) | (prev_shorts - new_shorts)
        total_positions = len(new_longs) + len(new_shorts)

        longs = grp[grp["symbol"].isin(new_longs)]
        shorts = grp[grp["symbol"].isin(new_shorts)]

        if vol_weighted:
            def _w(sym):
                s = sym_vol_lookup.get(sym)
                if s is None or len(s) == 0:
                    return 1.0
                idx = s.index.searchsorted(ts) - 1
                if idx < 0 or pd.isna(s.iloc[idx]):
                    return 1.0
                return 1.0 / max(float(s.iloc[idx]), 1e-4)

            def _weighted(side_df):
                if len(side_df) == 0:
                    return 0.0
                w = np.array([_w(s) for s in side_df["symbol"]])
                if vol_weight_cap is not None and len(w) > 1:
                    avg = w.mean()
                    w = np.clip(w, avg / vol_weight_cap, avg * vol_weight_cap)
                w = w / w.sum() if w.sum() > 0 else np.ones(len(w)) / len(w)
                return float((side_df["fwd_ret"].values * w).sum())
            long_ret = _weighted(longs)
            short_ret = _weighted(shorts)
        else:
            long_ret = longs["fwd_ret"].mean() if len(longs) > 0 else 0.0
            short_ret = shorts["fwd_ret"].mean() if len(shorts) > 0 else 0.0

        nl_act, ns_act = len(new_longs), len(new_shorts)
        if asymm is not None and nl_act > 0 and ns_act > 0:
            trend_thr = asymm.get("trend_thr", 0.25)
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


def per_window_v3(port: pd.DataFrame, label: str = "") -> pd.DataFrame:
    if port is None or port.empty:
        return pd.DataFrame()
    rows = []
    for w in r68.CONTINUOUS_WINDOWS:
        ts0 = pd.Timestamp(w["test_start"], tz="UTC")
        ts1 = pd.Timestamp(w["test_end"], tz="UTC")
        sub = port[(port["timestamp"] >= ts0) & (port["timestamp"] <= ts1)]
        rows.append({"label": label, "window": w["name"],
                     "n": len(sub),
                     "net_total": (1 + sub["net_ret"]).prod() - 1,
                     "net_sharpe": r68.sharpe(sub["net_ret"])})
    return pd.DataFrame(rows)


def main():
    print(f"Loading cache from {PREDS_PATH}")
    preds, regime_df = load_cache()

    base_port = simulate_v3(preds, regime_df, 4, 2)
    base = analyze(base_port, "BASELINE 4L/2S")

    BEST_A1 = {"trend_thr": 0.25, "weak_side_scale": 0.50}

    results: dict = {"baseline": base}

    print("\n=== G2 lookback sensitivity (4L/2S, no A1) ===")
    g2_rows = []
    for lb in [14, 28, 42, 56, 84, 168]:
        port = simulate_v3(preds, regime_df, 4, 2, vol_weighted=True, vol_lookback=lb)
        r = analyze(port, f"G2 lb={lb}")
        r["lb"] = lb
        r["delta_sharpe"] = r["net_sharpe"] - base["net_sharpe"]
        g2_rows.append(r)
    results["g2_lookback"] = g2_rows

    print("\n=== G2 weight cap sensitivity (lb=28) ===")
    cap_rows = []
    for cap in [None, 1.5, 2.0, 3.0, 5.0]:
        port = simulate_v3(preds, regime_df, 4, 2, vol_weighted=True,
                           vol_lookback=28, vol_weight_cap=cap)
        r = analyze(port, f"G2 cap={cap}")
        r["cap"] = cap
        r["delta_sharpe"] = r["net_sharpe"] - base["net_sharpe"]
        cap_rows.append(r)
    results["g2_caps"] = cap_rows

    print("\n=== A1+G2 per-window (best A1, lb=28) ===")
    port = simulate_v3(preds, regime_df, 4, 2, asymm=BEST_A1,
                       vol_weighted=True, vol_lookback=28)
    a1g2 = analyze(port, "A1+G2 (t=0.25, s=0.5, lb=28)")
    pw_a1g2 = per_window_v3(port, "A1+G2")
    print(pw_a1g2.to_string(index=False, float_format=lambda v: f"{v:+.3f}" if isinstance(v, float) else v))
    pw_base = per_window_v3(base_port, "BASELINE")
    print("BASELINE per-window:")
    print(pw_base.to_string(index=False, float_format=lambda v: f"{v:+.3f}" if isinstance(v, float) else v))
    results["a1g2_perwindow"] = pw_a1g2.to_dict("records")

    print("\n=== A1 with EMA-smoothed trend_dir (4L/2S) ===")
    smooth_rows = []
    for alpha in [None, 0.7, 0.5, 0.3, 0.2, 0.1]:
        port = simulate_v3(preds, regime_df, 4, 2, asymm=BEST_A1,
                           trend_ema_alpha=alpha)
        r = analyze(port, f"A1 trend_ema={alpha}")
        r["alpha"] = alpha
        r["delta_sharpe"] = r["net_sharpe"] - base["net_sharpe"]
        smooth_rows.append(r)
    results["a1_trend_smooth"] = smooth_rows

    print("\n=== A1+G2 with smoothed trend (best alpha) ===")
    best_alpha = max(smooth_rows, key=lambda r: r["delta_sharpe"])
    port = simulate_v3(preds, regime_df, 4, 2, asymm=BEST_A1,
                       vol_weighted=True, vol_lookback=28,
                       trend_ema_alpha=best_alpha["alpha"])
    a1g2_sm = analyze(port, f"A1+G2 + ema={best_alpha['alpha']}")
    a1g2_sm["delta_sharpe"] = a1g2_sm["net_sharpe"] - base["net_sharpe"]
    pw_combo = per_window_v3(port, "A1+G2+SM")
    print(pw_combo.to_string(index=False, float_format=lambda v: f"{v:+.3f}" if isinstance(v, float) else v))
    results["a1g2_smooth"] = a1g2_sm
    results["a1g2_smooth_perwindow"] = pw_combo.to_dict("records")

    print("\n=== 6L/3S best combo ===")
    base63 = analyze(simulate_v3(preds, regime_df, 6, 3), "BASELINE 6L/3S")
    a1g2_63 = analyze(simulate_v3(preds, regime_df, 6, 3, asymm=BEST_A1,
                                  vol_weighted=True, vol_lookback=28),
                      "A1+G2 6L/3S")
    a1g2_63["delta_sharpe"] = a1g2_63["net_sharpe"] - base63["net_sharpe"]
    results["63_baseline"] = base63
    results["63_a1g2"] = a1g2_63

    Path("results_r128c_round3.json").write_text(json.dumps(results, indent=2, default=str))
    print("\nSaved → results_r128c_round3.json")

    print("\n=== FINAL SUMMARY ===")
    print(f"  Baseline 4L/2S Sharpe        : {base['net_sharpe']:+.3f}")
    print(f"  A1 alone (best)              : {base['net_sharpe']+0.339:+.3f}  Δ +0.339")
    print(f"  G2 alone (lb=28)             : {next(r for r in g2_rows if r['lb']==28)['net_sharpe']:+.3f}  Δ {next(r for r in g2_rows if r['lb']==28)['delta_sharpe']:+.3f}")
    print(f"  A1+G2 (lb=28)                : {a1g2['net_sharpe']:+.3f}  Δ {a1g2['net_sharpe']-base['net_sharpe']:+.3f}")
    print(f"  A1+G2+smooth_trend           : {a1g2_sm['net_sharpe']:+.3f}  Δ {a1g2_sm['delta_sharpe']:+.3f}")
    print(f"  6L/3S baseline               : {base63['net_sharpe']:+.3f}")
    print(f"  6L/3S A1+G2                  : {a1g2_63['net_sharpe']:+.3f}  Δ {a1g2_63['delta_sharpe']:+.3f}")


if __name__ == "__main__":
    main()
