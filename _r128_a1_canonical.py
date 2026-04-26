"""R128 A1 asymmetric kelly overlay on CANONICAL 3.777 baseline.

Runs on VM where _research_r68_continuous_wf.py uses the cef6e2f simulate
(skip risk-off, n=688) that yields 4L/2S Net Sharpe 3.777.

Strategy:
  1. Load data + train ensemble (or reuse cache if exists).
  2. Reproduce baseline (sanity: must match 3.777 ±0.01).
  3. For each (trend_thr, weak_side_scale) in grid: monkey-patch simulate to
     scale weak side by `weak_side_scale` based on trend_direction sign and
     re-evaluate Sharpe.
  4. Print results table for both 4L/2S and 6L/3S.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from _preflight_check import check_versions
check_versions()

import _research_r68_continuous_wf as r68

warnings.filterwarnings("ignore")

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
PREDS_PATH = CACHE_DIR / "r128_canonical_preds.parquet"
REGIME_PATH = CACHE_DIR / "r128_canonical_regime.parquet"

PROD_CFG = r68.PROD_CFG
_cost_for_sym = r68._cost_for_sym


def build_or_load_cache() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if PREDS_PATH.exists() and REGIME_PATH.exists():
        print(f"  Loading cached predictions from {PREDS_PATH}")
        preds = pd.read_parquet(PREDS_PATH)
        rdf = pd.read_parquet(REGIME_PATH).set_index("timestamp")
        return preds, rdf
    print("  No cache — training ensemble (~6 min)...")
    df, regime_df = r68.load_data()
    feats = [f for f in r68.CHAMPION_FEAT_31 if f in df.columns]
    no_rank = set(getattr(r68, "MARKET_LEVEL_FEATURES", []))
    preds = r68.train_ensemble(df, feats, r68.CONTINUOUS_WINDOWS,
                                seeds=r68.SEEDS, cs_rank_exclude=no_rank)
    preds.to_parquet(PREDS_PATH)
    regime_df.reset_index().to_parquet(REGIME_PATH)
    return preds, regime_df


def simulate_a1(
    merged: pd.DataFrame,
    regime_df: pd.DataFrame,
    n_long: int,
    n_short: int,
    cfg: Dict[str, Any] = None,
    *,
    a1_trend_thr: float = None,
    a1_weak_scale: float = 0.5,
) -> pd.DataFrame:
    """Same as r68.simulate (cef6e2f: skip risk-off) but applies A1 overlay.

    A1: when trend_direction > a1_trend_thr -> short_weight *= a1_weak_scale
        when trend_direction < -a1_trend_thr -> long_weight *= a1_weak_scale
    Keeps gross exposure at original level (renormalize so long+short = 1.0).
    a1_trend_thr=None disables overlay -> exact baseline.
    """
    cfg = cfg or PROD_CFG
    trend_cutoff = cfg["trend_cutoff"]
    rebal_hours = cfg["rebal_hours"]
    ema_alpha = cfg.get("ema_alpha", None)
    hysteresis = cfg.get("hysteresis", 0)
    funding_per_12h = 0.00008

    all_rets = []
    prev_longs: Set[str] = set()
    prev_shorts: Set[str] = set()
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
            new_longs: Set[str] = set()
            new_shorts: Set[str] = set()
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
        long_ret = longs["fwd_ret"].mean() if len(longs) > 0 else 0
        short_ret = shorts["fwd_ret"].mean() if len(shorts) > 0 else 0

        nl_act, ns_act = len(new_longs), len(new_shorts)

        # ── A1 OVERLAY ───────────────────────────────────────────────
        # Default L/S weights (gross 1.0, half each side when both populated)
        if nl_act > 0 and ns_act > 0:
            w_l, w_s = 0.5, 0.5
            if a1_trend_thr is not None:
                if trend_dir > a1_trend_thr:
                    w_s *= a1_weak_scale
                elif trend_dir < -a1_trend_thr:
                    w_l *= a1_weak_scale
                # Renormalize to maintain original gross 1.0
                tot = w_l + w_s
                if tot > 0:
                    w_l /= tot
                    w_s /= tot
            gross_ret = w_l * long_ret - w_s * short_ret
        elif ns_act > 0:
            gross_ret = -short_ret
        else:
            gross_ret = long_ret
        gross_ret *= exposure

        if total_positions > 0:
            avg_weight = 1.0 / total_positions
            turnover_cost = sum(_cost_for_sym(sym) * avg_weight for sym in new_opened)
            turnover_cost += sum(_cost_for_sym(sym) * avg_weight for sym in closed)
            holding_cost = funding_per_12h * (rebal_hours / 12)
            total_cost = turnover_cost + holding_cost
        else:
            total_cost = 0.0

        net_ret = gross_ret - total_cost
        prev_longs, prev_shorts = new_longs, new_shorts

        all_rets.append({
            "timestamp": ts, "gross_ret": gross_ret, "net_ret": net_ret,
            "cost": total_cost, "n_long": nl_act, "n_short": ns_act,
            "turnover": len(new_opened) + len(closed),
        })

    return pd.DataFrame(all_rets) if all_rets else pd.DataFrame()


def sharpe(rets, periods_per_year=2*365):
    if len(rets) < 2:
        return 0.0
    eq = (1 + rets).cumprod()
    r = eq.pct_change().dropna()
    return r.mean() / (r.std() + 1e-10) * np.sqrt(periods_per_year)


def metrics(port: pd.DataFrame) -> Dict[str, float]:
    if port.empty:
        return {"net_sharpe": 0.0, "gross_sharpe": 0.0, "ret_pct": 0.0, "dd_pct": 0.0, "n": 0}
    gs = sharpe(port["gross_ret"])
    ns = sharpe(port["net_ret"])
    eq = (1 + port["net_ret"]).cumprod()
    total = float(eq.iloc[-1] - 1) * 100
    dd = float((eq / eq.cummax() - 1).min()) * 100
    return {
        "net_sharpe": round(ns, 3),
        "gross_sharpe": round(gs, 3),
        "ret_pct": round(total, 1),
        "dd_pct": round(dd, 1),
        "n": len(port),
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("  R128 A1 KELLY OVERLAY — CANONICAL 3.777 BASELINE")
    print("=" * 70)
    preds, regime_df = build_or_load_cache()
    print(f"  Predictions: {len(preds):,} rows, {preds['symbol'].nunique()} symbols")

    # Sanity: baseline
    print("\n[SANITY] Reproduce baseline (expect 4L/2S = 3.777, 6L/3S = 2.509)")
    for nl, ns in [(4, 2), (6, 3)]:
        port = simulate_a1(preds, regime_df, nl, ns, a1_trend_thr=None)
        m = metrics(port)
        print(f"  {nl}L/{ns}S baseline: Net={m['net_sharpe']:.3f} Gross={m['gross_sharpe']:.3f} "
              f"Ret={m['ret_pct']:.1f}% DD={m['dd_pct']:.1f}% n={m['n']}")

    # Grid search
    print("\n[A1 GRID] trend_thr × weak_side_scale")
    grid_thr = [0.20, 0.25, 0.30, 0.35]
    grid_scale = [0.40, 0.50, 0.60, 0.70]
    rows: List[Dict[str, Any]] = []
    for nl, ns in [(4, 2), (6, 3)]:
        for thr in grid_thr:
            for scale in grid_scale:
                port = simulate_a1(preds, regime_df, nl, ns,
                                   a1_trend_thr=thr, a1_weak_scale=scale)
                m = metrics(port)
                rows.append({
                    "config": f"{nl}L/{ns}S",
                    "trend_thr": thr,
                    "weak_scale": scale,
                    **m,
                })
                print(f"  {nl}L/{ns}S  thr={thr:.2f} scale={scale:.2f}  "
                      f"Net={m['net_sharpe']:.3f}  Gross={m['gross_sharpe']:.3f}  "
                      f"Ret={m['ret_pct']:>5.1f}%  DD={m['dd_pct']:>5.1f}%")

    df_res = pd.DataFrame(rows)
    out = "results_r128_a1_canonical.csv"
    df_res.to_csv(out, index=False)
    print(f"\n  Saved: {out}")

    # Summary winners
    print("\n[TOP 5 per config by Net Sharpe]")
    for cfg in ["4L/2S", "6L/3S"]:
        sub = df_res[df_res["config"] == cfg].nlargest(5, "net_sharpe")
        print(f"\n  --- {cfg} ---")
        for _, r in sub.iterrows():
            print(f"    thr={r.trend_thr:.2f} scale={r.weak_scale:.2f}  "
                  f"Net={r.net_sharpe:.3f}  Gross={r.gross_sharpe:.3f}  "
                  f"Ret={r.ret_pct:.1f}%  DD={r.dd_pct:.1f}%")
    print(f"\n  Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
