"""R128 multi-overlay backtest harness.

Reuses _research_r68_continuous_wf for data loading + training, then runs a
SWEEP of portfolio overlays over the same cached predictions.

Cached:
    cache/r128_preds_cont.parquet   merged dataframe (timestamp, symbol, pred, raw_prob, fwd_ret, window)
    cache/r128_regime.parquet       regime_df

Overlays implemented:
    A1 — asymmetric_kelly: scale short_weight by trend regime
    A2 — funding_filter:   skip long if funding > thr_long, skip short if funding < thr_short
    A3 — cooldown:         skip symbol for N rebalances after a losing trade
    A4 — cost_threshold:   require pred_rank percentile better than per-tier threshold

Each overlay can be combined; baseline = no overlays.

Usage:
    python _r128_overlay_sweep.py [--rebuild-cache]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Set

import numpy as np
import pandas as pd

from _preflight_check import check_versions
check_versions()

# Reuse from r68 module (training + simulate baseline)
import _research_r68_continuous_wf as r68

warnings.filterwarnings("ignore")

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
PREDS_PATH = CACHE_DIR / "r128_preds_cont.parquet"
REGIME_PATH = CACHE_DIR / "r128_regime.parquet"
FUNDING_PATH = Path("data/sentiment/binance_funding_rates.parquet")

# Symbol tiers (mirror r68)
TIER1_SYMS = r68.TIER1_SYMS
TIER2_SYMS = r68.TIER2_SYMS
TIER3_SYMS = r68.TIER3_SYMS
PROD_CFG = r68.PROD_CFG
_cost_for_sym = r68._cost_for_sym


# ----------------------------- DATA LOADING -----------------------------

def build_cache() -> tuple[pd.DataFrame, pd.DataFrame]:
    df, regime_df = r68.load_data()
    feats = [f for f in r68.CHAMPION_FEAT_31 if f in df.columns]
    no_rank = set(getattr(r68, "MARKET_LEVEL_FEATURES", []))
    print(f"  Training {len(r68.SEEDS)} seeds × {len(r68.CONTINUOUS_WINDOWS)} windows...")
    preds = r68.train_ensemble(df, feats, r68.CONTINUOUS_WINDOWS,
                                seeds=r68.SEEDS, cs_rank_exclude=no_rank)
    if preds is None:
        raise RuntimeError("train_ensemble returned None")
    preds.to_parquet(PREDS_PATH)
    regime_df.reset_index().to_parquet(REGIME_PATH)
    return preds, regime_df


def load_cache() -> tuple[pd.DataFrame, pd.DataFrame]:
    preds = pd.read_parquet(PREDS_PATH)
    rdf = pd.read_parquet(REGIME_PATH).set_index("timestamp")
    return preds, rdf


def load_funding() -> Optional[pd.DataFrame]:
    if not FUNDING_PATH.exists():
        return None
    f = pd.read_parquet(FUNDING_PATH)
    # columns expected: timestamp, symbol, funding_rate
    rename_map = {}
    for cand in ("fundingRate", "funding_rate_binance", "rate"):
        if cand in f.columns and "funding_rate" not in f.columns:
            rename_map[cand] = "funding_rate"
            break
    if rename_map:
        f = f.rename(columns=rename_map)
    if "funding_rate" not in f.columns:
        return None
    f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
    return f[["timestamp", "symbol", "funding_rate"]].sort_values(["symbol", "timestamp"])


# ----------------------------- CORE SIMULATOR ---------------------------

def simulate_overlay(
    merged: pd.DataFrame,
    regime_df: pd.DataFrame,
    n_long: int,
    n_short: int,
    cfg: Dict[str, Any] = None,
    *,
    overlay: Optional[Dict[str, Any]] = None,
    funding_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Like r68.simulate but accepts overlay config.

    overlay = {
        "asymm_kelly": {"trend_thr": 0.3, "weak_side_scale": 0.5},
        "funding_filter": {"thr_long": 0.0005, "thr_short": -0.0003},
        "cooldown": {"loss_thr": -0.03, "n_skip": 2},
        "cost_threshold": {"tier1_min": 0.0, "tier2_min": 0.0, "tier3_min": 0.6},
    }
    """
    cfg = cfg or PROD_CFG
    overlay = overlay or {}
    trend_cutoff = cfg["trend_cutoff"]
    rebal_hours = cfg["rebal_hours"]
    ema_alpha = cfg.get("ema_alpha", None)
    hysteresis = cfg.get("hysteresis", 0)
    funding_per_12h = 0.00008

    asymm = overlay.get("asymm_kelly")
    funding_cfg = overlay.get("funding_filter")
    cooldown_cfg = overlay.get("cooldown")
    cost_thr_cfg = overlay.get("cost_threshold")

    # Pre-resample funding to rebalance timestamps if needed
    funding_lookup: Dict = {}
    if funding_cfg is not None and funding_df is not None:
        # forward-fill last known funding per symbol
        funding_df = funding_df.copy()
        for sym, g in funding_df.groupby("symbol"):
            funding_lookup[sym] = g.set_index("timestamp")["funding_rate"].sort_index()

    all_rets = []
    prev_longs: Set[str] = set()
    prev_shorts: Set[str] = set()
    prev_preds: Dict[str, float] = {}
    cooldown_until: Dict[str, pd.Timestamp] = {}
    last_pos: Dict[str, str] = {}  # symbol -> 'L'/'S' last rebal

    timestamps_sorted = sorted(merged["timestamp"].unique())
    grouped = {ts: grp for ts, grp in merged.groupby("timestamp")}
    rebal_timestamps = timestamps_sorted[::rebal_hours]

    for i, ts in enumerate(rebal_timestamps):
        if ts not in regime_df.index or ts not in grouped:
            continue
        row = regime_df.loc[ts]
        trend_str = row.get("trend_strength", 0)
        trend_dir = row.get("trend_direction", 0) if "trend_direction" in row else 0

        # Risk-off skip
        if trend_str > trend_cutoff:
            if prev_longs or prev_shorts:
                n_prev = len(prev_longs) + len(prev_shorts)
                avg_w = 1.0 / n_prev
                close_cost = sum(_cost_for_sym(s) * avg_w for s in prev_longs | prev_shorts)
                all_rets.append({"timestamp": ts, "gross_ret": 0.0, "net_ret": -close_cost,
                                 "cost": close_cost, "n_long": 0, "n_short": 0,
                                 "turnover": n_prev})
            else:
                all_rets.append({"timestamp": ts, "gross_ret": 0.0, "net_ret": 0.0,
                                 "cost": 0.0, "n_long": 0, "n_short": 0, "turnover": 0})
            prev_longs, prev_shorts = set(), set()
            last_pos = {}
            continue

        grp = grouped[ts].copy()
        n = len(grp)

        nl = min(n_long, n // 3)
        ns = min(n_short, n // 3)
        if nl == 0 and ns == 0:
            continue

        # Exposure scaling (existing dyn_threshold)
        exposure = 1.0
        dyn_threshold = cfg.get("dyn_threshold", 0.5)
        if dyn_threshold is not None and trend_str > dyn_threshold:
            exposure = max(0.1, 1.0 - (trend_str - dyn_threshold) /
                           (trend_cutoff - dyn_threshold + 1e-10) * 0.5)

        # EMA pred smoothing
        if ema_alpha is not None and ema_alpha < 1.0:
            for idx, r in grp.iterrows():
                sym = r["symbol"]
                raw_pred = r["pred"]
                smoothed = ema_alpha * raw_pred + (1 - ema_alpha) * prev_preds.get(sym, raw_pred)
                prev_preds[sym] = smoothed
                grp.at[idx, "pred"] = smoothed

        grp["pred_rank"] = grp["pred"].rank(ascending=False)

        # Cooldown filter — eliminate symbols whose cooldown not yet expired
        if cooldown_cfg is not None and cooldown_until:
            grp = grp[~grp["symbol"].isin(
                {s for s, until in cooldown_until.items() if until > ts}
            )]
            if len(grp) < 4:
                # not enough to trade; skip period (record nothing)
                continue
            n = len(grp)
            nl = min(n_long, n // 3)
            ns = min(n_short, n // 3)
            grp["pred_rank"] = grp["pred"].rank(ascending=False)

        # Funding filter
        if funding_cfg is not None:
            thr_long = funding_cfg.get("thr_long", 0.0005)
            thr_short = funding_cfg.get("thr_short", -0.0003)
            def _funding_at(sym):
                s = funding_lookup.get(sym)
                if s is None or len(s) == 0:
                    return 0.0
                idx = s.index.searchsorted(ts) - 1
                if idx < 0:
                    return 0.0
                return float(s.iloc[idx])

            grp["funding"] = grp["symbol"].map(_funding_at)
            # mark ineligible-for-long, ineligible-for-short
            grp["ok_long"] = grp["funding"] <= thr_long
            grp["ok_short"] = grp["funding"] >= thr_short
        else:
            grp["ok_long"] = True
            grp["ok_short"] = True

        # Cost-aware threshold (per-tier minimum percentile)
        if cost_thr_cfg is not None:
            grp["pred_pct"] = grp["pred"].rank(pct=True)

            def _tier_min(sym):
                if sym in TIER1_SYMS: return cost_thr_cfg.get("tier1_min", 0.0)
                if sym in TIER3_SYMS: return cost_thr_cfg.get("tier3_min", 0.0)
                return cost_thr_cfg.get("tier2_min", 0.0)
            grp["min_long"] = grp["symbol"].map(_tier_min)
            # For long must be above (1-min_long) percentile, for short must be below min_short percentile
            # interpret tier_min as "edge required" — for longs need pct >= 1-tier_min, for shorts pct <= tier_min
            grp.loc[grp["pred_pct"] < (1 - grp["min_long"]), "ok_long"] = False
            grp.loc[grp["pred_pct"] > grp["min_long"], "ok_short"] = False

        # Selection (no hysteresis variant for clarity; keep original hysteresis logic)
        if hysteresis > 0 and (prev_longs or prev_shorts):
            new_longs: Set[str] = set()
            new_shorts: Set[str] = set()
            for idx, r in grp.iterrows():
                sym, rank = r["symbol"], r["pred_rank"]
                if sym in prev_longs and rank <= nl + hysteresis and r["ok_long"]:
                    new_longs.add(sym)
                elif sym in prev_shorts and rank > (n - ns - hysteresis) and r["ok_short"]:
                    new_shorts.add(sym)
            cand_long = grp[(~grp["symbol"].isin(new_longs | new_shorts)) & grp["ok_long"]]
            for _, r in cand_long.sort_values("pred_rank").head(nl - len(new_longs)).iterrows():
                new_longs.add(r["symbol"])
            cand_short = grp[(~grp["symbol"].isin(new_longs | new_shorts)) & grp["ok_short"]]
            for _, r in cand_short.sort_values("pred_rank", ascending=False).head(ns - len(new_shorts)).iterrows():
                new_shorts.add(r["symbol"])
        else:
            elig_long = grp[grp["ok_long"]].sort_values("pred_rank").head(nl)
            elig_short = grp[grp["ok_short"]].sort_values("pred_rank", ascending=False).head(ns)
            new_longs = set(elig_long["symbol"].tolist()) if nl > 0 else set()
            new_shorts = set(elig_short["symbol"].tolist()) if ns > 0 else set()

        new_opened = (new_longs - prev_longs) | (new_shorts - prev_shorts)
        closed = (prev_longs - new_longs) | (prev_shorts - new_shorts)
        total_positions = len(new_longs) + len(new_shorts)

        longs = grp[grp["symbol"].isin(new_longs)]
        shorts = grp[grp["symbol"].isin(new_shorts)]
        long_ret = longs["fwd_ret"].mean() if len(longs) > 0 else 0.0
        short_ret = shorts["fwd_ret"].mean() if len(shorts) > 0 else 0.0

        nl_act, ns_act = len(new_longs), len(new_shorts)

        # ---- Asymmetric Kelly weighting overlay ----
        if asymm is not None and nl_act > 0 and ns_act > 0:
            trend_thr = asymm.get("trend_thr", 0.3)
            scale = asymm.get("weak_side_scale", 0.5)
            # If bull regime (trend_dir > +thr) -> shorts are weak side
            if trend_dir > trend_thr:
                w_long, w_short = 0.5, 0.5 * scale
            elif trend_dir < -trend_thr:
                w_long, w_short = 0.5 * scale, 0.5
            else:
                w_long, w_short = 0.5, 0.5
            # renorm so sum stays = 1.0 to preserve unit gross
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

        # ---- Cooldown bookkeeping (per-symbol per-trade pnl) ----
        if cooldown_cfg is not None:
            loss_thr = cooldown_cfg.get("loss_thr", -0.03)
            n_skip = int(cooldown_cfg.get("n_skip", 2))
            # for each symbol that just CLOSED, get its individual pnl
            if i + 1 < len(rebal_timestamps):
                # use this period's fwd_ret as 12h ahead realized (ish)
                pass
            # simpler: penalize symbols whose realized period ret was bad
            for sym in new_longs:
                fr = float(grp[grp["symbol"] == sym]["fwd_ret"].iloc[0]) if (grp["symbol"] == sym).any() else 0.0
                if fr < loss_thr:
                    cooldown_until[sym] = ts + pd.Timedelta(hours=rebal_hours * (n_skip + 1))
            for sym in new_shorts:
                fr = float(grp[grp["symbol"] == sym]["fwd_ret"].iloc[0]) if (grp["symbol"] == sym).any() else 0.0
                if -fr < loss_thr:  # short loses when fwd_ret > 0
                    cooldown_until[sym] = ts + pd.Timedelta(hours=rebal_hours * (n_skip + 1))

        prev_longs, prev_shorts = new_longs, new_shorts

        all_rets.append({"timestamp": ts, "gross_ret": gross_ret, "net_ret": net_ret,
                         "cost": total_cost, "n_long": nl_act, "n_short": ns_act,
                         "turnover": len(new_opened) + len(closed)})

    return pd.DataFrame(all_rets) if all_rets else pd.DataFrame()


# ----------------------------- ANALYZE ---------------------------------

def analyze(port: pd.DataFrame, label: str) -> Dict[str, float]:
    if port is None or port.empty:
        print(f"  {label:<40s}: EMPTY")
        return {"label": label, "n": 0}
    n = len(port)
    gross_total = (1 + port["gross_ret"]).prod() - 1
    net_total = (1 + port["net_ret"]).prod() - 1
    gross_sharpe = r68.sharpe(port["gross_ret"])
    net_sharpe = r68.sharpe(port["net_ret"])
    avg_cost = port["cost"].mean()
    avg_turnover = port["turnover"].mean()
    n_active = (port["n_long"] + port["n_short"] > 0).sum()
    print(f"  {label:<40s}: n={n:>3d} act={n_active:>3d} "
          f"gross={gross_total:+.3f} net={net_total:+.3f} "
          f"S_g={gross_sharpe:+.3f} S_n={net_sharpe:+.3f} "
          f"cost={avg_cost*1e4:.1f}bp turn={avg_turnover:.2f}")
    return {"label": label, "n": n, "n_active": int(n_active),
            "gross_total": float(gross_total), "net_total": float(net_total),
            "gross_sharpe": float(gross_sharpe), "net_sharpe": float(net_sharpe),
            "avg_cost_bp": float(avg_cost * 1e4), "avg_turnover": float(avg_turnover)}


# ----------------------------- SWEEP DEFINITIONS -----------------------

def define_sweeps() -> list[dict]:
    return [
        {"label": "BASELINE (no overlay)", "overlay": None},

        # A1 — asymmetric kelly
        {"label": "A1.kelly_thr0.3_scale0.5", "overlay": {
            "asymm_kelly": {"trend_thr": 0.3, "weak_side_scale": 0.5}}},
        {"label": "A1.kelly_thr0.2_scale0.5", "overlay": {
            "asymm_kelly": {"trend_thr": 0.2, "weak_side_scale": 0.5}}},
        {"label": "A1.kelly_thr0.3_scale0.3", "overlay": {
            "asymm_kelly": {"trend_thr": 0.3, "weak_side_scale": 0.3}}},
        {"label": "A1.kelly_thr0.5_scale0.5", "overlay": {
            "asymm_kelly": {"trend_thr": 0.5, "weak_side_scale": 0.5}}},

        # A2 — funding filter (requires funding data)
        {"label": "A2.funding_long5bp_short3bp", "overlay": {
            "funding_filter": {"thr_long": 0.0005, "thr_short": -0.0003}}},
        {"label": "A2.funding_long3bp_short3bp", "overlay": {
            "funding_filter": {"thr_long": 0.0003, "thr_short": -0.0003}}},
        {"label": "A2.funding_long10bp_short5bp", "overlay": {
            "funding_filter": {"thr_long": 0.0010, "thr_short": -0.0005}}},

        # A3 — cooldown
        {"label": "A3.cooldown_-3pct_skip2", "overlay": {
            "cooldown": {"loss_thr": -0.03, "n_skip": 2}}},
        {"label": "A3.cooldown_-2pct_skip1", "overlay": {
            "cooldown": {"loss_thr": -0.02, "n_skip": 1}}},
        {"label": "A3.cooldown_-5pct_skip3", "overlay": {
            "cooldown": {"loss_thr": -0.05, "n_skip": 3}}},

        # Combos
        {"label": "A1+A2 kelly+funding", "overlay": {
            "asymm_kelly": {"trend_thr": 0.3, "weak_side_scale": 0.5},
            "funding_filter": {"thr_long": 0.0005, "thr_short": -0.0003}}},
        {"label": "A1+A3 kelly+cooldown", "overlay": {
            "asymm_kelly": {"trend_thr": 0.3, "weak_side_scale": 0.5},
            "cooldown": {"loss_thr": -0.03, "n_skip": 2}}},
    ]


# ----------------------------- MAIN -------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-cache", action="store_true",
                    help="Force re-train and rebuild cache")
    ap.add_argument("--n-long", type=int, default=4)
    ap.add_argument("--n-short", type=int, default=2)
    ap.add_argument("--out", default="results_r128_overlay_sweep.json")
    args = ap.parse_args()

    if args.rebuild_cache or not (PREDS_PATH.exists() and REGIME_PATH.exists()):
        print("Building cache (training)...")
        preds, regime_df = build_cache()
    else:
        print(f"Loading cache from {PREDS_PATH}")
        preds, regime_df = load_cache()
    print(f"  preds: {len(preds):,} rows, "
          f"{preds['timestamp'].nunique()} timestamps, "
          f"{preds['symbol'].nunique()} symbols")

    funding_df = load_funding()
    if funding_df is not None:
        print(f"  funding loaded: {len(funding_df):,} rows, "
              f"{funding_df['symbol'].nunique()} symbols")
    else:
        print("  funding NOT available (will skip A2)")

    sweeps = define_sweeps()
    print(f"\n=== Running {len(sweeps)} configs (n_long={args.n_long}, n_short={args.n_short}) ===\n")
    rows = []
    baseline = None
    for cfg in sweeps:
        # Skip funding-dependent configs if no data
        if funding_df is None and cfg["overlay"] and "funding_filter" in cfg["overlay"]:
            print(f"  {cfg['label']:<40s}: SKIP (no funding data)")
            continue
        port = simulate_overlay(preds, regime_df, args.n_long, args.n_short,
                                overlay=cfg["overlay"], funding_df=funding_df)
        r = analyze(port, cfg["label"])
        if cfg["overlay"] is None:
            baseline = r
        if baseline is not None and r.get("net_sharpe") is not None:
            r["delta_sharpe"] = r["net_sharpe"] - baseline["net_sharpe"]
            r["delta_net"] = r["net_total"] - baseline["net_total"]
        rows.append(r)

    Path(args.out).write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nSaved → {args.out}")

    # Pretty summary
    print("\n=== SUMMARY (sorted by ΔSharpe) ===")
    sorted_rows = sorted([r for r in rows if "delta_sharpe" in r],
                         key=lambda x: x["delta_sharpe"], reverse=True)
    for r in sorted_rows:
        flag = " ★" if r["delta_sharpe"] > 0.05 else ""
        print(f"  {r['label']:<40s} S_n={r['net_sharpe']:+.3f} "
              f"Δ={r['delta_sharpe']:+.3f}  Δret={r['delta_net']:+.3f}{flag}")


if __name__ == "__main__":
    sys.exit(main() or 0)
