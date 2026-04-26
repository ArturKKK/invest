"""R128 ALL overlays on CANONICAL 3.777 baseline (cef6e2f simulate).

Tests yesterday's 6 ideas on the correct baseline:
  A1 asymmetric kelly        — VM-confirmed +0.386 (4L/2S)
  A2 funding filter          — was REJECTED on wrong baseline; retest
  A3 cooldown                — was neutral on wrong baseline; retest
  A4 cost-aware threshold    — was marginal on wrong baseline; retest
  F2 confidence-skip         — was REJECTED on wrong baseline; retest
  G2 vol-weighted (REAL)     — was look-ahead; retest with past-only realized vol

Plus combos: A1+each.

Uses cef6e2f simulate (skip risk-off, n=688) — matches r68.simulate on VM and
matches canonical 4L/2S Net Sharpe 3.777.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
VOL_PATH = CACHE_DIR / "r128_realized_vol.parquet"
FUNDING_PATH = Path("data/sentiment/binance_funding_rates.parquet")

PROD_CFG = r68.PROD_CFG
_cost_for_sym = r68._cost_for_sym
TIER1_SYMS = getattr(r68, "TIER1_SYMS", set())
TIER3_SYMS = getattr(r68, "TIER3_SYMS", set())


# ----------------------------- DATA -------------------------------------

def build_or_load_cache() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if PREDS_PATH.exists() and REGIME_PATH.exists():
        print(f"  Loading cached preds from {PREDS_PATH}")
        preds = pd.read_parquet(PREDS_PATH)
        rdf = pd.read_parquet(REGIME_PATH).set_index("timestamp")
        return preds, rdf
    print("  Building cache (training ~12 min)...")
    df, regime_df = r68.load_data()
    feats = [f for f in r68.CHAMPION_FEAT_31 if f in df.columns]
    no_rank = set(getattr(r68, "MARKET_LEVEL_FEATURES", []))
    preds = r68.train_ensemble(df, feats, r68.CONTINUOUS_WINDOWS,
                                seeds=r68.SEEDS, cs_rank_exclude=no_rank)
    preds.to_parquet(PREDS_PATH)
    regime_df.reset_index().to_parquet(REGIME_PATH)
    return preds, regime_df


def build_or_load_realized_vol(syms: List[str]) -> pd.DataFrame:
    """Compute past-only realized vol per (ts, symbol) using close-to-close
    returns from raw OHLCV. NO look-ahead.

    Returns DataFrame with columns: timestamp, symbol, rv_24h, rv_72h.
    """
    if VOL_PATH.exists():
        print(f"  Loading realized vol cache: {VOL_PATH}")
        return pd.read_parquet(VOL_PATH)
    print("  Computing past-only realized vol from raw OHLCV...")
    rows = []
    for sym in syms:
        # Convert "BTC/USDT" -> "BTC_USDT" for filename lookup
        fname_sym = sym.replace("/", "_")
        path = Path(f"data/raw/{fname_sym}_1h.parquet")
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["ret"] = df["close"].pct_change()
        # Past-only rolling std at hour t uses returns through t-1 (shift+rolling)
        df["rv_24h"] = df["ret"].shift(1).rolling(24, min_periods=12).std()
        df["rv_72h"] = df["ret"].shift(1).rolling(72, min_periods=36).std()
        df["symbol"] = sym  # keep original "BTC/USDT" for join
        rows.append(df[["timestamp", "symbol", "rv_24h", "rv_72h"]])
    if not rows:
        print("  ! No raw OHLCV matched; G2 will be skipped")
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(VOL_PATH)
    print(f"  Saved realized vol: {len(out):,} rows, {out['symbol'].nunique()} syms")
    return out


def load_funding() -> Optional[pd.DataFrame]:
    if not FUNDING_PATH.exists():
        return None
    f = pd.read_parquet(FUNDING_PATH)
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


# ----------------------------- SIMULATE ---------------------------------

def simulate_full(
    merged: pd.DataFrame,
    regime_df: pd.DataFrame,
    n_long: int,
    n_short: int,
    cfg: Dict[str, Any] = None,
    *,
    overlay: Optional[Dict[str, Any]] = None,
    funding_lookup: Optional[Dict] = None,
    vol_lookup: Optional[Dict] = None,
) -> pd.DataFrame:
    """cef6e2f-style simulate (SKIP risk-off, n=688) + overlays.

    overlay = {
      'a1':       {trend_thr, weak_scale}
      'a2':       {thr_long, thr_short}                   # funding
      'a3':       {loss_thr, n_skip}                      # cooldown
      'a4':       {tier1_min_pct, tier2_min_pct, tier3_min_pct}  # require pred percentile gap
      'f2':       {min_edge}                              # skip if |raw_prob - 0.5| < min_edge
      'g2':       {window: '24h'|'72h', cap: float|None}  # vol-weighted weights
    }
    """
    cfg = cfg or PROD_CFG
    overlay = overlay or {}
    trend_cutoff = cfg["trend_cutoff"]
    rebal_hours = cfg["rebal_hours"]
    ema_alpha = cfg.get("ema_alpha", None)
    hysteresis = cfg.get("hysteresis", 0)
    funding_per_12h = 0.00008

    a1 = overlay.get("a1")
    a2 = overlay.get("a2")
    a3 = overlay.get("a3")
    a4 = overlay.get("a4")
    f2 = overlay.get("f2")
    g2 = overlay.get("g2")

    all_rets: List[Dict] = []
    prev_longs: Set[str] = set()
    prev_shorts: Set[str] = set()
    prev_preds: Dict[str, float] = {}
    cooldown_until: Dict[str, pd.Timestamp] = {}

    timestamps_sorted = sorted(merged["timestamp"].unique())
    grouped = {ts: grp for ts, grp in merged.groupby("timestamp")}
    rebal_timestamps = timestamps_sorted[::rebal_hours]

    for i, ts in enumerate(rebal_timestamps):
        if ts not in regime_df.index or ts not in grouped:
            continue
        row = regime_df.loc[ts]
        trend_str = row.get("trend_strength", 0)
        trend_dir = row.get("trend_direction", 0) if "trend_direction" in row else 0
        # cef6e2f path: SKIP risk-off (no zero-row record, no prev reset)
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

        # ── F2 confidence filter (skip symbols whose raw_prob is too close to 0.5)
        if f2 is not None and "raw_prob" in grp.columns:
            min_edge = f2.get("min_edge", 0.05)
            grp = grp[(grp["raw_prob"] - 0.5).abs() >= min_edge]
            if len(grp) < 4:
                continue
            n = len(grp)
            nl = min(n_long, n // 3)
            ns = min(n_short, n // 3)

        grp["pred_rank"] = grp["pred"].rank(ascending=False)

        # ── A3 cooldown: drop symbols still in cooldown
        if a3 is not None and cooldown_until:
            grp = grp[~grp["symbol"].isin(
                {s for s, until in cooldown_until.items() if until > ts}
            )]
            if len(grp) < 4:
                continue
            n = len(grp)
            nl = min(n_long, n // 3)
            ns = min(n_short, n // 3)
            grp["pred_rank"] = grp["pred"].rank(ascending=False)

        # ── A2 funding filter
        if a2 is not None and funding_lookup is not None:
            thr_long = a2.get("thr_long", 0.0005)
            thr_short = a2.get("thr_short", -0.0003)
            def _f_at(sym):
                s = funding_lookup.get(sym)
                if s is None or len(s) == 0:
                    return 0.0
                idx = s.index.searchsorted(ts) - 1
                if idx < 0:
                    return 0.0
                return float(s.iloc[idx])
            grp["funding"] = grp["symbol"].map(_f_at)
            grp["ok_long"] = grp["funding"] <= thr_long
            grp["ok_short"] = grp["funding"] >= thr_short
        else:
            grp["ok_long"] = True
            grp["ok_short"] = True

        # ── A4 cost-aware percentile threshold (per-tier)
        if a4 is not None:
            grp["pred_pct"] = grp["pred"].rank(pct=True)
            def _tier_min(sym):
                if sym in TIER1_SYMS: return a4.get("tier1_min_pct", 0.0)
                if sym in TIER3_SYMS: return a4.get("tier3_min_pct", 0.0)
                return a4.get("tier2_min_pct", 0.0)
            grp["tmin"] = grp["symbol"].map(_tier_min)
            grp.loc[grp["pred_pct"] < (1 - grp["tmin"]), "ok_long"] = False
            grp.loc[grp["pred_pct"] > grp["tmin"], "ok_short"] = False

        # ── Selection (with hysteresis)
        if hysteresis > 0 and (prev_longs or prev_shorts):
            new_longs: Set[str] = set()
            new_shorts: Set[str] = set()
            for idx, r in grp.iterrows():
                sym, rank = r["symbol"], r["pred_rank"]
                if sym in prev_longs and rank <= nl + hysteresis and r["ok_long"]:
                    new_longs.add(sym)
                elif sym in prev_shorts and rank > (n - ns - hysteresis) and r["ok_short"]:
                    new_shorts.add(sym)
            cand_l = grp[(~grp["symbol"].isin(new_longs | new_shorts)) & grp["ok_long"]]
            for _, r in cand_l.sort_values("pred_rank").head(nl - len(new_longs)).iterrows():
                new_longs.add(r["symbol"])
            cand_s = grp[(~grp["symbol"].isin(new_longs | new_shorts)) & grp["ok_short"]]
            for _, r in cand_s.sort_values("pred_rank", ascending=False).head(ns - len(new_shorts)).iterrows():
                new_shorts.add(r["symbol"])
        else:
            elig_l = grp[grp["ok_long"]].sort_values("pred_rank").head(nl)
            elig_s = grp[grp["ok_short"]].sort_values("pred_rank", ascending=False).head(ns)
            new_longs = set(elig_l["symbol"].tolist()) if nl > 0 else set()
            new_shorts = set(elig_s["symbol"].tolist()) if ns > 0 else set()

        new_opened = (new_longs - prev_longs) | (new_shorts - prev_shorts)
        closed = (prev_longs - new_longs) | (prev_shorts - new_shorts)
        total_positions = len(new_longs) + len(new_shorts)

        # ── Per-symbol weight computation
        nl_act, ns_act = len(new_longs), len(new_shorts)
        longs_df = grp[grp["symbol"].isin(new_longs)]
        shorts_df = grp[grp["symbol"].isin(new_shorts)]

        # G2: vol-weighted (1/realized_vol), past-only
        if g2 is not None and vol_lookup is not None and (nl_act + ns_act) > 0:
            vol_col = f"rv_{g2.get('window', '24h')}"
            cap = g2.get("cap", None)
            def _w(sym):
                v = vol_lookup.get(sym)
                if v is None or v.empty:
                    return 1.0
                idx = v.index.searchsorted(ts) - 1
                if idx < 0:
                    return 1.0
                rv = float(v.iloc[idx][vol_col]) if isinstance(v.iloc[idx], pd.Series) else float(v.iloc[idx])
                if not np.isfinite(rv) or rv <= 0:
                    return 1.0
                w = 1.0 / rv
                if cap is not None:
                    w = min(w, cap)
                return w
            long_ws = {s: _w(s) for s in new_longs}
            short_ws = {s: _w(s) for s in new_shorts}
            tot_l = sum(long_ws.values()) or 1.0
            tot_s = sum(short_ws.values()) or 1.0
            long_ret = sum(long_ws[s] / tot_l * float(longs_df[longs_df["symbol"] == s]["fwd_ret"].iloc[0])
                          for s in new_longs) if nl_act > 0 else 0.0
            short_ret = sum(short_ws[s] / tot_s * float(shorts_df[shorts_df["symbol"] == s]["fwd_ret"].iloc[0])
                           for s in new_shorts) if ns_act > 0 else 0.0
        else:
            long_ret = longs_df["fwd_ret"].mean() if nl_act > 0 else 0.0
            short_ret = shorts_df["fwd_ret"].mean() if ns_act > 0 else 0.0

        # A1 asymmetric kelly weights between L and S sides
        if nl_act > 0 and ns_act > 0:
            w_l, w_s = 0.5, 0.5
            if a1 is not None:
                trend_thr = a1.get("trend_thr", 0.25)
                scale = a1.get("weak_scale", 0.6)
                if trend_dir > trend_thr:
                    w_s *= scale
                elif trend_dir < -trend_thr:
                    w_l *= scale
                tot = w_l + w_s
                w_l /= tot
                w_s /= tot
            gross_ret = w_l * long_ret - w_s * short_ret
        elif ns_act > 0:
            gross_ret = -short_ret
        elif nl_act > 0:
            gross_ret = long_ret
        else:
            continue
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

        # A3 cooldown bookkeeping
        if a3 is not None:
            loss_thr = a3.get("loss_thr", -0.03)
            n_skip = int(a3.get("n_skip", 2))
            for sym in new_longs:
                fr = float(longs_df[longs_df["symbol"] == sym]["fwd_ret"].iloc[0]) if (longs_df["symbol"] == sym).any() else 0.0
                if fr < loss_thr:
                    cooldown_until[sym] = ts + pd.Timedelta(hours=rebal_hours * (n_skip + 1))
            for sym in new_shorts:
                fr = float(shorts_df[shorts_df["symbol"] == sym]["fwd_ret"].iloc[0]) if (shorts_df["symbol"] == sym).any() else 0.0
                if -fr < loss_thr:
                    cooldown_until[sym] = ts + pd.Timedelta(hours=rebal_hours * (n_skip + 1))

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


# ----------------------------- SWEEPS -----------------------------------

def define_sweeps() -> List[Dict[str, Any]]:
    A1_BEST = {"trend_thr": 0.25, "weak_scale": 0.60}  # 4L/2S best from grid
    return [
        # ── Baseline
        {"label": "BASELINE", "overlay": None},

        # ── A1 (already known winner; sanity)
        {"label": "A1.kelly thr0.25 scale0.60", "overlay": {"a1": A1_BEST}},

        # ── A2 funding filter
        {"label": "A2.funding L5bp S-3bp",
         "overlay": {"a2": {"thr_long": 0.0005, "thr_short": -0.0003}}},
        {"label": "A2.funding L3bp S-3bp",
         "overlay": {"a2": {"thr_long": 0.0003, "thr_short": -0.0003}}},
        {"label": "A2.funding L10bp S-5bp",
         "overlay": {"a2": {"thr_long": 0.0010, "thr_short": -0.0005}}},

        # ── A3 cooldown
        {"label": "A3.cool -3% skip2",
         "overlay": {"a3": {"loss_thr": -0.03, "n_skip": 2}}},
        {"label": "A3.cool -2% skip1",
         "overlay": {"a3": {"loss_thr": -0.02, "n_skip": 1}}},
        {"label": "A3.cool -5% skip3",
         "overlay": {"a3": {"loss_thr": -0.05, "n_skip": 3}}},

        # ── A4 cost-aware percentile threshold
        {"label": "A4.cost t1=0 t2=0.05 t3=0.15",
         "overlay": {"a4": {"tier1_min_pct": 0.0, "tier2_min_pct": 0.05, "tier3_min_pct": 0.15}}},
        {"label": "A4.cost t1=0 t2=0.10 t3=0.20",
         "overlay": {"a4": {"tier1_min_pct": 0.0, "tier2_min_pct": 0.10, "tier3_min_pct": 0.20}}},
        {"label": "A4.cost t1=0.05 t2=0.10 t3=0.20",
         "overlay": {"a4": {"tier1_min_pct": 0.05, "tier2_min_pct": 0.10, "tier3_min_pct": 0.20}}},

        # ── F2 confidence skip
        {"label": "F2.conf min_edge=0.02", "overlay": {"f2": {"min_edge": 0.02}}},
        {"label": "F2.conf min_edge=0.05", "overlay": {"f2": {"min_edge": 0.05}}},
        {"label": "F2.conf min_edge=0.10", "overlay": {"f2": {"min_edge": 0.10}}},

        # ── G2 vol-weighted (REAL past-only realized vol)
        {"label": "G2.vol 24h cap=None", "overlay": {"g2": {"window": "24h"}}},
        {"label": "G2.vol 72h cap=None", "overlay": {"g2": {"window": "72h"}}},
        {"label": "G2.vol 24h cap=8", "overlay": {"g2": {"window": "24h", "cap": 8.0}}},

        # ── Combos with A1 (the proven winner)
        {"label": "A1+A2 L5bp S-3bp",
         "overlay": {"a1": A1_BEST, "a2": {"thr_long": 0.0005, "thr_short": -0.0003}}},
        {"label": "A1+A3 -3% skip2",
         "overlay": {"a1": A1_BEST, "a3": {"loss_thr": -0.03, "n_skip": 2}}},
        {"label": "A1+A4 t2=0.05 t3=0.15",
         "overlay": {"a1": A1_BEST,
                     "a4": {"tier1_min_pct": 0.0, "tier2_min_pct": 0.05, "tier3_min_pct": 0.15}}},
        {"label": "A1+F2 me=0.02",
         "overlay": {"a1": A1_BEST, "f2": {"min_edge": 0.02}}},
        {"label": "A1+G2 24h",
         "overlay": {"a1": A1_BEST, "g2": {"window": "24h"}}},
        {"label": "A1+G2 72h",
         "overlay": {"a1": A1_BEST, "g2": {"window": "72h"}}},
    ]


def main():
    t0 = time.time()
    print("=" * 78)
    print("  R128 ALL OVERLAYS — CANONICAL 3.777 BASELINE (cef6e2f simulate)")
    print("=" * 78)

    preds, regime_df = build_or_load_cache()
    syms = sorted(preds["symbol"].unique().tolist())
    print(f"  Predictions: {len(preds):,} rows, {len(syms)} symbols")

    funding_df = load_funding()
    funding_lookup = None
    if funding_df is not None:
        funding_lookup = {sym: g.set_index("timestamp")["funding_rate"].sort_index()
                          for sym, g in funding_df.groupby("symbol")}
        print(f"  Funding: {len(funding_df):,} rows, {len(funding_lookup)} syms")
    else:
        print("  Funding: NOT AVAILABLE (A2 will be skipped)")

    vol_df = build_or_load_realized_vol(syms)
    vol_lookup = None
    if not vol_df.empty:
        vol_lookup = {sym: g.set_index("timestamp")[["rv_24h", "rv_72h"]].sort_index()
                      for sym, g in vol_df.groupby("symbol")}
        print(f"  Realized vol: {len(vol_df):,} rows, {len(vol_lookup)} syms")

    sweeps = define_sweeps()
    print(f"\n  Total sweeps: {len(sweeps)}")

    rows: List[Dict[str, Any]] = []
    for nl, ns in [(4, 2), (6, 3)]:
        print(f"\n{'='*78}\n  CONFIG {nl}L/{ns}S\n{'='*78}")
        baseline_ns = None
        for cfg in sweeps:
            ov = cfg["overlay"] or {}
            if ("a2" in ov) and funding_lookup is None:
                print(f"  {cfg['label']:<40s}  SKIP (no funding)")
                continue
            if ("g2" in ov) and vol_lookup is None:
                print(f"  {cfg['label']:<40s}  SKIP (no vol)")
                continue
            port = simulate_full(preds, regime_df, nl, ns,
                                 overlay=cfg["overlay"],
                                 funding_lookup=funding_lookup,
                                 vol_lookup=vol_lookup)
            m = metrics(port)
            if cfg["overlay"] is None:
                baseline_ns = m["net_sharpe"]
            delta = m["net_sharpe"] - baseline_ns if baseline_ns is not None else 0.0
            flag = " ★" if delta > 0.05 else (" ✗" if delta < -0.05 else "")
            print(f"  {cfg['label']:<40s}  Net={m['net_sharpe']:+.3f}  "
                  f"Δ={delta:+.3f}  Gross={m['gross_sharpe']:+.3f}  "
                  f"Ret={m['ret_pct']:>6.1f}%  DD={m['dd_pct']:>5.1f}%  n={m['n']}{flag}")
            rows.append({"config": f"{nl}L/{ns}S", "label": cfg["label"], **m, "delta_sharpe": round(delta, 3)})

    df_res = pd.DataFrame(rows)
    out = "results_r128_all_overlays_canonical.csv"
    df_res.to_csv(out, index=False)
    print(f"\n  Saved: {out}")

    # Summary
    print("\n" + "=" * 78)
    print("  TOP WINNERS PER CONFIG (Δ vs baseline)")
    print("=" * 78)
    for cfg in ["4L/2S", "6L/3S"]:
        sub = df_res[df_res["config"] == cfg].sort_values("delta_sharpe", ascending=False)
        print(f"\n  --- {cfg} ---")
        for _, r in sub.head(8).iterrows():
            mark = "★" if r.delta_sharpe > 0.05 else ("✗" if r.delta_sharpe < -0.05 else " ")
            print(f"    {mark} {r.label:<40s}  Net={r.net_sharpe:+.3f}  Δ={r.delta_sharpe:+.3f}")
    print(f"\n  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
