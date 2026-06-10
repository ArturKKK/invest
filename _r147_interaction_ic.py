#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
#  VM ONLY — heavy.  Calls r68.load_data() (full research frame, ~1.2M rows).
#  DO NOT run on the laptop (M3 Pro 18GB).  Designed for the MLC VM.
# ============================================================================
"""R147 — Interaction-feature IC screen (Track B).

Screens 25 INTERACTION candidates built ONLY from existing per-symbol columns
of the canonical research frame (r68.load_data()).  No new data sources, no
news / on-chain / social / market-level-constant features (all CLOSED per
MEGA_PROMPT final map).

Per candidate:
  * per-timestamp cross-sectional Spearman rank-IC vs fwd_ret_12h
    (vectorized: Pearson of within-timestamp ranks via groupby sums)
  * Newey-West(12) t-stat of the mean hourly IC (12h fwd-ret overlap makes the
    naive t inflated — same logic as _r143_pristine_oos.py)
  * 3 equal sub-window IC means + same-sign stability count
  * redundancy: max |corr| of the CS-ranked candidate vs the CS-ranked
    CHAMPION_FEAT_30 (sampled grid); flag redundant if > 0.7

Hygiene:
  * SCREEN_END = 2026-04-25 — the pristine OOS window (2026-04-26..06-08) is
    NOT touched, so it stays clean for later model-level validation.
  * cg_* features are NOT used as inputs (CoinGlass frozen 2026-05-06).
  * Per-timestamp Spearman is invariant to per-timestamp MONOTONE transforms,
    so sign(x)*sqrt(|x|)-style transforms of a single feature are pointless in
    an IC screen — only NON-monotone transforms (centered-rank squared =
    "extremity") and true 2-feature products carry new ranking information.
    Monotone transforms can still matter inside the tree model, but that is a
    model-level ablation, not an IC question.
  * R35b dead forms (oi_ret_divergence raw 12h, funding_ret_cross raw,
    vol_ret_confirm, ret_168h_x_disp) and the existing raw funding_x_mom_12h/24h
    columns are NOT re-proposed; all candidates below differ in form/horizon.

Output:
  * ranked table to stdout (sorted by |t_NW12|)
  * results_r147_interaction_ic.json

Usage (VM):
  python _r147_interaction_ic.py [--screen-end 2026-04-25] [--out results_r147_interaction_ic.json]
"""

import argparse
import gc
import json
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# r68 import runs _preflight_check at module level — VM only.
import _research_r68_continuous_wf as r68
from _research_r47_coinglass import CHAMPION_FEAT_30
from _research_r35_new_features import MARKET_LEVEL_FEATURES

# ── config ───────────────────────────────────────────────────────────────
SCREEN_END_DEFAULT = "2026-04-25"   # keep pristine OOS (2026-04-26..06-08) untouched
OUT_DEFAULT = "results_r147_interaction_ic.json"
TARGET = "fwd_ret_12h"
MIN_SYMS = 10          # min symbols per timestamp for a valid IC obs
NW_LAGS = 12           # Newey-West lags (12h fwd-ret overlap on hourly grid)
REDUNDANT_THRESH = 0.70
REDUNDANCY_GRID = 4    # sample every 4th hour for the redundancy matrix (RAM/CPU)
EPS = 1e-10

# t-thresholds for the verdict (25 candidates -> mild multiplicity; NW t>3 is
# the serious bar, t>2 is "weak, watch")
T_STRONG, T_PASS, T_WEAK = 4.0, 3.0, 2.0


# ── helpers ──────────────────────────────────────────────────────────────

def _nw_tstat(x, lags):
    """Newey-West (Bartlett) t-stat of the mean for an autocorrelated series.
    (copied from _r143_pristine_oos.py)"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 5:
        return np.nan
    d = x - x.mean()
    var = d @ d / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        var += 2.0 * w * (d[:-k] @ d[k:]) / n
    se = np.sqrt(max(var, 1e-18) / n)
    return x.mean() / (se + 1e-18)


def cs_ic_series(ts, x, y, min_syms=MIN_SYMS):
    """Per-timestamp cross-sectional Spearman IC of x vs y, vectorized.

    Spearman = Pearson of within-timestamp ranks (average-tie ranks, which is
    the tie-corrected Spearman).  Returns pd.Series indexed by timestamp.
    """
    sub = pd.DataFrame({"ts": ts.values, "x": x.values, "y": y.values}).dropna()
    if len(sub) == 0:
        return pd.Series(dtype=float)
    g = sub.groupby("ts")
    sub["rx"] = g["x"].rank()
    sub["ry"] = g["y"].rank()
    sub["rx2"] = sub["rx"] ** 2
    sub["ry2"] = sub["ry"] ** 2
    sub["rxy"] = sub["rx"] * sub["ry"]
    agg = sub.groupby("ts").agg(
        n=("rx", "size"), sx=("rx", "sum"), sy=("ry", "sum"),
        sxx=("rx2", "sum"), syy=("ry2", "sum"), sxy=("rxy", "sum"),
    )
    agg = agg[agg["n"] >= min_syms]
    num = agg["n"] * agg["sxy"] - agg["sx"] * agg["sy"]
    den2 = ((agg["n"] * agg["sxx"] - agg["sx"] ** 2)
            * (agg["n"] * agg["syy"] - agg["sy"] ** 2))
    ok = den2 > 0
    ic = num[ok] / np.sqrt(den2[ok])
    return ic.sort_index()


class RankCache:
    """Lazy per-timestamp CS pct-rank (centered): R(col) = rank_pct - 0.5.

    float32 to keep the cache small (~5MB per leg on ~1.2M rows).
    NaNs stay NaN (pandas groupby.rank leaves them).
    """

    def __init__(self, df):
        self.df = df
        self._cache = {}

    def __call__(self, col):
        if col not in self._cache:
            r = self.df.groupby("timestamp")[col].rank(pct=True) - 0.5
            self._cache[col] = r.astype(np.float32)
        return self._cache[col]


# ── candidate registry ───────────────────────────────────────────────────
# Each entry: name, family, inputs (existing frame columns), formula string,
# one-line economic rationale, fn(df, R) -> pd.Series.
# R(col) = per-timestamp centered CS pct-rank of col.

CANDIDATES = [
    # ── A. momentum x own-vol ────────────────────────────────────────────
    dict(
        name="vam_168h", family="mom_x_vol",
        inputs=["ret_168h", "rvol_168h"],
        formula="ret_168h / (rvol_168h*sqrt(168) + eps)",
        rationale="Sharpe-style weekly momentum: trend per unit of own risk persists; mom_z exists only at 12/24h.",
        fn=lambda df, R: df["ret_168h"] / (df["rvol_168h"] * np.sqrt(168) + EPS),
    ),
    dict(
        name="mom168_lowvol", family="mom_x_vol",
        inputs=["ret_168h", "rvol_24h"],
        formula="R(ret_168h) * (-R(rvol_24h))",
        rationale="Low-vol momentum premium: trends in quiet coins persist, high-vol movers mean-revert.",
        fn=lambda df, R: R("ret_168h") * (-R("rvol_24h")),
    ),
    dict(
        name="mom24_x_vov", family="mom_x_vol",
        inputs=["ret_24h", "vol_of_vol"],
        formula="R(ret_24h) * R(vol_of_vol)",
        rationale="Regime-fragile momentum: momentum built on unstable volatility is low quality and fades.",
        fn=lambda df, R: R("ret_24h") * R("vol_of_vol"),
    ),
    # ── B. funding x OI (crowded carry) ──────────────────────────────────
    dict(
        name="crowd_carry", family="funding_x_oi",
        inputs=["cum_funding_24h", "oi_chg_24h"],
        formula="R(cum_funding_24h) * R(oi_chg_24h)",
        rationale="Crowded-carry stress: high funding while OI grows = leveraged longs piling in, fragile to unwind.",
        fn=lambda df, R: R("cum_funding_24h") * R("oi_chg_24h"),
    ),
    dict(
        name="fund_oi_unwind", family="funding_x_oi",
        inputs=["funding_zscore", "oi_velocity"],
        formula="R(funding_zscore) * R(oi_velocity)",
        rationale="Funding extreme + OI acceleration = forced (de)leveraging stampede in progress.",
        fn=lambda df, R: R("funding_zscore") * R("oi_velocity"),
    ),
    dict(
        name="basis_stress", family="funding_x_basis",
        inputs=["funding_zscore", "premium_zscore"],
        formula="R(funding_zscore) * R(premium_zscore)",
        rationale="Both carry legs (funding + perp premium) stretched together = stronger crowding than either alone.",
        fn=lambda df, R: R("funding_zscore") * R("premium_zscore"),
    ),
    # ── C. taker flow x volume (informed flow) ───────────────────────────
    dict(
        name="informed_flow", family="flow_x_volume",
        inputs=["taker_cvd_12h", "rel_volume_cs"],
        formula="R(taker_cvd_12h) * R(rel_volume_cs)",
        rationale="Directional taker aggression on abnormal volume = informed trading; flow on thin volume is noise.",
        fn=lambda df, R: R("taker_cvd_12h") * R("rel_volume_cs"),
    ),
    dict(
        name="flow_absorption", family="flow_x_price",
        inputs=["taker_cvd_24h", "ret_24h"],
        formula="R(taker_cvd_24h) * (-R(ret_24h))",
        rationale="Heavy net taker buying without price progress = hidden supply absorbing it (and vice versa).",
        fn=lambda df, R: R("taker_cvd_24h") * (-R("ret_24h")),
    ),
    dict(
        name="flow_new_positions", family="flow_x_oi",
        inputs=["taker_cvd_12h", "oi_chg_12h"],
        formula="R(taker_cvd_12h) * R(oi_chg_12h)",
        rationale="Aggressive flow that opens new OI = conviction entries (continuation); flow closing OI = covering.",
        fn=lambda df, R: R("taker_cvd_12h") * R("oi_chg_12h"),
    ),
    # ── D. breakout quality (dist_from_high x skew / volume) ─────────────
    dict(
        name="breakout_skew", family="breakout_quality",
        inputs=["dist_from_high_24h", "ret_skew_168h"],
        formula="(-R(dist_from_high_24h)) * R(ret_skew_168h)",
        rationale="Near 24h-high with positive weekly skew = orderly breakout; negative skew near highs = blow-off top.",
        fn=lambda df, R: (-R("dist_from_high_24h")) * R("ret_skew_168h"),
    ),
    dict(
        name="breakout_volume", family="breakout_quality",
        inputs=["dist_from_high_24h", "rel_volume_cs"],
        formula="(-R(dist_from_high_24h)) * R(rel_volume_cs)",
        rationale="Breakout backed by cross-sectionally abnormal volume = genuine participation, not a thin-tape drift.",
        fn=lambda df, R: (-R("dist_from_high_24h")) * R("rel_volume_cs"),
    ),
    # ── E. basis-led speculation ─────────────────────────────────────────
    dict(
        name="perp_led_move", family="basis_x_price",
        inputs=["premium_zscore_12h", "ret_12h"],
        formula="R(premium_zscore_12h) * (-R(ret_12h))",
        rationale="Perp premium spiking without a spot move = derivative-led speculation that mean-reverts.",
        fn=lambda df, R: R("premium_zscore_12h") * (-R("ret_12h")),
    ),
    # ── F. idio vs beta momentum ─────────────────────────────────────────
    dict(
        name="idio_momentum", family="mom_x_btccorr",
        inputs=["ret_168h", "btc_corr_168h"],
        formula="R(ret_168h) * (-R(btc_corr_168h))",
        rationale="Momentum in BTC-decorrelated coins is own-narrative driven and persists; high-corr momentum is just beta.",
        fn=lambda df, R: R("ret_168h") * (-R("btc_corr_168h")),
    ),
    dict(
        name="corr_decoupling", family="mom_x_btccorr",
        inputs=["btc_corr_24h", "btc_corr_168h"],
        formula="btc_corr_24h - btc_corr_168h",
        rationale="Recent decoupling from BTC vs its own baseline = idiosyncratic event in progress.",
        fn=lambda df, R: df["btc_corr_24h"] - df["btc_corr_168h"],
    ),
    # ── G. positioning confirmation ──────────────────────────────────────
    dict(
        name="oi_conf_momentum", family="oi_x_price",
        inputs=["oi_zscore", "ret_12h"],
        formula="R(oi_zscore) * R(ret_12h)",
        rationale="Price move with stretched OI = positioning-confirmed move (continuation) or exhaustion at the extreme.",
        fn=lambda df, R: R("oi_zscore") * R("ret_12h"),
    ),
    dict(
        name="squeeze_fuel", family="positioning_x_price",
        inputs=["global_ls_ratio_zscore", "ret_12h"],
        formula="(-R(global_ls_ratio_zscore)) * R(ret_12h)",
        rationale="Crowd short (low global L/S z) while price rises = short-squeeze fuel for continuation.",
        fn=lambda df, R: (-R("global_ls_ratio_zscore")) * R("ret_12h"),
    ),
    # ── H. nonlinear (non-monotone only; monotone == same Spearman) ──────
    dict(
        name="ret24_extremity", family="nonlinear",
        inputs=["ret_24h"],
        formula="R(ret_24h)^2",
        rationale="U-shape: extreme movers in either direction (lottery/attention coins) behave unlike the middle of the pack.",
        fn=lambda df, R: R("ret_24h") ** 2,
    ),
    dict(
        name="funding_extremity", family="nonlinear",
        inputs=["cum_funding_24h"],
        formula="R(cum_funding_24h)^2",
        rationale="Extreme funding of either sign = crowding; squeeze risk is two-sided, linear funding misses the short side.",
        fn=lambda df, R: R("cum_funding_24h") ** 2,
    ),
    dict(
        name="skew_kurt_lottery", family="nonlinear",
        inputs=["ret_skew_24h", "ret_kurt_24h"],
        formula="R(ret_skew_24h) * R(ret_kurt_24h)",
        rationale="Fat tails + positive skew = lottery-preference coin, retail-overbought and underperforms.",
        fn=lambda df, R: R("ret_skew_24h") * R("ret_kurt_24h"),
    ),
    # ── I. vol structure ─────────────────────────────────────────────────
    dict(
        name="vol_term_structure", family="vol_structure",
        inputs=["rvol_12h", "rvol_168h"],
        formula="rvol_12h / (rvol_168h + eps)",
        rationale="Short vol above own long vol = event in progress; quiet-vs-self coins carry the carry.",
        fn=lambda df, R: df["rvol_12h"] / (df["rvol_168h"] + EPS),
    ),
    dict(
        name="upvol_share", family="vol_structure",
        inputs=["upvol_24h", "rvol_24h"],
        formula="upvol_24h / (rvol_24h + eps)",
        rationale="Share of volatility realized on the upside = accumulation vs distribution asymmetry.",
        fn=lambda df, R: df["upvol_24h"] / (df["rvol_24h"] + EPS),
    ),
    # ── J. carry-aware momentum ──────────────────────────────────────────
    dict(
        name="uncrowded_momentum", family="mom_x_funding",
        inputs=["ret_48h", "cum_funding_72h"],
        formula="R(ret_48h) * (-R(cum_funding_72h))",
        rationale="Momentum not yet paid for via funding has room to run; high-funding momentum is late-stage.",
        fn=lambda df, R: R("ret_48h") * (-R("cum_funding_72h")),
    ),
    # ── K. liquidity reversal / flow persistence / candle demand ────────
    dict(
        name="illiq_reversal", family="reversal_x_volume",
        inputs=["ret_12h", "rel_volume_cs"],
        formula="(-R(ret_12h)) * (-R(rel_volume_cs))",
        rationale="Short-horizon reversal concentrates in low-attention/low-volume names (liquidity-provision premium).",
        fn=lambda df, R: (-R("ret_12h")) * (-R("rel_volume_cs")),
    ),
    dict(
        name="flow_trend_persist", family="flow_x_autocorr",
        inputs=["taker_cvd_24h", "ret_autocorr_24h"],
        formula="R(taker_cvd_24h) * R(ret_autocorr_24h)",
        rationale="Taker flow in positively autocorrelated (trending) coins keeps pushing price the same way.",
        fn=lambda df, R: R("taker_cvd_24h") * R("ret_autocorr_24h"),
    ),
    dict(
        name="close_strength_vol", family="flow_x_volume",
        inputs=["buy_pressure", "rel_volume_cs"],
        formula="R(buy_pressure) * R(rel_volume_cs)",
        rationale="Closes near the highs on abnormal relative volume = real demand into the close, not drift.",
        fn=lambda df, R: R("buy_pressure") * R("rel_volume_cs"),
    ),
]

# Reference (existing champion) features evaluated identically — calibrates
# what a "good" NW t looks like on this exact screen.
REFERENCE_FEATURES = ["cs_rank_ma_5", "taker_cvd_12h", "cum_funding_24h", "ret_12h"]


# ── main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen-end", default=SCREEN_END_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    print("=" * 96)
    print(f"  R147 — INTERACTION IC SCREEN | {len(CANDIDATES)} candidates | "
          f"screen window: data start .. {args.screen_end} (pristine OOS untouched)")
    print("=" * 96)

    df, _regime = r68.load_data()
    del _regime
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[df["timestamp"] <= pd.Timestamp(args.screen_end, tz="UTC")]

    # Keep only the columns we need — frees ~1GB vs the full frame.
    input_cols = sorted({c for cand in CANDIDATES for c in cand["inputs"]})
    champ_cols = [f for f in CHAMPION_FEAT_30 if f in df.columns]
    keep = sorted(set(["timestamp", "symbol", TARGET] + input_cols + champ_cols
                      + REFERENCE_FEATURES) & set(df.columns))
    missing_inputs_global = [c for c in input_cols if c not in df.columns]
    df = df[keep].copy()
    gc.collect()
    print(f"  Frame: {len(df):,} rows, {df['symbol'].nunique()} symbols, "
          f"{df['timestamp'].min()} .. {df['timestamp'].max()}")
    if missing_inputs_global:
        print(f"  WARNING missing input columns: {missing_inputs_global}")

    R = RankCache(df)

    # ── redundancy base: CS-ranked champion features on a sampled grid ───
    # Market-level features (constant per timestamp) degenerate under CS-rank
    # (all tied -> zero variance) and are skipped automatically by corr NaN.
    grid_mask = (df["timestamp"].dt.hour % REDUNDANCY_GRID) == 0
    red_feats = [f for f in champ_cols if f not in MARKET_LEVEL_FEATURES]
    red_base = {}
    for f in red_feats:
        red_base[f] = R(f)[grid_mask]
    red_base = pd.DataFrame(red_base)
    print(f"  Redundancy base: {len(red_feats)} CS-ranked champion features "
          f"on {grid_mask.sum():,} sampled rows (every {REDUNDANCY_GRID}h)")

    def evaluate(name, series, family, formula, rationale, inputs):
        ic = cs_ic_series(df["timestamp"], series, df[TARGET])
        if len(ic) < 100:
            return dict(name=name, family=family, formula=formula,
                        rationale=rationale, inputs=inputs, status="NO_DATA",
                        n_ts=int(len(ic)))
        mean_ic = float(ic.mean())
        t_nw = float(_nw_tstat(ic.values, NW_LAGS))
        thirds = [float(np.mean(part)) for part in np.array_split(ic.values, 3)]
        same_sign = int(sum(np.sign(t) == np.sign(mean_ic) for t in thirds))
        # redundancy vs champion features (CS-ranked, sampled grid)
        cand_rank = (series.groupby(df["timestamp"]).rank(pct=True) - 0.5
                     ).astype(np.float32)[grid_mask]
        corrs = red_base.corrwith(cand_rank).abs().dropna()
        if len(corrs):
            max_corr = float(corrs.max())
            max_corr_feat = str(corrs.idxmax())
        else:
            max_corr, max_corr_feat = np.nan, ""
        redundant = bool(max_corr > REDUNDANT_THRESH) if np.isfinite(max_corr) else False
        at = abs(t_nw)
        if redundant:
            verdict = "REDUNDANT"
        elif at >= T_STRONG and same_sign == 3:
            verdict = "STRONG"
        elif at >= T_PASS and same_sign == 3:
            verdict = "PASS"
        elif at >= T_WEAK and same_sign >= 2:
            verdict = "WEAK"
        else:
            verdict = "DEAD"
        return dict(
            name=name, family=family, formula=formula, rationale=rationale,
            inputs=inputs, status="OK",
            n_ts=int(len(ic)), n_obs=int(series.notna().sum()),
            mean_ic=round(mean_ic, 5), t_nw12=round(t_nw, 2),
            ic_thirds=[round(t, 5) for t in thirds], same_sign_thirds=same_sign,
            max_abs_corr=round(max_corr, 3) if np.isfinite(max_corr) else None,
            max_corr_feat=max_corr_feat, redundant=redundant, verdict=verdict,
        )

    results = []

    # references first (calibration)
    for f in REFERENCE_FEATURES:
        if f not in df.columns:
            continue
        res = evaluate(f"REF_{f}", df[f], "reference", f"raw {f}",
                       "existing champion feature — calibration baseline", [f])
        results.append(res)
        print(f"  [ref ] {res['name']:<26} IC={res.get('mean_ic', float('nan')):+.4f} "
              f"t_NW12={res.get('t_nw12', float('nan')):+.2f}")
        gc.collect()

    for i, cand in enumerate(CANDIDATES, 1):
        miss = [c for c in cand["inputs"] if c not in df.columns]
        if miss:
            results.append(dict(name=cand["name"], family=cand["family"],
                                formula=cand["formula"], rationale=cand["rationale"],
                                inputs=cand["inputs"], status="MISSING_INPUTS",
                                missing=miss))
            print(f"  [{i:>2}/{len(CANDIDATES)}] {cand['name']:<26} SKIP missing {miss}")
            continue
        series = cand["fn"](df, R)
        res = evaluate(cand["name"], series, cand["family"], cand["formula"],
                       cand["rationale"], cand["inputs"])
        results.append(res)
        if res["status"] == "OK":
            print(f"  [{i:>2}/{len(CANDIDATES)}] {res['name']:<26} "
                  f"IC={res['mean_ic']:+.4f}  t_NW12={res['t_nw12']:+.2f}  "
                  f"thirds={res['same_sign_thirds']}/3  "
                  f"max|corr|={res['max_abs_corr']} ({res['max_corr_feat']})  "
                  f"-> {res['verdict']}")
        else:
            print(f"  [{i:>2}/{len(CANDIDATES)}] {res['name']:<26} {res['status']}")
        del series
        gc.collect()

    # ── ranked table ─────────────────────────────────────────────────────
    ok = [r for r in results if r.get("status") == "OK" and r["family"] != "reference"]
    ok.sort(key=lambda r: -abs(r["t_nw12"]))
    print("\n" + "=" * 96)
    print("  RANKED TABLE (by |t_NW12|, screen window only — NOT OOS)")
    print("=" * 96)
    hdr = (f"  {'rank':<4} {'name':<26} {'family':<20} {'IC':>8} {'t_NW12':>7} "
           f"{'3/3':>4} {'max|corr|':>9} {'vs':<20} {'verdict':<10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for rank, r in enumerate(ok, 1):
        print(f"  {rank:<4} {r['name']:<26} {r['family']:<20} "
              f"{r['mean_ic']:>+8.4f} {r['t_nw12']:>+7.2f} "
              f"{r['same_sign_thirds']:>3}/3 {str(r['max_abs_corr']):>9} "
              f"{r['max_corr_feat']:<20} {r['verdict']:<10}")

    n_strong = sum(1 for r in ok if r["verdict"] == "STRONG")
    n_pass = sum(1 for r in ok if r["verdict"] == "PASS")
    n_weak = sum(1 for r in ok if r["verdict"] == "WEAK")
    print(f"\n  STRONG={n_strong}  PASS={n_pass}  WEAK={n_weak}  "
          f"DEAD/REDUNDANT={len(ok) - n_strong - n_pass - n_weak}")
    print("  Next step for STRONG/PASS: model-level ablation (add to CHAMPION_FEAT_31,"
          " retrain WF) and only then pristine OOS.")

    payload = dict(
        meta=dict(
            run="R147_interaction_ic",
            generated_utc=datetime.now(timezone.utc).isoformat(),
            screen_end=args.screen_end,
            target=TARGET, min_syms=MIN_SYMS, nw_lags=NW_LAGS,
            redundant_thresh=REDUNDANT_THRESH,
            t_strong=T_STRONG, t_pass=T_PASS, t_weak=T_WEAK,
            n_rows=int(len(df)), n_symbols=int(df["symbol"].nunique()),
            data_start=str(df["timestamp"].min()),
            data_end=str(df["timestamp"].max()),
            note="screen window excludes pristine OOS 2026-04-26..06-08; "
                 "Spearman is monotone-invariant per timestamp, so only "
                 "non-monotone transforms / true interactions were screened",
        ),
        results=results,
    )
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n  Saved -> {args.out}")


if __name__ == "__main__":
    main()
