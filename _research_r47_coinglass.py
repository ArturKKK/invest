#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R47 — CoinGlass Feature Research

Protocol:
  1. Load base research frame (FEAT_28 + R35 features = champion FEAT_30)
  2. Add CoinGlass daily features (with 1-day lag to avoid lookahead)
  3. IC scan on TRAIN data for each WF window
  4. Redundancy check vs FEAT_30 (correlation matrix)
  5. Walk-forward test per-feature, then best pairs/triple

CoinGlass feature priority:
  #1  Liquidations  (~70% probability of edge, nothing like it in FEAT_30)
  #2  Taker flow    (~30%, partial overlap with taker_cvd_12h from Binance)
  #3  L/S ratio     (~20%, overlaps with ls_divergence from Binance)
  #4  OI/Funding    (~10%, high redundancy with Binance data — last in queue)

Key QA findings (from _research_r47_qa.py):
  - Candle OPENS at timestamp, covers [t, t+24h) (confirmed by OI chaining)
  - Use CG(date-1d) for model at time t → shift(1) in daily feature merge
  - corr(liq_lag1d, |fwd_ret|) = +0.028 → positive raw IC!
  - MATIC + FTM excluded (no funding/ls_ratio)
  - Anomaly: Oct 10 2025 massive liquidation event ($1.87B BTC) — real event, keep

Usage:
  python _research_r47_coinglass.py                # full run: IC + WF
  python _research_r47_coinglass.py --ic-only      # only IC scan
  python _research_r47_coinglass.py --wf-only      # only WF (need prior IC results)
  python _research_r47_coinglass.py --quick         # single symbol smoke test
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── project imports ───────────────────────────────────────────

from _research_round7 import WINDOWS, SYM_35
from _research_r30b_fixed import (
    compute_regime_extended,
    eval_with_costs,
    simulate_with_costs,
    train_ensemble,
)
from _research_r33_creative_features import FEAT_28
from _research_r35_new_features import (
    MARKET_LEVEL_FEATURES,
    add_r35_features,
    load_research_frame,
)

# ── config ─────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).resolve().parent
CG_DIR     = BASE_DIR / "data" / "raw" / "coinglass"    # 1d interval

CANONICAL_EXEC_CFG = {
    "n_long": 6, "n_short": 3, "rebal_hours": 12,
    "trend_cutoff": 0.9, "dyn_threshold": 0.7,
    "ema_alpha": 0.5, "hysteresis": 3,
}

# Champion feature set (R42 winner)
CHAMPION_FEAT_30 = FEAT_28 + ["ret_dispersion_12h", "cs_rank_ma_5"]

# Symbols with complete CG data (exclude MATIC, FTM for funding/ls_ratio)
CG_FULL_SYMS = [s for s in SYM_35 if s not in ("MATIC", "FTM")]

# Rolling window for zscores / moving stats (in DAYS)
ROLL_30D = 30
ROLL_7D  = 7


# ─────────────────────────────────────────────────────────────
#  SECTION 1: CoinGlass feature builder
# ─────────────────────────────────────────────────────────────

def load_cg_daily() -> Dict[str, pd.DataFrame]:
    """Load all 5 CoinGlass endpoints into clean DataFrames."""
    eps = {}
    for name in ["liq", "oi", "taker", "funding", "ls_ratio"]:
        path = CG_DIR / f"{name}.parquet"
        if not path.exists():
            print(f"  ⚠️  CG {name} not found: {path}")
            continue
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        # Normalize to midnight UTC (handle any stray 16:00 UTC funding rows)
        df["cg_date"] = df["timestamp"].dt.normalize()
        # Drop duplicate (symbol, date) keeping last (handles funding multi-row per day)
        df = df.drop_duplicates(subset=["symbol", "cg_date"], keep="last")
        eps[name] = df
        print(f"  Loaded {name}: {len(df):,} rows, {df['symbol'].nunique()} symbols, "
              f"{str(df['cg_date'].min())[:10]} → {str(df['cg_date'].max())[:10]}")
    return eps


def compute_cg_features(cg: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build daily CG feature table from the 5 endpoints.
    All features are keyed by (symbol, cg_date) — one row per (symbol, day).

    The caller merges this into the 1h OHLCV frame using:
        cg_date = ohlcv_timestamp.dt.normalize() - 1 day   # shift-1 lookahead safety
    """
    liq = cg.get("liq", pd.DataFrame())
    taker = cg.get("taker", pd.DataFrame())
    ls  = cg.get("ls_ratio", pd.DataFrame())
    # oi  = cg.get("oi", pd.DataFrame())   # not used in first pass (redundant w/ Binance)
    # funding = cg.get("funding", pd.DataFrame())  # idem

    frames = []

    # ── Liquidations ─────────────────────────────────────────
    if not liq.empty:
        df = liq[["symbol", "cg_date", "liq_long_usd", "liq_short_usd"]].copy()
        eps = 1e-10

        df["cg_liq_total"]     = df["liq_long_usd"] + df["liq_short_usd"]
        df["cg_liq_imbalance"] = ((df["liq_long_usd"] - df["liq_short_usd"])
                                   / (df["cg_liq_total"] + eps))

        # Per-symbol rolling stats for zscore + acceleration
        df = df.sort_values(["symbol", "cg_date"])
        for sym, g in df.groupby("symbol"):
            roll_mean = g["cg_liq_total"].rolling(ROLL_30D, min_periods=ROLL_7D).mean()
            roll_std  = g["cg_liq_total"].rolling(ROLL_30D, min_periods=ROLL_7D).std() + eps
            df.loc[g.index, "cg_liq_zscore"] = (g["cg_liq_total"] - roll_mean) / roll_std
            df.loc[g.index, "cg_liq_accel"]  = (g["cg_liq_total"]
                                                  / (g["cg_liq_total"].shift(1) + eps) - 1)

        frames.append(df[["symbol", "cg_date",
                           "cg_liq_total", "cg_liq_imbalance",
                           "cg_liq_zscore", "cg_liq_accel"]].set_index(["symbol", "cg_date"]))

    # ── Taker flow ────────────────────────────────────────────
    if not taker.empty:
        df = taker[["symbol", "cg_date", "taker_buy_usd", "taker_sell_usd"]].copy()
        eps = 1e-10

        taker_sum = df["taker_buy_usd"] + df["taker_sell_usd"]
        df["cg_taker_imb"] = (df["taker_buy_usd"] - df["taker_sell_usd"]) / (taker_sum + eps)

        df = df.sort_values(["symbol", "cg_date"])
        for sym, g in df.groupby("symbol"):
            roll_mean = g["cg_taker_imb"].rolling(ROLL_30D, min_periods=ROLL_7D).mean()
            roll_std  = g["cg_taker_imb"].rolling(ROLL_30D, min_periods=ROLL_7D).std() + eps
            df.loc[g.index, "cg_taker_imb_z"] = (g["cg_taker_imb"] - roll_mean) / roll_std

        frames.append(df[["symbol", "cg_date",
                           "cg_taker_imb", "cg_taker_imb_z"]].set_index(["symbol", "cg_date"]))

    # ── L/S Ratio ─────────────────────────────────────────────
    if not ls.empty:
        df = ls[["symbol", "cg_date", "ls_ratio"]].copy()
        eps = 1e-10

        df["cg_ls_ratio"] = df["ls_ratio"]

        df = df.sort_values(["symbol", "cg_date"])
        for sym, g in df.groupby("symbol"):
            roll_mean = g["cg_ls_ratio"].rolling(ROLL_30D, min_periods=ROLL_7D).mean()
            roll_std  = g["cg_ls_ratio"].rolling(ROLL_30D, min_periods=ROLL_7D).std() + eps
            df.loc[g.index, "cg_ls_zscore"] = (g["cg_ls_ratio"] - roll_mean) / roll_std

        frames.append(df[["symbol", "cg_date",
                           "cg_ls_ratio", "cg_ls_zscore"]].set_index(["symbol", "cg_date"]))

    if not frames:
        return pd.DataFrame()

    feat_daily = frames[0]
    for f in frames[1:]:
        feat_daily = feat_daily.join(f, how="outer")
    feat_daily = feat_daily.reset_index()

    print(f"  CG feature table: {len(feat_daily):,} rows, {len(feat_daily.columns)} cols")
    return feat_daily


def add_cg_features(df: pd.DataFrame, cg_feats: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Merge CG daily features into the 1h model frame.

    Shift rule (from QA):
      candle at cg_date covers [cg_date, cg_date+1d)
      → available at cg_date + 1d 00:00 UTC
      → for ohlcv row at time t: use cg_date = floor(t, 'D') - 1 day
                                             = t.dt.normalize() - 1d

    After merging, adds per-timestamp market-level (mkt_*) aggregates.
    Returns:
      df                  — enriched model frame
      per_sym_cols        — list of per-coin CG features (CS-rank these)
      mkt_cols            — list of market-level CG features (DO NOT CS-rank)
    """
    if cg_feats.empty:
        return df, [], []

    df = df.copy()

    # Build the merge key
    df["_cg_date"] = (df["timestamp"].dt.normalize()
                      - pd.Timedelta(days=1))

    # CG feature sets
    per_sym_raw = [c for c in cg_feats.columns if c.startswith("cg_")]

    merged = df.merge(
        cg_feats.rename(columns={"cg_date": "_cg_date"}),
        on=["symbol", "_cg_date"],
        how="left",
    )
    merged = merged.drop(columns=["_cg_date"])

    # liq_intensity: liq_total / daily volume (need daily volume from OHLCV)
    if "cg_liq_total" in merged.columns and "volume" in merged.columns:
        # daily volume per (symbol, day) — build once, merge back
        merged["_vol_day"] = merged["timestamp"].dt.normalize()
        daily_vol = (merged.groupby(["symbol", "_vol_day"])["volume"]
                     .sum().rename("_daily_vol").reset_index())
        merged = merged.merge(daily_vol, on=["symbol", "_vol_day"], how="left")
        merged["cg_liq_intensity"] = (merged["cg_liq_total"]
                                       / (merged["_daily_vol"] + 1e-10))
        merged = merged.drop(columns=["_vol_day", "_daily_vol"])
        per_sym_raw.append("cg_liq_intensity")

    # Replace inf / extreme values
    for col in per_sym_raw:
        if col in merged.columns:
            merged[col] = merged[col].replace([np.inf, -np.inf], np.nan)

    # ── Market-level aggregates (mkt_* = NOT CS-ranked) ───────
    mkt_cols: List[str] = []

    if "cg_liq_total" in merged.columns:
        merged["mkt_cg_liq_total"] = (merged.groupby("timestamp")["cg_liq_total"]
                                       .transform("sum"))
        # log1p for stability
        merged["mkt_cg_liq_log"] = np.log1p(merged["mkt_cg_liq_total"])
        mkt_cols += ["mkt_cg_liq_total", "mkt_cg_liq_log"]

    if "cg_liq_imbalance" in merged.columns:
        merged["mkt_cg_liq_imb"] = (merged.groupby("timestamp")["cg_liq_imbalance"]
                                     .transform("mean"))
        mkt_cols.append("mkt_cg_liq_imb")

    for col in mkt_cols:
        merged[col] = merged[col].replace([np.inf, -np.inf], np.nan)

    per_sym_cols = [c for c in per_sym_raw if c in merged.columns]

    return merged, per_sym_cols, mkt_cols


# ─────────────────────────────────────────────────────────────
#  SECTION 2: IC scan helpers
# ─────────────────────────────────────────────────────────────

def compute_ic_by_period(df: pd.DataFrame, feat_col: str, target_col: str = "fwd_ret_12h",
                          freq: str = "W") -> Tuple[float, float, int]:
    """Return (mean_IC, ICIR, n_periods). Spearman, grouped by time period."""
    sub = df[[feat_col, target_col, "timestamp"]].dropna()
    if len(sub) < 200:
        return np.nan, np.nan, 0

    sub_idx = sub.set_index("timestamp")
    ic_series = sub_idx.resample(freq).apply(
        lambda x: stats.spearmanr(x[feat_col], x[target_col])[0]
        if len(x) > 15 else np.nan
    )
    ic_series = ic_series.dropna()

    if len(ic_series) < 4:
        return np.nan, np.nan, 0

    mean_ic = float(ic_series.mean())
    icir = mean_ic / (float(ic_series.std()) + 1e-10)
    return mean_ic, icir, len(ic_series)


def run_ic_scan(df: pd.DataFrame, cg_cols: List[str], windows: list) -> pd.DataFrame:
    """
    Compute rank IC / ICIR for each CG feature on the TRAIN subset of each WF window.
    Returns a DataFrame with columns: feature, window, ic, icir, n_weeks, flag.
    """
    rows = []
    for w in windows:
        train_end = pd.to_datetime(w["train_end"], utc=True)
        test_start = pd.to_datetime(w["test_start"], utc=True)
        # Purge: exclude data between train_end and test_start
        train_df = df[df["timestamp"] < (test_start - pd.Timedelta(days=8))].copy()
        train_df = train_df[train_df["timestamp"] <= train_end]

        if len(train_df) < 1000:
            print(f"  ⚠️  {w['name']}: too few train rows ({len(train_df)}), skipping")
            continue

        for feat in cg_cols:
            if feat not in train_df.columns:
                continue
            ic, icir, n = compute_ic_by_period(train_df, feat)
            rows.append({
                "feature": feat,
                "window": w["name"],
                "ic": ic,
                "icir": icir,
                "n_weeks": n,
            })

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    # Aggregate across windows
    agg = (result.groupby("feature")
           .agg(mean_ic=("ic", "mean"),
                abs_ic=("ic", lambda x: x.abs().mean()),
                mean_icir=("icir", "mean"),
                n_windows=("window", "nunique"))
           .reset_index()
           .sort_values("abs_ic", ascending=False))
    agg["flag"] = agg.apply(
        lambda r: "🔥" if abs(r["mean_ic"]) > 0.03 and abs(r["mean_icir"]) > 0.10
                   else ("✅" if abs(r["mean_ic"]) > 0.015 else "  "),
        axis=1
    )
    return agg


def run_redundancy_check(df: pd.DataFrame, cg_cols: List[str],
                          existing_feats: List[str]) -> pd.DataFrame:
    """
    Return pairwise Spearman correlation between each new CG feature
    and each existing FEAT_30 feature.
    High |r| > 0.5 → CG feature is redundant.
    """
    existing_present = [c for c in existing_feats if c in df.columns]
    rows = []
    for new_feat in cg_cols:
        if new_feat not in df.columns:
            continue
        for old_feat in existing_present:
            sub = df[[new_feat, old_feat]].dropna()
            if len(sub) < 200:
                continue
            r = float(stats.spearmanr(sub[new_feat], sub[old_feat])[0])
            if abs(r) > 0.3:
                rows.append({
                    "new_feat": new_feat,
                    "old_feat": old_feat,
                    "corr": r,
                    "flag": "⚠️ redundant" if abs(r) > 0.5 else "↔️ partial"
                })
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("corr", key=abs, ascending=False)
    return result


# ─────────────────────────────────────────────────────────────
#  SECTION 3: Event study
# ─────────────────────────────────────────────────────────────

def run_event_study(df: pd.DataFrame) -> None:
    """
    Top-1% by cg_liq_intensity → mean fwd_ret_12h/24h split by liq_imbalance sign.
    Helps validate that liquidation spikes precede directional moves.
    """
    if "cg_liq_intensity" not in df.columns or "fwd_ret_12h" not in df.columns:
        print("  No cg_liq_intensity or fwd_ret_12h, skipping event study")
        return

    top1 = df["cg_liq_intensity"].quantile(0.99)
    extreme = df[df["cg_liq_intensity"] >= top1].copy()

    if len(extreme) < 20:
        print(f"  ⚠️  Too few extreme events ({len(extreme)}), skipping event study")
        return

    extreme["sign"] = np.where(extreme.get("cg_liq_imbalance", pd.Series(0, index=extreme.index)) > 0,
                                "long_liq_dom", "short_liq_dom")

    print(f"\n  Event study: top-1% liq_intensity ({len(extreme)} events, threshold={top1:.1e})")
    print(f"\n  {'Sign':<18} {'n':>5} {'fwd_ret_12h_mean':>18} {'fwd_ret_12h_med':>16} {'fwd_ret_12h_std':>16}")
    print(f"  {'─'*18} {'─'*5} {'─'*18} {'─'*16} {'─'*16}")
    for sign, g in extreme.groupby("sign"):
        sub = g["fwd_ret_12h"].dropna()
        if len(sub) < 5:
            continue
        t_stat = stats.ttest_1samp(sub, 0).statistic
        print(f"  {sign:<18} {len(sub):>5} {sub.mean():>+17.4f}  {sub.median():>+15.4f}  {sub.std():>15.4f}"
              f"  (t={t_stat:+.2f})")

    # vs rest of universe
    rest = df[df["cg_liq_intensity"] < top1]["fwd_ret_12h"].dropna()
    print(f"  {'rest_of_univ':<18} {len(rest):>5} {rest.mean():>+17.4f}  {rest.median():>+15.4f}  {rest.std():>15.4f}")


# ─────────────────────────────────────────────────────────────
#  SECTION 4: Walk-forward test
# ─────────────────────────────────────────────────────────────

def make_feature_set(extra: Sequence[str],
                     mkt_cols: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Return (full_feature_list, cs_rank_exclude)."""
    feats = list(CHAMPION_FEAT_30)
    no_rank = [c for c in CHAMPION_FEAT_30 if c in MARKET_LEVEL_FEATURES]
    for f in extra:
        if f not in feats:
            feats.append(f)
    for f in extra:
        if f in mkt_cols and f not in no_rank:
            no_rank.append(f)
    return feats, no_rank


def evaluate_predictions(preds: pd.DataFrame,
                          regime_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for window in ["W1", "W2", "W3", "ALL"]:
        subset = preds if window == "ALL" else preds[preds["window"] == window].copy()
        port = simulate_with_costs(subset, regime_df, CANONICAL_EXEC_CFG)
        out[window] = eval_with_costs(port, window)
    return out


def run_wf_ablation(df: pd.DataFrame,
                     regime_df: pd.DataFrame,
                     candidate_features: List[str],
                     mkt_cols: List[str],
                     ic_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Sequential per-feature WF test (exact pattern from R42).
    Returns summary DataFrame sorted by ALL_sh desc.
    """
    # Order by IC if available, else original order
    if ic_df is not None and not ic_df.empty:
        ordered = ic_df[ic_df["feature"].isin(candidate_features)].sort_values("abs_ic", ascending=False)["feature"].tolist()
        # append any that weren't in ic_df
        for f in candidate_features:
            if f not in ordered:
                ordered.append(f)
    else:
        ordered = list(candidate_features)

    rows = []

    # Baseline
    print("\n  [baseline] Champion_30f ...")
    feats, no_rank = make_feature_set([], mkt_cols)
    preds = train_ensemble(df, feats, WINDOWS, l2=1.0, rolling=False,
                            label="champion_30f", cs_rank_exclude=no_rank)
    if preds is not None and not preds.empty:
        results = evaluate_predictions(preds, regime_df)
        row = _make_row("champion_30f", [], results)
        rows.append(row)
        _print_row(row)
    else:
        print("  ⚠️  Baseline failed!")

    # Single features
    for feat in ordered:
        label = f"champion+{feat[-20:]}"  # truncate for display
        print(f"\n  [{feat}] ...")
        feats, no_rank = make_feature_set([feat], mkt_cols)
        preds = train_ensemble(df, feats, WINDOWS, l2=1.0, rolling=False,
                               label=label, cs_rank_exclude=no_rank)
        if preds is None or preds.empty:
            print(f"  ⚠️  {feat}: no predictions")
            continue
        results = evaluate_predictions(preds, regime_df)
        row = _make_row(label, [feat], results)
        rows.append(row)
        _print_row(row)

    # Best pairs from liquidation features
    liq_feats = [f for f in ordered if "liq" in f][:4]
    if len(liq_feats) >= 2:
        from itertools import combinations
        for a, b in combinations(liq_feats, 2):
            label = f"champion+{a[-12:]}+{b[-12:]}"
            print(f"\n  [{a} + {b}] ...")
            feats, no_rank = make_feature_set([a, b], mkt_cols)
            preds = train_ensemble(df, feats, WINDOWS, l2=1.0, rolling=False,
                                   label=label, cs_rank_exclude=no_rank)
            if preds is None or preds.empty:
                continue
            results = evaluate_predictions(preds, regime_df)
            row = _make_row(label, [a, b], results)
            rows.append(row)
            _print_row(row)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        baseline_all = float(summary.loc[summary["config"] == "champion_30f", "ALL_sh"].iloc[0]) if "champion_30f" in summary["config"].values else 0.0
        summary["delta_all"] = summary["ALL_sh"] - baseline_all
        summary = summary.sort_values("ALL_sh", ascending=False).reset_index(drop=True)

    return summary


def _make_row(config: str, extra_feats: List[str],
               results: Dict[str, Dict[str, float]]) -> Dict:
    row: Dict = {"config": config, "extra_feats": "|".join(extra_feats)}
    for window in ["W1", "W2", "W3", "ALL"]:
        m = results[window]
        row[f"{window}_sh"]    = m.get("sharpe", 0.0)
        row[f"{window}_sh_gr"] = m.get("sharpe_gross", 0.0)
        row[f"{window}_dd"]    = m.get("max_dd_pct", 0.0)
        row[f"{window}_cost"]  = m.get("total_cost_pct", 0.0)
        row[f"{window}_turn"]  = m.get("avg_turnover", 0.0)
    return row


def _print_row(row: Dict) -> None:
    w1 = row.get("W1_sh", 0), row.get("W2_sh", 0), row.get("W3_sh", 0), row.get("ALL_sh", 0)
    cost = row.get("ALL_cost", 0)
    delta = row.get("delta_all", 0)
    delta_str = f"Δ{delta:+.2f}" if delta else ""
    print(f"    W1={w1[0]:+.2f}  W2={w1[1]:+.2f}  W3={w1[2]:+.2f}  ALL={w1[3]:+.2f}  "
          f"cost={cost:.1f}%  {delta_str}")


# ─────────────────────────────────────────────────────────────
#  SECTION 5: main
# ─────────────────────────────────────────────────────────────

def main(ic_only: bool = False, wf_only: bool = False, quick: bool = False) -> None:
    print("=" * 80)
    print("R47 — COINGLASS FEATURE RESEARCH")
    print("=" * 80)

    # ── Load CG daily features ─────────────────────────────────
    print("\n[1] Loading CoinGlass daily data ...")
    cg = load_cg_daily()
    cg_feats_daily = compute_cg_features(cg)

    if cg_feats_daily.empty:
        print("❌ No CG features computed — check data/raw/coinglass/")
        return

    # ── Load base research frame ──────────────────────────────
    print("\n[2] Loading research frame (OHLCV + FEAT_30 + R35) ...")
    df, regime_df = load_research_frame()
    df, _ = add_r35_features(df)
    # regime_df already computed inside load_research_frame; don't recompute —
    # add_r35_features uses groupby().apply() which may drop 'symbol' in pandas 2.2+
    regime_df = regime_df.sort_index()
    print(f"  Base frame: {len(df):,} rows × {len(df.columns)} cols")

    if quick:
        print("  ⚡ Quick mode: subsetting to BTC/ETH/SOL ...")
        df = df[df["symbol"].isin(["BTC/USDT", "ETH/USDT", "SOL/USDT"])].copy()

    # ── Merge CG features ─────────────────────────────────────
    print("\n[3] Merging CG features (shift-1d lookahead-safe) ...")
    df, per_sym_cols, mkt_cols = add_cg_features(df, cg_feats_daily)

    # Drop cg_liq_intensity — r=0.89 with rel_volume_cs (redundant by formula)
    per_sym_cols = [c for c in per_sym_cols if c != "cg_liq_intensity"]

    all_cg_cols = per_sym_cols + mkt_cols
    print(f"  Per-symbol CG features: {per_sym_cols}")
    print(f"  Market-level CG features (no CS-rank): {mkt_cols}")
    print(f"  CG coverage: {df[per_sym_cols[0]].notna().mean()*100:.1f}% non-null (first feature)")

    # ── IC scan ───────────────────────────────────────────────
    print("\n[4] IC Scan (TRAIN data only, each WF window) ...")
    ic_df = run_ic_scan(df, all_cg_cols, WINDOWS)

    if not ic_df.empty:
        print(f"\n  {'Flag':<5} {'Feature':<28} {'mean_IC':>8} {'|IC|':>6} {'ICIR':>7} {'nWin':>5}")
        print(f"  {'─'*5} {'─'*28} {'─'*8} {'─'*6} {'─'*7} {'─'*5}")
        for _, r in ic_df.iterrows():
            print(f"  {r['flag']:<5} {r['feature']:<28} {r['mean_ic']:>+8.4f} "
                  f"{r['abs_ic']:>6.4f} {r['mean_icir']:>+7.3f} {int(r['n_windows']):>5}")

    # ── Redundancy check ──────────────────────────────────────
    print("\n[5] Redundancy check vs FEAT_30 ...")
    redund = run_redundancy_check(df, all_cg_cols, CHAMPION_FEAT_30)
    if redund.empty:
        print("  ✅ No high correlations found (|r| > 0.3)")
    else:
        print(f"\n  {'Flag':<15} {'new_feat':<25} {'old_feat':<28} {'corr':>6}")
        print(f"  {'─'*15} {'─'*25} {'─'*28} {'─'*6}")
        for _, r in redund.head(20).iterrows():
            print(f"  {r['flag']:<15} {r['new_feat']:<25} {r['old_feat']:<28} {r['corr']:>+6.3f}")

    # ── Event study ───────────────────────────────────────────
    print("\n[6] Event study: top-1% liquidation spikes ...")
    run_event_study(df)

    if ic_only:
        print("\n✅ IC-only mode, stopping before WF test.")
        return

    # ── WF test ───────────────────────────────────────────────
    # Select candidates: only features with IC > 0.01 and not redundant (|r| > 0.5)
    redundant_set = set()
    if not redund.empty:
        redundant_set = set(redund[redund["flag"].str.contains("redundant")]["new_feat"].tolist())

    if not ic_df.empty:
        candidates = ic_df[
            (ic_df["abs_ic"] > 0.01) & (~ic_df["feature"].isin(redundant_set))
        ]["feature"].tolist()
    else:
        # fallback: test all (sorted by liq priority)
        candidates = (
            [c for c in per_sym_cols if "liq" in c] +
            [c for c in per_sym_cols if "taker" in c] +
            [c for c in per_sym_cols if "ls" in c] +
            mkt_cols
        )

    if not candidates:
        print("\n⚠️  No candidates pass IC threshold — trying top-5 by |IC| anyway")
        candidates = ic_df.head(5)["feature"].tolist() if not ic_df.empty else all_cg_cols[:5]

    print(f"\n[7] Walk-Forward ablation ({len(candidates)} candidates): {candidates}")
    summary = run_wf_ablation(df, regime_df, candidates, mkt_cols, ic_df)

    # ── Save results ──────────────────────────────────────────
    out_path = BASE_DIR / "results_r47_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"\n[8] Results saved → {out_path}")

    if not summary.empty:
        print(f"\n{'='*80}")
        print(f"  TOP RESULTS (sorted by ALL Sharpe)")
        print(f"{'='*80}")
        cols = ["config", "W1_sh", "W2_sh", "W3_sh", "ALL_sh", "delta_all", "ALL_cost"]
        avail = [c for c in cols if c in summary.columns]
        print(summary.head(10)[avail].to_string(index=False))

        # Final verdict
        baseline_all = float(summary.loc[summary["config"] == "champion_30f", "ALL_sh"].iloc[0]) if "champion_30f" in summary["config"].values else 1.13
        best_all = float(summary.iloc[0]["ALL_sh"]) if not summary.empty else 0.0
        best_config = str(summary.iloc[0]["config"]) if not summary.empty else "N/A"
        print(f"\n  Baseline ALL Sharpe:  {baseline_all:.2f}")
        print(f"  Best ALL Sharpe:      {best_all:.2f}  ({best_config})")
        if best_all > baseline_all + 0.07:
            print(f"  🔥 IMPROVEMENT: +{best_all-baseline_all:.2f} → NEW CHAMPION CANDIDATE")
        elif best_all > baseline_all:
            print(f"  ✅ Marginal improvement: +{best_all-baseline_all:.2f}")
        else:
            print(f"  ❌ No improvement. CG data may be redundant (consider cancelling $29/mo subscription)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ic-only",  action="store_true", help="Only run IC scan, no WF test")
    parser.add_argument("--wf-only",  action="store_true", help="Only run WF test (skip IC scan)")
    parser.add_argument("--quick",    action="store_true", help="Smoke test on 3 symbols")
    args = parser.parse_args()

    main(ic_only=args.ic_only, wf_only=args.wf_only, quick=args.quick)
