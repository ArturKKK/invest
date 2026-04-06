#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R55 — CoinGlass Feature Expansion (OI/FR activation + Basis + Position Ratio)

Philosophy (AI consultation):
  - 1-2 features at a time, REPLACE not ADD (keep ≤33 total)
  - Prioritize disagreement features (CG 3-exchange vs Binance-only)
  - Basis = new axis "carry/sentiment" not in champion
  - Position ratio may beat account ratio → swap if better

Phases:
  Phase 1: Activate already-downloaded CG OI/FR → disagreement features
  Phase 2: Basis features (newly downloaded)
  Phase 3: Position ratio vs account ratio (newly downloaded)

Usage:
  python _research_r55_cg_features.py              # full run
  python _research_r55_cg_features.py --ic-only     # IC scan only
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── project imports ───────────────────────────────────────────

from _research_round7 import WINDOWS, SYM_35
from _research_r30b_fixed import (
    eval_with_costs,
    simulate_with_costs,
    train_ensemble,
)
from _research_r33_creative_features import FEAT_28
from _research_r35_new_features import (
    add_r35_features,
    load_research_frame,
)
from _research_r47_coinglass import (
    CG_DIR,
    CHAMPION_FEAT_30,
    CANONICAL_EXEC_CFG,
    add_cg_features,
    compute_cg_features,
    compute_ic_by_period,
    load_cg_daily,
    run_ic_scan,
    run_redundancy_check,
)

# ── config ─────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
BINANCE_METRICS = BASE_DIR / "data" / "sentiment" / "binance_futures_metrics.parquet"
BINANCE_FUNDING = BASE_DIR / "data" / "sentiment" / "binance_funding_rates.parquet"

ROLL_60D = 60
ROLL_30D = 30
ROLL_7D  = 7
EPS      = 1e-10

# TIER definitions (from run_trading.py)
TIER1 = {"BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"}
TIER3 = {"SAND/USDT", "MANA/USDT", "AXS/USDT", "THETA/USDT", "FLOW/USDT",
          "CHZ/USDT", "EGLD/USDT", "XTZ/USDT", "SNX/USDT"}


# ─────────────────────────────────────────────────────────────
#  SECTION 1: Load & align data
# ─────────────────────────────────────────────────────────────

def load_binance_daily_oi() -> pd.DataFrame:
    """Resample Binance hourly OI to daily (last snapshot per day)."""
    print("\n  Loading Binance hourly OI...")
    bm = pd.read_parquet(BINANCE_METRICS, columns=["timestamp", "symbol", "oi_value_usd"])
    bm["timestamp"] = pd.to_datetime(bm["timestamp"], utc=True)
    bm["date"] = bm["timestamp"].dt.normalize()

    # Last OI snapshot per (symbol, day) = daily close equivalent
    daily = (bm.sort_values("timestamp")
             .groupby(["symbol", "date"])
             .agg(bin_oi_close=("oi_value_usd", "last"))
             .reset_index())

    # Compute daily OI change
    daily = daily.sort_values(["symbol", "date"])
    daily["bin_oi_chg_1d"] = daily.groupby("symbol")["bin_oi_close"].pct_change()

    print(f"    Binance daily OI: {len(daily):,} rows, {daily['symbol'].nunique()} symbols")
    return daily


def load_binance_daily_fr() -> pd.DataFrame:
    """Resample Binance 8h funding to daily (mean of 3 snapshots per day)."""
    print("  Loading Binance 8h funding...")
    bfr = pd.read_parquet(BINANCE_FUNDING)
    bfr["timestamp"] = pd.to_datetime(bfr["timestamp"], utc=True)
    bfr["date"] = bfr["timestamp"].dt.normalize()

    daily = (bfr.groupby(["symbol", "date"])
             .agg(bin_fr_daily_mean=("funding_rate_binance", "mean"))
             .reset_index())

    print(f"    Binance daily FR: {len(daily):,} rows, {daily['symbol'].nunique()} symbols")
    return daily


def load_cg_oi_features() -> pd.DataFrame:
    """Build CG OI features from already-downloaded data."""
    path = CG_DIR / "oi.parquet"
    if not path.exists():
        print("  ⚠️  CG OI not found")
        return pd.DataFrame()

    print("  Loading CG OI...")
    oi = pd.read_parquet(path)
    oi["timestamp"] = pd.to_datetime(oi["timestamp"], utc=True)
    oi["cg_date"] = oi["timestamp"].dt.normalize()
    oi = oi.drop_duplicates(subset=["symbol", "cg_date"], keep="last")

    # Numeric conversion
    for c in ["oi_open", "oi_high", "oi_low", "oi_close"]:
        oi[c] = pd.to_numeric(oi[c], errors="coerce")

    # Features
    oi["cg_oi_chg_1d"] = oi["oi_close"] / (oi["oi_open"] + EPS) - 1
    oi["cg_oi_range"]   = (oi["oi_high"] - oi["oi_low"]) / (oi["oi_open"] + EPS)

    print(f"    CG OI features: {len(oi):,} rows, {oi['symbol'].nunique()} symbols")
    return oi[["symbol", "cg_date", "cg_oi_chg_1d", "cg_oi_range"]]


def load_cg_fr_features() -> pd.DataFrame:
    """Build CG Funding Rate features from already-downloaded data."""
    path = CG_DIR / "funding.parquet"
    if not path.exists():
        print("  ⚠️  CG Funding not found")
        return pd.DataFrame()

    print("  Loading CG Funding...")
    fr = pd.read_parquet(path)
    fr["timestamp"] = pd.to_datetime(fr["timestamp"], utc=True)
    fr["cg_date"] = fr["timestamp"].dt.normalize()
    fr = fr.drop_duplicates(subset=["symbol", "cg_date"], keep="last")

    for c in ["fr_open", "fr_high", "fr_low", "fr_close"]:
        fr[c] = pd.to_numeric(fr[c], errors="coerce")

    fr["cg_fr_close"]  = fr["fr_close"]
    fr["cg_fr_range"]  = fr["fr_high"] - fr["fr_low"]

    print(f"    CG FR features: {len(fr):,} rows, {fr['symbol'].nunique()} symbols")
    return fr[["symbol", "cg_date", "cg_fr_close", "cg_fr_range"]]


def load_cg_basis_features() -> pd.DataFrame:
    """Build Basis features from newly downloaded data."""
    path = CG_DIR / "basis.parquet"
    if not path.exists():
        print("  ⚠️  CG Basis not found — run: python src/data/download_coinglass_v4.py --only basis")
        return pd.DataFrame()

    print("  Loading CG Basis...")
    bs = pd.read_parquet(path)
    bs["timestamp"] = pd.to_datetime(bs["timestamp"], utc=True)
    bs["cg_date"] = bs["timestamp"].dt.normalize()
    bs = bs.drop_duplicates(subset=["symbol", "cg_date"], keep="last")

    for c in ["basis_open", "basis_close", "basis_open_chg", "basis_close_chg"]:
        if c in bs.columns:
            bs[c] = pd.to_numeric(bs[c], errors="coerce")

    bs["cg_basis_close"] = bs["basis_close"]
    bs["cg_basis_chg"]   = bs["basis_close"] - bs["basis_open"]

    # Rolling zscore of basis (60-day window)
    bs = bs.sort_values(["symbol", "cg_date"])
    zscores = []
    for sym, g in bs.groupby("symbol"):
        roll_mean = g["cg_basis_close"].rolling(ROLL_60D, min_periods=ROLL_7D).mean()
        roll_std  = g["cg_basis_close"].rolling(ROLL_60D, min_periods=ROLL_7D).std() + EPS
        z = (g["cg_basis_close"] - roll_mean) / roll_std
        zscores.append(z)
    bs["cg_basis_z_60d"] = pd.concat(zscores)

    print(f"    CG Basis features: {len(bs):,} rows, {bs['symbol'].nunique()} symbols")
    return bs[["symbol", "cg_date", "cg_basis_close", "cg_basis_chg", "cg_basis_z_60d"]]


def load_cg_pos_ratio_features() -> pd.DataFrame:
    """Build Position Ratio features from newly downloaded data."""
    path = CG_DIR / "pos_ratio.parquet"
    if not path.exists():
        print("  ⚠️  CG Pos Ratio not found — run: python src/data/download_coinglass_v4.py --only pos_ratio")
        return pd.DataFrame()

    print("  Loading CG Position Ratio...")
    pr = pd.read_parquet(path)
    pr["timestamp"] = pd.to_datetime(pr["timestamp"], utc=True)
    pr["cg_date"] = pr["timestamp"].dt.normalize()
    pr = pr.drop_duplicates(subset=["symbol", "cg_date"], keep="last")

    for c in ["pos_long_pct", "pos_short_pct", "pos_ls_ratio"]:
        if c in pr.columns:
            pr[c] = pd.to_numeric(pr[c], errors="coerce")

    pr["cg_pos_ls_ratio"] = pr["pos_ls_ratio"]

    # Rolling zscore
    pr = pr.sort_values(["symbol", "cg_date"])
    zscores = []
    for sym, g in pr.groupby("symbol"):
        roll_mean = g["cg_pos_ls_ratio"].rolling(ROLL_60D, min_periods=ROLL_7D).mean()
        roll_std  = g["cg_pos_ls_ratio"].rolling(ROLL_60D, min_periods=ROLL_7D).std() + EPS
        z = (g["cg_pos_ls_ratio"] - roll_mean) / roll_std
        zscores.append(z)
    pr["cg_pos_ls_z_60d"] = pd.concat(zscores)

    print(f"    CG PosRatio features: {len(pr):,} rows, {pr['symbol'].nunique()} symbols")
    return pr[["symbol", "cg_date", "cg_pos_ls_ratio", "cg_pos_ls_z_60d"]]


def build_disagreement_features(
    cg_oi: pd.DataFrame,
    cg_fr: pd.DataFrame,
    bin_oi: pd.DataFrame,
    bin_fr: pd.DataFrame,
) -> pd.DataFrame:
    """
    Disagreement = CG (aggregated 3 exchanges) vs Binance-only.
    CG daily aligned to Binance daily by (symbol, date).
    """
    print("\n  Building disagreement features...")
    frames = []

    # OI disagreement: cg_oi_chg vs binance_oi_chg
    if not cg_oi.empty and not bin_oi.empty:
        merged = cg_oi.merge(
            bin_oi[["symbol", "date", "bin_oi_chg_1d"]],
            left_on=["symbol", "cg_date"],
            right_on=["symbol", "date"],
            how="inner",
        )
        merged["cg_oi_disagreement"] = merged["cg_oi_chg_1d"] - merged["bin_oi_chg_1d"]
        # Clip extremes
        merged["cg_oi_disagreement"] = merged["cg_oi_disagreement"].clip(-1, 1)
        frames.append(merged[["symbol", "cg_date", "cg_oi_disagreement"]])
        print(f"    OI disagreement: {len(merged):,} rows")

    # FR disagreement: cg_fr_close vs binance_fr_daily_mean
    if not cg_fr.empty and not bin_fr.empty:
        merged = cg_fr.merge(
            bin_fr[["symbol", "date", "bin_fr_daily_mean"]],
            left_on=["symbol", "cg_date"],
            right_on=["symbol", "date"],
            how="inner",
        )
        merged["cg_fr_disagreement"] = merged["cg_fr_close"] - merged["bin_fr_daily_mean"]
        frames.append(merged[["symbol", "cg_date", "cg_fr_disagreement"]])
        print(f"    FR disagreement: {len(merged):,} rows")

    if not frames:
        return pd.DataFrame()

    result = frames[0]
    for f in frames[1:]:
        result = result.merge(f, on=["symbol", "cg_date"], how="outer")

    return result


def build_all_r55_features() -> pd.DataFrame:
    """Build unified R55 feature table: (symbol, cg_date) → feature columns."""
    # Load individual components
    cg_oi = load_cg_oi_features()
    cg_fr = load_cg_fr_features()
    cg_basis = load_cg_basis_features()
    cg_pos = load_cg_pos_ratio_features()
    bin_oi = load_binance_daily_oi()
    bin_fr = load_binance_daily_fr()

    # Disagreement features
    disagree = build_disagreement_features(cg_oi, cg_fr, bin_oi, bin_fr)

    # Merge all into single table
    all_frames = []
    for df in [cg_oi, cg_fr, cg_basis, cg_pos, disagree]:
        if not df.empty and "cg_date" in df.columns:
            all_frames.append(df.set_index(["symbol", "cg_date"]))

    if not all_frames:
        print("  ✗ No features built!")
        return pd.DataFrame()

    result = all_frames[0]
    for f in all_frames[1:]:
        result = result.join(f, how="outer")

    result = result.reset_index()

    # Clean: replace inf, clip extremes
    feat_cols = [c for c in result.columns if c.startswith("cg_")]
    for c in feat_cols:
        result[c] = result[c].replace([np.inf, -np.inf], np.nan)

    print(f"\n  ══ R55 feature table: {len(result):,} rows, "
          f"{result['symbol'].nunique()} syms, {len(feat_cols)} features ══")
    print(f"  Features: {feat_cols}")

    return result


# ─────────────────────────────────────────────────────────────
#  SECTION 2: Merge into 1h model frame & IC scan
# ─────────────────────────────────────────────────────────────

def merge_r55_into_model(df: pd.DataFrame, r55: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Merge R55 daily features into the 1h model frame.
    Same shift rule as R47: cg_date = floor(t) - 1 day (lookahead safety).
    """
    if r55.empty:
        return df, []

    df = df.copy()
    df["_cg_date"] = df["timestamp"].dt.normalize() - pd.Timedelta(days=1)

    feat_cols = [c for c in r55.columns if c.startswith("cg_")]

    merged = df.merge(
        r55.rename(columns={"cg_date": "_cg_date"}),
        on=["symbol", "_cg_date"],
        how="left",
    )
    merged = merged.drop(columns=["_cg_date"])

    # Leave NaN for basis/pos_ratio (LGB/XGB handle NaN natively).
    # cs_rank() also handles NaN correctly (ranks only non-NaN, leaves NaN as NaN).
    # Fill NaN with 0 only for disagreement features (NaN = no disagreement = neutral).
    disagree_cols = [c for c in feat_cols if "disagreement" in c]
    for c in disagree_cols:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0)

    present = [c for c in feat_cols if c in merged.columns]
    n_nan = {c: merged[c].isna().sum() for c in present if merged[c].isna().any()}
    print(f"  Merged {len(present)} R55 features into model frame ({len(merged):,} rows)")
    if n_nan:
        print(f"  NaN counts (native handling): {n_nan}")

    return merged, present


def ic_scan_by_tier(df: pd.DataFrame, feat_cols: List[str],
                     target: str = "fwd_ret_12h") -> pd.DataFrame:
    """IC scan split by tier (T1/T2/T3) and overall."""
    rows = []
    df = df.copy()
    df["tier"] = "T2"
    df.loc[df["symbol"].isin(TIER1), "tier"] = "T1"
    df.loc[df["symbol"].isin(TIER3), "tier"] = "T3"

    for feat in feat_cols:
        sub = df[[feat, target, "timestamp", "tier"]].dropna()
        if len(sub) < 500:
            continue

        # Overall IC
        ic_all, icir_all, n_all = compute_ic_by_period(sub, feat, target)
        rows.append({"feature": feat, "tier": "ALL", "ic": ic_all,
                      "icir": icir_all, "n": n_all})

        # Per-tier IC
        for tier_name, tier_df in sub.groupby("tier"):
            if len(tier_df) < 200:
                continue
            ic, icir, n = compute_ic_by_period(tier_df, feat, target)
            rows.append({"feature": feat, "tier": tier_name, "ic": ic,
                          "icir": icir, "n": n})

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    # Pretty print
    pivot = result.pivot_table(index="feature", columns="tier",
                                values="ic", aggfunc="first")
    pivot = pivot.reindex(columns=["ALL", "T1", "T2", "T3"])
    return result, pivot


def ic_scan_by_window(df: pd.DataFrame, feat_cols: List[str],
                       windows: list, target: str = "fwd_ret_12h") -> pd.DataFrame:
    """IC scan per WF window (on TRAIN portion only)."""
    rows = []
    for w in windows:
        test_start = pd.to_datetime(w["test_start"], utc=True)
        train_end = pd.to_datetime(w["train_end"], utc=True)
        # Purge gap
        train_df = df[(df["timestamp"] <= train_end)].copy()
        if len(train_df) < 5000:
            continue

        for feat in feat_cols:
            ic, icir, n = compute_ic_by_period(train_df, feat, target)
            rows.append({"feature": feat, "window": w["name"],
                          "ic": ic, "icir": icir, "n": n})

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    pivot = result.pivot_table(index="feature", columns="window",
                                values="ic", aggfunc="first")
    return result, pivot


# ─────────────────────────────────────────────────────────────
#  SECTION 3: WF ablation (replace worst existing feature)
# ─────────────────────────────────────────────────────────────

def run_wf_ablation(df: pd.DataFrame, new_feats: List[str],
                     champion_feats: List[str],
                     regime_df: pd.DataFrame = None) -> None:
    """
    For each candidate new feature:
      1. Run champion WF (baseline)
      2. Run champion WF with weakest feat replaced by new feat
      3. Compare Sharpe
    """
    # Current champion CG features (potential replacement targets)
    cg_existing = [f for f in champion_feats if f.startswith("cg_")]
    print(f"\n  ══ WF ABLATION ══")
    print(f"  Champion: {len(champion_feats)} features")
    print(f"  CG features (replacement targets): {cg_existing}")
    print(f"  New candidates: {new_feats}\n")

    # Baseline: run champion
    print("  [Baseline] Running champion WF...")
    baseline_results = _run_wf(df, champion_feats, regime_df)
    if baseline_results is None:
        print("  ✗ Baseline WF failed")
        return
    print(f"  Baseline: {_format_results(baseline_results)}")

    # For each new feature: try replacing each CG feature
    for new_f in new_feats:
        if new_f not in df.columns:
            continue
        print(f"\n  ── Testing: {new_f} ──")

        for old_f in cg_existing:
            test_feats = [new_f if f == old_f else f for f in champion_feats]
            # Verify all features exist
            missing = [f for f in test_feats if f not in df.columns]
            if missing:
                continue

            results = _run_wf(df, test_feats, regime_df)
            if results is None:
                continue

            delta = results["ALL"]["sharpe"] - baseline_results["ALL"]["sharpe"]
            flag = "🔥" if delta > 0.05 else ("✅" if delta > 0 else "  ")
            print(f"    {new_f} ↔ {old_f}: {_format_results(results)} "
                  f"(Δ={delta:+.3f}) {flag}")


def _run_wf(df: pd.DataFrame, features: List[str],
            regime_df: pd.DataFrame = None) -> dict | None:
    """Run walk-forward evaluation with given feature set."""
    # Verify features exist
    present = [f for f in features if f in df.columns]
    if len(present) < len(features) * 0.8:
        return None

    results = {}
    all_pnls = []

    for w in WINDOWS:
        train_end = pd.to_datetime(w["train_end"], utc=True)
        test_start = pd.to_datetime(w["test_start"], utc=True)
        test_end = pd.to_datetime(w["test_end"], utc=True)

        train_df = df[df["timestamp"] <= train_end].copy()
        test_df = df[(df["timestamp"] >= test_start) &
                     (df["timestamp"] <= test_end)].copy()

        if len(train_df) < 5000 or len(test_df) < 1000:
            continue

        try:
            preds = train_ensemble(train_df, present, WINDOWS)
            if preds is None:
                continue
            # Merge predictions into test_df
            test_preds = preds[preds["window"] == w["name"]].copy()
            merged = test_df.merge(
                test_preds[["timestamp", "symbol", "pred"]],
                on=["timestamp", "symbol"], how="left"
            )
            pnl = simulate_with_costs(merged, regime_df, CANONICAL_EXEC_CFG)
            metrics = eval_with_costs(pnl, w["name"])
            results[w["name"]] = metrics
            all_pnls.append(pnl)
        except Exception as e:
            print(f"    ⚠️  {w['name']} failed: {e}")
            continue

    if not all_pnls:
        return None

    # ALL: concat all window PnLs
    combined = pd.concat(all_pnls)
    results["ALL"] = eval_with_costs(combined, "ALL")
    return results


def _format_results(results: dict) -> str:
    """Format WF results as compact string."""
    parts = []
    for name in ["W1", "W2", "W3", "ALL"]:
        if name in results:
            s = results[name].get("sharpe", 0)
            parts.append(f"{name}={s:+.2f}")
    return " | ".join(parts)


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="R55 CoinGlass Feature Expansion")
    parser.add_argument("--ic-only", action="store_true", help="Only run IC scan")
    parser.add_argument("--wf-only", action="store_true", help="Only run WF ablation")
    args = parser.parse_args()

    print("=" * 70)
    print("  R55 — CoinGlass Feature Expansion")
    print("=" * 70)

    # ── Step 1: Build R55 features ───────────────────────────
    print("\n[1/4] Building R55 feature table...")
    r55_feats = build_all_r55_features()

    if r55_feats.empty:
        print("  ✗ No features built, exiting")
        return

    # ── Step 2: Load base model frame ────────────────────────
    print("\n[2/4] Loading base research frame...")
    base_df, regime_df = load_research_frame()
    base_df, r35_added = add_r35_features(base_df)

    # Also load existing CG features
    cg = load_cg_daily()
    cg_daily = compute_cg_features(cg)

    if not cg_daily.empty:
        base_df, cg_per_sym, cg_mkt = add_cg_features(base_df, cg_daily)

    # Merge R55 features
    df, r55_cols = merge_r55_into_model(base_df, r55_feats)

    print(f"\n  Model frame: {len(df):,} rows, {df['symbol'].nunique()} symbols")
    print(f"  R55 features available: {r55_cols}")

    # ── Step 3: IC scan ──────────────────────────────────────
    if not args.wf_only:
        print("\n" + "=" * 70)
        print("  [3/4] IC SCAN — per tier")
        print("=" * 70)

        tier_result, tier_pivot = ic_scan_by_tier(df, r55_cols)
        print("\n  IC by tier:")
        print(tier_pivot.to_string(float_format="{:.4f}".format))

        print("\n  IC by WF window:")
        win_result, win_pivot = ic_scan_by_window(df, r55_cols, WINDOWS)
        print(win_pivot.to_string(float_format="{:.4f}".format))

        # Redundancy vs champion
        print("\n  Redundancy check vs champion features:")
        redund = run_redundancy_check(df, r55_cols, CHAMPION_FEAT_30)
        if not redund.empty:
            print(redund.to_string(index=False))
        else:
            print("  No high correlations found (all |r| < 0.3)")

    # ── Step 4: WF ablation ──────────────────────────────────
    if not args.ic_only:
        print("\n" + "=" * 70)
        print("  [4/4] WF ABLATION")
        print("=" * 70)

        # Check which champion features are actually present
        champion_present = [f for f in CHAMPION_FEAT_30 if f in df.columns]
        print(f"  Champion features present: {len(champion_present)}/{len(CHAMPION_FEAT_30)}")

        # Select top candidates from IC scan (abs IC > 0.01)
        if not args.wf_only:
            good_feats = tier_result[
                (tier_result["tier"] == "ALL") &
                (tier_result["ic"].abs() > 0.01)
            ]["feature"].tolist()
        else:
            good_feats = r55_cols  # test all

        if good_feats:
            run_wf_ablation(df, good_feats, champion_present, regime_df)
        else:
            print("  No features passed IC threshold (|IC| > 0.01), skipping WF")

    print("\n" + "=" * 70)
    print("  R55 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
