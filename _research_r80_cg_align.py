#!/usr/bin/env python3
"""
R80 — CoinGlass Alignment & Lookahead Check

Phase 0 of DeepResearch v3: "CG Alpha + Risk Overlay on top of R68".

Loads all 5 CoinGlass daily parquets, merges into the 12h research frame,
and computes Spearman IC for each raw CG feature under two alignment modes:

  - direct  : cg_date = normalize(timestamp)          [same day — potential lookahead]
  - shift1  : cg_date = normalize(timestamp) - 1d     [1-day lag — prod-safe]

Sanity checks:
  - (symbol, cg_date) uniqueness per endpoint (after dedup)
  - Per-symbol coverage vs expected date range; flags < 0.95

IC scan:
  - Pooled Spearman IC (all rows pooled)
  - Mean-of-timestamps Spearman IC
  - Per test-window IC (W1 / W2 / W3 from R68 CONTINUOUS_WINDOWS)

Lookahead flag: abs(IC_direct) > 2× abs(IC_shift1) → warns

Output (always):
  - results/r80_ic_table.csv        — all IC rows with align_mode column
  - results/r80_summary.json        — metadata + per-feature coverage

Output (shift1 or both mode):
  - data/features/frame_12h_with_cg.parquet  — base 12h frame + raw CG (shift1)

Usage:
  python _research_r80_cg_align.py                         # both modes
  python _research_r80_cg_align.py --align_mode direct
  python _research_r80_cg_align.py --align_mode shift1
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

CG_DIR      = ROOT / "data" / "raw" / "coinglass"
FEAT_DIR    = ROOT / "data" / "features"
RESULTS_DIR = ROOT / "results"

# Raw features to build at Phase 0 (no z-scoring — that's Phase 2 / R82)
CG_RAW_FEATS = [
    "cg_taker_imb",    # (buy_usd - sell_usd) / (total + eps)
    "cg_oi_chg",       # oi_close.pct_change(1) per symbol
    "cg_fr",           # funding rate daily close
    "cg_liq_imb",      # (liq_long - liq_short) / (total + eps)
    "cg_liq_log",      # log1p(liq_long + liq_short)
    "cg_ls_ratio",     # long/short account ratio
]

# Test windows from R68 CONTINUOUS_WINDOWS (for per-window IC)
TEST_WINDOWS = [
    {"name": "W1", "start": "2024-10-15", "end": "2025-05-14"},
    {"name": "W2", "start": "2025-05-15", "end": "2025-11-14"},
    {"name": "W3", "start": "2025-11-15", "end": "2026-03-17"},
]

EPS                = 1e-10
COVERAGE_THRESHOLD = 0.95   # warn if per-symbol coverage below this
LOOKAHEAD_RATIO    = 2.0    # flag if direct IC > LOOKAHEAD_RATIO × shift1 IC


# ─── Step 1: Load raw CG parquets ─────────────────────────────────────────────

def load_cg_raw() -> dict:
    """
    Load all 5 CG parquets.
    Normalises timestamps to midnight UTC (cg_date).
    Drops duplicate (symbol, cg_date) rows, keeping last.
    Returns {name: DataFrame}.
    """
    result = {}
    for name in ["taker", "oi", "funding", "liq", "ls_ratio"]:
        path = CG_DIR / f"{name}.parquet"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping")
            continue
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["cg_date"]   = df["timestamp"].dt.normalize()
        before = len(df)
        df = df.drop_duplicates(subset=["symbol", "cg_date"], keep="last")
        after  = len(df)
        n_dropped = before - after
        result[name] = df
        print(
            f"  Loaded {name}: {after:,} rows, {df['symbol'].nunique()} syms, "
            f"{str(df['cg_date'].min())[:10]} → {str(df['cg_date'].max())[:10]}"
            + (f"  (dropped {n_dropped} dupes)" if n_dropped else "")
        )
    return result


# ─── Step 2: Sanity checks ────────────────────────────────────────────────────

def sanity_check(cg: dict) -> None:
    """
    Assert (symbol, cg_date) uniqueness.
    Report per-symbol coverage; flag below COVERAGE_THRESHOLD.
    """
    from _research_round7 import SYM_35

    print()
    for name, df in cg.items():
        dupes = df.duplicated(subset=["symbol", "cg_date"]).sum()
        assert dupes == 0, f"FAIL: {dupes} duplicate (symbol, cg_date) in {name} after dedup"

        min_date = df["cg_date"].min()
        max_date = df["cg_date"].max()
        n_days   = int((max_date - min_date).days) + 1

        per_sym  = df.groupby("symbol")["cg_date"].count()
        coverage = per_sym / n_days
        low      = coverage[coverage < COVERAGE_THRESHOLD]

        missing_sym = [s for s in SYM_35 if s not in df["symbol"].values]

        ok = len(low) == 0 and len(missing_sym) == 0
        print(f"  {name:<10}: uniqueness OK {'✓' if ok else '⚠'}"
              f"  (syms={df['symbol'].nunique()}, days={n_days})")

        if missing_sym:
            print(f"    Missing from SYM_35: {missing_sym}")
        for sym, cov in low.items():
            print(f"    Low coverage  {sym}: {cov:.1%}")


# ─── Step 3: Build CG daily feature table ─────────────────────────────────────

def build_cg_daily(cg: dict) -> pd.DataFrame:
    """
    Build (symbol, cg_date) → raw feature table from 5 CG endpoints.
    Joined on outer — symbols present in some but not all endpoints get NaN.
    """
    frames = []

    taker = cg.get("taker")
    if taker is not None:
        df    = taker[["symbol", "cg_date", "taker_buy_usd", "taker_sell_usd"]].copy()
        total = df["taker_buy_usd"] + df["taker_sell_usd"]
        df["cg_taker_imb"] = (df["taker_buy_usd"] - df["taker_sell_usd"]) / (total + EPS)
        frames.append(
            df[["symbol", "cg_date", "cg_taker_imb"]].set_index(["symbol", "cg_date"])
        )

    oi = cg.get("oi")
    if oi is not None:
        df = oi[["symbol", "cg_date", "oi_close"]].copy().sort_values(["symbol", "cg_date"])
        df["cg_oi_chg"] = df.groupby("symbol")["oi_close"].pct_change(1)
        frames.append(
            df[["symbol", "cg_date", "cg_oi_chg"]].set_index(["symbol", "cg_date"])
        )

    funding = cg.get("funding")
    if funding is not None:
        df         = funding[["symbol", "cg_date", "fr_close"]].copy()
        df["cg_fr"] = df["fr_close"]
        frames.append(
            df[["symbol", "cg_date", "cg_fr"]].set_index(["symbol", "cg_date"])
        )

    liq = cg.get("liq")
    if liq is not None:
        df    = liq[["symbol", "cg_date", "liq_long_usd", "liq_short_usd"]].copy()
        total = df["liq_long_usd"] + df["liq_short_usd"]
        df["cg_liq_imb"] = (df["liq_long_usd"] - df["liq_short_usd"]) / (total + EPS)
        df["cg_liq_log"] = np.log1p(total)
        frames.append(
            df[["symbol", "cg_date", "cg_liq_imb", "cg_liq_log"]]
            .set_index(["symbol", "cg_date"])
        )

    ls = cg.get("ls_ratio")
    if ls is not None:
        df                = ls[["symbol", "cg_date", "ls_ratio"]].copy()
        df["cg_ls_ratio"] = df["ls_ratio"]
        frames.append(
            df[["symbol", "cg_date", "cg_ls_ratio"]].set_index(["symbol", "cg_date"])
        )

    if not frames:
        raise RuntimeError("No CG data loaded — nothing to merge.")

    out = frames[0].copy()
    for f in frames[1:]:
        out = out.join(f, how="outer")
    out = out.reset_index()
    out = out.replace([np.inf, -np.inf], np.nan)

    print(f"  CG daily table: {len(out):,} rows × {out['symbol'].nunique()} syms "
          f"× {len(out.columns) - 2} features")
    return out


# ─── Step 4: Merge into 12h frame ─────────────────────────────────────────────

def merge_cg(df: pd.DataFrame, cg_daily: pd.DataFrame, align_mode: str) -> pd.DataFrame:
    """
    Merge CG daily features into the 12h research frame.

    direct  → merge key = timestamp.normalize()         (same calendar day)
    shift1  → merge key = timestamp.normalize() - 1day  (previous calendar day)
    """
    df = df.copy()
    base = df["timestamp"].dt.normalize()
    df["_cg_date"] = base if align_mode == "direct" else base - pd.Timedelta(days=1)

    merged = df.merge(
        cg_daily.rename(columns={"cg_date": "_cg_date"}),
        on=["symbol", "_cg_date"],
        how="left",
    ).drop(columns=["_cg_date"])
    return merged


# ─── Step 5: IC computation ───────────────────────────────────────────────────

def _ic_row(sub: pd.DataFrame, feat: str, window: str) -> dict:
    """Pooled Spearman IC + mean-of-timestamps Spearman IC for one slice."""
    valid = sub[[feat, "fwd_ret_12h", "timestamp"]].dropna()
    n_obs = len(valid)
    if n_obs < 50:
        return {
            "feature": feat, "window": window,
            "pooled_ic": np.nan, "mean_ts_ic": np.nan,
            "n_obs": n_obs, "coverage_pct": np.nan,
        }

    pooled_ic = float(stats.spearmanr(valid[feat], valid["fwd_ret_12h"])[0])

    ts_ics = []
    for _, grp in valid.groupby("timestamp"):
        g = grp[[feat, "fwd_ret_12h"]].dropna()
        if len(g) < 8 or g[feat].nunique() < 3:
            continue
        v = stats.spearmanr(g[feat], g["fwd_ret_12h"])[0]
        if pd.notna(v):
            ts_ics.append(float(v))
    mean_ts_ic = float(np.mean(ts_ics)) if ts_ics else np.nan

    total = len(sub[["timestamp", "symbol", "fwd_ret_12h"]].dropna())
    coverage = n_obs / total if total > 0 else 0.0

    return {
        "feature": feat, "window": window,
        "pooled_ic":   round(pooled_ic, 4),
        "mean_ts_ic":  round(mean_ts_ic, 4),
        "n_obs":       n_obs,
        "coverage_pct": round(coverage, 3),
    }


def ic_scan(merged: pd.DataFrame, feats: list) -> list:
    """IC over full dataset + per test window."""
    tz   = merged["timestamp"].dt.tz
    rows = [_ic_row(merged, f, "ALL") for f in feats if f in merged.columns]
    for w in TEST_WINDOWS:
        ts_s = pd.Timestamp(w["start"], tz=tz)
        ts_e = pd.Timestamp(w["end"],   tz=tz)
        wdf  = merged[(merged["timestamp"] >= ts_s) & (merged["timestamp"] <= ts_e)]
        for f in feats:
            if f in merged.columns:
                rows.append(_ic_row(wdf, f, w["name"]))
    return rows


def print_ic_table(rows: list, align_mode: str) -> None:
    df = pd.DataFrame(rows)
    print(f"\n  {'─'*76}")
    print(f"  IC TABLE  —  align_mode={align_mode}")
    print(f"  {'─'*76}")
    print(f"  {'Feature':<26} {'Win':<5} {'Pooled IC':>10} {'Mean-TS IC':>11} "
          f"{'Coverage':>10} {'N-obs':>8}")
    print(f"  {'─'*74}")
    for _, r in df.sort_values(["feature", "window"]).iterrows():
        cov = f"{r['coverage_pct']:.1%}" if pd.notna(r["coverage_pct"]) else "   N/A"
        pic = f"{r['pooled_ic']:.4f}"    if pd.notna(r["pooled_ic"])   else "    N/A "
        mic = f"{r['mean_ts_ic']:.4f}"   if pd.notna(r["mean_ts_ic"])  else "     N/A  "
        print(f"  {r['feature']:<26} {r['window']:<5} {pic:>10} {mic:>11} "
              f"{cov:>10} {int(r['n_obs']):>8,}")


def print_coverage(merged: pd.DataFrame, feats: list, label: str) -> None:
    tz          = merged["timestamp"].dt.tz
    test_slice  = merged[merged["timestamp"] >= pd.Timestamp("2024-10-15", tz=tz)]
    total       = len(test_slice)
    print(f"\n  Coverage on test period (≥ 2024-10-15)  —  {total:,} rows  [{label}]")
    for feat in feats:
        if feat in test_slice.columns:
            nonnull = test_slice[feat].notna().sum()
            pct     = nonnull / total if total else 0.0
            flag    = "  ⚠ LOW" if pct < COVERAGE_THRESHOLD else ""
            print(f"    {feat:<28}: {pct:.1%}  ({nonnull:,}/{total:,}){flag}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="R80: CG alignment & lookahead check")
    parser.add_argument(
        "--align_mode", choices=["direct", "shift1", "both"], default="both",
        help="Alignment mode to evaluate (default: both)",
    )
    args = parser.parse_args()

    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  R80 — CoinGlass Alignment & Lookahead Check")
    print("=" * 70)

    # ── [1/5] Load CG parquets ────────────────────────────────────────────────
    print("\n[1/5] Loading CoinGlass parquets …")
    cg_raw = load_cg_raw()

    # ── [2/5] Sanity checks ───────────────────────────────────────────────────
    print("\n[2/5] Sanity checks …")
    sanity_check(cg_raw)

    # ── [3/5] Build CG daily feature table ───────────────────────────────────
    print("\n[3/5] Building CG daily feature table …")
    cg_daily    = build_cg_daily(cg_raw)
    avail_feats = [f for f in CG_RAW_FEATS if f in cg_daily.columns]
    print(f"  Available raw features: {avail_feats}")

    # ── [4/5] Load 12h research frame ────────────────────────────────────────
    print("\n[4/5] Loading 12h research frame …")
    from _research_r35_new_features import load_research_frame, add_r35_features
    df, _regime_df = load_research_frame()
    df, _          = add_r35_features(df)
    print(f"  Frame: {len(df):,} rows × {df['symbol'].nunique()} symbols")
    print(f"  Range: {df['timestamp'].min()} → {df['timestamp'].max()}")

    # ── [5/5] Align + IC scan ────────────────────────────────────────────────
    print("\n[5/5] Running alignment IC scan …")
    modes    = ["direct", "shift1"] if args.align_mode == "both" else [args.align_mode]
    all_ic   = []
    merged_shift1 = None

    for mode in modes:
        print(f"\n  ══ {mode.upper()} ══")
        mdf  = merge_cg(df, cg_daily, mode)
        rows = ic_scan(mdf, avail_feats)
        for r in rows:
            r["align_mode"] = mode
        all_ic.extend(rows)
        print_ic_table(rows, mode)
        print_coverage(mdf, avail_feats, mode)
        if mode == "shift1":
            merged_shift1 = mdf

    # ── IC table to disk ─────────────────────────────────────────────────────
    ic_df   = pd.DataFrame(all_ic)
    ic_path = RESULTS_DIR / "r80_ic_table.csv"
    ic_df.to_csv(ic_path, index=False)
    print(f"\n  Saved IC table → {ic_path}")

    # ── Lookahead flag ───────────────────────────────────────────────────────
    if len(modes) == 2:
        print("\n=== LOOKAHEAD CHECK (direct vs shift1 — ALL window) ===")
        all_win = ic_df[ic_df["window"] == "ALL"].copy()
        piv = all_win.pivot(
            index="feature", columns="align_mode", values="pooled_ic"
        ).reset_index()
        if {"direct", "shift1"}.issubset(piv.columns):
            piv["ratio"] = piv["direct"].abs() / (piv["shift1"].abs() + EPS)
            for _, r in piv.iterrows():
                flag = "  ⚠  LOOKAHEAD SUSPECTED" if r["ratio"] > LOOKAHEAD_RATIO else "  ✓"
                d = f"{r['direct']:.4f}"  if pd.notna(r["direct"])  else "  N/A  "
                s = f"{r['shift1']:.4f}"  if pd.notna(r["shift1"])  else "  N/A  "
                print(f"  {r['feature']:<28}: direct={d}  shift1={s}  ratio={r['ratio']:.1f}{flag}")

    # ── Save merged frame (shift1) ───────────────────────────────────────────
    if merged_shift1 is not None:
        out_path = FEAT_DIR / "frame_12h_with_cg.parquet"
        merged_shift1.to_parquet(out_path, index=False)
        print(f"\n  Saved merged frame (shift1) → {out_path}")
        print(f"  Shape: {merged_shift1.shape}")
        print(f"  Columns: {[c for c in merged_shift1.columns if c.startswith('cg_')]}")

        # ── Summary JSON ──────────────────────────────────────────────────────
        tz         = merged_shift1["timestamp"].dt.tz
        test_slice = merged_shift1[
            merged_shift1["timestamp"] >= pd.Timestamp("2024-10-15", tz=tz)
        ]
        coverage = {
            f: round(test_slice[f].notna().mean(), 3)
            for f in avail_feats if f in test_slice.columns
        }
        ic_all_rows = ic_df[(ic_df["window"] == "ALL") & (ic_df["align_mode"] == "shift1")]
        ic_summary  = ic_all_rows.set_index("feature")[["pooled_ic", "mean_ts_ic"]].to_dict()

        summary = {
            "script":          "r80_cg_align",
            "align_saved":     "shift1",
            "n_rows":          len(merged_shift1),
            "n_symbols":       int(merged_shift1["symbol"].nunique()),
            "ts_start":        str(merged_shift1["timestamp"].min()),
            "ts_end":          str(merged_shift1["timestamp"].max()),
            "cg_features":     avail_feats,
            "coverage_test":   coverage,
            "ic_shift1_all":   ic_summary,
        }
        summary_path = RESULTS_DIR / "r80_summary.json"
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2, default=float)
        print(f"  Saved summary → {summary_path}")

    print("\n  DONE.")


if __name__ == "__main__":
    main()
