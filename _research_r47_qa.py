#!/usr/bin/env python3
"""
R47 — CoinGlass Data QA

Checks:
  1. Timestamp alignment   — all at 00:00 UTC (1d candles open at midnight)
  2. Candle direction       — OI chaining confirms candle covers [t, t+24h)
  3. Coverage per symbol    — rows, date range, % missing days, zero fraction
  4. Anomaly detection      — top-0.1% liquidations, date gaps > 1 day
  5. Lookahead (shift) test — corr(liq_total(t), fwd_ret_12h) vs
                              corr(liq_total(t-1d), fwd_ret_12h)
                              If shift(0) wins strongly → candles are CLOSED at t
                              If shift(1) wins → safer to use previous day only

Interpretation of 1d candle timestamps (confirmed by OI chaining):
  timestamp = 2022-01-01 00:00 UTC → candle covers [Jan 1 00:00, Jan 2 00:00)
  → candle is COMPLETE + available at Jan 2 00:00 UTC
  → for a model running at t = 00:00 UTC: use liq(t-1d) [previous day complete]
  → for a model running at t = 12:00 UTC: also use liq(t-1d) [same]
  → conservative approach: shift(1) in daily data = 1-day lag

Usage:
  python _research_r47_qa.py
  python _research_r47_qa.py --data12h    # also check 12h folder
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).parent
DATA_DIR = PROJECT / "data"
CG_1D = DATA_DIR / "raw" / "coinglass"
CG_12H = DATA_DIR / "raw" / "coinglass_12h"
RAW_OHLCV = DATA_DIR / "raw"

SYM_35 = [
    "BTC", "ETH", "SOL", "BNB", "XRP",
    "ADA", "DOGE", "AVAX", "DOT", "LINK",
    "MATIC", "UNI", "ATOM", "LTC", "NEAR",
    "FIL", "APT", "ARB", "OP", "AAVE",
    "INJ", "FTM", "ALGO", "SAND", "MANA",
    "AXS", "THETA", "RUNE", "EGLD", "XTZ",
    "FLOW", "CHZ", "CRV", "LDO", "SNX",
]

ENDPOINTS = ["liq", "oi", "taker", "funding", "ls_ratio"]


# ── helpers ───────────────────────────────────────────────────

def hr(ch="─", n=70):
    print(ch * n)


def load_cg_data(folder: Path) -> dict:
    """Load all CoinGlass parquet files from a folder."""
    dfs = {}
    for ep in ENDPOINTS:
        path = folder / f"{ep}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            dfs[ep] = df
        else:
            print(f"  ⚠️  {ep}.parquet not found in {folder}")
    return dfs


def load_ohlcv_daily() -> pd.DataFrame:
    """Load 1h OHLCV and resample to 24h bars for alignment checks."""
    frames = []
    for sym in SYM_35:
        path = RAW_OHLCV / f"{sym}_USDT_1h.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["symbol"] = f"{sym}/USDT"
        frames.append(df[["timestamp", "symbol", "close", "volume"]])

    if not frames:
        return pd.DataFrame()

    ohlcv = pd.concat(frames, ignore_index=True)
    ohlcv = ohlcv.sort_values(["symbol", "timestamp"])

    # Resample to daily — use close at 23:00 as approximate daily close,
    # and sum volume across 24h. We floor to midnight to align with CG timestamps.
    daily = []
    for sym, g in ohlcv.groupby("symbol"):
        g = g.set_index("timestamp")
        d = g["close"].resample("1D").last().rename("close_daily")
        v = g["volume"].resample("1D").sum().rename("vol_daily")
        day_df = pd.concat([d, v], axis=1).reset_index()
        day_df["symbol"] = sym
        daily.append(day_df)

    daily_df = pd.concat(daily, ignore_index=True)
    daily_df = daily_df.rename(columns={"timestamp": "date"})
    daily_df["date"] = daily_df["date"].dt.tz_convert("UTC")
    return daily_df


# ── QA checks ─────────────────────────────────────────────────

def check_timestamps(dfs: dict, label: str):
    """1. All timestamps at expected UTC boundaries."""
    hr()
    print(f"  [1] TIMESTAMP ALIGNMENT — {label}")
    hr()
    ok = True
    for ep, df in dfs.items():
        hours = df["timestamp"].dt.hour.unique()
        minutes = df["timestamp"].dt.minute.unique()
        if label == "1d":
            aligned = (hours == [0]).all() and (minutes == [0]).all()
        else:
            aligned = (set(hours) <= {0, 12}) and (minutes == [0]).all()
        status = "✅" if aligned else "❌"
        if not aligned:
            ok = False
        print(f"  {status} {ep:12s}: hours={sorted(hours)[:6]}  minutes={sorted(minutes)[:3]}")

    if ok:
        print(f"\n  → All timestamps aligned to {'midnight' if label=='1d' else '00:00/12:00'} UTC ✅")
    else:
        print(f"\n  → Misaligned timestamps detected! Check carefully ❌")
    return ok


def coverage_table(dfs: dict, label: str):
    """2. Coverage per symbol: rows, date range, missing, zeros."""
    hr()
    print(f"  [2] COVERAGE PER SYMBOL — {label}")
    hr()

    # Use liq for base coverage (most complete)
    liq = dfs.get("liq", pd.DataFrame())
    if liq.empty:
        print("  No liq data, skipping coverage table")
        return {}

    # Expected days
    all_dates = pd.date_range(
        liq["timestamp"].min(), liq["timestamp"].max(), freq="1D", tz="UTC"
    )
    expected_days = len(all_dates)

    print(f"\n  Date range: {liq['timestamp'].min().date()} → {liq['timestamp'].max().date()}")
    print(f"  Expected days: {expected_days}")
    print()
    print(f"  {'Symbol':<12} {'liq':>5} {'oi':>5} {'taker':>5} {'fund':>5} {'ls':>5}  {'zero_liq':>8}  {'coverage':>9}  {'status'}")
    print(f"  {'─'*12} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*5}  {'─'*8}  {'─'*9}  {'─'*8}")

    coverage = {}
    excluded = []
    for sym in SYM_35:
        sym_str = f"{sym}/USDT"
        rows = {}
        for ep in ENDPOINTS:
            df = dfs.get(ep, pd.DataFrame())
            rows[ep] = len(df[df["symbol"] == sym_str]) if not df.empty else 0

        liq_rows = rows["liq"]
        pct = 100 * liq_rows / expected_days if expected_days > 0 else 0

        # Zero fraction for liq
        sym_liq = liq[liq["symbol"] == sym_str]
        liq_total = sym_liq["liq_long_usd"] + sym_liq["liq_short_usd"] if len(sym_liq) else pd.Series()
        zero_pct = 100 * (liq_total == 0).sum() / len(liq_total) if len(liq_total) else 0

        missing_ep = [ep[:4] for ep in ENDPOINTS if rows.get(ep, 0) < 10]
        status = "⚠️ no " + "/".join(missing_ep) if missing_ep else "✅"

        print(f"  {sym:<12} {rows['liq']:>5} {rows['oi']:>5} {rows['taker']:>5} "
              f"{rows['funding']:>5} {rows['ls_ratio']:>5}  {zero_pct:>7.1f}%  {pct:>8.1f}%  {status}")

        coverage[sym] = rows
        if missing_ep:
            excluded.append(sym)

    print()
    if excluded:
        print(f"  ⚠️  Symbols with missing endpoints (exclude from CG tests): {excluded}")
    else:
        print("  ✅ All symbols have data for all endpoints")

    return coverage


def check_anomalies(dfs: dict, label: str):
    """3. Top-0.1% liquidations, date gaps."""
    hr()
    print(f"  [3] ANOMALY DETECTION — {label}")
    hr()

    liq = dfs.get("liq", pd.DataFrame())
    if liq.empty:
        return

    # Compute total liq
    liq = liq.copy()
    liq["liq_total"] = liq["liq_long_usd"] + liq["liq_short_usd"]

    threshold = liq["liq_total"].quantile(0.999)
    extreme = liq[liq["liq_total"] > threshold].sort_values("liq_total", ascending=False)

    print(f"\n  Top 0.1% liq threshold: ${threshold/1e6:.1f}M")
    print(f"  Top 10 extreme liquidation rows:")
    print(f"  {'Date':<14} {'Symbol':<14} {'liq_total_M':>12} {'long_M':>10} {'short_M':>10}")
    print(f"  {'─'*14} {'─'*14} {'─'*12} {'─'*10} {'─'*10}")
    for _, row in extreme.head(10).iterrows():
        d = str(row["timestamp"])[:10]
        t = row["liq_total"] / 1e6
        l = row["liq_long_usd"] / 1e6
        s = row["liq_short_usd"] / 1e6
        print(f"  {d:<14} {row['symbol']:<14} {t:>12.2f} {l:>10.2f} {s:>10.2f}")

    # Date gap check
    print(f"\n  Date gap check (any gaps > 1 day?):")
    for sym_str, g in liq.groupby("symbol"):
        g = g.sort_values("timestamp")
        gaps = g["timestamp"].diff().dt.days
        big_gaps = gaps[gaps > 1.5]
        if len(big_gaps) > 0:
            sym_short = sym_str.replace("/USDT", "")
            dates = g["timestamp"].iloc[big_gaps.index - 1].dt.date.tolist()[:3]
            print(f"  ⚠️  {sym_short}: {len(big_gaps)} gap(s) > 1 day at {dates}")

    # Zero fraction summary
    print(f"\n  Zero liq_total fraction (all symbols):")
    zero_frac = (liq["liq_total"] == 0).mean()
    print(f"  {zero_frac*100:.1f}% of all rows have zero total liquidation (normal for quiet periods)")


def lookahead_test(dfs: dict, label: str):
    """4. Lookahead test: does liq(t) predict FUTURE or PAST returns better?"""
    hr()
    print(f"  [4] LOOKAHEAD TEST — {label}")
    hr()
    print()
    print("  Hypothesis: if corr(liq(t), |fwd_ret|) >> corr(liq(t), |bwd_ret|) with shift=0,")
    print("  then liq(t) candle CLOSES at t (i.e., covers [t-24h, t)) → no shift needed.")
    print("  If roughly equal → candle OPENS at t (covers [t, t+24h)) → need shift(1).")
    print()

    liq = dfs.get("liq", pd.DataFrame())
    if liq.empty:
        print("  No liq data, skipping")
        return

    # Load OHLCV daily
    daily_ohlcv = load_ohlcv_daily()
    if daily_ohlcv.empty:
        print("  No OHLCV data found, skipping lookahead test")
        return

    # Compute daily returns
    daily_ohlcv = daily_ohlcv.sort_values(["symbol", "date"])
    daily_ohlcv["ret_fwd_1d"] = daily_ohlcv.groupby("symbol")["close_daily"].pct_change(1).shift(-1)
    daily_ohlcv["ret_bwd_1d"] = daily_ohlcv.groupby("symbol")["close_daily"].pct_change(1)

    # Merge liq with daily OHLCV
    liq_m = liq.copy()
    liq_m["liq_total"] = liq_m["liq_long_usd"] + liq_m["liq_short_usd"]
    liq_m["liq_imb"] = (liq_m["liq_long_usd"] - liq_m["liq_short_usd"]) / (liq_m["liq_total"] + 1e-10)
    liq_m["date"] = liq_m["timestamp"]

    merged = liq_m.merge(
        daily_ohlcv[["date", "symbol", "ret_fwd_1d", "ret_bwd_1d", "vol_daily"]],
        on=["date", "symbol"], how="inner"
    )

    if len(merged) < 100:
        print(f"  ⚠️  Too few merged rows ({len(merged)}), check symbol format alignment")
        return

    print(f"  Merged rows: {len(merged):,}")
    print()

    # Test shift(0) and shift(1)
    merged = merged.sort_values(["symbol", "date"])
    merged["liq_total_lag1"] = merged.groupby("symbol")["liq_total"].shift(1)

    # corr(liq(t), |fwd_ret(t+1d)|)
    # corr(liq(t-1d), |fwd_ret(t+1d)|)
    # corr(liq(t), |bwd_ret(t)|)

    sub = merged.dropna(subset=["liq_total", "liq_total_lag1", "ret_fwd_1d", "ret_bwd_1d"])

    c_same_fwd = stats.spearmanr(sub["liq_total"], sub["ret_fwd_1d"].abs())[0]
    c_lag1_fwd = stats.spearmanr(sub["liq_total_lag1"], sub["ret_fwd_1d"].abs())[0]
    c_same_bwd = stats.spearmanr(sub["liq_total"], sub["ret_bwd_1d"].abs())[0]
    c_lag1_bwd = stats.spearmanr(sub["liq_total_lag1"], sub["ret_bwd_1d"].abs())[0]

    print(f"  corr(liq(t),      |ret_fwd(t+1d)|) = {c_same_fwd:+.4f}")
    print(f"  corr(liq(t-1d),   |ret_fwd(t+1d)|) = {c_lag1_fwd:+.4f}")
    print(f"  corr(liq(t),      |ret_bwd(t)|)    = {c_same_bwd:+.4f}")
    print(f"  corr(liq(t-1d),   |ret_bwd(t)|)    = {c_lag1_bwd:+.4f}")

    print()
    # Decision
    if abs(c_same_bwd) > abs(c_same_fwd) * 1.3:
        verdict = "shift=0 — candle [t-24h, t) covering PAST, available at t → USE liq(t) directly for predicting [t, t+1d)"
        rec_shift = 0
    elif abs(c_same_fwd) > abs(c_same_bwd) * 1.3:
        verdict = "shift=1 — candle [t, t+24h) covering FUTURE, available at t+1d → USE liq(t-1d) in model"
        rec_shift = 1
    else:
        verdict = "ambiguous — cannot determine direction from correlation alone; use shift=1 (conservative)"
        rec_shift = 1

    print(f"  ⇒ Verdict: {verdict}")
    print(f"  ⇒ Recommended feature shift: {rec_shift} day(s)")
    print()

    # Additional: directional test (liq_imbalance as predictor)
    c_imb_fwd = stats.spearmanr(sub["liq_imb"], sub["ret_fwd_1d"])[0]
    c_imb_bwd = stats.spearmanr(sub["liq_imb"], sub["ret_bwd_1d"])[0]
    print(f"  corr(liq_imb(t),  ret_fwd(t+1d))   = {c_imb_fwd:+.4f}  (directional, positive=contrarion)")
    print(f"  corr(liq_imb(t),  ret_bwd(t))       = {c_imb_bwd:+.4f}")

    return rec_shift


def oi_chaining_test(dfs: dict, label: str):
    """Confirm OI candle direction: close(t) == open(t+1d)."""
    hr()
    print(f"  [OI CHAINING] — {label}")
    hr()

    oi = dfs.get("oi", pd.DataFrame())
    if oi.empty:
        print("  No OI data")
        return

    oi = oi.copy().sort_values(["symbol", "timestamp"])

    matches = []
    for sym, g in oi.groupby("symbol"):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            continue
        # close(t) should == open(t+1d)
        n_match = (g["oi_close"].iloc[:-1].values == g["oi_open"].iloc[1:].values).mean()
        matches.append(n_match)

    mean_chain = np.mean(matches)
    print(f"\n  OI close(t) == open(t+1d) match rate: {mean_chain*100:.1f}%")
    if mean_chain > 0.8:
        print("  ✅ Confirmed: candle OHLC opens at timestamp, covers [t, t+1d)")
        print("  → liq(t) covers the SAME interval: events from [t, t+1d)")
        print("  → For model at time t: use liq(t-1d) (previous day's COMPLETE candle)")
    else:
        print("  ⚠️  Lower match rate than expected — check data quality")


def zero_fraction_detail(dfs: dict, label: str):
    """Zero fraction per endpoint per symbol."""
    hr()
    print(f"  [5] ZERO / NaN FRACTIONS — {label}")
    hr()

    for ep in ["liq", "taker", "funding", "ls_ratio"]:
        df = dfs.get(ep, pd.DataFrame())
        if df.empty:
            continue

        numeric_cols = [c for c in df.columns if df[c].dtype in [float, "float64"] and c not in ["ls_long_pct", "ls_short_pct"]]
        total = len(df)
        for col in numeric_cols[:2]:  # first 2 numeric per endpoint
            nan_pct = df[col].isna().mean() * 100
            zero_pct = (df[col] == 0).mean() * 100
            print(f"  {ep:10s} {col:25s}: NaN={nan_pct:.1f}%  zero={zero_pct:.1f}%")


# ── main ─────────────────────────────────────────────────────

def run_qa(folder: Path, label: str):
    print(f"\n{'═'*70}")
    print(f"  CoinGlass QA — {label.upper()} ({folder})")
    print(f"{'═'*70}")

    dfs = load_cg_data(folder)
    if not dfs:
        print("  No data found!")
        return

    print(f"\n  Loaded: {', '.join(f'{ep}({len(df):,}r)' for ep, df in dfs.items())}")

    oi_chaining_test(dfs, label)
    align_ok = check_timestamps(dfs, label)
    coverage_table(dfs, label)
    check_anomalies(dfs, label)
    rec_shift = lookahead_test(dfs, label)
    zero_fraction_detail(dfs, label)

    print(f"\n{'═'*70}")
    print(f"  QA SUMMARY — {label.upper()}")
    print(f"{'═'*70}")
    print(f"  Timestamp alignment: {'✅' if align_ok else '❌'}")
    if rec_shift is not None:
        print(f"  Recommended feature shift: {rec_shift} day(s)")
        if rec_shift == 0:
            print("  → In feature code: use liq(t) directly (candle covers past)")
        else:
            print("  → In feature code: use liq.shift(1) = previous day (candle covers future)")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoinGlass Data QA")
    parser.add_argument("--data12h", action="store_true", help="Also run QA on 12h data")
    args = parser.parse_args()

    run_qa(CG_1D, "1d")

    if args.data12h:
        if CG_12H.exists() and any(CG_12H.glob("*.parquet")):
            run_qa(CG_12H, "12h")
        else:
            print(f"\n⚠️  12h data not found at {CG_12H} — run the download first")
