#!/usr/bin/env python3
"""
Analyze CryptoQuant on-chain data: compute IC, check redundancy.

Input:  data/cryptoquant/cryptoquant_raw.parquet (from download_cryptoquant.py)
Output: Prints IC table, redundancy heatmap, verdict

Can also fetch fresh prices from Binance if local OHLCV is stale.

Usage:
  python src/data/analyze_cryptoquant.py
  python src/data/analyze_cryptoquant.py --fetch-prices   # fetch last 14d prices from Binance
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
CQ_DIR = DATA_DIR / "cryptoquant"


# ─── Price data ──────────────────────────────────────────────────────────────

def load_prices_local() -> pd.DataFrame:
    """Load local OHLCV from data/raw/*.parquet."""
    raw_dir = DATA_DIR / "raw"
    frames = []
    for f in raw_dir.glob("*_USDT_1h.parquet"):
        sym = f.stem.replace("_USDT_1h", "") + "/USDT"
        df = pd.read_parquet(f)
        df["symbol"] = sym
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_prices_binance(days: int = 14) -> pd.DataFrame:
    """Fetch fresh 1h prices from Binance for last N days (no API key needed)."""
    try:
        import ccxt
    except ImportError:
        print("   ⚠️  ccxt not installed. pip install ccxt")
        return pd.DataFrame()

    from datetime import datetime, timezone, timedelta
    since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

    exchange = ccxt.binance({"enableRateLimit": True})
    symbols = [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
        'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'DOT/USDT', 'LINK/USDT',
        'MATIC/USDT', 'UNI/USDT', 'ATOM/USDT', 'LTC/USDT', 'ETC/USDT',
        'FIL/USDT', 'APT/USDT', 'ARB/USDT', 'OP/USDT', 'NEAR/USDT',
        'AAVE/USDT', 'MKR/USDT', 'GRT/USDT', 'INJ/USDT', 'FTM/USDT',
        'ALGO/USDT', 'SAND/USDT', 'MANA/USDT', 'AXS/USDT', 'THETA/USDT',
        'RUNE/USDT', 'EGLD/USDT', 'XTZ/USDT', 'FLOW/USDT', 'CHZ/USDT',
    ]

    frames = []
    for i, sym in enumerate(symbols):
        sys.stdout.write(f"\r   Fetching {sym} ({i+1}/{len(symbols)})...   ")
        try:
            ohlcv = exchange.fetch_ohlcv(sym, '1h', since=since, limit=1000)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df['symbol'] = sym
            frames.append(df)
        except Exception as e:
            print(f"\n   ⚠️  {sym}: {e}")
        time.sleep(0.3)

    sys.stdout.write(f"\r{' '*60}\r")
    if frames:
        prices = pd.concat(frames, ignore_index=True)
        # Cache for reuse
        cache_path = CQ_DIR / "_prices_cache.parquet"
        prices.to_parquet(cache_path, index=False)
        print(f"   💾 Cached {len(prices):,} rows → {cache_path}")
        return prices
    return pd.DataFrame()


def load_prices(fetch: bool = False) -> pd.DataFrame:
    """Load prices, preferring cached fresh data."""
    cache = CQ_DIR / "_prices_cache.parquet"
    if cache.exists():
        prices = pd.read_parquet(cache)
        latest = prices['timestamp'].max()
        print(f"   📋 Using cached prices (latest: {latest})")
        return prices

    if fetch:
        print("   📥 Fetching fresh prices from Binance...")
        return fetch_prices_binance(days=14)

    # Fall back to local
    prices = load_prices_local()
    if not prices.empty:
        latest = prices['timestamp'].max()
        print(f"   📋 Using local prices (latest: {latest})")
    return prices


# ─── IC Analysis ─────────────────────────────────────────────────────────────

def compute_forward_returns(prices: pd.DataFrame, horizon_h: int = 12) -> pd.DataFrame:
    """Compute forward returns for each symbol."""
    frames = []
    for sym, grp in prices.groupby("symbol"):
        grp = grp.sort_values("timestamp").copy()
        grp[f"fwd_ret_{horizon_h}h"] = grp["close"].shift(-horizon_h) / grp["close"] - 1
        frames.append(grp)
    return pd.concat(frames, ignore_index=True)


def compute_daily_cs_ic(feature_df: pd.DataFrame, ret_df: pd.DataFrame,
                        feature_col: str, ret_col: str = "fwd_ret_12h") -> pd.DataFrame:
    """
    Compute daily cross-sectional rank IC between feature and forward return.

    For each date, rank-correlate the feature across coins with forward returns.
    """
    # Ensure we have a date column
    feature_df = feature_df.copy()
    if "timestamp" not in feature_df.columns and "date" in feature_df.columns:
        feature_df["timestamp"] = pd.to_datetime(feature_df["date"])

    # Merge on symbol + date
    feature_df["date"] = pd.to_datetime(feature_df["timestamp"]).dt.date
    ret_df["date"] = pd.to_datetime(ret_df["timestamp"]).dt.date

    # For daily features, we match on date
    # For returns, take the 12:00 UTC snapshot (or daily close)
    daily_rets = ret_df.copy()
    daily_rets["hour"] = pd.to_datetime(daily_rets["timestamp"]).dt.hour
    daily_rets = daily_rets[daily_rets["hour"] == 0]  # midnight snapshot

    merged = feature_df.merge(
        daily_rets[["symbol", "date", ret_col]].dropna(),
        on=["symbol", "date"],
        how="inner"
    )

    if merged.empty:
        return pd.DataFrame()

    # Compute IC per date
    ics = []
    for date, grp in merged.groupby("date"):
        if len(grp) < 5:  # need at least 5 coins for meaningful IC
            continue
        feat = grp[feature_col]
        ret = grp[ret_col]
        if feat.std() == 0 or ret.std() == 0:
            continue
        ic, _ = stats.spearmanr(feat, ret)
        ics.append({"date": date, "ic": ic, "n_coins": len(grp)})

    return pd.DataFrame(ics)


def analyze_metric(cq_df: pd.DataFrame, prices_with_rets: pd.DataFrame,
                   metric_name: str, value_col: str) -> dict:
    """Analyze a single CQ metric: IC, ICIR, sign consistency."""
    # Get data for this metric
    mdf = cq_df[cq_df["metric_name"] == metric_name].copy()
    if mdf.empty or value_col not in mdf.columns:
        return None

    # Drop NaN values
    mdf = mdf.dropna(subset=[value_col])
    if len(mdf) < 5:
        return None

    # Compute daily IC
    ic_df = compute_daily_cs_ic(mdf, prices_with_rets, value_col)
    if ic_df.empty or len(ic_df) < 2:
        return None

    mean_ic = ic_df["ic"].mean()
    std_ic = ic_df["ic"].std()
    icir = mean_ic / std_ic if std_ic > 0 else 0
    hit_rate = (ic_df["ic"] > 0).mean()
    n_days = len(ic_df)
    n_coins_avg = ic_df["n_coins"].mean()

    return {
        "metric": metric_name,
        "value_col": value_col,
        "mean_ic": round(mean_ic, 4),
        "std_ic": round(std_ic, 4),
        "icir": round(icir, 3),
        "hit_rate": round(hit_rate, 3),
        "n_days": n_days,
        "n_coins_avg": round(n_coins_avg, 1),
    }


# ─── Redundancy Check ───────────────────────────────────────────────────────

def check_redundancy(cq_df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Check correlation between CQ metrics and existing features
    (ret_12h, ret_24h, volume, etc.)
    """
    # Compute some basic features from prices
    feats = []
    for sym, grp in prices.groupby("symbol"):
        grp = grp.sort_values("timestamp").copy()
        grp["ret_12h"] = grp["close"].pct_change(12)
        grp["ret_24h"] = grp["close"].pct_change(24)
        grp["vol_24h"] = grp["volume"].rolling(24).sum()
        grp["date"] = grp["timestamp"].dt.date
        # Daily aggregate
        daily = grp.groupby("date").agg(
            ret_12h_last=("ret_12h", "last"),
            ret_24h_last=("ret_24h", "last"),
            vol_24h_last=("vol_24h", "last"),
        ).reset_index()
        daily["symbol"] = sym
        feats.append(daily)

    prices_feats = pd.concat(feats, ignore_index=True)

    # For each CQ metric, compute correlation with ret_12h, ret_24h, volume
    results = []
    for metric_name in cq_df["metric_name"].unique():
        mdf = cq_df[cq_df["metric_name"] == metric_name].copy()
        mdf["date"] = pd.to_datetime(mdf["timestamp"]).dt.date

        # Find numeric columns (the actual values)
        num_cols = mdf.select_dtypes(include=[np.number]).columns
        num_cols = [c for c in num_cols if c not in ("timestamp",)]
        if not num_cols:
            continue

        val_col = num_cols[0]  # primary value column
        merged = mdf[["symbol", "date", val_col]].merge(
            prices_feats, on=["symbol", "date"], how="inner"
        )

        if len(merged) < 10:
            continue

        for ref_col in ["ret_12h_last", "ret_24h_last", "vol_24h_last"]:
            if ref_col in merged.columns:
                valid = merged[[val_col, ref_col]].dropna()
                if len(valid) > 5:
                    corr, _ = stats.spearmanr(valid[val_col], valid[ref_col])
                    results.append({
                        "cq_metric": metric_name,
                        "ref_feature": ref_col,
                        "correlation": round(corr, 3),
                        "n": len(valid),
                    })

    return pd.DataFrame(results)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze CryptoQuant IC")
    parser.add_argument("--fetch-prices", action="store_true",
                        help="Fetch fresh 14d prices from Binance (no API key needed)")
    args = parser.parse_args()

    print("=" * 60)
    print("  CryptoQuant IC Analysis")
    print("=" * 60)

    # Load CQ data
    cq_path = CQ_DIR / "cryptoquant_raw.parquet"
    if not cq_path.exists():
        print(f"\n❌ No CQ data found at {cq_path}")
        print("   Run: python src/data/download_cryptoquant.py")
        sys.exit(1)

    cq_df = pd.read_parquet(cq_path)
    print(f"\n📋 CQ data: {len(cq_df):,} rows, {cq_df['metric_name'].nunique()} metrics, {cq_df['symbol'].nunique()} coins")

    # Load prices
    prices = load_prices(fetch=args.fetch_prices)
    if prices.empty:
        print("\n❌ No price data available!")
        print("   Run with --fetch-prices to download from Binance")
        sys.exit(1)

    # Check date overlap
    cq_dates = pd.to_datetime(cq_df["timestamp"]).dt.date
    price_dates = pd.to_datetime(prices["timestamp"]).dt.date
    overlap = set(cq_dates.unique()) & set(price_dates.unique())
    print(f"   CQ dates:    {cq_dates.min()} → {cq_dates.max()}")
    print(f"   Price dates: {price_dates.min()} → {price_dates.max()}")
    print(f"   Overlap:     {len(overlap)} days")

    if len(overlap) < 2:
        print("\n⚠️  Not enough date overlap! Need --fetch-prices to get recent data.")
        if not args.fetch_prices:
            print("   Retry: python src/data/analyze_cryptoquant.py --fetch-prices")
            sys.exit(1)

    # Compute forward returns
    print("\n📊 Computing forward returns (12h)...")
    prices_with_rets = compute_forward_returns(prices, horizon_h=12)

    # Analyze each metric
    print("\n📊 Computing cross-sectional IC per metric...")
    results = []

    # Find all numeric value columns per metric
    for metric_name in sorted(cq_df["metric_name"].unique()):
        mdf = cq_df[cq_df["metric_name"] == metric_name]
        num_cols = mdf.select_dtypes(include=[np.number]).columns
        num_cols = [c for c in num_cols if c not in ("timestamp",)]

        for val_col in num_cols[:3]:  # top 3 numeric columns
            res = analyze_metric(cq_df, prices_with_rets, metric_name, val_col)
            if res:
                results.append(res)

    if not results:
        print("\n❌ No valid IC results. Check data overlap and coverage.")
        sys.exit(1)

    results_df = pd.DataFrame(results).sort_values("icir", ascending=False)

    # Print results
    print("\n" + "=" * 80)
    print("  IC RESULTS (sorted by ICIR)")
    print("=" * 80)
    print(f"{'Metric':<35} {'Value Col':<15} {'IC':>7} {'ICIR':>7} {'Hit%':>6} {'Days':>5} {'Coins':>6}")
    print("-" * 80)
    for _, row in results_df.iterrows():
        flag = "★" if abs(row["icir"]) > 0.15 else " "
        print(f"{flag} {row['metric']:<34} {row['value_col']:<15} {row['mean_ic']:>7.4f} {row['icir']:>7.3f} {row['hit_rate']:>5.1%} {row['n_days']:>5} {row['n_coins_avg']:>6.1f}")

    # Redundancy check
    print("\n" + "=" * 80)
    print("  REDUNDANCY CHECK (correlation with existing features)")
    print("=" * 80)
    redundancy = check_redundancy(cq_df, prices)
    if not redundancy.empty:
        print(f"{'CQ Metric':<35} {'Ref Feature':<20} {'Corr':>7} {'N':>5}")
        print("-" * 70)
        for _, row in redundancy.iterrows():
            flag = "⚠" if abs(row["correlation"]) > 0.5 else " "
            print(f"{flag} {row['cq_metric']:<34} {row['ref_feature']:<20} {row['correlation']:>7.3f} {row['n']:>5}")

    # Verdict
    print("\n" + "=" * 80)
    interesting = results_df[results_df["icir"].abs() > 0.10]
    non_redundant = []
    if not redundancy.empty and not interesting.empty:
        high_corr_metrics = set(redundancy[redundancy["correlation"].abs() > 0.5]["cq_metric"])
        non_redundant = interesting[~interesting["metric"].isin(high_corr_metrics)]

    print(f"\n  VERDICT:")
    print(f"  Total metrics tested:  {len(results_df)}")
    print(f"  |ICIR| > 0.10:        {len(interesting)}")
    print(f"  Non-redundant:         {len(non_redundant)}")

    if len(non_redundant) > 0:
        print(f"\n  ✅ PROMISING — {len(non_redundant)} metrics worth exploring with Professional tier (1yr history)")
        print(f"     Top candidates:")
        for _, row in non_redundant.head(5).iterrows():
            print(f"       {row['metric']} ({row['value_col']}): ICIR={row['icir']:.3f}")
    elif len(interesting) > 0:
        print(f"\n  ⚠️  SOME SIGNAL but redundant with existing features")
        print(f"     Not worth $99/mo — CoinGlass already covers this")
    else:
        print(f"\n  ❌ NO SIGNAL — don't buy Professional tier")
        print(f"     On-chain data confirmed weak for cross-sectional alpha (consistent with prior tests)")

    # Save results
    results_path = CQ_DIR / "ic_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n   💾 Saved IC results → {results_path}")


if __name__ == "__main__":
    main()
