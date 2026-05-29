#!/usr/bin/env python3
"""
IC Scanner: Test every feature's raw predictive power at multiple horizons.
Goal: Find which features (if any) have IC > 0.02 on TRUE OOS data.

This uses walk-forward windows to ensure no data leakage.
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT = Path(__file__).parent
DATA_DIR = PROJECT / "data"

# ── Walk-forward windows (proper OOS, no overlap) ──────────────────────
WINDOWS = [
    {"name": "W1", "train_end": "2024-07-01", "test_start": "2024-07-15", "test_end": "2024-12-31"},
    {"name": "W2", "train_end": "2025-01-01", "test_start": "2025-01-15", "test_end": "2025-06-30"},
    {"name": "W3", "train_end": "2025-07-01", "test_start": "2025-07-15", "test_end": "2026-03-17"},
]

HORIZONS = [1, 4, 12, 24, 48]  # hours ahead

TOP_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
    "MATIC/USDT", "UNI/USDT", "ATOM/USDT", "LTC/USDT", "NEAR/USDT",
    "FIL/USDT", "APT/USDT", "ARB/USDT", "OP/USDT", "AAVE/USDT",
]


def load_ohlcv():
    """Load and merge all OHLCV data."""
    frames = []
    raw_dir = DATA_DIR / "raw"
    for f in sorted(raw_dir.glob("*_1h.parquet")):
        symbol = f.stem.replace("_1h", "").replace("_", "/")
        df = pd.read_parquet(f)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["symbol"] = symbol
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return df


def load_derivatives():
    """Load all derivatives data."""
    sent_dir = DATA_DIR / "sentiment"

    dfs = {}

    # Funding rates (every 8h, we'll forward-fill to hourly)
    fr = pd.read_parquet(sent_dir / "binance_funding_rates.parquet")
    fr["timestamp"] = pd.to_datetime(fr["timestamp"], utc=True)
    dfs["funding"] = fr

    # Futures metrics (hourly)
    fm = pd.read_parquet(sent_dir / "binance_futures_metrics.parquet")
    fm["timestamp"] = pd.to_datetime(fm["timestamp"], utc=True)
    dfs["futures"] = fm

    # Premium index
    pi = pd.read_parquet(sent_dir / "binance_premium_index.parquet")
    pi["timestamp"] = pd.to_datetime(pi["timestamp"], utc=True)
    dfs["premium"] = pi

    # DVOL
    dv = pd.read_parquet(sent_dir / "deribit_dvol.parquet")
    dv["timestamp"] = pd.to_datetime(dv["timestamp"], utc=True)
    dfs["dvol"] = dv

    return dfs


def build_features_minimal(ohlcv, derivs):
    """
    Build a MINIMAL set of candidate features — focused on what MIGHT work:
    1. Momentum (returns at various lookbacks)
    2. Volatility
    3. Funding rate (carry)
    4. OI changes
    5. Taker flow
    6. Premium / basis
    7. Cross-asset (BTC residual)
    """
    df = ohlcv.copy()

    # ── Price features ──────────────────────────────────────────
    for h in [1, 4, 12, 24, 48, 168]:
        df[f"ret_{h}h"] = df.groupby("symbol")["close"].pct_change(h)

    # Realized vol
    df["ret_1h_sq"] = df["ret_1h"] ** 2
    for h in [12, 24, 168]:
        df[f"rvol_{h}h"] = df.groupby("symbol")["ret_1h_sq"].transform(
            lambda x: x.rolling(h, min_periods=h // 2).mean().apply(np.sqrt)
        )
    df.drop(columns=["ret_1h_sq"], inplace=True)

    # Volume ratio
    for h in [12, 24]:
        df[f"vol_ratio_{h}h"] = df.groupby("symbol")["volume"].transform(
            lambda x: x.rolling(h).mean() / x.rolling(168).mean()
        )

    # Momentum z-score (ret / rvol)
    for h in [12, 24]:
        df[f"mom_z_{h}h"] = df[f"ret_{h}h"] / (df[f"rvol_{h}h"] + 1e-10)

    # Mean-reversion signal (distance from 24h high/low)
    df["range_24h"] = df.groupby("symbol")["high"].transform(
        lambda x: x.rolling(24).max()
    ) - df.groupby("symbol")["low"].transform(
        lambda x: x.rolling(24).min()
    )
    df["dist_from_high_24h"] = (
        df.groupby("symbol")["high"].transform(lambda x: x.rolling(24).max()) - df["close"]
    ) / (df["range_24h"] + 1e-10)

    # ── BTC factor ──────────────────────────────────────────────
    btc = df[df["symbol"] == "BTC/USDT"][["timestamp", "close"]].rename(
        columns={"close": "btc_close"}
    )
    df = df.merge(btc, on="timestamp", how="left")
    for h in [1, 4, 12, 24]:
        df[f"btc_ret_{h}h"] = df.groupby("symbol")["btc_close"].pct_change(h)

    # Rolling beta
    df["coin_ret"] = df["ret_1h"]
    df["btc_ret"] = df[f"btc_ret_1h"]
    for sym, g in df.groupby("symbol"):
        cov = g["coin_ret"].rolling(168, min_periods=84).cov(g["btc_ret"])
        var = g["btc_ret"].rolling(168, min_periods=84).var()
        df.loc[g.index, "btc_beta_168h"] = cov / (var + 1e-10)

    # Residual return (alpha = ret - beta * btc_ret)
    for h in [12, 24]:
        df[f"residual_{h}h"] = df[f"ret_{h}h"] - df["btc_beta_168h"] * df[f"btc_ret_{h}h"]

    # ── Funding rate features ───────────────────────────────────
    fr = derivs["funding"].copy()
    fr = fr.sort_values(["symbol", "timestamp"])
    # Forward-fill to hourly
    fr_hourly = []
    for sym, g in fr.groupby("symbol"):
        g = g.set_index("timestamp").resample("1h").ffill()
        g["symbol"] = sym
        fr_hourly.append(g.reset_index())
    fr = pd.concat(fr_hourly, ignore_index=True)

    df = df.merge(fr[["timestamp", "symbol", "funding_rate_binance"]], on=["timestamp", "symbol"], how="left")
    df["funding_rate_binance"] = df.groupby("symbol")["funding_rate_binance"].ffill()

    # Cumulative funding (carry signal)
    for h in [24, 72, 168]:
        df[f"cum_funding_{h}h"] = df.groupby("symbol")["funding_rate_binance"].transform(
            lambda x: x.rolling(h, min_periods=h // 2).sum()
        )

    # Funding z-score
    fr_mean = df.groupby("symbol")["funding_rate_binance"].transform(
        lambda x: x.rolling(168, min_periods=84).mean()
    )
    fr_std = df.groupby("symbol")["funding_rate_binance"].transform(
        lambda x: x.rolling(168, min_periods=84).std()
    ) + 1e-10
    df["funding_zscore"] = (df["funding_rate_binance"] - fr_mean) / fr_std

    # Funding × momentum interaction (key: are longs paying high funding in an uptrend?)
    df["funding_x_mom_12h"] = df["funding_rate_binance"] * df["ret_12h"]
    df["funding_x_mom_24h"] = df["funding_rate_binance"] * df["ret_24h"]

    # ── OI features ─────────────────────────────────────────────
    fm = derivs["futures"].copy()
    fm = fm.sort_values(["symbol", "timestamp"])
    df = df.merge(
        fm[["timestamp", "symbol", "oi_value_usd", "taker_buy_sell_ratio",
            "top_ls_ratio", "global_ls_ratio"]],
        on=["timestamp", "symbol"], how="left"
    )

    # OI changes
    for h in [1, 4, 12, 24]:
        df[f"oi_chg_{h}h"] = df.groupby("symbol")["oi_value_usd"].pct_change(h)

    # OI z-score
    oi_mean = df.groupby("symbol")["oi_value_usd"].transform(
        lambda x: x.rolling(168, min_periods=84).mean()
    )
    oi_std = df.groupby("symbol")["oi_value_usd"].transform(
        lambda x: x.rolling(168, min_periods=84).std()
    ) + 1e-10
    df["oi_zscore"] = (df["oi_value_usd"] - oi_mean) / oi_std

    # OI × return divergence (OI up + price down = short buildup)
    df["oi_ret_diverge"] = df["oi_chg_12h"] - df["ret_12h"]

    # ── Taker flow ──────────────────────────────────────────────
    df["taker_imbalance"] = (df["taker_buy_sell_ratio"] - 1) / (df["taker_buy_sell_ratio"] + 1 + 1e-10)
    for h in [4, 12, 24]:
        df[f"taker_cvd_{h}h"] = df.groupby("symbol")["taker_imbalance"].transform(
            lambda x: x.rolling(h, min_periods=h // 2).sum()
        )

    # Taker z-score
    tk_mean = df.groupby("symbol")["taker_imbalance"].transform(
        lambda x: x.rolling(168, min_periods=84).mean()
    )
    tk_std = df.groupby("symbol")["taker_imbalance"].transform(
        lambda x: x.rolling(168, min_periods=84).std()
    ) + 1e-10
    df["taker_zscore"] = (df["taker_imbalance"] - tk_mean) / tk_std

    # ── L/S ratio features ──────────────────────────────────────
    for col in ["top_ls_ratio", "global_ls_ratio"]:
        mean = df.groupby("symbol")[col].transform(lambda x: x.rolling(168, min_periods=84).mean())
        std = df.groupby("symbol")[col].transform(lambda x: x.rolling(168, min_periods=84).std()) + 1e-10
        df[f"{col}_zscore"] = (df[col] - mean) / std

    df["ls_divergence"] = df["top_ls_ratio"] - df["global_ls_ratio"]

    # ── Premium index ───────────────────────────────────────────
    pi = derivs["premium"].copy()
    pi = pi.sort_values(["symbol", "timestamp"])
    df = df.merge(pi[["timestamp", "symbol", "premium_index"]], on=["timestamp", "symbol"], how="left")

    pi_mean = df.groupby("symbol")["premium_index"].transform(
        lambda x: x.rolling(168, min_periods=84).mean()
    )
    pi_std = df.groupby("symbol")["premium_index"].transform(
        lambda x: x.rolling(168, min_periods=84).std()
    ) + 1e-10
    df["premium_zscore"] = (df["premium_index"] - pi_mean) / pi_std

    # NOTE (R127, 2026-04-23): previous "Fix#1" did `replace([inf,-inf], nan)`
    # here to guard against OI/volume pct_change dividing by zero on illiquid
    # hours. Ablation (F10_F20 vs F11_F20) showed this costs 0.55 Sharpe.
    # On historical 2022-2026 data no inf actually appears; VPS prod runs
    # WITHOUT this cleanup (commit ccb3bc2) and matches backtest 3.777.
    # If LIVE ever produces inf from new illiquid symbols, add a safety net
    # in run_trading.py inference path (model.predict → fillna(0)) instead
    # of here. See PROGRESS.md R127.

    # ── Forward returns (targets) ───────────────────────────────
    for h in HORIZONS:
        df[f"fwd_ret_{h}h"] = df.groupby("symbol")["close"].transform(
            lambda x: x.pct_change(h).shift(-h)
        )

    return df


def compute_ic(feature, target, method="spearman"):
    """Compute information coefficient (rank correlation)."""
    mask = feature.notna() & target.notna()
    if mask.sum() < 100:
        return np.nan
    if method == "spearman":
        return stats.spearmanr(feature[mask], target[mask])[0]
    return stats.pearsonr(feature[mask], target[mask])[0]


def compute_ic_by_period(df, feat_col, target_col, freq="W"):
    """Compute IC per time period, return mean and t-stat."""
    df_sub = df[[feat_col, target_col, "timestamp"]].dropna()
    if len(df_sub) < 200:
        return np.nan, np.nan, 0

    df_sub = df_sub.set_index("timestamp")
    ic_series = df_sub.resample(freq).apply(
        lambda x: stats.spearmanr(x[feat_col], x[target_col])[0]
        if len(x) > 20 else np.nan
    )
    ic_series = ic_series.dropna()

    if len(ic_series) < 4:
        return np.nan, np.nan, 0

    mean_ic = ic_series.mean()
    ic_std = ic_series.std()
    t_stat = mean_ic / (ic_std / np.sqrt(len(ic_series))) if ic_std > 0 else 0
    return mean_ic, t_stat, len(ic_series)


def main():
    print("=" * 80)
    print("IC SCANNER: Finding features with real predictive power")
    print("=" * 80)

    # ── Load data ───────────────────────────────────────────────
    print("\n📊 Loading OHLCV data...")
    ohlcv = load_ohlcv()
    print(f"   OHLCV: {ohlcv.shape[0]:,} rows, {ohlcv['symbol'].nunique()} symbols")
    print(f"   Range: {ohlcv['timestamp'].min()} → {ohlcv['timestamp'].max()}")

    # Filter to top symbols (more liquid = cleaner signal)
    ohlcv = ohlcv[ohlcv["symbol"].isin(TOP_SYMBOLS)]
    print(f"   After filtering to top {len(TOP_SYMBOLS)}: {ohlcv.shape[0]:,} rows")

    print("\n📊 Loading derivatives data...")
    derivs = load_derivatives()

    print("\n🔧 Building features...")
    df = build_features_minimal(ohlcv, derivs)
    print(f"   Final: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # Identify feature columns
    exclude = {"timestamp", "symbol", "open", "high", "low", "close", "volume",
               "btc_close", "coin_ret", "btc_ret"}
    feat_cols = [c for c in df.columns
                 if c not in exclude
                 and not c.startswith("fwd_ret_")
                 and df[c].dtype in ["float64", "float32", "int64"]]
    print(f"   Features to test: {len(feat_cols)}")

    # ── IC Scan per walk-forward window ─────────────────────────
    results = []

    for w in WINDOWS:
        print(f"\n{'='*60}")
        print(f"WINDOW {w['name']}: test {w['test_start']} → {w['test_end']}")
        test = df[(df["timestamp"] >= w["test_start"]) & (df["timestamp"] <= w["test_end"])].copy()
        print(f"   Test rows: {test.shape[0]:,}")

        if test.shape[0] < 500:
            print("   ⚠️ Too few rows, skipping")
            continue

        for h in HORIZONS:
            target = f"fwd_ret_{h}h"
            if target not in test.columns or test[target].dropna().shape[0] < 100:
                continue

            for feat in feat_cols:
                mean_ic, t_stat, n_weeks = compute_ic_by_period(
                    test, feat, target, freq="W"
                )
                if pd.isna(mean_ic):
                    continue

                results.append({
                    "window": w["name"],
                    "horizon": h,
                    "feature": feat,
                    "mean_ic": mean_ic,
                    "t_stat": t_stat,
                    "n_weeks": n_weeks,
                    "abs_ic": abs(mean_ic),
                })

    if not results:
        print("\n❌ No results! Check data.")
        return

    res = pd.DataFrame(results)

    # ── Analysis ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RESULTS: Features ranked by average |IC| across windows")
    print("=" * 80)

    for h in HORIZONS:
        sub = res[res["horizon"] == h]
        if sub.empty:
            continue

        # Average across windows
        avg = sub.groupby("feature").agg(
            mean_ic=("mean_ic", "mean"),
            abs_ic=("abs_ic", "mean"),
            mean_tstat=("t_stat", "mean"),
            n_windows=("window", "nunique"),
        ).sort_values("abs_ic", ascending=False)

        # Filter: must appear in all windows
        avg = avg[avg["n_windows"] >= 2]

        print(f"\n{'─'*60}")
        print(f"  HORIZON: {h}h forward return")
        print(f"{'─'*60}")
        print(f"{'Feature':<30} {'Mean IC':>8} {'|IC|':>6} {'t-stat':>7} {'Win':>4}")
        print(f"{'─'*30} {'─'*8} {'─'*6} {'─'*7} {'─'*4}")

        for feat, row in avg.head(25).iterrows():
            flag = "✅" if row["abs_ic"] > 0.02 else "  "
            print(f"{flag} {feat:<28} {row['mean_ic']:>+8.4f} {row['abs_ic']:>6.4f} {row['mean_tstat']:>7.2f} {int(row['n_windows']):>4}")

    # ── Consistency check: features that work at MULTIPLE horizons ──
    print("\n" + "=" * 80)
    print("CONSISTENCY: Features with |IC| > 0.015 at 2+ horizons")
    print("=" * 80)

    avg_all = res.groupby(["feature", "horizon"]).agg(
        abs_ic=("abs_ic", "mean"),
        n_windows=("window", "nunique"),
    ).reset_index()
    avg_all = avg_all[avg_all["n_windows"] >= 2]

    good = avg_all[avg_all["abs_ic"] > 0.015]
    feat_counts = good.groupby("feature")["horizon"].count()
    multi_horizon = feat_counts[feat_counts >= 2].index.tolist()

    if multi_horizon:
        for feat in multi_horizon:
            sub = good[good["feature"] == feat].sort_values("horizon")
            ics = ", ".join([f"{int(r['horizon'])}h:{r['abs_ic']:.4f}" for _, r in sub.iterrows()])
            print(f"  {feat:<30} → {ics}")
    else:
        print("  ⚠️ No features consistently good at multiple horizons!")

    # ── Simple factor strategy backtest ─────────────────────────
    print("\n" + "=" * 80)
    print("FACTOR STRATEGIES: Simple long-short on single features")
    print("=" * 80)

    # Use last window for most recent OOS
    last_w = WINDOWS[-1]
    test = df[(df["timestamp"] >= last_w["test_start"]) & (df["timestamp"] <= last_w["test_end"])].copy()

    if test.shape[0] > 0:
        for h in [4, 12, 24]:
            target = f"fwd_ret_{h}h"
            if target not in test.columns:
                continue

            print(f"\n  --- Horizon: {h}h ---")

            # Test top 5 features by |IC|
            sub = res[(res["horizon"] == h) & (res["window"] == last_w["name"])]
            if sub.empty:
                continue
            top5 = sub.nlargest(5, "abs_ic")

            for _, row in top5.iterrows():
                feat = row["feature"]
                d = test[[feat, target, "timestamp", "symbol"]].dropna()
                if len(d) < 200:
                    continue

                # Long-short: go long top quintile, short bottom quintile
                d["rank"] = d.groupby("timestamp")[feat].rank(pct=True)
                long_ret = d[d["rank"] > 0.8][target].mean()
                short_ret = d[d["rank"] < 0.2][target].mean()
                spread = long_ret - short_ret

                # Sign-based IC
                ic = row["mean_ic"]
                direction = "long high" if ic > 0 else "short high"

                print(f"    {feat:<28} IC={ic:>+.4f}  L/S spread={spread*100:>+.3f}%  ({direction})")

    # ── Save results ────────────────────────────────────────────
    out_path = PROJECT / "ic_scan_results.csv"
    res.to_csv(out_path, index=False)
    print(f"\n📁 Full results saved to {out_path}")


if __name__ == "__main__":
    main()
