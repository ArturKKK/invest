#!/usr/bin/env python3
"""
Research: how to fix MaxDD in Ridge model.
Test multiple improvements to reduce the -40% monthly drawdown.
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

PROJECT = Path(__file__).parent

FEATURES = [
    "ret_12h", "ret_24h", "ret_48h",
    "residual_12h", "residual_24h",
    "mom_z_12h", "mom_z_24h",
    "dist_from_high_24h",
    "oi_chg_12h", "oi_chg_24h", "oi_zscore",
    "taker_cvd_12h", "taker_cvd_24h",
    "ls_divergence",
]

TOP_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
    "MATIC/USDT", "UNI/USDT", "ATOM/USDT", "LTC/USDT", "NEAR/USDT",
    "FIL/USDT", "APT/USDT", "ARB/USDT", "OP/USDT", "AAVE/USDT",
]

WINDOWS = [
    {"name": "W1",
     "train_end": "2024-06-01",
     "val_start": "2024-06-01", "val_end": "2024-09-30",
     "test_start": "2024-10-15", "test_end": "2025-01-31"},
    {"name": "W2",
     "train_end": "2025-01-01",
     "val_start": "2025-01-01", "val_end": "2025-04-30",
     "test_start": "2025-05-15", "test_end": "2025-08-31"},
    {"name": "W3",
     "train_end": "2025-07-01",
     "val_start": "2025-07-01", "val_end": "2025-10-31",
     "test_start": "2025-11-15", "test_end": "2026-03-17"},
]

HORIZON = 12
N_LONG, N_SHORT = 4, 4


def cs_rank(df, col):
    return df.groupby("timestamp")[col].rank(pct=True) - 0.5


def load_and_build():
    from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal
    ohlcv = load_ohlcv()
    ohlcv = ohlcv[ohlcv["symbol"].isin(TOP_SYMBOLS)]
    derivs = load_derivatives()
    return build_features_minimal(ohlcv, derivs)


def compute_regime_features(df):
    """Compute multiple regime signals."""
    btc = df[df["symbol"] == "BTC/USDT"][["timestamp", "close"]].copy()
    btc = btc.sort_values("timestamp").drop_duplicates("timestamp")

    # 1. Trend strength (original)
    btc["btc_ret_7d"] = btc["close"].pct_change(168)
    btc["btc_vol_7d"] = btc["close"].pct_change(1).rolling(168).std()
    btc["trend_strength"] = btc["btc_ret_7d"].abs() / (btc["btc_vol_7d"] * np.sqrt(168) + 1e-10)

    # 2. BTC direction (for asymmetric shorts)
    btc["btc_trend_dir"] = np.sign(btc["btc_ret_7d"])

    # 3. Cross-sectional dispersion (from all coins)
    rets_12h = df.groupby("symbol")["close"].pct_change(12)
    dispersion = rets_12h.groupby(df["timestamp"]).std()
    dispersion.name = "cs_dispersion"
    btc = btc.merge(dispersion.reset_index(), on="timestamp", how="left")
    btc["cs_dispersion"] = btc["cs_dispersion"].fillna(btc["cs_dispersion"].median())

    # Rolling median dispersion for relative measure
    btc["disp_median_30d"] = btc["cs_dispersion"].rolling(720, min_periods=100).median()
    btc["disp_ratio"] = btc["cs_dispersion"] / (btc["disp_median_30d"] + 1e-10)

    # 4. Average correlation (approximated by dispersion / vol)
    avg_vol = rets_12h.groupby(df["timestamp"]).mean().abs()
    avg_vol.name = "avg_abs_ret"
    btc = btc.merge(avg_vol.reset_index(), on="timestamp", how="left")

    return btc.set_index("timestamp")


def train_windows(df, feats):
    """Train Ridge models per window, return predictions + test data."""
    fwd_col = f"fwd_ret_{HORIZON}h"
    feat_r = [f"{f}_r" for f in feats]
    results = []

    for w in WINDOWS:
        train = df[df["timestamp"] < w["train_end"]].copy()
        val = df[(df["timestamp"] >= w["val_start"]) & (df["timestamp"] < w["val_end"])].copy()
        test = df[(df["timestamp"] >= w["test_start"]) & (df["timestamp"] <= w["test_end"])].copy()

        if len(train) < 5000 or len(test) < 200:
            continue

        for d in [train, val, test]:
            for feat in feats:
                d[f"{feat}_r"] = cs_rank(d, feat)
            d["target_rank"] = d.groupby("timestamp")[fwd_col].rank(pct=True) - 0.5

        train_c = train[feat_r + ["target_rank"]].dropna()
        val_c = val[feat_r + ["target_rank"]].dropna()
        test_c = test[feat_r + ["target_rank", "timestamp", "symbol"]].dropna()

        best_alpha, best_ic = 1.0, -999
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
            m = Ridge(alpha=alpha)
            m.fit(train_c[feat_r], train_c["target_rank"])
            ic = stats.spearmanr(m.predict(val_c[feat_r]), val_c["target_rank"])[0]
            if ic > best_ic:
                best_ic = ic
                best_alpha = alpha

        m = Ridge(alpha=best_alpha)
        X_all = pd.concat([train_c[feat_r], val_c[feat_r]])
        y_all = pd.concat([train_c["target_rank"], val_c["target_rank"]])
        m.fit(X_all, y_all)

        test_c = test_c.copy()
        test_c["pred"] = m.predict(test_c[feat_r])

        fwd_data = test[["timestamp", "symbol", fwd_col]].rename(
            columns={fwd_col: "fwd_ret"}).dropna()
        merged = test_c[["timestamp", "symbol", "pred"]].merge(
            fwd_data, on=["timestamp", "symbol"], how="inner")

        results.append(merged)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def simulate(merged, regime_df, config):
    """Run L/S simulation with configurable regime/filter rules."""
    name = config["name"]
    trend_cutoff = config.get("trend_cutoff", 999)  # trend_strength above this → go flat
    min_dispersion_ratio = config.get("min_disp_ratio", 0)  # skip if dispersion too low
    asymmetric_shorts = config.get("asymmetric_shorts", False)  # reduce shorts in uptrend
    short_scale_bull = config.get("short_scale_bull", 0.5)  # short weight multiplier in bull
    mr_scale_min = config.get("mr_scale_min", 0.2)
    mr_formula = config.get("mr_formula", "original")  # "original" or "aggressive"

    all_rets = []
    for ts, grp in merged.groupby("timestamp"):
        if ts not in regime_df.index:
            continue

        regime = regime_df.loc[ts]
        trend_str = regime.get("trend_strength", 0)
        btc_dir = regime.get("btc_trend_dir", 0)
        disp_ratio = regime.get("disp_ratio", 1.0)

        # Filter: skip if trend too strong
        if trend_str > trend_cutoff:
            continue

        # Filter: skip if dispersion too low
        if disp_ratio < min_dispersion_ratio:
            continue

        # MR scale
        if mr_formula == "aggressive":
            mr_scale = float(np.clip(1.2 - 0.8 * trend_str, 0.0, 1.0))
        else:
            mr_scale = float(np.clip(1.5 - 0.5 * trend_str, mr_scale_min, 1.0))

        if mr_scale < 0.05:
            continue

        # Rank and pick L/S
        grp = grp.copy()
        grp["pred_rank"] = grp["pred"].rank(ascending=False)
        n = len(grp)
        n_l = min(N_LONG, n // 3)
        n_s = min(N_SHORT, n // 3)

        long_mask = grp["pred_rank"] <= n_l
        short_mask = grp["pred_rank"] > (n - n_s)

        long_ret = grp.loc[long_mask, "fwd_ret"].mean() if long_mask.sum() > 0 else 0
        short_ret = grp.loc[short_mask, "fwd_ret"].mean() if short_mask.sum() > 0 else 0

        long_weight = 0.5
        short_weight = 0.5

        # Asymmetric: reduce shorts in bull trend
        if asymmetric_shorts and btc_dir > 0:
            short_weight *= short_scale_bull
            # Redistribute to long
            long_weight = 1.0 - short_weight

        port_ret = (long_weight * long_ret - short_weight * short_ret) * mr_scale

        all_rets.append({"timestamp": ts, "portfolio_ret": port_ret, "mr_scale": mr_scale})

    if not all_rets:
        return None

    port = pd.DataFrame(all_rets).sort_values("timestamp")
    sub = port.iloc[::HORIZON]
    rets = sub["portfolio_ret"]

    if len(rets) < 10:
        return None

    ppy = 8760 / HORIZON
    sharpe = rets.mean() / (rets.std() + 1e-10) * np.sqrt(ppy)
    cum = (1 + rets).cumprod()
    total = cum.iloc[-1] - 1
    maxdd = (cum / cum.cummax() - 1).min()
    hit = (rets > 0).mean()

    # Monthly worst
    sub_df = sub.copy()
    sub_df["month"] = sub_df["timestamp"].dt.to_period("M")
    monthly = sub_df.groupby("month")["portfolio_ret"].apply(
        lambda x: (1 + x * 3).prod() - 1)  # 3x leverage
    worst_month = monthly.min()
    best_month = monthly.max()

    return {
        "name": name,
        "sharpe": sharpe,
        "total": total,
        "maxdd": maxdd,
        "hit": hit,
        "n_periods": len(rets),
        "worst_month_3x": worst_month,
        "best_month_3x": best_month,
        "monthly": monthly,
        "rets": rets,
    }


def main():
    print("=" * 70)
    print("  RESEARCH: Reducing MaxDD in Ridge Model")
    print("=" * 70)

    df = load_and_build()
    feats = [f for f in FEATURES if f in df.columns]
    print(f"  Loaded: {df.shape[0]:,} rows, {len(feats)} features")

    print("\n  Training models (walk-forward)...")
    merged = train_windows(df, feats)
    print(f"  Predictions: {len(merged):,} rows")

    regime_df = compute_regime_features(df)
    print(f"  Regime features computed")

    # ── Test configurations ────────────────────────────────────
    configs = [
        # Baseline
        {"name": "0. Baseline (original)",
         "mr_scale_min": 0.2},

        # 1. Aggressive regime: shut down completely in strong trends
        {"name": "1. Aggressive regime (cutoff=1.5)",
         "mr_formula": "aggressive"},

        # 2. Hard trend cutoff
        {"name": "2. Hard cutoff (trend>1.2 → flat)",
         "trend_cutoff": 1.2},

        # 3. Even harder cutoff
        {"name": "3. Hard cutoff (trend>0.8 → flat)",
         "trend_cutoff": 0.8},

        # 4. Dispersion filter
        {"name": "4. Dispersion filter (ratio>0.7)",
         "min_disp_ratio": 0.7},

        # 5. Dispersion + aggressive regime
        {"name": "5. Disp>0.7 + aggressive regime",
         "min_disp_ratio": 0.7,
         "mr_formula": "aggressive"},

        # 6. Asymmetric shorts (halve shorts in bull)
        {"name": "6. Asymmetric shorts (0.5x in bull)",
         "asymmetric_shorts": True,
         "short_scale_bull": 0.5},

        # 7. Kill shorts in bull
        {"name": "7. No shorts in bull trend",
         "asymmetric_shorts": True,
         "short_scale_bull": 0.0},

        # 8. Combo: aggressive regime + no shorts in bull
        {"name": "8. Aggr regime + no shorts in bull",
         "mr_formula": "aggressive",
         "asymmetric_shorts": True,
         "short_scale_bull": 0.0},

        # 9. Combo: dispersion + aggressive + asymmetric
        {"name": "9. Disp>0.7 + aggr + no shorts in bull",
         "min_disp_ratio": 0.7,
         "mr_formula": "aggressive",
         "asymmetric_shorts": True,
         "short_scale_bull": 0.0},

        # 10. Combo: trend cutoff 1.0 + asymmetric
        {"name": "10. Cutoff 1.0 + asymmetric 0.3x",
         "trend_cutoff": 1.0,
         "asymmetric_shorts": True,
         "short_scale_bull": 0.3},

        # 11. Dispersion > 0.5 + cutoff 1.2 + asymmetric
        {"name": "11. Disp>0.5 + cutoff 1.2 + asym 0.3",
         "min_disp_ratio": 0.5,
         "trend_cutoff": 1.2,
         "asymmetric_shorts": True,
         "short_scale_bull": 0.3},
    ]

    print(f"\n{'─' * 100}")
    print(f"  {'Config':<45s} {'Sharpe':>7s} {'Total':>8s} {'MaxDD':>7s} "
          f"{'Hit':>6s} {'N':>5s} {'WorstM':>8s} {'BestM':>8s}")
    print(f"{'─' * 100}")

    results = []
    for cfg in configs:
        r = simulate(merged, regime_df, cfg)
        if r is None:
            print(f"  {cfg['name']:<45s}  (no data)")
            continue
        results.append(r)
        print(f"  {r['name']:<45s} {r['sharpe']:>+7.2f} {r['total']*100:>+7.1f}% "
              f"{r['maxdd']*100:>+6.1f}% {r['hit']:>5.1%} {r['n_periods']:>5d} "
              f"{r['worst_month_3x']*100:>+7.1f}% {r['best_month_3x']*100:>+7.1f}%")

    # Show top 3 by Sharpe/MaxDD tradeoff
    print(f"\n{'=' * 100}")
    print(f"  TOP 3 by Sharpe (with worst month > -25% at 3x leverage):")
    print(f"{'=' * 100}")
    filtered = [r for r in results if r['worst_month_3x'] > -0.25]
    if not filtered:
        filtered = sorted(results, key=lambda x: x['worst_month_3x'], reverse=True)[:3]
        print("  (none meet -25% filter, showing least-bad)")

    for r in sorted(filtered, key=lambda x: x['sharpe'], reverse=True)[:3]:
        print(f"\n  📊 {r['name']}")
        print(f"     Sharpe={r['sharpe']:.2f} | Total={r['total']*100:+.1f}% | "
              f"MaxDD={r['maxdd']*100:.1f}% | "
              f"Worst month (3x)={r['worst_month_3x']*100:+.1f}%")
        print(f"     Monthly P&L (3x, $100):")
        equity = 100
        for month, ret in r['monthly'].items():
            pnl = equity * ret
            marker = " ← worst" if ret == r['worst_month_3x'] else ""
            print(f"       {str(month):>10s}  {ret*100:>+7.1f}%  ${pnl:>+7.1f}{marker}")
            equity += pnl
        print(f"       Итого: ${equity:.1f}")


if __name__ == "__main__":
    main()
