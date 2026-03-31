#!/usr/bin/env python3
"""
Strict walk-forward simulation of Ridge model.
ZERO data leakage: train → val (alpha HPO) → test, no overlap.

Prints clear data boundaries and $ projections.
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

# Strict walk-forward: train_end < val_start, val_end + GAP < test_start
WINDOWS = [
    {"name": "W1",
     "train_end":  "2024-06-01",
     "val_start":  "2024-06-01",  "val_end":   "2024-09-30",
     "test_start": "2024-10-15",  "test_end":  "2025-01-31"},
    {"name": "W2",
     "train_end":  "2025-01-01",
     "val_start":  "2025-01-01",  "val_end":   "2025-04-30",
     "test_start": "2025-05-15",  "test_end":  "2025-08-31"},
    {"name": "W3",
     "train_end":  "2025-07-01",
     "val_start":  "2025-07-01",  "val_end":   "2025-10-31",
     "test_start": "2025-11-15",  "test_end":  "2026-03-17"},
]


def cs_rank(df, col):
    return df.groupby("timestamp")[col].rank(pct=True) - 0.5


def compute_regime(df):
    btc = df[df["symbol"] == "BTC/USDT"][["timestamp", "close"]].copy()
    btc = btc.sort_values("timestamp").drop_duplicates("timestamp")
    btc["btc_ret_7d"] = btc["close"].pct_change(168)
    btc["btc_vol_7d"] = btc["close"].pct_change(1).rolling(168).std()
    btc["trend_strength"] = btc["btc_ret_7d"].abs() / (btc["btc_vol_7d"] * np.sqrt(168) + 1e-10)
    btc["mr_scale"] = np.clip(1.5 - 0.5 * btc["trend_strength"], 0.2, 1.0)
    return btc[["timestamp", "mr_scale"]].set_index("timestamp")["mr_scale"]


def main():
    from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal

    print("=" * 70)
    print("  STRICT WALK-FORWARD SIMULATION — Ridge Mean-Reversion")
    print("  No data leakage. Train → Val → Test with gaps.")
    print("=" * 70)

    df = load_and_build()
    regime_series = compute_regime(df)
    feats = [f for f in FEATURES if f in df.columns]
    print(f"\n  Data: {df['symbol'].nunique()} symbols, {df.shape[0]:,} rows")
    print(f"  Date range: {df['timestamp'].min():%Y-%m-%d} → {df['timestamp'].max():%Y-%m-%d}")
    print(f"  Features: {len(feats)}/{len(FEATURES)}")

    HORIZON = 12
    fwd_col = f"fwd_ret_{HORIZON}h"
    N_LONG, N_SHORT = 4, 4

    all_window_rets = []

    for w in WINDOWS:
        print(f"\n{'─' * 70}")
        print(f"  {w['name']}:")
        print(f"    TRAIN:  everything < {w['train_end']}")
        print(f"    VAL:    {w['val_start']} → {w['val_end']}")
        print(f"    TEST:   {w['test_start']} → {w['test_end']}")
        print(f"    GAP:    {w['val_end']} → {w['test_start']} (15 days)")

        train = df[df["timestamp"] < w["train_end"]].copy()
        val = df[(df["timestamp"] >= w["val_start"]) & (df["timestamp"] < w["val_end"])].copy()
        test = df[(df["timestamp"] >= w["test_start"]) & (df["timestamp"] <= w["test_end"])].copy()

        # LEAK CHECK: no test timestamps in train or val
        test_ts = set(test["timestamp"])
        train_ts = set(train["timestamp"])
        val_ts = set(val["timestamp"])
        overlap_train = test_ts & train_ts
        overlap_val = test_ts & val_ts
        assert len(overlap_train) == 0, f"LEAK: {len(overlap_train)} test timestamps in train!"
        assert len(overlap_val) == 0, f"LEAK: {len(overlap_val)} test timestamps in val!"
        print(f"    ✅ No data leakage (0 overlapping timestamps)")
        print(f"    Rows: train={len(train):,}, val={len(val):,}, test={len(test):,}")

        if len(train) < 5000 or len(test) < 200:
            print(f"    ⚠️  Insufficient data, skipping")
            continue

        feat_r = [f"{f}_r" for f in feats]
        for d in [train, val, test]:
            for feat in feats:
                d[f"{feat}_r"] = cs_rank(d, feat)
            d["target_rank"] = d.groupby("timestamp")[fwd_col].rank(pct=True) - 0.5

        train_c = train[feat_r + ["target_rank"]].dropna()
        val_c = val[feat_r + ["target_rank"]].dropna()
        test_c = test[feat_r + ["target_rank", "timestamp", "symbol"]].dropna()

        # HPO on VAL only
        best_alpha, best_ic = 1.0, -999
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
            m = Ridge(alpha=alpha)
            m.fit(train_c[feat_r], train_c["target_rank"])
            ic = stats.spearmanr(m.predict(val_c[feat_r]), val_c["target_rank"])[0]
            if ic > best_ic:
                best_ic = ic
                best_alpha = alpha

        # Retrain on train+val, predict test
        m = Ridge(alpha=best_alpha)
        X_all = pd.concat([train_c[feat_r], val_c[feat_r]])
        y_all = pd.concat([train_c["target_rank"], val_c["target_rank"]])
        m.fit(X_all, y_all)

        test_c = test_c.copy()
        test_c["pred"] = m.predict(test_c[feat_r])

        # CS IC on test
        cs_ics = []
        for ts, grp in test_c.groupby("timestamp"):
            if len(grp) >= 5:
                rho, _ = stats.spearmanr(grp["pred"], grp["target_rank"])
                cs_ics.append(rho)
        mic = np.mean(cs_ics)
        print(f"    α={best_alpha} | val_IC={best_ic:.4f} | test CS IC={mic:.4f}")

        # Simulate L/S portfolio
        fwd_data = test[["timestamp", "symbol", fwd_col]].rename(
            columns={fwd_col: "fwd_ret"}).dropna()
        merged = test_c[["timestamp", "symbol", "pred"]].merge(
            fwd_data, on=["timestamp", "symbol"], how="inner")

        merged["pred_rank"] = merged.groupby("timestamp")["pred"].rank(ascending=False)
        n_syms = merged.groupby("timestamp")["pred_rank"].transform("count")

        merged["side"] = 0.0
        merged.loc[merged["pred_rank"] <= N_LONG, "side"] = 1.0 / N_LONG
        merged.loc[merged["pred_rank"] > (n_syms - N_SHORT), "side"] = -1.0 / N_SHORT

        positions = merged[merged["side"] != 0].copy()
        regime_map = regime_series.to_dict()
        positions["mr_scale"] = positions["timestamp"].map(regime_map).fillna(1.0)
        positions["weighted_ret"] = positions["side"] * positions["fwd_ret"] * positions["mr_scale"]

        portfolio = positions.groupby("timestamp").agg(
            portfolio_ret=("weighted_ret", "sum"),
            mr_scale=("mr_scale", "first"),
        ).reset_index().sort_values("timestamp")

        # Non-overlapping returns (every 12h)
        sub = portfolio.iloc[::HORIZON]
        rets = sub["portfolio_ret"]

        ppy = 8760 / HORIZON
        mean_ret = rets.mean()
        std_ret = rets.std()
        sharpe = mean_ret / (std_ret + 1e-10) * np.sqrt(ppy)
        cum = (1 + rets).cumprod()
        total = cum.iloc[-1] - 1
        max_dd = (cum / cum.cummax() - 1).min()
        hit = (rets > 0).mean()

        print(f"    📊 {len(sub)} periods | Sharpe={sharpe:.2f} | "
              f"Total={total*100:+.1f}% | MaxDD={max_dd*100:.1f}% | Hit={hit:.1%}")

        all_window_rets.append(sub[["timestamp", "portfolio_ret"]].copy().assign(window=w["name"]))

    # ── Aggregate results ──────────────────────────────────────
    if not all_window_rets:
        print("\n❌ No results")
        return

    all_rets = pd.concat(all_window_rets).sort_values("timestamp")
    rets_all = all_rets["portfolio_ret"]

    ppy = 8760 / HORIZON
    sharpe_all = rets_all.mean() / (rets_all.std() + 1e-10) * np.sqrt(ppy)
    cum_all = (1 + rets_all).cumprod()
    total_all = cum_all.iloc[-1] - 1
    maxdd_all = (cum_all / cum_all.cummax() - 1).min()
    hit_all = (rets_all > 0).mean()
    n_months = (all_rets["timestamp"].max() - all_rets["timestamp"].min()).days / 30.4

    print(f"\n{'=' * 70}")
    print(f"  AGGREGATE (all OOS windows, {n_months:.0f} months)")
    print(f"  Sharpe: {sharpe_all:.2f}")
    print(f"  Total return: {total_all*100:+.1f}%")
    print(f"  MaxDD: {maxdd_all*100:.1f}%")
    print(f"  Hit rate: {hit_all:.1%}")
    print(f"  Periods: {len(rets_all)}")
    print(f"{'=' * 70}")

    # ── $ PROJECTIONS ──────────────────────────────────────────
    # Based on per-period stats from OOS
    mean_per_period = rets_all.mean()  # mean 12h return
    periods_per_day = 24 / HORIZON

    CAPITAL = 100
    LEVERAGE = 3

    # Daily return (compounding 2 periods/day)
    daily_ret = (1 + mean_per_period * LEVERAGE) ** periods_per_day - 1
    daily_std = rets_all.std() * LEVERAGE * np.sqrt(periods_per_day)

    week_ret = (1 + daily_ret) ** 7 - 1
    month_ret = (1 + daily_ret) ** 30 - 1
    quarter_ret = (1 + daily_ret) ** 90 - 1

    # Conservative estimate: use median instead of mean (more robust)
    med_per_period = rets_all.median()
    daily_ret_med = (1 + med_per_period * LEVERAGE) ** periods_per_day - 1
    week_ret_med = (1 + daily_ret_med) ** 7 - 1
    month_ret_med = (1 + daily_ret_med) ** 30 - 1
    quarter_ret_med = (1 + daily_ret_med) ** 90 - 1

    # Worst-case: MaxDD with leverage
    worst_dd = maxdd_all * LEVERAGE

    print(f"\n{'=' * 70}")
    print(f"  💰 PROJECTIONS ($100 capital, 3x leverage)")
    print(f"{'=' * 70}")
    print(f"  Mean 12h return: {mean_per_period*100:+.3f}%")
    print(f"  Median 12h return: {med_per_period*100:+.3f}%")
    print(f"")
    print(f"  {'':20s} {'Optimistic (mean)':>20s}  {'Conservative (median)':>22s}")
    print(f"  {'1 неделя':20s} ${CAPITAL * week_ret:>+18.2f}  ${CAPITAL * week_ret_med:>+20.2f}")
    print(f"  {'1 месяц':20s} ${CAPITAL * month_ret:>+18.2f}  ${CAPITAL * month_ret_med:>+20.2f}")
    print(f"  {'3 месяца':20s} ${CAPITAL * quarter_ret:>+18.2f}  ${CAPITAL * quarter_ret_med:>+20.2f}")
    print(f"")
    print(f"  ⚠️  MaxDD (историч.): {worst_dd*100:.0f}% = -${abs(CAPITAL * worst_dd):.0f}")
    print(f"{'=' * 70}")

    # Per-month breakdown
    all_rets["month"] = all_rets["timestamp"].dt.to_period("M")
    monthly = all_rets.groupby("month")["portfolio_ret"].apply(
        lambda x: (1 + x * LEVERAGE).prod() - 1).reset_index()
    monthly.columns = ["month", "return"]
    print(f"\n  📅 ПОМЕСЯЧНАЯ ДОХОДНОСТЬ (3x leverage, $100):")
    print(f"  {'Месяц':>10s}  {'Return':>8s}  {'P&L':>8s}")
    equity = CAPITAL
    for _, row in monthly.iterrows():
        pnl = equity * row["return"]
        print(f"  {str(row['month']):>10s}  {row['return']*100:>+7.1f}%  ${pnl:>+7.1f}")
        equity += pnl
    print(f"\n  Итого equity: ${equity:.1f} (было ${CAPITAL})")


def load_and_build():
    from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal
    ohlcv = load_ohlcv()
    ohlcv = ohlcv[ohlcv["symbol"].isin(TOP_SYMBOLS)]
    derivs = load_derivatives()
    return build_features_minimal(ohlcv, derivs)


if __name__ == "__main__":
    main()
