#!/usr/bin/env python3
"""
Final mean-reversion model: clean, fast, production-ready.
Ridge model with 14 CS-IC-verified features + regime filter.
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

PROJECT = Path(__file__).parent

WINDOWS = [
    {"name": "W1", "train_end": "2024-06-01", "val_end": "2024-09-30",
     "test_start": "2024-10-15", "test_end": "2025-01-31"},
    {"name": "W2", "train_end": "2025-01-01", "val_end": "2025-04-30",
     "test_start": "2025-05-15", "test_end": "2025-08-31"},
    {"name": "W3", "train_end": "2025-07-01", "val_end": "2025-10-31",
     "test_start": "2025-11-15", "test_end": "2026-03-17"},
]

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


def load_and_build():
    from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal
    ohlcv = load_ohlcv()
    ohlcv = ohlcv[ohlcv["symbol"].isin(TOP_SYMBOLS)]
    derivs = load_derivatives()
    return build_features_minimal(ohlcv, derivs)


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


def fast_simulate(test_predictions, fwd_returns, regime_series, horizon, n_long=4, n_short=4):
    """
    Fast vectorized L/S simulation.
    test_predictions: DataFrame with columns [timestamp, symbol, pred]
    fwd_returns: DataFrame with columns [timestamp, symbol, fwd_ret]
    """
    # Merge predictions with forward returns
    merged = test_predictions.merge(fwd_returns, on=["timestamp", "symbol"], how="inner")
    if merged.empty:
        return pd.DataFrame()

    # Rank predictions per timestamp
    merged["pred_rank"] = merged.groupby("timestamp")["pred"].rank(ascending=False)
    n_syms = merged.groupby("timestamp")["pred_rank"].transform("count")

    # Top n_long = long, bottom n_short = short
    merged["side"] = 0.0
    merged.loc[merged["pred_rank"] <= n_long, "side"] = 1.0 / n_long
    merged.loc[merged["pred_rank"] > (n_syms - n_short), "side"] = -1.0 / n_short

    # Only positions
    positions = merged[merged["side"] != 0].copy()

    # Apply regime filter
    regime_map = regime_series.to_dict()
    positions["mr_scale"] = positions["timestamp"].map(regime_map).fillna(1.0)
    positions["weighted_ret"] = positions["side"] * positions["fwd_ret"] * positions["mr_scale"]

    # Aggregate per timestamp
    portfolio = positions.groupby("timestamp").agg(
        portfolio_ret=("weighted_ret", "sum"),
        n_pos=("side", lambda x: (x != 0).sum()),
        mr_scale=("mr_scale", "first"),
    ).reset_index()

    return portfolio


def backtest(portfolio_df, horizon, name=""):
    """Evaluate portfolio from per-timestamp returns."""
    port = portfolio_df.sort_values("timestamp").copy()
    # Non-overlapping subsample
    sub = port.iloc[::horizon]

    if len(sub) < 10:
        print(f"  {name}: too few periods ({len(sub)})")
        return {}

    rets = sub["portfolio_ret"]
    ppy = 8760 / horizon

    mean_ret = rets.mean()
    std_ret = rets.std()
    sharpe = mean_ret / (std_ret + 1e-10) * np.sqrt(ppy)

    cum = (1 + rets).cumprod()
    total = cum.iloc[-1] - 1
    max_dd = (cum / cum.cummax() - 1).min()
    hit = (rets > 0).mean()
    t = mean_ret / (std_ret / np.sqrt(len(sub))) if std_ret > 0 else 0

    print(f"\n  {name}")
    print(f"    n={len(sub)} periods | mean={mean_ret*100:+.4f}%/period | hit={hit:.1%}")
    print(f"    Sharpe={sharpe:.2f} | Total={total*100:+.1f}% | MaxDD={max_dd*100:.1f}% | t={t:.2f}")

    return {"sharpe": sharpe, "total": total, "max_dd": max_dd, "hit": hit, "t": t}


def main():
    print("=" * 80)
    print("FINAL MEAN-REVERSION MODEL v3")
    print("=" * 80)

    df = load_and_build()
    regime_series = compute_regime(df)
    feats = [f for f in FEATURES if f in df.columns]
    print(f" {len(feats)} features | {df['symbol'].nunique()} symbols | {df.shape[0]:,} rows")

    for HORIZON in [12, 24]:
        fwd_col = f"fwd_ret_{HORIZON}h"
        print(f"\n{'#'*80}")
        print(f"  HORIZON: {HORIZON}h")
        print(f"{'#'*80}")

        all_preds = []
        all_fwd = []

        for w in WINDOWS:
            train = df[df["timestamp"] < w["train_end"]].copy()
            val = df[(df["timestamp"] >= w["train_end"]) & (df["timestamp"] < w["val_end"])].copy()
            test = df[(df["timestamp"] >= w["test_start"]) & (df["timestamp"] <= w["test_end"])].copy()

            if train.shape[0] < 5000 or test.shape[0] < 200:
                continue

            feat_r = [f"{f}_r" for f in feats]
            for d in [train, val, test]:
                for feat in feats:
                    d[f"{feat}_r"] = cs_rank(d, feat)
                d["target_rank"] = d.groupby("timestamp")[fwd_col].rank(pct=True) - 0.5

            train_c = train[feat_r + ["target_rank"]].dropna()
            val_c = val[feat_r + ["target_rank"]].dropna()
            test_c = test[feat_r + ["target_rank", "timestamp", "symbol"]].dropna()

            # HPO
            best_alpha, best_ic = 1.0, -999
            for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
                m = Ridge(alpha=alpha)
                m.fit(train_c[feat_r], train_c["target_rank"])
                ic = stats.spearmanr(m.predict(val_c[feat_r]), val_c["target_rank"])[0]
                if ic > best_ic:
                    best_ic = ic
                    best_alpha = alpha

            # Train final
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
            ir = mic / (np.std(cs_ics) + 1e-10)

            print(f"\n  {w['name']} | α={best_alpha} | val_IC={best_ic:.4f} | test CS IC={mic:.4f} (IR={ir:.2f})")
            print(f"    Top weights: ", end="")
            for feat, c in sorted(zip(feats, m.coef_), key=lambda x: abs(x[1]), reverse=True)[:5]:
                print(f"{feat}={c:+.3f} ", end="")
            print()

            all_preds.append(test_c[["timestamp", "symbol", "pred"]])
            all_fwd.append(test[[f"timestamp", "symbol", fwd_col]].rename(columns={fwd_col: "fwd_ret"}).dropna())

        if not all_preds:
            continue

        preds = pd.concat(all_preds, ignore_index=True)
        fwds = pd.concat(all_fwd, ignore_index=True).drop_duplicates(["timestamp", "symbol"])

        # ── Simulations ───────────────────────────────────
        # No regime
        port_nr = fast_simulate(preds, fwds, pd.Series(1.0, index=regime_series.index), HORIZON)
        backtest(port_nr, HORIZON, f"[{HORIZON}h] L/S no regime")

        # With regime
        port_r = fast_simulate(preds, fwds, regime_series, HORIZON)
        backtest(port_r, HORIZON, f"[{HORIZON}h] L/S + regime filter")

        # Per-window breakdown
        for w in WINDOWS:
            sub = port_r[(port_r["timestamp"] >= w["test_start"]) & (port_r["timestamp"] <= w["test_end"])]
            if len(sub) > 50:
                backtest(sub, HORIZON, f"  └─ {w['name']} ({w['test_start'][:7]} → {w['test_end'][:7]})")

    # ── No-ML baseline: equal-weight factor ─────────────
    print(f"\n{'#'*80}")
    print("  NO-ML BASELINE: Simple factor average")
    print(f"{'#'*80}")

    SIMPLE = ["ret_12h", "ret_24h", "ret_48h", "oi_chg_12h", "oi_chg_24h"]
    SIMPLE = [f for f in SIMPLE if f in df.columns]

    for HORIZON in [12, 24]:
        fwd_col = f"fwd_ret_{HORIZON}h"
        test_parts = []
        for w in WINDOWS:
            t = df[(df["timestamp"] >= w["test_start"]) & (df["timestamp"] <= w["test_end"])].copy()
            if len(t) > 0:
                test_parts.append(t)
        if not test_parts:
            continue

        combined = pd.concat(test_parts, ignore_index=True)
        for feat in SIMPLE:
            combined[f"{feat}_r"] = cs_rank(combined, feat)

        # Composite = negative average rank (since all ICs are negative)
        rank_cols = [f"{f}_r" for f in SIMPLE]
        combined["pred"] = -combined[rank_cols].mean(axis=1)

        preds_simple = combined[["timestamp", "symbol", "pred"]].dropna()
        fwds_simple = combined[["timestamp", "symbol", fwd_col]].rename(columns={fwd_col: "fwd_ret"}).dropna()

        port = fast_simulate(preds_simple, fwds_simple, regime_series, HORIZON)
        backtest(port, HORIZON, f"[{HORIZON}h] No-ML factor + regime")

        port_nr = fast_simulate(preds_simple, fwds_simple, pd.Series(1.0, index=regime_series.index), HORIZON)
        backtest(port_nr, HORIZON, f"[{HORIZON}h] No-ML factor (no regime)")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
