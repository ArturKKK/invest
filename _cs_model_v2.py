#!/usr/bin/env python3
"""
Refined mean-reversion model v2:
1. Only features with proven CS IC > 0.02
2. Regime filter (reduce exposure in momentum)
3. Better portfolio construction
4. Multiple horizons tested
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

PROJECT = Path(__file__).parent

# Strict walk-forward: train → gap → test
WINDOWS = [
    {"name": "W1", "train_end": "2024-06-01", "val_start": "2024-06-15", "val_end": "2024-09-30",
     "test_start": "2024-10-15", "test_end": "2025-01-31"},
    {"name": "W2", "train_end": "2025-01-01", "val_start": "2025-01-15", "val_end": "2025-04-30",
     "test_start": "2025-05-15", "test_end": "2025-08-31"},
    {"name": "W3", "train_end": "2025-07-01", "val_start": "2025-07-15", "val_end": "2025-10-31",
     "test_start": "2025-11-15", "test_end": "2026-03-17"},
]

# Only features with proven CS IC > 0.02 at 12-24h
FEATURES_V2 = [
    # Short-term mean-reversion (strongest CS IC)
    "ret_12h", "ret_24h", "ret_48h",
    # Residual (market-neutral return)
    "residual_12h", "residual_24h",
    # Volatility-adjusted momentum (captures risk)
    "mom_z_12h", "mom_z_24h",
    # Distance from high (bounded, clean signal)
    "dist_from_high_24h",
    # OI (crowding)
    "oi_chg_12h", "oi_chg_24h", "oi_zscore",
    # Taker flow (order flow)
    "taker_cvd_12h", "taker_cvd_24h",
    # L/S positioning
    "ls_divergence",
]

TOP_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
    "MATIC/USDT", "UNI/USDT", "ATOM/USDT", "LTC/USDT", "NEAR/USDT",
    "FIL/USDT", "APT/USDT", "ARB/USDT", "OP/USDT", "AAVE/USDT",
]


def load_and_build():
    """Load data and build features."""
    from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal
    ohlcv = load_ohlcv()
    ohlcv = ohlcv[ohlcv["symbol"].isin(TOP_SYMBOLS)]
    derivs = load_derivatives()
    df = build_features_minimal(ohlcv, derivs)
    return df


def cs_rank(df, col):
    """Cross-sectional rank to [-0.5, 0.5]."""
    return df.groupby("timestamp")[col].rank(pct=True) - 0.5


def compute_regime(df, lookback=168):
    """
    Regime detection: momentum vs mean-reversion.
    Uses BTC trend strength as indicator.
    """
    btc = df[df["symbol"] == "BTC/USDT"][["timestamp", "close"]].copy()
    btc = btc.sort_values("timestamp").drop_duplicates("timestamp")
    btc["btc_ret_7d"] = btc["close"].pct_change(lookback)
    btc["btc_vol_7d"] = btc["close"].pct_change(1).rolling(lookback).std()
    btc["trend_strength"] = btc["btc_ret_7d"].abs() / (btc["btc_vol_7d"] * np.sqrt(lookback) + 1e-10)

    # Past 7d cross-sectional dispersion (mean-reversion works when dispersion is high)
    cs_disp = df.groupby("timestamp")["ret_168h"].std().reset_index()
    cs_disp.columns = ["timestamp", "cs_dispersion"]

    regime = btc[["timestamp", "trend_strength", "btc_ret_7d"]].merge(cs_disp, on="timestamp", how="left")

    # Regime: mean-reversion works when trend is WEAK
    # Scale factor: 1.0 when trend_strength < 1σ, 0.3 when > 2σ
    regime["mr_scale"] = np.clip(1.5 - 0.5 * regime["trend_strength"], 0.2, 1.0)

    return regime[["timestamp", "trend_strength", "mr_scale", "cs_dispersion"]]


def build_portfolio(predictions, n_long=4, n_short=4):
    """
    Build L/S portfolio: long top-n predictions, short bottom-n.
    Weight by prediction magnitude (stronger signal = bigger position).
    """
    positions = {}
    for ts, grp in predictions.groupby("timestamp"):
        if len(grp) < n_long + n_short:
            continue

        sorted_g = grp.sort_values("pred", ascending=False)
        longs = sorted_g.head(n_long)
        shorts = sorted_g.tail(n_short)

        # Equal weight within long/short leg
        for _, row in longs.iterrows():
            positions.setdefault(ts, []).append({
                "symbol": row["symbol"],
                "weight": 1.0 / n_long,
                "side": "long",
                "pred": row["pred"],
            })
        for _, row in shorts.iterrows():
            positions.setdefault(ts, []).append({
                "symbol": row["symbol"],
                "weight": -1.0 / n_short,
                "side": "short",
                "pred": row["pred"],
            })

    return positions


def simulate_ls(df, positions, horizon, regime_df=None):
    """Simulate L/S portfolio returns."""
    fwd_col = f"fwd_ret_{horizon}h"
    returns = []

    for ts, pos_list in sorted(positions.items()):
        mr_scale = 1.0
        if regime_df is not None:
            r = regime_df[regime_df["timestamp"] == ts]
            if len(r) > 0:
                mr_scale = r.iloc[0]["mr_scale"]

        period_ret = 0
        n_pos = 0
        for pos in pos_list:
            row = df[(df["timestamp"] == ts) & (df["symbol"] == pos["symbol"])]
            if len(row) == 0 or pd.isna(row[fwd_col].iloc[0]):
                continue
            actual_ret = row[fwd_col].iloc[0]
            period_ret += pos["weight"] * actual_ret * mr_scale
            n_pos += 1

        if n_pos > 0:
            returns.append({"timestamp": ts, "portfolio_ret": period_ret,
                            "n_pos": n_pos, "mr_scale": mr_scale})

    return pd.DataFrame(returns)


def evaluate_portfolio(ret_df, horizon, name=""):
    """Compute portfolio metrics from L/S returns."""
    # Non-overlapping sub-sample
    ret_df = ret_df.sort_values("timestamp").copy()
    sub = ret_df.iloc[::horizon]

    if len(sub) < 10:
        return {}

    rets = sub["portfolio_ret"]
    periods_per_year = 8760 / horizon

    mean_ret = rets.mean()
    std_ret = rets.std()
    sharpe = mean_ret / (std_ret + 1e-10) * np.sqrt(periods_per_year)

    cum = (1 + rets).cumprod()
    total_ret = cum.iloc[-1] - 1
    max_dd = (cum / cum.cummax() - 1).min()

    hit_rate = (rets > 0).mean()
    n_periods = len(sub)

    t_stat = mean_ret / (std_ret / np.sqrt(n_periods)) if std_ret > 0 else 0

    print(f"\n  {name}")
    print(f"    Periods: {n_periods} (non-overlap, {horizon}h)")
    print(f"    Mean ret/period: {mean_ret*100:>+.4f}%")
    print(f"    Hit rate: {hit_rate:.1%}")
    print(f"    Sharpe: {sharpe:.2f}")
    print(f"    Total return: {total_ret*100:.1f}%")
    print(f"    Max drawdown: {max_dd*100:.1f}%")
    print(f"    t-stat: {t_stat:.2f}")

    return {"sharpe": sharpe, "total_ret": total_ret, "max_dd": max_dd,
            "hit_rate": hit_rate, "t_stat": t_stat, "n": n_periods}


def main():
    print("=" * 80)
    print("REFINED MEAN-REVERSION MODEL v2")
    print("=" * 80)

    print("\n Loading data...")
    df = load_and_build()

    # Compute regime
    print(" Computing regime...")
    regime_df = compute_regime(df)

    FEATURES = [f for f in FEATURES_V2 if f in df.columns]
    print(f" Features: {len(FEATURES)} → {FEATURES}")

    for HORIZON in [12, 24]:
        fwd_col = f"fwd_ret_{HORIZON}h"
        print(f"\n{'#'*80}")
        print(f"  HORIZON: {HORIZON}h")
        print(f"{'#'*80}")

        all_test_preds = []

        for w in WINDOWS:
            train = df[df["timestamp"] < w["train_end"]].copy()
            val = df[(df["timestamp"] >= w["val_start"]) & (df["timestamp"] < w["val_end"])].copy()
            test = df[(df["timestamp"] >= w["test_start"]) & (df["timestamp"] <= w["test_end"])].copy()

            if train.shape[0] < 5000 or test.shape[0] < 200:
                print(f"  {w['name']}: skip")
                continue

            # CS-rank features
            for feat in FEATURES:
                for d in [train, val, test]:
                    d[f"{feat}_r"] = cs_rank(d, feat)

            # CS-rank target
            for d in [train, val, test]:
                d["target_rank"] = d.groupby("timestamp")[fwd_col].rank(pct=True) - 0.5

            feat_r = [f"{f}_r" for f in FEATURES]

            train_clean = train[feat_r + ["target_rank"]].dropna()
            val_clean = val[feat_r + ["target_rank"]].dropna()
            test_clean = test[feat_r + ["target_rank", "timestamp", "symbol", fwd_col]].dropna()

            X_train = train_clean[feat_r].values
            y_train = train_clean["target_rank"].values
            X_val = val_clean[feat_r].values
            y_val = val_clean["target_rank"].values

            # HPO on val
            best_alpha, best_ic = 1.0, -999
            for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
                m = Ridge(alpha=alpha)
                m.fit(X_train, y_train)
                pred = m.predict(X_val)
                ic = stats.spearmanr(pred, y_val)[0]
                if ic > best_ic:
                    best_ic = ic
                    best_alpha = alpha

            # Final model: train on train+val
            m = Ridge(alpha=best_alpha)
            combined_X = np.vstack([X_train, X_val])
            combined_y = np.concatenate([y_train, y_val])
            m.fit(combined_X, combined_y)

            pred_test = m.predict(test_clean[feat_r].values)
            test_clean = test_clean.copy()
            test_clean["pred"] = pred_test

            # CS IC on test
            cs_ics = []
            for ts, grp in test_clean.groupby("timestamp"):
                if len(grp) < 5:
                    continue
                rho, _ = stats.spearmanr(grp["pred"], grp["target_rank"])
                cs_ics.append(rho)
            mean_cs_ic = np.mean(cs_ics)
            std_cs_ic = np.std(cs_ics)
            ir = mean_cs_ic / (std_cs_ic + 1e-10)

            print(f"\n  {w['name']} | alpha={best_alpha} | val_IC={best_ic:.4f}")
            print(f"    Test CS IC: {mean_cs_ic:.4f} ± {std_cs_ic:.4f} (IR={ir:.2f})")
            print(f"    Feature weights (sorted by |w|):")
            for feat, coef in sorted(zip(FEATURES, m.coef_), key=lambda x: abs(x[1]), reverse=True)[:8]:
                print(f"      {feat:<28} {coef:>+.4f}")

            all_test_preds.append(test_clean)

        if not all_test_preds:
            continue

        combined_test = pd.concat(all_test_preds, ignore_index=True)

        # Portfolio simulation
        print(f"\n  {'─'*60}")
        print(f"  PORTFOLIO SIMULATION ({HORIZON}h)")
        print(f"  {'─'*60}")

        # Build positions
        positions_raw = build_portfolio(combined_test, n_long=4, n_short=4)

        # No regime filter
        ret_no_regime = simulate_ls(df, positions_raw, HORIZON, regime_df=None)
        evaluate_portfolio(ret_no_regime, HORIZON, f"[{HORIZON}h] No regime filter")

        # With regime filter
        ret_regime = simulate_ls(df, positions_raw, HORIZON, regime_df=regime_df)
        evaluate_portfolio(ret_regime, HORIZON, f"[{HORIZON}h] With regime filter")

        # Long-only (just top quintile)
        positions_long = {}
        for ts, pos_list in positions_raw.items():
            positions_long[ts] = [p for p in pos_list if p["side"] == "long"]
        ret_long = simulate_ls(df, positions_long, HORIZON, regime_df=None)
        evaluate_portfolio(ret_long, HORIZON, f"[{HORIZON}h] Long-only (top 4)")

        # Short-only
        positions_short = {}
        for ts, pos_list in positions_raw.items():
            positions_short[ts] = [p for p in pos_list if p["side"] == "short"]
        ret_short = simulate_ls(df, positions_short, HORIZON, regime_df=None)
        evaluate_portfolio(ret_short, HORIZON, f"[{HORIZON}h] Short-only (bottom 4)")

    # ── Equal-weight factor model (no ML) ───────────────────────
    print(f"\n{'#'*80}")
    print("  BONUS: EQUAL-WEIGHT FACTOR (no ML, just average rank)")
    print(f"{'#'*80}")

    # Simple: average CS-rank of top features → portfolio
    SIMPLE_FEATURES = ["ret_12h", "ret_24h", "ret_48h", "oi_chg_12h", "oi_chg_24h"]
    SIMPLE_FEATURES = [f for f in SIMPLE_FEATURES if f in df.columns]

    for HORIZON in [12, 24]:
        fwd_col = f"fwd_ret_{HORIZON}h"
        # Only use test period of all windows
        test_periods = []
        for w in WINDOWS:
            t = df[(df["timestamp"] >= w["test_start"]) & (df["timestamp"] <= w["test_end"])].copy()
            if len(t) > 0:
                test_periods.append(t)

        if not test_periods:
            continue

        combined = pd.concat(test_periods, ignore_index=True)

        # Simple composite: average CS-rank of features (INVERTED since all have negative IC)
        for feat in SIMPLE_FEATURES:
            combined[f"{feat}_r"] = cs_rank(combined, feat)

        rank_cols = [f"{f}_r" for f in SIMPLE_FEATURES]
        combined["composite"] = -combined[rank_cols].mean(axis=1)  # negative because IC is negative

        # Add fwd return
        if fwd_col not in combined.columns:
            continue

        combined_clean = combined[["timestamp", "symbol", "composite", fwd_col]].dropna()

        # Build positions from composite
        positions = {}
        for ts, grp in combined_clean.groupby("timestamp"):
            if len(grp) < 8:
                continue
            sorted_g = grp.sort_values("composite", ascending=False)
            longs = sorted_g.head(4)
            shorts = sorted_g.tail(4)
            pos_list = []
            for _, row in longs.iterrows():
                pos_list.append({"symbol": row["symbol"], "weight": 0.25, "side": "long", "pred": row["composite"]})
            for _, row in shorts.iterrows():
                pos_list.append({"symbol": row["symbol"], "weight": -0.25, "side": "short", "pred": row["composite"]})
            positions[ts] = pos_list

        ret_simple = simulate_ls(df, positions, HORIZON)
        evaluate_portfolio(ret_simple, HORIZON, f"[{HORIZON}h] Simple equal-weight factor (no ML)")

        # With regime
        ret_simple_r = simulate_ls(df, positions, HORIZON, regime_df=regime_df)
        evaluate_portfolio(ret_simple_r, HORIZON, f"[{HORIZON}h] Simple factor + regime filter")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
