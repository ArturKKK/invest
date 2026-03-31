#!/usr/bin/env python3
"""
Cross-sectional IC verification + Simple mean-reversion model.
Computes IC per TIMESTAMP (pure cross-section), not pooled.
Then: builds a Ridge-based simple model with top features.
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

PROJECT = Path(__file__).parent

# Walk-forward windows (strict OOS)
WINDOWS = [
    {"name": "W1", "train_end": "2024-07-01", "val_end": "2024-10-01",
     "test_start": "2024-10-15", "test_end": "2024-12-31"},
    {"name": "W2", "train_end": "2025-01-01", "val_end": "2025-04-01",
     "test_start": "2025-04-15", "test_end": "2025-06-30"},
    {"name": "W3", "train_end": "2025-07-01", "val_end": "2025-10-01",
     "test_start": "2025-10-15", "test_end": "2026-03-17"},
]

HORIZONS = [4, 12, 24, 48]

TOP_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
    "MATIC/USDT", "UNI/USDT", "ATOM/USDT", "LTC/USDT", "NEAR/USDT",
    "FIL/USDT", "APT/USDT", "ARB/USDT", "OP/USDT", "AAVE/USDT",
]


def load_and_build():
    """Load data and build features (reuse logic from IC scanner)."""
    from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal
    ohlcv = load_ohlcv()
    ohlcv = ohlcv[ohlcv["symbol"].isin(TOP_SYMBOLS)]
    derivs = load_derivatives()
    df = build_features_minimal(ohlcv, derivs)
    return df


def cross_sectional_ic(df, feat_col, target_col):
    """Compute IC per timestamp (pure cross-sectional), return series."""
    ic_per_ts = []
    for ts, grp in df.groupby("timestamp"):
        x = grp[feat_col].dropna()
        y = grp[target_col].reindex(x.index).dropna()
        common = x.index.intersection(y.index)
        if len(common) < 5:
            continue
        rho, _ = stats.spearmanr(x[common], y[common])
        ic_per_ts.append({"timestamp": ts, "ic": rho})
    return pd.DataFrame(ic_per_ts).set_index("timestamp")["ic"]


def cross_sectional_rank_per_ts(df, col):
    """Rank a column cross-sectionally per timestamp to [-0.5, 0.5]."""
    return df.groupby("timestamp")[col].rank(pct=True) - 0.5


def main():
    print("=" * 80)
    print("CROSS-SECTIONAL IC + SIMPLE MODEL")
    print("=" * 80)

    print("\n Loading data & building features...")
    df = load_and_build()
    print(f"   Shape: {df.shape[0]:,} rows")

    # Features to focus on (from IC scanner results)
    CANDIDATE_FEATURES = [
        # Mean-reversion (past returns)
        "ret_168h", "ret_48h", "ret_24h", "ret_12h",
        # Momentum z-scores
        "mom_z_24h", "mom_z_12h",
        # Distance from high
        "dist_from_high_24h",
        # OI
        "oi_zscore", "oi_chg_24h", "oi_chg_12h",
        # Funding / carry
        "funding_rate_binance", "cum_funding_24h", "cum_funding_72h",
        "funding_zscore",
        # Positioning
        "top_ls_ratio_zscore", "ls_divergence",
        # Premium
        "premium_index", "premium_zscore",
        # Taker flow
        "taker_cvd_12h", "taker_cvd_24h",
        # Volume
        "vol_ratio_12h",
        # Residual
        "residual_24h",
        # BTC
        "btc_ret_24h",
    ]
    CANDIDATE_FEATURES = [f for f in CANDIDATE_FEATURES if f in df.columns]
    print(f"   Candidate features: {len(CANDIDATE_FEATURES)}")

    # ── Step 1: Cross-sectional IC ──────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 1: PURE CROSS-SECTIONAL IC (per-timestamp)")
    print("=" * 80)

    for h in HORIZONS:
        target = f"fwd_ret_{h}h"
        if target not in df.columns:
            continue

        # Use most recent OOS window
        w = WINDOWS[-1]
        test = df[(df["timestamp"] >= w["test_start"]) & (df["timestamp"] <= w["test_end"])].copy()
        if test.shape[0] < 500:
            continue

        print(f"\n  --- Horizon: {h}h | Window: {w['name']} ({w['test_start']} → {w['test_end']}) ---")
        print(f"  {'Feature':<28} {'CS IC':>8} {'t-stat':>7} {'Hit%':>6} {'n_ts':>6}")
        print(f"  {'─'*28} {'─'*8} {'─'*7} {'─'*6} {'─'*6}")

        ic_results = []
        for feat in CANDIDATE_FEATURES:
            ic_series = cross_sectional_ic(test, feat, target)
            if len(ic_series) < 20:
                continue
            mean_ic = ic_series.mean()
            std_ic = ic_series.std()
            t_stat = mean_ic / (std_ic / np.sqrt(len(ic_series))) if std_ic > 0 else 0
            hit_rate = (ic_series > 0).mean() if mean_ic > 0 else (ic_series < 0).mean()
            flag = "✅" if abs(mean_ic) > 0.02 else "  "
            print(f"  {flag} {feat:<26} {mean_ic:>+8.4f} {t_stat:>+7.2f} {hit_rate:>5.0%} {len(ic_series):>6}")
            ic_results.append({"feature": feat, "cs_ic": mean_ic, "t_stat": t_stat,
                               "hit_rate": hit_rate, "horizon": h})

        ic_df = pd.DataFrame(ic_results)
        if not ic_df.empty:
            top = ic_df.reindex(ic_df["cs_ic"].abs().nlargest(5).index)
            print(f"\n  Top 5 by |CS IC|: {', '.join(top['feature'].tolist())}")

    # ── Step 2: Simple model (Ridge on CS-ranked features) ──────
    print("\n" + "=" * 80)
    print("STEP 2: SIMPLE RIDGE MODEL (walk-forward)")
    print("=" * 80)

    # Select strongest features (top 10 by IC from scanner)
    MODEL_FEATURES = [
        "ret_168h", "ret_48h", "oi_zscore", "cum_funding_24h",
        "funding_rate_binance", "top_ls_ratio_zscore",
        "oi_chg_24h", "mom_z_24h", "dist_from_high_24h",
        "premium_zscore",
    ]
    MODEL_FEATURES = [f for f in MODEL_FEATURES if f in df.columns]
    print(f"\n  Model features ({len(MODEL_FEATURES)}): {MODEL_FEATURES}")

    all_test_results = []

    for h in [12, 24, 48]:
        target = f"fwd_ret_{h}h"
        fwd_col = target
        print(f"\n  {'='*60}")
        print(f"  HORIZON: {h}h")
        print(f"  {'='*60}")

        for w in WINDOWS:
            # Training data: CS-rank features, train target as CS rank
            train = df[(df["timestamp"] < w["train_end"])].copy()
            val = df[(df["timestamp"] >= w["train_end"]) & (df["timestamp"] < w["val_end"])].copy()
            test = df[(df["timestamp"] >= w["test_start"]) & (df["timestamp"] <= w["test_end"])].copy()

            if train.shape[0] < 5000 or test.shape[0] < 200:
                print(f"    {w['name']}: skip (train={train.shape[0]}, test={test.shape[0]})")
                continue

            # CS-rank features per timestamp
            for feat in MODEL_FEATURES:
                train[f"{feat}_r"] = cross_sectional_rank_per_ts(train, feat)
                val[f"{feat}_r"] = cross_sectional_rank_per_ts(val, feat)
                test[f"{feat}_r"] = cross_sectional_rank_per_ts(test, feat)

            # CS-rank target
            for d in [train, val, test]:
                d["target_rank"] = d.groupby("timestamp")[fwd_col].rank(pct=True) - 0.5

            feat_r = [f"{f}_r" for f in MODEL_FEATURES]

            # Drop NaN
            train_clean = train[feat_r + ["target_rank"]].dropna()
            val_clean = val[feat_r + ["target_rank"]].dropna()
            test_clean = test[feat_r + ["target_rank", "timestamp", "symbol", fwd_col]].dropna()

            if len(train_clean) < 1000 or len(test_clean) < 100:
                print(f"    {w['name']}: skip after dropna (train={len(train_clean)}, test={len(test_clean)})")
                continue

            X_train, y_train = train_clean[feat_r].values, train_clean["target_rank"].values
            X_val, y_val = val_clean[feat_r].values, val_clean["target_rank"].values
            X_test = test_clean[feat_r].values
            y_test = test_clean["target_rank"].values

            # Find best alpha on val
            best_alpha, best_ic = 0.1, -999
            for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
                m = Ridge(alpha=alpha)
                m.fit(X_train, y_train)
                pred = m.predict(X_val)
                ic_val = stats.spearmanr(pred, y_val)[0]
                if ic_val > best_ic:
                    best_ic = ic_val
                    best_alpha = alpha

            # Train on train+val, test on test
            m = Ridge(alpha=best_alpha)
            combined_X = np.vstack([X_train, X_val])
            combined_y = np.concatenate([y_train, y_val])
            m.fit(combined_X, combined_y)

            pred = m.predict(X_test)
            test_clean["pred"] = pred

            # Metrics
            overall_ic = stats.spearmanr(pred, y_test)[0]

            # Per-timestamp IC
            cs_ics = []
            for ts, grp in test_clean.groupby("timestamp"):
                if len(grp) < 5:
                    continue
                rho, _ = stats.spearmanr(grp["pred"], grp["target_rank"])
                cs_ics.append(rho)
            mean_cs_ic = np.mean(cs_ics) if cs_ics else 0
            cs_ic_std = np.std(cs_ics) if cs_ics else 1
            cs_ir = mean_cs_ic / (cs_ic_std + 1e-10)  # IC IR

            # L/S returns
            for ts, grp in test_clean.groupby("timestamp"):
                if len(grp) < 5:
                    continue
                top_q = grp.nlargest(max(1, len(grp) // 5), "pred")
                bot_q = grp.nsmallest(max(1, len(grp) // 5), "pred")
                all_test_results.append({
                    "window": w["name"],
                    "horizon": h,
                    "timestamp": ts,
                    "long_ret": top_q[fwd_col].mean(),
                    "short_ret": bot_q[fwd_col].mean(),
                    "n_symbols": len(grp),
                })

            # Feature weights
            print(f"\n    {w['name']} | alpha={best_alpha} | val_IC={best_ic:.4f}")
            print(f"    Test: overall IC={overall_ic:.4f}, CS IC={mean_cs_ic:.4f}, IC IR={cs_ir:.2f}")
            print(f"    Feature weights:")
            for feat, coef in sorted(zip(MODEL_FEATURES, m.coef_), key=lambda x: abs(x[1]), reverse=True):
                print(f"      {feat:<28} {coef:>+8.4f}")

    # ── Step 3: L/S performance ─────────────────────────────────
    if all_test_results:
        print("\n" + "=" * 80)
        print("STEP 3: LONG/SHORT PERFORMANCE")
        print("=" * 80)

        res_df = pd.DataFrame(all_test_results)

        for h in res_df["horizon"].unique():
            for w_name in res_df["window"].unique():
                sub = res_df[(res_df["horizon"] == h) & (res_df["window"] == w_name)]
                if sub.empty:
                    continue

                # Compute L/S spread per rebalance
                sub = sub.copy()
                sub["ls_spread"] = sub["long_ret"] - sub["short_ret"]

                # Since we rebalance hourly but hold for h hours, we have overlapping returns
                # Subsample every h hours for non-overlapping
                sub_sorted = sub.sort_values("timestamp")
                sub_nonoverlap = sub_sorted.iloc[::h]

                mean_spread = sub_nonoverlap["ls_spread"].mean()
                std_spread = sub_nonoverlap["ls_spread"].std()
                n = len(sub_nonoverlap)
                t_stat = mean_spread / (std_spread / np.sqrt(n)) if std_spread > 0 else 0

                # Annualized
                periods_per_year = 8760 / h  # hours per year / holding period
                ann_ret = mean_spread * periods_per_year
                ann_vol = std_spread * np.sqrt(periods_per_year)
                sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

                hit_rate = (sub_nonoverlap["ls_spread"] > 0).mean()

                print(f"\n  {w_name} | {h}h horizon | n={n} non-overlap periods")
                print(f"  Mean L/S spread: {mean_spread*100:>+.4f}% per period")
                print(f"  Hit rate: {hit_rate:.1%}")
                print(f"  Annualized: ret={ann_ret*100:.1f}%, vol={ann_vol*100:.1f}%, Sharpe={sharpe:.2f}")
                print(f"  t-stat: {t_stat:.2f}")

                # Cumulative
                cum_ret = (1 + sub_nonoverlap["ls_spread"]).cumprod()
                total_ret = cum_ret.iloc[-1] - 1 if len(cum_ret) > 0 else 0
                max_dd = (cum_ret / cum_ret.cummax() - 1).min()
                print(f"  Total return: {total_ret*100:.1f}%, Max DD: {max_dd*100:.1f}%")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
