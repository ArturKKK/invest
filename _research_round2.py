#!/usr/bin/env python3
"""
Research round 2: more improvement ideas for Ridge model.
1. More symbols (30+)
2. Multi-horizon blend (12h + 24h)
3. Momentum overlay in trends
4. Dynamic position count
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

SYMBOLS_20 = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
    "MATIC/USDT", "UNI/USDT", "ATOM/USDT", "LTC/USDT", "NEAR/USDT",
    "FIL/USDT", "APT/USDT", "ARB/USDT", "OP/USDT", "AAVE/USDT",
]

# Expanded universe
SYMBOLS_35 = SYMBOLS_20 + [
    "ALGO/USDT", "FTM/USDT", "SAND/USDT", "MANA/USDT", "AXS/USDT",
    "CRV/USDT", "DYDX/USDT", "IMX/USDT", "INJ/USDT", "TIA/USDT",
    "SUI/USDT", "SEI/USDT", "WLD/USDT", "PEPE/USDT", "WIF/USDT",
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


def cs_rank(df, col):
    return df.groupby("timestamp")[col].rank(pct=True) - 0.5


def load_data(symbols):
    from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal
    ohlcv = load_ohlcv()
    ohlcv = ohlcv[ohlcv["symbol"].isin(symbols)]
    derivs = load_derivatives()
    return build_features_minimal(ohlcv, derivs)


def compute_regime(df):
    btc = df[df["symbol"] == "BTC/USDT"][["timestamp", "close"]].copy()
    btc = btc.sort_values("timestamp").drop_duplicates("timestamp")
    btc["btc_ret_7d"] = btc["close"].pct_change(168)
    btc["btc_vol_7d"] = btc["close"].pct_change(1).rolling(168).std()
    btc["trend_strength"] = btc["btc_ret_7d"].abs() / (btc["btc_vol_7d"] * np.sqrt(168) + 1e-10)
    btc["btc_trend_dir"] = np.sign(btc["btc_ret_7d"])
    return btc.set_index("timestamp")


def train_and_predict(df, feats, horizon):
    """Train Ridge models, return predictions per test window."""
    fwd_col = f"fwd_ret_{horizon}h"
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


def simulate_ls(merged, regime_df, horizon, n_long, n_short,
                trend_cutoff=0.8, momentum_overlay=False):
    """L/S simulation with hard trend cutoff + optional momentum overlay."""
    all_rets = []

    for ts, grp in merged.groupby("timestamp"):
        if ts not in regime_df.index:
            continue

        regime = regime_df.loc[ts]
        trend_str = regime.get("trend_strength", 0)
        btc_dir = regime.get("btc_trend_dir", 0)

        # Momentum overlay: in trend, reverse the signal (go WITH trend)
        if momentum_overlay and trend_str > trend_cutoff:
            # Skip — or we could flip MR to momentum
            # For now: skip, same as cutoff
            continue
        elif trend_str > trend_cutoff:
            continue

        grp = grp.copy()
        grp["pred_rank"] = grp["pred"].rank(ascending=False)
        n = len(grp)
        nl = min(n_long, n // 3)
        ns = min(n_short, n // 3)

        if nl == 0 or ns == 0:
            continue

        long_mask = grp["pred_rank"] <= nl
        short_mask = grp["pred_rank"] > (n - ns)

        long_ret = grp.loc[long_mask, "fwd_ret"].mean()
        short_ret = grp.loc[short_mask, "fwd_ret"].mean()

        port_ret = 0.5 * long_ret - 0.5 * short_ret
        all_rets.append({"timestamp": ts, "portfolio_ret": port_ret})

    if not all_rets:
        return None

    port = pd.DataFrame(all_rets).sort_values("timestamp")
    sub = port.iloc[::horizon]
    return sub


def eval_rets(sub, horizon, leverage=3, capital=100):
    """Compute stats from non-overlapping returns."""
    rets = sub["portfolio_ret"]
    if len(rets) < 10:
        return None

    ppy = 8760 / horizon
    sharpe = rets.mean() / (rets.std() + 1e-10) * np.sqrt(ppy)
    cum = (1 + rets).cumprod()
    total = cum.iloc[-1] - 1
    maxdd = (cum / cum.cummax() - 1).min()
    hit = (rets > 0).mean()

    sub_df = sub.copy()
    sub_df["month"] = sub_df["timestamp"].dt.to_period("M")
    monthly = sub_df.groupby("month")["portfolio_ret"].apply(
        lambda x: (1 + x * leverage).prod() - 1)
    worst_m = monthly.min()

    equity = capital
    for ret in monthly:
        equity *= (1 + ret)

    return {
        "sharpe": sharpe,
        "total": total,
        "maxdd": maxdd,
        "hit": hit,
        "n": len(rets),
        "worst_m": worst_m,
        "final_equity": equity,
        "monthly": monthly,
    }


def print_result(name, r):
    if r is None:
        print(f"  {name:<55s}  (no data)")
        return
    print(f"  {name:<55s} Sharpe={r['sharpe']:>+5.2f} Total={r['total']*100:>+6.1f}% "
          f"MaxDD={r['maxdd']*100:>+5.1f}% Hit={r['hit']:.0%} N={r['n']:>4d} "
          f"WrstM={r['worst_m']*100:>+6.1f}% → ${r['final_equity']:.0f}")


def main():
    print("=" * 100)
    print("  RESEARCH ROUND 2: More Improvement Ideas")
    print("=" * 100)

    # ── Load data for both universes ──
    print("\n  Loading 20-symbol universe...")
    df20 = load_data(SYMBOLS_20)
    feats = [f for f in FEATURES if f in df20.columns]
    regime_df = compute_regime(df20)
    print(f"    {df20.shape[0]:,} rows, {df20['symbol'].nunique()} symbols")

    print("  Loading 35-symbol universe...")
    df35 = load_data(SYMBOLS_35)
    n35 = df35['symbol'].nunique()
    print(f"    {df35.shape[0]:,} rows, {n35} symbols")

    # ── 1. Baseline: 20 symbols, 12h, cutoff 0.8 ──
    print("\n  Training 20-sym 12h models...")
    pred_20_12h = train_and_predict(df20, feats, 12)

    print("  Training 20-sym 24h models...")
    pred_20_24h = train_and_predict(df20, feats, 24)

    print("  Training 35-sym 12h models...")
    pred_35_12h = train_and_predict(df35, feats, 12)

    print("  Training 35-sym 24h models...")
    pred_35_24h = train_and_predict(df35, feats, 24)

    print(f"\n{'─' * 100}")
    print(f"  {'Config':<55s} {'Sharpe':>6s} {'Total':>7s} {'MaxDD':>6s} {'Hit':>4s} "
          f"{'N':>5s} {'WrstM':>7s}  {'$100→':>5s}")
    print(f"{'─' * 100}")

    # A. Baseline
    sub = simulate_ls(pred_20_12h, regime_df, 12, 4, 4, trend_cutoff=0.8)
    r_baseline = eval_rets(sub, 12) if sub is not None else None
    print_result("A. Baseline: 20sym, 12h, 4L/4S, cutoff=0.8", r_baseline)

    # B. More positions: 5L/5S
    sub = simulate_ls(pred_20_12h, regime_df, 12, 5, 5, trend_cutoff=0.8)
    r = eval_rets(sub, 12) if sub is not None else None
    print_result("B. 20sym, 12h, 5L/5S", r)

    # C. More positions: 6L/6S
    sub = simulate_ls(pred_20_12h, regime_df, 12, 6, 6, trend_cutoff=0.8)
    r = eval_rets(sub, 12) if sub is not None else None
    print_result("C. 20sym, 12h, 6L/6S", r)

    # D. 35 symbols, 12h, 4L/4S
    sub = simulate_ls(pred_35_12h, regime_df, 12, 4, 4, trend_cutoff=0.8)
    r = eval_rets(sub, 12) if sub is not None else None
    print_result("D. 35sym, 12h, 4L/4S", r)

    # E. 35 symbols, 12h, 6L/6S
    sub = simulate_ls(pred_35_12h, regime_df, 12, 6, 6, trend_cutoff=0.8)
    r = eval_rets(sub, 12) if sub is not None else None
    print_result("E. 35sym, 12h, 6L/6S", r)

    # F. 35 symbols, 12h, 8L/8S
    sub = simulate_ls(pred_35_12h, regime_df, 12, 8, 8, trend_cutoff=0.8)
    r = eval_rets(sub, 12) if sub is not None else None
    print_result("F. 35sym, 12h, 8L/8S", r)

    # G. 20sym, 24h horizon, 4L/4S
    sub = simulate_ls(pred_20_24h, regime_df, 24, 4, 4, trend_cutoff=0.8)
    r = eval_rets(sub, 24) if sub is not None else None
    print_result("G. 20sym, 24h, 4L/4S", r)

    # H. 35sym, 24h, 6L/6S
    sub = simulate_ls(pred_35_24h, regime_df, 24, 6, 6, trend_cutoff=0.8)
    r = eval_rets(sub, 24) if sub is not None else None
    print_result("H. 35sym, 24h, 6L/6S", r)

    # I. Multi-horizon blend: average 12h + 24h pred, trade at 12h
    print(f"\n  Multi-horizon blends:")
    if len(pred_20_12h) > 0 and len(pred_20_24h) > 0:
        blend_20 = pred_20_12h[["timestamp", "symbol", "pred", "fwd_ret"]].copy()
        blend_20 = blend_20.rename(columns={"pred": "pred_12h"})
        p24 = pred_20_24h[["timestamp", "symbol", "pred"]].rename(columns={"pred": "pred_24h"})
        blend_20 = blend_20.merge(p24, on=["timestamp", "symbol"], how="inner")
        blend_20["pred"] = 0.6 * blend_20["pred_12h"] + 0.4 * blend_20["pred_24h"]

        sub = simulate_ls(blend_20, regime_df, 12, 4, 4, trend_cutoff=0.8)
        r = eval_rets(sub, 12) if sub is not None else None
        print_result("I. 20sym, blend 12h+24h (0.6/0.4), 4L/4S", r)

    if len(pred_35_12h) > 0 and len(pred_35_24h) > 0:
        blend_35 = pred_35_12h[["timestamp", "symbol", "pred", "fwd_ret"]].copy()
        blend_35 = blend_35.rename(columns={"pred": "pred_12h"})
        p24 = pred_35_24h[["timestamp", "symbol", "pred"]].rename(columns={"pred": "pred_24h"})
        blend_35 = blend_35.merge(p24, on=["timestamp", "symbol"], how="inner")
        blend_35["pred"] = 0.6 * blend_35["pred_12h"] + 0.4 * blend_35["pred_24h"]

        sub = simulate_ls(blend_35, regime_df, 12, 6, 6, trend_cutoff=0.8)
        r = eval_rets(sub, 12) if sub is not None else None
        print_result("J. 35sym, blend 12h+24h, 6L/6S", r)

        sub = simulate_ls(blend_35, regime_df, 12, 8, 8, trend_cutoff=0.8)
        r = eval_rets(sub, 12) if sub is not None else None
        print_result("K. 35sym, blend 12h+24h, 8L/8S", r)

    # L. Tighter cutoff
    sub = simulate_ls(pred_20_12h, regime_df, 12, 4, 4, trend_cutoff=0.6)
    r = eval_rets(sub, 12) if sub is not None else None
    print_result("L. 20sym, 12h, cutoff=0.6 (tighter)", r)

    # M. Looser cutoff
    sub = simulate_ls(pred_20_12h, regime_df, 12, 4, 4, trend_cutoff=1.0)
    r = eval_rets(sub, 12) if sub is not None else None
    print_result("M. 20sym, 12h, cutoff=1.0 (looser)", r)

    # N. 35sym blend with tighter cutoff
    if len(pred_35_12h) > 0 and len(pred_35_24h) > 0:
        sub = simulate_ls(blend_35, regime_df, 12, 6, 6, trend_cutoff=0.6)
        r = eval_rets(sub, 12) if sub is not None else None
        print_result("N. 35sym, blend, 6L/6S, cutoff=0.6", r)

    # Show monthly breakdown for best configs
    print(f"\n{'=' * 100}")
    print(f"  MONTHLY BREAKDOWN for A (baseline) vs best:")
    print(f"{'=' * 100}")
    if r_baseline:
        print(f"\n  A. Baseline (20sym, 12h, 4L/4S, cutoff=0.8):")
        equity = 100
        for month, ret in r_baseline['monthly'].items():
            pnl = equity * ret
            print(f"    {str(month):>10s}  {ret*100:>+7.1f}%  ${pnl:>+7.1f}  (equity: ${equity+pnl:.0f})")
            equity += pnl


if __name__ == "__main__":
    main()
