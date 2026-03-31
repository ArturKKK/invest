#!/usr/bin/env python3
"""
Research round 3: push for maximum returns.
Test: all 50 symbols, edge-weighted positions, multi-horizon blends,
dynamic cutoffs. Report at 5x leverage.
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

SYM_20 = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
    "MATIC/USDT", "UNI/USDT", "ATOM/USDT", "LTC/USDT", "NEAR/USDT",
    "FIL/USDT", "APT/USDT", "ARB/USDT", "OP/USDT", "AAVE/USDT",
]

SYM_ALL = [
    "AAVE/USDT", "ADA/USDT", "ALGO/USDT", "APT/USDT", "ARB/USDT",
    "ATOM/USDT", "AVAX/USDT", "AXS/USDT", "BAT/USDT", "BNB/USDT",
    "BTC/USDT", "CHZ/USDT", "COMP/USDT", "CRV/USDT", "DOGE/USDT",
    "DOT/USDT", "EGLD/USDT", "ENJ/USDT", "ENS/USDT", "ETC/USDT",
    "ETH/USDT", "FIL/USDT", "FLOW/USDT", "FTM/USDT", "GALA/USDT",
    "GRT/USDT", "ICX/USDT", "IMX/USDT", "INJ/USDT", "IOTA/USDT",
    "LDO/USDT", "LINK/USDT", "LTC/USDT", "MANA/USDT", "MATIC/USDT",
    "MKR/USDT", "NEAR/USDT", "ONE/USDT", "OP/USDT", "RUNE/USDT",
    "SAND/USDT", "SNX/USDT", "SOL/USDT", "SUSHI/USDT", "THETA/USDT",
    "UNI/USDT", "XRP/USDT", "XTZ/USDT", "YFI/USDT", "ZIL/USDT",
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
    return btc.set_index("timestamp")


def train_and_predict(df, feats, horizon):
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


def simulate(merged, regime_df, horizon, n_long, n_short,
             trend_cutoff=0.8, edge_weight=False):
    """L/S simulation."""
    all_rets = []

    for ts, grp in merged.groupby("timestamp"):
        if ts not in regime_df.index:
            continue
        trend_str = regime_df.loc[ts].get("trend_strength", 0)
        if trend_str > trend_cutoff:
            continue

        grp = grp.copy()
        n = len(grp)
        nl = min(n_long, n // 3)
        ns = min(n_short, n // 3)
        if nl == 0 or ns == 0:
            continue

        grp["pred_rank"] = grp["pred"].rank(ascending=False)
        longs = grp[grp["pred_rank"] <= nl]
        shorts = grp[grp["pred_rank"] > (n - ns)]

        if edge_weight:
            # Weight proportional to |pred - median|
            med = grp["pred"].median()
            lw = (longs["pred"] - med).abs()
            lw = lw / (lw.sum() + 1e-10)
            sw = (shorts["pred"] - med).abs()
            sw = sw / (sw.sum() + 1e-10)
            long_ret = (longs["fwd_ret"] * lw).sum()
            short_ret = (shorts["fwd_ret"] * sw).sum()
        else:
            long_ret = longs["fwd_ret"].mean()
            short_ret = shorts["fwd_ret"].mean()

        port_ret = 0.5 * long_ret - 0.5 * short_ret
        all_rets.append({"timestamp": ts, "portfolio_ret": port_ret})

    if not all_rets:
        return None
    port = pd.DataFrame(all_rets).sort_values("timestamp")
    return port.iloc[::horizon]


def eval_and_show(sub, horizon, name, leverage=5, capital=100, show_monthly=False):
    if sub is None or len(sub) < 10:
        print(f"  {name:<60s}  (no data)")
        return None

    rets = sub["portfolio_ret"]
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
    best_m = monthly.max()

    equity = capital
    month_data = []
    for month, ret in monthly.items():
        pnl = equity * ret
        month_data.append({"month": str(month), "ret": ret, "pnl": pnl, "equity": equity + pnl})
        equity += pnl

    # 6-month return (last 6 months, or all if shorter)
    recent_6m = monthly.tail(6)
    ret_6m = 1.0
    for r in recent_6m:
        ret_6m *= (1 + r)
    ret_6m -= 1

    avg_monthly = monthly.mean()

    print(f"  {name:<60s} Sh={sharpe:>+5.2f} Tot={total*100:>+5.1f}% "
          f"DD={maxdd*100:>+5.1f}% Wr={worst_m*100:>+6.1f}% → ${equity:.0f}")

    if show_monthly:
        print(f"    {'Месяц':>10s}  {'Return':>8s}  {'P&L':>8s}  {'Equity':>8s}")
        for md in month_data:
            print(f"    {md['month']:>10s}  {md['ret']*100:>+7.1f}%  "
                  f"${md['pnl']:>+7.1f}  ${md['equity']:>7.0f}")
        print(f"    Avg monthly: {avg_monthly*100:+.1f}% | "
              f"Last 6m: {ret_6m*100:+.1f}%")

    return {
        "name": name, "sharpe": sharpe, "total": total, "maxdd": maxdd,
        "worst_m": worst_m, "equity": equity, "monthly": monthly,
        "avg_monthly": avg_monthly, "ret_6m": ret_6m,
        "month_data": month_data,
    }


def main():
    LEV = 5
    CAP = 100

    print("=" * 100)
    print(f"  RESEARCH ROUND 3: Push for Maximum Returns ({LEV}x leverage, ${CAP})")
    print("=" * 100)

    # Load all data sets
    print("\n  Loading data...")
    df20 = load_data(SYM_20)
    feats = [f for f in FEATURES if f in df20.columns]
    regime_df = compute_regime(df20)
    print(f"    20 symbols: {df20.shape[0]:,} rows")

    df50 = load_data(SYM_ALL)
    n50 = df50['symbol'].nunique()
    print(f"    {n50} symbols: {df50.shape[0]:,} rows")

    # Train models
    print("\n  Training models...")
    p20_12 = train_and_predict(df20, feats, 12)
    p20_24 = train_and_predict(df20, feats, 24)
    p50_12 = train_and_predict(df50, feats, 12)
    p50_24 = train_and_predict(df50, feats, 24)
    print(f"    20sym: 12h={len(p20_12):,}, 24h={len(p20_24):,}")
    print(f"    {n50}sym: 12h={len(p50_12):,}, 24h={len(p50_24):,}")

    # Make blends
    def make_blend(p12, p24, w12=0.6):
        b = p12[["timestamp", "symbol", "pred", "fwd_ret"]].copy()
        b = b.rename(columns={"pred": "pred_12h"})
        q = p24[["timestamp", "symbol", "pred"]].rename(columns={"pred": "pred_24h"})
        b = b.merge(q, on=["timestamp", "symbol"], how="inner")
        b["pred"] = w12 * b["pred_12h"] + (1 - w12) * b["pred_24h"]
        return b

    blend20 = make_blend(p20_12, p20_24) if len(p20_12) > 0 and len(p20_24) > 0 else None
    blend50 = make_blend(p50_12, p50_24) if len(p50_12) > 0 and len(p50_24) > 0 else None

    # ── Run all configs ────────────────────────────────────
    print(f"\n{'─' * 100}")
    print(f"  All configs at {LEV}x leverage, ${CAP} start:")
    print(f"{'─' * 100}")

    results = []

    # 20 sym baselines
    for nl, ns in [(4, 4), (5, 5), (6, 6)]:
        for co in [0.8, 0.6]:
            sub = simulate(p20_12, regime_df, 12, nl, ns, trend_cutoff=co)
            r = eval_and_show(sub, 12, f"20sym 12h {nl}L/{ns}S co={co}", LEV, CAP)
            if r: results.append(r)

    # 50 sym variants
    for nl, ns in [(4, 4), (6, 6), (8, 8), (10, 10)]:
        for co in [0.8, 0.6]:
            sub = simulate(p50_12, regime_df, 12, nl, ns, trend_cutoff=co)
            r = eval_and_show(sub, 12, f"{n50}sym 12h {nl}L/{ns}S co={co}", LEV, CAP)
            if r: results.append(r)

    # Edge-weighted
    for nl, ns in [(6, 6), (8, 8)]:
        sub = simulate(p50_12, regime_df, 12, nl, ns, trend_cutoff=0.8, edge_weight=True)
        r = eval_and_show(sub, 12, f"{n50}sym 12h {nl}L/{ns}S co=0.8 EDGE-W", LEV, CAP)
        if r: results.append(r)

    # 24h horizon
    for nl, ns in [(4, 4), (6, 6)]:
        sub = simulate(p50_24, regime_df, 24, nl, ns, trend_cutoff=0.8)
        r = eval_and_show(sub, 24, f"{n50}sym 24h {nl}L/{ns}S co=0.8", LEV, CAP)
        if r: results.append(r)

    # Blends
    if blend50 is not None:
        for nl, ns in [(6, 6), (8, 8), (10, 10)]:
            for co in [0.8, 0.6]:
                sub = simulate(blend50, regime_df, 12, nl, ns, trend_cutoff=co)
                r = eval_and_show(sub, 12, f"{n50}sym BLEND 12+24h {nl}L/{ns}S co={co}", LEV, CAP)
                if r: results.append(r)

        # Edge-weighted blend
        sub = simulate(blend50, regime_df, 12, 8, 8, trend_cutoff=0.8, edge_weight=True)
        r = eval_and_show(sub, 12, f"{n50}sym BLEND 8L/8S co=0.8 EDGE-W", LEV, CAP)
        if r: results.append(r)

    if blend20 is not None:
        for nl, ns in [(4, 4), (5, 5)]:
            sub = simulate(blend20, regime_df, 12, nl, ns, trend_cutoff=0.8)
            r = eval_and_show(sub, 12, f"20sym BLEND 12+24h {nl}L/{ns}S co=0.8", LEV, CAP)
            if r: results.append(r)

    # ── TOP 5 ────────────────────────────────────────────
    if not results:
        print("\n  ❌ No results")
        return

    # Sort by final equity
    results.sort(key=lambda x: x["equity"], reverse=True)

    print(f"\n{'=' * 100}")
    print(f"  🏆 TOP 5 BY FINAL EQUITY ({LEV}x leverage, ${CAP} → ?):")
    print(f"{'=' * 100}")

    for i, r in enumerate(results[:5]):
        print(f"\n  #{i+1}: {r['name']}")
        print(f"      Sharpe={r['sharpe']:.2f} | Total(1x)={r['total']*100:+.1f}% | "
              f"MaxDD(1x)={r['maxdd']*100:.1f}%")
        print(f"      Worst month ({LEV}x): {r['worst_m']*100:+.1f}% | "
              f"Avg month: {r['avg_monthly']*100:+.1f}% | "
              f"Last 6m: {r['ret_6m']*100:+.1f}%")
        print(f"      ${CAP} → ${r['equity']:.0f} за 17 мес")
        print(f"      📅 Помесячно:")
        for md in r["month_data"]:
            marker = ""
            if md["ret"] == r["worst_m"]:
                marker = " ← worst"
            elif md["ret"] == r["monthly"].max():
                marker = " ← best"
            print(f"         {md['month']:>10s}  {md['ret']*100:>+7.1f}%  "
                  f"${md['pnl']:>+8.1f}  equity=${md['equity']:>7.0f}{marker}")

    # ── PROJECTIONS ────────────────────────────────────────
    best = results[0]
    print(f"\n{'=' * 100}")
    print(f"  💰 PROJECTIONS для лучшего варианта: {best['name']}")
    print(f"{'=' * 100}")
    avg_m = best["avg_monthly"]
    # Conservative: median monthly
    med_m = best["monthly"].median()
    # Pessimistic: 25th percentile
    p25_m = best["monthly"].quantile(0.25)

    for label, mret in [("Оптимистичный (avg)", avg_m),
                         ("Реалистичный (median)", med_m),
                         ("Пессимистичный (p25)", p25_m)]:
        w1 = (1 + mret) ** (7/30) - 1
        m1 = mret
        m3 = (1 + mret) ** 3 - 1
        m6 = (1 + mret) ** 6 - 1
        y1 = (1 + mret) ** 12 - 1
        print(f"\n  {label} ({mret*100:+.1f}%/мес):")
        print(f"    1 неделя:   ${CAP * w1:>+8.1f}  (equity: ${CAP * (1+w1):.0f})")
        print(f"    1 месяц:    ${CAP * m1:>+8.1f}  (equity: ${CAP * (1+m1):.0f})")
        print(f"    3 месяца:   ${CAP * m3:>+8.1f}  (equity: ${CAP * (1+m3):.0f})")
        print(f"    6 месяцев:  ${CAP * m6:>+8.1f}  (equity: ${CAP * (1+m6):.0f})")
        print(f"    12 месяцев: ${CAP * y1:>+8.1f}  (equity: ${CAP * (1+y1):.0f})")

    print(f"\n  ⚠️  Worst month: {best['worst_m']*100:+.1f}% = ${CAP * best['worst_m']:+.0f}")
    print(f"  ⚠️  MaxDD (1x): {best['maxdd']*100:.1f}%")


if __name__ == "__main__":
    main()
