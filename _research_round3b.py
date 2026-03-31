#!/usr/bin/env python3
"""
Research round 3B: sophisticated risk management for 5x leverage.
Focus: volatility targeting, dynamic exposure, 35-symbol sweet spot.
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

# 35 symbol sweet spot from round 2
SYM_35 = SYM_20 + [
    "INJ/USDT", "FTM/USDT", "ALGO/USDT", "SAND/USDT", "MANA/USDT",
    "AXS/USDT", "THETA/USDT", "RUNE/USDT", "EGLD/USDT", "XTZ/USDT",
    "FLOW/USDT", "CHZ/USDT", "CRV/USDT", "LDO/USDT", "SNX/USDT",
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
    # Also compute recent portfolio vol proxy (BTC vol over last 48h)
    btc["btc_vol_48h"] = btc["close"].pct_change(1).rolling(48).std()
    btc["vol_regime"] = btc["btc_vol_48h"] / btc["btc_vol_48h"].rolling(720).mean()
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


def simulate_advanced(merged, regime_df, horizon, n_long, n_short,
                      trend_cutoff=0.8, vol_target=False,
                      dynamic_exposure=False, edge_weight=False,
                      asymmetric_regime=False):
    """Advanced L/S simulation with risk management."""
    all_rets = []

    for ts, grp in merged.groupby("timestamp"):
        if ts not in regime_df.index:
            continue
        row = regime_df.loc[ts]
        trend_str = row.get("trend_strength", 0)
        vol_regime = row.get("vol_regime", 1.0)

        # Regime filter: go flat if trend too strong
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

        # Volatility targeting: scale inversely with recent vol
        if vol_target and not np.isnan(vol_regime) and vol_regime > 0:
            # When vol_regime > 1 (vol above average), scale down
            # When vol_regime < 1 (vol below average), scale up (capped at 1.5x)
            scale = min(1.5, 1.0 / max(0.5, vol_regime))
            port_ret *= scale

        # Dynamic exposure: trade less aggressively when trend is medium
        if dynamic_exposure:
            if trend_str > 0.5:
                # Reduce exposure linearly from 0.5 to cutoff
                exposure = 1.0 - (trend_str - 0.5) / (trend_cutoff - 0.5) * 0.5
                port_ret *= exposure

        # Asymmetric regime: long-only when BTC trending up
        if asymmetric_regime:
            btc_ret = row.get("btc_ret_7d", 0)
            if btc_ret > 0 and trend_str > 0.3:
                # Bull trend: go long-biased (70% long, 30% short)
                port_ret = 0.7 * long_ret - 0.3 * short_ret

        all_rets.append({"timestamp": ts, "portfolio_ret": port_ret})

    if not all_rets:
        return None
    port = pd.DataFrame(all_rets).sort_values("timestamp")
    return port.iloc[::horizon]


def eval_config(sub, horizon, name, leverage=5, capital=100):
    if sub is None or len(sub) < 10:
        return None

    rets = sub["portfolio_ret"]
    ppy = 8760 / horizon
    sharpe = rets.mean() / (rets.std() + 1e-10) * np.sqrt(ppy)
    cum = (1 + rets).cumprod()
    total = cum.iloc[-1] - 1
    maxdd = (cum / cum.cummax() - 1).min()

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

    avg_monthly = monthly.mean()
    med_monthly = monthly.median()

    return {
        "name": name, "sharpe": sharpe, "total": total, "maxdd": maxdd,
        "worst_m": worst_m, "equity": equity, "monthly": monthly,
        "avg_monthly": avg_monthly, "med_monthly": med_monthly,
        "month_data": month_data,
    }


def print_config(r, lev, cap):
    print(f"  {r['name']:<65s} Sh={r['sharpe']:>+5.2f} "
          f"Wr={r['worst_m']*100:>+6.1f}% → ${r['equity']:.0f}")


def main():
    LEV = 5
    CAP = 100

    print("=" * 100)
    print(f"  RESEARCH ROUND 3B: Advanced Risk Management ({LEV}x, ${CAP})")
    print("=" * 100)

    print("\n  Loading data...")
    df20 = load_data(SYM_20)
    feats = [f for f in FEATURES if f in df20.columns]
    regime_df = compute_regime(df20)
    n20 = df20['symbol'].nunique()
    print(f"    {n20} symbols: {df20.shape[0]:,} rows")

    df35 = load_data(SYM_35)
    n35 = df35['symbol'].nunique()
    print(f"    {n35} symbols: {df35.shape[0]:,} rows")

    print("\n  Training models...")
    p20_12 = train_and_predict(df20, feats, 12)
    p35_12 = train_and_predict(df35, feats, 12)
    print(f"    20sym 12h: {len(p20_12):,} preds")
    print(f"    {n35}sym 12h: {len(p35_12):,} preds")

    results = []

    print(f"\n{'─' * 100}")
    print(f"  SECTION A: Volatility Targeting (scale by 1/vol_regime)")
    print(f"{'─' * 100}")

    for sym_name, preds in [("20sym", p20_12), (f"{n35}sym", p35_12)]:
        for nl, ns in [(4, 4), (5, 5), (6, 6), (8, 8)]:
            if sym_name == "20sym" and nl > 6:
                continue
            for co in [0.8]:
                sub = simulate_advanced(preds, regime_df, 12, nl, ns,
                                        trend_cutoff=co, vol_target=True)
                name = f"{sym_name} 12h {nl}L/{ns}S co={co} VOL-TARGET"
                r = eval_config(sub, 12, name, LEV, CAP)
                if r:
                    results.append(r)
                    print_config(r, LEV, CAP)

    print(f"\n{'─' * 100}")
    print(f"  SECTION B: Dynamic Exposure (fade when trend > 0.5)")
    print(f"{'─' * 100}")

    for sym_name, preds in [("20sym", p20_12), (f"{n35}sym", p35_12)]:
        for nl, ns in [(4, 4), (5, 5), (6, 6), (8, 8)]:
            if sym_name == "20sym" and nl > 6:
                continue
            sub = simulate_advanced(preds, regime_df, 12, nl, ns,
                                    trend_cutoff=0.8, dynamic_exposure=True)
            name = f"{sym_name} 12h {nl}L/{ns}S DYN-EXPOSURE"
            r = eval_config(sub, 12, name, LEV, CAP)
            if r:
                results.append(r)
                print_config(r, LEV, CAP)

    print(f"\n{'─' * 100}")
    print(f"  SECTION C: Vol-Target + Dynamic Exposure (combo)")
    print(f"{'─' * 100}")

    for sym_name, preds in [("20sym", p20_12), (f"{n35}sym", p35_12)]:
        for nl, ns in [(4, 4), (5, 5), (6, 6), (8, 8)]:
            if sym_name == "20sym" and nl > 6:
                continue
            sub = simulate_advanced(preds, regime_df, 12, nl, ns,
                                    trend_cutoff=0.8,
                                    vol_target=True, dynamic_exposure=True)
            name = f"{sym_name} 12h {nl}L/{ns}S COMBO(vol+dyn)"
            r = eval_config(sub, 12, name, LEV, CAP)
            if r:
                results.append(r)
                print_config(r, LEV, CAP)

    print(f"\n{'─' * 100}")
    print(f"  SECTION D: Asymmetric Regime (long-biased in bull)")
    print(f"{'─' * 100}")

    for sym_name, preds in [("20sym", p20_12), (f"{n35}sym", p35_12)]:
        for nl, ns in [(4, 4), (6, 6)]:
            sub = simulate_advanced(preds, regime_df, 12, nl, ns,
                                    trend_cutoff=0.8, asymmetric_regime=True)
            name = f"{sym_name} 12h {nl}L/{ns}S ASYM-REGIME"
            r = eval_config(sub, 12, name, LEV, CAP)
            if r:
                results.append(r)
                print_config(r, LEV, CAP)

    print(f"\n{'─' * 100}")
    print(f"  SECTION E: COMBO + Edge-Weight + Asymmetric (kitchen sink)")
    print(f"{'─' * 100}")

    for sym_name, preds in [("20sym", p20_12), (f"{n35}sym", p35_12)]:
        for nl, ns in [(4, 4), (5, 5), (6, 6)]:
            sub = simulate_advanced(preds, regime_df, 12, nl, ns,
                                    trend_cutoff=0.8,
                                    vol_target=True, dynamic_exposure=True,
                                    edge_weight=True, asymmetric_regime=True)
            name = f"{sym_name} 12h {nl}L/{ns}S KITCHEN-SINK"
            r = eval_config(sub, 12, name, LEV, CAP)
            if r:
                results.append(r)
                print_config(r, LEV, CAP)

    print(f"\n{'─' * 100}")
    print(f"  SECTION F: Different cutoffs (lower = more conservative)")
    print(f"{'─' * 100}")

    for sym_name, preds in [("20sym", p20_12), (f"{n35}sym", p35_12)]:
        for co in [0.5, 0.6, 0.7, 0.9, 1.0]:
            nl, ns = 4, 4
            sub = simulate_advanced(preds, regime_df, 12, nl, ns, trend_cutoff=co)
            name = f"{sym_name} 12h {nl}L/{ns}S co={co} (plain)"
            r = eval_config(sub, 12, name, LEV, CAP)
            if r:
                results.append(r)
                print_config(r, LEV, CAP)

    # ── RANK ALL BY risk-adjusted metric ──
    # We want: high returns BUT worst_month > -20% at 5x
    # Score = equity * (1 + worst_m) to penalize bad drawdowns
    print(f"\n{'=' * 100}")
    print(f"  🏆 RANKING BY RISK-ADJUSTED EQUITY (equity × safety)")
    print(f"{'=' * 100}")

    for r in results:
        # Penalize configs where worst month at 5x is worse than -20%
        safety = max(0.3, 1.0 + r["worst_m"])  # 1.0 if no drawdown, 0.8 if -20%
        r["score"] = r["equity"] * safety

    results.sort(key=lambda x: x["score"], reverse=True)

    for i, r in enumerate(results[:8]):
        safe = "✅" if r["worst_m"] > -0.15 else ("⚠️" if r["worst_m"] > -0.25 else "❌")
        print(f"\n  #{i+1} {safe} {r['name']}")
        print(f"      Sharpe={r['sharpe']:.2f} | Worst month: {r['worst_m']*100:+.1f}% | "
              f"Avg month: {r['avg_monthly']*100:+.1f}% | Med month: {r['med_monthly']*100:+.1f}%")
        print(f"      ${CAP} → ${r['equity']:.0f} за {len(r['month_data'])} мес | "
              f"Score: {r['score']:.0f}")
        for md in r["month_data"]:
            marker = " ← worst" if md["ret"] == r["worst_m"] else ""
            print(f"         {md['month']:>10s}  {md['ret']*100:>+7.1f}%  "
                  f"equity=${md['equity']:>7.0f}{marker}")

    # ── Also show TOP 3 BY EQUITY (pure greed) ──
    results.sort(key=lambda x: x["equity"], reverse=True)
    print(f"\n{'=' * 100}")
    print(f"  💰 TOP 3 BY RAW EQUITY (maximum greed):")
    print(f"{'=' * 100}")
    for i, r in enumerate(results[:3]):
        print(f"  #{i+1} {r['name']}: ${CAP} → ${r['equity']:.0f} | "
              f"Worst: {r['worst_m']*100:+.1f}% | Sh={r['sharpe']:.2f}")

    # ── And TOP 3 BY WORST MONTH (safest) ──
    results.sort(key=lambda x: x["worst_m"], reverse=True)
    print(f"\n{'=' * 100}")
    print(f"  🛡️ TOP 3 BY SAFETY (smallest worst month):")
    print(f"{'=' * 100}")
    for i, r in enumerate(results[:3]):
        print(f"  #{i+1} {r['name']}: Worst: {r['worst_m']*100:+.1f}% | "
              f"${CAP} → ${r['equity']:.0f} | Sh={r['sharpe']:.2f}")

    # ── PROJECTIONS for best risk-adjusted config ──
    results.sort(key=lambda x: x["score"], reverse=True)
    best = results[0]
    print(f"\n{'=' * 100}")
    print(f"  📊 PROJECTIONS: {best['name']}")
    print(f"{'=' * 100}")
    for label, mret in [("Оптимистичный (avg)", best["avg_monthly"]),
                         ("Реалистичный (median)", best["med_monthly"]),
                         ("Пессимистичный (p25)", best["monthly"].quantile(0.25))]:
        m1 = mret
        m3 = (1 + mret) ** 3 - 1
        m6 = (1 + mret) ** 6 - 1
        y1 = (1 + mret) ** 12 - 1
        print(f"\n  {label} ({mret*100:+.1f}%/мес):")
        print(f"    1 месяц:    ${CAP * m1:>+8.1f}  → ${CAP * (1+m1):.0f}")
        print(f"    3 месяца:   ${CAP * m3:>+8.1f}  → ${CAP * (1+m3):.0f}")
        print(f"    6 месяцев:  ${CAP * m6:>+8.1f}  → ${CAP * (1+m6):.0f}")
        print(f"    12 месяцев: ${CAP * y1:>+8.1f}  → ${CAP * (1+y1):.0f}")


if __name__ == "__main__":
    main()
