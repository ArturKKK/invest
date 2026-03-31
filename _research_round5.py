#!/usr/bin/env python3
"""
Research round 5: deeper alpha hunt.
Ideas tested:
  A) Enhanced EQ-MOM: scale UP after recovery, not just down in DD
  B) Kelly-inspired sizing: adjust L/S split by predicted spread
  C) Signal agreement: two independent Ridge models (12h, 24h), trade when aligned
  D) Adaptive trend cutoff (rolling percentile instead of fixed 0.8)
  E) Sector-neutral: L1/DeFi/Gaming buckets → pick top/bottom WITHIN each sector
  F) Time-of-day filter: skip certain hours
  G) Drawdown floor: hard stop at -X% monthly DD
  H) Mega-combos of best ideas
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

SYM_35 = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
    "MATIC/USDT", "UNI/USDT", "ATOM/USDT", "LTC/USDT", "NEAR/USDT",
    "FIL/USDT", "APT/USDT", "ARB/USDT", "OP/USDT", "AAVE/USDT",
    "INJ/USDT", "FTM/USDT", "ALGO/USDT", "SAND/USDT", "MANA/USDT",
    "AXS/USDT", "THETA/USDT", "RUNE/USDT", "EGLD/USDT", "XTZ/USDT",
    "FLOW/USDT", "CHZ/USDT", "CRV/USDT", "LDO/USDT", "SNX/USDT",
]

SYM_20 = SYM_35[:20]

# Sector groupings for sector-neutral strategy
SECTORS = {
    "L1": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT",
           "AVAX/USDT", "DOT/USDT", "NEAR/USDT", "ATOM/USDT", "ALGO/USDT",
           "FTM/USDT", "EGLD/USDT", "XTZ/USDT", "FLOW/USDT"],
    "DeFi": ["UNI/USDT", "AAVE/USDT", "CRV/USDT", "LDO/USDT", "SNX/USDT",
             "INJ/USDT", "LINK/USDT", "RUNE/USDT"],
    "Gaming": ["AXS/USDT", "SAND/USDT", "MANA/USDT", "CHZ/USDT",
               "GALA/USDT", "ENJ/USDT", "THETA/USDT"],
    "Infra": ["FIL/USDT", "ARB/USDT", "OP/USDT", "APT/USDT", "LTC/USDT",
              "XRP/USDT", "DOGE/USDT", "MATIC/USDT"],
}

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
    # Rolling percentile of trend_strength (for adaptive cutoff)
    btc["trend_pctl_30d"] = btc["trend_strength"].rolling(720).rank(pct=True)
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


def simulate(merged, regime_df, horizon, cfg, merged_24h=None):
    """Unified simulation engine."""
    n_long = cfg.get("n_long", 5)
    n_short = cfg.get("n_short", 5)
    trend_cutoff = cfg.get("trend_cutoff", 0.8)
    dyn_threshold = cfg.get("dyn_threshold", 0.5)
    eq_mom = cfg.get("eq_mom", False)
    eq_mom_boost = cfg.get("eq_mom_boost", False)  # Enhanced: scale UP after recovery
    confidence_weight = cfg.get("confidence_weight", False)
    adaptive_cutoff = cfg.get("adaptive_cutoff", False)  # Use rolling percentile
    adaptive_pctl = cfg.get("adaptive_pctl", 0.8)  # percentile threshold
    signal_agreement = cfg.get("signal_agreement", False)  # Require 12h+24h agree
    sector_neutral = cfg.get("sector_neutral", False)
    dd_floor = cfg.get("dd_floor", None)  # Monthly DD hard stop (e.g., -0.15)
    kelly_sizing = cfg.get("kelly_sizing", False)

    all_rets = []
    equity_curve = [1.0]
    month_equity_start = {}  # Track monthly starting equity for dd_floor

    timestamps_sorted = sorted(merged["timestamp"].unique())
    grouped = {ts: grp for ts, grp in merged.groupby("timestamp")}
    grouped_24h = {}
    if merged_24h is not None and len(merged_24h) > 0:
        grouped_24h = {ts: grp for ts, grp in merged_24h.groupby("timestamp")}

    for ts in timestamps_sorted:
        if ts not in regime_df.index or ts not in grouped:
            continue
        row = regime_df.loc[ts]
        trend_str = row.get("trend_strength", 0)

        # Adaptive cutoff: use rolling percentile instead of fixed
        if adaptive_cutoff:
            trend_pctl = row.get("trend_pctl_30d", 0.5)
            if trend_pctl > adaptive_pctl:
                continue
        else:
            if trend_str > trend_cutoff:
                continue

        # Monthly DD floor
        if dd_floor is not None:
            month_key = ts.to_period("M")
            if month_key not in month_equity_start:
                month_equity_start[month_key] = equity_curve[-1]
            month_start = month_equity_start[month_key]
            current_eq = equity_curve[-1]
            month_dd = (current_eq - month_start) / (month_start + 1e-10)
            if month_dd < dd_floor:
                continue  # Stop trading this month

        grp = grouped[ts].copy()
        n = len(grp)

        # Signal agreement: only trade if 12h and 24h models agree on direction
        if signal_agreement and ts in grouped_24h:
            grp24 = grouped_24h[ts]
            grp = grp.merge(grp24[["symbol", "pred"]].rename(columns={"pred": "pred_24h"}),
                            on="symbol", how="inner")
            # Only keep symbols where both models agree on direction
            grp["agree"] = np.sign(grp["pred"]) == np.sign(grp["pred_24h"])
            grp = grp[grp["agree"]].copy()
            n = len(grp)

        # Sector-neutral: pick top/bottom within each sector
        if sector_neutral:
            sym_sector = {}
            for sec, syms in SECTORS.items():
                for s in syms:
                    sym_sector[s] = sec
            grp["sector"] = grp["symbol"].map(sym_sector)
            grp = grp.dropna(subset=["sector"])

            longs_list = []
            shorts_list = []
            for sec, sec_grp in grp.groupby("sector"):
                if len(sec_grp) < 4:
                    continue
                sec_grp = sec_grp.copy()
                nper = max(1, min(2, len(sec_grp) // 3))
                sec_grp["sec_rank"] = sec_grp["pred"].rank(ascending=False)
                longs_list.append(sec_grp[sec_grp["sec_rank"] <= nper])
                shorts_list.append(sec_grp[sec_grp["sec_rank"] > (len(sec_grp) - nper)])
            if not longs_list or not shorts_list:
                continue
            longs = pd.concat(longs_list)
            shorts = pd.concat(shorts_list)
        else:
            nl = min(n_long, n // 3)
            ns = min(n_short, n // 3)
            if nl == 0 or ns == 0:
                continue

            grp["pred_rank"] = grp["pred"].rank(ascending=False)
            longs = grp[grp["pred_rank"] <= nl]
            shorts = grp[grp["pred_rank"] > (n - ns)]

        if confidence_weight:
            lw = longs["pred"].abs()
            lw = lw / (lw.sum() + 1e-10)
            sw = shorts["pred"].abs()
            sw = sw / (sw.sum() + 1e-10)
            long_ret = (longs["fwd_ret"] * lw).sum()
            short_ret = (shorts["fwd_ret"] * sw).sum()
        else:
            long_ret = longs["fwd_ret"].mean()
            short_ret = shorts["fwd_ret"].mean()

        # Kelly-inspired sizing: adjust L/S ratio by predicted spread
        if kelly_sizing:
            pred_spread = longs["pred"].mean() - shorts["pred"].mean()
            # Scale between 30% and 70% long allocation based on spread
            long_alloc = np.clip(0.5 + pred_spread * 5, 0.3, 0.7)
            port_ret = long_alloc * long_ret - (1 - long_alloc) * short_ret
        else:
            port_ret = 0.5 * long_ret - 0.5 * short_ret

        # Dynamic exposure
        if dyn_threshold is not None and not adaptive_cutoff:
            if trend_str > dyn_threshold:
                exposure = 1.0 - (trend_str - dyn_threshold) / (trend_cutoff - dyn_threshold + 1e-10) * 0.5
                port_ret *= max(0.1, exposure)

        # Enhanced equity momentum
        if (eq_mom or eq_mom_boost) and len(equity_curve) > 48:
            recent_eq = equity_curve[-1]
            peak_eq = max(equity_curve[-48:])
            dd = (recent_eq - peak_eq) / (peak_eq + 1e-10)
            if dd < -0.05:
                scale = max(0.3, 1.0 + dd * 3)
                port_ret *= scale
            elif eq_mom_boost and dd > -0.01:
                # Recovery boost: scale up when near/at new highs
                trough_eq = min(equity_curve[-48:])
                recovery = (recent_eq - trough_eq) / (trough_eq + 1e-10)
                if recovery > 0.05:
                    boost = min(1.5, 1.0 + recovery * 0.5)
                    port_ret *= boost

        all_rets.append({"timestamp": ts, "portfolio_ret": port_ret})
        equity_curve.append(equity_curve[-1] * (1 + port_ret))

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

    equity = capital
    month_data = []
    for month, ret in monthly.items():
        pnl = equity * ret
        month_data.append({"month": str(month), "ret": ret, "pnl": pnl, "equity": equity + pnl})
        equity += pnl

    avg_monthly = monthly.mean()
    med_monthly = monthly.median()
    calmar = avg_monthly / (abs(worst_m) + 1e-10)
    # Sortino-like: downside deviation
    neg = monthly[monthly < 0]
    downside_std = neg.std() if len(neg) > 1 else 0.01
    sortino = avg_monthly / (downside_std + 1e-10)

    return {
        "name": name, "sharpe": sharpe, "total": total, "maxdd": maxdd,
        "worst_m": worst_m, "equity": equity, "monthly": monthly,
        "avg_monthly": avg_monthly, "med_monthly": med_monthly,
        "calmar": calmar, "sortino": sortino, "month_data": month_data,
    }


def show(r):
    if r is None:
        return
    safe = "✅" if r["worst_m"] > -0.15 else ("⚠️" if r["worst_m"] > -0.25 else "❌")
    print(f"  {safe} {r['name']:<62s} Sh={r['sharpe']:>+5.2f} "
          f"Wr={r['worst_m']*100:>+6.1f}% → ${r['equity']:.0f}")


def main():
    LEV = 5
    CAP = 100

    print("=" * 100)
    print(f"  RESEARCH ROUND 5: Deeper Alpha Hunt ({LEV}x, ${CAP})")
    print("=" * 100)

    print("\n  Loading data...")
    df35 = load_data(SYM_35)
    df20 = load_data(SYM_20)
    feats = [f for f in FEATURES if f in df35.columns]
    regime_df = compute_regime(df35)
    n35 = df35["symbol"].nunique()
    print(f"    {n35} sym loaded")

    print("\n  Training models...")
    p35_12 = train_and_predict(df35, feats, 12)
    p35_24 = train_and_predict(df35, feats, 24)
    p20_12 = train_and_predict(df20, feats, 12)
    p20_24 = train_and_predict(df20, feats, 24)
    print(f"    {n35}sym: 12h={len(p35_12):,} 24h={len(p35_24):,}")
    print(f"    20sym: 12h={len(p20_12):,} 24h={len(p20_24):,}")

    results = []

    # ── BASELINE ──
    print(f"\n{'─' * 100}")
    print(f"  BASELINE: R4 winner (35sym 5L/5S dyn+EQ-MOM)")
    print(f"{'─' * 100}")
    cfg_base = {"n_long": 5, "n_short": 5, "trend_cutoff": 0.8,
                "dyn_threshold": 0.5, "eq_mom": True}
    sub = simulate(p35_12, regime_df, 12, cfg_base)
    r = eval_config(sub, 12, "BASELINE: 35sym 5L/5S dyn+EQ-MOM", LEV, CAP)
    if r:
        results.append(r)
        show(r)

    # ── A: Enhanced EQ-MOM (boost on recovery) ──
    print(f"\n{'─' * 100}")
    print(f"  A: Enhanced EQ-MOM (scale UP after recovery)")
    print(f"{'─' * 100}")

    for label, preds, syms in [(f"{n35}sym", p35_12, SYM_35), ("20sym", p20_12, SYM_20)]:
        for nl, ns in [(4, 4), (5, 5), (6, 6)]:
            cfg = {"n_long": nl, "n_short": ns, "trend_cutoff": 0.8,
                   "dyn_threshold": 0.5, "eq_mom_boost": True}
            sub = simulate(preds, regime_df, 12, cfg)
            name = f"{label} {nl}L/{ns}S dyn+EQ-BOOST"
            r = eval_config(sub, 12, name, LEV, CAP)
            if r:
                results.append(r)
                show(r)

    # ── B: Kelly-inspired sizing ──
    print(f"\n{'─' * 100}")
    print(f"  B: Kelly-inspired L/S sizing")
    print(f"{'─' * 100}")

    for label, preds in [(f"{n35}sym", p35_12), ("20sym", p20_12)]:
        for nl, ns in [(5, 5), (6, 6)]:
            cfg = {"n_long": nl, "n_short": ns, "trend_cutoff": 0.8,
                   "dyn_threshold": 0.5, "eq_mom": True, "kelly_sizing": True}
            sub = simulate(preds, regime_df, 12, cfg)
            name = f"{label} {nl}L/{ns}S dyn+EQ+KELLY"
            r = eval_config(sub, 12, name, LEV, CAP)
            if r:
                results.append(r)
                show(r)

    # ── C: Signal agreement (12h + 24h) ──
    print(f"\n{'─' * 100}")
    print(f"  C: Signal agreement (trade only when 12h & 24h agree)")
    print(f"{'─' * 100}")

    for label, p12, p24 in [(f"{n35}sym", p35_12, p35_24), ("20sym", p20_12, p20_24)]:
        for nl, ns in [(5, 5), (6, 6)]:
            cfg = {"n_long": nl, "n_short": ns, "trend_cutoff": 0.8,
                   "dyn_threshold": 0.5, "eq_mom": True, "signal_agreement": True}
            sub = simulate(p12, regime_df, 12, cfg, merged_24h=p24)
            name = f"{label} {nl}L/{ns}S dyn+EQ+AGREE"
            r = eval_config(sub, 12, name, LEV, CAP)
            if r:
                results.append(r)
                show(r)

    # ── D: Adaptive cutoff (rolling percentile) ──
    print(f"\n{'─' * 100}")
    print(f"  D: Adaptive trend cutoff (rolling percentile)")
    print(f"{'─' * 100}")

    for label, preds in [(f"{n35}sym", p35_12), ("20sym", p20_12)]:
        for nl, ns in [(5, 5), (6, 6)]:
            for pctl in [0.70, 0.75, 0.80, 0.85]:
                cfg = {"n_long": nl, "n_short": ns,
                       "adaptive_cutoff": True, "adaptive_pctl": pctl,
                       "eq_mom": True}
                sub = simulate(preds, regime_df, 12, cfg)
                name = f"{label} {nl}L/{ns}S adapt-p{int(pctl*100)}+EQ"
                r = eval_config(sub, 12, name, LEV, CAP)
                if r:
                    results.append(r)
                    show(r)

    # ── E: Sector-neutral ──
    print(f"\n{'─' * 100}")
    print(f"  E: Sector-neutral L/S")
    print(f"{'─' * 100}")

    for label, preds in [(f"{n35}sym", p35_12)]:
        cfg = {"sector_neutral": True, "trend_cutoff": 0.8,
               "dyn_threshold": 0.5, "eq_mom": True}
        sub = simulate(preds, regime_df, 12, cfg)
        name = f"{label} SECTOR-NEUTRAL dyn+EQ"
        r = eval_config(sub, 12, name, LEV, CAP)
        if r:
            results.append(r)
            show(r)

        cfg2 = {"sector_neutral": True, "trend_cutoff": 0.8,
                "dyn_threshold": 0.5, "eq_mom_boost": True}
        sub = simulate(preds, regime_df, 12, cfg2)
        name = f"{label} SECTOR-NEUTRAL dyn+EQ-BOOST"
        r = eval_config(sub, 12, name, LEV, CAP)
        if r:
            results.append(r)
            show(r)

    # ── F: Monthly DD floor ──
    print(f"\n{'─' * 100}")
    print(f"  F: Monthly DD floor (stop trading if month DD > X%)")
    print(f"{'─' * 100}")

    for label, preds in [(f"{n35}sym", p35_12), ("20sym", p20_12)]:
        for floor in [-0.10, -0.15, -0.20]:
            cfg = {"n_long": 5, "n_short": 5, "trend_cutoff": 0.8,
                   "dyn_threshold": 0.5, "eq_mom": True, "dd_floor": floor}
            sub = simulate(preds, regime_df, 12, cfg)
            name = f"{label} 5L/5S dyn+EQ+FLOOR({int(floor*100)}%)"
            r = eval_config(sub, 12, name, LEV, CAP)
            if r:
                results.append(r)
                show(r)

    # ── G: MEGA COMBOS ──
    print(f"\n{'─' * 100}")
    print(f"  G: Mega combos")
    print(f"{'─' * 100}")

    combos = [
        {"n_long": 5, "n_short": 5, "trend_cutoff": 0.8, "dyn_threshold": 0.5,
         "eq_mom_boost": True, "kelly_sizing": True,
         "label": "EQ-BOOST+KELLY"},
        {"n_long": 5, "n_short": 5, "trend_cutoff": 0.8, "dyn_threshold": 0.5,
         "eq_mom_boost": True, "dd_floor": -0.15,
         "label": "EQ-BOOST+FLOOR15"},
        {"n_long": 5, "n_short": 5, "trend_cutoff": 0.8, "dyn_threshold": 0.5,
         "eq_mom_boost": True, "confidence_weight": True,
         "label": "EQ-BOOST+CONF-W"},
        {"n_long": 5, "n_short": 5, "trend_cutoff": 0.8, "dyn_threshold": 0.5,
         "eq_mom_boost": True, "kelly_sizing": True, "dd_floor": -0.15,
         "label": "EQ-BOOST+KELLY+FLOOR15"},
        {"n_long": 5, "n_short": 5, "adaptive_cutoff": True, "adaptive_pctl": 0.80,
         "eq_mom_boost": True,
         "label": "ADAPT-p80+EQ-BOOST"},
        {"n_long": 5, "n_short": 5, "adaptive_cutoff": True, "adaptive_pctl": 0.75,
         "eq_mom_boost": True, "dd_floor": -0.15,
         "label": "ADAPT-p75+EQ-BOOST+FLOOR15"},
        {"n_long": 6, "n_short": 6, "trend_cutoff": 0.8, "dyn_threshold": 0.5,
         "eq_mom_boost": True, "kelly_sizing": True,
         "label": "6L6S EQ-BOOST+KELLY"},
        {"n_long": 5, "n_short": 5, "trend_cutoff": 0.8, "dyn_threshold": 0.5,
         "eq_mom_boost": True, "signal_agreement": True,
         "label": "EQ-BOOST+AGREE"},
        {"n_long": 5, "n_short": 5, "trend_cutoff": 0.8, "dyn_threshold": 0.5,
         "eq_mom_boost": True, "confidence_weight": True, "dd_floor": -0.15,
         "label": "EQ-BOOST+CONF-W+FLOOR15"},
    ]

    for label, p12, p24 in [(f"{n35}sym", p35_12, p35_24), ("20sym", p20_12, p20_24)]:
        for combo in combos:
            cfg = {k: v for k, v in combo.items() if k != "label"}
            need_24h = cfg.get("signal_agreement", False)
            sub = simulate(p12, regime_df, 12, cfg, merged_24h=p24 if need_24h else None)
            name = f"{label} {combo['label']}"
            r = eval_config(sub, 12, name, LEV, CAP)
            if r:
                results.append(r)
                show(r)

    # ── GRAND RANKING ──────────────────────────────────────
    if not results:
        print("  No results")
        return

    for r in results:
        safety = max(0.3, 1.0 + r["worst_m"])
        r["score"] = r["equity"] * safety * (max(0.01, r["calmar"]) ** 0.3)

    results.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n{'=' * 100}")
    print(f"  🏆 TOP 10 RISK-ADJUSTED ({LEV}x, ${CAP})")
    print(f"{'=' * 100}")
    for i, r in enumerate(results[:10]):
        safe = "✅" if r["worst_m"] > -0.15 else ("⚠️" if r["worst_m"] > -0.25 else "❌")
        print(f"\n  #{i+1} {safe} {r['name']}")
        print(f"      Sh={r['sharpe']:.2f} | Wr={r['worst_m']*100:+.1f}% | "
              f"Avg/m={r['avg_monthly']*100:+.1f}% | Med/m={r['med_monthly']*100:+.1f}% | "
              f"Sortino={r['sortino']:.2f}")
        print(f"      ${CAP} → ${r['equity']:.0f} ({len(r['month_data'])} мес) | "
              f"Score={r['score']:.0f}")
        for md in r["month_data"]:
            marker = " ← worst" if md["ret"] == r["worst_m"] else ""
            print(f"         {md['month']:>10s}  {md['ret']*100:>+7.1f}%  "
                  f"equity=${md['equity']:>7.0f}{marker}")

    # ── TOP 5 BY EQUITY ──
    results.sort(key=lambda x: x["equity"], reverse=True)
    print(f"\n{'=' * 100}")
    print(f"  💰 TOP 5 RAW EQUITY:")
    print(f"{'=' * 100}")
    for i, r in enumerate(results[:5]):
        safe = "✅" if r["worst_m"] > -0.15 else ("⚠️" if r["worst_m"] > -0.25 else "❌")
        print(f"  #{i+1} {safe} {r['name']}: ${CAP}→${r['equity']:.0f} | "
              f"Wr={r['worst_m']*100:+.1f}% | Sh={r['sharpe']:.2f}")

    # ── SAFE picks (worst > -15%) ──
    safe_results = [r for r in results if r["worst_m"] > -0.15]
    safe_results.sort(key=lambda x: x["equity"], reverse=True)
    print(f"\n{'=' * 100}")
    print(f"  🛡️ TOP 5 SAFE (worst > -15%):")
    print(f"{'=' * 100}")
    for i, r in enumerate(safe_results[:5]):
        print(f"  #{i+1} {r['name']}: ${CAP}→${r['equity']:.0f} | "
              f"Wr={r['worst_m']*100:+.1f}% | Sh={r['sharpe']:.2f}")

    # ── PROJECTIONS ──
    results.sort(key=lambda x: x["score"], reverse=True)
    best = results[0]
    print(f"\n{'=' * 100}")
    print(f"  📊 PROJECTIONS: {best['name']}")
    print(f"{'=' * 100}")
    for label, mret in [("Avg", best["avg_monthly"]),
                         ("Median", best["med_monthly"]),
                         ("p25", best["monthly"].quantile(0.25))]:
        m6 = (1 + mret) ** 6 - 1
        y1 = (1 + mret) ** 12 - 1
        print(f"  {label:>7s} ({mret*100:+.1f}%/мес): "
              f"1м=${CAP*(1+mret):.0f} | 6м=${CAP*(1+m6):.0f} | 12м=${CAP*(1+y1):.0f}")

    if safe_results and safe_results[0]["name"] != best["name"]:
        sb = safe_results[0]
        print(f"\n  📊 SAFE PICK: {sb['name']}")
        for label, mret in [("Avg", sb["avg_monthly"]),
                             ("Median", sb["med_monthly"]),
                             ("p25", sb["monthly"].quantile(0.25))]:
            m6 = (1 + mret) ** 6 - 1
            y1 = (1 + mret) ** 12 - 1
            print(f"  {label:>7s} ({mret*100:+.1f}%/мес): "
                  f"1м=${CAP*(1+mret):.0f} | 6м=${CAP*(1+m6):.0f} | 12м=${CAP*(1+y1):.0f}")


if __name__ == "__main__":
    main()
