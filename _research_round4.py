#!/usr/bin/env python3
"""
Research round 4: advanced alpha enhancements.
Ideas:
  A) Dispersion-aware sizing — more positions when CS dispersion is high
  B) Rolling IC filter — only trade when model "works" (recent IC > 0)
  C) Equity momentum — scale down after drawdowns (anti-tilt)
  D) Fine-tuned dynamic exposure thresholds
  E) Asymmetric L/S sizing per regime
  F) Confidence-weighted (prediction magnitude → position weight)
  G) Best combos
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

# ─────────── helpers ─────────────────────────────────────


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


# ─────────── simulation engine ─────────────────────────────


def simulate(merged, regime_df, horizon, cfg):
    """Unified simulation with configurable risk management."""
    n_long = cfg.get("n_long", 5)
    n_short = cfg.get("n_short", 5)
    trend_cutoff = cfg.get("trend_cutoff", 0.8)
    dyn_threshold = cfg.get("dyn_threshold", None)  # start fading at this trend
    edge_weight = cfg.get("edge_weight", False)
    confidence_weight = cfg.get("confidence_weight", False)
    dispersion_scale = cfg.get("dispersion_scale", False)
    rolling_ic_filter = cfg.get("rolling_ic_filter", False)
    equity_momentum = cfg.get("equity_momentum", False)

    all_rets = []
    ic_window = []
    IC_LOOKBACK = 24 * 5  # 5 days lookback for rolling IC
    equity_curve = [1.0]

    timestamps_sorted = sorted(merged["timestamp"].unique())
    grouped = {ts: grp for ts, grp in merged.groupby("timestamp")}

    for ts in timestamps_sorted:
        if ts not in regime_df.index or ts not in grouped:
            continue
        row = regime_df.loc[ts]
        trend_str = row.get("trend_strength", 0)

        # Trend cutoff
        if trend_str > trend_cutoff:
            continue

        grp = grouped[ts].copy()
        n = len(grp)
        nl = min(n_long, n // 3)
        ns = min(n_short, n // 3)
        if nl == 0 or ns == 0:
            continue

        # Rolling IC filter: skip if recent IC < 0
        if rolling_ic_filter and len(ic_window) >= IC_LOOKBACK:
            recent_ic = np.mean(ic_window[-IC_LOOKBACK:])
            if recent_ic < -0.01:
                continue

        # Determine positions
        grp["pred_rank"] = grp["pred"].rank(ascending=False)
        longs = grp[grp["pred_rank"] <= nl].copy()
        shorts = grp[grp["pred_rank"] > (n - ns)].copy()

        if confidence_weight:
            # Weight by absolute prediction magnitude
            lw = longs["pred"].abs()
            lw = lw / (lw.sum() + 1e-10)
            sw = shorts["pred"].abs()
            sw = sw / (sw.sum() + 1e-10)
            long_ret = (longs["fwd_ret"] * lw).sum()
            short_ret = (shorts["fwd_ret"] * sw).sum()
        elif edge_weight:
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

        # Dynamic exposure: fade between dyn_threshold and cutoff
        if dyn_threshold is not None and trend_str > dyn_threshold:
            exposure = 1.0 - (trend_str - dyn_threshold) / (trend_cutoff - dyn_threshold + 1e-10) * 0.5
            port_ret *= max(0.1, exposure)

        # Dispersion-based scaling
        if dispersion_scale:
            cs_disp = grp["fwd_ret"].std()
            # Use pred dispersion as proxy (fwd_ret is future, but pred_std is known)
            pred_disp = grp["pred"].std()
            if pred_disp > 0:
                # Scale up when high dispersion (more spread = more opportunity)
                scale = min(1.5, pred_disp / (grp["pred"].rolling(1).std().mean() + 1e-10))
                # Simpler: just use pred std relative to its mean
                port_ret *= min(1.5, max(0.5, pred_disp * 10))

        # Equity momentum: scale down after recent drawdown
        if equity_momentum and len(equity_curve) > 48:
            recent_eq = equity_curve[-1]
            peak_eq = max(equity_curve[-48:])
            dd = (recent_eq - peak_eq) / (peak_eq + 1e-10)
            if dd < -0.05:
                # In drawdown > 5%, scale down proportionally
                scale = max(0.3, 1.0 + dd * 3)  # e.g., -10% dd → 0.7x
                port_ret *= scale

        # Track IC for rolling filter
        if len(longs) > 0 and len(shorts) > 0:
            ts_ic = stats.spearmanr(grp["pred"], grp["fwd_ret"])[0]
            if not np.isnan(ts_ic):
                ic_window.append(ts_ic)

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
    # Calmar-like: avg_monthly / abs(worst_m)
    calmar = avg_monthly / (abs(worst_m) + 1e-10)

    return {
        "name": name, "sharpe": sharpe, "total": total, "maxdd": maxdd,
        "worst_m": worst_m, "equity": equity, "monthly": monthly,
        "avg_monthly": avg_monthly, "med_monthly": med_monthly,
        "calmar": calmar, "month_data": month_data,
    }


def show(r):
    if r is None:
        return
    safe = "✅" if r["worst_m"] > -0.15 else ("⚠️" if r["worst_m"] > -0.25 else "❌")
    print(f"  {safe} {r['name']:<62s} Sh={r['sharpe']:>+5.2f} "
          f"Wr={r['worst_m']*100:>+6.1f}% → ${r['equity']:.0f}")


# ─────────── main ─────────────────────────────────────────


def main():
    LEV = 5
    CAP = 100

    print("=" * 100)
    print(f"  RESEARCH ROUND 4: Advanced Alpha Enhancements ({LEV}x, ${CAP})")
    print("=" * 100)

    print("\n  Loading data...")
    df35 = load_data(SYM_35)
    df20 = load_data(SYM_20)
    feats = [f for f in FEATURES if f in df35.columns]
    regime_df = compute_regime(df35)
    n35 = df35["symbol"].nunique()
    print(f"    {n35} symbols: {df35.shape[0]:,} rows")

    print("\n  Training models...")
    p35 = train_and_predict(df35, feats, 12)
    p20 = train_and_predict(df20, feats, 12)
    print(f"    {n35}sym 12h: {len(p35):,} | 20sym 12h: {len(p20):,}")

    results = []

    # Baseline (round 3 winner): 35sym 5L/5S DYN-EXPOSURE
    print(f"\n{'─' * 100}")
    print(f"  BASELINE (round 3 winner)")
    print(f"{'─' * 100}")
    cfg_base = {"n_long": 5, "n_short": 5, "trend_cutoff": 0.8, "dyn_threshold": 0.5}
    sub = simulate(p35, regime_df, 12, cfg_base)
    r = eval_config(sub, 12, "BASELINE: 35sym 5L/5S dyn(0.5→0.8)", LEV, CAP)
    if r:
        results.append(r)
        show(r)

    # ── SECTION A: Fine-tune dynamic exposure thresholds ──
    print(f"\n{'─' * 100}")
    print(f"  A: Dynamic exposure threshold sweep")
    print(f"{'─' * 100}")

    for nsym, preds, label in [(n35, p35, f"{n35}sym"), (20, p20, "20sym")]:
        for nl, ns in [(4, 4), (5, 5), (6, 6)]:
            for dyn_th in [0.3, 0.4, 0.5, 0.6]:
                cfg = {"n_long": nl, "n_short": ns, "trend_cutoff": 0.8,
                       "dyn_threshold": dyn_th}
                sub = simulate(preds, regime_df, 12, cfg)
                name = f"{label} {nl}L/{ns}S dyn({dyn_th}→0.8)"
                r = eval_config(sub, 12, name, LEV, CAP)
                if r:
                    results.append(r)
                    show(r)

    # ── SECTION B: Rolling IC filter ──
    print(f"\n{'─' * 100}")
    print(f"  B: Rolling IC filter (skip when model is cold)")
    print(f"{'─' * 100}")

    for nsym, preds, label in [(n35, p35, f"{n35}sym"), (20, p20, "20sym")]:
        for nl, ns in [(5, 5), (6, 6)]:
            cfg = {"n_long": nl, "n_short": ns, "trend_cutoff": 0.8,
                   "dyn_threshold": 0.5, "rolling_ic_filter": True}
            sub = simulate(preds, regime_df, 12, cfg)
            name = f"{label} {nl}L/{ns}S dyn+IC-filter"
            r = eval_config(sub, 12, name, LEV, CAP)
            if r:
                results.append(r)
                show(r)

    # ── SECTION C: Equity momentum (anti-tilt) ──
    print(f"\n{'─' * 100}")
    print(f"  C: Equity momentum (scale down in drawdown)")
    print(f"{'─' * 100}")

    for nsym, preds, label in [(n35, p35, f"{n35}sym"), (20, p20, "20sym")]:
        for nl, ns in [(5, 5), (6, 6)]:
            cfg = {"n_long": nl, "n_short": ns, "trend_cutoff": 0.8,
                   "dyn_threshold": 0.5, "equity_momentum": True}
            sub = simulate(preds, regime_df, 12, cfg)
            name = f"{label} {nl}L/{ns}S dyn+EQ-MOM"
            r = eval_config(sub, 12, name, LEV, CAP)
            if r:
                results.append(r)
                show(r)

    # ── SECTION D: Confidence-weighted positions ──
    print(f"\n{'─' * 100}")
    print(f"  D: Confidence-weighted positions")
    print(f"{'─' * 100}")

    for nsym, preds, label in [(n35, p35, f"{n35}sym"), (20, p20, "20sym")]:
        for nl, ns in [(5, 5), (6, 6)]:
            cfg = {"n_long": nl, "n_short": ns, "trend_cutoff": 0.8,
                   "dyn_threshold": 0.5, "confidence_weight": True}
            sub = simulate(preds, regime_df, 12, cfg)
            name = f"{label} {nl}L/{ns}S dyn+CONF-W"
            r = eval_config(sub, 12, name, LEV, CAP)
            if r:
                results.append(r)
                show(r)

    # ── SECTION E: Asymmetric L/S ──
    print(f"\n{'─' * 100}")
    print(f"  E: Asymmetric L/S sizing")
    print(f"{'─' * 100}")

    for nsym, preds, label in [(n35, p35, f"{n35}sym"), (20, p20, "20sym")]:
        for nl, ns in [(7, 3), (3, 7), (6, 4), (4, 6), (8, 4), (4, 8)]:
            cfg = {"n_long": nl, "n_short": ns, "trend_cutoff": 0.8,
                   "dyn_threshold": 0.5}
            sub = simulate(preds, regime_df, 12, cfg)
            name = f"{label} {nl}L/{ns}S dyn"
            r = eval_config(sub, 12, name, LEV, CAP)
            if r:
                results.append(r)
                show(r)

    # ── SECTION F: Best combos ──
    print(f"\n{'─' * 100}")
    print(f"  F: Best combinations")
    print(f"{'─' * 100}")

    combos = [
        {"n_long": 5, "n_short": 5, "trend_cutoff": 0.8, "dyn_threshold": 0.5,
         "rolling_ic_filter": True, "equity_momentum": True,
         "label": "dyn+IC+EQ-MOM"},
        {"n_long": 5, "n_short": 5, "trend_cutoff": 0.8, "dyn_threshold": 0.5,
         "confidence_weight": True, "equity_momentum": True,
         "label": "dyn+CONF-W+EQ-MOM"},
        {"n_long": 5, "n_short": 5, "trend_cutoff": 0.8, "dyn_threshold": 0.4,
         "rolling_ic_filter": True,
         "label": "dyn(0.4)+IC"},
        {"n_long": 5, "n_short": 5, "trend_cutoff": 0.8, "dyn_threshold": 0.4,
         "confidence_weight": True,
         "label": "dyn(0.4)+CONF-W"},
        {"n_long": 6, "n_short": 6, "trend_cutoff": 0.8, "dyn_threshold": 0.4,
         "rolling_ic_filter": True, "equity_momentum": True,
         "label": "6L6S dyn(0.4)+IC+EQ"},
        {"n_long": 6, "n_short": 4, "trend_cutoff": 0.8, "dyn_threshold": 0.5,
         "rolling_ic_filter": True,
         "label": "6L4S dyn+IC"},
        {"n_long": 4, "n_short": 6, "trend_cutoff": 0.8, "dyn_threshold": 0.5,
         "rolling_ic_filter": True,
         "label": "4L6S dyn+IC"},
        {"n_long": 5, "n_short": 5, "trend_cutoff": 0.7, "dyn_threshold": 0.4,
         "label": "dyn(0.4→0.7) tighter"},
        {"n_long": 5, "n_short": 5, "trend_cutoff": 0.9, "dyn_threshold": 0.5,
         "label": "dyn(0.5→0.9) looser"},
    ]

    for nsym, preds, sym_label in [(n35, p35, f"{n35}sym"), (20, p20, "20sym")]:
        for combo in combos:
            cfg = {k: v for k, v in combo.items() if k != "label"}
            sub = simulate(preds, regime_df, 12, cfg)
            name = f"{sym_label} {combo['label']}"
            r = eval_config(sub, 12, name, LEV, CAP)
            if r:
                results.append(r)
                show(r)

    # ── GRAND RANKING ──────────────────────────────────────
    if not results:
        print("  No results")
        return

    # Risk-adjusted score: equity * (1 + worst_month) * calmar^0.3
    for r in results:
        safety = max(0.3, 1.0 + r["worst_m"])
        r["score"] = r["equity"] * safety * (r["calmar"] ** 0.3 + 0.01)

    results.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n{'=' * 100}")
    print(f"  🏆 TOP 10 RISK-ADJUSTED ({LEV}x, ${CAP})")
    print(f"{'=' * 100}")
    for i, r in enumerate(results[:10]):
        safe = "✅" if r["worst_m"] > -0.15 else ("⚠️" if r["worst_m"] > -0.25 else "❌")
        print(f"\n  #{i+1} {safe} {r['name']}")
        print(f"      Sh={r['sharpe']:.2f} | Wr={r['worst_m']*100:+.1f}% | "
              f"Avg/m={r['avg_monthly']*100:+.1f}% | Med/m={r['med_monthly']*100:+.1f}% | "
              f"Calmar={r['calmar']:.2f}")
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

    # ── TOP 5 BY SAFETY (worst_m > -15%) ──
    safe_results = [r for r in results if r["worst_m"] > -0.20]
    safe_results.sort(key=lambda x: x["equity"], reverse=True)
    print(f"\n{'=' * 100}")
    print(f"  🛡️ TOP 5 SAFE (worst month > -20%):")
    print(f"{'=' * 100}")
    for i, r in enumerate(safe_results[:5]):
        print(f"  #{i+1} {r['name']}: ${CAP}→${r['equity']:.0f} | "
              f"Wr={r['worst_m']*100:+.1f}% | Sh={r['sharpe']:.2f}")

    # ── PROJECTIONS for overall best ──
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

    # Also show projections for safest high-equity config
    if safe_results:
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
