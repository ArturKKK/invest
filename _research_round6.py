#!/usr/bin/env python3
"""
Research round 6: fresh ideas beyond position management.
Focus on SIGNAL quality, not just risk management.

New ideas:
  A) Spread gate — only trade when L-S predicted spread > threshold
  B) Regime-dependent N — more positions in low-vol, fewer in high-vol
  C) Vol-normalized returns — rank by risk-adjusted fwd_ret
  D) Momentum-of-strategy — use recent strategy P&L as meta-signal
  E) Rebalance frequency — 8h, 16h, 24h vs baseline 12h
  F) Top-K confidence — only trade top/bottom K% of prediction range
  G) Correlation-aware weighting — inversely weight by pairwise corr
  H) Mega combos from best
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
    # Vol regime for adaptive N
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


def compute_recent_vol(merged):
    """Compute per-symbol trailing 48h vol for correlation-aware weighting."""
    merged = merged.sort_values(["symbol", "timestamp"])
    merged["sym_vol"] = merged.groupby("symbol")["fwd_ret"].transform(
        lambda x: x.rolling(48, min_periods=12).std())
    return merged


def simulate(merged, regime_df, horizon, cfg):
    """Round 6 simulation engine."""
    n_long = cfg.get("n_long", 5)
    n_short = cfg.get("n_short", 5)
    trend_cutoff = cfg.get("trend_cutoff", 0.8)
    dyn_threshold = cfg.get("dyn_threshold", 0.5)
    eq_mom_boost = cfg.get("eq_mom_boost", True)  # default ON (R5 winner)
    kelly_sizing = cfg.get("kelly_sizing", True)   # default ON (R5 winner)

    # New R6 ideas
    spread_gate = cfg.get("spread_gate", None)       # min L-S spread to trade
    adaptive_n = cfg.get("adaptive_n", False)         # vary N by vol regime
    strategy_momentum = cfg.get("strategy_momentum", False)  # meta-signal
    strat_mom_lookback = cfg.get("strat_mom_lookback", 72)   # hours
    topk_pct = cfg.get("topk_pct", None)              # only trade top/bot K% of preds
    corr_weight = cfg.get("corr_weight", False)       # inverse-correlation weighting
    rebal_hours = cfg.get("rebal_hours", horizon)     # actual rebalance freq

    all_rets = []
    equity_curve = [1.0]
    strategy_rets = []  # for strategy momentum

    timestamps_sorted = sorted(merged["timestamp"].unique())
    grouped = {ts: grp for ts, grp in merged.groupby("timestamp")}

    for i_ts, ts in enumerate(timestamps_sorted):
        if ts not in regime_df.index or ts not in grouped:
            continue
        row = regime_df.loc[ts]
        trend_str = row.get("trend_strength", 0)
        vol_regime = row.get("vol_regime", 1.0)

        if trend_str > trend_cutoff:
            continue

        grp = grouped[ts].copy()
        n = len(grp)

        # Dynamic exposure
        if dyn_threshold is not None and trend_str > dyn_threshold:
            exposure = max(0.1, 1.0 - (trend_str - dyn_threshold) / (trend_cutoff - dyn_threshold + 1e-10) * 0.5)
        else:
            exposure = 1.0

        # Strategy momentum: skip if strategy has been losing recently
        if strategy_momentum and len(strategy_rets) >= strat_mom_lookback:
            recent = strategy_rets[-strat_mom_lookback:]
            cum = np.prod([1 + r for r in recent])
            if cum < 0.97:  # strategy down > 3% recently → reduce
                exposure *= max(0.3, cum)

        # Adaptive N: fewer positions in high vol, more in low vol
        if adaptive_n:
            if not np.isnan(vol_regime) and vol_regime > 0:
                scale = 1.0 / max(0.5, min(2.0, vol_regime))
                nl = max(3, int(round(n_long * scale)))
                ns = max(3, int(round(n_short * scale)))
            else:
                nl, ns = n_long, n_short
        else:
            nl, ns = n_long, n_short

        nl = min(nl, n // 3)
        ns = min(ns, n // 3)
        if nl == 0 or ns == 0:
            continue

        grp["pred_rank"] = grp["pred"].rank(ascending=False)
        longs = grp[grp["pred_rank"] <= nl].copy()
        shorts = grp[grp["pred_rank"] > (n - ns)].copy()

        # Spread gate
        if spread_gate is not None:
            pred_spread = longs["pred"].mean() - shorts["pred"].mean()
            if pred_spread < spread_gate:
                continue

        # Top-K pct: only trade if pred is in extreme tails
        if topk_pct is not None:
            pred_range = grp["pred"].max() - grp["pred"].min()
            if pred_range > 0:
                # long pred must be in top topk_pct
                long_threshold = grp["pred"].quantile(1 - topk_pct)
                short_threshold = grp["pred"].quantile(topk_pct)
                longs = longs[longs["pred"] >= long_threshold]
                shorts = shorts[shorts["pred"] <= short_threshold]
                if len(longs) == 0 or len(shorts) == 0:
                    continue

        # Correlation-aware weighting
        if corr_weight and "sym_vol" in grp.columns:
            # Weight inversely by vol (low-vol = more weight = more diversification)
            lw = 1.0 / (longs["sym_vol"] + 1e-6)
            lw = lw / (lw.sum() + 1e-10)
            sw = 1.0 / (shorts["sym_vol"] + 1e-6)
            sw = sw / (sw.sum() + 1e-10)
            long_ret = (longs["fwd_ret"] * lw).sum()
            short_ret = (shorts["fwd_ret"] * sw).sum()
        else:
            long_ret = longs["fwd_ret"].mean()
            short_ret = shorts["fwd_ret"].mean()

        # Kelly sizing
        if kelly_sizing:
            pred_spread = longs["pred"].mean() - shorts["pred"].mean()
            long_alloc = np.clip(0.5 + pred_spread * 5, 0.3, 0.7)
            port_ret = long_alloc * long_ret - (1 - long_alloc) * short_ret
        else:
            port_ret = 0.5 * long_ret - 0.5 * short_ret

        port_ret *= exposure

        # EQ-MOM boost
        if eq_mom_boost and len(equity_curve) > 48:
            recent_eq = equity_curve[-1]
            peak_eq = max(equity_curve[-48:])
            dd = (recent_eq - peak_eq) / (peak_eq + 1e-10)
            if dd < -0.05:
                scale = max(0.3, 1.0 + dd * 3)
                port_ret *= scale
            elif dd > -0.01:
                trough_eq = min(equity_curve[-48:])
                recovery = (recent_eq - trough_eq) / (trough_eq + 1e-10)
                if recovery > 0.05:
                    boost = min(1.5, 1.0 + recovery * 0.5)
                    port_ret *= boost

        all_rets.append({"timestamp": ts, "portfolio_ret": port_ret})
        equity_curve.append(equity_curve[-1] * (1 + port_ret))
        strategy_rets.append(port_ret)

    if not all_rets:
        return None
    port = pd.DataFrame(all_rets).sort_values("timestamp")
    return port.iloc[::rebal_hours]


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
    # Win months
    win_months = (monthly > 0).sum()
    total_months = len(monthly)

    return {
        "name": name, "sharpe": sharpe, "total": total, "maxdd": maxdd,
        "worst_m": worst_m, "equity": equity, "monthly": monthly,
        "avg_monthly": avg_monthly, "med_monthly": med_monthly,
        "calmar": calmar, "month_data": month_data,
        "win_months": win_months, "total_months": total_months,
    }


def show(r):
    if r is None:
        return
    safe = "✅" if r["worst_m"] > -0.15 else ("⚠️" if r["worst_m"] > -0.25 else "❌")
    wm = f"{r['win_months']}/{r['total_months']}"
    print(f"  {safe} {r['name']:<58s} Sh={r['sharpe']:>+5.2f} "
          f"Wr={r['worst_m']*100:>+6.1f}% WM={wm} → ${r['equity']:.0f}")


def main():
    LEV = 5
    CAP = 100

    print("=" * 100)
    print(f"  RESEARCH ROUND 6: Signal Quality & Meta-Strategies ({LEV}x, ${CAP})")
    print("=" * 100)

    print("\n  Loading data...")
    df35 = load_data(SYM_35)
    feats = [f for f in FEATURES if f in df35.columns]
    regime_df = compute_regime(df35)
    n35 = df35["symbol"].nunique()
    print(f"    {n35} sym loaded")

    print("\n  Training models (12h)...")
    p35 = train_and_predict(df35, feats, 12)
    p35 = compute_recent_vol(p35)
    print(f"    {len(p35):,} predictions")

    results = []

    # ── BASELINE (R5 winner) ──
    print(f"\n{'─' * 100}")
    print(f"  BASELINE: R5 winner (35sym 5L/5S EQ-BOOST+KELLY)")
    print(f"{'─' * 100}")
    cfg_base = {"n_long": 5, "n_short": 5, "trend_cutoff": 0.8,
                "dyn_threshold": 0.5, "eq_mom_boost": True, "kelly_sizing": True}
    sub = simulate(p35, regime_df, 12, cfg_base)
    r = eval_config(sub, 12, "BASELINE: EQ-BOOST+KELLY", LEV, CAP)
    if r: results.append(r); show(r)

    # ── A: Spread gate — only trade high-conviction timestamps ──
    print(f"\n{'─' * 100}")
    print(f"  A: Spread gate (min predicted L-S spread)")
    print(f"{'─' * 100}")

    for gate in [0.001, 0.005, 0.01, 0.02, 0.03, 0.05]:
        cfg = {**cfg_base, "spread_gate": gate}
        sub = simulate(p35, regime_df, 12, cfg)
        name = f"EQ+K spread>{gate}"
        r = eval_config(sub, 12, name, LEV, CAP)
        if r: results.append(r); show(r)

    # ── B: Regime-dependent N ──
    print(f"\n{'─' * 100}")
    print(f"  B: Adaptive N (more positions in low-vol, fewer in high-vol)")
    print(f"{'─' * 100}")

    for nl, ns in [(5, 5), (6, 6), (7, 7), (8, 8)]:
        cfg = {**cfg_base, "n_long": nl, "n_short": ns, "adaptive_n": True}
        sub = simulate(p35, regime_df, 12, cfg)
        name = f"EQ+K adaptN base={nl}L/{ns}S"
        r = eval_config(sub, 12, name, LEV, CAP)
        if r: results.append(r); show(r)

    # ── C: Strategy momentum (meta-signal) ──
    print(f"\n{'─' * 100}")
    print(f"  C: Strategy momentum (reduce when strategy losing)")
    print(f"{'─' * 100}")

    for lookback in [48, 72, 120]:
        cfg = {**cfg_base, "strategy_momentum": True, "strat_mom_lookback": lookback}
        sub = simulate(p35, regime_df, 12, cfg)
        name = f"EQ+K strat-mom(lb={lookback}h)"
        r = eval_config(sub, 12, name, LEV, CAP)
        if r: results.append(r); show(r)

    # ── D: Top-K confidence (extreme tails only) ──
    print(f"\n{'─' * 100}")
    print(f"  D: Top-K confidence (trade only extreme prediction tails)")
    print(f"{'─' * 100}")

    for topk in [0.10, 0.15, 0.20, 0.25, 0.30]:
        cfg = {**cfg_base, "topk_pct": topk}
        sub = simulate(p35, regime_df, 12, cfg)
        name = f"EQ+K topK={int(topk*100)}%"
        r = eval_config(sub, 12, name, LEV, CAP)
        if r: results.append(r); show(r)

    # ── E: Correlation-aware weighting ──
    print(f"\n{'─' * 100}")
    print(f"  E: Correlation-aware (inverse-vol) weighting")
    print(f"{'─' * 100}")

    for nl, ns in [(5, 5), (6, 6)]:
        cfg = {**cfg_base, "n_long": nl, "n_short": ns, "corr_weight": True}
        sub = simulate(p35, regime_df, 12, cfg)
        name = f"EQ+K {nl}L/{ns}S CORR-W"
        r = eval_config(sub, 12, name, LEV, CAP)
        if r: results.append(r); show(r)

    # ── F: Rebalance frequency ──
    print(f"\n{'─' * 100}")
    print(f"  F: Rebalance frequency sweep")
    print(f"{'─' * 100}")

    for rebal in [6, 8, 16, 24]:
        cfg = {**cfg_base, "rebal_hours": rebal}
        sub = simulate(p35, regime_df, 12, cfg)
        name = f"EQ+K rebal={rebal}h"
        r = eval_config(sub, 12, name, LEV, CAP)
        if r: results.append(r); show(r)

    # ── G: Different base N with R5 setup ──
    print(f"\n{'─' * 100}")
    print(f"  G: N sweep with full R5 stack")
    print(f"{'─' * 100}")

    for nl, ns in [(3, 3), (4, 4), (6, 6), (7, 7), (8, 8), (3, 5), (5, 3), (4, 6), (6, 4)]:
        cfg = {**cfg_base, "n_long": nl, "n_short": ns}
        sub = simulate(p35, regime_df, 12, cfg)
        name = f"EQ+K {nl}L/{ns}S"
        r = eval_config(sub, 12, name, LEV, CAP)
        if r: results.append(r); show(r)

    # ── H: MEGA COMBOS ──
    print(f"\n{'─' * 100}")
    print(f"  H: Mega combos (best of R6)")
    print(f"{'─' * 100}")

    combos = [
        {"spread_gate": 0.005, "adaptive_n": True, "n_long": 6, "n_short": 6,
         "label": "adaptN6+spread005"},
        {"spread_gate": 0.01, "adaptive_n": True, "n_long": 6, "n_short": 6,
         "label": "adaptN6+spread01"},
        {"spread_gate": 0.005, "corr_weight": True,
         "label": "corrW+spread005"},
        {"spread_gate": 0.01, "strategy_momentum": True, "strat_mom_lookback": 72,
         "label": "stratMom+spread01"},
        {"topk_pct": 0.20, "corr_weight": True,
         "label": "topK20+corrW"},
        {"spread_gate": 0.005, "topk_pct": 0.20,
         "label": "spread005+topK20"},
        {"adaptive_n": True, "n_long": 7, "n_short": 7, "corr_weight": True,
         "label": "adaptN7+corrW"},
        {"spread_gate": 0.005, "adaptive_n": True, "n_long": 7, "n_short": 7,
         "corr_weight": True, "label": "adaptN7+corrW+spread005"},
        {"spread_gate": 0.01, "rebal_hours": 8,
         "label": "rebal8h+spread01"},
        {"spread_gate": 0.005, "n_long": 6, "n_short": 4,
         "label": "6L4S+spread005"},
        {"spread_gate": 0.005, "n_long": 4, "n_short": 6,
         "label": "4L6S+spread005"},
        {"strategy_momentum": True, "strat_mom_lookback": 72,
         "corr_weight": True, "label": "stratMom+corrW"},
    ]

    for combo in combos:
        cfg = {**cfg_base}
        lab = combo.pop("label")
        cfg.update(combo)
        combo["label"] = lab  # restore
        sub = simulate(p35, regime_df, 12, cfg)
        name = f"EQ+K {lab}"
        r = eval_config(sub, 12, name, LEV, CAP)
        if r: results.append(r); show(r)

    # ── GRAND RANKING ──
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
        wm = f"{r['win_months']}/{r['total_months']}"
        print(f"\n  #{i+1} {safe} {r['name']}")
        print(f"      Sh={r['sharpe']:.2f} | Wr={r['worst_m']*100:+.1f}% | "
              f"Avg/m={r['avg_monthly']*100:+.1f}% | Med/m={r['med_monthly']*100:+.1f}% | "
              f"WM={wm}")
        print(f"      ${CAP} → ${r['equity']:.0f} ({len(r['month_data'])} мес) | "
              f"Score={r['score']:.0f}")
        for md in r["month_data"]:
            marker = " ← worst" if md["ret"] == r["worst_m"] else ""
            print(f"         {md['month']:>10s}  {md['ret']*100:>+7.1f}%  "
                  f"equity=${md['equity']:>7.0f}{marker}")

    results.sort(key=lambda x: x["equity"], reverse=True)
    print(f"\n{'=' * 100}")
    print(f"  💰 TOP 5 RAW EQUITY:")
    print(f"{'=' * 100}")
    for i, r in enumerate(results[:5]):
        safe = "✅" if r["worst_m"] > -0.15 else ("⚠️" if r["worst_m"] > -0.25 else "❌")
        wm = f"{r['win_months']}/{r['total_months']}"
        print(f"  #{i+1} {safe} {r['name']}: ${CAP}→${r['equity']:.0f} | "
              f"Wr={r['worst_m']*100:+.1f}% | WM={wm} | Sh={r['sharpe']:.2f}")

    safe_r = [r for r in results if r["worst_m"] > -0.10]
    safe_r.sort(key=lambda x: x["equity"], reverse=True)
    print(f"\n{'=' * 100}")
    print(f"  🛡️ TOP 5 ULTRA-SAFE (worst > -10%):")
    print(f"{'=' * 100}")
    for i, r in enumerate(safe_r[:5]):
        wm = f"{r['win_months']}/{r['total_months']}"
        print(f"  #{i+1} {r['name']}: ${CAP}→${r['equity']:.0f} | "
              f"Wr={r['worst_m']*100:+.1f}% | WM={wm} | Sh={r['sharpe']:.2f}")

    # Projections
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


if __name__ == "__main__":
    main()
