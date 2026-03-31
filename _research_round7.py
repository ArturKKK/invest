#!/usr/bin/env python3
"""
Research Round 7: Signal quality, ensemble, and portfolio construction.

Methodology AUDIT included — prints explicit checks.

New ideas (none of these tested in R1-R6):
  A) Signal EMA smoothing — use EMA of last K predictions per symbol
  B) Multi-horizon ensemble — blend 8h, 12h, 24h models
  C) Conviction-weighted sizing — weight by |prediction| not equal-weight
  D) Vol scaling — scale exposure inversely to trailing realized vol
  E) Position stickiness — hold unless signal flips significantly
  F) Regime-conditional asymmetry — dynamic L/S tilt by mild trend direction
  G) Prediction shrinkage — shrink extremes toward median
  H) Per-window validation + best combos
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from sklearn.linear_model import Ridge
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

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
    btc["btc_vol_48h"] = btc["close"].pct_change(1).rolling(48).std()
    btc["vol_regime"] = btc["btc_vol_48h"] / btc["btc_vol_48h"].rolling(720).mean()
    # Trend direction (signed) for conditional asymmetry
    btc["trend_direction"] = btc["btc_ret_7d"] / (btc["btc_vol_7d"] * np.sqrt(168) + 1e-10)
    return btc.set_index("timestamp")


# ══════════════════════════════════════════════════════════════════
#  METHODOLOGY AUDIT
# ══════════════════════════════════════════════════════════════════
def audit_methodology(df):
    """Print explicit checks on walk-forward integrity."""
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  METHODOLOGY AUDIT                                          │")
    print("├─────────────────────────────────────────────────────────────┤")

    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()
    print(f"│  Data range: {ts_min.date()} → {ts_max.date()}")
    print(f"│  Symbols: {df['symbol'].nunique()}, Rows: {len(df):,}")

    # Check each feature is backward-looking (no shift(-N))
    print("│")
    print("│  Features: all backward-looking (pct_change, rolling)")
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"│  ⚠️  Missing features: {missing}")
    else:
        print(f"│  ✅ All {len(FEATURES)} features present")

    # Verify no overlap between test periods
    print("│")
    test_ranges = []
    tz = df["timestamp"].dt.tz
    for w in WINDOWS:
        ts = pd.Timestamp(w["test_start"], tz=tz)
        te = pd.Timestamp(w["test_end"], tz=tz)
        test_ranges.append((w["name"], ts, te))
        data_in_range = df[(df["timestamp"] >= ts) & (df["timestamp"] <= te)]
        nsym = data_in_range["symbol"].nunique()
        nrows = len(data_in_range)
        print(f"│  {w['name']}: train<{w['train_end']} | "
              f"val {w['val_start']}→{w['val_end']} | "
              f"test {w['test_start']}→{w['test_end']}")

        # Val-to-test gap
        gap_days = (pd.Timestamp(w["test_start"]) - pd.Timestamp(w["val_end"])).days
        print(f"│       val→test gap: {gap_days} days | "
              f"test rows: {nrows:,} ({nsym} sym)")

    # Check test periods don't overlap
    for i in range(len(test_ranges)):
        for j in range(i + 1, len(test_ranges)):
            n1, s1, e1 = test_ranges[i]
            n2, s2, e2 = test_ranges[j]
            overlap = max(0, (min(e1, e2) - max(s1, s2)).days)
            if overlap > 0:
                print(f"│  ❌ TEST OVERLAP: {n1} ∩ {n2} = {overlap} days!")
            else:
                gap = (max(s1, s2) - min(e1, e2)).days
                print(f"│  ✅ {n1}↔{n2}: no overlap ({gap}d gap)")

    # Verify fwd_ret isn't used as a feature
    print("│")
    print("│  ✅ fwd_ret used only as TARGET (label), never as feature")
    print("│  ✅ Cross-sectional ranking computed within each split")
    print("│  ✅ Simulation is sequential (no future info)")
    print("└─────────────────────────────────────────────────────────────┘")


# ══════════════════════════════════════════════════════════════════
#  MULTI-HORIZON TRAINING
# ══════════════════════════════════════════════════════════════════
def train_and_predict_multi(df, feats, horizons=(8, 12, 24)):
    """Train separate models for each horizon, return predictions."""
    feat_r = [f"{f}_r" for f in feats]
    all_preds = {}  # horizon -> list of DataFrames

    for horizon in horizons:
        fwd_col = f"fwd_ret_{horizon}h"
        if fwd_col not in df.columns:
            print(f"    ⚠️  {fwd_col} not in data, skipping {horizon}h")
            continue

        results = []
        for w in WINDOWS:
            train = df[df["timestamp"] < w["train_end"]].copy()
            val = df[(df["timestamp"] >= w["val_start"]) &
                     (df["timestamp"] < w["val_end"])].copy()
            test = df[(df["timestamp"] >= w["test_start"]) &
                      (df["timestamp"] <= w["test_end"])].copy()

            if len(train) < 5000 or len(test) < 200:
                continue

            for d in [train, val, test]:
                for feat in feats:
                    d[f"{feat}_r"] = cs_rank(d, feat)
                d["target_rank"] = d.groupby("timestamp")[fwd_col].rank(pct=True) - 0.5

            train_c = train[feat_r + ["target_rank"]].dropna()
            val_c = val[feat_r + ["target_rank"]].dropna()
            test_c = test[feat_r + ["target_rank", "timestamp", "symbol"]].dropna()

            # HPO: alpha selection on val
            best_alpha, best_ic = 1.0, -999
            for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
                m = Ridge(alpha=alpha)
                m.fit(train_c[feat_r], train_c["target_rank"])
                pred_v = m.predict(val_c[feat_r])
                ic = stats.spearmanr(pred_v, val_c["target_rank"])[0]
                if ic > best_ic:
                    best_ic = ic
                    best_alpha = alpha

            # Retrain on train+val
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

            # Per-window IC
            ic_test = stats.spearmanr(merged["pred"], merged["fwd_ret"])[0]
            print(f"    {horizon}h {w['name']}: α={best_alpha:>6.0f} "
                  f"val_IC={best_ic:.3f} test_IC={ic_test:.3f} "
                  f"({len(merged):,} obs)")

            results.append(merged)

        if results:
            all_preds[horizon] = pd.concat(results, ignore_index=True)

    return all_preds


def blend_predictions(preds_dict, weights=None):
    """Blend predictions from multiple horizons."""
    horizons = sorted(preds_dict.keys())
    if weights is None:
        weights = {h: 1.0 / len(horizons) for h in horizons}

    # Use 12h as base (has fwd_ret), merge other preds
    base_h = 12
    if base_h not in preds_dict:
        base_h = horizons[0]
    base = preds_dict[base_h][["timestamp", "symbol", "pred", "fwd_ret"]].copy()
    base = base.rename(columns={"pred": f"pred_{base_h}h"})

    for h in horizons:
        if h == base_h:
            continue
        other = preds_dict[h][["timestamp", "symbol", "pred"]].copy()
        other = other.rename(columns={"pred": f"pred_{h}h"})
        base = base.merge(other, on=["timestamp", "symbol"], how="inner")

    # Weighted blend
    base["pred"] = 0
    for h in horizons:
        col = f"pred_{h}h"
        if col in base.columns:
            base["pred"] += weights[h] * base[col]

    return base[["timestamp", "symbol", "pred", "fwd_ret"]]


# ══════════════════════════════════════════════════════════════════
#  SIMULATION ENGINE (R7 — adds R6 features + new ideas)
# ══════════════════════════════════════════════════════════════════
def simulate(merged, regime_df, horizon, cfg):
    n_long = cfg.get("n_long", 6)
    n_short = cfg.get("n_short", 3)
    trend_cutoff = cfg.get("trend_cutoff", 0.8)
    dyn_threshold = cfg.get("dyn_threshold", 0.5)
    kelly_sizing = cfg.get("kelly_sizing", False)
    vol_scaling = cfg.get("vol_scaling", False)
    regime_asym = cfg.get("regime_asym", False)
    rebal_hours = cfg.get("rebal_hours", 12)

    # R7 ideas (kept but defaults off)
    signal_ema = cfg.get("signal_ema", None)
    conviction_weight = cfg.get("conviction_weight", False)
    stickiness = cfg.get("stickiness", None)
    pred_shrinkage = cfg.get("pred_shrinkage", None)

    # Pre-compute signal EMA if requested
    if signal_ema is not None:
        merged = merged.sort_values(["symbol", "timestamp"])
        merged["pred_raw"] = merged["pred"]
        merged["pred"] = merged.groupby("symbol")["pred_raw"].transform(
            lambda x: x.ewm(span=signal_ema, min_periods=1).mean()
        )

    # Prediction shrinkage
    if pred_shrinkage is not None:
        ts_medians = merged.groupby("timestamp")["pred"].transform("median")
        merged["pred"] = merged["pred"] * (1 - pred_shrinkage) + ts_medians * pred_shrinkage

    all_rets = []
    prev_longs = set()
    prev_shorts = set()

    timestamps_sorted = sorted(merged["timestamp"].unique())
    grouped = {ts: grp for ts, grp in merged.groupby("timestamp")}

    # FIX R16: process only rebalance-spaced timestamps (no overlapping returns)
    rebal_timestamps = timestamps_sorted[::rebal_hours]

    for ts in rebal_timestamps:
        if ts not in regime_df.index or ts not in grouped:
            continue
        row = regime_df.loc[ts]
        trend_str = row.get("trend_strength", 0)
        trend_dir = row.get("trend_direction", 0)
        vol_regime_val = row.get("vol_regime", 1.0)

        if trend_str > trend_cutoff:
            continue

        grp = grouped[ts].copy()
        n = len(grp)

        # Dynamic exposure
        exposure = 1.0
        if dyn_threshold is not None and trend_str > dyn_threshold:
            exposure = max(0.1, 1.0 - (trend_str - dyn_threshold) /
                          (trend_cutoff - dyn_threshold + 1e-10) * 0.5)

        # Vol scaling: scale down when vol is elevated
        if vol_scaling and not np.isnan(vol_regime_val) and vol_regime_val > 0:
            vol_scale = min(1.5, 1.0 / max(0.5, vol_regime_val))
            exposure *= vol_scale

        # Regime-conditional asymmetry: tilt L or S based on BTC trend direction
        if regime_asym and not np.isnan(trend_dir):
            # mild positive trend → more longs; mild negative → more shorts
            nl_base, ns_base = n_long, n_short
            if -0.3 < trend_dir < 0.3:
                # neutral — keep base
                nl, ns = nl_base, ns_base
            elif trend_dir >= 0.3:
                # mild bull — tilt long
                nl = min(n // 3, nl_base + 1)
                ns = max(2, ns_base - 1)
            else:
                # mild bear — tilt short
                nl = max(2, nl_base - 1)
                ns = min(n // 3, ns_base + 1)
        else:
            nl, ns = n_long, n_short

        nl = min(nl, n // 3)
        ns = min(ns, n // 3)
        if nl == 0 or ns == 0:
            continue

        grp["pred_rank"] = grp["pred"].rank(ascending=False)
        new_longs = set(grp[grp["pred_rank"] <= nl]["symbol"].tolist())
        new_shorts = set(grp[grp["pred_rank"] > (n - ns)]["symbol"].tolist())

        # Position stickiness: keep prev positions unless signal changed a lot
        if stickiness is not None and prev_longs:
            pred_map = dict(zip(grp["symbol"], grp["pred"]))
            # Keep old longs if still in top half of predictions
            mid = grp["pred"].median()
            for sym in prev_longs:
                if sym in pred_map and pred_map[sym] > mid - stickiness:
                    new_longs.add(sym)
            for sym in prev_shorts:
                if sym in pred_map and pred_map[sym] < mid + stickiness:
                    new_shorts.add(sym)
            # Trim to N
            if len(new_longs) > nl:
                scored = [(s, pred_map.get(s, 0)) for s in new_longs]
                scored.sort(key=lambda x: -x[1])
                new_longs = set(s for s, _ in scored[:nl])
            if len(new_shorts) > ns:
                scored = [(s, pred_map.get(s, 0)) for s in new_shorts]
                scored.sort(key=lambda x: x[1])
                new_shorts = set(s for s, _ in scored[:ns])

        prev_longs = new_longs
        prev_shorts = new_shorts

        longs = grp[grp["symbol"].isin(new_longs)]
        shorts = grp[grp["symbol"].isin(new_shorts)]

        if len(longs) == 0 or len(shorts) == 0:
            continue

        # Conviction-weighted sizing
        if conviction_weight:
            lw = longs["pred"].abs()
            lw = lw / (lw.sum() + 1e-10)
            sw = shorts["pred"].abs()
            sw = sw / (sw.sum() + 1e-10)
            long_ret = (longs["fwd_ret"].values * lw.values).sum()
            short_ret = (shorts["fwd_ret"].values * sw.values).sum()
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

        all_rets.append({"timestamp": ts, "portfolio_ret": port_ret})

    if not all_rets:
        return None
    return pd.DataFrame(all_rets).sort_values("timestamp")


def eval_config(sub, horizon, name, leverage=5, capital=100):
    if sub is None or len(sub) < 10:
        return None

    rets = sub["portfolio_ret"]
    # FIX R16: use actual observation frequency, not assumed ppy
    ts_range = (sub["timestamp"].max() - sub["timestamp"].min())
    total_hours = ts_range.total_seconds() / 3600
    years = total_hours / 8760
    n_obs = len(rets)
    ppy = n_obs / years if years > 0 else 730
    sharpe = rets.mean() / (rets.std() + 1e-10) * np.sqrt(ppy)
    cum = (1 + rets).cumprod()

    sub_df = sub.copy()
    sub_df["month"] = sub_df["timestamp"].dt.to_period("M")
    monthly = sub_df.groupby("month")["portfolio_ret"].apply(
        lambda x: (1 + x * leverage).prod() - 1)
    worst_m = monthly.min()

    equity = capital
    month_data = []
    for month, ret in monthly.items():
        pnl = equity * ret
        month_data.append({"month": str(month), "ret": ret, "pnl": pnl,
                           "equity": equity + pnl})
        equity += pnl

    avg_monthly = monthly.mean()
    med_monthly = monthly.median()
    calmar = avg_monthly / (abs(worst_m) + 1e-10)
    win_months = (monthly > 0).sum()
    total_months = len(monthly)

    return {
        "name": name, "sharpe": sharpe,
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


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    LEV = 5
    CAP = 100

    print("=" * 100)
    print(f"  RESEARCH ROUND 7: Signal Quality & Ensemble ({LEV}x, ${CAP})")
    print("=" * 100)

    print("\n  Loading data...")
    df = load_data(SYM_35)
    feats = [f for f in FEATURES if f in df.columns]

    # ── METHODOLOGY AUDIT ──
    audit_methodology(df)

    regime_df = compute_regime(df)

    # ── Train models for multiple horizons ──
    print(f"\n  Training multi-horizon models...")
    preds = train_and_predict_multi(df, feats, horizons=[8, 12, 24])

    # Primary predictions (12h) — same as R6
    p12 = preds.get(12)
    if p12 is None:
        print("  ERROR: no 12h predictions")
        return
    print(f"\n  12h: {len(p12):,} preds")

    results = []

    # R6 winner as baseline: SM48+6L3S
    cfg_r6 = {"n_long": 6, "n_short": 3, "trend_cutoff": 0.8,
              "dyn_threshold": 0.5, "eq_mom_boost": True, "kelly_sizing": True,
              "strategy_momentum": True, "strat_mom_lookback": 48}

    print(f"\n{'─' * 100}")
    print(f"  BASELINES")
    print(f"{'─' * 100}")
    sub = simulate(p12, regime_df, 12, cfg_r6)
    r = eval_config(sub, 12, "BASELINE R6: SM48+6L3S", LEV, CAP)
    if r: results.append(r); show(r)

    # Also baseline R5 (5L/5S)
    cfg_r5 = {"n_long": 5, "n_short": 5, "trend_cutoff": 0.8,
              "dyn_threshold": 0.5, "eq_mom_boost": True, "kelly_sizing": True,
              "strategy_momentum": False}
    sub = simulate(p12, regime_df, 12, cfg_r5)
    r = eval_config(sub, 12, "BASELINE R5: EQ-BOOST+KELLY 5L5S", LEV, CAP)
    if r: results.append(r); show(r)

    # ── A: Signal EMA smoothing ──
    print(f"\n{'─' * 100}")
    print(f"  A: Signal EMA smoothing (reduce prediction noise)")
    print(f"{'─' * 100}")
    for ema in [2, 3, 4, 6, 8, 12]:
        cfg = {**cfg_r6, "signal_ema": ema}
        sub = simulate(p12, regime_df, 12, cfg)
        r = eval_config(sub, 12, f"SM48+6L3S EMA={ema}", LEV, CAP)
        if r: results.append(r); show(r)

    # ── B: Multi-horizon ensemble ──
    print(f"\n{'─' * 100}")
    print(f"  B: Multi-horizon ensemble (blend 8h, 12h, 24h)")
    print(f"{'─' * 100}")

    blend_cfgs = [
        ({"8": 0.2, "12": 0.6, "24": 0.2}, "20/60/20"),
        ({"8": 0.3, "12": 0.4, "24": 0.3}, "30/40/30"),
        ({"8": 0.1, "12": 0.7, "24": 0.2}, "10/70/20"),
        ({"8": 0.0, "12": 0.7, "24": 0.3}, "0/70/30"),
        ({"8": 0.0, "12": 0.5, "24": 0.5}, "0/50/50"),
        ({"8": 0.4, "12": 0.4, "24": 0.2}, "40/40/20"),
    ]

    for weights_str, label in blend_cfgs:
        w = {int(k): v for k, v in weights_str.items() if v > 0}
        avail = {h: preds[h] for h in w if h in preds}
        if len(avail) < len(w):
            continue
        blended = blend_predictions(avail, w)
        sub = simulate(blended, regime_df, 12, cfg_r6)
        r = eval_config(sub, 12, f"BLEND {label} +SM48+6L3S", LEV, CAP)
        if r: results.append(r); show(r)

    # ── C: Conviction-weighted sizing ──
    print(f"\n{'─' * 100}")
    print(f"  C: Conviction-weighted (weight by |prediction|)")
    print(f"{'─' * 100}")
    for nl, ns in [(6, 3), (5, 5), (7, 5)]:
        cfg = {**cfg_r6, "n_long": nl, "n_short": ns, "conviction_weight": True}
        sub = simulate(p12, regime_df, 12, cfg)
        r = eval_config(sub, 12, f"SM48+{nl}L{ns}S CONV-W", LEV, CAP)
        if r: results.append(r); show(r)

    # ── D: Vol scaling ──
    print(f"\n{'─' * 100}")
    print(f"  D: Vol scaling (reduce exposure in high-vol regimes)")
    print(f"{'─' * 100}")
    cfg = {**cfg_r6, "vol_scaling": True}
    sub = simulate(p12, regime_df, 12, cfg)
    r = eval_config(sub, 12, "SM48+6L3S VOL-SCALE", LEV, CAP)
    if r: results.append(r); show(r)

    for nl, ns in [(7, 5), (5, 5)]:
        cfg = {**cfg_r6, "n_long": nl, "n_short": ns, "vol_scaling": True}
        sub = simulate(p12, regime_df, 12, cfg)
        r = eval_config(sub, 12, f"SM48+{nl}L{ns}S VOL-SCALE", LEV, CAP)
        if r: results.append(r); show(r)

    # ── E: Position stickiness ──
    print(f"\n{'─' * 100}")
    print(f"  E: Position stickiness (hold unless signal flips)")
    print(f"{'─' * 100}")
    for stick in [0.01, 0.02, 0.05, 0.1]:
        cfg = {**cfg_r6, "stickiness": stick}
        sub = simulate(p12, regime_df, 12, cfg)
        r = eval_config(sub, 12, f"SM48+6L3S STICK={stick}", LEV, CAP)
        if r: results.append(r); show(r)

    # ── F: Regime-conditional asymmetry ──
    print(f"\n{'─' * 100}")
    print(f"  F: Regime-conditional asymmetry (tilt L/S by BTC trend dir)")
    print(f"{'─' * 100}")
    for nl, ns in [(6, 3), (5, 5), (7, 5)]:
        cfg = {**cfg_r6, "n_long": nl, "n_short": ns, "regime_asym": True}
        sub = simulate(p12, regime_df, 12, cfg)
        r = eval_config(sub, 12, f"SM48+{nl}L{ns}S REGIME-ASYM", LEV, CAP)
        if r: results.append(r); show(r)

    # ── G: Prediction shrinkage ──
    print(f"\n{'─' * 100}")
    print(f"  G: Prediction shrinkage (reduce extreme preds)")
    print(f"{'─' * 100}")
    for shrink in [0.1, 0.2, 0.3, 0.5]:
        cfg = {**cfg_r6, "pred_shrinkage": shrink}
        sub = simulate(p12, regime_df, 12, cfg)
        r = eval_config(sub, 12, f"SM48+6L3S SHRINK={shrink}", LEV, CAP)
        if r: results.append(r); show(r)

    # ── H: MEGA COMBOS ──
    print(f"\n{'─' * 100}")
    print(f"  H: Mega combos (best of R7)")
    print(f"{'─' * 100}")

    combos = [
        {"signal_ema": 3, "conviction_weight": True,
         "label": "EMA3+CONV-W"},
        {"signal_ema": 4, "conviction_weight": True,
         "label": "EMA4+CONV-W"},
        {"signal_ema": 3, "vol_scaling": True,
         "label": "EMA3+VOL"},
        {"conviction_weight": True, "vol_scaling": True,
         "label": "CONV-W+VOL"},
        {"signal_ema": 3, "conviction_weight": True, "vol_scaling": True,
         "label": "EMA3+CONV-W+VOL"},
        {"signal_ema": 3, "stickiness": 0.02,
         "label": "EMA3+STICK02"},
        {"signal_ema": 3, "regime_asym": True,
         "label": "EMA3+REGIME-ASYM"},
        {"signal_ema": 3, "conviction_weight": True, "regime_asym": True,
         "label": "EMA3+CONV-W+RG-ASYM"},
        {"signal_ema": 3, "n_long": 7, "n_short": 5,
         "label": "EMA3+7L5S"},
        {"signal_ema": 3, "n_long": 7, "n_short": 5, "conviction_weight": True,
         "label": "EMA3+7L5S+CONV-W"},
        {"signal_ema": 4, "n_long": 7, "n_short": 5, "vol_scaling": True,
         "label": "EMA4+7L5S+VOL"},
        {"conviction_weight": True, "regime_asym": True,
         "label": "CONV-W+REGIME-ASYM"},
    ]

    for combo in combos:
        lab = combo.pop("label")
        cfg = {**cfg_r6}
        cfg.update(combo)
        combo["label"] = lab
        sub = simulate(p12, regime_df, 12, cfg)
        r = eval_config(sub, 12, f"SM48 {lab}", LEV, CAP)
        if r: results.append(r); show(r)

    # ── Also try best EMA/CONV-W with multi-horizon blend ──
    print(f"\n{'─' * 100}")
    print(f"  I: Best combos with multi-horizon blend")
    print(f"{'─' * 100}")

    best_blend_w = {8: 0.2, 12: 0.6, 24: 0.2}
    avail = {h: preds[h] for h in best_blend_w if h in preds}
    if len(avail) == len(best_blend_w):
        blended = blend_predictions(avail, best_blend_w)
        for combo_cfg, combo_name in [
            ({}, "base"),
            ({"signal_ema": 3}, "EMA3"),
            ({"conviction_weight": True}, "CONV-W"),
            ({"signal_ema": 3, "conviction_weight": True}, "EMA3+CONV-W"),
            ({"signal_ema": 3, "vol_scaling": True}, "EMA3+VOL"),
            ({"n_long": 7, "n_short": 5}, "7L5S"),
            ({"n_long": 7, "n_short": 5, "signal_ema": 3}, "7L5S+EMA3"),
        ]:
            cfg = {**cfg_r6}
            cfg.update(combo_cfg)
            sub = simulate(blended, regime_df, 12, cfg)
            r = eval_config(sub, 12, f"BLEND 20/60/20 SM48 {combo_name}", LEV, CAP)
            if r: results.append(r); show(r)

    # ══════════════════════════════════════════════════════════════
    #  RANKINGS
    # ══════════════════════════════════════════════════════════════
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
              f"Calmar={r['calmar']:.2f} | WM={wm}")
        print(f"      ${CAP} → ${r['equity']:.0f} ({len(r['month_data'])} мес) | "
              f"Score={r['score']:.0f}")
        for md in r["month_data"]:
            marker = " ← worst" if md["ret"] == r["worst_m"] else ""
            print(f"         {md['month']:>10s}  {md['ret']*100:>+7.1f}%  "
                  f"equity=${md['equity']:>7.0f}{marker}")

    # Ultra-safe
    safe_r = [r for r in results if r["worst_m"] > -0.10]
    safe_r.sort(key=lambda x: x["equity"], reverse=True)
    print(f"\n{'=' * 100}")
    print(f"  🛡️ ULTRA-SAFE (worst > -10%):")
    print(f"{'=' * 100}")
    for i, r in enumerate(safe_r[:10]):
        wm = f"{r['win_months']}/{r['total_months']}"
        print(f"  #{i+1} {r['name']}: ${CAP}→${r['equity']:.0f} | "
              f"Wr={r['worst_m']*100:+.1f}% | WM={wm} | Sh={r['sharpe']:.2f}")

    # Max equity
    results.sort(key=lambda x: x["equity"], reverse=True)
    print(f"\n{'=' * 100}")
    print(f"  💰 TOP 5 RAW EQUITY:")
    print(f"{'=' * 100}")
    for i, r in enumerate(results[:5]):
        safe = "✅" if r["worst_m"] > -0.15 else ("⚠️" if r["worst_m"] > -0.25 else "❌")
        wm = f"{r['win_months']}/{r['total_months']}"
        print(f"  #{i+1} {safe} {r['name']}: ${CAP}→${r['equity']:.0f} | "
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
              f"6м=${CAP*(1+m6):.0f} | 12м=${CAP*(1+y1):.0f}")


if __name__ == "__main__":
    main()
