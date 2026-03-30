#!/usr/bin/env python3
"""
Research Round 8: New Feature Discovery + Feature Expansion.

Premise: Current model uses 14 features (all from IC scanner's ~50 candidates).
There are untapped data sources and feature ideas that could improve IC.

APPROACH:
  Phase 1 — IC scan on ALL candidate features (50+ from scanner + new ones)
  Phase 2 — Build new features from untapped data (DVOL, macro, funding combos)
  Phase 3 — IC scan the new features
  Phase 4 — Train Ridge with expanded feature set, backtest with R7 winner config

NEW FEATURE IDEAS:
  A) DVOL features — BTC/ETH implied vol, term structure, vol regime
  B) Macro features — VIX, DXY, SPX, Gold, Yields (daily, interpolated to 1h)
  C) Funding carry — cumulative funding as carry signal
  D) Funding × momentum interaction — crowded longs paying funding in an uptrend
  E) Volume momentum — volume trend as leading indicator
  F) OI-price divergence — OI up + price down = short buildup
  G) Premium/basis — futures basis as sentiment indicator
  H) Cross-coin momentum dispersion — market breadth signal
  I) Multi-timeframe momentum — 168h ret not in current 14
  J) Relative strength vs BTC — already have residuals, but not relative strength rank
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

PROJECT = Path(__file__).parent
DATA_DIR = PROJECT / "data"

# Import from existing modules
from _ic_scanner import load_ohlcv, load_derivatives, compute_ic, compute_ic_by_period
from _research_round7 import (
    SYM_35, WINDOWS, FEATURES as FEATURES_14, cs_rank,
    compute_regime, simulate
)

HORIZONS_IC = [4, 12, 24, 48]  # horizons for IC scan


# ══════════════════════════════════════════════════════════════════
#  EXTENDED FEATURE BUILDER
# ══════════════════════════════════════════════════════════════════
def build_features_extended(ohlcv, derivs):
    """
    Build ALL candidate features: original scanner features + new ideas.
    Returns df with ~80+ features + forward returns.
    """
    from _ic_scanner import build_features_minimal
    df = build_features_minimal(ohlcv, derivs)

    print("  🔧 Building extended features...")

    # ── A) DVOL features (BTC/ETH implied vol from Deribit) ────
    try:
        dvol = derivs["dvol"].copy()
        dvol = dvol.sort_values("timestamp")

        # BTC DVOL
        btc_dvol = dvol[dvol["currency"] == "BTC"][
            ["timestamp", "dvol_close"]
        ].rename(columns={"dvol_close": "btc_dvol"}).drop_duplicates("timestamp")
        btc_dvol = btc_dvol.set_index("timestamp").resample("1h").ffill().reset_index()
        df = df.merge(btc_dvol, on="timestamp", how="left")
        df["btc_dvol"] = df["btc_dvol"].ffill()

        # DVOL z-score (is implied vol elevated?)
        df["dvol_zscore"] = (
            df["btc_dvol"] - df["btc_dvol"].rolling(720, min_periods=168).mean()
        ) / (df["btc_dvol"].rolling(720, min_periods=168).std() + 1e-10)

        # DVOL change (is IV increasing or decreasing?)
        df["dvol_change_24h"] = df["btc_dvol"].pct_change(24)
        df["dvol_change_168h"] = df["btc_dvol"].pct_change(168)

        # DVOL vs realized vol spread (vol risk premium)
        # Use BTC realized vol as reference
        btc_mask = df["symbol"] == "BTC/USDT"
        btc_rvol = df.loc[btc_mask, ["timestamp", "rvol_24h"]].rename(
            columns={"rvol_24h": "btc_rvol_24h"}
        ).drop_duplicates("timestamp")
        df = df.merge(btc_rvol, on="timestamp", how="left")
        df["btc_rvol_24h"] = df["btc_rvol_24h"].ffill()
        # Convert DVOL from annualized % to hourly: dvol/100/sqrt(8760)
        df["dvol_rv_spread"] = df["btc_dvol"] / 100 / np.sqrt(8760) - df["btc_rvol_24h"].fillna(0)

        n_dvol = df["btc_dvol"].notna().sum()
        print(f"    ✅ DVOL features: 4 features, {n_dvol:,} non-null rows")
    except Exception as e:
        print(f"    ⚠️  DVOL features failed: {e}")
        for col in ["btc_dvol", "dvol_zscore", "dvol_change_24h", "dvol_change_168h", "dvol_rv_spread"]:
            if col not in df.columns:
                df[col] = np.nan

    # ── B) Macro features (VIX, DXY, SPX, Gold, Yields) ───────
    try:
        macro_path = DATA_DIR / "sentiment" / "macro_daily.parquet"
        if macro_path.exists():
            macro = pd.read_parquet(macro_path)
            macro["timestamp"] = pd.to_datetime(macro["date"], utc=True)
            macro = macro.sort_values("timestamp")

            # Select key columns
            macro_cols = {}
            for prefix, col_name in [("vix", "vix_close"), ("dxy", "dxy_close"),
                                      ("spx", "spx_close"), ("gold", "gold_close"),
                                      ("yield10y", "yield_10y_close")]:
                if col_name in macro.columns:
                    macro_cols[prefix] = col_name

            macro_hourly = macro[["timestamp"] + list(macro_cols.values())].drop_duplicates("timestamp")
            macro_hourly = macro_hourly.set_index("timestamp").resample("1h").ffill().reset_index()
            df = df.merge(macro_hourly, on="timestamp", how="left")

            for prefix, col in macro_cols.items():
                df[col] = df[col].ffill()
                # Returns
                df[f"{prefix}_ret_24h"] = df[col].pct_change(24)
                df[f"{prefix}_ret_168h"] = df[col].pct_change(168)
                # Z-score
                mean = df[col].rolling(720, min_periods=168).mean()
                std = df[col].rolling(720, min_periods=168).std() + 1e-10
                df[f"{prefix}_zscore"] = (df[col] - mean) / std

            print(f"    ✅ Macro features: {len(macro_cols)*3} features ({list(macro_cols.keys())})")
        else:
            print(f"    ⚠️  Macro data not found")
    except Exception as e:
        print(f"    ⚠️  Macro features failed: {e}")

    # ── C) Extended funding features ───────────────────────────
    # cum_funding already in scanner, add interaction and z-score features
    # Funding carry: cumulative funding is a carry signal
    # High funding = shorts will get squeezed OR longs are overleveraged
    for h in [24, 72]:
        col = f"cum_funding_{h}h"
        if col in df.columns:
            # Cross-sectional rank of cumulative funding
            df[f"{col}_cs"] = df.groupby("timestamp")[col].rank(pct=True) - 0.5

    # ── D) Volume momentum (volume trend as leading indicator) ─
    for h in [12, 24]:
        col = f"vol_ratio_{h}h"
        if col in df.columns:
            # Volume momentum z-score
            mean = df.groupby("symbol")[col].transform(
                lambda x: x.rolling(168, min_periods=84).mean()
            )
            std = df.groupby("symbol")[col].transform(
                lambda x: x.rolling(168, min_periods=84).std()
            ) + 1e-10
            df[f"vol_mom_z_{h}h"] = (df[col] - mean) / std

    # ── E) Cross-coin momentum dispersion (market breadth) ─────
    # Higher dispersion = more differentiation = better for L/S
    ret_12h_median = df.groupby("timestamp")["ret_12h"].transform("median")
    ret_12h_std = df.groupby("timestamp")["ret_12h"].transform("std")
    df["ret_dispersion_12h"] = ret_12h_std
    df["ret_vs_median_12h"] = df["ret_12h"] - ret_12h_median

    # ── F) Multi-timeframe momentum (168h not in current 14) ───
    # Already computed by scanner as ret_168h, add z-score version
    if "ret_168h" in df.columns:
        mean = df.groupby("symbol")["ret_168h"].transform(
            lambda x: x.rolling(720, min_periods=168).mean()
        )
        std = df.groupby("symbol")["ret_168h"].transform(
            lambda x: x.rolling(720, min_periods=168).std()
        ) + 1e-10
        df["mom_z_168h"] = (df["ret_168h"] - mean) / std

    # ── G) Relative strength index (pure rank-based RS) ────────
    # How does the coin rank in recent returns vs all coins?
    df["rs_rank_12h"] = df.groupby("timestamp")["ret_12h"].rank(pct=True)
    df["rs_rank_24h"] = df.groupby("timestamp")["ret_24h"].rank(pct=True)

    # Lagged rank (was this coin strong/weak previously too?)
    df["rs_rank_12h_lag12"] = df.groupby("symbol")["rs_rank_12h"].shift(12)
    df["rs_rank_change_12h"] = df["rs_rank_12h"] - df["rs_rank_12h_lag12"]

    # ── H) OI-funding interaction ──────────────────────────────
    # OI rising + funding rising = aggressive longs entering
    if "oi_chg_12h" in df.columns and "funding_rate_binance" in df.columns:
        df["oi_funding_interaction"] = df["oi_chg_12h"] * df["funding_rate_binance"] * 1e4

    # ── I) Taker flow momentum (change in taker flow) ──────────
    if "taker_cvd_12h" in df.columns:
        df["taker_flow_accel"] = df.groupby("symbol")["taker_cvd_12h"].diff(12)

    # ── J) Basis features ──────────────────────────────────────
    if "premium_index" in df.columns:
        # Basis momentum
        df["basis_mom_12h"] = df.groupby("symbol")["premium_index"].diff(12)
        # Basis-funding convergence (both signals agree?)
        if "funding_rate_binance" in df.columns:
            df["basis_funding_agree"] = np.sign(df["premium_index"]) * np.sign(df["funding_rate_binance"])

    print(f"  ✅ Extended features built. Total columns: {len(df.columns)}")
    return df


# ══════════════════════════════════════════════════════════════════
#  PHASE 1: IC SCAN ALL FEATURES
# ══════════════════════════════════════════════════════════════════
def run_ic_scan(df):
    """Compute IC for every numeric feature across walk-forward windows."""
    # Identify candidate features
    exclude_prefixes = ("fwd_ret_", "timestamp", "symbol", "close", "open",
                        "high", "low", "volume", "btc_close", "btc_rvol_24h",
                        "coin_ret", "btc_ret_1h", "btc_ret", "oi_value_usd",
                        "taker_buy_sell_ratio", "top_ls_ratio_raw",
                        "global_ls_ratio_raw", "premium_index_raw",
                        "funding_rate_raw")
    # Keep raw columns that are features
    keep_raw = {"funding_rate_binance", "taker_imbalance", "premium_index",
                "top_ls_ratio", "global_ls_ratio", "btc_dvol"}

    candidates = []
    for col in df.columns:
        if col in keep_raw:
            candidates.append(col)
            continue
        if any(col.startswith(p) for p in exclude_prefixes):
            continue
        if col in ("ret_1h_sq",):
            continue
        if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            candidates.append(col)

    # Remove columns that are mostly NaN
    candidates = [c for c in candidates if df[c].notna().mean() > 0.3]

    print(f"\n{'='*70}")
    print(f"  IC SCAN: {len(candidates)} candidate features × {len(HORIZONS_IC)} horizons")
    print(f"{'='*70}")

    results = []
    for feat in sorted(candidates):
        for h in HORIZONS_IC:
            target = f"fwd_ret_{h}h"
            if target not in df.columns:
                continue

            ics = []
            for w in WINDOWS:
                test = df[(df["timestamp"] >= w["test_start"]) &
                         (df["timestamp"] <= w["test_end"])]
                if len(test) < 200:
                    continue

                # Cross-sectional IC per timestamp, then mean
                ts_ics = []
                for ts, grp in test.groupby("timestamp"):
                    f_vals = grp[feat].values
                    t_vals = grp[target].values
                    mask = ~np.isnan(f_vals) & ~np.isnan(t_vals)
                    if mask.sum() >= 10:
                        ic, _ = stats.spearmanr(f_vals[mask], t_vals[mask])
                        if not np.isnan(ic):
                            ts_ics.append(ic)

                if len(ts_ics) >= 20:
                    ics.append(np.mean(ts_ics))

            if len(ics) >= 2:
                mean_ic = np.mean(ics)
                std_ic = np.std(ics)
                # Consistent sign across windows?
                sign_consistent = all(x > 0 for x in ics) or all(x < 0 for x in ics)
                results.append({
                    "feature": feat,
                    "horizon": h,
                    "mean_ic": mean_ic,
                    "std_ic": std_ic,
                    "n_windows": len(ics),
                    "consistent": sign_consistent,
                    "per_window": ics,
                    "in_current_14": feat in FEATURES_14,
                })

    results_df = pd.DataFrame(results)

    # Print top features at 12h horizon
    print(f"\n{'─'*70}")
    print(f"  TOP FEATURES at 12h horizon (sorted by |IC|)")
    print(f"{'─'*70}")

    filt_12h = results_df[results_df["horizon"] == 12].copy()
    filt_12h["abs_ic"] = filt_12h["mean_ic"].abs()
    filt_12h = filt_12h.sort_values("abs_ic", ascending=False)

    print(f"  {'Feature':<30} {'IC':>8} {'Std':>8} {'Win':>4} {'Cons':>5} {'In14':>5}")
    print(f"  {'─'*66}")
    for _, row in filt_12h.head(40).iterrows():
        marker = "✅" if row["in_current_14"] else "🆕"
        cons = "✓" if row["consistent"] else " "
        print(f"  {marker} {row['feature']:<28} {row['mean_ic']:>+.4f} "
              f"{row['std_ic']:>.4f} {row['n_windows']:>4} {cons:>5} "
              f"{'Y' if row['in_current_14'] else 'N':>5}")

    # Features with |IC| > 0.02 at 12h that are NOT in current 14
    new_candidates = filt_12h[
        (filt_12h["abs_ic"] > 0.015) & (~filt_12h["in_current_14"])
    ].copy()

    print(f"\n  🆕 NEW features with |IC| > 0.015 at 12h: {len(new_candidates)}")
    for _, row in new_candidates.iterrows():
        ws = ", ".join(f"{x:+.3f}" for x in row["per_window"])
        print(f"     {row['feature']:<30} IC={row['mean_ic']:+.4f} [{ws}]")

    # Multi-horizon: features good at multiple horizons
    print(f"\n{'─'*70}")
    print(f"  MULTI-HORIZON CONSISTENCY (features with |IC|>0.015 at 2+ horizons)")
    print(f"{'─'*70}")

    multi = results_df[results_df["mean_ic"].abs() > 0.015].copy()
    multi_count = multi.groupby("feature").size().reset_index(name="n_horizons")
    multi_count = multi_count[multi_count["n_horizons"] >= 2].sort_values("n_horizons", ascending=False)

    for _, row in multi_count.iterrows():
        feat = row["feature"]
        in14 = "✅" if feat in FEATURES_14 else "🆕"
        horizons = multi[multi["feature"] == feat].sort_values("horizon")
        h_str = " | ".join(f"{int(h['horizon'])}h:{h['mean_ic']:+.4f}" for _, h in horizons.iterrows())
        print(f"  {in14} {feat:<30} [{h_str}]")

    return results_df, new_candidates


# ══════════════════════════════════════════════════════════════════
#  PHASE 2: TRAIN AND BACKTEST WITH EXPANDED FEATURES
# ══════════════════════════════════════════════════════════════════
def train_and_predict(df, feats, horizon=12):
    """Train Ridge per window, return merged predictions."""
    feat_r = [f"{f}_r" for f in feats]
    fwd_col = f"fwd_ret_{horizon}h"
    all_preds = []

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

        # HPO alpha
        best_alpha, best_ic = 1000, -1
        for alpha in [10, 100, 500, 1000, 2000, 5000]:
            m = Ridge(alpha=alpha, fit_intercept=False)
            m.fit(train_c[feat_r], train_c["target_rank"])
            p = m.predict(val_c[feat_r])
            ic = stats.spearmanr(p, val_c["target_rank"])[0]
            if ic > best_ic:
                best_ic = ic
                best_alpha = alpha

        # Retrain on train + val
        full = pd.concat([train_c, val_c])
        m = Ridge(alpha=best_alpha, fit_intercept=False)
        m.fit(full[feat_r], full["target_rank"])

        # Test predictions
        test_c = test[feat_r + ["target_rank"]].dropna()
        test_preds = m.predict(test_c[feat_r])
        test_ic = stats.spearmanr(test_preds, test_c["target_rank"])[0]

        test_out = test.loc[test_c.index, ["timestamp", "symbol", fwd_col]].copy()
        test_out["pred"] = test_preds
        test_out = test_out.rename(columns={fwd_col: "fwd_ret"})
        all_preds.append(test_out)

        print(f"    {w['name']}: α={best_alpha}, val_IC={best_ic:.4f}, "
              f"test_IC={test_ic:.4f}, n={len(test_c):,}")

    if not all_preds:
        return None
    return pd.concat(all_preds, ignore_index=True)


def eval_config(preds, regime_df, cfg, label=""):
    """Run simulation and print results."""
    port = simulate(preds, regime_df, horizon=12, cfg=cfg)
    if port is None or len(port) == 0:
        print(f"  {label}: no trades")
        return None

    rets = port["portfolio_ret"]
    eq = (1 + rets).cumprod()

    port_ts = port.set_index("timestamp")
    monthly = port_ts["portfolio_ret"].groupby(port_ts.index.to_period("M")).apply(
        lambda x: (1 + x).prod() - 1
    )
    worst = monthly.min()
    sharpe = rets.mean() / (rets.std() + 1e-10) * np.sqrt(365 * 24 / 12)
    winning_months = (monthly > 0).sum()
    calmar = (eq.iloc[-1] - 1) / (abs(worst) + 1e-10) if worst < 0 else 999

    print(f"  {label:<45} "
          f"Eq=${eq.iloc[-1]*100:.0f}  Wr={worst*100:+.1f}%  "
          f"Sh={sharpe:.2f}  Cal={calmar:.1f}  WM={winning_months}/{len(monthly)}")

    return {
        "label": label,
        "equity": eq.iloc[-1] * 100,
        "worst_month": worst,
        "sharpe": sharpe,
        "calmar": calmar,
        "winning_months": f"{winning_months}/{len(monthly)}",
    }


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  RESEARCH ROUND 8: New Feature Discovery")
    print("=" * 70)

    # ── Load data ──
    print("\n📊 Loading data...")
    ohlcv = load_ohlcv()
    ohlcv = ohlcv[ohlcv["symbol"].isin(SYM_35)]
    derivs = load_derivatives()

    print("🔧 Building extended features...")
    df = build_features_extended(ohlcv, derivs)
    print(f"   Shape: {df.shape}")

    # ── Phase 1: IC scan ──
    print("\n" + "=" * 70)
    print("  PHASE 1: IC SCAN (all features)")
    print("=" * 70)
    results_df, new_candidates = run_ic_scan(df)

    # ── Phase 2: Train models with different feature sets ──
    print("\n" + "=" * 70)
    print("  PHASE 2: TRAIN & BACKTEST")
    print("=" * 70)

    regime_df = compute_regime(df)

    # R7 winner config (baseline)
    cfg_r7 = {
        "n_long": 6, "n_short": 3,
        "trend_cutoff": 0.8, "dyn_threshold": 0.5,
        "eq_mom_boost": True, "kelly_sizing": True,
        "strategy_momentum": True, "strat_mom_lookback": 48,
        "regime_asym": True, "vol_scaling": True,
        "signal_ema": 2, "rebal_hours": 12,
    }

    # A) Baseline: 14 features (R7 winner)
    print("\n── A) Baseline: 14 features ──")
    preds_14 = train_and_predict(df, FEATURES_14)
    res_baseline = None
    if preds_14 is not None:
        res_baseline = eval_config(preds_14, regime_df, cfg_r7, "BASELINE (14 feats)")

    # B) Pick new features with |IC| > 0.015 at 12h
    new_feat_names = list(new_candidates["feature"].unique()) if len(new_candidates) > 0 else []
    # Filter to features that actually exist in df
    new_feat_names = [f for f in new_feat_names if f in df.columns]

    if new_feat_names:
        # B1) 14 + all new features
        feats_expanded = FEATURES_14 + new_feat_names
        print(f"\n── B) Expanded: 14 + {len(new_feat_names)} new = {len(feats_expanded)} features ──")
        print(f"   New: {new_feat_names}")
        preds_exp = train_and_predict(df, feats_expanded)
        if preds_exp is not None:
            eval_config(preds_exp, regime_df, cfg_r7, f"EXPANDED ({len(feats_expanded)} feats)")

        # B2) Try adding features one by one (ablation)
        print(f"\n── C) Feature ablation: add one at a time ──")
        ablation_results = []
        for feat in new_feat_names:
            feats_plus1 = FEATURES_14 + [feat]
            preds_p1 = train_and_predict(df, feats_plus1)
            if preds_p1 is not None:
                res = eval_config(preds_p1, regime_df, cfg_r7, f"+{feat}")
                if res:
                    ablation_results.append(res)

        # B3) Top 3 individual features — combine them
        if ablation_results:
            ablation_results.sort(key=lambda x: x["equity"], reverse=True)
            print(f"\n── D) Best new features ranked by equity ──")
            for r in ablation_results:
                print(f"   {r['label']:<40} Eq=${r['equity']:.0f}  "
                      f"Wr={r['worst_month']*100:+.1f}%  Sh={r['sharpe']:.2f}")

            # Combine top 3
            top_feats = [r["label"].replace("+", "") for r in ablation_results[:3]]
            top_feats = [f for f in top_feats if f in df.columns]
            if top_feats:
                feats_top3 = FEATURES_14 + top_feats
                print(f"\n── E) Top-3 combined: 14 + {top_feats} ──")
                preds_top3 = train_and_predict(df, feats_top3)
                if preds_top3 is not None:
                    eval_config(preds_top3, regime_df, cfg_r7, f"TOP-3 ({len(feats_top3)} feats)")

            # Combine top 5
            top5_feats = [r["label"].replace("+", "") for r in ablation_results[:5]]
            top5_feats = [f for f in top5_feats if f in df.columns]
            if len(top5_feats) > len(top_feats):
                feats_top5 = FEATURES_14 + top5_feats
                print(f"\n── F) Top-5 combined: 14 + {top5_feats} ──")
                preds_top5 = train_and_predict(df, feats_top5)
                if preds_top5 is not None:
                    eval_config(preds_top5, regime_df, cfg_r7, f"TOP-5 ({len(feats_top5)} feats)")

    else:
        print("\n   ⚠️  No new features with IC > 0.015 found")

    # ── Phase 3: Try alternative combos of EXISTING scanner features ──
    print("\n" + "=" * 70)
    print("  PHASE 3: ALTERNATIVE FEATURE COMBOS (from scanner pool)")
    print("=" * 70)

    # Features from scanner that had decent IC but weren't in top 14
    scanner_extras = [
        "ret_168h", "ret_4h", "ret_1h",
        "rvol_12h", "rvol_24h",
        "vol_ratio_12h", "vol_ratio_24h",
        "cum_funding_24h", "cum_funding_72h", "cum_funding_168h",
        "funding_zscore", "funding_x_mom_12h", "funding_x_mom_24h",
        "oi_chg_1h", "oi_chg_4h", "oi_ret_diverge",
        "taker_imbalance", "taker_cvd_4h", "taker_zscore",
        "top_ls_ratio_zscore", "global_ls_ratio_zscore",
        "premium_index", "premium_zscore",
    ]
    scanner_extras = [f for f in scanner_extras if f in df.columns]

    # Test scanner extras one-by-one
    print(f"\n── Scanner extras ablation ({len(scanner_extras)} features) ──")
    scanner_results = []
    for feat in scanner_extras:
        feats_p1 = FEATURES_14 + [feat]
        preds_p1 = train_and_predict(df, feats_p1)
        if preds_p1 is not None:
            res = eval_config(preds_p1, regime_df, cfg_r7, f"+{feat}")
            if res:
                scanner_results.append(res)

    if scanner_results:
        scanner_results.sort(key=lambda x: x["equity"], reverse=True)
        print(f"\n── Best scanner extras by equity ──")
        for r in scanner_results:
            print(f"   {r['label']:<40} Eq=${r['equity']:.0f}  "
                  f"Wr={r['worst_month']*100:+.1f}%  Sh={r['sharpe']:.2f}")

        # Top 3 scanner extras combined
        top3_scan = [r["label"].replace("+", "") for r in scanner_results[:3]]
        top3_scan = [f for f in top3_scan if f in df.columns]
        if top3_scan:
            feats_scan3 = FEATURES_14 + top3_scan
            print(f"\n── Top-3 scanner combined: {top3_scan} ──")
            preds_scan3 = train_and_predict(df, feats_scan3)
            if preds_scan3 is not None:
                eval_config(preds_scan3, regime_df, cfg_r7, f"SCAN-TOP3 ({len(feats_scan3)} feats)")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  R8 SUMMARY")
    print("=" * 70)
    if res_baseline:
        print(f"  Baseline (14 feats): Eq=${res_baseline['equity']:.0f}, "
              f"Wr={res_baseline['worst_month']*100:+.1f}%, "
              f"Sh={res_baseline['sharpe']:.2f}")
    print(f"\n  Check results above for improvements over baseline.")
    print(f"  If new features improve IC consistently, consider retraining production model.")


if __name__ == "__main__":
    main()
