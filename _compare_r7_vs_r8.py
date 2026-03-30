#!/usr/bin/env python3
"""
Direct apples-to-apples comparison: R7 (14 feats) vs R8 TOP-3 (17 feats).

IDENTICAL conditions:
  - Same WINDOWS (R7 windows: Oct24-Jan25, May-Aug25, Nov25-Mar26)
  - Same eval pipeline (R7 eval_config with leverage=5, capital=$100)
  - Same R7 winner cfg (regime-asym, vol-scale, EMA=2, 6L/3S, etc.)
  - Same 35 symbols, same Ridge HPO
  - Only difference: feature set (14 vs 17)
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

from _research_round7 import (
    SYM_35, WINDOWS, FEATURES as FEATURES_14, cs_rank,
    compute_regime, simulate, eval_config, show,
    train_and_predict_multi, load_data,
)
from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal

# R8 TOP-3 additions
NEW_FEATURES = ["range_24h", "btc_beta_168h", "global_ls_ratio_zscore"]
FEATURES_17 = FEATURES_14 + NEW_FEATURES


def load_data_full(symbols):
    """Load data — build_features_minimal already includes all 3 new features."""
    ohlcv = load_ohlcv()
    ohlcv = ohlcv[ohlcv["symbol"].isin(symbols)]
    derivs = load_derivatives()
    df = build_features_minimal(ohlcv, derivs)
    return df


def check_features(df, feats, label):
    missing = [f for f in feats if f not in df.columns]
    available = [f for f in feats if f in df.columns]
    print(f"  [{label}] {len(available)}/{len(feats)} features available")
    if missing:
        print(f"    ⚠️  MISSING: {missing}")
    return available


def run_comparison():
    LEV = 5
    CAP = 100

    print("=" * 80)
    print("  DIRECT COMPARISON: R7 (14 feats) vs R8 TOP-3 (17 feats)")
    print("  IDENTICAL: windows, leverage=5, capital=$100, ridge HPO, cfg")
    print("=" * 80)

    print("\n📊 Loading data...")
    df = load_data_full(SYM_35)
    print(f"   Shape: {df.shape}, symbols: {df['symbol'].nunique()}")

    # Check feature availability
    print("\n🔍 Feature check:")
    feats_14 = check_features(df, FEATURES_14, "R7 14-feat")
    feats_17 = check_features(df, FEATURES_17, "R8 17-feat")

    if len(feats_14) < 10:
        print("❌ Not enough R7 features — aborting")
        return

    regime_df = compute_regime(df)

    # R7 winner config (deployed to prod)
    cfg_r7_winner = {
        "n_long": 6, "n_short": 3,
        "trend_cutoff": 0.8, "dyn_threshold": 0.5,
        "eq_mom_boost": True, "kelly_sizing": True,
        "strategy_momentum": True, "strat_mom_lookback": 48,
        "regime_asym": True, "vol_scaling": True,
        "signal_ema": 2, "rebal_hours": 12,
    }

    print("\n🏋️  Training models...")
    print("  → R7 (14 features)...")
    preds_14 = train_and_predict_multi(df, feats_14, horizons=[12])

    print("  → R8 (17 features)...")
    preds_17 = train_and_predict_multi(df, feats_17, horizons=[12])

    if preds_14 is None or preds_17 is None:
        print("❌ Training failed")
        return

    p12_14 = preds_14[12]
    p12_17 = preds_17[12]

    print(f"\n  Predictions: R7={len(p12_14):,}, R8={len(p12_17):,}")

    # ── Simulate both ───────────────────────────────────────────
    print("\n📈 Simulating...")
    sub_14 = simulate(p12_14, regime_df, 12, cfg_r7_winner)
    sub_17 = simulate(p12_17, regime_df, 12, cfg_r7_winner)

    # ── Evaluate ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  RESULTS (leverage=5x, capital=$100, R7 windows)")
    print("=" * 80)

    r14 = eval_config(sub_14, 12, "R7  — 14 features (PRODUCTION)", LEV, CAP)
    r17 = eval_config(sub_17, 12, "R8  — 17 features (TOP-3 from IC scan)", LEV, CAP)
    show(r14)
    show(r17)

    # ── Monthly breakdown ───────────────────────────────────────
    if r14 and r17:
        print("\n── Monthly breakdown ──")
        print(f"{'Month':<10}  {'R7 (14f)':<12}  {'R8 (17f)':<12}  {'Δ':<8}")
        print("-" * 45)

        r14_m = {md["month"]: md["ret"] for md in r14["month_data"]}
        r17_m = {md["month"]: md["ret"] for md in r17["month_data"]}
        all_months = sorted(set(r14_m) | set(r17_m))

        r8_wins = 0
        r7_wins = 0
        for m in all_months:
            v14 = r14_m.get(m, float("nan"))
            v17 = r17_m.get(m, float("nan"))
            delta = v17 - v14 if not np.isnan(v17) and not np.isnan(v14) else float("nan")
            marker = "✅ R8" if delta > 0 else ("⚠️  R7" if delta < 0 else "=")
            if delta > 0:
                r8_wins += 1
            elif delta < 0:
                r7_wins += 1
            print(f"{m:<10}  {v14*100:>+8.1f}%    {v17*100:>+8.1f}%    {delta*100:>+6.1f}%  {marker}")

        print("-" * 45)
        print(f"R8 better in {r8_wins}/{len(all_months)} months, R7 better in {r7_wins}/{len(all_months)}")

        # ── Summary verdict ─────────────────────────────────────
        print("\n" + "=" * 80)
        print("  VERDICT")
        print("=" * 80)

        metrics = {
            "Equity ($)": (r14["equity"], r17["equity"], True),
            "Sharpe": (r14["sharpe"], r17["sharpe"], True),
            "Worst month (%)": (r14["worst_m"] * 100, r17["worst_m"] * 100, True),
            "Calmar": (r14["calmar"], r17["calmar"], True),
            "Win months": (r14["win_months"], r17["win_months"], True),
        }

        r8_score = 0
        for metric, (v14, v17, higher_better) in metrics.items():
            better = v17 > v14 if higher_better else v17 < v14
            winner = "R8 ✅" if better else "R7 ✅"
            if better:
                r8_score += 1
            delta = v17 - v14
            print(f"  {metric:<22}  R7={v14:>8.2f}  R8={v17:>8.2f}  Δ={delta:>+8.2f}  {winner}")

        print()
        if r8_score >= 4:
            print(f"  🏆 WINNER: R8 (17 features) wins {r8_score}/5 metrics")
            print(f"  → RECOMMEND: deploy 17-feature model")
        elif r8_score <= 1:
            print(f"  🏆 WINNER: R7 (14 features) wins {5 - r8_score}/5 metrics")
            print(f"  → RECOMMEND: stay with production model")
        else:
            print(f"  ⚖️  MIXED: R8 wins {r8_score}/5, R7 wins {5 - r8_score}/5")
            print(f"  → RECOMMEND: evaluate risk preference (R8 Sharpe vs R7 equity)")


if __name__ == "__main__":
    run_comparison()
