#!/usr/bin/env python3
"""
Train Ridge mean-reversion model for production.
Saves model coefficients to results_ridge_prod/model.json.

Uses walk-forward: train on all data up to val_cutoff, validate on rest,
then retrain on ALL data for final production model.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
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

TOP_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
    "MATIC/USDT", "UNI/USDT", "ATOM/USDT", "LTC/USDT", "NEAR/USDT",
    "FIL/USDT", "APT/USDT", "ARB/USDT", "OP/USDT", "AAVE/USDT",
]

HORIZON = 12


def cs_rank(df, col):
    return df.groupby("timestamp")[col].rank(pct=True) - 0.5


def main():
    from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal

    print("=" * 60)
    print("  TRAIN RIDGE PRODUCTION MODEL")
    print("=" * 60)

    print("\n📊 Loading data...")
    ohlcv = load_ohlcv()
    ohlcv = ohlcv[ohlcv["symbol"].isin(TOP_SYMBOLS)]
    derivs = load_derivatives()

    print("🔧 Building features...")
    df = build_features_minimal(ohlcv, derivs)

    feats = [f for f in FEATURES if f in df.columns]
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"  ⚠️  Missing features: {missing}")
    print(f"  {len(feats)}/{len(FEATURES)} features | "
          f"{df['symbol'].nunique()} symbols | {df.shape[0]:,} rows")

    fwd_col = f"fwd_ret_{HORIZON}h"

    # Split: train on older data, validate on recent
    val_cutoff = "2025-10-31"
    train = df[df["timestamp"] < val_cutoff].copy()
    val = df[df["timestamp"] >= val_cutoff].copy()
    print(f"  Train: {len(train):,} rows (→ {val_cutoff})")
    print(f"  Val:   {len(val):,} rows ({val_cutoff} →)")

    # CS-rank features
    feat_r = [f"{f}_r" for f in feats]
    for d in [train, val]:
        for feat in feats:
            d[f"{feat}_r"] = cs_rank(d, feat)
        d["target_rank"] = d.groupby("timestamp")[fwd_col].rank(pct=True) - 0.5

    train_c = train[feat_r + ["target_rank"]].dropna()
    val_c = val[feat_r + ["target_rank"]].dropna()

    # HPO: find best alpha
    print("\n🔍 Alpha search:")
    best_alpha, best_ic = 1.0, -999
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        m = Ridge(alpha=alpha)
        m.fit(train_c[feat_r], train_c["target_rank"])
        preds = m.predict(val_c[feat_r])
        ic = stats.spearmanr(preds, val_c["target_rank"])[0]
        marker = " ← best" if ic > best_ic else ""
        print(f"    α={alpha:>8.2f}  val_IC={ic:.4f}{marker}")
        if ic > best_ic:
            best_ic = ic
            best_alpha = alpha

    # Final model on ALL data
    print(f"\n✅ Best alpha: {best_alpha} (val IC: {best_ic:.4f})")
    print("🏋️  Training final model on ALL data...")

    all_data = df.copy()
    for feat in feats:
        all_data[f"{feat}_r"] = cs_rank(all_data, feat)
    all_data["target_rank"] = all_data.groupby("timestamp")[fwd_col].rank(pct=True) - 0.5
    all_c = all_data[feat_r + ["target_rank"]].dropna()

    m = Ridge(alpha=best_alpha)
    m.fit(all_c[feat_r], all_c["target_rank"])

    # Show weights
    print("\n📊 Model weights:")
    for feat, c in sorted(zip(feats, m.coef_), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {feat:>25s}  {c:+.4f}")
    print(f"    {'intercept':>25s}  {m.intercept_:+.4f}")

    # Verify on val set
    val_preds = m.predict(val_c[feat_r])
    final_ic = stats.spearmanr(val_preds, val_c["target_rank"])[0]
    print(f"\n  Final model val IC: {final_ic:.4f}")

    # Save
    out_dir = PROJECT / "results_ridge_prod"
    out_dir.mkdir(exist_ok=True)

    model_data = {
        "coef": m.coef_.tolist(),
        "intercept": float(m.intercept_),
        "alpha": best_alpha,
        "features": feats,
        "horizon": HORIZON,
        "train_rows": len(all_c),
        "val_ic": round(best_ic, 4),
        "final_ic": round(final_ic, 4),
        "train_end": str(df["timestamp"].max().date()),
    }

    model_path = out_dir / "model.json"
    with open(model_path, "w") as f:
        json.dump(model_data, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  ✅ Saved to {model_path}")
    print(f"  Features: {len(feats)} | Alpha: {best_alpha} | Val IC: {best_ic:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
