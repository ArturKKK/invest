#!/usr/bin/env python3
"""
Direct model quality check — NO simulation needed.
Compares predictions vs actual returns from test_predictions files.
"""
import pandas as pd
import numpy as np
import os

BASE = "/Users/a.s.tabakov/Developer/invest"

# All production model predictions
MODELS = {
    "CatBoost (prod)":  f"{BASE}/results_catboost_prod/test_predictions_catboost.parquet",
    "LGB v6 (prod)":    f"{BASE}/results_v6_prod/test_predictions_v6.parquet",
    "LGB v7 (prod)":    f"{BASE}/results_v7_prod/test_predictions_v6.parquet",
    "XGBoost (prod)":   f"{BASE}/results_xgboost_prod/test_predictions_xgboost.parquet",
    "LGB v6 Huber":     f"{BASE}/results_v6_huber_prod/test_predictions_v6.parquet",
}

# Also check DVOL (overnight v7 models) if they exist
DVOL_MODELS = {
    "CatBoost DVOL":    f"{BASE}/results_catboost_dvol_prod/test_predictions_catboost.parquet",
    "LGB v6 DVOL":      f"{BASE}/results_v6_dvol_prod/test_predictions_v6.parquet",
    "LGB v7 DVOL":      f"{BASE}/results_v7_dvol_prod/test_predictions_v6.parquet",
    "XGBoost DVOL":     f"{BASE}/results_xgboost_dvol_prod/test_predictions_xgboost.parquet",
}


def find_pred_col(df):
    """Find the prediction column name."""
    for c in df.columns:
        if c.startswith("pred_"):
            return c
    return None


def find_target_col(df):
    """Find the target column name."""
    for c in df.columns:
        if "target" in c.lower() and "ret" in c.lower():
            return c
    return None


def evaluate_model(name, path):
    """Evaluate a single model's predictive quality."""
    if not os.path.exists(path):
        return None

    df = pd.read_parquet(path)
    pred_col = find_pred_col(df)
    target_col = find_target_col(df)

    if pred_col is None or target_col is None:
        print(f"  {name}: columns={list(df.columns)} — can't find pred/target")
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    target = df[target_col]
    pred = df[pred_col]

    # IC = Pearson correlation between prediction and actual return
    ic = target.corr(pred)

    # Rank IC = Spearman correlation (more robust)
    rank_ic = target.rank().corr(pred.rank())

    # Direction accuracy
    # If pred is probability (0-1), threshold at 0.5; if raw score, threshold at 0
    if pred.min() >= 0 and pred.max() <= 1:
        pred_long = pred > 0.5
    else:
        pred_long = pred > pred.median()
    dir_acc = ((target > 0) == pred_long).mean()

    # Top/bottom decile spread (the real money metric)
    pred_rank = pred.rank(pct=True)
    top_decile_ret = target[pred_rank >= 0.9].mean()
    bot_decile_ret = target[pred_rank <= 0.1].mean()
    spread = top_decile_ret - bot_decile_ret

    # Monthly IC breakdown
    monthly = df.set_index("timestamp").resample("ME").apply(
        lambda g: g[target_col].corr(g[pred_col]) if len(g) > 10 else np.nan
    ).dropna()

    # IC stability: fraction of months with positive IC
    ic_positive_frac = (monthly > 0).mean() if len(monthly) > 0 else 0

    return {
        "name": name,
        "n_samples": len(df),
        "date_range": f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()}",
        "IC": ic,
        "Rank_IC": rank_ic,
        "Dir_Acc": dir_acc,
        "Top10_ret": top_decile_ret,
        "Bot10_ret": bot_decile_ret,
        "L/S_spread": spread,
        "IC_stability": ic_positive_frac,
        "monthly_ics": monthly,
        "pred_col": pred_col,
        "pred": pred,
        "target": target,
        "timestamp": df["timestamp"],
        "symbol": df["symbol"],
    }


def build_ensemble(results):
    """Build ensemble by averaging normalized predictions from all models."""
    # Find common timestamps+symbols
    dfs = []
    for r in results:
        tmp = pd.DataFrame({
            "timestamp": r["timestamp"],
            "symbol": r["symbol"],
            "pred": r["pred"],
            "target": r["target"],
        })
        # Normalize pred to z-score per timestamp
        tmp["pred_z"] = tmp.groupby("timestamp")["pred"].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-10)
        )
        tmp = tmp.rename(columns={"pred_z": f"pred_{r['name']}"})
        dfs.append(tmp[["timestamp", "symbol", f"pred_{r['name']}", "target"]])

    # Merge all predictions on timestamp+symbol
    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.merge(
            d.drop(columns="target"),
            on=["timestamp", "symbol"],
            how="inner"
        )

    # Ensemble = mean of z-scored predictions
    pred_cols = [c for c in merged.columns if c.startswith("pred_")]
    merged["ensemble_pred"] = merged[pred_cols].mean(axis=1)

    # Evaluate ensemble
    target = merged["target"]
    pred = merged["ensemble_pred"]

    ic = target.corr(pred)
    rank_ic = target.rank().corr(pred.rank())
    pred_long = pred > pred.median()
    dir_acc = ((target > 0) == pred_long).mean()

    pred_rank = pred.rank(pct=True)
    top_ret = target[pred_rank >= 0.9].mean()
    bot_ret = target[pred_rank <= 0.1].mean()

    merged["ts"] = merged["timestamp"]
    monthly = merged.set_index("ts").resample("ME").apply(
        lambda g: g["target"].corr(g["ensemble_pred"]) if len(g) > 10 else np.nan
    ).dropna()

    return {
        "name": "ENSEMBLE (all models)",
        "n_samples": len(merged),
        "n_models": len(pred_cols),
        "models_used": [r["name"] for r in results],
        "IC": ic,
        "Rank_IC": rank_ic,
        "Dir_Acc": dir_acc,
        "Top10_ret": top_ret,
        "Bot10_ret": bot_ret,
        "L/S_spread": top_ret - bot_ret,
        "IC_stability": (monthly > 0).mean() if len(monthly) > 0 else 0,
        "monthly_ics": monthly,
    }


# ─── Main ────────────────────────────────────────────────────────
print("=" * 80)
print("  MODEL QUALITY CHECK — Predictions vs Reality (NO simulation)")
print("=" * 80)
print()

all_results = []

# Evaluate individual models
for name, path in {**MODELS, **DVOL_MODELS}.items():
    r = evaluate_model(name, path)
    if r:
        all_results.append(r)
        print(f"✅ {name}")
        print(f"   Samples: {r['n_samples']:,}  |  Period: {r['date_range']}")
        print(f"   IC: {r['IC']:.4f}  |  Rank IC: {r['Rank_IC']:.4f}  |  Dir Acc: {r['Dir_Acc']:.1%}")
        print(f"   Top 10% avg ret: {r['Top10_ret']*100:.3f}%  |  Bot 10%: {r['Bot10_ret']*100:.3f}%")
        print(f"   L/S spread/step: {r['L/S_spread']*100:.3f}%  |  IC stability: {r['IC_stability']:.0%} months positive")
        print()

# Build and evaluate ensembles
if len(all_results) >= 2:
    print("\n" + "=" * 80)
    print("  ENSEMBLE COMBINATIONS")
    print("=" * 80)

    # All models ensemble
    ens_all = build_ensemble(all_results)
    print(f"\n🔵 {ens_all['name']} ({ens_all['n_models']} models)")
    print(f"   Models: {', '.join(ens_all['models_used'])}")
    print(f"   IC: {ens_all['IC']:.4f}  |  Rank IC: {ens_all['Rank_IC']:.4f}  |  Dir Acc: {ens_all['Dir_Acc']:.1%}")
    print(f"   L/S spread: {ens_all['L/S_spread']*100:.3f}%  |  IC stability: {ens_all['IC_stability']:.0%}")

    # Ensemble of just prod models (v6, v7, cb, xgb)
    prod_results = [r for r in all_results if "(prod)" in r["name"]]
    if len(prod_results) >= 2:
        ens_prod = build_ensemble(prod_results)
        ens_prod["name"] = "ENSEMBLE (4 prod GBDT)"
        print(f"\n🟢 {ens_prod['name']} ({len(prod_results)} models)")
        print(f"   Models: {', '.join(r['name'] for r in prod_results)}")
        print(f"   IC: {ens_prod['IC']:.4f}  |  Rank IC: {ens_prod['Rank_IC']:.4f}  |  Dir Acc: {ens_prod['Dir_Acc']:.1%}")
        print(f"   L/S spread: {ens_prod['L/S_spread']*100:.3f}%  |  IC stability: {ens_prod['IC_stability']:.0%}")

    # DVOL models ensemble
    dvol_results = [r for r in all_results if "DVOL" in r["name"]]
    if len(dvol_results) >= 2:
        ens_dvol = build_ensemble(dvol_results)
        ens_dvol["name"] = "ENSEMBLE (DVOL models)"
        print(f"\n🟡 {ens_dvol['name']} ({len(dvol_results)} models)")
        print(f"   Models: {', '.join(r['name'] for r in dvol_results)}")
        print(f"   IC: {ens_dvol['IC']:.4f}  |  Rank IC: {ens_dvol['Rank_IC']:.4f}  |  Dir Acc: {ens_dvol['Dir_Acc']:.1%}")
        print(f"   L/S spread: {ens_dvol['L/S_spread']*100:.3f}%  |  IC stability: {ens_dvol['IC_stability']:.0%}")

    # All unique ensemble combinations of size 2-3 for top models
    from itertools import combinations
    print(f"\n{'─'*80}")
    print("  COMBO SEARCH — best 2-3 model ensembles by IC")
    print(f"{'─'*80}")
    best_combos = []
    for size in [2, 3]:
        for combo in combinations(all_results, size):
            ens = build_ensemble(list(combo))
            best_combos.append((ens["IC"], ens["Rank_IC"], ens["L/S_spread"], [r["name"] for r in combo]))
    best_combos.sort(key=lambda x: x[0], reverse=True)
    print("\nTop 10 combos by IC:")
    for ic, ric, spread, names in best_combos[:10]:
        print(f"  IC={ic:.4f}  RankIC={ric:.4f}  spread={spread*100:.3f}%  ← {', '.join(names)}")


# ─── Reference thresholds ────────────────────────────────────────
print(f"\n{'='*80}")
print("  REFERENCE THRESHOLDS (quant finance)")
print(f"{'='*80}")
print("  IC < 0.02  →  noise, no signal")
print("  IC 0.02-0.05  →  weak but possibly tradeable with low costs")
print("  IC > 0.05  →  solid alpha, worth trading")
print("  IC > 0.10  →  excellent (rare)")
print()
print("  L/S spread < 0.1%/step  →  eaten by costs")
print("  L/S spread 0.1-0.3%    →  marginal, need very low turnover")
print("  L/S spread > 0.3%      →  tradeable")
print()
print("  Direction accuracy 50-51%  →  noise")
print("  Direction accuracy 52-55%  →  weak edge")
print("  Direction accuracy > 55%   →  strong")
