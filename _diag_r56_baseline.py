#!/usr/bin/env python3
"""Diagnostic: why R56 baseline=0.91 vs R48 baseline=1.66"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from _research_r35_new_features import load_research_frame, add_r35_features, MARKET_LEVEL_FEATURES
from _research_r47_coinglass import (
    CANONICAL_EXEC_CFG, CHAMPION_FEAT_30, add_cg_features, compute_cg_features,
    load_cg_daily, make_feature_set,
)
from _research_r55_cg_features import build_all_r55_features, merge_r55_into_model
from _research_round7 import WINDOWS

print("=" * 60)
print("  DIAGNOSTIC: R56 baseline regression")
print("=" * 60)

# ── 1. Load data exactly as R48 ──
print("\n[1] Loading base frame (same as R48)...")
df, regime_df = load_research_frame()
df, _ = add_r35_features(df)
print(f"  After load_research_frame + R35: {len(df):,} rows, {df['symbol'].nunique()} symbols")

cg = load_cg_daily()
cg_daily = compute_cg_features(cg)
df, per_sym_cols, mkt_cols = add_cg_features(df, cg_daily)
print(f"  After add_cg_features: {len(df):,} rows")

# ── 2. Check what make_feature_set returns ──
feats, no_rank = make_feature_set(["cg_taker_imb"], mkt_cols)
print(f"\n[2] make_feature_set results:")
print(f"  Features ({len(feats)}): {feats}")
print(f"  no_rank (cs_rank_exclude): {no_rank}")
print(f"  MARKET_LEVEL_FEATURES: {MARKET_LEVEL_FEATURES}")

# Which champion features are market-level?
champ_mkt = [f for f in CHAMPION_FEAT_30 if f in MARKET_LEVEL_FEATURES]
print(f"  Champion features that are market-level: {champ_mkt}")

# ── 3. R55 merge row count check ──
print(f"\n[3] R55 merge row count check...")
rows_before = len(df)
r55_feats = build_all_r55_features()
print(f"  R55 features shape: {r55_feats.shape}")

# Check for duplicate keys in R55
if not r55_feats.empty and "cg_date" in r55_feats.columns:
    dupes = r55_feats.groupby(["symbol", "cg_date"]).size()
    n_dupes = (dupes > 1).sum()
    print(f"  R55 duplicate (symbol, cg_date) keys: {n_dupes}")
    if n_dupes > 0:
        print(f"  ⚠️  DUPLICATE KEYS FOUND! Max count: {dupes.max()}")
        print(f"  Examples: {dupes[dupes > 1].head(5)}")

df_merged, r55_cols = merge_r55_into_model(df, r55_feats)
rows_after = len(df_merged)
print(f"  Rows before merge: {rows_before:,}")
print(f"  Rows after merge:  {rows_after:,}")
print(f"  Δrows: {rows_after - rows_before:,}")
if rows_after != rows_before:
    print(f"  ⚠️  ROW COUNT CHANGED! Ratio: {rows_after/rows_before:.4f}")
else:
    print(f"  ✓ Row count unchanged")

# ── 4. Quick baseline comparison ──
from _research_r30b_fixed import train_ensemble, eval_with_costs
from _research_r48_cost import simulate_with_hybrid_costs
from _research_r22_models import log

def eval_hybrid(preds, regime_df, cfg, label=""):
    results = {}
    for wname in ["W1", "W2", "W3"]:
        sub = preds[preds["window"] == wname]
        if len(sub) < 10:
            results[wname] = {"sharpe": 0}
            continue
        port = simulate_with_hybrid_costs(sub, regime_df, cfg)
        r = eval_with_costs(port, f"{label}_{wname}")
        results[wname] = r
        print(f"  {wname}: Sh={r['sharpe']:>5.2f} (gross={r['sharpe_gross']:>5.2f})")
    port_all = simulate_with_hybrid_costs(preds, regime_df, cfg)
    r_all = eval_with_costs(port_all, label)
    results["ALL"] = r_all
    print(f"  ALL: Sh={r_all['sharpe']:>5.2f} (gross={r_all['sharpe_gross']:>5.2f})")
    return results

print(f"\n[4] Run A: R48-exact (no R55 merge, with cs_rank_exclude)")
preds_a = train_ensemble(df, feats, WINDOWS, l2=1.0, rolling=False,
                         label="diag_r48_exact", cs_rank_exclude=no_rank)
if preds_a is not None:
    res_a = eval_hybrid(preds_a, regime_df, CANONICAL_EXEC_CFG, "diag_r48_exact")
    sharpe_a = res_a["ALL"]["sharpe"]
else:
    sharpe_a = None
    print("  FAILED")

print(f"\n[5] Run B: R56-as-run (R55 merged, NO cs_rank_exclude)")
preds_b = train_ensemble(df_merged, feats, WINDOWS, l2=1.0, rolling=False,
                         label="diag_r56_asis")
if preds_b is not None:
    res_b = eval_hybrid(preds_b, regime_df, CANONICAL_EXEC_CFG, "diag_r56_asis")
    sharpe_b = res_b["ALL"]["sharpe"]
else:
    sharpe_b = None
    print("  FAILED")

print(f"\n[6] Run C: R55 merged + WITH cs_rank_exclude (isolate R55 effect)")
preds_c = train_ensemble(df_merged, feats, WINDOWS, l2=1.0, rolling=False,
                         label="diag_r55_merge_only", cs_rank_exclude=no_rank)
if preds_c is not None:
    res_c = eval_hybrid(preds_c, regime_df, CANONICAL_EXEC_CFG, "diag_r55_merge")
    sharpe_c = res_c["ALL"]["sharpe"]
else:
    sharpe_c = None
    print("  FAILED")

print(f"\n[7] Run D: NO R55 merge, NO cs_rank_exclude (isolate cs_rank effect)")
preds_d = train_ensemble(df, feats, WINDOWS, l2=1.0, rolling=False,
                         label="diag_no_csrank_excl")
if preds_d is not None:
    res_d = eval_hybrid(preds_d, regime_df, CANONICAL_EXEC_CFG, "diag_no_csrank")
    sharpe_d = res_d["ALL"]["sharpe"]
else:
    sharpe_d = None
    print("  FAILED")

# ── Summary ──
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  A (R48-exact):        ALL Sharpe = {sharpe_a}")
print(f"  B (R56-as-run):       ALL Sharpe = {sharpe_b}")
print(f"  C (R55+cs_rank_fix):  ALL Sharpe = {sharpe_c}")
print(f"  D (no-R55+no-csrank): ALL Sharpe = {sharpe_d}")
print()
if sharpe_a and sharpe_b:
    print(f"  Total gap A-B:     {sharpe_a - sharpe_b:+.2f}")
if sharpe_a and sharpe_d:
    print(f"  cs_rank effect:    {sharpe_a - sharpe_d:+.2f}  (A vs D)")
if sharpe_a and sharpe_c:
    print(f"  R55 merge effect:  {sharpe_a - sharpe_c:+.2f}  (A vs C)")
if sharpe_c and sharpe_b:
    print(f"  cs_rank + R55:     {sharpe_c - sharpe_b:+.2f}  (C vs B)")
print()
print("  Expected: A ≈ 1.66, B ≈ 0.91")
print("  If A ≈ 1.66 → root cause is cs_rank + R55 merge")
print("  If A ≈ 0.91 → something else changed (data? timestamps?)")
