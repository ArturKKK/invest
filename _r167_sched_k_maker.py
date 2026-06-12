#!/usr/bin/env python3
"""R167 — sim-side boosters on the frozen 10-seed stack. VM ONLY.

(1) Re-train champion30 s10 on W2W3 and CACHE preds (R166 trained but never
    saved them; every future sim-side experiment needs this cache).
(2) Sanity: reproduce R166 stack (3.080; threading noise tolerance ~0.15).
(3) Pre-registered regime-scheduled k (plan P3). PRIMARY arm: k=0.75 when
    td_persist_720h < expanding-median (choppy regime -> specialist up-weighted),
    k=0.25 otherwise, default 0.5 while median undefined. The MIRROR arm is
    DIAGNOSTIC ONLY and cannot be adopted regardless of its number.
    Adopt PRIMARY iff paired P(sched > fixed-0.5) >= 0.85.
(4) Maker T2/T3 cost variants on the frozen stack — CONDITIONAL numbers (the
    prod maker redo per review spec is NOT deployed; these state what the same
    trades would net IF T2/T3 execution moves maker-first).
(5) k-grid {0.4, 0.6} gated diagnostic on s10 (stability check around 0.5).
"""
from _preflight_check import check_versions
check_versions()

import json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import _research_r68_continuous_wf as r68
from _research_r68_continuous_wf import CHAMPION_FEAT_31, CONTINUOUS_WINDOWS, sharpe, train_ensemble
from _research_r35_new_features import MARKET_LEVEL_FEATURES
from _research_r121_realistic_costs import R114B_CFG
from src.costs import cost_prod_blended, TIER1_SYMS, TIER3_SYMS
from _r136_s6_retest import simulate_r136, A1_FROZEN, L_FROZEN, Q_FROZEN
import _r129_persistence_gate as r129

SEEDS10 = [0, 7, 13, 42, 99, 1, 8, 14, 43, 100]
W23 = CONTINUOUS_WINDOWS[1:]  # W2, W3
R166_REF = 3.080


def cost_maker23_cons(sym):
    """Maker-first T2/T3, conservative fills: T2 70% maker -> 3.5bp, T3 50% -> 7bp."""
    if sym in TIER1_SYMS:
        return 0.90 * 0.0002 + 0.10 * 0.0006     # 2.4bp unchanged
    if sym in TIER3_SYMS:
        return 0.50 * 0.0004 + 0.50 * 0.0010     # 7.0bp (maker fee2+adv2 / taker10)
    return 0.70 * 0.0002 + 0.30 * 0.0007         # 3.5bp


def cost_maker23_aggr(sym):
    """Maker-first T2/T3, optimistic fills: T2 85% -> 2.75bp, T3 70% -> 5.1bp."""
    if sym in TIER1_SYMS:
        return 0.90 * 0.0002 + 0.10 * 0.0006     # 2.4bp unchanged
    if sym in TIER3_SYMS:
        return 0.70 * 0.0003 + 0.30 * 0.0010     # 5.1bp
    return 0.85 * 0.0002 + 0.15 * 0.0007         # 2.75bp


def boot_paired(a, b, n_boot=1000, block=14, seed=167):
    m = a[["timestamp", "net_ret"]].rename(columns={"net_ret": "x"}).merge(
        b[["timestamp", "net_ret"]].rename(columns={"net_ret": "y"}), on="timestamp")
    x, y = m["x"].values, m["y"].values; n = len(x)
    rng_ = np.random.RandomState(seed); wins = 0
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s, min(s + block, n))
                              for s in rng_.randint(0, n - block, size=n // block + 1)])[:n]
        sx = (x[idx].sum() / (x[idx].std() + 1e-12)) / np.sqrt(len(idx))
        sy = (y[idx].sum() / (y[idx].std() + 1e-12)) / np.sqrt(len(idx))
        wins += (sx > sy)
    return wins / n_boot


print("Loading frame...")
df, regime_df = r68.load_data()
if "timestamp" in regime_df.columns:
    regime_df = regime_df.set_index("timestamp")
regime_aug = r129.add_persistence(regime_df, lookback=L_FROZEN)
persist = regime_aug[f"td_persist_{L_FROZEN}h"]
thr = r129.expanding_quantile_threshold(persist, Q_FROZEN, min_periods=720)
gate = (persist < thr)

feats30 = [f for f in CHAMPION_FEAT_31 if f in df.columns and f != "cg_taker_imb"]


def run_gated(preds, label, cost_fn=cost_prod_blended):
    port = simulate_r136(preds, regime_aug, 4, 2, dict(R114B_CFG),
                         cutoff_on=0.9, cutoff_off=0.8, min_risk_off_periods=2,
                         cost_fn=cost_fn, funding_per_12h=0.00012,
                         exec_delay_penalty=0.0003, a1_cfg=A1_FROZEN, gate_series=gate)
    ns = sharpe(port["net_ret"])
    ret = ((1 + port["net_ret"]).prod() - 1) * 100
    dd = ((1 + port["net_ret"]).cumprod() / (1 + port["net_ret"]).cumprod().cummax() - 1).min() * 100
    print(f"  {label:40s} Net={ns:+.3f}  Ret={ret:+.1f}%  DD={dd:+.1f}%  n={len(port)}", flush=True)
    return ns, port


# ── (1) champion s10: train ONCE, cache IMMEDIATELY ──────────────────────
print("\n=== TRAIN champion30 s10 W2W3 (the slow part) ===", flush=True)
champ = train_ensemble(df, feats30, W23, seeds=SEEDS10,
                       cs_rank_exclude=[f for f in feats30 if f in MARKET_LEVEL_FEATURES])
champ.to_parquet("cache/r167_champ30_s10_w23_preds.parquet", index=False)
print("champion s10 preds CACHED -> cache/r167_champ30_s10_w23_preds.parquet", flush=True)

spec = pd.read_parquet("cache/r166_spec_venue5_s10_preds.parquet")
mg = champ.merge(spec[["timestamp", "symbol", "pred"]].rename(columns={"pred": "spred"}),
                 on=["timestamp", "symbol"], how="left")
mg["spred"] = mg["spred"].fillna(0.0)

results = {}

# ── (2) sanity: reproduce R166 ────────────────────────────────────────────
print("\n=== SANITY: reproduce R166 stack ===")
bl = mg.copy(); bl["pred"] = bl["pred"] + 0.5 * bl["spred"]
ns_fix, p_fix = run_gated(bl, "STACK k=0.5 + GATED_A1 s10")
results["stack_fixed"] = round(float(ns_fix), 3)
if abs(ns_fix - R166_REF) > 0.2:
    print(f"  WARN: reproduced {ns_fix:.3f} vs R166 {R166_REF} — investigate before trusting deltas")

# ── (3) regime-scheduled k (pre-registered) ───────────────────────────────
print("\n=== SCHEDULED k (P3, pre-registered) ===")
med = r129.expanding_quantile_threshold(persist, 0.5, min_periods=720)
for arm, hi_when_choppy in (("PRIMARY", True), ("mirror-diag", False)):
    klo, khi = (0.25, 0.75)
    cond = (persist < med) if hi_when_choppy else (persist >= med)
    k_ser = pd.Series(np.where(cond, khi, klo), index=persist.index)
    k_ser[med.isna()] = 0.5
    ks = mg["timestamp"].map(k_ser).fillna(0.5)
    bl2 = mg.copy(); bl2["pred"] = bl2["pred"] + ks.values * bl2["spred"]
    ns_s, p_s = run_gated(bl2, f"sched-k {arm} (0.75 {'choppy' if hi_when_choppy else 'trendy'})")
    pwin = boot_paired(p_s, p_fix)
    print(f"     -> delta {ns_s - ns_fix:+.3f}, P(sched>fixed) = {pwin:.3f}")
    results[f"sched_{arm}"] = {"ns": round(float(ns_s), 3), "p": round(float(pwin), 3)}

# ── (4) maker T2/T3 conditional costs ─────────────────────────────────────
print("\n=== MAKER T2/T3 (conditional — prod redo not deployed) ===")
for nm, fn in (("maker23_conservative", cost_maker23_cons), ("maker23_aggressive", cost_maker23_aggr)):
    ns_m, p_m = run_gated(bl, f"stack + {nm}", cost_fn=fn)
    results[nm] = round(float(ns_m), 3)

# ── (5) k-grid diagnostic ─────────────────────────────────────────────────
print("\n=== k-grid diagnostic (gated, s10) ===")
for k in (0.4, 0.6):
    blk = mg.copy(); blk["pred"] = blk["pred"] + k * blk["spred"]
    ns_k, p_k = run_gated(blk, f"stack k={k} + GATED_A1")
    results[f"kgrid_{k}"] = round(float(ns_k), 3)

with open("results_r167_sched_maker.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nGATE: adopt sched-k iff PRIMARY P>=0.85; maker numbers are conditional on prod redo.")
print("R167 done.")
