#!/usr/bin/env python3
"""R144 — two quick cache-based checks (no retraining):

A) GATED_A1 out-of-sample: apply the only surviving overlay (persistence-gated
   asymmetric kelly, frozen L=720/q=0.20) to the saved R143 pred caches on the
   pristine window 2026-04-26..2026-06-08. Does the overlay still help OOS?

B) cost_prod_blended_v2 (D6-recalibrated tiers) impact on the CANONICAL
   champion cache: baseline 2.831 re-simulated under v2 costs.
"""
from _preflight_check import check_versions
check_versions()

import warnings
warnings.filterwarnings("ignore")
import pandas as pd

import _research_r68_continuous_wf as r68
from _research_r68_continuous_wf import sharpe
from _research_r121_realistic_costs import R114B_CFG
from _research_r113_trend_cutoff_reopt import analyze_config
from src.costs import cost_prod_blended, cost_prod_blended_v2
from _r136_s6_retest import simulate_r136, A1_FROZEN, L_FROZEN, Q_FROZEN
import _r129_persistence_gate as r129

PRISTINE = (pd.Timestamp("2026-04-26", tz="UTC"), pd.Timestamp("2026-06-08", tz="UTC"))


def run(preds, regime_df, label, cost_fn=cost_prod_blended, a1_cfg=None, gate_series=None):
    port = simulate_r136(
        preds, regime_df, 4, 2, dict(R114B_CFG),
        cutoff_on=0.9, cutoff_off=0.8, min_risk_off_periods=2,
        cost_fn=cost_fn, funding_per_12h=0.00012, exec_delay_penalty=0.0003,
        a1_cfg=a1_cfg, gate_series=gate_series,
    )
    ns = sharpe(port["net_ret"]) if len(port) > 2 else float("nan")
    ret = ((1 + port["net_ret"]).prod() - 1) * 100 if len(port) else float("nan")
    dd = ((1 + port["net_ret"]).cumprod() / (1 + port["net_ret"]).cumprod().cummax() - 1).min() * 100 if len(port) else float("nan")
    print(f"  {label:46s} Net={ns:+.3f}  Ret={ret:+.1f}%  DD={dd:+.1f}%  n={len(port)}")
    return ns


print("=" * 86)
print("  R144-A — GATED_A1 OOS on pristine window (saved R143 caches, S6 costs)")
print("=" * 86)
# Regime frame from full fresh data (needed for the persistence gate + state machine)
df, regime_df = r68.load_data()
del df
if "timestamp" in regime_df.columns:
    regime_df = regime_df.set_index("timestamp")
regime_aug = r129.add_persistence(regime_df, lookback=L_FROZEN)
persist_col = f"td_persist_{L_FROZEN}h"
thr = r129.expanding_quantile_threshold(regime_aug[persist_col], Q_FROZEN, min_periods=720)
gate_series = (regime_aug[persist_col] < thr)
print(f"  gate_on fraction: {gate_series.mean()*100:.1f}%")

for tag in ["V2_2026-01_R132", "V3_fresh_2026-02-25"]:
    path = f"cache/r143_{tag}_30f_preds.parquet"
    try:
        preds = pd.read_parquet(path)
    except Exception as e:
        print(f"  {tag}: cache missing ({e})")
        continue
    sub = preds[(preds["timestamp"] >= PRISTINE[0]) & (preds["timestamp"] <= PRISTINE[1])].copy()
    print(f"\n  {tag} (pristine rows={len(sub):,})")
    b = run(sub, regime_aug, f"{tag} BASE")
    g = run(sub, regime_aug, f"{tag} GATED_A1", a1_cfg=A1_FROZEN, gate_series=gate_series)
    print(f"  -> GATED_A1 OOS delta: {g - b:+.3f}")

print()
print("=" * 86)
print("  R144-B — v2 costs (D6 re-tier) on CANONICAL champion cache")
print("=" * 86)
preds_c = pd.read_parquet("cache/r128_canonical_preds.parquet")
regime_c = pd.read_parquet("cache/r128_canonical_regime.parquet")
if "timestamp" in regime_c.columns:
    regime_c = regime_c.set_index("timestamp")
s6 = run(preds_c, regime_c, "canonical S6 cost_prod_blended (ref 2.831)")
v2 = run(preds_c, regime_c, "canonical cost_prod_blended_v2 (D6 re-tier)", cost_fn=cost_prod_blended_v2)
print(f"  -> v2-costs delta vs S6: {v2 - s6:+.3f}")
print("\nR144 done.")
