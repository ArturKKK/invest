#!/usr/bin/env python3
"""R148 — OOS validation of the R146 sweep winner (dyn_threshold=0.8) on the
pristine window, from saved R143 pred caches. VM ONLY (needs June regime).

R146 (canonical cache, honest S6): dyn=0.8 -> 2.874, d=+0.043, P=0.824 — the
only gate-passing cell. Check: does it also help on 2026-04-26..06-08 OOS?
Also checks dyn=0.8 stacked with GATED_A1 (the promoted overlay candidate).
"""
from _preflight_check import check_versions
check_versions()

import warnings
warnings.filterwarnings("ignore")
import pandas as pd

import _research_r68_continuous_wf as r68
from _research_r68_continuous_wf import sharpe
from _research_r121_realistic_costs import R114B_CFG
from src.costs import cost_prod_blended
from _r136_s6_retest import simulate_r136, A1_FROZEN, L_FROZEN, Q_FROZEN
import _r129_persistence_gate as r129

PRISTINE = (pd.Timestamp("2026-04-26", tz="UTC"), pd.Timestamp("2026-06-08", tz="UTC"))

df, regime_df = r68.load_data()
del df
if "timestamp" in regime_df.columns:
    regime_df = regime_df.set_index("timestamp")
regime_aug = r129.add_persistence(regime_df, lookback=L_FROZEN)
thr = r129.expanding_quantile_threshold(regime_aug[f"td_persist_{L_FROZEN}h"], Q_FROZEN, min_periods=720)
gate_series = (regime_aug[f"td_persist_{L_FROZEN}h"] < thr)


def run(preds, label, dyn, a1=False):
    cfg = dict(R114B_CFG)
    cfg["dyn_threshold"] = dyn
    port = simulate_r136(
        preds, regime_aug, 4, 2, cfg,
        cutoff_on=0.9, cutoff_off=0.8, min_risk_off_periods=2,
        cost_fn=cost_prod_blended, funding_per_12h=0.00012,
        exec_delay_penalty=0.0003,
        a1_cfg=A1_FROZEN if a1 else None,
        gate_series=gate_series if a1 else None,
    )
    ns = sharpe(port["net_ret"])
    ret = ((1 + port["net_ret"]).prod() - 1) * 100
    print(f"  {label:44s} Net={ns:+.3f}  Ret={ret:+.1f}%  n={len(port)}", flush=True)
    return ns


print("=" * 84)
print("  R148 — dyn_threshold 0.7 vs 0.8 (+GATED_A1 stack) on pristine OOS")
print("=" * 84)
for tag in ["V2_2026-01_R132", "V3_fresh_2026-02-25"]:
    preds = pd.read_parquet(f"cache/r143_{tag}_30f_preds.parquet")
    sub = preds[(preds["timestamp"] >= PRISTINE[0]) & (preds["timestamp"] <= PRISTINE[1])].copy()
    print(f"\n{tag}:")
    b7 = run(sub, "dyn=0.7 (prod)", 0.7)
    b8 = run(sub, "dyn=0.8 (R146 winner)", 0.8)
    g7 = run(sub, "dyn=0.7 + GATED_A1", 0.7, a1=True)
    g8 = run(sub, "dyn=0.8 + GATED_A1", 0.8, a1=True)
    print(f"  -> dyn0.8 delta: {b8-b7:+.3f} | gated stack delta vs prod: {g8-b7:+.3f} | gated(0.7) {g7-b7:+.3f}")
print("\nR148 done.")
