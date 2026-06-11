#!/usr/bin/env python3
"""R150c — score surviving seed-ensemble pred caches (sim only, no retrain).
VM ONLY (needs regime via load_data). Base + GATED_A1 for seeds10/seeds20.
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

df, regime_df = r68.load_data()
del df
if "timestamp" in regime_df.columns:
    regime_df = regime_df.set_index("timestamp")
regime_aug = r129.add_persistence(regime_df, lookback=L_FROZEN)
thr = r129.expanding_quantile_threshold(regime_aug[f"td_persist_{L_FROZEN}h"], Q_FROZEN, min_periods=720)
gate = (regime_aug[f"td_persist_{L_FROZEN}h"] < thr)

for n in (10, 20):
    try:
        preds = pd.read_parquet(f"cache/r150_seeds{n}_preds.parquet")
    except Exception as e:
        print(f"seeds{n}: no cache ({e})")
        continue
    for label, a1 in (("base", False), ("GATED_A1", True)):
        port = simulate_r136(
            preds, regime_aug, 4, 2, dict(R114B_CFG),
            cutoff_on=0.9, cutoff_off=0.8, min_risk_off_periods=2,
            cost_fn=cost_prod_blended, funding_per_12h=0.00012, exec_delay_penalty=0.0003,
            a1_cfg=A1_FROZEN if a1 else None, gate_series=gate if a1 else None,
        )
        ns = sharpe(port["net_ret"])
        ret = ((1 + port["net_ret"]).prod() - 1) * 100
        dd = ((1 + port["net_ret"]).cumprod() / (1 + port["net_ret"]).cumprod().cummax() - 1).min() * 100
        print(f"  seeds{n:2d} {label:9s}: Net={ns:+.3f}  Ret={ret:+.1f}%  DD={dd:+.1f}%  n={len(port)}", flush=True)
print("R150c done.")
