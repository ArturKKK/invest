#!/usr/bin/env python3
"""R143 — pristine OOS + retrain-cadence test on June-refreshed data.

The window 2026-04-26 -> 2026-06-08 was never seen by any model or selection
decision (all prior selection used data <= 2026-04-26). Trains 3 cadence
variants (different train_end), scores per-timestamp rank-IC (far more power
than a ~80-period Sharpe) + Net Sharpe (S6 prod_blended) on the pristine window
and (for V1/V2) the old April sub-window 2026-03-18..04-25 as a continuity
check vs the April numbers (V1~-0.27, V2~+1.93).

Usage: python _r143_pristine_oos.py [30f|31f]
  30f drops cg_taker_imb so the FULL pristine window is clean despite CoinGlass
      dying 2026-05-06 (use if R142 shows cg_taker_imb is unimportant).
  31f keeps all features (cg_taker_imb stale after 2026-05-06 -> degraded tail).
"""
from _preflight_check import check_versions
check_versions()

import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import _research_r68_continuous_wf as r68
from _research_r68_continuous_wf import CHAMPION_FEAT_31, sharpe
from _research_r35_new_features import MARKET_LEVEL_FEATURES
from _research_r22_models import SEEDS
from _research_r121_realistic_costs import R114B_CFG
from src.costs import cost_prod_blended
from _r136_s6_retest import simulate_r136

FEATURESET = sys.argv[1] if len(sys.argv) > 1 else "30f"

# Per-variant test_start: V1/V2 include old April (clean continuity); V3's val
# ends 2026-04-25 so it tests pristine only (no leak).
VARIANTS = [
    {"name": "V1_stale_2025-07", "train_end": "2025-07-01",
     "val_start": "2025-07-01", "val_end": "2025-10-31", "test_start": "2026-03-18"},
    {"name": "V2_2026-01_R132", "train_end": "2026-01-01",
     "val_start": "2026-01-01", "val_end": "2026-03-15", "test_start": "2026-03-18"},
    {"name": "V3_fresh_2026-02-25", "train_end": "2026-02-25",
     "val_start": "2026-02-25", "val_end": "2026-04-25", "test_start": "2026-04-26"},
]
TEST_END = "2026-06-08"
PRISTINE = ("2026-04-26", "2026-06-08")
OLD_APRIL = ("2026-03-18", "2026-04-25")


def _ts(x):
    return pd.Timestamp(x, tz="UTC")


def _ic_series(preds, lo, hi):
    sub = preds[(preds["timestamp"] >= _ts(lo)) & (preds["timestamp"] <= _ts(hi))]
    out = []
    for ts, g in sub.groupby("timestamp"):
        if g["pred"].nunique() > 2 and g["fwd_ret"].nunique() > 2:
            ic = spearmanr(g["pred"], g["fwd_ret"]).correlation
            if not np.isnan(ic):
                out.append((ts, ic))
    return pd.Series(dict(out)).sort_index()


def _nw_tstat(x, lags):
    """Newey-West (Bartlett) t-stat of the mean for an autocorrelated series."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 5:
        return np.nan
    d = x - x.mean()
    var = d @ d / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        var += 2.0 * w * (d[:-k] @ d[k:]) / n
    se = np.sqrt(max(var, 1e-18) / n)
    return x.mean() / (se + 1e-18)


def rank_ic(preds, lo, hi):
    """Returns (mean_ic, t_naive_hourly, n_hourly, t_nw12, t_grid12, n_grid).

    fwd_ret is 12h while preds are hourly -> adjacent hourly ICs overlap and
    the naive t is inflated. t_nw12 = Newey-West(12 lags) on the hourly series;
    t_grid12 = plain t on the non-overlapping every-12th-hour grid (worst case,
    averaged over the 12 possible grid offsets).
    """
    s = _ic_series(preds, lo, hi)
    if len(s) < 5:
        return np.nan, np.nan, len(s), np.nan, np.nan, 0
    mean_ic = s.mean()
    t_naive = mean_ic / (s.std(ddof=1) / np.sqrt(len(s)) + 1e-12)
    t_nw = _nw_tstat(s.values, lags=12)
    grid_ts, grid_ns = [], []
    for off in range(12):
        g = s.values[off::12]
        if len(g) >= 5:
            grid_ts.append(np.mean(g) / (np.std(g, ddof=1) / np.sqrt(len(g)) + 1e-12))
            grid_ns.append(len(g))
    t_grid = float(np.mean(grid_ts)) if grid_ts else np.nan
    n_grid = int(np.mean(grid_ns)) if grid_ns else 0
    return mean_ic, t_naive, len(s), t_nw, t_grid, n_grid


def sharpe_window(preds, lo, hi):
    sub = preds[(preds["timestamp"] >= _ts(lo)) & (preds["timestamp"] <= _ts(hi))].copy()
    if len(sub) == 0:
        return np.nan, 0, np.nan
    port = simulate_r136(
        sub, regime_df, 4, 2, dict(R114B_CFG),
        cutoff_on=0.9, cutoff_off=0.8, min_risk_off_periods=2,
        cost_fn=cost_prod_blended, funding_per_12h=0.00012,
        exec_delay_penalty=0.0003,
    )
    if len(port) < 3:
        return np.nan, len(port), np.nan
    ns = sharpe(port["net_ret"])
    ret = (1 + port["net_ret"]).prod() - 1
    return ns, len(port), ret * 100


df, regime_df = r68.load_data()
if "timestamp" in regime_df.columns:
    regime_df = regime_df.set_index("timestamp")

feats_all = [f for f in CHAMPION_FEAT_31 if f in df.columns]
feats = [f for f in feats_all if f != "cg_taker_imb"] if FEATURESET == "30f" else feats_all
no_rank = [f for f in feats if f in MARKET_LEVEL_FEATURES]

print("=" * 96)
print(f"  R143 — PRISTINE OOS + cadence  | FEATURESET={FEATURESET} ({len(feats)} feats) | "
      f"pristine {PRISTINE[0]}..{PRISTINE[1]}")
print(f"  Data range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
print("=" * 96)
print("  PRE-REGISTERED GATE: restart only if best variant has Ret>0 AND rank-IC t>2 on PRISTINE")
print("=" * 96)

rows = []
for v in VARIANTS:
    w = {"name": v["name"], "train_end": v["train_end"], "val_start": v["val_start"],
         "val_end": v["val_end"], "test_start": v["test_start"], "test_end": TEST_END}
    preds = r68.train_ensemble(df, feats, [w], seeds=SEEDS, cs_rank_exclude=no_rank)
    if preds is None or len(preds) == 0:
        print(f"\n{v['name']}: NO PREDS")
        continue
    cache_path = f"cache/r143_{v['name']}_{FEATURESET}_preds.parquet"
    preds.to_parquet(cache_path, index=False)
    pic, pt, pn, pt_nw, pt_grid, png = rank_ic(preds, *PRISTINE)
    psh, pnp, pret = sharpe_window(preds, *PRISTINE)
    line = (f"\n{v['name']} (train_end {v['train_end']}, val {v['val_start']}..{v['val_end']}):\n"
            f"  PRISTINE  rankIC={pic:+.4f}  t_naive={pt:+.2f} (n={pn})  "
            f"t_NW12={pt_nw:+.2f}  t_grid12={pt_grid:+.2f} (n={png})\n"
            f"            Sharpe={psh:+.3f} Ret={pret:+.1f}% (n={pnp})   [saved {cache_path}]")
    if v["test_start"] == "2026-03-18":
        aic, at, an, at_nw, at_grid, ang = rank_ic(preds, *OLD_APRIL)
        ash, anp, aret = sharpe_window(preds, *OLD_APRIL)
        line += (f"\n  OLD-APRIL rankIC={aic:+.4f} t_NW12={at_nw:+.2f} t_grid12={at_grid:+.2f}  "
                 f"Sharpe={ash:+.3f} Ret={aret:+.1f}% (n={anp})  [Apr 31f refs: V1~-0.27 V2~+1.93]")
    print(line)
    rows.append({"variant": v["name"], "pristine_ic": pic, "pristine_ic_t": pt,
                 "pristine_ic_t_nw": pt_nw, "pristine_ic_t_grid": pt_grid,
                 "pristine_sharpe": psh, "pristine_ret": pret})

print("\n" + "=" * 96)
print("  SUMMARY (pristine window)")
print("=" * 96)
res = pd.DataFrame(rows)
if len(res):
    print(res.to_string(index=False))
    # Gate on the OVERLAP-HONEST t (Newey-West 12 lags) — naive hourly t is
    # inflated by the 12h-horizon overlap and must not decide the gate.
    best = res.sort_values("pristine_ic_t_nw", ascending=False).iloc[0]
    gate = (best["pristine_ret"] > 0) and (best["pristine_ic_t_nw"] > 2)
    print(f"\n  BEST by NW t-stat: {best['variant']} (t_NW12={best['pristine_ic_t_nw']:+.2f}, "
          f"t_grid12={best['pristine_ic_t_grid']:+.2f}, Ret={best['pristine_ret']:+.1f}%)")
    print(f"  RESTART GATE (overlap-honest): {'PASS' if gate else 'FAIL'} "
          f"(need Ret>0 AND t_NW12>2; got Ret={best['pristine_ret']:+.1f}%, "
          f"t_NW12={best['pristine_ic_t_nw']:+.2f})")
