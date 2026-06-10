"""R137 — PROPER hysteresis-aware trend_cutoff grid (pre-registered).

Prior evidence was conflicting:
  - R113 grid used the state machine but tested cutoff_on 0.9 then 1.0 (skipped
    0.95); 0.9 vs 1.0 nearly tied, 1.0 worse on DD.
  - May-29 sweep crowned 0.95 but was methodologically weak: single threshold
    (no hysteresis), 6L/3S, lenient costs, fresh retrain.
Memory verdict: "0.95 is a HYPOTHESIS for a proper hysteresis-aware test".

PRE-REGISTERED DESIGN (fixed before running, no deviation):
  Grid: cutoff_on in {0.90, 0.95, 1.00} x cutoff_off in {on-0.05, on-0.10},
        min_risk_off_periods=2 fixed. Baseline cell = (0.90, 0.80) = production.
  Config: 4L/2S R114B_CFG, canonical cached preds (cache/r128_canonical_*.parquet),
        S6 prod_blended costs from src/costs.py, simulate_r121's EXACT state
        machine + include-flat accounting (function imported, not copied —
        it already exposes cutoff_on/cutoff_off/min_risk_off_periods).
  Inference: block bootstrap (block length 14 periods, 1000 resamples), PAIRED
        (same block indices for cell and baseline), delta Net Sharpe vs baseline.
  DECISION RULE: recommend changing production cutoffs ONLY if
        P(delta Sharpe > 0 vs baseline) >= 0.80 AND maxDD not worse by >1pp.
        Otherwise: keep 0.9/0.8.

Secondary diagnostics (NOT part of the decision rule):
  - same grid under lenient_r68 costs (to explain the May-29 "0.95 wins" claim)
  - Net Sharpe without the 3bp exec noise (noise draws misalign across cells
    because the rng is only consumed on active periods).
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from _research_r121_realistic_costs import simulate_r121, R114B_CFG
from _research_r121_realistic_costs import cost_prod_blended as _r121_prod
from _research_r113_trend_cutoff_reopt import analyze_config
from _research_r68_continuous_wf import sharpe
from src.costs import cost_prod_blended, cost_lenient_r68, FUNDING_PER_12H

PREDS = "cache/r128_canonical_preds.parquet"
REGIME = "cache/r128_canonical_regime.parquet"

GRID = [(0.90, 0.85), (0.90, 0.80),
        (0.95, 0.90), (0.95, 0.85),
        (1.00, 0.95), (1.00, 0.90)]
BASELINE = (0.90, 0.80)
MIN_OFF = 2
BLOCK_LEN = 14
N_BOOT = 1000
BOOT_SEED = 137

# ── sanity: unified costs == validated r121-local costs ──────────────────
preds = pd.read_parquet(PREDS)
regime_df = pd.read_parquet(REGIME).set_index("timestamp")
all_syms = preds["symbol"].unique()
assert all(abs(cost_prod_blended(s) - _r121_prod(s)) < 1e-15 for s in all_syms), \
    "src.costs.cost_prod_blended diverges from validated r121 cost model"
assert FUNDING_PER_12H == 0.00012

print(f"preds rows={len(preds):,}  syms={preds['symbol'].nunique()}  "
      f"range {preds['timestamp'].min()} .. {preds['timestamp'].max()}")


def run_cell(on, off, cost_fn, funding, noise=0.0003):
    return simulate_r121(
        preds, regime_df, 4, 2, dict(R114B_CFG),
        cutoff_on=on, cutoff_off=off,
        min_risk_off_periods=MIN_OFF, min_risk_on_periods=0,
        cost_fn=cost_fn, funding_per_12h=funding,
        exec_delay_penalty=noise,
    )


# ── run all 6 cells under S6 (primary) ───────────────────────────────────
print("\n" + "=" * 100)
print("PRIMARY: S6 prod_blended costs, exec noise 3bp seed 42, include-flat 1013 periods")
print("=" * 100)
ports, metrics, nonoise_ns = {}, {}, {}
for on, off in GRID:
    port = run_cell(on, off, cost_prod_blended, FUNDING_PER_12H)
    m = analyze_config(port, f"on={on:.2f}/off={off:.2f}")
    ports[(on, off)] = port
    metrics[(on, off)] = m
    pn = run_cell(on, off, cost_prod_blended, FUNDING_PER_12H, noise=0.0)
    nonoise_ns[(on, off)] = sharpe(pn["net_ret"])

base_port = ports[BASELINE]
base_m = metrics[BASELINE]

# sanity gate: baseline cell must reproduce canonical 2.831 exactly
assert base_m["net_sharpe"] == 2.831, \
    f"HARNESS BROKEN: baseline cell gives {base_m['net_sharpe']}, expected 2.831"
assert base_m["n_periods"] == 1013
print(f"\n[SANITY OK] baseline cell (0.90/0.80) Net Sharpe = "
      f"{base_m['net_sharpe']} on {base_m['n_periods']} periods (canonical 2.831)")

# timestamps must align across cells for paired bootstrap
for key, port in ports.items():
    assert port["timestamp"].equals(base_port["timestamp"]), f"ts misalign {key}"

hdr = (f"{'on/off':>10} | {'NetSh':>6} {'NetSh_noNoise':>13} {'GrossSh':>7} "
       f"{'Ret%':>7} {'MaxDD%':>7} {'Calmar':>6} | {'n_flat':>6} {'%flat':>5} "
       f"{'offEv':>5} {'avgDur':>6} | {'Cost%':>6} {'Turn':>5}")
print("\n" + hdr)
print("-" * len(hdr))
for on, off in GRID:
    m = metrics[(on, off)]
    tag = " *BASE*" if (on, off) == BASELINE else ""
    print(f"{on:.2f}/{off:.2f} | {m['net_sharpe']:6.3f} {nonoise_ns[(on, off)]:13.3f} "
          f"{m['gross_sharpe']:7.3f} {m['total_ret_pct']:7.1f} {m['max_dd_pct']:7.1f} "
          f"{m['calmar']:6.2f} | {m['n_flat']:6d} {m['pct_flat']:5.1f} "
          f"{m['n_off_events']:5d} {m['avg_off_duration']:6.1f} | "
          f"{m['total_cost_pct']:6.2f} {m['avg_turnover']:5.2f}{tag}")

# ── paired moving-block bootstrap vs baseline ────────────────────────────
print("\n" + "=" * 100)
print(f"PAIRED MOVING-BLOCK BOOTSTRAP: block={BLOCK_LEN}, resamples={N_BOOT}, "
      f"seed={BOOT_SEED}, delta Net Sharpe vs baseline (0.90/0.80)")
print("=" * 100)

n = len(base_port)
rng = np.random.RandomState(BOOT_SEED)
n_blocks = int(np.ceil(n / BLOCK_LEN))
# one shared index matrix -> identical resamples for every cell (paired)
starts = rng.randint(0, n - BLOCK_LEN + 1, size=(N_BOOT, n_blocks))
idx_mat = (starts[:, :, None] + np.arange(BLOCK_LEN)[None, None, :]) \
    .reshape(N_BOOT, -1)[:, :n]

base_rets = base_port["net_ret"].to_numpy()


def boot_sharpes(rets):
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        out[b] = sharpe(pd.Series(rets[idx_mat[b]]))
    return out


base_boot = boot_sharpes(base_rets)

boot_rows = {}
print(f"\n{'on/off':>10} | {'dNetSh':>7} | {'P(improve)':>10} | "
      f"{'90% CI of delta':>22} | survives P>=0.80?")
print("-" * 75)
for on, off in GRID:
    if (on, off) == BASELINE:
        continue
    cell_boot = boot_sharpes(ports[(on, off)]["net_ret"].to_numpy())
    delta = cell_boot - base_boot
    p_imp = float((delta > 0).mean())
    lo, hi = np.percentile(delta, [5, 95])
    d_point = metrics[(on, off)]["net_sharpe"] - base_m["net_sharpe"]
    boot_rows[(on, off)] = dict(p_improve=p_imp, ci_lo=lo, ci_hi=hi,
                                d_point=d_point)
    print(f"{on:.2f}/{off:.2f} | {d_point:+7.3f} | {p_imp:10.3f} | "
          f"[{lo:+.3f}, {hi:+.3f}]      | {'YES' if p_imp >= 0.80 else 'no'}")

# ── decision rule (mechanical) ───────────────────────────────────────────
print("\n" + "=" * 100)
print("DECISION RULE: change ONLY if P(delta>0) >= 0.80 AND maxDD not worse by >1pp")
print("=" * 100)
base_dd = base_m["max_dd_pct"]
winners = []
for (on, off), br in boot_rows.items():
    dd = metrics[(on, off)]["max_dd_pct"]
    dd_ok = dd >= base_dd - 1.0          # dd is negative; worse = more negative
    p_ok = br["p_improve"] >= 0.80
    verdict = "PASSES BOTH GATES" if (p_ok and dd_ok) else \
        f"fails ({'P' if not p_ok else ''}{'+' if not p_ok and not dd_ok else ''}{'DD' if not dd_ok else ''})"
    print(f"  {on:.2f}/{off:.2f}: P(improve)={br['p_improve']:.3f} "
          f"(gate>=0.80 {'OK' if p_ok else 'FAIL'}), "
          f"maxDD={dd:.1f}% vs base {base_dd:.1f}% "
          f"(gate {'OK' if dd_ok else 'FAIL'}) -> {verdict}")
    if p_ok and dd_ok:
        winners.append((on, off))

if winners:
    print(f"\nVERDICT: candidate(s) {winners} pass the pre-registered gates.")
else:
    print("\nVERDICT: keep production cutoffs 0.90/0.80 — no cell passes the "
          "pre-registered gates.")

# ── secondary diagnostic: same grid under lenient_r68 costs ──────────────
print("\n" + "=" * 100)
print("SECONDARY (diagnostic only, NOT decision-relevant): lenient_r68 costs, "
      "funding 0.8bp/12h")
print("=" * 100)
print(f"\n{'on/off':>10} | {'NetSh':>6} {'Ret%':>7} {'MaxDD%':>7} {'Calmar':>6}")
print("-" * 45)
for on, off in GRID:
    pl = run_cell(on, off, cost_lenient_r68, 0.00008)
    ml = analyze_config(pl, f"lenient on={on}/off={off}")
    tag = " *BASE*" if (on, off) == BASELINE else ""
    print(f"{on:.2f}/{off:.2f} | {ml['net_sharpe']:6.3f} {ml['total_ret_pct']:7.1f} "
          f"{ml['max_dd_pct']:7.1f} {ml['calmar']:6.2f}{tag}")

print("\nR137 done.")
