"""R146 — TRACK A: honest portfolio-construction sweeps on the canonical cache.

The K / ema_alpha / hysteresis / dyn_threshold knobs were last swept in the
R65/R113 era under OLD accounting + lenient costs. This redoes them honestly:
simulate_r136 (verbatim simulate_r121 semantics: include-flat 1013 periods,
risk-off state machine cutoff_on=0.90/cutoff_off=0.80/min_off=2, costs on both
legs + risk-off close + funding 1.2bp/12h, exec noise N(0,3bp) seed 42) with
S6 cost_prod_blended on cache/r128_canonical_preds.parquet.

PRE-REGISTERED DESIGN (fixed before running, no deviation):
  GRID 1 — K (n_long, n_short): (4,2)=BASE, (3,2), (5,2), (3,3), (4,3), (5,3), (6,3)
  GRID 2 — ema_alpha {0.3, 0.4, 0.5=BASE, 0.7, 1.0=off} at hysteresis=3
  GRID 3 — hysteresis {0, 2, 3=BASE, 4, 5} at ema_alpha=0.5
  GRID 4 — dyn_threshold {0.5, 0.6, 0.7=BASE, 0.8, None=off}
  All other knobs frozen at the R114B/S6 production config.

  Sanity gate: BASE cell must reproduce Net Sharpe 2.831 EXACTLY (analyze_config
  rounding) on 1013 periods, else abort.

  Inference: paired moving-block bootstrap of per-period net_ret
  (block=14 periods = 7d, 1000 resamples, ONE shared index matrix so resamples
  are identical across cells), delta Net Sharpe vs BASE.
  DECISION RULE: a cell is a CANDIDATE only if point delta > 0 AND
  P(delta Sharpe > 0) >= 0.80 AND maxDD not worse than base by >1pp.
  Everything else is labelled "noise".

  IMPORTANT: candidates are HYPOTHESES until validated on the pristine-OOS pred
  caches (cache/r143_*.parquet, on the VM — orchestrator validates).

State machine, rebalance grid and active-period set are identical across all
cells (cutoffs fixed; K/ema/hysteresis/dyn only affect selection/weighting),
so the exec-noise rng stream aligns period-by-period -> comparisons are paired.
"""
import warnings
warnings.filterwarnings("ignore")

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from _r136_s6_retest import simulate_r136
from _research_r121_realistic_costs import R114B_CFG
from _research_r113_trend_cutoff_reopt import analyze_config
from _research_r68_continuous_wf import sharpe
from src.costs import cost_prod_blended, FUNDING_PER_12H

ROOT = Path(__file__).resolve().parent
PREDS = ROOT / "cache" / "r128_canonical_preds.parquet"
REGIME = ROOT / "cache" / "r128_canonical_regime.parquet"
OUT_JSON = ROOT / "results_r146_sweeps.json"

BASELINE_NS = 2.831
BLOCK_LEN = 14
N_BOOT = 1000
BOOT_SEED = 146
P_GATE = 0.80
DD_GATE_PP = 1.0

BASE = dict(nl=4, ns=2, ema=0.5, hyst=3, dyn=0.7)

GRIDS = [
    ("GRID1_K (n_long,n_short)", [
        dict(BASE, nl=4, ns=2), dict(BASE, nl=3, ns=2), dict(BASE, nl=5, ns=2),
        dict(BASE, nl=3, ns=3), dict(BASE, nl=4, ns=3), dict(BASE, nl=5, ns=3),
        dict(BASE, nl=6, ns=3),
    ]),
    ("GRID2_ema_alpha (hyst=3)", [
        dict(BASE, ema=0.3), dict(BASE, ema=0.4), dict(BASE, ema=0.5),
        dict(BASE, ema=0.7), dict(BASE, ema=1.0),
    ]),
    ("GRID3_hysteresis (ema=0.5)", [
        dict(BASE, hyst=0), dict(BASE, hyst=2), dict(BASE, hyst=3),
        dict(BASE, hyst=4), dict(BASE, hyst=5),
    ]),
    ("GRID4_dyn_threshold", [
        dict(BASE, dyn=0.5), dict(BASE, dyn=0.6), dict(BASE, dyn=0.7),
        dict(BASE, dyn=0.8), dict(BASE, dyn=None),
    ]),
]


def cell_key(p):
    return (p["nl"], p["ns"], p["ema"], p["hyst"], p["dyn"])


def cell_label(p):
    dyn = "None" if p["dyn"] is None else f"{p['dyn']:.1f}"
    ema = f"{p['ema']:.1f}" + ("(off)" if p["ema"] == 1.0 else "")
    return f"K={p['nl']}L/{p['ns']}S ema={ema} hyst={p['hyst']} dyn={dyn}"


def main():
    t0 = time.time()
    assert FUNDING_PER_12H == 0.00012
    print("=" * 110)
    print("  R146 — honest portfolio-construction sweeps (S6 prod_blended, "
          "canonical cache, cutoffs 0.90/0.80 min_off=2, 4L/2S base)")
    print("=" * 110)

    preds = pd.read_parquet(PREDS)
    regime_df = pd.read_parquet(REGIME).set_index("timestamp")
    print(f"  preds {len(preds):,} rows / {preds['symbol'].nunique()} syms; "
          f"regime {len(regime_df):,} rows; "
          f"range {preds['timestamp'].min()} .. {preds['timestamp'].max()}")

    ports, metrics = {}, {}

    def run_cell(p):
        key = cell_key(p)
        if key in ports:
            return ports[key], metrics[key]
        cfg = dict(R114B_CFG)
        cfg["ema_alpha"] = p["ema"]
        cfg["hysteresis"] = p["hyst"]
        cfg["dyn_threshold"] = p["dyn"]
        port = simulate_r136(
            preds, regime_df, p["nl"], p["ns"], cfg,
            cutoff_on=0.9, cutoff_off=0.8,
            min_risk_off_periods=2, min_risk_on_periods=0,
            cost_fn=cost_prod_blended, funding_per_12h=FUNDING_PER_12H,
            exec_delay_penalty=0.0003)
        m = analyze_config(port, cell_label(p))
        ports[key], metrics[key] = port, m
        return port, m

    # ── SANITY GATE: base cell must reproduce canonical 2.831 exactly ──
    base_port, base_m = run_cell(BASE)
    assert base_m["n_periods"] == 1013, f"base periods {base_m['n_periods']} != 1013"
    assert base_m["net_sharpe"] == BASELINE_NS, \
        f"HARNESS BROKEN: base cell Net Sharpe {base_m['net_sharpe']} != {BASELINE_NS}"
    print(f"\n  [SANITY OK] BASE cell ({cell_label(BASE)}) Net Sharpe = "
          f"{base_m['net_sharpe']:.3f} on {base_m['n_periods']} periods "
          f"(canonical {BASELINE_NS})  maxDD={base_m['max_dd_pct']:.1f}%")

    # ── shared paired bootstrap index matrix ──
    n = len(base_port)
    rng = np.random.RandomState(BOOT_SEED)
    n_blocks = int(np.ceil(n / BLOCK_LEN))
    starts = rng.randint(0, n - BLOCK_LEN + 1, size=(N_BOOT, n_blocks))
    idx_mat = (starts[:, :, None] + np.arange(BLOCK_LEN)[None, None, :]) \
        .reshape(N_BOOT, -1)[:, :n]

    def boot_sharpes(rets):
        out = np.empty(N_BOOT)
        for b in range(N_BOOT):
            out[b] = sharpe(pd.Series(rets[idx_mat[b]]))
        return out

    base_boot = boot_sharpes(base_port["net_ret"].to_numpy())
    base_dd = base_m["max_dd_pct"]

    results = []
    candidates = []

    for grid_name, cells in GRIDS:
        print("\n" + "=" * 110)
        print(f"  {grid_name}")
        print("=" * 110)
        hdr = (f"  {'cell':<42s} {'NetSh':>6s} {'dNetSh':>7s} {'P(imp)':>7s} "
               f"{'90%CI of delta':>18s} {'GrossSh':>7s} {'Ret%':>7s} "
               f"{'MaxDD%':>7s} {'Cost%':>6s} {'Turn':>5s}  verdict")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for p in cells:
            port, m = run_cell(p)
            is_base = cell_key(p) == cell_key(BASE)
            assert m["n_periods"] == 1013, f"{cell_label(p)}: {m['n_periods']} != 1013"
            assert port["timestamp"].equals(base_port["timestamp"]), \
                f"ts misalign {cell_label(p)} — paired bootstrap invalid"
            d_point = round(m["net_sharpe"] - base_m["net_sharpe"], 3)
            if is_base:
                boot = None
                verdict = "(BASE)"
                p_imp, lo, hi = float("nan"), float("nan"), float("nan")
            else:
                cell_boot = boot_sharpes(port["net_ret"].to_numpy())
                delta = cell_boot - base_boot
                p_imp = float((delta > 0).mean())
                lo, hi = (float(x) for x in np.percentile(delta, [5, 95]))
                boot = {"p_improve": round(p_imp, 3),
                        "ci90_lo": round(lo, 3), "ci90_hi": round(hi, 3)}
                dd_ok = m["max_dd_pct"] >= base_dd - DD_GATE_PP
                if d_point > 0 and p_imp >= P_GATE and dd_ok:
                    verdict = "CANDIDATE (OOS hypothesis)"
                    candidates.append((grid_name, cell_label(p), d_point, p_imp,
                                       m["max_dd_pct"]))
                elif d_point > 0:
                    why = []
                    if p_imp < P_GATE:
                        why.append(f"P={p_imp:.2f}<{P_GATE}")
                    if not dd_ok:
                        why.append(f"DD {m['max_dd_pct']:.1f} vs {base_dd:.1f}")
                    verdict = "noise (" + ", ".join(why) + ")"
                else:
                    verdict = "noise (negative)"
            ci_s = "" if is_base else f"[{lo:+.3f},{hi:+.3f}]"
            p_s = "" if is_base else f"{p_imp:.3f}"
            print(f"  {cell_label(p):<42s} {m['net_sharpe']:>6.3f} {d_point:>+7.3f} "
                  f"{p_s:>7s} {ci_s:>18s} {m['gross_sharpe']:>7.3f} "
                  f"{m['total_ret_pct']:>7.1f} {m['max_dd_pct']:>7.1f} "
                  f"{m['total_cost_pct']:>6.2f} {m['avg_turnover']:>5.2f}  {verdict}")
            results.append({
                "grid": grid_name, "label": cell_label(p),
                "params": {k: p[k] for k in ("nl", "ns", "ema", "hyst", "dyn")},
                "is_base": is_base,
                "net_sharpe": m["net_sharpe"], "delta_vs_base": d_point,
                "gross_sharpe": m["gross_sharpe"],
                "total_ret_pct": m["total_ret_pct"],
                "max_dd_pct": m["max_dd_pct"], "calmar": m["calmar"],
                "total_cost_pct": m["total_cost_pct"],
                "avg_turnover": m["avg_turnover"],
                "n_periods": m["n_periods"], "n_flat": m["n_flat"],
                "boot": boot, "verdict": verdict,
            })

    # ── summary ──
    print("\n" + "=" * 110)
    print(f"  DECISION RULE (pre-registered): point delta>0 AND P(improve)>={P_GATE} "
          f"AND maxDD not worse by >{DD_GATE_PP:.0f}pp vs base ({base_dd:.1f}%)")
    print("=" * 110)
    if candidates:
        print("  CANDIDATES (hypotheses — must be validated on pristine-OOS "
              "cache/r143_*.parquet on the VM before any production change):")
        for g, lab, d, pi, dd in candidates:
            print(f"    {g:<28s} {lab:<42s} d={d:+.3f}  P(improve)={pi:.3f}  "
                  f"maxDD={dd:.1f}%")
    else:
        print("  NO cell passes the pre-registered gates -> keep production config "
              f"({cell_label(BASE)}).")
    print("\n  NOTE: every positive cell here is in-sample on the canonical cache; "
          "winners are HYPOTHESES until checked on cache/r143_*.parquet (VM).")

    meta = {
        "baseline_net_sharpe": BASELINE_NS,
        "base_cell": cell_label(BASE),
        "base_max_dd_pct": base_dd,
        "cost_model": "cost_prod_blended (S6)", "funding_per_12h": 0.00012,
        "exec_noise_bp": 3, "cutoffs": "on=0.90/off=0.80/min_off=2",
        "block_len": BLOCK_LEN, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
        "p_gate": P_GATE, "dd_gate_pp": DD_GATE_PP,
        "candidates": [
            {"grid": g, "label": lab, "delta": d, "p_improve": pi,
             "max_dd_pct": dd} for g, lab, d, pi, dd in candidates],
    }
    with open(OUT_JSON, "w") as f:
        json.dump({"meta": meta, "cells": results}, f, indent=2, default=str)
    print(f"\n  Saved: {OUT_JSON}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
