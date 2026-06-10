"""R136 — Re-test all R128-era overlay claims under HONEST accounting + S6 costs.

Every R128-R135 overlay result was measured with the lenient r68 cost model
(Tier1 = -0.36bp NET REBATE) and on record-zero/skip accounting WITHOUT the
production risk-off state machine (cutoff_on=0.9 / cutoff_off=0.8 / min_off=2).

This script ports the overlays onto the honest simulate_r121 methodology
(include-flat 1013 periods, state machine, costs on both legs + risk-off close
+ funding, exec noise N(0,3bp) seed 42) and runs each experiment under BOTH
cost models (cost_lenient_r68 and cost_prod_blended from src/costs.py).

Experiments:
  BASE_CLOSE          canonical baseline (must reproduce 2.831 under S6)
  BASE_SKIP           SKIP risk-off: hold prev positions through risk-off,
                      record their fwd_ret minus funding, no turnover cost,
                      rebalance resumes on exit. ALL 1013 periods recorded.
  A1 t=0.25 s=0.50    best from r128b finegrid (claimed +0.339 lenient)
  A1 t=0.25 s=0.60    R131 frozen params (claimed +0.338 lenient)
  GATED_A1            persistence-gated A1, frozen L=720 / q=0.20 (R131)
  G2 lb=14/28         vol-weighting, LEGACY variant (reproduces r128c lookup,
                      which reads rolling std of OVERLAPPING fwd_ret 1h back —
                      contains look-ahead) and HONEST variant (vol available
                      only once the fwd window has closed, ts-12h)
  A1+G2               headline combo (claimed +0.591 lenient)

For every overlay: paired moving-block bootstrap of per-period net_ret
(block=14 periods=7d, 1000 resamples) -> P(delta Sharpe > 0) vs same-cost
baseline.
"""
from __future__ import annotations

import json
import time
import warnings
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from _preflight_check import check_versions
check_versions()

from src.costs import cost_lenient_r68, cost_prod_blended
from _research_r121_realistic_costs import R114B_CFG
from _research_r113_trend_cutoff_reopt import analyze_config
from _research_r68_continuous_wf import sharpe
import _r129_persistence_gate as r129

PREDS = "cache/r128_canonical_preds.parquet"
REGIME = "cache/r128_canonical_regime.parquet"

COST_SETUPS = {
    # label: (cost_fn, funding_per_12h)
    "lenient_r68": (cost_lenient_r68, 0.00008),
    "prod_blended": (cost_prod_blended, 0.00012),
}

A1_BEST = {"trend_thr": 0.25, "weak_scale": 0.50}    # r128b finegrid best
A1_FROZEN = {"trend_thr": 0.25, "weak_scale": 0.60}  # R131 frozen
L_FROZEN, Q_FROZEN = 720, 0.20

BLOCK_LEN = 14
N_BOOT = 1000


# ─────────────────────────────────────────────────────────────────────
# G2 vol lookups
# ─────────────────────────────────────────────────────────────────────

def build_g2_lookup(merged: pd.DataFrame, lookback: int, honest: bool) -> Dict[str, pd.Series]:
    """Per-symbol rolling std of fwd_ret (hourly rows, exactly as _r128c_round3).

    legacy (honest=False): series indexed at raw hour h; sim reads value at the
      last index strictly BEFORE ts (searchsorted(ts)-1 = h
      <= ts-1h). fwd_ret at h spans (h, h+12h), so the value read at ts still
      contains returns up to ts+11h -> LOOK-AHEAD. Kept only to reproduce the
      r128c claimed numbers.
    honest (honest=True): value computed at hour h becomes available at h+12h
      (when its last fwd window has fully closed); sim reads last available <= ts.
    """
    out: Dict[str, pd.Series] = {}
    for sym, g in merged.groupby("symbol"):
        g = g.sort_values("timestamp")
        roll = g["fwd_ret"].rolling(lookback, min_periods=max(2, min(8, lookback))).std()
        ts_idx = pd.to_datetime(g["timestamp"].values, utc=True)
        if honest:
            ts_idx = ts_idx + pd.Timedelta(hours=12)
        out[sym] = pd.Series(roll.values, index=ts_idx)
    return out


def g2_weight(sym: str, ts, lookup: Dict[str, pd.Series], honest: bool) -> float:
    s = lookup.get(sym)
    if s is None or len(s) == 0:
        return 1.0
    if honest:
        idx = s.index.searchsorted(ts, side="right") - 1
    else:
        idx = s.index.searchsorted(ts) - 1  # exact r128c lookup (leaky)
    if idx < 0 or pd.isna(s.iloc[idx]):
        return 1.0
    return 1.0 / max(float(s.iloc[idx]), 1e-4)


# ─────────────────────────────────────────────────────────────────────
# Honest simulator: simulate_r121 copied verbatim and extended with
# a1_cfg / gate_series / g2 / risk_off_mode. With all extensions off and
# risk_off_mode="close" it must be bit-compatible with simulate_r121.
# ─────────────────────────────────────────────────────────────────────

def simulate_r136(merged, regime_df, n_long, n_short, cfg,
                  cutoff_on=0.9, cutoff_off=0.8,
                  min_risk_off_periods=2, min_risk_on_periods=0,
                  cost_fn=cost_prod_blended,
                  funding_per_12h=0.00012,
                  exec_delay_penalty=0.0003,
                  *,
                  a1_cfg: Optional[Dict] = None,
                  gate_series: Optional[pd.Series] = None,
                  g2_lookup: Optional[Dict[str, pd.Series]] = None,
                  g2_honest: bool = False,
                  risk_off_mode: str = "close"):
    assert risk_off_mode in ("close", "skip")
    rebal_hours   = cfg["rebal_hours"]
    ema_alpha     = cfg.get("ema_alpha", None)
    hysteresis    = cfg.get("hysteresis", 0)
    dyn_threshold = cfg.get("dyn_threshold", 0.5)

    all_rets = []
    prev_longs: Set[str]  = set()
    prev_shorts: Set[str] = set()
    prev_preds: Dict[str, float] = {}
    prev_w: Dict[str, float] = {}   # signed per-symbol weights of held book
    risk_off = False
    periods_in_off = 0
    periods_in_on  = 999

    timestamps_sorted = sorted(merged["timestamp"].unique())
    grouped = {ts: grp for ts, grp in merged.groupby("timestamp")}
    rebal_timestamps = timestamps_sorted[::rebal_hours]

    rng = np.random.RandomState(42)  # deterministic noise, same as simulate_r121

    for ts in rebal_timestamps:
        if ts not in regime_df.index or ts not in grouped:
            continue
        row = regime_df.loc[ts]
        trend_str = row.get("trend_strength", 0)
        trend_dir = row.get("trend_direction", 0) if "trend_direction" in row else 0
        grp = grouped[ts].copy()

        # ── EMA smoothing (before state machine, as in simulate_r121) ──
        if ema_alpha is not None and ema_alpha < 1.0:
            for idx, r in grp.iterrows():
                sym = r["symbol"]
                raw_pred = r["pred"]
                smoothed = (ema_alpha * raw_pred
                            + (1 - ema_alpha) * prev_preds.get(sym, raw_pred))
                prev_preds[sym] = smoothed
                grp.at[idx, "pred"] = smoothed

        def _held_row():
            """SKIP mode: hold prev book through this risk-off period.
            Gross = held signed weights x this period's fwd_ret. Funding only,
            NO turnover cost. Period IS recorded."""
            if prev_w:
                fwd = dict(zip(grp["symbol"], grp["fwd_ret"]))
                gross = float(sum(w * fwd.get(s, 0.0) for s, w in prev_w.items()))
                hold_cost = funding_per_12h * (rebal_hours / 12)
                return {"timestamp": ts, "gross_ret": gross,
                        "net_ret": gross - hold_cost, "cost": hold_cost,
                        "n_long": len(prev_longs), "n_short": len(prev_shorts),
                        "turnover": 0, "risk_off": True}
            return {"timestamp": ts, "gross_ret": 0.0, "net_ret": 0.0,
                    "cost": 0.0, "n_long": 0, "n_short": 0,
                    "turnover": 0, "risk_off": True}

        # ── State machine (identical transitions in both modes) ──
        if cutoff_on is not None:
            if risk_off:
                periods_in_off += 1
                can_exit = (trend_str < cutoff_off
                            and periods_in_off >= min_risk_off_periods)
                if can_exit:
                    risk_off = False
                    periods_in_on = 0
                else:
                    if risk_off_mode == "skip":
                        all_rets.append(_held_row())
                    else:
                        all_rets.append({
                            "timestamp": ts, "gross_ret": 0.0, "net_ret": 0.0,
                            "cost": 0.0, "n_long": 0, "n_short": 0,
                            "turnover": 0, "risk_off": True,
                        })
                    continue
            else:
                periods_in_on += 1
                can_enter = (trend_str > cutoff_on
                             and periods_in_on >= min_risk_on_periods)
                if can_enter:
                    risk_off = True
                    periods_in_off = 0
                    periods_in_on = 0
                    if risk_off_mode == "skip":
                        all_rets.append(_held_row())
                        continue
                    if prev_longs or prev_shorts:
                        n_prev = len(prev_longs) + len(prev_shorts)
                        avg_w = 1.0 / n_prev
                        close_cost = sum(cost_fn(s) * avg_w
                                         for s in prev_longs | prev_shorts)
                        all_rets.append({
                            "timestamp": ts, "gross_ret": 0.0,
                            "net_ret": -close_cost, "cost": close_cost,
                            "n_long": 0, "n_short": 0,
                            "turnover": n_prev, "risk_off": True,
                        })
                    else:
                        all_rets.append({
                            "timestamp": ts, "gross_ret": 0.0,
                            "net_ret": 0.0, "cost": 0.0,
                            "n_long": 0, "n_short": 0,
                            "turnover": 0, "risk_off": True,
                        })
                    prev_longs, prev_shorts = set(), set()
                    prev_w = {}
                    continue

        # ── Portfolio construction (verbatim simulate_r121) ──
        n = len(grp)
        nl = min(n_long, n // 3)
        ns = min(n_short, n // 3)
        if nl == 0 and ns == 0:
            all_rets.append({
                "timestamp": ts, "gross_ret": 0.0, "net_ret": 0.0,
                "cost": 0.0, "n_long": 0, "n_short": 0,
                "turnover": 0, "risk_off": False,
            })
            continue

        exposure = 1.0
        if (cutoff_on is not None and dyn_threshold is not None
                and trend_str > dyn_threshold):
            exposure = max(0.1, 1.0 - (trend_str - dyn_threshold) /
                           (cutoff_on - dyn_threshold + 1e-10) * 0.5)

        grp["pred_rank"] = grp["pred"].rank(ascending=False)

        if hysteresis > 0 and (prev_longs or prev_shorts):
            new_longs: Set[str] = set()
            new_shorts: Set[str] = set()
            for idx, r in grp.iterrows():
                sym, rank = r["symbol"], r["pred_rank"]
                if sym in prev_longs and rank <= nl + hysteresis:
                    new_longs.add(sym)
                elif sym in prev_shorts and rank > (n - ns - hysteresis):
                    new_shorts.add(sym)
            remaining = grp[~grp["symbol"].isin(new_longs | new_shorts)]
            for _, r in remaining.sort_values("pred_rank").head(
                    nl - len(new_longs)).iterrows():
                new_longs.add(r["symbol"])
            for _, r in remaining.sort_values(
                    "pred_rank", ascending=False).head(
                    ns - len(new_shorts)).iterrows():
                new_shorts.add(r["symbol"])
        else:
            new_longs = (set(grp[grp["pred_rank"] <= nl]["symbol"].tolist())
                         if nl > 0 else set())
            new_shorts = (set(grp[grp["pred_rank"] > (n - ns)]["symbol"].tolist())
                          if ns > 0 else set())

        new_opened = (new_longs - prev_longs) | (new_shorts - prev_shorts)
        closed     = (prev_longs - new_longs) | (prev_shorts - new_shorts)
        total_positions = len(new_longs) + len(new_shorts)

        longs  = grp[grp["symbol"].isin(new_longs)]
        shorts = grp[grp["symbol"].isin(new_shorts)]
        nl_act, ns_act = len(new_longs), len(new_shorts)

        # ── A1 side weights (gated) ──
        gate_on = True
        if gate_series is not None:
            g = gate_series.get(ts, False)
            gate_on = bool(g) if pd.notna(g) else False
        w_l_side, w_s_side = 0.5, 0.5
        if (a1_cfg is not None and gate_on and nl_act > 0 and ns_act > 0):
            if trend_dir > a1_cfg["trend_thr"]:
                w_s_side *= a1_cfg["weak_scale"]
            elif trend_dir < -a1_cfg["trend_thr"]:
                w_l_side *= a1_cfg["weak_scale"]
            tot = w_l_side + w_s_side
            w_l_side /= tot
            w_s_side /= tot

        # ── Within-side weights (G2 or equal) ──
        if g2_lookup is not None:
            lw = np.array([g2_weight(s, ts, g2_lookup, g2_honest)
                           for s in longs["symbol"]]) if nl_act else np.array([])
            sw = np.array([g2_weight(s, ts, g2_lookup, g2_honest)
                           for s in shorts["symbol"]]) if ns_act else np.array([])
            lw = lw / lw.sum() if lw.size and lw.sum() > 0 else lw
            sw = sw / sw.sum() if sw.size and sw.sum() > 0 else sw
            long_ret = float((longs["fwd_ret"].values * lw).sum()) if nl_act else 0.0
            short_ret = float((shorts["fwd_ret"].values * sw).sum()) if ns_act else 0.0
            side_lw = dict(zip(longs["symbol"], lw)) if nl_act else {}
            side_sw = dict(zip(shorts["symbol"], sw)) if ns_act else {}
        else:
            long_ret  = longs["fwd_ret"].mean() if nl_act > 0 else 0
            short_ret = shorts["fwd_ret"].mean() if ns_act > 0 else 0
            side_lw = {s: 1.0 / nl_act for s in new_longs} if nl_act else {}
            side_sw = {s: 1.0 / ns_act for s in new_shorts} if ns_act else {}

        if nl_act > 0 and ns_act > 0:
            gross_ret = w_l_side * long_ret - w_s_side * short_ret
        elif ns_act > 0:
            gross_ret = -short_ret
            w_l_side, w_s_side = 0.0, 1.0
        else:
            gross_ret = long_ret
            w_l_side, w_s_side = 1.0, 0.0
        gross_ret *= exposure

        # ── Execution delay noise (same rng stream as simulate_r121) ──
        if exec_delay_penalty > 0 and total_positions > 0:
            noise = rng.normal(0, exec_delay_penalty)
            gross_ret += noise

        # ── Costs (count-based, verbatim simulate_r121) ──
        if total_positions > 0:
            avg_weight = 1.0 / total_positions
            turnover_cost = sum(cost_fn(sym) * avg_weight for sym in new_opened)
            turnover_cost += sum(cost_fn(sym) * avg_weight for sym in closed)
            holding_cost = funding_per_12h * (rebal_hours / 12)
            total_cost = turnover_cost + holding_cost
        else:
            total_cost = 0.0

        net_ret = gross_ret - total_cost
        prev_longs, prev_shorts = new_longs, new_shorts
        prev_w = {}
        for s in new_longs:
            prev_w[s] = w_l_side * side_lw.get(s, 0.0) * exposure
        for s in new_shorts:
            prev_w[s] = -w_s_side * side_sw.get(s, 0.0) * exposure

        all_rets.append({
            "timestamp": ts, "gross_ret": gross_ret, "net_ret": net_ret,
            "cost": total_cost, "n_long": nl_act, "n_short": ns_act,
            "turnover": len(new_opened) + len(closed), "risk_off": False,
        })

    return pd.DataFrame(all_rets) if all_rets else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────
# Paired moving-block bootstrap
# ─────────────────────────────────────────────────────────────────────

def block_boot_p_delta(base: np.ndarray, alt: np.ndarray,
                       block_len: int = BLOCK_LEN, n_boot: int = N_BOOT,
                       seed: int = 7) -> Dict[str, float]:
    base = np.asarray(base, dtype=float)
    alt = np.asarray(alt, dtype=float)
    n = len(base)
    assert len(alt) == n
    n_blocks = int(np.ceil(n / block_len))
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    offs = np.arange(block_len)
    for k in range(n_boot):
        starts = rng.integers(0, n - block_len + 1, size=n_blocks)
        idx = (starts[:, None] + offs[None, :]).ravel()[:n]
        rb, ra = base[idx], alt[idx]
        sb = rb.mean() / (rb.std() + 1e-10) * np.sqrt(2 * 365)
        sa = ra.mean() / (ra.std() + 1e-10) * np.sqrt(2 * 365)
        deltas[k] = sa - sb
    return {"mean_delta": float(deltas.mean()),
            "p_pos": float((deltas > 0).mean()),
            "ci_low": float(np.percentile(deltas, 2.5)),
            "ci_high": float(np.percentile(deltas, 97.5))}


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 100)
    print("  R136 — R128-era claims re-tested: HONEST accounting (simulate_r121, "
          "1013 periods) x BOTH cost models")
    print("=" * 100)

    preds = pd.read_parquet(PREDS)
    regime_df = pd.read_parquet(REGIME).set_index("timestamp")
    print(f"  preds {len(preds):,} rows / {preds['symbol'].nunique()} syms; "
          f"regime {len(regime_df):,} rows")

    # Gate (frozen params, past-only)
    print(f"  Building persistence gate L={L_FROZEN} q={Q_FROZEN} (expanding quantile)...")
    regime_aug = r129.add_persistence(regime_df, lookback=L_FROZEN)
    persist_col = f"td_persist_{L_FROZEN}h"
    thr = r129.expanding_quantile_threshold(regime_aug[persist_col], Q_FROZEN,
                                            min_periods=720)
    gate_series = (regime_aug[persist_col] < thr)  # NaN-safe: NaN -> False
    print(f"  gate_on fraction (all hourly ts): {gate_series.mean()*100:.1f}%")

    # G2 lookups
    print("  Building G2 vol lookups (lb=14/28, legacy + honest)...")
    g2 = {
        ("lb14", False): build_g2_lookup(preds, 14, honest=False),
        ("lb14", True):  build_g2_lookup(preds, 14, honest=True),
        ("lb28", False): build_g2_lookup(preds, 28, honest=False),
        ("lb28", True):  build_g2_lookup(preds, 28, honest=True),
    }

    EXPERIMENTS: List[Dict[str, Any]] = [
        {"label": "BASE_CLOSE",                 "kw": {}},
        {"label": "BASE_SKIP",                  "kw": {"risk_off_mode": "skip"}},
        {"label": "A1 t0.25 s0.50",             "kw": {"a1_cfg": A1_BEST}},
        {"label": "A1 t0.25 s0.60 (frozen)",    "kw": {"a1_cfg": A1_FROZEN}},
        {"label": "GATED_A1 L720 q0.20",        "kw": {"a1_cfg": A1_FROZEN,
                                                       "gate_series": gate_series}},
        {"label": "G2 lb14 LEGACY(leak)",       "kw": {"g2_lookup": g2[("lb14", False)],
                                                       "g2_honest": False}},
        {"label": "G2 lb14 HONEST",             "kw": {"g2_lookup": g2[("lb14", True)],
                                                       "g2_honest": True}},
        {"label": "G2 lb28 LEGACY(leak)",       "kw": {"g2_lookup": g2[("lb28", False)],
                                                       "g2_honest": False}},
        {"label": "G2 lb28 HONEST",             "kw": {"g2_lookup": g2[("lb28", True)],
                                                       "g2_honest": True}},
        {"label": "A1+G2 lb28 LEGACY(leak)",    "kw": {"a1_cfg": A1_BEST,
                                                       "g2_lookup": g2[("lb28", False)],
                                                       "g2_honest": False}},
        {"label": "A1+G2 lb28 HONEST",          "kw": {"a1_cfg": A1_BEST,
                                                       "g2_lookup": g2[("lb28", True)],
                                                       "g2_honest": True}},
    ]

    results: List[Dict[str, Any]] = []
    ports: Dict[tuple, pd.DataFrame] = {}

    for cost_label, (cost_fn, funding) in COST_SETUPS.items():
        print("\n" + "=" * 100)
        print(f"  COST MODEL: {cost_label}  (funding {funding*1e4:.1f}bp/12h)")
        print("=" * 100)
        base_port = None
        for exp in EXPERIMENTS:
            port = simulate_r136(
                preds, regime_aug, 4, 2, dict(R114B_CFG),
                cutoff_on=0.9, cutoff_off=0.8,
                min_risk_off_periods=2, min_risk_on_periods=0,
                cost_fn=cost_fn, funding_per_12h=funding,
                exec_delay_penalty=0.0003,
                **exp["kw"])
            assert len(port) == 1013, \
                f"{exp['label']} ({cost_label}): {len(port)} periods != 1013"
            ports[(cost_label, exp["label"])] = port
            m = analyze_config(port, exp["label"])
            ns = m["net_sharpe"]
            if exp["label"] == "BASE_CLOSE":
                base_port = port
                base_ns = ns
                if cost_label == "prod_blended":
                    assert abs(ns - 2.831) < 1e-9 or round(ns, 3) == 2.831, \
                        f"S6 baseline {ns} != 2.831 — simulator not faithful!"
            delta = ns - base_ns
            boot = None
            if exp["label"] != "BASE_CLOSE":
                mrg = port[["timestamp", "net_ret"]].rename(columns={"net_ret": "alt"}).merge(
                    base_port[["timestamp", "net_ret"]].rename(columns={"net_ret": "base"}),
                    on="timestamp", how="inner")
                assert len(mrg) == 1013
                boot = block_boot_p_delta(mrg["base"].values, mrg["alt"].values)
            bs = (f"  boot: P(d>0)={boot['p_pos']:.3f} "
                  f"meanD={boot['mean_delta']:+.3f} "
                  f"CI[{boot['ci_low']:+.3f},{boot['ci_high']:+.3f}]") if boot else ""
            print(f"  {exp['label']:<26s} Net={ns:+.3f}  d={delta:+.3f}  "
                  f"Gross={m['gross_sharpe']:+.3f}  Ret={m['total_ret_pct']:>6.1f}%  "
                  f"DD={m['max_dd_pct']:>5.1f}%  Cost={m['total_cost_pct']:>5.2f}%  "
                  f"n={m['n_periods']}{bs}")
            results.append({
                "cost_model": cost_label, "label": exp["label"],
                "net_sharpe": ns, "delta_vs_base": round(delta, 3),
                "gross_sharpe": m["gross_sharpe"],
                "total_ret_pct": m["total_ret_pct"], "max_dd_pct": m["max_dd_pct"],
                "total_cost_pct": m["total_cost_pct"], "n_periods": m["n_periods"],
                "n_flat": m["n_flat"],
                "boot": boot,
            })

    # ── Final table ──
    print("\n" + "=" * 100)
    print("  FINAL TABLE (delta vs same-cost BASE_CLOSE; survives = S6 delta>0 AND "
          "P(d>0)>=0.85 under S6)")
    print("=" * 100)
    print(f"  {'experiment':<26s} {'lenient':>9s} {'d_len':>7s} {'S6':>7s} "
          f"{'d_S6':>7s} {'P(d>0)S6':>9s}  verdict")
    by = {(r["cost_model"], r["label"]): r for r in results}
    for exp in EXPERIMENTS:
        lab = exp["label"]
        rl, rs = by[("lenient_r68", lab)], by[("prod_blended", lab)]
        p = rs["boot"]["p_pos"] if rs["boot"] else float("nan")
        if lab == "BASE_CLOSE":
            verdict = "(reference)"
        elif rs["delta_vs_base"] > 0 and p >= 0.85:
            verdict = "SURVIVES"
        elif rs["delta_vs_base"] > 0:
            verdict = "positive but NOT significant"
        else:
            verdict = "DEAD under honest test"
        print(f"  {lab:<26s} {rl['net_sharpe']:>+9.3f} {rl['delta_vs_base']:>+7.3f} "
              f"{rs['net_sharpe']:>+7.3f} {rs['delta_vs_base']:>+7.3f} "
              f"{p:>9.3f}  {verdict}")

    with open("results_r136_s6_retest.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: results_r136_s6_retest.json   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
