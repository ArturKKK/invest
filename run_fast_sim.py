#!/usr/bin/env python3
"""
Fast Historical Simulation — backtest the live pipeline on recent data.

Walks through recent data in configurable steps, generates LGB signals,
and tracks portfolio PnL with realistic costs and hold-aware execution.

Key features:
  - Rebalance every N hours (default 12h, optimal from sweep analysis)
  - HOLD positions that remain in portfolio (save on costs)
  - Only pay transaction costs on position CHANGES
  - Realistic cost model: taker + slippage per side + funding for leverage
  - Vol scaling + DD circuit-breaker
  - Edge-based selectivity: filter by |score − median| (P75/P90)
  - Leverage support for futures trading

Usage:
  python run_fast_sim.py                                  # 14d, $1000, 12h rebal
  python run_fast_sim.py --days 30 --capital 500          # more days
  python run_fast_sim.py --days 30 --rebal 8 --npos 3    # custom params
  python run_fast_sim.py --leverage 3 --edge-pct 75       # 3x leverage, P75 edge filter
  python run_fast_sim.py --ensemble --leverage 3 --rebal 24 --edge-boost  # recommended
"""

import os, sys, json, argparse, warnings
import pandas as pd, numpy as np
from datetime import datetime, timezone

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_trading import (
    SYMBOLS, EXCLUDE_COLS, DEFAULT_RISK, HORIZON,
    fetch_ohlcv, build_features, cross_sectional_rank, load_lgb_models,
    load_catboost_models,
)

COST_SIDE = 0.0003 + 0.0001          # taker 3bps + slippage 1bp
FUNDING_PER_8H = 0.0001              # ~1bp per 8h funding cost for leveraged positions

# ── Macro event calendar (FOMC, CPI, major crypto events) ─────────
# These are UTC dates of high-impact events where we reduce/skip positions
# to avoid tail risk.  Updated periodically.
MACRO_EVENTS = {
    # 2025 FOMC rate decisions (announcement ~18:00 UTC)
    '2025-01-29', '2025-03-19', '2025-05-07', '2025-06-18',
    '2025-07-30', '2025-09-17', '2025-10-29', '2025-12-17',
    # 2025 US CPI releases (~12:30 UTC)
    '2025-01-15', '2025-02-12', '2025-03-12', '2025-04-10',
    '2025-05-13', '2025-06-11', '2025-07-11', '2025-08-12',
    '2025-09-10', '2025-10-14', '2025-11-12', '2025-12-10',
    # 2026 FOMC (projected)
    '2026-01-28', '2026-03-18', '2026-04-29', '2026-06-17',
    '2026-07-29', '2026-09-16', '2026-10-28', '2026-12-16',
    # 2026 US CPI (projected)
    '2026-01-14', '2026-02-11', '2026-03-11', '2026-04-14',
    '2026-05-12', '2026-06-10', '2026-07-14', '2026-08-12',
    '2026-09-11', '2026-10-13', '2026-11-12', '2026-12-10',
}

def is_near_event(ts, hours_before=18, hours_after=6):
    """Check if timestamp is within danger zone around a macro event.
    Default: skip 18h before event (day before) to 6h after.
    With 24h rebalance, this means we go flat for the step spanning the event.
    """
    ts_dt = pd.Timestamp(ts).tz_localize(None) if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts)
    for evt_str in MACRO_EVENTS:
        evt = pd.Timestamp(evt_str)
        delta_h = (ts_dt - evt).total_seconds() / 3600
        if -hours_before <= delta_h <= hours_after:
            return True, evt_str
    return False, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days",    type=int,   default=14)
    ap.add_argument("--capital", type=float, default=1000.0)
    ap.add_argument("--config",  type=str,   default=None)
    ap.add_argument("--model-dir", type=str, default=None,
                    help="Model directory (default: results_v8 > results_v7 > results_v6)")
    ap.add_argument("--rebal",   type=int,   default=12,
                    help="Rebalance interval in hours (default: 12)")
    ap.add_argument("--npos",    type=int,   default=None,
                    help="Positions per side (overrides config)")
    ap.add_argument("--kelly",   type=float, default=None,
                    help="Kelly fraction (overrides config)")
    ap.add_argument("--leverage", type=float, default=1.0,
                    help="Leverage multiplier (default: 1.0, e.g. 3 for 3x)")
    ap.add_argument("--edge-pct", type=int,  default=0, choices=[0, 50, 75, 90],
                    help="Edge percentile filter: 0=off, 75=P75 (recommended)")
    ap.add_argument("--min-edge", type=float, default=0.0,
                    help="Manual min edge threshold (overrides --edge-pct)")
    ap.add_argument("--ensemble", action="store_true",
                    help="Ensemble v6+v7 models (average scores from both)")
    ap.add_argument("--edge-boost", action="store_true",
                    help="Edge-proportional sizing: high-edge positions get more weight")
    ap.add_argument("--no-conf", action="store_true",
                    help="Disable confidence weighting (for A/B testing)")
    ap.add_argument("--adaptive-rebal", action="store_true",
                    help="Adaptive rebalance: base period + early rebal on strong signals")
    ap.add_argument("--dynamic-lev", action="store_true",
                    help="Dynamic leverage: base lev normally, scale up on strong edge")
    ap.add_argument("--max-lev", type=float, default=7.0,
                    help="Max leverage for dynamic-lev mode (default: 7)")
    ap.add_argument("--event-filter", action="store_true",
                    help="Reduce positions near FOMC/CPI events to avoid tail risk")
    ap.add_argument("--warmup",  type=int,   default=720)
    args = ap.parse_args()
    root = os.path.dirname(os.path.abspath(__file__))

    # ── risk cfg ──────────────────────────────────────────────────
    risk = DEFAULT_RISK.copy()
    for p in [args.config,
              os.path.join(root, "results_risk_study", "optimal_config.json")]:
        if p and os.path.exists(p):
            with open(p) as f: risk.update(json.load(f))
            print(f"   Risk config: {os.path.basename(p)}")
            break

    n_pos   = args.npos  or risk["n_long"]
    kelly   = args.kelly or risk["kelly_frac"]
    vol_tgt = risk["vol_target"]
    dd_stop = risk["dd_stop"]
    dd_resume = risk["dd_resume"]
    vol_lb  = risk.get("vol_lookback", 50)
    rebal_h = args.rebal
    leverage = args.leverage
    min_edge = args.min_edge            # will be calibrated later if edge_pct > 0

    total_h = args.warmup + args.days * 24
    sim_h   = args.days * 24

    lev_str = f"{leverage:.0f}x" if leverage >= 1 else f"{leverage:.1f}x"
    edge_str = f"P{args.edge_pct}" if args.edge_pct > 0 else (
        f"edge>{min_edge:.4f}" if min_edge > 0 else "off")
    boost_str = "boost" if args.edge_boost else ""
    adapt_str = "adaptive" if args.adaptive_rebal else ""
    dynlev_str = f"dynlev→{args.max_lev:.0f}x" if args.dynamic_lev else ""
    evtfilt_str = "evtfilt" if args.event_filter else ""
    mode_parts = [s for s in [edge_str if edge_str != 'off' else '', boost_str, adapt_str, dynlev_str, evtfilt_str] if s]
    mode_str = '+'.join(mode_parts) if mode_parts else 'baseline'
    print("=" * 70)
    print(f"  FAST SIMULATION")
    print(f"  {args.days}d | ${args.capital:,.0f} | rebal={rebal_h}h | "
          f"N={n_pos}+{n_pos} | kelly={kelly:.0%} | cost={COST_SIDE*1e4:.0f}bp/side")
    print(f"  leverage={lev_str} | mode={mode_str}")
    print("=" * 70)

    # ── 1  data ───────────────────────────────────────────────────
    print(f"\n📊 Fetching {total_h}h …")
    raw = fetch_ohlcv(SYMBOLS, total_h)
    if raw is None or len(raw) == 0:
        print("❌ fetch failed"); return
    print(f"   {raw.shape}, {raw['symbol'].nunique()} symbols")

    # ── 2  features ───────────────────────────────────────────────
    print("🔧 Features …")
    df = build_features(raw)
    fc = [c for c in df.columns if c not in EXCLUDE_COLS
          and not c.startswith("target_")
          and df[c].dtype in ("float64","float32","int64","int32")]
    df = cross_sectional_rank(df, fc)

    # ── 3  models ─────────────────────────────────────────────────
    print("📡 Models …")
    model_groups = []   # list of (models, feature_names) tuples

    if args.ensemble:
        # Load LGB v6, v7 and CatBoost models
        for d in ["results_v6", "results_v7"]:
            p = os.path.join(root, d)
            if os.path.isdir(p):
                ms = load_lgb_models(p)
                if ms:
                    mf_g = ms[0].feature_name()
                    for c in [c for c in mf_g if c not in df.columns]:
                        df[c] = 0.0
                    model_groups.append((ms, mf_g))
                    print(f"   {d}: {len(ms)} LGB models, {len(mf_g)} feats")
        # CatBoost ensemble member
        cb_dir = os.path.join(root, "results_catboost")
        if os.path.isdir(cb_dir):
            try:
                ms = load_catboost_models(cb_dir)
                if ms:
                    fn_path = os.path.join(cb_dir, 'feature_names.json')
                    if os.path.exists(fn_path):
                        with open(fn_path) as _f:
                            mf_g = json.load(_f)
                    else:
                        mf_g = ms[0].feature_names_
                    for c in [c for c in mf_g if c not in df.columns]:
                        df[c] = 0.0
                    model_groups.append((ms, mf_g))
                    print(f"   results_catboost: {len(ms)} CB models, {len(mf_g)} feats")
            except ImportError:
                print("   ⚠️  catboost not installed, skipping CatBoost models")
        if not model_groups:
            print("❌ no models for ensemble"); return
    else:
        model_dir = args.model_dir
        if not model_dir:
            for d in ["results_v8", "results_v7", "results_v6"]:
                p = os.path.join(root, d)
                if os.path.isdir(p) and any(f.endswith('.txt') for f in os.listdir(p)):
                    model_dir = p; break
        if not model_dir:
            model_dir = os.path.join(root, "results_v6")
        models = load_lgb_models(model_dir)
        if not models:
            print("❌ no models"); return
        mf = models[0].feature_name()
        for c in [c for c in mf if c not in df.columns]:
            df[c] = 0.0
        model_groups.append((models, mf))
        print(f"   {len(models)} models, {len(mf)} feats")

    def predict_ensemble(snap_df):
        """Average predictions across all model groups. Returns (scores, confidence)."""
        all_scores = []
        all_individual = []  # individual model predictions
        for ms, mf_g in model_groups:
            X = snap_df[mf_g].values
            preds = [m.predict(X) for m in ms]
            all_individual.extend(preds)
            scores = np.mean(preds, axis=0)
            all_scores.append(scores)
        mean_scores = np.mean(all_scores, axis=0)
        # Confidence = model agreement. Normalize each model's preds before computing std
        if len(all_individual) > 1:
            normed = []
            for p in all_individual:
                normed.append((p - p.mean()) / (p.std() + 1e-10))
            model_std = np.std(normed, axis=0)
            confidence = 1.0 / (1.0 + model_std)
        else:
            confidence = np.ones_like(mean_scores) * 0.5
        return mean_scores, confidence

    # ── 4  timestamps (rebal_h apart) ─────────────────────────────
    all_ts = sorted(df["timestamp"].unique())
    sim_start = max(0, len(all_ts) - sim_h)
    steps = all_ts[sim_start::rebal_h]      # every rebal_h hours
    print(f"   {steps[0]} → {steps[-1]}  ({len(steps)} steps, {rebal_h}h apart)")

    # ── 4b  calibrate edge threshold ──────────────────────────────
    edge_p75 = 0.0   # used by edge-boost sizing
    need_calibrate = (args.edge_pct > 0 and min_edge == 0.0) or args.edge_boost or args.adaptive_rebal
    if need_calibrate:
        label = f"P{args.edge_pct}" if args.edge_pct > 0 else "P75 (for boost/adaptive)"
        print(f"\n📐 Calibrating edge distribution ({label}) ...")
        edge_samples = []
        cal_steps = steps[-min(30, len(steps)):]  # last 30 steps (avoid warmup)
        for ts in cal_steps:
            snap = df[df["timestamp"] == ts]
            if len(snap) < 20:
                continue
            scores, _ = predict_ensemble(snap)
            median_s = np.median(scores)
            edges_abs = np.abs(scores - median_s)
            edge_samples.extend(edges_abs.tolist())
        if edge_samples:
            if args.edge_pct > 0:
                min_edge = float(np.percentile(edge_samples, args.edge_pct))
            edge_p75 = float(np.percentile(edge_samples, 75))
            edge_p90 = float(np.percentile(edge_samples, 90))
            print(f"   P75 edge = {edge_p75:.5f}, P90 = {edge_p90:.5f}")
            if min_edge > 0:
                print(f"   Filter threshold (P{args.edge_pct}) = {min_edge:.5f}")
            print(f"   ({len(edge_samples)} samples, {len(cal_steps)} steps)")
        else:
            print("   ⚠️  No calibration data, edge features disabled")
            min_edge = 0.0
            edge_p75 = 0.0
            edge_p90 = 0.0

    # ── 5  simulate ───────────────────────────────────────────────
    print(f"\n{'─'*70}")
    equity   = args.capital
    peak     = args.capital
    stopped  = False
    ret_buf: list[float] = []
    skip_count = 0                    # steps where edge filter blocked all positions
    early_rebal_count = 0             # adaptive early rebalances triggered
    event_reduce_count = 0            # steps where event filter reduced leverage

    held_L: dict[str, float] = {}     # symbol → entry_price
    held_S: dict[str, float] = {}
    results: list[dict] = []
    cum_cost = 0.0
    tot_trades = 0

    # Build step schedule for adaptive rebalance
    if args.adaptive_rebal:
        # Base rebalance every rebal_h; also check at half-intervals for P90+ opportunities
        check_interval = max(rebal_h // 2, 4)  # check every half-period (min 4h)
        all_check_ts = all_ts[sim_start::check_interval]
        # Mark which are "base" rebalance times vs "check" times
        base_set = set(all_ts[sim_start::rebal_h])
        step_schedule = [(ts, ts in base_set) for ts in all_check_ts]
    else:
        step_schedule = [(ts, True) for ts in steps]  # all are base rebalances

    for si in range(len(step_schedule) - 1):
        ts0 = step_schedule[si][0]
        is_base = step_schedule[si][1]

        # Find next timestamp in schedule
        ts1 = step_schedule[si + 1][0]

        snap0 = df[df["timestamp"] == ts0]
        snap1 = df[df["timestamp"] == ts1]
        if len(snap0) < 20 or len(snap1) < 20:
            continue

        px0 = dict(zip(snap0["symbol"], snap0["close"]))
        px1 = dict(zip(snap1["symbol"], snap1["close"]))

        # ── predict & rank at ts0 ────────────────────────────────
        scores, confidence = predict_ensemble(snap0)
        syms = snap0["symbol"].values
        median_score = np.median(scores)
        edges = scores - median_score
        abs_edges = np.abs(edges)

        # Adaptive rebalance check: skip non-base steps unless strong signal
        if args.adaptive_rebal and not is_base:
            max_edge = np.max(abs_edges)
            if max_edge < edge_p90:
                # No exceptional signal → don't rebalance, just hold
                # Still compute PnL for held positions
                mtm_pnl = 0.0
                n_held = len(held_L) + len(held_S)
                if n_held > 0:
                    alloc_per = (equity * kelly * leverage) / n_held
                    for sym, ep in held_L.items():
                        p0 = px0.get(sym); p1 = px1.get(sym, p0)
                        if p0 and p1 and ep:
                            mtm_pnl += alloc_per * (p1 - p0) / p0
                    for sym, ep in held_S.items():
                        p0 = px0.get(sym); p1 = px1.get(sym, p0)
                        if p0 and p1 and ep:
                            mtm_pnl -= alloc_per * (p1 - p0) / p0
                    # Funding cost for held period
                    if leverage > 1:
                        hours_held = check_interval
                        fc = (equity * kelly * leverage) * FUNDING_PER_8H * (hours_held / 8.0)
                        mtm_pnl -= fc
                    equity += mtm_pnl
                    peak = max(peak, equity)
                    # Update entry prices
                    held_L = {s: px1.get(s, 0) for s in held_L}
                    held_S = {s: px1.get(s, 0) for s in held_S}
                continue
            else:
                early_rebal_count += 1

        # Edge filtering: select positions based on edge
        order_desc = np.argsort(-scores)
        order_asc  = np.argsort(scores)

        # ── Dynamic leverage: scale leverage based on edge strength ──
        if args.dynamic_lev and edge_p75 > 0 and edge_p90 > 0:
            max_abs_edge = np.max(abs_edges)
            # Require: edge > P90 AND recent returns positive (momentum)
            recent_ok = len(ret_buf) >= 3 and np.mean(ret_buf[-3:]) > 0
            # Also: current DD must be shallow (not in drawdown recovery)
            dd_now = equity / peak - 1 if peak > 0 else 0
            dd_ok = dd_now > -0.08  # only scale up if DD < 8%

            if max_abs_edge >= edge_p90 and recent_ok and dd_ok:
                # Gradual scale: P90 edge → base+25%, P90×1.5 → max_lev
                overshoot = (max_abs_edge - edge_p90) / (edge_p90 * 0.5 + 1e-10)
                lev_ratio = min(overshoot, 1.0)
                step_leverage = leverage + lev_ratio * (args.max_lev - leverage)
            else:
                step_leverage = leverage
        else:
            step_leverage = leverage

        # ── Event filter: reduce leverage near macro events ──────
        if args.event_filter:
            near_evt, evt_name = is_near_event(ts0)
            if near_evt:
                step_leverage = max(1.0, step_leverage * 0.3)  # reduce to 30% of planned lev
                event_reduce_count += 1

        if min_edge > 0:
            # Select only positions with sufficient edge
            long_idx = []
            for idx in order_desc:
                if edges[idx] >= min_edge:
                    long_idx.append(idx)
                if len(long_idx) >= n_pos:
                    break
            short_idx = []
            for idx in order_asc:
                if edges[idx] <= -min_edge:
                    short_idx.append(idx)
                if len(short_idx) >= n_pos:
                    break
            new_L = set(syms[long_idx]) if long_idx else set()
            new_S = set(syms[short_idx]) if short_idx else set()
            nl = len(long_idx)
        else:
            n = len(syms)
            nl = min(n_pos, n // 3)
            new_L = set(syms[order_desc[:nl]])
            new_S = set(syms[order_asc[:nl]])

        if len(new_L) == 0 and len(new_S) == 0:
            # No positions pass the edge filter — skip step
            skip_count += 1
            results.append(dict(step=si, ts=str(ts0), pnl=0,
                                eq=round(equity, 2), dd=round(equity/peak-1, 4),
                                nL=0, nS=0, turn=0, skipped=True))
            held_L.clear(); held_S.clear()
            continue

        # Confidence-weighted sizing with optional edge boost
        score_dict = dict(zip(syms, scores))
        edge_dict = dict(zip(syms, abs_edges))
        conf_dict = dict(zip(syms, confidence))

        def compute_weights(symbols, is_long=True):
            """Compute position weights with optional edge-boost × confidence."""
            if len(symbols) == 0:
                return {}
            syms_list = list(symbols)
            if args.edge_boost and edge_p75 > 0:
                # Edge-proportional: high-edge positions get more weight
                # boost = 1 for edge at P50, ~2 at P75, ~3 at P90
                raw_w = []
                for s in syms_list:
                    e = edge_dict.get(s, 0)
                    ratio = e / edge_p75           # 1.0 at P75
                    boost = 1.0 + min(ratio, 3.0)  # cap at 4x (to avoid over-concentration)
                    # Multiply by confidence: high-agreement → more capital
                    c = conf_dict.get(s, 0.5) if not getattr(args, 'no_conf', False) else 1.0
                    raw_w.append(boost * c)
                raw_w = np.array(raw_w)
                w = raw_w / raw_w.sum()
            else:
                # Original softmax-like weighting
                if is_long:
                    arr = np.array([score_dict[s] for s in syms_list])
                else:
                    arr = np.array([-score_dict[s] for s in syms_list])
                arr = arr - arr.mean()
                w = np.exp(arr * 2)
                w = w / w.sum()
            return dict(zip(syms_list, w))

        weight_L = compute_weights(new_L, is_long=True)
        weight_S = compute_weights(new_S, is_long=False)

        # ── compute changes (costs only on traded positions) ──────
        open_L  = new_L - set(held_L)
        close_L = set(held_L) - new_L
        open_S  = new_S - set(held_S)
        close_S = set(held_S) - new_S

        # Compute actual hours between ts0→ts1 for funding calc
        hours_between = max(1, int((ts1 - ts0).total_seconds() / 3600)) if hasattr(ts1, 'total_seconds') else rebal_h

        total_alloc = equity * kelly * step_leverage
        half_alloc = total_alloc / 2  # half for longs, half for shorts

        # Costs: estimate average position size for cost calc
        n_active = max(len(new_L) + len(new_S), 1)
        usd_per_avg = total_alloc / n_active
        step_cost = (len(open_L) + len(close_L) + len(open_S) + len(close_S)) * usd_per_avg * COST_SIDE

        # Funding cost for leveraged positions (proportional to hold time)
        if leverage > 1:
            funding_periods = rebal_h / 8.0  # how many 8h funding intervals
            funding_cost = total_alloc * FUNDING_PER_8H * funding_periods
            step_cost += funding_cost
        cum_cost += step_cost
        tot_trades += len(open_L) + len(close_L) + len(open_S) + len(close_S)

        # ── PnL from ts0→ts1 for NEW portfolio ───────────────────
        fwd_pnl = 0.0
        for sym in new_L:
            p0 = px0.get(sym, 0); p1 = px1.get(sym, p0)
            w = weight_L.get(sym, 1.0 / max(len(new_L), 1))
            if p0 > 0: fwd_pnl += half_alloc * w * (p1 - p0) / p0
        for sym in new_S:
            p0 = px0.get(sym, 0); p1 = px1.get(sym, p0)
            w = weight_S.get(sym, 1.0 / max(len(new_S), 1))
            if p0 > 0: fwd_pnl += half_alloc * w * (-(p1 - p0) / p0)

        equity += fwd_pnl - step_cost
        peak = max(peak, equity)
        dd = equity / peak - 1

        if fwd_pnl != 0:
            prev_eq = equity - fwd_pnl + step_cost
            if prev_eq > 0:
                ret_buf.append((fwd_pnl - step_cost) / prev_eq)
                ret_buf = ret_buf[-200:]

        # ── dd breaker ────────────────────────────────────────────
        if stopped:
            if dd > dd_resume: stopped = False
            else:
                results.append(dict(step=si, ts=str(ts0), pnl=0,
                                    eq=round(equity,2), dd=round(dd,4),
                                    nL=0, nS=0, turn=0, stopped=True))
                held_L.clear(); held_S.clear()
                continue
        if dd < dd_stop:
            stopped = True
            held_L.clear(); held_S.clear()
            continue

        turn = len(open_L) + len(close_L) + len(open_S) + len(close_S)
        results.append(dict(step=si, ts=str(ts0),
                            pnl=round(fwd_pnl - step_cost, 2),
                            eq=round(equity, 2), dd=round(dd, 4),
                            nL=len(new_L), nS=len(new_S), turn=turn))

        # Update held for next step mtm
        held_L = {s: px1.get(s, 0) for s in new_L}
        held_S = {s: px1.get(s, 0) for s in new_S}

        if si % 5 == 0 or si == len(step_schedule) - 2:
            print(f"   {si:>4d}/{len(step_schedule)-1} | ${equity:>8,.2f} | "
                  f"DD {dd:>6.1%} | L{len(new_L)} S{len(new_S)} | "
                  f"Δ{turn}")

    # ── 6  summary ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  RESULTS — {args.days}d, rebal={rebal_h}h, N={n_pos}+{n_pos}, "
          f"lev={lev_str}, mode={mode_str}")
    print(f"{'='*70}")

    tot_ret = equity / args.capital - 1
    max_dd  = min(r["dd"] for r in results) if results else 0
    pnls    = [r["pnl"] for r in results if r["pnl"] != 0]

    if pnls:
        w = [p for p in pnls if p > 0]
        l = [p for p in pnls if p < 0]
        wr = len(w) / len(pnls)
        a  = np.array(pnls)
        sh = np.mean(a) / (np.std(a) + 1e-10) * np.sqrt(365 * 24 / rebal_h)

        ann_ret = tot_ret * (365 / args.days)
        calmar = ann_ret / (abs(max_dd) + 1e-10) if max_dd < 0 else 999

        print(f"\n   Start:      ${args.capital:,.0f}")
        print(f"   End:        ${equity:,.2f}")
        print(f"   Return:     {tot_ret:+.1%}  (ann. ~{ann_ret:+.0%})")
        print(f"   Max DD:     {max_dd:.1%}")
        print(f"   Sharpe:     {sh:+.2f}")
        print(f"   Calmar:     {calmar:.2f}")
        print(f"   Win Rate:   {wr:.0%}  ({len(w)}W / {len(l)}L)")
        if w: print(f"   Avg Win:    ${np.mean(w):+.2f}")
        if l: print(f"   Avg Loss:   ${np.mean(l):+.2f}")
        if l: print(f"   PF:         {sum(w)/(abs(sum(l))+1e-10):.2f}")
        print(f"   Trades:     {tot_trades}")
        print(f"   Costs:      ${cum_cost:,.2f}  ({cum_cost/args.capital*100:.1f}%)")
        if skip_count > 0:
            print(f"   Skipped:    {skip_count} steps (no edge)")
        if early_rebal_count > 0:
            print(f"   Early rebal:{early_rebal_count} (adaptive P90+ triggers)")
        if event_reduce_count > 0:
            print(f"   Event filt: {event_reduce_count} steps (leverage reduced near FOMC/CPI)")
        if leverage > 1:
            print(f"   Leverage:   {lev_str}")
            liq_dd = -1.0 / leverage  # approximate liquidation DD
            print(f"   Liq. level: {liq_dd:.0%} DD (approx)")
    else:
        print("\n   No trades.")

    # ── save ──────────────────────────────────────────────────────
    outdir = os.path.join(root, "trading_logs"); os.makedirs(outdir, exist_ok=True)
    ep = os.path.join(outdir, "fast_sim_equity.csv")
    pd.DataFrame(results).to_csv(ep, index=False)

    # ── ascii chart ───────────────────────────────────────────────
    if len(results) > 3:
        eqs = [r["eq"] for r in results]
        s = eqs[::max(1, len(eqs)//50)]
        mn, mx = min(s), max(s)
        rng = mx - mn or 1
        print(f"\n   📈 Equity (${mn:.0f}–${mx:.0f}):")
        for i, e in enumerate(s):
            f_ = int((e - mn) / rng * 40)
            if i % max(1, len(s)//8) == 0 or i == len(s)-1:
                print(f"      ${e:>8,.0f} |{'█'*f_}{'░'*(40-f_)}|")

    print(f"\n   Saved: {ep}")
    print(f"\n{'='*70}")
    if tot_ret > 0.02 and max_dd > -0.15:
        print("   🟢 PROFITABLE — go live")
    elif tot_ret > 0 and max_dd > (-1.0 / leverage if leverage > 1 else -0.3):
        print("   🟡 Marginal — keep monitoring")
    elif leverage > 1 and max_dd < -1.0 / leverage:
        print("   💀 LIQUIDATED — reduce leverage!")
    else:
        print("   🔴 Unprofitable")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
