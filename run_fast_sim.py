#!/usr/bin/env python3
"""
Fast Historical Simulation — backtest the live pipeline on recent data.

Walks through recent data in configurable steps, generates LGB signals,
and tracks portfolio PnL with realistic costs and hold-aware execution.

Key features:
  - Rebalance every N hours (default 12h, optimal from sweep analysis)
  - HOLD positions that remain in portfolio (save on costs)
  - Only pay transaction costs on position CHANGES
  - Realistic cost model: taker + slippage per side
  - Vol scaling + DD circuit-breaker

Usage:
  python run_fast_sim.py                                  # 14d, $1000, 12h rebal
  python run_fast_sim.py --days 30 --capital 500          # more days
  python run_fast_sim.py --days 30 --rebal 8 --npos 3    # custom params
"""

import os, sys, json, argparse, warnings
import pandas as pd, numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_trading import (
    SYMBOLS, EXCLUDE_COLS, DEFAULT_RISK, HORIZON,
    fetch_ohlcv, build_features, cross_sectional_rank, load_lgb_models,
)

COST_SIDE = 0.0003 + 0.0001          # taker 3bps + slippage 1bp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days",    type=int,   default=14)
    ap.add_argument("--capital", type=float, default=1000.0)
    ap.add_argument("--config",  type=str,   default=None)
    ap.add_argument("--rebal",   type=int,   default=12,
                    help="Rebalance interval in hours (default: 12)")
    ap.add_argument("--npos",    type=int,   default=None,
                    help="Positions per side (overrides config)")
    ap.add_argument("--kelly",   type=float, default=None,
                    help="Kelly fraction (overrides config)")
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

    total_h = args.warmup + args.days * 24
    sim_h   = args.days * 24

    print("=" * 70)
    print(f"  FAST SIMULATION")
    print(f"  {args.days}d | ${args.capital:,.0f} | rebal={rebal_h}h | "
          f"N={n_pos}+{n_pos} | kelly={kelly:.0%} | cost={COST_SIDE*1e4:.0f}bp/side")
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
    models = load_lgb_models(os.path.join(root, "results_v6"))
    if not models:
        print("❌ no models"); return
    mf = models[0].feature_name()
    for c in [c for c in mf if c not in df.columns]:
        df[c] = 0.0
    print(f"   {len(models)} models, {len(mf)} feats")

    # ── 4  timestamps (rebal_h apart) ─────────────────────────────
    all_ts = sorted(df["timestamp"].unique())
    sim_start = max(0, len(all_ts) - sim_h)
    steps = all_ts[sim_start::rebal_h]      # every rebal_h hours
    print(f"   {steps[0]} → {steps[-1]}  ({len(steps)} steps, {rebal_h}h apart)")

    # ── 5  simulate ───────────────────────────────────────────────
    print(f"\n{'─'*70}")
    equity   = args.capital
    peak     = args.capital
    stopped  = False
    ret_buf: list[float] = []

    held_L: dict[str, float] = {}     # symbol → entry_price
    held_S: dict[str, float] = {}
    results: list[dict] = []
    cum_cost = 0.0
    tot_trades = 0

    for si in range(len(steps) - 1):
        ts0, ts1 = steps[si], steps[si + 1]
        snap0 = df[df["timestamp"] == ts0]
        snap1 = df[df["timestamp"] == ts1]
        if len(snap0) < 20 or len(snap1) < 20:
            continue

        px0 = dict(zip(snap0["symbol"], snap0["close"]))
        px1 = dict(zip(snap1["symbol"], snap1["close"]))

        # ── mark-to-market held positions (from previous step) ────
        mtm_pnl = 0.0
        for sym, ep in held_L.items():
            p = px0.get(sym)
            if p and ep: mtm_pnl += (p - ep) / ep
        for sym, ep in held_S.items():
            p = px0.get(sym)
            if p and ep: mtm_pnl -= (p - ep) / ep

        n_held = len(held_L) + len(held_S)
        usd_old = (equity * kelly) / max(n_held, 1) if n_held else 0
        dollar_mtm = mtm_pnl * usd_old

        # ── predict & rank at ts0 ────────────────────────────────
        X = snap0[mf].values
        scores = np.mean([m.predict(X) for m in models], axis=0)
        syms = snap0["symbol"].values
        order = np.argsort(-scores)
        n = len(syms)
        nl = min(n_pos, n // 3)

        new_L = set(syms[order[:nl]])
        new_S = set(syms[order[-nl:]])

        # ── compute changes (costs only on traded positions) ──────
        open_L  = new_L - set(held_L)
        close_L = set(held_L) - new_L
        open_S  = new_S - set(held_S)
        close_S = set(held_S) - new_S

        usd_per = (equity * kelly) / max(2 * nl, 1)
        step_cost = (len(open_L) + len(close_L) + len(open_S) + len(close_S)) * usd_per * COST_SIDE
        cum_cost += step_cost
        tot_trades += len(open_L) + len(close_L) + len(open_S) + len(close_S)

        # ── PnL from ts0→ts1 for NEW portfolio ───────────────────
        fwd_pnl = 0.0
        for sym in new_L:
            p0 = px0.get(sym, 0); p1 = px1.get(sym, p0)
            if p0 > 0: fwd_pnl += usd_per * (p1 - p0) / p0
        for sym in new_S:
            p0 = px0.get(sym, 0); p1 = px1.get(sym, p0)
            if p0 > 0: fwd_pnl += usd_per * (-(p1 - p0) / p0)

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

        if si % 5 == 0 or si == len(steps) - 2:
            print(f"   {si:>4d}/{len(steps)-1} | ${equity:>8,.2f} | "
                  f"DD {dd:>6.1%} | L{len(new_L)} S{len(new_S)} | "
                  f"Δ{turn}")

    # ── 6  summary ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  RESULTS — {args.days}d, rebal={rebal_h}h, N={n_pos}+{n_pos}")
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
    elif tot_ret > 0:
        print("   🟡 Marginal — keep monitoring")
    else:
        print("   🔴 Unprofitable")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
