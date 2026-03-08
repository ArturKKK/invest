#!/usr/bin/env python3
"""
Leverage + Selectivity Simulator.

Idea: instead of always trading top-5/bottom-5, only trade when the model
is VERY confident (high edge from median + seed agreement).  Fewer but
better trades → higher win rate.  Compensate reduced exposure with leverage.

Sweeps:
  - leverage: 1x, 2x, 3x, 5x, 7x, 10x
  - min_edge: minimum deviation from median score to include in portfolio
  - n_positions: 3, 5 per side

Outputs a grid: leverage × edge → {return, sharpe, maxdd, winrate, prob_ruin}
"""

import os, sys, json, warnings, itertools
import pandas as pd, numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_trading import (
    SYMBOLS, EXCLUDE_COLS, DEFAULT_RISK, HORIZON,
    fetch_ohlcv, build_features, cross_sectional_rank, load_lgb_models,
)

COST_SIDE = 0.0003 + 0.0001    # taker 3bps + slippage 1bp
FUNDING_8H = 0.0001             # funding rate per 8h (both sides)

ROOT = os.path.dirname(os.path.abspath(__file__))


def run_sim(
    df, models, mf, steps,
    capital=100.0, leverage=1.0, n_pos=5, kelly=1.0,
    min_edge=0.0,               # minimum |score - median| to include position
    max_seed_std=999.0,         # max per-symbol seed std (filter noisy predictions)
    rebal_h=12,
    liq_threshold=0.90,
):
    """Run one simulation pass. Returns dict of metrics."""
    equity = capital
    peak   = capital
    held_L: dict[str, float] = {}
    held_S: dict[str, float] = {}
    cum_cost = 0.0
    tot_trades = 0
    pnls: list[float] = []
    equities = [capital]
    skip_count = 0
    trade_count = 0
    liquidated = False

    for si in range(len(steps) - 1):
        ts0, ts1 = steps[si], steps[si + 1]
        snap0 = df[df["timestamp"] == ts0]
        snap1 = df[df["timestamp"] == ts1]
        if len(snap0) < 20 or len(snap1) < 20:
            continue

        px0 = dict(zip(snap0["symbol"], snap0["close"]))
        px1 = dict(zip(snap1["symbol"], snap1["close"]))

        # Predict with all seeds
        X = snap0[mf].values
        all_preds = np.array([m.predict(X) for m in models])  # (n_seeds, n_symbols)
        scores = all_preds.mean(axis=0)
        seed_std = all_preds.std(axis=0)    # per-symbol disagreement
        syms = snap0["symbol"].values

        # Edge = deviation from median
        median_score = np.median(scores)
        edges = scores - median_score

        # Select long candidates: positive edge > min_edge, low seed disagreement
        order_desc = np.argsort(-scores)
        order_asc  = np.argsort(scores)

        long_idx = []
        for idx in order_desc:
            if edges[idx] >= min_edge and seed_std[idx] <= max_seed_std:
                long_idx.append(idx)
            if len(long_idx) >= n_pos:
                break

        short_idx = []
        for idx in order_asc:
            if edges[idx] <= -min_edge and seed_std[idx] <= max_seed_std:
                short_idx.append(idx)
            if len(short_idx) >= n_pos:
                break

        nl = len(long_idx)
        ns = len(short_idx)
        if nl == 0 and ns == 0:
            skip_count += 1
            equities.append(equity)
            held_L.clear(); held_S.clear()
            continue

        new_L = set(syms[long_idx]) if nl > 0 else set()
        new_S = set(syms[short_idx]) if ns > 0 else set()

        # Confidence-weighted sizing (softmax)
        score_dict = dict(zip(syms, scores))

        def soft_weights(sym_set, sign=1):
            if not sym_set:
                return {}
            arr = np.array([sign * score_dict[s] for s in sym_set])
            arr = arr - arr.mean()
            w = np.exp(arr * 2)  # temperature=0.5
            w = w / w.sum()
            return dict(zip(sym_set, w))

        weight_L = soft_weights(new_L, sign=1)
        weight_S = soft_weights(new_S, sign=-1)

        # Costs (only on changes)
        open_L  = new_L - set(held_L)
        close_L = set(held_L) - new_L
        open_S  = new_S - set(held_S)
        close_S = set(held_S) - new_S

        total_alloc = equity * kelly * leverage
        n_total = max(nl + ns, 1)
        usd_per_avg = total_alloc / n_total
        n_changes = len(open_L) + len(close_L) + len(open_S) + len(close_S)
        step_cost = n_changes * usd_per_avg * COST_SIDE

        # Funding costs (only with leverage, both sides)
        if leverage > 1:
            funding_periods = rebal_h / 8
            step_cost += total_alloc * FUNDING_8H * funding_periods

        cum_cost += step_cost
        tot_trades += n_changes
        trade_count += 1

        # Allocate: proportional to number of positions per side
        alloc_L = total_alloc * nl / n_total if n_total > 0 else 0
        alloc_S = total_alloc * ns / n_total if n_total > 0 else 0

        # Forward PnL
        fwd_pnl = 0.0
        for sym in new_L:
            p0 = px0.get(sym, 0); p1 = px1.get(sym, p0)
            w = weight_L.get(sym, 1.0 / max(nl, 1))
            if p0 > 0:
                fwd_pnl += alloc_L * w * (p1 - p0) / p0
        for sym in new_S:
            p0 = px0.get(sym, 0); p1 = px1.get(sym, p0)
            w = weight_S.get(sym, 1.0 / max(ns, 1))
            if p0 > 0:
                fwd_pnl += alloc_S * w * (-(p1 - p0) / p0)

        equity += fwd_pnl - step_cost
        peak = max(peak, equity)

        if fwd_pnl != 0:
            pnls.append(fwd_pnl - step_cost)

        equities.append(equity)

        # Liquidation check
        if equity <= capital * (1 - liq_threshold):
            liquidated = True
            break

        # Update held
        held_L = {s: px1.get(s, 0) for s in new_L}
        held_S = {s: px1.get(s, 0) for s in new_S}

    # ── Metrics ───────────────────────────────────────────────────
    if not pnls or trade_count == 0:
        return None

    tot_ret = equity / capital - 1
    max_dd = min(
        (equities[i] / max(equities[:i] + [capital]) - 1)
        for i in range(1, len(equities))
    ) if len(equities) > 1 else 0
    w_trades = [p for p in pnls if p > 0]
    l_trades = [p for p in pnls if p < 0]
    wr = len(w_trades) / len(pnls) if pnls else 0
    a = np.array(pnls)
    sh = np.mean(a) / (np.std(a) + 1e-10) * np.sqrt(365 * 24 / rebal_h)
    ann_ret = tot_ret * (365 / 60)
    calmar = ann_ret / (abs(max_dd) + 1e-10) if max_dd < 0 else 999
    pf = sum(w_trades) / (abs(sum(l_trades)) + 1e-10) if l_trades else 999

    return {
        "equity_final": round(equity, 2),
        "return_pct": round(tot_ret * 100, 1),
        "ann_return_pct": round(ann_ret * 100, 0),
        "sharpe": round(sh, 2),
        "max_dd_pct": round(max_dd * 100, 1),
        "calmar": round(calmar, 2),
        "win_rate": round(wr * 100, 1),
        "profit_factor": round(pf, 2),
        "n_trades": len(pnls),
        "n_skipped": skip_count,
        "costs": round(cum_cost, 2),
        "liquidated": liquidated,
        "equities": equities,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--capital", type=float, default=100.0)
    ap.add_argument("--model-dir", type=str, default=None)
    ap.add_argument("--warmup", type=int, default=720)
    ap.add_argument("--rebal", type=int, default=12)
    args = ap.parse_args()

    # ── Load data & models ────────────────────────────────────────
    total_h = args.warmup + args.days * 24
    print(f"📊 Fetching {total_h}h of data ...")
    raw = fetch_ohlcv(SYMBOLS, total_h)
    if raw is None or len(raw) == 0:
        print("❌ fetch failed"); return
    print(f"   {raw.shape}, {raw['symbol'].nunique()} symbols")

    print("🔧 Building features ...")
    df = build_features(raw)
    fc = [c for c in df.columns if c not in EXCLUDE_COLS
          and not c.startswith("target_")
          and df[c].dtype in ("float64","float32","int64","int32")]
    df = cross_sectional_rank(df, fc)

    model_dir = args.model_dir
    if not model_dir:
        for d in ["results_v8", "results_v7", "results_v6"]:
            p = os.path.join(ROOT, d)
            if os.path.isdir(p) and any(f.endswith('.txt') for f in os.listdir(p)):
                model_dir = p; break
    if not model_dir:
        model_dir = os.path.join(ROOT, "results_v6")
    models = load_lgb_models(model_dir)
    mf = models[0].feature_name()
    for c in [c for c in mf if c not in df.columns]:
        df[c] = 0.0
    print(f"📡 {len(models)} models from {os.path.basename(model_dir)}, {len(mf)} features")

    all_ts = sorted(df["timestamp"].unique())
    sim_h = args.days * 24
    sim_start = max(0, len(all_ts) - sim_h)
    steps = all_ts[sim_start::args.rebal]
    print(f"   {len(steps)} rebalance steps")

    # ── Calibrate edge distribution ───────────────────────────────
    print("\n📐 Calibrating score distribution ...")
    edge_samples = []
    std_samples = []
    # Use LAST 30 steps for calibration (beginning may be warmup with sparse data)
    cal_start = max(0, len(steps) - 31)
    for si in range(cal_start, min(cal_start + 30, len(steps)-1)):
        ts0 = steps[si]
        snap = df[df["timestamp"] == ts0]
        if len(snap) < 20:
            continue
        X = snap[mf].values
        all_preds = np.array([m.predict(X) for m in models])
        scores = all_preds.mean(axis=0)
        median_s = np.median(scores)
        edges = np.abs(scores - median_s)
        edge_samples.extend(edges.tolist())
        std_samples.extend(all_preds.std(axis=0).tolist())

    if not edge_samples:
        print("   WARNING: no valid calibration steps, using defaults")
        edge_p50 = 0.001
        edge_p75 = 0.002
        edge_p90 = 0.003
        std_p50 = 0.01
    else:
        edge_arr = np.array(edge_samples)
        std_arr = np.array(std_samples)
    print(f"   Edge |score - median|: mean={edge_arr.mean():.5f}, "
          f"P50={np.percentile(edge_arr, 50):.5f}, "
          f"P75={np.percentile(edge_arr, 75):.5f}, "
          f"P90={np.percentile(edge_arr, 90):.5f}")
    print(f"   Seed std: mean={std_arr.mean():.5f}, "
          f"P50={np.percentile(std_arr, 50):.5f}, "
          f"P90={np.percentile(std_arr, 90):.5f}")

    edge_p50 = np.percentile(edge_arr, 50)
    edge_p75 = np.percentile(edge_arr, 75)
    edge_p90 = np.percentile(edge_arr, 90)
    std_p50 = np.percentile(std_arr, 50)

    # ── Sweep grid ────────────────────────────────────────────────
    leverages = [1, 2, 3, 5, 7, 10]
    edge_configs = [
        ("no_filter",  0.0,       999.0),
        ("P50_edge",   edge_p50,  999.0),
        ("P75_edge",   edge_p75,  999.0),
        ("P90_edge",   edge_p90,  999.0),
        ("P75+agree",  edge_p75,  std_p50),
    ]
    n_positions = [3, 5]

    print(f"\n{'='*95}")
    print(f"  LEVERAGE x SELECTIVITY SWEEP   |   ${args.capital:.0f} start   |   {args.days}d   |   {os.path.basename(model_dir)}")
    print(f"{'='*95}")

    results_grid = []

    for n_pos in n_positions:
        for edge_name, min_edge, max_std in edge_configs:
            print(f"\n  -- N={n_pos}+{n_pos}, filter={edge_name} (edge>{min_edge:.5f}, std<{max_std:.4f}) --")
            hdr = (f"  {'Lev':>4s} | {'Return':>8s} | {'Ann.':>6s} | {'Sharpe':>7s} | {'MaxDD':>7s} | "
                   f"{'Calmar':>7s} | {'WinRate':>7s} | {'PF':>5s} | {'#Tr':>4s} | {'Skip':>4s} | "
                   f"{'End$':>7s} | {'$Cost':>6s} |")
            print(hdr)
            print(f"  {'='*len(hdr)}")

            for lev in leverages:
                r = run_sim(df, models, mf, steps,
                            capital=args.capital, leverage=lev, n_pos=n_pos,
                            min_edge=min_edge, max_seed_std=max_std,
                            rebal_h=args.rebal)
                if r is None:
                    print(f"  {lev:>3}x | {'no trades':>8s}")
                    continue

                tag = " DEAD" if r["liquidated"] else ""
                star = ""
                if r["win_rate"] >= 65 and r["sharpe"] >= 1.5 and not r["liquidated"]:
                    star = " ***"
                elif r["win_rate"] >= 60 and r["sharpe"] >= 2 and not r["liquidated"]:
                    star = " **"
                elif r["win_rate"] >= 58 and r["sharpe"] >= 1.5 and not r["liquidated"]:
                    star = " *"

                print(f"  {lev:>3}x | {r['return_pct']:>+7.1f}% | {r['ann_return_pct']:>+5.0f}% | "
                      f"{r['sharpe']:>+7.2f} | {r['max_dd_pct']:>6.1f}% | "
                      f"{r['calmar']:>7.2f} | {r['win_rate']:>6.1f}% | {r['profit_factor']:>5.2f} | "
                      f"{r['n_trades']:>4d} | {r['n_skipped']:>4d} | "
                      f"${r['equity_final']:>6.0f} | ${r['costs']:>5.1f} |{tag}{star}")

                results_grid.append({
                    "n_pos": n_pos, "edge_filter": edge_name,
                    "min_edge": round(min_edge, 6), "max_std": round(max_std, 6),
                    "leverage": lev, **{k: v for k, v in r.items() if k != "equities"}
                })

    # ── Best configurations ───────────────────────────────────────
    print(f"\n{'='*95}")
    print("  BEST CONFIGURATIONS (not liquidated, DD > -80%)")
    print(f"{'='*95}")

    valid = [r for r in results_grid if not r["liquidated"] and r["max_dd_pct"] > -80]
    if valid:
        best_sh = max(valid, key=lambda r: r["sharpe"])
        print(f"\n  Best Sharpe: {best_sh['leverage']}x, N={best_sh['n_pos']}, "
              f"{best_sh['edge_filter']} -> Sharpe {best_sh['sharpe']}, "
              f"Ret {best_sh['return_pct']:+.1f}%, WR {best_sh['win_rate']:.0f}%, "
              f"DD {best_sh['max_dd_pct']:.1f}%")

        high_wr = [r for r in valid if r["sharpe"] > 1]
        if high_wr:
            best_wr = max(high_wr, key=lambda r: r["win_rate"])
            print(f"  Best WinRate (Sh>1): {best_wr['leverage']}x, N={best_wr['n_pos']}, "
                  f"{best_wr['edge_filter']} -> WR {best_wr['win_rate']:.0f}%, "
                  f"Sharpe {best_wr['sharpe']}, Ret {best_wr['return_pct']:+.1f}%, "
                  f"DD {best_wr['max_dd_pct']:.1f}%")

        safe = [r for r in valid if r["max_dd_pct"] > -30]
        if safe:
            best_ret = max(safe, key=lambda r: r["return_pct"])
            print(f"  Best Return (DD>-30%): {best_ret['leverage']}x, N={best_ret['n_pos']}, "
                  f"{best_ret['edge_filter']} -> Ret {best_ret['return_pct']:+.1f}%, "
                  f"DD {best_ret['max_dd_pct']:.1f}%, Sharpe {best_ret['sharpe']}, "
                  f"WR {best_ret['win_rate']:.0f}%")

        play = [r for r in valid if r["max_dd_pct"] > -50]
        if play:
            def play_score(r):
                return r["return_pct"] * (1 + max(r["sharpe"], 0)) / (1 + abs(r["max_dd_pct"]))
            best_play = max(play, key=play_score)
            print(f"\n  $100 Play Money Pick: {best_play['leverage']}x, "
                  f"N={best_play['n_pos']}, {best_play['edge_filter']}")
            print(f"     -> ${best_play['equity_final']:.0f} ({best_play['return_pct']:+.1f}%), "
                  f"WR {best_play['win_rate']:.0f}%, DD {best_play['max_dd_pct']:.1f}%, "
                  f"Sharpe {best_play['sharpe']}, PF {best_play['profit_factor']:.2f}")

    # ── Monte Carlo ───────────────────────────────────────────────
    base = run_sim(df, models, mf, steps,
                   capital=args.capital, leverage=1.0, n_pos=5,
                   min_edge=0.0, max_seed_std=999.0, rebal_h=args.rebal)
    if base and len(base["equities"]) > 10:
        base_eq = base["equities"]
        base_rets = [(base_eq[i] - base_eq[i-1]) / base_eq[i-1]
                     for i in range(1, len(base_eq)) if base_eq[i-1] > 0]
        if len(base_rets) > 5:
            print(f"\n{'='*95}")
            print("  MONTE CARLO: P(Ruin) over 500 steps (equity < $10)")
            print(f"{'='*95}")
            print(f"  {'Leverage':>8s} | {'P(Ruin)':>10s} | {'Median$':>10s} | {'P5$':>8s} | {'P95$':>8s} | {'Risk':>15s}")
            sep = f"  {'─'*8}-+-{'─'*10}-+-{'─'*10}-+-{'─'*8}-+-{'─'*8}-+-{'─'*15}"
            print(sep)

            ret_arr = np.array(base_rets)
            for lev in leverages:
                n_sims = 10000
                ruin = 0
                finals = []
                for _ in range(n_sims):
                    eq = float(args.capital)
                    path = np.random.choice(ret_arr, size=500, replace=True)
                    for r in path:
                        eq *= (1 + r * lev)
                        if eq < 10:
                            ruin += 1
                            break
                    finals.append(max(eq, 0))
                p_ruin = ruin / n_sims
                med = np.median(finals)
                p5 = np.percentile(finals, 5)
                p95 = np.percentile(finals, 95)
                risk = ("SAFE" if p_ruin < 0.05 else
                        "MODERATE" if p_ruin < 0.20 else
                        "DANGEROUS" if p_ruin < 0.50 else
                        "SUICIDAL")
                print(f"  {lev:>7}x | {p_ruin:>9.1%} | ${med:>9.0f} | ${p5:>7.0f} | ${p95:>7.0f} | {risk}")

    # ── Save ──────────────────────────────────────────────────────
    outdir = os.path.join(ROOT, "trading_logs")
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "leverage_sweep.json")
    with open(out_path, "w") as f:
        json.dump(results_grid, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print()


if __name__ == "__main__":
    main()
