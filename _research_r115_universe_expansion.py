#!/usr/bin/env python3
"""
R115 — Universe Expansion (35 → 80+ coins).

Steps:
  1. Discover all Binance USDT-M perpetuals via ccxt
  2. Download OHLCV for symbols missing from data/raw/
  3. Load expanded universe (skip SYM_35 filter)
  4. Build all features (CG only available for ~50 coins → NaN ok for trees)
  5. Add point-in-time ADV (7d rolling dollar volume) column
  6. Simulate with volume-filtered portfolio selection
  7. Compare vs R113 baseline (SYM_35, 4L/2S)

Grid:
  - min_adv_usd ∈ {5M, 10M, 20M}  (different effective universes)
  - n_long/n_short ∈ {(4, 2), (6, 3)}
  - cutoff_on = 0.9 (fixed)

Acceptance:
  - Sharpe >= baseline + 0.05  OR  Calmar +10% at Sharpe >= baseline - 0.05
  - Costs not dramatically worse (cost% +<15% relative)
"""
import time, json, os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from typing import Set, Dict, List
warnings.filterwarnings("ignore")

from _research_r22_models import SEEDS, log
from _research_round7 import SYM_35
from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal
from _research_r22_models import add_new_features, build_r19_features
from _research_r30b_fixed import add_extra_features_clean, compute_regime_extended
from _research_r33_creative_features import add_r33_features
from _research_r35_new_features import add_r35_features, MARKET_LEVEL_FEATURES
from _research_r47_coinglass import load_cg_daily, compute_cg_features
from _research_r68_continuous_wf import (
    add_cg_features, CHAMPION_FEAT_31, CONTINUOUS_WINDOWS, PROD_CFG,
    train_ensemble, sharpe, _cost_for_sym,
    TIER1_SYMS, TIER2_SYMS,
)
from _research_r113_trend_cutoff_reopt import simulate_v2, analyze_config, print_result


DATA_DIR = Path("data/raw")


# ─── Phase 1: Download expanded universe ────────────────────────

def download_expanded_universe():
    """Download OHLCV for all Binance USDT-M perp coins not yet in data/raw/."""
    import ccxt

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Existing symbols
    existing = set()
    for f in DATA_DIR.glob("*_1h.parquet"):
        sym = f.stem.replace("_1h", "").replace("_", "/")
        existing.add(sym)
    log(f"  Existing OHLCV symbols: {len(existing)}")

    # Discover Binance USDT-M perpetuals
    try:
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},
        })
        markets = exchange.load_markets()
    except Exception as e:
        log(f"  ⚠ Cannot load Binance markets: {e}")
        log(f"  Proceeding with {len(existing)} existing symbols")
        return existing

    perp_bases = set()
    for sym, info in markets.items():
        if (info.get('active') and info.get('linear')
                and info.get('quote') == 'USDT'):
            perp_bases.add(info['base'])

    target = {f"{b}/USDT" for b in perp_bases}
    missing = sorted(target - existing)
    log(f"  Binance USDT-M perps: {len(perp_bases)} coins")
    log(f"  Missing OHLCV: {len(missing)}")

    if not missing:
        return existing | target

    # Download missing OHLCV (spot)
    spot = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'},
    })
    SINCE = spot.parse8601('2021-01-01T00:00:00Z')
    downloaded = 0

    for sym in missing:
        fname = DATA_DIR / f"{sym.replace('/', '_')}_1h.parquet"
        try:
            all_data = []
            current_since = SINCE
            while True:
                ohlcv = spot.fetch_ohlcv(sym, '1h', since=current_since,
                                         limit=1000)
                if not ohlcv:
                    break
                all_data.extend(ohlcv)
                last_ts = ohlcv[-1][0]
                if last_ts == current_since:
                    break
                current_since = last_ts + 1
                time.sleep(0.1)

            if len(all_data) < 2000:
                continue  # skip coins with < ~3 months data

            d = pd.DataFrame(all_data,
                             columns=['timestamp', 'open', 'high', 'low',
                                      'close', 'volume'])
            d['timestamp'] = pd.to_datetime(d['timestamp'], unit='ms', utc=True)
            d = (d.drop_duplicates(subset='timestamp')
                   .sort_values('timestamp')
                   .reset_index(drop=True))
            d.to_parquet(fname, index=False)
            downloaded += 1
            if downloaded % 10 == 0:
                log(f"  Downloaded {downloaded} symbols...")
        except Exception:
            continue  # skip unavailable pairs

    log(f"  Downloaded {downloaded} new symbols")

    # Reload
    all_available = set()
    for f in DATA_DIR.glob("*_1h.parquet"):
        sym = f.stem.replace("_1h", "").replace("_", "/")
        all_available.add(sym)
    return all_available


# ─── Phase 2: Load expanded data ────────────────────────────────

def load_data_expanded(symbol_list=None):
    """
    Load data like load_data() but with expanded (or custom) symbol list.
    CoinGlass features NaN for non-CG symbols → tree models handle it.
    """
    log("=" * 70)
    log("  LOADING DATA (expanded universe)")
    log("=" * 70)

    ohlcv = load_ohlcv()
    all_syms = sorted(ohlcv["symbol"].unique().tolist())
    log(f"  All OHLCV symbols: {len(all_syms)}")

    if symbol_list is not None:
        ohlcv = ohlcv[ohlcv["symbol"].isin(symbol_list)]
        log(f"  Filtered to {ohlcv['symbol'].nunique()} symbols")

    derivs = load_derivatives()
    df = build_features_minimal(ohlcv, derivs)
    df = build_r19_features(df)
    df, _ = add_new_features(df)
    # Skip SYM_35 filter
    df = add_extra_features_clean(df)
    df = add_r33_features(df)

    # Regime from BTC only (independent of universe)
    regime_df = compute_regime_extended(df)

    df, _ = add_r35_features(df)

    # CoinGlass (available for ~50 coins, NaN for rest)
    try:
        cg = load_cg_daily()
        cg_feats = compute_cg_features(cg)
        df, _, _ = add_cg_features(df, cg_feats)
    except Exception as e:
        log(f"  CoinGlass features skipped: {e}")

    present = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    missing_f = [f for f in CHAMPION_FEAT_31 if f not in df.columns]
    if missing_f:
        log(f"  WARNING: Missing features: {missing_f}")

    n_sym = df["symbol"].nunique()
    log(f"  Frame: {len(df):,} rows, {n_sym} symbols")
    log(f"  Features: {len(present)}/{len(CHAMPION_FEAT_31)}")

    # Compute ADV (7-day rolling avg daily $ volume)
    log("  Computing ADV...")
    df["dollar_vol_1h"] = df["close"] * df["volume"]
    df["adv_7d"] = df.groupby("symbol")["dollar_vol_1h"].transform(
        lambda x: x.rolling(168, min_periods=84).mean() * 24)
    log(f"  ADV computed. Median ADV: ${df['adv_7d'].median()/1e6:.1f}M")

    return df, regime_df, sorted(df["symbol"].unique().tolist())


# ─── Phase 3: Volume-filtered simulate ──────────────────────────

def simulate_v2_volfilter(merged, regime_df, n_long, n_short, cfg,
                          cutoff_on=0.9, cutoff_off=None,
                          min_adv_usd=10e6):
    """
    R113 simulate_v2 with point-in-time ADV filter.
    At each rebalance, only coins with adv_7d >= min_adv_usd are eligible.
    """
    if cutoff_off is None and cutoff_on is not None:
        cutoff_off = cutoff_on - 0.1

    rebal_hours  = cfg["rebal_hours"]
    ema_alpha    = cfg.get("ema_alpha", None)
    hysteresis   = cfg.get("hysteresis", 0)
    dyn_threshold = cfg.get("dyn_threshold", 0.5)
    funding_per_12h = 0.00008

    all_rets = []
    prev_longs: Set[str] = set()
    prev_shorts: Set[str] = set()
    prev_preds: Dict[str, float] = {}
    risk_off = False

    timestamps_sorted = sorted(merged["timestamp"].unique())
    grouped = {ts: grp for ts, grp in merged.groupby("timestamp")}
    rebal_timestamps = timestamps_sorted[::rebal_hours]
    universe_sizes = []

    for ts in rebal_timestamps:
        if ts not in regime_df.index or ts not in grouped:
            continue
        row = regime_df.loc[ts]
        trend_str = row.get("trend_strength", 0)
        grp = grouped[ts].copy()

        # ── Volume filter ──
        if "adv_7d" in grp.columns:
            grp = grp[grp["adv_7d"] >= min_adv_usd].copy()
        universe_sizes.append(len(grp))

        if len(grp) == 0:
            all_rets.append({
                "timestamp": ts, "gross_ret": 0.0, "net_ret": 0.0,
                "cost": 0.0, "n_long": 0, "n_short": 0,
                "turnover": 0, "risk_off": True, "n_universe": 0,
            })
            continue

        # ── Update EMA ──
        if ema_alpha is not None and ema_alpha < 1.0:
            for idx, r in grp.iterrows():
                sym = r["symbol"]
                raw_pred = r["pred"]
                smoothed = (ema_alpha * raw_pred
                            + (1 - ema_alpha) * prev_preds.get(sym, raw_pred))
                prev_preds[sym] = smoothed
                grp.at[idx, "pred"] = smoothed

        # ── Risk-off state machine ──
        if cutoff_on is not None:
            if not risk_off and trend_str > cutoff_on:
                risk_off = True
                if prev_longs or prev_shorts:
                    n_prev = len(prev_longs) + len(prev_shorts)
                    avg_w = 1.0 / n_prev
                    close_cost = sum(_cost_for_sym(s) * avg_w
                                     for s in prev_longs | prev_shorts)
                    all_rets.append({
                        "timestamp": ts, "gross_ret": 0.0,
                        "net_ret": -close_cost, "cost": close_cost,
                        "n_long": 0, "n_short": 0,
                        "turnover": n_prev, "risk_off": True,
                        "n_universe": len(grp),
                    })
                else:
                    all_rets.append({
                        "timestamp": ts, "gross_ret": 0.0, "net_ret": 0.0,
                        "cost": 0.0, "n_long": 0, "n_short": 0,
                        "turnover": 0, "risk_off": True,
                        "n_universe": len(grp),
                    })
                prev_longs, prev_shorts = set(), set()
                continue
            if risk_off:
                if trend_str < cutoff_off:
                    risk_off = False
                else:
                    all_rets.append({
                        "timestamp": ts, "gross_ret": 0.0, "net_ret": 0.0,
                        "cost": 0.0, "n_long": 0, "n_short": 0,
                        "turnover": 0, "risk_off": True,
                        "n_universe": len(grp),
                    })
                    continue

        # ── Portfolio construction ──
        n = len(grp)
        nl = min(n_long, n // 3)
        ns = min(n_short, n // 3)
        if nl == 0 and ns == 0:
            all_rets.append({
                "timestamp": ts, "gross_ret": 0.0, "net_ret": 0.0,
                "cost": 0.0, "n_long": 0, "n_short": 0,
                "turnover": 0, "risk_off": False, "n_universe": n,
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
            for _, r in remaining.sort_values("pred_rank", ascending=False).head(
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
        long_ret  = longs["fwd_ret"].mean() if len(longs) > 0 else 0
        short_ret = shorts["fwd_ret"].mean() if len(shorts) > 0 else 0

        nl_act, ns_act = len(new_longs), len(new_shorts)
        if nl_act > 0 and ns_act > 0:
            gross_ret = 0.5 * long_ret - 0.5 * short_ret
        elif ns_act > 0:
            gross_ret = -short_ret
        else:
            gross_ret = long_ret
        gross_ret *= exposure

        if total_positions > 0:
            avg_weight = 1.0 / total_positions
            turnover_cost = sum(_cost_for_sym(sym) * avg_weight
                                for sym in new_opened)
            turnover_cost += sum(_cost_for_sym(sym) * avg_weight
                                 for sym in closed)
            holding_cost = funding_per_12h * (rebal_hours / 12)
            total_cost = turnover_cost + holding_cost
        else:
            total_cost = 0.0

        net_ret = gross_ret - total_cost
        prev_longs, prev_shorts = new_longs, new_shorts

        all_rets.append({
            "timestamp": ts, "gross_ret": gross_ret, "net_ret": net_ret,
            "cost": total_cost, "n_long": nl_act, "n_short": ns_act,
            "turnover": len(new_opened) + len(closed), "risk_off": False,
            "n_universe": n,
        })

    port = pd.DataFrame(all_rets) if all_rets else pd.DataFrame()

    # Attach universe stats to metadata
    if universe_sizes:
        port.attrs["avg_universe"] = np.mean(universe_sizes)
        port.attrs["min_universe"] = min(universe_sizes)
        port.attrs["max_universe"] = max(universe_sizes)

    return port


# ─── Main ────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log("=" * 70)
    log("R115 — Universe Expansion")
    log("=" * 70)

    os.makedirs("results", exist_ok=True)

    # ── Phase 1: Download ──
    log("\nPhase 1: Discover & download expanded universe")
    try:
        all_available = download_expanded_universe()
    except Exception as e:
        log(f"  Download failed: {e}")
        log("  Proceeding with existing data only")
        all_available = set()
        for f in DATA_DIR.glob("*_1h.parquet"):
            sym = f.stem.replace("_1h", "").replace("_", "/")
            all_available.add(sym)
    log(f"  Total available: {len(all_available)} symbols")

    # ── Phase 2: Load expanded data ──
    log("\nPhase 2: Load expanded universe")
    df, regime_df, all_syms = load_data_expanded(symbol_list=None)
    n_total = len(all_syms)
    n_in_35 = len([s for s in all_syms if s in set(SYM_35)])
    n_new = n_total - n_in_35
    log(f"  Universe: {n_total} total ({n_in_35} original + {n_new} new)")

    # Features available
    base_feats = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    no_rank = [f for f in base_feats if f in MARKET_LEVEL_FEATURES]

    # ── Phase 3: Train on expanded universe ──
    log("\nPhase 3: Training ensemble on expanded universe...")
    t1 = time.time()
    preds = train_ensemble(df, base_feats, CONTINUOUS_WINDOWS,
                           seeds=SEEDS, cs_rank_exclude=no_rank)
    log(f"  Trained in {time.time()-t1:.0f}s")

    # ── Merge ADV into preds ──
    adv_lookup = (df[["timestamp", "symbol", "adv_7d"]]
                  .drop_duplicates(subset=["timestamp", "symbol"]))
    preds = preds.merge(adv_lookup, on=["timestamp", "symbol"], how="left")

    # ── Phase 4: R113 baseline (SYM_35, no vol filter) ──
    log("\n" + "=" * 70)
    log("R113 baseline (SYM_35 only, n_long=4, n_short=2)")
    log("=" * 70)

    # Run baseline on SYM_35-only preds
    preds_35 = preds[preds["symbol"].isin(set(SYM_35))].copy()
    cfg = dict(PROD_CFG)
    port_base = simulate_v2(preds_35, regime_df, 4, 2, cfg, cutoff_on=0.9)
    m_base = analyze_config(port_base, "R113_SYM35_4L2S")
    print_result(m_base)

    # ── Phase 5: Grid search ──
    log("\n" + "=" * 70)
    log("R115 Grid: expanded universe with volume filter")
    log("=" * 70)

    MIN_ADV_GRID = [5e6, 10e6, 20e6]
    NL_NS_GRID   = [(4, 2), (6, 3)]

    results = [m_base]

    for min_adv in MIN_ADV_GRID:
        for nl, ns in NL_NS_GRID:
            label = f"adv{int(min_adv/1e6)}M_{nl}L{ns}S"
            log(f"\n  {label}...")

            port = simulate_v2_volfilter(
                preds, regime_df, nl, ns, cfg,
                cutoff_on=0.9, min_adv_usd=min_adv)

            m = analyze_config(port, label)
            m["min_adv_usd"] = int(min_adv)
            m["n_long_cfg"] = nl
            m["n_short_cfg"] = ns
            if hasattr(port, 'attrs'):
                m["avg_universe"] = round(port.attrs.get("avg_universe", 0), 1)
                m["min_universe"] = int(port.attrs.get("min_universe", 0))
                m["max_universe"] = int(port.attrs.get("max_universe", 0))
            else:
                m["avg_universe"] = 0
                m["min_universe"] = 0
                m["max_universe"] = 0
            print_result(m)
            log(f"    Universe: avg={m['avg_universe']:.0f}, "
                f"min={m['min_universe']}, max={m['max_universe']}")
            results.append(m)

    # ── Results table ──
    log("\n" + "=" * 70)
    log("R115 RESULTS")
    log("=" * 70)

    hdr = (f"  {'Config':<22} {'NetSh':>7} {'GrSh':>7} {'Ret%':>7} "
           f"{'DD%':>7} {'Calmar':>7} {'%flat':>6} {'Cost%':>6} "
           f"{'AvgN':>5}")
    sep = (f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} "
           f"{'-'*6} {'-'*6} {'-'*5}")
    log(hdr)
    log(sep)

    for m in results:
        avg_n = m.get("avg_universe", "n/a")
        avg_n_str = f"{avg_n:>5.0f}" if isinstance(avg_n, (int, float)) else f"{avg_n:>5}"
        log(f"  {m['label']:<22} {m['net_sharpe']:>7.3f} "
            f"{m['gross_sharpe']:>7.3f} {m['total_ret_pct']:>7.1f} "
            f"{m['max_dd_pct']:>7.1f} {m['calmar']:>7.2f} "
            f"{m['pct_flat']:>5.1f}% {m['total_cost_pct']:>6.2f} "
            f"{avg_n_str}")

    # ── Acceptance check ──
    base_s = m_base["net_sharpe"]
    base_c = m_base["calmar"]
    base_cost = m_base["total_cost_pct"]

    accepted = []
    for m in results[1:]:
        crit_a = m["net_sharpe"] >= base_s + 0.05
        crit_b = (m["calmar"] >= base_c * 1.1
                  and m["net_sharpe"] >= base_s - 0.05)
        cost_ok = m["total_cost_pct"] <= base_cost * 1.15
        if (crit_a or crit_b) and cost_ok:
            accepted.append(m)

    log(f"\n  Accepted (Sharpe +0.05 OR Calmar +10%, cost <+15%): "
        f"{len(accepted)}/{len(results)-1}")

    if accepted:
        best = max(accepted, key=lambda x: x["calmar"])
        log(f"  BEST: {best['label']} → Sharpe={best['net_sharpe']:.3f}, "
            f"DD={best['max_dd_pct']:.1f}%, Calmar={best['calmar']:.2f}")
    else:
        best = m_base
        log(f"  No configs beat acceptance. R113 baseline stays.")

    # ── Comparison ──
    log(f"\n  {'Metric':<22} {'R113':>12} {'Best R115':>12} {'Delta':>10}")
    log(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*10}")
    for k in ['net_sharpe', 'gross_sharpe', 'total_ret_pct', 'max_dd_pct',
              'calmar', 'pct_flat', 'total_cost_pct']:
        v0 = m_base[k]
        v1 = best[k]
        log(f"  {k:<22} {v0:>12.3f} {v1:>12.3f} {v1 - v0:>+10.3f}")

    # ── Save ──
    df_res = pd.DataFrame(results)
    df_res.to_csv("results/r115_grid.csv", index=False)
    with open("results/r115_best.json", "w") as f:
        json.dump(best, f, indent=2, default=str)

    log(f"\nSaved: results/r115_grid.csv, r115_best.json")
    log(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
