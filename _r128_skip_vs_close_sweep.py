#!/usr/bin/env python3
"""
R128 — SKIP vs CLOSE risk-off mode comparison with trend_cutoff sweep
Tests both modes across cutoff values 0.8, 0.85, 0.9, 0.95, 1.0
"""

import sys, warnings, time
from _preflight_check import check_versions
check_versions()

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

from _research_round7 import SYM_35
from _research_r22_models import SEEDS, log, cs_rank_cols
from _research_r35_new_features import add_r35_features, load_research_frame, MARKET_LEVEL_FEATURES
from _research_r47_coinglass import CHAMPION_FEAT_30, add_cg_features, compute_cg_features, load_cg_daily

CHAMPION_FEAT_31 = CHAMPION_FEAT_30 + ["cg_taker_imb"]

TIER1_SYMS = {"BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"}
TIER3_SYMS = {"SAND/USDT", "LDO/USDT", "INJ/USDT", "APT/USDT", "ARB/USDT", "GALA/USDT", "FTM/USDT", "MATIC/USDT"}
TIER2_SYMS = set(SYM_35) - TIER1_SYMS - TIER3_SYMS

def _cost_for_sym(sym):
    if sym in TIER1_SYMS: return 0.92 * (-0.0001) + 0.08 * 0.0007
    elif sym in TIER2_SYMS: return 0.75 * 0.0001 + 0.25 * 0.0007
    else: return 0.0005 + 0.0002

CONTINUOUS_WINDOWS = [
    {"name": "W1", "train_end": "2024-06-01", "val_start": "2024-06-01", "val_end": "2024-09-30", "test_start": "2024-10-15", "test_end": "2025-05-14"},
    {"name": "W2", "train_end": "2025-01-01", "val_start": "2025-01-01", "val_end": "2025-04-30", "test_start": "2025-05-15", "test_end": "2025-11-14"},
    {"name": "W3", "train_end": "2025-07-01", "val_start": "2025-07-01", "val_end": "2025-10-31", "test_start": "2025-11-15", "test_end": "2026-03-17"},
]

PROD_CFG = {"n_long": 6, "n_short": 3, "rebal_hours": 12, "trend_cutoff": 0.9, "dyn_threshold": 0.7, "ema_alpha": 0.5, "hysteresis": 3}

LGB_PARAMS = {"objective": "binary", "metric": "auc", "learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 100, "subsample": 0.8, "colsample_bytree": 0.8, "lambda_l2": 1.0, "verbose": -1, "n_jobs": -1}
XGB_PARAMS = {"objective": "binary:logistic", "eval_metric": "auc", "learning_rate": 0.03, "max_depth": 6, "min_child_weight": 100, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "n_jobs": -1, "verbosity": 0}
N_ROUNDS, EARLY_STOP = 600, 40

def load_data():
    print("  LOADING DATA")
    df, regime_df = load_research_frame()
    df, _ = add_r35_features(df)
    cg = load_cg_daily()
    cg_feats = compute_cg_features(cg)
    df, _, _ = add_cg_features(df, cg_feats)
    present = [f for f in CHAMPION_FEAT_31 if f in df.columns]
    print(f"  Frame: {len(df):,} rows, {df['symbol'].nunique()} symbols, Features: {len(present)}/31")
    return df, regime_df

def train_ensemble(df, feats, windows, seeds=SEEDS, cs_rank_exclude=None):
    avail = [f for f in feats if f in df.columns]
    rank_exclude = set(cs_rank_exclude or [])
    rank_feats = [f for f in avail if f not in rank_exclude]
    tz = df["timestamp"].dt.tz
    all_lgb, all_xgb = [], []

    for seed in seeds:
        p_lgb, p_xgb = {**LGB_PARAMS, "seed": seed}, {**XGB_PARAMS, "seed": seed}
        for w in windows:
            te_end = pd.Timestamp(w["test_end"], tz=tz)
            te_start = pd.Timestamp(w["test_start"], tz=tz)
            tr_end = pd.Timestamp(w["train_end"], tz=tz)
            va_start = pd.Timestamp(w["val_start"], tz=tz)
            va_end = pd.Timestamp(w["val_end"], tz=tz)

            train_ = df[df["timestamp"] < tr_end].copy()
            val_ = df[(df["timestamp"] >= va_start) & (df["timestamp"] < va_end)].copy()
            test_ = df[(df["timestamp"] >= te_start) & (df["timestamp"] <= te_end)].copy()
            if len(train_) < 5000 or len(test_) < 200: continue
            if rank_feats:
                train_ = cs_rank_cols(train_, rank_feats)
                val_ = cs_rank_cols(val_, rank_feats)
                test_ = cs_rank_cols(test_, rank_feats)
            for d in [train_, val_, test_]:
                d["target_binary"] = (d["fwd_ret_12h"] > 0).astype(int)
            for col in avail:
                for d in [train_, val_, test_]:
                    if d[col].isna().any(): d[col] = d[col].fillna(0)

            tr = train_[avail + ["target_binary"]].dropna()
            va = val_[avail + ["target_binary"]].dropna()
            te = test_[avail + ["target_binary", "timestamp", "symbol"]].dropna()
            fwd = test_[["timestamp", "symbol", "fwd_ret_12h"]].rename(columns={"fwd_ret_12h": "fwd_ret"}).dropna()
            for d in [tr, va, te]:
                d.replace([np.inf, -np.inf], np.nan, inplace=True)
            tr, va, te = tr.dropna(), va.dropna(), te.dropna()
            if len(te) == 0: continue

            dt = lgb.Dataset(tr[avail], label=tr["target_binary"])
            dv = lgb.Dataset(va[avail], label=va["target_binary"])
            m = lgb.train(p_lgb, dt, num_boost_round=N_ROUNDS, valid_sets=[dv], callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(-1)])
            p = m.predict(te[avail])
            rec = te[["timestamp", "symbol"]].copy()
            rec["pred_lgb"] = p
            rec = rec.merge(fwd, on=["timestamp", "symbol"], how="inner")
            rec["window"], rec["seed"] = w["name"], seed
            all_lgb.append(rec)

            dt_x = xgb.DMatrix(tr[avail], label=tr["target_binary"])
            dv_x = xgb.DMatrix(va[avail], label=va["target_binary"])
            m_x = xgb.train(p_xgb, dt_x, num_boost_round=N_ROUNDS, evals=[(dv_x, "val")], early_stopping_rounds=EARLY_STOP, verbose_eval=False)
            p_x = m_x.predict(xgb.DMatrix(te[avail]))
            rec2 = te[["timestamp", "symbol"]].copy()
            rec2["pred_xgb"] = p_x
            rec2 = rec2.merge(fwd, on=["timestamp", "symbol"], how="inner")
            rec2["window"], rec2["seed"] = w["name"], seed
            all_xgb.append(rec2)

            if seed == seeds[0]:
                log(f"  {w['name']}/s{seed}: train={len(tr):,} test={len(te):,}")

    if not all_lgb: return None
    lgb_df = pd.concat(all_lgb)
    xgb_df = pd.concat(all_xgb)
    lgb_avg = lgb_df.groupby(["timestamp", "symbol"]).agg(pred_lgb=("pred_lgb", "mean"), fwd_ret=("fwd_ret", "first"), window=("window", "first")).reset_index()
    xgb_avg = xgb_df.groupby(["timestamp", "symbol"]).agg(pred_xgb=("pred_xgb", "mean")).reset_index()
    merged = lgb_avg.merge(xgb_avg, on=["timestamp", "symbol"], how="inner")
    merged["raw_prob"] = 0.5 * merged["pred_lgb"] + 0.5 * merged["pred_xgb"]
    merged["rank_lgb"] = merged.groupby("timestamp")["pred_lgb"].rank(pct=True) - 0.5
    merged["rank_xgb"] = merged.groupby("timestamp")["pred_xgb"].rank(pct=True) - 0.5
    merged["pred"] = 0.5 * merged["rank_lgb"] + 0.5 * merged["rank_xgb"]
    return merged[["timestamp", "symbol", "pred", "raw_prob", "fwd_ret", "window"]]

def simulate(merged, regime_df, n_long, n_short, cfg):
    trend_cutoff = cfg["trend_cutoff"]
    rebal_hours = cfg["rebal_hours"]
    risk_off_mode = cfg.get("risk_off_mode", "close")
    ema_alpha = cfg.get("ema_alpha", None)
    hysteresis = cfg.get("hysteresis", 0)
    funding_per_12h = 0.00008

    all_rets = []
    prev_longs, prev_shorts, prev_preds = set(), set(), {}

    timestamps_sorted = sorted(merged["timestamp"].unique())
    grouped = {ts: grp for ts, grp in merged.groupby("timestamp")}
    rebal_timestamps = timestamps_sorted[::rebal_hours]

    for ts in rebal_timestamps:
        if ts not in regime_df.index or ts not in grouped: continue
        row = regime_df.loc[ts]
        trend_str = row.get("trend_strength", 0)

        if trend_str > trend_cutoff:
            if risk_off_mode == "close":
                if prev_longs or prev_shorts:
                    n_prev = len(prev_longs) + len(prev_shorts)
                    avg_weight = 1.0 / n_prev if n_prev > 0 else 0
                    close_cost = sum(_cost_for_sym(s) * avg_weight for s in prev_longs | prev_shorts)
                    all_rets.append({"timestamp": ts, "gross_ret": 0.0, "net_ret": -close_cost, "cost": close_cost, "n_long": 0, "n_short": 0, "turnover": n_prev})
                else:
                    all_rets.append({"timestamp": ts, "gross_ret": 0.0, "net_ret": 0.0, "cost": 0.0, "n_long": 0, "n_short": 0, "turnover": 0})
                prev_longs, prev_shorts = set(), set()
            elif risk_off_mode == "skip":
                # SKIP (prod): don't rebalance, KEEP holding previous positions,
                # record their fwd_ret this period. No turnover cost (we didn't trade),
                # only holding/funding cost. prev_longs/prev_shorts stay unchanged.
                if prev_longs or prev_shorts:
                    gh = grouped[ts]
                    held_l = gh[gh["symbol"].isin(prev_longs)]
                    held_s = gh[gh["symbol"].isin(prev_shorts)]
                    lr = held_l["fwd_ret"].mean() if len(held_l) > 0 else 0
                    sr = held_s["fwd_ret"].mean() if len(held_s) > 0 else 0
                    nl_h, ns_h = len(prev_longs), len(prev_shorts)
                    if nl_h > 0 and ns_h > 0:
                        gr = 0.5 * lr - 0.5 * sr
                    elif ns_h > 0:
                        gr = -sr
                    else:
                        gr = lr
                    holding_cost = funding_per_12h * (rebal_hours / 12)
                    all_rets.append({"timestamp": ts, "gross_ret": gr, "net_ret": gr - holding_cost, "cost": holding_cost, "n_long": nl_h, "n_short": ns_h, "turnover": 0})
                else:
                    all_rets.append({"timestamp": ts, "gross_ret": 0.0, "net_ret": 0.0, "cost": 0.0, "n_long": 0, "n_short": 0, "turnover": 0})
            continue

        grp = grouped[ts].copy()
        n = len(grp)
        nl = min(n_long, n // 3)
        ns = min(n_short, n // 3)
        if nl == 0 and ns == 0: continue

        exposure = 1.0
        dyn_threshold = cfg.get("dyn_threshold", 0.5)
        if dyn_threshold is not None and trend_str > dyn_threshold:
            exposure = max(0.1, 1.0 - (trend_str - dyn_threshold) / (trend_cutoff - dyn_threshold + 1e-10) * 0.5)

        if ema_alpha is not None and ema_alpha < 1.0:
            for idx, r in grp.iterrows():
                sym = r["symbol"]
                raw_pred = r["pred"]
                smoothed = ema_alpha * raw_pred + (1 - ema_alpha) * prev_preds.get(sym, raw_pred)
                prev_preds[sym] = smoothed
                grp.at[idx, "pred"] = smoothed

        grp["pred_rank"] = grp["pred"].rank(ascending=False)

        if hysteresis > 0 and (prev_longs or prev_shorts):
            new_longs = set()
            new_shorts = set()
            for idx, r in grp.iterrows():
                sym, rank = r["symbol"], r["pred_rank"]
                if sym in prev_longs and rank <= nl + hysteresis:
                    new_longs.add(sym)
                elif sym in prev_shorts and rank > (n - ns - hysteresis):
                    new_shorts.add(sym)
            for _, r in grp[~grp["symbol"].isin(new_longs | new_shorts)].sort_values("pred_rank").head(nl - len(new_longs)).iterrows():
                new_longs.add(r["symbol"])
            for _, r in grp[~grp["symbol"].isin(new_longs | new_shorts)].sort_values("pred_rank", ascending=False).head(ns - len(new_shorts)).iterrows():
                new_shorts.add(r["symbol"])
        else:
            new_longs = set(grp[grp["pred_rank"] <= nl]["symbol"].tolist()) if nl > 0 else set()
            new_shorts = set(grp[grp["pred_rank"] > (n - ns)]["symbol"].tolist()) if ns > 0 else set()

        new_opened = (new_longs - prev_longs) | (new_shorts - prev_shorts)
        closed = (prev_longs - new_longs) | (prev_shorts - new_shorts)
        total_positions = len(new_longs) + len(new_shorts)

        longs = grp[grp["symbol"].isin(new_longs)]
        shorts = grp[grp["symbol"].isin(new_shorts)]
        long_ret = longs["fwd_ret"].mean() if len(longs) > 0 else 0
        short_ret = shorts["fwd_ret"].mean() if len(shorts) > 0 else 0

        nl_act, ns_act = len(new_longs), len(new_shorts)
        if nl_act > 0 and ns_act > 0:
            gross_ret = 0.5 * long_ret - 0.5 * short_ret
        elif ns_act > 0: gross_ret = -short_ret
        else: gross_ret = long_ret
        gross_ret *= exposure

        if total_positions > 0:
            avg_weight = 1.0 / total_positions
            turnover_cost = sum(_cost_for_sym(sym) * avg_weight for sym in new_opened)
            turnover_cost += sum(_cost_for_sym(sym) * avg_weight for sym in closed)
            holding_cost = funding_per_12h * (rebal_hours / 12)
            total_cost = turnover_cost + holding_cost
        else: total_cost = 0.0

        net_ret = gross_ret - total_cost
        prev_longs, prev_shorts = new_longs, new_shorts

        all_rets.append({"timestamp": ts, "gross_ret": gross_ret, "net_ret": net_ret, "cost": total_cost, "n_long": nl_act, "n_short": ns_act, "turnover": len(new_opened) + len(closed)})

    return pd.DataFrame(all_rets) if all_rets else pd.DataFrame()

def sharpe(rets_series, periods_per_year=2*365):
    if len(rets_series) < 2: return 0.0
    eq = (1 + rets_series).cumprod()
    r = eq.pct_change().dropna()
    return r.mean() / (r.std() + 1e-10) * np.sqrt(periods_per_year)

def analyze(port, label):
    if len(port) == 0:
        return {"mode": label, "net_sharpe": 0, "periods": 0, "return": 0}
    ns = sharpe(port["net_ret"])
    return {"mode": label, "net_sharpe": ns, "periods": len(port), "return": (1 + port["net_ret"]).prod() - 1}

if __name__ == "__main__":
    print("=" * 90)
    print("  R128 — SKIP vs CLOSE RISK-OFF MODE COMPARISON (trend_cutoff sweep)")
    print("=" * 90)

    df, regime_df = load_data()
    print("\nTraining ensemble...")
    merged = train_ensemble(df, [f for f in CHAMPION_FEAT_31 if f in df.columns], CONTINUOUS_WINDOWS, cs_rank_exclude=[f for f in CHAMPION_FEAT_31 if f in MARKET_LEVEL_FEATURES])

    results = []
    print("\n" + "=" * 90)
    print("  TESTING: trend_cutoff sweep 0.8 → 1.0")
    print("=" * 90)

    for cutoff in [0.8, 0.85, 0.9, 0.95, 1.0]:
        print(f"\ntrend_cutoff={cutoff:.2f}")

        # CLOSE mode
        cfg_close = PROD_CFG.copy()
        cfg_close['trend_cutoff'] = cutoff
        cfg_close['risk_off_mode'] = 'close'
        port_close = simulate(merged, regime_df, 6, 3, cfg_close)
        r_close = analyze(port_close, f'CLOSE')
        r_close['cutoff'] = cutoff
        results.append(r_close)
        print(f"  CLOSE: Sharpe={r_close['net_sharpe']:.3f}, periods={r_close['periods']}, return={r_close['return']*100:.1f}%")

        # SKIP mode
        cfg_skip = PROD_CFG.copy()
        cfg_skip['trend_cutoff'] = cutoff
        cfg_skip['risk_off_mode'] = 'skip'
        port_skip = simulate(merged, regime_df, 6, 3, cfg_skip)
        r_skip = analyze(port_skip, f'SKIP')
        r_skip['cutoff'] = cutoff
        results.append(r_skip)
        print(f"  SKIP:  Sharpe={r_skip['net_sharpe']:.3f}, periods={r_skip['periods']}, return={r_skip['return']*100:.1f}%")
        print(f"  Δ: {r_skip['net_sharpe'] - r_close['net_sharpe']:+.3f}")

    df_res = pd.DataFrame(results).sort_values('net_sharpe', ascending=False)
    print("\n" + "=" * 90)
    print("  RESULTS (sorted by Sharpe)")
    print("=" * 90)
    print(df_res[['mode', 'cutoff', 'net_sharpe', 'periods', 'return']].to_string(index=False))

    best = df_res.iloc[0]
    print(f"\n✅ BEST: {best['mode']} cutoff={best['cutoff']:.2f} → Sharpe {best['net_sharpe']:.3f}")

    # Group by mode and find best for each
    close_best = df_res[df_res['mode'] == 'CLOSE'].iloc[0] if (df_res['mode'] == 'CLOSE').any() else None
    skip_best = df_res[df_res['mode'] == 'SKIP'].iloc[0] if (df_res['mode'] == 'SKIP').any() else None

    if close_best is not None and skip_best is not None:
        delta = skip_best['net_sharpe'] - close_best['net_sharpe']
        print(f"\n  CLOSE best: cutoff={close_best['cutoff']:.2f} → {close_best['net_sharpe']:.3f}")
        print(f"  SKIP best:  cutoff={skip_best['cutoff']:.2f} → {skip_best['net_sharpe']:.3f}")
        print(f"  SKIP advantage: {delta:+.3f} ({delta/close_best['net_sharpe']*100:+.1f}%)")
