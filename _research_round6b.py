#!/usr/bin/env python3
"""
Research round 6b: Cross-combos of best R6 findings.

Best individual from R6:
  - Strategy momentum (lb=48h) → $1521, Wr=-8.0%, Sh=3.55
  - 5L/3S (long-heavy) → $1323, Wr=-7.1%
  - 6L/4S → $1272, Wr=-6.8%
  - Adaptive N base=8 → $1083, Wr=-6.0%
  - Adaptive N base=7 → $1270, Wr=-10.2%, Sh=3.71

Cross-combos to test:
  strat-mom + asymmetric L/S
  strat-mom + adaptive N
  strat-mom + rebal=8h
  asymmetric L/S + adaptive N
  strat-mom + asymmetric + adaptive
  Best + bigger N range
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

# Reuse everything from round 6
from _research_round6 import (
    FEATURES, SYM_35, WINDOWS, load_data, compute_regime,
    train_and_predict, compute_recent_vol, simulate, eval_config, show
)


def main():
    LEV = 5
    CAP = 100

    print("=" * 100)
    print(f"  RESEARCH ROUND 6B: Cross-combos ({LEV}x, ${CAP})")
    print("=" * 100)

    print("\n  Loading data...")
    df35 = load_data(SYM_35)
    feats = [f for f in FEATURES if f in df35.columns]
    regime_df = compute_regime(df35)
    print(f"    {df35['symbol'].nunique()} sym")

    print("\n  Training models (12h)...")
    p35 = train_and_predict(df35, feats, 12)
    p35 = compute_recent_vol(p35)
    print(f"    {len(p35):,} predictions")

    results = []

    # Base config (R5 winner)
    cfg_base = {"n_long": 5, "n_short": 5, "trend_cutoff": 0.8,
                "dyn_threshold": 0.5, "eq_mom_boost": True, "kelly_sizing": True}

    # ── Baseline ──
    sub = simulate(p35, regime_df, 12, cfg_base)
    r = eval_config(sub, 12, "BASELINE R5", LEV, CAP)
    if r: results.append(r); show(r)

    configs = [
        # STRAT-MOM combos
        {"strategy_momentum": True, "strat_mom_lookback": 48,
         "label": "SM48"},
        {"strategy_momentum": True, "strat_mom_lookback": 48,
         "n_long": 5, "n_short": 3, "label": "SM48+5L3S"},
        {"strategy_momentum": True, "strat_mom_lookback": 48,
         "n_long": 6, "n_short": 4, "label": "SM48+6L4S"},
        {"strategy_momentum": True, "strat_mom_lookback": 48,
         "n_long": 6, "n_short": 3, "label": "SM48+6L3S"},
        {"strategy_momentum": True, "strat_mom_lookback": 48,
         "n_long": 7, "n_short": 5, "label": "SM48+7L5S"},
        {"strategy_momentum": True, "strat_mom_lookback": 48,
         "n_long": 7, "n_short": 4, "label": "SM48+7L4S"},
        {"strategy_momentum": True, "strat_mom_lookback": 48,
         "adaptive_n": True, "n_long": 7, "n_short": 7,
         "label": "SM48+adaptN7"},
        {"strategy_momentum": True, "strat_mom_lookback": 48,
         "adaptive_n": True, "n_long": 8, "n_short": 8,
         "label": "SM48+adaptN8"},
        {"strategy_momentum": True, "strat_mom_lookback": 48,
         "adaptive_n": True, "n_long": 6, "n_short": 6,
         "label": "SM48+adaptN6"},
        {"strategy_momentum": True, "strat_mom_lookback": 48,
         "adaptive_n": True, "n_long": 7, "n_short": 5,
         "label": "SM48+adaptN7L5S"},

        # Asymmetric without strat-mom
        {"n_long": 6, "n_short": 3, "label": "6L3S"},
        {"n_long": 7, "n_short": 4, "label": "7L4S"},
        {"n_long": 7, "n_short": 5, "label": "7L5S"},
        {"n_long": 8, "n_short": 5, "label": "8L5S"},

        # Adaptive N + asymmetric
        {"adaptive_n": True, "n_long": 8, "n_short": 5,
         "label": "adaptN 8L5S"},
        {"adaptive_n": True, "n_long": 7, "n_short": 5,
         "label": "adaptN 7L5S"},
        {"adaptive_n": True, "n_long": 7, "n_short": 4,
         "label": "adaptN 7L4S"},

        # STRAT-MOM 120h combos (also strong)
        {"strategy_momentum": True, "strat_mom_lookback": 120,
         "n_long": 5, "n_short": 3, "label": "SM120+5L3S"},
        {"strategy_momentum": True, "strat_mom_lookback": 120,
         "n_long": 6, "n_short": 4, "label": "SM120+6L4S"},
        {"strategy_momentum": True, "strat_mom_lookback": 120,
         "adaptive_n": True, "n_long": 7, "n_short": 7,
         "label": "SM120+adaptN7"},

        # Rebal=8h
        {"strategy_momentum": True, "strat_mom_lookback": 48,
         "rebal_hours": 8, "label": "SM48+rebal8h"},
        {"n_long": 6, "n_short": 4, "rebal_hours": 8,
         "label": "6L4S+rebal8h"},
        {"strategy_momentum": True, "strat_mom_lookback": 48,
         "n_long": 6, "n_short": 4, "rebal_hours": 8,
         "label": "SM48+6L4S+rebal8h"},

        # Triple combo: SM + asymmetric + adaptive
        {"strategy_momentum": True, "strat_mom_lookback": 48,
         "adaptive_n": True, "n_long": 8, "n_short": 5,
         "label": "SM48+adaptN8L5S"},
        {"strategy_momentum": True, "strat_mom_lookback": 48,
         "adaptive_n": True, "n_long": 7, "n_short": 4,
         "label": "SM48+adaptN7L4S"},
    ]

    for combo in configs:
        lab = combo.pop("label")
        cfg = {**cfg_base}
        cfg.update(combo)
        combo["label"] = lab
        sub = simulate(p35, regime_df, 12, cfg)
        r = eval_config(sub, 12, lab, LEV, CAP)
        if r: results.append(r); show(r)

    if not results:
        print("  No results")
        return

    # Scoring
    for r in results:
        safety = max(0.3, 1.0 + r["worst_m"])
        r["score"] = r["equity"] * safety * (max(0.01, r["calmar"]) ** 0.3)

    results.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n{'=' * 100}")
    print(f"  🏆 TOP 10 RISK-ADJUSTED ({LEV}x, ${CAP})")
    print(f"{'=' * 100}")
    for i, r in enumerate(results[:10]):
        safe = "✅" if r["worst_m"] > -0.15 else ("⚠️" if r["worst_m"] > -0.25 else "❌")
        wm = f"{r['win_months']}/{r['total_months']}"
        print(f"\n  #{i+1} {safe} {r['name']}")
        print(f"      Sh={r['sharpe']:.2f} | Wr={r['worst_m']*100:+.1f}% | "
              f"Avg/m={r['avg_monthly']*100:+.1f}% | Med/m={r['med_monthly']*100:+.1f}% | "
              f"Calmar={r['calmar']:.2f} | WM={wm}")
        print(f"      ${CAP} → ${r['equity']:.0f} ({len(r['month_data'])} мес) | "
              f"Score={r['score']:.0f}")
        for md in r["month_data"]:
            marker = " ← worst" if md["ret"] == r["worst_m"] else ""
            print(f"         {md['month']:>10s}  {md['ret']*100:>+7.1f}%  "
                  f"equity=${md['equity']:>7.0f}{marker}")

    # Ultra-safe ranking
    safe_r = [r for r in results if r["worst_m"] > -0.10]
    safe_r.sort(key=lambda x: x["equity"], reverse=True)
    print(f"\n{'=' * 100}")
    print(f"  🛡️ ULTRA-SAFE (worst > -10%):")
    print(f"{'=' * 100}")
    for i, r in enumerate(safe_r[:10]):
        wm = f"{r['win_months']}/{r['total_months']}"
        print(f"  #{i+1} {r['name']}: ${CAP}→${r['equity']:.0f} | "
              f"Wr={r['worst_m']*100:+.1f}% | WM={wm} | Sh={r['sharpe']:.2f}")

    # Equity ranking
    results.sort(key=lambda x: x["equity"], reverse=True)
    print(f"\n{'=' * 100}")
    print(f"  💰 TOP 5 RAW EQUITY:")
    print(f"{'=' * 100}")
    for i, r in enumerate(results[:5]):
        safe = "✅" if r["worst_m"] > -0.15 else ("⚠️" if r["worst_m"] > -0.25 else "❌")
        wm = f"{r['win_months']}/{r['total_months']}"
        print(f"  #{i+1} {safe} {r['name']}: ${CAP}→${r['equity']:.0f} | "
              f"Wr={r['worst_m']*100:+.1f}% | WM={wm} | Sh={r['sharpe']:.2f}")

    # Projections for top risk-adj
    results.sort(key=lambda x: x["score"], reverse=True)
    best = results[0]
    print(f"\n{'=' * 100}")
    print(f"  📊 PROJECTIONS: {best['name']}")
    print(f"{'=' * 100}")
    for label, mret in [("Avg", best["avg_monthly"]),
                         ("Median", best["med_monthly"]),
                         ("p25", best["monthly"].quantile(0.25))]:
        m6 = (1 + mret) ** 6 - 1
        y1 = (1 + mret) ** 12 - 1
        print(f"  {label:>7s} ({mret*100:+.1f}%/мес): 6м=${CAP*(1+m6):.0f} | 12м=${CAP*(1+y1):.0f}")


if __name__ == "__main__":
    main()
