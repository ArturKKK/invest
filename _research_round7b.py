#!/usr/bin/env python3
"""
Research Round 7B: Cross-combos of best R7 findings.

Best individual from R7:
  - SHRINK=0.1 → $1829, Wr=-2.4%, Calmar=12.53 (SAFEST EVER)
  - REGIME-ASYM → $2061, Wr=-6.4%
  - VOL-SCALE (7L5S) → $2208, Wr=-8.8% (HIGHEST EQUITY)
  - EMA=2 → $1917, Wr=-6.1%, 11/13 WM
  - CONV-W (7L5S) → $1769, Wr=-8.2%, 11/13 WM

Cross-combos:
  - SHRINK + REGIME-ASYM
  - SHRINK + VOL-SCALE
  - SHRINK + EMA
  - REGIME-ASYM + VOL-SCALE
  - REGIME-ASYM + EMA
  - VOL-SCALE + EMA
  - Triple combos
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

from _research_round7 import (
    FEATURES, SYM_35, WINDOWS, load_data, compute_regime,
    train_and_predict_multi, simulate, eval_config, show
)


def main():
    LEV = 5
    CAP = 100

    print("=" * 100)
    print(f"  RESEARCH ROUND 7B: Cross-combos ({LEV}x, ${CAP})")
    print("=" * 100)

    print("\n  Loading data...")
    df = load_data(SYM_35)
    feats = [f for f in FEATURES if f in df.columns]
    regime_df = compute_regime(df)
    print(f"    {df['symbol'].nunique()} sym")

    print("\n  Training models...")
    preds = train_and_predict_multi(df, feats, horizons=[12])
    p12 = preds[12]
    print(f"    {len(p12):,} predictions")

    results = []

    # SM48+6L3S base (R6 winner)
    cfg_base = {"n_long": 6, "n_short": 3, "trend_cutoff": 0.8,
                "dyn_threshold": 0.5, "eq_mom_boost": True, "kelly_sizing": True,
                "strategy_momentum": True, "strat_mom_lookback": 48}

    # Baseline
    sub = simulate(p12, regime_df, 12, cfg_base)
    r = eval_config(sub, 12, "BASELINE SM48+6L3S", LEV, CAP)
    if r: results.append(r); show(r)

    configs = [
        # ── Doubles ──
        {"pred_shrinkage": 0.1, "regime_asym": True,
         "label": "SHRINK01+RG-ASYM 6L3S"},
        {"pred_shrinkage": 0.1, "vol_scaling": True,
         "label": "SHRINK01+VOL 6L3S"},
        {"pred_shrinkage": 0.1, "signal_ema": 2,
         "label": "SHRINK01+EMA2 6L3S"},
        {"pred_shrinkage": 0.1, "conviction_weight": True,
         "label": "SHRINK01+CONV-W 6L3S"},
        {"regime_asym": True, "vol_scaling": True,
         "label": "RG-ASYM+VOL 6L3S"},
        {"regime_asym": True, "signal_ema": 2,
         "label": "RG-ASYM+EMA2 6L3S"},
        {"regime_asym": True, "conviction_weight": True,
         "label": "RG-ASYM+CONV-W 6L3S"},
        {"vol_scaling": True, "signal_ema": 2,
         "label": "VOL+EMA2 6L3S"},
        {"vol_scaling": True, "conviction_weight": True,
         "label": "VOL+CONV-W 6L3S"},
        {"signal_ema": 2, "conviction_weight": True,
         "label": "EMA2+CONV-W 6L3S"},

        # ── Triples ──
        {"pred_shrinkage": 0.1, "regime_asym": True, "vol_scaling": True,
         "label": "SHRINK01+RG-ASYM+VOL 6L3S"},
        {"pred_shrinkage": 0.1, "regime_asym": True, "signal_ema": 2,
         "label": "SHRINK01+RG-ASYM+EMA2 6L3S"},
        {"pred_shrinkage": 0.1, "vol_scaling": True, "signal_ema": 2,
         "label": "SHRINK01+VOL+EMA2 6L3S"},
        {"regime_asym": True, "vol_scaling": True, "signal_ema": 2,
         "label": "RG-ASYM+VOL+EMA2 6L3S"},
        {"regime_asym": True, "vol_scaling": True, "conviction_weight": True,
         "label": "RG-ASYM+VOL+CONV-W 6L3S"},

        # ── Quads ──
        {"pred_shrinkage": 0.1, "regime_asym": True, "vol_scaling": True,
         "signal_ema": 2, "label": "SHRINK01+RG-ASYM+VOL+EMA2 6L3S"},
        {"pred_shrinkage": 0.1, "regime_asym": True, "vol_scaling": True,
         "conviction_weight": True,
         "label": "SHRINK01+RG-ASYM+VOL+CONV-W 6L3S"},

        # ── Best ideas with 7L5S ──
        {"n_long": 7, "n_short": 5, "pred_shrinkage": 0.1,
         "label": "SHRINK01 7L5S"},
        {"n_long": 7, "n_short": 5, "regime_asym": True,
         "label": "RG-ASYM 7L5S"},
        {"n_long": 7, "n_short": 5, "vol_scaling": True, "regime_asym": True,
         "label": "VOL+RG-ASYM 7L5S"},
        {"n_long": 7, "n_short": 5, "pred_shrinkage": 0.1, "regime_asym": True,
         "label": "SHRINK01+RG-ASYM 7L5S"},
        {"n_long": 7, "n_short": 5, "pred_shrinkage": 0.1, "vol_scaling": True,
         "regime_asym": True, "label": "SHRINK01+VOL+RG-ASYM 7L5S"},
        {"n_long": 7, "n_short": 5, "pred_shrinkage": 0.1, "vol_scaling": True,
         "label": "SHRINK01+VOL 7L5S"},

        # ── Shrinkage sweep with REGIME-ASYM ──
        {"pred_shrinkage": 0.05, "regime_asym": True,
         "label": "SHRINK005+RG-ASYM 6L3S"},
        {"pred_shrinkage": 0.15, "regime_asym": True,
         "label": "SHRINK015+RG-ASYM 6L3S"},
        {"pred_shrinkage": 0.2, "regime_asym": True,
         "label": "SHRINK02+RG-ASYM 6L3S"},

        # ── 7L3S (even more long-heavy + best ideas) ──
        {"n_long": 7, "n_short": 3, "pred_shrinkage": 0.1,
         "label": "SHRINK01 7L3S"},
        {"n_long": 7, "n_short": 3, "regime_asym": True,
         "label": "RG-ASYM 7L3S"},
        {"n_long": 7, "n_short": 3, "pred_shrinkage": 0.1, "regime_asym": True,
         "label": "SHRINK01+RG-ASYM 7L3S"},
        {"n_long": 8, "n_short": 4, "regime_asym": True,
         "label": "RG-ASYM 8L4S"},
        {"n_long": 8, "n_short": 4, "pred_shrinkage": 0.1, "regime_asym": True,
         "label": "SHRINK01+RG-ASYM 8L4S"},
    ]

    for combo in configs:
        lab = combo.pop("label")
        cfg = {**cfg_base}
        cfg.update(combo)
        combo["label"] = lab
        sub = simulate(p12, regime_df, 12, cfg)
        r = eval_config(sub, 12, lab, LEV, CAP)
        if r: results.append(r); show(r)

    if not results:
        print("  No results")
        return

    for r in results:
        safety = max(0.3, 1.0 + r["worst_m"])
        r["score"] = r["equity"] * safety * (max(0.01, r["calmar"]) ** 0.3)

    results.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n{'=' * 100}")
    print(f"  🏆 TOP 15 RISK-ADJUSTED ({LEV}x, ${CAP})")
    print(f"{'=' * 100}")
    for i, r in enumerate(results[:15]):
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

    safe_r = [r for r in results if r["worst_m"] > -0.10]
    safe_r.sort(key=lambda x: x["equity"], reverse=True)
    print(f"\n{'=' * 100}")
    print(f"  🛡️ ULTRA-SAFE (worst > -10%):")
    print(f"{'=' * 100}")
    for i, r in enumerate(safe_r[:10]):
        wm = f"{r['win_months']}/{r['total_months']}"
        print(f"  #{i+1} {r['name']}: ${CAP}→${r['equity']:.0f} | "
              f"Wr={r['worst_m']*100:+.1f}% | WM={wm} | Sh={r['sharpe']:.2f}")

    results.sort(key=lambda x: x["equity"], reverse=True)
    print(f"\n{'=' * 100}")
    print(f"  💰 TOP 5 RAW EQUITY:")
    print(f"{'=' * 100}")
    for i, r in enumerate(results[:5]):
        safe = "✅" if r["worst_m"] > -0.15 else ("⚠️" if r["worst_m"] > -0.25 else "❌")
        wm = f"{r['win_months']}/{r['total_months']}"
        print(f"  #{i+1} {safe} {r['name']}: ${CAP}→${r['equity']:.0f} | "
              f"Wr={r['worst_m']*100:+.1f}% | WM={wm} | Sh={r['sharpe']:.2f}")

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
        print(f"  {label:>7s} ({mret*100:+.1f}%/мес): "
              f"6м=${CAP*(1+m6):.0f} | 12м=${CAP*(1+y1):.0f}")


if __name__ == "__main__":
    main()
