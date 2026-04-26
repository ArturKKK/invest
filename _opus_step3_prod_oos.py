"""STEP 3: Honest OOS comparison of PROD-style portfolio logic (R114b) vs.
canonical skip-risk-off equal-weight, on identical OOS preds.

Window: 2026-03-18 → 2026-04-25 (39 days, OOS for all trained models).
Models compared (all = CHAMPION_FEAT_31, 5 seeds, LGB+XGB rank ensemble,
identical to R48 production weights):
  - R128_W4  (trained-through 2025-07-01)
  - R132_W4  (trained-through 2026-01-01)
  - R134_W4  (trained-through 2026-03-15) ← closest proxy for R48 (2026-03-07)

Portfolio configs:
  - R128 canonical: skip risk-off, equal-weight 4L/2S
  - R114b PROD:     state-machine + ema + hysteresis (matches deployed
                    run_trading.py --cls behavior at portfolio-shape level,
                    minus kelly/vol_scale which are exposure scalers, not
                    Sharpe-affecting at constant exposure)
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd

from _preflight_check import check_versions
check_versions()

import _research_r68_continuous_wf as r68
import _r128_all_overlays_canonical as r128
import _r130_validate_r129 as r130


def line(c="="):
    print(c * 78)


def hdr(t):
    line(); print(f"  {t}"); line()


def metrics(label, rets):
    arr = np.asarray(rets, dtype=float)
    if len(arr) < 2:
        print(f"  {label:<46} EMPTY")
        return None
    S = r130.sharpe(arr)
    So = r130.sortino(arr)
    DD = r130.max_drawdown(arr) * 100
    sum_pct = arr.sum() * 100
    mean_bp = arr.mean() * 1e4
    print(f"  {label:<46} S={S:+.3f}  Sortino={So:+.3f}  DD={DD:+.2f}%  "
          f"sum={sum_pct:+.2f}%  mean={mean_bp:+.2f}bp  n={len(arr)}")
    return dict(label=label, sharpe=S, sortino=So, dd_pct=DD,
                sum_pct=sum_pct, mean_bp=mean_bp, n=len(arr))


def main():
    t0 = time.time()
    hdr("OPUS STEP 3 — OOS PROD-LIKE (R48/R114b state-machine) vs CANONICAL")
    print("\n  Window: 2026-03-18 → 2026-04-25 (39 days OOS)\n")

    reg = pd.read_parquet("cache/opus_oos_regime.parquet")
    if "timestamp" in reg.columns:
        reg = reg.set_index("timestamp")

    PRED_FILES = {
        "R128_W4 (train≤2025-07-01)": "cache/opus_r128_w4_preds.parquet",
        "R132_W4 (train≤2026-01-01)": "cache/opus_r132_w4_preds.parquet",
        "R134_W4 (train≤2026-03-15)": "cache/opus_r134_w4_preds.parquet",
    }

    R114B_CFG = {  # same shape as run_trading.py prod (--cls + meta.json)
        "n_long": 4, "n_short": 2, "rebal_hours": 12,
        "trend_cutoff": 0.9, "dyn_threshold": 0.7,
        "ema_alpha": 0.5, "hysteresis": 3,
    }

    rows = []
    for tag, path in PRED_FILES.items():
        preds = pd.read_parquet(path)
        preds["timestamp"] = pd.to_datetime(preds["timestamp"], utc=True)
        n_ts = preds.timestamp.nunique()
        n_sym = preds.symbol.nunique()
        ic = preds[["pred", "fwd_ret"]].corr(method="spearman").iloc[0, 1]
        print(f"\n  {tag}")
        print(f"    n_rows={len(preds):,} n_ts={n_ts} n_sym={n_sym}  "
              f"range {preds.timestamp.min().date()}→{preds.timestamp.max().date()}  "
              f"IC(spearman)={ic:+.4f}")

        # 1. Canonical skip-risk-off + equal-weight 4L/2S
        port_canon = r128.simulate_full(preds, reg, 4, 2)
        m = metrics(f"  CANON 4L/2S skip-risk-off (R128 sim)", port_canon.net_ret.values)
        if m: m["model"] = tag; m["sim"] = "canon_skip"; rows.append(m)

        # 2. PROD-like R114b state-machine 4L/2S
        port_prod = r68.simulate(preds, reg, 4, 2, cfg=R114B_CFG)
        m = metrics(f"  R114b PROD 4L/2S (state-machine+ema+hyst)",
                     port_prod.net_ret.values)
        if m: m["model"] = tag; m["sim"] = "r114b_prod"; rows.append(m)

        # 3. PROD-like R114b state-machine 6L/3S (sanity, matches meta.json)
        port_prod6 = r68.simulate(preds, reg, 6, 3, cfg=R114B_CFG)
        m = metrics(f"  R114b PROD 6L/3S (sanity, matches meta.json)",
                     port_prod6.net_ret.values)
        if m: m["model"] = tag; m["sim"] = "r114b_6l3s"; rows.append(m)

    # ── FINAL TABLE ─────────────────────────────────────────────
    hdr("FINAL TABLE — sorted by Sharpe desc")
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)
        for _, r in df.iterrows():
            print(f"  S={r.sharpe:+.3f}  Sortino={r.sortino:+.3f}  "
                  f"DD={r.dd_pct:+.2f}%  sum={r.sum_pct:+.2f}%  "
                  f"n={r.n}  | {r.sim:<14} | {r.model}")
        df.to_csv("cache/opus_step3_prod_oos.csv", index=False)
        print(f"\n  Saved → cache/opus_step3_prod_oos.csv")

    print(f"\n  Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
