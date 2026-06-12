#!/usr/bin/env python3
"""R179 — vol-targeting overlay on the s30 deploy artifact. LOCAL, instant.

PRE-REGISTERED (declared before results were seen):
  PRIMARY config: vol_t = std(net_ret, trailing 30 periods); ref_t = expanding
  median of vol_t (min 60 periods); scale_t = clip(ref_t / vol_t, 0.5, 1.5);
  scale defaults to 1.0 while undefined. Levered return = L * scale_t * r_t.
  ADOPTION RULE (3x living config): adopt iff at L=3 BOTH maxDD and worst
  single period improve AND Sharpe degrades by < 0.10 vs unscaled.
  Sensitivity grid (N in {20,48}, clip [0.33,2.0]) is DIAGNOSTIC ONLY.
Caveat: scaling changes turnover; cost effect is second-order at clip<=1.5
(costs already inside net_ret scale ~linearly with notional).
"""
import json
import numpy as np
import pandas as pd

port = pd.read_parquet("cache/r178_s30_port.parquet").sort_values("timestamp").reset_index(drop=True)
r = port["net_ret"].astype(float)
months = pd.to_datetime(port["timestamp"]).dt.to_period("M").nunique()
PPY = 2 * 365


def sharpe(x):
    return float(x.mean() / (x.std() + 1e-12) * np.sqrt(PPY))


def stats(x, L):
    eq = np.cumprod(1 + L * np.asarray(x, dtype=float))
    total = (eq[-1] - 1) * 100
    ann = ((1 + total / 100) ** (12.0 / months) - 1) * 100
    dd = ((eq / np.maximum.accumulate(eq)) - 1).min() * 100
    return {"sharpe": round(sharpe(x), 3), "total": round(float(total), 1),
            "ann": round(float(ann), 1), "dd": round(float(dd), 1),
            "worst": round(float((L * x).min()) * 100, 2)}


def vt_scale(r, n, lo, hi, min_ref=60):
    vol = r.rolling(n, min_periods=n).std()
    ref = vol.expanding(min_periods=min_ref).median()
    sc = (ref / vol).clip(lo, hi)
    return sc.fillna(1.0)


results = {}
print(f"{'конфиг':22s} {'L':>2s} | {'Sharpe':>6s} | {'итог':>8s} | {'годовых':>8s} | {'maxDD':>6s} | {'худш.период':>10s}")
print("-" * 78)
for L in (1, 3):
    base = stats(r, L)
    results[f"base_{L}x"] = base
    print(f"{'без VT':22s} {L}x | {base['sharpe']:6.3f} | {base['total']:+7.1f}% | {base['ann']:+7.1f}% | {base['dd']:+5.1f}% | {base['worst']:+9.2f}%")
    sc = vt_scale(r, 30, 0.5, 1.5)
    vt = stats(r * sc, L)
    results[f"vt_primary_{L}x"] = vt
    print(f"{'VT ПРАЙМЕРИ n30 [0.5,1.5]':22s} {L}x | {vt['sharpe']:6.3f} | {vt['total']:+7.1f}% | {vt['ann']:+7.1f}% | {vt['dd']:+5.1f}% | {vt['worst']:+9.2f}%")
    for n, lo, hi, tag in ((20, 0.5, 1.5, "diag n20"), (48, 0.5, 1.5, "diag n48"),
                           (30, 0.33, 2.0, "diag clip[.33,2]")):
        sc2 = vt_scale(r, n, lo, hi)
        d = stats(r * sc2, L)
        results[f"vt_{tag.replace(' ', '_')}_{L}x"] = d
        print(f"{tag:22s} {L}x | {d['sharpe']:6.3f} | {d['total']:+7.1f}% | {d['ann']:+7.1f}% | {d['dd']:+5.1f}% | {d['worst']:+9.2f}%")
    print("-" * 78)

b3, v3 = results["base_3x"], results["vt_primary_3x"]
adopt = (v3["dd"] > b3["dd"]) and (v3["worst"] > b3["worst"]) and (v3["sharpe"] >= b3["sharpe"] - 0.10)
results["adopt_rule_3x"] = bool(adopt)
print(f"\nПравило (заранее объявленное): улучшить maxDD И худший период на 3x, шарп не хуже −0.10")
print(f"ВЕРДИКТ: {'ПРИНИМАЕМ vol-targeting для 3x' if adopt else 'НЕ принимаем (правило не выполнено)'}")
sc = vt_scale(r, 30, 0.5, 1.5)
print(f"средний масштаб: {sc.mean():.2f}; время на полном размере: {(sc >= 1.49).mean()*100:.0f}%; на минимуме: {(sc <= 0.51).mean()*100:.0f}%")

with open("results_r179_vol_target.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("R179 done.")
