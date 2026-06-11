"""TRACK C — verification math for R143 statistics.

1) 95% CI half-width of an annualized Sharpe measured on n=87 12h periods.
2) Why t_naive/t_NW12 ~ 2.2-2.3 is consistent with (and below) the sqrt(12)
   mechanical-overlap bound: theory + Monte Carlo with the exact _nw_tstat
   from _r143_pristine_oos + empirical check on the canonical preds cache
   (same hourly-pred / 12h-fwd_ret structure as the R143 OOS run).
No training, no load_data — cache only.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ── exact copy of _r143_pristine_oos._nw_tstat (avoid heavy import chain) ──
def _nw_tstat(x, lags):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 5:
        return np.nan
    d = x - x.mean()
    var = d @ d / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        var += 2.0 * w * (d[:-k] @ d[k:]) / n
    se = np.sqrt(max(var, 1e-18) / n)
    return x.mean() / (se + 1e-18)


print("=" * 78)
print("1) Sharpe CI on n=87 12h periods")
print("=" * 78)
n = 87
half = 1.96 * np.sqrt(730 / n)
print(f"   SE(annualized Sharpe) ~ sqrt(730/n) = sqrt(730/87) = "
      f"{np.sqrt(730/n):.3f}")
print(f"   95% CI half-width = 1.96*sqrt(730/87) = {half:.2f}")
print(f"   V2 pristine Sharpe +5.19 -> CI [{5.19-half:+.2f}, {5.19+half:+.2f}]"
      f"  (includes 0)")

print()
print("=" * 78)
print("2) t_naive / t_NW12 inflation: theory")
print("=" * 78)
# Mechanical fully-overlapping 12h windows sampled hourly => IC_t ~ MA(11),
# rho_k = (12-k)/12 for k<12. True large-n VIF of the mean:
rho = np.array([(12 - k) / 12 for k in range(1, 12)])
vif_true = 1 + 2 * rho.sum()
# NW-Bartlett(12) tapers lag-k autocov by w_k = 1-k/13, so even for PURE
# mechanical overlap the MEASURED ratio is below sqrt(12):
w = np.array([1 - k / 13 for k in range(1, 12)])
vif_nw = 1 + 2 * (w * rho).sum()
print(f"   true VIF (mechanical MA(11) overlap) = {vif_true:.2f} "
      f"-> t-ratio upper bound sqrt(12) = {np.sqrt(vif_true):.3f}")
print(f"   NW-Bartlett(12)-measured VIF on same series = {vif_nw:.3f} "
      f"-> measured ratio = {np.sqrt(vif_nw):.3f}")
for r in (2.2, 2.3):
    implied = (r**2 - 1) / 2
    print(f"   observed ratio {r:.1f} -> implied sum(w_k*rho_k) = "
          f"{implied:.2f} = {implied/(w*rho).sum()*100:.0f}% of mechanical")

print()
print("=" * 78)
print("3) Monte Carlo with exact _nw_tstat (n=1033 hourly, 400 reps)")
print("=" * 78)
rng = np.random.default_rng(0)
N, REPS = 1033, 400
ratios = []
for _ in range(REPS):
    z = rng.standard_normal(N + 11)
    x = np.convolve(z, np.ones(12) / 12, mode="valid") + 0.01  # MA(11)+drift
    t_naive = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    ratios.append(t_naive / _nw_tstat(x, 12))
ratios = np.array(ratios)
print(f"   pure mechanical overlap: median measured ratio = "
      f"{np.median(ratios):.2f} (IQR {np.percentile(ratios,25):.2f}-"
      f"{np.percentile(ratios,75):.2f}); sqrt(12)={np.sqrt(12):.2f} "
      f"never reached by NW12")

print()
print("=" * 78)
print("4) Empirical: canonical preds cache (hourly preds, 12h fwd_ret)")
print("=" * 78)
preds = pd.read_parquet("cache/r128_canonical_preds.parquet")
ics = []
for ts, g in preds.groupby("timestamp"):
    if g["pred"].nunique() > 2 and g["fwd_ret"].nunique() > 2:
        ic = spearmanr(g["pred"], g["fwd_ret"]).correlation
        if not np.isnan(ic):
            ics.append(ic)
s = np.asarray(ics)
t_naive = s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))
t_nw = _nw_tstat(s, 12)
# raw lag autocorrelations of the IC series vs mechanical (12-k)/12
d = s - s.mean()
ac = [float(d[:-k] @ d[k:] / (d @ d)) for k in range(1, 12)]
print(f"   n={len(s)} hourly ICs  mean={s.mean():+.4f}  "
      f"t_naive={t_naive:+.2f}  t_NW12={t_nw:+.2f}  "
      f"ratio={t_naive/t_nw:.2f}")
print(f"   IC autocorr lag1/3/6/11 = {ac[0]:.2f}/{ac[2]:.2f}/"
      f"{ac[5]:.2f}/{ac[10]:.2f}  vs mechanical "
      f"{rho[0]:.2f}/{rho[2]:.2f}/{rho[5]:.2f}/{rho[10]:.2f}")
