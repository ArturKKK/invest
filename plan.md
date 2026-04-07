# R110–R112 Plan: Neutralization + Spillover + Factor-Mimicking

## Статус: 8 апреля 2026 — В РАБОТЕ

---

## Контекст

R68 (directional ML, 4L/2S continuous WF, Sharpe=3.777) — единственная production стратегия.
R105-R108 (funding arb) — FAIL/UNPRACTICAL. R109 (macro features) — FAIL (IC < 0.03).
Цель: найти либо Sharpe uplift, либо Calmar/DD uplift без деградации Sharpe.

## Общие инварианты (как R68)
- Canonical loader/sim: те же universe, costs, funding, trend filter, rebalance schedule
- Continuous WF как в R68
- Никаких новых lookahead: все фичи/факторы строго shift1 относительно таргета
- Для каждого R11X сохранять:
  - results/r11X_summary.json
  - results/r11X_equity.csv
  - results/r11X_monthly.csv
  - results/r11X_bootstrap.json (если сравниваем с R68)
  - results/r11X_params.csv / grid.csv

## Acceptance (единое)
- **PASS-A (alpha)**: Sharpe >= R68 + 0.05 и bootstrap P(Sharpe better) > 0.80
- **PASS-B (risk)**: Sharpe >= R68 - 0.05 и Calmar >= R68 + 5% и bootstrap P(Calmar better) > 0.80

---

## R110 — Partial Neutralization Sweep (Numerai-style) поверх R68

### Идея
Уменьшить нежелательные экспозиции prediction-скоринга к "режим/риск" драйверам, не меняя модель.

### R110.1 Выбор экспозиций (3 набора)
- **SET1** (минимальный): beta_to_btc_60, ret_48h
- **SET2** (risk+liquidity): SET1 + rel_volume_cs, rvol_20
- **SET3** (derivs regime): SET2 + cum_funding_24h, oi_velocity

### R110.2 Нейтрализация (cross-sectional ridge per timestamp)
b = (X'X + lambda*I)^{-1} X' p ; p_neut = p - X b
Grid lambda: {0, 1e-3, 1e-2, 1e-1}

### R110.3 Смешивание
p_mix = (1-a)*p + a*p_neut, a in {0, 0.25, 0.5, 0.75, 1.0}

### R110.4 Портфель — строго R68 (4L/2S, trend filter, costs)

### R110.5 Метрики — Sharpe/MaxDD/Return/Calmar, turnover, corr(p_mix, p)

### R110.6 Bootstrap — Block bootstrap B=10, N=1000

Deliverable: results/r110_grid.csv

---

## R111 — Spillover-head (межмонетные лаги + market factors) как новые фичи

### R111.1 Market factors (shift1)
- mkt_ret_12h = mean_cs(ret_12h)
- btc_ret_12h, eth_ret_12h (лидер-лаги)
- dispersion = std_cs(ret_12h) (market stress)
- pc1_ret (PCA на cs-ретёрнах, 1st component)

### R111.2 Per-coin spillover фичи
- beta_i = cov(ret_i, btc_ret)/var(btc_ret) на W=60, shift1
- spill_btc = beta_i * btc_ret_{t-1}
- spill_mkt = corr_i_to_mkt * mkt_ret_{t-1}

### R111.3 IC scan gate — |IC| >= 0.03, stability >= 2/3

### R111.4 Add-only WF (если IC pass)

Deliverable: results/r111_ic_report.csv, results/r111_ablation.csv

---

## R112 — Factor-Mimicking Portfolios (FMP)

### R112.1 Универсальная функция FMP
For each t: normalize z по cs, weights w = z_norm / sum|z_norm|, f_{t+1} = sum(w * r)

### R112.2 FMP-фичи (all lag1)
- fmp_level, fmp_z120, fmp_mom (3-period sum), fmp_tail (>q95 flag)

### R112.3 Прогон на имеющихся proxy (cum_funding_24h, oi_velocity, rel_volume_cs)

### R112.4 Подключение CryptoQuant позже

Deliverable: results/r112_fmp_factors.csv, results/r112_ablation.csv

---

## Execution order
Day 1: R110 (neutralization grid + bootstrap) + R111 (spillover build + IC gate)
Day 2: R111 add-only WF + R112 (FMP pipeline + sanity + add-only)
