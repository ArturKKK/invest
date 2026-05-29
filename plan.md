# R105–R108 — Funding Rate Arbitrage: План и Результаты

## Статус: 8 апреля 2026 — ЗАВЕРШЕНО, НЕ ДЕПЛОИМ

---

## Контекст

R68 (directional ML, Sharpe=3.777) — единственная production стратегия. R93 (4h ML) не улучшает R68 при комбинации (R99-R104 все REJECTED). Исследуем новое ортогональное направление: **market-neutral funding arbitrage**.

**Суть**: short perp + long spot = hedge price risk → зарабатываем funding payments.
**Отличие от R91**: R91 использовал FR как ценовой предиктор (провалился). Funding arb **не предсказывает цену** — собирает funding как carry.

**Данные**: Binance 8h funding rates (294K rows, 50 символов, 2020-01 → 2026-03), Binance premium index (496K rows, 48 sym, 2021-12 → 2026-03), OKX funding (14.9K rows, cross-validation).
**Комиссии**: spot taker 0.05%, perp taker 0.03%, round-trip = 0.16%.
**Капитал**: $100 ($50 spot + $50 perp, 1x leverage).

---

## R105 — Historical Analysis ✅ PASS

**Файл**: `_research_r105_funding_analysis.py`

### Top-5 carry coins (annualized)
| Symbol | Mean FR | Ann Carry | % positive | % >0.02% |
|---|---:|---:|---:|---:|
| FTM/USDT | 0.0169% | 18.5% | 88.0% | 14.6% |
| XRP/USDT | 0.0144% | 15.8% | 81.8% | 14.9% |
| LTC/USDT | 0.0144% | 15.8% | 83.1% | 14.3% |
| AAVE/USDT | 0.0141% | 15.4% | 86.2% | 10.8% |
| MKR/USDT | 0.0138% | 15.1% | 88.0% | 12.0% |

### Opportunity frequency
| Threshold | Opps/mo | Coin-opps/mo | % periods with opp | Avg coins |
|---:|---:|---:|---:|---:|
| 0.01% | 45 | 643 | 26.8% | 14.3 |
| 0.03% | 37 | 462 | 19.6% | 12.4 |
| 0.05% | 33 | 357 | 14.1% | 10.9 |
| 0.10% | 22 | 206 | 6.4% | 9.4 |

### FR Persistence
- AC(lag1=8h) = **0.711** — highly persistent
- AC(lag3=24h) = 0.574
- AC(lag6=48h) = 0.490
- AC(lag12=96h) = 0.412

### Regime stability по годам
| Year | Mean FR | Ann Carry | % >0.02% |
|---:|---:|---:|---:|
| 2020 | 0.0108% | 11.8% | 23.4% |
| **2021** | **0.0335%** | **36.7%** | **35.4%** |
| **2022** | **-0.0027%** | **-2.9%** | **0.0%** |
| 2023 | 0.0057% | 6.2% | 5.8% |
| 2024 | 0.0112% | 12.3% | 13.1% |
| **2025** | **0.0017%** | **1.9%** | **0.1%** |
| **2026** | **-0.0148%** | **-16.3%** | **0.6%** |

### Theoretical carry (entry > threshold, hold N periods, pay 0.16% RT)
| Threshold | Hold | Net carry% | Win% | Entries/mo |
|---:|---:|---:|---:|---:|
| 0.05% | 24h | +0.145% | 77.6% | 138 |
| 0.05% | 48h | +0.404% | 93.7% | 138 |
| 0.08% | 24h | +0.251% | 93.7% | 75 |
| 0.08% | 96h | +1.171% | 97.9% | 75 |

Cross-validation Binance↔OKX: correlation=0.679.
**VERDICT: PASS** — 37 opps/month at 0.03% (criterion: ≥5).

---

## R106 — Backtest (без basis risk) ✅ PASS

**Файл**: `_research_r106_funding_arb_backtest.py`

Grid: 5 entry × 3 exit × 4 hold × 3 positions = 156 valid configs.

### Top-5 by Sharpe
| Entry | Exit | Hold | Pos | Sharpe | Ret% | DD% | Win% | Trades |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.080% | 0.005% | 24 | 3 | **6.638** | +14.7% | -0.8% | 66.7% | 147 |
| 0.050% | 0.010% | 24 | 3 | 6.586 | +15.2% | -1.1% | 65.2% | 204 |
| 0.050% | 0.005% | 24 | 3 | 6.537 | +15.2% | -1.0% | 66.2% | 201 |
| 0.080% | 0.010% | 24 | 3 | 6.378 | +14.3% | -0.8% | 65.3% | 150 |
| 0.050% | 0.010% | 24 | 2 | 5.885 | +15.3% | -1.5% | 61.7% | 149 |

Best: entry=0.080%, exit=0.005%, hold=24×8h=192h, max_positions=3.
Funding earned=$22.57, Costs=$7.84, **Hedge P&L=$0.00** (идеальный хедж, basis risk не учтён).

⚠️ Sharpe 6.6 завышен — volatility ≈0 потому что hedge_pnl моделируется как идеальный (spot_pnl + perp_pnl = 0 математически). Нужен R107.

**VERDICT: PASS** — Sharpe=6.638 ≥ 1.0 (criterion).

---

## R107 — Hedge Quality & Basis Risk ✅ PASS

**Файл**: `_research_r107_hedge_quality.py`

### Basis distribution (ΔBasis = изменение premium за 8h)
| Symbol | Basis µ% | ΔBasis σ% | |ΔBasis| µ% |
|---|---:|---:|---:|
| BTC/USDT | -0.029 | 0.023 | 0.014 |
| ETH/USDT | -0.028 | 0.030 | 0.018 |
| ADA/USDT | -0.026 | 0.037 | 0.026 |
| XRP/USDT | -0.027 | 0.037 | 0.022 |

### Worst-case basis moves
| Hold | Mean |Δ|% | σ% | P1/P99% | Worst% |
|---:|---:|---:|---:|---:|
| 8h | 0.027 | 0.054 | -0.10/+0.10 | -11.4 |
| 24h | 0.029 | 0.066 | -0.11/+0.11 | -11.2 |
| 192h | 0.033 | 0.154 | -0.11/+0.11 | -21.3 |

### Basis risk vs funding income — КЛЮЧЕВАЯ ТАБЛИЦА
| Filter | Mean FR% | Basis σ% | Net PnL% | Ratio σ/FR | Ann Sharpe | Win% |
|---|---:|---:|---:|---:|---:|---:|
| All FR>0 | 0.011 | 0.043 | +0.013 | 4.0x | 9.16 | 65% |
| FR>0.03% | 0.049 | 0.063 | +0.054 | **1.3x** | 26.53 | 85% |
| FR>0.05% | 0.066 | 0.077 | +0.079 | **1.2x** | 31.61 | 91% |
| FR>0.08% | 0.098 | 0.127 | +0.150 | **1.3x** | 37.28 | 99% |

→ При high FR (>0.05%) basis risk составляет ~1.2-1.3x от FR, но net PnL остаётся положительным.

### Revised R106 backtest (С basis risk)
| Config | Sharpe | Return% | MaxDD% | Vol% | Trades |
|---|---:|---:|---:|---:|---:|
| Best (entry=0.08%, hold=24) | **2.421** | +1.47% | -0.13% | 0.24% | 12 |
| Alt (entry=0.05%, hold=24) | **2.625** | +2.14% | -0.19% | 0.32% | — |
| Conservative (hold=6) | 0.657 | +0.49% | -0.15% | 0.30% | — |

Sharpe упал с 6.64 → **2.42** (basis risk добавляет реальную volatility).
Всего **12 trades за 4+ года**. Yearly: 2022=0%, 2023=+0.09%, 2024=**+1.36%**, 2025=0%, 2026=0%.

**VERDICT: PASS** — revised Sharpe=2.421 ≥ 1.0, basis_ratio=1.3x < 2x (criterion).

---

## R108 — Paper Trading ❌ FAIL

**Файл**: `_research_r108_funding_arb_paper.py`

### Результат: ZERO opportunities за последние 30 дней

| Config | Entries | Closed Trades | Return |
|---|---:|---:|---:|
| R106_best (entry>0.08%) | **0** | 0 | 0.000% |
| R106_alt (entry>0.05%) | **0** | 0 | 0.000% |

Причина: 2026 FR ≈ -0.015% (mean). Рынок bear/flat → лонги не платят funding → стратегия неактивна.
Deviation from backtest: **-100%** (ожидалось >0, получено 0).

**VERDICT: FAIL** — paper 100% deviation (criterion: kill >50%).

---

## Сводная таблица

| Step | Criterion | Result | Verdict |
|---|---|---|---|
| R105 | ≥5 opps/month | 37 opps/month | ✅ PASS |
| R106 | Sharpe ≥ 1.0 | Sharpe=6.64 (без basis) | ✅ PASS |
| R107 | basis < 2× FR | ratio=1.3x, revised Sharpe=2.42 | ✅ PASS |
| R108 | ≤30% deviation from backtest | **-100% deviation** (0 trades) | ❌ FAIL |

---

## Заключение

Funding rate arbitrage **теоретически работает** в bull/neutral market:
- 2020-2021, 2024: positive FR, 12-37% annualized gross carry
- Sharpe 2.4 после учёта basis risk
- Market-neutral, ортогонально R68

Но **практически бесполезна сейчас**:
- 2022, 2025-2026: FR ≈ 0 или negative
- **0 opportunities** за последние 30 дней
- 12 trades за 4 года при high threshold — слишком редко
- $1-2/year net на $100 капитала — micro-scale

### Решение: **НЕ ДЕПЛОИТЬ**

R68 остаётся единственной production стратегией. Скрипты сохранены для мониторинга — если bull market вернёт positive FR, можно пересмотреть.

Следующие направления для исследования:
1. Новые data sources (CryptoQuant exchange flows — $49/mo)
2. Улучшение feature engineering в R68
3. Другие market-neutral подходы (statistical arbitrage, cross-exchange)

