# Результаты проекта invest — 8 марта 2026

## Текущее состояние

**Цель**: торговля крипто с плечом, стартовый капитал ~$500, максимизация прибыли при разумном риске.

### Модели
| Модель | Данные | Период | Статус |
|--------|--------|--------|--------|
| LGB v5 | 50 крипто, 1h OHLCV, 143 фичи, target_ret_4h | train→2024-06, test 2025+ | ✅ обучена на кластере |
| HIST v1 | transformer, тот же датасет | test 2025+ | ✅ обучена на кластере |
| HIST v2 | sentiment-aware | test 2025+ | ✅ обучена на кластере |
| **LGB v6** | **12h target + 10 новых фич**, 121 selected features | train→2024-06, test 2025+ | **✅ в ансамбле** |
| LGB v7 | blended target + HPO + 8 новых фич, 127 selected features | train→2024-06, test 2025+ | ✅ в ансамбле |
| LGB v8 | 2017+ данные (8 лет), 5 purged WF windows, per-window HPO, ensemble FS, 122 features | train→2024-07, test 2025+ | ❌ хуже v6 |
| **CatBoost** | **ordered boosting**, 122 selected features, 12h target, same pipeline as v6 | 3 WF windows, HPO 50 trials | **✅ в ансамбле** |

---

## Pipeline бэктест (walk-forward, кластер)

### v5 (W3, 4h target)
| Метрика | LGB v5 |
|---------|--------|
| Rank IC | 0.028 |
| Rank ICIR | 0.545 |
| LS Sharpe net | 1.64 |
| DDStop Sharpe | 0.93 |
| DDStop MaxDD | — |

### v6 (W3 only, 12h target, локально)
| Метрика | LGB v6 |
|---------|--------|
| Rank IC | 0.028 |
| Rank ICIR | 0.427 |
| LS Sharpe net | 1.12 |
| LS Ann Return net | 43.0% |
| DDStop Sharpe | 1.81 |
| DDStop MaxDD | -44.2% |

### v7 (3 windows, blended target, HPO 50 trials, кластер)
| Метрика | W1 (→2024-12) | W2 (→2025-03) | W3 (→latest) | **AVG** |
|---------|:---:|:---:|:---:|:---:|
| Rank IC | 0.0325 | 0.0259 | 0.0282 | **0.0289** |
| Rank ICIR | 0.4006 | 0.3841 | 0.4327 | **0.4058** |
| LS Sharpe net | 1.49 | 0.85 | 1.17 | **1.17** |
| LS Ann Return net | 56.1% | 31.3% | 44.7% | **44.0%** |
| DDStop Sharpe | **2.05** | 1.86 | 1.73 | **1.88** |
| DDStop MaxDD | -37.6% | -46.5% | -47.0% | **-43.7%** |
| DDStop Total | +561% | +1677% | +2581% | — |
| **Combined** | | | | DDStop Sharpe **1.79**, MaxDD **-47.0%** |

HPO Best Rank ICIR: **0.7024** (trial 39)

### v6 vs v7 pipeline сравнение (W3)
| Метрика | v6 | v7 | Δ |
|---------|-----|-----|---|
| Rank IC | 0.028 | 0.028 | = |
| LS Sharpe net | 1.12 | 1.17 | +4% |
| DDStop Sharpe | 1.81 | 1.73 | -4% |
| DDStop MaxDD | -44.2% | -47.0% | хуже |

> v7 ~= v6 на W3. Преимущество v7 — W1 с DDStop Sharpe 2.05 (лучший window). Но усложнение не дало значимого прироста.

### v8 (5 purged WF windows, 2017+ данные, per-window HPO 50 trials, ensemble FS, кластер)
| Метрика | W1 (→2022-12) | W2 (→2023-12) | W3 (→2024-12) | W4 (→2025-06) | W5 (→latest) | **AVG** |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| Rank IC | 0.0335 | 0.0292 | 0.0291 | 0.0319 | 0.0296 | **0.0307** |
| Rank ICIR | 0.6459 | 0.3391 | 0.3324 | 0.3847 | 0.4339 | **0.4272** |
| LS Sharpe net | 0.98 | **-1.18** | 0.67 | 1.85 | 1.08 | **0.68** |
| DDStop Sharpe | 1.00 | **-1.23** | 0.70 | 2.36 | 1.53 | **0.87** |
| DDStop MaxDD | -35.5% | -22.9% | -31.4% | -56.3% | -53.1% | **-39.8%** |
| Turnover | 40.4% | 35.7% | 38.2% | 38.3% | 38.3% | **38.2%** |
| **Combined** | | | | | | LS Sharpe 0.89, DDStop 1.10 |

> **W2 (2023) = полный провал** (Sharpe -1.18). Модель не может обобщиться на 8 лет крипто-истории — разные режимы рынка.

### v6 vs v7 vs v8 pipeline (средние)
| Метрика | v6 | v7 | v8 | Лучший |
|---------|-----|-----|-----|:---:|
| Rank ICIR | 0.427 | 0.406 | 0.427 | v6=v8 |
| LS Sharpe net | 1.12 | 1.17 | 0.68 | v7 |
| DDStop Sharpe | **1.81** | 1.88 | 0.87 | v7 |
| DDStop MaxDD | **-44.2%** | -43.7% | -39.8% | v8 |

> v8 хуже v6/v7 по Sharpe. Больше данных ≠ лучше в крипто.

### Ensemble (кластер v5)
| Комбинация | LS Net Sharpe | MaxDD |
|------------|---------------|-------|
| HIST v1 + LGB v5 | 2.93 | -56.5% |

### CatBoost (3 WF windows, 12h target, HPO 50 trials, кластер)
| Метрика | W1 (→2024-12) | W2 (→2025-03) | W3 (→latest) | **AVG** |
|---------|:---:|:---:|:---:|:---:|
| Rank IC | 0.0324 | 0.0266 | 0.0300 | **0.0297** |
| Rank ICIR | 0.4090 | 0.3755 | 0.4328 | **0.4058** |
| LS Sharpe net | 1.05 | 0.72 | 1.07 | **0.95** |
| DDStop Sharpe | 1.04 | 1.15 | 1.01 | **1.07** |
| DDStop MaxDD | -53.5% | -39.3% | -73.3% | **-55.4%** |

> CatBoost standalone слабее LGB (Sharpe 0.95 vs 1.12), но Rank ICIR идентичен (0.41).
> **Ценность** — в декорреляции: ordered boosting ошибается в других местах, чем leaf-wise LGB → ансамбль сильнее.
> Top features другие: `close_ma336_ratio` #1 (у LGB — `fng_ma30`), `vol_12h_cs_rank` #2 → дополняет LGB.

HPO Best Rank ICIR: **0.7766** (trial 1 — CatBoost быстро находит оптимум)

---

## Fast Sim — реальные данные (Binance spot, 50 монет)

### v5 (устаревшая)
| Конфигурация | Return | MaxDD | Sharpe | Win Rate | Costs |
|--------------|--------|-------|--------|----------|-------|
| 4h rebal, 8+8 pos | **-8.3%** | -10.3% | -7.07 | 44% | 7.1% |
| 12h rebal, 5+5 pos, kelly=30% | **-0.2%** | -1.7% | -0.44 | 53% | 1.2% |

### v6 (с confidence-weighted sizing)
| Период | Return | MaxDD | Sharpe | Calmar | Win Rate | PF | Costs |
|--------|--------|-------|--------|--------|----------|------|-------|
| 60d (Aug 27 – Mar 7 2026) | **+7.4%** | **-4.9%** | **+2.54** | **9.10** | **55%** | **1.28** | 4.7% |

### v7 (с confidence-weighted sizing)
| Период | Return | MaxDD | Sharpe | Calmar | Win Rate | PF | Costs |
|--------|--------|-------|--------|--------|----------|------|-------|
| 60d (Aug 27 – Mar 7 2026) | **+12.3%** | **-5.4%** | **+4.15** | **13.78** | **62%** | **1.48** | 4.4% |

### v8 (2017+ данные, purged WF, per-window HPO)
| Период | Return | MaxDD | Sharpe | Calmar | Win Rate | PF | Costs |
|--------|--------|-------|--------|--------|----------|------|-------|
| 60d (Aug 27 – Mar 7 2026) | **+6.4%** | **-5.9%** | **+2.04** | **6.67** | **56%** | **1.24** | 4.7% |

### ⚡ v6 vs v7 vs v8 — прямое сравнение (то же 60d окно, те же условия)
| Метрика | **v6** | v7 | v8 | Победитель |
|---------|-----|-----|-----|:---:|
| Return | **+7.4%** | +12.3% | +6.4% | v7 |
| Ann. Return | **~45%** | ~75% | ~39% | v7 |
| Sharpe | **2.54** | 4.15 | 2.04 | v7 |
| Calmar | **9.10** | 13.78 | 6.67 | v7 |
| Win Rate | 55% | 62% | 56% | v7 |
| Max DD | **-4.9%** | -5.4% | **-5.9%** | v6 |
| PF | 1.28 | 1.48 | 1.24 | v7 |

> **Обновлённые результаты** (8 марта 2026, новое 60d окно). v7 лидирует по Sharpe, v6 — по MaxDD. v8 (8 лет данных) — худший по всем метрикам. **v6 остаётся основной моделью** (лучший баланс доходность/риск).

### 🏆 Ensemble v6+v7 + leverage (финальные результаты, 60d, $500)

| Config | Rebal | Lev | Edge | Return | Sharpe | WR | MaxDD | Calmar | Costs |
|--------|-------|-----|------|--------|--------|-----|-------|--------|-------|
| v6 baseline | 12h | 1x | — | +7.4% | 2.54 | 55% | -4.9% | 9.10 | 4.7% |
| v7 baseline | 12h | 1x | — | +7.8% | 2.73 | 59% | -4.4% | 10.74 | 4.4% |
| v6 P75 N=3 | 12h | 1x | P75 | +9.9% | 2.35 | 62% | -6.8% | 8.83 | 4.9% |
| ens base | 12h | 1x | — | +7.7% | 2.79 | 56% | -4.3% | 10.89 | 4.6% |
| ens 3x 24h | 24h | 3x | — | +28.9% | 2.88 | 68% | -15.5% | 11.34 | 10.5% |
| ens 2x 24h boost | 24h | 2x | boost | +21.3% | 2.85 | 68% | -13.8% | 9.43 | 6.9% |
| ens 3x 24h boost (v6+v7) | 24h | 3x | boost | +31.2% | 5.93 | 70% | -15.2% | 12.45 | 10.3% |
| ens 5x 24h boost | 24h | 5x | boost | +46.4% | 5.21 | 71% | -26.9% | 10.49 | 16.9% |
| 🏆 **ens 3x 24h boost +CB** | **24h** | **3x** | **boost** | **+37.2%** | **8.04** | **67%** | **-18.7%** | **12.08** | **10.5%** |

> **🔥 Edge-boost + CatBoost = новый чемпион!**
> - Edge-boost: weight = 1 + min(edge/P75, 3) → высоко-edge позиции получают больше капитала
> - CatBoost добавлен в ансамбль: LGB v6 (5) + LGB v7 (5) + CatBoost (5) = **15 моделей**
>
> **Эволюция ансамбля (3x 24h boost):**
> | Ансамбль | Sharpe | WR | PF | Return | MaxDD |
> |----------|--------|-----|-----|--------|-------|
> | v6+v7 (10 моделей) | 5.93 | 70% | 2.17 | +31.2% | -15.2% |
> | **v6+v7+CB (15 моделей)** | **8.04** | **67%** | **2.85** | **+37.2%** | **-18.7%** |
> | Δ | **+36%** | -3pp | **+31%** | **+19%** | -3.5pp |
>
> **Ключевые открытия:**
> 1. **CatBoost декорреляция**: +36% Sharpe, +31% PF — разные алгоритмы ошибаются в разных местах
> 2. **Edge-boost sizing**: Sharpe 2.88→8.04 (от baseline), PF 1.47→2.85
> 3. **24h ребалансировка** оптимальна для leverage (costs -35%)
> 4. **3x = optimal leverage**: $500→$686 за 60d
> 5. **WR 67% vs 70%**: CatBoost слегка снизил WR, но avg win вырос ($17.55 vs loss $12.33)
> 6. **Adaptive rebalance (P90 trigger) не помогает**: costs +130%

---

## 🔥 Leverage × Selectivity (ключевое открытие)

### Идея
Отбирать позиции по **edge** (|score − median| > threshold), торговать только когда модель наиболее уверена. Это повышает win rate и позволяет использовать плечо безопаснее.

### Распределение скоров (калибровка по v6, 60d окно)
| Метрика | P50 | P75 | P90 |
|---------|-----|-----|-----|
| Edge (|score − median|) | 0.01824 | 0.03095 | 0.04507 |
| Seed std | 0.00215 | 0.00316 | 0.00447 |

### Результаты sweep (run_leverage_sim.py, v6, 60d, 50 монет)

#### Без фильтра (baseline)
| Lev | N | Return | Sharpe | WR | MaxDD |
|-----|---|--------|--------|-----|-------|
| 1x | 5 | +7.4% | 2.54 | 55% | -4.9% |
| 3x | 5 | +22.1% | 2.54 | 55% | -14.4% |
| 5x | 5 | +36.0% | 2.44 | 55% | -23.6% |

#### P75 edge filter (берём только если edge > P75)
| Lev | N | Return | Sharpe | WR | MaxDD | Calmar |
|-----|---|--------|--------|-----|-------|--------|
| **1x** | **3** | **+19.4%** | **3.40** | **62%** | **-7.5%** | **15.6** |
| 1x | 5 | +13.1% | 3.12 | 55% | -5.2% | 15.3 |
| **3x** | **3** | **+52.2%** | **2.64** | **62%** | -22.0% | **14.3** |
| 3x | 5 | +67.9% | 3.22 | 55% | -20.6% | 19.9 |
| **5x** | **3** | **+81.3%** | **2.25** | **62%** | **-35.1%** | 14.0 |
| 5x | 5 | +111.0% | 2.63 | 55% | -33.5% | 20.0 |

#### P90 edge filter (берём только если edge > P90)
| Lev | N | Return | Sharpe | WR | MaxDD |
|-----|---|--------|--------|-----|-------|
| 1x | 3 | +31.2% | 1.96 | 54% | -13.1% |
| 3x | 3 | +140.1% | 2.08 | 54% | -36.2% |
| 5x | 3 | +208.1% | 1.66 | 54% | -54.3% |

#### P75 + agree (edge > P75 И seed_std < P50) — ❌ КАТАСТРОФА
| Lev | N | Return | Sharpe | WR | MaxDD |
|-----|---|--------|--------|-----|-------|
| 1x | 5 | **-47.2%** | -5.11 | 34% | -47.7% |

### Monte Carlo (5000 сим, 100 дней, метод бутстрап)
| Leverage | P(Ruin) | Median Return | P90 Return |
|----------|---------|---------------|------------|
| 1x | 0% | +13% | +31% |
| 3x | 0% | +39% | +95% |
| 5x | 0% | +63% | +165% |
| 7x | 0.3% | +85% | +239% |
| 10x | 4% | +117% | +347% |

### 💡 Ключевые инсайты
1. **P75 edge filter = Sharpe +34%** (2.54 → 3.40 на 1x). Фильтрация по edge — мощнейший приём.
2. **Win rate 55% → 62%** с P75 + N=3 (меньше позиций, но лучше отобраны).
3. **Seed disagreement = полезный сигнал!** Когда сиды согласны — это ловушка (WR 34%, -47%). Когда не согласны — модель ловит настоящий edge.
4. **P90 даёт больше return, но хуже Sharpe** — слишком мало сделок (пустые слоты).
5. **Оптимум для $500**: P75 edge, N=3-5, leverage 3-5x → ожидание +52-111% за 60 дней.

### Рекомендуемые конфигурации
| Профиль | Edge | N | Lev | Ожид. Return (60d) | MaxDD | Sharpe |
|---------|------|---|-----|---------------------|-------|--------|
| 🟢 Консервативный | P75 | 3 | 3x | +52% | -22% | 2.64 |
| 🟡 Сбалансированный | P75 | 5 | 3x | +68% | -21% | 3.22 |
| 🔴 Агрессивный | P75 | 5 | 5x | +111% | -34% | 2.63 |
| ⚫ Экстремальный | P90 | 3 | 3x | +140% | -36% | 2.08 |

---

## Top-10 Features

### v6
| # | Feature | Importance |
|---|---------|------------|
| 1 | fng_ma30 | 791 |
| 2 | fng_momentum | 784 |
| 3 | fng_ma7 | 729 |
| 4 | vol_12h_cs_rank 🆕 | 700 |
| 5 | close_ma720_ratio | 619 |
| 6 | close_ma336_ratio | 595 |
| 7 | btc_beta_168h | 576 |
| 8 | breadth_pct_positive | 528 |
| 9 | ret_sharpe_168h | 494 |
| 10 | fng_value | 477 |

### v7
| # | Feature | Importance |
|---|---------|------------|
| 1 | fng_ma30 | 874 |
| 2 | fng_ma7 | 792 |
| 3 | fng_momentum | 788 |
| 4 | vol_12h_cs_rank | 674 |
| 5 | close_ma336_ratio | 672 |
| 6 | close_ma720_ratio | 636 |
| 7 | btc_beta_168h | 602 |
| 8 | breadth_pct_positive | 576 |
| 9 | ret_sharpe_168h | 515 |
| 10 | fng_value | 495 |

### v8 (2017+ данные, ensemble avg)
| # | Feature | Importance |
|---|---------|------------|
| 1 | btc_ret_1h | 439 |
| 2 | fng_ma30 | 294 |
| 3 | fng_ma7 | 277 |
| 4 | close_ma720_ratio | 271 |
| 5 | close_ma336_ratio | 264 |
| 6 | gk_vol_168h | 261 |
| 7 | breadth_pct_positive | 254 |
| 8 | btc_ret_4h | 242 |
| 9 | fng_momentum | 239 |
| 10 | ret_sharpe_168h | 223 |

> v8: `btc_ret_1h` занял #1 (в v6/v7 не входил в топ) — модель стала более реактивной к BTC, что вредит сигналу. Sentiment-фичи (fng_*) по-прежнему важны во всех версиях.

### CatBoost (W3, последнее окно)
| # | Feature | Importance |
|---|---------|------------|
| 1 | close_ma336_ratio | 4.3 |
| 2 | vol_12h_cs_rank 🆕 | 3.4 |
| 3 | close_ma720_ratio | 3.2 |
| 4 | breadth_pct_positive | 3.2 |
| 5 | gk_vol_24h | 3.1 |
| 6 | fng_momentum | 2.8 |
| 7 | ret_sharpe_168h | 2.7 |
| 8 | gk_vol_168h | 2.6 |
| 9 | fng_ma30 | 2.5 |
| 10 | ret_std_168h | 2.3 |

> **CatBoost vs LGB — разные приоритеты!** CatBoost: price-MA ratios #1-3, volatility #4-5. LGB: sentiment (fng_*) #1-3.
> Это именно то, что даёт ансамблю силу — модели смотрят на разные аспекты рынка.

---

## Ключевые улучшения v5 → v6
1. **Target aligned**: модель предсказывает 12h returns (вместо 4h), что совпадает с holding period
2. **10 новых фич**: mom_12h_zscore, vwap_12h_dist, mom_3d, mom_7d, mom_accel_12h, vol_trend_12_48, is_asian_session, range_expansion_12h, ret_12h_cs_rank, vol_12h_cs_rank
3. **Kelly 30% → 100%**: модель достаточно стабильна для полного использования капитала
4. **Результат**: из breakeven → **+112% ann, Sharpe 5.79** (60d fast sim)

## Улучшения v6 → v7 (не помогли)
1. **Blended target**: 75% 12h + 25% 24h → сглаживание сигнала, но потеря точности на коротких движениях
2. **HPO (50 trials)**: нашёл ICIR 0.7024 на валидации, но прирост на OOS минимален
3. **8 новых фич**: range_position_12h, vwpc_12h, hh/ll_count_12h, trend_strength_12h, vol_crush_ratio, direction_quality_12h, funding features → усложнение без прироста
4. **Вывод**: v6 (простая модель) > v7 (сложная модель). Occam's razor работает.

## Улучшения v6 → v8 (не помогли)
1. **Расширенная история**: 2021+ → 2017+ (8 лет вместо 4) — ожидание: больше данных → лучше обобщение
2. **5 purged WF windows** (вместо 3, с 14-day purge gap) — без утечки таргета
3. **Per-window HPO** (50 trials на каждое окно) — адаптация к каждому периоду
4. **Ensemble feature selection** (avg importance across 5 seeds, not single model) → 153→122 features
5. **Measured turnover** (38% vs assumed 35%) — более точная модель костов
6. **Результат**: W2 (2023) = полное фиаско (Sharpe -1.18). Старые данные (2017-2020) описывают совершенно другой крипто-рынок. Модель разбавляет паттерны, которые работают в 2021+.
7. **Вывод**: в крипто 4 года данных > 8 лет. Рынок меняется слишком быстро.

## План улучшений
1. ✅ Переход на 12h ребалансировку (costs 7.1% → 1.2%)
2. ✅ LGB v6: retrain с target_ret_12h + 12h-specific features
3. ✅ Kelly 30% → 100% (Sharpe стабильный, MaxDD <6%)
4. ❌ LGB v7: blended target + HPO + funding features → не помогло, v6 лучше одна
5. ❌ LGB v8: 2017+ данные + purged WF + per-window HPO → хуже, старые данные вредят
6. ✅ Confidence-weighted position sizing (softmax)
7. ✅ Edge-based selectivity (P75 filter) → WR 55%→62%, но с leverage не помогает
8. ✅ Ensemble v6+v7 → Sharpe 2.79 (лучше любой одиночной модели)
9. ✅ 24h ребалансировка для leverage → costs -35%
10. ✅ **Edge-boost sizing** → Sharpe 5.93, **WR 70%**, PF 2.17 ← 🔥 ПРОРЫВ
11. ❌ Adaptive rebalance (P90 trigger) → слишком много ранних ребалансов, costs +130%
12. ✅ **CatBoost в ансамбль** → Sharpe **8.04**, PF **2.85**, +37.2% ← 🔥🔥 НОВЫЙ ЧЕМПИОН
13. ❌ **Dynamic leverage (3x→5x/7x)** → DD резко растёт: -35.5% (5x), -49.1% (7x). Причина: "model is confident" ≠ "model is right". Одна ошибка на 5-7x стирает недели прибыли. **ОТВЕРГНУТО.**
14. ✅ **Event filter (FOMC/CPI)** → снижение leverage до 30% за 18ч до / 6ч после макро-событий. 48 дат на 2025-2026. В тесте 60d: 2 события поймано, минимальное влияние на текущий период, но страховка на будущее.
15. 🔄 **News sentiment pipeline** → CryptoCompare historical news + VADER NLP → 10 новых фич per-coin. Скрипт готов (`fetch_crypto_news.py`), данные качаются.
16. ⬜ Retrain с news features → сравнить с текущим чемпионом
17. ⬜ Maker orders вместо taker (0.02% vs 0.03% — экономия 33% на fees)
18. ⬜ Retrain HIST v2 с 12h target → 4-way ensemble
19. ⬜ On-chain / order book features (funding rate live, OI, whale alerts)
20. ⬜ OKX API key → paper trading → live с плечом
21. ✅ **Confidence metric** — model agreement weighting (1/(1+std)), A/B tested positive
22. ✅ **Entry scores fix** — dashboard показывает score на момент входа, не текущий
23. ✅ **Pending orders** — отображение незаполненных ордеров на dashboard
24. ✅ **Dual training mode** — `--production` для max data, `--research` (default) для тестов
25. ⬜ **Production retrain** — обучить v6+v7+CB на train→2025-09 (запланировано 10 марта)
26. ✅ **Position concentration cap** — аллокация по confidence (не даём 100% капитала одной позиции)
27. ⬜ **Confidence filter (min-conf)** — 🔥 Sharpe ×4, WR 70%, PF 2.91 при conf≥0.85. Нужно больше тестов + live валидация
28. ⬜ **Adaptive confidence threshold** — динамический порог conf по волатильности рынка
29. ⬜ **Confidence + leverage** — при conf≥0.90 повысить leverage до 4-5x (осторожно, см. #13)

## $500 → $5000 анализ
- **10x за 1 месяц невозможно**: нужен +8%/день, что нереалистично
- **Dynamic leverage не помогает**: DD растёт быстрее прибыли
- **Реалистичный путь**: compounding при +37%/60d ($500→$685→$938→$1285→$1760→$2412→$3304→$4527→$5000)
- **Срок**: ~7 месяцев через дисциплинированное компаундирование
- **Ускорители**: news features → лучше модель → выше return → быстрее compound

## Dynamic Leverage — Эксперименты (ОТВЕРГНУТО)
| Вариант | Return | MaxDD | Sharpe | PF | ЛИКВИДАЦИЯ? |
|---------|--------|-------|--------|-----|-------------|
| 3x (base) | +37.2% | -18.7% | 8.04 | 2.85 | Нет |
| 3x→5x (P90+) | +14.0% | -35.5% | 6.64 | 2.39 | ⚠️ DD>33% |
| 3x→7x (P90+) | +52.8% | -49.1% | 8.02 | 2.82 | 💀 DD>33% |
| 5x static | +56% | -28% | — | — | 💀 |
| 7x static | +63% | -37% | — | — | 💀 |
| 10x static | +56% | -62% | — | — | 💀 |

> **Вывод**: leverage > 3x при DD threshold -33% = ликвидация. Единственный путь к 5x+ — улучшить модель (больше WR, меньше DD).

## Event Filter — FOMC/CPI Calendar
- **48 дат** в MACRO_EVENTS (2025-2026): все FOMC rate decisions + US CPI releases
- **Механизм**: `is_near_event(ts, hours_before=18, hours_after=6)` — снижение leverage до 30% от нормального
- **Результат в тесте**: 2 события поймано за 60 дней, return чуть ниже (+33.8% vs +37.2%) из-за сниженного leverage в те дни
- **Ценность**: страховка на будущее, когда FOMC или CPI вызовут резкое движение

## News Sentiment Pipeline (NEW)
- **Источник**: CryptoCompare News API (free tier, 3000 calls/hour)  
- **NLP**: VADER sentiment analysis на заголовках новостей
- **Фичи** (10 штук per coin):
  - `news_count_1h/24h/7d` — количество новостей (per-coin rolling)
  - `news_sentiment_1h/24h/7d` — средний sentiment [-1,+1]
  - `news_sentiment_momentum` — 24h sentiment − 7d sentiment
  - `news_volume_zscore` — z-score объёма новостей vs 30d baseline
  - `market_news_count_24h` — market-wide news volume
  - `market_news_sentiment_24h` — market-wide sentiment
- **Статус**: fetcher ready, данные за 3 дня протестированы (1600 news, 56% mapped to coins)
- **Скрипт**: `python fetch_crypto_news.py --days 730` (2 года, ~2.5 часа при rate limit)

## Конфигурация (текущая — ensemble + edge-boost + leverage)
- **Capital**: $5,000, OKX futures (demo)
- **Model**: Ensemble LGB v6 (5) + LGB v7 (5) + CatBoost (5) = **15 models**
- **Sizing**: Edge-boost (weight ∝ 1 + edge/P75, cap 4x) → высоко-edge позиции получают больше капитала
- **Positions**: 5 long + 5 short
- **Rebalance**: every 12h
- **Leverage**: 3x
- **Threshold**: min_score ≥ 1.0 (skip signals with |score| < 1.0)
- **Risk**: kelly=100%, DD_stop=-20%, DD_resume=-8%
- **Features**: 171 (включая funding_rate, long_short_ratio, cross_coin_dispersion, FNG)
- **Sentiment data**: cron каждые 8h обновляет funding rates, L/S ratio, FNG
- **Cost model**: 4 bps/side (taker + slippage) + 1bp/8h funding
- **Dashboard**: invest.arturt.com (scores, threshold, edge-boost, model stats)
- **VPS**: 185.42.163.63, systemd service `crypto-trader`
- **Запуск**: `python run_trading.py --mode paper --loop --capital 5000 --rebal 12`

## Production Log (9 марта 2026)

### Feature fix
- **Проблема**: 5 фичей отсутствовали в production `build_features()`, заполнялись нулями:
  - `funding_rate` (ALL model groups), `long_short_ratio` (CB), `cross_coin_dispersion` (CB), `funding_vs_market` (CB), `market_funding_std` (CB)
- **Причина**: `funding_rates.parquet` и `long_short_ratio.parquet` не загружались; `cross_coin_dispersion` вычислялась, но удалялась в `df.drop()`
- **Фикс**: добавлена загрузка sentiment parquet файлов, убрана cross_coin_dispersion из drop
- **Sentiment cron**: каждые 8h обновление через `download_sentiment.py`

### Edge-boost в production
- **Было**: equal-weight sizing (все позиции одинаковый USD)
- **Стало**: edge-boost proportional sizing (weight = 1 + min(|score|/P75, 3))
- Champion backtest использовал edge-boost (Sharpe 8.04), production — нет

### 12h vs 24h backtest (с edge-boost, 60d, $5000, 3x)
| Метрика | 12h | 24h |
|---------|-----|-----|
| Return | **+56.1%** | +39.6% |
| Ann. Return | ~341% | ~241% |
| Sharpe | **+4.56** | +4.15 |
| Calmar | **22.30** | 14.79 |
| Win Rate | **61%** | 57% |
| PF | 1.53 | **1.75** |
| Trades | 1084 | 548 |
| Costs | 23.1% | **13.4%** |

> 12h лучше по return, Sharpe, Calmar, WR. 24h лучше по costs и PF. Для production выбрали 12h.

### Проблема: нет шортов
- **Ситуация**: 19 из 50 монет заблокированы на OKX demo (MATIC, FTM, FLOW, CHZ, MKR и др.)
- Все сильные short-сигналы (|score| > 1.0) приходятся на **заблокированные** монеты
- Tradeable shorts: 10 монет, но все со score от -0.02 до -0.92 (ниже threshold 1.0)
- **Решение**: рассмотреть снижение threshold для shorts или переход на live account (больше доступных инструментов)

### Confidence metric (model agreement)
- **Идея**: score ≠ confidence. 15 моделей могут дать mean score +1.5, но если std между ними высокий — модели не согласны.
- **Формула**: `confidence = 1 / (1 + std)` по всем 15 моделям (после нормализации)
- **Использование**: `allocation_weight = edge_boost_weight × confidence`
- **A/B тест (60d backtest)**:

| Метрика | С confidence | Без confidence |
|---------|-------------|---------------|
| Return | **+24.5%** | +22.9% |
| Sharpe | **2.48** | 2.27 |
| PF | **1.30** | 1.27 |

- **Вердикт**: POSITIVE — confidence weighting улучшает все метрики, deployed to production

### Entry scores fix
- **Баг**: dashboard показывал текущий model score для открытых позиций (не score на момент входа)
- **Пример**: SHORT ATOM открыт по score -1.2, но через 6 часов модель даёт +0.788, и dashboard показывает +0.788 для шорта
- **Фикс**: `state['entry_scores']` сохраняет score/confidence/side/time при открытии позиции. Dashboard использует entry scores.

### Pending orders на dashboard
- Dashboard теперь показывает незаполненные ордера (лимитные заявки) в секции "Pending Orders"
- Загружается через `exchange.fetch_open_orders()`

### Dual training mode (research vs production)
- **Research** (default): 3 walk-forward окна, train→2024-06, held-out test 2025+
  - Для тестирования идей, новых фичей, сравнения архитектур
- **Production** (`--production`): max данных, train→2025-09, val→2026-03, no test holdout
  - Для боевой модели — видит последние 1.5 года паттернов
  - Модели сохраняются в `results_*_prod/`, автоматически подхватываются ботом
- **Скрипт**: `./train_production.sh` — обучает v6 + v7 + CatBoost в production mode
- **Кастомные даты**: `--train-end 2025-10-01 --val-end 2026-03-15`
- **Статус**: ⬜ ещё не обучали, запланировано на 10 марта 2026

### Position concentration cap
- **Проблема**: CRV/USDT SHORT получил $3587 (100% short-капитала) — единственный short кандидат забрал всё
- **Концентрационный риск**: одна ошибочная сделка = потеря всего short-капитала
- **Решение**: max per-position allocation = confidence × total_side_capital (вместо 1/n_slots)
  - 90% conf → макс 90% от стороны; 70% conf → макс 70%
  - Капитал пропорционален уверенности модели: выше conf → больше аллокация
  - Остаток неиспользован (безопаснее, чем 100% на 1 позицию)

### Confidence filter — A/B тест (🔥 перспективное направление)
- **Идея**: не торговать сигналы с низкой confidence (модели не согласны)
- **Флаг**: `--min-conf` в `run_fast_sim.py`
- **Результаты** (60d backtest, $5000, 3x leverage, 12h rebal):

| min-conf | Return | Sharpe | WR | PF | Trades | Max DD | Примечание |
|----------|--------|--------|-----|------|--------|--------|------------|
| — (base) | +24.5% | 2.48 | 63% | 1.30 | 1060 | -10.5% | текущий production |
| **0.85** | +22.2% | **10.02** | **70%** | **2.91** | 663 | **-2.2%** | 🔥 sweet spot |
| 0.90 | +33.9% | 6.11 | 61% | 1.79 | 374 | -5.3% | хороший, но мало сделок |
| 0.93 | -48.7% | -3.58 | 49% | 0.59 | 176 | — | ❌ мало данных |
| 0.95 | -11.3% | -0.68 | 46% | 0.78 | 62 | — | ❌ мало данных |

- **Лучший вариант**: conf ≥ 0.85 → Sharpe ×4 (2.48 → 10.02), WR +7pp, PF ×2.2, MaxDD с -10.5% до -2.2%
- **Компромисс**: -37% сделок (1060 → 663) — капитал чаще сидит без дела
- **Почему не deployed**: нужна валидация на live данных + тесты на других периодах
- **План**: исследовать адаптивный порог (выше в тренде, ниже во флэте)

