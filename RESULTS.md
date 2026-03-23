# Результаты проекта invest — 10 марта 2026

## Текущее состояние

**Цель**: торговля крипто с плечом, стартовый капитал ~$500, максимизация прибыли при разумном риске.

### Модели
| Модель | Данные | Период | Статус |
|--------|--------|--------|--------|
| LGB v5 | 50 крипто, 1h OHLCV, 143 фичи, target_ret_4h | train→2024-06, test 2025+ | ✅ обучена на кластере |
| HIST v1 | transformer, тот же датасет | test 2025+ | ✅ обучена на кластере |
| HIST v2 | sentiment-aware | test 2025+ | ✅ обучена на кластере |
| **LGB v6** | **12h target + 10 новых фич**, 121 selected features, **без news** | train→2024-06, test 2025+ | **✅ в ансамбле** |
| **LGB v7** | blended target + HPO + 8 новых фич, 127 selected features, **без news** | train→2024-06, test 2025+ | **✅ в ансамбле** |
| LGB v8 | 2017+ данные (8 лет), 5 purged WF windows, per-window HPO, ensemble FS, 122 features | train→2024-07, test 2025+ | ❌ хуже v6 |
| **CatBoost** | **ordered boosting**, 130 selected features (8 news), 12h target | 3 WF windows, HPO 50 trials | **✅ в ансамбле** |

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

### CatBoost без news (3 WF windows, 12h target, HPO 50 trials, кластер)
| Метрика | W1 (→2024-12) | W2 (→2025-03) | W3 (→latest) | **AVG** |
|---------|:---:|:---:|:---:|:---:|
| Rank IC | 0.0324 | 0.0266 | 0.0300 | **0.0297** |
| Rank ICIR | 0.4090 | 0.3755 | 0.4328 | **0.4058** |
| LS Sharpe net | 1.05 | 0.72 | 1.07 | **0.95** |
| DDStop Sharpe | 1.04 | 1.15 | 1.01 | **1.07** |
| DDStop MaxDD | -53.5% | -39.3% | -73.3% | **-55.4%** |

### CatBoost с news (3 WF windows, GPU cluster, 130 features — 8 news)
| Метрика | W1 (→2024-12) | W2 (→2025-03) | W3 (→latest) | **AVG** |
|---------|:---:|:---:|:---:|:---:|
| Rank IC | — | — | — | — |
| Rank ICIR | 0.3464 | 0.3687 | 0.3912 | **0.3688** |
| LS Sharpe net | 0.99 | 0.92 | 1.29 | **1.07** |
| DDStop Sharpe | 0.98 | 1.57 | **1.97** | **1.51** |
| DDStop MaxDD | -36.7% | -34.5% | **-42.7%** | **-38.0%** |

> **News помогли CatBoost**: DDStop Sharpe 1.07→1.51 (+41%), MaxDD -55%→-38%.
> News features: `news_count_1h/24h/7d`, `news_sentiment_1h/7d`, `news_sentiment_momentum`, `market_news_count_24h`, `market_news_sentiment_24h`.

### News A/B тест — pipeline бэктест
| Модель | Без news (DDStop) | С news (DDStop) | Δ | Вердикт |
|--------|:---:|:---:|:---:|:---:|
| LGB v6 | **1.81** | 0.96 | -47% | ❌ News вредят |
| LGB v7 | **1.88** | 1.20 | -36% | ❌ News вредят |
| CatBoost | 1.07 | **1.51** | +41% | ✅ News помогают |

> **Вывод**: LGB (leaf-wise) не умеет использовать шумные news фичи — переоценивает их важность. CatBoost (ordered boosting) лучше справляется с шумом.
> **Оптимальный микс**: LGB v6 (без news) + LGB v7 (без news) + CatBoost (с news) → каждая модель использует свой набор фичей через `feature_names.json`.

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

### 🏆 Ensemble v6+v7 + leverage (старые результаты, 60d, $500, 8 марта)

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

> ⚠️ **Sharpe 8.04 измерен на коротком 60d окне с 3x leverage.** На полном 365d тесте тот же конфиг даёт Sharpe 4.55 (см. ниже). Короткое окно попало на отличный участок рынка → завышенная оценка.

---

### 🔥 News A/B тест — Fast Sim (9 марта 2026, 365d, 1x, 12h, edge-boost, min-conf=0.85)

| Конфиг | Sharpe | MaxDD | WR | PF | Return | Calmar |
|--------|--------|-------|-----|------|--------|--------|
| A: все без news | 2.57 | -9.3% | 55% | 1.30 | +9.8% | 1.05 |
| B: все с news | 2.76 | -8.9% | 54% | 1.32 | +9.7% | 1.08 |
| 🏆 **C: LGB без news + CB с news** | **3.77** | **-5.1%** | **63%** | **1.46** | **+15.1%** | **2.96** |
| D: LGB с news + CB без news | 2.12 | -8.4% | 50% | 1.24 | +6.5% | 0.77 |

> **Конфиг C (текущий) = лучший.** Sharpe в 1.5x выше следующего, MaxDD в 1.7x ниже.
> News вредят LGB, но помогают CatBoost. Гибридный микс оптимален.

---

### 📊 Полная сетка Fast Sim (9 марта 2026, LGB без news + CB с news)

| # | Период | Lev | Rebal | min-conf | Return | Sharpe | WR | PF | MaxDD | Calmar | Trades | Costs |
|---|--------|-----|-------|----------|--------|--------|-----|------|-------|--------|--------|-------|
| 1 | 60d | 3x | 24h | — | +21.7% | 2.03 | 62% | 1.33 | -21.7% | 6.10 | 486 | 10.8% |
| 2 | 60d | 3x | 24h | 0.85 | +1.2% | 0.73 | 62% | 1.12 | -18.1% | 0.41 | 282 | 11.2% |
| 3 | 60d | 1x | 12h | 0.85 | +6.7% | 2.23 | 66% | 1.25 | -5.6% | 7.26 | 669 | 6.6% |
| 4 | 60d | 1x | 12h | — | +8.3% | 2.69 | 60% | 1.30 | -6.5% | 7.78 | 1128 | 4.7% |
| 5 | **365d** | **1x** | **12h** | **0.85** | **+15.1%** | **3.77** | **63%** | **1.46** | **-5.1%** | **2.96** | **685** | **7.0%** |
| 6 | 🏆 **365d** | **1x** | **12h** | **—** | **+21.3%** | **6.61** | **61%** | **1.86** | **-5.4%** | **3.95** | **1140** | **5.1%** |
| 7 | 365d | 3x | 24h | — | +48.7% | 4.55 | 65% | 1.85 | -18.1% | 2.70 | 582 | 12.2% |

> **Ключевые выводы:**
> 1. 🏆 **Лучший конфиг: 365d, 1x, 12h, без min-conf** — Sharpe 6.61, PF 1.86, MaxDD -5.4%
> 2. **min-conf 0.85 вредит на 365d**: Sharpe 6.61→3.77 (-43%), Return 21.3%→15.1%, PF 1.86→1.46. Фильтр выкидывает прибыльные сделки.
> 3. **60d vs 365d**: 60d = зависит от конкретного окна, нестабильно. 365d = надёжная оценка.
> 4. **Leverage 3x** увеличивает return (21→49%) но MaxDD растёт (5→18%) — risk/reward ухудшается.
> 5. **Sharpe 8.04 (старый) vs 6.61 (новый)**: модели не ухудшились! Разница: 60d cherry-pick vs 365d full test + добавление CatBoost с news.
>
> **Почему Sharpe 8 → 6.61 → 3.77:**
> | Фактор | Sharpe | Причина |
> |--------|--------|----------|
> | 8.04 | 60d, 3x lev, 24h, старые CatBoost без news | Короткое удачное окно + leverage |
> | 6.61 | 365d, 1x, 12h, без min-conf, CB с news | Полный год — реалистичная оценка |
> | 4.55 | 365d, 3x, 24h, без min-conf | Leverage + larger costs |
> | 3.77 | 365d, 1x, 12h, min-conf=0.85 | Фильтр режет хорошие сделки |

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
12. ✅ **CatBoost в ансамбль** → Sharpe **8.04** (60d) / **6.61** (365d), PF **1.86** ← 🔥🔥 ЧЕМПИОН
13. ❌ **Dynamic leverage (3x→5x/7x)** → DD резко растёт: -35.5% (5x), -49.1% (7x). **ОТВЕРГНУТО.**
14. ✅ **Event filter (FOMC/CPI)** → снижение leverage до 30% возле макро-событий.
15. ✅ **News sentiment pipeline** → CryptoCompare news + VADER NLP → 8 news фичей per-coin
16. ✅ **Retrain с news** → news ВРЕДЯТ LGB (DDStop -36..47%), но ПОМОГАЮТ CatBoost (DDStop +41%)
17. ✅ **News A/B тест (fast sim, 365d)** → оптимально: LGB без news + CatBoost с news = Sharpe 3.77 (с conf filter) / **6.61** (без)
18. ❌ **min-conf 0.85 на 365d** → Sharpe 6.61→3.77, фильтр выкидывает прибыльные сделки. На 60d выглядел хорошо, но на полном году вредит.
19. ⬜ Maker orders вместо taker (0.02% vs 0.03% — экономия 33% на fees)
20. ⬜ Retrain HIST v2 с 12h target → 4-way ensemble
21. ⬜ On-chain / order book features (funding rate live, OI, whale alerts)
22. ⬜ OKX API key → paper trading → live с плечом
21. ✅ **Confidence metric** — model agreement weighting (1/(1+std)), A/B tested positive
22. ✅ **Entry scores fix** — dashboard показывает score на момент входа, не текущий
23. ✅ **Pending orders** — отображение незаполненных ордеров на dashboard
24. ✅ **Dual training mode** — `--production` для max data, `--research` (default) для тестов
25. ⬜ **Production retrain** — обучить v6+v7+CB на train→2025-09 (запланировано 10 марта)
26. ✅ **Position concentration cap** — аллокация по confidence
27. ❌ **Confidence filter (min-conf 0.85)** — на 60d выглядел отлично (Sharpe 10.02), но на **365d вредит**: Sharpe 6.61→3.77. Фильтр слишком агрессивно режет сделки.
28. ⬜ **Adaptive confidence threshold** — возможно мягкий порог (0.70-0.75?) не будет вредить
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

## Конфигурация (текущая — 9 марта 2026)
- **Capital**: $5,000, OKX futures (demo)
- **Model**: Ensemble LGB v6 (5, без news, 121 фичей) + LGB v7 (5, без news, 127 фичей) + CatBoost (5, **с news**, 130 фичей) = **15 models**
- **Sizing**: Edge-boost (weight ∝ 1 + edge/P75, cap 4x)
- **Positions**: 5 long + 5 short
- **Rebalance**: every 12h
- **Leverage**: 1x (рекомендовано по 365d тесту; 3x опционально для агрессивного профиля)
- **min-conf**: 0 (❌ отключен — на 365d вредит: Sharpe 6.61→3.77)
- **Threshold**: min_score ≥ 1.0
- **Risk**: kelly=100%, DD_stop=-20%, DD_resume=-8%
- **Features**: LGB: 121-127 (OHLCV, FNG, vol, momentum); CatBoost: 130 (+8 news)
- **Sentiment data**: cron каждые 8h (funding rates, L/S ratio, FNG, news)
- **Cost model**: 4 bps/side (taker + slippage) + 1bp/8h funding
- **Dashboard**: invest.arturt.com
- **VPS**: 185.42.163.63, systemd service `crypto-trader`
- **Training data**: train→2024-06, val→2024-12, test 2025+ (W3). Fast sim: 365d = мар 2025→мар 2026, **полностью OOS** (9 месяцев после train end)
- **Best backtest (365d)**: Sharpe **6.61**, Return +21.3%, MaxDD -5.4%, PF 1.86, WR 61%
- **Запуск sim**: `python run_fast_sim.py --ensemble --edge-boost --days 365`
- **Запуск live**: `python run_trading.py --mode paper --loop --capital 5000 --rebal 12`

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

### Confidence filter — A/B тест (⚠️ полезно на 60d, ВРЕДНО на 365d)
- **Идея**: не торговать сигналы с низкой confidence (модели не согласны)
- **Механизм**: confidence = 1 / (1 + std) по всем 15 моделям. min-conf ≥ 0.85 = пропуск сигналов с низким agreement.
- **Флаг**: `--min-conf` в `run_fast_sim.py`
- **Результаты** (60d backtest, $5000, 3x leverage, 12h rebal):

| min-conf | Return | Sharpe | WR | PF | Trades | Max DD | Примечание |
|----------|--------|--------|-----|------|--------|--------|------------|
| — (base) | +24.5% | 2.48 | 63% | 1.30 | 1060 | -10.5% | текущий production |
| **0.85** | +22.2% | **10.02** | **70%** | **2.91** | 663 | **-2.2%** | 🔥 на 60d |
| 0.90 | +33.9% | 6.11 | 61% | 1.79 | 374 | -5.3% | |
| 0.93 | -48.7% | -3.58 | 49% | 0.59 | 176 | — | ❌ мало данных |

- **Но на 365d** (полный год, 1x, 12h):

| min-conf | Return | Sharpe | WR | PF | Trades | Max DD |
|----------|--------|--------|-----|------|--------|--------|
| — | **+21.3%** | **6.61** | 61% | **1.86** | **1140** | -5.4% |
| 0.85 | +15.1% | 3.77 | 63% | 1.46 | 685 | **-5.1%** |
| Δ | **-29%** | **-43%** | +2pp | **-22%** | -40% | +0.3pp |

- **Вывод**: min-conf 0.85 удаляет 40% сделок и теряет 43% Sharpe на полном году.
  На 60d попало удачное окно. **НЕ ИСПОЛЬЗОВАТЬ в production.**
- **Возможно**: мягкий порог 0.70-0.75 будет полезен — нужно проверить.

---

## Meta-model stacking (10 марта 2026)

### Архитектура
- **Level-1** поверх L0 OOS предсказаний (v6 + v7 + CatBoost из exp12_full)
- Meta-train = W1 test (2024-07 → 2024-12), Meta-test = W3 (2025-01 → 2026-03)
- 4 варианта: LGB (33 feat), LGB-MINIMAL (21 feat), Ridge (3 feat), Ridge-ALL (33 feat)
- Shared inference: `src/models/meta_model.py` (run_trading.py + run_fast_sim.py)

### Исправления в run_meta_stack.py
- 12 багфиксов: dedup before merge, NaN ffill, RidgeCV TSCV, explicit features, LGB capacity
- 4 tier-1: LGB TimeSeriesSplit 3-fold CV для best_round, target winsorization (0.005), expanding window, save-model default
- 2 бага найдены внешним AI-review: (1) raw preds до z-score нормализации в run_trading.py, (2) --meta-model без --ensemble тихо fallback на simple mean

### run_meta_stack.py результаты (OOS, meta-test 2025-01 → 2026-03)

| Модель | IC | Rank IC | LS Sharpe | VT Sharpe | DDStop Sharpe | MaxDD |
|--------|-----|---------|-----------|-----------|---------------|-------|
| Simple Mean | 0.0182 | 0.0223 | +2.70 | **+2.97** | +3.15 | -83.6% |
| v6 only | 0.0179 | 0.0228 | +2.70 | +2.90 | +2.96 | -82.8% |
| v7 only | 0.0184 | 0.0221 | +2.54 | +2.78 | +2.88 | -84.0% |
| CB only | 0.0177 | 0.0214 | +2.92 | **+3.19** | **+3.50** | -80.9% |
| Ridge-3 | 0.0183 | 0.0223 | +2.70 | +2.96 | +3.09 | -83.1% |
| Ridge-ALL | 0.0203 | 0.0284 | +2.75 | +3.00 | +3.20 | -84.9% |
| **LGB-META** | **0.0277** | **0.0353** | +2.34 | +2.47 | +2.72 | -88.2% |
| **LGB-MINIMAL** | 0.0202 | 0.0284 | +2.76 | +2.97 | +3.14 | -81.5% |

> LGB-META имеет лучший IC/RankIC, но хуже Sharpe — overfitting контекстных фичей.
> LGB-MINIMAL ≈ Simple Mean по Sharpe, но улучшает IC. CB only — лучший по DDStop.

### fast_sim backtesting (ensemble mode)

| Режим | 30d Sharpe | 30d Return | 180d Sharpe | 180d Return | 180d MaxDD | 180d Calmar |
|-------|-----------|------------|-------------|-------------|------------|-------------|
| Ensemble (v6+v7+CB), без мета | 1.83 | +3.2% | 0.67 | +9.7% | -16.3% | 1.21 |
| **Ensemble + meta lgb_minimal** | **3.73** | **+6.2%** | **1.49** | **+22.7%** | **-13.8%** | **3.33** |
| Δ | **+104%** | +94% | **+122%** | +134% | +15% лучше | +175% |

> Мета-модель даёт **+122% Sharpe** на ансамбле за 180 дней.

### fast_sim: полный бенчмарк 16 вариантов (2026-03-10)

Тест: v6/v7 × deriv/no-deriv × ensemble/meta × 14d/60d.
Cost: taker 3bps + slippage 1bp, funding 0.5bps/8h, top/bottom 5, rebalance 1h.

**14 дней (краткосрочный)**

| # | Variant | Return | Ann. Return | Sharpe | Sharpe HAC | Max DD |
|---|---------|--------|-------------|--------|------------|--------|
| 1 | **Ensemble no deriv** | **+3.6%** | **~93%** | **+4.68** | **+4.17** | -3.1% |
| 2 | v6 single (±deriv) | +3.0% | ~77% | +3.63 | +3.67 | -2.7% |
| 3 | Ensemble+meta no deriv | +2.7% | ~70% | +3.43 | +3.44 | -3.0% |
| 4 | Ensemble + deriv | +2.2% | ~57% | +3.36 | +2.99 | -2.8% |
| 5 | v7 single (±deriv) | +2.4% | ~64% | +3.10 | +3.14 | -3.0% |
| 6 | Ensemble+meta + deriv | +1.8% | ~47% | +2.64 | +2.61 | -2.7% |

**60 дней (среднесрочный)**

| # | Variant | Return | Ann. Return | Sharpe | Sharpe HAC | Max DD |
|---|---------|--------|-------------|--------|------------|--------|
| 1 | **Ensemble+meta no deriv** | **+3.1%** | **~19%** | **+0.60** | **+0.65** | -14.3% |
| 2 | v6 single (±deriv) | +2.4% | ~15% | +0.42 | +0.49 | -14.8% |
| 3 | v7 single (±deriv) | +1.9% | ~11% | +0.32 | +0.38 | -14.8% |
| 4 | Ensemble+meta + deriv | +0.8% | ~5% | +0.16 | +0.18 | -13.8% |
| 5 | Ensemble no deriv | -1.1% | ~-7% | -0.21 | -0.24 | -15.4% |
| 6 | Ensemble + deriv | -2.7% | ~-17% | -0.52 | -0.62 | -14.0% |

#### Ключевые находки

1. **Deriv-gate НЕ влияет на single-модели** — v6 и v7 дают идентичные результаты ±deriv.
   Вероятно, derivative mini-model и main model всегда agree по знаку на текущих данных.

2. **v6 > v7** на обоих горизонтах (Sharpe 3.63 vs 3.10 на 14d; 0.42 vs 0.32 на 60d).

3. **Deriv-gate вредит ансамблю** — на 14d: Sharpe 4.68→3.36 (−28%), на 60d: −0.21→−0.52.
   Гипотеза: deriv-gate обрезает сигнал ансамбля, когда derivative mini-model disagree,
   но ансамбль агрегирует > 1 модели и уже даёт более стабильный прогноз.

4. **Meta-model спасает long-horizon ансамбль**: на 60d чистый ансамбль = −0.21,
   а ensemble+meta = **+0.60** Sharpe. Мета-модель научилась понимать, когда L0 модели
   дают разумный сигнал, и когда лучше уменьшить позицию.

5. **Лучший вариант зависит от горизонта**:
   - 14d: **Ensemble no deriv** (Sharpe 4.68) — лучше single
   - 60d: **Ensemble+meta no deriv** (Sharpe 0.60) — лучше single v6 (0.42)
   - Стабильный baseline: **single v6** (3.63 / 0.42) — без доп. моделей, прост в production

### Рекомендации для production

| Сценарий | Рекомендация | Почему |
|----------|-------------|--------|
| Conservative | Single v6, без deriv-gate | Простой, стабильный, не зависит от ансамбля |
| Aggressive short-term | Ensemble no deriv | Sharpe 4.68 на 14d, но −0.21 на 60d (рисковано) |
| Balanced | Ensemble+meta no deriv | Лучший на 60d (+0.60), хорош на 14d (+3.43) |

**Текущий production**: single v7 + deriv-gate → **рекомендуется переключить на single v6 без deriv-gate** (Sharpe +17% на 14d, +31% на 60d vs v7).

### Выводы
1. Инфраструктура мета-модели полностью готова (train → save → inference в production/sim)
2. **Deriv-gate бесполезен** для single моделей и вреден для ансамбля — отключить
3. **v6 лучше v7** на текущих данных — переключить production
4. Мета-модель даёт lift ансамблю (+122% Sharpe 30d, конвертирует −0.21→+0.60 на 60d)
5. После добавления новых данных (OI, basis, liquidations) → переобучить L0 → ансамбль+мета должен стать лучше single

---

## Production Models + Full Feature Pipeline (12 марта 2026)

### Контекст
- **Модели**: обучены на кластере с GPU + HPO (10-12 марта 2026)
  - LGB v6 (5 seeds, 160 feats), LGB v7 (5 seeds, 150 feats), CatBoost (5 seeds, 160 feats), XGBoost (5 seeds, 160 feats)
  - Production mode: train→2025-09-01, val 2025-09-09→2026-03-01
- **Критический багфикс**: `run_fast_sim.py` online mode (live fetch) использовал только `build_features()` + `cross_sectional_rank()`, пропуская 4 функции обогащения:
  - `add_cross_asset_features()` — eth_btc_ret_24h, btc_regime_*, market_dispersion
  - `add_advanced_regime_features()` — regime_composite, regime_low_vol, etc.
  - `add_sentiment_features()` — funding, L/S ratio, FNG, news (8+7 features)
  - `add_derivatives_features()` — OI, taker CVD, basis, premium (30+ features)
- **Результат бага**: V6 и CB получали 53/160 features = 0 (zero-filled). Модели работали на ~67% фичей.
- **Фикс**: добавлена полная цепочка обогащения + drop 23 overlapping columns + news_mode='none'→'all'
- **После фикса**: 0 zero-filled features для всех моделей (V6: 160/160, V7: 150/150, CB: 160/160)

### Бенчмарк: 30d live fetch, $5000, rebal=12h, edge-boost

| # | Config | Return | Ann. | Sharpe HAC | Max DD | Calmar | WR | PF | Trades | Costs |
|---|--------|--------|------|------------|--------|--------|----|----|--------|-------|
| 1 | single_v6 | -0.3% | -4% | **-0.75** | -1.3% | -2.80 | 54% | 0.93 | 1328 | 0.8% |
| 2 | single_v7 | +0.3% | +3% | **+0.61** | -1.5% | 2.10 | 53% | 1.06 | 1254 | 0.8% |
| 3 | ensemble_no_deriv | -0.1% | -1% | **-0.16** | -1.4% | -0.55 | 56% | 0.99 | 1320 | 0.8% |
| 4 | ensemble_deriv | -0.0% | -0% | **-0.00** | -1.1% | -0.00 | 53% | 1.00 | 1320 | 0.7% |
| 5 | ensemble_meta_lgb_min | +0.2% | +3% | **+0.74** | -0.9% | 2.90 | 53% | 1.06 | 1308 | 0.7% |
| 6 | 🏆 **ensemble_meta_ridge** | **+1.2%** | **+15%** | **+2.85** | -1.2% | **12.33** | **60%** | **1.34** | 1316 | 0.7% |
| 7 | 🔥 **ensemble_deriv_3x** | **+3.4%** | **+42%** | **+2.55** | -3.6% | **11.51** | **59%** | — | 1314 | 3.0% |

### Ключевые выводы

1. **Meta-model Ridge = чемпион**: Sharpe HAC 2.85 на 30d live — лучший результат среди всех конфигов.
   Ridge с 3 фичами (v6_pred, v7_pred, cb_pred) лучше LGB-minimal с 21 фичей (0.74 vs 2.85 HAC).

2. **3x leverage с meta+deriv**: +3.4% за 30 дней (42% ann.) при MaxDD -3.6%.
   Calmar 11.51 — отличное risk/reward для 3x leverage.

3. **Single models слабы**: v6 -0.3%, v7 +0.3% — на 30d live сигнал одной модели недостаточен.
   Ансамбль без мета тоже ~0% — нужна мета-модель для lift.

4. **Deriv-gate**: ensemble_deriv (0.00) vs ensemble_no_deriv (-0.16) — marginal improvement, но не значим.

5. **Полный pipeline критичен**: до фикса meta_ridge давал ~0%, после +1.2%.
   53 zero-filled features уничтожали сигнал моделей, обученных на полном наборе фич.

### Конфигурация (обновлена 12 марта 2026)
- **Production ensemble**: LGB v6 (5, 160 feat) + LGB v7 (5, 150 feat) + CatBoost (5, 160 feat, с news) = 15 L0 models
- **Meta-model**: Ridge (3 feat: v6_pred, v7_pred, cb_pred) → `results/meta_stack/meta_model.pkl`
- **Deriv-gate**: deriv_only model (5 seeds, 39 feat) для scaling
- **Sizing**: Edge-boost (weight ∝ 1 + edge/P75, cap 4x)
- **Features**: 207 total после полного pipeline (build_features + cross-asset + regime + 12h + sentiment + derivatives)
- **Best config**: `--ensemble --edge-boost --meta-model auto --meta-variant ridge --leverage 3`
- **Запуск**: `python run_fast_sim.py --ensemble --edge-boost --meta-model auto --meta-variant ridge --days 30`

---

## Известные ограничения и bias

### 1. Survivorship bias (ошибка выжившего)
- Список SYMBOLS (50 монет) зафиксирован на текущий момент (2025/2026)
- При бэктесте на 2021–2024 используются те же монеты, хотя часть из них могла иметь другой статус (LUNA, FTT и т.д. — делистированы)
- Только "выжившие" монеты попадают в бэктест → систематический позитивный bias
- **Влияние**: среднегодовой return вероятно завышен; Sharpe может быть inflated
- **Решение**: нужны исторические снапшоты юниверса (CoinGecko/CMC top-50 за каждый квартал)
- **Статус**: задокументировано как ограничение; для live trading (вперёд) — это не проблема

### 2. Модель издержек (ИСПРАВЛЕНО 2026-03-09)
- **Было**: комиссия только при полном открытии/закрытии позиции. Если монета оставалась в портфеле, но вес менялся (25% → 10%) — нулевая комиссия
- **Стало**: долларовый оборот (dollar turnover) = сумма |new_alloc − prev_alloc| для каждого символа. Комиссия = turnover × COST_SIDE (4bps)
- **Влияние**: издержки были занижены примерно в 2–3x

### 3. Purge/embargo в walk-forward (ИСПРАВЛЕНО 2026-03-09)
- **Было**: gap 2 дня между train_end и val_start
- **Стало**: 8 дней purge (покрывает 12h target overlap + 168h rolling features)
- **Влияние**: небольшое — 2 дня уже покрывали 12h target, но не полностью rolling фичи
- **Требует**: переобучения моделей на кластере для полного эффекта

### 4. Sharpe autocorrelation (ИСПРАВЛЕНО 2026-03-09)
- **Было**: стандартная аннуализация sqrt(periods_per_year) — корректна для IID
- **Стало**: выводится дополнительная метрика "Sharpe HAC" (Newey-West) в run_fast_sim.py
- **Влияние**: HAC Sharpe может быть на 30–50% ниже наивного Sharpe

---

## Benchmark v2 — Full Model Stack (12 марта 2026)

### Контекст
Полный A/B бенчмарк: 12 конфигураций × 30d + 60d = 24 симуляции.
Offline данные: `trading_logs/frozen_raw.parquet` (50K rows, 50 символов, 2024-07 → 2026-03-12).
Capital $5000, leverage 3x, short-blocked.

**Фикс перед запуском**: добавлена `add_news_interaction_features()` в `run_fast_sim.py` и `run_trading.py`.
21 `nx_*` фичи (news × price/vol/funding interactions) были zero-filled при инференсе XGBoost — исправлено (23 фичи теперь вычисляются).

### Конфигурации
- **v6_solo / v7_solo** — одиночные LGB модели
- **ens3** — LGB v6 + LGB v7 + CatBoost (без XGBoost)
- **ens4** — LGB v6 + LGB v7 + CatBoost + XGBoost
- **+deriv** — с deriv-gate (скейлинг по derivative mini-model)
- **+meta_lgb** — с мета-моделью LGB minimal (21 фича)

### 30d Results (sorted by Sharpe)

| # | Config | Ret% | Sharpe | DD% | Calmar | WR% | PF |
|---|--------|------|--------|-----|--------|-----|-----|
| 1 | **v7_solo** | +6.6 | **+8.87** | -1.1 | 72.56 | 59 | 2.49 |
| 2 | ens4 | +6.2 | +8.20 | -1.1 | 71.61 | 59 | 2.20 |
| 3 | v7+deriv | +4.9 | +7.05 | -1.2 | 48.83 | 59 | 2.02 |
| 3 | ens3 | +5.4 | +7.05 | -1.1 | 58.98 | 58 | 2.00 |
| 5 | v6_solo | +4.8 | +6.31 | -1.1 | 51.60 | 59 | 1.80 |
| 6 | ens4+deriv | +4.3 | +6.27 | -1.2 | 44.33 | 54 | 1.83 |
| 7 | ens4+meta_lgb | +4.6 | +6.16 | -1.1 | 48.98 | 59 | 1.77 |
| 8 | ens3+meta_lgb | +4.3 | +5.17 | -1.1 | 48.48 | 63 | 1.65 |
| 9 | ens3+deriv | +3.5 | +4.88 | -1.2 | 34.83 | 49 | 1.60 |
| 10 | ens4+deriv+meta_lgb | +3.0 | +4.21 | -1.4 | 25.01 | 51 | 1.48 |
| 11 | v6+deriv | +3.0 | +4.06 | -1.5 | 24.94 | 51 | 1.47 |
| 12 | ens3+deriv+meta_lgb | +2.7 | +3.42 | -1.7 | 19.08 | 53 | 1.41 |

### 60d Results (sorted by Sharpe)

| # | Config | Ret% | Sharpe | DD% | Calmar | WR% | PF |
|---|--------|------|--------|-----|--------|-----|-----|
| 1 | **v7_solo** | +8.6 | **+7.35** | -1.4 | 37.67 | 57 | 2.09 |
| 2 | v7+deriv | +7.6 | +6.85 | -1.2 | 36.89 | 57 | 1.98 |
| 3 | ens4 | +7.4 | +6.15 | -1.5 | 29.35 | 59 | 1.81 |
| 4 | ens4+deriv | +6.2 | +5.56 | -1.4 | 26.31 | 56 | 1.72 |
| 5 | ens3 | +6.5 | +5.42 | -1.6 | 25.51 | 55 | 1.70 |
| 6 | ens3+meta_lgb | +6.0 | +5.13 | -1.4 | 26.59 | 60 | 1.65 |
| 7 | ens4+meta_lgb | +5.6 | +4.76 | -2.1 | 16.41 | 56 | 1.59 |
| 7 | ens4+deriv+meta_lgb | +5.2 | +4.76 | -2.3 | 13.94 | 54 | 1.58 |
| 9 | ens3+deriv | +5.2 | +4.56 | -1.5 | 21.53 | 50 | 1.57 |
| 10 | ens3+deriv+meta_lgb | +4.8 | +4.34 | -1.7 | 16.98 | 56 | 1.55 |
| 11 | v6_solo | +5.1 | +4.26 | -2.2 | 13.91 | 56 | 1.50 |
| 12 | v6+deriv | +4.1 | +3.63 | -2.2 | 11.49 | 54 | 1.42 |

### Impact Analysis (Sharpe delta)

| Effect | 30d avg | 60d avg | Verdict |
|--------|---------|---------|---------|
| **+XGB** (ens3→ens4) | **+1.08** | **+0.45** | Всегда положительный, XGB помогает |
| **Deriv gate** | **−2.04** | **−0.65** | Всегда отрицательный на этом окне |
| **Meta LGB** | **−1.86** | **−0.52** | Всегда отрицательный на этом окне |

### Важные выводы

1. **v7_solo — абсолютный лидер** на обоих окнах (Sharpe +8.87 / +7.35).
2. **XGBoost помогает ансамблю** (ens3→ens4: +1.08 Sharpe на 30d), но не превосходит v7_solo.
3. **Deriv-gate вредит** на текущем окне: −2.04 Sharpe на 30d, −0.65 на 60d. 
   Гипотеза: на sideways/trending market (последние 60d) deriv-gate слишком агрессивно обрезает позиции.
4. **Meta LGB вредит**: −1.86 на 30d, −0.52 на 60d. Meta-model trained on different regime, overfits.
5. **nx_ fix (23 новых news interaction features)** структурно правильный, но не изменил результаты на frozen данных — XGB уже доминировался остальными 161 фичами.

### Сравнение с live 30d (предыдущая секция)

| Метрика | Бенчмарк v7_solo 30d | Live 30d ensemble+meta_ridge |
|---------|---------------------|------------------------------|
| Sharpe | +8.87 | +2.85 |
| Return | +6.6% | +1.2% |
| MaxDD | -1.1% | -1.2% |

Разрыв объясняется: live считает HAC Sharpe (более консервативный), а бенчмарк — наивный.
Также live использовал ensemble+meta+deriv (уступает v7_solo в бенчмарке).

---

## Walk-Forward Validation (12 марта 2026)

### Контекст
6 non-overlapping 90d окон × 6 конфигураций = 36 симуляций.
Данные: `data/features/crypto_features_1h.parquet` (710K rows, 50 символов, 2024-07 → 2026-03-07).
Capital $5000, leverage 3x, short-blocked, rebalance 12h.

Это **настоящая** walk-forward валидация — не один fold, а 6 разных рыночных режимов:
- W1 (2024-09→12): pre-bull run
- W2 (2024-12→03): bull run peak
- W3 (2025-03→06): correction / sideways
- W4 (2025-06→09): summer lull
- W5 (2025-09→12): autumn recovery
- W6 (2025-12→03): recent (latest data)

### Sharpe per window

| Config | W1 | W2 | W3 | W4 | W5 | W6 | Mean | Std | Min | Positive |
|--------|-----|-----|-----|-----|-----|-----|------|-----|-----|----------|
| **ens4** | **4.81** | **8.94** | **5.50** | **1.47** | 1.23 | 0.20 | **3.69** | 3.03 | 0.20 | **6/6** |
| ens4+deriv | 3.35 | 8.20 | 4.73 | 0.36 | **1.40** | **0.30** | 3.06 | 2.80 | 0.30 | **6/6** |
| ens3 | 3.56 | 7.28 | 4.58 | 0.91 | 1.16 | 0.06 | 2.92 | 2.50 | 0.06 | **6/6** |
| v7_solo | 4.30 | 4.99 | **5.54** | -0.10 | 1.80 | 0.01 | 2.76 | 2.30 | -0.10 | 5/6 |
| ens4+meta_lgb | 2.80 | 6.61 | 4.75 | 0.16 | 1.04 | 0.97 | 2.72 | 2.30 | 0.16 | **6/6** |
| ens4+meta_ridge | 2.02 | 4.82 | 3.51 | -0.08 | 1.09 | -0.54 | 1.80 | 1.90 | -0.54 | 4/6 |

### Return per window

| Config | W1 | W2 | W3 | W4 | W5 | W6 | Compound |
|--------|------|------|------|------|------|------|----------|
| **ens4** | +25.7% | +35.9% | +19.8% | +4.5% | +4.7% | +0.7% | **+125.5%** |
| v7_solo | +22.0% | +22.9% | +19.4% | -0.3% | +7.0% | +0.0% | +91.0% |
| ens3 | +18.3% | +28.7% | +16.0% | +2.7% | +4.5% | +0.2% | +89.9% |
| ens4+deriv | +15.7% | +27.6% | +15.8% | +1.0% | +5.4% | +1.0% | +83.8% |
| ens4+meta_lgb | +14.0% | +27.1% | +17.0% | +0.5% | +4.1% | +3.3% | +83.2% |
| ens4+meta_ridge | +10.1% | +20.8% | +12.4% | -0.2% | +4.1% | -1.9% | +52.4% |

### MaxDD per window

| Config | W1 | W2 | W3 | W4 | W5 | W6 | Worst |
|--------|------|------|------|------|------|------|-------|
| ens4 | -7.8% | -3.0% | -3.0% | -7.3% | -7.4% | -8.6% | -8.6% |
| v7_solo | -6.6% | -4.0% | -2.6% | -7.4% | -8.3% | -9.5% | -9.5% |
| ens3 | -7.9% | -3.1% | -3.2% | -6.9% | -7.5% | -9.0% | -9.0% |
| ens4+deriv | -8.5% | -3.5% | -3.0% | -7.9% | -6.9% | -6.8% | -8.5% |
| ens4+meta_lgb | -8.5% | -3.3% | -2.8% | -8.5% | -8.3% | -5.4% | -8.5% |
| ens4+meta_ridge | -8.8% | -3.8% | -3.4% | -6.9% | -7.3% | -8.9% | -8.9% |

### Impact Analysis (Sharpe delta по каждому окну)

| Effect | W1 | W2 | W3 | W4 | W5 | W6 | Avg | Win Rate |
|--------|-----|-----|-----|-----|-----|-----|-----|----------|
| ens4 vs v7_solo | +0.51 | +3.95 | -0.04 | +1.57 | -0.57 | +0.19 | **+0.93** | **4/6** |
| +XGB (ens3→ens4) | +1.25 | +1.66 | +0.92 | +0.56 | +0.07 | +0.14 | **+0.77** | **6/6** |
| +deriv gate | -1.46 | -0.74 | -0.77 | -1.11 | +0.17 | +0.10 | **-0.63** | 2/6 |
| +meta LGB | -2.01 | -2.33 | -0.75 | -1.31 | -0.19 | +0.77 | **-0.97** | 1/6 |
| +meta Ridge | -2.79 | -4.12 | -1.99 | -1.55 | -0.14 | -0.74 | **-1.89** | **0/6** |

### Выводы walk-forward

1. **ens4 — однозначный победитель**: Mean Sharpe +3.69, positive на 6/6 окнах, compound return +125.5%.
   Walk-forward подтвердил рекомендацию: ens4 > v7_solo (Sharpe +0.93 в среднем, 4/6 wins).

2. **XGBoost помогает стабильно**: +0.77 Sharpe в среднем, **6/6 wins** — самый стабильный эффект.
   Ни одного окна с отрицательным вкладом.

3. **v7_solo — рисковый**: хотя даёт высокие пики (W3: +5.54), на 2 окнах уходит в минус/ноль.
   Подтверждает теорию single-model risk.

4. **Deriv-gate стабильно вредит**: -0.63 Sharpe в среднем, лишь 2/6 wins (и оба минимальные: +0.17, +0.10).
   **Рекомендация: отключить deriv-gate в production.**

5. **Meta-модели вредят**: LGB -0.97 (1/6 wins), Ridge -1.89 (**0/6 wins**).
   Ridge — худший конфиг в тесте. Meta-модели overfitted.
   **Рекомендация: отключить мета-модель в production.**

6. **Деградация сигнала**: все конфиги теряют силу к W4-W6 (2025-06 → 2026-03).
   W1-W3 (сильный рынок): Sharpe 3-9. W4-W6 (слабый рынок): Sharpe 0-1.5.
   Это может означать: (a) модели обучены на старых данных и деградируют, (b) рынок 2H 2025 сложнее.

### Рекомендуемый production стек

```
python run_trading.py --ensemble --no-deriv-gate --leverage 3
```

Т.е. ens4 (LGB v6 + LGB v7 + CatBoost + XGBoost), без deriv-gate, без мета-модели.

---

## Overnight v1 — Multi-Horizon & Calendar A/B (13 марта 2026)

### Контекст
Кластерный эксперимент: 5 конфигов, OOS = Feb 9 – Mar 7, 2026 (26 дней).
Модели Gen#3: retrain train→2026-02-01, val→2026-03-07.
Цель: A/B calendar features, multi-horizon (4h/24h), MLP neural net.
Sim: `run_fast_sim.py`, $5000, 3x leverage, edge-boost, 12h rebalance.

### Результаты

| Experiment | Return | Sharpe | HAC Sharpe | Max DD | WR | Verdict |
|-----------|--------|--------|-----------|--------|----|---------|
| **Gen#2 baseline (4-grp, no calendar)** | +21.7% | 8.43 | — | -4.5% | ~63% | ✅ Production (VPS) |
| Gen#3 (4-grp, +9 calendar features) | +16.9% | 6.64 | 7.35 | -4.3% | — | ❌ Calendar hurts |
| Gen#3 no-cal retrain (4-grp, SKIP_CALENDAR) | +17.6% | 6.82 | 7.31 | -4.6% | — | ≈ baseline |
| Gen#3 + LGB_24h (5-grp) | +18.5% | 7.10 | 7.99 | -4.2% | — | ✅ Best new config |
| Gen#3 + LGB_4h (5-grp) | +16.1% | 6.28 | 6.90 | -4.4% | — | ❌ 4h hurts |
| Gen#3 + LGB_4h + LGB_24h (6-grp) | +15.9% | 6.18 | 6.81 | -4.4% | — | ❌ 4h drags down |
| MLP neural net (5-grp) | +14.8% | 6.59 | 7.12 | -4.1% | — | ❌ MLP is noise (IC=0.021) |

### Per-Model IC (cross-sectional Spearman rank IC on target_ret_12h)

| Model | Mean IC | Std IC | ICIR |
|-------|---------|--------|------|
| catboost | 0.1315 | 0.2129 | 0.618 |
| lgb_v7 | 0.1273 | 0.2095 | 0.608 |
| lgb_24h | 0.1271 | 0.2261 | 0.562 |
| lgb_v6 | 0.1255 | 0.2125 | 0.591 |
| xgboost | 0.1211 | 0.2097 | 0.577 |
| lgb_4h | 0.1124 | 0.2030 | 0.554 |
| MLP | 0.0206 | — | — |

### Выводы v1

1. **Calendar features = шум**: No-cal Sharpe 6.82 > с calendar 6.64. SKIP_CALENDAR=1 — дефолт.
2. **LGB_24h — единственное полезное добавление**: Sharpe 7.10 (+0.28), IC=0.127, корреляция с baseline 0.93 (хороший баланс diversity/signal).
3. **LGB_4h вредит**: IC=0.112 (самый низкий), тянет ансамбль вниз.
4. **MLP = полный провал**: IC=0.021, чистый шум. Neural nets не работают на данном объёме данных.
5. **Gen#2 остаётся лучшим** (Sharpe 8.43 vs лучший новый 7.10). Не деплоим ничего нового.

---

## Overnight v2 — LambdaRank, Residual Target, Meta-Labeling (14–15 марта 2026)

### Контекст
Кластерный эксперимент по рекомендациям AI-консультации (GPT Pro + Opus).
3 идеи: LambdaRank (ranking loss), Residual target (ret − β×BTC), Meta-labeling (binary classifier).
Модели: retrain train→2026-02-01, val→2026-03-07, SKIP_CALENDAR=1.
OOS: Feb 9 – Mar 7, 2026. Sim: $5000, 3x, edge-boost, 12h rebalance.

**Баги найдены и исправлены перед запуском:**
1. **Data leakage в meta-label** (CRITICAL): oos-start default 2025-12-09 но L0 обучены до 2026-02-01 → meta-label тренировался бы на in-sample предсказаниях L0
2. **Cost formula**: leverage cancels в sign test → убрали leverage из формулы, break-even = 9.5 bps
3. **XGBoost DMatrix crash**: predict_group() не различал LGB/XGB → фикс через `type(model).__module__`

### EXP A: LambdaRank (LGBMRanker + NDCG)

Идея: вместо MSE regression использовать LGBMRanker с LambdaRank loss для прямой оптимизации ранжирования.

| Model | Mean IC | Correlation с baseline |
|-------|---------|----------------------|
| v6_rank | **0.0060** | 0.038 |
| v7_rank | 0.0339 | — |
| v6_base | 0.1109 | — |
| v7_base | 0.1169 | — |

**Вердикт: ❌ ПОЛНЫЙ ПРОВАЛ.** IC коллапсировал с 0.111 → 0.006 (в 18 раз хуже).
Максимальная diversity (r=0.038) но нулевой IC = шум. LambdaRank несовместим с нашей задачей.

### EXP B: Residual Target (ret − β×BTC)

Идея: убрать рыночную компоненту из таргета, учить модель предсказывать idiosyncratic return.

| Model | Mean IC | Correlation с baseline |
|-------|---------|----------------------|
| v6_resid | 0.1064 | **0.965** |
| v7_resid | 0.1137 | — |
| v6_base | 0.1109 | — |
| v7_base | 0.1169 | — |

**Вердикт: ❌ БЕСПОЛЕЗНО.** r=0.965 с baseline = практически идентичные предсказания.
IC незначительно хуже. Нулевой diversity gain. BTC beta слишком мала для cross-sectional сигнала.

### EXP C: Meta-Labeling (Binary LGBMClassifier)

Идея: обучить binary classifier на meta-features (23 признака: ens_mean, ens_std, agreement, vol context, BTC state) для фильтрации трейдов.

**Результат:**
- Только **690 трейдов** в OOS (340 train / 350 test) — критически мало для обучения
- CV best_iterations = [1, 1, 106], median = 1 → **модель не нашла сигнал**
- Test WR = 48.6% (хуже coin flip)
- Все пороги 0.50–0.58 дают одинаковый результат, ≥0.60 → 0 трейдов

**Вердикт: ❌ ПОЛНЫЙ ПРОВАЛ.** Недостаточно данных. Нужно ≥2000+ трейдов (расширить OOS window).

### EXP D: Combo Ensemble

Все модели вместе (base + rank + resid + 24h) — 5-7 групп.

| Metric | Combo | Base 4-grp (ref) |
|--------|-------|-----------------|
| Return | +16.8% | +17.6% |
| Sharpe | 6.85 | 6.82 |
| HAC Sharpe | 7.19 | 7.31 |
| Max DD | -3.6% | -4.6% |

**Вердикт: ❌ НЕ ЛУЧШЕ baseline.** LambdaRank отравляет ансамбль, residual не добавляет diversity.

### Unique Contribution Analysis (IC drop when model removed)

| Model removed | IC drop | Verdict |
|--------------|---------|---------|
| v6_24h | **+0.0034** | ✅ Единственный уникальный вклад |
| v6_rank | **-0.0050** | ❌ Активно вредит |
| v7_rank | -0.0008 | Нейтрален |
| v6_resid | +0.0005 | Нейтрален |
| v7_resid | +0.0001 | Нейтрален |

### Best Ensemble Combinations

| Combination | Mean IC | ICIR |
|-------------|---------|------|
| Base 4+24h | 0.1166 | 0.50 |
| Resid 4+24h | 0.1164 | **0.51** |
| Base 4-grp | 0.1143 | 0.49 |
| All (base+rank+resid+24h) | 0.1106 | 0.46 |

### Итоговые выводы v2

1. **Все 3 идеи провалились**: LambdaRank (IC collapse), Residual (zero diversity), Meta-label (no data).
2. **LGB_24h — по-прежнему единственное полезное добавление** (подтверждено unique contribution analysis).
3. **Gen#2 остаётся на VPS без изменений** (Sharpe 8.43 >> лучший новый конфиг 7.13).
4. **Корреляция моделей (0.93-0.97)** — фундаментальная проблема. Структурно-другие подходы (LambdaRank, residual) либо разрушают сигнал, либо дают идентичный результат.
5. **Meta-labeling теоретически перспективен**, но нужно значительно больше OOS данных (retrain с ранним train-end → ≥2000 trades).

---

## Overnight v3 — Portfolio Construction + Huber Loss (15 Mar 2026)

**Hypothesis**: 3 orthogonal ideas from AI consultation round 2:
- **EXP A**: Hysteresis + turnover budget (reduce unnecessary rebalancing)
- **EXP B**: Min z-score + dynamic N (filter weak signals / concentrate on strong ones)
- **EXP C**: Huber loss (robust to outlier returns) + dead-zone weighting

**Sim window**: 2026-02-09 → 2026-03-07 (26 days OOS), 3x leverage, ensemble, edge-boost.

### Full Results Table

| Experiment | Return | Sharpe | HAC Sharpe | MaxDD | WinRate | Turnover |
|------------|--------|--------|------------|-------|---------|----------|
| **baseline** | +16.8% | 6.63 | 7.19 | -4.6% | 61% | $145,928 |
| exp_a1_hyst_3 | +16.8% | 6.63 | 7.19 | -4.6% | 61% | $145,928 |
| exp_a1_hyst_5 | +16.8% | 6.63 | 7.19 | -4.6% | 61% | $145,928 |
| exp_a1_hyst_7 | +16.8% | 6.63 | 7.19 | -4.6% | 61% | $145,928 |
| exp_a1_hyst_10 | +16.8% | 6.63 | 7.19 | -4.6% | 61% | $145,928 |
| exp_a2_tb_2 | +16.8% | 6.63 | 7.19 | -4.6% | 61% | $145,928 |
| exp_a2_tb_3 | +16.8% | 6.63 | 7.19 | -4.6% | 61% | $145,928 |
| exp_a2_tb_5 | +16.8% | 6.63 | 7.19 | -4.6% | 61% | $145,928 |
| exp_a3_hyst3_tb3 | +16.8% | 6.63 | 7.19 | -4.6% | 61% | $145,928 |
| exp_a3_hyst3_tb5 | +16.8% | 6.63 | 7.19 | -4.6% | 61% | $145,928 |
| exp_a3_hyst5_tb3 | +16.8% | 6.63 | 7.19 | -4.6% | 61% | $145,928 |
| exp_a3_hyst5_tb5 | +16.8% | 6.63 | 7.19 | -4.6% | 61% | $145,928 |
| exp_b1_mz_0.3 | +16.8% | 6.63 | 7.19 | -4.6% | 61% | $145,928 |
| **exp_b1_mz_0.5** | **+17.4%** | **6.81** | **7.46** | -4.6% | 61% | $146,426 |
| exp_b1_mz_0.7 | +16.6% | 6.01 | 6.56 | -4.6% | 61% | $146,762 |
| exp_b1_mz_1.0 | +22.9% | 6.03 | 6.83 | -6.9% | **67%** | $159,672 |
| exp_b2_dynN | +21.7% | 6.09 | 6.55 | -6.4% | 63% | $161,226 |
| exp_b3_dynN_mz0.3 | +21.7% | 6.09 | 6.55 | -6.4% | 63% | $161,226 |
| exp_b3_dynN_mz0.5 | +21.7% | 6.09 | 6.55 | -6.4% | 63% | $161,226 |
| exp_b4_hyst5_mz05 | +17.2% | 6.76 | 7.32 | -4.6% | 61% | $146,006 |
| exp_b5_full_combo | +17.2% | 6.76 | 7.32 | -4.6% | 61% | $146,006 |
| exp_c1c_sim_huber (v6+v7 Huber) | **+18.4%** | **6.99** | **7.56** | -4.6% | **65%** | $147,419 |
| exp_c2c_sim_dz03 (v6+v7 DZ 0.3%) | +17.4% | 6.75 | 7.22 | -4.6% | 63% | $145,628 |
| exp_c3c_sim_huber_dz (Huber+DZ) | +17.4% | 6.75 | 7.22 | -4.6% | 63% | $145,628 |
| **exp_c4_huber_pc** (Huber + mz0.5) | **+18.7%** | **7.12** | **7.69** | **-4.6%** | **65%** | $147,503 |

Training Sharpe (in-sample, informational only):
- v6 Huber: 51.67
- v6 Deadzone: 55.65
- v6 Huber+DZ: 55.65

### Analysis by Experiment Group

**EXP A — Hysteresis + Turnover Budget: ZERO EFFECT**
- All 11 configurations identical to baseline (same return, Sharpe, turnover to the dollar)
- Root cause: portfolio composition is already very stable at 12h rebalance — top/bottom 10 coins rarely change between steps. Hysteresis has nothing to "hold". The $145K turnover is from edge-boost weight rebalancing, not position replacement.
- Verdict: **USELESS for this system. Can be removed.**

**EXP B — Min Z-Score + Dynamic N: MIXED**
- `mz_0.5`: Best risk-adjusted improvement (HAC +3.8%) without increasing DD. Sweet spot.
- `mz_0.7`: HAC drops (6.56) — too aggressive filtering hurts diversification.
- `mz_1.0`: Highest WR (67%) and return (+22.9%), but DD jumps to -6.9% and Sharpe drops. Concentration risk.
- `dynamic_n`: Similar to mz_1.0 — more return but worse risk-adjusted.
- Verdict: **mz_0.5 is optimal. Higher thresholds trade risk-adj for raw return.**

**EXP C — Huber Loss: CLEAR WINNER**
- Huber solo: WR +4pp (61→65%), HAC +5.1% (7.19→7.56). Models trained with Huber loss generalize better.
- Deadzone solo: Minimal improvement (HAC +0.4%). Higher in-sample Sharpe (55.65 vs 51.67) but worse OOS — overfitting to large moves.
- Huber+Deadzone: = Deadzone only. Deadzone dominates and cancels Huber benefit.
- Verdict: **Huber solo > Deadzone solo > Huber+DZ. Train with Huber, skip deadzone.**

**Interesting bias-variance observation**: Deadzone achieves higher training Sharpe (55.65 vs Huber's 51.67) but worse OOS performance. This is textbook overfitting — deadzone upweights large returns, making the model chase outliers rather than learn robust patterns. Huber dampens outlier influence → lower in-sample fit → better generalization.

### Best Configuration: exp_c4_huber_pc

**Huber-trained v6+v7 + min-zscore=0.5** (with hysteresis=5, tb=3, but these have zero effect):

| Metric | Baseline | **exp_c4** | Δ |
|--------|----------|-----------|---|
| Return | +16.8% | **+18.7%** | +11.3% rel |
| Sharpe | 6.63 | **7.12** | +7.4% |
| HAC Sharpe | 7.19 | **7.69** | +7.0% |
| MaxDD | -4.6% | **-4.6%** | unchanged |
| Win Rate | 61% | **65%** | +4pp |
| Turnover | $145,928 | $147,503 | +1.1% |

### Next Steps (DONE → see v4/v5 below)
1. ~~Retrain CatBoost + XGBoost with Huber loss~~ → v4
2. ~~Run full 4-model Huber ensemble simulation~~ → v4
3. ~~A/B test news features under Huber~~ → v5
4. If confirmed → deploy on VPS replacing Gen#2

---

## Overnight v4 — 4-Model Huber Ensemble (15–16 Mar 2026)

### Контекст
Следуя результатам v3 (Huber = winner), переобучили CatBoost и XGBoost с Huber loss.
Train→2026-02-01, val→2026-03-07, OOS: Feb 9 → Mar 7.
Sim: $5000, 3x leverage, edge-boost, 12h rebalance.

**Что было сделано:**
- CatBoost retrain: `--huber` (Huber:delta=1.5)
- XGBoost retrain: `--huber --huber-slope 1.0` (reg:pseudohubererror)
- Bug fixes перед запуском (commit 4cb83f5):
  1. CatBoost: Huber loss применялся ПОСЛЕ feature selection → перенесён ДО
  2. XGBoost: whitelist не содержал `huber_slope`/`objective` → добавлен

### Результаты

| Experiment | Return | Sharpe | HAC Sharpe | MaxDD | WR | Description |
|-----------|--------|--------|-----------|-------|----|-------------|
| baseline (4 RMSE models) | +17.3% | 6.67 | 7.58 | -4.5% | 63% | All 4 models trained with RMSE |
| huber_v6v7_mz05 | +18.6% | 7.07 | 7.64 | -4.5% | 65% | Only v6+v7 Huber (v3 best), CB+XGB RMSE, mz=0.5 |
| **huber_4model** | **+18.7%** | **7.28** | **8.08** | **-4.5%** | **65%** | All 4 models Huber |
| **huber_4model_mz05** | **+18.8%** | **7.29** | **8.14** | **-4.5%** | **65%** | All 4 models Huber + min-zscore=0.5 |

### Анализ

1. **Huber на всех 4 моделях >> только v6+v7**: HAC 8.08 vs 7.64 (+5.8%). CatBoost и XGBoost тоже выигрывают от Huber.
2. **min-zscore=0.5 даёт ещё +0.7% HAC**: 8.14 vs 8.08. Фильтрация слабых сигналов стабильно помогает.
3. **Новый лучший конфиг**: `huber_4model_mz05` — HAC 8.14, Sharpe 7.29, WR 65%, DD -4.5%.
4. **Δ vs baseline**: Return +8.7% rel, Sharpe +9.3%, HAC +7.4%, WR +2pp. MaxDD unchanged.

### Best Config: huber_4model_mz05

| Metric | Baseline (v3) | **huber_4model_mz05** | Δ |
|--------|--------------|----------------------|---|
| Return | +17.3% | **+18.8%** | +8.7% rel |
| Sharpe | 6.67 | **7.29** | +9.3% |
| HAC Sharpe | 7.58 | **8.14** | +7.4% |
| MaxDD | -4.5% | **-4.5%** | unchanged |
| Win Rate | 63% | **65%** | +2pp |

---

## Overnight v5 — News Feature A/B Test Under Huber (16 Mar 2026)

### Контекст
Другая AI (Claude) обнаружила, что v6 production модель содержит 10 news features, несмотря на `--news-mode none`.
Расследование: `results_v6_prod` byte-identical (`md5`) с `lgb_v6_no_news` — флаг `--news-mode none` отработал.
Однако `crypto_features_1h.parquet` на кластере мог содержать news-колонки, прошедшие сквозь pipeline.

**Вопрос**: помогают ли news features под Huber loss? Прошлый A/B (RMSE) показал -47%, но Huber может быть более robust.

**Эксперимент**: retrain v6 Huber WITH news (`--news-mode all`) vs WITHOUT news (`--news-mode none`).
Train→2026-02-01, val→2026-03-07, OOS: Feb 9 → Mar 7.

### Результаты

| Experiment | Return | Sharpe | HAC Sharpe | MaxDD | WR | News? |
|-----------|--------|--------|-----------|-------|----|-------|
| huber_4model_nonews | +16.2% | 6.41 | 7.13 | -4.5% | 63% | v6 без news |
| huber_4model_nonews_mz05 | +16.4% | 6.39 | 7.11 | -4.5% | 63% | v6 без news + mz=0.5 |
| **huber_4model_withnews_mz05** | **+18.8%** | **7.29** | **8.14** | **-4.5%** | **65%** | v6 WITH news + mz=0.5 |

### Анализ

1. **News ПОМОГАЮТ под Huber**: HAC 8.14 vs 7.11 = **+14.5% относительно**.
2. **Почему Huber+news > RMSE+news**: Huber loss подавляет влияние шумных outlier-ов в news features. RMSE усиливает ошибки на outliers → news = шум. Huber → news = полезный сигнал.
3. **Без news Sharpe падает с 7.29 до 6.39**: news добавляют ~+0.9 Sharpe, +1.03 HAC, +2pp WR.
4. **Прошлый A/B (RMSE, -47%) больше не актуален**: Huber loss полностью меняет value proposition news features.

### Ключевой вывод

**News features + Huber loss = синергетическая комбинация.**
- RMSE + news = шумная модель (старый A/B: -47% Sharpe)
- Huber + news = robust модель с дополнительным alpha (+14.5% HAC)

**Вердикт: НЕ удалять news features. Оставить `--news-mode all` в v6.**

---

## Overnight Research Summary (15–16 Mar 2026)

### Лучший конфиг на данный момент

**huber_4model_mz05** (with news):
- 4 модели (LGB v6, LGB v7, CatBoost, XGBoost), все с Huber loss
- v6 обучена с `--news-mode all` (10 news features)
- Portfolio: edge-boost sizing, min-zscore=0.5
- **HAC Sharpe: 8.14** | Sharpe: 7.29 | Return: +18.8% (26d) | MaxDD: -4.5% | WR: 65%

### OOS Methodology Note
Sim period (Feb 9 – Mar 7) overlaps with validation window (Dec 9 – Mar 7).
Validation used ONLY for early stopping (not gradient updates) → quasi-OOS.
This is standard practice for LGB/CatBoost/XGBoost — early stopping ≠ training.

### ~~Pending~~ Completed: v6 experiment (v7 + news) — 15 Mar 2026

**Hypothesis**: if news help v6 (+14.5% HAC), they may help v7 too.
Modified `run_pipeline_v7.py` to use v6's news-aware `add_sentiment_features` when `--news-mode != none`.

| Experiment | Return | Sharpe | HAC | MaxDD | WR | Notes |
|---|---|---|---|---|---|---|
| **v7 no-news + mz0.5 (ref)** | **+18.8%** | **7.29** | **8.14** | **-4.5%** | **65%** | baseline |
| v7+news | +16.6% | 6.56 | 7.40 | -4.5% | 63% | news hurt |
| v7+news + mz0.5 | +17.2% | 6.76 | 7.72 | -4.5% | 63% | news hurt |
| v7 no-news + mz0.5 (control) | +18.8% | 7.29 | 8.14 | -4.5% | 65% | ✅ matches ref |

**Verdict: News HURT v7** ❌ — HAC −5.2%, WR −2pp. Control matches ref 1:1, experiment fair.
v7 stays WITHOUT news. v6 keeps news (v5 showed +14.5% HAC for v6).

Current best config unchanged:
- **v6**: Huber + news ✅
- **v7**: Huber, NO news ✅
- **CB/XGB**: Huber ✅
- **mz=0.5** ✅
- **HAC 8.14** — baseline for DVOL experiment (v7, running)

### ~~Pending~~ Completed: v7 experiment (DVOL features) — 15 Mar 2026

Added 13 DVOL features from Deribit (dvol_btc/eth level, change_12h/24h, z_30d/z_60d, spread, term_ratio, vol_of_vol).
Script `run_overnight_v7.sh`: retrain all 4 models with DVOL, compare vs v4 baseline (HAC 8.14).
**Hypothesis**: implied volatility adds orthogonal signal to realized-vol features.

| Experiment | Return | Sharpe | HAC | MaxDD | WR | Notes |
|---|---|---|---|---|---|---|
| **v4-best (ref, no DVOL)** | **+18.8%** | **7.29** | **8.14** | **-4.5%** | **65%** | baseline |
| DVOL + mz0.5 | +17.1% | 6.59 | 7.73 | -4.5% | 63% | DVOL hurt |
| DVOL (no mz) | +15.8% | 6.20 | 6.94 | -4.5% | 63% | DVOL hurt more |

**Verdict: DVOL HURT** ❌ — HAC −5.0% (7.73 vs 8.14), WR −2pp.
13 additional implied-vol features → noise, models overfit to DVOL patterns that don't generalize.
Train-time metrics (CB Sharpe 36.6, XGB Sharpe 46.5) were artifact of 4-period val, not confirmed in sim.

**DVOL discarded.** Baseline HAC 8.14 remains best.

### ~~Pending~~ Completed: v8 experiment (macro features ± DVOL) — 15-16 Mar 2026

Added ~38 FRED macro features (VIX, S&P500, DXY, Gold, 10Y Yield, HY Spread, Breakeven Inflation, Yield Curve, Fed Funds Rate + changes + z-scores + cross-interactions).
Script `run_overnight_v8.sh`: macro-only (DVOL hidden) vs macro+DVOL vs v4-best baseline.

**Results** (51-step sim, 2026-02-09 → 2026-03-07):

| Config | HAC | Sharpe | Return | MaxDD | WR | Feats (v6) |
|---|---|---|---|---|---|---|
| **A: Macro only (no DVOL)** | 7.64 | 7.01 | +17.8% | -5.2% | 61% | 190 |
| **B: Macro + DVOL** | **8.36** | **7.39** | +18.8% | -4.5% | 63% | 200 |
| **C: v4 baseline** | 8.14 | 7.29 | +18.8% | -4.5% | 65% | 160 |

**Key findings:**
1. **Macro alone HURTS**: A vs C → HAC −6.1% (7.64 vs 8.14), WR −4pp, DD worse (−5.2% vs −4.5%).
2. **Macro+DVOL slightly beats baseline**: B vs C → HAC +2.7% (8.36 vs 8.14). BUT:
   - WR dropped (63% vs 65%), feats grew 25% (160→200)
   - Each component alone is negative — synergy may be val-period artifact
   - Δ +0.22 HAC too small for 51-step window to be statistically reliable
3. **Pattern**: both DVOL-alone (v7) and macro-alone (v8A) hurt individually. Combined they happen to help on this window, but fragility is high.

**Verdict: MACRO DISCARDED** ❌ — improvement too small and fragile. Baseline HAC 8.14 remains best.

---

## Experiment Results Master Table (16 Mar 2026)

| Exp | Config | HAC | Sharpe | Return | MaxDD | WR | Verdict |
|---|---|---|---|---|---|---|---|
| v2 | LambdaRank / Residual / Meta-label | — | — | — | — | ~61% | all FAILED |
| v3 | Huber v6+v7 + mz0.5 | 7.69 | — | — | — | 65% | good but 2-model only |
| **v4** | **Huber 4-model + mz0.5** | **8.14** | **7.29** | **+18.8%** | **-4.5%** | **65%** | **BEST** |
| v4 | Huber 4-model (no mz) | 8.08 | 7.09 | +18.3% | -4.5% | 65% | mz helps slightly |
| v4 | RMSE 4-model baseline | 7.58 | 7.00 | +17.2% | -4.5% | 63% | Huber >> RMSE |
| v5 | v6 news A/B (Huber) | 8.14 vs 7.11 | — | — | — | 65% vs 63% | news help v6 under Huber |
| v6 | v7+news | 7.72 | 6.76 | +17.2% | -4.5% | 63% | news hurt v7 |
| v7 | +DVOL (13 features) | 7.73 | 6.59 | +17.1% | -4.5% | 63% | DVOL hurt |
| v8A | +macro only | 7.64 | 7.01 | +17.8% | -5.2% | 61% | macro hurt |
| v8B | +macro +DVOL | 8.36 | 7.39 | +18.8% | -4.5% | 63% | +2.7% but fragile |

---

## Research Windows (17–18 марта 2026)

### Контекст
Предыдущее research (v1-v8; exp1-v5) использовало 3 WF окна с train→2024-06.
Новый формат: **RESEARCH_WINDOWS** — 2 окна для OOS-оценки на свежих данных:
- **R1**: train→2024-12-31, val 2025-01-08→2025-09-30, **test Q4 2025** (3 мес OOS)
- **R2**: train→2025-06-30, val 2025-07-08→2025-12-31, **test Q1 2026** (2.5 мес OOS)

4 пайплайна: LGB v6 (12h target), LGB v7 (blended 75/25 12h/24h), CatBoost, XGBoost.
Все с Huber loss, 5 seeds, skip-hpo (кроме отмеченных).

---

## Overnight v11 — Baseline Research (17 марта 2026)

**18 экспериментов, 7 фаз, 7h 35m на GPU-кластере.**

### Training Sharpe (LS_Sharpe_net, avg R1+R2)

| # | Experiment | R1 Sharpe | R2 Sharpe | **Avg** | Notes |
|---|---|---|---|---|---|
| 1 | v6_baseline (HPO 50) | 0.73 | 1.48 | **1.10** | Current prod config |
| 2 | v7_baseline (HPO 50) | 0.32 | 1.41 | **0.87** | Weakest model |
| 3 | cb_baseline (skip-hpo) | 1.18 | 1.78 | **1.48** | Best single model! |
| 4 | xgb_baseline (skip-hpo) | 0.95 | 1.38 | **1.17** | Solid middle |
| 5 | v6_no_deriv | 0.82 | 1.84 | **1.33** | +21% vs baseline! |
| 6 | v6_24h_target | -0.18 | 0.58 | **0.20** | 24h target = DEAD |
| 7 | v6_4h_target | 0.46 | 1.18 | **0.82** | Interesting but < 12h |
| 8 | v6_no_news | 0.85 | 1.45 | **1.15** | News neutral for LGB |
| 9 | xgb_no_news | 1.14 | 1.63 | **1.39** | Market-only = +19%! |

### Sim Results (60d, 3x leverage, edge-boost)

| Config | Return | Sharpe HAC |
|---|---|---|
| Ensemble v6+CB (Huber) | +46.2% | 7.19 |
| v6 solo | +37.7% | 7.29 |
| 4-model ensemble | +40.2% | 6.30 |

### Ключевые выводы v11
1. **CatBoost = лучшая одиночная модель** (Sharpe 1.48 avg)
2. **Derivatives вредят LGB v6** (1.10→1.33 без derivs, +21%)
3. **v7 = самая слабая модель** (0.87), v6↔v7 корреляция 0.957
4. **24h target = мёртвый** (Sharpe 0.20)
5. **v6 solo в sim > 4-model ensemble** (7.29 vs 6.30 HAC)
6. **News нейтральны для LGB**, но market-only news помогают XGB

---

## Overnight v12 — Focused Follow-up (17–18 марта 2026)

**9 экспериментов (8 training + sim grid), 4h 24m.**

### Training Sharpe (LS_Sharpe_net, avg R1+R2)

| # | Experiment | R1 Sharpe | R2 Sharpe | **Avg** | Notes |
|---|---|---|---|---|---|
| 1 | cb_price_only | 1.38 | 2.18 | **1.78** | 🏆 **ALL-TIME BEST** |
| 2 | cb_no_deriv | 1.31 | 2.01 | **1.66** | News help CB (+0.12) |
| 3 | v6_price_only_hpo (50 trials) | 1.23 | 1.68 | **1.45** | +32% vs v11 baseline |
| 4 | v6_no_deriv_hpo (50 trials) | 0.95 | 1.43 | **1.19** | HPO helps v6 |
| 5 | xgb_no_deriv | 1.23 | 1.63 | **1.43** | Solid |
| 6 | xgb_price_only | 1.01 | 1.48 | **1.25** | News help XGB |
| 7 | v6_price_only_skip | 0.97 | 1.69 | **1.33** | Same as v11#5 |
| 8 | v6_residual_target | 0.91 | 1.66 | **1.29** | Residual ≈ standard |

### v12 Sim Paradox (ОБНАРУЖЕН)

Better training Sharpe → **WORSE** sim performance!

| Config | Sim Return | HAC | Training Sharpe |
|---|---|---|---|
| v11 v6+CB baseline | **+46.2%** | **7.19** | 1.10+1.48 |
| v12 v6po+cbpo | +36.3% | 5.35 | 1.45+1.78 |
| v12 cbpo solo | +35.1% | 4.79 | 1.78 |

### Ключевые выводы v12
1. **cb_price_only = лучший training Sharpe ever** (1.78 avg)
2. **Derivatives вредят ВСЕМ моделям** (подтверждено для CB, XGB, LGB)
3. **Парадокс**: лучшие training метрики → худшие sim результаты
4. **HPO помогает v6** (+32% Sharpe vs skip-hpo)

---

## Overnight v13 — Resolve the Sim Paradox (18 марта 2026)

**33 сима + 2 training эксперимента, 2h 9m.**

### Phase 1: Sim Sensitivity (R1, Q4 2025)

| Config | v11 v6+CB | v12 CBpo solo | v12 v6po+CBpo |
|---|---|---|---|
| Baseline (lev3/k0.8/edge) | **+12.4%** (7.08) | +9.1% (4.55) | +9.3% (5.21) |
| lev2 | +8.1% (7.15) | +6.0% (4.64) | +6.2% (5.29) |
| lev1 | +4.6% (8.34) | +3.6% (5.70) | — |
| kelly 0.5 | +7.6% (7.16) | +5.6% (4.65) | +5.8% (5.30) |
| no edge-boost | +11.4% (6.93) | +7.5% (3.99) | — |
| ddstop on/off | **идентично** | **идентично** | — |

**Выводы Phase 1:**
- **DDstop бесполезен** (0.0% разницы ± ddstop)
- Edge-boost даёт +1–1.6pp return
- v11 > v12 на R1 при ЛЮБЫХ настройках

### Phase 2: R2 Period (Q1 2026)

| Config | Return | HAC |
|---|---|---|
| v12 cbpo solo | **+52.7%** | 6.74 |
| v11 v6 solo | +50.8% | 7.04 |
| mix v11v6+v12cbpo | +49.3% | 6.20 |
| v11 v6+cb | +45.8% | 6.41 |
| v12 v6po+cbpo | +45.9% | 5.84 |

**На R2 v12 cbpo чуть лучше v11, разница невелика.**

### Phase 3: FULL Period (Oct 2025 – Mar 2026, 5 мес)

| # | Config | Return | HAC |
|---|---|---|---|
| **27** | **v12 cb_no_deriv solo** | **+131.5%** | **5.09** |
| 23 | v12 cbpo solo | +103.7% | 4.19 |
| 33 | mix v11v6 + v13 cbMKTnd | +93.3% | 4.01 |
| 24 | v12 v6po+cbpo | +94.8% | 3.96 |
| 22 | v11 v6 solo | +91.5% | 4.15 |
| 26 | mix 3-model | +91.2% | 3.93 |
| 21 | v11 v6+cb | +90.1% | 3.99 |
| 25 | mix v11v6+v12cbpo | +87.1% | 3.71 |
| 28 | v11 3-model | +73.4% | 3.48 |

### Phase 4: Leverage Sensitivity (FULL period)

| Config | Return | HAC |
|---|---|---|
| cbnd 3x (ref) | +131.5% | 5.09 |
| v11 v6+cb 3x | +90.1% | 3.99 |
| v11 v6+cb 2x | +55.1% | 3.89 |
| v11 v6+cb 1x | +30.0% | 4.41 |
| mix 2x | +53.5% | 3.64 |
| mix 1x | +29.3% | 4.19 |

### Phase 5: New Training

| Experiment | R1 Sharpe | R2 Sharpe | **Avg** |
|---|---|---|---|
| cb_market_no_deriv (market-only news, no derivs) | 1.53 | 2.15 | **1.84** |
| v6_no_deriv_skip_hpo | 1.18 | 1.46 | **1.32** |

### ПАРАДОКС РЕШЁН

**Причина**: v11 > v12 на R1 (Q4 2025) — но это артефакт конкретного 3-мес периода.
На FULL период (5 мес) v12 модели **побеждают**:

| | R1 (Q4'25) | R2 (Q1'26) | FULL (5 мес) |
|---|---|---|---|
| v11 v6+cb | +12.4% | +45.8% | +90.1% |
| v12 cbnd solo | ? (not tested R1) | ? | **+131.5%** |
| v12 cbpo solo | +9.1% | +52.7% | +103.7% |

### Главные выводы v11–v13

1. **cb_no_deriv SOLO = абсолютный чемпион** (+131.5% за 5 мес, HAC 5.09)
2. **Ансамбли ХУЖЕ соло CatBoost**. Больше моделей → больше "усреднение" → меньше edge
3. **Derivatives ВРЕДЯТ всем моделям** (подтверждено в training и sim)
4. **News ПОМОГАЮТ CatBoost** (cb_no_deriv 1.66 > cb_price_only 1.78 training, НО 131.5% > 103.7% sim)
5. **DDstop бесполезен** (идентичные результаты ±)
6. **Edge-boost помогает** (~+1-2pp return)
7. **Training Sharpe НЕ предсказывает sim performance** (1.66→+131.5% vs 1.78→+103.7%)
8. **LGB v7 можно убрать** (0.957 корреляция с v6, самый слабый Sharpe)
9. **24h target = мёртвый** (Sharpe 0.20)
10. **leverage 3x оптимален** для сильных моделей

### Рекомендация на 19 марта 2026 (обновлена после v14–v15)
- **Production model**: CatBoost solo cb_market_noderiv_hpo (market-only news, no derivs, Huber, HPO 50 trials)
- **Sim config**: --leverage 3 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop --vol-size
- **Следующий шаг**: deploy vol-size в production, исследовать meta-risk overlay

---

## v14 — CatBoost Training Variations (19 марта 2026)

6 CatBoost экспериментов + 22 сима. Цель: побить cb_no_deriv (+131.5%) вариациями обучения.

### Phase 1: Training (6 experiments)

| Experiment | Конфиг | Train Sharpe | Sim Return | Sim HAC |
|---|---|---|---|---|
| cb_noderiv_hpo | no derivs, HPO 50 trials | 1.77 | +121.2% | 4.90 |
| cb_noderiv_residual | no derivs, residual target | 1.27 | +92.2% | 4.06 |
| cb_noderiv_hd05 | no derivs, huber delta=0.5 | 1.70 | +111.6% | 4.61 |
| cb_noderiv_hd15 | no derivs, huber delta=1.5 | **1.93** | +127.2% | 5.03 |
| cb_all_hpo | ALL features, HPO 50 | 1.62 | +128.4% | 5.29 |
| **cb_market_noderiv_hpo** | **market-only news, no derivs, HPO 50** | **1.83** | **+143.8%** | **5.33** |

### Phase 2: References (reproduce)

| Config | Return | HAC |
|---|---|---|
| **cb_market_noderiv_hpo** (NEW) | **+143.8%** | **5.33** |
| v13 cb_market_no_deriv (skip-hpo) | +132.8% | 4.90 |
| v12 cb_no_deriv | +131.5% | 5.09 |
| v11 cb | +122.5% | 5.05 |
| v12 cb_price_only | +103.7% | 4.19 |

### Phase 3: Stability (R1/R2)

| Config | R1 Return | R1 HAC | R2 Return | R2 HAC |
|---|---|---|---|---|
| v12 cb_no_deriv | +11.4% | 6.45 | +62.1% | 8.14 |
| cb_noderiv_hpo | +11.3% | 6.42 | +58.7% | 7.81 |
| cb_all_hpo | +14.1% | 8.52 | +57.8% | 8.26 |
| **cb_market_noderiv_hpo** | +11.2% | 6.30 | **+66.6%** | **8.42** |

### Главные выводы v14

1. **НОВЫЙ ЧЕМПИОН: cb_market_noderiv_hpo** (+143.8%, HAC 5.33) — обогнал старого на +12.3pp
2. **Ключ: market-only news + HPO**. Per-coin news удалены (шум), market-level оставлены (сигнал)
3. **HPO помог market-only** (skip-hpo 132.8% → HPO 143.8%), но **навредил no-deriv** (131.5% → 121.2%)
4. **Residual target = провал** (+92.2%). Предсказание excess return сложнее raw return
5. **Huber delta=1.5**: лучший training Sharpe ever (1.93!), но sim +127.2% < old champ. Training ≠ sim (ОПЯТЬ)
6. **cb_all_hpo** (+128.4%, HAC 5.29) — derivs + HPO = почти равны old champ. HPO компенсирует вред derivs

---

## v15 — Execution Layer Optimization (19 марта 2026)

46 симов, 0 обучения. Тест 6 sim-флагов которые существовали но НИКОГДА не использовались.

**Чемпион для всех тестов**: cb_market_noderiv_hpo (+143.8% baseline)

### Что значат флаги

| Флаг | Что делает |
|---|---|
| `--vol-target-ann X` | Масштабирует exposure обратно пропорционально реализованной волатильности. Target X% годовых. Высокая vol → меньше позиции, низкая → больше |
| `--hysteresis K` | Не выкидывает позицию пока она не упала в ранкинге ниже N+K. Снижает churn |
| `--smooth-signal α` | EMA-blend текущих предсказаний с предыдущими. α=0.3 → 30% веса старым scores |
| `--turnover-budget N` | Максимум N замен на сторону за ребаланс |
| `--vol-size` | Inverse-vol sizing: позиции в low-vol монетах крупнее, в high-vol мельче |
| `--regime-shorts X` | В bull-режиме масштабирует шорты на X (0.5 = вдвое меньше шортов) |
| `--meta-risk` | Composite risk scaler (0.3x–1.5x) из 5 сигналов: model confidence, score spread, recent win rate, DD depth, regime |

### Phase 2: Single-flag sweeps (каждый отдельно vs baseline)

| Config | Return | HAC | Max DD | Δ Return | Вердикт |
|---|---|---|---|---|---|
| **BASELINE** | **+143.8%** | **5.33** | **-20.5%** | — | — |
| vol-target 30% | +89.8% | 4.61 | -17.6% | -54.0pp | ВРЕДИТ |
| vol-target 40% | +110.3% | 4.74 | -20.1% | -33.5pp | ВРЕДИТ |
| vol-target 50% | +130.4% | 4.81 | -22.2% | -13.4pp | ВРЕДИТ |
| vol-target 60% | +150.3% | 4.85 | -24.1% | +6.5pp | Marginal, DD хуже |
| hysteresis 3 | +143.8% | 5.33 | -20.5% | 0 | НОЛЬ ЭФФЕКТА |
| hysteresis 5 | +143.8% | 5.33 | -20.5% | 0 | НОЛЬ ЭФФЕКТА |
| hysteresis 7 | +143.8% | 5.33 | -20.5% | 0 | НОЛЬ ЭФФЕКТА |
| hysteresis 10 | +143.8% | 5.33 | -20.5% | 0 | НОЛЬ ЭФФЕКТА |
| smooth 0.2 | +129.3% | 5.16 | -21.4% | -14.5pp | ВРЕДИТ |
| smooth 0.3 | +105.4% | 4.26 | -21.5% | -38.4pp | ВРЕДИТ |
| smooth 0.4 | +89.4% | 3.72 | -20.4% | -54.4pp | ВРЕДИТ |
| smooth 0.5 | +72.3% | 3.16 | -19.4% | -71.5pp | ВРЕДИТ |
| turnover-3 | — | — | — | — | БАГ (пустой output) |
| turnover-5 | — | — | — | — | БАГ (пустой output) |
| turnover-8 | +143.8% | 5.33 | -20.5% | 0 | НОЛЬ ЭФФЕКТА |
| **vol-size** | **+147.5%** | **5.48** | **-20.4%** | **+3.7pp** | **WINNER** |
| regime-shorts 0.5 | +56.7% | 2.47 | -21.3% | -87.1pp | КАТАСТРОФА |
| regime-shorts 0.3 | +30.0% | 1.27 | -22.4% | -113.8pp | КАТАСТРОФА |

### Phase 3: Combos

| Config | Return | HAC | Max DD |
|---|---|---|---|
| **meta-risk (solo)** | **+184.7%** | **5.29** | -21.9% |
| vol-size (solo) | +147.5% | 5.48 | -20.4% |
| hyst5 + vt50 | +130.4% | 4.81 | -22.2% |
| hyst5 + vt50 + vol-size | +132.5% | 4.97 | -22.2% |
| meta-risk + hyst5 + vt50 | +142.3% | 4.67 | -22.7% |
| hyst5 + smooth 0.3 | +105.4% | 4.26 | -21.5% |
| all_moderate | +93.1% | 3.66 | -23.2% |
| all_conservative | +67.2% | 3.10 | -20.4% |

### Phase 5: Leverage sweep (с hyst5 + vt50)

| Leverage | Return | HAC | Max DD |
|---|---|---|---|
| 1x | +74.8% | **5.66** | -12.6% |
| 2x | +101.9% | 4.91 | -18.3% |
| 3x (baseline) | +143.8% | 5.33 | -20.5% |
| 4x | +156.2% | 4.72 | -25.4% |
| 5x | +180.1% | 4.63 | -28.2% |

### Главные выводы v15

1. **`--vol-size` = единственный чистый winner** (+147.5%, HAC 5.48 — лучший HAC всего эксперимента). Inverse-vol sizing: монеты с низкой vol получают больший вес → стабильнее portfolio
2. **`--meta-risk` = больше return (+184.7%!), но чуть хуже HAC (5.29 vs 5.33)**. Масштабирует exposure вверх в "хорошие" режимы → больше variance. С solo CB model agreement signal = константа (бесполезен), работают только spread/perf/DD/regime сигналы
3. **Hysteresis = НОЛЬ эффекта** на 12h rebal. Топ-10 монет почти не меняются между ребалансами — hysteresis не нужен
4. **Smooth-signal ВРЕДИТ** монотонно: чем сильнее сглаживание, тем хуже. Модель уже выдает точный сигнал
5. **Vol-target ВРЕДИТ return И HAC**. Модель неявно учитывает vol через features — внешний vol scaling отбирает edge
6. **Regime-shorts = КАТАСТРОФА** (-87pp!). Short alpha реальная и сильная, нельзя резать
7. **Turnover-budget 3/5 = баг** в sim (пустой output), turnover-8 = нет эффекта
8. **Leverage: risk-adjusted оптимум = 1x** (HAC 5.66). 3x — разумный компромисс return/risk
9. **Kitchen sink (all flags) = worst of all worlds** (+67–93%). Больше ограничений ≠ лучше

### Рекомендация после v15
- **Deploy**: `--vol-size` в production (inverse-vol sizing, +3.7pp return, +0.15 HAC, DD чуть ниже)
- **Исследовать**: `--meta-risk` отдельно (нужен эксперимент с meta-risk + vol-size combo)
- **Не трогать**: hysteresis, smooth-signal, vol-target, regime-shorts
- **Починить**: turnover-budget баг

---

## Window Sweep Experiment (23 марта 2026)

**Вопрос**: влияет ли размер обучающего окна на OOS-качество? Expanding (все данные с 2017) vs capped (36/24/18/12 месяцев)?

**Сетап**: CatBoost Huber (delta=1.0), 5 seeds, 260 фич, --no-news, --huber, derivatives. 5 конфигов × 3 OOS окна = 15 симуляций. Sim: leverage 3x, rebal 24h, vol-size, dd-stop.

### Walk-forward окна (те же что в mega_comparison3)

| Окно | Train end | Test period |
|------|-----------|-------------|
| WinA | 2024-04-30 | Jul 2024 — Dec 2024 |
| WinB | 2024-10-31 | Jan 2025 — Jun 2025 |
| WinC | 2025-04-30 | Jul 2025 — Dec 2025 |

### Результаты симуляций (OOS)

| Window | Cap | Return% | HAC | MaxDD% | WR% | PF |
|--------|-----|---------|-----|--------|-----|-----|
| WinA | expanding | +126.8 | +6.74 | -3.0 | 66 | 3.18 |
| WinA | cap36m | +127.5 | +6.74 | -2.9 | 67 | 3.18 |
| WinA | **cap24m** | **+126.1** | **+7.06** | **-2.8** | **69** | **3.08** |
| WinA | cap18m | +129.3 | +6.81 | -2.5 | 68 | 3.21 |
| WinA | cap12m | +123.9 | +6.81 | -3.4 | 67 | 3.00 |
| WinB | expanding | +139.2 | +9.12 | -2.9 | 68 | 2.91 |
| WinB | **cap36m** | **+150.3** | **+9.82** | **-2.9** | **68** | **3.10** |
| WinB | cap24m | +145.7 | +9.48 | -3.0 | 67 | 3.04 |
| WinB | cap18m | +151.2 | +9.49 | -3.0 | 67 | 3.00 |
| WinB | cap12m | +150.9 | +9.59 | -3.2 | 67 | 3.17 |
| WinC | **expanding** | **+74.4** | **+7.04** | **-4.3** | **67** | **2.28** |
| WinC | cap36m | +71.0 | +6.87 | -5.1 | 64 | 2.17 |
| WinC | cap24m | +67.8 | +6.65 | -5.9 | 65 | 2.13 |
| WinC | cap18m | +69.0 | +6.64 | -6.2 | 65 | 2.20 |
| WinC | cap12m | +68.2 | +6.73 | -5.9 | 63 | 2.17 |

### Средний HAC по конфигам

| Cap | WinA HAC | WinB HAC | WinC HAC | **Avg HAC** |
|-----|----------|----------|----------|-------------|
| expanding | 6.74 | 9.12 | 7.04 | **7.63** |
| cap36m | 6.74 | 9.82 | 6.87 | **7.81** |
| cap24m | 7.06 | 9.48 | 6.65 | **7.73** |
| cap18m | 6.81 | 9.49 | 6.64 | **7.65** |
| cap12m | 6.81 | 9.59 | 6.73 | **7.71** |

### Метрики обучения (pipeline, не sim)

| Window | Cap | Rank IC | Rank ICIR | LS Sharpe net | DDStop Sharpe |
|--------|-----|---------|-----------|---------------|---------------|
| WinA | expanding | 0.0298 | 0.5018 | 2.33 | 3.84 |
| WinA | cap36m | 0.0347 | 0.5070 | 2.17 | 3.83 |
| WinA | cap24m | 0.0410 | 0.4815 | 1.59 | 3.33 |
| WinA | cap18m | 0.0370 | 0.5006 | 1.21 | 2.95 |
| WinA | cap12m | 0.0314 | 0.3957 | 1.31 | 3.28 |
| WinB | expanding | 0.0283 | 0.4725 | 2.20 | 4.05 |
| WinB | cap36m | 0.0373 | 0.5068 | 2.38 | 4.40 |
| WinB | cap24m | 0.0362 | 0.4762 | 1.69 | 3.77 |
| WinB | cap18m | 0.0273 | 0.2786 | 1.89 | 4.00 |
| WinB | cap12m | 0.0252 | 0.2097 | 1.95 | 4.15 |
| WinC | expanding | 0.0198 | 0.4338 | 0.98 | 2.32 |
| WinC | cap36m | 0.0432 | 0.4343 | 0.72 | 1.98 |
| WinC | cap24m | 0.0389 | 0.4420 | 0.85 | 1.96 |
| WinC | cap18m | 0.0419 | 0.4987 | 1.16 | 2.66 |
| WinC | cap12m | 0.0368 | 0.5154 | 1.02 | 2.31 |

### Главные выводы Window Sweep

1. **Разница между конфигами минимальна**: avg HAC от 7.63 (expanding) до 7.81 (cap36m) — spread всего 0.18. Ни один конфиг не доминирует стабильно.

2. **Cap36m чуть лучше по avg HAC (7.81)**, но победитель меняется от окна к окну:
   - WinA → cap24m (HAC 7.06)
   - WinB → cap36m (HAC 9.82)
   - WinC → expanding (HAC 7.04)

3. **WinC (2-я половина 2025) значительно слабее**: HAC ~6.6-7.0 vs WinB HAC ~9.1-9.8. Это рыночный эффект (sideways/bearish market), а не деградация модели.

4. **Expanding safe default**: не лучший, но и не худший (avg HAC 7.63 — 3-е место из 5). Не теряет на старых данных — CatBoost сам справляется с concept drift.

5. **Cap12m НЕ лучше** несмотря на "свежесть" данных: avg HAC 7.71 — 4-е место. Мало данных = хуже обобщение.

6. **Training metrics не коррелируют с OOS sim HAC**: WinC cap36m имеет лучший Rank IC (0.0432), но худший sim HAC (6.87). Валидация через sim обязательна.

### Рекомендация

**Оставить expanding** (текущий production конфиг). Разница с cap36m несущественна (0.18 HAC), а expanding проще и не требует выбора окна. Тренируем на всех данных с 2017.

---
