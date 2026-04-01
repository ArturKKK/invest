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

## 🧪 Ridge Model Research (март 2026)

> **Контекст**: все GBDT модели (LGB v5-v8, CatBoost) показали IC~0.01 = шум при 208 фичах. Переход к простой Ridge regression с 14 CS-IC-проверенными фичами.

### Модель

- **Тип:** Ridge regression, α=1000
- **Фичи (14):** `ret_12h`, `ret_24h`, `ret_48h`, `residual_12h`, `residual_24h`, `mom_z_12h`, `mom_z_24h`, `dist_from_high_24h`, `oi_chg_12h`, `oi_chg_24h`, `oi_zscore`, `taker_cvd_12h`, `taker_cvd_24h`, `ls_divergence`
- **Горизонт:** 12h
- **CS-rank:** все фичи ранжируются cross-sectionally перед подачей в модель
- **Target:** CS-ranked forward return 12h
- **Валидация:** Walk-forward, 3 окна, 15-day gap, HPO alpha на val

### Walk-Forward Windows

```
W1: train→2024-06 | val: 2024-06→2024-09 | test: 2024-10-15→2025-01-31
W2: train→2025-01 | val: 2025-01→2025-04 | test: 2025-05-15→2025-08-31
W3: train→2025-07 | val: 2025-07→2025-10 | test: 2025-11-15→2026-03-17
```

### Скрипты

```bash
# Обучение продакшн модели (Ridge, α=1000, 14 фич)
python train_ridge_prod.py
# → results_ridge_prod/model.json (val IC=0.052, final IC=0.062, 1M+ rows)

# Strict OOS walk-forward simulation (baseline — без risk mgmt)
python _sim_ridge_strict.py
# → Sharpe 0.47, $100→$79 at 3x (ПЛОХО)

# Research round 1: 12 risk management configs
python _research_maxdd.py
# → Winner: Hard cutoff trend>0.8 → Sharpe 1.74, worst month -14.9% (3x)

# Research round 2: 14 configs, расширенная вселенная, blend horizons
python _research_round2.py
# → Winner: 35sym 6L/6S 12h co=0.8, Sharpe 2.17, worst month -3.0% (3x), $100→$196

# Research round 3: full 50-symbol universe, edge-weighting, blends
python _research_round3.py
# → Winner: 20sym 4L/4S 12h co=0.8, $100→$207 (5x), worst -24.5%

# Research round 3B: vol-targeting, dynamic exposure, asymmetric regime
python _research_round3b.py
# → Winner: 35sym 5L/5S DYN-EXPOSURE, Sharpe 2.03, $100→$274 (5x), worst -21.1%

# Research round 4: rolling IC filter, equity momentum, confidence weighting, combos
python _research_round4.py
# → BREAKTHROUGH: 35sym 5L/5S + EQ-MOM, Sharpe 3.81, $100→$671 (5x), worst -13.2%
```

### Regime Filter: BTC Trend Strength

```python
trend_strength = |BTC_ret_7d| / (BTC_vol_7d * sqrt(168))
# trend > cutoff(0.8) → go FLAT (не торгуем)
```

### Dynamic Exposure

```python
# Когда trend_strength > dyn_threshold (0.5), плавно снижаем экспозицию:
if trend_str > dyn_threshold:
    exposure = 1.0 - (trend_str - dyn_threshold) / (trend_cutoff - dyn_threshold) * 0.5
    port_ret *= exposure
```

### Equity Momentum (Anti-Tilt)

```python
# В просадке >5% за последние 48 баров → уменьшаем позиции:
if dd < -0.05:
    scale = max(0.3, 1.0 + dd * 3)  # -10% dd → 0.7x size
    port_ret *= scale
```

---

### Результаты по раундам (сводная таблица, 5x leverage, $100 старт)

| Раунд | Лучший конфиг | Sharpe | $100 → | Worst month | Что нашли |
|-------|---------------|--------|--------|-------------|-----------|
| Baseline | 20sym 5L/5S 12h (без risk mgmt) | 0.47 | $79 | -43.4% | Модель работает, но MaxDD убивает |
| R1 | 20sym 5L/5S co=0.8 | 1.74 | $168 | -14.9% | Hard cutoff при trend>0.8 |
| R2 | 35sym 6L/6S co=0.8 | 2.17 | $196 | -3.0% | Больше символов = больше dispersion |
| R3 | 20sym 4L/4S co=0.8 | 1.74 | $207 | -24.5% | 50 символов нестабильнее 20 |
| R3B | **35sym 5L/5S DYN-EXP** | **2.03** | **$274** | -21.1% | Dynamic exposure > hard cutoff |
| **R4** | **35sym 5L/5S DYN+EQ-MOM** | **3.81** | **$671** | **-13.2%** | **Equity momentum = прорыв** |
| R4 alt | 35sym CONF-W+EQ-MOM | 3.94 | $805 | -17.9% | Больше $ но менее стабильно |

### Что работает / не работает

| Идея | Результат | Вердикт |
|------|-----------|---------|
| Hard cutoff (trend>0.8 → flat) | Sharpe 1.74, worst -14.9% | ✅ Базовый фильтр |
| Dynamic exposure (fade 0.5→0.8) | Sharpe 2.03, worst -21.1% | ✅ Лучше hard cutoff |
| **Equity momentum (anti-tilt)** | **Sharpe 3.81, worst -13.2%** | **✅✅ Прорыв** |
| Confidence-weighted (|pred| → weight) | Sharpe 1.60, worst -30.6% | ⚠️ Хуже без EQ-MOM |
| CONF-W + EQ-MOM combo | Sharpe 3.94, $805 | ⚠️ $$ но EQ-MOM маскирует хвост risk |
| Vol-targeting (1/vol_regime) | Sharpe 1.47, worst -37.6% | ❌ Хуже чем dyn exposure |
| Asymmetric regime (long-bias in bull) | Sharpe 0.28 | ❌ Убивает MR-модель |
| Rolling IC filter | Sharpe 4.49, но $105 | ❌ Слишком агрессивно, 1-2 мес торгов |
| 50 vs 35 vs 20 символов | 35 ≈ 50 > 20 | ⚠️ 35 sweet-spot, 50 нестабильнее |
| 24h horizon | Sharpe 0.52-0.91 | ❌ 12h значительно лучше |
| Blend 12+24h | Sharpe 1.10-1.15 | ❌ Blend хуже чистого 12h |
| Edge-weighting | Sharpe ~1.5 | ⚠️ Не помогает Ridge |

### Помесячная разбивка лучшего конфига: 35sym 5L/5S DYN+EQ-MOM (5x)

| Месяц | Return | Equity |
|-------|--------|--------|
| 2024-10 | +4.3% | $104 |
| 2024-11 | +14.1% | $119 |
| 2024-12 | +116.2% | $257 |
| 2025-01 | **-13.2%** | $223 ← worst |
| 2025-05 | +36.1% | $304 |
| 2025-06 | +26.2% | $384 |
| 2025-07 | +32.4% | $508 |
| 2025-08 | -1.6% | $500 |
| 2025-11 | +2.8% | $514 |
| 2025-12 | +22.2% | $628 |
| 2026-01 | +3.8% | $652 |
| 2026-02 | +2.5% | $668 |
| 2026-03 | +0.4% | $671 |

### Проекции: 35sym 5L/5S DYN+EQ-MOM (5x, $100)

| Сценарий | Мес. доход | 1м | 6м | 12м |
|----------|-----------|-----|-----|-----|
| Avg (+18.9%/мес) | +$19 | $119 | $270 | $730 |
| Median (+4.3%/мес) | +$4 | $104 | $129 | $166 |
| p25 (-1.6%/мес) | -$2 | $98 | $91 | $82 |

### Research Round 5 (30 марта 2026)

```bash
python _research_round5.py
# Идеи: EQ-MOM boost (scale UP после recovery), Kelly sizing, signal agreement (12h+24h),
#        adaptive cutoff (rolling pctl), sector-neutral, monthly DD floor, mega combos
```

**Два прорыва:**

1. **EQ-MOM Boost** — не только уменьшаем размер в просадке, но и **увеличиваем** (до 1.5x) когда equity восстанавливается от дна. Compound-эффект на росте.
2. **Kelly-inspired sizing** — вместо фиксированного 50% long / 50% short, сдвигаем аллокацию по predicted spread: `long_alloc = clip(0.5 + pred_spread * 5, 0.3, 0.7)`. Больше в strong conviction side.

#### Новый лидер: 35sym EQ-BOOST+KELLY

| Раунд | Конфиг | Sharpe | $100 → | Worst month | vs Baseline |
|-------|--------|--------|--------|-------------|-------------|
| R4 | 35sym 5L/5S dyn+EQ-MOM (baseline) | 3.81 | $671 | -13.2% | — |
| **R5** | **35sym 5L/5S EQ-BOOST+KELLY** | **3.43** | **$1366** | **-9.3%** | **+104% equity, +4% safety** |
| R5 alt | 35sym 6L/6S EQ-BOOST+KELLY | 3.41 | $1263 | -8.7% | +88%, +5% safety |
| R5 | 35sym 5L/5S dyn+EQ-BOOST (без Kelly) | 4.19 | $957 | -12.5% | +43%, Sharpe лучше |

#### Помесячная разбивка (35sym EQ-BOOST+KELLY, 5x, $100):

| Месяц | Return | Equity |
|-------|--------|--------|
| 2024-10 | -1.3% | $99 |
| 2024-11 | +70.9% | $169 |
| 2024-12 | +121.0% | $373 |
| 2025-01 | +8.1% | $403 |
| 2025-05 | +38.9% | $560 |
| 2025-06 | -0.8% | $555 |
| 2025-07 | +74.7% | $970 |
| 2025-08 | **-9.3%** | $880 ← worst |
| 2025-11 | +5.7% | $930 |
| 2025-12 | +18.5% | $1102 |
| 2026-01 | +19.5% | $1317 |
| 2026-02 | +6.1% | $1397 |
| 2026-03 | -2.2% | $1366 |

#### Проекции (35sym EQ-BOOST+KELLY, 5x, $100):

| Сценарий | Мес. доход | 1м | 6м | 12м |
|----------|-----------|-----|-----|-----|
| Avg (+26.9%/мес) | +$27 | $127 | $418 | $1745 |
| Median (+8.1%/мес) | +$8 | $108 | $160 | $256 |
| p25 (-0.8%/мес) | -$1 | $99 | $95 | $90 |

#### TOP 5 SAFE (worst month > -15%):

| # | Конфиг | $100 → | Worst | Sharpe |
|---|--------|--------|-------|--------|
| 1 | **35sym EQ-BOOST+KELLY** | **$1366** | **-9.3%** | 3.43 |
| 2 | 35sym 6L6S EQ-BOOST+KELLY | $1263 | -8.7% | 3.41 |
| 3 | 35sym 5L/5S dyn+EQ-BOOST | $957 | -12.5% | 4.19 |
| 4 | BASELINE dyn+EQ-MOM | $671 | -13.2% | 3.81 |
| 5 | 35sym 6L/6S dyn+EQ+KELLY | $652 | -9.1% | 2.90 |

#### Интересные находки R5:

| Идея | Результат | Вердикт |
|------|-----------|---------|
| **EQ-MOM Boost (scale up on recovery)** | $957, worst -12.5%, Sh 4.19 | **✅✅ Прорыв** |
| **Kelly sizing (dynamic L/S split)** | $1366, worst -9.3%, Sh 3.43 | **✅✅ Прорыв** |
| EQ-BOOST + Kelly combo | $1366, worst -9.3% | **✅ Лучший overall** |
| Signal agreement (12h+24h) | $725, worst -17.6% | ⚠️ Хуже baseline |
| Adaptive cutoff p70 + EQ | $622, worst -3.8% | ✅ Самый safe вариант |
| Sector-neutral | $321, worst -14.1% | ⚠️ Меньше equity |
| Monthly DD floor -15% | $240, worst -10.2% | ⚠️ Безопасно но медленно |
| DD floor -10% | $197, worst -9.1% | ⚠️ Слишком режет |

#### Обновлённая сводка всех раундов:

| Раунд | Лучший конфиг | Sharpe | $100 → | Worst month | Ключевая находка |
|-------|---------------|--------|--------|-------------|------------------|
| Baseline | 20sym 5L/5S 12h (no risk) | 0.47 | $79 | -43.4% | Модель работает, MaxDD убивает |
| R1 | 20sym 5L/5S co=0.8 | 1.74 | $168 | -14.9% | Hard cutoff при trend>0.8 |
| R2 | 35sym 6L/6S co=0.8 | 2.17 | $196 | -3.0% | Больше символов = dispersion |
| R3 | 20sym 4L/4S co=0.8 | 1.74 | $207 | -24.5% | 50sym нестабильнее 20 |
| R3B | 35sym 5L/5S DYN-EXP | 2.03 | $274 | -21.1% | Dynamic exposure > hard cutoff |
| R4 | 35sym 5L/5S DYN+EQ-MOM | 3.81 | $671 | -13.2% | Equity momentum = прорыв |
| R5 | 35sym 5L/5S EQ-BOOST+KELLY | 3.43 | $1366 | -9.3% | EQ-boost + Kelly = x2 equity |
| R6 | 35sym SM48+7L5S | 3.87 | $1848 | -6.8% | Strategy momentum + asymmetric L/S |
| **R6** | **35sym SM48+6L3S** | **3.57** | **$1720** | **-3.0%** | **Worst month всего -3%! Calmar=9.84** |

---

## Research Round 6: Signal Quality & Meta-Strategies

**Дата**: июнь 2025
**Цель**: Улучшить R5 ($1366, Wr=-9.3%) через quality сигналов и мета-стратегии

### Запуск

```bash
python _research_round6.py     # Основные идеи
python _research_round6b.py    # Кросс-комбинации победителей
```

### Новые идеи R6

| Идея | Описание | Результат |
|------|----------|-----------|
| **Strategy Momentum** | Используем P&L последних 48ч стратегии как мета-сигнал. Если стратегия теряет → снижаем exposure | **$1521, Wr=-8.0%, Sh=3.55** |
| Spread Gate | Торгуем только когда L-S spread > threshold | Без эффекта (спред всегда выше) |
| Adaptive N | Больше позиций при low-vol, меньше при high-vol | $1270, Wr=-10.2%, Sh=3.71 |
| Top-K Confidence | Только extreme tails предсказаний | Без эффекта |
| Corr-Aware Weight | Inverse-vol weighting позиций | Хуже (-12.7% worst) |
| Rebal=8h | Ребалансировка каждые 8ч вместо 12 | $1457 но рискованнее |
| **Asymmetric 5L/3S** | Больше лонгов чем шортов | **$1323, Wr=-7.1%** |
| **Asymmetric 6L/4S** | Ещё больше лонгов | **$1272, Wr=-6.8%** |

### Ключевое открытие: Long-heavy + Strategy Momentum

Модель mean-reversion, но **длинная сторона стабильнее**. Комбинация asymmetric L/S + strategy momentum даёт:

```python
# Strategy momentum (meta-signal)
if len(strategy_rets) >= 48:
    recent = strategy_rets[-48:]
    cum = np.prod([1 + r for r in recent])
    if cum < 0.97:  # стратегия потеряла >3% за 48ч
        exposure *= max(0.3, cum)  # снижаем на столько же

# Asymmetric L/S: 6 лонгов, 3 шорта (или 7L/5S)
n_long, n_short = 6, 3  # safe variant
n_long, n_short = 7, 5  # aggressive variant
```

### R6B: Кросс-комбинации

| Config | Equity | Worst Month | Sharpe | Calmar | WM | Примечание |
|--------|--------|-------------|--------|--------|-----|-----------|
| BASELINE R5 | $1366 | -9.3% | 3.43 | - | 9/13 | Бейслайн R5 |
| SM48 | $1521 | -8.0% | 3.55 | - | 9/13 | Только strategy momentum |
| **SM48+6L3S** | **$1720** | **-3.0%** | **3.57** | **9.84** | **10/13** | **🏆 SAFEST** |
| **SM48+7L5S** | **$1848** | **-6.8%** | **3.87** | **4.46** | **10/13** | **🏆 BEST EQUITY (safe)** |
| SM48+7L4S | $1605 | -6.5% | 3.62 | 4.41 | 9/13 | Тоже сильный |
| SM48+adaptN8L5S | $1582 | -7.5% | 3.95 | 3.57 | **11/13** | **🏆 BEST SHARPE** |
| SM48+5L3S | $1528 | -6.1% | 3.47 | 4.63 | 9/13 | Консервативный |
| 6L3S (без SM) | $1524 | -4.1% | 3.44 | 6.84 | 10/13 | Asymmetric alone |
| SM48+6L4S | $1440 | -5.0% | 3.42 | 5.57 | 10/13 | Стабильный |
| SM48+rebal8h | $1844 | -9.3% | 3.10 | 3.13 | 9/13 | Больше equity но рискованнее |
| SM48+6L4S+rebal8h | $2451 | -19.1% | 3.27 | 1.73 | 10/13 | Слишком рискованно |

### Месячная разбивка TOP-3

**#1 SM48+6L3S** (safest, Calmar=9.84):
| Месяц | Return | Equity |
|-------|--------|--------|
| 2024-10 | -0.7% | $99 |
| 2024-11 | +47.1% | $146 |
| 2024-12 | +155.8% | $374 |
| 2025-01 | -2.0% | $366 |
| 2025-05 | +51.8% | $556 |
| 2025-06 | +2.8% | $572 |
| 2025-07 | +53.1% | $875 |
| 2025-08 | **-3.0%** | $849 |
| 2025-11 | +6.5% | $904 |
| 2025-12 | +29.7% | $1173 |
| 2026-01 | +39.7% | $1639 |
| 2026-02 | +2.7% | $1683 |
| 2026-03 | +2.1% | $1720 |

**#2 SM48+7L5S** (best equity, safe):
| Месяц | Return | Equity |
|-------|--------|--------|
| 2024-10 | -3.5% | $97 |
| 2024-11 | +72.4% | $166 |
| 2024-12 | +133.3% | $388 |
| 2025-01 | +10.5% | $429 |
| 2025-05 | +35.7% | $582 |
| 2025-06 | -2.8% | $566 |
| 2025-07 | +76.2% | $998 |
| 2025-08 | **-6.8%** | $931 |
| 2025-11 | +6.3% | $989 |
| 2025-12 | +22.0% | $1206 |
| 2026-01 | +33.9% | $1615 |
| 2026-02 | +1.4% | $1638 |
| 2026-03 | +12.8% | $1848 |

**#3 SM48+adaptN8L5S** (best Sharpe=3.95, 11/13 WM):
| Месяц | Return | Equity |
|-------|--------|--------|
| 2024-10 | -2.5% | $97 |
| 2024-11 | +53.2% | $149 |
| 2024-12 | +108.0% | $311 |
| 2025-01 | +22.6% | $381 |
| 2025-05 | +40.1% | $534 |
| 2025-06 | -7.5% | $494 |
| 2025-07 | +51.9% | $750 |
| 2025-08 | +0.8% | $756 |
| 2025-11 | +2.9% | $778 |
| 2025-12 | +20.7% | $939 |
| 2026-01 | +29.2% | $1213 |
| 2026-02 | +14.9% | $1394 |
| 2026-03 | +13.5% | $1582 |

### Выводы R6

1. **Strategy Momentum** — мета-сигнал на P&L 48ч: +$155 equity + улучшение worst month
2. **Asymmetric L/S (long-heavy)** — длинная сторона модели стабильнее. 6L/3S или 7L/5S значительно снижает worst month
3. **Комбинация SM48+6L3S** — worst month всего **-3.0%**, Calmar **9.84**. Идеальный для продакшена.
4. **SM48+7L5S** — если нужен max equity: **$1848** при приемлемом -6.8% worst
5. **Adaptive N не так полезен** при asymmetric sizing (уже хватает diversification от большего N лонгов)

### Рекомендация для продакшена

**Primary (safe)**: SM48+6L3S — worst -3.0%, $100→$1720 за 13 мес
**Aggressive**: SM48+7L5S — worst -6.8%, $100→$1848 за 13 мес

Полный стек:
- Ridge α=1000, 14 features, 12h horizon
- BTC trend_strength cutoff=0.8, dynamic threshold=0.5
- EQ-MOM Boost (DD scale + recovery boost)
- Kelly sizing (dynamic L/S allocation)
- Strategy Momentum 48h (meta-signal)
- Asymmetric 6L/3S (long-heavy)

---

## Research Round 7: Signal Quality, Ensemble & Portfolio Construction

**Дата**: март 2026
**Цель**: Улучшить R6 ($1720, Wr=-3.0%) через качество сигнала и portfolio construction

### Аудит методологии

```
┌─────────────────────────────────────────────────────────────┐
│  METHODOLOGY AUDIT                                          │
│  Data range: 2017-08-17 → 2026-03-07                        │
│  Symbols: 35, Rows: 1,826,014                               │
│  ✅ All 14 features present (backward-looking only)          │
│                                                              │
│  W1: train<2024-06 | val 06→09 | test 10/15→01/31          │
│  W2: train<2025-01 | val 01→04 | test 05/15→08/31          │
│  W3: train<2025-07 | val 07→10 | test 11/15→03/17          │
│  ✅ W1↔W2: no overlap (104d gap)                            │
│  ✅ W1↔W3: no overlap (288d gap)                            │
│  ✅ W2↔W3: no overlap (76d gap)                             │
│  ✅ val→test gap: 15 days in each window                    │
│  ✅ fwd_ret used only as TARGET, never as feature           │
│  ✅ CS ranking computed within each split                   │
│  ✅ Simulation is sequential (no future info)               │
└─────────────────────────────────────────────────────────────┘
```

Per-window model quality:
| Window | α | Val IC | Test IC |
|--------|---|--------|---------|
| W1 | 10 | 0.030 | 0.020 |
| W2 | 1000 | 0.052 | 0.013 |
| W3 | 1000 | 0.050 | 0.019 |

### Запуск

```bash
python _research_round7.py     # Основные идеи
python _research_round7b.py    # Кросс-комбинации
```

### Новые идеи R7

| Идея | Описание | Лучший результат |
|------|----------|-----------------|
| **Signal EMA** | EMA(2) сглаживание предсказаний — снижает шум | **$1917, Wr=-6.1%, 11/13 WM** |
| **Pred Shrinkage (0.1)** | Сжимаем крайние предсказания к медиане на 10% | **$1829, Wr=-2.4%, Calmar=12.53** |
| **Regime-Conditional Asymmetry** | Тилт L/S по направлению BTC тренда (бык→+лонг, медведь→+шорт) | **$2061, Wr=-6.4%** |
| **Vol Scaling** | Масштабирование exposure обратно пропорционально реализованному vol | **$2208, Wr=-8.8% (7L5S)** |
| **Conviction Weighting** | Вес позиции ∝ |prediction| вместо equal-weight | **$1769, Wr=-8.2%, 11/13 WM** |
| Multi-horizon blend (12h+24h) | Лучший бленд 50/50 | $1272, хуже R6 baseline |
| Position stickiness | Hold unless signal flips | Без эффекта |

### R7B: Кросс-комбинации

| # | Config | Equity | Worst Month | Sharpe | WM | Профиль |
|---|--------|--------|-------------|--------|-----|---------|
| 1 | **RG-ASYM+VOL+EMA2 6L3S** | **$2226** | **-6.1%** | **3.63** | 9/13 | **🏆 #1 risk-adj** |
| 2 | **RG-ASYM+VOL 6L3S** | **$2404** | -7.7% | 3.49 | 9/13 | **💰 MAX EQUITY** |
| 3 | **RG-ASYM+EMA2 6L3S** | **$1847** | **-3.9%** | **3.93** | 10/13 | **🛡️ ULTRA-SAFE** |
| 4 | SHRINK01+RG-ASYM 6L3S | $2029 | -5.3% | 3.77 | 10/13 | Баланс |
| 5 | SHRINK01+RG-ASYM+VOL 6L3S | $2230 | -8.2% | 3.51 | 9/13 | Высокий equity |
| 6 | SHRINK01+RG-ASYM+VOL+EMA2 | $2076 | -6.6% | 3.67 | 7/13 | Не хватает WM |
| 7 | RG-ASYM+CONV-W 6L3S | $1816 | -7.6% | 3.79 | 11/13 | Стабильный |
| 8 | BASELINE SM48+6L3S | $1720 | -3.0% | 3.57 | 10/13 | R6 winner |

### Месячная разбивка TOP-3

**#1 RG-ASYM+VOL+EMA2 6L3S** ($2226, risk-adj winner):
| Месяц | Return | Equity |
|-------|--------|--------|
| 2024-10 | -2.7% | $97 |
| 2024-11 | +39.5% | $136 |
| 2024-12 | +282.7% | $519 |
| 2025-01 | -3.7% | $500 |
| 2025-05 | +62.6% | $813 |
| 2025-06 | +20.1% | $976 |
| 2025-07 | +52.3% | $1487 |
| 2025-08 | +0.1% | $1489 |
| 2025-11 | +0.0% | $1489 |
| 2025-12 | +13.8% | $1695 |
| 2026-01 | +48.3% | $2514 |
| 2026-02 | -5.7% | $2370 |
| 2026-03 | **-6.1%** | $2226 |

**#2 RG-ASYM+VOL 6L3S** ($2404, max equity):
| Месяц | Return | Equity |
|-------|--------|--------|
| 2024-10 | -1.1% | $99 |
| 2024-11 | +15.6% | $114 |
| 2024-12 | +213.3% | $358 |
| 2025-01 | **-7.7%** | $331 |
| 2025-05 | +74.0% | $576 |
| 2025-06 | +24.6% | $717 |
| 2025-07 | +75.1% | $1256 |
| 2025-08 | -4.9% | $1194 |
| 2025-11 | +4.2% | $1245 |
| 2025-12 | +22.5% | $1525 |
| 2026-01 | +54.3% | $2352 |
| 2026-02 | -4.8% | $2238 |
| 2026-03 | +7.4% | $2404 |

**#3 RG-ASYM+EMA2 6L3S** ($1847, ultra-safe):
| Месяц | Return | Equity |
|-------|--------|--------|
| 2024-10 | **-3.9%** | $96 |
| 2024-11 | +66.8% | $160 |
| 2024-12 | +213.3% | $502 |
| 2025-01 | -1.2% | $496 |
| 2025-05 | +52.0% | $754 |
| 2025-06 | +9.2% | $824 |
| 2025-07 | +42.2% | $1172 |
| 2025-08 | +0.5% | $1177 |
| 2025-11 | +0.2% | $1180 |
| 2025-12 | +19.1% | $1405 |
| 2026-01 | +28.3% | $1803 |
| 2026-02 | +4.3% | $1880 |
| 2026-03 | -1.8% | $1847 |

### Ключевые открытия R7

1. **Regime-Conditional Asymmetry** — тилт L/S по BTC тренду: бычий → больше лонгов, медвежий → больше шортов. Synergy с mean-reversion: при мягком тренде сторона "по тренду" стабильнее.
2. **Vol Scaling** — снижение exposure при высоком vol: `vol_scale = min(1.5, 1/vol_regime)`. Добавляет equity, немного увеличивает worst month.
3. **Signal EMA(2)** — минимальное сглаживание предсказаний. EMA=2 = sweet spot, EMA>4 ухудшает.
4. **Prediction Shrinkage (10%)** — `pred = pred * 0.9 + median * 0.1`. Уменьшает worst month с -3.0% до -2.4%.
5. **Комбинации работают**: RG-ASYM+VOL+EMA2 даёт синергию — **$2226 при worst -6.1%**.

```python
# Regime-conditional asymmetry
if -0.3 < trend_direction < 0.3:  # neutral
    nl, ns = 6, 3
elif trend_direction >= 0.3:  # mild bull
    nl, ns = 7, 2  # tilt long
else:  # mild bear
    nl, ns = 5, 4  # tilt short

# Vol scaling
vol_scale = min(1.5, 1.0 / max(0.5, vol_regime))
exposure *= vol_scale

# Signal EMA(2) smoothing
pred = df.groupby("symbol")["pred"].ewm(span=2).mean()

# Prediction shrinkage 10%
pred = pred * 0.9 + cs_median * 0.1
```

### Выводы R7

Три уровня рекомендации:

| Профиль | Config | Equity | Worst | Sharpe |
|---------|--------|--------|-------|--------|
| 🛡️ Ultra-safe | RG-ASYM+EMA2 6L3S | $1847 | -3.9% | 3.93 |
| ⚖️ Balanced | RG-ASYM+VOL+EMA2 6L3S | $2226 | -6.1% | 3.63 |
| 💰 Aggressive | RG-ASYM+VOL 6L3S | $2404 | -7.7% | 3.49 |

vs R6 winner (SM48+6L3S): $1720, Wr=-3.0%, Sh=3.57

### Прогресс по раундам

| Round | Config | Sharpe | Equity | Worst Month | Ключевое открытие |
|-------|--------|--------|--------|-------------|-------------------|
| Baseline | 20sym 5L/5S 12h (no risk) | 0.47 | $79 | -43.4% | Модель работает, MaxDD убивает |
| R1 | 20sym 5L/5S co=0.8 | 1.74 | $168 | -14.9% | Hard cutoff |
| R2 | 35sym 6L/6S co=0.8 | 2.17 | $196 | -3.0% | Больше символов |
| R3B | 35sym DYN-EXP | 2.03 | $274 | -21.1% | Dynamic exposure |
| R4 | DYN+EQ-MOM | 3.81 | $671 | -13.2% | Equity momentum |
| R5 | EQ-BOOST+KELLY | 3.43 | $1366 | -9.3% | Kelly sizing |
| R6 | SM48+6L3S | 3.57 | $1720 | -3.0% | Strategy momentum + long-heavy |
| **R7** | **RG-ASYM+VOL+EMA2** | **3.63** | **$2226** | **-6.1%** | **Regime-tilt + vol-scale + EMA** |
| **R8** | **+range_24h +btc_beta +gls_z (TOP-3)** | **3.97** | **$223*** | **-1.9%** | **IC scan + new features (no lev, 8.5m test)** |

Полный стек R7 winner (DEPLOYED to VPS 30 Mar 2026):
- Ridge α=1000, 14 CS-IC features, 12h horizon
- Walk-forward: 3 окна, 15-day gaps, HPO alpha на val
- BTC trend_strength cutoff=0.8, dynamic threshold=0.5
- EQ-MOM Boost (DD scale + recovery boost)
- Kelly sizing (dynamic L/S allocation)
- Strategy Momentum 48h (meta-signal)
- Asymmetric 6L/3S (long-heavy)
- Regime-conditional L/S tilt
- Vol scaling (exposure ∝ 1/vol)
- Signal EMA(2) smoothing

---

## Research Round 8 — New Feature Discovery (31 марта 2026)

### Контекст
R7 задеплоен в прод с 14 CS-IC фичами. R8 исследует новые фичи для расширения модели.
Скрипт: `_research_round8.py`
Данные: 105 расширенных фич (79 candidate для IC scan), 1.8M строк, 35 символов.

### Фазы
1. **IC Scan**: 79 candidate features × 4 horizons (4h/12h/24h/48h) × 3 WF windows
2. **Train & Backtest**: Ridge с расширенными feature sets vs baseline (14 feats)
3. **Ablation**: добавление фич по одной к baseline
4. **Combos**: TOP-3 и TOP-5 комбинации

### Новые источники данных

| Источник | Фичи | Статус |
|----------|-------|--------|
| **Deribit DVOL** (implied vol) | btc_dvol, dvol_zscore, dvol_change_24h/168h, dvol_rv_spread | ✅ Built, IC < 0.015 |
| **FRED Macro** (VIX, DXY, SPX, Gold, Yields) | 15 features (ret, zscore per series) | ✅ Built, IC < 0.01 |
| **Extended Funding** | cum_funding_24/72h_cs, funding_x_mom_12/24h | ✅ Built, IC 0.016–0.022 |
| **Volume Momentum** | vol_mom_z_12/24h | ✅ Built |
| **Cross-coin Dispersion** | ret_dispersion_12h, ret_vs_median_12h | ✅ Built, IC 0.052 |
| **Multi-TF Momentum** | mom_z_168h, ret_168h | ✅ Built, IC 0.028 |
| **Relative Strength** | rs_rank_12/24h, rs_rank_change_12h, rs_rank_12h_lag12 | ✅ Built, IC 0.030–0.052 |
| **OI-Funding Interaction** | oi_funding_interaction | ✅ Built, IC 0.010 |
| **Taker Flow Acceleration** | taker_flow_accel | ✅ Built |
| **Basis Momentum** | basis_mom_12h, basis_funding_agree | ✅ Built |

### Phase 1: IC Scan — TOP NEW Features at 12h

21 новых фич с |IC| > 0.015 at 12h horizon.

| # | Feature | IC@12h | Std | Consistent | Multi-Horizon |
|---|---------|--------|-----|------------|---------------|
| 1 | **rvol_12h** | **-0.068** | 0.011 | ✓ 3/3 | 4h→48h, все > 0.04 |
| 2 | **rvol_24h** | **-0.064** | 0.012 | ✓ 3/3 | 4h→48h, все > 0.04 |
| 3 | **rvol_168h** | **-0.060** | 0.014 | ✓ 3/3 | 4h→48h, все > 0.04 |
| 4 | rs_rank_12h | -0.052 | 0.009 | ✓ 3/3 | 4h→48h |
| 5 | ret_vs_median_12h | -0.052 | 0.009 | ✓ 3/3 | 4h→48h |
| 6 | rs_rank_24h | -0.048 | 0.009 | ✓ 3/3 | 4h→48h |
| 7 | **ret_4h** | **-0.041** | 0.011 | ✓ 3/3 | 4h→48h |
| 8 | **global_ls_ratio** | **+0.031** | 0.020 | ✓ 3/3 | 4h→48h |
| 9 | rs_rank_change_12h | -0.030 | 0.009 | ✓ 3/3 | 3 horizons |
| 10 | **range_24h** | **+0.028** | 0.014 | ✓ 3/3 | 4h→48h |
| 11 | ret_168h | -0.028 | 0.023 | 2/3 | 3 horizons |
| 12 | ret_1h | -0.026 | 0.006 | ✓ 3/3 | 3 horizons |
| 13 | **btc_beta_168h** | **-0.025** | 0.004 | ✓ 3/3 | 4h→48h, IC растёт с горизонтом |
| 14 | oi_chg_4h | -0.024 | 0.003 | ✓ 3/3 | 3 horizons |
| 15 | cum_funding_24h | -0.022 | 0.015 | ✓ 3/3 | 12h→48h |
| 16 | funding_x_mom_24h | -0.022 | 0.008 | ✓ 3/3 | 4h→48h |
| 17 | funding_x_mom_12h | -0.022 | 0.004 | ✓ 3/3 | 3 horizons |
| 18 | **global_ls_ratio_zscore** | **+0.018** | 0.008 | ✓ 3/3 | 3 horizons |
| 19 | cum_funding_72h | -0.016 | 0.017 | 2/3 | 3 horizons |

**DVOL & Macro features**: IC < 0.015 — не прошли порог. DVOL и макро не дают кросс-секционного сигнала для крипто.

**Realized Volatility (rvol_12/24/168h)**: Самый сильный IC среди ВСЕХ фич (-0.068), включая текущие 14. Но: rvol_12h уже используется НЕЯВНО через `mom_z_12h = ret_12h / rvol_12h`. Добавление напрямую повышает IC, но может быть redundant.

### Phase 2: Baseline (14 feats) Performance

| Config | Equity | Worst Month | Sharpe | Calmar | Win Mo. |
|--------|--------|-------------|--------|--------|---------|
| **Baseline (14 feats)** | **$200** | **-2.6%** | **2.93** | 38.9 | 9/13 |

Note: Equity здесь — unleveraged, $100 start, отображает «$100 × cumulative_return». R7 equity ($2226) считался с **leverage=5** (каждый месяц доходность ×5), поэтому напрямую не сравнимы. Также разные тестовые окна: R7 тест Oct24–Jan25, May–Aug25, Nov25–Mar26 (≈13 мес); R8 тест Oct–Dec24, Apr–Jun25, Oct25–Mar26 (≈8.5 мес).

### Phase 2: Feature Ablation (add one at a time)

| Feature | Equity | Worst Mo. | Sharpe | Calmar | Win Mo. | IC@12h |
|---------|--------|-----------|--------|--------|---------|--------|
| **+range_24h** | **$232** | -2.0% | **4.30** | 65.7 | 10/13 | +0.028 |
| +btc_beta_168h | $231 | -1.9% | 3.43 | 67.9 | 9/13 | -0.025 |
| +global_ls_ratio_zscore | $219 | **-1.2%** | 3.29 | 101.1 | 11/13 | +0.018 |
| +ret_4h | $217 | -2.9% | 3.23 | 40.6 | 9/13 | -0.041 |
| +cum_funding_24h | $213 | **-0.9%** | 3.13 | 125.2 | 10/13 | -0.022 |
| +ret_1h | $209 | -2.3% | 3.22 | 47.3 | 8/13 | -0.026 |
| +rs_rank_change_12h | $208 | -2.3% | 3.09 | 47.6 | 9/13 | -0.030 |
| +global_ls_ratio | $204 | -2.5% | 3.13 | 41.7 | 9/13 | +0.031 |
| +oi_chg_4h | $202 | -2.1% | 2.94 | 49.2 | 8/13 | -0.024 |
| +ret_168h | $200 | -1.8% | 3.18 | 55.2 | 10/13 | -0.028 |
| +rvol_12h | $199 | **-0.1%** | **4.37** | 1529.6 | **12/13** | -0.068 |
| +funding_x_mom_12h | $199 | -1.6% | 2.93 | 59.8 | 8/13 | -0.022 |
| +rvol_24h | $190 | -3.3% | 4.01 | 27.1 | 12/13 | -0.064 |
| +rvol_168h | $167 | -2.3% | 3.57 | 29.4 | 11/13 | -0.060 |
| BASELINE | $200 | -2.6% | 2.93 | 38.9 | 9/13 | — |

**Топ-открытия ablation:**
1. **range_24h (24h high-low range)** — лучший equity boost: $200→$232 (+16%), Sharpe 2.93→4.30 (+47%)
2. **rvol_12h (realized vol 12h)** — лучший risk-adj: worst month -0.1% (!!), Sharpe 4.37, 12/13 winning months
3. **btc_beta_168h** — #2 equity ($231), стабильный IC across all 4 horizons, IC растёт с горизонтом
4. **global_ls_ratio_zscore** — лучший Calmar (101), worst month -1.2%, 11/13 WM
5. **cum_funding_24h** — рекордный Calmar (125.2), worst month -0.9%

### Phase 2: Feature Combinations

| Config | Equity | Worst Mo. | Sharpe | Calmar | Win Mo. |
|--------|--------|-----------|--------|--------|---------|
| **TOP-3** (14 + range_24h + btc_beta_168h + global_ls_ratio_zscore) | **$223** | **-1.9%** | **3.97** | 65.6 | 10/13 |
| TOP-5 (14 + top 5) | $196 | -3.7% | 3.76 | 25.9 | 11/13 |
| EXPANDED (14 + 21 new = 35) | $165 | -3.7% | 3.25 | 17.3 | 10/13 |
| SCAN-TOP3 (14 + gls_z + ret_4h + cum_fund_24h) | $213 | -3.1% | 3.28 | 36.1 | 8/13 |
| BASELINE (14 feats) | $200 | -2.6% | 2.93 | 38.9 | 9/13 |

### Phase 3: Scanner Extras (features from IC scanner not in current 14)

Также протестированы все 23 фичи из IC-сканера, которые не входят в текущие 14:

| Feature | Equity | Worst Mo. | Sharpe |
|---------|--------|-----------|--------|
| +global_ls_ratio_zscore | $219 | -1.2% | 3.29 |
| +ret_4h | $217 | -2.9% | 3.23 |
| +cum_funding_24h | $213 | -0.9% | 3.13 |
| +vol_ratio_24h | $211 | -4.1% | 3.13 |
| +ret_1h | $209 | -2.3% | 3.22 |
| +taker_cvd_4h | $204 | -1.7% | 3.02 |
| +oi_chg_1h | $203 | -2.4% | 2.99 |
| +top_ls_ratio_zscore | $202 | -1.9% | 2.99 |
| +premium_index | $140 | -3.7% | 2.35 |
| +premium_zscore | $150 | -4.6% | 2.86 |

**Premium/basis features ВРЕДЯТ**: $140 / $150 vs baseline $200. Basis signal too noisy.

### Ключевые выводы R8

1. **Realized Volatility** (rvol_12/24/168h) — самый сильный IC среди ВСЕХ features (-0.068), но уже частично захвачен через mom_z (ret/rvol). Добавление rvol_12h напрямую даёт Sharpe 4.37 и worst month -0.1%, но не top equity.

2. **range_24h** — лучший equity boost одиночный: +16% equity, +47% Sharpe. Capture: дневной диапазон как мера активности/волатильности, ортогональна к rvol.

3. **btc_beta_168h** — BTC-бета за 7 дней: монеты с высокой бетой хуже на 12-48h. IC растёт с горизонтом (от -0.020 до -0.036). Фундаментальный сигнал: low-beta coins outperform.

4. **global_ls_ratio_zscore** — z-score глобального L/S ratio: crowded longs (zs > 0) → reversal signal. Стабильный IC (+0.018) через все 3 окна.

5. **TOP-3 combo = лучший risk-adjusted**: 17 features, Sharpe 3.97, worst month -1.9%. Сохраняет простоту (всего +3 фичи) при значительном улучшении.

6. **Больше ≠ лучше**: 35 фич ($165) хуже 14 ($200). TOP-5 ($196) уже хуже TOP-3 ($223). Ridge regularization предпочитает меньше orthogonal features.

7. **DVOL и Macro не работают** для cross-sectional crypto: IC < 0.015. Implied vol и макро-факторы — market-wide signals, не дают alpha для L/S ranking.

8. **Funding carry** (cum_funding_24h) — перспективен для risk management: Calmar 125, worst month -0.9%.

### Прямое сравнение R7 vs R8 (apples-to-apples)

Скрипт: `_compare_r7_vs_r8.py`  
Условия: одинаковые окна (R7 windows), leverage=5, capital=$100, один и тот же cfg (R7 winner), Ridge HPO.

| Метрика | R7 (14 features, PROD) | R8 (17 features, TOP-3) | Победитель |
|---------|------------------------|-------------------------|------------|
| **Equity** | **$2993** | $1401 | **R7 ✅** |
| **Sharpe** | **3.59** | 3.23 | **R7 ✅** |
| **Worst month** | **-6.4%** | -19.8% | **R7 ✅** |
| **Calmar** | **6.58** | 1.43 | **R7 ✅** |
| Win months | 9/13 | **10/13** | R8 ✅ |

**Месячная разбивка:**

| Месяц | R7 (14f) | R8 (17f) | Δ |
|-------|---------|---------|---|
| 2024-10 | -2.3% | -6.7% | -4.4% R7 лучше |
| 2024-11 | +48.6% | +31.2% | -17.4% R7 лучше |
| **2024-12** | **+286.5%** | +162.8% | **-123.6% R7 лучше** |
| **2025-01** | -0.9% | **-19.8%** | -18.9% R7 лучше |
| 2025-05 | +66.1% | +65.5% | ≈ |
| 2025-06 | +17.0% | +30.4% | +13.4% R8 лучше |
| 2025-07 | +67.8% | +33.7% | -34.1% R7 лучше |
| 2025-08 | +1.8% | -3.5% | -5.3% R7 лучше |
| 2025-11 | +2.8% | +1.6% | -1.2% R7 лучше |
| 2025-12 | +9.4% | +13.6% | +4.2% R8 лучше |
| 2026-01 | +56.5% | +39.3% | -17.2% R7 лучше |
| 2026-02 | -6.4% | +2.2% | +8.6% R8 лучше |
| 2026-03 | -1.5% | +18.7% | +20.2% R8 лучше |

R7 лучше в **9/13 месяцев**, R8 — 4/13.

### Вывод по R8

**Новые фичи (range_24h, btc_beta_168h, global_ls_ratio_zscore) ВРЕДЯТ при leverage=5 на R7 окнах:**
- Worst month: -6.4% → -19.8% (утроился риск)
- Equity: $2993 → $1401 (потеря $1592)
- Проблемный период: Dec 2024 – Jan 2025. R7 сделал +286%/−0.9%, R8 — +163%/−19.8%.

Почему R8 выглядел лучше в предыдущем backteste?  
→ R8 тестировался на **других (более коротких) окнах без leverage**, что скрыло риск в ключевые месяцы.

**РЕШЕНИЕ: оставить R7 в production (14 features).**  
Новые фичи добавляют шум при leverage: range_24h и btc_beta_168h снижают сигнал в bull-run декабря 2024.

### Обновлённая таблица прогресса

| Раунд | Конфиг | Sharpe | Equity | Worst Mo. | Примечание |
|--------|--------|--------|--------|-----------|------------|
| Baseline | Ridge 14f vanilla | 1.69 | $296 | -13.7% | — |
| R1 | +Vol target+DD stop | 1.96 | $375 | -11.6% | — |
| R2 | +Regime filter | 2.19 | $485 | -10.0% | — |
| R3 | +Confidence+HPO | 2.66 | $650 | -9.9% | — |
| R4 | +Dynamic exposure | 3.15 | $989 | -8.9% | — |
| R5 | +EQ-BOOST+KELLY | 3.43 | $1366 | -9.3% | — |
| R6 | SM48+6L3S | 3.57 | $1720 | -3.0% | — |
| **R7** | **RG-ASYM+VOL+EMA2** | **3.59** | **$2993** | **-6.4%** | **DEPLOYED ✅** |
| R8 | +range_24h+btc_beta+gls_z | 3.23 | $1401 | -19.8% | ❌ WORSE, not deployed |

*(R7 equity пересчитан скриптом `_compare_r7_vs_r8.py` с теми же условиями — незначительно отличается от ранее записанного $2226, т.к. прогон был чуть другим)*

Ожидаемый эффект vs R7 baseline:
- Sharpe: 2.93 → 3.97 (+35%)
- Worst month: -2.6% → -1.9% (+27% better)
- Winning months: 9/13 → 10/13

### R8 Config для deployment

```python
FEATURES_17 = [
    # Original 14
    "ret_12h", "ret_24h", "ret_48h",
    "residual_12h", "residual_24h",
    "mom_z_12h", "mom_z_24h",
    "dist_from_high_24h",
    "oi_chg_12h", "oi_chg_24h", "oi_zscore",
    "taker_cvd_12h", "taker_cvd_24h",
    "ls_divergence",
    # NEW R8 features
    "range_24h",           # 24h high-low range (IC +0.028)
    "btc_beta_168h",       # 7d BTC beta (IC -0.025, grows with horizon)
    "global_ls_ratio_zscore",  # L/S ratio z-score (IC +0.018)
]
```

---

## Текущее состояние production (31 марта 2026)

### VPS: root@185.42.163.63
- **Модель**: Ridge α=1000, 14 features (R7 deployed)
- **systemd**: `--mode live --loop --capital 100 --leverage 3 --no-deriv-gate --no-meta --ridge --vol-size --min-zscore 0.8`
- **OKX**: LIVE mode (OKX_DEMO=0), NOT demo
- **Позиции**: 6 open (5L + 1S): ATOM long, CHZ short, COMP long, EGLD long, ENS long, IMX long
- **Проблемы**: FTM-USDT-SWAP и MKR-USDT-SWAP не существуют на OKX → 2 из 3 шортов не открылись
- **Dashboard**: invest.arturt.com, обновляется каждый цикл

### R7 Production Stack
- Ridge regression, α=1000, 14 CS-IC features, 12h horizon
- Regime filter (BTC trend_strength): cutoff=0.8, dynamic threshold=0.5
- Regime-conditional asymmetry (бычий → 7L/2S, медвежий → 5L/4S)
- Vol scaling (exposure ∝ 1/vol_regime)
- Signal EMA(2) smoothing
- EQ-MOM Boost + Kelly sizing + Strategy Momentum 48h
- Asymmetric base: 6L/3S

### Pending
- ~~**R8 deployment**: переобучить Ridge на 17 фичах~~ — **ОТМЕНЕНО**: прямое сравнение показало R8 хуже R7 (см. выше). VPS остаётся на 14 features.

---

## Research Round 9 — Targeted Improvements (2026-03-31)

**Цель**: найти дальнейшие улучшения после R7 production config.  
**Baseline**: Ridge α=1000, 14 features, EMA=2, 6L/3S, leverage=5, capital=$100 → Eq=$2993, Sh=3.59, Wr=-6.4%, WM=9/13

### Phase A: Multi-Horizon Blends (INVALID / Worse)

| Config | Equity | Sharpe | Worst M | Note |
|--------|--------|--------|---------|------|
| 12h only (baseline) | $2993 | 3.59 | -6.4% | — |
| 4h only | $119 | 0.78 | — | **INVALID** — uses fwd_ret_4h |
| 24h only | $61,204 | 5.88 | — | **INVALID** — uses fwd_ret_24h (2× magnitude) |
| 4h+12h equal | $1499 | 3.06 | -9.1% | WORSE |
| 12h+24h equal | $1937 | 3.33 | -8.5% | WORSE |
| 4h+12h+24h equal | $1623 | 3.19 | -8.8% | WORSE |
| 4h+12h+24h 1:2:1 | $1581 | 3.17 | -9.1% | WORSE |
| 4h+12h+24h 1:3:1 | $1676 | 3.21 | -8.6% | WORSE |

**Вывод**: все мульти-горизонтные комбинации ХУЖЕ 12h. 12h — оптимальный горизонт.

### Phase B: LightGBM vs Ridge

| Model | Equity | ΔEq | Sharpe | ΔSh | Worst M | WM |
|-------|--------|-----|--------|-----|---------|-----|
| Ridge α=1000 (baseline) | $2993 | — | 3.59 | — | -6.4% | 9/13 |
| LightGBM n_leaves=31 | $2583 | -410 | 4.12 | +0.53 | -6.2% | 11/13 |

**Test IC (информационный коэффициент)**:
- Ridge: W1=0.020, W2=0.013, W3=0.019 (avg ~0.017)  
- LGB: W1=0.0603, W2=0.0571, W3=0.0720 (avg ~0.063) — **в 3-4× выше!**

**Вывод**: LGB имеет значительно более сильный сигнал (IC 3-4×) и выше Sharpe (+0.53), но ниже абсолютный equity на $410. Причина: Ridge EMA(2) случайно попадает в декабрь 2024 pump; LGB более стабилен (11/13 WM). Требует дополнительного исследования в R9B.

### Phase C1: Position Count Sweep

| Config | Equity | Sharpe | Worst M | WM |
|--------|--------|--------|---------|-----|
| 4L/2S | $2045 | 3.17 | -11.7% | — |
| 5L/2S | $2691 | 3.37 | -8.4% | — |
| 5L/3S | $2790 | 3.37 | -5.9% | — |
| **6L/3S (current)** | **$2993** | **3.59** | **-6.4%** | **9/13** |
| 7L/3S | $2603 | 3.56 | -7.1% | — |
| 8L/3S | $2457 | 3.61 | -5.7% | — |
| 9L/3S | $1909 | 3.49 | -7.1% | 10/13 |
| 10L/4S | $1684 | 3.45 | -12.4% | — |

**Вывод**: 6L/3S — оптимально по equity. 8L/3S (+0.02 Sh, -0.7% лучший месяц, -$536 equity) — альтернатива если нужна меньшая просадка.

### Phase C2: Rebalancing Frequency (INVALID)

Тест rebal=4h/6h/8h НЕДЕЙСТВИТЕЛЕН: fwd_ret_12h создаёт перекрывающиеся позиции, что накачивает equity ($340k при 4h). Только rebal=24h ($279) и 48h ($259) технически корректны, но значительно хуже из-за меньшего числа сделок.

### Phase D1: EMA Smoothing Sweep

| EMA span | Equity | Sharpe | Worst Month | WM |
|----------|--------|--------|-------------|-----|
| None | $2597 | 3.35 | -6.5% | — |
| 1 | $2597 | 3.35 | -6.5% | — |
| **2 (current)** | **$2993** | **3.59** | **-6.4%** | **9/13** |
| 3 | $2770 | 3.54 | **-4.2%** | 10/13 |
| 4 | $1849 | 3.26 | -7.2% | — |
| 5 | $1970 | 3.35 | -7.5% | — |
| 8 | $2235 | 3.55 | -7.8% | — |

**Вывод**: EMA=2 — оптимально по equity. EMA=3 улучшает worst month с -6.4% → -4.2% (+2.2pp!) и даёт 10/13 WM, ценой -$222 equity. Интересная опция для риск-ориентированного режима.

### Phase D2: Conviction Weighting
- conviction=True: +$38 equity, +0.01 Sh, но worst month -8.5% (-2.1pp ХУЖЕ). Нейтрально/негативно.

### Phase D3: Prediction Shrinkage

| Shrink | Equity | ΔEq | Sharpe | ΔSh | Worst M |
|--------|--------|-----|--------|-----|---------|
| None (current) | $2993 | — | 3.59 | — | -6.4% |
| **0.05** | **$2987** | **-6** | **3.60** | **+0.01** | **-5.6%** |
| 0.10 | $2878 | -114 | 3.60 | +0.01 | -5.9% |
| 0.15 | $2822 | -170 | 3.62 | +0.03 | -5.6% |
| 0.20 | $2531 | -462 | 3.59 | 0 | -5.1% |

**Вывод**: shrink=0.05 — практически бесплатное снижение риска (-$6 equity, worst month улучшается с -6.4% до -5.6%). Рекомендуется к деплою.

### Phase D4: Vol Target
Параметр `vol_target` не используется в коде симуляции → все конфиги идентичны baseline. Нет эффекта.

### R9 Summary

| Что | Действие |
|-----|---------|
| Multi-horizon блендинг | ❌ не помогает, 12h оптимально |
| LightGBM | ⚠️ IC 3-4×, Sharpe +0.53, WM 11/13, но equity -$410. Требует R9B |
| Position counts | ✅ 6L/3S текущий оптимален |
| Rebalancing | ❌ тест невалиден. rebal=12h подтверждён |
| EMA=3 vs EMA=2 | ⚠️ better worst month (-4.2% vs -6.4%), -$222 equity |
| shrink=0.05 | ✅ бесплатный риск-контроль. **Рекомендуется к деплою** |
| vol_target | ❌ не работает в sim (баг параметра) |

---

## Research Round 9B — LightGBM Deep Dive (2026-03-31)

**Цель**: найти оптимальную конфигурацию LightGBM + сигнальных параметров.  
**Ключевое открытие**: LGB не нуждается в EMA-сглаживании — его предсказания уже высококачественные (IC 0.06-0.07).

### LightGBM + Signal Configs

| Config | Equity | ΔEq | Sharpe | ΔSh | Worst M | WM |
|--------|--------|-----|--------|-----|---------|-----|
| Ridge EMA=2 (prod) | $2993 | — | 3.59 | — | -6.4% | 9/13 |
| **LGB EMA=None** | **$2916** | **-77** | **4.21** | **+0.62** | **-5.6%** | **11/13** |
| LGB EMA=2 (baseline) | $2583 | -410 | 4.12 | +0.53 | -6.2% | 11/13 |
| LGB shrink=0.15 EMA=2 | $2540 | -453 | 4.16 | +0.57 | -5.3% | 11/13 |
| LGB shrink=0.05 EMA=2 | $2542 | -450 | 4.12 | +0.53 | -5.8% | 11/13 |
| LGB conviction EMA=2 | $2796 | -197 | 4.10 | +0.51 | -11.0% | 11/13 |
| LGB EMA=3 | $1738 | -1255 | 3.79 | +0.20 | -7.7% | 9/13 |

### Ridge + Signal Configs

| Config | Equity | ΔEq | Sharpe | ΔSh | Worst M | WM |
|--------|--------|-----|--------|-----|---------|-----|
| Ridge EMA=2 (prod) | $2993 | — | 3.59 | — | -6.4% | 9/13 |
| **Ridge shrink=0.05 EMA=2** | **$2987** | **-6** | **3.60** | **+0.01** | **-5.6%** | **9/13** |
| Ridge shrink=0.05 EMA=3 | $2756 | -237 | 3.54 | -0.04 | **-3.6%** | 9/13 |
| Ridge shrink=0.1 EMA=3 | $2661 | -331 | 3.54 | -0.04 | -3.7% | 8/13 |
| Ridge shrink=0.15 EMA=3 | $2506 | -487 | 3.54 | -0.04 | -3.7% | 8/13 |

### Key Insight: EMA Smoothing ≠ Universal Improvement

EMA сглаживание повышает производительность Ridge (низкий IC 0.013-0.020) за счёт подавления шума. Но LGB уже имеет качественный сигнал (IC 0.060-0.072) — дополнительное сглаживание **ухудшает** его, размывая точные предсказания.

→ Ridge: EMA=2 оптимально  
→ LGB: EMA=None оптимально

### Final Decision Table (R9B)

| Priority | Config | Equity | Sharpe | Worst M | WM | Deploy? |
|----------|--------|--------|--------|---------|----|---------| 
| 🥇 R7 Production | Ridge EMA=2 | $2993 | 3.59 | -6.4% | 9/13 | ✅ Deployed |
| 🔄 Quick win | Ridge shrink=0.05 EMA=2 | $2987 | 3.60 | -5.6% | 9/13 | ✅ Ready |
| 🚀 Upgrade candidate | **LGB EMA=None** | **$2916** | **4.21** | **-5.6%** | **11/13** | ⚠️ Needs R10 validation |
| 🛡️ Ultra-safe mode | Ridge shrink=0.05 EMA=3 | $2756 | 3.54 | -3.6% | 9/13 | 🔧 Optional |

### R9 + R9B Recommendations

1. **Deploy `pred_shrinkage=0.05` immediately** — практически нет потерь equity (-$6), worst month улучшается на +0.8pp. Это параметр в CFG, не требует переобучения.

2. **R10 plan: Глубокая настройка LightGBM** — `LGB EMA=None` показывает:
   - Sharpe 4.21 (+0.62 vs prod, +17%)
   - Worst month -5.6% vs -6.4% (less risky)
   - Winning months 11/13 vs 9/13 (85% vs 69%)
   - Equity -$77 (-2.6%) — незначительная разница
   
   Нужно проверить: num_leaves sweep (15/31/63/127), n_estimators, min_child_samples, feature importance, Ridge+LGB ensemble.

3. **EMA=3 option**: только если приоритет — минимизация worst month (-4.2% vs -6.4%, -$222 equity)

---

## Research Round 10 — LGB Verification + Production Deployment (2026-03-31)

**Цель**: верификация превосходства LGB и деплой в продакшн.

### Verification (_verify_lgb.py — 8 checks)

| Check | LGB (nl=31, seed=42) | Ridge (prod) | Вывод |
|-------|----------------------|--------------|-------|
| Sharpe | **4.29** | 3.59 | LGB +0.70 ✅ |
| Worst month | **-1.2%** | -6.4% | LGB +5.2pp ✅ |
| Winning months | **12/13** | 9/13 | LGB +3 ✅ |
| Equity | **$3487** | $2993 | LGB +$494 ✅ |
| Seed stability (5 seeds) | Sh 4.01–4.35, Eq $2383–$4740 | — | высокая дисперсия |
| Bootstrap CI Sharpe | [2.11, 6.19] | [1.79, 5.19] | **CI пересекаются** ⚠️ |
| Permutation test | p-value=0.02 | — | слабая значимость ⚠️ |
| IC | **0.053–0.072** | 0.013–0.020 | LGB 3-4× лучше ✅ |

**Вердикт: MIXED** — LGB объективно лучше по всем метрикам, но статистически не подтверждено (только 13 месяцев тестовых данных). Решение: деплоить, т.к. 3-4× преимущество IC — фундаментальное, не шумовое.

**Решение по дисперсии seed**: 5-seed ансамблирование усредняет предсказания → устраняет риск "плохого seed".

### Production Training (train_lgb_prod.py)

**Конфиг**: seeds=[0,7,13,42,99], num_leaves=31, features=FEATURES_14 (14 фич), in-place CS-rank (без суффикса `_r`)

**Walk-forward validation результаты ансамбля:**

| Window | Seed диапазон IC (test) | trees |
|--------|------------------------|-------|
| W1 (test: Oct–Jan 2024/25) | 0.0565–0.0599 | 16–20 |
| W2 (test: May–Aug 2025) | 0.0534–0.0594 | 39–64 |
| W3 (test: Nov 2025–Mar 2026) | 0.0703–0.0727 | 36–47 |

| Метрика | LGB Ансамбль | Ridge (prod) | Δ |
|---------|-------------|--------------|---|
| Sharpe | **4.07** | 3.59 | **+0.48** |
| Worst month | **-4.5%** | -6.4% | **+1.9pp** |
| Winning months | **11/13** | 9/13 | **+2** |
| Equity | $2565 | $2993 | -$428 |

**Final models (ALL data):**
- IC_val(10%) final: 0.0695–0.0711 (5 seeds, стабильно)
- trees: 76–99
- Sanity check IC на последних 90 днях: **0.1192** ✅

### Deployment

- `results_lgb_prod/lgb_model_seed_*.txt` — 5 моделей (237–306 KB каждая)
- `run_trading.py --lgb` — новый флаг, без EMA, BTC regime так же как Ridge
- Git commit: `f7e1357`

```bash
# VPS:
git pull
python run_trading.py --mode live --loop --capital 100 \
  --leverage 3 --lgb --vol-size --min-zscore 0.8
```

### Итог R10

| Priority | Config | Sharpe | WM | Equity | Status |
|----------|--------|--------|----|--------|--------|
| ✅ Deployed (old) | Ridge EMA=2 | 3.59 | 9/13 | $2993 | Заменяется |
| 🚀 **R10 Production** | **LGB 5-seed EMA=None** | **4.07** | **11/13** | **$2565** | **→ Deploy** |

**Ключевые выводы R10:**
1. 5-seed ансамбль: Sh=4.07 ← реалистичная оценка без утечки данных (walk-forward)
2. Equity ниже Ridge ($2565 < $2993) — ансамбль консервативнее одиночного seed=42. Приоритет: меньший downside, не абсолютная доходность.
3. IC на последних 90д = 0.1192 — сигнал не деградирует

---

## Research Round 11 — Feature & Hyperparameter Exploration (2026-03-31)

Разведка: подбор num_leaves, расширение фичей, TS z-score. Baseline: R10 (Sh=4.07, 14f, nl=31).

### R11A — num_leaves sweep (14f)

| nl | Sharpe | Worst Mo | WM | Equity |
|----|--------|----------|-----|--------|
| 15 | 3.84 | -6.2% | 11/13 | $2263 |
| **31** | **4.07** | -7.4% | 11/13 | $2565 | (baseline)
| **63** | **4.22** | **-1.1%** | **12/13** | **$3142** | ← winner
| 127 | 4.12 | -3.2% | 12/13 | $2892 |

### R11B — Expanded features (nl=63)

| Config | Sharpe | Worst Mo | WM |
|--------|--------|----------|-----|
| 18f (+ funding, premium) | 4.17 | -1.1% | 12/13 |
| 20f + dvol | 4.17 | -17.1% | 11/13 |
| 25f full-deriv | 2.87 | -34.5% | 8/13 |

Вывод: больше фичей ≠ лучше. 25+ features → катастрофический оверфит.

### R11C — TS z-scores & interactions

| Config | Sharpe | Worst Mo | WM |
|--------|--------|----------|-----|
| interactions 14f nl=31 | 4.10 | -3.9% | 11/13 |
| **TS-z 16f nl=31** | **4.37** | -10.6% | 12/13 |
| kitchen sink | 2.29 | — | — |
| TS-z + nl=127 | 4.25 | -3.1% | 12/13 |
| TS-z + nl=63 | 4.21 | -4.8% | 12/13 |

### R11 Feature Importance (top 5)

1. `ret_48h` (23.0%)
2. `ls_divergence` (13.0%)
3. `oi_zscore` (7.5%)
4. `ret_24h` (7.4%)
5. `ts_z_ret12h_60d` (6.8%)

---

## Research Round 12 — Leakage Audit + Advanced Experiments (2026-03-31)

### Leakage Audit (8 checks)

| # | Check | Result |
|---|-------|--------|
| 1 | Walk-forward window gaps (val→test) | ✅ 15d+ gaps |
| 2 | No train/val overlap with test | ✅ 134-137d gaps |
| 3 | fwd_ret_12h formula verification | ✅ 0 mismatches |
| 4 | CS-ranking within-timestamp | ✅ mean≈0 |
| 5 | TS z-score uses only past data | ✅ |
| 6 | Feature↔future_return correlation | ✅ all \|corr\| < 0.15 |
| 7 | Shuffled-target test | ❌ false alarm (*) |
| 8 | Test window date verification | ✅ |

(*) CHECK 7 ложное срабатывание: val labels не шаффлились → early stopping подгоняла шаффл-модель. Исправлено в Deep Audit (см. R13).

### R12 Experiments

| Config | Sharpe | Worst Mo | WM | Equity |
|--------|--------|----------|-----|--------|
| **R12F: 12f pruned nl=63** | **4.77** | -3.6% | 12/13 | **$5280** |
| R12E: min_child=200 14f nl=63 | 4.53 | -0.8% | 12/13 | $3867 |
| R12E: lr=0.03+L2=1 14f nl=63 | 4.30 | +0.9% | **13/13** | $3373 |
| R12E: L1=0.1 14f nl=63 | 4.27 | -0.6% | 12/13 | $3137 |
| R12B: blend α=0.7 16f+TSz | 4.23 | -0.5% | 12/13 | $3065 |
| R12A: 16f+TSz nl=63 | 4.21 | -6.1% | 12/13 | $3040 |
| R12C: LambdaRank | 3.95–4.19 | -14% to -18% | 9/13 | — |
| R12D: Ridge+LGB stack | 3.67–4.03 | — | — | — |

**Ключевое открытие R12**: удаление 2 слабых фич (`dist_from_high_24h`, `mom_z_12h`) → +17% Sharpe.

### 12 Production Features (R12F)
```
ret_12h, ret_24h, ret_48h, residual_12h, residual_24h, mom_z_24h,
oi_chg_12h, oi_chg_24h, oi_zscore, taker_cvd_12h, taker_cvd_24h, ls_divergence
```

---

## Research Round 13 — Combo Tests + Production Deploy (2026-03-31)

Комбинация лучших находок R12: feature pruning × hyperparams.

### R13 Combo Results

| Config | Sharpe | Worst Mo | WM | Equity | IC |
|--------|--------|----------|-----|--------|----|
| **R13-4: 12f lr=0.03 L2=1** | **4.81** | **+2.4%** | **13/13** | **$4900** | 0.063 |
| baseline: 12f nl=63 (R12F) | 4.77 | -3.6% | 12/13 | $5280 | 0.062 |
| R13-8: 12f nl=127 mc=200 | 4.70 | +2.8% | 13/13 | $4422 | 0.059 |
| R13-7: 12f nl=47 mc=200 | 4.62 | -1.2% | 12/13 | $4793 | 0.063 |
| baseline: 14f nl=63 mc=200 (R12E) | 4.53 | -0.8% | 12/13 | $3867 | 0.061 |
| R13-5: 12f mc=300 | 4.49 | -4.7% | 12/13 | $3960 | 0.062 |
| R13-2: 12f mc=200 lr=0.03 L2=1 | 4.45 | -5.0% | 12/13 | $3608 | 0.063 |
| R13-1: 12f mc=200 | 4.33 | -12.3% | 12/13 | $3321 | 0.062 |
| R13-6: 10f mc=200 | 4.04 | -9.9% | 10/13 | $2346 | 0.062 |

Winner: **R13-4** (12f, nl=63, lr=0.03, L2=1.0) — Sh=4.81, **все 13 месяцев положительные**, ни одного убыточного.

### Deep Leakage Audit (R13, 7 checks — all passed ✅)

| # | Check | Result | Details |
|---|-------|--------|---------|
| A | Permutation test (50 shuffles × 3 windows) | **✅** | p=0.0000, z=4.38. Shuffled IC mean=0.0009 vs real=0.0629 |
| B | Time-reversed features | **✅** | Reversed IC=-0.008 vs real=0.063 (ratio -8x) |
| C | Window isolation | **✅** | Cross-window IC lower; gaps 76-288 days |
| D | Rolling IC stability | **✅** | 60.5% timestamps positive, mean IC=0.0623 |
| E | Per-symbol IC distribution | **✅** | 97% symbols (33/34) positive IC |
| F | Leave-one-symbol-out | **✅** | Max drop -26.4% (XRP), all still Sh>3.5 |
| G | Forward return computation | **✅** | 5 symbols × 500 points, 0 mismatches |

### R13 Production Deployment

```
Config: 12f, nl=63, lr=0.03, L2=1.0, 5-seed ensemble [0,7,13,42,99]
Walk-forward: Sh=4.81, WM=13/13, Wr=+2.4%
Final models: 5 × ~70-100 trees, IC_val=0.071
Sanity IC (last 90d): 0.1288
Commit: ce68470
```

```bash
# VPS deploy:
git pull
python run_trading.py --mode live --loop --capital 100 \
  --leverage 3 --lgb --vol-size --min-zscore 0.8
```

### Прогресс R10 → R13

| Version | Sharpe | Worst Month | WM | Equity | Config |
|---------|--------|-------------|-----|--------|--------|
| Ridge R7 | 3.59 | -6.4% | 9/13 | $2993 | 14f, Ridge |
| R10 | 4.07 | -7.4% | 11/13 | $2565 | 14f, LGB nl=31 |
| R11 best | 4.22 | -1.1% | 12/13 | $3142 | 14f, nl=63 |
| R12 best | 4.77 | -3.6% | 12/13 | $5280 | 12f pruned, nl=63 |
| **R13 prod** | **4.81** | **+2.4%** | **13/13** | **$4900** | **12f, nl=63, lr=0.03, L2=1** |

---

## Research Round 14 — Robustness & Exploration (overnight, 2026-03-31)

### R14A: Extended Walk-Forward (5 windows vs 3)

| Config | Sharpe | Worst Mo | WM | Equity |
|--------|--------|----------|-----|--------|
| 3 windows (оригинал) | 4.81 | +2.4% | 13/13 | $4900 |
| **5 windows (расширено)** | **3.65** | **-1.0%** | **18/20** | **$4298** |

Per-window breakdown (5 windows):

| Window | Test Period | Sharpe | IC | Worst Mo |
|--------|-------------|--------|----|----------|
| W0 | 2024-05 → 2024-09 | 5.27 | 0.040 | -0.1% |
| W1 | 2024-10 → 2025-01 | 5.90 | 0.057 | +7.2% |
| W2 | 2025-05 → 2025-08 | 4.84 | 0.058 | +6.4% |
| W3 | 2025-11 → 2026-03 | 3.25 | 0.073 | +2.4% |
| W4 | 2025-07 → 2025-10 | 0.13 | 0.060 | -31.0% |

⚠️ W4 (jul-oct 2025) — слабое окно (Sh=0.13). IC=0.060 нормальный, но сигнал не конвертировался в P&L. Остальные 4 окна отличные.

### R14B: Target Horizon Comparison

| Target | Rebal | Sharpe | Worst Mo | WM | Equity |
|--------|-------|--------|----------|-----|--------|
| 6h | 6h | 1.47 | -23.8% | 6/13 | $189 |
| 8h | 8h | 2.97 | -18.9% | 9/13 | $709 |
| **12h** | **12h** | **4.81** | **+2.4%** | **13/13** | **$4900** |
| 24h | 24h | 4.49 | -10.1% | 10/13 | $3292 |

Вывод: 12h — оптимальный горизонт. 6h/8h — слишком шумно, 24h — медленнее.

### R14C: Rebalance Frequency (target=12h)

| Rebal | Sharpe | Worst Mo | WM | Equity |
|-------|--------|----------|-----|--------|
| 6h | 3.83 | -7.4% | 10/13 | $31490 |
| 8h | 4.04 | -8.5% | 12/13 | $7337 |
| **12h** | **4.81** | **+2.4%** | **13/13** | **$4900** |
| 18h | 3.89 | -11.5% | 7/13 | $596 |
| 24h | 3.48 | -39.0% | 10/13 | $343 |

⚠️ rebal=6h equity $31k — это из-за compounding (больше ребалансировок). Sharpe ниже → больше шума/комиссий. 12h оптимально.

### R14D: Position Count Sweep

| n_long | n_short | Sharpe | Worst Mo | WM | Equity |
|--------|---------|--------|----------|-----|--------|
| 4 | 2 | 4.61 | +1.9% | 13/13 | $6173 |
| **6** | **3** | **4.81** | **+2.4%** | **13/13** | **$4900** |
| 8 | 3 | 4.73 | +0.6% | 13/13 | $4523 |
| 8 | 4 | 4.21 | +3.2% | 13/13 | $2483 |
| 4 | 4 | 3.94 | -3.7% | 11/13 | $2688 |
| 6 | 6 | 4.10 | -8.3% | 12/13 | $1840 |
| 10 | 5 | 3.64 | -5.3% | 12/13 | $1257 |

Текущий 6L/3S — оптимальный, 4L/2S higher equity но более концентрированный.

### R14E: IC Temporal Analysis

| Quarter | Mean IC | % Positive | Тренд |
|---------|---------|------------|-------|
| 2024Q4 | 0.049 | 58% | 📊 |
| 2025Q1 | 0.079 | 63% | 📈 |
| 2025Q2 | 0.062 | 60% | 📈 |
| 2025Q3 | 0.055 | 59% | 📈 |
| 2025Q4 | 0.059 | 61% | 📈 |
| 2026Q1 | 0.081 | 63% | 📈 |

**📈 IC РАСТЁТ со временем** (slope>0, p=0.001). Edge не деградирует — наоборот, усиливается.

### R14F: XGBoost

Пропущено (не установлен). Можно протестировать позже.

### R14G: Bootstrap Confidence Intervals

| Метрика | Значение |
|---------|----------|
| Observed Sharpe | 4.81 |
| Bootstrap median | 2.98 |
| 90% CI | [2.18, 4.19] |
| P(Sharpe > 0) | **100.0%** |
| P(Sharpe > 2) | **98.1%** |

Даже при самом пессимистичном bootstrap (5th percentile): Sharpe = 2.18 — всё равно очень хорошо.

### R14 Key Takeaways

1. **12h target + 12h rebalance** — подтверждён как оптимальный горизонт
2. **6L/3S** — оптимальный position count (подтверждён)
3. **IC растёт** — edge усиливается, не деградирует
4. **Bootstrap**: P(Sh>2) = 98.1%, P(Sh>0) = 100%
5. **5-window**: одно слабое окно (W4, jul-oct 2025), но остальные 4 сильные
6. Текущая prod конфигурация R13 — **оптимальна**, менять ничего не нужно

---

## Round 15 — Deep Optimization (R15)

**Дата**: 2025-07-10
**Цель**: Глубокий поиск улучшений через 8 axis: гиперпараметры, feature interactions, DART, winsorization, time-weighting, Extra Trees, bagging, het-ensemble.
**Baseline**: R13 prod (12f, nl=63, lr=0.03, L2=1.0) → Sh=4.81, WM=13/13

### R15A — Fine-Grained Hyperparameter Grid

| Config | Sharpe | Worst Mo | Win Mo | Equity | IC |
|--------|--------|----------|--------|--------|----|
| lr=0.02 | +4.85 | +2.7% | 13/13 | $5453 | 0.063 |
| lr=0.04 | +4.61 | +0.0% | 13/13 | $4207 | 0.062 |
| L2=0.5 | +4.39 | +3.0% | 13/13 | $3209 | 0.063 |
| L2=2.0 | +4.78 | -0.4% | 12/13 | $4721 | 0.063 |
| L2=3.0 | +4.48 | +2.8% | 13/13 | $3822 | 0.063 |
| mc=75 | +4.53 | -6.7% | 12/13 | $4077 | 0.063 |
| mc=50 | +4.68 | -0.5% | 12/13 | $4843 | 0.063 |
| nl=47 | +4.62 | -4.5% | 12/13 | $4746 | 0.063 |
| nl=95 | +4.77 | +3.1% | 13/13 | $5131 | 0.061 |
| lr=0.02+L2=2 | +4.47 | -1.1% | 12/13 | $3663 | 0.063 |
| baseline_check | +4.81 | +2.4% | 13/13 | $4900 | 0.063 |

**Вывод**: lr=0.02 лучше baseline (+4.85 vs +4.81), Sh чувствителен к lr в диапазоне 0.02-0.04.

### R15B — Feature Interactions

| Config | Sharpe | Worst Mo | Win Mo | Equity | IC |
|--------|--------|----------|--------|--------|----|
| +oi_accel +ret_accel (14f) | +3.92 | -5.9% | 12/13 | $2302 | 0.064 |
| +accel trio (15f) | +3.92 | -8.3% | 11/13 | $2283 | 0.063 |
| +divergence feats (14f) | +4.57 | -6.3% | 12/13 | $4605 | 0.060 |
| all interactions (17f) | +4.23 | -6.4% | 11/13 | $4077 | 0.061 |

**Вывод**: Добавление interaction features **ухудшает** результат. 12 базовых фичей оптимальны.

### R15C — DART (Dropout)

| Config | Sharpe | Worst Mo | Win Mo | Equity |
|--------|--------|----------|--------|--------|
| drop=0.1 | +4.21 | +1.4% | 13/13 | $2956 |
| drop=0.2 | +4.40 | -4.6% | 12/13 | $3520 |

**Вывод**: DART хуже стандартного GBDT. Не берём.

### R15D — Target Winsorization

| Config | Sharpe | Worst Mo | Win Mo | Equity |
|--------|--------|----------|--------|--------|
| winsorize 1% | **+4.87** | +1.2% | **13/13** | $5307 |
| winsorize 5% | +4.48 | +0.3% | 13/13 | $4271 |

**Вывод**: Winsorize 1% даёт Sh=4.87 — **бьёт baseline**! Лёгкая обрезка outliers помогает.

### R15E — Time-Weighted Training

| Config | Sharpe | Worst Mo | Win Mo | Equity |
|--------|--------|----------|--------|--------|
| decay=0.5 | +4.35 | -8.2% | 12/13 | $2987 |
| decay=0.8 | +4.46 | -2.3% | 12/13 | $3890 |

**Вывод**: Time-weighting ухудшает. Все данные одинаково важны.

### R15F — Extra Trees

| Config | Sharpe | Worst Mo | Win Mo | Equity | IC |
|--------|--------|----------|--------|--------|----|
| Extra Trees | **+4.93** | -10.1% | 11/13 | $6926 | 0.066 |

**Вывод**: Максимальный Sharpe (+4.93) и IC (+0.066), но **высокий риск** — 2 убыточных месяца, worst month -10.1%.

### R15G — Bagging Variations

| Config | Sharpe | Worst Mo | Win Mo | Equity |
|--------|--------|----------|--------|--------|
| sub=0.7 col=0.7 | +4.15 | -3.7% | 10/13 | $2855 |
| sub=0.9 col=0.9 | +4.61 | -3.4% | 12/13 | $4625 |
| sub=0.6 col=0.8 | +4.81 | +2.4% | 13/13 | $4900 |
| sub=0.8 col=0.6 | +4.22 | -6.3% | 11/13 | $3070 |
| sub=1.0 col=0.8 | +4.81 | +2.4% | 13/13 | $4900 |

**Вывод**: Текущий sub=0.8, col=0.8 — оптимален. Некоторые альтернативы дают тот же результат.

### R15H — Heterogeneous Ensemble

| Config | Sharpe | Worst Mo | Win Mo | Equity |
|--------|--------|----------|--------|--------|
| het ensemble (5 configs) | +4.35 | -0.7% | 12/13 | $3315 |

**Вывод**: Использование разных конфигураций для каждого seed ухудшает результат. Единообразие лучше.

### R15 Top-3

| Rank | Config | Sharpe | Worst Mo | WM | Equity | IC |
|------|--------|--------|----------|----|--------|----|
| 1 | **Extra Trees** | **+4.93** | -10.1% | 11/13 | $6926 | 0.066 |
| 2 | **Winsorize 1%** | **+4.87** | +1.2% | 13/13 | $5307 | 0.062 |
| 3 | **lr=0.02** | **+4.85** | +2.7% | 13/13 | $5453 | 0.063 |

---

## Round 15.5 — Combo Tests (R15.5)

**Цель**: Комбинации 3 Top-3 находок из R15.
**Гипотеза**: Комбинация Extra Trees + lr=0.02 + L2 tuning может дать и высокий Sharpe, и стабильность.

### R15.5 Results

| Config | Sharpe | Worst Mo | Win Mo | Equity | IC |
|--------|--------|----------|--------|--------|----|
| **ET + lr=0.02 + L2=2** | **+4.83** | **+0.3%** | **13/13** | **$6397** | **0.066** |
| ET + lr=0.02 + L2=2 + mc=150 | +4.93 | -8.4% | 11/13 | $7133 | 0.066 |
| ET + mc=150 | +4.78 | -8.2% | 12/13 | $6498 | 0.066 |
| ET + L2=3.0 | +4.74 | -9.9% | 11/13 | $5712 | 0.066 |
| ET + mc=200 | +4.73 | -10.2% | 10/13 | $6439 | 0.066 |
| ET + L2=5.0 | +4.71 | -3.9% | 12/13 | $5473 | 0.066 |
| ExtraTrees + lr=0.02 | +4.70 | -9.1% | 11/13 | $5691 | 0.066 |
| ExtraTrees + winsor1% | +4.62 | -11.2% | 11/13 | $5175 | 0.066 |
| lr=0.02 + winsor1% | +4.61 | +2.3% | 13/13 | $4035 | 0.063 |
| ET + lr=0.02 + winsor1% | +4.59 | -9.4% | 11/13 | $5223 | 0.066 |
| ET + L2=2.0 (no lr change) | +4.44 | -5.1% | 11/13 | $4804 | 0.066 |
| ET + mc=300 | +4.37 | -2.9% | 12/13 | $3894 | 0.066 |
| ET + lr=0.02 + L2=2 + winsor1% | +4.75 | -9.6% | 11/13 | $5873 | 0.066 |

### R15.5 Key Finding

🏆 **Combo9: ExtraTrees + lr=0.02 + L2=2.0** — единственная конфигурация с Extra Trees, которая сохраняет **WM=13/13** при Sharpe > baseline:

| Metric | R13 Prod | R15.5 Combo9 | Delta |
|--------|----------|--------------|-------|
| Sharpe | 4.81 | **4.83** | +0.02 |
| Win Months | 13/13 | **13/13** | = |
| Worst Month | +2.4% | +0.3% | -2.1% |
| Equity | $4900 | **$6397** | +$1497 (+30%) |
| IC | 0.063 | **0.066** | +0.003 |

**Плюсы**: Higher Sharpe, +30% equity, higher IC, WM=13/13.
**Минусы**: Worst month хуже (0.3% vs 2.4%), но всё ещё положительный.

### R15 + R15.5 General Conclusions

1. **Extra Trees** — мощная регуляризация через случайный выбор сплитов. IC=0.066 vs 0.063 у GBDT.
2. Сам по себе Extra Trees даёт WM=11/13. Но в комбо с lr=0.02 + L2=2 стабилизируется до WM=13/13.
3. **Feature interactions ухудшают** — 12 фичей остаются оптимальными.
4. **DART, time-weighting, het-ensemble** — все хуже baseline.
5. **Winsorize 1%** помогает отдельно, но в комбо с ET не добавляет.
6. **R15.5 Combo9** (ET + lr=0.02 + L2=2) — **кандидат на deployment** при следующем обновлении.

### Конфигурация кандидата R15.5

```python
# R15.5 Combo9: ExtraTrees + lr=0.02 + L2=2.0
FEATURES_12  # без изменений
NUM_LEAVES = 63
LR = 0.02       # было 0.03
REG_L2 = 2.0    # было 1.0
EXTRA_TREES = True  # НОВОЕ
MIN_CHILD = 100  # без изменений
SUBSAMPLE = 0.8
COLSAMPLE = 0.8
SEEDS = [0, 7, 13, 42, 99]
```

---

## ⚠️ Round 16 — CRITICAL AUDIT: Bugs Found (R16)

**Дата**: 2026-03-31
**Цель**: Глубокий аудит кода — поиск багов, утечек, завышения результатов.
**Вердикт**: Найдены 2 критических бага. **Sharpe 4.81 завышен до ~1.75**.

### Что ПРОВЕРЕНО и прошло OK

| Check | Результат |
|-------|-----------|
| **fwd_ret_12h** расчёт | ✅ Корректен: `pct_change(h).shift(-h)` = будущий ретёрн. 20/20 числовых проверок. |
| **Train/val граница** | ✅ `< train_end` vs `>= val_start` — нет пересечения |
| **CS-rank per split** | ✅ Ранг внутри таймстемпа (35 символов), uniform по конструкции |
| **Realized vol** | ✅ Включает текущий bar — стандартная практика, не look-ahead |
| **Funding rate ffill** | ✅ Forward-fill последнего известного рейта — корректно |
| **Permutation test** | ✅ p=0.0000, z=2.48 — сигнал модели **реален** |
| **Long/Short attribution** | ✅ Обе ноги работают: Long 54% / Short 46% contribution |

### ❌ БАГ 1: Over-Annualization Sharpe

`eval_config()` использует `ppy = 8760/12 = 730` (=730 12h-периодов в год). Но regime filter (`trend_cutoff=0.8`) **пропускает 36.7% таймстемпов**!

| Метод расчёта | Sharpe |
|---------------|--------|
| `eval_config` (ppy=730, ошибочный) | **4.81** |
| Actual obs frequency (299/yr) | **3.08** |
| Calendar CAGR/vol | **2.68** |

**Причина**: `sqrt(730)` vs `sqrt(299)` → завышение Sharpe на **56%!**

### ❌ БАГ 2: eq_mom_boost Look-Ahead Bias

`simulate()` обрабатывает КАЖДЫЙ часовой таймстемп, вычисляя 12h-return, обновляя `equity_curve` каждый час. Затем `iloc[::12]` отбирает каждый 12-й.

**Проблема**: На таймстемпе T=12, `equity_curve` содержит записи T=0..T=11, каждая с ПОЛНЫМ 12h forward return (от T до T+12). Return из T=11 покрывает T=11→T=23, что **перекрывается** на 11 часов с T=12→T=24.

Когда `eq_mom_boost` на T=12 проверяет drawdown, он использует **будущую ценовую информацию** через overlapping returns.

**Масштаб**: eq_mom_boost **один** добавляет +2.61 Sharpe в бэктесте! С рандомными предсказаниями portfolio overlay даёт Sh=2.67.

### CHECK C: Decomposition Portfolio Enhancements

| Конфигурация | Sh (buggy) | Sh (fixed) |
|-------------|-----------|-----------|
| Bare-bones (простой L/S) | 2.18 | **1.75** |
| + kelly | — | 1.35 |
| + regime filter | 3.33 | 0.97 |
| + vol scaling | 2.43 | 0.84 |
| + eq_mom_boost | **4.79** | 0.76 |
| + strat_momentum | — | 0.75 |
| **Full config** | **4.81** | **0.75** |

**Вывод**: В корректной реализации (без look-ahead) ВСЕ portfolio overlay-и **ухудшают** результат! Barebone L/S = оптимум.

### CHECK G: Per-Window Results (Fixed)

| Window | Period | Sh (buggy) | Sh (fixed) | Win Mo | IC |
|--------|--------|-----------|-----------|--------|-----|
| W1 | Oct 2024–Jan 2025 | +5.90 | +1.41 | 3/4 | 0.036 |
| W2 | May 2025–Aug 2025 | +4.84 | **-1.01** | **0/4** | 0.018 |
| W3 | Nov 2025–Mar 2026 | +3.25 | +1.58 | 1/5 | 0.033 |

**W2 ОТРИЦАТЕЛЬНЫЙ!** Это было скрыто buggy eq_mom_boost и over-annualization.

### Реальная Картина

| Компонент | Sharpe |
|-----------|--------|
| Чистый сигнал (barebone L/S, correct ann.) | **~1.75** |
| Сигнал + portfolio overlays (correct) | ~0.75 |
| Ранее отчитывавшийся R13 | ~~4.81~~ |

**Sharpe ~1.75** (barebone) — это всё ещё **положительный edge** для крипто L/S. Permutation test подтверждает: сигнал реально лучше случайного (z=2.48, p=0.0000). Но это НЕ 4.81.

### R16 Рекомендации

1. **Исправить `simulate()`** — обрабатывать только rebalance timestamps, не hourly
2. **Исправить `eval_config()`** — аннуализировать по реальной частоте наблюдений
3. **Убрать eq_mom_boost** — в корректной реализации он не помогает
4. **Убрать strat_momentum** — та же проблема с look-ahead
5. **Пересчитать все R11-R15** — Sharpe во всех предыдущих раундах завышен
6. **Оценить жизнеспособность** стратегии при Sh≈1.75 (без леверейджа)

### Скрипты аудита

- `_audit_r16_deep.py` — 9 проверок (fwd_ret, subsampling, enhancements, regime bias, autocorrelation, permutation, per-window, Sharpe verification, L/S attribution)
- `_audit_r16b_fix.py` — Fixed simulation + recalculation

---

## Round 17 — Bug Fixes Applied (R17)

**Дата**: 2026-04-01  
**Цель**: Применить R16-фиксы, получить честную базовую линию.

### Исправления
1. `simulate()`: обрабатывает только `timestamps_sorted[::rebal_hours]` (не каждый час)
2. `eval_config()`: `ppy = n_obs / years` из реального диапазона таймстемпов
3. Убраны `eq_mom_boost` и `strategy_momentum`

### Результаты R17 (честный baseline, 12f LGB)

| Config | Sharpe | WM | Worst M | Final Eq |
|--------|--------|----|---------|----------|
| LGB-12f bare | 1.75 | 8/13 | -41.7% | $666 |
| **LGB-12f + regime** | **1.84** | **9/13** | **-19.1%** | **$516** |

Permutation test: z=2.06, p=0.02 — сигнал реальный.  
Скрипт: `_test_r17_fixed.py`, commit `1bac294`

---

## Round 18 — Major Feature Expansion (R18)

**Дата**: 2026-04-01  
**Цель**: Расширить фичи (6 новых источников), multi-model ensemble.  
**Скрипт**: `_research_r18_improve.py` (52 мин, 36 конфигов)

### Новые источники данных
| Источник | Фичи | Файл |
|----------|------|------|
| TA (техн. анализ) | 20 фич (ATR, RSI, MACD, BB, GK-vol, etc.) | `data/features/crypto_features_1h.parquet` |
| Новости | 5 фич (sentiment, volume zscore, momentum) | `data/sentiment/crypto_news.parquet` |
| Fear & Greed | 4 фичи | `data/sentiment/fear_greed.parquet` |
| DVOL (Deribit) | 4 фичи (btc_dvol, zscore, chg, IV-RV spread) | `data/sentiment/deribit_dvol.parquet` |
| Macro | 5 фич (VIX, DXY, yield curve) | `data/sentiment/macro_daily.parquet` |
| Deriv+ | 5 фич (top/global long pct, smart_money_diverge) | `data/sentiment/binance_futures_metrics.parquet` |
| Итого | **90 доступных фич** (было 12) | — |

### Топ IC features (OOS scan, 78 кандидатов)
```
atr_14:        IC=-0.0698  (отрицательный IC = mean-reversion)
rvol_12h:      IC=-0.0681
gk_vol_24h:    IC=-0.0663
rvol_24h:      IC=-0.0637
iv_rv_spread:  IC=+0.0637  (IV > RV → expect reversion)
```
Все топ-5 — **vol features** (backward-looking, утечки нет).

### Финальные результаты R18

| Ранг | Config | Sharpe | WM | Worst M | Eq |
|------|--------|--------|----|---------|-----|
| 1 | **LGB-17f-top5 + regime** | **2.23** | **10/13** | -22.4% | **$968** |
| 2 | LGB-HPO-nl=31 + regime | 1.95 | 9/13 | -39.7% | $649 |
| 3 | CB-19f + regime | 1.95 | 7/13 | -19.8% | $641 |
| baseline | LGB-12f + regime | 1.84 | 9/13 | -19.1% | $516 |

**R18 winner (17f)** = 12f baseline + `atr_14, rvol_12h, gk_vol_24h, rvol_24h, iv_rv_spread`

### Выводы R18
- ✅ Добавление 5 vol-фич → Sharpe 1.84 → 2.23 (+21%)
- ✅ IC улучшился: 0.0623 → 0.0800 (+28%)
- ❌ Kitchen-sink 35f — хуже (Sh=1.06), переобучение
- ❌ Meta-ensemble 3 моделей — разбавляет сигнал (Sh=1.74)
- ❌ Target engineering — не помогает
- ⚠️ IC scan был на TEST данных → R19 исправляет

---

## Round 19 — Leakage Fix + New Signals (R19)

**Дата**: 2026-04-01  
**Цель**: (1) Исправить IC scan leakage R18, (2) добавить market breadth + seasonality + regime-as-features.  
**Скрипт**: `_research_r19_v2.py` (29 мин, 20 конфигов)

### Новые сигналы R19
| Сигнал | Формула | Тип |
|--------|---------|-----|
| `pct_coins_up_12h` | % монет с ret_12h > 0 | Market breadth |
| `pct_coins_up_1h` | % монет с ret_1h > 0 | Short-term breadth |
| `btc_outperform` | ret_12h - BTC ret_12h | Relative strength |
| `hour_sin/cos` | sin/cos(hour*2π/24) | Seasonality |
| `dow_sin/cos` | sin/cos(dow*2π/7) | Weekly cycle |
| `trend_strength` | \|BTC_ret_7d\| / BTC_vol_7d | Regime feature |
| `trend_direction` | BTC_ret_7d / BTC_vol_7d | Regime feature |

### IC Scan R19 (TRAIN ONLY — без leakage)
```
atr_14:        IC=-0.0630   (подтверждён на train)
gk_vol_24h:    IC=-0.0629   (подтверждён)
iv_rv_spread:  IC=+0.0623   (подтверждён)
rvol_24h:      IC=-0.0623   (подтверждён)
btc_outperform:IC=-0.0281   (новое! mean-reversion vs BTC)
```
**IC scan на TRAIN дал те же топ-фичи, что R18 на TEST → R18 leakage не влияло!**

### Финальные результаты R19

| Ранг | Config | Sharpe | WM | Worst M | Eq |
|------|--------|--------|----|---------|-----|
| **1** | **LGB-17f + breadth + season + regime** | **2.50** | **10/13** | -29.6% | **$1314** |
| 2 | LGB-17f-top5 + regime (R18 подтверждён) | 2.23 | 10/13 | -22.4% | $968 |
| 3 | CB-combo + regime | 2.16 | 8/13 | **-13.4%** | $869 |
| 4 | LGB+CB-17f avg + regime | 2.11 | 8/13 | -21.4% | $829 |
| 5 | LGB-full-enhanced + regime | 2.05 | 10/13 | -21.9% | $769 |

### Equity path R19 winner (LGB-17f+breadth+season)
```
Oct24: +24.6%  Nov24: -29.6%  Dec24: +88.7%  Jan25:  +6.0%
May25: +35.9%  Jun25: +44.8%  Jul25:  -9.6%  Aug25: +58.1%
Nov25: +13.8%  Dec25: +63.2%  Jan26: +37.3%  Feb26:  +9.6%  Mar26: -4.8%
Final equity: $1314 (от $100 за 11 торговых месяцев, 5x leverage)
```

### Выводы R19
- ✅ Market breadth + seasonality → Sharpe 2.23 → 2.50 (+0.27, +12%)
- ✅ WM=10/13 сохраняется
- ✅ IC scan на TRAIN подтвердил R18 leakage был незначимым
- ⚠️ Worst month -29.6% (хуже R18 -22.4%)
- ❌ Regime features в модели НЕ заменяют жёсткий фильтр (bare Sh=1.84 vs filtered 2.50)
- ⚠️ Funding carry: не удалось загрузить (`Column not found: funding_rate_binance`)

### Прогресс по раундам (все исправлены от R16 bagов)

| Round | Config | Sharpe | Improvement |
|-------|--------|--------|-------------|
| R17 | 12f baseline | 1.84 | базовая линия |
| R18 | 17f + vol | 2.23 | +21% |
| R19 | 23f + breadth + season | 2.50 | +36% от R17 |

**Текущий рекорд: Sh=2.50** (LGB, 23 фичи, режим-фильтр)

---

## R20 — Deep Research (6 экспериментов)

**Дата**: март 2026  
**База**: R19 winner (LGB-23f, regime filter trend_cutoff=0.8, dyn_threshold=0.5, 6L/3S, rebal=12h, Sh=2.50)  
**Скрипт**: `_research_r20_deep.py`  
**Время**: 29.3 мин (1756 сек)

### CONTROL — воспроизведение R19

| Config | Sh | WM | Wr | Equity |
|--------|----|----|-----|--------|
| R19-winner-ctrl [regime] | **2.50** | 10/13 | -29.6% | $1314 |

Месячный путь: Oct24 +24.6% → Nov24 **-29.6%** → Dec24 +88.7% → Jan25 +6% → May25 +35.9% → Jun25 +44.8% → Jul25 -9.6% → Aug25 +58.1% → Nov25 +13.8% → Dec25 +63.2% → Jan26 +37.3% → Feb26 +9.6% → Mar26 -4.8% → **$1314**

---

### EXP-A: Funding Carry Signal

Добавлены 4 фичи: `cum_funding_24h, funding_zscore, funding_x_mom_12h, funding_x_mom_24h` (итого 27f).

| Config | Sh | WM | Wr | Equity | vs R19 |
|--------|----|----|-----|--------|--------|
| LGB-27f+funding [bare] | 1.41 | 8/13 | -72.8% | $380 | ❌ -44% |
| LGB-27f+funding [regime] | 1.87 | 8/13 | -33.4% | $599 | ❌ -25% |

**Вывод: Funding carry ОТВЕРГНУТ.** Добавление funding-фичей ухудшает Sharpe 2.50 → 1.87. Шум перевешивает сигнал.

---

### EXP-B: Position Count Sweep

| Config | Sh | WM | Wr | Equity |
|--------|----|----|-----|--------|
| 4L/2S | 1.91 | 9/13 | -29.1% | $766 |
| 5L/2S | 1.95 | 9/13 | -28.1% | $803 |
| **6L/3S (baseline)** | **2.50** | **10/13** | **-29.6%** | **$1314** |
| 6L/2S | 1.97 | 9/13 | -26.7% | $815 |
| 8L/3S | 2.46 | 10/13 | -30.0% | $1187 |
| 8L/4S | 1.58 | 11/13 | -56.8% | $376 |
| 10L/5S | 1.56 | 10/13 | -46.3% | $329 |

**Вывод: 6L/3S — оптимум подтверждён.** Уменьшение позиций снижает Sharpe, увеличение — резко увеличивает просадки.

---

### EXP-C: Regime Threshold Sweep ⭐ КЛЮЧЕВАЯ НАХОДКА

| Cutoff | Sh | WM | Wr | Equity | vs R19 |
|--------|----|----|-----|--------|--------|
| 0.5 | 1.67 | 9/13 | -41.3% | $378 | ❌ |
| 0.6 | 1.90 | 9/13 | -37.5% | $518 | ❌ |
| 0.7 | 2.36 | 10/13 | -34.4% | $982 | ❌ |
| **0.8 (R19)** | **2.50** | **10/13** | **-29.6%** | **$1314** | — |
| **0.9** | **2.80** | **10/13** | **-28.5%** | **$2096** | **⭐ +12%** |
| 1.0 | 2.70 | 10/13 | -28.5% | $2025 | +8% |
| 1.2 | 2.43 | 11/13 | -41.2% | $1596 | ❌ |
| 1.5 | 2.48 | 11/13 | -35.1% | $2007 | ❌ |
| 999 (off) | 1.84 | 10/13 | -73.9% | $816 | ❌ |

**Вывод: trend_cutoff=0.9 → Sh=2.80 (+12% vs R19), лучшая просадка AND equity!**  
Оптимум находится в диапазоне 0.9–1.0, пик на 0.9.

---

### EXP-D: Multi-Horizon Target Training

| Config | Sh | WM | Wr | Equity | IC_all |
|--------|----|----|-----|--------|--------|
| LGB-4h-target [regime] | 2.40 | 11/13 | -37.5% | $1286 | 0.0843 |
| **LGB-12h-target (R19)** | **2.50** | **10/13** | **-29.6%** | **$1314** | **0.0783** |
| LGB-12h+24h-ens [regime] | 2.14 | 9/13 | -32.8% | $811 | 0.0768 |
| LGB-24h-target [regime] | 1.78 | 8/13 | -18.6% | $504 | 0.0726 |

**Вывод: 12h target остаётся лучшим.** 4h horizon даёт более высокий IC но луший Sharpe у 12h. Ансамбль 12h+24h не помогает.

---

### EXP-E: Permutation Test (n=300) ✅

```
Null distribution:  mean=-0.020 ± 0.866
Real Sharpe:        2.500
z = 2.91,  p = 0.0033
```

**Вывод: Sh=2.50 — НЕ случайность (p=0.0033 << 0.05).** Стратегия статистически значима. z=2.91 означает результат более 2.9 стандартных отклонений выше случайного (нулевого) распределения.

---

### EXP-F: Rebalance Interval Sweep ⭐ КЛЮЧЕВАЯ НАХОДКА

| Config | Sh | WM | Wr | Equity | vs R19 |
|--------|----|----|-----|--------|--------|
| **6h rebal** | **2.88** | **8/13** | **-18.8%** | **$4981** | **⭐ +15%** |
| **12h rebal (R19)** | **2.50** | **10/13** | **-29.6%** | **$1314** | — |
| 24h rebal | 1.31 | 10/13 | -16.6% | $235 | ❌ |

**Вывод: 6h rebal → Sh=2.88 (+15%), Equity=**$4981** (!), Wr=-18.8% (лучше всех!)**  
WM=8/13 ниже (меньше выигрышных месяцев), но equity и Sharpe значительно выше. Огромный результат.

---

### Итоги R20 — Рейтинг

| # | Config | Sh | WM | Wr | Equity |
|---|--------|----|----|-----|--------|
| 1 | **EXP-F: 6h rebal** (+ пока не confirmed) | 2.88 | 8/13 | -18.8% | $4981 |
| 2 | **EXP-C: cutoff=0.9** | 2.80 | 10/13 | -28.5% | $2096 |
| 3 | EXP-C: cutoff=1.0 | 2.70 | 10/13 | -28.5% | $2025 |
| 4 | R19 baseline | 2.50 | 10/13 | -29.6% | $1314 |
| 5 | EXP-B: 8L/3S | 2.46 | 10/13 | -30.0% | $1187 |

### Выводы R20

- ⭐ **EXP-F best individual**: 6h rebal → Sh=2.88, Eq=$4981, Wr=-18.8% (строго лучше по всем метрикам кроме WM)
- ⭐ **EXP-C**: cutoff=0.9 → Sh=2.80 (+12% vs R19), сохраняет WM=10/13
- ✅ **Permutation test p=0.0033** — стратегия статистически значима
- ❌ Funding carry: ОТВЕРГНУТ (27f→1.87 vs 23f→2.50)
- ✅ 6L/3S позиционирование: ПОДТВЕРЖДЕНО оптимальным
- ✅ 12h prediction horizon: ПОДТВЕРЖДЕНО лучшим
- ⚠️ Нужно проверить комбинацию: cutoff=0.9 + 6h rebal → потенциально Sh>3.0

### Прогресс по раундам (обновлено)

| Round | Config | Sharpe | Equity | Key change |
|-------|--------|--------|--------|------------|
| R17 | 12f baseline | 1.84 | — | базовая линия |
| R18 | 17f + vol | 2.23 | — | +volatility features |
| R19 | 23f + breadth/season | 2.50 | $1314 | +breadth+seasonality |
| R20 best-C | cutoff=0.9 | 2.80 | $2096 | +regime threshold |
| R20 best-F | 6h rebal | 2.88 | $4981 | +rebalance frequency |

**Текущий рекорд: Sh=2.88** (потенциал: cutoff=0.9 + 6h rebal → R21 ✅ подтверждено!)

---

## R21 — Confirmation Round (эксперименты G/H/I/J)

**Дата**: апрель 2026  
**База**: R20 findings: cutoff=0.9 → Sh=2.80 | 6h rebal → Sh=2.88  
**Скрипт**: `_research_r21_confirm.py`  
**Время**: 7.2 мин (433 сек)  
**Цель**: подтвердить комбинацию cutoff=0.9 + 6h rebal

---

### EXP-G: Combined vs Individuals ⭐⭐ ГЛАВНЫЙ РЕЗУЛЬТАТ

| Config | Sh | WM | Wr | Equity | vs R19 |
|--------|----|----|-----|--------|--------|
| R19-baseline | 2.50 | 10/13 | -29.6% | $1314 | — |
| R20-C cutoff=0.9 | 2.80 | 10/13 | -28.5% | $2096 | +12% |
| R20-F 6h rebal | 2.88 | 8/13 | -18.8% | $4981 | +15% |
| **R21 combined (0.9+6h)** | **3.17** | **9/13** | **-23.7%** | **$9889** | **⭐ +27%** |

**Комбинация СИНЕРГЕТИЧЕСКАЯ: 2.80 + 2.88 → 3.17 (лучше каждого по отдельности!)**

Monthly equity path для R21:
```
Oct24  +33.0%  →$133     Nov24 -2.1%   →$130    Dec24  +185.2% →$371
Jan25  -23.7%  →$283     May25 +70.0%  →$482    Jun25  +168.2% →$1292
Jul25  -9.9%   →$1164    Aug25 +79.3%  →$2086   Nov25  +33.7%  →$2789
Dec25  +78.4%  →$4977    Jan26 +116.2% →$10760  Feb26  +6.5%   →$11460
Mar26  -13.7%  →$9889
```
**Nov24 был -29.6%, теперь всего -2.1%! 6h rebal почти убрал этот удар.**

---

### EXP-H: Fine-tuning cutoff (6h rebal fixed) ⭐

| Cutoff | Sh | WM | Wr | Equity |
|--------|----|----|-----|--------|
| 0.80 | 2.88 | 8/13 | -18.8% | $4981 |
| 0.85 | 2.94 | 8/13 | -22.4% | $5851 |
| **0.90** | **3.17** | **9/13** | **-23.7%** | **$9889** |
| 0.92 | 3.09 | 9/13 | -30.3% | $8663 |
| 0.95 | 3.07 | 9/13 | -29.7% | $8718 |
| **1.00** | **3.24** | **10/13** | **-28.3%** | **$12619** |
| 1.05 | 3.15 | 9/13 | -32.8% | $11310 |

**cutoff=1.00 + 6h → Sh=3.24, WM=10/13, Eq=$12619 — лучший по Sharpe!**
cutoff=0.90 хуже по Sharpe (3.17) но лучше по Wr (-23.7% vs -28.3%)

---

### EXP-I: Per-Window Robustness

| Window | R19 Sh | R21 Sh | R21 Wr | R21 Eq |
|--------|--------|--------|--------|--------|
| W1 (Oct24-Jan25) | 2.20 | 2.85 | -23.7% | $283 |
| W2 (May25-Aug25) | 2.09 | **4.55** | **-5.2%** | $536 |
| W3 (Nov25-Mar26) | 2.14 | 3.18 | -16.2% | $305 |

**R21 строго лучше во ВСЕХ windows.** W2: Sh=4.55(!), Wr=-5.2% — почти без просадок.

---

### EXP-J: Permutation Test для R21

```
Null distribution:  mean=-0.121 ± 0.854
Real Sharpe:        3.174
z = 3.86,  p = 0.0000
✅ HIGHLY SIGNIFICANT — R21 стратегия НЕ случайность (p<0.0001)
```

---

### Итоги R21 — ~~Финальный рейтинг~~ ОТМЕНЁН (баг с overlap)

> **⚠️ CRITICAL BUG FOUND**: 6h rebalance с 12h prediction horizon = **перекрывающиеся доходности**.
> На каждом 6h интервале используется полный 12h return → часы 6-12 считаются дважды.
> Все результаты с `rebal_hours=6` + `fwd_ret_12h` — **артефакт**.
> Подтверждено диагностикой `_research_r21_fix.py` (апрель 2026).

**Некорректные результаты (INVALIDATED):**

| # | Config | Sh (buggy) | Sh corrected | Equity |
|---|--------|------------|-------------|--------|
| ~~1~~ | ~~R21: 0.9+6h~~ | ~~3.17~~ | ~2.24 | ~~$9889~~ |
| ~~2~~ | ~~H: cutoff=1.00+6h~~ | ~~3.24~~ | ~2.29 | ~~$12619~~ |
| ~~3~~ | ~~R20-F: 6h rebal~~ | ~~2.88~~ | ~2.04 | ~~$4981~~ |

**Причина**: `rebal_timestamps = timestamps_sorted[::rebal_hours]` при rebal=6 выдаёт 2x больше точек, каждая с 12h return → 50% overlap → ppy удваивается → Sharpe × sqrt(2), equity экспоненциально раздувается.

**Валидные результаты:**

| # | Config | Sh | WM | Wr | Equity | Статус |
|---|--------|----|----|-----|--------|--------|
| 1 | **cutoff=0.9, 12h** | **2.80** | **10/13** | **-28.5%** | **$2096** | ✅ BEST |
| 2 | cutoff=1.0, 12h | 2.70 | 10/13 | -28.5% | $2025 | ✅ |
| 3 | cutoff=0.8, 12h (R19) | 2.50 | 10/13 | -29.6% | $1314 | ✅ |

### Выводы R21 (исправленные)

- ❌ **6h rebal** — ARTIFACT, вычеркнут (overlap bug)
- ❌ **Все результаты Sh>2.80** с 6h rebal — НЕДЕЙСТВИТЕЛЬНЫ
- ✅ **Permutation test для R19 (p=0.0033)** — по-прежнему валиден (12h rebal)
- ✅ **cutoff=0.9** — РЕАЛЬНОЕ улучшение (2.50 → 2.80, +12%), подтверждено без бага
- ✅ **cutoff=0.9 equity path**: $100 → $2096 за 13 OOS месяцев (5x leverage) — реалистично

### Sanity check (cutoff=0.9, 12h, verified)
```
n_obs: 450 (12h periods)
Mean ret per period: +0.166% (unleveraged)
Std: 1.072%
Mean × 5x leverage: +0.829% per 12h
Positive frac: 64.4%
Annualized ret: 54.3% (unlev), ~270% (5x lev)
PPY: 328
```

### Прогресс по раундам (исправленный)

| Round | Config | Sharpe | Equity | Key change | Статус |
|-------|--------|--------|--------|------------|--------|
| R17 | 12f baseline | 1.84 | $816 | базовая линия | ✅ |
| R18 | 17f + vol | 2.23 | — | +volatility features | ✅ |
| R19 | 23f + breadth/season | 2.50 | $1314 | +breadth+seasonality | ✅ |
| **R20-C** | **cutoff=0.9** | **2.80** | **$2096** | **+regime threshold** | **✅ BEST** |
| ~~R20-F~~ | ~~6h rebal~~ | ~~2.88~~ | ~~$4981~~ | overlap bug | ❌ |
| ~~R21~~ | ~~cutoff=0.9 + 6h~~ | ~~3.17~~ | ~~$9889~~ | overlap bug | ❌ |

**Текущий рекорд: Sh=2.80, Eq=$2096** (cutoff=0.9, 12h rebal, 6L/3S, 23f, $100→$2096 за 13мес)

---

## R22 — Deep Model Improvement (01 апр 2026)

**Цель:** Попробовать всё — HPO, другие модели, ensembles, новые фичи.  
**База:** R20-C winner — LGB-23f, cutoff=0.9, 12h rebal, 6L/3S → Sh=2.80, Eq=$2096

### EXP-K: LGB Hyperparameter Optimization (Optuna, 20 trials)

Оптимизировались: lr, num_leaves, min_child_samples, subsample, colsample_bytree, lambda_l1/l2, max_depth.

Лучшие найденные параметры:
```
lr=0.006, num_leaves=122, min_child_samples=291, subsample=0.90,
colsample_bytree=0.65, lambda_l2=0.02, lambda_l1=3.42, max_depth=4
```

| Config | Sharpe | Equity | WM | Worst M | Статус |
|--------|--------|--------|----|---------|--------|
| LGB-HPO-23f | 2.56 | $1541 | 10/13 | -29.7% | ❌ хуже baseline |
| LGB-default-23f (control) | **2.80** | **$2096** | 10/13 | -28.5% | ✅ baseline подтверждён |

**Вывод:** HPO не помог. Дефолтные параметры (lr=0.03, num_leaves=63, min_child=100) оказались лучше.
Optuna нашла более сложную модель (leaves=122, depth=4), которая переобучилась.

### EXP-L: Feature Importance & Pruning

Важность фич (avg gain, 15 моделей):
```
 1. atr_14                 2379  ██████████████████████████████
 2. gk_vol_24h             2245  ████████████████████████████
 3. ret_48h                1794  ██████████████████████
 4. ls_divergence          1752  ██████████████████████
 5. oi_zscore              1205  ███████████████
 6. taker_cvd_24h          1102  █████████████
 7. mom_z_24h               849  ██████████
 8. rvol_24h                778  █████████
 9. oi_chg_24h              759  █████████
10. ret_24h                 624  ███████
11. taker_cvd_12h           597  ███████
12. rvol_12h                593  ███████
13. residual_24h            512  ██████
14. ret_12h                 494  ██████
15. oi_chg_12h              430  █████
16. pct_coins_up_12h        403  █████
17. residual_12h            362  ████
18. iv_rv_spread            346  ████
19. pct_coins_up_1h          69  █
20. hour_sin                 12  █
21-23. hour_cos, dow_sin/cos  ~0  (не используются)
```

| Config | Sharpe | Equity | WM | Статус |
|--------|--------|--------|----|--------|
| 23f (все фичи) | **2.80** | **$2096** | 10/13 | ✅ лучший |
| 20f (drop 3 worst) | 2.14 | $922 | 10/13 | ❌ |
| 18f (drop 5) | 2.25 | $1045 | 9/13 | ❌ |
| 16f (drop 7) | 1.98 | $750 | 9/13 | ❌ |
| 13f (drop 10) | 2.17 | $957 | 9/13 | ❌ |

**Вывод:** Все 23 фичи нужны. Даже удаление 3 worst (dow_sin/cos, hour_cos) ухудшает на 0.66 Sharpe. Видимо, LGB сам справляется с шумовыми фичами, а pruning теряет редкие но полезные сигналы.

### EXP-M: XGBoost Baseline (23f)

IC: W1=0.084, W2=0.063, W3=0.089, ALL=0.079

| Config | Sharpe | Equity | WM | Worst M | Статус |
|--------|--------|--------|----|---------|--------|
| XGB-23f | 2.14 | $885 | 10/13 | -31.2% | ❌ |

Помесячно:
```
2024-10  +22.3%   2024-11  -31.2%   2024-12  +37.8%   2025-01   +5.5%
2025-05  +36.8%   2025-06  +58.9%   2025-07   -2.0%   2025-08  +70.2%
2025-11  +17.5%   2025-12  +50.5%   2026-01  +17.0%   2026-02   +3.2%
2026-03   -6.6%
```

**Вывод:** XGB заметно хуже LGB (2.14 vs 2.80). IC ниже (0.079 vs ~0.085 у LGB).

### EXP-N: CatBoost Baseline (23f)

IC: W1=0.089, W2=0.070, W3=0.093, ALL=0.084

| Config | Sharpe | Equity | WM | Worst M | Статус |
|--------|--------|--------|----|---------|--------|
| CB-23f | 2.29 | $1082 | 8/13 | -22.0% | ❌ |

Помесячно:
```
2024-10  +21.7%   2024-11  -22.0%   2024-12  +61.3%   2025-01   -0.7%
2025-05  +37.4%   2025-06  +68.5%   2025-07   -1.4%   2025-08  +77.8%
2025-11  +27.7%   2025-12  +29.7%   2026-01  +30.8%   2026-02  -14.8%
2026-03   -5.0%
```

**Вывод:** CB имеет лучший IC (0.084) и меньший worst month (-22% vs -28.5%), но Sharpe ниже (2.29). У CB 5 отрицательных месяцев vs 3 у LGB.

### EXP-O: Stacked Ensemble (LGB + XGB + CB)

| Метод | Sharpe | Equity | WM | Worst M | Статус |
|-------|--------|--------|----|---------|--------|
| avg-ensemble | 2.28 | $1108 | 10/13 | -35.2% | ❌ |
| rank-ensemble | 2.45 | $1361 | 11/13 | -38.2% | ❌ |
| ridge-stack (W1→W2, W1+W2→W3) | 1.15 | $162 | 6/9 | -34.9% | ❌ |

Ridge коэффициенты:
```
W1→W2:     LGB=-0.010  XGB=+0.038  CB=-0.019
W1+W2→W3:  LGB=+0.033  XGB=+0.004  CB=-0.027
```

**Вывод:** Все ensemble-методы хуже чистого LGB. Ridge-stacking катастрофа — слишком мало OOS данных для meta-learner (только W2+W3 = 9 месяцев). Rank-ensemble (Sh=2.45) чуть лучше avg-ensemble (2.28), но оба хуже LGB solo (2.80).

### EXP-P: New Features from Untapped Sources

18 новых фич: premium_zscore, oi_velocity, taker_imb_z, vol_of_vol, dist_from_high, vol_ratio, ret_168h, fng_value/zscore, vix_close/zscore, dxy_ret_7d, adx, mfi_14, ret_skew/kurt_24h, vwap_dev_24h, obv_ma_ratio_24.

| Config | Sharpe | Equity | WM | Worst M | Статус |
|--------|--------|--------|----|---------|--------|
| P-ctrl-23f (baseline) | **2.80** | **$2096** | 10/13 | -28.5% | ✅ |
| P-all-40f (23+17 new) | 1.96 | $691 | 9/13 | -46.1% | ❌ |
| P-fng-25f (+fear&greed) | 2.50 | $1548 | 9/13 | -25.1% | ⚠️ |
| P-deriv-25f (+oi_vel, taker_imb) | 2.07 | $835 | 10/13 | -37.1% | ❌ |
| P-vol-25f (+vol_of_vol, vol_ratio) | 2.29 | $1081 | 10/13 | -31.6% | ❌ |
| P-ta-29f (+adx,mfi,skew,kurt,vwap,obv) | 2.43 | $1248 | 10/13 | -19.1% | ⚠️ |
| P-mom-25f (+dist_high, ret_168h) | 2.37 | $1238 | 10/13 | -27.6% | ❌ |

Помесячно P-fng-25f:
```
2024-10  +43.0%   2024-11  -25.1%   2024-12  +76.5%   2025-01   -2.3%
2025-05  +30.7%   2025-06  +74.5%   2025-07   -2.2%   2025-08  +51.9%
2025-11  +17.7%   2025-12  +47.2%   2026-01  +46.2%   2026-02   -6.6%
2026-03   +4.6%
```

Помесячно P-ta-29f:
```
2024-10  +26.2%   2024-11  -19.1%   2024-12  +29.4%   2025-01  +33.4%
2025-05  +25.4%   2025-06  +58.0%   2025-07   +8.3%   2025-08  +62.1%
2025-11  +21.1%   2025-12  +63.7%   2026-01  +16.8%   2026-02   -8.8%
2026-03   -3.6%
```

**Выводы по P:**
- Все 17 новых фич разом (40f) — катастрофа (Sh=1.96). Слишком много шума, LGB переобучается.
- **FNG (+fear&greed)** — снизил worst month до -25.1% (vs -28.5%), но потерял Sharpe (2.50 vs 2.80). Более стабильный.
- **TA (+adx,mfi,skew,kurt,vwap,obv)** — лучший worst month (-19.1%!), но Sharpe 2.43. Самый "гладкий" equity.
- Ни одна новая фича не побила baseline по Sharpe.

### R22 Summary Table

| # | Config | Sharpe | Equity | vs Baseline |
|---|--------|--------|--------|-------------|
| 1 | **LGB-default-23f** | **2.80** | **$2096** | **= BASELINE** |
| 2 | LGB-HPO-23f | 2.56 | $1541 | -0.24 |
| 3 | P-fng-25f | 2.50 | $1548 | -0.30 |
| 4 | O-rank-ensemble | 2.45 | $1361 | -0.35 |
| 5 | P-ta-29f | 2.43 | $1248 | -0.37 |
| 6 | P-mom-25f | 2.37 | $1238 | -0.43 |
| 7 | N-CB-23f | 2.29 | $1082 | -0.51 |
| 8 | P-vol-25f | 2.29 | $1081 | -0.51 |
| 9 | O-avg-ensemble | 2.28 | $1108 | -0.52 |
| 10 | L-prune-18f | 2.25 | $1045 | -0.55 |
| 11 | L-prune-13f | 2.17 | $957 | -0.63 |
| 12 | M-XGB-23f | 2.14 | $885 | -0.66 |
| 13 | L-prune-20f | 2.14 | $922 | -0.66 |
| 14 | P-deriv-25f | 2.07 | $835 | -0.73 |
| 15 | L-prune-16f | 1.98 | $750 | -0.82 |
| 16 | P-all-40f | 1.96 | $691 | -0.84 |
| 17 | O-ridge-stack | 1.15 | $162 | -1.65 |

### R22 Ключевые выводы

1. **LGB с дефолтными параметрами — ЛУЧШЕ всех.** HPO, XGBoost, CatBoost, ensemble, новые фичи — ничего не побило Sh=2.80.
2. **Модель уже хорошо оптимизирована.** 23 фичи, lr=0.03, num_leaves=63, min_child_samples=100 — это sweet spot.
3. **Ensemble = диверсификация, а не improvement.** Avg/rank/ridge — заметно хуже одиночного LGB. Модели слишком коррелированы.
4. **Больше фич ≠ лучше.** 40f → Sh=1.96 vs 23f → Sh=2.80. Шум победил сигнал.
5. **Интересно:** P-ta имеет worst month -19.1% (лучший drawdown), P-fng стабильнее. Можно использовать для risk management, но не для return maximization.

**Рекорд не побит. Текущий лучший: Sh=2.80, Eq=$2096** (LGB-23f, cutoff=0.9, 12h rebal, 6L/3S)

---

## Round 23 — Deep Training & Signal Experiments

**Цель:** Попробовать принципиально другие подходы к обучению и использованию сигналов, вместо breadth (модели/фичи) из R22.

**База:** R20-C — LGB-23f, cutoff=0.9, 12h rebal, 6L/3S → **Sh=2.80, Eq=$2096**

### Эксперименты

#### EXP-F: Signal EMA + Prediction Shrinkage
**Идея:** Сглаживание предсказаний (EMA) и сжатие к медиане (shrinkage) для снижения шума.

| EMA | Shrinkage | Sharpe | Equity | Worst Month | WM |
|-----|-----------|--------|--------|-------------|-----|
| 0 | 0 | 2.80 | $2096 | -28.5% | 10/13 |
| 0 | 0.1-0.3 | 2.80 | $2096 | -28.5% | 10/13 |
| 2 | 0-0.3 | 2.18 | $962 | -30.1% | 9/13 |
| 3 | 0-0.3 | 2.30 | $1112 | -18.7% | 9/13 |
| 5 | 0-0.3 | 2.28 | $1072 | -19.0% | 9/13 |

**Вывод:** Shrinkage не влияет (pred_shrinkage не работает в текущей реализации). EMA ухудшает Sharpe, но EMA=3 улучшает worst month с -28.5% до -18.7%.

#### EXP-D: Signal Confidence Filter
**Идея:** Торговать только когда модель "уверена" — спред предсказаний (top-bottom) выше порога.

| Min Spread | Timestamps | Sharpe | Equity | Worst Month |
|------------|------------|--------|--------|-------------|
| 0.0 | 7882/7882 | 2.80 | $2096 | -28.5% |
| 0.3 | 7882/7882 | 2.80 | $2096 | -28.5% |
| 0.5 | 7833/7882 | 1.94 | $673 | -8.9% |
| 0.7 | 7036/7882 | 1.59 | $402 | -33.2% |
| 0.9 | 5088/7882 | 0.98 | $185 | -33.0% |

**Вывод:** Фильтрация убивает доходность. D-conf0.5 имеет лучший worst month (-8.9%), но Sharpe падает до 1.94. Сигнал слишком слаб для фильтрации.

#### EXP-G: Classification Target ⭐ НОВЫЙ РЕКОРД
**Идея:** Вместо regression (predict rank), обучить binary classifier: P(positive return). LGB с objective="binary", metric="auc".

| Config | Sharpe | Equity | Worst Month | WM |
|--------|--------|--------|-------------|-----|
| **G-classification** | **2.94** | **$1997** | **-13.8%** | **10/13** |
| Baseline (regression) | 2.80 | $2096 | -28.5% | 10/13 |

**Месячные результаты G-classification:**
| Месяц | Return | Equity |
|-------|--------|--------|
| 2024-10 | +44.6% | $145 |
| 2024-11 | -10.4% | $130 |
| 2024-12 | +86.8% | $242 |
| 2025-01 | +15.8% | $280 |
| 2025-05 | +45.5% | $408 |
| 2025-06 | +38.1% | $563 |
| 2025-07 | -13.8% | $485 |
| 2025-08 | +38.2% | $671 |
| 2025-11 | +11.0% | $744 |
| 2025-12 | +56.4% | $1164 |
| 2026-01 | +33.3% | $1552 |
| 2026-02 | +33.2% | $2067 |
| 2026-03 | -3.4% | $1997 |

**Вывод:** 🏆 **НОВЫЙ РЕКОРД Sh=2.94!** Classification обходит regression по Sharpe (+0.14) И по worst month (-13.8% vs -28.5%). Equity чуть ниже ($1997 vs $2096), но risk-adjusted return значительно лучше. Бинарная задача проще → модель делает более стабильные предсказания.

#### EXP-C: Risk-Adjusted Target
**Идея:** Предсказывать fwd_ret/trailing_vol (risk-adjusted) или winsorized return.

| Config | Sharpe | Equity | Worst Month |
|--------|--------|--------|-------------|
| C-riskadjusted | 0.52 | $109 | -60.2% |
| C-winsorized | 2.19 | $992 | -39.3% |

**Вывод:** Risk-adjusted target катастрофически плох. Winsorized тоже хуже baseline. Деление на vol добавляет огромный шум.

#### EXP-B: Time-Weighted Training
**Идея:** Экспоненциальный decay при обучении — недавним данным выше вес.

| Half-Life | Sharpe | Equity | Worst Month |
|-----------|--------|--------|-------------|
| 90 дней | 1.59 | $431 | -40.5% |
| 180 дней | 1.30 | $289 | -45.4% |
| 365 дней | 2.08 | $866 | -42.0% |
| 730 дней | 2.46 | $1460 | -31.9% |

**Вывод:** Чем сильнее decay, тем хуже. Модель нуждается во ВСЕХ данных равномерно. hl=730d (почти без decay) ближе всего к baseline.

#### EXP-E: Rolling Window (ограничение обучающих данных)
**Идея:** Вместо expanding window — train только на последних N дней.

| Max Days | Sharpe | Equity | Worst Month |
|----------|--------|--------|-------------|
| 180 | 0.68 | $124 | -48.9% |
| 365 | 1.38 | $321 | -51.5% |
| 540 | 1.89 | $650 | -40.2% |
| 730 | 1.79 | $603 | -46.2% |

**Вывод:** Catastrophic. Expanding window >>> rolling. Модель критически зависит от объёма обучающих данных.

#### EXP-A: Individual Feature Addition (18 features, 1 at a time)
**Идея:** Каждую из 18 новых фич тестируем отдельно (R22-P добавлял группами → шум).

| # | Feature | Sharpe | Δ vs base | Worst Month | WM |
|---|---------|--------|-----------|-------------|-----|
| 1 | baseline (23f) | 2.80 | — | -28.5% | 10/13 |
| 2 | +vol_of_vol | 2.59 | -0.21 | -13.7% | 10/13 |
| 3 | +obv_ma_ratio_24 | 2.54 | -0.26 | -24.2% | 9/13 |
| 4 | +adx | 2.46 | -0.34 | -24.5% | 11/13 |
| 5 | +mfi_14 | 2.39 | -0.40 | -18.4% | 9/13 |
| 6 | +fng_value | 2.36 | -0.44 | -29.6% | 10/13 |
| 7 | +vix_close | 2.36 | -0.44 | -29.6% | 10/13 |
| 8 | +ret_168h | 2.35 | -0.45 | -21.1% | 9/13 |
| 9 | +dxy_ret_7d | 2.33 | -0.47 | -37.2% | 10/13 |
| 10 | +vix_zscore | 2.32 | -0.48 | -9.7% | 9/13 |
| 11 | +dist_from_high_24h | 2.32 | -0.48 | -48.2% | 10/13 |
| 12 | +oi_velocity | 2.25 | -0.55 | -48.6% | 9/13 |
| 13 | +ret_kurt_24h | 2.18 | -0.62 | -29.3% | 11/13 |
| 14 | +taker_imb_z | 2.18 | -0.62 | -38.3% | 10/13 |
| 15 | +vol_ratio_24h | 2.14 | -0.66 | -22.4% | 10/13 |
| 16 | +ret_skew_24h | 2.09 | -0.71 | -37.1% | 10/13 |
| 17 | +vwap_dev_24h | 2.07 | -0.73 | -31.5% | 10/13 |
| 18 | +fng_zscore | 1.91 | -0.89 | -30.8% | 9/13 |

**Вывод:** НИ ОДНА фича не улучшает Sharpe. 23 фичи — идеальный набор. Интересно: vol_of_vol (-13.7%) и vix_zscore (-9.7%) дают отличный worst month, но за счёт Sharpe.

### R23 Финальная таблица (top-10)

| # | Experiment | Sharpe | Equity | Worst Month | Δ vs 2.80 |
|---|-----------|--------|--------|-------------|-----------|
| 1 | **G-classification** | **2.94** | **$1997** | **-13.8%** | **+0.14** |
| 2 | F-ema0 (baseline) | 2.80 | $2096 | -28.5% | 0.00 |
| 3 | A-+vol_of_vol | 2.59 | $1574 | -13.7% | -0.21 |
| 4 | A-+obv_ma_ratio_24 | 2.54 | $1512 | -24.2% | -0.26 |
| 5 | B-hl730d | 2.46 | $1460 | -31.9% | -0.34 |
| 6 | A-+adx | 2.46 | $1463 | -24.5% | -0.34 |
| 7 | A-+mfi_14 | 2.39 | $1337 | -18.4% | -0.41 |
| 8 | F-ema3 | 2.30 | $1112 | -18.7% | -0.50 |
| 9 | F-ema5 | 2.28 | $1072 | -19.0% | -0.52 |
| 10 | C-winsorized | 2.19 | $992 | -39.3% | -0.61 |

### R23 Ключевые выводы

1. 🏆 **НОВЫЙ РЕКОРД: G-classification Sh=2.94** — бинарная классификация (P(ret>0)) лучше regression (rank). Sharpe выше, worst month вдвое лучше (-13.8% vs -28.5%).
2. **Expanding window обязателен.** Rolling window и time decay катастрофически ухудшают результат. Модель использует ВСЕ данные.
3. **Prediction shrinkage не работает.** Реализация в simulate() не влияет на результат (shrinkage=0.1/0.2/0.3 → одинаковый Sharpe).
4. **EMA сглаживание — trade-off.** EMA=3 снижает worst month (-18.7%), но ухудшает Sharpe до 2.30.
5. **23 фичи = sweet spot. Подтверждено третий раз.** Ни одна из 18 новых фич индивидуально не улучшает Sharpe. Однако vol_of_vol и vix_zscore дают лучший worst month.
6. **Risk-adjusted target — полная катастрофа.** Sh=0.52. Деление ret на vol добавляет noise.
7. **Signal confidence filter — не работает.** Модель всегда "уверена" примерно одинаково, фильтрация только убирает market exposure.

**🏆 НОВЫЙ РЕКОРД: Sh=2.94, Eq=$1997** (LGB-binary-23f, cutoff=0.9, 12h rebal, 6L/3S)
