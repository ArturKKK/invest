# Консультация: анализ регрессии производительности ансамбля моделей

## Контекст

Система: crypto algo-trading, 50 монет, 1h таймфрейм, 12h ребалансировка.
Ансамбль: LightGBM v6 + LightGBM v7 + CatBoost (по 5 сидов каждый = 15 моделей).
Дополнительно: derivatives-only mini-model (5 моделей, 29 фичей из Binance Futures OI/funding/taker).
Инференс: `mean(mean(v6_seeds), mean(v7_seeds), mean(CB_seeds))` → cross-sectional rank → top-N long / bottom-N short.

## Проблема

**9 марта 2026**: прогон fast simulation на 60d окне (Jan 6 – Mar 6, 3x leverage, 24h rebal) показал:
- Return: **+21.7%**, Sharpe: 2.03, WR: 62%, MaxDD: -21.7%

**10 марта 2026**: модели были перетренированы (DDStop fix + W2/W3 overlap fix + расширенное окно данных). Старые модели перезаписаны.

**Сейчас**: тот же 60d период, те же данные, но с НОВЫМИ моделями (10 марта):

| Конфиг | Код | Return | Sharpe HAC | MaxDD | WR |
|--------|-----|--------|-----------|-------|-----|
| OLD code, vanilla (v6+v7+CB+deriv as member) | pre-refactor | **LIQUIDATED** (-18.7%) | -4.16 | -36% | 40% |
| OLD code + edge-boost | pre-refactor | **LIQUIDATED** (-10%) | -5.10 | -40% | 40% |
| NEW code, vanilla v6+v7+CB (no extras) | post-refactor | **-6.4%** | -4.31 | -36% | 40% |
| NEW code + deriv risk gate | post-refactor | **-3.7%** | -3.10 | -33.7% | 56% |
| NEW code + edge-boost + short-blocked + deriv | post-refactor | **-18.5%** | -7.79 | -42% | 40% |

**Вывод**: рефакторинг кода НЕ виноват (old code ещё хуже с теми же моделями). Проблема в перетренированных моделях.

## Что изменилось при перетренировке (10 марта)

### 1. DDStop fix
Раньше: DDStop вычислялся с багом — при drawdown > threshold тренировка продолжалась, но equity замораживалась (flat line). Это **занижало** DDStop и позволяло модели продолжать торговать в просадке.

Теперь: DDStop правильно срезает exposure. Модель стопается в просадке, потом **возобновляется с СЛЕДУЮЩЕГО бара** (не текущего, избегая lookahead). 

**Гипотеза**: исправленный DDStop делает модели более консервативными → меньше alpha на благоприятных участках.

### 2. W2/W3 non-overlapping windows
Раньше: train/val/test окна могли перекрываться (W2 overlap). Это вносило data leakage.

Теперь: строгое разделение без overlap + purge gap.

**Гипотеза**: fix устранил data leakage → модели показывают РЕАЛЬНУЮ, более низкую предсказательную силу. Старый +21.7% был частично от leakage.

### 3. Расширенное окно данных
`train_end = 2025-12-01`, `val_end = 2026-03-07` — больше данных для validation.

**Гипотеза**: модели могли переоптимизироваться на validation window, который включает часть test периода.

## Конкретные вопросы

### Q1: Data leakage как причина старого результата
Если W2/W3 overlap давал leakage, то +21.7% был завышен. Насколько реалистично, что fix leakage вызывает такую большую деградацию (от +21.7% до ликвидации)?

Для оценки: используется walk-forward (expanding window) с 5 группами сидов. Каждый сид — отдельный train/val/test split. В старом коде W2 (val) и W3 (test) могли overlap на несколько месяцев.

### Q2: DDStop bias
Старый DDStop с багом фактически давал моделям "бесплатный" выход из просадки (equity zamerezhivalas', но позиции оставались). Это могло:
- a) Создать модели, которые агрессивнее берут риск (зная что DDStop "прощает")
- b) Дать leakage в val/test метриках (DDStop метрики были optimistic)

Вопрос: если модели теперь с правильным DDStop (real-time stop + resume), должны ли мы ожидать краткосрочную деградацию? Или модели должны быть ЛУЧШЕ в long run?

### Q3: Edge-boost sizing
Edge-boost: position weight ∝ |score − median| / P75 edge (capped at 2x).
На 60d окне с edge-boost: -16% vs +8% без него (с новым кодом/моделями).

Раньше edge-boost документировался как "Sharpe 2.79→5.93, WR 56%→70%".

**Парадокс**: production сейчас работает С edge-boost. Стоит ли его отключить?

Вопрос: является ли edge-boost принципиально ущербным (concentration risk в хвостах), или это режим-зависимая вещь (плохо в mean-reversion рынке, хорошо в trending)?

### Q4: Deriv-only model — member vs gate
Старый код: `mean(v6, v7, CB, deriv)` — deriv как 4-й равноправный member.
Новый код: `mean(v6, v7, CB) × deriv_gate(0.3–1.0)` — deriv как risk filter.

Результаты показывают, что deriv-as-gate (+deriv gate: -3.7%) лучше deriv-as-member (old code: -18.7%), но разница может быть от других факторов.

Вопрос: что теоретически правильнее для модели с 29 фичами + Rank_IC 0.025 (слабый signal) — быть averaging member или risk gate?

### Q5: Общая рекомендация
Данные:
- 14d тест (свежий рынок): +8.4%, Sharpe 5.14 (с deriv gate)
- 60d тест: -3.7% (лучший конфиг) до -18.5% (с edge-boost)
- 365d тест (старые модели, 1x): +21.3%, Sharpe 6.61, MaxDD -5.4%

Нужно ли:
a) Откатить модели к pre-fix (если это возможно) и проверить гипотезу leakage?
b) Запустить ретренировку без DDStop fix но с W2/W3 fix (изолировать фактор)?
c) Просто подождать — 60d окно нерепрезентативно (14d показывает хорошо)?
d) Что-то ещё?

## Архитектура системы (для контекста)

```
Features (178):
  - Base OHLCV features: returns, volatility, momentum
  - Cross-asset: BTC/ETH correlation, market-wide features
  - Regime: bull/bear detection, volatility regime
  - 12h-specific: holding period features
  - Sentiment: synthetic (no real news in current data)
  - Derivatives: OI, funding rate, taker buy/sell ratio, L/S ratio
  
Models:
  - LGB v6: 5 seeds, 142 features, residual-hybrid target
  - LGB v7: 5 seeds, 148 features, residual-hybrid + advanced features
  - CatBoost: 5 seeds, 152 features, with news (ordered boosting)
  - Deriv-only: 5 seeds, 29 features, derivatives-only subset
  
Ensemble:
  mean(group_mean(v6), group_mean(v7), group_mean(CB)) → rank → top/bottom N
  
Sizing options:
  - Equal weight (default)
  - Edge-boost: weight ∝ |score − median|/P75 (max 2x)
  
Risk:
  - DDStop: -20% drawdown → stop, resume at -8%
  - Deriv gate: scale positions 0.3–1.0 based on derivatives signal agreement
  - Vol targeting: 30% annual portfolio vol target
```

## Метаданные для воспроизведения

### КРИТИЧЕСКАЯ НАХОДКА: Live vs Offline feature mismatch

**Live-режим** (`build_features()` в `run_trading.py`) **не имеет 21 фичу**, которую модели ожидают:
- Все derivatives: `oi_change_*`, `oi_value_usd`, `oi_zscore_7d`, `taker_cvd_*`, `top_ls_*`, `global_ls_*`
- `funding_rate_binance`, `funding_surprise`
- `news_coverage_ok`, `news_event`

Эти фичи заполняются нулями → модели предсказывают мусор в live.

**Offline-режим** (`--data` + pipeline enrichment) не имеет 15 фичей (в основном news).

**Результат A/B на одном 14d окне:**
| Режим | Return | Sharpe |
|-------|--------|--------|
| Live (21 missing → zeros) | **-3.4%** | -1.37 |
| Offline (pipeline feats) | **+4.9%** | +2.34 |

**Вывод**: production bot (run_trading.py) сломан — ему не хватает derivatives features. Все предыдущие "live" результаты на проде были с неполными фичами.

### Невалидность 365d и 60d тестов

Новые модели: `val_end = 2026-03-07`. Данные в parquet: до `2026-03-07`.
- **365d тест**: 100% данных видели при обучении → leakage → невалиден
- **60d тест**: Jan 6 - Mar 7, целиком в validation window → невалиден
- **14d offline**: Feb 21 - Mar 7, внутри val → невалиден
- **Реальный OOS**: только данные ПОСЛЕ Mar 7 (3 дня, нет в parquet)

```bash
# Run fast sim with current models (March 10)
python run_fast_sim.py --ensemble --leverage 3 --rebal 24 \
  --data data/features/crypto_features_1h.parquet --days 60

# With deriv gate
python run_fast_sim.py --ensemble --leverage 3 --rebal 24 \
  --data data/features/crypto_features_1h.parquet --days 60 --deriv-gate

# With edge-boost
python run_fast_sim.py --ensemble --leverage 3 --rebal 24 --edge-boost \
  --data data/features/crypto_features_1h.parquet --days 60
```
