# Валидация результатов ML-торговой системы — проверка на ошибки

Ты эксперт по quantitative finance, ML backtesting и статистическим ловушкам. Мне нужна **жёсткая проверка**: где мы могли ошибиться, есть ли в результатах red flags, можем ли мы доверять нашим метрикам.

---

## Контекст

Мы строим ML long-short стратегию на 50 криптовалютах. Ребалансировка каждые 12 часов, top-5 long + bottom-5 short. Текущий ансамбль: 15 моделей (LightGBM v6 ×5 seeds + LightGBM v7 ×5 seeds + CatBoost ×5 seeds).

Мы провели масштабный эксперимент (exp12) — 20 вариантов моделей через 3-window walk-forward validation. Ниже полные результаты. **Проверь на ошибки и bias.**

---

## Методология

### Walk-Forward Windows

| Window | Train end | Purge (8d) | Val start | Val end | Test start | Test end |
|--------|-----------|------------|-----------|---------|------------|----------|
| W1 | 2023-06-30 | → 2023-07-08 | 2023-07-08 | 2024-06-30 | 2024-07-01 | 2024-12-31 |
| W2 | 2024-01-01 | → 2024-01-09 | 2024-01-09 | 2024-12-31 | 2025-01-01 | 2025-03-31 |
| W3 | 2024-06-30 | → 2024-07-08 | 2024-07-08 | 2025-06-30 | 2025-01-01 | 2026+ |

- Purge gap: 8 дней между train/val/test
- Данные: hourly OHLCV с 2017 (2.5M строк, 50 символов, Binance)
- Target: `target_rank` = cross-sectional percentile rank (0-1) of forward 12h return per timestamp

### Cost Model
```
taker_fee = 0.04%
slippage = 0.02%
round_trip = (0.04 + 0.02) × 2 = 0.12%
turnover = 50% per rebalance (assumption)
funding = 0.01% per 8h
```

### DDStop (key metric)
DDStop Sharpe — Sharpe ratio with circuit breaker:
- Position paused at DD < -20%
- Resumed at DD > -8%
- Annualized Sharpe of the resulting equity curve

---

## Полные результаты exp12 (20 вариантов)

### Рейтинг по Avg DDStop Sharpe

| # | Variant | DDStop Sh | ±std | W1 | W2 | W3 | DDStop MaxDD% | AnnRet% | Rank_IC | ICIR |
|--:|---------|----------:|-----:|----:|----:|----:|------:|--------:|--------:|-----:|
| 1 | v7_baseline | 2.12 | 0.22 | 1.84 | 2.14 | 2.38 | -38.3 | 49.0 | 0.0243 | 0.234 |
| 2 | v7_res_hyb | 1.84 | 0.92 | 0.55 | 2.62 | 2.36 | -38.3 | 32.7 | 0.0271 | 0.231 |
| 3 | v7_res_hyb_null | 1.83 | 0.92 | 0.54 | 2.44 | 2.52 | -35.6 | 30.1 | 0.0267 | 0.221 |
| 4 | catboost_baseline | 1.76 | 0.45 | 1.18 | 1.81 | 2.29 | -46.3 | 48.1 | 0.0243 | 0.240 |
| 5 | xgboost_res_hyb | 1.76 | 0.73 | 0.72 | 2.34 | 2.21 | -36.8 | 39.6 | 0.0277 | 0.225 |
| 6 | v6_res_hyb | 1.67 | 1.09 | 0.14 | 2.55 | 2.32 | -39.6 | 37.0 | 0.0280 | 0.229 |
| 7 | v6_residual | 1.67 | 0.91 | 0.39 | 2.21 | 2.41 | -37.6 | 39.3 | 0.0248 | 0.235 |
| 8 | v6_res_hyb_no_news | 1.66 | 1.15 | 0.04 | 2.45 | 2.50 | -34.3 | 32.6 | 0.0283 | 0.230 |
| 9 | cb_res_hyb_no_news | 1.64 | 0.82 | 0.52 | 2.45 | 1.95 | -38.1 | 33.3 | 0.0288 | 0.238 |
| 10 | v6_hybrid | 1.63 | 0.57 | 0.83 | 1.97 | 2.10 | -41.0 | 40.3 | 0.0273 | 0.227 |
| 11 | v6_baseline | 1.61 | 0.38 | 1.07 | 1.90 | 1.87 | -41.9 | 41.4 | 0.0248 | 0.228 |
| 12 | v6_res_hyb_coin | 1.57 | 0.90 | 0.31 | 2.06 | 2.34 | -38.8 | 36.6 | 0.0282 | 0.233 |
| 13 | v6_res_hyb_market | 1.53 | 1.09 | -0.01 | 2.23 | 2.38 | -41.1 | 34.3 | 0.0278 | 0.233 |
| 14 | xgb_res_hyb_no_news | 1.41 | 1.49 | -0.69 | 2.66 | 2.25 | -42.0 | 34.2 | 0.0280 | 0.222 |
| 15 | v6_res_hyb_null | 1.39 | 1.32 | -0.48 | 2.34 | 2.31 | -49.6 | 34.4 | 0.0280 | 0.223 |
| 16 | catboost_res_hyb | 1.32 | 0.64 | 0.44 | 1.94 | 1.57 | -48.7 | 40.4 | 0.0284 | 0.242 |
| 17 | v6_no_deriv | 1.09 | 0.47 | 0.44 | 1.53 | 1.31 | -52.8 | 39.2 | 0.0273 | 0.235 |
| 18 | v6_no_news_no_deriv | 1.03 | 0.37 | 0.51 | 1.28 | 1.31 | -54.4 | 38.4 | 0.0281 | 0.239 |
| 19 | v7_lambdarank | -0.89 | 0.22 | -1.14 | -0.60 | -0.92 | -31.2 | -41.9 | -0.0152 | 0.026 |
| 20 | v6_lambdarank | -0.98 | 0.19 | -1.24 | -0.89 | -0.81 | -39.6 | -43.2 | -0.0130 | 0.020 |

### Наши выводы (проверь)

1. **v7_baseline — лидер** (DDStop 2.12, std=0.22 — самый стабильный, 49% ann ret)
2. **LambdaRank — катастрофа** (отрицательный Sharpe во всех окнах). LambdaRank objective + наш rank target = не работает.
3. **Derivatives features — главный фактор**: с ними DDStop 1.67, без них 1.09 (−0.58)
4. **News бесполезны для LGB** (1.67 vs 1.66), **вредят CatBoost** (1.32 vs 1.64)
5. **Residual/Hybrid target добавляет variance** без стабильного улучшения
6. **Null-importance фильтрация бесполезна**

---

## Дополнительные результаты: offline backtest (run_fast_sim.py)

Прогнали ансамбль (v6+v7+CB, edge-boost) через независимый быстрый бэктест (НЕ walk-forward):

| Период | Lev | Return | MaxDD | Sharpe HAC | WR |
|--------|-----|--------|-------|------------|-----|
| 60 дней (Янв-Мар 2026) | 1x | +14.4% | -12.5% | +2.80 | 61% |
| 60 дней | 3x | +35.8% | -35.8% | +4.12 | 58% |
| 180 дней (Сен 2025–Мар 2026) | 1x | +3.6% | -23.1% | -1.41 | 52% |
| 180 дней | 2x | -4.0% | -43.7% | -2.98 | 54% |
| 365 дней | 1x | +34.4% | -23.1% | +0.56 | 61% |

**Проблема**: 6-месячный бэктест показывает Sharpe -1.41, при том что 60-дневный даёт +2.80. Весь профит в последних 2 месяцах.

При этом live demo (OKX testnet, работает 2 месяца) даёт: Sharpe 6.61, Return +21.3%, WR 61%.

---

## Ключевые детали реализации

### Нормализация фичей
- Большинство фичей: **cross-sectional rank** (0-1) per timestamp per coin
- Vol/OI/funding features: **time-series zscore** (rolling 168h, winsorized ±3σ)
- Regime features (binary/composite): без нормализации

### Ансамбль
```
final_score = mean([mean(v6_seeds), mean(v7_seeds), mean(cb_seeds)])
confidence = 1 / (1 + std(all_15_normalized_predictions))
position_weight = edge_boost × confidence
edge_boost = 1 + min(|edge| / P75_edge, 3.0)
```

### Top Feature Importance
- **LGB**: fng_ma30 > fng_momentum > fng_ma7 > vol_12h_cs_rank > close_ma720_ratio
- **CatBoost**: close_ma336_ratio > vol_12h_cs_rank > close_ma720_ratio > breadth_pct_positive > gk_vol_24h

### Данные с потенциальными проблемами
- **50 символов фиксированы** (список из 2025, используется на данных 2017+) → survivorship bias
- **19/50 символов** заблокированы для шорта на OKX → live не может шортить эти монеты, а бэктест может
- **News покрытие**: 87% строк покрыты, 49% из покрытых имеют новости, остальные = NaN (не 0)
- **Derivatives data**: только с Dec 2021 (Binance), покрытие ~70% строк для OI, ~64% для taker

---

## Вопросы для проверки

### 1. Валидность DDStop Sharpe
DDStop Sharpe 2.12 — это реалистично для 12h crypto L/S на 50 монетах? Или это подозрительно высоко?

### 2. Методология walk-forward
3 окна, 8-дневный purge, expanding window. Достаточно ли 3 окон? Purge 8 дней для 12h таргета — хватает? Есть ли утечка?

### 3. Survivorship bias
50 монет из 2025 на данных 2017+. Насколько это критично? Может ли это объяснить +49% ann return?

### 4. Расхождение метрик
- Walk-forward DDStop Sharpe 2.12 vs offline backtest Sharpe HAC 0.56 (365d)
- Live demo Sharpe 6.61 vs offline 2.80 (60d)
Откуда расхождение? Это нормально или red flag?

### 5. Паттерн "W1 слабый, W2-W3 сильные"
Во всех вариантах W1 (тест 2024-07 → 2024-12) значительно слабее W2 и W3. Это model decay? Или что-то не так с данными?

### 6. News вредят
News ВРЕДЯТ CatBoost (1.32 vs 1.64) — при том что CatBoost specifically designed для categorical/noisy features. Это нормально?

### 7. LambdaRank полностью сломан
DDStop -0.89 / -0.98 с отрицательным Rank_IC. Это баг или LambdaRank принципиально не подходит?

### 8. Cost model
0.04% taker + 0.02% slip + 50% turnover + 0.01%/8h funding — реалистично? Или мы занижаем costs?

### 9. Что мы точно делаем неправильно?
Какие классические ошибки quant backtesting ты видишь? Look-ahead bias? Information leakage? Overfitting to in-sample?

### 10. Можем ли доверять результатам достаточно, чтобы торговать на реальные деньги?
У нас $500 для старта. При текущих результатах — стоит ли? Какие guardrails нужны?

---

## ОТВЕТЫ на твои вопросы (после первого ревью)

### RE: #0 — W3 val/test пересечение
**Это была опечатка в документации.** В коде:
```python
# W3
'val_start': '2024-07-07',
'val_end': '2024-12-30',     # ← val заканчивается здесь
'test_start': '2025-01-01',  # ← test начинается здесь
'test_end': '2026-12-31',
```
Val и test НЕ пересекаются. Purge gap ~2 дня (30 дек → 1 янв).

Однако ты прав, что **W2 test (2025-01→2025-12) и W3 test (2025-01→2026+) перекрываются** — W2/W3 не полностью независимы.

### RE: #1 — DDStop: ты нашёл реальную проблему!

Проверил код `drawdown_stop_returns()`:
```python
if is_stopped:
    equity *= (1 + net_rets[i])   # ← equity ПРОДОЛЖАЕТ двигаться!
    dd = equity / peak - 1
    if dd > recovery_threshold:
        is_stopped = False
        stopped_rets[i] = net_rets[i]
```

DDStop в pipeline **не закрывает позиции** — equity трекается по реальным returnам, а `stopped_rets` обнуляется. Это "бумажный" DDStop.

В fast_sim (run_fast_sim.py) DDStop реализован **реально**: `held_L.clear(); held_S.clear()` — позиции закрываются.

Поэтому DDStop Sharpe 2.12 из pipeline и Sharpe из fast_sim — **разные метрики**. MaxDD -38%…-52% в pipeline DDStop объясняется тем, что equity продолжает падать во время "паузы".

**Мы это принимаем:** для ранжирования вариантов между собой DDStop метрика всё ещё валидна (одна и та же формула для всех), но абсолютные значения завышены vs реальную торговлю.

### RE: #2 — W2 test 3 месяца слишком короткий
Согласны. W2 test на самом деле 2025-01 → 2025-12 (12 месяцев в коде, не 3 как в документации). Ещё одна ошибка в промпте, sorry.

### RE: #3 — Survivorship bias
Принимаем. Точная оценка эффекта неизвестна, но мы понимаем что ann return завышен.

### RE: #7 — LambdaRank
Проверили: группы формируются через `_compute_groups()` по timestamp, target = `target_rank` (уже ранжированный 0-1). Попробуем твой совет: raw returns как labels, но это low priority (LambdaRank отброшен).

### RE: #8 — Cost model
Turnover 50% — это грубое допущение из pipeline оценки. В fast_sim turnover считается точно по реальным позициям (open/close). Поэтому fast_sim цифры надёжнее.

### RE: #4 — Расхождение метрик
Теперь понятно почему расходятся:
- Pipeline DDStop Sharpe = "бумажный" DDStop + упрощённый turnover 50%
- Fast_sim = реальный DDStop + реальный turnover + per-step costs
- Live demo = реальный OKX execution + short constraints
Три разных симулятора = три разных результата. Ты прав что нужен "single source of truth".

---

## Новые результаты: exp13 (derivatives-only модель)

Обучили отдельную мини-модель на ТОЛЬКО деривативных фичах (29 features: OI, taker flow, L/S ratio, funding + минимальный контекст). Идея: декоррелированный эксперт для ансамбля.

Результаты walk-forward:
(результаты будут добавлены после завершения обучения)

---

## Обновлённые вопросы

### 11. DDStop как метрика для ранжирования
Мы используем DDStop Sharpe для СРАВНЕНИЯ вариантов (все считаются одной формулой). Для абсолютных ожиданий используем fast_sim. Это корректный подход?

### 12. Декоррелированный деривативный эксперт
Имеет ли смысл мини-модель на 29 фичах (OI/taker/funding) как дополнительный голос в ансамбле 15 моделей? Или лучше просто оставить деривативы как часть основных моделей?

### 13. Приоритеты перед деплоем
Из твоих замечаний, что КРИТИЧНО исправить перед реальными деньгами, а что можно итерировать после?

---

## Что мы уже исправили по твоим замечаниям

### Fix 1: DDStop теперь "flat equity"
```python
def drawdown_stop_returns(net_rets, max_dd_threshold=-0.25, recovery_threshold=-0.10):
    # Когда stopped: equity FLAT (позиции закрыты).
    # Отдельный shadow_equity отслеживает рынок для решения о resume.
    for i in range(n):
        if is_stopped:
            shadow_equity *= (1 + net_rets[i])  # только shadow двигается
            shadow_dd = shadow_equity / peak - 1
            if shadow_dd > recovery_threshold:
                is_stopped = False
                equity *= (1 + net_rets[i])     # resume
            # else: equity stays flat
        else:
            equity *= (1 + net_rets[i])
            shadow_equity *= (1 + net_rets[i])  # синхронизированы
            ...
```

### Fix 2: W2/W3 тесты больше не пересекаются
```
W1: test 2024-07-01 → 2024-12-31  (H2 2024)
W2: test 2025-01-01 → 2025-06-30  (H1 2025)
W3: test 2025-07-01 → 2026-12-31  (H2 2025+)
```
Теперь 3 полностью независимых OOS теста. W3 val расширен до 2025-06-29 чтобы покрыть больше данных.

---

## Новый вопрос: какие фичи добавить?

Win rate нашей стратегии ~61% на коротком горизонте, но 52% на 6 месяцах. Модель теряет edge при смене режима. Мы хотим добавить фичи которые помогут моделям лучше видеть market microstructure и regime shifts.

### Текущий набор фичей (~160 для v6, ~165 для v7):

| Категория | Кол-во | Примеры |
|-----------|--------|---------|
| Price/OHLCV | ~24 | ret_1h..168h, close_ma6..720_ratio, candle ratios |
| Volume | ~25 | vol_ma ratios, vol_mom, vwap_dev, obv, vol_surge |
| Momentum/Trend | ~25 | RSI, MACD, ADX, stoch, CCI, reversal scores, mom_zscore |
| Volatility | ~30 | GK vol, ret_std/skew/kurt, BB bands, ATR |
| Cross-Asset | ~14 | btc_ret, eth_ret, btc_beta, market_dispersion |
| Regime | ~12 | btc_regime, ma_above, breadth, regime_composite |
| Fear & Greed | ~6 | fng_value, fng_ma7/30, fng_momentum |
| Funding (OKX) | ~6 | funding_rate, market_avg_funding, L/S ratio |
| News/Sentiment | ~15 | news_count, news_sentiment, political_news (v6 only) |
| Derivatives | ~24 | OI changes, taker flow, top/global L/S, funding_binance |
| Time | ~1 | is_asian_session |

### Данные которые у нас ЕСТЬ и можем использовать:
- Binance 1h OHLCV (50 монет, 8 лет)
- Binance Futures: OI, taker buy/sell volume, top/global L/S (с Dec 2021)
- Binance + OKX funding rates
- Fear & Greed Index (daily)
- Crypto news с sentiment (87% покрытие)

### Данные которых у нас НЕТ (но можем попробовать достать):
- On-chain data (whale transactions, exchange inflows/outflows)
- Liquidation data
- Orderbook depth / bid-ask spread
- Cross-exchange arbitrage / basis / premium
- Social media sentiment (Twitter, Reddit volume)
- Options data (implied vol, put/call ratio)
- Macro market data (DXY, S&P500, bonds)

### Вопросы:
1. **Какие фичи ты бы добавил в первую очередь** из тех данных что у нас УЖЕ есть? Может мы не используем какие-то очевидные комбинации/взаимодействия?

2. **Какие новые данные дадут наибольший marginal edge** для 12h crypto L/S? Что реалистично достать и стоит усилий?

3. **Что БЕСПОЛЕЗНО** и не стоит тратить время? (чтобы мы не гнались за shiny objects)

4. У нас модель теряет edge в **боковике / mean-reverting режиме**. Есть ли фичи которые помогают именно в таких режимах?
