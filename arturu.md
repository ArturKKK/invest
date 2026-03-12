# Arturu — Шпаргалка по торговой системе

## Общая архитектура

```
OHLCV данные (50 монет, 800 часов)
        │
        ▼
┌─────────────────────────────┐
│   Feature Engineering       │
│   207 фичей:                │
│   - технические (MA, RSI…)  │
│   - cross-asset (BTC/ETH)   │
│   - режимные (regime)       │
│   - sentiment (FNG, news)   │
│   - derivatives (OI, fund)  │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────┐
│   L0 — Базовые модели (каждая группа = 5 сидов) │
│                                                  │
│   lgb_v6_no_news  × 5  → pred_v6               │
│   lgb_v7_no_news  × 5  → pred_v7               │
│   catboost_with_news × 5 → pred_cb             │
│   xgboost         × 5  → pred_xgb  ⚠️ НЕ подключён │
│   deriv_only      × ?  → pred_deriv ❌ НЕ обучена  │
└─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│   L1 — Meta-model (LGB-MINIMAL)    │
│                                     │
│   Вход: pred_v6, pred_v7, pred_cb  │
│         + рыночные фичи            │
│   Выход: финальный скор            │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│   Z-score нормализация      │
│   (cross-sectional rank)    │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│   Portfolio Construction    │
│   10 Long + 10 Short       │
│   × $225 каждая             │
│   = $4500 из $5000 капитала │
│   При 3x leverage           │
└─────────────────────────────┘
```

---

## Что такое L0 модели

Это базовые ML-модели, каждая обучена предсказывать `target_12h` (доходность монеты через 12 часов).

| Группа | Алгоритм | Фичей | Кол-во моделей | Статус |
|--------|----------|-------|----------------|--------|
| `lgb_v6_no_news` | LightGBM | 160 | 5 (разные random seed) | ✅ На проде |
| `lgb_v7_no_news` | LightGBM | 150 | 5 | ✅ На проде |
| `catboost_with_news` | CatBoost | 160 | 5 | ✅ На проде |
| `xgboost` | XGBoost | ~160 | 5 | ⚠️ Обучена, лежит на проде, но **не вызывается** в коде |
| `deriv_only` | ? | ? | 0 | ❌ Никогда не обучалась, папка пустая |

Каждая группа из 5 моделей усредняет свои прогнозы → один `pred_XXX`.

**Файлы моделей**: `results/production/<группа>/`

---

## Что такое Meta-model (L1)

Meta-model берёт прогнозы L0-моделей и комбинирует их в один финальный скор. Это «модель над моделями».

На вход подаётся: `pred_v6`, `pred_v7`, `pred_cb` (+ `pred_xgb` = 0, т.к. не подключён).

При тестировании (`run_meta_stack.py`) сравнивались разные варианты мета-комбинации:

| Вариант | Что делает | DDStop_Sharpe | Rank_IC |
|---------|-----------|---------------|---------|
| **cb only** | Просто берёт pred_cb | **3.13** | 0.0228 |
| **LGB-MINIMAL** ← на проде | LightGBM на прогнозах + базовые рыночные фичи | **2.95** | 0.0269 |
| Ridge-4 | Ridge-регрессия на 4 прогнозах | 2.93 | 0.0238 |
| Ridge-ALL | Ridge на прогнозах + спреды + ранки | 2.84 | 0.0277 |
| Simple Mean | Среднее всех прогнозов | 2.81 | 0.0236 |
| LGB-META | LightGBM на всех 37 мета-фичах | 2.75 | 0.0328 |

**Файл мета-модели**: `results/meta_stack/meta_model.pkl`

> **Sharpe** — мера риск/доходность (чем выше, тем лучше).
> **Rank_IC** — корреляция предсказанного ранга с реальным (чем выше, тем точнее прогноз).

---

## Что такое deriv_only

Планировалось как отдельная L0-модель, обученная **только на деривативных фичах** (OI, funding, taker ratio, L/S ratio, basis). Идея — специализированная модель, которая ловит сигналы из рынка деривативов.

Была запланирована как Phase 3 в `run_prod_training.sh`, но скрипт до неё не дошёл. Результат: папка `results/production/deriv_only/` пустая, модели не обучены.

В `generate_signal()` есть код «deriv gate» — он должен был гейтить (фильтровать) сигналы через deriv_only модель, но т.к. модели нет, это no-op.

---

## Фичи (207 штук)

### Категории

| Категория | Кол-во | Источник |
|-----------|--------|----------|
| Технические | ~120 | OHLCV (returns, MA, RSI, ADX, Bollinger, ATR, volume) |
| Cross-asset | 17 | Корреляции с BTC/ETH, ret_vs_btc |
| Regime | 8 | BTC выше/ниже MA, дропдаун-режимы |
| Sentiment | 18 | Fear & Greed Index, новости, OKX funding |
| Derivatives | 27 | Binance: OI, taker buy/sell, L/S ratio, funding, basis |
| 12h-specific | ~17 | Фичи для 12-часового горизонта (v7) |

### Известные проблемы с фичами

**15 all-zero фичей** (из 207):

| Фича | Причина |
|------|---------|
| `oi_change_1h/4h/12h/24h` | data.binance.vision имеет лаг ~1 день. Последние часы маппятся на одно значение OI → pct_change = 0 |
| `oi_ret_interaction`, `oi_ret_interaction_12h` | Зависят от oi_change → тоже 0 |
| `agg_oi_change_12h`, `agg_oi_total_change_12h` | Агрегаты по oi_change → 0 |
| `top_ls_change_12h/24h` | L/S ratio из тех же CSV → тот же лаг |
| `close_ma720_ratio`, `vol_ma720_ratio` | MA(720) на 800 точках — на границе окна |
| `btc_regime_24`, `btc_regime_72` | Structural zero — зависит от режима BTC |
| `fng_extreme_greed` | Корректно: FNG=16 (extreme fear), поэтому extreme_greed=0 |

---

## Торговый цикл

```
Каждые 12 часов (00:05 UTC и 12:05 UTC):

1. Fetch OHLCV (50 монет × 800 часов) — с Binance через ccxt
2. Build features (207 фичей)
3. L0 inference → pred_v6, pred_v7, pred_cb
4. Meta-model → финальный скор
5. Z-score нормализация (cross-sectional)
6. Portfolio: TOP-10 = Long, BOTTOM-10 = Short
7. Close all existing positions
8. Wait for margin (OKX нужно время на освобождение маржи)
9. Open 20 new positions × $225 each
10. Update dashboard
11. Sleep until next cycle
```

---

## Инфраструктура

| Компонент | Детали |
|-----------|--------|
| **VPS** | 185.42.163.63, user: `trader`, `/home/trader/invest/` |
| **Биржа** | OKX Demo (sandbox), 3x leverage, isolated margin |
| **Капитал** | $5000, из них $4500 в позициях |
| **Service** | systemd `crypto-trader`, auto-restart |
| **Python** | `/home/trader/invest/venv/bin/python` |
| **Git** | github.com/ArturKKK/invest, branch: main |

### Cron-задачи (trader user)

| Время | Что делает |
|-------|-----------|
| `0 */6 * * *` | News update (fetch_crypto_news) |
| `35 5,11,17,23 * * *` | OKX sentiment (funding, FNG) |
| `40 5,11,17,23 * * *` | Binance derivatives (OI, taker, L/S, premium) |
| `0 3 * * *` | Log rotation (>7 дней удаляются) |
| `*/5 * * * *` | Watchdog: перезапуск если сервис упал |

---

## Ключевые файлы

| Файл | Описание |
|------|----------|
| `run_trading.py` | Главный бот — fetch, features, inference, trading |
| `run_pipeline_v6.py` | Feature engineering (enrichment functions) |
| `run_meta_stack.py` | Обучение и тестирование мета-модели |
| `run_prod_training.sh` | Скрипт обучения прод-моделей (Phase 1-3) |
| `deploy/crontab.txt` | Cron-расписание |
| `deploy/deploy.sh` | Деплой на VPS |
| `telegram_bot.py` | Телеграм-уведомления |
| `dashboard/index.html` | Веб-дашборд |
| `src/data/download_binance_futures.py` | Загрузка деривативов с Binance |
| `src/data/download_sentiment.py` | Загрузка sentiment данных |

### Директории моделей

```
results/production/
├── lgb_v6_no_news/        ← 5 LGB моделей ✅
├── lgb_v7_no_news/        ← 5 LGB моделей ✅
├── catboost_with_news/    ← 5 CB моделей ✅
├── catboost_no_news/      ← 5 CB моделей (не используются)
├── xgboost/               ← 5+5 XGB моделей (обучены, НЕ подключены) ⚠️
└── deriv_only/            ← ПУСТО ❌

results/meta_stack/
├── meta_model.pkl         ← LGB-MINIMAL мета-модель
├── meta_stack_results.json ← Бенчмарк всех вариантов
└── meta_retrain_info.json ← Инфо о последнем переобучении
```

---

## Баги, которые мы починили (12 марта 2026)

1. **Score compression** — z-score нормализация отсутствовала после деплоя. L/S spread упал с ~3.0 до ~0.08. Починили: добавили `z_score_normalize()` + правильный meta variant.

2. **Order 51008 (insufficient balance)** — две причины:
   - Нет паузы между `close_all()` и `execute()`: OKX не мгновенно освобождает маржу → добавили `wait_for_margin()`.
   - Неправильная конвертация USD → контракты: отправляли `amount=225` (225 контрактов BTC = $156k вместо $225). Починили: `_usd_to_contracts()` учитывает `contractSize × price`.

3. **Crontab `source` не работал** — cron использует `/bin/sh` (dash), а `source` — bashism. Заменили на абсолютный путь к python. Также перевели расписание на XX:35/XX:40 чтобы свежие данные были готовы до цикла бота.

---

## TODO / Потенциальные улучшения

- [ ] Подключить XGBoost в `generate_signal()` — модели обучены, мета-модель ожидает `pred_xgb`
- [ ] Попробовать `cb only` мета-вариант (Sharpe 3.13 vs текущие 2.95)
- [ ] Обучить `deriv_only` (Phase 3 из run_prod_training.sh)
- [ ] Добавить REST API fallback для OI/taker данных (убирает 10 из 15 нулевых фичей)
- [ ] Переобучить мета-модель с подключённым XGBoost
