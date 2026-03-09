# Полный контекст проекта invest — ML Trading System для криптовалют

**Дата**: 9 марта 2026  
**Автор**: Артур Табаков  
**Цель**: автоматическая торговля криптовалютами с плечом, $500 стартовый капитал → максимизация прибыли

---

## 1. Что это за проект

ML-система для прогнозирования 12-часовых доходностей 50 криптовалют. Модель даёт скор для каждой монеты → берём топ-5 в лонг, худшие 5 в шорт → ребалансируем каждые 12 часов. Торгуем через OKX futures (пока demo).

**Стек**: Python, LightGBM, CatBoost, XGBoost, Optuna, CCXT (биржевое API), VADER (NLP для новостей). VPS на 185.42.163.63, дашборд на invest.arturt.com. Обучение на GPU-кластере.

---

## 2. Данные

### Источники
- **OHLCV**: 50 криптовалют, часовые свечи с 2021 года (через CCXT/Binance). ~2.5M строк.
- **Fear & Greed Index**: ежедневный индикатор настроений рынка (alternative.me API)
- **Funding rates (OKX)**: каждые 8 часов, funding rate
- **Binance Futures Derivatives** (NEW, Dec 2021+, data.binance.vision):
  - `binance_futures_metrics.parquet` (81MB, 1.8M rows) — OI, top-trader L/S ratio, global L/S ratio, taker buy/sell ratio
  - `binance_funding_rates.parquet` (1.2MB, 294K rows) — funding rates from Jan 2020
  - 50 символов, 1h частота (resample из 5min)
  - Фичи: OI change (1/4/12/24h), OI zscore, OI×return interaction, taker imbalance, taker CVD, taker flow zscore, top L/S changes, funding surprise, L/S divergence
- **Long/Short ratio**: соотношение лонг/шорт позиций
- **News**: CryptoCompare News API → VADER sentiment analysis → 10 фичей per coin:
  - `news_count_1h/24h/7d` — количество новостей  
  - `news_sentiment_1h/24h/7d` — средний sentiment [-1, +1]
  - `news_sentiment_momentum` — 24h sentiment − 7d sentiment
  - `news_volume_zscore` — z-score объёма новостей vs 30d
  - `market_news_count_24h`, `market_news_sentiment_24h` — market-wide

### Feature engineering (186 фичей у XGBoost)
- **Price/Volume**: returns (1h, 4h, 12h, 24h, 3d, 7d), volatility (Garman-Klass), VWAP distance, MA ratios (72h, 168h, 336h, 720h), volume surge, range expansion
- **Momentum**: z-scores, acceleration, trend strength, direction quality
- **Cross-asset**: BTC beta (48h, 168h), cross-coin dispersion, breadth (% positive), correlation
- **Sentiment**: FNG value/MA7/MA30/momentum, funding rate, L/S ratio, synthetic positioning (reversal signals)
- **Session/Regime**: is_asian_session, regime features (volatility regime, trend regime)
- **News (10 фичей)**: per-coin и market-wide sentiment + volume
- **News interactions (23 фичи, `nx_*`, только XGBoost)**: sentiment×volume, news burst, sentiment-price divergence, sentiment×momentum, coin-vs-market sentiment, sentiment×volatility, sentiment×fear&greed, news cluster detection, sentiment acceleration, funding×news, cross-coin news asymmetry

### Target
- `target_ret_12h` — доходность через 12 часов (rank-normalized per timestamp)

### Train/Val/Test split (walk-forward)
- **Window 1**: Train → 2023-06-30, Val 2023-07 → 2024-06, Test 2024-07 → 2024-12
- **Window 2**: Train → 2024-01-01, Val 2024-01 → 2024-12, Test 2025-01 → 2025-03
- **Window 3**: Train → 2024-06-29, Val 2024-07 → 2024-12, Test 2025-01 → latest
- **Production**: Train → 2025-09, Val → 2026-03 (max data, no test holdout)
- **Fast Sim 365d**: март 2025 → март 2026 (полностью OOS, 9 месяцев после конца train)

---

## 3. Модели — что пробовали, что работает

### Текущий ансамбль (15 моделей, live)

| Модель | Алгоритм | Фичей | News? | Seeds | DDStop Sharpe (avg) |
|--------|----------|-------|-------|-------|---------------------|
| **LGB v6** | LightGBM (leaf-wise) | 121 | Нет | 5 | **1.81** |
| **LGB v7** | LightGBM (leaf-wise) | 127 | Нет | 5 | **1.88** |
| **CatBoost** | CatBoost (ordered/symmetric) | 130 | **Да (8 news)** | 5 | **1.51** |

Финальный сигнал = `mean(15 предсказаний)`. Каждая модель обучена с 5 разными random seeds.

### XGBoost + News Interactions (обучается сейчас, эксперимент #9)

| Модель | Алгоритм | Фичей | News? | Уникальность |
|--------|----------|-------|-------|--------------|
| **XGBoost** | XGBoost (level-wise) | 186 | **Да (10 news + 23 interactions)** | Явные interaction features |

23 news interaction features (`nx_*`):
- `nx_sent_x_count_*` — sentiment × объём новостей (много позитивных новостей = сильный сигнал)
- `nx_burst_ratio`, `nx_is_burst`, `nx_burst_x_sent` — обнаружение всплеска новостей
- `nx_sent_price_div` — позитивные новости + падение цены = контрарный сигнал
- `nx_sent_mom_align` — sentiment alignment с ценовым моментумом
- `nx_sent_vs_market`, `nx_count_vs_market` — coin vs market sentiment gap
- `nx_sent_x_vol` — sentiment × волатильность
- `nx_sent_x_fear` — sentiment в контексте Fear & Greed
- `nx_high_vol_positive/negative` — новостные кластеры
- `nx_sent_accel`, `nx_sent_accel_7d` — ускорение sentiment
- `nx_funding_x_sent`, `nx_funding_sent_div` — funding rate × news
- `nx_news_in_dispersion` — новости в период ценовой дисперсии

**Ожидаемый ансамбль после**: 20 моделей (LGB v6×5 + v7×5 + CatBoost×5 + XGBoost×5). Три разных GBDT алгоритма (leaf-wise, symmetric, level-wise) для максимальной декорреляции ошибок.

### Что пробовали и отвергли

| Эксперимент | Результат | Почему не работает |
|-------------|-----------|-------------------|
| **LGB v8** (8 лет данных, 2017+) | Sharpe 0.68 (vs 1.17 у v7) | Крипторынок 2017-2020 ≠ 2021+. Старые паттерны разбавляют полезные |
| **HIST v1/v2** (Transformer) | LS Sharpe 2.93 в паре с v5 | Сложная архитектура, нестабильная. Не retraiнили с 12h target |
| **LGB с news** | DDStop Sharpe -36..47% хуже | LGB leaf-wise переоценивает шумные news фичи |
| **v7 blended target** (75% 12h + 25% 24h) | ≈ v6 | Усложнение без прироста |
| **min-conf 0.85** | Sharpe 6.61→3.77 на 365d | Режет 40% сделок, многие прибыльные. На 60d выглядит хорошо (10.02), но на полном году вредит |
| **Dynamic leverage** (3x→5x/7x) | MaxDD -35..49% | DD растёт быстрее прибыли, риск ликвидации |
| **P75 + seed agree filter** | WR 34%, -47% | Когда сиды согласны — ловушка, а не уверенность |
| **P90 edge filter** | Sharpe ↓ vs P75 | Слишком мало сделок, пустые слоты |
| **Adaptive rebalance** (P90 trigger) | Costs +130% | Слишком много ранних ребалансов |

### Что работает

| Идея | Эффект | Статус |
|------|--------|--------|
| **12h target** (aligned с holding period) | Sharpe +4x vs v5 | ✅ live |
| **Edge-boost sizing** (weight ∝ edge) | Sharpe 2.79→5.93, WR 56%→70% | ✅ live |
| **CatBoost в ансамбль** (с news) | Sharpe 5.93→8.04 (60d) / 6.61 (365d) | ✅ live |
| **Confidence weighting** (1/(1+std)) | Sharpe 2.27→2.48 | ✅ live |
| **Гибридный news**: LGB без news + CB с news | Sharpe +45% vs all-with-news | ✅ live |
| **Concentration cap** (max alloc = confidence) | Убирает 100% на 1 позицию | ✅ live |
| **Event filter** (FOMC/CPI) | Страховка от макро-шоков | ✅ live |

---

## 4. Текущие результаты (лучший бэктест)

### Pipeline бэктест (walk-forward, кластер)

**LGB v6 (W3)**: Rank IC 0.028, ICIR 0.427, LS Sharpe 1.12, DDStop Sharpe **1.81**  
**LGB v7 (avg 3W)**: Rank IC 0.029, ICIR 0.406, LS Sharpe 1.17, DDStop Sharpe **1.88**  
**CatBoost с news (avg 3W)**: ICIR 0.369, LS Sharpe 1.07, DDStop Sharpe **1.51**

### Fast Sim — реальные данные, Binance spot

#### Лучший конфиг: 365d, 1x leverage, 12h rebal, без min-conf
| Метрика | Значение |
|---------|----------|
| **Sharpe** | **6.61** |
| Return | +21.3% (ann ~21%) |
| Max DD | -5.4% |
| Win Rate | 61% |
| Profit Factor | 1.86 |
| Trades | 1140 |
| Costs | 5.1% |

#### С leverage 3x, 365d
| Метрика | Значение |
|---------|----------|
| Sharpe | 4.55 |
| Return | +48.7% |
| Max DD | -18.1% |

#### Полная сетка конфигов (9 марта 2026)

| # | Период | Lev | Rebal | min-conf | Return | Sharpe | WR | PF | MaxDD |
|---|--------|-----|-------|----------|--------|--------|-----|------|-------|
| 1 | 60d | 3x | 24h | — | +21.7% | 2.03 | 62% | 1.33 | -21.7% |
| 2 | 60d | 3x | 24h | 0.85 | +1.2% | 0.73 | 62% | 1.12 | -18.1% |
| 3 | 60d | 1x | 12h | 0.85 | +6.7% | 2.23 | 66% | 1.25 | -5.6% |
| 4 | 60d | 1x | 12h | — | +8.3% | 2.69 | 60% | 1.30 | -6.5% |
| 5 | 365d | 1x | 12h | 0.85 | +15.1% | 3.77 | 63% | 1.46 | -5.1% |
| **6** | **365d** | **1x** | **12h** | **—** | **+21.3%** | **6.61** | **61%** | **1.86** | **-5.4%** |
| 7 | 365d | 3x | 24h | — | +48.7% | 4.55 | 65% | 1.85 | -18.1% |

### Историческая 60d симуляция ($500 start, 3x, ensemble с CatBoost)
| Метрика | Значение |
|---------|----------|
| Sharpe | 8.04 |
| Return | +37.2% |
| WR | 67% |
| PF | 2.85 |
| Max DD | -18.7% |

> ⚠️ Sharpe 8.04 — на коротком удачном 60d окне. На 365d реалистичная оценка: 6.61. Не обманывайся коротким окном.

---

## 5. Ключевые инсайты (выученное)

1. **В крипто 4 года данных > 8 лет.** Рынок 2017-2020 совершенно другой, старые паттерны вредят (v8 провал).
2. **News помогают CatBoost, но вредят LGB.** Ordered boosting лучше справляется с шумными фичами. Гибрид (LGB без news + CB с news) оптимален.
3. **min-conf filter — ловушка.** На 60d выглядит великолепно (Sharpe 10.02), на 365d вредит (3.77). Удаляет 40% сделок, среди которых много прибыльных.
4. **Edge-boost sizing > equal weight.** Давать больше капитала высоко-уверенным сигналам = Sharpe +113%.
5. **Seed agreement ≠ уверенность.** Когда все 5 seeds модели согласны — это часто ловушка (WR 34%).
6. **Leverage > 3x = ликвидация.** При DD threshold -33% любой leverage выше 3x рано или поздно убивает.
7. **12h holding period оптимален.** 4h = слишком много costs, 24h = упущенные возможности.
8. **Ансамбль > одиночная модель.** v6+v7 = Sharpe 2.79 vs 2.54 (v6) / 2.73 (v7).
9. **Больше фичей ≠ лучше.** v7 (127 фичей) ≈ v6 (121 фича). v8 (153 фичи) хуже.
10. **Sentiment (FNG) — топовые фичи** в LGB. close_ma_ratios и volatility — топовые в CatBoost. Разные модели смотрят на разное → ансамбль сильнее.
11. **Residual-target и hybrid-norm улучшают IC, но вредят DDStop Sharpe.** Rank_IC +0.003, но DDStop до -47%. Baseline остаётся лучше по P&L.
12. **News features добавляют ~10 фичей средней важности** (news_count_7d в top-25 CatBoost), но снижают DDStop Sharpe у всех LGB на 30-47% vs no-news baseline. CatBoost устойчив к news.

---

## 6. Текущая production конфигурация

```
Capital:        $5,000 (OKX demo)
Models:         15 (LGB v6×5 + v7×5 + CatBoost×5)
Features:       LGB: 121-127 (без news), CatBoost: 130 (с news)
Positions:      5 long + 5 short
Rebalance:      12h
Leverage:       1x (3x опционально)
Sizing:         edge-boost (weight ∝ 1 + edge/P75, cap 4x)
min-conf:       отключён (вредит на 365d)
Risk:           kelly=100%, DD_stop=-20%, DD_resume=-8%
Event filter:   FOMC/CPI → leverage 30%
Costs:          ~4 bps/side (taker+slippage) + 1bp/8h funding
Dashboard:      invest.arturt.com
VPS:            185.42.163.63
Sentiment cron: каждые 8h
```

---

## 7. Архитектура кода

```
run_pipeline_v6.py          — LGB v6 train pipeline (base)
run_pipeline_v7.py          — LGB v7 train pipeline (blended target)
run_pipeline_catboost.py    — CatBoost train pipeline (ordered boosting)
run_pipeline_xgboost.py     — XGBoost train pipeline (news interactions)
run_trading.py              — Live/paper trading bot (OKX)
run_fast_sim.py             — Fast backtest simulator (ensemble)
run_leverage_sim.py         — Leverage/edge sweep tool
run_train_all.sh            — Train all models: bash run_train_all.sh <exp_name>
fetch_crypto_news.py        — News fetcher (CryptoCompare → parquet)
src/data/download_binance_futures.py — Binance Futures derivatives downloader
dashboard/                  — Web dashboard (invest.arturt.com)
data/features/              — crypto_features_1h.parquet (2.5M rows, 107 cols)
data/sentiment/             — news, binance metrics/funding, fear&greed, etc.
data/raw/                   — Per-symbol OHLCV parquets
results/production/         — Текущие production модели
  lgb_v6_no_news/           — LGB v6 (5 seeds)
  lgb_v7_no_news/           — LGB v7 (5 seeds)
  catboost_with_news/       — CatBoost (5 seeds)
results/exp10_.../          — A/B тест: residual/hybrid/null-importance
results/exp11_ablation/     — Derivatives A/B тест (12 конфигураций)
results/archive/            — Старые эксперименты (exp02-exp07)
```

---

## 8. Текущие задачи и план

### AI Architecture Review (10 марта 2026)
Запрошен ревью у внешней AI. Ключевые рекомендации:

#### Самое важное (делаем сейчас):
1. ⏳ **Production retrain до 2025-09 + exp11 derivatives** — модели устарели (train→2024-06), derivatives дали +42%.
2. ⏳ **LambdaRank в LGB** — objective=lambdarank, group=timestamp. Дёшево, может улучшить ранжирование top/bottom → WR↑.
3. ⏳ **Derivatives-only мини-модель** — Ridge/маленький GBDT только на OI/taker/LS/funding → декоррелированный 4-й голос в ансамбле.
4. ⏳ **Short constraints в симуляции** — 19/50 монет нельзя шортить на OKX demo. Сим должен это отражать.

#### Следующий приоритет:
5. ⏳ **Meta-model как risk scaler** (НЕ stacking) — РЕАЛИЗОВАНО в run_fast_sim.py `--meta-risk`
   - Управляет **gross exposure** (0.3x…1.5x) на основе 5 сигналов:
   - Model agreement (confidence), score spread, recent WR, DD depth, regime
   - Weighted composite → scale factor, applied ПОСЛЕ vol targeting
   - **Тестирование**: `python run_fast_sim.py --ensemble --edge-boost --meta-risk`

6. ⏳ **Vol targeting** — РЕАЛИЗОВАНО в run_fast_sim.py `--vol-target-ann 0.30`
   - leverage_t = target_vol / realized_vol_portfolio (rolling, clipped 0.2x–2.0x)
   - Дефолтный lookback = vol_lookback из risk config (50 шагов)
   - **Рекомендуемый запуск**: `--vol-target-ann 0.30 --meta-risk` (вместе)

#### Когда-нибудь потом:
7. ⬜ **Liquidation / basis / funding premium** фичи — нужны данные (CoinGlass? Coinalyze?)
8. ⬜ **On-chain market-wide** (exchange netflow BTC/ETH, stablecoin flows) — как regime filter
9. ⬜ **TFT / Transformer** — последовательный сигнал, некоррелированный с GBDT (3-5 дней)
10. ⬜ **Survivorship bias fix** — динамический universe по листингам/ликвидности
11. ⬜ **News ablation (exp12)** — 4-way с полными данными. Низкий приоритет: news вредят LGB, CB уже с news.

#### Insights из ревью:
- **Sharpe 6.61** может быть завышен survivorship bias + OKX short constraints
- **News вредят LGB** из-за leaf-wise boosting overweighting noisy splits; ordered boosting CatBoost robust
- **Gain-based feature selection** может выбрасывать стабилизаторы → перейти на permutation importance OOS
- **WR 61→65%** — менять не модель, а слой принятия решений (meta risk scaler + regime short budget)
- **Kelly criterion** — не в чистом виде (нестационарные данные), лучше fractional Kelly + clips
- **Residual target** — использовать как отдельный эксперт или компонент meta-model, не как основной target
- **Vol targeting** часто снижает DD и повышает WR больше, чем тонкая настройка DDStop

### Статус данных
- ✅ **exp11_ablation завершён** — v7_baseline DDStop 2.12 (+42% vs exp10). Подробности в секции 11.
- ✅ **News data полностью скачан** — 950,488 статей, 67/67 месяцев ≥3000 шт.
- ✅ **Binance derivatives data** — OI, taker, L/S, funding с Dec 2021 (81MB + 1.2MB).

#### Другое
- ⬜ Maker orders (экономия 33% на fees: 0.02% vs 0.03%)
- ⬜ Live trading с реальными $500 (после стабилизации)

---

## 9. Top фичи по моделям

### LGB v6/v7 (sentiment-driven)
1. fng_ma30 (Fear & Greed 30d MA)
2. fng_momentum
3. fng_ma7
4. vol_12h_cs_rank
5. close_ma720_ratio
6. close_ma336_ratio
7. btc_beta_168h
8. breadth_pct_positive
9. ret_sharpe_168h
10. fng_value

### CatBoost (price-structure-driven)
1. close_ma336_ratio
2. vol_12h_cs_rank
3. close_ma720_ratio
4. breadth_pct_positive
5. gk_vol_24h
6. fng_momentum
7. ret_sharpe_168h
8. gk_vol_168h
9. fng_ma30
10. ret_std_168h

> LGB фокусируется на sentiment (FNG), CatBoost — на price structure и volatility. Это именно то, что делает ансамбль сильнее одиночной модели.

### XGBoost — ждём результатов обучения

---

## 10. Метрики которые мы отслеживаем

| Метрика | Что значит | Хорошее значение |
|---------|-----------|------------------|
| **Rank IC** | Spearman корреляция предсказания с реальным ранговым return | > 0.025 |
| **Rank ICIR** | IC / std(IC) — стабильность IC | > 0.35 |
| **LS Sharpe net** | Sharpe long-short стратегии за вычетом costs | > 1.0 |
| **DDStop Sharpe** | Sharpe с drawdown stop (-25% → выход, -10% → вход) | > 1.5 |
| **DDStop MaxDD** | Max drawdown с DD stop | > -50% |
| **Win Rate (WR)** | % прибыльных ребалансировок | > 60% |
| **Profit Factor (PF)** | Сумма wins / сумма losses | > 1.5 |
| **Fast Sim Sharpe** | Sharpe на реальных OOS данных (365d) | > 3.0 |
| **Max DD** | Максимальная просадка в fast sim | > -10% |

---

## 11. Хронология экспериментов

| # | Дата | Эксперимент | Результат | Статус |
|---|------|-------------|-----------|--------|
| 1 | Фев 2026 | LGB v5 (4h target) | Sharpe 1.64, fast sim -8.3% | ❌ Не работает |
| 2 | Фев 2026 | HIST v1/v2 (Transformer) | LS Sharpe 2.93 в ансамбле с v5 | ⚠️ Не retrainили |
| 3 | Мар 2026 | LGB v6 (12h target, новые фичи) | DDStop Sharpe 1.81, fast sim +7.4% | ✅ В ансамбле |
| 4 | Мар 2026 | LGB v7 (blended target, HPO) | DDStop Sharpe 1.88, fast sim +12.3% | ✅ В ансамбле |
| 5 | Мар 2026 | LGB v8 (8 лет данных) | LS Sharpe 0.68, W2 провал | ❌ Отвергнуто |
| 6 | Мар 2026 | CatBoost (ordered boosting) | DDStop Sharpe 1.51, fast sim Sharpe 6.61 | ✅ В ансамбле |
| 7 | 8 Мар | News A/B тест | LGB без news + CB с news = лучший | ✅ Применено |
| 8 | 9 Мар | Full sim grid (7 конфигов) | 365d 1x 12h no filter = Sharpe 6.61 | ✅ Документировано |
| 9 | 9 Мар | XGBoost + news interactions | DDStop Sharpe 0.97 — слабее ансамбля, отложен | ⚠️ |
| 10 | 9 Мар | A/B тесты: residual-target, hybrid-norm, null-importance | News фичи снижают DDStop всех LGB на 21-60%. Подробности ниже. | ✅ Задокументировано |
| 11 | 9 Мар | Binance Futures derivatives data | Скачано: OI, taker, L/S, funding с Dec 2021 (data.binance.vision). 50 символов, 1.8M строк metrics + 294K funding. Новые фичи добавлены в pipeline. | ✅ Данные есть |
| 12 | 10 Мар | exp11_ablation — derivatives A/B | **v7_baseline DDStop 2.12 (+42% vs exp10)**, v6_res_hyb_null 1.64, CatBoost 1.54. Derivatives МАССИВНО помогают LGB. | ✅ Задокументировано |
| 13 | 10 Мар | News data gap-fill | 950k статей, 67/67 месяцев ≥3000 шт. Готово для exp12. | ✅ Данные готовы |
| 14 | 10 Мар | AI Architecture Review | Внешний AI: meta-model risk scaler, LambdaRank, derivatives-only, vol targeting, short constraints | ✅ Рекомендации записаны |
| 15 | 10 Мар | Реализация п.1-4: LambdaRank, short-blocked, derivatives-only pipeline | Код написан, тестируется на кластере (exp12) | ⏳ Ждём результаты |
| 16 | 10 Мар | Реализация п.5-6: meta-risk + vol targeting | A/B тест на старых моделях — **meta-risk Sharpe +57%** vs baseline. Подробности ниже. | ✅ Задокументировано |

### Результаты A/B теста meta-risk + vol targeting (10 марта 2026)

**Условия**: 365 дней, offline данные (crypto_features_1h.parquet), production модели (старые, без derivatives). Ensemble v6+v7+CB, edge-boost, 1x leverage, rebal=12h.

| Variant | Return | MaxDD | Sharpe | Sharpe HAC | Calmar | WR | PF | Avg meta-risk |
|---------|--------|-------|--------|-----------|--------|----|----|---------------|
| **baseline** (boost) | +34.4% | -23.1% | 0.56 | 0.55 | 1.49 | 54% | 1.06 | — |
| **+meta-risk** | +51.1% | -25.4% | **0.88** | **0.88** | **2.01** | 54% | 1.09 | 1.25x |
| **+vol-target 30%** | +27.8% | -27.7% | 0.69 | 0.68 | 1.00 | 55% | 1.07 | — |
| **+meta+vol** | +39.4% | -27.8% | **0.93** | **0.92** | 1.42 | 55% | 1.10 | 1.23x |
| **+meta+short-blocked** | +19.4% | -19.7% | 0.18 | 0.18 | 0.98 | 55% | 1.02 | 1.27x |

**Выводы**:
1. **Meta-risk — основной выигрыш**: Sharpe +57% (0.56→0.88), Return +49% (34→51%), Calmar +35%.
2. **Vol targeting** добавляет меньше (+23% Sharpe), но улучшает WR на 1pp.
3. **Вместе** дают лучший Sharpe (0.93) и WR (55%), но return ниже чем meta-risk alone.
4. **Short-blocked** сильно режет прибыль (19.4% vs 51.1%) — OKX ограничения критичны.
5. **Avg meta-risk scale ~1.25x** — система в среднем увеличивает позиции (модели уверены + рынок растёт).
6. **Осторожно**: meta-risk scale 1.25x может быть overfitted к бычьему рынку. Перепроверить на bear-market окне.

### Подробные результаты exp10 — A/B тест news features (9 марта 2026)

**Цель**: проверить, помогают ли news features разным моделям/подходам.

**Прогоны**: 12 конфигураций (1 упал — v6_lambdarank). 3 окна walk-forward, 5 seeds.

**Ключевые метрики** (DDStop Sharpe по окнам, Combined DDStop Sharpe):

| Run | Модель | Rank_IC | Rank_ICIR | DDStop W1 | DDStop W2 | DDStop W3 | DDStop AVG | Comb DDStop |
|-----|--------|---------|-----------|-----------|-----------|-----------|------------|-------------|
| **PROD v7 no-news** | **LGB** | **0.0289** | **0.4058** | **2.05** | **1.86** | **1.73** | **1.88** | **1.79** |
| **PROD v6 no-news** | **LGB** | **0.0279** | **0.4267** | — | — | **1.81** | **1.81*** | **1.81** |
| **PROD CatBoost w/news** | **CB** | **0.0258** | **0.3690** | **0.98** | **1.57** | **1.98** | **1.51** | — |
| exp10/v7_baseline | LGB | 0.0246 | 0.3920 | 1.68 | 1.05 | 1.75 | 1.49 | 1.61 |
| exp10/catboost_baseline | CB | 0.0249 | 0.3814 | 1.02 | 1.33 | 2.11 | 1.49 | 1.62 |
| exp10/v7_res_hyb_null | LGB | 0.0267 | 0.3946 | 0.91 | 1.04 | 1.34 | 1.10 | 1.12 |
| exp10/catboost_res_hyb | CB | 0.0283 | 0.3875 | 0.49 | 1.89 | 1.31 | 1.23 | 1.28 |
| exp10/v6_hybrid | LGB | 0.0275 | 0.3761 | 0.36 | 1.49 | 1.42 | 1.09 | 1.27 |
| exp10/v6_res_hyb | LGB | 0.0278 | 0.3861 | 0.59 | 1.40 | 1.11 | 1.03 | 1.16 |
| exp10/v7_res_hyb | LGB | 0.0266 | 0.3914 | 0.71 | 1.15 | 1.05 | 0.97 | 1.10 |
| exp10/xgboost_res_hyb | XGB | 0.0270 | 0.3823 | 0.49 | 1.19 | 1.23 | 0.97 | 1.09 |
| exp10/v6_res_hyb_null | LGB | 0.0276 | 0.3872 | 0.18 | 1.24 | 1.20 | 0.87 | 1.07 |
| exp10/v6_baseline | LGB | 0.0251 | 0.3777 | -0.19 | 1.06 | 1.48 | 0.78 | 1.05 |
| exp10/v6_residual | LGB | 0.0255 | 0.3925 | -0.10 | 1.31 | 0.98 | 0.73 | 0.95 |
| exp10/v6_lambdarank | LGB | — | — | — | — | — | — | FAILED |

*\* PROD v6 обучен только на W3*

**Сравнение с baseline (DDStop Sharpe avg, % деградации)**:
- exp10/v6_baseline (с news) vs PROD v6 (без news): 0.78 vs 1.81 = **−57%**
- exp10/v6_hybrid: 1.09 vs 1.81 = **−40%**
- exp10/v6_res_hyb: 1.03 vs 1.81 = **−43%**
- exp10/v6_residual: 0.73 vs 1.81 = **−60%**
- exp10/v7_baseline (с news) vs PROD v7 (без news): 1.49 vs 1.88 = **−21%**
- exp10/v7_res_hyb: 0.97 vs 1.88 = **−48%**
- exp10/catboost_baseline (с news) vs PROD CB (с news): 1.49 vs 1.51 = **−2%** (стабилен)

**Подтверждение из exp07** (архив): LGB v6 с news тогда тоже дал DDStop avg 0.96, LGB v7 с news — 1.20. Оба ниже no-news baseline.

**Провал v6_lambdarank**: `LightGBMError: label should be int type (met 0.947368) for ranking task`. Нужно квантизировать label в int.

### Подробные результаты exp11_ablation — Binance Derivatives (10 марта 2026)

**Цель**: проверить, как Binance derivatives фичи (OI, taker buy/sell, L/S ratios, funding rate) влияют на все модели.

**Изменения**: +25 фичей (163→188 для v6/CB, 160→185 для v7, 186→211 для XGB).

**Прогоны**: 11 конфигураций × 3 окна walk-forward × 5 seeds.

| Run | Модель | Rank_IC | DDStop W1 | DDStop W2 | DDStop W3 | DDStop AVG | DDStop std | Comb DDStop | Ann Ret |
|-----|--------|---------|-----------|-----------|-----------|------------|------------|-------------|--------|
| **exp11/v7_baseline** | **LGB** | **0.0243** | **1.84** | **2.14** | **2.38** | **2.12** | **0.22** | **2.18** | **49%** |
| exp11/v7_res_hyb | LGB | 0.0271 | 0.55 | 2.62 | 2.36 | 1.84 | 0.92 | 1.97 | 33% |
| exp11/v7_res_hyb_null | LGB | 0.0267 | 0.54 | 2.44 | 2.52 | 1.83 | 0.92 | 2.06 | 30% |
| exp11/v6_res_hyb_null | LGB | 0.0278 | 0.44 | 2.24 | 2.24 | 1.64 | 0.85 | 2.03 | 27% |
| exp11/v6_res_hyb | LGB | 0.0280 | -0.09 | 2.33 | 2.40 | 1.55 | 1.16 | 2.10 | 30% |
| exp11/catboost_baseline | CB | 0.0243 | 0.79 | 1.86 | 1.96 | 1.54 | 0.53 | 1.80 | 43% |
| exp11/v6_residual | LGB | 0.0256 | 0.38 | 1.93 | 2.16 | 1.49 | 0.79 | 1.89 | 34% |
| exp11/v6_hybrid | LGB | 0.0268 | 0.27 | 2.27 | 1.91 | 1.48 | 0.87 | 1.77 | 38% |
| exp11/xgboost_res_hyb | XGB | 0.0270 | -0.40 | 2.15 | 2.03 | 1.26 | 1.17 | 1.84 | 29% |
| exp11/v6_baseline | LGB | 0.0252 | -0.24 | 1.86 | 2.06 | 1.23 | 1.04 | 1.73 | 35% |
| exp11/catboost_res_hyb | CB | 0.0286 | -0.23 | 2.16 | 1.63 | 1.19 | 1.02 | 1.52 | 35% |

**Сравнение exp11 vs exp10 (DDStop Sharpe avg)**:
- v7_baseline: 2.12 vs 1.49 = **+42%** 🚀
- v7_res_hyb: 1.84 vs 0.97 = **+90%** 🚀
- v6_baseline: 1.23 vs 0.78 = **+57%** 🚀
- v6_res_hyb: 1.55 vs 1.03 = **+50%** 🚀
- catboost_baseline: 1.54 vs 1.49 = **+3%** (минимальное улучшение)
- catboost_res_hyb: 1.19 vs 1.23 = **−3%** (без изменений)

**Сравнение exp11 vs PROD (DDStop Sharpe)**:
- v7_baseline: 2.12 vs 1.88 = **+13%** — НОВЫЙ ЛУЧШИЙ РЕЗУЛЬТАТ
- v6_res_hyb_null: 1.64 vs 1.81 = −9% (пока не бьёт PROD v6)
- catboost_baseline: 1.54 vs 1.51 = +2% (стабилен)

**Выводы exp11**:
1. **v7_baseline — безоговорочный лидер**: DDStop 2.12 avg, 2.18 combined, std 0.22 (самый стабильный), 49% Ann Ret.
2. **Derivatives МАССИВНО помогают LGB** (+42-90% DDStop), но **почти не влияют на CatBoost** (+3%). CatBoost уже использовал ordered boosting для noisy features.
3. **v7_baseline бьёт PROD** (2.12 vs 1.88, +13%) — первый раз эксперимент превосходит production.
4. **W1 (→2024-12) остаётся слабым** для всех вариантов. Маленький тест-сет + ранний период.
5. **res_hyb варианты улучшают Combined DDStop** для v6 (2.10 vs 1.73), но добавляют variance.
6. **XGBoost по-прежнему слабее** (1.26) — сложные interaction features не компенсируют.

---

**Выводы exp10**:
1. **News features ВРЕДЯТ LightGBM** — DDStop Sharpe падает на 21-60%. Особенно сильно v6 (−57%).
2. **CatBoost устойчив к news** — деградация всего 2%.
3. **Residual-target и hybrid-norm улучшают Rank_IC** (+0.002-0.003), но **ухудшают DDStop** — модель лучше предсказывает ранг, но хуже зарабатывает.
4. **Null-importance фильтрация не помогает** — результат хуже или на уровне baseline.
5. **Лучшая конфигурация — текущая продакшн**: LGB v6+v7 без news + CatBoost с news.
6. **XGBoost с news interactions** — DDStop 0.97, не оправдывает усложнение.
7. **W1 (→2024-12) — самое слабое окно** для почти всех exp10 прогонов. Гипотеза: news features вносят шум в более раннем периоде.

---

## 12. Открытые вопросы

1. **Win Rate** сейчас 61%. Цель — 65-70%. Как поднять без потери Sharpe?
2. **Leverage**: 1x safe но медленно, 3x рискованно. Как найти оптимум?
3. **Production retrain**: модели обучены на данных до 2024-06. Нужно retrain на данных до 2025-09 для улучшения.
4. **Live trading**: когда переходить с demo на реальные деньги?
5. **Новые данные**: on-chain, order book — какие API использовать?
6. **Новые архитектуры**: TFT/LSTM давали бы некоррелированный сигнал, но сложнее в разработке.

---

## 13. Ключевые файлы для понимания кода

- **`run_pipeline_v6.py`** (1265 строк) — базовый pipeline, все shared функции: `add_sentiment_features`, `add_multi_horizon_targets`, `evaluate_model`, walk-forward windows, constants
- **`run_pipeline_xgboost.py`** (721 строка) — новая модель с news interaction features
- **`run_fast_sim.py`** (~1000 строк) — симулятор, загрузка ансамбля, predict_ensemble, meta-risk, vol-target
- **`run_trading.py`** (2200+ строк) — live/paper trading bot, build_features, generate_signal
- **`RESULTS.md`** (570 строк) — все результаты, таблицы, графики

---

## 14. Backlog: рекомендации из AI-ревью v2 (10 марта 2026)

### Приоритет 1 — после exp12
- [ ] **Фикс meta-risk scaler**: заменить `recent WR (10 шагов)` на EMA top-bottom spread (40-60 шагов) или Rank IC EMA — текущий WR(10) = 5 дней, чистый шум
- [ ] **Vol-target "только вниз"**: `scale = min(1, target/vol)` — не увеличивать в спокой, только резать в стресс
- [ ] **Stress cap по деривативам**: если btc_vol высока ИЛИ funding_surprise экстремален → gross ≤ 0.5-0.8x
- [ ] **Short budget режимно**: в сильном bull уменьшать шорты (уже частично есть через regime_shorts)

### Приоритет 2 — новые данные/фичи для следующего ретренинга
- [ ] **Liquidation data** (Binance): long_liq, short_liq, liq_imbalance, zscore — сильный сигнал forced flows
- [ ] **Basis / perp premium** (perp vs spot/index): zscore, mean reversion signal
- [ ] **Market-wide stress** агрегаты: aggregate OI change, aggregate taker imbalance, funding dispersion
- [ ] **Cross-exchange spread** (Binance vs OKX) по BTC/ETH как системный маркер
- [ ] **On-chain** (только BTC/ETH): exchange flows + stablecoin flows (coin-level для 50 альтов не окупается)

### Приоритет 3 — архитектура/ensemble
- [ ] **Constrained linear stacking (Ridge)**: на OOF walk-forward predictions + disagreement + режим
- [ ] **Dynamic weighting по режиму** (Mixture-of-Experts): trend → больше LGB, panic → больше CB + меньше gross
- [ ] **Derivatives-only model** как отдельный эксперт в ансамбле (декорреляция)
- [ ] **Recency weighting** при обучении: exp weights с half-life 90-180 дней
- [ ] **Rolling training window** (последние 18-30 мес) вместо expanding (2017+)

### Приоритет 4 — мониторинг/production
- [ ] **Degradation monitor**: OOS Rank IC по неделям, z-score vs история, автоалерт
- [ ] **Top-bottom spread** мониторинг (realized L/S return differential)
- [ ] **Ensemble correlation tracking**: если модели начинают ошибаться синхронно → алерт
- [ ] **Unified backtester**: один source of truth с OKX-ограничениями (blocked shorts), одинаковый cost model
- [ ] **Production kill-switch**: max daily loss, max intraday DD, max leverage hard cap

### Красные флаги из ревью (помнить)
- Survivorship bias (50 монет из 2025 на данных 2021+) — остаётся
- Short constraints (19/50 blocked) — симулятор должен это учитывать
- Binance features → OKX execution: в стресс расхождения могут расти
- Kelly criterion НЕ рекомендуется (нестационарный edge → переплечо)
