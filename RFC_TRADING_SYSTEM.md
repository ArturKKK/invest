# RFC: AI-Powered Quantitative Trading System

**Автор:** A.S. Tabakov  
**Дата:** 2026-03-07  
**Статус:** Draft  

---

## 1. Executive Summary

Строим production-grade систему для заработка на фондовом рынке, используя SOTA deep learning подходы. На основе проведённого research выбираем **Microsoft Qlib** как фреймворк-основу (38k+ stars, MIT, поддержка Microsoft Research Asia) и комбинацию из трёх SOTA архитектур:

1. **MASTER** (AAAI 2024) — Market-Guided Stock Transformer, адаптированный под крипто
2. **HIST** (KDD 2021) — Hidden State Transformer для моделирования межтокеновых связей  
3. **RL-агент (PPO/SAC)** — для оптимизации исполнения ордеров и portfolio management
4. **FinBERT Sentiment (Phase 2)** — новостной sentiment как дополнительный фактор

**Рынок:** Crypto (Binance)  
**Стиль:** Intraday (1h ребалансировка)  
**Биржа:** OKX (основная), Bybit (запасная) — доступны для резидентов РФ  
**Капитал:** $100-1,000 (spot only, без leverage)  
**Ожидаемый результат:** Sharpe > 1.5, Annualized Return > 30%, Max Drawdown < 20%.

---

## 2. Почему именно этот подход — обоснование

### 2.1. Что говорят бенчмарки Qlib (CSI300, 20 runs)

Из официальных бенчмарков Microsoft Qlib на датасете **Alpha360** (raw price/volume features):

| Модель | IC | ICIR | Rank IC | Rank ICIR | Ann. Return | Info Ratio | Max DD |
|--------|-------|-------|---------|-----------|-------------|------------|--------|
| LightGBM | 0.040 | 0.304 | 0.050 | 0.404 | 5.6% | 0.763 | -6.6% |
| LSTM | 0.045 | 0.347 | 0.055 | 0.437 | 6.5% | 0.896 | -8.8% |
| GRU | 0.049 | 0.377 | 0.058 | 0.464 | 7.2% | 0.973 | -8.2% |
| GATs | 0.048 | 0.351 | 0.060 | 0.460 | 8.2% | 1.108 | -8.9% |
| ALSTM | 0.050 | 0.383 | 0.060 | 0.474 | 6.3% | 0.865 | -9.9% |
| TRA (KDD 2021) | 0.049 | 0.379 | 0.059 | 0.476 | 9.2% | 1.279 | -8.3% |
| IGMTF | 0.048 | 0.359 | 0.061 | 0.477 | 9.5% | 1.351 | -7.2% |
| **HIST** | **0.052** | **0.353** | **0.067** | **0.458** | **9.9%** | **1.373** | **-6.8%** |

**Вывод:** Deep learning модели (HIST, TRA, IGMTF) значительно превосходят gradient boosting на raw features. HIST — лидер по Annualized Return и Information Ratio.

На датасете **Alpha158** (engineered features / 158 факторов):

| Модель | Ann. Return | Info Ratio | Max DD |
|--------|-------------|------------|--------|
| LightGBM | 9.0% | 1.016 | -10.4% |
| **DoubleEnsemble** | **11.6%** | **1.343** | **-9.2%** |
| TRA | 7.2% | 1.084 | -7.6% |

**Вывод:** На feature-engineered данных ensemble-методы (DoubleEnsemble на базе LightGBM) конкурентоспособны. Оптимальная стратегия — **комбинировать оба подхода**.

### 2.2. Почему MASTER (AAAI 2024) — upgrade над HIST

**MASTER** (MArkert-Guided Stock TransformER) решает 2 ключевые проблемы предыдущих моделей:

1. **Momentary & cross-time корреляции** — акции коррелируют не только в один момент времени, но и с лагами. MASTER чередует intra-stock и inter-stock attention.
2. **Dynamic feature selection** — рыночный режим (бычий/медвежий) влияет на то, какие признаки важны. MASTER использует market information для автоматического выбора фич.

Результат: MASTER превосходит HIST в экспериментах авторов на CSI300 и CSI800.

### 2.3. Почему не чистый RL

RL-подходы (FinRL и подобные) хороши для:
- Order execution (как именно исполнять заявку, минимизируя market impact)
- Portfolio rebalancing (как распределять веса между активами)

Но **НЕ** для prediction (предсказание куда пойдёт цена) — supervised learning здесь объективно сильнее по бенчмаркам.

**Наш подход: комбинация** — supervised model предсказывает alpha, RL-агент оптимизирует execution.

---

## 3. Архитектура системы

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRADING SYSTEM                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  DATA LAYER  │───▶│ ALPHA LAYER  │───▶│ EXECUTION LAYER  │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│         │                    │                     │            │
│         ▼                    ▼                     ▼            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ Qlib Data    │    │ Multi-model  │    │ Portfolio Opt    │  │
│  │ Server       │    │ Ensemble     │    │ + RL Executor    │  │
│  │              │    │              │    │                  │  │
│  │ • Yahoo/API  │    │ • MASTER     │    │ • TopK Strategy  │  │
│  │ • MOEX data  │    │ • HIST       │    │ • Risk Mgmt      │  │
│  │ • Alt data   │    │ • DoubleEns  │    │ • PPO executor   │  │
│  │ • News/NLP   │    │ • LightGBM   │    │ • Slippage ctrl  │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  ADAPTATION LAYER                         │  │
│  │  • DDG-DA (concept drift detection & adaptation)          │  │
│  │  • Rolling retraining with walk-forward validation        │  │
│  │  • Online model update pipeline                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  MONITORING & BACKTEST                     │  │
│  │  • Sharpe / Sortino / Calmar ratios                       │  │
│  │  • Max Drawdown tracking                                  │  │
│  │  • IC / ICIR monitoring                                   │  │
│  │  • Position & risk exposure dashboard                     │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Компоненты — детально

### 4.1. Data Layer

#### Источники данных

| Источник | Данные | Частота | Назначение |
|----------|--------|---------|------------|
| **Binance API** | Crypto OHLCV + order book | 1h, 15min, 1min | Исторические данные (бесплатно без аккаунта) |
| **OKX API** | Crypto OHLCV + order book | 1h, 15min, 1min | **Основной источник + execution** |
| **Bybit API** | Crypto OHLCV | 1h, 15min | Запасная биржа |
| **Binance Funding Rate** | Perpetual futures funding | 8h | Sentiment фьючерсного рынка |
| **CoinGecko / CoinMarketCap** | Market cap, dominance | 1d | Кросс-секционные фичи |
| **Glassnode / CryptoQuant** | On-chain метрики | 1d | Active addresses, exchange flows |
| **CryptoPanic / NewsAPI** | Крипто-новости | Realtime | Sentiment (Phase 2, FinBERT) |
| **FRED** | DXY, ставки ФРС | 1d | Macro regime detection |
| **Fear & Greed Index** | Рыночный sentiment | 1d | Дополнительный фактор |

#### Feature Engineering — Alpha158+ расширенный набор

**Базовые (Alpha360 — 6 × 60 = 360 фич):**
- OHLCV нормализованные: `close/close_lag_1`, `high/low`, `volume/volume_ma_20`, etc.
- Окна: 5, 10, 20, 30, 60 дней

**Расширенные (Alpha158+):**
- Технические индикаторы: RSI(6,12,24), MACD, Bollinger Bands, ATR, OBV, ADX, CCI, Stochastic
- Volume profile: VWAP deviation, volume momentum, order flow imbalance
- Volatility: realized vol, Garman-Klass, Parkinson, rolling skewness/kurtosis
- Cross-sectional: rank momentum, industry-relative strength, market cap quintile
- Macro regime: VIX level, yield curve slope, credit spreads (feature selection via MASTER)

**On-chain Features (крипто-специфичные):**
- Exchange inflow/outflow (крупные переводы на биржу = давление на продажу)
- Active addresses momentum
- Funding rate (positive = перегретый long, negative = перегретый short)
- Open interest changes
- Whale transaction count
- Stablecoin supply ratio (SSR)

**NLP/Sentiment Features (Phase 2 — добавляем после базовой модели):**
- FinBERT sentiment score от крипто-новостей (CryptoPanic feed)
- Aggregated sentiment за 1h / 6h / 24h окна
- News volume spike detection (аномальное кол-во новостей = событие)
- Social media sentiment (Reddit, Twitter/X crypto mentions)
- Fear & Greed Index как фича

**Почему sentiment во Phase 2, а не сразу:**
Цена уже содержит ~80% информации из новостей. Добавление sentiment улучшит модель на 5-15%, но утроит сложность pipeline. Сначала добиваемся profit на чистых ценовых данных, потом усиливаем.

**Всего: ~400-600 фич перед feature selection (Phase 1), ~700-900 фич (Phase 2 с NLP).**

### 4.2. Alpha Layer — Multi-Model Ensemble

#### Model 1: MASTER (Primary — Transformer)

```
Architecture:
  - Input: [batch, num_stocks, seq_len, features]
  - Intra-stock Temporal Attention (capture patterns within each stock)
  - Inter-stock Cross-Attention (capture momentary & cross-time correlations)
  - Market-guided Gating (dynamic feature selection based on market state)
  - Output: predicted return ranking for each stock

Hyperparameters:
  - d_model: 128
  - n_heads: 8
  - n_layers: 4 (alternating intra/inter)
  - seq_len: 60 days
  - dropout: 0.1
  - lr: 1e-4 (cosine annealing)
  - batch_size: динамический (all stocks per day)

Training:
  - Loss: IC-based loss (maximize information coefficient)
  - Optimizer: AdamW with gradient clipping
  - GPU: A100 — estimated 2-4 hours per full training run
```

#### Model 2: HIST (Secondary — Transformer)

```
Architecture:
  - Predefined Concept Module (industry, sector features)
  - Hidden Concept Module (learned stock relationships)
  - Temporal attention + cross-stock attention

Hyperparameters:
  - d_model: 128
  - n_concepts: 10 hidden concepts
  - seq_len: 60

Training:
  - Loss: MSE on excess return
  - GPU: A100 — ~1-2 hours
```

#### Model 3: DoubleEnsemble + LightGBM (Fallback)

```
Architecture:
  - LightGBM base learners
  - DoubleEnsemble: sample reweighting + feature selection
  - Works on Alpha158 tabular features (no sequence modeling)

Hyperparameters:
  - n_estimators: 2000
  - max_depth: 8
  - learning_rate: 0.05
  - feature_fraction: 0.7
  - bagging_fraction: 0.8

Training:
  - CPU-only, ~5 minutes
  - Стабильный baseline без GPU зависимости
```

#### Ensemble Strategy

```python
# Weighted ensemble based on rolling IC performance
alpha_final = (
    w_master * alpha_master +    # ~0.45 (primary signal)
    w_hist   * alpha_hist   +    # ~0.35 (secondary signal)  
    w_lgbm   * alpha_lgbm        # ~0.20 (stability anchor)
)

# Weights are dynamically adjusted each month based on:
# - Rolling 20-day IC of each model
# - Correlation between model predictions (diversity bonus)
# - Model stability (lower IC variance → higher weight)
```

### 4.3. Execution Layer

#### Portfolio Construction — TopK + Risk Parity

```
Strategy:
  1. Rank all stocks by alpha_final
  2. Long: top K stocks (K = 30-50)
  3. Short: bottom K stocks (if allowed; neutral otherwise)
  4. Weight by: alpha score × inverse volatility (risk parity)
  5. Constraints:
     - Max single position: 5% of portfolio
     - Max sector exposure: 25%
     - Max turnover per day: 20% (control transaction costs)
     - Target volatility: 15% annualized
```

#### RL Order Execution (Phase 2)

```
Agent: PPO/SAC
State: [order_remaining, time_remaining, spread, volume_profile, volatility]
Action: fraction of order to execute this timestep
Reward: -execution_cost (slippage + market impact)

Purpose: для крупных заказов (>1% ADV) — минимизация market impact
```

### 4.4. Adaptation Layer — DDG-DA

Рынок непостоянен (concept drift). DDG-DA (AAAI 2022) решает это:

```
1. Data-driven Distribution Generation (DDG):
   - Meta-model определяет текущий рыночный режим
   - Генерирует "виртуальный" датасет, похожий на текущее распределение
   
2. Domain Adaptation (DA):
   - Переобучает модель на комбинации исторических и сгенерированных данных
   - Fine-tune каждые 1-4 недели

3. Rolling Retraining (baseline):
   - Полное переобучение модели каждый квартал
   - Walk-forward: train → validate → test сдвигается каждый месяц
```

---

## 5. Данные и объём датасета

### 5.1. Минимальный датасет для production-quality модели

| Параметр | Рекомендация |
|----------|-------------|
| **Количество инструментов** | 300-800 акций (CSI300/500, S&P500, или MOEX + крипто) |
| **Исторический период** | 10-15 лет daily (2010-2025) |
| **Количество строк** | ~500 stocks × 3750 days ≈ **1.9M строк** |
| **Количество фич** | 360-800 per row |
| **Размер данных** | ~2-5 GB (raw), ~10-20 GB (с features) |
| **Test period** | 2023-2025 (walk-forward OOS) |
| **Validation period** | 2021-2022 |
| **Training period** | 2010-2020 |

### 5.2. Наш основной датасет — Crypto (Binance)

| Параметр | Рекомендация |
|----------|-------------|
| **Инструменты** | Top 50-100 койнов по ликвидности на Binance (BTC, ETH, SOL, BNB, ...) |
| **Период** | 3-5 лет (2021-2026), BTC/ETH доступны с 2017 |
| **Частота** | **1h** (основная) + 15min/4h для multi-timeframe features |
| **Строки** | 80 coins × 35,000 hours ≈ **2.8M строк** |
| **Размер** | ~3-5 GB с features |
| **Плюсы** | 24/7, нет выходных, высокая волатильность, бесплатные данные, старт с $100 |

**Почему 1h а не 1min/5min (не скальпинг):**
- На минутках побеждает latency (HFT-фирмы с colocation) — мы проиграем
- 1h свечи: достаточно сигнала, терпимые комиссии, ML-модели работают хорошо
- 1 свеча = 1 решение. 24 решения в день × 80 монет = 1920 сигналов/день
- Комиссия 0.1% × ~5-10 сделок/день = терпимо при 0.3-1% дневной доходности

### 5.3. Скачивание данных (Binance — всё бесплатно)

```bash
# Установка
pip install python-binance pandas ccxt

# Скачивание исторических 1h свечей для всех пар
# Binance хранит данные с момента листинга
# BTC/USDT доступен с 2017, большинство альтов с 2019-2021

# Через ccxt (универсальный интерфейс для 100+ бирж):
import ccxt
binance = ccxt.binance()
ohlcv = binance.fetch_ohlcv('BTC/USDT', '1h', limit=1000)

# Или через официальный python-binance:
from binance.client import Client
client = Client()  # API key не нужен для исторических данных
klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1HOUR, "1 Jan 2021")
```

---

## 6. Обучение — время и ресурсы

### 6.1. На A100 (80GB)

| Компонент | Время 1 run | Полный pipeline (20 seeds) |
|-----------|------------|---------------------------|
| MASTER | 2-4 часа | ~2-3 дня |
| HIST | 1-2 часа | ~1-2 дня |
| DoubleEnsemble/LightGBM | 5 мин (CPU) | 2 часа |
| RL executor | 4-8 часов | ~1 неделя |
| Full backtest | 30 мин | 10 часов |
| **HPO (Optuna, 100 trials)** | ~1 неделя | N/A |

### 6.2. Полный цикл разработки

Все модели (HIST, TRA, LightGBM, DoubleEnsemble, etc.) уже реализованы в Qlib и запускаются через yaml-конфиг одной командой. MASTER — отдельный open-source репо с готовым кодом. Писать модели с нуля НЕ нужно.

| Этап | Время | Описание |
|------|-------|----------|
| **Phase 0: Setup** | 1-2 часа | `pip install pyqlib`, скачать данные через CLI |
| **Phase 1: Baseline** | 10-30 мин | `qrun` LightGBM + Alpha158, первый backtest |
| **Phase 2: HIST** | 2-4 часа | `qrun` HIST на Alpha360 (A100), сравнить с baseline |
| **Phase 3: MASTER** | 2-4 часа | Запуск MASTER, интеграция в pipeline |
| **Phase 4: Ensemble** | 1-2 часа | Объединение моделей, финальный backtest |
| **Phase 5: HPO (опционально)** | 1-3 дня | Optuna перебор гиперпараметров для каждой модели |
| **Phase 6: Paper trading** | 1-3 мес | Торговля на бумаге, проверка в реальном времени |
| **ИТОГО до первых backtest результатов** | **~1-2 дня** | |
| **ИТОГО до paper trading** | **~3-5 дней** | |

---

## 7. Валидация — как проверить что система работает

### 7.1. Anti-Overfitting Protocol

**Это САМАЯ важная часть.** Большинство "прибыльных" моделей оказываются overfit.

```
ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА:

1. НИКОГДА не смотреть на test период до финальной оценки
2. Walk-forward validation ТОЛЬКО:
   - Train: [t-5y, t-1y]
   - Val:   [t-1y, t]
   - Test:  [t, t+6m]
   - Сдвигаем окно каждые 3-6 месяцев
   
3. Множественное тестирование:
   - 20 runs с разными seeds → берём mean ± std
   - Если std > 50% от mean → модель нестабильна, не торгуем
   
4. Economy of hypothesis:
   - Deflated Sharpe Ratio (учитываем сколько backtest'ов мы запустили)
   - Minimum Backtest Length (MBL) — минимум 2 года OOS
   
5. Reality checks:
   - Transaction costs: 0.1-0.3% per trade (включая spread + commission)
   - Slippage model: ±1 tick для акций, ±0.05% для крипты
   - No look-ahead bias: фичи T используют данные только ≤ T-1
   - Survivorship bias: включаем delisted/bankrupt акции
```

### 7.2. Ключевые метрики

| Метрика | Минимум для торговли | Хорошо | Отлично |
|---------|---------------------|--------|---------|
| **Sharpe Ratio** (net of costs) | > 1.0 | > 1.5 | > 2.0 |
| **Information Ratio** | > 0.8 | > 1.2 | > 1.5 |
| **Max Drawdown** | < 15% | < 10% | < 5% |
| **Win Rate** (monthly) | > 55% | > 60% | > 65% |
| **Profit Factor** | > 1.3 | > 1.5 | > 2.0 |
| **IC** (daily) | > 0.03 | > 0.05 | > 0.07 |
| **ICIR** | > 0.3 | > 0.4 | > 0.5 |
| **Calmar Ratio** | > 0.5 | > 1.0 | > 2.0 |
| **Turnover** (daily) | < 30% | < 20% | < 10% |

### 7.3. Staged Deployment

```
Stage 1: Backtest (historical data)
  → Пройти все метрики выше
  → Minimum 2 года OOS с Sharpe > 1.0

Stage 2: Paper Trading (live data, fake money)  
  → 1-3 месяца без real money
  → Проверить что live performance ≈ backtest performance
  → Допустимая деградация: Sharpe drop не более 30%

Stage 3: Small Capital ($1K-10K)
  → 1-3 месяца с минимальным капиталом
  → Проверить execution quality, slippage реальный

Stage 4: Scale Up
  → Увеличиваем капитал на 2x каждый месяц
  → при условии что метрики стабильны
  → Hard stop: если drawdown > 15% — останавливаем, анализируем
```

---

## 8. Tech Stack

```yaml
Framework:       Microsoft Qlib (core platform)
Language:        Python 3.10+
Deep Learning:   PyTorch 2.x (for MASTER/HIST)
Boosting:        LightGBM, XGBoost
RL:              Stable-Baselines3 / ElegantRL
Data:            Binance API + CCXT + custom collectors → Qlib format
HPO:             Optuna / Ray Tune
Experiment:      MLflow / Qlib Recorder
Backtesting:     Qlib built-in + vectorbt (comparison)
Visualization:   Plotly + Qlib analysis tools
Scheduling:      Prefect / cron (для hourly prediction pipeline)
Broker API:      CCXT + OKX SDK (основная биржа) + Bybit SDK (запасная)
Sentiment:       FinBERT + CryptoPanic API (Phase 2)
Infra:           A100 GPU, Docker
```

### Requirements

```
# Core
pyqlib>=0.9.6
torch>=2.0
lightning>=2.0
lightgbm>=4.0
xgboost>=2.0

# Data
yfinance
python-binance
moexalgo
requests

# RL
stable-baselines3
gymnasium

# Optimization
optuna
ray[tune]

# Analysis  
vectorbt
plotly
mlflow

# NLP (Phase 2)
transformers
finbert-embedding
```

---

## 9. Риски и митигации

| Риск | Вероятность | Impact | Митигация |
|------|------------|--------|-----------|
| **Overfitting** | Высокая | Критический | Walk-forward, multiple seeds, deflated Sharpe |
| **Concept drift** | Высокая | Высокий | DDG-DA, rolling retraining, regime detection |
| **Data quality** | Средняя | Высокий | Cross-validate sources, health checks |
| **Execution gap** (backtest ≠ live) | Средняя | Высокий | Conservative slippage model, paper trading phase |
| **Crowded signals** | Средняя | Средний | Alternative data (NLP), unique features, multi-horizon |
| **Black swan** | Низкая | Критический | Hard stop-loss, max drawdown circuit breaker, hedging |
| **Infrastructure failure** | Низкая | Средний | Redundancy, fail-safe flat positions |

---

## 10. План действий — что делаем прямо сейчас

### Неделя 1: Foundation
```
[ ] Установить Qlib, скачать данные (US/CN или Crypto)
[ ] Запустить LightGBM baseline за 1 час
[ ] Получить первые IC / backtest числа
[ ] Понять структуру данных и pipeline
```

### Неделя 2-3: HIST Implementation
```
[ ] Запустить HIST из Qlib model zoo
[ ] HPO: lr, d_model, seq_len, dropout
[ ] Сравнить с LightGBM baseline
[ ] Walk-forward validation на 2023-2025
```

### Неделя 3-4: MASTER Implementation
```
[ ] Имплементировать MASTER (код доступен на GitHub)
[ ] Интеграция в Qlib pipeline
[ ] Тренировка на A100, HPO
[ ] Сравнение с HIST и baseline
```

### Неделя 5: Ensemble + Adaptation
```
[ ] Построить ensemble из 3 моделей
[ ] Имплементировать DDG-DA
[ ] Rolling retraining pipeline
[ ] Финальный backtest на полном OOS периоде
```

### Неделя 6-8: Paper Trading + Analysis
```
[ ] Подключить к broker API (paper trading mode)
[ ] Daily pipeline: data → predict → trade
[ ] Мониторинг метрик в реальном времени
[ ] Если стабильно → переход к real money
```

---

## 11. References

### Papers
1. **MASTER**: Tong Li et al. "MASTER: Market-Guided Stock Transformer for Stock Price Forecasting" — AAAI 2024. [arXiv:2312.15235](https://arxiv.org/abs/2312.15235)
2. **HIST**: Wentao Xu et al. "HIST: A Graph-based Framework for Stock Trend Forecasting via Mining Concept-Oriented Shared Information" — KDD 2021
3. **TRA**: Hengxu Lin et al. "Learning Multiple Stock Trading Patterns with Temporal Routing Adaptor and Optimal Transport" — KDD 2021
4. **DoubleEnsemble**: Chuheng Zhang et al. "DoubleEnsemble: A New Ensemble Method Based on Sample Reweighting and Feature Selection for Financial Data Analysis" — ICDM 2020
5. **DDG-DA**: Wendi et al. "DDG-DA: Data Distribution Generation for Predictable Concept Drift Adaptation" — AAAI 2022
6. **Qlib**: Xiao Yang et al. "Qlib: An AI-oriented Quantitative Investment Platform" — [arXiv:2009.11189](https://arxiv.org/abs/2009.11189)
7. **FinRL**: Xiao-Yang Liu et al. "FinRL: Deep Reinforcement Learning Framework to Automate Trading" — ICAIF 2021

### Code Repositories
- Microsoft Qlib: https://github.com/microsoft/qlib (38k+ ⭐)
- FinRL: https://github.com/AI4Finance-Foundation/FinRL (14k+ ⭐)
- MASTER: https://github.com/SJTU-Quant/MASTER
- RD-Agent: https://github.com/microsoft/RD-Agent (LLM-driven auto quant research)

---

## 12. TL;DR

**Что строим:** Multi-model ensemble (MASTER + HIST + LightGBM) на платформе Qlib с RL execution и адаптацией к market drift.

**Почему это лучший путь:**
- MASTER/HIST — признанный SOTA на бенчмарках (AAAI/KDD top-tier conferences)
- Qlib — production-ready платформа от Microsoft с полным пайплайном
- Ensemble даёт стабильность, которой нет у одиночных моделей
- DDG-DA решает проблему concept drift (почему модели "умирают" через месяцы)

**Рынок:** Crypto (OKX — основная биржа), 1h intraday, top 50-100 койнов

**Капитал:** $100-1,000 (spot, без leverage)

**Сколько данных:** ~2.8M строк (80 койнов × 35K часов), скачивается бесплатно через Binance/OKX API

**Сколько обучение:** 2-4 часа на A100 за один run, ~3 дня на полный HPO cycle

**Новости/Sentiment:** Phase 2 (после базовой модели). FinBERT + CryptoPanic. Добавит ~5-15% улучшения.

**Когда первые деньги:** backtest за 1-2 дня, paper trading ~1 мес, real money с $100-1K

**Главный риск:** Overfitting. Борьба с ним — 70% всей работы.
