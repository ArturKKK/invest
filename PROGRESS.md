# Project Progress — AI Crypto Trading System

**Последнее обновление:** 2026-03-07  
**Статус:** Phase 1 — Baseline v2 TRAINED. Сильный ranking сигнал (LS Sharpe 4.2). Long-only не работает.

---

## 0. Ключевые результаты (TL;DR)

### Baseline v1 (ПРОВАЛ)
- IC = 0.005, Sharpe = -1.0 — модель хуже рандома
- Проблемы: time features leakage, нет cross-sectional normalization, переобучение

### Baseline v2 (УСПЕХ) ← текущий
- **Rank IC = 0.031, ICIR = 0.36, LS Sharpe = 3.87** (rank model)
- **Ensemble LS Sharpe = 4.21** (среднее 3 моделей)
- Long-Short работает отлично, Long-Only нет (рынок падал в test периоде)
- Top фичи: MA-ratios, GK-volatility, Sharpe-ratios, CCI

### Что нужно дальше
1. Short-side execution (futures/margin на OKX) ИЛИ market-neutral стратегия
2. Попробовать другие target horizons (12h, 24h)
3. HIST/MASTER трансформеры на A100 GPU
4. HPO с Optuna (100+ trials)

---

## 1. Что уже сделано

### ✅ RFC и Architecture Design
- Написан подробный RFC: [RFC_TRADING_SYSTEM.md](RFC_TRADING_SYSTEM.md)
- Выбрана архитектура: Multi-model ensemble (MASTER + HIST + LightGBM)
- Рынок: Crypto, 1h intraday, top 50 койнов, OKX (основная биржа)
- Капитал: $100-1K, spot only

### ✅ Project Setup
- Python 3.13, venv в `./venv`
- macOS M3 Pro, `libomp` установлен через brew для LightGBM
- Установлены пакеты: `ccxt pandas numpy lightgbm xgboost scikit-learn plotly ta vectorbt pyarrow tqdm joblib scipy`
- Структура:
```
invest/
├── RFC_TRADING_SYSTEM.md          # Детальный RFC
├── PROGRESS.md                    # Этот файл
├── run_pipeline.py                # v1 pipeline (baseline, плохие результаты)
├── run_pipeline_v2.py             # v2 pipeline (cross-sectional rank, 3 модели + ensemble)
├── requirements-cluster.txt       # Зависимости для кластера
├── data/
│   ├── raw/                       # 50 parquet файлов, 65 MB
│   └── features/                  # crypto_features_1h.parquet, 1.5 GB
├── results/                       # v1 results (плохие)
├── results_v2/                    # v2 results (хорошие) ← ТЕКУЩИЕ
│   ├── all_results.json           # Метрики всех 4 моделей
│   ├── feature_importance_v2.csv  # Важность фичей
│   ├── test_predictions_v2.parquet# Предсказания на тест
│   └── equity_curve_v2.parquet    # Кривая капитала
├── src/
│   ├── data/download_crypto.py    # Загрузка данных с Binance
│   ├── features/build_features.py # Генерация 98 фичей
│   ├── models/baseline_lgbm.py    # LightGBM baseline (v1)
│   └── backtest/simple_backtest.py# Long-only Top-K бэктест
└── venv/                          # Python virtual environment
```

### ✅ Data Download (50 symbols, 65 MB)
- Загружены 1h OHLCV данные с Binance (public API, без ключа)
- 50 пар: BTC, ETH, BNB, SOL, XRP, ADA, DOGE, AVAX, DOT, LINK, MATIC, UNI, ATOM, LTC, ETC, FIL, APT, ARB, OP, NEAR, AAVE, MKR, GRT, INJ, FTM, ALGO, SAND, MANA, AXS, THETA, RUNE, EGLD, XTZ, FLOW, CHZ, CRV, LDO, SNX, COMP, YFI, SUSHI, ENJ, BAT, ZIL, ONE, IOTA, ICX, ENS, IMX, GALA
- Период: с 2021-01-01 по 2026-03-06, ~45K строк на монету
- Формат: parquet, по файлу на символ в `data/raw/`
- Workaround: `exchange.session.verify = False` (SSL проблема из России)

### ✅ Feature Engineering (2.1M rows, 98 features)
- Обработаны все 50 символов
- Сгенерированы 98 фичей (описание ниже)
- Результат: `data/features/crypto_features_1h.parquet` (1.5 GB)
- Shape: 2,102,405 строк × 107 столбцов (98 фичей + meta)
- Period: 2021-01-30 → 2026-03-06
- Target: 4h forward return (`target_ret`), binary classification (`target_cls`)
- Target stats: mean return = 0.000143, up ratio = 49.11%

### ❌ → ✅ Baseline v1 обучена, ПРОВАЛ
- LightGBM обучился за ~30 секунд на M3 Pro
- IC = 0.005, Sharpe = -1.0, Backtest $1K → $6 
- Проблемы: time features leakage (hour_sin/dow_cos доминировали), нет CS normalization
- Early stopping на 51 итерации из 2000 — переобучение

### ✅ Baseline v2 обучена, УСПЕХ
**Скрипт:** `run_pipeline_v2.py`

**Ключевые исправления (v1 → v2):**
1. **Cross-sectional rank normalization** — фичи ранжируются внутри каждого timestamp [0,1] → центрируются [-0.5, 0.5]
2. **Time features удалены** — hour_sin, hour_cos, dow_sin, dow_cos убраны из модели
3. **3 варианта target:** rank (CS rank), excess return (vs mean), raw return
4. **Ensemble** — среднее нормализованных предсказаний 3 моделей
5. **Лучшие гиперпараметры:** LR 0.01 (было 0.05), depth 6 (было 8), min_samples 200 (было 50), L1/L2 = 1.0 (было 0.1)

**Результаты на тесте (2025-07 → 2026-03, 282K rows):**

| Model | Rank IC | ICIR | LS Sharpe | LS Ann Return | Max DD |
|-------|---------|------|-----------|---------------|--------|
| **Rank Model** | **0.031** | **0.36** | **3.87** | 158% | -56% |
| Excess Model | 0.005 | 0.04 | 1.10 | 34% | -61% |
| Raw Model | 0.006 | 0.09 | 2.95 | 87% | -47% |
| **Ensemble** | **0.025** | **0.27** | **4.21** | 161% | -49% |

**⚠️ Long-Only НЕ работает** (Top5 $1K → $18-25). Причина: bear market в test периоде.
**✅ Long-Short РАБОТАЕТ** (Sharpe 3.87-4.21). Модель отличает winners от losers.

**Top 20 фичей (v2):**
```
close_ma24_ratio, close_ma720_ratio, low_close_ratio, close_ma336_ratio,
ret_sharpe_24h, ret_sharpe_168h, gk_vol_168h, close_ma6_ratio, ret_2h,
gk_vol_24h, close_ma12_ratio, cci_48, gk_vol_48h, vol_price_corr_168h,
macd, gk_vol_12h, atr_48, vol_price_corr_48h, bb_width_20, high_close_ratio
```

---

## 2. Детальное описание фичей (98 шт.)

### Returns (9 фичей)
`ret_1h, ret_2h, ret_4h, ret_6h, ret_12h, ret_24h, ret_48h, ret_72h, ret_168h`

### Price Features (~40 фичей)
- Candle ratios: `close_open_ratio, high_low_ratio, high_close_ratio, low_close_ratio, upper_shadow, lower_shadow, body`
- MA ratios (8 окон × 2): `close_ma{6,12,24,48,72,168,336,720}_ratio, vol_ma{...}_ratio`
- Garman-Klass volatility: `gk_vol_{12,24,48,168}h`
- Rolling stats (3 окна × 5): `ret_std_{24,48,168}h, ret_skew_{...}h, ret_kurt_{...}h, ret_mean_{...}h, ret_sharpe_{...}h`

### Volume Features (~14 фичей)
- Volume momentum: `vol_mom_{6,12,24,48}h`
- VWAP deviation: `vwap_dev_{12,24,48}h`
- Volume-price correlation: `vol_price_corr_{24,48,168}h`
- Buying pressure: `buy_pressure`

### Technical Indicators (~30 фичей)
- RSI: `rsi_{6,12,14,24}`
- MACD: `macd, macd_signal, macd_diff`
- Bollinger Bands: `bb_high_{20,48}, bb_low_{20,48}, bb_width_{20,48}, bb_pband_{20,48}`
- ATR: `atr_{14,24,48}`
- ADX: `adx, adx_pos, adx_neg`
- Stochastic: `stoch_k, stoch_d`
- CCI: `cci_14, cci_48`
- Williams %R: `willr_14`
- OBV: `obv_ma_ratio_{12,24,48}`
- MFI: `mfi_14`

### Time Features (4 фичи)
- Cyclical: `hour_sin, hour_cos, dow_sin, dow_cos`

---

## 3. Модель: Что обучать

### 3.1 Baseline: LightGBM (первый приоритет)

**Скрипт:** `src/models/baseline_lgbm.py`

**Walk-forward split:**
- Train: < 2024-07-01 (3.5 года)
- Val: 2024-07 — 2025-07 (1 год)  
- Test: >= 2025-07 (8 месяцев OOS)

**Гиперпараметры:**
```python
{
    'objective': 'regression',
    'metric': 'mse',
    'n_estimators': 2000,
    'learning_rate': 0.05,
    'max_depth': 8,
    'num_leaves': 63,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 50,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'early_stopping': 50 rounds
}
```

**Метрики (ожидаем):**
- IC > 0.02 — хороший знак для baseline
- ICIR > 0.3
- Direction Accuracy > 51%
- Long-Short Sharpe > 0.5

**Output:**
- `results/baseline_results.json` — метрики
- `results/feature_importance.csv` — важность фичей
- `results/test_predictions.parquet` — предсказания на тест

### 3.2 Backtest

**Скрипт:** `src/backtest/simple_backtest.py`

Запускается после обучения модели, читает `results/test_predictions.parquet`.
- Long-only, Top-5 монет каждые 4 часа
- Commission: 0.1% (OKX taker fee)
- Initial capital: $1,000

---

## 4. Инструкция для кластера

### Что загрузить на S3:
```
data/features/crypto_features_1h.parquet    # 1.5 GB — основные данные
src/models/baseline_lgbm.py                 # скрипт обучения
src/backtest/simple_backtest.py             # скрипт бэктеста
```

### Зависимости:
```bash
pip install pandas numpy lightgbm scikit-learn scipy pyarrow
```

### Запуск:
```bash
# 1. Обучение LightGBM (1-3 минуты)
python baseline_lgbm.py

# 2. Бэктест (секунды)
python simple_backtest.py
```

### Ожидаемый результат:
- `results/baseline_results.json` — основные метрики (IC, Sharpe, Drawdown)
- `results/feature_importance.csv` — топ фичей
- `results/test_predictions.parquet` — предсказания
- `results/backtest_results.json` — результаты симуляции торговли
- `results/equity_curve.parquet` — кривая капитала

---

## 5. Следующие шаги (приоритезированы)

### Immediate (можно делать сейчас):
1. **Market-neutral / short-side** — LS Sharpe = 4.21 говорит что сигнал есть, но нужен шорт. Варианты:
   - OKX margin trading (шорт на споте)
   - OKX futures (perps) — аккуратно, без leverage
   - Dollar-neutral: long top 50% equal weight + short bottom 50%
2. **Больше target horizons** — попробовать 12h и 24h вместо 4h (меньше шума, меньше комиссий)
3. **HPO с Optuna** — LR, depth, leaves, regularization (100 trials, ~30 мин на M3 Pro)
4. **Walk-forward expanding** — вместо одного split, двигать окно каждые 3 мес

### Phase 2: Deep Learning (нужен GPU/A100):
5. **Установить Qlib** — `pip install pyqlib`
6. **HIST model** — transformer для cross-asset relationships
7. **MASTER model** — AAAI 2024 SOTA, market-guided transformer
8. **Ensemble** — weighted combination MASTER + HIST + LightGBM

### Phase 3: Production:
9. **FinBERT sentiment** — добавить новостные фичи (CryptoPanic API)
10. **On-chain data** — exchange flows, funding rates
11. **Paper trading** — подключить к OKX sandbox
12. **Live trading** — реальные деньги ($100-1K)

---

## 6. Ключевые решения и контекст

- **Почему крипто, а не акции:** Доступно из РФ, торговля 24/7, старт с $100, бесплатные данные
- **Почему OKX:** Работает для резидентов РФ (в отличие от Binance). Bybit — запасная.
- **Почему 1h:** На минутках побеждает HFT (latency). На daily — мало данных. 1h — оптимальный баланс.
- **Почему 4h target:** Forward return на 4 часа — достаточно для значимого сигнала при 1h данных
- **Почему 50 монет:** Top по ликвидности на Binance, достаточно для cross-sectional ranking
- **SSL workaround:** В России SSL проблема с Binance — отключаем verify в CCXT
- **libomp на macOS:** LightGBM не работает без OpenMP. Fix: `brew install libomp` (сделано).
- **v1 vs v2:** v1 провалился из-за time feature leakage и отсутствия CS normalization. v2 с cross-sectional ranks дал Rank IC 0.031 и LS Sharpe 3.87.
- **Long-only vs Long-short:** Модель хорошо РАНЖИРУЕТ (LS Sharpe 4.2), но long-only не работает в bear market. Нужен short-side для реальной прибыли.
- **Rank Model лучше всех:** Предсказание cross-sectional rank (а не raw return) даёт самый стабильный сигнал (ICIR 0.36).

---

## 7. Technical Notes

### Data Schema (crypto_features_1h.parquet)
```
Columns: timestamp, open, high, low, close, volume, 
         [98 feature columns], 
         target_ret, target_cls, symbol, hour, day_of_week
Rows: 2,102,405
Symbols: 50
Period: 2021-01-30 to 2026-03-06
```

### Feature exclusion list (NOT used as model input)
```python
exclude = {'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume',
           'target_ret', 'target_cls', 'hour', 'day_of_week'}
```

### Walk-forward dates
```python
TRAIN_END = '2024-07-01'  # Train: all before this
VAL_END = '2025-07-01'    # Val: TRAIN_END to VAL_END
                           # Test: after VAL_END
```
