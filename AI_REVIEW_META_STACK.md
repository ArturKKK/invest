# Промпт для ревью мета-модели стэкинга

## Контекст проекта

Это crypto-trading система. 3 L0 модели (LightGBM v6, LightGBM v7, CatBoost) генерируют предсказания по ~40 монетам каждые 12 часов. Поверх них обучен Level-1 мета-модель (стэкинг), который комбинирует L0 OOS предсказания в финальный сигнал.

**Архитектура:**
- L0: 5 seeds × 3 модели = 15 моделей total, обученные walk-forward
- Meta: Ridge (baseline) + LightGBM (основной), варианты: lgb (33 feat), lgb_minimal (21 feat), ridge (3 feat), ridge_all (33 feat)
- Production: rebal каждые 12h, long/short top/bottom 20%, taker 3bps + slippage 1bp, funding 0.5bps/8h, turnover 35%

**Walk-forward:**
- W1 test = 2024-07 → 2024-12 (используется как meta-train)
- W3 test = 2025-01 → 2026-03 (meta-test, out-of-sample)
- W2/W3 overlap разрешён через dedup (keep='last', т.е. W3 модели = больше данных)

## Задание

Проведи **подробный code review** следующих 4 файлов. Ищи именно **баги, утечки данных (leakage), ошибки walk-forward, логические ошибки**. Не нужно стилистических замечаний или рефакторинг ради рефакторинга.

## Файлы для ревью

### 1. `run_meta_stack.py` (~738 строк) — тренировка мета-модели

Основной скрипт: загрузка L0 OOS предсказаний → сборка мета-фич → обучение Ridge + LGB → evaluation.

**Что было сделано (фиксы и улучшения):**
- Dedup per-model BEFORE merge (не после) — fix cross-product explosion
- NaN ffill/bfill для контекстных фич (не dropna целых периодов)
- RidgeCV с TimeSeriesSplit вместо random CV
- Explicit feature lists (META_FEATURES_FULL, META_FEATURES_MINIMAL) вместо discovery
- LGB: num_leaves=15, max_depth=5, min_child_samples=500 (было 31/auto/200)
- LGB: TimeSeriesSplit CV (3 fold) для определения best_num_boost_round, потом train на ALL meta-train data
- Target winsorization: `--winsorize 0.005` клипает extreme returns до rank target
- `--expanding` флаг: expanding window (все данные до cutoff) вместо только W1
- `--save-model` default=True
- Функция vol_target_returns исправлена: cost вычитается как `cost * 2 * scale`
- max_dd: исправлен для корректной работы (min != 0 baseline fix)
- Cost per period учитывает funding_per_8h корректно

**Ключевые вопросы для ревью:**
1. Нет ли look-ahead bias в `build_meta_features()`? Контекстные фичи (gk_vol_24h, rsi_14 и т.д.) берутся из features parquet — корректен ли join по timestamp?
2. LGB TimeSeriesSplit CV: мы определяем best_iter через median 3 folds, потом тренируем на ALL meta-train data с фиксированным числом раундов. Это правильно?
3. Winsorization: правильно ли клипать target_ret_12h ДО ранжирования? Не теряем ли информацию?
4. Expanding window: использовать все данные до 2025-01-01 vs только W1 — какие риски?
5. Нет ли утечки через деdup? keep='last' при W2/W3 overlap — корректно ли?

### 2. `src/models/meta_model.py` (~263 строки) — shared inference модуль

Используется в production (run_trading.py) и backtesting (run_fast_sim.py).

**Что проверить:**
1. `build_meta_features_live()` — полностью ли совпадает логика с `build_meta_features()` в run_meta_stack.py? (train-serving skew)
2. `MetaModelInference.load()` — корректно ли обрабатывает все 4 варианта?
3. `predict()` — заполнение missing features нулями: не опасно ли?
4. BTC context features: корректно ли берутся из snap_df? Нет ли NaN propagation?
5. `root` parameter в `load()`: при pkl_path='auto' корректно ли вычисляется путь через `__file__`?

### 3. `run_fast_sim.py` (~1180 строк) — historical backtesting simulator  

**Что было сделано:**
- Удалена локальная `_build_meta_features_live()` (~80 строк), заменена на `from src.models.meta_model import MetaModelInference, build_meta_features_live`
- Удалена ручная загрузка мета-модели через joblib, заменена на `MetaModelInference.load(..., root=root)`
- `predict_ensemble()` использует `_meta_model_inf.predict()` вместо прямого вызова моделей

**Что проверить:**
1. DRY рефакторинг: не сломалась ли логика при переходе на shared модуль?
2. `predict_ensemble()`: как передаются pred_v6, pred_v7, pred_cb в мета-модель? Корректно ли разделение на группы?
3. `--meta-model auto --meta-variant lgb_minimal`: CLI args парсятся и используются правильно?
4. `root` передаётся в `MetaModelInference.load()` — это project root или что?

### 4. `run_trading.py` (~2379 строк) — production trading

**ВАЖНО:** В этом файле ДВЕ функции `main()` (исторический артефакт). Только ВТОРАЯ (последняя) вызывается. Мета-модель интегрирована во вторую.

**Что было сделано:**
- Import `MetaModelInference` (с try/except fallback)
- `generate_signal()` принимает `meta_model=None`, если передан — использует для финального scoring
- Во второй `main()`: `--meta-model` и `--meta-variant` CLI args
- Meta-model loading block после загрузки L0 моделей
- Передача meta_model в generate_signal()

**Что проверить:**
1. Как мета-модель используется в `generate_signal()`? Перезаписывает ли она L0 простое среднее?
2. Корректно ли передаются snap_df, pred_v6, pred_v7, pred_cb из production pipeline?
3. Нет ли race condition при 12h rebalancing + мета-модель?
4. Fallback на простое среднее если мета-модель не загрузилась — работает?

## Текущие результаты (meta-test, OOS)

```
Model               IC      RankIC  LS_Sharpe  VT_Sharpe  DDStop
Simple Mean         0.0248  0.0306  +1.42      +2.97      +3.15
LGB-MINIMAL (21f)   0.0285  0.0340  +1.39      +2.97      +3.14
LGB-META (33f)      0.0271  0.0316  +1.36      +2.47      +2.72
Ridge-ALL (33f)     0.0254  0.0307  +1.47      +3.00      +3.20
Ridge-3 (3f)        0.0250  0.0307  +1.44      +2.85      +2.93
```

**Наблюдение:** LGB-MINIMAL почти не улучшает Simple Mean по VT_Sharpe (2.97 vs 2.97), но улучшает IC/RankIC. LGB-META с контекстом хуже — возможный overfitting. Ridge-ALL немного лучше.

## Формат ответа

Пожалуйста, для каждого файла:
1. **Баги** (P0) — то что может сломать production или дать неправильные результаты
2. **Leakage / bias** (P1) — утечки данных, look-ahead bias, train-test contamination
3. **Логические проблемы** (P2) — спорные решения, потенциальные проблемы
4. **Мелкие замечания** (P3) — некритичные улучшения

В конце дай общую оценку: готов ли мета-модель к production?
