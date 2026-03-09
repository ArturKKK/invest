# AI Architecture Review — Crypto ML Trading System (v2, March 2026)

Ты эксперт в quantitative finance, ML portfolio management и альтернативных данных. Проанализируй мою торговую систему и дай конкретные рекомендации.

---

## Система

**Цель**: Автоматическая торговля 50 криптовалютами с leverage. Модели предсказывают 12h returns, после чего формируется long-short портфель (top-5 long, bottom-5 short). Ребалансировка каждые 12h. Торговля через OKX futures.

**Текущий live результат (OOS, 365 дней, 9 месяцев после конца training):**
- Sharpe: 6.61, Return: +21.3%, MaxDD: -5.4%, WR: 61%, PF: 1.86
- Капитал: $5,000 (demo), 1x leverage, OKX testnet

**Текущий backtest результат (offline, последние 60 дней данных, ensemble + edge-boost):**
- 1x lev: Return +14.4%, MaxDD -12.5%, Sharpe HAC +2.80, WR 61%
- 3x lev: Return +35.8%, MaxDD -35.8%, Sharpe HAC +4.12, WR 58%

**ПРОБЛЕМА — бэктест на 6 месяцах показывает нестабильность:**
- 1x lev (6 мес): Return +3.6%, MaxDD -23.1%, Sharpe HAC -1.41, WR 52%
- 2x lev (6 мес): Return -4.0%, MaxDD -43.7%, Sharpe HAC -2.98
- 3x lev (6 мес): **ЛИКВИДАЦИЯ** (DD > 50%)

Весь профит сконцентрирован в последних 2 месяцах. Период Сент-Дек 2025 — модель теряет деньги.

---

## Данные

| Источник | Описание |
|----------|----------|
| OHLCV | 50 крипто, 1h свечи с 2017 (~2.5M строк, Binance) |
| Fear & Greed Index | daily, 0-100 |
| OKX Funding rates | per-coin, каждые 8h |
| Binance Derivatives | Dec 2021+, 50 symbols, 1h: OI, L/S ratio, taker buy/sell, funding (1.8M строк) |
| News | CryptoCompare, 950K статей, 67 месяцев, VADER sentiment |

---

## Фичи (~190 для LGB, ~211 для XGBoost)

### Price/MA (6)
`ret_1h/4h/12h/24h/3d/7d`, `close_ma72/168/336/720_ratio`, `vwap_12h_dist`

### Volume (4)
`vol_surge_12h/24h/48h`, `vol_trend_12_48`, `vol_12h_cs_rank`

### Volatility (4)
`gk_vol_24h/168h`, `ret_std_168h`, `range_expansion_12h`, `vol_crush_ratio` (v7)

### Momentum (6)
`mom_12h_zscore`, `mom_3d/7d`, `mom_accel_12h`, `ret_sharpe_168h`, `ret_12h_cs_rank`

### v7 Momentum Enhanced (5)
`range_position_12h`, `vwpc_12h`, `hh_count_12h/ll_count_12h`, `trend_strength_12h`, `direction_quality_12h`

### Cross-Asset (10)
`btc_ret_1h..168h`, `eth_ret_1h..24h`, `btc_vol_24h`, `eth_btc_ret_24h`, `btc_beta_48h/168h`, `market_dispersion`, `ret_vs_btc_24h`

### Regime (10, NOT ranked)
`btc_regime_24/72/168`, `regime_btc_above_ma336/720`, `regime_btc_ma720_slope`, `regime_btc_not_crashed`, `regime_btc_dd_720`, `regime_low_vol`, `regime_breadth_bullish`, `breadth_pct_positive`, `regime_composite`

### Sentiment (7, NOT ranked)
`fng_value`, `fng_extreme_fear/greed`, `fng_ma7/30`, `fng_momentum`, `market_avg_funding/std/skew`

### Per-Coin Positioning (5)
`funding_rate`, `funding_vs_market`, `long_short_ratio`, `is_asian_session`

### Synthetic Positioning (6)
`reversal_4v24/12v48/24v168`, `cross_coin_dispersion/disp_ma24`, `dispersion_regime`, `ret_skew_48h/168h_cs`

### Derivatives — Binance Futures (25)
**Open Interest**: `oi_change_1h/4h/12h/24h`, `oi_zscore_7d`, `oi_ret_interaction/12h`, `oi_change_12h_cs`
**Taker**: `taker_buy_sell_ratio`, `taker_imbalance`, `taker_cvd_12h/24h`, `taker_flow_zscore`, `taker_imbalance_cs`
**L/S**: `top_ls_ratio`, `top_long_pct`, `top_ls_change_12h/24h`, `top_ls_zscore`, `global_ls_ratio`, `global_long_pct`, `ls_divergence`
**Funding**: `funding_rate_binance`, `funding_surprise`

### News (10, только CatBoost в production)
`news_count_1h/24h/7d`, `news_sentiment_1h/24h/7d`, `news_sentiment_momentum`, `news_volume_zscore`, `market_news_count/sentiment_24h`

---

## Модели

### LightGBM v6 (production, 5 seeds)
- objective: regression (на rank target), n_est: 5000, lr: 0.01, max_depth: 6, num_leaves: 31
- feature_fraction: 0.5, bagging_fraction: 0.7, min_child: 200, L1/L2: 1.0
- **БЕЗ news** в production

### LightGBM v7 (production, 5 seeds)
- Как v6, но target = 0.75 × ret_12h + 0.25 × ret_24h (blended)
- +8 дополнительных фичей (range_position, vwpc, trend_strength...)
- **БЕЗ news** в production

### CatBoost (production, 5 seeds)
- iterations: 5000, lr: 0.01, depth: 6, l2_leaf_reg: 3, SymmetricTree
- **С NEWS** в production (единственная модель)

### XGBoost (экспериментальный, НЕ в production)
- Как LGB, но с 23 news interaction фичами — слабее ансамбля

---

## Target

- `target_rank` = cross-sectional percentile rank (0..1) of forward 12h return per timestamp
- v7: rank of blended (75% 12h + 25% 24h forward return)
- Purge gap: 8 дней между train и validation/test

---

## Нормализация

- **Cross-sectional rank** (0-1) для большинства фичей
- **Time-series zscore** (rolling 168h, winsorized ±3σ) для: vol surges, OI changes, funding, news counts
- **Без нормализации** для regime фичей (бинарные/composite)

---

## Walk-Forward Validation (3 окна)

| Window | Train end | Test start | Test end |
|--------|-----------|------------|----------|
| W1 | 2023-06 | 2024-07 | 2024-12 |
| W2 | 2024-01 | 2025-01 | 2025-12 |
| W3 | 2024-06 | 2025-01 | 2026+ |

---

## Ансамбль

```
final_score = mean([mean(v6_seeds), mean(v7_seeds), mean(cb_seeds)])
confidence = 1 / (1 + std(all_15_normalized_predictions))
position_weight = edge_boost × confidence
edge_boost = 1 + min(|edge| / P75_edge, 3.0)
```

Top-5 long, bottom-5 short. DDStop: exit при DD < -20%, resume при DD > -8%.

---

## Результаты экспериментов

### Production Walk-Forward (DDStop Sharpe)
| Model | DDStop Sharpe | MaxDD |
|-------|---------------|-------|
| LGB v7 (no news) | 1.88 | -43.7% |
| LGB v6 (no news) | 1.81 | -44.2% |
| CatBoost (with news) | 1.51 | -38.0% |

### exp10 — News A/B тест
- **News ВРЕДЯТ LightGBM**: DDStop -21% to -57%
- **CatBoost устойчив к news**: ±2%

### exp11 — Binance Derivatives A/B
| Variant | DDStop Sharpe | vs exp10 | vs PROD |
|---------|---------------|----------|---------|
| **v7_baseline** | **2.12** (std=0.22) | **+42%** | **+13%** |
| v7_res_hyb | 1.84 | +90% | — |
| catboost_baseline | 1.54 | +3% | +2% |
| v6_baseline | 1.23 | +57% | — |

**Ключевой вывод**: Derivatives МАССИВНО помогают LGB, почти не влияют на CatBoost.

### exp12 — Полный ретренинг (СЕЙЧАС ОБУЧАЕТСЯ на H100)
20 вариантов: v6/v7 × lambdarank/rank × news_ablation × derivatives_ablation + CatBoost + XGBoost

---

## Новые компоненты (реализованы, тестируются)

### Meta-Risk Scaler
5-сигнальный композит для масштабирования gross exposure (0.3x–1.5x):
- **Confidence** (25%): средняя уверенность ансамбля выбранных позиций
- **Score spread** (20%): абсолютный спред скоров (широкий = хорошо)
- **Recent WR** (25%): win rate за последние 10 шагов
- **DD depth** (20%): текущая глубина просадки → уменьшает размер при DD
- **Regime** (10%): бычий/медвежий режим BTC

### Vol Targeting
- Целевая волатильность портфеля (годовая → per-step)
- Масштабирует позиции: `vol_target / realized_vol`, clipped 0.2x–2.0x
- Lookback: 48 шагов (24 дня при 12h rebal)

### A/B тесты meta-risk / vol targeting (60 дней, 1x leverage, $1000)

| Вариант | Return | MaxDD | Sharpe HAC | WR |
|---------|--------|-------|------------|-----|
| Baseline (ensemble+boost) | +14.4% | -12.5% | +2.80 | 61% |
| **+meta-risk** | **+21.6%** | -14.8% | **+3.35** | 61% |
| +vol-target 30% | +10.4% | -14.4% | +2.14 | 59% |
| +meta+vol | +16.2% | -15.1% | +3.01 | 60% |

С 3x leverage:

| Вариант | Return | MaxDD | Sharpe HAC |
|---------|--------|-------|------------|
| Baseline 3x | +35.8% | -35.8% | +4.12 |
| **+meta-risk 3x** | **+54.3%** | -39.1% | +3.82 |

---

## Ключевая проблема: НЕСТАБИЛЬНОСТЬ НА ДЛИННЫХ ПЕРИОДАХ

| Период | Lev | Return | MaxDD | Sharpe HAC |
|--------|-----|--------|-------|------------|
| 60 дней (Янв-Мар 2026) | 1x | +14.4% | -12.5% | +2.80 |
| 60 дней | 3x | +35.8% | -35.8% | +4.12 |
| **180 дней (Сен 2025 – Мар 2026)** | 1x | **+3.6%** | **-23.1%** | **-1.41** |
| 180 дней | 2x | **-4.0%** | **-43.7%** | **-2.98** |
| 180 дней | 3x | **LIQUIDATED** | >50% | — |
| 180 дней + meta-risk | 3x | +0.6% | -60.3% | -2.08 |
| 365 дней | 1x | +34.4% | -23.1% | +0.56 |
| 365 дней + meta-risk | 1x | +51.1% | -25.4% | +0.88 |

**Модели обучены до 2024-06. Период Сен-Дек 2025 — модель теряет деньги. Последние 2 месяца (Янв-Мар 2026) — весь профит.**

Это говорит о:
- Model decay / concept drift через 15+ месяцев без ретренинга
- Возможном regime mismatch (модель обучена на bull/bear, а Сен-Дек 2025 — боковик?)
- Необходимости регулярного ретренинга (сейчас запущен exp12)

---

## Что работает / Не работает

### ✅ Работает
- 12h target aligned с holding period (Sharpe +4x vs 4h target)
- Edge-boost sizing (Sharpe 2.79→5.93)
- Hybrid news: LGB no-news + CB with-news (+45% Sharpe)
- Derivatives features (+42% DDStop для LGB)
- DDStop circuit breaker (-20% stop, -8% resume)
- Event filter (FOMC/CPI → leverage 30%)
- Meta-risk scaler (+50% return при +57% Sharpe на 1 год)

### ❌ Не работает
- News в LGB (DDStop -36% как минимум)
- Vol targeting (снижает returns без пропорционального снижения DD)
- Leverage >1x на длинных периодах (DD >50% при 3x)
- min-conf 0.85 threshold (режет 40% трейдов, многие прибыльные)
- Данные до 2017 (старая крипта ≠ текущая)
- XGBoost news interactions (сложность без пользы)
- Residual target (лучше prediction, хуже profit)

---

## Top Feature Importance

**LGB** (sentiment-driven): fng_ma30 > fng_momentum > fng_ma7 > vol_12h_cs_rank > close_ma720_ratio
**CatBoost** (price-structure-driven): close_ma336_ratio > vol_12h_cs_rank > close_ma720_ratio > breadth_pct_positive > gk_vol_24h

---

## Известные проблемы

1. **Survivorship bias**: фиксированные 50 монет из 2025, применяются к 2021+
2. **Model decay**: обучены до 2024-06, сейчас +21 мес OOS — Сен-Дек 2025 явный decay
3. **WR 61%**: хотим 65-70% без потери Sharpe
4. **19/50 монет заблокированы** для шорта на OKX demo → сильные short сигналы пропадают
5. **Ещё не на реальных деньгах** (OKX demo testnet)
6. **Нет on-chain data** (whale flows, exchange reserves) — только derivatives

---

## Вопросы (обновлённые, с фокусом на главных проблемах)

### 1. Нестабильность на длинных периодах
Модель даёт +14% за 2 мес, но всего +3.6% за 6 мес (1x). Весь профит в последних 2 месяцах. Как это исправить?
- Чаще ретренить? Каждые 3 мес? 6 мес?
- Online learning / incremental updates?
- Regime-aware switching (одна модель для бока, другая для тренда)?
- Ensemble weighting по recent performance?

### 2. Leverage management
При 1x — профит мал (14% за 2 мес). При 3x — DD 36-60%. Как найти оптимум?
- Kelly criterion для dynamic leverage?
- Vol-of-vol based sizing?
- Time-varying leverage по режиму рынка?
- Наш vol-target (30% ann) ухудшает результат — почему и что делать?

### 3. Architecture Next Steps
Текущий ансамбль: equal-weight LGB_v6(5) + LGB_v7(5) + CatBoost(5) = 15 моделей.
- Стоит ли добавлять meta-model (stacking)? Наш простой meta-risk scaler даёт +57% Sharpe.
- TFT (Temporal Fusion Transformer) — добавит ли что-то к деревьям? Или слишком затратно для маргинального улучшения?
- Какой подход к ensemble weighting рекомендуешь (вместо equal weight)?

### 4. Feature engineering gaps
Что критически упущено?
- On-chain (whale flows, exchange reserves, NVT ratio)?
- Liquidation data (каскады ликвидаций как сигнал)?
- Order book depth / imbalance?
- Macro (DXY, rates, M2)?
- Cross-exchange basis / arbitrage spreads?

### 5. Почему news вредят LGB но помогают CatBoost?
Наша гипотеза: LGB leaf-wise boosting переобучается на шумных news splits. CatBoost с ordered boosting более robust. Верно? Как исправить? Residual target пробовали — лучше Rank IC, хуже DDStop.

### 6. Risk management optimal setup
- DDStop -20%/-8% выбрано эвристически. Как оптимизировать?
- Meta-risk scaler: 5-signal composite (confidence, spread, WR, DD, regime) → 0.3x-1.5x. Как улучшить?
- Position-level stop-loss нужен? Или портфельный DDStop достаточен?

### 7. Что я делаю КРИТИЧЕСКИ неправильно?
Какие bias/ошибки/наивности ты видишь? Что бы тебя конкретно удивило или настораживает?

### 8. Production readiness
Система работает 2 месяца на demo. Что нужно проверить перед переходом на реальные деньги? Какие подводные камни при переходе demo → live?
