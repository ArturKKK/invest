# AI Architecture Review — Crypto ML Trading System

Ты эксперт в quantitative finance, ML portfolio management и альтернативных данных. Проанализируй мою торговую систему и дай конкретные рекомендации.

---

## Система

**Цель**: Автоматическая торговля 50 криптовалютами с leverage. Модели предсказывают 12h returns, после чего формируется long-short портфель (top-5 long, bottom-5 short). Ребалансировка каждые 12h. Торговля через OKX futures.

**Текущий результат (OOS, 365 дней, 9 месяцев после конца training):**
- Sharpe: 6.61, Return: +21.3%, MaxDD: -5.4%, WR: 61%, PF: 1.86
- Капитал: $5,000 (demo), 1x leverage

---

## Данные

| Источник | Описание |
|----------|----------|
| OHLCV | 50 крипто, 1h свечи с 2021, ~2.5M строк (Binance) |
| Fear & Greed Index | daily, 0-100 |
| OKX Funding rates | per-coin, каждые 8h |
| Binance Derivatives | Dec 2021+, 50 symbols, 1h: OI, L/S ratio (top-trader + global), taker buy/sell, funding (1.8M строк) |
| News | CryptoCompare, 950K статей, 67 месяцев, VADER sentiment |

---

## Фичи (188 для LGB, 211 для XGBoost)

### Price/MA (6)
`ret_1h/4h/12h/24h/3d/7d`, `close_ma72/168/336/720_ratio`, `vwap_12h_dist`

### Volume (4)
`vol_surge_12h/24h/48h`, `vol_trend_12_48`, `vol_12h_cs_rank`

### Volatility (4)
`gk_vol_24h/168h`, `ret_std_168h`, `range_expansion_12h`, `vol_crush_ratio` (v7)

### Momentum (6)
`mom_12h_zscore`, `mom_3d/7d`, `mom_accel_12h`, `ret_sharpe_168h`, `ret_12h_cs_rank`

### v7 Momentum Enhanced (5)
`range_position_12h`, `vwpc_12h`, `hh_count_12h`, `ll_count_12h`, `trend_strength_12h`, `direction_quality_12h`

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

### Derivatives — NEW, exp11 (25)
**Open Interest**: `oi_change_1h/4h/12h/24h`, `oi_zscore_7d`, `oi_ret_interaction/12h`, `oi_change_12h_cs`
**Taker**: `taker_buy_sell_ratio`, `taker_imbalance`, `taker_cvd_12h/24h`, `taker_flow_zscore`, `taker_imbalance_cs`
**L/S**: `top_ls_ratio`, `top_long_pct`, `top_ls_change_12h/24h`, `top_ls_zscore`, `global_ls_ratio`, `global_long_pct`, `ls_divergence`
**Funding**: `funding_rate_binance`, `funding_surprise`

### News (10, только CatBoost в prод)
`news_count_1h/24h/7d`, `news_sentiment_1h/24h/7d`, `news_sentiment_momentum`, `news_volume_zscore`, `market_news_count/sentiment_24h`

### News Interactions (23, только XGBoost, не работает)
`nx_sent_x_count`, `nx_burst_*`, `nx_sent_price_div`, `nx_sent_ret_product`, и т.д.

---

## Модели

### LightGBM v6 (production, 5 seeds)
- objective: regression (на rank), n_estimators: 5000, lr: 0.01, max_depth: 6, num_leaves: 31
- feature_fraction: 0.5, bagging_fraction: 0.7, min_child: 200, L1/L2: 1.0
- Feature selection: gain-based, bottom 20% pruned → ~150 фичей
- **БЕЗ news** в production

### LightGBM v7 (production, 5 seeds)
- Как v6, но target = 0.75 × ret_12h + 0.25 × ret_24h (blended)
- +8 дополнительных фичей (range_position, vwpc, trend_strength...)
- **БЕЗ news** в production

### CatBoost (production, 5 seeds)
- iterations: 5000, lr: 0.01, depth: 6, l2_leaf_reg: 3, SymmetricTree
- С NEWS в production (единственная модель)

### XGBoost (экспериментальный, НЕ в production)
- Как LGB, но с 23 news interaction фичами
- DDStop Sharpe 0.97-1.26, слабее ансамбля

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

Top-5 long, bottom-5 short. Leverage 1x. DDStop: exit при DD < -20%, resume при DD > -8%.

---

## Результаты экспериментов

### Production (DDStop Sharpe, walk-forward)
| Model | DDStop Sharpe | MaxDD |
|-------|---------------|-------|
| LGB v7 (no news) | 1.88 | -43.7% |
| LGB v6 (no news) | 1.81 | -44.2% |
| CatBoost (with news) | 1.51 | -38.0% |

### exp10 — News A/B тест (9 марта)
- **News ВРЕДЯТ LightGBM**: DDStop -21% to -57%
- **CatBoost устойчив к news**: ±2%
- Residual-target: лучше Rank_IC, но хуже DDStop
- Null-importance: бесполезно

### exp11 — Binance Derivatives A/B (10 марта)
| Variant | DDStop Sharpe | vs exp10 | vs PROD |
|---------|---------------|----------|---------|
| **v7_baseline** | **2.12** (std=0.22) | **+42%** | **+13%** |
| v7_res_hyb | 1.84 | +90% | — |
| catboost_baseline | 1.54 | +3% | +2% |
| v6_baseline | 1.23 | +57% | — |
| xgboost_res_hyb | 1.26 | — | — |

**Ключевой вывод**: Derivatives МАССИВНО помогают LGB, почти не влияют на CatBoost.

---

## Что работает / Не работает

### ✅ Работает
- 12h target aligned с holding period (Sharpe +4x vs 4h target)
- Edge-boost sizing (Sharpe 2.79→5.93)
- Hybrid news: LGB no-news + CB with-news (+45% Sharpe)
- Derivatives features (+42% DDStop для LGB)
- DDStop circuit breaker
- Event filter (FOMC/CPI → leverage 30%)

### ❌ Не работает
- News в LGB (DDStop -36% как минимум)
- min-conf 0.85 threshold (режет 40% трейдов, многие прибыльные)
- Более 8 лет данных (2017 ≠ 2021+ крипта)
- XGBoost news interactions (сложность без пользы)
- Residual target (лучше prediction, хуже profit)
- Dynamic leverage выше 1x (DD растёт быстрее profit)

---

## Top Feature Importance

**LGB** (sentiment-driven): fng_ma30 > fng_momentum > fng_ma7 > vol_12h_cs_rank > close_ma720_ratio
**CatBoost** (price-structure-driven): close_ma336_ratio > vol_12h_cs_rank > close_ma720_ratio > breadth_pct_positive > gk_vol_24h

---

## Известные проблемы

1. **Survivorship bias**: фиксированные 50 монет из 2025, применяются к 2021+
2. **Модели устарели**: обучены на данных до 2024-06, нужен retrain до 2025-09
3. **WR 61%**: хотим 65-70% без потери Sharpe
4. **19/50 монет заблокированы** для шорта на OKX demo → сильные short сигналы пропадают
5. **Ещё не на реальных деньгах**

---

## Вопросы

1. **Архитектура ансамбля**: Сейчас equal weight (v6, v7, CB). Есть ли лучший способ комбинировать? Stacking? Dynamic weighting по режиму рынка? Какой meta-model подход рекомендуешь?

2. **Новые модели для ансамбля**: TFT (Temporal Fusion Transformer) видит последовательности — имеет смысл? Или TabNet, NODE, другие? Что даст максимум при минимуме усилий?

3. **Feature engineering**: Какие фичи я упускаю? On-chain (whale flows, exchange reserves)? Order book imbalance? Liquidation cascades? Cross-exchange arbitrage gaps?

4. **Почему news вредят LGB но не CatBoost?** Наша гипотеза: leaf-wise boosting LGB overweights noisy news splits. CatBoost с ordered boosting более robust. Это правильно? Как можно сделать news полезными для LGB?

5. **Target engineering**: Rank target (0-1) работает. Стоит ли пробовать что-то другое? Quantile targets? Classification (buy/sell/hold)? Multi-horizon совместное обучение?

6. **Risk management**: DDStop -20%/-8% выбрано эвристически. Как оптимизировать? Kelly criterion for sizing? Regime-dependent sizing?

7. **Что добавить в production для максимального impact при минимальных усилиях?**

8. **Что я делаю неправильно?** Какие bias/ошибки ты видишь в текущем подходе?
