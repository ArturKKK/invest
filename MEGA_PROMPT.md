# MEGA PROMPT — Полная история крипто ML-торговой системы

> Этот документ содержит ВСЮ историю проекта: все модели, все фичи, все результаты, все ошибки.
> Используй как полный контекст при работе с любой AI-моделью.
> Последнее обновление: 6 апреля 2026 (после R48 COMPLETE. Новый чемпион: 31f + hybrid tiered costs → ALL=1.66).

---

## Оглавление
1. [Описание системы](#1-описание-системы)
2. [Данные](#2-данные)
3. [Продакшн система](#3-продакшн-система-run_tradingpy)
4. [Модели в production registry](#4-модели-в-production-registry)
5. [Полная история фич](#5-полная-история-фич)
6. [Pipeline v1–v8: Эволюция пайплайна](#6-pipeline-v1v8-эволюция-пайплайна)
7. [Research Rounds R1–R39: Полная хронология](#7-research-rounds-r1r39-полная-хронология)
8. [История утечек данных (Data Leakage)](#8-история-утечек-данных-data-leakage)
9. [Execution Layer эксперименты (Overnight v10–v15)](#9-execution-layer-эксперименты-overnight-v10v15)
10. [AI Consultation Documents](#10-ai-consultation-documents)
11. [Установленные факты и выводы](#11-установленные-факты-и-выводы)
12. [Текущее состояние и нерешённые проблемы](#12-текущее-состояние-и-нерешённые-проблемы)
13. [ML Core — удалённые вычисления](#13-ml-core--удалённые-вычисления)

---

## 1. Описание системы

**Цель**: Систематическая крипто long/short портфельная стратегия. Предсказываем 12-часовой forward return для 35 монет, ранжируем cross-sectionally, идём long top-N / short bottom-N. Ребалансировка каждые 12 часов на OKX perpetual futures.

**Стек**:
- Python 3.13, macOS M3 Pro (dev), VPS 185.42.163.63 (production)
- **pandas ≤2.x ОБЯЗАТЕЛЬНО** (pandas 3.0 ломает groupby.apply → тихо меняет все фичи, Sharpe 1.66→1.02)
- LightGBM 4.6.0, XGBoost, CatBoost, scikit-learn
- Данные: Binance OHLCV + Derivatives + Funding + OI + Taker Flow
- Биржа: OKX demo → OKX live (perpetual futures)

**Ключевые параметры**:
- `LEVERAGE = 5`, `CAPITAL = 100 USDT`
- `HORIZON = 12` (предсказание на 12 часов вперёд)
- `REBAL_HOURS = 12` (ребалансировка каждые 12 часов)
- `N_LONG = 6`, `N_SHORT = 3`
- `SEEDS = [0, 7, 13, 42, 99]`
- Walk-forward валидация на 3 окнах (W1, W2, W3)

**VPS**: root@185.42.163.63 (SSH через SOCKS5 proxy), `/home/trader/invest/`
- Текущий статус: RUNNING (R114b champion deployed, state machine fix: cutoff_off=0.8, moff=2)
- ⚠️ НИКОГДА не деплоить на VPS без явного запроса пользователя. Каждый редеплой стоит ~$5 в комиссиях.

---

## 2. Данные

### 2.1. Основные данные

| Источник | Файл | Частота | Покрытие |
|----------|------|---------|----------|
| Binance Spot OHLCV | `data/raw/*_1h.parquet` | 1h свечи | 50 монет, 2021+, ~2.5M строк |
| Binance Futures Metrics | `binance_futures_metrics.parquet` | 1h | Dec 2021+, 1.8M строк |
| Binance Funding Rates | `binance_funding_rates.parquet` | 8h | Jan 2020+, 294K строк |
| Binance Premium Index | `binance_premium_index.parquet` | varies | Имеется |
| Fear & Greed Index | `fear_greed.parquet` | daily | Полная история |
| Macro (FRED) | `macro_daily.parquet` | daily | VIX, SPX, DXY, Gold, Yields, HY spread |
| Deribit DVOL | `deribit_dvol.parquet` | hourly | BTC+ETH implied vol |
| CryptoCompare News | `crypto_news.parquet` | hourly | 950K статей, 67 месяцев |
| On-chain | `btc_onchain.parquet`, `onchain_daily.parquet` | daily | BTC метрики |
| DeFi TVL | `defi_tvl_daily.parquet` | daily | Protocol TVL |
| Stablecoin Supply | `stablecoin_supply.parquet` | daily | Supply data |
| Google Trends | `data/features/google_trends.parquet` | daily | Имеется |
| CC Social | `data/features/cc_social_daily.parquet` | daily | Social metrics |

### 2.2. Монеты

**49 символов** (полный список):
```
BTC, ETH, BNB, SOL, XRP, ADA, DOGE, AVAX, DOT, LINK,
UNI, ATOM, LTC, FIL, APT, ARB, OP, NEAR, AAVE, INJ,
FTM, ALGO, SAND, MANA, AXS, THETA, RUNE, EGLD, XTZ, FLOW,
CHZ, CRV, LDO, SNX, COMP, YFI, SUSHI, ENJ, BAT, ZIL,
ONE, IOTA, ICX, ENS, IMX, GALA, MKR, GRT, ETC
```

**OKX заблокированные** (19 шт — нет на OKX demo): MATIC, UNI, APT, FTM, MANA, RUNE, EGLD, FLOW, SNX, ENJ, BAT, ONE, ICX, ENS, GALA, GRT, CHZ, MKR

**Рабочий набор для research (SYM_35)**: 35 монет (49 минус заблокированные и неликвидные)

---

## 3. Продакшн система (run_trading.py)

### 3.1. Архитектура

```
Main Loop:
  Fetch OHLCV → build_features() → enrich(cross-asset, regime, derivatives,
  sentiment, news) → generate_signal() → construct_portfolio() →
  partial_rebalance(limit+market) → position_ledger_sync → trades.csv →
  dashboard → state_persist → sleep(next_rebalance_boundary)
```

### 3.2. Генерация сигналов

Три генератора сигналов (последовательно эволюционировали):

1. **`generate_signal()`** — регрессионный ансамбль:
   - LGB v6 (5 seeds) + LGB v7 (5 seeds) + CatBoost (5 seeds) + XGBoost (5 seeds)
   - → mean prediction → z-score
   - ~165 фич (включая news, macro, TA, derivatives)

2. **`generate_signal_cls()`** — ТЕКУЩИЙ ЛУЧШИЙ, бинарная классификация:
   - LGB classifier (5 seeds) + XGB classifier (5 seeds)
   - Таргет: P(ret_12h > 0)
   - → mean probability → rank → z-score
   - 23-28 фич (FEATURES_23 или FEAT_28)

3. **`generate_signal_ridge()`** — линейная модель с режимным масштабированием:
   - Ridge alpha=1000, 14 фич
   - mr_scale для режимного масштабирования

### 3.3. Конструкция портфеля

```python
DEFAULT_RISK = {
    'n_long': 6, 'n_short': 3,
    'vol_target': 0.008, 'vol_lookback': 48,
    'kelly_frac': 0.8,
    'dd_stop': -0.15, 'dd_resume': -0.06,
}
```

**Текущий лучший конфиг портфеля (6L3S_ema05_h3)**:
```python
{
    "n_long": 6, "n_short": 3,
    "rebal_hours": 12,
    "trend_cutoff": 0.9,       # BTC trend filter
    "dyn_threshold": 0.7,      # dynamic threshold
    "ema_alpha": 0.5,          # EMA сглаживание сигнала
    "hysteresis": 3,           # позиции держатся 3 цикла минимум
}
```

**Элементы портфельной конструкции**:
- Edge-boost sizing: weight ∝ (1 + edge/P75), cap 4x
- Regime-conditional asymmetry: bull → 7L/2S, bear → 5L/4S
- Vol scaling, DD circuit breaker
- Partial rebalance (только изменения — HOLD saves costs)

### 3.4. Стоимость торговли (Cost Model)

```python
taker_fee    = 0.0005   # 5 bps
slippage     = 0.0002   # 2 bps
funding_12h  = 0.00008  # ~1 bp per 12h
cost_one_way = 0.0007   # 7 bps total
```

**⚠️ Издержки = 40-47% drag от gross до net Sharpe!**
- Gross Sharpe 3.82 → Net Sharpe 2.88 (типичный случай)
- Turnover ~4-5 позиций/ребалансировка при ema+hysteresis
- Turnover ~10-11 без сглаживания (12h rebal)

---

## 4. Модели в production registry

### model_registry.json — 7 поколений

| Gen | Имя | Train End | Модели | Результат | Проблемы |
|-----|-----|-----------|--------|-----------|----------|
| 1 | v2.0-gbdt-no-calendar | 2025-12-01 | LGB v6+v7, CB, XGB × 5 seeds | OOS Sharpe 2.25 | — |
| 2 | v2.0-fresh-data | 2026-02-01 | LGB v6+v7, CB, XGB × 5 seeds | **OOS Sharpe 8.43, +21.7%** | Предыдущий gen1 имел leakage |
| 3 | gen3-calendar | 2026-02-01 | +9 calendar feats, +MLP | — | Calendar фичи потом оказались вредны |
| 4 | v4-huber-4model-mz05 | — | Huber loss для всех 4 моделей | HAC 8.14, Sharpe 7.29, WR 65% | — |
| 5 | v10-expanded-features | — | 200 фич | HAC 8.98, +22.3%, WR 69% | Раздутый feature set |
| 6 | v14-champion-cb-solo | — | CatBoost solo, HPO 50 trials | +147.5%, HAC 5.48 | — |
| **7** | **v15-cb-huber-fresh** | — | CatBoost Huber, 208 фич | **HAC: WA=6.91, WB=9.22, WC=7.17** | Текущий |

**current_gen: 7** (v15-cb-huber-fresh-deploy)

---

## 5. Полная история фич

### 5.1. Эволюция feature set

```
14 фич (Ridge baseline, R1-R16)
  → 17 фич (R18: +vol features)
  → 23 фич (R19: +breadth, seasonality) ← MAIN PRODUCTION SET
  → 26 фич (R31: +ret_168h, cum_funding_24h, dist_from_high_24h)
  → 27 фич (R32: +rel_volume_cs)
  → 28 фич (R32: +ret_skew_168h) ← ТЕКУЩИЙ ЛУЧШИЙ для walk-forward
  → 29-30 фич (R33: +btc_corr) — НЕ помогли W3, но стабилизируют W2
```

### 5.2. RIDGE_FEATURES (14 фич, исходный набор)

```python
RIDGE_FEATURES = [
    'ret_12h', 'ret_24h', 'ret_48h',                    # Returns
    'residual_12h', 'residual_24h',                       # BTC residuals
    'mom_z_12h', 'mom_z_24h',                             # Momentum z-scores
    'dist_from_high_24h',                                  # Price distance from high
    'oi_chg_12h', 'oi_chg_24h', 'oi_zscore',             # Open Interest
    'taker_cvd_12h', 'taker_cvd_24h',                     # Taker CVD
    'ls_divergence',                                       # Long/Short divergence
]
```

### 5.3. FEATURES_23 (основной production set, R19+)

```python
FEATURES_23 = [
    # --- Returns & Momentum (6) ---
    "ret_12h", "ret_24h", "ret_48h",
    "residual_12h", "residual_24h",
    "mom_z_24h",
    # --- Derivatives (6) ---
    "oi_chg_12h", "oi_chg_24h", "oi_zscore",
    "taker_cvd_12h", "taker_cvd_24h", "ls_divergence",
    # --- Volatility (5) ---
    "atr_14", "rvol_12h", "gk_vol_24h", "rvol_24h", "iv_rv_spread",
    # --- Market Breadth (2) ---
    "pct_coins_up_12h", "pct_coins_up_1h",
    # --- Seasonality (4) ---
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
```

### 5.4. FEAT_26 (R31: +3 high-IC фичи)

```python
FEAT_26 = FEATURES_23 + [
    "ret_168h",             # IC=-0.044, ICIR=-0.80 — strongest new
    "cum_funding_24h",      # IC=-0.051, ICIR=-1.03
    "dist_from_high_24h",   # IC=+0.020, ICIR=+0.36
]
```

### 5.5. FEAT_28 (R32: +2 Kaggle features) ← ТЕКУЩИЙ ЛУЧШИЙ

```python
FEAT_28 = FEAT_26 + [
    "rel_volume_cs",        # log(vol) - cs_mean(log(vol)), ICIR=-0.106, ОРТОГОНАЛЬНЫЙ
    "ret_skew_168h",        # rolling 168h skewness, ICIR=-0.103
]
```

### 5.6. Фичи R33 (тестированы, НЕ добавлены)

```python
# Тестировались, НЕ улучшили W3:
"btc_corr_168h"    # ICIR=+0.172, rolling 168h correlation with BTC
"btc_corr_24h"     # ICIR=+0.156, rolling 24h correlation with BTC
"upvol_24h"        # ICIR=-0.163, upside volatility
# Эти фичи СТАБИЛИЗИРУЮТ W2 (с -0.98 до +1.26), но снижают W3 peak
```

### 5.7. Фичи после R33 (тестированы, но не promoted в champion)

```python
# R35a — лучший новый bundle после R33
"ret_dispersion_12h"   # market-level dispersion, strongest scan score, чинит W2
"cs_rank_ma_5"         # rank persistence / CS momentum
"oi_chg_12h_cs"        # cross-sectional OI change
"taker_cvd_12h_cs"     # cross-sectional taker flow
"cum_funding_24h_cs"   # cross-sectional funding level

# R39 stablecoin pilot — сильный regime signal, но не direct champion add-on
"stable_total_supply_chg7d"
"stable_total_supply_chg30d"
"stable_supply_accel"
"stable_usdt_vs_usdc_chg7d"
"stable_total_supply_chg7d_z"
"stable_usdt_dom_z"
```

**Статус:**
- `R35a` даёт лучший post-R33 feature uplift: `W2=-0.98 -> +3.03`, `W3=1.91`, `ALL=0.64`
- `R35b` interactions блестяще работают только в `W3` (`3.34`), но не держат `ALL`
- `R35c` temporal bundle умеренно полезен (`ALL=0.62`)
- `R35d` market-only bundle подтвердил старый урок: market-level regime features в CS-модели легко превращаются в noise
- stablecoin bundle из `R39` мощно чинит `W2`, но ухудшает `W3/ALL`, значит это скорее gate/regime signal, а не прямая добавка к champion model

### 5.8. Полный каталог ВСЕХ тестированных фич и их статус

#### ✅ В production (FEAT_28):
| Фича | Группа | Описание | IC/ICIR | Статус |
|------|--------|----------|---------|--------|
| ret_12h | Returns | 12h return | — | Baseline |
| ret_24h | Returns | 24h return | — | Baseline |
| ret_48h | Returns | 48h return | — | Baseline |
| ret_168h | Returns | 7d return | IC=-0.044 | R31 added |
| residual_12h | Alpha | BTC residual 12h | — | Baseline |
| residual_24h | Alpha | BTC residual 24h | — | Baseline |
| mom_z_24h | Momentum | Momentum z-score 24h | — | Baseline |
| dist_from_high_24h | Price | Distance from 24h high | IC=+0.020 | R31 added |
| oi_chg_12h | Derivatives | OI change 12h | — | Baseline |
| oi_chg_24h | Derivatives | OI change 24h | — | Baseline |
| oi_zscore | Derivatives | OI z-score | — | Baseline |
| taker_cvd_12h | Flow | Taker CVD 12h | — | Baseline |
| taker_cvd_24h | Flow | Taker CVD 24h | — | Baseline |
| ls_divergence | Positioning | Long/Short divergence | — | Baseline |
| cum_funding_24h | Funding | Cumulative funding 24h | IC=-0.051 | R31 added |
| atr_14 | Volatility | ATR 14-period | — | R19 added |
| rvol_12h | Volatility | Realized vol 12h | — | R19 added |
| gk_vol_24h | Volatility | Garman-Klass vol 24h | — | R19 added |
| rvol_24h | Volatility | Realized vol 24h | — | R19 added |
| iv_rv_spread | Volatility | Implied-Realized vol spread | — | R19 added |
| pct_coins_up_12h | Breadth | % coins up 12h | — | R19 added |
| pct_coins_up_1h | Breadth | % coins up 1h | — | R19 added |
| hour_sin | Season | Hour of day (sin) | — | R19 added |
| hour_cos | Season | Hour of day (cos) | — | R19 added |
| dow_sin | Season | Day of week (sin) | — | R19 added |
| dow_cos | Season | Day of week (cos) | — | R19 added |
| rel_volume_cs | Volume | CS relative volume | ICIR=-0.106 | R32 added |
| ret_skew_168h | Distribution | Rolling 7d skewness | ICIR=-0.103 | R32 added |

#### ❌ Тестированы, не помогли / навредили:

| Фича | IC/ICIR | Почему отброшена |
|------|---------|------------------|
| btc_corr_168h | ICIR=+0.172 | W3 -0.67 vs baseline |
| btc_corr_24h | ICIR=+0.156 | W3 -0.13 vs baseline |
| upvol_24h | ICIR=-0.163 | W3 -1.20 vs baseline |
| mom_z_12h | — | Pruned in R12 (redundant) |
| funding_rate_binance | IC=-0.057 | R28: CATASTROPHE with extended features |
| global_ls_ratio | IC=+0.76Δ | R29d: data contamination found |
| taker_buy_sell_ratio | IC=+0.63Δ | R29d: data contamination found |
| buy_pressure | — | R30b: marginal |
| ret_std_24h | — | R30b: marginal |
| rsi_14, cci_14 | — | R30b: marginal |
| fng_value, fng_zscore | — | Noise for CS models |
| vix_close, dxy_close | — | Macro = noise for CS crypto |
| adx, mfi_14 | — | Marginal, not worth complexity |
| premium_zscore_12h | — | Not enough data coverage |
| oi_velocity | — | Noisy |
| taker_imb_z | — | Noisy |
| vol_of_vol | — | Noisy |
| fng_momentum | — | Noise |

#### ❌ Крупные feature sets которые ПРОВАЛИЛИСЬ целиком:

| Feature Set | Кол-во | Результат | Урок |
|-------------|--------|-----------|------|
| R28 FEAT_35 (+12 derivatives) | 35f | Ens Sh=0.06 | **КАТАСТРОФА** — больше фич = хуже |
| R28 FEAT_50 (+macro/TA) | 50f | LGB Sh=-0.03 | **КАТАСТРОФА** |
| R28 FEAT_65 (+momentum/vol) | 65f | LGB Sh=-0.69 | **САМЫЙ ХУДШИЙ** |
| R28 FEAT_75 (+engineered) | 75f | Ens Sh=-0.38 | Все провалилось |
| v8 (8 лет данных) | 165f | W2 Sh=-1.18 | Старые данные 2017-2020 = яд |

### 5.9. Production pipeline фичи (v6, ~165 фич — для регрессионных моделей)

Для полноты — в продакшн регрессионной модели используется ~165 фич:

**Returns (9)**: 1h, 2h, 4h, 6h, 12h, 24h, 48h, 72h, 168h
**Price shape (7)**: close_open_ratio, high_low_ratio, upper/lower shadow, body size
**MA ratios (11)**: close_ma{6,12,24,48,72,168,336,720}_ratio + vol_ma equivalents
**GK volatility (4)**: 12h, 24h, 48h, 168h
**Rolling stats (15)**: std/skew/kurt/mean/sharpe at 24h/48h/168h
**Volume (7)**: vol_mom, vwap_dev, vol_price_corr, buy_pressure
**TA (22)**: RSI (6,12,14,24), MACD, Bollinger (20,48), ATR (14,24,48), ADX, Stochastic, CCI, Williams %R, OBV, MFI
**Cross-asset (14)**: btc_ret_*, eth_ret_*, btc_vol_24h, market_dispersion, ret_vs_btc_24h
**Regime (5)**: btc_above_ma720, btc_dd_720, btc_not_crashed
**Macro (38)**: VIX/SPX/DXY/Gold/Yields/HY raw + changes + z-scores + interactions
**FNG (4)**: value, extreme_fear/greed, ma7/30, momentum
**Positioning (6)**: reversal_{4v24,12v48,24v168}, vol_surge_{12,24}h
**BTC beta (2)**: 48h, 168h
**News (9)**: sentiment, count, interactions (только CatBoost)
**Derivatives (17)**: funding, OI, taker flow, premium, long/short ratio
**DVOL (12)**: BTC/ETH implied vol, changes, spreads

---

## 6. Pipeline v1–v8: Эволюция пайплайна

### v1 — ТОТАЛЬНЫЙ ПРОВАЛ
- IC=0.005, Sharpe=-1.0, $1K→$6
- **Проблема**: time features (hour_sin/dow_cos) доминировали модель → leakage
- Нет CS normalization, overfitting (early stop at 51/2000 iterations)
- **Урок**: Time features = data leakage в CS models

### v2 — ПРОРЫВ
- Cross-sectional rank normalization, time features removed
- Rank IC=0.031, ICIR=0.36, **LS Sharpe=3.87** (gross)
- Ensemble 3 моделей (rank/excess/raw) → Sharpe=4.21
- **Урок**: CS rank normalization = essential. Long-Only = catastrophe.

### v3
- Multi-horizon targets (4h/12h/24h), cross-asset features, BTC regime filter
- 109 features, best=4h horizon (Sharpe 3.82)
- **Урок**: Regime filter USELESS (49.9% ON = coin flip)

### v4
- Feature selection (118→94), composite regime, 5-seed ensemble
- Best: Ensemble Sharpe=4.00

### v5 + HIST Transformer
- Sentiment (FNG, funding), risk overlay (vol targeting, DD breaker)
- HIST transformer: IC 0.0752 — cross-stock attention works
- HIST+LGB ensemble: Sharpe 4.38 (best gross на тот момент)
- **Урок**: Transformer работает, но сложная инфра, LGB проще и стабильнее

### v6 — PRODUCTION WORKHORSE (регрессия)
- **Критический фикс**: target aligned 12h (был 4h), matching 12h rebalance
- HORIZON=12, 5 seeds, PURGE_DAYS=8
- LGB: lr=0.01, max_depth=6, num_leaves=31, feature_fraction=0.5, bagging=0.7, min_child=200, L1/L2=1.0, n_estimators=5000
- ~165 features (NO news in production), gain-based pruning (bottom 20%)
- DDStop Sharpe 1.81

### v7
- Blended target: 75% ret_12h + 25% ret_24h
- +8 features: range_position, vwpc, hh/ll_count, trend_strength, vol_crush, direction_quality
- **Вывод**: v6 > v7. Correlation 0.957 — barely different. Occam's razor.

### v8 — ПРОВАЛ
- 8 лет данных (2017+), 5 purged WF windows
- W2 (2023 test): Sharpe **-1.18**
- **Урок**: В крипте 4 года > 8 лет. Рынок меняется слишком быстро. Старые данные = ЯД.

### CatBoost
- Ordered boosting, iterations=5000, lr=0.01, depth=6, l2_leaf_reg=3
- **С news в production** — единственная модель где news помогают
- DDStop Sharpe 1.51
- **Урок**: CatBoost лучше обрабатывает шумные фичи (news, derivatives)

### XGBoost
- GPU hist, 186 features (включая 23 news interactions nx_*)
- DDStop Sharpe 0.97-1.26, слабее чем ensemble

---

## 7. Research Rounds R1–R39: Полная хронология

### Фаза 1: Ridge Model + Portfolio Construction (R2–R7B)

#### R2
- Ridge model, 14 фич, alpha grid [0.01-1000], CS rank target
- Тесты: больше символов (30+), multi-horizon blend, momentum overlay, dynamic position count
- Regime filter с BTC trend strength

#### R3 / R3B
- 50 символов, edge-weighted positions, 5x leverage
- SYM_35 confirmed as sweet spot
- vol_regime для dynamic exposure

#### R4
- Dispersion-aware sizing, rolling IC filter, equity momentum, fine-tuned dynamic exposure

#### R5
- Enhanced EQ-MOM, Kelly sizing, signal agreement, adaptive trend cutoff, sector-neutral, time-of-day filter

#### R6 / R6B
- Spread gate, regime-dependent N, strategy_momentum meta-signal
- Best R6: SM48+6L3S → $1720
- ~25 конфигурационных комбинаций

#### R7 — КЛЮЧЕВОЙ ПОВОРОТ
- Methodology AUDIT с explicit `audit_methodology()` function
- Best config: 6L/3S, cutoff=0.8, dyn=0.5, eq_mom, kelly, SM48, regime_asym, vol_scale, ema=2
- **Sharpe baseline: ~$2226**

#### R7B
- Combos best settings: SHRINK=0.1 ($1829 SAFEST), VOL-SCALE 7L5S ($2208 HIGHEST)

### Фаза 2: Feature Expansion + LightGBM (R8–R9B)

#### R8
- Расширение 14→80+ фич: DVOL, Macro, Extended funding, Volume momentum, Cross-coin dispersion
- `build_features_extended()` with ~80+ candidates

#### R9 — ОТКРЫТИЕ LGBM 🏆
- **KEY FINDING**: LightGBM IC 0.060-0.072 vs Ridge IC 0.013-0.027 (**5x лучше!**)
- LGB params: lr=0.05, num_leaves=31, min_child=100, subsample=0.8, colsample=0.8
- Result: Sh=4.21 (+0.62 vs baseline)
- **Урок**: Нелинейные модели кардинально лучше Ridge для этих данных

#### R9B
- LGB + signal tweaks: EMA=None/3/4, shrink=0.05-0.15

### Фаза 3: LGB Optimization + Feature Pruning (R11–R15.5)

#### R11
- Baseline: R10 LGB 5-seed nl=31, 14 feats → Sh=4.07
- Seeds: [0, 7, 13, 42, 99], lr=0.05, 500 rounds, early_stop=30
- Тесты: num_leaves sweep, expanded features, interaction features

#### R12 — LEAKAGE AUDIT
- 5 проверок: window gaps, overlap, fwd_ret formula, CS-ranking, features backward-looking
- Feature pruning: 14→12 (dropped dist_from_high_24h, mom_z_12h)
- Best: nl=63, TS-z 16f → Sh=4.37

#### R13
- FEATURES_12 confirmed
- PROD_PARAMS: lr=0.03, nl=63, min_child=100, L2=1.0
- Best: Sh=4.77, Eq=$5280

#### R14
- Extended walk-forward (5 windows), XGBoost comparison, bootstrap CI
- Target alternatives: 6h/8h/24h → 12h confirmed best

#### R15 / R15.5
- Fine-grained HP grid, DART mode, target winsorization, Extra Trees
- Best: Extra Trees (Sh=4.93), Winsorize 1% (Sh=4.87)

### ⚠️ R16 — КРИТИЧЕСКИЙ АУДИТ

**Найдены 2 бага:**
1. **Over-annualization**: Sharpe был завышен на 56%
2. **eq_mom_boost look-ahead bias**: заглядывание в будущее

**Реальный Sharpe ≈ 1.75** (barebone L/S), не 4.77!

### Фаза 4: Leakage Fix + Classification Discovery (R18–R21)

#### R18
- Расширение фич: News sentiment, TA, FNG, DVOL, Macro, derivatives
- Extended от 12 до множества кандидатов

#### R19 — ФИКС УТЕЧКИ В IC SCAN
- **Проблема**: IC scan считался на TEST data → selection bias
- **Фикс**: IC scan только на TRAIN
- FEATURES_17 → FEATURES_23 (новые: breadth, seasonality)
- R19 winner: LGB-23f, Sh=2.50

#### R20
- cutoff=0.9 → Sh=2.80, permutation p=0.0033
- 6h rebal → Sh=2.88 (казалось лучше)

#### R21 — КРИТИЧЕСКИЙ БАГ
- **simulate() uses fwd_ret_12h but rebalances every 6h → hours 6-12 counted TWICE → Sharpe inflated by √2**
- Все результаты 6h rebal = артефакт!
- **РЕАЛЬНЫЙ BEST**: cutoff=0.9, 12h rebal → Sh=2.80

### Фаза 5: Multi-Model + Ensemble (R22–R27)

#### R22
- HPO Optuna (40 trials), XGBoost, CatBoost, stacking
- Ничего не побеждает LGB-23f Sh=2.80

#### R23 — ПРОРЫВ: Classification 🏆
- **Binary classification P(ret>0) → Sh=2.94** vs regression Sh=2.80
- Worst month: -13.8% (вдвое лучше regression)
- **Урок**: Для портфеля важен RANK, не точное значение → classification лучше

#### R24
- HPO for classification, CLS+EMA, blend, LambdaRank
- Best: 5L/3S → Sh=2.98

#### R25 — LGB+XGB ENSEMBLE 🏆
- **LGB+XGB binary cls, 5L/3S → Sh=3.36, worst=-5.7%**
- Формула: `0.5 * rank(LGB_prob) + 0.5 * rank(XGB_prob)`

#### R26
- 6L/3S → Sh=3.39 (marginally better)
- Multi-Class: Sh=1.36 (fail), Focal Loss: crash, Interactions: Sh=1.74
- **Вывод**: Изменения модели ВСЕ ПРОВАЛИЛИСЬ. Только portfolio config marginal.

#### R27 — SIGNAL PLATEAU
- Multi-horizon blending, temporal weighting, meta-stacking
- ALL ≤ Sh=2.90 (хуже baseline 3.39)
- **ВЫВОД: Sharpe plateau at ~3.39. ВСЕ изменения модели/сигнала не помогают.**

### Фаза 6: Feature Expansion + Realistic Costs (R28–R33)

#### R28 — FEATURE CURSE 🚫
- FEAT_35 (baseline+12 derivatives): LGB Sh=-0.00, Ens Sh=0.06 → **КАТАСТРОФА**
- FEAT_50 (+macro/TA): LGB Sh=-0.03 → КАТАСТРОФА
- FEAT_65 (+momentum/vol): LGB Sh=-0.69 → **WORST EVER**
- **Урок**: Больше фич = хуже. funding_rate_binance доминирует gain importance но не помогает OOS.

#### R29 / R29b / R29c / R29d — Forward Selection
- 62 кандидата тестированы по одному
- Лучшие: global_ls_ratio (+0.76Δ), taker_buy_sell_ratio (+0.63Δ)
- **НО R29d обнаружил data contamination** — FEAT25 не бьёт чистый baseline Sh=3.39

#### R30 / R30b — Realistic Costs
- Модель потеряла $14 за 2 дня live
- **Открытие**: Costs = 40-47% drag!
- Gross Sh=3.01, **Net Sh=1.29** (6L3S, 12h rebal without smoothing)
- **R30b fixed**: double-leverage on costs bug
- EMA smoothing + hysteresis → turnover 4-5 вместо 10-11 → значительно лучше net

#### R31 — High-IC Features
- IC analysis на TRAIN:
  - ret_168h (IC=-0.044, STRONGEST)
  - cum_funding_24h (IC=-0.051)
  - dist_from_high_24h (IC=+0.020)
- FEAT_26 = FEAT_23 + [ret_168h, cum_funding_24h, dist_from_high_24h]
- Improved over 23f, but FEAT_29/32 = overfitting

#### R32 — Kaggle Features 🏆
- rel_volume_cs (ICIR=-0.106, **ORTHOGONAL** ко всем 26 фичам!)
- ret_skew_168h (ICIR=-0.103)
- FEAT_28 = FEAT_26 + [rel_volume_cs, ret_skew_168h]
- **Results**: A_26f W3=1.75 → D_28f W3=2.88 (+1.13!) 🎯

#### R33 — Creative Features (FINAL)
- 45 creative features scanned, top: btc_corr_168h (ICIR=+0.172), btc_corr_24h (ICIR=+0.156)
- **Walk-forward results** (6L3S_ema05_h3):

| Experiment | W1 | W2 | W3 | ALL | Eq$ | DD% |
|-----------|-----|------|------|------|-----|------|
| **A_28f (baseline)** | -0.69 | -0.98 | **2.88** | 0.47 | $201 | -29.5% |
| C_29f_24 (+btc_corr_24h) | -0.59 | **1.53** | 2.75 | 0.94 | $198 | -27.2% |
| D_30f (+both btc_corr) | **0.57** | **1.26** | 2.35 | **1.00** | $176 | -43.0% |
| B_29f_168 (+btc_corr_168h) | -0.13 | 1.05 | 2.21 | 0.64 | $164 | -32.3% |
| E_30f_alt (+btc_corr+upvol) | -0.53 | 0.68 | 1.68 | 0.90 | $141 | -44.6% |

**ВЕРДИКТ R33**: Новые фичи НЕ помогают W3. 28f всё ещё лучший.
- btc_corr фичи драматически фиксят W2 (с -0.98 до +1.26)
- Но снижают W3 peak (с 2.88 до 2.35)
- Tradeoff: stability vs peak performance

**Monthly IC для A_28f**:
```
2024-10: +0.0571  2024-11: +0.0422  2024-12: +0.0348
2025-01: +0.0060  2025-05: +0.0320  2025-06: +0.0236
2025-07: +0.0097  2025-08: +0.0161  2025-11: +0.0440
2025-12: +0.0321  2026-01: +0.0411  2026-02: +0.0191
2026-03: +0.0171
Mean IC: 0.0288, IC>0: 13/13, ICIR: 1.99
```

### Фаза 7: Diagnostics + Feature Search + Execution (R34–R39)

#### R34 — W2 Attribution ✅
- **Ключевая находка**: `W2` ломается из-за long leg, а не из-за short leg.
- `long_leg_sharpe=-1.71`, `short_leg_sharpe=+0.38`, `portfolio_sharpe=-1.01`.
- Official cross-check подтвердил корректность trace-диагностики:
    - `W1`: official `-0.69` vs trace `-0.65`
    - `W2`: official `-0.98` vs trace `-1.01`
    - `W3`: official `2.88` vs trace `2.88`
- Conditional IC по режимам:
    - `dist_from_high_24h` и `iv_rv_spread` стабильно лучшие почти во всех биннингах
    - high-gain `W2` features (`cum_funding_24h`, `mom_z_24h`, `ret_24h`, `gk_vol_24h`) имеют отрицательный post-hoc test IC
- Токсичные `W2` long names: `XRP`, `ADA`, `SAND`, `APT`
- Лучшие `W2` shorts: `LDO`, `INJ`, `SNX`, `ARB`
- Rank correlation `W2` vs `W3` почти одинаковая (`~0.56`), но bottom-book churn в `W2` заметно хуже (`0.664` vs `0.577`)
- **Вывод**: проблема `W2` — не общий развал ранжирования, а плохая long-leg экспозиция и regime-mismatch у части market-sensitive features.

#### R35 — New Features From Existing Data ★★★
- Протестированы 4 группы новых фич:
    - `R35a`: CS second-order
    - `R35b`: interactions
    - `R35c`: temporal structure
    - `R35d`: market-level derivatives
- **R35a (CS second-order) — лучший новый bundle после R33**:
    - `ret_dispersion_12h`, `cs_rank_ma_5`, `oi_chg_12h_cs`, `taker_cvd_12h_cs`, `cum_funding_24h_cs`
    - `W1=-0.45`, `W2=3.03`, `W3=1.91`, `ALL=0.64`
- `R35b interactions`: `W2=2.55`, `W3=3.34`, но `ALL=-0.00`
- `R35c temporal`: `W2=0.85`, `W3=1.90`, `ALL=0.62`
- `R35d market`: `W1=-3.01`, `ALL=-0.13`
- **Вывод**:
    - `R35a` реально добавляет ортогональный сигнал и становится главным кандидатом на дальнейшую абляцию
    - market-level-only bundle снова подтверждает, что общий режимный шум плохо заходит прямо в CS-модель

#### R36 — Regime Gating
- Сравнивались 3 эксперта:
    - `expert_base = A_28f`
    - `expert_stability = D_30f`
    - `expert_stable_flow = A_28f + stable_flow4`
- Лучший standalone по `ALL`: `expert_stability = D_30f`
    - `W1=-0.47`, `W2=1.06`, `W3=2.84`, `ALL=0.92`
- Лучший hard gate по `W2`: `gate_stable_base_vs_flow`
    - `W2=3.72`, но `ALL=0.29`
- `gate_tri_regime`: `W2=3.17`, `W3=2.24`, `ALL=0.15`
- **Вывод**: regime gating реальный, но текущие hard-switch rules слишком хрупкие; по `ALL` они не бьют простой стабильный эксперт.

#### R37 — Cost-Aware Execution
- Тестированы no-trade bands, edge thresholds, liquidity floors (`liq60`, `liq70`) и комбинации.
- Ключевой победитель: `liq70`
    - `W2=1.47`, `W3=2.87`
    - `cost_pct`: `W2=3.86%`, `W3=4.05%`
    - turnover резко ниже baseline (`~1.5-1.7` vs `~4.2-4.4`)
- Но `ALL` у `liq70` оказался `-0.04`, а baseline path внутри этого sweep всё ещё лучший по `ALL=0.74`
- **Вывод**: liquidity filter = сильный execution lever для отдельных окон и costs, но сам по себе не гарантирует лучший `ALL`; его надо тестировать в комбинации с лучшим signal bundle.

#### R38 — Target Engineering 🚫
- Проверены:
    - `P(ret > 0.5/1.0/1.5/2.0%)`
    - `ret - btc_ret`
    - temporal decay weighting (`90d`, `180d`)
- Результаты:
    - baseline `P(ret > 0)` всё ещё лучший по `ALL=0.71`
    - threshold targets = катастрофа, особенно в `W2` (`-2.95` ... `-5.06`)
    - `excess_vs_btc`: `ALL=-0.56`
    - decay variants слегка помогают `W2`, но ухудшают `ALL`
- **Вывод**: target engineering не ломает representation limit; текущий binary target остаётся лучшим.

#### R39 — Dead Data Activation / Stablecoin Pilot
- `R39.1` inspection:
    - `stablecoin_supply.parquet` — лучший из dead-data источников: полный daily regime candidate
    - `defi_tvl_daily.parquet` — частично покрытый breadth/regime source
    - `onchain_daily.parquet` — usable only for subset / BTC-ETH-heavy regime info
- `R39 stablecoin pilot`:
    - `stable_flow4`: `W1=-0.67`, `W2=2.46`, `W3=1.78`, `ALL=0.17`
    - `stable_regime6`: `W1=-2.95`, `W2=2.27`, `W3=0.14`, `ALL=-0.72`
- По ходу `R39` найден и исправлен pipeline issue:
    - market-level features схлопывались после cross-sectional ranking
    - в `train_ensemble()` добавлен `cs_rank_exclude`, чтобы regime features не занулялись
- Дополнительно разблокирован `D5`:
    - `stablecoins.llama.fi` downloader успешно сохранил snapshots + global/chain history locally
- **Вывод**:
    - stablecoin flows содержат реальный regime signal
    - но как direct feature add-on они вредят champion configuration по `W3/ALL`

### Итог после R34-R39

| Config | W1 | W2 | W3 | ALL |
|--------|-----|------|------|------|
| **A_28f baseline** | -0.69 | -0.98 | **2.88** | 0.47 |
| **R35a (CS second-order)** | -0.45 | **3.03** | 1.91 | 0.64 |
| **D_30f (stability expert)** | -0.47 | 1.06 | 2.84 | **0.92** |
| **stable_flow4** | -0.67 | 2.46 | 1.78 | 0.17 |

**Новый урок фазы R34-R39**:
- `W2` можно чинить, но почти все fixes платятся ухудшением `W3` или `ALL`
- лучший новый signal bundle = `R35a`
- лучший stable overall expert = `D_30f`
- лучший execution lever = `liq70`
- target engineering и direct market-level feature add-ons не являются главным путём вперёд

### R41–R46: финальный research sprint (апрель 2026)

**R41 — Consolidation matrix**: Попарные комбинации A_28f / R35a / D_30f / liq70 не стакаются.
Best = A_28f + D30 overlap features → ALL 0.74. R35a + D_30f → ALL 0.64.
**Вывод**: перекрытие сильнее синергии.

**R42 — Ablation R35a (🏆 НОВЫЙ ЧЕМПИОН)**:
R35a = 5 features: `ret_dispersion_12h`, `cs_rank_ma_5`, `oi_chg_12h_cs`, `taker_cvd_12h_cs`, `cum_funding_24h_cs`.
Тестировали все C(5,2)=10 пар → `dispersion + rankma` = **ALL 1.13** (W2=3.22, W3=2.50).
Вторая пара: `dispersion + funding_cs` = ALL 0.98.
Full R35a (все 5) = ALL 0.64 — 3 лишних фичи добавляли noise.
**Урок**: minimal feature set >> saturated bundle.

**R43 — Dynamic exposure**: 5L/4S и 4L/4S при слабом breadth.
Best = 5L4S ALL 0.87. Но 6L3S baseline = 1.13. Уменьшение net long отдаёт alpha.
**Вывод**: long bias IS the alpha.

**R44 — Dynamic universe / quality filter**: Hard volume и trailing-IC фильтры.
Volume top-20 = ALL 0.82. IC-weighted = 0.65. Baseline 35 coins = 1.13.
**Вывод**: hard pruning контрпродуктивно, модель сама умеет downweight слабые.

**R45 — Calibrated soft gate / expert blend**:
`expert_stability` (D_30f) standalone = ALL 0.92.
Soft blend 70/30 champion × stability = ALL 0.85. EMA blend = 0.78.
**Вывод**: blending не побил лидера. Standalone пики не аддитивны.

**R46 — Separate long/short models**:
P(ret>median) для long, P(ret<p25) для short. Turnover 2.4 (vs 4.5), cost 12.2% (vs 19.2%).
Но ALL = 0.54. Потеря alpha >> экономия на costs.
**Вывод**: unified model лучше. Asymmetric learning не нужен.

### Итог после R41-R46

| Config | W1 | W2 | W3 | ALL |
|--------|-----|------|------|------|
| **A_28f + dispersion + rankma** | 0.01 | **3.22** | **2.50** | **1.13** |
| **D_30f (expert_stability)** | -0.47 | 1.06 | 2.84 | 0.92 |
| Dynamic 5L4S | -0.11 | 2.70 | 2.14 | 0.87 |
| Volume-20 universe | -0.12 | 2.50 | 1.90 | 0.82 |
| A_28f baseline | -0.69 | -0.98 | 2.88 | 0.47 |
| Asymmetric L/S | -1.17 | 1.80 | 2.08 | 0.54 |

**Ключевой урок R41-R46**: diminishing returns от model/feature engineering. Путь вперёд — НОВЫЕ ДАННЫЕ + execution improvement.

### R47 — CoinGlass Feature Research (🏆 НОВЫЙ CHAMPION CANDIDATE)

**Цель**: Протестировать CoinGlass derivatives data (ликвидации, taker buy/sell, L/S ratio) на incremental alpha.

**Данные**: CoinGlass API, 5 endpoints × 35 symbols:
- 1d interval: 259,412 rows total (5 endpoints), 2022-01-01 → 2026-04-05 (покрывает все 3 WF окна)
- 12h interval: 25,200 rows/endpoint, 2025-04-11 → 2026-04-05

**Протокол**: QA → IC scan (TRAIN only) → redundancy check → event study → per-feature WF ablation.

**QA findings**:
- Свечи CoinGlass ОТКРЫВАЮТСЯ в timestamp, покрывают [t, t+24h). Shift=1 день (lookahead-safe).
- `cg_date = floor(ohlcv_timestamp, 'D') - 1 day` — берём вчерашний полный daily candle
- MATIC + FTM исключены (нет funding/ls_ratio)
- 4.4% нулей (норма), аномалия Oct 10 2025 $1.87B BTC liq — реальное событие

**Построены фичи (11 штук)**:
- Per-symbol (CS-ranked): `cg_liq_total`, `cg_liq_imbalance`, `cg_liq_zscore`, `cg_liq_accel`, `cg_taker_imb`, `cg_taker_imb_z`, `cg_ls_ratio`, `cg_ls_zscore`
- Market-level (НЕ CS-ranked): `mkt_cg_liq_total`, `mkt_cg_liq_log`, `mkt_cg_liq_imb`
- Исключена: `cg_liq_intensity` (r=0.89 с `rel_volume_cs` — артефакт формулы)

**IC Scan (TRAIN, all 35 sym, 3 WF windows)**:

| Flag | Feature | mean_IC | ICIR |
|------|---------|---------|------|
| 🔥 | `mkt_cg_liq_imb` | +0.086 | +0.479 |
| 🔥 | `cg_liq_imbalance` | +0.063 | +0.509 |
| 🔥 | `cg_taker_imb` | -0.032 | -0.324 |
| 🔥 | `cg_taker_imb_z` | -0.031 | -0.292 |
| ✅ | `cg_liq_zscore` | +0.019 | +0.151 |
| ✅ | `cg_ls_zscore` | +0.018 | +0.207 |

**Redundancy**: `mkt_cg_liq_imb` ~ `ret_48h` (r=-0.57), `cg_taker_imb` ~ `ret_48h` (r=+0.39, менее redundant).

**Event study**: short_liq_dom → fwd_ret_12h=+0.0008, t=+4.16 (статистически значимо — short squeeze effect).

**Walk-Forward Ablation**:

| Config | W1 | W2 | W3 | ALL | Δ ALL |
|--------|-----|------|------|------|-------|
| **champion+cg_taker_imb** | **0.60** | 2.86 | 2.03 | **1.31** | **+0.18** |
| champion_30f (baseline) | 0.01 | 3.22 | 2.50 | 1.13 | 0.00 |
| champion+liq_log+liq_total | -0.60 | 2.33 | 1.76 | 1.03 | -0.10 |
| champion+cg_ls_zscore | -0.91 | 4.03 | 2.17 | 0.54 | -0.59 |
| champion+mkt_cg_liq_log | -1.50 | -0.56 | 1.77 | 0.07 | -1.06 |
| champion+cg_liq_zscore | -1.49 | 2.79 | 2.33 | -0.04 | -1.17 |

**R47 ключевые выводы**:
1. `cg_taker_imb` — единственная CG фича, улучшившая ALL (1.13→1.31, +16%). Формула: `(buyVol-sellVol)/(buyVol+sellVol)`, CS-ranked, shift=1 day.
2. **Парадокс**: высший IC (liq_imbalance IC=0.086) УБИВАЕТ модель в WF (ALL→0.07), а скромный taker_imb (IC=-0.032) — побеждает. Причина: redundancy с ret_48h + multicollinearity.
3. W1 прыгнул 0.01→0.60 (самое проблемное окно). W2/W3 чуть ослабли.
4. Market-level CG features вредят (в отличие от ret_dispersion_12h).
5. CoinGlass подписка оправдана ($29/мес): `cg_taker_imb` даёт +0.18 Sharpe.

**Файлы**: `_research_r47_qa.py` (QA), `_research_r47_coinglass.py` (IC+WF), `results_r47.log`, `results_r47_summary.csv`

### R48 — Hybrid Costs + Feature Validation (🏆 НОВЫЙ CHAMPION ALL=1.66)

**Цель**: Валидация champion_31f, тестирование производных от cg_taker_imb, residualized liquidations, и гибридная модель стоимости исполнения.

**Скрипты**: `_research_r48_validation.py`, `_research_r48_features.py`, `_research_r48_cost.py`, `_research_r48_combo.py`

#### R48.0 — Validation champion_31f
- Timestamp check: CoinGlass shift=1d → lookahead-safe ✅
- **Bootstrap**: P(ΔSharpe>0) = 69.6% — MARGINAL (ниже порога 80%, но не катастрофа)
- Monthly stability: 10/18 win months (56%), top-2 months = 57% → ✅ распределено
- Mean monthly IC = +0.022, 9/13 положительных месяцев
- Clip sensitivity: raw=1.31, clip_98=1.31, winsorize=0.94, gaussian_rank=0.22 → raw лучший, умеренно устойчив
- **Вывод**: champion_31f принят с оговоркой (bootstrap маргинальный)

#### R48.1 — Taker derivatives (Phase 1)

| Config | W1 | W2 | W3 | ALL | Δ ALL |
|--------|-----|------|------|------|-------|
| baseline champion_31f | +0.60 | +2.86 | +2.03 | **+1.31** | 0.00 |
| +cg_taker_imb_ma3 | -0.50 | +3.37 | +3.13 | +0.94 | -0.37 ❌ |
| +cg_taker_imb_delta | +1.09 | +1.04 | +2.76 | +1.23 | -0.08 ❌ |
| +cg_taker_imb_cs_demean | -0.45 | +2.68 | +3.00 | +0.73 | -0.58 ❌ |

**Вывод**: Никакие taker derivatives не улучшают ALL=1.31. Оригинальный cg_taker_imb оптимален.

#### R48.2 — Residualized liquidations (Phase 2)

Попытка убрать redundancy между cg_liq_imbalance и ret_48h через residualization:

| Config | IC | ICIR | WF ALL | Δ ALL |
|--------|-----|------|---------|-------|
| cg_liq_imb_resid_bin | +0.060 | +0.48 | +0.71 | -0.60 ❌ |
| cg_liq_imb_resid_roll | -0.002 | -0.02 | +0.61 | -0.70 ❌ |
| mkt_cg_liq_imb_resid | +0.059 | +0.33 | +0.13 | -1.18 ❌ |

**Вывод**: Residualization НЕ спасает liquidation features. Несмотря на хороший IC, все хуже baseline. cg_liq_imbalance фундаментально redundant с ret_48h.

#### R48.3 — Hybrid Tiered Cost Model (Phase 3) 🏆

Тестировалось три модели стоимости для champion_31f:

| Cost Model | W1 | W2 | W3 | ALL | Cost% |
|-----------|-----|------|------|------|-------|
| Uniform 7bps | +0.60 | +2.86 | +2.03 | +1.31 | 19.2% |
| **Hybrid Tiered** | **+0.98** | **+3.53** | **+2.52** | **+1.66** | **9.8%** |
| Liq-Weighted | +0.17 | +0.18 | +1.95 | +0.53 | 27.3% |

Hybrid тiers:
- TIER1 (BTC/ETH/SOL/BNB/XRP): 92% maker fill @ -1bp + 8% taker @ 7bp → ~0.4bp effective
- TIER2 (ликвидные altcoins): 75% maker @ 1bp + 25% taker @ 7bp → ~2.5bp effective
- TIER3 (остальные): taker + slippage = 7bp

**Вывод**: Hybrid costs = ПРОРЫВ. Стоимость 19.2%→9.8% (-9.4%), Sharpe 1.31→1.66 (+27%). Liq-weighted хуже (overfits small caps).

#### R48.4 — Best Combo (Phase 4)

Финальное сравнение с hybrid costs:

| Config | W1 | W2 | W3 | ALL | Cost% |
|--------|-----|------|------|------|-------|
| **A: champion_31f + hybrid** | **+0.98** | **+3.53** | **+2.52** | **+1.66** | **9.8%** |
| E: champion_30f + hybrid (ref) | +0.40 | +3.89 | +2.99 | +1.51 | 9.7% |

**🏆 NEW CHAMPION: champion_31f + hybrid tiered costs → ALL=1.66**
- cg_taker_imb остаётся essential: 30f=1.51, 31f=1.66 (+0.15 при том же cost)
- Hybrid costs = основной драйвер (1.31→1.66, -48% cost drag)

**Итоговая таблица R48**:

| Config | W1 | W2 | W3 | ALL | Cost% |
|--------|-----|------|------|------|-------|
| **R48 Champion**: 31f hybrid | +0.98 | +3.53 | +2.52 | **1.66** | **9.8%** |
| Prev champion: 31f uniform | +0.60 | +2.86 | +2.03 | 1.31 | 19.2% |
| 30f hybrid (no CG ref) | +0.40 | +3.89 | +2.99 | 1.51 | 9.7% |
| 30f uniform | +0.01 | +3.22 | +2.50 | 1.13 | 19.2% |

**Ключевые R48 уроки**:
1. Hybrid maker/taker cost model = крупнейший execution improvement в истории проекта
2. Taker derivatives (ma3/delta/demean) не помогают — оригинал оптимален
3. Residualization не спасает liq_imbalance — redundancy structure фундаментальная
4. cg_taker_imb по-прежнему valuable: +0.15 Sharpe на hybrid costs
5. Bootstrap маргинален (69.6%), но strategy реальна (monthly stability подтверждена)

---

### R55 — CoinGlass Basis/FR/OI IC Scan (✅ pandas 2.3.3, 2026-04-06)

**Цель**: IC scan новых CoinGlass фич (`cg_basis_z_60d`, `cg_fr_close`, `cg_fr_disagreement`, `cg_basis_close`, `cg_oi_chg_1d`) на 35-монетном universe.

**IC результаты** (cross-val на train split, R56 baseline IC):

| Feature | Coverage | IC W1 | IC W2 | IC W3 | IC ALL | Заметки |
|---------|----------|--------|--------|--------|--------|---------|
| `cg_basis_z_60d` | 27/35 (77%) | +0.052 | +0.054 | +0.056 | **+0.054** | winner, но неполное покрытие |
| `cg_basis_close` | 27/35 (77%) | — | — | — | +0.045 | коварна с basis_z |
| `cg_fr_close` | 33/35 (94%) | — | — | — | -0.064 | r=0.86 с cum_funding (redundant!) |
| `cg_fr_disagreement` | 33/35 (94%) | — | — | — | -0.047 | novel, но отрицательный IC |
| `cg_oi_chg_1d` | 33/35 (94%) | — | — | — | -0.056 | отрицательный IC |

**Ключевые уроки R55**:
1. `cg_basis_z_60d` — единственный winner по IC (+0.054 ALL), но покрытие 27/35 — риск
2. `cg_fr_close` коллинеарен с `cum_funding_24h` (r=0.86) — redundant, не добавляет информацию
3. Отрицательный IC ≠ бесполезная фича (модель может выучить обратный знак), но слабый сигнал

---

### R56 — CG Feature Substitution WF Ablation (✅ pandas 2.3.3, 2026-04-06)

**Цель**: Заменить champion фичи на candidate CG фичи (R55) в walk-forward симуляции. Все на 35-coin universe vs baseline_35=+1.66.

**Feature importance Phase 0** (LGB gain, champion 31f):

| Фича | Gain | Примечание |
|------|------|----------|
| `cum_funding_24h` | 18831 | **#1**, незаменимая |
| `pct_coins_up_12h` | 13509 | #2 |
| `mom_z_24h` | 9542 | #3 |
| `dow_cos` | 0 | мёртвая |
| `dow_sin` | 0 | мёртвая |
| `hour_cos` | 0 | мёртвая |

**Coverage penalty**: baseline_27 (только монеты с basis данными) = ALL=+0.09 vs baseline_35=+1.66 → **Δ=-1.57**. 8 монет без basis датчика несут огромную alpha.

**Результаты замены фич** (все сравнения vs baseline_35=+1.66):

| Exp | Замена | ALL | W1 | W2 | W3 | ΔSharpe | Решение |
|-----|--------|-----|----|----|-----|---------|---------|
| baseline | 35-coin champion 31f | +1.66 | +0.98 | +3.53 | +2.52 | — | **чемпион** |
| 2.1 | `dow_cos` → `basis_z` | +0.91 | -0.35 | +2.30 | +2.03 | -0.75 | ❌ |
| 2.2 | `cum_funding` → `basis_z` | +0.97 | -0.29 | +2.30 | +1.94 | -0.69 | ❌ |
| 3.1 | `cum_funding` → `fr_close` | +1.36 | +0.35 | +2.67 | +0.46 | -0.30 | ❌ |
| 3.2 | `cum_funding` → `fr_disagree` | +0.75 | -0.45 | +2.16 | +1.45 | -0.91 | ❌ |
| 3.3 | `oi_zscore` → `oi_chg_1d` | +0.32 | -1.13 | +2.93 | +2.60 | -1.34 | ❌ |

**Ключевые R56 уроки**:
1. **Ни одна CG/R55 фича НЕ улучшает champion при замене.** Чемпион 31f hybrid ALL=+1.66 остаётся.
2. Basis фичи (27/35 coverage) сильно штрафуются: 8 монет без basis дают Δ=-1.57 alpha.
3. `cum_funding_24h` #1 по важности — незаменима. Любая замена сильно ухудшает.
4. Добавление 32-й фичи (вместо замены) — **не тестировалось**, открытый вопрос для R57.

---

### R56b — Dead Feature Swap (✅ pandas 2.3.3, 2026-04-06)

**Цель**: Заменить мёртвые фичи (gain=0: `dow_cos`, `hour_cos`, `dow_sin`) на CG кандидатов — consultant recommendation "replace dead, not top-1".

| Exp | Замена | ALL | W1 | W2 | W3 | ΔSharpe | Решение |
|-----|--------|-----|----|----|-----|---------|---------|
| B0 | baseline 31f | +1.66 | +0.98 | +3.53 | +2.52 | — | **чемпион** |
| B1 | `dow_cos` → `fr_disagree` | +0.94 | -0.47 | +1.51 | +2.89 | -0.72 | ❌ |
| B2 | `hour_cos` → `fr_disagree` | +0.78 | -0.98 | +1.86 | +3.10 | -0.88 | ❌ |
| B3 | `dow_cos` → `oi_chg_1d` | +0.76 | -0.22 | +3.66 | +1.72 | -0.90 | ❌ |

**Ключевые R56b уроки**:
1. Даже замена мёртвых фич (gain=0) на CG кандидатов даёт -0.72 … -0.90 Sharpe.
2. `cg_fr_disagreement` разрушает W1 и W2 вне зависимости от заменяемой фичи.
3. `cg_oi_chg_1d` показывает хорошо в W2 (+3.66 > baseline +3.53), но ломает W1/W3.
4. **Итог R55+R56+R56b**: Ни одна CG фича не проходит WF-тест на замену. Champion 31f hybrid ALL=+1.66 остаётся финальным чемпионом.

### R60 — Portfolio Construction Optimization (✅ 2026-04-06)

5 режимов портфельной конструкции поверх gen8 champion 31f. ORIGINAL_WINDOWS, hybrid tiered costs, 5 seeds.

| Mode | Sharpe | Ret% | MaxDD | WR% | Комментарий |
|------|--------|------|-------|-----|-------------|
| **grid_4L2S** | **2.98** | +77.4% | -14.1% | 60% | **WINNER** — 4L/2S |
| baseline 6L3S | 1.78 | +33.9% | -15.0% | 58% | Текущий прод |
| edge_cost_filter | 1.60 | — | — | — | Провал |
| prob_weighting | 1.58 | — | — | — | Провал |
| dynamic_K | 1.51 | — | — | — | Провал |

**Вывод**: Уменьшение K с 6L/3S до 4L/2S — единственное осмысленное улучшение портфельной конструкции. Позиции 5-6 в лонге добавляют шум.

### R61 — Temporal Features (✅ 2026-04-06)

12 temporal features (lags + rolling) в 5 ablation-вариантах.

| Вариант | Features | Sharpe | Δ | Комментарий |
|---------|----------|--------|---|-------------|
| **+cg_temporal** | 31+4=35f | **1.89** | +0.11 | cg_taker_imb lags помогают |
| baseline_31f | 31f | 1.78 | — | |
| +oi_temporal | 34f | 1.39 | -0.39 | Hurt |
| all_43f | 43f | 1.01 | -0.77 | Overfit |
| +ret_temporal | 36f | 0.48 | -1.30 | Catastrophic |

**Вывод**: ret_12h лаги категорически вредны. cg_taker_imb lags дают маргинальный +0.11, но в combined run (R64) ухудшают.

### R62 — Meta-Stacking LogReg+GRU (✅ 2026-04-06)

OOF predictions от LogReg (p_lin) и GRU micro-model (p_seq) как доп. фичи.

| Вариант | Sharpe | Δ |
|---------|--------|---|
| baseline_31f | 1.78 | — |
| +p_seq (GRU) | 1.48 | -0.29 |
| +p_lin+p_seq | 0.50 | -1.28 |
| +p_lin (LogReg) | -0.38 | -2.16 |

**Вывод**: Meta-stacking не работает. OOF от более слабых моделей = шум для LGB/XGB. Тема закрыта.

### R63 — Uncertainty Gating / Seed Disagreement (✅ 2026-04-06)

p_std по 10 моделям (5 seeds × LGB+XGB) как фильтр неуверенных позиций.

| Вариант | Sharpe | Δ | Комментарий |
|---------|--------|---|-------------|
| filter_std003 | 1.83 | +0.05 | Marginal, порог 0.03 |
| baseline | 1.78 | — | |
| scaling | 1.73 | -0.05 | |
| filter_std002 | -0.09 | -1.87 | Catastrophic over-filter |

**Вывод**: Ensemble diversity слишком низкая (LGB≈XGB), p_std < 0.03 почти никогда не срабатывает. Фиксированный порог бесполезен.

### R64 — Combined Verification (✅ 2026-04-06)

Совместный запуск лучших из R60+R63+R61.

| Config | Sharpe | Комментарий |
|--------|--------|-------------|
| **grid_4L2S** | **1.84** | +0.07 vs baseline — WINNER |
| grid4L2S+filter | 1.84 | Filter не добавляет |
| baseline_6L3S | 1.77 | |
| filter_std003 | 1.77 | 0 эффект |
| grid4L2S+cg_temp | 1.20 | cg_temporal hurt в комбо |

### R65 — Gross vs Net Sharpe: 4L/2S vs 6L/3S (✅ 2026-04-07)

Ключевой вопрос: улучшение 4L/2S — от лучшего alpha или от меньших costs?

| Config | Gross Sharpe | Net Sharpe | Δ(cost) | AvgCost | AvgPos |
|--------|-------------|-----------|---------|---------|--------|
| **4L/2S** | **3.443** | **2.984** | 0.459 | 2.1bp | 6.0 |
| 6L/3S | 2.317 | 1.779 | 0.538 | 2.1bp | 9.0 |
| 3L/3S | 2.088 | 1.567 | 0.521 | 2.2bp | 6.0 |
| 8L/4S | 1.961 | 1.323 | 0.638 | 2.1bp | 12.0 |

**Gross Sharpe delta 4L/2S vs 6L/3S: +1.126** — improvement от **BETTER ALPHA**, не от costs. Costs почти одинаковые (2.1 vs 2.1 bps). Модель хорошо ранжирует top-4, позиции 5-6 добавляют шум/анти-альфу.

Квартальная стабильность 4L/2S (net):

| Quarter | Net Sharpe | Net Ret% |
|---------|-----------|---------|
| 2024Q4 | 5.26 | +34.5% |
| 2025Q1 | -5.19 | -11.1% |
| 2025Q2 | 6.30 | +15.5% |
| 2025Q3 | 1.53 | +6.9% |
| 2025Q4 | 4.87 | +10.3% |
| 2026Q1 | 2.09 | +9.1% |

Total net return 4L/2S: **+77.4%** (vs +33.9% для 6L3S). MaxDD: -14.1%.

### R67 — Reject Option / Score-Gap Threshold (✅ 2026-04-07)

Вместо фиксированного K=4L/2S — переменный K через порог на raw_prob: long если p > 0.5+t, short если p < 0.5-t. Max cap = 4L/2S.

| Threshold | Gross Sharpe | Net Sharpe | Ret% | MaxDD | WR% | AvgPos | Periods |
|-----------|-------------|-----------|------|-------|-----|--------|---------|
| baseline_4L2S | 3.443 | 2.984 | 77.4% | -14.1% | 60.4% | 4.2 | 450 |
| t=0.01 | 2.257 | 1.619 | 86.2% | -33.0% | 57.8% | 2.7 | 450 |
| t=0.02 | 1.410 | 1.073 | 38.9% | -38.7% | 54.4% | 2.1 | 443 |
| t=0.03 | 0.735 | 0.446 | -10.0% | -60.6% | 54.8% | 1.7 | 301 |
| ≥0.04 | negative | — | — | — | — | — | <136 |

**Вывод**: Reject option НЕ помогает. Baseline 4L/2S подтверждён = R65 (Gross 3.443, Net 2.984). Фильтрация по raw_prob threshold разрушает диверсификацию (AvgPos падает до 1.7-2.7) и ухудшает Sharpe. При t≥0.03 слишком мало позиций → катастрофический DD. Фиксированный 4L/2S лучше.

### R68 — Continuous Walk-Forward (✅ 2026-04-07)

Торговля в gap-периодах последней моделью (как в реале), без "0% месяцев".

| Config | Gross Sharpe | Net Sharpe | Ret% | MaxDD | Periods |
|--------|-------------|-----------|------|-------|---------|
| **4L/2S continuous** | **4.297** | **3.777** | **179.3%** | -13.9% | 688 |
| 4L/2S original | 3.443 | 2.984 | 77.4% | -14.1% | 450 |
| 6L/3S continuous | 3.102 | 2.509 | 81.9% | -15.0% | 688 |
| 6L/3S original | 2.317 | 1.779 | 33.9% | -15.0% | 450 |

**Вывод**: В continuous WF 4L/2S Net Sharpe = **3.777** (vs 2.984 на gapped windows). Improvement 4L/2S vs 6L/3S стабильно: +1.268 в continuous (3.777 vs 2.509), +1.205 в original (2.984 vs 1.779). Gap-периоды были профитабельны. Q1/2025 в continuous: Sharpe=+2.65 (vs -5.19 на gapped!) — последняя модель работала в gap.

Квартальная 4L/2S continuous:

| Quarter | Net Sharpe | Net Ret% |
|---------|-----------|---------|
| 2024Q4 | 5.26 | +34.5% |
| 2025Q1 | 2.65 | +14.7% |
| 2025Q2 | 5.48 | +21.2% |
| 2025Q3 | 0.30 | +2.6% |
| 2025Q4 | 7.13 | +33.5% |
| 2026Q1 | 2.09 | +9.1% |

### R69 — Percentile Uncertainty Gating (✅ 2026-04-07)

Adaptive filtering: на каждом timestamp убираем монеты с p_std выше определённого квантиля. Идея: relative threshold вместо абсолютного 0.03.

| Percentile | Gross Sharpe | Net Sharpe | Ret% | MaxDD | WR% | Filtered% |
|-----------|-------------|-----------|------|-------|-----|-----------|
| baseline_4L2S | 3.443 | 2.984 | 77.4% | -14.1% | 60.4% | 0% |
| pct=0.9 | 1.263 | 0.608 | 8.5% | -18.8% | 53.8% | 12% |
| pct=0.8 | 0.356 | -0.364 | -8.5% | -19.0% | 51.8% | 21% |
| pct=0.7 | -0.223 | -0.976 | -18.4% | -26.3% | 53.8% | 30% |
| pct=0.6 | -2.459 | -3.287 | -46.0% | -46.2% | 49.8% | 40% |
| pct=0.5 | -0.554 | -1.550 | -23.9% | -27.4% | 49.1% | 49% |

**Вывод**: Percentile gating КАТАСТРОФИЧЕСКИ ухудшает. Даже при pct=0.9 (фильтруем только 12% самых неуверенных) → Sharpe падает с 2.98 до 0.61. Причина: фильтрация убирает монеты ИМЕННО из top/bottom, которые и генерируют alpha. Тема uncertainty gating закрыта окончательно.

### R70 — LambdaRank Objective (✅ 2026-04-07)

LightGBM lambdarank + XGBRanker вместо binary classification. Группировка по timestamp, NDCG@4/@2 для оптимизации top-K ranking quality.

| Config | Gross Sh | Net Sh | NDCG@4 | NDCG@2 | Ret% | DD% |
|--------|----------|--------|--------|--------|------|-----|
| binary_4L2S | 3.443 | 2.984 | 0.5670 | 0.5563 | 77.4% | -14.1% |
| binary_6L3S | 2.317 | 1.779 | 0.5670 | 0.5563 | 33.9% | -15.0% |
| rank_4L2S | 1.546 | 0.796 | 0.5253 | 0.5167 | 10.6% | -14.5% |
| rank_6L3S | 0.874 | 0.042 | 0.5253 | 0.5167 | -0.7% | -17.6% |

**Вывод**: LambdaRank хуже binary по ВСЕМ метрикам. Binary baseline подтверждён = R65 (Gross 3.443, Net 2.984). NDCG@4 ниже (0.525 vs 0.567), Sharpe rank_4L2S=0.80 vs binary_4L2S=2.98. Binary classification уже хорошо оптимизирует ranking quality через probability calibration — переход на ranking loss не даёт улучшения. Тема закрыта.

**Итог R60-R70**: Из 25+ вариантов — единственное надёжное улучшение: **4L/2S** (по gross alpha, не по costs). В continuous WF: Net Sharpe **3.777**, ret **+179%**. Reject option, uncertainty gating, и LambdaRank разрушают performance. Система на локальном оптимуме — дальнейшее улучшение через feature/model engineering имеет <5% шанс.

### Аудит замечаний AI-консультанта (2026-04-07)

Консультант поднял 4 concerns по результатам R65-R70:

| # | Concern | Вердикт | Детали |
|---|---------|---------|--------|
| 1 | Look-ahead bias в continuous WF | **НЕТ** | Даты хардкожены (ORIGINAL_WINDOWS). train < val < 15d gap < test. Фильтрация: `df[df["timestamp"] < tr_end]` |
| 2 | Leverage inflates Sharpe/return | **НЕТ** | simulate() = 1x leverage. `gross_port_ret = 0.5*long_ret - 0.5*short_ret`. Return +179% = unleveraged |
| 3 | R69 "alpha от seed disagreement" | **НЕВЕРНО** | Код фильтрует OUT high p_std (controversial). Фильтрация любых монет ломает ranking → причина в сужении юниверса |
| 4 | Sharpe 3.78 подозрительно высоко | **ВАЛИДНО** | SE ≈ 0.38, CI [3.04, 4.52]. 500 12h-периодов, 3 WF окна. Перекрытие окон в continuous WF может inflate через non-independence |

### Deploy 4L/2S на VPS (✅ 2026-04-07)

- `run_trading.py`: DEFAULT_RISK n_long=6→4, n_short=3→2; CLS mode override: аналогично
- Shadow logging: каждый цикл записывает что выбрал бы 6L/3S → `trading_logs/shadow_6L3S.jsonl`
- Первый цикл 4L/2S: BNB(L), ETH(L), LTC(L), XRP(L), FIL(S), THETA(S)
- Shadow 6L/3S добавил бы: BTC(L), XTZ(L), ALGO(S)
- Пакеты VPS: pandas=2.3.3, lgb=4.6.0, xgb=3.2.0 ✅

---

## 8. История утечек данных (Data Leakage)

### 6 крупных инцидентов:

| # | Описание | Как нашли | Влияние | Фикс |
|---|----------|-----------|---------|------|
| 1 | **v1 time features**: hour_sin/dow_cos доминировали модель | IC analysis → time dominated | IC inflated, Sharpe=-1.0 в production | Removed time features from model input |
| 2 | **R18 IC scan на test data**: Feature selection по test IC | Audit methodology | Selection bias → artificially strong features | R19: IC scan ТОЛЬКО на TRAIN |
| 3 | **R21 rebalance overlap**: 6h rebal с 12h target → часы 6-12 дважды | Manual code review | Sharpe inflated by √2 | rebal_hours ≥ horizon |
| 4 | **Gen2 OOS leakage**: sim start Dec 9 but train_end=Feb 1 | Timeline review | Test data in training | Honest OOS window (Feb 9→Mar 7) |
| 5 | **R30 double cost**: Costs multiplied в simulate() И eval() | Debugging live losses | Costs understated 2x | R30b corrected |
| 6 | **Val-test overlap (pre-v6)**: Validation overlapped test | Walk-forward audit | Inflated early stopping | train_end moved 2mo before test_start + purge gap |

### Правила предотвращения утечек:
- IC scan ТОЛЬКО на train data
- 8 days purge gap между train и test (168h features + 12h target ≈ 7d, округлено)
- rebal_hours ≥ horizon (НИКОГДА 6h rebal с 12h target)
- Expanding window: train_end строго < test_start - purge
- Validation set: ends BEFORE test_start - purge

---

## 9. Execution Layer эксперименты (Overnight v10–v15)

### v14 (CatBoost variations)

| Experiment | Train Sharpe | Sim Return | Sim HAC |
|-----------|-------------|-----------|---------|
| cb_noderiv_hpo | 1.77 | +121.2% | 4.90 |
| cb_noderiv_hd05 | 1.70 | +111.6% | 4.61 |
| cb_noderiv_hd15 | **1.93** | +127.2% | 5.03 |
| cb_all_hpo | 1.62 | +128.4% | 5.29 |
| **cb_market_noderiv_hpo** | **1.83** | **+143.8%** | **5.33** |

**CHAMPION**: cb_market_noderiv_hpo. Per-coin news = noise, market-level news = signal.

### v15 (46 simulations, execution flags)

| Flag | Return | HAC | Verdict |
|------|--------|-----|---------|
| vol-target 0.30-0.60 | -54 to +6pp | хуже | **ВРЕДИТ** |
| hysteresis 3-10 | identical | identical | **НУЛЕВОЙ ЭФФЕКТ** (при 12h rebal) |
| smooth-signal 0.2-0.5 | -15 to -71pp | хуже | **ВРЕДИТ** |
| **vol-size** | **+3.7pp** | **+0.15** | **ЕДИНСТВЕННЫЙ ПОБЕДИТЕЛЬ** |
| regime-shorts 0.3-0.5 | -87 to -114pp | catastrophe | **КАТАСТРОФА** |
| meta-risk | +41pp | -0.04 | Больше return, хуже HAC |

**Ключевые сюрпризы v15:**
- Hysteresis = нулевой эффект при 12h ребалансировке
- Signal smoothing ВРЕДИТ — предсказания модели уже точные, сглаживание размывает
- Vol targeting ВРЕДИТ — модель уже учитывает vol через фичи
- Short alpha РЕАЛЬНЫЙ и сильный — урезание шортов разрушает performance
- **Только --vol-size (inverse-vol position sizing) помог**

### Benchmark Results (14d)

| Config | Return | Sharpe HAC |
|--------|--------|-----------|
| v6 no deriv | +3.0% | 3.67 |
| v7 no deriv | +2.4% | 3.14 |
| Ens no deriv | +2.3% | 2.65 |
| **Ens+meta no deriv** | **+3.7%** | **4.69** |

**Deriv gate ВРЕДИТ** (avg scale 0.84x): v6 HAC 3.67→1.74

---

## 10. AI Consultation Documents

11 файлов-консультаций, хронология:

| Файл | Фокус | Ключевой вывод |
|------|-------|----------------|
| AI_CONSULTATION_PROMPT.md | Full system review (188 features) | Architecture overview |
| v2 | 6-month instability | Sept-Dec 2025 = модель теряет деньги, профит только в последние 2 месяца |
| v3 | Meta-model stacking | LGB-MINIMAL best meta: DDStop Sh=2.35, но simple mean ≈ equal |
| v4 | What's next? Full exp history | Champion=CatBoost solo +131.5%, derivatives HURT |
| v5 | v14-v15 results | --vol-size единственный победитель, short alpha real |
| v6 | Walk-forward + deployment | Val-test overlap bug fixed. CatBoost no_news best |
| v7 | Feature engineering | **System is representation-limited, not model-limited** |
| v8 | Rolling vs Expanding window | WinC notably weaker. 6-24mo testing. |
| DATA | New data sources | On-chain, orderbook depth, stablecoin flows = gaps |
| IMPROVEMENT | How to improve | LambdaRank FAIL, Residual target useless, Meta-labeling failed |
| REGRESSION | Performance regression | Retrained models: +21.7% → LIQUIDATION. DDStop fix + overlap fix revealed true lower alpha. |

---

## 11. Установленные факты и выводы

### Что работает:
1. **LGB+XGB binary ensemble** — лучшая модель среди всех протестированных
2. **Cross-sectional rank normalization** — essential, убирает market beta
3. **FEAT_28 (28 фич)** — оптимальный feature set, добавление фич ухудшает
4. **6L/3S + ema_alpha=0.5 + hysteresis=3** — лучший portfolio config
5. **12h rebalance** — optimal (6h = leakage bug, 24h = теряет сигнал)
6. **Inverse-vol sizing (--vol-size)** — единственный execution flag который помог
7. **5 seeds ensemble** — стабильнее одного seed
8. **Expanding window** — лучше rolling для этих данных
9. **Short alpha РЕАЛЬНЫЙ** — нельзя урезать шорты
10. **R35a CS second-order features** — лучший новый feature bundle после R33, особенно для ремонта `W2`
11. **Liquidity filter (`liq70`)** — сильный execution lever: costs и turnover резко вниз, `W2/W3` вверх
12. **`D_30f` stability expert** — лучший текущий компромисс по `ALL`
13. **`cg_taker_imb`** — CoinGlass taker buy/sell imbalance, shift-1 day, CS-ranked. Fixes W1 dramatically.
14. **Hybrid tiered cost model (R48)** — TIER1~0.4bp / TIER2~2.5bp / TIER3~7bp. Costs 19.2%→9.8%, ALL 1.31→1.66

### Что НЕ работает:
1. **Больше фич = хуже** (feature curse, R28 catastrophe)
2. **Signal smoothing EMA** ВРЕДИТ — модель уже точная
3. **Vol targeting** ВРЕДИТ — модель учитывает vol через фичи
4. **Regime-shorts** — КАТАСТРОФА
5. **Macro фичи** (VIX, DXY, SPX) — noise для CS crypto
6. **Per-coin news** ВРЕДИТ LGB (помогает только CatBoost)
7. **Per-coin derivatives** ВРЕДЯТ всем моделям (market-level = marginal)
8. **Multi-horizon blending** — не помогает
9. **LambdaRank** — ТОТАЛЬНЫЙ ПРОВАЛ (IC 0.111→0.006)
10. **Residual target** — useless (r=0.965 с обычным target)
11. **Meta-labeling** — failed (690 trades too few)
12. **Старые данные (2017-2020)** — яд для crypto models
13. **Calendar features** — hurt
14. **Target engineering** (`R38`) — threshold/excess/decay не бьют baseline binary target
15. **Market-level macro / stablecoin features как direct CS add-on** — иногда чинят `W2`, но чаще вредят `W3/ALL`
16. **Hard regime switching** — помогает `W2`, но пока не побеждает лучший standalone expert по `ALL`
17. **cg_taker_imb_ma3 / delta / cs_demean (R48.1)** — все производные от taker_imb хуже оригинала
18. **cg_liq_imb residualization (R48.2)** — residual bin/roll/mkt все хуже baseline; redundancy фундаментальная
19. **Liq-weighted portfolio (R48.3 / R49c)** — ALL=0.53 при cost=27.3%, хуже uniform. R49c диагноз: NOT a bug — CS alpha генуинно в T2/T3 (IC=+0.028/+0.022), T1 IC≈0 (-0.004). Доп. причина: volume = coin units, BTC downweighted vs DOGE.

### Ключевые метрики:
- **Signal plateau**: Sh≈3.39 (gross, research sim) — model changes не помогают
- **Cost drag**: 40-47% — net Sharpe ~1.3 без smoothing, ~2.88 с ema+hysteresis
- **Model correlation**: r=0.93-0.97 между LGB/XGB/CB/CatBoost
- **IC**: Mean 0.029, ICIR 1.99, IC>0 в 13/13 месяцев
- **System is representation-limited, not model-limited**

### Гиперпараметры лучшей модели:

**LGB Classification**:
```python
params_lgb = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda_l2": 1.0,
    "n_estimators": 600,
    "early_stopping_rounds": 40,
}
```

**XGB Classification**:
```python
params_xgb = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 600,
    "early_stopping_rounds": 40,
}
```

**Ensemble**: `signal = 0.5 * rank(lgb_prob) + 0.5 * rank(xgb_prob)`

---

## 12. Текущее состояние и нерешённые проблемы

### Текущее состояние после R48:

| Config | W1 | W2 | W3 | ALL | Cost% | Статус |
|--------|-----|------|------|------|-------|--------|
| **31f + hybrid costs** | **+0.98** | **+3.53** | **+2.52** | **+1.66** | **9.8%** | **🏆 CHAMPION (R48)** |
| 31f + uniform 7bps | +0.60 | +2.86 | +2.03 | +1.31 | 19.2% | prev champion (R47) |
| 30f + hybrid costs (no CG) | +0.40 | +3.89 | +2.99 | +1.51 | 9.7% | reference |
| A_28f + dispersion + rankma | 0.01 | 3.22 | 2.50 | 1.13 | — | prev (R42) |
| D_30f (expert_stability) | -0.47 | 1.06 | 2.84 | 0.92 | — | stability baseline |

**Champion (R48)**: `A_28f + ret_dispersion_12h + cs_rank_ma_5 + cg_taker_imb` (31 features) + **hybrid tiered costs**
- `cg_taker_imb` = (buyVol - sellVol)/(buyVol + sellVol), CoinGlass taker data, shift-1 day, CS-ranked
- `ret_dispersion_12h` — market-level feature, MUST use `cs_rank_exclude`
- `cs_rank_ma_5` — smoothed 5-period rank, stabilizer
- **Hybrid cost model**: TIER1 (BTC/ETH/SOL/BNB/XRP ~0.4bp), TIER2 (~2.5bp), TIER3 (7bp)
- ALL Sharpe = 1.66, cost drag = 9.8% (down from 19.2%!), W1=0.98 / W2=3.53 / W3=2.52
- Bootstrap P(ΔSharpe>0) = 69.6% (marginal but accepted)

### Полная матрица R41–R46 экспериментов:

| Эксперимент | Гипотеза | ALL Sharpe | Вердикт |
|-------------|----------|------------|---------|
| **R41** Consolidation matrix | Комбинации R35/D30/liq70 дадут синергию | 0.74 best | ❌ Не стакаются |
| **R42** Ablation R35a features | Минимальный subset из 5 CS фич | **1.13** | ✅ **ЧЕМПИОН**: dispersion+rankma |
| **R43** Dynamic exposure (5L4S/4L4S) | Уменьшить net long при слабом бредте | 0.87 best | ❌ Отдают alpha |
| **R44** Dynamic universe / quality filter | Volume/IC фильтры | 0.82 best | ❌ Hard pruning слишком грубый |
| **R45** Calibrated soft gate / expert blend | soft blend R42 × stability × stablecoin | 0.92 (standalone) | ❌ Blend не побил standalone |
| **R46** Separate long/short models | P(ret>median) long + P(ret<p25) short | 0.54 | ❌ Alpha loss >> cost saving |
| **R47** CoinGlass features | Liq/taker/LS ratio from CoinGlass API | **1.31** | ✅ +cg_taker_imb (+0.18 Sharpe) |
| **R48** Hybrid costs + validation | Validate 31f, taker derivatives, residualized liq, hybrid costs | **1.66** | ✅ **🏆 CHAMPION**: hybrid tiered costs (cost 19.2%→9.8%) |

### R42 детали (ablation R35a):

R35a bundle = 5 фич: ret_dispersion_12h, cs_rank_ma_5, oi_chg_12h_cs, taker_cvd_12h_cs, cum_funding_24h_cs

| Config | W1 | W2 | W3 | ALL |
|--------|-----|------|------|------|
| **A_28f + dispersion + rankma** | 0.01 | **3.22** | **2.50** | **1.13** |
| A_28f + dispersion + funding_cs | 0.21 | 3.09 | 2.89 | 0.98 |
| A_28f + rankma + taker_cs + funding_cs | 0.10 | 1.34 | 2.92 | 0.95 |
| Full R35a (all 5) | -0.45 | 3.03 | 1.91 | 0.64 |
| A_28f baseline | 0.05 | 0.21 | 2.35 | 0.74 |

### R46 детали (asymmetric long/short):

| Config | W1 | W2 | W3 | ALL | Cost% | Turnover |
|--------|-----|------|------|------|-------|----------|
| **Unified baseline** | 0.01 | **3.22** | **2.50** | **1.13** | 19.2% | 4.5 |
| Asymmetric L/S | -1.17 | 1.80 | 2.08 | 0.54 | 12.2% | 2.4 |

### Данные — новые источники (D1–D7):

| Источник | Статус | Результат |
|----------|--------|-----------|
| **D1** CoinGlass (liq/taker/LS) | ✅ **TESTED (R47)** | `cg_taker_imb` → ALL=1.31 (+0.18). Liq features high IC но вредят WF |
| **D5** DefiLlama stablecoins | ✅ Downloaded | 360 assets, 3K+ history. Использован в R39 |
| **D6** Orderbook depth (Binance) | ⏳ Daemon копит | Инфра готова, нужно 2-3 недели |
| **D7** Social/Trends scan | ✅ Scanned | Reddit subs IC≈0.05, gtrends IC≈0.04. Покрытие 17/35 |
| On-chain (CoinMetrics) | ✅ Scanned | TxCnt_chg7d ICIR=0.145, только 9 монет |
| **Santiment** | 🆕 Free tier зарегались | Social volume/sentiment per coin, 1000+ assets |

### Провайдеры данных — решения по покупке (апрель 2026):

**КУПИЛИ / ПОКУПАЕМ:**
1. **CoinGlass Hobbyist** — $29/мес
   - 80+ endpoints, 30 req/min
   - Liquidations, L/S ratio, taker buy/sell, OI, funding, CVD, orderbook — PER COIN
   - History: 12h interval = 360 days, daily = all-time
   - Endpoint для ликвидаций: `GET /api/futures/liquidation/history`

2. **Santiment Free** — $0
   - 1,000 API calls/мес, 1 год истории (без последних 30 дней)
   - Social volume, sentiment, dev activity per coin. 1000+ криптоактивов

**ПРОВЕРИЛИ И ОТЛОЖИЛИ:**
- **CoinAPI Flat Files** — pay-as-you-go: $10/1000 GET requests, $1-3/GB transfer, $250/1000 OHLCV requests. Дорого для нашего масштаба, нет derivatives-specific data (no liquidations/L-S ratio).
- **Hyperliquid S3** — `s3://hyperliquid-archive/market_data/` L2 book snapshots + `s3://hl-mainnet-node-data/node_fills_by_block` trade fills. БЕСПЛАТНО (requester-pays S3). Только Hyperliquid DEX, не Binance/OKX, и обновляется ~1 раз в месяц. Полезно как дополнительный DEX orderbook source.
- **Kaggle dataset** (ollibolli/btc-historical-leverage-liquidations-order-data) — 723 MB, BTC only, 3 месяца от Hyperliquid. Слишком мало и только BTC.
- **CryptoDataDownload** — Free: OHLCV CSV. API $49.99/мес: 1100+ spot assets, 330+ futures, funding/liquidations/OI. Но для derivatives пытаются покрыть то же что CoinGlass за дороже.
- **CoinDesk Data** (data.coindesk.com) — enterprise-grade, 300+ бирж, tick-level. Нет публичных цен → "Book a call". Точно > $100/мес.
- **CryptoQuant** — $29/мес Advanced: 100 req/day, 7 дней API history = бесполезно. Professional $99/мес: 1 год. CoinGlass покрывает то же за $29.
- **LunarCrush** — $5/день ($150/мес) за social data. Santiment даёт то же бесплатно.
- **Glassnode** — on-chain, $49-999/мес. On-chain мы уже тестировали: ICIR < 0.15, максимум 9 монет.

### Что доказано работающим (финальный список):

1. **12h binary classification** — оптимальный target
2. **LGB+XGB ensemble** — оптимальная модель
3. **EMA + hysteresis** — снижает cost drag с 57% → 35%
4. **`ret_dispersion_12h`** — ключевой W2 repair lever (market-level, cs_rank_exclude!)
5. **`cs_rank_ma_5`** — стабилизатор, без него dispersion одна не работает
6. **Regime filter** — BTC trend_strength scale
7. **`cg_taker_imb`** — CoinGlass taker buy/sell imbalance, shift-1 day. Fixes W1 (0.01→0.60), ALL 1.13→1.31
8. **Hybrid tiered cost model (R48)** — TIER1~0.4bp / TIER2~2.5bp / TIER3~7bp. ALL 1.31→1.66, cost drag 19.2%→9.8%

### Что доказано БЕСПОЛЕЗНЫМ (финальный список):

- ❌ Target engineering (R38) — threshold, excess-vs-BTC, temporal decay
- ❌ Stablecoin/macro features напрямую в CS модель (R39, R28)
- ❌ Hard regime switching (R36)
- ❌ Feature sets >30 (noise dominates)
- ❌ Multi-horizon / meta-stacking (R27)
- ❌ Dynamic exposure rules 5L4S/4L4S (R43)
- ❌ Hard quality/liquidity pruning (R44)
- ❌ Separate long/short models (R46)
- ❌ Soft gate expert blending (R45)
- ❌ On-chain exchange flows (ICIR < 0.03)
- ❌ DeFi TVL features (ICIR < 0.10)
- ❌ Market-level CG liquidation features (mkt_cg_liq_log/total → killed ALL, R47)
- ❌ CG liq_imbalance напрямую в модель (IC=0.086 но redundant с ret_48h, WF destroys, R47)
- ❌ CG liq_intensity (r=0.89 с rel_volume_cs — артефакт формулы, R47)
- ❌ cg_taker_imb_ma3/delta/cs_demean (R48.1) — все производные хуже оригинального cg_taker_imb
- ❌ cg_liq_imb residualization (R48.2) — bin/roll/mkt все хуже; IC≈0.06 не спасает от WF failure
- ❌ Liq-weighted portfolio (R48.3) — ALL=0.53, cost=27.3%, значительно хуже hybrid

### Главная проблема:

**Cost gap sim→live.** Sim ALL=1.66 при hybrid costs (9.8% drag). С hybrid tiered execution (maker fills для top-5 пар) sim→live gap сокращается. Нужен:
- (a) Реализовать hybrid cost execution на live OKX — TIER1 maker orders (BTC/ETH/SOL/BNB/XRP)
- (b) Больше capital ($86 → sizing problem для 9 позиций)
- (c) D6 orderbook depth — daemon копит, нужно ещё 2-3 недели

### Нерешённые проблемы:

1. **Hybrid cost model на live**: R48.3 доказал 9.8% cost drag в sim. Нужна реализация maker fills для TIER1 на OKX live.
2. **Bootstrap маргинален** (69.6%): cg_taker_imb не proof-positive статистически, но monthly IC стабильна и W1 repair убедительный.
3. **Liquidation IC paradox**: IC=0.086 но WF уничтожается redundancy. Residualization не помогла (R48.2). Liquidations = not usable.
4. **Capital**: $86 → невозможно нормально sizing 9 позиций.
5. **VPS остановлен**: live trading приостановлена. D6 daemon не на VPS (SSH ключ rejected).

### R48 — ЗАВЕРШЁН ✅

**R48.0 — Validation**: ✅ Bootstrap 69.6% (маргинальный), monthly IC 9/13 positive, clip_98=1.31 ✅
**R48.1 — Taker derivatives**: ✅ Все хуже: ma3=-0.37, delta=-0.08, demean=-0.58. Оригинал оптимален.
**R48.2 — Residualized liq**: ✅ Все хуже: bin=-0.60, roll=-0.70, mkt=-1.18. IC≠WF signal.
**R48.3 — Hybrid costs**: ✅ ПРОРЫВ: ALL 1.31→1.66, cost 19.2%→9.8%. Liq-weighted = хуже.
**R48.4 — Best combo**: ✅ 31f hybrid = лучший. 30f reference 1.51. cg_taker_imb +0.15 на hybrid.

### AI Consultation после R48 (ключевые выводы)

- Hybrid cost model 9.8% = гипотеза. Если maker fill rate 40-60% вместо 92%, +0.35 Sharpe исчезнет → нужен live пилот
- DD -52% при 5× leverage = почти гарантированный stop. Нужен crash protocol (при 1× DD уже -13.6% = ок)
- liq-weighted (ALL=0.53) — диагностировано как GENUINE: CS alpha в T2/T3, не в T1
- bootstrap 69.6% — достаточно для research, для live нужен живой A/B тест

### R49c — Liq-Weighted Diagnostic (ЗАВЕРШЁН)

Диагноз: NOT a bug. Genuine signal distribution.

| Проверка | Результат |
|----------|----------|
| Volume alignment | ✅ 100% timestamps aligned |
| T1 weight vs T2/T3 | T1=0.878, T2=1.037 — **T1 DOWN-weighted** (coin vol ≠ USD vol) |
| corr(IC, log_vol) | -0.314 (p=0.07) — CS alpha в T2/T3, не в T1 |
| T1 IC avg | -0.004 (почти ноль), T2 IC=+0.028, T3 IC=+0.022 |
| T1 short returns | +0.0013 (wrong direction) — T1 shorts теряют P&L |

**Вывод**: Liq-weighting genuinely hurts. CS alpha живёт в mid/small caps. Добавлено в список `proven useless`. Есть также частичный баг: volume в coin units, не USD — BTC downweighted vs DOGE. Но исправление бага не поможет: T1 IC ≈ 0.

### ⚠️ КРИТИЧЕСКИЙ БАГ: pandas 3.0 → РЕШЕНО (2026-04-06)

**Проблема**: pandas 3.0 удалил `include_groups=True` из `groupby.apply`. Grouping column (`symbol`) автоматически дропается. Это ТИХО ломает:
- `cs_rank_cols()` — ranking без symbol column
- `add_r35_features()` → `per_symbol_features()` — теряет symbol
- Все `groupby("symbol")` transforms в feature pipeline

**Симптомы**: ICs меняются (W1: 0.0697→0.0365), Sharpe падает 1.66→1.02. Без ошибок, без warnings.

**Решение**: `.venv` откачен на pandas 2.3.3. НЕ обновлять pandas до 3.x.

**Правило**: Перед любым `pip install --upgrade` → проверить `python -c "import pandas; assert int(pandas.__version__.split('.')[0]) < 3"`

**Затронутые запуски** (запускались с pandas 3.0.1 → результаты невалидны, перезапущены):
- R49c, R50 — диагностика, выводы стабильны, перезапуск не критичен
- R55 — ✅ перезапущен 2026-04-06, результаты валидны (см. раздел R55)
- R56 — ✅ перезапущен 2026-04-06, результаты валидны (см. раздел R56)
- R56b — ✅ перезапущен 2026-04-06, результаты валидны (см. раздел R56b)

### R49 — Maker-First Execution (IMPLEMENTATION READY)

**run_trading.py** обновлён:
- Добавлены константы `_TIER1_SYMS`, `_TIER3_SYMS`, `MAKER_TTL_SECONDS=90`, `MAKER_MAX_RETRIES=3`
- Новая функция `_maker_first_limit()`: post-only limit → 3 попытки с нарастающей агрессией → market fallback
- Новая функция `_log_execution()`: пишет per-trade metrics в `trading_logs/execution_log.csv`
- TIER1 (BTC/ETH/SOL/BNB/XRP) → `_maker_first_limit()` во всех 4 точках (close/resize/open/retry)
- TIER2/TIER3 → без изменений (`_limit_with_fallback`)

**Следующий шаг**: Пилот 7-14 дней на OKX demo/микролотах → измерить реальный maker fill rate.

KPI пилота (из `trading_logs/execution_log.csv`):
- maker fill rate по notional
- avg effective_bps по TIER1
- pct unfilled → market fallback

### R50 — Risk Protocol (COMPLETED)

**Результаты** (champion_31f + hybrid costs, 1× leverage):

| Protocol | ALL Sharpe | Max DD | DD change |
|---------|-----------|--------|----------|
| Baseline (hybrid) | +1.02 | -13.6% | — |
| Crash (rvol>85%, ×30% gross) | best ALL | -9.4% | **+4.2pp** |
| DD breaker (threshold -12%) | не активируется при 1× | — | — |

**Ключевые выводы R50**:
1. При 1× leverage MaxDD = -13.6% уже в пределах нормы. Проблема DD -52% = проблема 5× leverage.
2. Crash protocol (BTC rvol > p85, gross ×30%, полный портфель) **улучшает** Sharpe и снижает DD.
3. DD breaker при -12% threshold = 0% активаций на тестовом наборе → сработает как safety net при live 5× leverage.
4. Имплементация: `_research_r50_risk_protocol.py`. Для деплоя на live — добавить crash detector в `run_trading.py`.

**Лучшие параметры R50**:
- `CRASH_RVOL_THRESHOLD = 0.85` (p85 30d rolling)
- `CRASH_GROSS_REDUCTION = 0.30` (reduce to 30%)
- `CRASH_TIER1_ONLY = False` (не ограничивать монеты, только размер)
- `DD_BREAKER_THRESHOLD = -0.12` (-12% от пика)

### Следующие шаги (R49.2 / D6):

**R49.2 — OKX Demo Pilot** (СЛЕДУЮЩИЙ ШАГ):
- [ ] Запустить `run_trading.py --mode paper` с R49.1 лимит-логикой
- [ ] 7-14 дней сбор `trading_logs/execution_log.csv`
- [ ] Анализ: real maker fill rate vs sim 92%

**R49.3 — Calibrate Sim** (after pilot):
- [ ] Заменить 92/8 maker/taker реальными числами из пилота
- [ ] Перегнать WF с calibrated costs → новый "честный" ALL Sharpe

**D6 — Orderbook Depth** (ждём данных):
- [x] Daemon запущен на VPS ✅ (cron `35 * * * *`)
- [ ] Накопить → IC scan (~20 апреля)

**Параллельно:**
- [ ] Добавить crash detector в `run_trading.py` с параметрами R50 (CRASH_RVOL_THRESHOLD=0.85)
- [ ] Santiment API — 1000 calls/мес free, scan social ICIR

### Walk-Forward Windows

```python
WINDOWS = [
    ('W1', train_end='2024-07-01', purge='2024-07-15', test='2024-07-15 to 2024-12-31'),
    ('W2', train_end='2025-01-01', purge='2025-01-15', test='2025-01-15 to 2025-06-30'),
    ('W3', train_end='2025-07-01', purge='2025-07-15', test='2025-07-15 to 2026-03-17'),
]
```

### Ключевые файлы проекта

```
run_trading.py              — Production trading code (3136 lines)
_ic_scanner.py              — IC analysis tool
_research_r22_models.py     — FEATURES_23, SEEDS, base model definitions
_research_r30b_fixed.py     — simulate_with_costs, train_ensemble
_research_r31_new_features.py  — R31 study (FEAT_26)
_research_r32_kaggle_features.py — R32 study (FEAT_28)
_research_r33_creative_features.py — R33 study (btc_corr)
_research_r47_qa.py         — R47 CoinGlass QA (timestamps, coverage, anomalies)
_research_r47_coinglass.py  — R47 CoinGlass features + IC + WF ablation
_research_r48_validation.py — R48.0 bootstrap + monthly stability + clip sensitivity
_research_r48_features.py   — R48.1+2 taker derivatives + residualized liq ablation
_research_r48_cost.py       — R48.3 hybrid tiered costs vs uniform vs liq-weighted
_research_r48_combo.py      — R48.4 best combo (31f hybrid = new champion)
model_registry.py           — CLI for tracking model generations
model_registry.json         — All 7 model generations
RESULTS.md                  — Complete research history (~4720 lines)
PROGRESS.md                 — Progress tracking (~607 lines)
CONTEXT_FULL.md             — Project summary
PROJECT_NOTES.md            — VPS, API keys, operational runbook
AI_CONSULTATION_*.md        — 11 consultation documents for AI review
```

### Feature Pipeline Chain

```
load_ohlcv() + load_derivatives()     [_ic_scanner.py]
  → build_features_minimal()           [_ic_scanner.py]
  → build_r19_features()               [_research_r22_models.py]
  → add_new_features()                 [_research_r22_models.py] (FNG, macro, TA)
  → add_extra_features_clean()         [_research_r30b_fixed.py]
  → add_kaggle_features()              [_research_r32_kaggle_features.py]
  → add_r33_features()                 [_research_r33_creative_features.py]
  → add_cg_features()                  [_research_r47_coinglass.py] (CoinGlass taker/liq/LS, shift-1d)
  → cs_rank_cols()                     [перед моделью]
```

---

## 13. ML Core — удалённые вычисления

Тяжёлые эксперименты (обучение, WF-симуляции) запускаем на ML Core вместо локального ноутбука.

### Инфраструктура

- **Платформа**: ML Core (mlc), проект `macos-build-infra`
- **Ноутбук**: `invest` (регион `ix-m5-sm12`, флейвор `8cpu-64ram`, образ `notebook-python-3.11`)
- **CLI**: `/usr/local/bin/mlc` (уже установлен, настроен: `project=macos-build-infra`, `detach=true`, `auto_relogin=true`)
- **Выхода в интернет НЕТ** — нельзя pip install из PyPI, git clone, fetch данных с бирж

### Файловая система на ноутбуке

| Путь | Размер | Назначение |
|------|--------|------------|
| `/workdir/` | 8 GiB | Рабочая директория, код проекта. **Мало места!** |
| `/data/` | Внешний S3 | Большие данные (датасеты, модели, архивы). Монтируется из S3 bucket `macos-build-infra-astabakov` (кластер `s3msk`) в `/data/datasets/` |
| `/data/.venv/` | В /data/ | Виртуальное окружение Python (~5 GiB). Симлинк `/workdir/invest/.venv → /data/.venv` |

**Правило**: большие файлы (data/, .venv, results/, модели) хранить в `/data/`, в `/workdir/` — только код и конфиги. Симлинки для удобства.

### Управление ноутбуком

```bash
# Статус
mlc notebook get invest

# Запуск / остановка
mlc notebook start invest
mlc notebook stop invest

# Список всех ноутбуков
mlc notebook ls
```

### Подключение к ноутбуку (терминал)

```bash
# SSH через VS Code (Remote-SSH → Connect to Host → invest.macos-build-infra.mlc)
# или через CLI:
mlc qwarium connect invest --open-in-vs-code
```

### Типичный workflow на ноутбуке

```bash
# 1. Запустить ноутбук (если остановлен)
mlc notebook start invest

# 2. Подключиться (VS Code Remote-SSH)
# Host: invest.macos-build-infra.mlc

# 3. На ноутбуке: подготовить данные
cd /workdir/invest
git pull                              # обновить код
tar xzf /data/datasets/data.tar.gz -C /data/datasets/   # распаковать данные в /data
ln -sf /data/datasets/data /workdir/invest/data          # симлинк data
ln -sf /data/.venv /workdir/invest/.venv                 # симлинк venv (~5GB, хранится в /data)
source .venv/bin/activate                                # активировать окружение

# 4. Запустить эксперимент
python _research_r60_portfolio_opt.py 2>&1 | tee results_r60.log

# 6. Забрать результаты
# Скопировать в /data/ для сохранности, т.к. /workdir/ может пересоздаться
cp results_r60.log /data/datasets/

# 7. Остановить ноутбук (экономим ресурсы)
mlc notebook stop invest
```

### Запуск команд на ноутбуке удалённо (без SSH)

```bash
# Узнать имя текущего job ноутбука
mlc notebook get invest    # → current job: invest-XXXXXX

# Выполнить команду
mlc job exec invest-XXXXXX -- bash -c "cd /workdir/invest && source .venv/bin/activate && python script.py"

# Интерактивный shell
mlc job exec invest-XXXXXX -- bash
```

### Запуск job'ов (альтернатива ноутбуку)

Для автономных скриптов можно отправлять job:

```bash
# Запуск job
mlc job submit notebook-python-3.11 python _research_r60_portfolio_opt.py \
  --flavor 8cpu-64ram \
  --region ix-m5-sm12 \
  --name r60-portfolio-opt \
  -i "type=s3,src=macos-build-infra-astabakov/data.tar.gz,dst=/data/datasets/" \
  -w /workdir

# Логи
mlc job logs r60-portfolio-opt

# Список job'ов
mlc job ls

# Отмена
mlc job cancel r60-portfolio-opt
```

### Важные ограничения

1. **Нет интернета** — все зависимости должны быть в Docker-образе или pip cache
2. **pandas ≤2.x ОБЯЗАТЕЛЬНО** — pandas 3.0 ломает groupby.apply (Sharpe 1.66→1.02). Текущий: 2.3.3
3. **/workdir = 8 GiB** — не хранить данные и venv здесь, только код
4. **Venv в /data/.venv** (~5 GiB) — слишком большой для /workdir, симлинк: `ln -sf /data/.venv /workdir/invest/.venv`
5. **Данные в S3** — bucket `macos-build-infra-astabakov`, кластер `s3msk`. Архив `data.tar.gz` распаковывать в `/data/datasets/`, симлинк: `ln -sf /data/datasets/data /workdir/invest/data`
6. **git pull** — обязательно перед каждым запуском
7. **Результаты копировать в /data/** — иначе потеряются при остановке
8. **Работаем только в /workdir/ и /data/** — в другие папки не лазить

---

## 14. DeepResearch v3: CG Alpha + Risk Overlay (R80–R86) — ЗАВЕРШЕНО

> Обновление: 8 апреля 2026. Полный цикл CG alpha + vol overlay research.

### Контекст

После R48 (champion 31f, Sharpe=1.66 на original WF) была введена **continuous WF** (R68):
- 3 окна без гэпов, wall-to-wall test coverage Oct 2024 – Mar 2026
- Config: 4L/2S, 12h rebalance, PROD_CFG (trend_cutoff=0.9, dyn_threshold=0.7, ema_alpha=0.5, hysteresis=3)
- **R68 baseline (честный, R113 fix): Net Sharpe = 3.057, MaxDD = -11.2%, Return = 183.9%, 1013 periods, Calmar = 16.47**
- *(Старый нечестный baseline: Sharpe 3.777, 688 periods — 32% flat-периодов пропускались через `continue`)*

```python
CONTINUOUS_WINDOWS = [
    {"name": "W1", "train_end": "2024-06-01", "test_start": "2024-10-15", "test_end": "2025-05-14"},
    {"name": "W2", "train_end": "2025-01-01", "test_start": "2025-05-15", "test_end": "2025-11-14"},
    {"name": "W3", "train_end": "2025-07-01", "test_start": "2025-11-15", "test_end": "2026-03-17"},
]
```

### R80 — CG Data Alignment (✅)

**Скрипт**: `_research_r80_cg_align.py`

Все raw CG фичи имеют catastrophic lookahead при direct-merge:

| Feature | Direct IC | Shift1 IC | Ratio |
|---------|-----------|-----------|-------|
| cg_liq_imb | -0.333 | +0.002 | 145× |
| cg_taker_imb | +0.276 | +0.005 | 61× |
| cg_oi_chg | +0.347 | -0.020 | 17× |
| cg_fr | +0.108 | -0.015 | 7× |
| cg_liq_log | -0.048 | +0.008 | 6× |
| cg_ls_ratio | -0.004 | -0.008 | 0.5× |

**Вывод**: CG timestamp = конец периода → direct merge = утечка будущего. Всегда использовать shift1.

### R81 — Vol Overlay Grid (✅)

**Скрипт**: `_research_r81_vol_overlay.py`

Grid: L∈{20,40} × vol_tgt∈{median,p25} × s_min∈{0.25,0.35} × s_max∈{1.25,1.50} = 16 configs.
DD overlay: dd>10% → scale×0.7, dd>15% → scale×0.5.

Best config: `L20_p25_smin035_smax15`
- Sharpe: 3.730 (ΔSh = -0.047)
- MaxDD: **-10.41%** (было -13.95%, ↓25.4%)
- Calmar: 35.8 (было 27.1, ↑32.2%)
- Return: 130.7% (было 179.3%)

8/16 configs прошли acceptance. Все L20_p25 → DD=-10.41%, все L40_p25 → DD=-11.55%.

### R82 — CG Feature Factory (✅)

13 z-score/momentum фичей (shift1, rolling 120 periods):
- TAKER: cg_taker_imb_z120, cg_taker_flow_z120
- LIQ: cg_liq_imb_z120, cg_liq_log_z120, cg_liq_spike
- OI: cg_oi_z120, cg_oi_notional_chg, cg_oi_surge
- FUNDING: cg_fr_z120, cg_fr_accel, cg_fr_accel_z120
- LS: cg_ls_z120, cg_ls_chg_z120

Все 13/13 прошли coverage gate ≥0.95 (98.9–100%).

### R83 — IC Scan (✅)

| Feature | Pooled IC | Stability | MaxCorr | Gate |
|---------|-----------|-----------|---------|------|
| cg_liq_log_z120 | 0.0197 | 0.67 | 0.518 | ✗ (IC<0.03) |
| cg_fr_z120 | -0.0085 | 1.00 | 0.510 | ✗ (IC<0.03) |
| cg_taker_imb_z120 | 0.0034 | 0.67 | 0.921 | ✗ (redundant) |
| остальные 10 | |IC|<0.021 | 0.0–0.33 | — | ✗ |

Gate: |IC|≥0.03, stability≥2/3, max_corr<0.7, coverage≥0.95. **0/13 прошли.**

### R84 — Baseline Re-run (✅)

⚠️ **Обнаружен баг**: оркестратор (`_research_orchestrate.py`) загружал данные своим путём, не через R68 `load_data()`. Train sizes отличались (на 800-1200 строк). Результат: Sharpe=-0.077 — **невалиден**.

R86 fix: использовал каноничный `load_data()` из R68. Baseline: **Sharpe=3.777**, MaxDD=-13.95%, n_periods=688 — **совпадает** с оригиналом.

### R85 — Bootstrap (✅, пересчитан в R86)

Block bootstrap (block=10, N=1000, 688 periods):

| Comparison | P(exp>base) | Median ΔSh | Verdict |
|------------|-------------|------------|---------|
| R81 best vs R68 | **0.372** | **-0.129** | **✗ REJECT** |

Vol overlay не проходит bootstrap. Sharpe R81 (3.730) < R68 (3.777), 63% ресэмплов baseline лучше.

### Выводы DeepResearch v3

1. **CG Alpha — ЗАКРЫТО** (как линейная альфа). Z-score/momentum IC 0.002–0.020 после shift1. Ни одна не прошла |IC|≥0.03.
2. **Vol Overlay — НЕ ПОДТВЕРЖДЁН.** MaxDD↓25%, но Sharpe↓0.047. Bootstrap P=0.372 < 0.8.
3. **R68 baseline стабилен.** Повторный запуск = точно 3.777.
4. **Баг data loading:** оркестратор давал Sharpe=-0.077 из-за другого пути загрузки данных.

### Добавлено в "proven useless":
- ❌ Raw CG values без shift1 (lookahead, R80)
- ❌ CG z-score/momentum линейная альфа (IC<0.03, R83)
- ❌ Vol overlay как Sharpe improvement (bootstrap reject, R85/R86)
- ❌ Temporal features, meta-stacking, LambdaRank, reject option (R60–R70)

---

## 15. DeepResearch: Parallel strategies + 4h ML (R90–R100) — результаты

> Обновление: 8 апреля 2026.

### Контекст
- R68 baseline (champion, prod): 4L/2S continuous WF, gen8 hybrid LGB+XGB (5 seeds), 31 фича
  **Sharpe=3.777, MaxDD=-13.95%, Return=179.3%, n_periods=688** (Oct 2024 – Mar 2026, 12h, 33 монеты)
- DeepResearch v3 показал:
  - CoinGlass z-score/momentum линейные фичи не дают IC после устранения lookahead
  - Vol overlay снижает DD, но не проходит bootstrap по Sharpe (под Sharpe-целью)

### R90 — Data Audit ✅
- OHLCV: 1h timeframe, **1,826,014 строк**, **35 символов**, диапазон 2017–2026
- В frame уже есть предрасчитанные: `fwd_ret_1h`, `fwd_ret_4h`, `fwd_ret_12h`, `fwd_ret_24h`, `fwd_ret_48h`
- CoinGlass parquet: `taker`, `oi`, `funding`, `liq`, `ls_ratio` (история с 2022)

### R91 — Funding Carry (rule-based) ✗ FAIL
Carry_score = -funding_rate (как ценовой сигнал).
Все конфиги убыточны:
- 2L/2S 12h: Sharpe -0.295, MaxDD -62.0%, Return -53.8%
- 3L/3S 12h: Sharpe -1.017, MaxDD -73.7%, Return -72.1%
Причина: funding payments не входят в `fwd_ret_12h` (ценовой ретёрн), поэтому "carry" здесь превращается в слабый предиктор цены.
Corr(R91, R68) = -0.341 (диверсификация есть, но стратегия убыточна).

### R92 — Liquidation Events Mean-Reversion ✅ (слабая, но диверсифицирует)
Лучшее:
- thr2.0_H3_K2L2S_cd0: Sharpe 0.535, MaxDD -53%, Hit 0.528
- Bootstrap P(Sharpe>0) = 0.812 ✅
- Corr(R92, R68) = -0.031 (почти ноль)

### R93 — 4h-target ML engine ✅ (принципиально другой горизонт)
Модель обучается на `fwd_ret_4h` (1h данные, 4h таргет), но торгует с ребалансом 12h/4h.
Результаты:
- **Best: 4L2S_12h**: Sharpe **3.190**, MaxDD **-10.1%**, Return **34.8%**, Win 57.3%, N=689
- 4L2S_4h: Sharpe 2.508, MaxDD -17.2%, Return 105.8% (turnover/cost выше)
Ключевое:
- Corr(R93, R68) = 0.456 (умеренная корреляция ⇒ потенциальный диверсификатор)
- Trend filter критичен: без него Sharpe у R93 валится (в одном тесте: -1.074)

### R94 — Strategy Mix (DD↓, Sharpe не бьёт R68)
Best: 80/10/10 (R68/R91/R92) — **MaxDD с 13.9% до 9.8% (↓30%)**, Sharpe 3.461 (падает с 3.78).
Bootstrap P(mix Sharpe > R68) = 0.239 → REJECT.

### R95 — Bootstrap
| Тест | P | ΔSharpe | Verdict |
|------|---|---------|---------|
| R91 vs cash | 0.370 | -0.277 | ✗ |
| R92 vs cash | 0.812 | +0.558 | ✅ |
| R94 mix vs R68 | 0.239 | -0.341 | ✗ REJECT |

### R97 — Attribution (R68 vs R93) ✅
| Метрика | R68 | R93 | Δ |
|---|---:|---:|---:|
| Gross alpha | 17.7 bps/period | 7.0 bps | -10.7 |
| Net alpha | 15.6 bps | 4.6 bps | -11.0 |
| Volatility | 111.4 bps | 66.1 bps | -45.3 |
| Turnover | 3.14 | 3.89 | +0.75 |
| Cost drag | 44.2% | 24.7% | -19.5 |
| Compounding bonus | 72.5% | 3.4% | -69.1 |

Вывод: R93 реально **low-vol + low-alpha** относительно R68 ⇒ для честного микса нужен vol-scaling, иначе вклад R93 "тонет".
R93 улучшается со временем: W1 Sharpe=1.39 → W2=1.96 → W3=2.51.

### R100 — Rank Ensemble ⚠️ (НЕканоничный baseline)
| α (вес R68) | Sharpe | MaxDD | Return |
|---|---:|---:|---:|
| 0.00 (pure R93) | 3.174 | -17.9% | 140.4% |
| 0.25 | 3.285 | -12.8% | 153.2% |
| **0.50** | **3.604** | **-12.7%** | **178.0%** |
| 0.75 | 3.491 | -13.3% | 156.6% |
| 1.00 (pure R68) | 3.311 | -15.9% | 137.4% |

Bootstrap P(best > R68) = 0.681 (не значимо).
**Баг**: "pure R68" в R100 даёт 3.311, а каноничный R68 = 3.777. Причина: R100 пере-ранжировал уже rank-blended score (double-ranking), что нарушило EMA/hysteresis поведение. Эксперимент R100 невалиден — нужен R102 fix.

### R99 — Return Mix с vol-scaling ❌ (мелкий эффект, без значимости)
Best w93=0.25: Calmar 28.05 (+3.6% vs R68), DD -13.1%, Sharpe 3.687.
Bootstrap: P(Calmar better)=0.684, P(Sharpe better)=0.421 ⇒ не значимо.

### Итог R90–R100
- R68 остаётся чемпионом.
- R93 — сильная параллельная 4h-target модель (Sharpe~3.19, DD лучше), но пока **не доказано**, что она улучшает R68 при миксе/ансамбле.
- R100 невалиден (baseline drift) → нужен R102 fix + R103/R104 корректные ensemble.

### Добавлено в "proven useless":
- ❌ Funding carry как ценовая стратегия (FR ≠ price predictor, R91)
- ❌ Liq events standalone (Sharpe=0.54, не масштабируется, R92)
- ❌ Return mix R68+R93 без значимости (R99, P(Sharpe)=0.42)
- ❌ Rank ensemble с double-ranking (R100, деградирует baseline)

### R102 — Baseline Equivalence ✅ PASS
Cached R68 predictions → R68 simulate(4,2) → **Sharpe=3.7771, MaxDD=-13.95%, N=688** — бит-в-бит совпадение с каноничным R68.
Left-join R68+R93 с pred=R68 → equity correlation=**1.000000**, max abs diff=0.

### R103 — Logit Ensemble ❌ REJECTED
| α | Sharpe | MaxDD | Return |
|---|---:|---:|---:|
| 0.00 (R93 logit) | 2.941 | -18.4% | 138.8% |
| 0.25 | 2.446 | -17.2% | 107.4% |
| 0.50 | 2.116 | -18.7% | 83.1% |
| 0.75 | 2.639 | -14.6% | 119.7% |
| 1.00 (R68 logit) | 2.797 | -15.1% | 134.6% |

α=1.0 → Sharpe 2.797 (каноничный R68=3.777). Logit-трансформация `raw_prob` вместо rank-blend `pred` разрушает EMA-smoothing behaviour. Все α деградируют.
Bootstrap P(Sharpe better)=0.151. **REJECTED.**

### R104 — Tiebreak Ensemble ❌ REJECTED
| M (long pool) | N (short pool) | Sharpe | MaxDD |
|---|---|---:|---:|
| 6 | 3 | 2.653 | -18.0% |
| 6 | 4 | **3.023** | -24.0% |
| 6 | 5 | 2.776 | -15.5% |
| 8 | 3 | 2.449 | -17.2% |
| 10 | 4 | 3.010 | -20.4% |

Best M=6 N=4: Sharpe=3.023 — все конфиги хуже R68 (3.777). R93 tiebreak ухудшает R68-отобранных кандидатов.
Bootstrap P(Sharpe better)=0.221. **REJECTED.**

### Итог R102–R104
R93 (4h-target) **не улучшает** R68 ни в одном из подходов:
- Logit ensemble: logit(raw_prob) ≠ rank-blend, ломает EMA
- Tiebreak: R93 ранжирует монеты иначе чем R68 (разные targets), подмешивание ухудшает
- R93 оставить на мониторинг (shadow). R68 = champion без изменений.

### Добавлено в "proven useless":
- ❌ Logit ensemble R68+R93 (R103, baseline degradation)
- ❌ Tiebreak ensemble R68+R93 (R104, R93 ranking incompatible)

---

## R105–R106 — Funding Rate Arbitrage (market-neutral carry)

**Новое направление**: short perp + long spot = hedge price risk → собираем funding payments.
Ортогонально R68 (directional ML). Принципиально отличается от R91 (R91 использовал FR как ценовой предиктор — провалился).

### R105 — Historical Analysis ✅ PASS
Data: Binance 8h FR, 294K rows, 50 символов, 2020-01 → 2026-03.

**Per-coin FR (top-5 carry):**
| Symbol | Mean FR | Ann Carry | % positive | % >0.02% |
|---|---:|---:|---:|---:|
| FTM/USDT | 0.0169% | 18.5% | 88.0% | 14.6% |
| XRP/USDT | 0.0144% | 15.8% | 81.8% | 14.9% |
| LTC/USDT | 0.0144% | 15.8% | 83.1% | 14.3% |
| AAVE/USDT | 0.0141% | 15.4% | 86.2% | 10.8% |
| MKR/USDT | 0.0138% | 15.1% | 88.0% | 12.0% |

**Opportunity frequency (threshold → opps/month):**
| Threshold | Opps/mo | Coin-opps/mo | % periods | Avg coins |
|---:|---:|---:|---:|---:|
| 0.01% | 45 | 643 | 26.8% | 14.3 |
| 0.03% | 37 | 462 | 19.6% | 12.4 |
| 0.05% | 33 | 357 | 14.1% | 10.9 |
| 0.10% | 22 | 206 | 6.4% | 9.4 |

**Persistence**: AC(lag1=8h)=0.711, AC(lag3=24h)=0.574 — FR highly persistent, good for hold.

**Regime stability (по годам):**
| Year | Mean FR | Ann Carry | % >0.02% |
|---:|---:|---:|---:|
| 2020 | 0.0108% | 11.8% | 23.4% |
| 2021 | 0.0335% | **36.7%** | 35.4% |
| 2022 | -0.0027% | **-2.9%** | 0.0% |
| 2023 | 0.0057% | 6.2% | 5.8% |
| 2024 | 0.0112% | 12.3% | 13.1% |
| 2025 | 0.0017% | 1.9% | 0.1% |
| 2026 | -0.0148% | -16.3% | 0.6% |

⚠️ **2022 bear market: negative carry.** 2025-2026 тоже слабые. Стратегия зависит от бычьего/нейтрального рынка.

**Theoretical carry (entry > threshold, hold N periods, pay 0.16% RT):**
| Threshold | Hold | Net carry% | Win% | Entries/mo |
|---:|---:|---:|---:|---:|
| 0.05% | 24h | +0.145% | 77.6% | 138 |
| 0.05% | 48h | +0.404% | 93.7% | 138 |
| 0.08% | 24h | +0.251% | 93.7% | 75 |
| 0.08% | 96h | +1.171% | 97.9% | 75 |

Cross-validation Binance↔OKX: corr=0.679, mean_abs_diff=0.022%.
**VERDICT: PASS** — 37 opps/month at 0.03% (need ≥5).

### R106 — Backtest ✅ PASS
Grid: 5 entry × 3 exit × 4 hold × 3 positions = 156 valid configs.
Capital=$100, round-trip=0.16%.

**Top-10 by Sharpe:**
| Entry | Exit | Hold | Pos | Sharpe | Ret% | DD% | Win% | Trades |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.080% | 0.005% | 24 | 3 | **6.638** | +14.7% | -0.8% | 66.7% | 147 |
| 0.050% | 0.010% | 24 | 3 | 6.586 | +15.2% | -1.1% | 65.2% | 204 |
| 0.050% | 0.005% | 24 | 3 | 6.537 | +15.2% | -1.0% | 66.2% | 201 |
| 0.080% | 0.010% | 24 | 3 | 6.378 | +14.3% | -0.8% | 65.3% | 150 |
| 0.050% | 0.010% | 24 | 2 | 5.885 | +15.3% | -1.5% | 61.7% | 149 |

Best: entry=0.080%, exit=0.005%, hold=24 (192h), max_positions=3.
- Sharpe=6.638, Return=+14.7% (6y), MaxDD=-0.8%, 147 trades
- Funding earned=$22.57, Costs=$7.84
- **Hedge P&L = $0.00** (теоретический хедж идеальный — basis risk НЕ моделирован!)

⚠️ **Caveats**:
1. Hedge_pnl=0 — в реальности spot≠perp pricing, нужен R107 для basis risk
2. Regime-dependent: 2022+2025-2026 убыточные (bear/flat market = negative FR)
3. Sharpe 6.6 завышен из-за (a) идеального хеджа, (b) no slippage на entry/exit
4. На $100 капитала чистый return $14.7 за 6 лет ≈ $2.4/год — micro-scale

**VERDICT: PASS** — Sharpe=6.638 ≥ 1.0. Proceed to R107 (basis risk).

### R107 — Hedge Quality & Basis Risk ✅ PASS
Data: Binance premium_index (perp-spot basis), 496K rows, 48 sym, 2021-12 → 2026-03.

**Basis distribution (top coins, sorted by ΔBasis σ):**
| Symbol | Basis µ% | ΔBasis σ% | |ΔBasis| µ% |
|---|---:|---:|---:|
| BTC/USDT | -0.029 | 0.023 | 0.014 |
| ETH/USDT | -0.028 | 0.030 | 0.018 |
| ADA/USDT | -0.026 | 0.037 | 0.026 |
| XRP/USDT | -0.027 | 0.037 | 0.022 |

**Worst-case basis moves (all coins aggregate):**
| Hold | Mean |ΔBasis|% | σ% | P1/P99% | Worst% |
|---:|---:|---:|---:|---:|
| 8h | 0.027 | 0.054 | -0.10/+0.10 | -11.4 |
| 24h | 0.029 | 0.066 | -0.11/+0.11 | -11.2 |
| 192h | 0.033 | 0.154 | -0.11/+0.11 | -21.3 |

**Basis risk vs funding income (ключевая таблица):**
| Filter | Mean FR% | Basis σ% | Net PnL% | Ratio | Ann Sharpe | Win% |
|---|---:|---:|---:|---:|---:|---:|
| All FR>0 | 0.011 | 0.043 | +0.013 | 4.0x | 9.16 | 65% |
| FR>0.03% | 0.049 | 0.063 | +0.054 | **1.3x** | 26.53 | 85% |
| FR>0.05% | 0.066 | 0.077 | +0.079 | **1.2x** | 31.61 | 91% |
| FR>0.08% | 0.098 | 0.127 | +0.150 | **1.3x** | 37.28 | 99% |

→ При FR>0.05-0.08% basis risk ≈ 1.2-1.3x от FR, но net PnL всё равно положительный. Win rate 91-99%.

**Revised R106 backtest (с basis risk):**
| Config | Sharpe | Return% | MaxDD% | Vol% | Trades | Win% |
|---|---:|---:|---:|---:|---:|---:|
| Best (entry=0.08%, hold=24) | **2.421** | +1.47% | -0.13% | 0.24% | 12 | 100% |
| Alt (entry=0.05%, hold=24) | **2.625** | +2.14% | -0.19% | 0.32% | — | — |
| Conservative (hold=6) | 0.657 | +0.49% | -0.15% | 0.30% | — | — |

⚠️ Sharpe упал с 6.64 → **2.42** (basis risk добавил реальную vol). Всего 12 trades за 4+ лет.
Yearly: 2022=0%, 2023=+0.09%, 2024=+1.36%, 2025=0%, 2026=0%. Почти весь return — 2024.

**VERDICT: PASS** — revised Sharpe=2.421 ≥ 1.0, basis_ratio=1.3x < 2x.

### R108 — Paper Trading Monitor ❌ FAIL
Data: last 30 days (Feb-Mar 2026), Binance FR + premium.

**Result: ZERO opportunities.**
- R106_best (entry>0.08%): **0 entries** за 30 дней
- R106_alt (entry>0.05%): **0 entries** за 30 дней
- Paper return: **0.000%** (ничего не произошло)
- Deviation from backtest: **-100%** (ожидалось >0, получено 0)

Причина: 2026 FR ≈ -0.015% (negative). Рынок bear/flat → лонги не платят funding → нет opportunities.

**VERDICT: FAIL** — paper 100% deviation (>50% kill). Стратегия неактивна в текущем режиме.

### Итог R105–R108: Funding Rate Arbitrage
| Step | Verdict | Key Metric |
|---|---|---|
| R105 Analysis | ✅ PASS | 37 opps/month historically |
| R106 Backtest | ✅ PASS | Sharpe=6.64 (без basis) |
| R107 Basis Risk | ✅ PASS | Revised Sharpe=2.42 (с basis) |
| R108 Paper | ❌ FAIL | 0 opportunities in current market |

**Заключение**: Funding arb работает в bull/neutral market (2020-2021, 2024) но **мертва в bear/flat (2022, 2025-2026)**. При $100 капитала чистый доход ~$1-2/год в хорошие годы. Стратегия:
- ✅ Market-neutral, ортогональна R68
- ✅ Sharpe 2.4 с учётом basis risk
- ❌ Regime-dependent (negative FR в bear market)
- ❌ Low absolute return ($1-2/year per $100)
- ❌ Текущий рынок: 0 opportunities

**Решение**: НЕ деплоить. Оставить скрипты на мониторинг. Если FR вернётся к positive (bull market) → пересмотреть. R68 остаётся единственной production стратегией.

### Post-Audit Validation (R105–R108)

AI-рецензент выявил 4 несоответствия в отчёте. Проведена независимая валидация (`_validate_r105_r108.py`).

#### Issue 1: "37 opps/month" (R105) vs "12 trades за 4+ года" (R107)

**Root cause: несовпадение данных.**

| Dataset | Symbols | Unique periods | Date range |
|---|---|---|---|
| R105 (full Binance FR) | 50 | 10,251 | 2020-01 → 2026-03 |
| R107 (FR ∩ Premium Index) | 48 | 2,738 | 2021-12 → 2026-03 |

- 2 символа (FTM, MKR) есть в FR, но нет в Premium Index
- При >0.08% threshold: **8,532** coin-opps в full data vs **67** в merged data
- Из 67 coin-level сигналов: **12 entered**, 34 rejected (capacity full), 21 rejected (overlap)
- Entry rate: 12/67 = 17.9%

**Вывод**: 37 opps/month и 12 trades — оба корректны для своих определений. Проблема — R105 анализировал полные данные (2020+), а R107 backtest работал на пересечении (с 2021-12). Capacity blocking (max_pos=3, hold=192h) отклонил 82% сигналов.

#### Issue 2: "Worst basis -21%" vs "MaxDD -0.13%"

**Root cause: позиционный размер + selection bias при n=12.**

- Position size per leg = $16.67 ($100/3/2)
- Worst basis -21% × $16.67 = max $3.55 потеря = 3.6% от капитала
- Но: basis change **во время реальных 12 позиций** был значительно меньше:

| Metric | Value |
|---|---|
| N observations during positions | 285 |
| Mean basis change | -0.0071% |
| Std | 0.0688% |
| Min / Max | -0.8136% / +0.1569% |
| P1 / P99 | -0.1790% / +0.1097% |
| Max cumulative adverse (single trade) | -0.8136% |
| In dollar terms | $0.14 |

**MaxDD -0.13% реален**, но при n=12 trades **статистически бессмысленен** — не репрезентативен для tail risk.

#### Check 1: FR > 0.05% за последние 90 дней

| Threshold | Periods (of 1,246) | Coin-opps | Top coin |
|---|---|---|---|
| > 0.03% | 51 (4.1%) | 51 | FLOW/USDT (0.35%) |
| > 0.05% | 30 (2.4%) | 30 | FLOW/USDT (0.35%) |
| > 0.08% | 13 (1.0%) | 13 | FLOW/USDT (0.35%) |

Opportunities есть, но **только на FLOW** и с низкой частотой. При max_pos=3 и hold=24 это ~1-2 trades/month.

#### Check 3: Liquidation distance

- При 1x leverage, margin $16.67 per position
- Liquidation = 80%+ adverse price move → **negligible risk**
- Для hedged позиции (spot+perp) risk = basis change, не price change
- Max basis change при наших trades: -0.81% → $0.14 потеря
- BTC worst 192h drawdown: -54.9%, но это не релевантно для hedged position

#### Corrected Summary Table

| Step | Verdict | Key Metric | Correction |
|---|---|---|---|
| R105 Analysis | ✅ PASS | 37 opps/month historically | Корректно для full data |
| R106 Backtest | ✅ PASS | Sharpe=6.64 (без basis) | Корректно, но без friction |
| R107 Basis Risk | ⚠️ UNPRACTICAL | Revised Sharpe=2.42, **n=12 trades** | Sharpe на 12 trades бессмысленен |
| R108 Paper | ❌ FAIL | 0 opportunities in current market | Подтверждено: только FLOW, ~1 opp/mo |

**Усиленное заключение**: НЕ ДЕПЛОИТЬ. Причина не только "текущий рынок плохой" (R108), но и **недостаточная статистическая база** — 12 trades за 4 года не позволяют сделать выводы о Sharpe/DD. Стратегия теоретически корректна, но непрактична на текущих данных.

## R109 — Macro Features IC Scan ❌ FAIL

**Гипотеза**: Macro-экономические фичи (DXY, VIX, SPX, US10Y, Gold) могут добавить предсказательную силу к R68.

**Данные**: yfinance (бесплатно). Daily, shift(1) для lookahead prevention, forward-fill на hourly.

**12 features tested:**

| Feature | Pooled IC | Mean TS IC | Stability | Max Corr (vs R68) | Gate |
|---|---|---|---|---|---|
| btc_dxy_corr_20d | -0.0177 | -0.0021 | 0.67 | 0.058 | ❌ IC |
| us10y_chg_5d | -0.0157 | -0.0198 | 0.33 | 0.070 | ❌ IC |
| us10y_level | -0.0120 | -0.3503 | 0.33 | 0.450 | ❌ IC |
| vix_level | +0.0114 | -0.3503 | 0.33 | 0.201 | ❌ IC |
| spx_ret_20d | +0.0076 | +0.3503 | 0.33 | 0.309 | ❌ IC |
| btc_spx_corr_20d | +0.0075 | +0.0046 | 0.67 | 0.115 | ❌ IC |
| gold_ret_5d | -0.0066 | +0.0198 | 0.33 | 0.121 | ❌ IC |
| vix_chg_5d | -0.0059 | -0.3503 | 1.00 | 0.252 | ❌ IC |
| dxy_ret_5d | -0.0049 | -0.3503 | 0.33 | 0.127 | ❌ IC |
| dxy_ret_20d | -0.0033 | -0.3503 | 0.33 | 0.075 | ❌ IC |
| spx_ret_5d | -0.0026 | +0.3503 | 0.67 | 0.269 | ❌ IC |
| vix_z60 | +0.0005 | -0.3503 | 0.33 | 0.219 | ❌ IC |

**Gate**: |IC| ≥ 0.03, stability ≥ 2/3, coverage ≥ 95%, redundancy < 0.70.

**Result**: 0/12 features pass gate. Best |IC| = 0.0177 (btc_dxy_corr_20d) — ниже порога 0.03.

**VERDICT: ❌ FAIL** — macro features не имеют предсказательной силы для crypto returns на 12h горизонте. WF test не проводился. R68 остаётся champion без изменений.

**Time**: 2.9 минуты. **Cost**: $0.

## R110 — Partial Neutralization Sweep ❌ FAIL

**Гипотеза**: Убрать нежелательные экспозиции R68 к risk/regime факторам через cross-sectional ridge regression (Numerai-style) может улучшить Sharpe или Calmar.

**Grid**: 3 exposure sets × 4 λ × 5 α = 60 комбинаций.

| Exposure Set | Exposures |
|---|---|
| SET1 (minimal) | btc_beta_168h, ret_48h |
| SET2 (risk+liq) | + rel_volume_cs, rvol_24h |
| SET3 (derivs) | + cum_funding_24h, oi_velocity |

**Baseline**: R68 4L/2S Sharpe=3.777, DD=-13.9%, Calmar=12.86

**Лучшие результаты** (из 48 non-baseline):

| Config | Sharpe | DD | Calmar | P(Sh↑) | P(Cal↑) |
|---|---|---|---|---|---|
| SET2/λ=0.1/α=0.25 | 2.650 | -17.5% | 5.43 | 0.06 | 0.07 |
| SET1/λ=0.1/α=0.5 | 2.056 | -15.7% | 4.21 | 0.04 | 0.10 |
| SET3/λ=0.1/α=0.75 | 1.699 | -10.2% | 3.37 | 0.08 | 0.11 |

**PASS-A (Sharpe uplift)**: 0/48. **PASS-B (Risk uplift)**: 0/48.

Нейтрализация **всегда ухудшает** Sharpe (от -1.1 до -3.7). Даже при минимальном α=0.25 Sharpe падает с 3.78 до ~2.0-2.6. Корреляция с оригинальным signal 0.87-0.99 — нейтрализация слишком агрессивна для этого типа модели.

**VERDICT: ❌ FAIL** — neutralization destroys signal. R68 signal не имеет вредной экспозиции к BTC beta / momentum / vol — они *нужны* для alpha.

**Time**: 8.9 минут.

## R111 — Spillover Features (inter-coin lags + market factors) ❌ FAIL

**Гипотеза**: Межмонетные лаги (BTС→altcoins), market factors (PCA, dispersion), и spillover (β×btc_ret_lag) добавят предсказательную силу.

**9 features tested:**

| Feature | IC | Stability | Max Corr (vs existing) | Gate |
|---|---|---|---|---|
| pc1_ret_lag1 | -0.0273 | 0.00 | 0.796 (pct_coins_up_12h) | ❌ cov=0.24 |
| dispersion_12h | +0.0216 | 0.67 | **0.945** (ret_dispersion_12h) | ❌ redundant |
| btc_ret_12h_lag1 | -0.0146 | 0.33 | 0.704 (pct_coins_up_12h) | ❌ IC+redundant |
| eth_ret_12h_lag1 | -0.0125 | 0.67 | 0.751 (pct_coins_up_12h) | ❌ IC+redundant |
| spill_btc | -0.0117 | 0.33 | 0.686 (pct_coins_up_12h) | ❌ IC |
| mkt_ret_12h | -0.0015 | 0.33 | 0.845 (pct_coins_up_12h) | ❌ IC+redundant |
| mkt_ret_12h_exBTC | -0.0010 | 0.33 | 0.844 (pct_coins_up_12h) | ❌ IC+redundant |
| spill_mkt | +0.0009 | 0.33 | 0.838 (pct_coins_up_12h) | ❌ IC+redundant |
| beta_btc_60 | -0.0008 | 0.33 | 0.210 (iv_rv_spread) | ❌ IC |

**Gate**: |IC| ≥ 0.03, stability ≥ 2/3, coverage ≥ 70%, redundancy < 0.70.

**Result**: 0/9 pass gate. Best |IC| = 0.027 (pc1_ret_lag1), но coverage только 24.5%.

**Ключевой insight**: Большинство spillover фич **высоко коррелированы** с уже существующими breadth features (pct_coins_up_12h, ret_dispersion_12h). R68 уже захватывает market-level информацию через эти features. Нового сигнала нет.

**VERDICT: ❌ FAIL** — spillover features либо redundant (corr > 0.70 с existing), либо IC < 0.03. WF test не проводился.

**Time**: 3.9 минуты.

## R112 — Factor-Mimicking Portfolios (FMP) ❌ FAIL

**Гипотеза**: Построить FMP из характеристик (cum_funding_24h, oi_velocity, rel_volume_cs) → time-series features (level, z120, momentum, skewness) → добавить как market-level фичи.

**Методология FMP**: На каждом timestamp t: z-score характеристики → веса w = z/Σ|z| (dollar-neutral) → fmp_ret = Σ(w × ret_12h). Затем из ряда FMP returns строим 4 производные фичи × 3 характеристики = 12 фич. Все features shift(1).

**⚠️ Обнаружена и исправлена утечка данных**: Первая версия использовала fwd_ret_12h (forward return) для FMP — IC=0.23 (!!), Sharpe 5.48. После исправления на ret_12h (past return) — IC упал до 0.004-0.018. **Всегда использовать realized returns для FMP!**

**12 features tested (corrected version):**

| Feature | IC | Stability | Max Corr | Gate |
|---|---|---|---|---|
| fmp_level_cum_funding_24h | -0.0183 | 0.67 | 0.069 | ❌ IC |
| fmp_mom_rel_volume_cs | +0.0164 | 0.33 | 0.244 | ❌ IC+stab |
| fmp_level_rel_volume_cs | -0.0165 | 0.33 | 0.304 | ❌ IC+stab |
| fmp_tail_oi_velocity | +0.0161 | 0.67 | 0.082 | ❌ IC |
| fmp_z120_cum_funding_24h | +0.0113 | 1.00 | 0.063 | ❌ IC |
| fmp_mom_cum_funding_24h | +0.0104 | 1.00 | 0.028 | ❌ IC |
| fmp_mom_oi_velocity | -0.0086 | 0.00 | 0.139 | ❌ IC+stab |
| fmp_z120_oi_velocity | -0.0046 | 0.00 | 0.152 | ❌ IC+stab |
| fmp_tail_cum_funding_24h | -0.0041 | 0.67 | 0.055 | ❌ IC |
| fmp_level_oi_velocity | -0.0041 | 0.67 | 0.576 | ❌ IC |
| fmp_z120_rel_volume_cs | +0.0039 | 1.00 | 0.372 | ❌ IC |
| fmp_tail_rel_volume_cs | +0.0029 | 0.33 | 0.120 | ❌ IC+stab |

**Result**: 0/12 pass gate. Best |IC| = 0.018. FMP approach в crypto hourly data не даёт IC ≥ 0.03.

**VERDICT: ❌ FAIL** — WF test не проводился.

**Time**: 5.4 минуты.

### Добавлено в "proven useless":
- ❌ Funding arb в текущем режиме (R108, zero opportunities 2025-2026)
- ❌ Macro features (DXY/VIX/SPX/US10Y/Gold): IC < 0.02 для всех 12 фич (R109)
- ❌ Prediction neutralization (Numerai-style): Sharpe всегда падает, 0/48 PASS (R110)
- ❌ Spillover features (inter-coin lags, PCA, market factors): redundant с existing breadth (R111)
- ❌ Factor-Mimicking Portfolios (FMP): IC < 0.02 для всех 12 фич после fix lookahead (R112)

## R113 — P0 Fix + Trend Cutoff Reoptimization ✅

### Предыстория: критический баг sim-vs-live (P0)

**Баг**: `simulate()` при `trend_str > trend_cutoff` делала `continue` — пропуск периода. В симуляции = 0% return (кэш). В live боте `construct_portfolio()` возвращала `[]`, но позиции **оставались открытыми** на бирже и дрифтили без управления.

Старый Sharpe 3.777 был **нечестным** — 325 из 1013 периодов (32%) просто не учитывались в расчёте. Реальная доходность привязана к 688 периодам, а риск — к 1013.

### P0 Fix: simulate_v2() — risk-off state machine

Новая архитектура разделяет два состояния:
- **selection_state** (EMA/prev_preds) — **живёт через risk_off**. При выходе из flat-периода EMA-smoothing продолжает с правильными значениями, hysteresis корректен.
- **positions_state** (prev_longs/prev_shorts) — очищается при enter_off. Re-entry = все позиции "новые" (честные costs).

Ключевые отличия от наивного фикса:
1. **Каждый** rebal timestamp записывает return (no `continue` skip). Periods = calendar.
2. Risk-off с **гистерезисом**: enter_off когда `trend_str > cutoff_on`, exit_off когда `trend_str < cutoff_off` (= cutoff_on − 0.1).
3. EMA **не сбрасывается** в risk_off → нет деградации gross returns на активных периодах.
4. Dynamic exposure (0.7–cutoff диапазон) привязан к cutoff_on.

**Sanity check**: Periods = 1013 = expected calendar periods → **PASS**.

### Версии пакетов — КРИТИЧНО для воспроизводимости

Обнаружено: незапиненные пакеты (pandas 3.0.2 вместо 2.3.3) давали Net Sharpe 2.228 вместо 3.777 на том же коде. pandas 3.0 тихо ломает `groupby.apply` (дропает grouping column). **requirements.txt теперь полностью запинен:**
- numpy==2.4.3, pandas==2.3.3, scipy==1.17.1, lightgbm==4.6.0, xgboost==3.2.0, scikit-learn==1.8.0

### R113 Grid Results (cutoff_on sweep)

| cutoff_on | cutoff_off | Net Sharpe | Gross Sharpe | Return | MaxDD | Calmar | %flat | #off events | Avg Off Dur |
|---|---|---|---|---|---|---|---|---|---|
| **0.9** | **0.8** | **3.057** | **3.510** | **183.9%** | **-11.2%** | **16.47** | 33.9% | 71 | 4.8 |
| 1.0 | 0.9 | 3.061 | 3.526 | 188.8% | -12.2% | 15.49 | 28.7% | 60 | 4.8 |
| 1.2 | 1.1 | 2.975 | 3.431 | 201.3% | -14.7% | 13.68 | 21.9% | 45 | 4.9 |
| 1.5 | 1.4 | 2.135 | 2.585 | 135.5% | -17.7% | 7.66 | 12.3% | 32 | 3.9 |
| 1.8 | 1.7 | 1.913 | 2.355 | 123.1% | -23.4% | 5.25 | 6.5% | 18 | 3.7 |
| 2.0 | 1.9 | 1.940 | 2.377 | 130.2% | -24.2% | 5.37 | 3.8% | 9 | 4.3 |
| 2.5 | 2.4 | 1.740 | 2.170 | 115.3% | -25.4% | 4.53 | 1.1% | 2 | 5.5 |
| 3.0 | 2.9 | 1.721 | 2.142 | 116.5% | -26.1% | 4.46 | 0.7% | 2 | 3.5 |
| None | — | 1.565 | 1.956 | 110.6% | -28.5% | 3.88 | 0% | 0 | 0 |

### Выводы R113

1. **Чёткая монотонность**: ниже cutoff → больше flat → лучше Sharpe/DD/Calmar. Trend filter реально защищает: без него MaxDD -28.5% и Sharpe 1.565.
2. **cutoff_on=0.9/0.8** — лучший risk-adjusted: Calmar 16.47, DD -11.2%. cutoff_on=1.0 чуть выше по Sharpe (+0.004) но хуже по DD и Calmar.
3. **Честный baseline**: Net Sharpe **3.057** (vs 3.777 старый нечестный). Разница = учёт flat-периодов с нулевым return.
4. **Фикс live бота**: `run_trading.py` теперь при risk_off закрывает все позиции через `rebalance_positions(exchange, [], ...)`. State machine с `state['trend_risk_off']` flag.
5. **Файл**: `_research_r113_trend_cutoff_reopt.py`, результаты: `results/r113_grid.csv`, `results/r113_best.json`.

### Обновление production baseline:
- **Старый baseline (нечестный)**: Net Sharpe = 3.777, 688 periods (32% пропущено)
- **Новый baseline (честный)**: Net Sharpe = **3.057**, 1013 periods, MaxDD = -11.2%, Calmar = 16.47, Return = 183.9%
- Config: `cutoff_on=0.9, cutoff_off=0.8`, всё остальное без изменений (4L/2S, ema_alpha=0.5, hysteresis=3)

---

## R114 — Continuous Trend Sizing ❌ FAIL

### Гипотеза
Вместо бинарного cutoff (100% или 0%) использовать линейное масштабирование размера позиции:
`size_mult = clip(1 - (trend_str - dyn_start) / (flat_threshold - dyn_start), 0, 1)`

Идея: уменьшить flat-время (33.9%) за счёт частичных позиций в "серой зоне" trend → вернуть diluted Sharpe.

### Grid search: flat_threshold × dyn_start (18 конфигов)

| Config | NetSh | GrSh | Ret% | DD% | Calmar | %flat | %reduced | AvgSize |
|---|---|---|---|---|---|---|---|---|
| **R113 baseline** | **3.057** | **3.510** | **183.9** | **-11.2** | **16.47** | **33.9%** | **n/a** | **n/a** |
| ft1.5_ds0.3 | 2.683 | 3.112 | 139.2 | -12.7 | 10.94 | 11.0% | 61.0% | 0.646 |
| ft1.5_ds0.5 | 2.763 | 3.195 | 162.9 | -13.0 | 12.51 | 11.0% | 44.5% | 0.703 |
| ft1.5_ds0.7 | 2.745 | 3.181 | 175.3 | -14.4 | 12.13 | 11.0% | 31.6% | 0.752 |
| ft2.0_ds0.3 | 2.452 | 2.875 | 137.9 | -16.5 | 8.35 | 3.7% | 68.2% | 0.730 |
| ft2.0_ds0.5 | 2.458 | 2.881 | 151.9 | -17.9 | 8.47 | 3.7% | 51.7% | 0.779 |
| ft2.0_ds0.7 | 2.381 | 2.803 | 155.7 | -19.7 | 7.91 | 3.7% | 38.7% | 0.821 |
| ft2.5_ds0.3 | 2.226 | 2.641 | 131.6 | -19.4 | 6.79 | 1.0% | 70.9% | 0.786 |
| ft3.0_ds0.3 | 2.080 | 2.494 | 127.1 | -21.1 | 6.01 | 0.5% | 71.4% | 0.824 |
| ft4.0_ds0.3 | 1.927 | 2.335 | 122.8 | -23.2 | 5.30 | 0.0% | 71.9% | 0.872 |
| ft5.0_ds0.3 | 1.844 | 2.249 | 120.4 | -24.3 | 4.95 | 0.0% | 71.9% | 0.899 |

### Выводы R114

1. **R113 binary cutoff доминирует ВСЕ continuous sizing конфиги** по Sharpe, DD и Calmar.
2. Лучший continuous: ft1.5_ds0.5 → NetSh=2.763, DD=-13.0%, Calmar=12.51 — хуже R113 по всем метрикам.
3. **Чёткая монотонность**: чем больше exposure в "серой зоне" (выше flat_threshold, выше dyn_start) → хуже DD (от -12.7% до -25.9%).
4. Периоды с высоким trend_strength действительно токсичны — даже частичная экспозиция вредит.
5. **Binary cutoff оптимален**: когда trend_strength > 0.9, правильное действие = полностью выйти в cash.
6. **Файл**: `_research_r114_continuous_sizing.py`, результаты: `results/r114_grid.csv`, `results/r114_best.json`.

### Статус: R113 baseline остаётся лучшим. Двигаемся к R115 (universe expansion) или R116 (8h rebalance).

---

## R114b — Risk-off Churn Reduction ✅ WIN

### Гипотеза
Сохранить R113 edge, но уменьшить unnecessary off-events через timing hysteresis:
- `min_risk_off_periods`: once in risk_off, stay at least N periods before exit
- `min_risk_on_periods`: once in risk_on, stay at least N periods before can re-enter off

### Grid (36 конфигов): cutoff_off × min_off × min_on

| Config | NetSh | GrSh | Ret% | DD% | Calmar | %flat | #off | AvgDur | Cost% |
|---|---|---|---|---|---|---|---|---|---|
| **R113 baseline** | **3.057** | **3.510** | **183.9** | **-11.2** | **16.47** | **33.9%** | **71** | **4.8** | **16.21** |
| **off0.8_moff2_mon0** | **3.266** | **3.707** | **199.5** | **-10.9** | **18.25** | **36.6%** | **68** | **5.4** | **15.51** |
| off0.8_moff2_mon1 | 3.266 | 3.707 | 199.5 | -10.9 | 18.25 | 36.6% | 68 | 5.4 | 15.51 |
| off0.8_moff3_mon0 | 3.196 | 3.627 | 186.8 | -10.9 | 17.09 | 39.0% | 63 | 6.2 | 14.88 |
| off0.75_moff2_mon0 | 3.140 | 3.568 | 184.0 | -11.3 | 16.25 | 38.3% | 63 | 6.1 | 14.92 |
| off0.8_moff3_mon2 | 3.143 | 3.577 | 183.2 | -11.4 | 16.03 | 37.5% | 59 | 6.3 | 15.07 |

### Выводы R114b

1. **off0.8_moff2_mon0 — новый чемпион**: Sharpe 3.266 (+0.21), DD -10.9% (+0.3pp), Calmar 18.25 (+1.78), Return 199.5% (+15.6pp).
2. `min_risk_off_periods=2` (24h минимальный stay in risk_off) — ключевое улучшение. Предотвращает "flicker" — быстрый выход/вход в risk_off.
3. `min_risk_on_periods=0/1` эквивалентны (no effect). `mon=2` слегка вредит.
4. Снижение cutoff_off (0.75, 0.7, 0.65) увеличивает flat% и ухудшает Sharpe — original 0.8 оптимален.
5. Churn reduction -25% не достигнуто (68 vs 71), но метрики однозначно лучше.
6. **Файл**: `_research_r114b_churn_reduction.py`, результаты: `results/r114b_grid.csv`, `results/r114b_best.json`.

### Обновление production config:
- **Старый**: `cutoff_on=0.9, cutoff_off=0.8, min_risk_off=1, min_risk_on=0`
- **Новый**: `cutoff_on=0.9, cutoff_off=0.8, min_risk_off_periods=2, min_risk_on_periods=0`

---

## R115 — Universe Expansion (35 → 50) ❌ INCONCLUSIVE

### Что было сделано
- Binance API недоступен на VM → использовали 50 existing символов (35 + 15 дополнительных из download_crypto.py)
- Модель обучена на ВСЕХ 50 символах (не 35)
- Point-in-time ADV filter (7d rolling dollar volume)
- Grid: min_adv ∈ {5M, 10M, 20M} × n_long/n_short ∈ {(4,2), (6,3)}

### Результаты

| Config | NetSh | GrSh | Ret% | DD% | Calmar | Cost% | AvgN |
|---|---|---|---|---|---|---|---|
| R113 SYM_35 (50-sym model) | 1.386 | 1.847 | 58.9 | -24.3 | 2.43 | 17.02 | 35 |
| adv5M_4L2S | 1.637 | 2.161 | 74.5 | -16.8 | 4.43 | 19.37 | 32 |
| adv10M_4L2S | 1.833 | 2.303 | 83.2 | -11.5 | 7.24 | 16.65 | 26 |
| **adv20M_4L2S** | **2.563** | **2.991** | **119.7** | **-10.1** | **11.83** | **13.83** | **19** |
| adv5M_6L3S | 0.802 | 1.415 | 24.3 | -18.0 | 1.35 | 19.45 | 32 |
| adv10M_6L3S | 1.683 | 2.216 | 64.2 | -18.0 | 3.57 | 16.87 | 26 |
| adv20M_6L3S | 1.397 | 1.866 | 46.0 | -22.4 | 2.05 | 13.73 | 19 |

### Ключевое наблюдение: обучение на 50 символах УБИЛО модель!
- **R113 с моделью обученной на 35 symb**: Sharpe 3.057, DD -11.2%, Calmar 16.47
- **R113 с моделью обученной на 50 symb**: Sharpe 1.386, DD -24.3%, Calmar 2.43
- Деградация = 15 дополнительных мелких альтов добавили noise в training

### Выводы R115
1. **НЕ расширять training universe** — extra low-cap coins = noise, degraded predictions.
2. Volume filter работает: adv20M (19 coins avg) лучше adv5M (32 coins avg). Качество > количество.
3. 4L/2S стабильно лучше 6L/3S при любом ADV пороге.
4. Правильный подход для будущего: train на SYM_35, но **predict на 50+** (split universes). Требует модификации train_ensemble.
5. Binance API не работает на MLC VM → нужно качать данные локально.
6. **Файл**: `_research_r115_universe_expansion.py`, результаты: `results/r115_grid.csv`.

---

## R116 — 8h Rebalance A/B Test ❌ FAIL

### Гипотеза
Больше decision points (3/day vs 2) → потенциально лучше timing. Модель та же (12h target), только частота ребалансировки ↑.

### Результаты

| Config | NetSh | GrSh | Ret% | DD% | Calmar | Cost% |
|---|---|---|---|---|---|---|
| **12h co=0.9 (baseline)** | **3.057** | **3.510** | **183.9** | **-11.2** | **16.47** | **16.21** |
| 12h co=1.0 | 3.061 | 3.526 | 188.8 | -12.2 | 15.49 | 16.94 |
| 8h co=0.9 | 0.503 | 0.918 | 19.6 | -23.3 | 0.84 | 18.97 |
| 8h co=1.0 | 0.531 | 0.945 | 22.0 | -23.0 | 0.95 | 19.88 |
| 8h co=None | -0.090 | 0.276 | -16.1 | -51.0 | -0.32 | 25.07 |

### Выводы R116

1. **Катастрофический FAIL**: 8h rebalance Sharpe 0.5 vs 12h Sharpe 3.06. Calmar 0.95 vs 16.47.
2. Модель обучена предсказывать 12h returns. При 8h rebalance используем fwd_ret_8h, который захватывает лишь часть предсказанного move.
3. 8h returns = меньший signal + те же costs → signal-to-noise ratio коллапсирует.
4. Подтверждает R93: shorter timeframe = worse risk-adjusted performance.
5. **12h rebalance оптимален** для текущей модели.
6. **Файл**: `_research_r116_8h_rebalance.py`, результаты: `results/r116_grid.csv`.

---

## R114c — Validation of R114b Champion ✅ ALL PASS

**Цель**: Проверить R114b champion (min_risk_off=2) тремя независимыми валидациями.

### CHECK 1: Per-window stability (W1/W2/W3)

| Window | Period | R113 Sharpe | R114b Sharpe | ΔSharpe | R113 Calmar | R114b Calmar | ΔCalmar | Result |
|--------|--------|-------------|--------------|---------|-------------|--------------|---------|--------|
| W1 | 2024-10 → 2025-05 | 2.342 | 2.308 | -0.034 | 3.88 | 4.32 | +0.44 | LOSE (marginal) |
| W2 | 2025-05 → 2025-11 | 4.452 | 4.834 | +0.382 | 11.46 | 12.06 | +0.60 | WIN |
| W3 | 2025-11 → 2026-03 | 2.478 | 3.050 | +0.572 | 1.81 | 2.30 | +0.49 | WIN |

**Результат**: R114b wins 2/3 windows. W1 проигрывает на Sharpe лишь на -0.034 (в пределах шума), но выигрывает на Calmar. **PASS**.

### CHECK 2: Per-year stability

| Period | R113 Sharpe | R114b Sharpe | ΔSharpe | Result |
|--------|-------------|--------------|---------|--------|
| 2024-H2 | 3.408 | 3.519 | +0.111 | WIN |
| 2025-H1 | 2.607 | 2.792 | +0.185 | WIN |
| 2025-H2 | 4.156 | 4.323 | +0.167 | WIN |
| 2026-Q1 | 1.012 | 1.476 | +0.464 | WIN |

**Результат**: R114b wins ALL 4/4 periods. Наибольший uplift в 2026-Q1 (+0.464). **PASS**.

### CHECK 3: Block Bootstrap ΔSharpe/ΔCalmar (5000 resamples, block=30)

| Metric | Median Δ | P(Δ > 0) | 90% CI |
|--------|----------|----------|--------|
| ΔSharpe | +0.186 | 0.880 | [-0.073, +0.448] |
| ΔCalmar | +1.50 | 0.848 | [-1.14, +5.02] |

**Результат**: Median ΔSharpe = +0.186, P(positive) = 88%. **PASS**.

### Финальный вердикт R114c

**OVERALL PASS — R114b is production-grade champion.**
- Выигрывает 2/3 windows и ВСЕ 4/4 year-periods
- Bootstrap: 88% confidence что Sharpe improvement реальный
- Механизм sound: min_risk_off_periods=2 предотвращает flicker

**Файл**: `_research_r114c_validation.py`, результаты: `results/r114c_validation.json`.

---

## R117 — Dynamic K (Confidence-based Position Sizing) ✅ WIN

**Цель**: Вместо фиксированных 4L/2S, масштабировать позиции по "уверенности" модели.
**Confidence**: std(predictions) across coins at each timestamp (expanding window quantiles).

### Лучшие конфигурации

| Config | Method | Low→K | Mid→K | High→K | NetSh | GrSh | Ret% | DD% | Calmar | Cost% |
|--------|--------|-------|-------|--------|-------|------|------|-----|--------|-------|
| **R114b baseline** | fixed | – | 4L2S | – | 3.266 | 3.707 | 199.5 | -10.9 | 18.25 | 15.51 |
| **std_q30_4L2S_else2L1S** | std | ≤30%→2L1S | 4L2S | – | **3.906** | 4.379 | **212.9** | -11.9 | 17.82 | 14.42 |
| **std_q40_4L2S_else2L1S** | std | ≤40%→2L1S | 4L2S | – | 3.460 | 3.905 | 185.5 | **-10.1** | **18.33** | 14.07 |
| rng_q70_30_6L3S_2L1S | range | ≤30%→2L1S | 4L2S | ≥70%→6L3S | 3.188 | 3.692 | 138.5 | -9.0 | 15.34 | 14.43 |

### Ключевые выводы R117

1. **EXPANDING K (6L/3S при high confidence) УХУДШАЕТ**: все конфиги с 6L/3S → хуже baseline
2. **SHRINKING K (2L/1S при low confidence) РАБОТАЕТ**: все варианты с low→2L/1S улучшают Sharpe
3. **Best по Calmar**: `std_q40_4L2S_else2L1S` → Calmar 18.33 (+0.08), DD -10.1% (+0.8pp), Sharpe 3.460 (+0.19)
4. **Best по Sharpe**: `std_q30_4L2S_else2L1S` → Sharpe 3.906 (+0.64!), Return 212.9% (+13.4pp), Cost% 14.42 (-1.09)
5. **K distribution** (q40): warmup=49, mid=585 (58%), low=379 (37%) → 37% time in reduced exposure
6. **Mechanism sound**: когда модель не уверена (predictions clustered), уменьшить exposure → меньше шума

### Delta vs R114b

| Metric | R114b | Best (q40) | Δ |
|--------|-------|------------|---|
| Net Sharpe | 3.266 | 3.460 | +0.194 |
| Calmar | 18.25 | 18.33 | +0.08 |
| MaxDD | -10.9% | -10.1% | +0.8pp |
| Return | 199.5% | 185.5% | -14.0pp |
| Cost% | 15.51 | 14.07 | -1.44pp |

**Файл**: `_research_r117_dynamic_k.py`, результаты: `results/r117_grid.csv`, `results/r117_best.json`.

---

## R115b — Frozen Split-Universe (Train 35, Predict 50) ❌ INCONCLUSIVE

**Цель**: Train model на SYM_35 only, predict на 50 symbols, select from larger pool.
**Frozen normalization**: market-level features computed from SYM_35 anchor only.

### Результаты

| Config | NetSh | GrSh | Ret% | DD% | Calmar | Cost% | AvgUniverse |
|--------|-------|------|------|-----|--------|-------|-------------|
| R114b_SYM35_4L2S (baseline) | 2.557 | 2.949 | 151.1 | -14.0 | 10.75 | 14.97 | 35 |
| split_adv10M_4L2S | 2.643 | 3.041 | 153.1 | -16.9 | 9.07 | 14.88 | 26 |
| **split_adv10M_6L3S** | **3.046** | **3.529** | 144.6 | **-10.5** | **13.71** | 14.85 | 26 |
| split_adv20M_4L2S | 2.232 | 2.569 | 113.3 | -17.8 | 6.38 | 12.22 | 19 |
| split_adv20M_6L3S | 2.867 | 3.283 | 123.4 | -11.3 | 10.89 | 12.21 | 19 |
| split_adv50M_4L2S | 1.968 | 2.235 | 81.4 | -13.9 | 5.85 | 8.67 | 12 |
| split_adv50M_6L3S | 0.970 | 1.249 | 30.2 | -23.8 | 1.27 | 8.61 | 12 |

### Критическая проблема

**Baseline degradation**: R115b baseline (SYM_35, 4L2S) = Sharpe 2.557, но настоящий R114b = 3.266.
Разница = 0.71 Sharpe (!). Причина: frozen normalization заморозила только MARKET_LEVEL_FEATURES,
но cross-sectional ranked features (`oi_chg_12h_cs`, `taker_cvd_12h_cs`, `cum_funding_24h_cs`, `cs_rank_ma_5`)
были пересчитаны по 50 символам → распределения для SYM_35 сместились → модель деградировала.

### Выводы R115b

1. **Split-universe ВНУТРИ своей модели**: expanding от 4L2S к 6L3S помогает (+0.49 Sharpe, +2.96 Calmar)
2. **НО модель деградировала**: baseline 2.557 vs real R114b 3.266 = -0.71 Sharpe loss
3. **Frozen normalization неполная**: нужно замораживать ВСЕ cs-features, не только market-level
4. **Правильный подход**: compute ALL cs features from SYM_35 anchor only, then extrapolate to expanded symbols
5. Binance API заблокировано на VM → только 50 symbols (35 + 15 existing)

**Файл**: `_research_r115b_split_universe.py`, результаты: `results/r115b_grid.csv`, `results/r115b_best.json`.

---

## Консенсус AI‑консультантов (9 апреля 2026)

Два независимых консультанта проанализировали результаты R114c/R117/R115b.

### Согласованные решения (100% консенсус)

1. **Deploy R114b NOW** — validated (R114c ALL 3 CHECKS PASS), деплоить немедленно
2. **R117 q40 > q30** — лучший risk profile (DD -10.1%, Calmar 18.33), q30 = aggressive/overfitting risk
3. **R117c обязательно** — перед продом обязательна валидация по протоколу R114c + sensitivity sweep
4. **Combo (R117+R115b) — только после отдельной валидации**

### Расхождение: R115b

- **Консультант 1**: Чинить → R115c full-frozen cs (freeze ALL cs features, non-regression gate)
- **Консультант 2**: **Закрыть R115 полностью** — проблема глубже normalization: joint distribution 35 coins, extrapolation опасна

**Принятое решение**: **R115 CLOSED.** Universe = 35. Консультант 2 аргументировал убедительнее:
даже full-frozen cs не решает joint distribution problem, модель обучена на 35 символов.

### Доп. инсайт от консультанта 2 (стратегический)

```
R68 original (dishonest): Sharpe 3.777
R113 (honest fix):        Sharpe 3.057  (-0.72)
R114b (churn reduction):  Sharpe 3.266  (+0.21)
R117 q40 (dynamic K):     Sharpe 3.460  (+0.19, pending validation)

Total honest recovery:    3.057 → 3.460 = +0.40
Dilution from 34% flat:   Sharpe 3.46 × 1/sqrt(0.66) ≈ 4.26 equivalent
→ Active-period Sharpe STRONGER than original R68
→ Research map ALMOST FULLY CLOSED within current paradigm
```

### Action plan (в процессе)

- [x] Deploy R114b на VPS (state machine fix: cutoff_off=0.8, min_risk_off_periods=2) — DONE
- [x] R117c validation — **FAIL** (1/3 windows, catastrophic W3, bootstrap P=0.61)
- **VERDICT: R114b remains champion.**

### R117c Validation Results (9 апреля 2026)

```
CHECK 0 Sensitivity sweep: PASS (3/4 q values beat baseline)
  q=0.30: Sharpe +0.640, Calmar -0.43
  q=0.35: Sharpe +0.310, Calmar +0.91 (best Calmar=19.16!)
  q=0.40: Sharpe +0.194, Calmar +0.08
  q=0.45: Sharpe -0.492, Calmar -7.40 → CLIFF

CHECK 1 Per-window: FAIL (1/3 wins + CATASTROPHIC in W3)
  W1: ΔSharpe=+1.129  WIN (big)
  W2: ΔSharpe=-0.397  LOSE (marginal, Calmar +2.24)
  W3: ΔSharpe=-1.340  LOSE CATASTROPHIC (3.050→1.710)

CHECK 2 Per-year: PASS (2/4 marginal)
  2024-H2: +2.535 WIN | 2025-H1: +0.472 WIN
  2025-H2: -0.653 LOSE | 2026-Q1: -0.681 LOSE

CHECK 3 Bootstrap: FAIL
  P(ΔSharpe>0) = 0.611 (need >0.80)
  P(ΔCalmar>0) = 0.604 (need >0.80)
  Median ΔSharpe = +0.152, 90% CI: [-0.678, +1.027]
```

**Инсайт**: Dynamic K помогал в ранних периодах (W1, 2024-H2) но вредил в поздних (W3, 2026-Q1).
Эффект regime-dependent → не generalize → keep R114b fixed 4L/2S.

### Закрытые направления (финальная карта)

```
CLOSED: Features (31), Models (LGB+XGB), Portfolio base (4L/2S),
        Trend filter (binary 0.9), Churn (min_off=2), Ensemble,
        Rebalance (12h), Neutralization, Spillover, FMP,
        Continuous sizing, Expanding K (6L/3S worse),
        8h rebalance, Universe expansion (R115),
        Dynamic K — R117c FAIL (regime-dependent)

OPEN:
  1. CryptoQuant exchange flows (external data, untested)
  2. Extend maker-first execution to Tier2/Tier3 (R121: +0.32 Sharpe potential)
  3. OKX fee tier optimization (referral discount 20%, OKB holding for Lv2)
```

## R121 — Realistic Cost Model Audit ✅ DONE (14 апреля 2026)

**Проблема**: Бэктест занижал комиссии. Функция `_cost_for_sym()` предполагала 92% maker fills (лимитные ордера), но продакшен использует смесь: Tier1 через maker-first (post_only), Tier2 через aggressive limit, Tier3 через market. Комиссия taker на OKX Regular Lv1 = 0.05% per side, maker = 0.02%. При обороте 2.5x за период это существенная разница.

**Метод**: Walk-forward ensemble (3 окна × 5 сидов × LGB+XGB = 30 моделей), 6 сценариев с разными моделями костов. Прогон на MLC VM с правильными пакетами (numpy 2.4.3, scipy 1.17.1).

**Сценарии**:
- **S0 original** — старая модель костов из бэктеста (92% maker assumption). Заниженные комиссии.
- **S2 okx taker + delay** — нижняя граница: все ордера маркетные, OKX taker fee 0.05%, фандинг 1.2bp/12h, шум исполнения 3bp.
- **S4 okx maker + delay** — верхняя граница: все лимитные ордера, OKX maker fee 0.02%, тот же фандинг и шум.
- **S5 pessimistic** — worst case: максимальные комиссии (5bp/side + wide spread), фандинг 1.5bp/12h, шум 5bp.
- **S6 prod blended** — **реальный прод**: Tier1 (BTC,ETH,SOL,BNB,XRP) через maker-first (90% maker 2bp + 10% taker 6bp = 2.4bp), Tier2 (средние альты) через aggressive limit (50% maker 4bp + 50% taker 7bp = 5.5bp), Tier3 (мелкие) через market (10bp).

**Результаты**:

```
Scenario                    NetSh    Ret%     DD%  Calmar  Cost%
S0_original (старый бэктест) 3.266   199.5   -10.9   18.25  15.5%
S2_okx_taker+delay (нижняя)  2.612   138.3   -11.5   12.02  38.3%
S6_prod_blended (РЕАЛЬНЫЙ)   2.831   157.3   -10.9   14.38  30.6%
S4_okx_maker+delay (верхняя) 3.147   187.4   -10.9   17.27  19.6%
S5_pessimistic (worst case)  2.057    96.4   -13.5    7.15  57.6%
```

**По окнам (Net Sharpe)**:

```
         S0      S2(нижняя)  S6(РЕАЛЬНЫЙ)  S4(верхняя)  S5(worst)
W1:     2.305    1.675        1.880         2.200        1.122
W2:     4.834    4.065        4.334         4.709        3.411
W3:     3.050    2.496        2.677         2.904        2.063
```

Все окна прибыльны во всех сценариях.

**Декомпозиция влияния**:
- Маркетные ордера vs старая модель: **-0.651 Sharpe** (главный фактор)
- Execution delay (3bp шум): **-0.003 Sharpe** (пренебрежимо)
- Реальный prod execution mix vs 100% taker: **+0.219 Sharpe** (maker-first на Tier1 работает)
- Переход всех ордеров на лимитки: **+0.535 Sharpe** (потолок)

**Выводы**:
1. **Реальный прод (S6) даёт Sharpe 2.83, Calmar 14.4** — наиболее точная оценка live performance.
2. **Бэктест завышал Sharpe на 0.44** из-за неправильной модели костов (с учётом что прод уже использует maker-first).
3. **Maker-first на Tier1 уже экономит**: S6 (2.83) vs S2 (2.61) = +0.22 Sharpe от существующего maker-first execution.
4. **Расширение maker-first на все тиры** может поднять ещё на +0.32 Sharpe (до 3.15).
5. **Execution delay не проблема**: при ребалансировке раз в 12h slippage пренебрежим.
6. **Даже worst case (Sharpe 2.06, Calmar 7.15)** — система прибыльна.

**ПРИМЕЧАНИЕ**: В R121 cost functions используют OKX Regular Lv1 fees: taker 0.05% (5 bps), maker 0.02% (2 bps). НЕ VIP, НЕ rebate. Код: строки 53, 65 файла `_research_r121_realistic_costs.py`.

**Файл**: `_research_r121_realistic_costs.py` (commit 768f93c)
**Результаты**: `results/r121_cost_audit.csv`, `results/r121_realistic.json`

**VERDICT: ✅ DONE** — система валидна с реальными костами. Реалистичный Sharpe = 2.83 (S6 prod blended). Следующий шаг: расширить maker-first execution на Tier2/Tier3 для ещё +0.32 Sharpe.

## R122 — Directional BTC During Risk-Off ❌ INCONCLUSIVE (14 апреля 2026)

**Гипотеза**: 36.6% времени модель в кэше (risk-off при trend_strength > 0.9). Может ли directional BTC/ETH стратегия генерировать доход в эти периоды?

**Метод**: 
1. Извлечь все risk-off периоды (2463 hourly bars, 382 спелла, средн. длительность 77h)
2. Naive baseline: long BTC при uptrend, short при downtrend, 50% exposure
3. LGB модель на 27 BTC-specific фичах (momentum, vol, derivatives, macro, sentiment), binary classification
4. Walk-forward на тех же 3 окнах, 5 сидов
5. Объединить: risk-on = main L/S модель, risk-off = BTC directional

**Статистика risk-off**:
- BTC mean return: +0.162% per 12h (positive bias → BTC trend continuation works)
- 52.8% positive returns (слабый edge)
- Trend UP: mean +0.281%, Trend DOWN: mean -0.011% (short не работает)
- "Always long" standalone Sharpe ≈ 1.53

**Результаты**:

```
Strategy                    NetSh    Ret%     DD%  Calmar  ΔSharpe
S6_baseline (main only)     2.831   157.3   -10.9   14.38     —
Naive_combined              2.390   136.2   -19.5    6.98   -0.441
LGB_combined                2.888   162.5   -10.9   14.85   +0.057
```

**По окнам (LGB combined)**:
- W1: 2.002 (was 1.880, +0.122) — единственное улучшение
- W2: 4.334 (unchanged)
- W3: 2.677 (unchanged)

**Анализ**:
- **Naive вредит**: DD ухудшается с -10.9% до -19.5%. Шорт BTC при даунтрендах убыточен.
- **LGB маргинально помогает (+0.057 Sharpe)**, но только потому что модель неуверена (mean confidence 0.055 → крошечные позиции). Accuracy 53.8% — почти монетка.
- Прирост +0.057 = **шум, не сигнал**. Не прошёл бы bootstrap.
- Улучшение только в W1, W2/W3 нетронуты → не generalize.

**Инсайт**: Risk-off периоды — это когда BTC делает резкие движения. Ловить тренд continuation на 12h горизонте при 53% accuracy невозможно. Модель правильно "отказывается" торговать (low confidence), что фактически = оставаться в кэше.

**VERDICT: ❌ INCONCLUSIVE** — +0.057 Sharpe нестатистически значим. Risk-off = правильно быть в кэше. Не внедрять.

**Файл**: `_research_r122_riskoff_btc.py` (commit d4ec8cd)

## R123 — News Sentiment Feature Evaluation ❌ NEGATIVE (14 апреля 2026)

**Гипотеза**: Добавление news sentiment фичей к 31 champion features улучшит Sharpe.

**Данные**: 2.4M строк crypto_news.parquet (2020-09 → 2026-03), VADER-scored, 10 фичей (8 per-coin + 2 market-level). Политические фичи отсутствуют.

**IC Scan** (14 фичей, включая 4 interaction):
```
Feature                               MeanIC    W1       W2       W3    Stable
nx_mkt_sent_x_vol                    -0.0375  -0.031  -0.132  +0.050     ✓
market_news_sentiment_24h            -0.0374  -0.050  -0.118  +0.056     ✓
market_news_count_24h                -0.0336  +0.002  -0.091  -0.012     ✗
nx_sent_divergence                   +0.0257  +0.040  +0.082  -0.045     ✓
nx_mkt_count_zscore                  -0.0246  +0.048  -0.077  -0.044     ✓
(rest: |IC| < 0.022, unstable)
```
4/14 фичей прошли IC gate (|IC|>0.02, стабильные в ≥2/3 окон). Но IC нестабильны: W2 доминирует, W1/W3 слабые или противоположного знака.

**Эксперименты** (WF 3 окна × 5 сидов, S6 prod_blended costs):
```
Experiment             NetSh    Ret%     DD%  Calmar  ΔSharpe  P(imp)
A_baseline (31f)       2.831   157.3   -10.9   14.38     —       —
B_market (+2f)         1.881    85.8   -17.8    4.82   -0.950   0.104
C_mkt_pol (=B)         1.881    85.8   -17.8    4.82   -0.950   0.104
D_all_news (+10f)      1.095    40.8   -22.1    1.85   -1.736   0.019
E_ic_pass (+4f)        0.624    20.1   -24.5    0.82   -2.207   0.003
F_interact (+6f)       0.802    28.6   -19.5    1.47   -2.029   0.006
```

**Анализ**:
- **Все эксперименты хуже baseline**, включая market-only (B: -0.950 Sharpe)
- DD ухудшается с -10.9% до -17.8...−24.5%
- IC-passing фичи (E) — **худший результат** (0.624). IC scan обманчив: W2 доминирует, модель переобучается
- Парадокс: IC фичи значимы, но при добавлении в ансамбль **вредят** — noise injection в LGB/XGB
- Bootstrap: P(improvement) < 0.11 для всех вариантов = baseline однозначно лучше
- Возможная причина: VADER слишком грубый (60% accuracy), нужен FinBERT/CryptoBERT или LLM

**VERDICT: ❌ NEGATIVE** — News sentiment features вредят LGB+XGB ensemble. Не внедрять. 
Для будущего: LLM-filtered sentiment (GPT/Claude) мог бы дать лучший сигнал, но требует API costs.

**Файл**: `_research_r123_news_sentiment.py` (commit 1d6895f)

## R124 — OKX Fee Optimization ✅ POSITIVE (15 апреля 2026)

**Гипотеза**: Снижение комиссий OKX (referral cashback, VIP tier, maker-first execution) увеличит net Sharpe без изменения модели.

**Метод**: Один раз обучить baseline ensemble (31 feature, 3 WF окна × 3 seeds), затем прогнать 8 стоимостных сценариев через тот же набор predictions. Меняется только cost function в `simulate_r121()`.

**Параметрическая cost function**:
```
cost_fn(sym) =
  Tier1 (BTC,ETH,SOL,BNB,XRP): maker_pct * maker_fee + (1-maker_pct) * (taker_fee + spread_T1)
  Tier2 (mid-cap):              maker_pct * (maker_fee + spread_T2) + (1-maker_pct) * (taker_fee + spread_T2)
  Tier3 (SAND,LDO,INJ,...):     taker_fee + spread_T3  (pure market orders)
```

Базовые ставки OKX Regular Lv1: maker=2bp, taker=5bp. Funding=1.2bp/12h. Exec delay=N(0,3bp).

**8 сценариев** (S6_current = текущий прод):

| Сценарий | Maker | Taker | Maker% T1/T2 | Описание |
|---|---|---|---|---|
| S6_current | 2bp | 5bp | 90%/50% | Текущий prod_blended |
| REF10 | 1.8bp | 4.5bp | 90%/50% | 10% referral cashback |
| REF20 | 1.6bp | 4bp | 90%/50% | 20% referral cashback |
| REF30 | 1.4bp | 3.5bp | 90%/50% | 30% referral (max promo) |
| MAKER_OPT | 2bp | 5bp | 95%/70% | Улучшенное исполнение |
| REF20_MAKER | 1.6bp | 4bp | 95%/70% | REF20 + улучшенное исполнение |
| VIP1 | 1.8bp | 4.5bp | 90%/50% | VIP1 tier (>$100K assets) |
| VIP1_MAKER | 1.8bp | 4.5bp | 95%/70% | VIP1 + улучшенное исполнение |

**Результаты**:
```
Scenario            Sharpe*  ΔSharpe  Return%   DD%   Cost%
S6_current            3.691     —      157.3  -10.9   23.6
REF10                 3.736   +0.035   161.1  -10.9   22.5
REF20                 3.780   +0.068   164.9  -10.9   21.4
REF30                 3.825   +0.103   168.8  -10.9   20.3
MAKER_OPT             3.746   +0.042   162.0  -10.9   22.1
REF20_MAKER           3.824   +0.102   168.7  -10.9   20.3
VIP1                  3.736   +0.035   161.1  -10.9   22.5
VIP1_MAKER            3.785   +0.072   165.4  -10.9   21.3
```
*Sharpe computed on trading periods only (excl. risk_off). Scale factor to full-method: ×0.767. Deltas are comparable.*

DD не меняется (-10.9%) → risk_off экзогенный (BTC trend), не зависит от costs.

**⚠️ Замечания из внешнего ревью**:
1. **baseline maker_pct=90%/50%**: если прод реально market orders (100% taker), то S6 baseline уже оптимистичен. Нужен дополнительный TAKER_ONLY baseline (maker_pct=0) для честной оценки.
2. **Exec delay как N(0,3bp)**: нулевое среднее не моделирует directional slippage. Лучше: `slippage = k × rvol × turnover` или Monte-Carlo по seeds.
3. **Sharpe на trading periods only**: если risk_off экзогенный — дельты корректны. Но лучше печатать оба (full + active).

**VERDICT: ✅ POSITIVE** — Referral 20% cashback = +0.068 Sharpe (≈+2.4% годовой доходности), бесплатно. DD не меняется. Maker-first execution + referral = +0.102 Sharpe потенциал.

**Файл**: `_research_r124_fee_optimization.py` (commit f4bebe0)

## R125 — FinBERT News Sentiment ❌ NEGATIVE (15 апреля 2026)

**Гипотеза**: Замена VADER (60% accuracy) на FinBERT (87% accuracy, ProsusAI/finbert) для scoring 954K новостей даст полезные features.

**Метод**:
1. Пере-scoring всех 954K новостей FinBERT на H100 80GB GPU (~5 мин)
2. Score = weighted sum: Σ(w_c × P(c|title)), w_pos=+1, w_neg=-1, w_neu=0
3. Rebuild features (per-coin 8 + market 2 + interactions 4 = 14 features)
4. Те же 6 экспериментов A-F что в R123 (WF 3 окна × 3 seeds, S6 costs)

**FinBERT scoring stats** (954,551 items, torch 2.5.1+cu121, fp16):
- Positive (>0.1): 44.8%, Negative (<-0.1): 24.6%, Neutral: 30.6%
- Mean: +0.080, Std: 0.529 (vs VADER: mean≈0, std≈0.3 — FinBERT более "экспрессивный")

**IC Scan (FinBERT)**:
```
Feature                          MeanIC    W1       W2       W3    Stable
nx_mkt_sent_x_vol               -0.0554  -0.025  -0.121  -0.020     ✓
market_news_sentiment_24h        -0.0530  -0.028  -0.110  -0.021     ✓
nx_sent_divergence               +0.0436  +0.030  +0.085  +0.016     ✓
nx_mkt_count_zscore              -0.0246  +0.048  -0.077  -0.044     ✓
(rest: |IC| < 0.034, not stable)
```
4/14 features прошли IC gate (vs 0 с VADER). Но IC scan **методологически некорректен** для market-level фичей (см. ниже).

**Эксперименты (S6 prod_blended costs)**:
```
Experiment           NetSh    Ret%    DD%   ΔSharpe  P(imp)
A_baseline (31f)     2.831   157.3  -10.9     —        —
B_market (+2f)       1.650    77.3  -18.9   -1.181    0.048
C_mkt_pol (=B)       1.650    77.3  -18.9   -1.181    0.048
D_all_news (+10f)    1.289    53.0  -20.0   -1.542    0.032
E_ic_pass (+4f)      1.104    43.7  -19.2   -1.727    0.011
F_interact (+6f)     0.527    16.4  -20.9   -2.304    0.002
```
Per-window: B: W1=0.183, W2=4.542, W3=0.762 (baseline: W1=1.880, W2=4.334, W3=2.677). W1 и W3 полностью ломаются, W2 чуть лучше.

**Сравнение VADER (R123) vs FinBERT (R125)**:
- VADER best: Sharpe ~2.83 (≈baseline) — не помог, не навредил
- FinBERT best: Sharpe 1.65 (−1.18) — **активно вредит**
- Парадокс: FinBERT IC значения выше, но PnL результат в 6× хуже

**⚠️ Критические замечания из внешнего ревью**:

1. **IC scan для market-level фичей некорректен**: `market_news_sentiment_24h` одинаков для всех ~35 монет в один timestamp. IC считается по всем (coin,time) строкам → 35× дублирование → **artificial inflation IC и ложная стабильность**. Правильно: time-series IC (corr market_feature[t] vs cs_mean_return[t+1]) или corr в рамках одного timestamp.

2. **Availability/lag mismatch**: Новости агрегируются в hour=t и используются для сигнала на ребалансе t → возможно lookahead bias (новости, появившиеся после момента принятия решения). Нужен lag-shift тест: сдвиг news features на +1h/+2h.

3. **C = B**: Political features отсутствуют в данных → C тестирует то же что B.

4. **Предложение**: перед полным закрытием темы — одна быстрая итерация: убрать market-level news из panel модели, оставить только per-coin divergence/coverage. Если и это ≤ 0 — закрыть окончательно.

**VERDICT: ❌ NEGATIVE** — FinBERT features **активно вредят** модели (Sharpe −1.18..−2.30). IC scan для market-level фичей методологически ошибочен (duplicate rows). Тема news-as-panel-features закрыта, но lag-shift + per-coin-only тест может быть полезен для финальной точки.

**Файл**: `_research_r123_news_sentiment.py` (reused for R125), `_run_r125_on_vm.sh`, `fetch_crypto_news.py --scorer finbert`

## R124b — TAKER_ONLY Baseline (ответ на замечание ревью) ✅ COMPLETED (15 апреля 2026)

**Гипотеза**: Ревьюер спросил: "а что если продакшн делает taker (market) orders?" Нужен baseline.

**Dual Sharpe methodology**: `sharpe_active` (только периоды с позицией) + `sharpe_full` (все периоды).

| Scenario | Sh_Active | ΔAct | Sh_Full | ΔFull | Ret% | DD% | Cost% |
|---|---|---|---|---|---|---|---|
| S6_current (maker-first) | 3.691 | +0.000 | 2.831 | +0.000 | 157.3% | -10.9% | 23.6% |
| TAKER_ONLY (maker_pct=0) | 3.464 | -0.227 | 2.612 | -0.219 | 138.3% | -11.5% | 29.6% |
| TAKER_REF20 (taker+ref) | 3.593 | -0.098 | 2.733 | -0.098 | 148.7% | -11.1% | 26.3% |
| REF20 (maker+ref) | 3.780 | +0.089 | 2.914 | +0.083 | 164.9% | -10.9% | 21.4% |

Per-trade costs: S6=T1:2.4bp T2:5.5bp T3:10bp; TAKER_ONLY=T1:6bp T2:7bp T3:10bp

**Ключевой вывод**: Maker-first execution benefit = **+0.227 Sharpe_active / +0.219 Sharpe_full**.  
Это наибольшее бесплатное улучшение. Если прод сейчас taker-only → maker-first приоритет #1.  
REF20 (maker + 20% referral) = 3.780 — верхняя планка при текущей модели.

**Файл**: `_research_r124b_taker_baseline.py`, результаты: `results/r124b_taker_baseline.json`

## R126 — Review Fixes: IC Scan + Lag Test + Per-Coin Only ❌ NEGATIVE (15 апреля 2026)

**Ревью нашло 3 методологических замечания к R123/R124/R125. Все трое исправлены и протестированы.**

### Fix 1: IC scan для market-level features

**Проблема (ревью)**: Рыночные фичи дублируются 35× (по числу монет) → IC рассчитывается неверно.  
**Исправление**: Time-series IC (collapse to 1 obs per timestamp, Spearman с cross-sectional mean return).

| Feature | Old IC | New IC | Δ | Pass? |
|---|---|---|---|---|
| nx_mkt_sent_x_vol | -0.0554 | -0.0607 | -0.0053 | ✓ both |
| market_news_sentiment_24h | -0.0530 | -0.0594 | -0.0064 | ✓ both |
| nx_mkt_count_zscore | -0.0246 | -0.0266 | -0.0020 | ✓ both |
| nx_sent_divergence (per-coin) | +0.0436 | +0.0436 | 0 | ✓ both |
| nx_mkt_sent_x_ret12 | -0.0061 | -0.0000 | +0.0061 | ✗ both |

**Результат**: Те же 4 фичи проходят IC-фильтр в обоих методах. Time-series IC фактически **чуть сильнее** (не слабее, как ожидал ревьюер). Гипотеза об артифициальной инфляции IC за счёт дублирования **НЕ подтвердилась** в данных.

### Fix 2: Lag-shift test (lookahead bias check)

**Проблема (ревью)**: Возможен lookahead — новости, пришедшие после свечи, попадают в фичи той же свечи.  
**Тест**: Обучение с per-coin news features, shifted +1h и +2h внутри каждого символа.

| Experiment | Gross Sharpe | Net Sharpe | Δ vs Baseline | Вывод |
|---|---|---|---|---|
| A (baseline, no news) | 3.703 | 2.831 | 0 | Reference |
| LAG1 (+1h shift, per-coin) | 3.361 | 2.404 | -0.342 | Сигнал слабеет |
| LAG2 (+2h shift, per-coin) | 1.807 | 0.944 | -1.896 | Сигнал рушится |

**Bootstrap**: LAG1 p_improve=0.239, LAG2 p_improve=0.001  
**Вывод**: Сигнал резко деградирует при сдвиге → per-coin новости действительно имеют lookahead bias. Даже LAG1 хуже baseline. Направление **ЗАКРЫТО**.

### Fix 3: Per-coin only experiments (без рыночных фичей)

**Проблема (ревью)**: Нада проверить per-coin фичи изолированно, без market-level дублирования.  
**Тест**: Только per-coin фичи (8 штук), затем + `nx_sent_divergence` (9 штук).

| Experiment | Gross Sharpe | Net Sharpe | Δ vs Baseline | Bootstrap p |
|---|---|---|---|---|
| A (31 базовых, без новостей) | 3.703 | 2.831 | 0 | — |
| G (per-coin 8 фичей) | 2.649 | 1.782 | -1.054 | p=0.06 |
| H (per-coin + divergence 9 фичей) | 3.140 | 2.265 | -0.563 | p=0.214 |
| I (4 IC-passing фичей из fixed scan) | 1.957 | 1.104 | -1.746 | p=0.011 |

**Bootstrap mean Δ Sharpe**: G=-1.056, H=-0.564, I=-1.727  
**Вывод**: Все варианты хуже baseline. Per-coin новостные фичи достоверно **вредят** модели (G: p=6%, I: p=1.1%). H немного лучше за счёт divergence, но всё равно -0.56 Sharpe. Направление **ЗАКРЫТО**.

**ИТОГОВЫЙ ВЕРДИКТ R126**: ❌ NEGATIVE на всех трёх проверках.
- IC scan bug исправлен: те же фичи проходят, тот же вывод
- Lookahead bias подтверждён для per-coin features: +1h → -0.34 Sharpe, +2h → -1.90 Sharpe  
- Per-coin news features изолированно — все варианты хуже baseline  
- **News sentiment (VADER, VADER-based interactions) навсегда ЗАКРЫТО.**

**Файлы**: `_research_r126_review_fixes.py`, `_research_r124b_taker_baseline.py`, результаты: `results/r126_review_fixes.json`, `results/r124b_taker_baseline.json`

### Закрытые направления (финальная карта, после R126)

```
CLOSED: Features (31), Models (LGB+XGB), Portfolio base (4L/2S),
        Trend filter (binary 0.9), Churn (min_off=2), Ensemble,
        Rebalance (12h), Neutralization, Spillover, FMP,
        Continuous sizing, Expanding K (6L/3S worse),
        8h rebalance, Universe expansion (R115),
        Dynamic K — R117c FAIL (режим-зависимо),
        Risk-off BTC directional — R122 INCONCLUSIVE (+0.057 noise),
        News sentiment VADER — R123 NEGATIVE (все варианты хуже baseline),
        News sentiment FinBERT — R125 NEGATIVE (хуже VADER),
        News sentiment (review fixes) — R126 NEGATIVE:
          - IC scan bug: те же 4 фичи, тот же вывод
          - Lookahead bias: LAG1=-0.34, LAG2=-1.90 → per-coin новости нечистые
          - Per-coin only: G=-1.05, H=-0.56, I=-1.75 → все варианты хуже

OPEN:
  1. CryptoQuant exchange flows (external data, untested)
  2. Extend maker-first execution to Tier2/Tier3 (R121: +0.32 Sharpe potential)
  3. OKX referral cashback 20% — R124 POSITIVE (+0.068 Sharpe) → DEPLOY now
  4. Maker-first execution (если прод taker-only) — R124b: worth +0.227 Sharpe_active → PRIORITY #1
  5. LLM-filtered news (GPT/Claude API) — возможно чисто, но lookahead риск тот же
```

---

*Конец документа. Используй как полный контекст для любой AI-модели.*
