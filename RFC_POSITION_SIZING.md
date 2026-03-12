# RFC: Оптимизация Position Sizing (размера ставки)

**Автор:** AI Research  
**Дата:** 2026-03-12  
**Статус:** Draft  
**Приоритет:** Высокий — напрямую влияет на P&L и risk/reward

---

## 1. Проблема

Текущая система нестабильно определяет размер позиций: одна ставка = $400, другая = $2800 (при 3x плече). Пользователь не понимает логику, почему так происходит, и хочет более осмысленный подход к position sizing.

### 1.1 Как работает position sizing СЕЙЧАС (production)

**Файл:** `run_trading.py`, функция `construct_portfolio()` (строка 699).

**Конфиг на проде (systemd):**
```
--mode paper --loop --capital 5000 --leverage 3
```

**DEFAULT_RISK в коде:**
```python
DEFAULT_RISK = {
    'n_long': 10,       # 10 лонгов
    'n_short': 10,      # 10 шортов  
    'vol_target': 0.008, # 0.8% vol target
    'vol_lookback': 48,
    'kelly_frac': 0.3,   # 30% Kelly
    'dd_stop': -0.15,
    'dd_resume': -0.06,
    'confidence_threshold': 0.0,
}
```

**Формула (равные веса):**
```
vol_scale = clip(vol_target / realized_vol, 0.1, 3.0)
effective_kelly = kelly_frac × vol_scale
total_alloc = capital × effective_kelly × leverage
per_position = total_alloc / total_positions
per_position = min(per_position, capital × leverage × 0.15)
```

**Конкретные числа:**
```
capital = $5,000
leverage = 3x
kelly_frac = 0.3
total_positions = 20 (10L + 10S)

Базовый: total_alloc = 5000 × 0.3 × vol_scale × 3 = 4500 × vol_scale
per_position = 4500 × vol_scale / 20 = $225 × vol_scale

При vol_scale = 1.0: per_position = $225
При vol_scale = 0.1: per_position = $22.5 (минимальная)
При vol_scale = 3.0: per_position = $675 (максимальная)
Cap: capital × leverage × 0.15 = $2,250
```

### 1.2 Откуда разброс $400–$2800?

Главный виновник — **`vol_scale`** (диапазон 0.1–3.0, т.е. **30-кратный** разброс):

| Рыночный режим | Realized Vol | vol_scale | per_position (1 из 20) |
|----------------|-------------|-----------|----------------------|
| Низкая волатильность | 0.003 | 2.67 | $600 |
| Средняя | 0.008 | 1.00 | $225 |
| Высокая волатильность | 0.025 | 0.32 | $72 |
| Крайне низкая | 0.001 | 3.00 (cap) | $675 |

Если `optimal_config.json` перезаписывает DEFAULT_RISK (например, kelly_frac=1.0, n_long=5, n_short=5), тогда:
```
total_alloc = 5000 × 1.0 × vol_scale × 3 = 15000 × vol_scale
per_position = 15000 × vol_scale / 10

vol_scale = 1.0: $1,500
vol_scale = 0.3: $450
vol_scale = 1.8: $2,700 (→ cap $2,250)
```

**Это объясняет диапазон $400–$2800.**

### 1.3 Ключевые проблемы текущего подхода

1. **Equal weight в production** — все позиции получают одинаковый $ вне зависимости от силы сигнала. В симуляторе (`run_fast_sim.py`) есть edge-boost, но в trading bot его **нет**.

2. **vol_scale диапазон слишком широкий (0.1–3.0)** — создаёт 30x разброс. В спокойном рынке позиции раздуваются до 3x, а при первом stress-событии происходит резкое сокращение.

3. **Kelly fraction неоправдан** — Kelly criterion предполагает стационарный edge, а в крипте edge нестационарен. Fractional Kelly = 0.3 — это уже хорошо (консервативно), но vol_scale сверху может сделать effective_kelly = 0.3 × 3.0 = 0.9 — почти full Kelly.

4. **Нет различия между позициями** — монета с score +0.95 и монета с score +0.51 (оба в top-5) получают одинаковый $. Score +0.95 = сильный сигнал, +0.51 = едва попавший.

5. **Нет учёта волатильности отдельных монет** — $1500 в BTC ≠ $1500 в DOGE по риску. DOGE-позиция несёт в 3-5x больше volatility risk.

---

## 2. Исследование: методы position sizing

### 2.1 Классические методы

#### A. Fixed Fractional (самый простой)
```
position_size = equity × fixed_pct / n_positions
```
- **Плюсы:** Простой, предсказуемый, масштабируется с equity
- **Минусы:** Не учитывает ни силу сигнала, ни волатильность
- **Годится как:** Текущий baseline (по сути, это и есть то, что сейчас работает без vol_scale)

#### B. Kelly Criterion / Fractional Kelly
```
f* = (p × b - q) / b
f_actual = fraction × f*

где p = win_rate, b = avg_win/avg_loss, q = 1-p
```
- **Плюсы:** Теоретически оптимален для максимизации долгосрочного роста
- **Минусы:** Требует точную оценку p и b, которые нестационарны в крипте. При ошибке в оценке edge → сильный overshoot → drawdown
- **Статус:** Частично используется (kelly_frac=0.3 = 30% Kelly). **Рекомендация из AI-ревью:** использовать как верхний cap, но не как основной sizing-метод

#### C. Volatility Targeting (уже частично реализован)
```
position_vol_target = portfolio_vol_target / n_positions
position_size = position_vol_target / coin_volatility
```
- **Плюсы:** Выравнивает risk contribution каждой позиции, стабилизирует equity curve
- **Минусы:** В спокойном рынке раздувает позиции перед штормом (vol clustering)
- **Статус:** В коде есть, но работает на уровне портфеля (не per-coin). С диапазоном 0.1–3.0 — слишком агрессивно.

#### D. Risk Parity (Inverse Volatility Weighting)
```
weight_i = (1/σ_i) / Σ(1/σ_j)
position_size_i = total_alloc × weight_i
```
- **Плюсы:** Каждая монета вносит ~одинаковый risk in portfolio. BTC получает больше $, DOGE меньше — уравнивается по риску.
- **Минусы:** Не учитывает силу сигнала. BTC с edge=0.01 получит больше чем SHIB с edge=0.10.
- **В коде:** Есть в `run_fast_sim.py` как `--vol-size` опция.

#### E. Mean-Variance Optimization (Markowitz)
```
w* = argmax(w'μ - λ/2 × w'Σw)
```
- **Плюсы:** Теоретически оптимален для risk-adjusted return
- **Минусы:** Крайне чувствителен к оценке μ (expected returns) и Σ (covariance). В крипте с 50 монетами → матрица 50×50, estimation error → экстремальные веса. Требует regularization (shrinkage, constraints).
- **Рекомендация:** Не использовать в чистом виде. Дорого, нестабильно, overfit-prone.

#### F. Hierarchical Risk Parity (HRP, López de Prado 2016)
```
1. Hierarchical clustering по корреляционной матрице
2. Quasi-diagonalization
3. Recursive bisection для аллокации
```
- **Плюсы:** Не требует инверсии ковариационной матрицы (в отличие от Markowitz), более стабильные веса
- **Минусы:** Сложная реализация, не учитывает expected returns (только risk), для 10 позиций overhead vs inverse-vol может быть минимальным
- **Рекомендация:** Возможно для полного портфеля (50 монет), для 10 позиций — overkill.

### 2.2 Сигнал-зависимые методы (то что нам нужно)

#### G. Edge-Proportional Sizing (уже в симуляторе)
```
edge_i = |score_i - median|
weight_i = 1 + min(edge_i / P75_edge, 3.0)
weight_i *= confidence_i  # seed agreement
position_size_i = total_alloc × (weight_i / Σweight_j)
Cap: weight_i ≤ confidence_i (0.15–0.40)
```
- **Плюсы:** Монеты с более сильным сигналом получают больше капитала. Sharpe 2.79→5.93 в бэктесте (+113%).
- **Минусы:** P75_edge калибруется на первых 30 шагов → может не отражать текущий режим.
- **Статус:** ✅ Реализован в `run_fast_sim.py`, ❌ НЕ в `run_trading.py`.
- **⚠️ Это первое, что нужно сделать — перенести в production.**

#### H. Edge × Inverse-Vol (гибрид G + D)
```
weight_i = edge_boost_i × (1/σ_i)
normalize: weight_i /= Σweight_j
```
- **Плюсы:** Сочетает силу сигнала + risk equalization. Сильный сигнал на стабильной монете → максимальная ставка. Сильный сигнал на волатильной → умеренная ставка.
- **В коде:** Есть как `--edge-boost --vol-size` в `run_fast_sim.py`.
- **Рекомендация:** Это наш лучший кандидат для production sizing.

#### I. Confidence-Scaled Sizing
```
confidence_i = 1 / (1 + std(predictions_15_models_i))
weight_i ∝ confidence_i
```
- **Плюсы:** Когда 15 моделей согласны → бо́льшая ставка. Когда расходятся → меньше.
- **Минусы:** Seed agreement ≠ уверенность (результат exp10: WR 34% когда сиды согласны). Но confidence weighting всё равно дал Sharpe 2.27→2.48 (+9%).
- **Статус:** ✅ В `run_fast_sim.py`, ❌ НЕ в `run_trading.py`.

### 2.3 ML-based Position Sizing (обучить отдельную модель)

#### J. Supervised Sizing Model
**Идея:** обучить модель, которая предсказывает не направление, а **оптимальный размер позиции**.

**Варианты target:**

| Target | Описание | Формула |
|--------|----------|---------|
| `abs_return_12h` | Абсолютный return | `|ret_coin_12h|` |
| `signed_edge` | Предсказанный edge × реальный знак | `sign(predicted) × actual_return` |
| `sharpe_per_trade` | Sharpe отдельной сделки | `ret / vol_12h` |
| `optimal_f` | Оптимальная доля | Решение из Kelly по исторической выборке |

**Фичи для sizing model:**
- Все 127+ стандартных фичей
- **Meta-фичи:** score spread, model agreement, recent WR, DD depth, regime
- **Volatility фичи:** per-coin vol, cross-coin dispersion, VIX-proxy
- **Market structure:** OI change, taker imbalance, funding surprise, L/S ratio

**Архитектура:**
```
Sizing model: маленький GBDT (LightGBM, 50 trees)
Input: [standard_features, meta_features, vol_features]
Output: scale_factor ∈ (0.2, 2.0) — множитель к базовому размеру
Loss: MSE(predicted_scale, optimal_scale)
```

**Проблемы:**
1. **Label engineering сложен** — нет однозначного "правильного размера". Нужно определить oracle policy.
2. **Lookahead bias** — optimal_f определяется по будущему return, тренировать нужно на walk-forward OOF.
3. **Sample size** — ~730 шагов на 365 дней (12h rebal), 10 позиций = 7300 сэмплов. Мало для надёжного обучения.
4. **Overfitting risk** — модель может научиться "ставить больше когда рынок растёт" = leverage momentum, что убивает в reversal.

**Вердикт:** Можно попробовать, но **не как первый шаг**. Сначала нужно внедрить алгоритмические методы (G+H+I), а ML-sizing — как эксперимент позже.

#### K. Reinforcement Learning для sizing
**Идея:** RL-агент, который учится оптимальному allocation через reward = PnL.

**Преимущества над supervised:**
- Не нужен label engineering — reward = итоговый P&L
- Учитывает последовательность решений (state-dependent)
- Может выучить нелинейные зависимости (увеличивать ставку после серии маленьких побед)

**Проблемы:**
- Требует ~100K+ шагов для сходимости, у нас ~730/год
- Нестационарная среда = постоянный concept drift
- Крайне сложно отлаживать
- На маленьком датасете → catastrophic overfitting

**Вердикт:** ❌ Не рекомендуется при текущем объёме данных и частоте ребалансировки.

---

## 3. Рекомендуемый план действий

### Phase 1: Quick Wins (1 день, без ретренинга) ← НАЧАТЬ СЕЙЧАС

#### 1A. Перенести edge-boost sizing в production
**Что:** Скопировать логику `compute_weights()` из `run_fast_sim.py` в `construct_portfolio()` в `run_trading.py`.

**Ожидаемый эффект:** Sharpe +113% (по бэктесту 2.79→5.93).

**Изменение в `construct_portfolio()`:**
```python
# Вместо:
per_position = total_alloc / total_positions  # equal weight

# Сделать:
# 1. Calibrate edge_p75 (можно закешировать при первом запуске)
# 2. Для каждой позиции:
#    boost = 1 + min(|score - median| / edge_p75, 3.0)
#    weight = boost × confidence
# 3. Normalize → dollar allocation
```

#### 1B. Перенести inverse-vol weighting  
**Что:** Для каждой монеты рассчитать 24h realized vol, аллоцировать обратно пропорционально.

**Ожидаемый эффект:** Снижение MaxDD на 15-30% (risk parity effect).

#### 1C. Ужать vol_scale диапазон
**Что:** Изменить `clip(vol_target / realized_vol, 0.1, 3.0)` → `clip(..., 0.5, 1.5)`.

**Ожидаемый эффект:** Позиции станут стабильнее (разброс 3x вместо 30x). В тихом рынке не раздувает перед crash.

### Phase 2: Meta-Risk Integration (2-3 дня)

#### 2A. Перенести meta-risk scaler в production
**Что:** Портировать `compute_meta_risk()` из `run_fast_sim.py`.

**5 сигналов:**
1. Model agreement (mean confidence top/bottom)
2. Score spread (P90-P10 scores)
3. Recent performance (EMA win rate, 40-60 шагов — НЕ 10 шагов)
4. DD depth (глубже DD → меньше risk)
5. Regime (bull/bear → разный base)

**Результат:** Gross exposure scale × 0.3–1.5. Sharpe +57% по бэктесту.

#### 2B. Stress cap по деривативам
**Что:** Если btc_vol_24h > 2σ OR funding_surprise extreme → gross ≤ 0.5x.

#### 2C. Asymmetric vol targeting ("только вниз")
```python
vol_scale = min(1.0, vol_target / realized_vol)
```
Никогда не увеличивать позиции выше базового, только уменьшать в стрессе.

### Phase 3: ML Sizing Experiment (1-2 недели)

#### 3A. Position Sizing как регрессия
```
Label: optimal_f = sign(predicted) × actual_ret / vol × scaling
Features: [model_score, confidence, vol_24h, OI_change, taker_imbalance, 
           FNG, regime, recent_wr, dd_depth, funding_rate]
Model: LightGBM (50 trees, lr=0.05, max_depth=4)
Validation: Walk-forward OOF, та же сетка что для основных моделей
```

**Критерий успеха:** DDStop Sharpe > Phase2 на walk-forward test windows.

#### 3B. Signal-Adaptive Kelly
```
kelly_i = f(score_spread, confidence, vol_regime)
kelly_portfolio = mean(kelly_i) × fractional_multiplier
```
Идея: Kelly fraction не фиксирован, а адаптируется под текущий рыночный режим. В высоковолатильном стресс-режиме — kelly↓, в тихом тренде — kelly↑.

### Phase 4: Advanced (опционально)

#### 4A. Hierarchical Risk Parity (HRP)
- Для полного портфеля из 10 позиций
- Учитывает корреляции между монетами (BTC-ETH коррелированы → суммарно меньше)

#### 4B. Online learning sizing
- Bayesian updating: prior = calibrated edge, posterior updated каждый ребаланс
- Thompson Sampling для exploration/exploitation в sizing

---

## 4. Конкретный план имплементации Phase 1

### 4.1 Изменения в `run_trading.py`

**Новая функция `compute_position_weights()`:**

```python
def compute_position_weights(signals, risk_cfg, edge_p75=None):
    """
    Compute per-position dollar weights using:
    1. Edge-proportional boost (weight ∝ 1 + edge/P75)
    2. Confidence weighting (× model agreement)
    3. Inverse-volatility scaling (÷ coin vol)
    4. Concentration cap (per position ≤ confidence-based max)
    
    Returns dict: symbol → weight (normalized to sum=1 per side)
    """
    scores = signals['score'].values
    symbols = signals['symbol'].values
    median_score = np.median(scores)
    
    # Edge P75 calibration (can be cached)
    abs_edges = np.abs(scores - median_score)
    if edge_p75 is None or edge_p75 == 0:
        edge_p75 = np.percentile(abs_edges, 75)
    
    n_long = risk_cfg['n_long']
    n_short = risk_cfg['n_short']
    
    sorted_idx = np.argsort(-scores)
    long_idx = sorted_idx[:n_long]
    short_idx = sorted_idx[-n_short:]
    
    def _weights(indices, is_long=True):
        ws = {}
        for i in indices:
            sym = symbols[i]
            edge = abs_edges[i]
            # Edge boost: 1.0 + min(edge/P75, 3.0)
            boost = 1.0 + min(edge / (edge_p75 + 1e-10), 3.0)
            # Confidence (if available, else 1.0)
            # For now use 1.0; will be overridden when confidence data available
            conf = 1.0
            ws[sym] = boost * conf
        # Normalize
        total = sum(ws.values())
        if total > 0:
            ws = {s: w/total for s, w in ws.items()}
        return ws
    
    weight_L = _weights(long_idx, is_long=True)
    weight_S = _weights(short_idx, is_long=False)
    
    return weight_L, weight_S
```

**Изменения в `construct_portfolio()`:**

```python
# Вместо equal weight:
weight_L, weight_S = compute_position_weights(signals, risk_cfg)

half_alloc = total_alloc / 2

for sym, w in weight_L.items():
    usd = round(half_alloc * w, 2)
    if usd < 5: continue
    positions.append({'symbol': sym, 'side': 'long', 'usd': usd, ...})

for sym, w in weight_S.items():
    usd = round(half_alloc * w, 2)
    if usd < 5: continue
    positions.append({'symbol': sym, 'side': 'short', 'usd': usd, ...})
```

### 4.2 vol_scale cap

```python
# Было:
vol_scale = np.clip(risk_cfg['vol_target'] / realized_vol, 0.1, 3.0)

# Стало (asymmetric: только вниз):
vol_scale = np.clip(risk_cfg['vol_target'] / realized_vol, 0.3, 1.2)
```

Это значит:
- При высокой волатильности: уменьшаем до 0.3x (было 0.1x — слишком агрессивное сжатие)
- При низкой волатильности: **не раздуваем** выше 1.2x (было 3.0x — слишком агрессивное увеличение)

### 4.3 Пример: как изменятся позиции

**Текущий production (equal weight):**
| Позиция | Score | USD |
|---------|-------|-----|
| BTC long | +0.92 | $1,500 |
| ETH long | +0.78 | $1,500 |
| SOL long | +0.65 | $1,500 |
| DOGE long | +0.53 | $1,500 |
| XRP long | +0.51 | $1,500 |

**С edge-boost + inverse-vol:**
| Позиция | Score | Edge | Boost | Vol | Adj Weight | USD |
|---------|-------|------|-------|-----|-----------|-----|
| BTC long | +0.92 | 0.42 | 2.68 | 2.1% | 30% | $2,235 |
| ETH long | +0.78 | 0.28 | 2.12 | 3.0% | 17% | $1,260 |
| SOL long | +0.65 | 0.15 | 1.60 | 4.5% | 8% | $630 |
| DOGE long | +0.53 | 0.03 | 1.12 | 6.0% | 4% | $340 |
| XRP long | +0.51 | 0.01 | 1.04 | 3.5% | 7% | $535 |

→ Сильный сигнал + низкая вола = $2,235 (BTC)  
→ Слабый сигнал + высокая вола = $340 (DOGE)  
→ Это **осмысленное** распределение капитала.

---

## 5. Оценка рисков

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Edge-boost overfits к P75 калибрации | Medium | High | Обновлять P75 каждые N циклов (rolling) |
| Concentration risk (BTC получает 30%+) | Medium | Medium | Hard cap 25% per position |
| vol_scale cap → упущенная прибыль в тихом рынке | Low | Low | 1.2x всё ещё даёт небольшой boost |
| Meta-risk scale = 1.5x перед crash | Medium | High | Asymmetric: max up = 1.3x, max down = 0.3x |
| ML sizing model overfits | High | Medium | Walk-forward validation obligatory |

---

## 6. Метрики для оценки

| Метрика | Baseline (current) | Target (Phase 1) | Target (Phase 2) |
|---------|-------------------|-------------------|-------------------|
| DDStop Sharpe | — (нет edge-boost в prod) | +50-100% vs equal weight | +100-150% |
| MaxDD | текущий | ≤ текущий | ≤ 80% текущего |
| Разброс sizes | 30x (0.1–3.0 vol_scale) | ≤ 4x | ≤ 4x |
| Win Rate | ~61% | ≥ 61% | ≥ 63% |
| Avg trade size stability | σ/μ > 1.0 | σ/μ < 0.5 | σ/μ < 0.4 |

---

## 7. Checklist для имплементации

### Phase 1 (Quick Wins)
- [ ] Перенести `compute_weights()` из `run_fast_sim.py` в `run_trading.py`
- [ ] Добавить edge-boost в `construct_portfolio()`
- [ ] Добавить inverse-vol weighting (coin vol 24h calculation)
- [ ] Сузить vol_scale clip до (0.3, 1.2)
- [ ] Добавить hard cap 25% per position
- [ ] Backtesting: сравнить equal vs edge-boost на 365d fast sim
- [ ] Деплой на VPS (paper trading, 1-2 недели мониторинга)

### Phase 2 (Meta-Risk)
- [ ] Портировать `compute_meta_risk()` в production
- [ ] Исправить recent WR (EMA 40-60 шагов вместо rolling 10)
- [ ] Добавить stress cap по derivatives
- [ ] Asymmetric vol targeting (только вниз)
- [ ] A/B тест: 365d sim с/без meta-risk

### Phase 3 (ML Sizing)
- [ ] Определить oracle policy для label engineering
- [ ] Собрать meta-features dataset (walk-forward OOF)
- [ ] Обучить LightGBM sizing model
- [ ] Walk-forward validation на 3 окнах
- [ ] Сравнить с Phase 2 по DDStop Sharpe

---

## 8. Резюме

**Главная находка:** Production бот использует equal-weight sizing, в то время как симулятор уже имеет edge-boost (+113% Sharpe), confidence weighting (+9% Sharpe), и meta-risk (+57% Sharpe). Эти проверенные механизмы просто не перенесены в production.

**Рекомендация:**
1. **Немедленно (Phase 1):** перенести edge-boost + inverse-vol из симулятора в production, сузить vol_scale. Это бесплатный прирост, уже проверенный в бэктесте.
2. **Затем (Phase 2):** интегрировать meta-risk scaler, который адаптирует gross exposure по 5 сигналам.
3. **Позже (Phase 3):** попробовать обучить ML-модель для sizing, но только после того как исчерпаны алгоритмические методы.

**Обучать ML для sizing — можно, но преждевременно.** Сначала нужно закрыть разрыв production ↔ simulator, и только потом усложнять.
