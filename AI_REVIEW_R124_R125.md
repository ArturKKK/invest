# AI Review: R124 (OKX Fee Optimization) + R125 (FinBERT News Sentiment)

## Для ревьюеров: контекст системы

Крипто ML торговая система. 50 монет, USDT-M perpetual фьючерсы OKX.
- **Модель**: LightGBM + XGBoost ensemble, 31 feature, walk-forward на 3 окнах (W1/W2/W3)
- **Стратегия**: Long 4 / Short 2 монеты, ребаланс каждые 12 часов, risk-off при сильных трендах
- **Champion (R114b)**: Sharpe 2.831 net (после комиссий S6 prod_blended)
- **Baseline**: R114b с S6 prod_blended cost model = Sharpe 2.831 (проверено многократно)

---

## R124 — OKX Fee Optimization

### Идея

Комиссии съедают ~23.6% от gross прибыли. Если снизить комиссии — Sharpe растёт автоматически без изменения модели. Вопрос: **сколько именно** Sharpe можно получить от реалистичных сценариев снижения комиссий на OKX?

### Реализация

**Подход**: Однократно обучить baseline ensemble на 31 champion feature, затем прогнать 8 сценариев стоимости через один и тот же набор predictions. Меняется только cost function в симуляторе.

**Код cost function (параметрическая)**:
```python
def _make_cost_fn(maker, taker, spread_tier1=0.0001, spread_tier2=0.0002,
                  spread_tier3=0.0005, maker_pct_t1=0.90, maker_pct_t2=0.50):
    def cost_fn(sym):
        if sym in TIER1_SYMS:  # BTC, ETH, SOL, BNB, XRP
            maker_cost = maker  # maker fee, ~0 spread (you are the spread)
            taker_cost = taker + spread_tier1  # taker fee + spread
            return maker_pct_t1 * maker_cost + (1 - maker_pct_t1) * taker_cost
        elif sym in TIER3_SYMS:  # SAND, LDO, INJ, APT, ARB, GALA, FTM, MATIC
            return taker + spread_tier3  # pure market orders
        else:  # Tier2 — mid-cap
            maker_like = maker + spread_tier2
            taker_cost = taker + spread_tier2
            return maker_pct_t2 * maker_like + (1 - maker_pct_t2) * taker_cost
    return cost_fn
```

**Формула one-way cost per trade для Tier1 (baseline S6)**:
$$C_{T1} = 0.90 \times 0.0002 + 0.10 \times (0.0005 + 0.0001) = 0.00024 = 2.4 \text{ bps}$$

**Формула для referral cashback (e.g. REF20)**:
$$C_{T1}^{ref20} = 0.90 \times (0.0002 \times 0.80) + 0.10 \times (0.0005 \times 0.80 + 0.0001) = 0.000194 = 1.94 \text{ bps}$$

**Cost deduction в симуляторе** (из `simulate_r121`):

```python
# При каждом ребалансе:
avg_weight = 1.0 / total_positions   # total_positions = n_long + n_short (обычно 6)
turnover_cost = sum(cost_fn(sym) * avg_weight for sym in new_opened)  # открытые
turnover_cost += sum(cost_fn(sym) * avg_weight for sym in closed)     # закрытые
holding_cost = funding_per_12h * (rebal_hours / 12)                   # ~1.2 bps/12h
total_cost = turnover_cost + holding_cost
net_ret = gross_ret - total_cost
```

**Execution delay penalty**: random noise $\mathcal{N}(0, 0.0003)$ на gross_ret каждый период (моделирует 5-мин задержку исполнения).

**8 сценариев** (все используют funding=1.2bp/12h, exec_delay=3bp):

| Сценарий | Maker fee | Taker fee | Maker% T1 | Maker% T2 | Описание |
|---|---|---|---|---|---|
| S6_current | 2bp | 5bp | 90% | 50% | Текущий прод |
| REF10 | 1.8bp | 4.5bp | 90% | 50% | 10% referral cashback |
| REF20 | 1.6bp | 4bp | 90% | 50% | 20% referral cashback |
| REF30 | 1.4bp | 3.5bp | 90% | 50% | 30% referral cashback (max) |
| MAKER_OPT | 2bp | 5bp | 95% | 70% | Улучшенное исполнение |
| REF20_MAKER | 1.6bp | 4bp | 95% | 70% | 20% referral + улучшенное исполнение |
| VIP1 | 1.8bp | 4.5bp | 90% | 50% | VIP1 tier (>$100K) |
| VIP1_MAKER | 1.8bp | 4.5bp | 95% | 70% | VIP1 + улучшенное исполнение |

**Sharpe computation в R124** (из `compute_metrics`):
```python
trading = port_df[~port_df["risk_off"]]
trading_rets = trading["net_ret"].values
ann = np.sqrt(365 * 24 / 12)  # √(730) ≈ 27.02 (12h periods → annual)
sharpe = mean(trading_rets) / std(trading_rets) * ann
```

### Результаты R124

| Сценарий | Net Sharpe* | Δ Sharpe | Return% | DD% | Cost% |
|---|---|---|---|---|---|
| S6_current | 3.691 | — | 157.3% | -10.9% | 23.6% |
| REF10 | 3.736 | +0.035 | 161.1% | -10.9% | 22.5% |
| REF20 | 3.780 | +0.068 | 164.9% | -10.9% | 21.4% |
| REF30 | 3.825 | +0.103 | 168.8% | -10.9% | 20.3% |
| MAKER_OPT | 3.746 | +0.042 | 162.0% | -10.9% | 22.1% |
| REF20_MAKER | 3.824 | +0.102 | 168.7% | -10.9% | 20.3% |
| VIP1 | 3.736 | +0.035 | 161.1% | -10.9% | 22.5% |
| VIP1_MAKER | 3.785 | +0.072 | 165.4% | -10.9% | 21.3% |

**⚠️ ВАЖНАЯ ЗАМЕТКА О SHARPE**:

`compute_metrics` считает Sharpe **только по торговым периодам** (исключая risk_off), что даёт 3.691 вместо 2.831 для baseline. Оригинальный `sharpe()` из R68 считает по всем периодам (включая risk_off с ret=0). Масштабный коэффициент: 2.831 / 3.691 = 0.767. **Дельты (Δ) между сценариями корректны** — одна и та же формула применяется ко всем.

**Вывод R124**: Referral 20% cashback = +0.068 Sharpe (или ~+0.052 в масштабе full-method), бесплатно. DD не меняется. REF20+MAKER_OPT = +0.102, но требует доработки execution engine.

---

## R125 — FinBERT News Sentiment

### Идея

R123 показал что VADER sentiment (rule-based, ~60% accuracy на финтексте) не добавляет value к модели. Гипотеза: **более качественный scorer** (FinBERT, ~87% accuracy) может извлечь полезный сигнал из тех же 954K новостей.

### Реализация

**Step 1: Пере-scoring всех новостей с FinBERT**

Модель: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) — BERT pre-trained на Financial PhraseBank (4846 финансовых предложений, размеченных 16 экспертами).

Три класса: positive, negative, neutral.

**Формула scoring для каждого headline**:
$$\text{score} = \text{clip}\left(\sum_{c \in \{pos, neg, neu\}} w_c \cdot P(c | \text{title}), -1, +1\right)$$

где $w_{pos}=+1, w_{neg}=-1, w_{neu}=0$, и $P(c|\text{title})$ — softmax вероятности из FinBERT.

```python
# label_map: {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
# Пример: P(pos)=0.6, P(neg)=0.1, P(neu)=0.3
# score = 1.0*0.6 + (-1.0)*0.1 + 0.0*0.3 = +0.5

for res in results:  # results = pipeline(texts, return_all_scores=True)
    weighted = sum(label_map.get(r["label"], 0.0) * r["score"] for r in res)
    scores.append(np.clip(weighted, -1.0, 1.0))
```

**Параметры inference**:
- GPU: NVIDIA H100 80GB HBM3
- torch 2.5.1+cu121, transformers 4.44.2
- float16 (half precision на GPU)
- batch_size=64, max_length=512
- mega-batches по 10K items для progress tracking
- 954,551 новостей за ~5 минут

**Распределение FinBERT scores**:
- Positive (>0.1): 427,918 (44.8%)
- Negative (<-0.1): 234,743 (24.6%)
- Neutral: 291,890 (30.6%)
- Mean: +0.080, Std: 0.529

**Step 2: Построение features из scored новостей**

Те же features что в R123, формулы:

Per-coin (8 features на монету):
```
news_count_1h[sym, t]       = count(news where coin=sym, hour=t)
news_count_24h[sym, t]      = Σ news_count_1h[sym, t-23..t]
news_count_7d[sym, t]       = Σ news_count_1h[sym, t-167..t]
news_sentiment_1h[sym, t]   = mean(score | coin=sym, hour=t)
news_sentiment_24h[sym, t]  = rolling_mean(news_sentiment_1h, 24h)
news_sentiment_7d[sym, t]   = rolling_mean(news_sentiment_1h, 168h)
sentiment_momentum[sym, t]  = news_sentiment_24h - news_sentiment_7d
news_volume_zscore[sym, t]  = (news_count_24h - μ_30d) / σ_30d
```

Market-level (2 features, одинаковы для всех монет):
```
market_news_count_24h[t]    = rolling_sum(all_crypto_news_per_hour, 24h)
market_news_sentiment_24h[t]= rolling_mean(all_crypto_sentiment_per_hour, 24h)
```

Political (5 features) — отсутствали в данных (GDELT не подгружался), все 5 = NaN.

Interaction features (созданы в R123 на лету):
```python
nx_mkt_sent_x_ret12   = market_news_sentiment_24h * (-ret_12h)         # contrarian
nx_mkt_sent_x_vol     = market_news_sentiment_24h * rvol_24h           # sentiment × volatility
nx_mkt_count_zscore    = zscore(market_news_count_24h, rolling_720h)   # attention spike
nx_sent_divergence     = news_sentiment_24h[coin] - market_news_sentiment_24h  # divergence
```

**NaN handling и preprocessing при merge в основной df**:
```python
# Log1p transform для count features (heavy-tailed)
for col in ["news_count_1h","news_count_24h","news_count_7d","market_news_count_24h"]:
    df[col] = np.log1p(df[col])

# Coverage flag: 86.2% rows имели news coverage
df["news_coverage_ok"] = (~df["news_count_24h"].isna()).astype(float)
```

**Step 3: IC Scan** (Spearman rank correlation с fwd_ret_12h на test-окнах)

$$IC = \rho_{Spearman}\left(\text{feature}_{rank}, \text{fwd\_ret\_12h}_{rank}\right)$$

Gate: |IC| > 0.02 в ≥2 из 3 окон = "stable".

**IC results (FinBERT)**:

| Feature | Mean IC | W1 | W2 | W3 | Stable |
|---|---|---|---|---|---|
| nx_mkt_sent_x_vol | -0.0554 | -0.0254 | -0.1211 | -0.0198 | ✓ |
| market_news_sentiment_24h | -0.0530 | -0.0284 | -0.1102 | -0.0205 | ✓ |
| nx_sent_divergence | +0.0436 | +0.0297 | +0.0852 | +0.0159 | ✓ |
| nx_mkt_count_zscore | -0.0246 | +0.0477 | -0.0773 | -0.0442 | ✓ |
| market_news_count_24h | -0.0336 | +0.0022 | -0.0914 | -0.0117 | ✗ |
| news_volume_zscore | -0.0219 | +0.0155 | -0.0684 | -0.0129 | ✗ |
| news_sentiment_24h | -0.0112 | -0.0060 | -0.0261 | -0.0013 | ✗ |

**4 features прошли gate** (vs 0 с VADER в R123) — FinBERT действительно создает более информативные IC.

**Step 4: 6 экспериментов**

Каждый эксперимент: train LGB+XGB ensemble на 3 walk-forward окнах × 3 random seeds = 18 моделей:
```python
# Энсембль: LGB + XGB, усреднение предсказаний
preds = train_ensemble(df, features, CONTINUOUS_WINDOWS, seeds=[42,123,777],
                       cs_rank_exclude=market_level_features)
```

Cross-sectional ranking: per-coin features ранкируются внутри timestamp (percentile 0-1), market-level features — нет (одинаковы для всех монет).

Simulation: long 4 / short 2, rebalance 12h, S6 prod_blended costs, trend risk-off (cutoff_on=0.9, cutoff_off=0.8, hysteresis=3).

**Step 5: Bootstrap comparison**
$$P(\text{improvement}) = \frac{1}{N_{boot}} \sum_{b=1}^{5000} \mathbb{1}\left[Sharpe_{test}^{(b)} > Sharpe_{base}^{(b)}\right]$$

где $(b)$ — bootstrap sample (with replacement) из aligned returns.

### Результаты R125

| Experiment | Features | Net Sharpe | Δ vs base | DD% | P(imp) |
|---|---|---|---|---|---|
| **A baseline** | 31 champion | **2.831** | — | -10.9% | — |
| B +market | +2 (market_news_count/sent_24h) | 1.650 | **-1.181** | -18.9% | 0.048 |
| C +mkt+pol | +2 (political=NaN, =B) | 1.650 | -1.181 | -18.9% | 0.048 |
| D +all news | +10 all available | 1.289 | -1.542 | -20.0% | 0.032 |
| E +IC-pass | +4 IC-stable | 1.104 | **-1.727** | -19.2% | 0.011 |
| F +interact | +6 (market + interactions) | 0.527 | **-2.304** | -20.9% | 0.002 |

**Per-window breakdown экспериментa A (baseline)**:
- W1: Sharpe=1.880, Ret=32.6%
- W2: Sharpe=4.334, Ret=57.9%
- W3: Sharpe=2.677, Ret=21.7%

**Per-window breakdown эксперимента B (+market)**:
- W1: Sharpe=0.183, Ret=0.5%
- W2: Sharpe=4.542, Ret=64.9%
- W3: Sharpe=0.762, Ret=5.4%

Вердикт: **NEGATIVE** — все варианты хуже baseline, Sharpe падает на 1.18 до 2.30.

---

## Сравнение R123 (VADER) vs R125 (FinBERT)

| | R123 (VADER) | R125 (FinBERT) |
|---|---|---|
| Scorer accuracy | ~60% | ~87% |
| IC-passing features | 0/14 | 4/14 |
| Best variant Sharpe | ~2.83 (=baseline) | 1.65 (-1.18) |
| Worst variant Sharpe | ~2.49 | 0.53 |
| DD baseline | -10.9% | -10.9% |
| DD worst variant | ~-13% | -20.9% |

**Парадокс**: FinBERT дает в 4 раза больше IC-passing features, но результат в 6 раз хуже (не -0 а -1.18).

---

## Self-Review: Что может быть неправильно

### 🔴 Критическое: Возможное уничтожение VADER scores при перезаписи crypto_news.parquet

**Проблема**: `fetch_crypto_news.py --skip-fetch --scorer finbert` перезаписывает `data/sentiment/crypto_news.parquet` FinBERT scores. Потом `_research_r123_news_sentiment.py` (скрипт для R123) читает этот же файл. Значит:
- Эксперимент A (baseline) тренировался на тех же данных (31 feature, без news) → **OK, baseline не зависит от scorer**
- Эксперименты B-F читали news features из **FinBERT-scored** файла → **OK, это то что мы хотели**
- ⚠️ **Но скрипт называется R123, а не R125**. Он перезаписал `results/r123_news_sentiment.json` вместо создания нового файла. Потом мы скопировали: `cp results/r123_news_sentiment.json results/r125_finbert.json`. **Не является ошибкой в данных, только в naming.**

### 🟡 Важное: C = B из-за отсутствия political data

Эксперимент C добавляет 5 political features (`political_news_count_24h`, etc.), но в данных их нет (GDELT не подгружался). `Missing: ['political_news_count_24h', ...]`. В коде:
```python
extra_avail = [f for f in extra_feats if f in df.columns]  # → фильтрует только доступные
```
Итого: C использует ровно те же 2 features что и B → **одинаковые результаты** (1.650 = 1.650). Это не баг, но **C не тестирует political features** — оно тестирует то же самое что B.

### 🟡 Важное: Sharpe calculation mismatch в R124

R124 `compute_metrics()` считает Sharpe **только по торговым периодам** (exclude risk_off), а стандартный `analyze_config()` из R113 (используемый в R123/R125) считает по всем. Поэтому:
- R124 baseline Sharpe = 3.691
- R125 baseline Sharpe = 2.831

Оба правильны, но с разными формулами. **Дельты сопоставимы внутри каждого эксперимента**.

### 🟡 Важное: Per-window instability = red flag

R125 B показал: W1=0.183, W2=4.542, W3=0.762. Это значит модель с news features **полностью ломается на W1 и W3**, а W2 даже чуть лучше baseline (4.542 vs 4.334). Это не "news features не работают" — это "news features вызывают **переобучение** в 2 из 3 окон".

Почему: LightGBM/XGBoost легко запоминают шумовые корреляции. При 33 features (31+2) vs 31 model capacity не сильно меняется, но 2 новых market-level фичи с IC~-0.05 позволяют модели "сфокусироваться" на них в train, что не обобщается на test.

### 🟡 Количество features: 10/15 доступно, не все 15

`Available: 10/15 features, Missing: political (5)`. При "all 15 news features" реально добавляется 10. Это known и корректно — скрипт фильтрует.

### 🟢 Minor: FinBERT trained на Financial PhraseBank, а не на crypto

FinBERT обучался на корпоративных финансовых текстах ("EPS increased 15%", "profits declined"). Crypto news ("Bitcoin pumps 10% after whale accumulation") — другой домен. CryptoBERT (ElKulako/cryptobert, обучен на 3.2M crypto tweets) мог бы быть лучше, но мы его не тестировали в R125. **Не запустили потому что FinBERT показал такой сильный negative result.**

### 🟢 Minor: batch_size=64 для H100

H100 80GB может легко делать batch_size=256-512 для модели 110M параметров (FinBERT). batch_size=64 не влияет на результаты (только на скорость), но можно было быстрее.

### 🟢 Minor: execution delay noise (exec_delay_penalty)

`rng = np.random.RandomState(42)` — детерминистичный seed для noise. Это хорошо (reproducibility), но один seed = один realization шума. Можно было бы Monte Carlo с N realizations и усреднить. На практике noise с σ=3bp при Sharpe>2 minor impact.

### ❓ Вопрос для ревью: Почему IC-passing features ухудшают Sharpe?

Это не trivial. IC > 0.02 (stable across windows) должен означать prerdictive power. Возможные объяснения:
1. **IC computed unconditionally** — spearmanr по всему test set. Но модель использует features в interaction с другими 31 features. IC маргинального feature ≠ conditional value.
2. **IC temporal regime** — IC=-0.1211 в W2, но -0.0198 в W3. Модель обучается на train, но IC меняется.
3. **Multicollinearity** — `market_news_sentiment_24h` может коррелировать с существующими features (vix_zscore, fear_greed). Добавление коррелированных features = noise injection.
4. **Leakage direction** — IC отрицательный означает "позитивный sentiment → negative return". Если модель учит "позитивный sentiment → short", а потом market mode меняется — катастрофа.

### ❓ Вопрос для ревью: Правильна ли формула cost deduction?

```python
turnover_cost = sum(cost_fn(sym) * avg_weight for sym in new_opened)
turnover_cost += sum(cost_fn(sym) * avg_weight for sym in closed)
```

`avg_weight = 1/total_positions` — это вес каждой позиции. Cost начисляется только на changed positions (opened + closed). **Это правильно**: если из 6 позиций 2 меняются, мы платим cost на 2/6 = 33% портфеля. Funding начисляется на весь портфель.

Но: holding_cost = funding_per_12h * (rebal_hours / 12). При rebal_hours=12 это ровно funding_per_12h=1.2bp. **Funding списывается раз в 8 часов на OKX (3 раза в сутки)**. За 12h hold = ~1.5 funding events. Мы моделируем это как flat 1.2bp/12h. Это approximation, реальный funding variable (может быть 0.01% или -0.01%). **Среднее по историческим данным ~0.01%/8h для BTC → 0.015%/12h = 1.5bp**. Мы используем 1.2bp — **slightly optimistic, но в разумном range.**

---

## Общий вердикт

| Исследование | Результат | Actionable |
|---|---|---|
| R124 Fee Opt | POSITIVE | 20% referral cashback = +0.068 Sh, бесплатно |
| R125 FinBERT | NEGATIVE | Все варианты хуже baseline, Sharpe -1.18 до -2.30 |

**R124**: Зарегистрировать реферальную ссылку — мгновенный бесплатный буст ~2.4% годовой доходности.

**R125**: Закрыть направление news sentiment. Ни VADER, ни FinBERT не помогают а вредят. 31 champion features уже содержат достаточно informative power. News sentiment шумный на крипто и модель переобучается на него.

---

## Воспроизводимость

```bash
# R124 (на любой машине с данными):
python _research_r124_fee_optimization.py

# R125 (нужен GPU для FinBERT, но R123 скрипт работает и на CPU через VADER data):
# Step 1: Re-score news with FinBERT (GPU)
python fetch_crypto_news.py --skip-fetch --scorer finbert
# Step 2: Run experiments
python _research_r123_news_sentiment.py
```
