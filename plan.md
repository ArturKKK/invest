# Plan: R60-R63 Model Improvement Experiments

## TL;DR
Серия из 4 экспериментов для улучшения gen8 champion (Sharpe 1.66 → цель 2.0+).
Приоритет: dynamic K + weighting (R60) → uncertainty gating (R63) → temporal features (R61) → alt model stacking (R62).
Все эксперименты используют шаблон R58 (WF + hybrid costs), метрика: net Sharpe, maxDD, win rate по окнам.

---

## Phase 1: R60 — Dynamic K + Edge>Cost + Weighting

**Файл:** `_research_r60_portfolio_opt.py` (новый, на базе R58)

- [x] Скопировать `_research_r58_continuous_wf.py` → `_research_r60_portfolio_opt.py`
- [x] Модифицировать `simulate_with_hybrid_costs` — добавить `sizing_mode` параметр
- [x] Реализовать 5 вариантов портфельной конструкции:
  - [x] **baseline**: 6L/3S equal-weight (текущий)
  - [x] **grid_K**: 4L/2S, 6L/3S, 8L/4S, 3L/3S (чувствительность к K)
  - [x] **dynamic_K**: K_long = clip(2..8) по `strength = mean(top_K_score) - median(score)`; K_short аналогично. Пороги: strength > 0.3 → +1, > 0.6 → +2 к базовому K
  - [x] **edge_cost_filter**: открывать позицию только если `|p - 0.5| * notional > estimated_cost_bps`. Для этого нужны raw probs (до z-norm)
  - [x] **prob_weighting**: `w_long_i ∝ (p_i - 0.5)`, `w_short_i ∝ (0.5 - p_i)`, нормализовать отдельно L/S book
- [-] Для каждого варианта прогнать WF (ORIGINAL_WINDOWS, 5 seeds) с hybrid costs ← **DONE (23 min)**
- [x] Собрать таблицу: mode | Sharpe | MaxDD | WinRate | Turnover | Win_W1 | Win_W2 | Win_W3

**R60 RESULTS:**
```
grid_4L2S   Sharpe=2.98  Ret=+77.4%  DD=-14.1%  WR=60%  W1=2.37 W2=3.68 W3=3.28 ← WINNER
baseline    Sharpe=1.78  Ret=+33.9%  DD=-15.0%  WR=58% ← BASELINE  
edc_filter  Sharpe=1.60
prob_weight Sharpe=1.58
dynamic_K   Sharpe=1.51
```
**Winner: grid_4L2S (4 long, 2 short). Sharpe +1.20 vs baseline!**

**Ключевые модификации в simulate:**
- `_research_r48_cost.py:simulate_with_hybrid_costs` строка ~170: заменить `long_ret = longs["fwd_ret"].mean()` на взвешенный
- Добавить raw_probs в merged для prob_weighting и edge_cost
- Dynamic K: вычислять strength каждый timestamp из pred колонки

**Файлы (read-only):**
- `_research_r48_cost.py` — НЕ трогать, скопировать simulate в R60
- `_release_champion.py` — train_ensemble(), eval_with_costs()
- `_research_r58_continuous_wf.py` — шаблон

**Зависимости:** нет

---

## Phase 2: R63 — Uncertainty Gating (Seed Disagreement)

**Файл:** `_research_r63_uncertainty.py` (новый)

- [x] Модифицировать train_ensemble — возвращать per-seed predictions (не усреднённые)
  - Сейчас: `lgb_probs = mean(seed_preds)` → одно число
  - Нужно: сохранить lgb_seed_0_prob, ... + xgb_seed_*
  - Вычислить: `p_mean = mean(all_10_seeds)`, `p_std = std(all_10_seeds)`
- [x] Реализовать 3 варианта gating:
  - [x] **uncertainty_filter**: торговать только если `p_std < threshold` (пороги: 0.02, 0.03, 0.05)
  - [x] **uncertainty_scaling**: `weight_i *= (1 - clip(p_std_i / max_std, 0, 0.7))` — снижаем вес неуверенных
  - [x] **agreement_K**: dynamic K где K уменьшается если mean(p_std_topK) высок
- [-] WF на ORIGINAL_WINDOWS, hybrid costs ← **DONE (23 min)**
- [x] Таблица: mode | threshold | Sharpe | MaxDD | WinRate | AvgPositions

**R63 RESULTS:**
```
filter_std003  Sharpe=1.83  Ret=+35.1%  DD=-15.7%  thr=0.03 ← BEST (+0.05)
baseline       Sharpe=1.78  Ret=+33.9%  DD=-15.0% ← BASELINE
scaling        Sharpe=1.73  DD=-13.0%  (lower DD)
filter_std002  Sharpe=-0.09 (catastrophic - over-filtered)
```
**Winner: filter_std003. Marginal improvement +0.05 only.**

**Ключевое:** не нужно переобучать модели — просто сохранить 10 отдельных предсказаний и использовать std как фильтр

**Зависимости:** нет (параллельно с R60)

---

## Phase 3: R61 — Temporal Features (Lags + Rolling)

**Файл:** `_research_r61_temporal.py` (новый)

- [x] Добавить temporal features:
  - [x] `ret_12h`: lag2 (=ret_36h), lag4 (=ret_60h), rolling_std_5
  - [x] `cg_taker_imb`: lag1, lag2, rolling_mean_5, rolling_std_5
  - [x] `oi_chg_12h`: lag1, lag2, rolling_mean_5
  - [x] `sign_persistence_10 = rolling_mean(sign(ret_12h) == sign(ret_12h.shift(1)), 10)`
  - [x] `reversal_count_10 = rolling_sum(sign(ret_12h) != sign(ret_12h.shift(1)), 10)`
- [x] Всего ~12 новых фичей. Добавить в feature list (31 → ~43)
- [-] Обучить LGB+XGB на расширенном наборе, WF ← **DONE (26 min)**
- [x] Сравнить с baseline (31 feat champion)
- [x] Если улучшение — ablation: какие temporal фичи дают прирост

**R61 RESULTS:**
```
+cg_temporal  31+4=35f  Sharpe=1.89  DD=-11.8%  (Δ+0.11) ← BEST - cg lags help!
baseline_31f  31f       Sharpe=1.78  DD=-15.0% ← BASELINE
+oi_temporal  34f       Sharpe=1.39  (Δ-0.39) hurts
all_43f       43f       Sharpe=1.01  (Δ-0.77) overfit
+ret_temporal 36f       Sharpe=0.48  (Δ-1.30) catastrophic
```
**Winner: +cg_temporal (cg_taker_imb lags/rolling). Small improvement +0.11. ret_12h lags HURT.**

**Важно:**
- cg_taker_imb — дневные данные, лаги в днях
- Новые temporal фичи добавить в cs_rank_exclude (не ранкировать лаги cross-sectionally)
- НЕ использовать cum_funding_24h (missing на VPS)

**Зависимости:** нет (параллельно с R60/R63)

---

## Phase 4: R62 — Alternative Model as Feature (Stacking)

**Файл:** `_research_r62_stacking.py` (новый)

- [x] **p_lin (Ridge/LogReg)**: обучить LogReg на 8 лучших фичах. OOF-предсказания (5-fold temporal CV).
  - Фичи: ret_12h, ret_24h, mom_z_24h, oi_chg_12h, taker_cvd_12h, atr_14, pct_coins_up_12h, cg_taker_imb
- [x] **p_seq (GRU micro)**:
  - Input: последние 8 баров × 5 фичей (ret_12h, rvol_12h, cg_taker_imb, oi_chg_12h, pct_coins_up_12h)
  - Architecture: GRU(hidden=16) → Dense(1, sigmoid)
  - OOF для train, full-train pred для test
- [x] Добавить p_lin и/или p_seq как новые фичи (31 → 32-33)
- [-] Обучить LGB+XGB с новыми фичами, WF ← **DONE (33 min)**
- [x] Сравнить: baseline vs +p_lin vs +p_seq vs +both

**R62 RESULTS:**
```
baseline_31f   31f  Sharpe=1.78 ← BASELINE
+p_seq_32f     32f  Sharpe=1.48 (Δ-0.29) - GRU hurts
+p_lin+p_seq   33f  Sharpe=0.50 (Δ-1.28)
+p_lin_32f     32f  Sharpe=-0.38 (Δ-2.16) - LogReg catastrophic
```
**Вывод: meta-stacking не помогает. Скорее всего OOF LogReg/GRU не даёт дополнительного сигнала поверх LGB на этих же фичах.**

**Предупреждения:**
- OOF обязателен — иначе утечка
- GRU требует torch
- Temporal CV внутри train: split по времени, не random

**Зависимости:** лучше после R61

---

## Verification Checklist

- [x] Каждый эксперимент сохраняет результаты в `results_r6X_*.csv` — все 4 сохранены
- [x] Таблица сравнения: все варианты в одной таблице (в этом файле)
- [x] Per-window breakdown (W1/W2/W3 представлены в каждом эксперименте)
- [ ] Monthly returns для визуальной проверки (pending)
- [x] Sanity checks:
  - Sharpe > 1.0: ✅ grid_4L2S=2.98, filter_std003=1.83, +cg_temporal=1.89
  - MaxDD < 60%: ✅ все < 20%
  - Win rate 55-65%: ✅ 57-60% у лучших
- [ ] Финальная верификация: лучший R60 + лучший R63 → combined run

## Финальный рейтинг по всем экспериментам

| Эксперимент | Лучший вариант | Sharpe | Действие |
|---|---|---|---|
| R60 | **grid_4L2S** | **2.98** | ⭐ Уменьшить K до 4L/2S |
| R61 | +cg_temporal | 1.89 | Добавить cg_taker_imb лаги |
| R63 | filter_std003 | 1.83 | Gating по std<0.03 |
| R62 | никто | - | Stacking не помогает |

**Очевидный следующий шаг**: combined run grid_4L2S + filter_std003 + опционально +cg_temporal фичи.

## Decisions (не менять)
- ORIGINAL_WINDOWS (с gap-ами) для сравнимости с baseline Sharpe 1.66
- N_ROUNDS=600, EARLY_STOP=40
- 5 seeds [0, 7, 13, 42, 99]
- Hybrid tiered costs обязательно
- Шаблон: R58 (`_research_r58_continuous_wf.py`)

## Execution Order
R60 и R63 параллельно (независимы). R61 параллельно. R62 последним.
Если последовательно: R60 → R63 → R61 → R62
Ожидаемое время: R60 ~40min, R63 ~30min, R61 ~40min, R62 ~60min (GRU)
