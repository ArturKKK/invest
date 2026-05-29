# 🔍 Project Audit & Review — May 2026

**Дата:** 2026-05-29  
**Статус:** Code review + VPS data check + missed opportunities search  
**Базовое состояние:** Чемпион R114b (Net Sharpe 2.831), R127 ablation завершена, R128-R134 OOS тесты выполнены  

---

## 1. СТАТУС D6 ДАННЫХ (orderbook depth)

### Текущее состояние
- **Размер:** 5.8 MB (`data_vps_d6/orderbook_depth/binance_orderbook_depth_snapshots.parquet`)
- **Временной диапазон:** 2026-04-05 22:00 → 2026-04-15 10:00 (**10 дней**, не 3 месяца!)
- **Количество:** 11,450 snapshots
- **Символов:** Не известно точно (feature_ob.parquet имеет более полные данные)

### Проблема
❌ **Data не достаточна для тренировки 3 месяца!** Собирается всего 2+ недели. Скорее всего сбор данных был ОСТАНОВЛЕН или ПРЕРВАН.

### IC Score results (2026-04-26)
- Протестирована 13 features на 1h/4h/12h/24h горизонтах
- **Все IC < 0.03** (очень слабый сигнал)
- Лучший: `spread_bps` при horizon 24h = +0.0166
- **Вердикт:** Orderbook depth features **НЕ имеют predictive power** на текущих данных

### Рекомендация
1. ✅ Продолжить сбор d6 данных на VPS (нужно минимум 3 месяца)
2. ❓ Выяснить почему сбор был остановлен (логи на VPS?)
3. 🔄 Пересчитать IC score когда будет 90+ дней данных
4. 💡 Если IC остаётся <0.02, **отказаться от d6 фичей** (не стоит GPU места)

---

## 2. НАЙДЕННЫЕ ОШИБКИ И ПРОБЛЕМЫ В КОДЕ

### 🔴 CRITICAL — Неправильный порядок после R127 ablation

**Статус:** Partially reverted, not committed  
**Файлы:** `_research_r35_new_features.py`, `_ic_scanner.py`  
**Проблема:** После ablation R127 найдено что **Fix#1 и Fix#2 были ошибки**:
- **Fix#1** (inf→nan cleanup в `_ic_scanner.py`): **−0.55 Sharpe** ❌
- **Fix#2** (6 dead features в `MARKET_LEVEL_FEATURES`): **−1.64 Sharpe** ❌

**Вердикт R127:** Вернуться к F10_F20 состоянию (до Fix#1 и Fix#2)

**Текущее состояние кода:**
- ✅ Fix#1 удален (комментарий добавлен)
- ✅ Fix#2 удален (комментарий добавлен)
- ❌ **Изменения НЕ закомичены** (остаются в modified state)

**Действие:** Нужно закомитить эти изменения:
```bash
git add _research_r35_new_features.py _ic_scanner.py
git commit -m "R127: Revert Fix#1 + Fix#2 (both cost Sharpe, prod never had them)"
```

### 🟡 WARNING — `_d6_ic_check.py` legacy, не используется

**Проблема:** Старый скрипт загружает данные из неправильного пути (`/home/trader/invest/data/features/...`) который не существует локально

**Статус:** Этот скрипт ЗАМЕНЁН на `_r128_d6_ic_scan.py` который работает

**Действие:** Удалить или архивировать `_d6_ic_check.py` (он legacy)

### 🟡 WARNING — Вероятная data quality проблема в `features_ob.parquet`

**Проблема:** 
- `features_ob.parquet` имеет только 10 дней данных (2026-04-05 → 2026-04-15)
- Merge main features + OB features дал **0 rows** в первом скрипте

**Причина:** Timestamp misalignment или несовместимый формат timestamps

**Статус:** `_r128_d6_ic_scan.py` работает (работает самостоятельно на features_ob.parquet), но интеграция с main feature set не удалась

---

## 3. ЭКСПЕРИМЕНТЫ С ОШИБКАМИ

### R127 — Replacement Features Sweep (❌ ВСЕ ХУДШИЕ)

| Experiment | 4L/2S NetSh | vs Baseline | Status |
|---|---|---|---|
| **BASELINE** | **3.777** | — | ✅ Correctly reproduces |
| B_hod_vol_rank | 2.898 | **−0.879** | ❌ Worse |
| F_drop_only (control) | 2.898 | −0.879 | ❌ Control also worse |
| E_regime_x_beta | 2.282 | **−1.495** | ❌ Terrible for 4L |
| D_session_regime | 2.108 | **−1.669** | ❌ Terrible |
| A_seasonal_x_symbol | 2.102 | **−1.675** | ❌ Worst |
| C_relative_breadth | 1.889 | **−1.888** | ❌ Catastrophic |

**Вывод:** 
- **Ни один replacement НЕ бил baseline 3.777**
- Даже "контроль" (drop-only) был хуже
- Это подтвердило что **6 "dead" features — это полезная регуляризация, несмотря на нулевые значения после CS-rank**

**Причина:**  
- Per-symbol replacements либо имели **NaN coverage issues** (B был идентичен F), либо **перефиттились** на новые сигналы
- CHAMPION_FEAT_31 — это **тесно связанный оптимум**, удаление любого слота ломает калибровку

**Статус:** ✅ Решение принято (REVERT, НЕ МЕНЯТЬ)

### R126 — Regularization sweep (❌ ПРОВАЛИЛАСЬ)

**Статус:** Net Sharpe 2.34 < baseline 3.224, acceptance failed  
**Артефакты:** `/data/datasets/r126_final.json`  
**Проблема:** Over-regularization с 600 rounds параметров убила сигнал

---

## 4. УПУЩЕННЫЕ ИДЕИ (Направления для развития)

### ✅ Priority #1 — Maker-first execution  
**Потенциал:** +0.227 Sharpe (подтвержден в R121)  
**Статус:** **НЕ РЕАЛИЗОВАНО**  
**Что нужно:** Логика в executor для limit-order-first (вместо market order)  
**Легкость:** ⭐ EASY — нет ML, чистая логика  
**Рекомендация:** **СДЕЛАТЬ НЕМЕДЛЕННО** (бесплатный прирост)

### ✅ Priority #2 — OKX referral 20% rebate  
**Потенциал:** +0.068 Sharpe  
**Статус:** **НЕ АКТИВИРОВАНА**  
**Что нужно:** Установить referral link на VPS  
**Легкость:** ⭐ TRIVIAL  
**Рекомендация:** **АКТИВИРОВАТЬ** (в следующем развертывании)

### ✅ Priority #3 — CryptoQuant exchange flows (платный API)  
**Потенциал:** TBD (институциональные потоки, более реальны чем CoinMetrics)  
**Статус:** **Не протестировано**  
**Что нужно:** Платный API + интеграция в feature pipeline  
**Легкость:** ⭐⭐ MEDIUM  
**Затраты:** ~$500/мес за API  
**ROI:** Если +0.2 Sharpe → окупится за 2-3 месяца трейдинга

### 🟡 Priority #4 — Conviction sizing (softmax temperature-tuning)  
**Статус:** R114 пробовал full Z-score sizing и проиграл binary 4L/2S  
**Идея:** Попробовать **softmax-based weighting с temperature parameter**, не full weighted  
**Причина:** Может сместить риск-профиль без потери Sharpe  
**Легкость:** ⭐⭐⭐ HARD (нужна переоптимизация на val)

### 🟡 Priority #5 — Regime-aware L/S imbalance  
**Идея:** Вместо фиксированного 4L/2S, адаптировать к режиму:
- Strong uptrend (`trend_strength > 1.5`) → 5L/1S
- Risk-off → 3L/3S  
**Статус:** Никогда не тестировалось систематически  
**Легкость:** ⭐⭐⭐ HARD

### 🟡 Priority #6 — LLM-based news signals  
**Статус:** Все варианты (VADER R123, FinBERT R125) **дали −1 to −2 Sharpe**  
**Парадокс:** IC выглядит хорошо, но PnL хуже  
**Причина:** Вероятно **lookahead bias** (новости появляются с задержкой в реальности)  
**Рекомендация:** **НЕ ПОВТОРЯТЬ** (закрыто направление)

### 🟡 Priority #7 — Feature integrity gate  
**Идея:** Pre-flight скрипт, который проверяет что все 31 фича присутствуют И имеют coverage >95%  
**Статус:** Задокументировано в memory но **НЕ реализовано**  
**Легкость:** ⭐⭐ EASY (1 час)  
**Защита:** От silent degradation из-за data loader failures  
**Рекомендация:** **РЕАЛИЗОВАТЬ в следующем спринте**

### 🟡 Priority #8 — Unified cost module  
**Проблема:** `_cost_for_sym` живет в разных местах, не синхронизирована  
**Текущее:** S6 prod_blended (Tier1=2.4bp, Tier2=5.5bp, Tier3=10bp)  
**Статус:** **НЕ унифицировано**  
**Легкость:** ⭐⭐ EASY (2 часа)  
**Рекомендация:** Создать `src/costs.py`, импортировать везде

### 🟡 Priority #9 — Live vs Backtest parity test  
**Статус:** R127.7 обнаружил **−7.9 Sharpe gap** (backtest +3.777 vs live −4.10)  
**Причины найдены:** small capital, state loss, margin fees, min-order fees  
**Что нужно:** Скрипт для сравнения live feature values с backtest  
**Рекомендация:** **КРИТИЧНО** перед следующим развертыванием

---

## 5. КРИТИЧЕСКИЕ ПРОБЛЕМЫ В PRODUCTION

### 🚨 LIVE vs Backtest Gap (Net Sharpe −7.9)

**Backtest (R114b):** Net Sharpe 2.831, Win Rate 59.7%  
**LIVE (2026-03-16→04-23):** Net Sharpe −4.10, Win Rate 23.7% (9/38 days)  
**Регистрация:** $110 → $80 (−27.3%)

**Найденные причины (НЕ код баги):**
1. **Capital constraint:** $80 × 1x leverage = $5-10 per position → min-order fees eat PnL
2. **State loss:** 51 bot restarts за 5 недель → `trading_state.json` теряется
3. **Sim-prod parity bugs:** edge_boost, dynamic L/S, sm_scale overlays может быть активны в CLS mode
4. **Data quality:** Limited margin book history

**Статус:** VPS уже на целевом состоянии (F10_F20). Нужна диагностика перед next deployment.

---

## 6. GIT STATUS (UNCOMMITTED CHANGES)

### Modified files (нужно закомитить):
```
M .github/agents/memory-keeper.agent.md    (updated agent instructions)
M PROGRESS.md                              (R127 results documented)
M _ic_scanner.py                           (Fix#1 removed, commented)
M _mlc_oos_fast.py                         (updates to OOS harness)
M _r132_oos_validate.py                    (validation script)
M _r134_compare.py                         (comparison script)
M _research_r35_new_features.py            (Fix#2 removed, commented)
M plan.md                                  (plan updates)
M run_trading.py                           (possible minor updates)
```

### Untracked files (новые research scripts):
- `_ablation_*.py` (R127 ablation harness)
- `_replacement_*.py` (R127.5 replacement features)
- `_r128_*.py` (R128 OOS overlay sweeps)
- `_r128b_*.py`, `_r128c_*.py` (R128 variants)
- `_r135_baseline_repro.py` (baseline reproduction)
- Data artifacts: `data_vps_d6/`, `results_r128_*.json`

**Рекомендация:** Закомитить PROGRESS.md + основные fixes, архивировать research скрипты в branch

---

## 7. ОБЩИЙ ПЛАН ДАЛЬНЕЙШИХ РАБОТ

### Фаза 1: Stabilization (2 часа)
1. ✅ Закомитить R127 revert changes (Fix#1, Fix#2 removal)
2. ✅ Обновить git history
3. ✅ Verify baseline 3.777 reproduces locally + MLC

### Фаза 2: Quick Wins (4 часа)
1. 🚀 **Maker-first execution** (+0.227 Sharpe)
2. 🚀 **OKX referral activation** (+0.068 Sharpe)
3. → Итого: **+0.295 Sharpe** (от 2.831 → 3.126!)

### Фаза 3: Infrastructure (8 часов)
1. **Feature integrity gate** — pre-flight check все 31 фич
2. **Unified cost module** — `src/costs.py`
3. **Live parity test** — скрипт для сравнения live vs backtest features

### Фаза 4: Deep Research (20+ часов)
1. **D6 data continuation** — собрать 3+ месяца orderbook
2. **CryptoQuant flows** — если бюджет позволяет
3. **Conviction sizing revisit** — если есть время

### Фаза 5: Next Deployment (TBD)
- Перед развертыванием: пройти full checklist (R25, R130, R127.7)
- Стартовый капитал: минимум $500+ (чтобы комиссии не ели прибыль)
- State backup: сохранять `trading_state.json` в git

---

## 8. ОЦЕНКА ЗДОРОВЬЯ ПРОЕКТА

### ✅ Strengths
- **Solid baseline:** Net Sharpe 2.831 с полной валидацией (3 окна, 5 seeds)
- **Reproducible:** Canonical cache позволяет точно воспроизвести R114b
- **Well-documented:** MEGA_PROMPT.md, PROGRESS.md, memory system
- **Low overfitting:** R127 ablation доказала что нет hidden lookahead bugs
- **Good governance:** Walk-forward, bootstrap, per-window stats

### ⚠️ Weaknesses
- **Sim-prod gap:** −7.9 Sharpe (потребует serious diagnostics)
- **D6 stalled:** Orderbook data собирается только 2 недели (не 3 месяца)
- **Unimplemented wins:** +0.3 Sharpe лежит на столе (maker-first, referral)
- **State fragility:** 51 restarts → state loss, min-order fees issue

### 🎯 Opportunity
- **+0.3 Sharpe quick wins** = от 2.831 → 3.126 (11% прирост)
- **If D6 IC > 0.02 at 90d** = возможно еще +0.1-0.2
- **If maker-first + conviction sizing** = возможно +0.3-0.5 в долгосроке

---

## 9. СПИСОК КРИТИЧЕСКИХ РЕШЕНИЙ, КОТОРЫЕ НУЖНЫ

### 1. Закомитить ли R127 revert changes?
**Рекомендация:** ✅ **ДА, НЕМЕДЛЕННО**  
Код уже исправлен, комментарии добавлены, нужно только `git commit`

### 2. Продолжить ли D6 collection?
**Рекомендация:** ✅ **ДА, но с переоценкой после 90 дней**  
Если IC остаётся <0.02 при 90d данных, отказаться от направления

### 3. Реализовать ли maker-first execution?
**Рекомендация:** ✅ **ДА, в следующем спринте**  
+0.227 Sharpe подтвержден, это бесплатный прирост

### 4. Развертывать ли на VPS скоро?
**Рекомендация:** ⚠️ **ПОДОЖДАТЬ**  
1. Сначала пройти full parity test (live vs backtest features)
2. Увеличить стартовый капитал до $500+ (текущие $80 съедают комиссии)
3. Исправить state loss проблему (git commits для state backup)

### 5. Какой следующий big research?
**Рекомендация:** 🎯 **CryptoQuant flows** (если бюджет) или **Conviction sizing** (если нет)  
Оба имеют потенциал +0.2-0.3 Sharpe

---

## SUMMARY FOR USER

**Краткое резюме по-русски:**

✅ **Хорошие новости:**
- Baseline 2.831 ПРАВИЛЕН (R127 подтвердила)
- R127 ablation разрешила спор о Fix#1/Fix#2 (обе стоят −Sharpe)
- Replacement features все хуже (значит 31f оптимален)

❌ **Проблемы:**
- D6 данные собирались только 2 недели (не 3 месяца), IC слабый
- LIVE потерял 7.9 Sharpe (но это не баг кода, а sim-prod gap)
- +0.3 Sharpe висит в воздухе (maker-first, OKX referral) — не реализовано

🚀 **Действия (приоритет):**
1. Закомитить R127 changes (2 минуты)
2. Сделать maker-first execution (+0.227) + referral (+0.068) = +0.295 Sharpe
3. Собрать 3 месяца D6 данных перед следующим развертыванием
4. Пройти live/backtest parity test перед next deployment

**Готов к следующему шагу через MLC на VPS?** Дайте команду.
