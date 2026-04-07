# Plan: Текущий статус и следующие шаги

## Завершённые эксперименты (R60-R69)

### Ключевые результаты:

| # | Эксперимент | Net Sharpe | Вердикт |
|---|-------------|-----------|---------|
| R65 | **Gross vs Net: 4L/2S** | **2.984** | **WINNER — alpha, не costs** |
| R68 | **4L/2S continuous WF** | **3.777** | **Лучший результат — непрерывная торговля** |
| R60 | grid_4L2S (gapped) | 2.984 | = R65 baseline |
| R64 | Combined (4L2S+filter) | 1.84 | Marginal |
| R63 | Uncertainty filter std<0.03 | 1.83 | Шум |
| R61 | +cg_temporal 35f | 1.89 | Hurt в комбо |
| R62 | Meta-stacking LogReg+GRU | -0.38..1.48 | Провал |
| R67 | Reject option (prob threshold) | 1.507 best | Провал — снижает positions |
| R69 | Percentile uncertainty gating | 0.608 best | Катастрофа — фильтрует alpha-генераторы |

### Закрытые направления (не возвращаться):
- ❌ Temporal features (ret lags catastrophic, cg_temporal hurt combined)
- ❌ Meta-stacking (OOF от слабых моделей = шум)
- ❌ Uncertainty gating (любой вариант — fixed/percentile — разрушает)
- ❌ Reject option / score-gap threshold (kills diversification)
- ❌ dynamic_K, edge_cost_filter, prob_weighting (R60 failures)

---

## R70 — LambdaRank Objective (🔄 RUNNING)

LightGBM lambdarank + XGBRanker вместо binary classification.
Оптимизирует NDCG@K — прямо top-K ranking quality.

- [x] Скрипт написан (фикс int labels)
- [x] Задеплоен на MLC (invest-y5u733)
- [x] Запущен (PID 1438)
- [ ] Результаты собраны

---

## Следующие шаги (после R70)

### 1. Deploy 4L/2S в прод (VPS)
- Поменять PROD_CFG n_long=6→4, n_short=3→2
- Shadow-лог старого 6L/3S (какие позиции выбрал бы + expected edge)
- Через 2-4 недели: атрибуция "4L/2S лучше потому что..."

### 2. Сбор live данных 1-3 месяца
- Live execution logs
- Sim vs live расхождения
- Какие trades убыточны? Какие монеты wrong? Какие режимы убивают alpha?

### 3. Если R70 LambdaRank покажет результат → deploy
Если нет → система в финальном оптимуме, переход к live monitoring.

---

## Технические заметки

**MLC:** invest-y5u733, Python 3.11.15, pandas 2.3.3 (КРИТИЧНО — не обновлять!)
**Venv:** /data/datasets/.venv (абсолютный путь, symlink через .venv не работает в mlc exec)
**Запуск:** `mlc job exec invest-y5u733 -- bash -c 'cd /workdir/invest && /data/datasets/.venv/bin/python script.py > /data/datasets/log.log 2>&1 && echo DONE'`
