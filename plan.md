# Plan: R102–R104 — Fix R100 + Conservative Ensemble

## Статус: 8 апреля 2026

**Контекст**: R100 rank ensemble невалиден (α=1.0 → Sharpe 3.311, каноничный R68 = 3.777).
Причина: double-ranking pred уже rank-blended score, нарушение EMA/hysteresis.
R99 return mix тоже не значим (P(Sharpe)=0.42).

**Цель**: Корректно проверить, даёт ли R93 (4h-target) улучшение R68 через information combination,
не ломая baseline и не добавляя costs.

---

## R102 — Fix baseline invariance for ensemble (обязательно)

**Файл**: `_research_r102_baseline_equivalence.py`

**Требование**: в любом "ensemble" эксперименте при α=1.0 результат должен быть
**бит-в-бит** как R68 (Sharpe=3.777, DD=-13.95%, N=688).

Checklist:
- Использовать тот же `load_data()` и тот же `simulate()`, что R68
- Использовать тот же scoring/weighting/selection, что в R68
- Совпадение: n_periods, timestamps, avg turnover, avg cost, equity correlation ~1.0

Deliverable:
- `results/r102_baseline_equivalence.json` с флагом PASS/FAIL и диагностикой
- `results/r102_equity.csv`

---

## R103 — Ensemble v2: logit combination (без rank-induced drift)

**Файл**: `_research_r103_logit_ensemble.py`

Вместо `rank_cs(p)` → комбинация в **logit-пространстве** (без деградации baseline):

```
s68 = logit(clip(raw_prob_68, 1e-6, 1-1e-6))
s93 = logit(clip(raw_prob_93, 1e-6, 1-1e-6))
s = α * s68 + (1-α) * s93
```

longs = top4 by `s`, shorts = bottom2 by `s`

Grid α: {0.0, 0.25, 0.5, 0.75, 1.0}

Acceptance:
- Sharpe >= R68 + 0.05 with bootstrap P>0.8
- OR MaxDD improves ≥15% with Sharpe >= R68 - 0.05 and bootstrap P(Calmar better)>0.8

Deliverables:
- `results/r103_summary.json`, `results/r103_equity.csv`
- `results/r103_bootstrap.json`, `results/r103_grid.csv`

---

## R104 — Conservative tie-break ensemble (safer, high-EV)

**Файл**: `_research_r104_tiebreak_ensemble.py`

Идея: не "мешать всё", а использовать R93 как tie-breaker внутри кандидатов R68.

Алгоритм:
1. По R68 score выбрать кандидатов:
   - long_pool = top M (M ∈ {6, 8, 10})
   - short_pool = bottom N (N ∈ {3, 4, 5})
2. Внутри pool ранжировать по R93 score:
   - финальные longs = top4 по R93 среди long_pool
   - финальные shorts = bottom2 по R93 среди short_pool
3. Исполнение и sizing — как R68.

Grid: M×N = 3×3 = 9 конфигов.

Acceptance: как в R103.

Deliverables:
- `results/r104_summary.json`, `results/r104_equity.csv`
- `results/r104_bootstrap.json`, `results/r104_grid.csv`

---

## Bootstrap

- Block bootstrap block=10, N=1000
- Считать P(Sharpe better) и P(Calmar better) vs R68
- Порог: P>0.8

---

## Если R103/R104 не дают значимости

- R93 оставить как отдельную стратегию "на мониторинг" (paper/live shadow)
- Основной путь улучшения — D6 microstructure (отдельный roadmap)

---

## Execution Pipeline

```
R102 (baseline check)  →  PASS required
       ↓
R103 (logit ensemble)  →  grid α, bootstrap
       ↓
R104 (tiebreak)        →  grid M×N, bootstrap
       ↓
Best of R103/R104 vs R68 → DEPLOY or REJECT
```

