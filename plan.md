# Plan: R68 + R93 Combination — Attribution & Ensemble (R97–R100)

## Статус: 8 апреля 2026

**Контекст**: R93 (4h ML) показал Sharpe=3.190, MaxDD=-10.1%, P(Sharpe>0)=0.962. 
R68 (12h ML) — Sharpe=3.777, MaxDD=-13.95%, Return=179%.
Corr(R68, R93) = 0.456 — умеренная, есть потенциал для комбинации.

**Аномалия**: R93 Return=34.8% при Sharpe=3.19 vs R68 Return=179% при Sharpe=3.78.
Нужно понять ПОЧЕМУ, прежде чем смешивать.

**Порядок**: R97 Attribution → R100 Rank Ensemble → R99 Simple Mix (если нужен).
**Пропуск**: R98 (merged в R99), R101 (overfitting risk).

---

## R97 — Attribution Analysis (MANDATORY FIRST)

**Файл**: `_research_r97_attribution.py`

**Цель**: Понять разницу R68 Return=179% vs R93 Return=34.8% при похожем Sharpe.

### Метрики (считаем отдельно для R68 и R93 на aligned timestamps):

| Metric | Описание |
|--------|----------|
| n_trading_periods | Кол-во ребалансировок |
| pct_risk_off | Доля периодов без позиций (all cash) |
| mean_gross_ret | Средний return за период (до костов) |
| std_ret | Стд return за период |
| avg_turnover | Средний turnover (доля позиций, которые менялись) |
| avg_cost_bps | Средний cost per period (bps) |
| net_ret_per_period | mean_gross - avg_cost |

### Breakdown:
- Per-window (W1, W2, W3)
- Per-quarter (Q4'24, Q1'25, Q2'25, Q3'25, Q4'25, Q1'26)

### Гипотезы для проверки:
1. R93 сидит в кэше больше (pct_risk_off выше)?
2. R93 торгует чаще → больше костов (turnover)?
3. R93 имеет меньший gross_ret per period?
4. R93 concentration: торгует меньше символов?

### Acceptance:
- Чёткое объяснение ΔReturn в виде decomposition
- Понимание, нужен ли vol-match перед mixing

---

## R100 — Rank Ensemble (PRIORITY 1)

**Файл**: `_research_r100_rank_ensemble.py`

**Идея**: Один портфель из комбинированных rank-сигналов.

```
r_combined = α * rank_cs(p_12h) + (1-α) * rank_cs(p_4h)
```

где `rank_cs` = cross-sectional rank (0 to 1), `p_12h` = R68 predictions, `p_4h` = R93 predictions.

### Требования:
- **Alignment**: нужны predictions R68 и R93 на одних и тех же timestamp'ах
- R93 predictions: `results/r93_predictions.parquet`
- R68 predictions: нужно сохранить (или пересчитать) из `_research_r68_continuous_wf.py`

### Grid:
- α ∈ {0.0, 0.25, 0.50, 0.75, 1.0}
- α=1.0 = pure R68, α=0.0 = pure R93 (benchmark)

### Selection:
- TOP 4 / BOTTOM 2 (4L/2S) из combined_rank
- Simulate с standard R68 costs, 12h rebalance

### Bootstrap:
- Best α config vs R68 (α=1.0)
- P(ΔSharpe > 0) > 0.80 → ACCEPT

### Acceptance:
- Sharpe ≥ R68 (3.78) AND MaxDD < R68 (-13.95%)
- OR Sharpe ≥ 3.5 AND MaxDD < -10%

---

## R99 — Simple Return Mix (ONLY IF R97 shows vol-match needed)

**Файл**: `_research_r99_return_mix.py`

**Идея**: Два отдельных портфеля, returns складываются с весами.

```
ret_mix = (1 - w93) * ret_R68 + w93 * ret_R93_scaled
```

### Vol-scaling (если R97 показал разницу vol):
```
σ_R68 = std(ret_R68)
σ_R93 = std(ret_R93)
ret_R93_scaled = ret_R93 * (σ_R68 / σ_R93)
```

### Grid:
- w93 ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50}

### Optimisation target: **Calmar ratio** (return / MaxDD), не Sharpe!

### Bootstrap:
- Best w93 vs R68-only
- P(ΔCalmar > 0) > 0.80 → ACCEPT

### Acceptance:
- Calmar ↑ при сохранении Sharpe ≥ 3.5

---

## Execution Pipeline

```
R97 (attribution)  →  understand vol/return difference
       ↓
R100 (rank ensemble)  →  combine signals into one portfolio
       ↓
R99 (return mix, if needed)  →  fallback: two portfolios with weights
```

## Decision Tree

```
IF R97 shows vol-match NOT needed:
    → R100 rank ensemble (preferred, one portfolio)
    → IF R100 beats R68 → DEPLOY
    → IF R100 fails → R99 return mix

IF R97 shows vol-match needed:
    → R99 first (vol-scaled mix)
    → R100 with vol-scaled predictions
    → Best of R99/R100 → DEPLOY
```

## Порядок: R90 → R91 + R92 → R94 → R95

