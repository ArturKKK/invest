# Plan: Parallel Systems — Funding Carry + Liq Mean-Reversion (R90–R95)

## Статус: 8 апреля 2026

**Контекст**: DeepResearch v3 (R80–R86) показал, что CG alpha как линейные фичи в ML не работает (0/13 прошли IC gate). Vol overlay не проходит bootstrap (P=0.372). R68 baseline (Sharpe=3.777) устойчив.

**Новая стратегия**: Вместо улучшения R68 — строим 1–2 **параллельные, принципиально другие системы** и смешиваем их с R68. Это диверсификация на уровне стратегий, не фич.

---

## R90 — Data Audit (prerequisite)

**Файл**: `_research_r90_data_audit.py`

| Check | Описание |
|-------|----------|
| OHLCV | load_research_frame() — rows, symbols, date range |
| CG funding | funding.parquet — rows, symbols, coverage |
| CG liq | liq.parquet — rows, symbols, coverage |
| CG oi | oi.parquet — rows/symbols |
| CG taker | taker.parquet — rows/symbols |
| CG ls | ls_ratio.parquet — rows/symbols |
| Shift1 | Verify cg_date + 1d alignment |

**Acceptance**: Все datasets загружаются. Coverage ≥ 80% по символам.

---

## R91 — Funding Carry Strategy (rule-based, NO ML)

**Файл**: `_research_r91_funding_carry.py`

**Идея**: Coins с низким FR → long (получаем funding), coins с высоким FR → short (получаем funding).

**Сигнал**: `carry_score = -fr_close` (shift1)

**Grid**:
- K ∈ {2L/2S, 3L/3S, 4L/2S}
- rebal_hours ∈ {12, 24}

**Simulation**: R68 `simulate()` с carry_score вместо ML pred

**Acceptance**: Sharpe > 0, корреляция с R68 < 0.3

---

## R92 — Liquidation Event Mean-Reversion

**Файл**: `_research_r92_liq_events.py`

**Событие**: `liq_zscore > threshold`
**Направление**: liq_long >> liq_short → go long (и наоборот)

**Grid**:
- threshold ∈ {2.0, 2.5, 3.0}
- hold H ∈ {1, 2, 3} × 12h
- K ∈ {1L/1S, 2L/1S, 2L/2S}
- cooldown ∈ {0, 1, 2}

**Acceptance**: Sharpe > 0, hit rate > 50%, corr с R68 < 0.3

---

## R94 — Strategy Mix

**Файл**: `_research_r94_strategy_mix.py`

**Grid weights**: (0.7,0.15,0.15), (0.6,0.2,0.2), (0.5,0.25,0.25), (0.8,0.1,0.1), risk_parity

**Acceptance**: Mix Sharpe > R68 × 1.05, Mix MaxDD < R68 MaxDD × 0.85

---

## R95 — Bootstrap Significance

**Файл**: `_research_r95_bootstrap.py`

Block bootstrap (B=10, N=1000). R94-best vs R68.

**Acceptance**: P(mix > R68) > 0.80, Median ΔSh > 0.08

---

## Порядок: R90 → R91 + R92 → R94 → R95

