# RESULTS R48 — Running Log

> Auto-updated by monitoring agent during overnight run.
> Last updated: see timestamps inside.

---

## Status

| Phase | Script | Status | Time |
|-------|--------|--------|------|
| 0 | validation | 🔄 RUNNING (0.4) | 03:43 start |
| 1+2 | features | ⏳ WAITING | — |
| 3 | cost model | ⏳ WAITING | — |
| 4 | combo | ⏳ WAITING | — |

---

## Phase 0 — Validation

**Статус**: 🔄 выполняется (0.4 clip sensitivity)

### 0.2 — Paired Block Bootstrap

| Metric | Value |
|--------|-------|
| Baseline Sharpe (30f) | +1.683 |
| Candidate Sharpe (31f=+cg_taker_imb) | +1.953 |
| Observed Δ Sharpe | **+0.270** |
| P(Δ Sharpe > 0) | **69.6%** ❌ |
| 90% CI | [-0.818, +1.559] |
| Median Δ | +0.349 |
| **Вывод** | **FAIL — ниже порога 80%** |

> ⚠️ CI включает 0 → improvement статистически нестабилен. Но медиана Δ=+0.35 говорит о положительной тенденции. Все фазы продолжаются.

### 0.3 — Monthly Stability

- Win months: **10/18 (56%)**
- Mean monthly IC: **+0.022** (9/13 positive)
- Top-2 months contribution: 57% (< 60% — ОК)
- **Вывод**: ✅ стабильность умеренная, нет явной концентрации в 1-2 месяцах

### 0.4 — Clip Sensitivity

*(ещё идёт)*

---

## Phase 1+2 — Taker Derivatives + Residualized Liq

*(будет заполнено по завершению)*

---

## Phase 3 — Hybrid Cost Model

*(будет заполнено по завершению)*

---

## Phase 4 — Best Combo

*(будет заполнено по завершению)*

---

## Final Champion

*(будет заполнено по завершению Phase 4)*
