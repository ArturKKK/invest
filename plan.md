# Plan: CG Alpha + Risk Overlay on top of R68

## Текущий статус (7 апр 2026)

| Phase | Script | Статус | Результат |
|-------|--------|--------|-----------|
| **Phase 0** | `_research_r80_cg_align.py` | ✅ DONE | Lookahead подтверждён — raw CG фичи непригодны, нужны z-scores/momentum |
| **Phase 1** | `_research_r81_vol_overlay.py` | ⬜ NEXT | Vol targeting + DD overlay, grid search |
| **Phase 2** | `_research_r82_cg_feature_factory.py` | ⬜ TODO | Z-scores, momentum, liq/oi/fr феатуры |
| **Phase 3** | `_research_r83_cg_ic_scan.py` | ⬜ TODO | IC-gate, redundancy check |
| **Phase 4** | `_research_r84_cg_addonly_wf.py` | ⬜ TODO | Add-only WF test, ΔSharpe gate |
| **Phase 5** | `_research_r85_bootstrap.py` | ⬜ TODO | Block bootstrap significance |

---

## Phase 0: R80 — результаты (✅ DONE)

**Файл**: `_research_r80_cg_align.py` — commit `d198488`

**Вывод: все raw CG фичи имеют lookahead при direct-merge:**

| Feature | Direct IC | Shift1 IC | Ratio | Вердикт |
|---------|-----------|-----------|-------|---------|
| `cg_liq_imb` | -0.333 | +0.002 | 145× | ⚠ LOOKAHEAD |
| `cg_taker_imb` | +0.276 | +0.005 | 61× | ⚠ LOOKAHEAD |
| `cg_oi_chg` | +0.347 | -0.020 | 17× | ⚠ LOOKAHEAD |
| `cg_fr` | +0.108 | -0.015 | 7× | ⚠ LOOKAHEAD |
| `cg_liq_log` | -0.048 | +0.008 | 6× | ⚠ LOOKAHEAD |
| `cg_ls_ratio` | -0.004 | -0.008 | 0.5× | ✓ ок |

Реальный сигнал = **rolling z-scores/momentum** (30-day window), а не сырые дневные значения.
Покрытие на тест-периоде: 99–100% по всем 6 фичам.

Артефакты:
- `results/r80_ic_table.csv` — полная IC-таблица
- `results/r80_summary.json` — метаданные + coverage
- `data/features/frame_12h_with_cg.parquet` — 1.8M × 129 cols (shift1)

---

## Phase 1: R81 — Vol Targeting + DD Overlay (⬜ NEXT)

**Файл**: `_research_r81_vol_overlay.py`

Идея: поверх R68-сигнала (4L/2S) добавить масштабирование позиций:
- `vol_t = std(net_ret_{t-L:t-1})`, L ∈ {20, 40}
- `vol_target = median(vol_t)` [A] или `percentile(25%)` [B]
- `scale_t = clip(vol_target / vol_t, s_min, s_max)`, сетка: `s_min ∈ {0.25, 0.35}`, `s_max ∈ {1.25, 1.50}`
- DD overlay: `dd > 10%` → `scale × 0.7`, `dd > 15%` → `scale × 0.5`

**Acceptance**: MaxDD ↓ ≥ 20% при Sharpe ≥ R68 baseline − 0.05 **ИЛИ** Calmar ↑ ≥ 20% при Sharpe ≥ baseline − 0.10

Артефакты: `results/r81_grid.csv`, `results/r81_best_equity.csv`

---

## Phase 2: R82 — CG Feature Factory (⬜ TODO)

**Файл**: `_research_r82_cg_feature_factory.py`

Фичи для IC-скрина (shift1, rolling 120 periods):
- **TAKER**: `taker_imb_z120`, `taker_flow_z120`
- **LIQ**: `liq_imb_z120`, `liq_int_z120`, `liq_spike`
- **OI**: `oi_z120`, `oi_surge`, `oi_notional_chg`
- **FUNDING**: `fr_z120`, `fr_accel`, `fr_accel_z120` ← НЕ заменять `cum_funding_24h`
- **LS RATIO**: `ls_z120`, `ls_chg_z120`

Coverage gate: < 0.95 на test-периоде → отбросить.

Артефакт: `data/features/frame_12h_with_cg_features.parquet`

---

## Phase 3: R83 — IC Scan + Redundancy Gate (⬜ TODO)

**Файл**: `_research_r83_cg_ic_scan.py`

- IC_ALL + per-window (W1/W2/W3) + `stability_score = n_windows_|IC|≥0.02 / 3`
- Redundancy: `|corr(F, cg_taker_imb)| < 0.7`, `|corr(F, cum_funding_24h)| < 0.7`
- Gate: `|IC_ALL| ≥ 0.03`, stability ≥ 2/3, coverage ≥ 0.95, redundancy < 0.7
- Top-2 по `IC_ALL × stability_score`

Артефакт: `results/r83_ic_table.csv`

---

## Phase 4: R84 — Add-only WF test (⬜ TODO)

**Файл**: `_research_r84_cg_addonly_wf.py`

- Exp1: 31f + best_CG (32f)
- Exp2: 31f + top-2 CG (33f) — только если Exp1 выиграл
- CS-rank: `*_chg/*_accel/*_surge/*_spike` — НЕ ранкировать; z-score/level — ранкировать
- **Acceptance**: ΔSharpe ≥ +0.10 к R68, **или** MaxDD ↓ ≥ 15% при Sharpe ≥ baseline − 0.05

Артефакты: `results/r84_summary.json`, equity/monthly/quarterly CSV

---

## Phase 5: R85 — Bootstrap significance (⬜ TODO)

**Файл**: `_research_r85_bootstrap.py`

- Block bootstrap: block=10, N=1000 ресэмплов
- Сравнить Sharpe distributions: R68 baseline vs best exp (R81 + R84)
- Принять если `P(Sharpe_exp > Sharpe_base) > 0.8` и `median ΔSharpe > 0.08`

---

## Output contract (единый формат для всех скриптов)
- `results/<RID>_summary.json` — sharpe, maxdd, calmar, turnover, cost%, winrate, per_window
- `results/<RID>_equity.csv` — timestamp, gross_ret, net_ret, cost, n_pos
- `results/<RID>_monthly.csv`, `results/<RID>_quarterly.csv`

---

## Baseline (R68): 4L/2S continuous WF
- Net Sharpe: **3.777**
- Производство: VPS `root@185.42.163.63`, capital=80, leverage=1
- 31 фича: CHAMPION_FEAT_31 = CHAMPION_FEAT_30 + ["cg_taker_imb"]
- 5 LGB + 5 XGB seeds, continuous windows W1/W2/W3

---

## Закрытые направления (не возвращаться)
- ❌ Temporal features (ret lags catastrophic)
- ❌ Meta-stacking, LambdaRank, reject option, uncertainty gating
- ❌ dynamic_K, edge_cost_filter, prob_weighting
- ❌ Raw daily CG values без z-score (lookahead подтверждён R80)

---

## Завершённые эксперименты (R60-R70)

### Ключевые результаты:

| # | Эксперимент | Net Sharpe | Вердикт |
|---|-------------|-----------|---------|
| R68 | **4L/2S continuous WF** | **3.777** | **CHAMPION — в проде** |
| R65 | Gross vs Net: 4L/2S | 2.984 | baseline |
| R67 | Reject option | 1.619 | Провал |
| R70 | LambdaRank | 0.796 | Провал |
| R69 | Percentile gating | 0.608 | Катастрофа |
| R62 | Meta-stacking | -0.38..1.48 | Провал |

---

## Технические заметки

**MLC:** invest-y5u733, Python 3.11, venv `/data/datasets/.venv`
**Запуск:** `mlc job exec invest-y5u733 -- bash -c 'cd /workdir/invest && /data/datasets/.venv/bin/python script.py > /data/datasets/log.log 2>&1 && echo DONE'`
**VPS:** `ssh -o "ProxyCommand=nc -X 5 -x 192.168.1.1:1080 %h %p" -p 22 root@185.42.163.63`



---

## Технические заметки

**MLC:** invest-y5u733, Python 3.11.15, pandas 2.3.3 (КРИТИЧНО — не обновлять!)
**Venv:** /data/datasets/.venv (абсолютный путь, symlink через .venv не работает в mlc exec)
**Запуск:** `mlc job exec invest-y5u733 -- bash -c 'cd /workdir/invest && /data/datasets/.venv/bin/python script.py > /data/datasets/log.log 2>&1 && echo DONE'`
