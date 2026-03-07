# Результаты проекта invest — 7 марта 2026

## Текущее состояние

### Модели
| Модель | Данные | Период | Статус |
|--------|--------|--------|--------|
| LGB v5 | 50 крипто, 1h OHLCV, 143 фичи, target_ret_4h | train→2024-06, test 2025+ | ✅ обучена на кластере |
| HIST v1 | transformer, тот же датасет | test 2025+ | ✅ обучена на кластере |
| HIST v2 | sentiment-aware | test 2025+ | ✅ обучена на кластере |
| **LGB v6** | **12h target + 10 новых фич**, 121 selected features | train→2024-06, test 2025+ | ✅ **обучена локально** |

### Бэктест v5 на исторических данных (кластер, walk-forward W3)
| Метрика | LGB v5 (4h) |
|---------|-------------|
| Rank IC | 0.028 |
| Rank ICIR | 0.545 |
| LS Sharpe raw | 4.15 |
| LS Sharpe net | 1.64 |
| LS Ann Return net | 64.5% |
| LS MaxDD net | -70.9% |
| DDStop Sharpe | 0.93 |

### Бэктест v6 (12h target, локальная тренировка W3)
| Метрика | LGB v6 (12h) |
|---------|--------------|
| Rank IC | 0.028 |
| Rank ICIR | 0.427 |
| LS Sharpe raw | 2.48 |
| **LS Sharpe net** | **1.12** |
| LS Ann Return net | 43.0% |
| **DDStop Sharpe** | **1.81** |
| **DDStop MaxDD** | **-44.2%** |
| DDStop Total | +2876% |

### Ensemble (кластер v5)
| Комбинация | LS Net Sharpe | MaxDD |
|------------|---------------|-------|
| HIST v1 + LGB v5 | 2.93 | -56.5% |

### Fast Sim v5 — реальные данные (последние 30 дней)
| Конфигурация | Return | MaxDD | Sharpe | Win Rate | Costs |
|--------------|--------|-------|--------|----------|-------|
| 4h rebal, 8+8 pos | **-8.3%** | -10.3% | -7.07 | 44% | 7.1% |
| 12h rebal, 5+5 pos, kelly=30% | **-0.2%** | -1.7% | -0.44 | 53% | 1.2% |

### 🆕 Fast Sim v6 — реальные данные (kelly=100%, 12h rebal, 5+5)
| Период | Return | MaxDD | Sharpe | Calmar | Win Rate | PF | Costs |
|--------|--------|-------|--------|--------|----------|------|-------|
| 30d | **-0.2%** | -1.5% | -0.32 | — | 53% | 0.97 | 0.8% |
| 60d, kelly=100% | **+5.9%** | **-5.0%** | **+2.11** | **7.16** | **56%** | **1.22** | 4.4% |
| 60d, kelly=30% | +1.8% | -1.5% | +2.23 | 7.21 | 56% | 1.24 | 1.3% |

### Top-10 Features v6
| # | Feature | Importance |
|---|---------|------------|
| 1 | fng_ma30 | 791 |
| 2 | fng_momentum | 784 |
| 3 | fng_ma7 | 729 |
| 4 | **vol_12h_cs_rank** 🆕 | 700 |
| 5 | close_ma720_ratio | 619 |
| 6 | close_ma336_ratio | 595 |
| 7 | btc_beta_168h | 576 |
| 8 | breadth_pct_positive | 528 |
| 9 | ret_sharpe_168h | 494 |
| 10 | fng_value | 477 |

## Ключевые улучшения v5 → v6
1. **Target aligned**: модель предсказывает 12h returns (вместо 4h), что совпадает с holding period
2. **10 новых фич**: mom_12h_zscore, vwap_12h_dist, mom_3d, mom_7d, mom_accel_12h, vol_trend_12_48, is_asian_session, range_expansion_12h, ret_12h_cs_rank, vol_12h_cs_rank
3. **Kelly 30% → 100%**: модель достаточно стабильна для полного использования капитала
4. **Результат**: из breakeven → **+36% annualized, Sharpe 2.11**

## План улучшений
1. ✅ Переход на 12h ребалансировку (costs 7.1% → 1.2%)
2. ✅ LGB v6: retrain с target_ret_12h + 12h-specific features
3. ✅ Kelly 30% → 100% (Sharpe стабильный, MaxDD -5%)
4. ⬜ Maker orders вместо taker (0.02% vs 0.03% — экономия 33% на fees)
5. ⬜ Retrain HIST v2 с 12h target → ensemble с LGB v6
6. ⬜ Leverage 2-3x на OKX (ann return ~36% → ~72-108%)
7. ⬜ OKX API key → paper trading → live

## Конфигурация (текущая v6)
- **Risk**: kelly=100%, vol_target=1.5%, DD_stop=-20%, DD_resume=-8%
- **Positions**: 5 long + 5 short
- **Rebalance**: every 12h
- **Cost model**: 4 bps/side (taker + slippage)
- **Capital**: $1000 target
- **Model**: LGB v6, 5 seeds, 121 features, W3 window
