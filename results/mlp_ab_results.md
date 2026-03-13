# MLP A/B Test Results (2026-03-14)

## Setup
- OOS window: Feb 9 - Mar 7, 2026 (honest, train_end = Feb 1)
- Gen#3 models: all with calendar features
- MLP: AlphaMLP(128,64,32,32), dropout=0.44, 177 feats, 5 seeds

## _analyze_mlp.py Results

### Correlation Matrix (Pearson)
| | lgb_v6 | lgb_v7 | catboost | xgboost | mlp |
|---|---|---|---|---|---|
| lgb_v6 | 1.0 | 0.972 | 0.972 | 0.964 | **0.033** |
| lgb_v7 | | 1.0 | 0.955 | 0.935 | **0.034** |
| catboost | | | 1.0 | 0.942 | **0.031** |
| xgboost | | | | 1.0 | **0.028** |
| mlp | | | | | 1.0 |

**MLP vs GBDT ensemble: Pearson 0.032, Spearman 0.035** - virtually uncorrelated.

### Per-Model IC (OOS)
| Model | Mean IC | Std IC | ICIR | N_periods |
|---|---|---|---|---|
| lgb_v6 | 0.1255 | 0.2125 | 0.5905 | 625 |
| lgb_v7 | 0.1273 | 0.2095 | 0.6079 | 625 |
| catboost | 0.1315 | 0.2129 | 0.6177 | 625 |
| xgboost | 0.1211 | 0.2097 | 0.5773 | 625 |
| **mlp** | **0.0206** | 0.1794 | 0.1149 | 625 |

### Ensemble IC
| Config | Mean IC | ICIR |
|---|---|---|
| 4-group GBDT | **0.1281** | **0.6048** |
| 5-group +MLP | 0.1263 | 0.5964 |
| 80/20 blend | 0.1263 | 0.5964 |

## Backtest Comparison (OOS Feb 9 - Mar 7)
| Metric | 4-group (GBDT only) | 5-group (+MLP) |
|---|---|---|
| Return | **+16.9%** | +14.8% |
| Sharpe | **6.64** | 6.59 |
| Max DD | -4.3% | **-4.1%** |
| PF | 1.84 | 1.84 |
| Win Rate | 61% | 64% |

## Verdict: DO NOT add MLP to production

**Reasons:**
1. MLP IC = 0.021 - below 0.03 threshold, essentially noise
2. MLP-GBDT correlation = 0.03 - predicts something completely different (and wrong)
3. 5-group IC (0.1263) < 4-group IC (0.1281) - MLP dilutes good signal
4. Backtest: -2.1pp return, -0.05 Sharpe with MLP
5. Only upside: marginally lower DD (-4.1% vs -4.3%)

**Root cause hypothesis:** MLP RankIC loss + gradient-based feature selection
may not be suitable for this cross-sectional alpha signal where GBDT tree splits
naturally capture non-linear feature interactions better.
