# Consultation: How to Improve Crypto L/S Trading System

## Role

You are a senior quantitative researcher reviewing a crypto long/short trading system. We've exhausted several improvement ideas and need fresh directions. Analyze our complete experiment history, identify what went wrong, and propose the next set of concrete, actionable experiments — prioritized by expected impact and implementation complexity.

## Current System Overview

**What it does**: Hourly crypto long/short portfolio (10 longs + 10 shorts from 50 coins), rebalanced every 12h on OKX Demo ($5000 capital, 3x leverage).

**Signal generation**: Ensemble of 4 GBDT model groups × 5 seeds each = 20 models total:
- LightGBM v6 (165 features, regression on `target_ret_12h`)
- LightGBM v7 (159 features, regression on `target_ret_12h`)
- CatBoost (167 features, regression on `target_ret_12h`)
- XGBoost + News Interactions (188 features, regression on `target_ret_12h`)

Final score = simple mean of 4 group averages → cross-sectional z-score → top/bottom 10 → portfolio.

**Features** (~170 unique across groups):
- Price/volume technicals: MA ratios (6h-720h), volatility (Garman-Klass, ATR), momentum (ret_2h-168h), Sharpe ratios
- Cross-asset: BTC/ETH returns, BTC beta (48h/168h), market breadth, dispersion
- Regime: BTC drawdown from ATH, cross-coin correlation
- Derivatives: OI, funding rate, taker buy/sell ratio, long/short ratios, basis/premium (Binance Futures)
- Sentiment: Fear & Greed index (raw + MA7/MA30 + momentum), crypto news (coin-level + market-level counts/sentiment)
- News interactions: news × volume, news × volatility (XGBoost only)
- All features are cross-sectionally rank-normalized (percentile within each timestamp)

**Target**: `target_ret_12h` = 12-hour forward return per coin. Regression (MSE loss).

**Position sizing**: Edge-boost — position weight ∝ |z-score|. Stronger conviction → larger allocation.

**Risk controls**: Vol-targeting (30% annual), Kelly fraction (0.8), leverage 3x.

**Cost model**: Break-even = 9.5 bps per trade (8 bps round-trip + 1.5 bps funding).

## Complete Experiment History

All results are on honest out-of-sample period: Feb 9 – Mar 7, 2026 (26 days, ~51 rebalance periods). Train cutoff: Feb 1, 2026.
Sim: $5000, 3x leverage, edge-boost sizing, 12h rebalance, short-blocked (long-only market implementation).

### Baseline Performance (Generation 2 — current production on VPS)
| Metric | Value |
|--------|-------|
| Return | +21.7% |
| Sharpe | 8.43 |
| Max DD | -4.5% |
| PF | 2.21 |
| Win Rate | ~63% |

### Overnight Batch 1: Calendar, Multi-Horizon, MLP (13 Mar 2026)

| # | Experiment | Return | Sharpe | HAC Sharpe | Max DD | Verdict |
|---|-----------|--------|--------|-----------|--------|---------|
| 1 | **Gen#2 baseline (4-grp, no calendar)** | +21.7% | 8.43 | — | -4.5% | ✅ Current production |
| 2 | Gen#3 (4-grp, +9 calendar features) | +16.9% | 6.64 | 7.35 | -4.3% | ❌ Calendar hurts |
| 3 | Gen#3 no-cal retrain (4-grp, SKIP_CALENDAR) | +17.6% | 6.82 | 7.31 | -4.6% | ≈ baseline |
| 4 | **Gen#3 + LGB_24h (5-grp)** | **+18.5%** | **7.10** | **7.99** | **-4.2%** | ✅ Best new config |
| 5 | Gen#3 + LGB_4h (5-grp) | +16.1% | 6.28 | 6.90 | -4.4% | ❌ 4h hurts |
| 6 | Gen#3 + LGB_4h + LGB_24h (6-grp) | +15.9% | 6.18 | 6.81 | -4.4% | ❌ 4h drags down |
| 7 | MLP neural net (5-grp) | +14.8% | 6.59 | 7.12 | -4.1% | ❌ MLP is noise (IC=0.021) |

**Batch 1 conclusions:**
- Calendar features (day-of-week, hour-of-day, monthly expiry) = noise. Rejected.
- 24h horizon LGB = only useful new model. Lower correlation (r=0.93), solid IC=0.127.
- 4h horizon = too noisy (IC=0.112, worst of all models).
- MLP (3-layer, 128-64-32) = total failure. IC=0.021, pure noise.

### Overnight Batch 2: LambdaRank, Residual Target, Meta-Labeling (14–15 Mar 2026)

These 3 ideas came from a prior AI consultation (GPT Pro + Opus consensus).

#### A. LambdaRank (LGBMRanker + NDCG)
Idea: Replace MSE regression with LGBMRanker (LambdaRank loss) to directly optimize cross-sectional ranking.

| Model | Mean IC | Correlation with baseline |
|-------|---------|--------------------------|
| v6_rank | **0.006** | 0.038 |
| v7_rank | 0.034 | — |
| v6_base | 0.111 | — |
| v7_base | 0.117 | — |

**Result: ❌ TOTAL FAILURE.** IC collapsed from 0.111 → 0.006 (18× worse). Maximum diversity achieved (r=0.038 with baseline) but zero predictive power. LambdaRank/NDCG is fundamentally incompatible with this task — optimizing ranking of 50 coins at each timestamp destroys the signal that MSE regression captures.

#### B. Residual Target (ret − β×BTC)
Idea: Remove market component from target, train on idiosyncratic returns only.

| Model | Mean IC | Correlation with baseline |
|-------|---------|--------------------------|
| v6_resid | 0.106 | **0.965** |
| v7_resid | 0.114 | — |
| v6_base | 0.111 | — |
| v7_base | 0.117 | — |

**Result: ❌ USELESS.** r=0.965 with baseline = practically identical predictions. IC marginally worse. Zero diversity gain. Cross-sectional rank normalization already removes most market beta, so subtracting β×BTC changes almost nothing.

#### C. Meta-Labeling (Binary LGBMClassifier)
Idea: Train binary classifier on 23 meta-features (ens_mean, ens_std, rank agreement, confidence, vol context, BTC state) to filter profitable trades.

**Result: ❌ TOTAL FAILURE.**
- Only **690 trades** in OOS (340 train / 350 test) — critically insufficient
- CV best_iterations = [1, 1, 106], median = 1 → model found no signal beyond intercept
- Test WR = 48.6% (worse than coin flip)
- All thresholds 0.50-0.58 produce identical results, ≥0.60 → 0 trades
- Root cause: insufficient data. Need ≥2000+ trades (longer OOS window or retrain with earlier train-end)

#### D. Combo Ensemble (all models together)

| Metric | Combo (5-7 grp) | Base 4-grp |
|--------|-----------------|------------|
| Return | +16.8% | +17.6% |
| Sharpe | 6.85 | 6.82 |
| HAC Sharpe | 7.19 | 7.31 |
| Max DD | -3.6% | -4.6% |

No improvement. LambdaRank garbage poisons the ensemble.

#### Unique Contribution Analysis (IC drop when model removed from full ensemble)

| Model removed | IC drop | Interpretation |
|--------------|---------|---------------|
| v6_24h | **+0.0034** | ✅ Only unique contributor |
| v6_rank | **−0.0050** | ❌ Actively hurts |
| v7_rank | −0.0008 | Neutral |
| v6_resid | +0.0005 | Neutral |
| v7_resid | +0.0001 | Neutral |

### Model Correlation Matrix (Pearson, OOS predictions)
```
            lgb_v6  lgb_v7  catboost  xgboost  lgb_24h  lgb_4h
lgb_v6      1.000   0.972    0.972    0.964    0.934    0.905
lgb_v7      0.972   1.000    0.955    0.935    0.955    0.874
catboost    0.972   0.955    1.000    0.942    0.930    0.889
xgboost     0.964   0.935    0.942    1.000    0.896    0.884
lgb_24h     0.934   0.955    0.930    0.896    1.000    0.786
lgb_4h      0.905   0.874    0.889    0.884    0.786    1.000
```

Additional from Batch 2:
```
v6_rank ↔ v6_base: 0.038  (maximally diverse, but zero signal)
v6_resid ↔ v6_base: 0.965 (identical predictions)
v6_24h ↔ v6_base: 0.904   (good diversity/signal balance)
```

### Per-Model IC Summary (cross-sectional Spearman rank IC on target_ret_12h)
```
Model          Mean IC    Std IC    ICIR     N
catboost       0.1315     0.2129    0.618    625
lgb_v7         0.1273     0.2095    0.608    625
lgb_24h        0.1271     0.2261    0.562    625
lgb_v6         0.1255     0.2125    0.591    625
xgboost        0.1211     0.2097    0.577    625
v7_resid       0.1137     —         —        —
lgb_4h         0.1124     0.2030    0.554    625
v6_resid       0.1064     —         —        —
xgb_base       0.1054     —         —        —
v7_rank        0.0339     —         —        —
MLP            0.0206     —         —        625
v6_rank        0.0060     —         —        —
```

## Key Patterns & Failure Analysis

1. **Correlation ceiling (0.93-0.97)**: All GBDT regression models produce near-identical predictions despite different architectures (LGB/CB/XGB), feature sets, and hyperparameters. The only thing that reduced correlation was a different target horizon (24h → r=0.93).

2. **Diversity-signal tradeoff**: When we achieve true diversity (LambdaRank r=0.038), signal collapses. When we preserve signal (Residual r=0.965), we get zero diversity. 24h horizon is the only sweet spot found (r=0.90, IC preserved).

3. **Cross-sectional rank normalization dominates**: We rank-normalize all features within each timestamp. This means BTC-beta subtraction (residual target) has almost no effect — the ranking is already market-neutral.

4. **Neural nets fail**: MLP with 128-64-32 produced IC=0.021 (noise). Likely insufficient data for the model complexity, or wrong architecture for tabular cross-sectional data.

5. **Short OOS window**: 26 days (51 rebalance periods) limits what can be evaluated. Meta-labeling failed partly because only 690 trades available.

6. **What works**: Simple MSE regression with diverse target horizons. Everything else we tried made things worse or added nothing.

## What We Have NOT Tried Yet

From our internal discussion, candidate ideas that remain untested:
1. **Dead-zone target**: Drop |return| < 0.3-0.5% from training (reduce noise from flat periods)
2. **Huber loss / quantile regression**: Robust loss functions instead of MSE
3. **Turnover penalty in portfolio construction**: Penalize excessive position changes
4. **Dynamic N-positions**: Vary number of longs/shorts based on signal strength (instead of fixed 10/10)
5. **Beta-neutralization at portfolio level** (model-free, no need to change targets)
6. **Expand universe**: 50 → 100+ coins for more cross-sectional signal
7. **Feature interaction engineering**: Explicit pairwise interactions between top features
8. **Time-varying ensemble weights**: Weight models by recent IC performance
9. **Longer training window**: Currently train from 2024-07; could go back to 2017+ (tried for v8, was worse — but v8 also had other issues)

## Specific Questions

Given our complete experiment history and the failure patterns above:

1. **What should we try next?** Prioritize 3-5 concrete experiments by expected impact. For each, explain WHY you think it would work given our specific failure modes.

2. **Why did LambdaRank fail so badly?** Is this expected for cross-sectional crypto with 50 assets? Is there a correct way to use learning-to-rank for portfolio construction?

3. **How to actually break the correlation ceiling?** We tried: different GBDT frameworks (LGB/CB/XGB), different feature sets, different targets (MSE/ranking/residual), neural net (MLP), different horizons (4h/12h/24h). Only 24h horizon worked. What structural changes could genuinely produce diverse models that ALSO retain signal?

4. **Portfolio construction vs signal improvement**: Given that our signal IC ≈ 0.13 seems close to a ceiling for GBDT on this data, should we shift focus to portfolio construction (sizing, turnover, dynamic N, risk management)? What's the expected Sharpe improvement from better portfolio construction?

5. **Is 50 coins enough?** Would expanding to 100-200 coins help with cross-sectional signal? Or is the bottleneck something else?

## Constraints
- 50-coin universe (top liquid alts on OKX) — expandable to 100+
- Hourly data, 12h rebalance
- 3x leverage, ~9.5 bps break-even cost per trade (8 bps round-trip + 1.5 bps funding)
- OOS validation window: 26 days (can expand by retraining with earlier cutoff)
- Infrastructure: Python, LightGBM/CatBoost/XGBoost, PyTorch, GPU cluster for training
- VPS for live trading (limited compute, inference only)
- Currently profitable on VPS (Gen#2, Sharpe 8.43) — changes must not degrade production
