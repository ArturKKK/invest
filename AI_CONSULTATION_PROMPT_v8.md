# Consultation: Rolling vs Expanding Training Window for Crypto L/S System

## System Summary
Solo quant. Systematic crypto long/short on OKX perpetual swaps. 49 coins, 3x leverage, 12h rebalance. Target: predict 12h forward cross-sectional returns. Coins ranked per timestamp, top-10 long / bottom-10 short, inverse-vol position sizing.

Model: CatBoost (ordered boosting, Huber loss delta=1.0, 5 seeds averaged, walk-forward 3-fold validation). Training data starts Dec 2021.

**Currently deployed live with real money ($100, 3x leverage).**

---

## The Question: Training Window Length

Currently I use an **expanding window** — all data from Dec 2021 to `train_end`. As time goes on, the training set grows. My most recent production model was trained with `train_end=2026-02-01` (~4 years of data).

I want to explore whether a **rolling (fixed-length) training window** would work better — for example, using only the most recent 12–18 months of data instead of the full 4+ years.

### Why I'm Thinking About This

1. **Crypto regimes change fast.** The 2022 bear market, 2023 recovery, 2024 bull run, and 2025 consolidation are very different environments. Old data may dilute current patterns.

2. **Evidence that more data ≠ better:** When I tried expanding training to 2017+ (8 years), it was a disaster — WinC (2023 test) had Sharpe -1.18. The signal from 2017-2020 was actively harmful. Old data literally poisoned the model.

3. **My H2 2025 performance degraded** (LS Sharpe ~1.1 vs ~2.3 in earlier periods). This could partially be explained by the expanding window: as the training set grows, the model gives equal weight to old and new patterns, even when the market has structurally changed.

4. **I already "discovered" this intuitively** — the idea of "rolling training window (last 18-30 months) instead of expanding" was noted as a TODO in my project context file but never implemented.

### My Current Walk-Forward Setup (for simulation/backtesting)

Three validation windows (each ~6 months OOS):
- **WinA**: train_end=2023-06-30, val=2023-08-07→2024-01-01 
- **WinB**: train_end=2024-01-01, val=2024-02-08→2024-06-29
- **WinC**: train_end=2024-06-29, val=2024-08-06→2025-04-30

With an expanding window:
- WinA training: ~18 months (Dec 2021 → Jun 2023)
- WinB training: ~24 months (Dec 2021 → Jan 2024)
- WinC training: ~30 months (Dec 2021 → Jun 2024)

So WinA is already effectively a "short window" model, while WinC has 2.5 years of data.

### What I Have NOT Tested Yet
- Rolling window (e.g., always last 15 months regardless of train_end)
- Different window lengths (6, 9, 12, 15, 18, 24 months)
- Combining expanding + rolling (ensemble of both)
- Weighted samples (exponential decay giving recent data higher weight)

---

## My Current Results (Champion = CatBoost Huber, expanding window)

### mega_comparison3 (183 simulations, exhaustive A/B testing):
- **cb_solo_huber**: HAC avg +7.77 across 3 windows (champion)
- Across windows: WinA ~6.9, WinB ~9.2, WinC ~7.2

### Individual window OOS metrics (CatBoost Huber, 3x leverage, vol-size):
| Window | LS Sharpe | CAGR | MaxDD | IC (rank) |
|--------|-----------|------|-------|-----------|
| WinA   | ~2.1      | ~35% | ~8%   | 0.045     |
| WinB   | ~2.8      | ~55% | ~5%   | 0.052     |
| WinC   | ~1.1      | ~15% | ~12%  | 0.041     |

WinC is notably weaker — the question is whether a shorter training window could help specifically for recent market regimes.

---

## Technical Details

### Model Architecture
- CatBoost with `loss_function='Huber:delta=1.0'`
- ~200 features (technical, cross-asset, regime, macro, sentiment, derivatives, calendar)
- Cross-sectional ranking of per-coin features at each timestamp
- 5 random seeds → simple mean ensemble
- Purge gap: 8 days between train and val

### Feature Categories (~200 total)
- Per-coin technical: returns, volatility (GK, ATR), momentum, price structure (MA ratios, RSI, BB), volume patterns
- Cross-asset: BTC/ETH returns + regime, ETH/BTC ratio, market dispersion
- Regime: BTC drawdown, MA slopes, breadth, volatility regime
- Calendar: hour/dow/month cyclical, US session, weekend, expiry
- Macro: VIX, SPX, DXY, gold, yields, HY spread (daily → ffill)
- Sentiment: Fear & Greed, funding rates, long/short ratios
- Derivatives: OI, taker flow, top trader L/S, basis, liquidations
- DVOL: BTC/ETH implied volatility from Deribit

### Data Available
- Hourly OHLCV: Dec 2021 → present (~4.5 years)
- Macro data: similar coverage
- Derivatives: mid 2023 → present (~3 years)
- DVOL: mid 2024 → present (~2 years)

### Simulation Infrastructure
- Walk-forward: 3 windows (each ~6 months OOS)
- HAC (Heteroscedasticity and Autocorrelation Consistent) returns as main metric
- Standardized A/B comparison script: run identical model with different configs across all 3 windows

---

## Specific Questions

### 1. Rolling vs Expanding: What Does the Literature Say?

For gradient-boosted tree models on non-stationary financial time series:
- Is there empirical consensus on rolling vs expanding windows?
- What are the theoretical arguments for each?
- Is crypto's non-stationarity worse or different than equities?
- What window length is typical for similar crypto ML systems (if any research exists)?

### 2. What Window Lengths Should I Test?

Given my data (Dec 2021 → present, 4.5 years):
- What lengths are worth testing? I'm thinking: 9, 12, 15, 18, 24 months + expanding (full history) = 6 configurations
- Should I test this across my existing 3 walk-forward windows?
- How do I handle the fact that short windows may have insufficient data for derivatives/DVOL features (which start later)?
- Is 9 months too short for 200 features? (overfitting risk from small N)

### 3. Sample Weighting as an Alternative

Instead of hard-truncating old data, I could use sample weights:
- Exponential decay: `weight = exp(-λ × months_ago)`
- Linear decay: `weight = max(0, 1 - months_ago / max_months)`
- CatBoost supports `sample_weight` — easy to implement

Is this better than hard truncation? What decay rates / half-lives are worth testing?
Should I combine weighted samples with a window cap?

### 4. Ensemble of Multiple Window Lengths

Rather than picking ONE optimal window, I could:
- Train 3 models: 12-month window, 18-month window, expanding window
- Average their predictions (like I average seeds)

Does this add value or just add complexity? In equity quant, multi-horizon ensembles sometimes work — but is this different?

### 5. Practical Experiment Design

I want to run this on my VPS cluster. What's the most efficient experiment?

Option A: Simple — just test 5 window lengths × 3 WF windows = 15 simulations
Option B: Include sample weighting variants = 15 + ~10 weighting configs = ~25 sims
Option C: Also test ensembles = ~35 sims

Given each simulation takes ~12 min (train + simulate), what would you prioritize?

### 6. Interaction with Feature Count

With a rolling 12-month window on hourly data for 49 coins:
- Training samples: ~12 × 30 × 24 × 49 ≈ 425K rows
- Features: ~200

Is this enough for CatBoost? My expanding window at WinC has ~900K rows.
Could a shorter window benefit from feature selection to reduce dimensionality?

### 7. What Could Go Wrong?

What are the risks of switching from expanding to rolling?
- Survivorship/look-ahead bias in my WF windows with rolling?
- Loss of rare events (e.g., Luna crash in 2022) that teach the model about tail risks?
- Variance increase from smaller training sets?
- Is there a "U-shaped" curve where medium windows outperform both short and long?

---

## What I'm Looking For

1. **Clear recommendation**: should I prioritize rolling window experiments, sample weighting, or both?
2. **Specific experiment plan**: what to test first, with expected number of simulations
3. **Hypotheses about optimal window**: given my WinC degradation, what window length is most likely to help?
4. **Red flags**: anything that could make this experiment misleading or waste of time
5. **Implementation hints**: any CatBoost-specific tricks for handling non-stationarity (e.g., `has_time=True`, `auto_class_weights`, `model_size_reg`)
