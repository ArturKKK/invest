# Consultation: Crypto L/S System — Walk-Forward Results + Live Deployment Decision

## Who I Am
Solo quant dev. Systematic crypto long/short on OKX perpetual futures. 50 symbols,
3x leverage, 12h rebalance. ~2 years of research, 15+ experiment rounds.

---

## What the System Does

At each 12h bar:
1. Model predicts 12h forward rank-IC for each of 50 coins
2. Top-N coins → long, bottom-N coins → short (equal-weight within each leg)
3. Rebalance every 12h

Features: price/vol/momentum (ATR, BB, MA ratios), regime (breadth, trend), macro
(SPX, VIX, gold, DXY), cross-asset (BTC/ETH returns), sentiment (FNG, funding rate,
market-level news sentiment). ~165–200 features depending on config.

Models tested: LightGBM v6 (rank IC loss), LightGBM v7 (blended 12h+24h target),
CatBoost (ordered boosting, huber loss), XGBoost (with news interactions).

---

## Critical Fixed Bug: Data Leakage

Previous experiments had a **fatal flaw**: the validation set (used for early stopping
in LGB/CatBoost) overlapped with the test period. This gave inflated Sharpe ~5–6 in
backtest but model underperformed live. **This was the reason for live losses.**

Fix applied: train_end moved 2 months before test_start. Proper timeline:
- Train data: everything up to train_end  
- Val set: train_end+8d → train_end+2mo (for early stopping only, never in sim)  
- Test (OOS): train_end+2mo → train_end+8mo  

This is the first honest walk-forward test.

---

## Walk-Forward Experiment (3 windows, all OOS)

| Window | Train cutoff | Val period | Test period (OOS) |
|--------|-------------|------------|-------------------|
| A | 2024-04-30 | May–Jun 2024 | Jul–Dec 2024 (6 mo) |
| B | 2024-10-31 | Nov–Dec 2024 | Jan–Jun 2025 (6 mo) |
| C | 2025-04-30 | May–Jun 2025 | Jul–Dec 2025 (6 mo) |

Total honest OOS: 18 months.

Full simulation results not yet available (still running, ~183 sims × ~3 min).
What I have now: **training quality metrics on the val set**.

---

## Training Quality Results (LS Sharpe on val set)

LS Sharpe = Sharpe of L/S portfolio using model predictions as signal, computed on
the hold-out validation period *before* the test window. Measures raw signal quality
before any execution logic. ICIR = Information Coefficient IR (consistency of rank IC).

### Window A — val: May–Jun 2024

| Model | ICIR | LS Sharpe |
|-------|------|-----------|
| catboost_no_news | 0.487 | **2.37** |
| catboost_market_news | 0.490 | 2.35 |
| xgboost_with_news | 0.496 | 2.27 |
| catboost_with_news | 0.483 | 2.25 |
| catboost_huber | 0.489 | 2.25 |
| xgboost_no_news | 0.491 | 2.20 |
| v6_base | 0.485 | 2.15 |
| v7_base | 0.490 | 2.12 |
| v6_with_news | 0.482 | 2.04 |
| v7_with_news | 0.489 | 1.98 |
| catboost_huber_no_deriv | 0.476 | 1.80 |
| catboost_no_deriv | 0.467 | 1.73 |
| v6_no_deriv | 0.459 | 1.65 |

### Window B — val: Nov–Dec 2024

| Model | ICIR | LS Sharpe |
|-------|------|-----------|
| catboost_no_news | 0.472 | **2.46** |
| catboost_with_news | 0.478 | 2.40 |
| catboost_huber | 0.483 | 2.36 |
| xgboost_no_deriv | **0.510** | 2.11 |
| xgboost_no_news | 0.497 | 2.19 |
| v6_base | 0.479 | 2.05 |
| v7_base | 0.480 | 1.88 |
| catboost_no_deriv | 0.464 | 1.99 |

### Window C — val: May–Jun 2025

| Model | ICIR | LS Sharpe |
|-------|------|-----------|
| catboost_news_no_deriv | 0.465 | **1.19** |
| catboost_huber | 0.458 | 1.16 |
| catboost_with_news | 0.463 | 1.14 |
| catboost_no_deriv | 0.439 | 1.14 |
| catboost_no_news | 0.457 | 1.13 |
| catboost_market_news | 0.462 | 1.08 |
| v6_no_deriv | 0.477 | 0.93 |
| v7_no_deriv | 0.480 | 0.84 |
| v6_with_news | 0.484 | 0.82 |
| v7_with_news | 0.488 | 0.70 |
| v6_base | 0.483 | 0.57 |
| v7_base | 0.483 | **0.55** |

*(XGBoost WinC still training)*

---

## Key Observations

### 1. Massive signal degradation in Window C (H1 2025 train → H2 2025 test)

LGB models: LS Sharpe dropped from ~2.0 (WinA/B) to **0.55–0.93** (WinC). That's
a 3–4x collapse. CatBoost: 2.25–2.46 → **1.08–1.19** (still a halving, but much
less severe).

The val period for WinC is **May–Jun 2025** — this was a period of strong BTC rally
(BTC went from ~80K to ~100K+ after Trump "crypto-friendly" pivot). It's plausible
that 12h cross-sectional alpha collapsed during the bull run.

### 2. CatBoost consistently more resilient than LGB

In WinA, LGB was competitive with CatBoost. In WinC, CatBoost is 2x better
by LS Sharpe. CatBoost handles this regime change better. Possible reasons:
- Ordered boosting is more robust to distributional shift
- CatBoost tends to regularize more aggressively
- LGB v6/v7 may have overfit to a regime that ended in early 2025

### 3. ICIR stays high even when LS Sharpe collapses

In WinC: LGB ICIR = 0.483–0.488 (barely changed from WinA/B ~0.48–0.49). But
LS Sharpe dropped 4x. This means the model is making *consistent* but *small*
predictions. The magnitude of the alpha collapsed, not the consistency. 
The coin ranking is still mildly correct, but differences are tiny.

### 4. Removing derivatives helps more in WinC

For CatBoost: `catboost_no_deriv` = 1.14 vs base variants ~1.08–1.14.
In WinA/B: no-deriv variants were consistently **worse** by ~0.5 LS Sharpe.
This reversal suggests derivative features (OI, funding) became noisy signals 
in the regime change of 2025.

---

## My Current Live Setup

Right now: **not running** (paused after losses with old leaky models).
Previous live model: CatBoost solo, train_end=2025-01-31, but WITH the leakage
bug — so val overlapped test, model was indirectly overfit to recent data.
When I deployed live: lost real money (small amount). This was the leakage problem.

Now I have correct models (no leakage) but the WinC degradation makes me worried:
even honest models may have weak signal in the current market.

---

## Questions

**1. The WinC degradation**

LS Sharpe halved (CatBoost: 2.3→1.1, LGB: 2.1→0.6) in the H1 2025 val period.
Is this level of degradation (CatBoost 1.1, LGB 0.6) still worth deploying live?
What would you set as a minimum LS Sharpe threshold to greenlight a live deployment?
Is "ICIR stays high but LS Sharpe collapses" a well-known pattern, and what causes it?

**2. Regime detection for live gating**

How do practitioners detect whether you're in a "model works" vs "model breaks" regime
in real time? Ideas I'm considering:
- Monitor rolling 20-bar win rate drop below threshold → reduce size or pause
- DD-stop: cut to 0 if drawdown exceeds X%
- Signal dispersion gate: if P90-P10 of model predictions < threshold → don't trade
- Equity curve momentum filter (are we above N-bar rolling MA of equity?)

Which of these actually work? Are there smarter approaches?

**3. The conservative deployment option**

Option A: Deploy CatBoost now, full 3x leverage. Signal is weaker but still positive
(LS Sharpe ~1.1 = above zero). Live trading at reduced leverage > not trading.

Option B: Wait for full OOS simulation results (tonight/tomorrow), then decide.
If WinC OOS Sharpe > 1.0, deploy. If < 0.5, stay flat.

Option C: Deploy at 1x leverage only, scale up if live performance is good.

Which framework would you use here? Is "signal above zero = deploy at minimum size"
the right mental model, or is there a better decision rule?

**4. CatBoost vs Ensemble decision**

Previous experiments (with leakage, so data is suspect) showed:
- Solo CatBoost (cb_no_deriv): +131.5% backtest, HAC 5.09 — but leaked
- Now honest WinC val: CatBoost LS Sharpe ~1.1, LGB ~0.6

If CatBoost has 2x better signal than LGB in the current regime, should I just 
deploy CatBoost solo and forget about the ensemble? The whole point of ensemble
was diversification, but if LGB signal is near-zero, it would dilute a decent signal
with noise.

**5. Rolling retraining decision**

Currently: static model. As the test period progresses (Jul–Dec 2025), the model
was trained on data up to Apr 2025. By December the model is 8 months stale.

What's the retraining frequency decision rule? Monthly? Rolling 3-month window?
Event-driven (triggered by drawdown or performance drop)?

**6. The fundamental question**

Given:
- Honest OOS validation quality: CatBoost val LS Sharpe 1.1 (decent but not strong)
- Previous live losses were explained by data leakage (now fixed)  
- I don't know WinC OOS sim Sharpe yet (waiting for results)
- The market did something unusual in H1 2025 (strong BTC rally, altcoin season)

Is this a case of temporary regime change (deploy, ride it out) or structural 
alpha decay (find new features before deploying)?

What signals would tell you which it is, and how long to wait for them?

---

## System Specs for Context

- Data: CryptoCompare, 1h OHLCV, 50 perpetual coins, since 2018
- Features: ~165–200 depending on config
- Model: CatBoost, 5 seeds, ordered boosting, huber delta=1.0, walk-forward 3-fold CV
- Pipeline: no look-ahead (all features computed at bar open using prior data)
- Costs: 0.03% taker + 0.01% slip + 0.005%/8h funding
- Exchange: OKX, ~$10k account (testing scale, not production scale)
