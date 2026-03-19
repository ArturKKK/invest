# Consultation: Crypto Trading System — What's Next?

## Who I Am
Solo developer building a systematic crypto L/S trading system. Running on OKX perpetual futures, 50 symbols, ~$500 starting capital with 3x leverage. I've done 13+ rounds of experiments over the past week and need fresh eyes on my situation.

## System Architecture

### Data
- **Universe**: 50 crypto perpetual futures on OKX
- **Features**: hourly OHLCV → ~200 engineered features (price MAs, volatility, momentum, cross-asset, sentiment, derivatives, macro, news NLP)
- **Data range**: 2017–2026-03-07, 2.5M rows
- **Target**: cross-sectional rank of 12h forward returns (rank normalization)
- **Cost model**: 0.03% taker + 0.01% slippage + 0.005%/8h funding

### Models Tried
| Model | Type | Best Training Sharpe | Sim Performance | Status |
|---|---|---|---|---|
| LGB v5 | LightGBM, 4h target | 1.64 | breakeven | retired |
| LGB v6 | LightGBM, 12h target, 121 feat | 1.10 | +91.5% (5mo) | active |
| LGB v7 | LightGBM, blended 12h/24h | 0.87 | — | retired (corr 0.957 with v6) |
| LGB v8 | LightGBM, 8yr history, 5 WF | 0.68 | +6.4% (60d) | retired (old data hurts) |
| **CatBoost** | Ordered boosting, 12h, Huber | **1.78** (price-only) | **+131.5%** (5mo) | **champion** |
| XGBoost | GPU hist, 12h, Huber | 1.43 | — | tested, not champion |
| HIST | Transformer (H100), cross-sectional | IC 0.075 | — | retired (infra complex) |
| GRU | BiGRU temporal per-coin | IC 0.035 | — | retired (weak signal) |
| Meta-stack | Ridge/LGB on L0 predictions | — | +1.2% (30d) | tested, mostly neutral |
| Deriv-gate | LGB on derivative features, scales signals | — | 0% to negative | retired |

### Evaluation Framework
- **Training metric**: Walk-forward (2 windows), LS Sharpe net (long top-5, short bottom-5, with costs)
  - R1: train→2024-12, test Q4 2025
  - R2: train→2025-06, test Q1 2026
- **Sim metric**: `run_fast_sim.py` — realistic backtester with Binance/OKX spot data, proper costs, leverage, Kelly sizing, edge-boost
- **Key sim flags**: --leverage 3 --kelly 0.8 --edge-boost --no-deriv-gate --no-ddstop

## Full Experiment History (Condensed)

### Early Phase (v1–v5, Feb–Mar 2026)
- v1: total failure (IC 0.005, Sharpe -1.0) — time leakage, no cross-sectional normalization
- v2: breakthrough (Rank IC 0.031, LS Sharpe 3.87) — cross-sectional rank model works
- v3: best horizon = 4h, cross-asset features work, regime filter useless
- v4: LGB with feature selection (118→94 feat), Sharpe 4.00
- v5: added sentiment (FNG, funding, synthetic proxies) — marginal
- HIST transformer: IC 0.075 (2.6x LGB!) but hard to deploy, eventually retired
- v5→v6: aligned 12h target with 12h rebalance → dramatic improvement

### Model Engineering Phase (v6–v8, Mar 8–10)
- v6: simple 12h LGB, 121 features → Sharpe 1.12 (WF), +7.4% (60d sim)
- v7: blended target + HPO → Sharpe 1.17, corr 0.957 with v6 (barely different)
- v8: 8 years of data, 5 WF windows → Sharpe 0.68, 2023 window = -1.18 (old crypto data = poison)
- CatBoost added: ordered boosting handles noisy features better than leaf-wise LGB
- News A/B test: news HURT LGB (-36% DDStop), but HELP CatBoost (+41%)

### Ensemble + Meta Phase (Mar 10–12)
- Ensemble v6+v7+CB = slightly better than v6 solo (~+5% Sharpe)
- Meta-model (Ridge on L0 preds): Sharpe 2.85 on 30d live, but fragile
- Full pipeline bugfix: 53/160 features were zero-filled in production → 0 after fix
- deriv-gate model: neutral to harmful, retired

### Production Deploy + Problem (Mar 12–16)
- Deployed v6+v7+CB+XGB ensemble with meta-model and deriv-gate
- **Real trading result: -16% in production** despite 69% backtest win rate
- Root cause: weak per-trade edge (raw preds 0.50–0.58), v6↔v7 correlation 0.957 (no diversity), 3x leverage amplifies errors

### Research Windows Phase (v11–v13, Mar 17–18)
Systematic re-evaluation with fresh OOS periods (Q4'25 and Q1'26).

**v11 (18 experiments):**
- CatBoost = best single model (Sharpe 1.48 avg)
- Derivatives hurt LGB: 1.10 → 1.33 without derivs (+21%)
- 24h target dead (Sharpe 0.20), v7 weakest (0.87)
- v6 solo sim > 4-model ensemble

**v12 (9 experiments):**
- cb_price_only = new best training Sharpe ever (1.78!)
- Derivatives hurt ALL models (confirmed CB, XGB, LGB)
- Paradox discovered: better training → worse sim
  - v11 models (Sharpe 1.10+1.48) → +46.2% sim
  - v12 models (Sharpe 1.45+1.78) → +35-40% sim

**v13 (33 sims + 2 training): PARADOX RESOLVED**
- R1 (Q4'25): v11 > v12 — but this is period-specific
- R2 (Q1'26): v12 slightly > v11
- FULL (5 months): v12 WINS decisively

**FULL period sim results (Oct'25 – Mar'26, lev3, kelly 0.8, edge-boost):**

| Config | Return | HAC Sharpe |
|---|---|---|
| **cb_no_deriv solo** | **+131.5%** | **5.09** |
| cb_price_only solo | +103.7% | 4.19 |
| mix v11_v6 + v13_cbMKTnd | +93.3% | 4.01 |
| v12 v6po + cbpo | +94.8% | 3.96 |
| v11 v6 solo | +91.5% | 4.15 |
| v11 v6+cb ensemble | +90.1% | 3.99 |
| mix v11v6+v12cbpo | +87.1% | 3.71 |
| v11 3-model (v6+cb+xgb) | +73.4% | 3.48 |

### Additional insights from v13:
- DDstop is useless (0.0% difference)
- Edge-boost helps +1-2pp
- Ensembles consistently WORSE than solo CatBoost
- cb_no_deriv (training Sharpe 1.66, WITH news) → +131.5%
- cb_price_only (training Sharpe 1.78, NO news) → +103.7%
- **Training Sharpe does NOT predict sim performance**. News hurt training metric but help trading.

## Current State (March 19, 2026)

### What's working
- CatBoost solo with Huber loss, no derivatives, all news → **best ever** (+131.5% / 5 months)
- 3x leverage with Kelly 0.8 and edge-boost sizing
- 12h rebalance period
- Cross-sectional rank target on 12h forward returns
- ~200 features (price/vol/momentum/cross-asset/sentiment/news/macro)

### What's NOT working
- Ensembles (add noise, reduce edge)
- Derivative features (hurt all models)
- Meta-models (fragile, marginal lift)
- DDstop (does nothing)
- LGB v7 (too correlated with v6)
- 24h/4h targets (12h is optimal)

### What worries me
1. **Single model risk**: all eggs in CatBoost basket. If CatBoost signal degrades, I have nothing.
2. **Backtest ≠ production**: had -16% in production before. New research is more rigorous (5-month OOS), but real trading can still surprise.
3. **Training vs sim disconnect**: training Sharpe doesn't predict trading. Hard to know what's good without running full sim.
4. **Feature count**: ~200 features for 50 symbols × ~3 years training. Possible overfit despite cross-validation and Huber loss.

## Questions For You

### 1. Model Diversification
CatBoost solo is my champion but 1-model systems are fragile. What models should I consider that would be truly UNCORRELATED to gradient boosting? I tried:
- Transformer (HIST): high IC but hard to deploy, eventually retired
- GRU: weak per-coin signal
- Meta-stacking: marginal
What about:
- **Linear models** (Ridge/Elastic Net) as a stable baseline?
- **Neural ODE / temporal fusion transformers** for time-series?
- **Graph neural networks** for cross-coin correlation?
- Something else entirely?

### 2. The Training ≠ Sim Problem
My training metric (LS Sharpe on walk-forward) doesn't predict real trading performance. cb_no_deriv has LOWER training Sharpe (1.66) than cb_price_only (1.78) but +27% better in sim. Why? Possible explanations:
- News features add information that helps position timing but hurts cross-sectional ranking
- LS Sharpe measures top-N/bottom-N spread, but sim measures actual portfolio P&L with sizing
- Overfitting to ranking → underfitting to trading signal
What evaluation metric should I use instead of LS Sharpe to better predict real performance?

### 3. Feature Engineering
Top features are FNG sentiment (ma30, momentum, ma7), volatility, price MA ratios. Derivatives features (OI, funding, taker CVD) HURT despite seeming theoretically useful. Why? Should I:
- Drop derivatives entirely?
- Engineer them differently (e.g., cross-sectional rank of funding changes)?
- Use them only for risk overlay (not for alpha signal)?

### 4. Ensemble Architecture
My ensembles (average of model predictions) consistently UNDERPERFORM CatBoost solo. This is opposite to typical ML experience. Hypotheses:
- Low model diversity (all see same features, same target)
- Averaging dilutes CB's strong signal with weaker signals
- Position sizing doesn't benefit from averaging
What ensemble methods might actually help? Stacking? Switching? Signal gating?

### 5. Risk Management
- leverage 3x is optimal in backtest, but real trading lost -16%
- DDstop does nothing in backtest
- How should I think about position sizing, drawdown management, and regime detection?
- Should I use a separate model for "risk-on/risk-off" switching?

### 6. Crypto-Specific Ideas
- Order flow features (orderbook imbalance, whale alerts, mempool)
- On-chain metrics (exchange inflows, active addresses, NVT ratio)
- Basis/premium between spot and perps
- Liquidation cascade detection
- Any other alpha sources you'd recommend for crypto?

### 7. Production Architecture
Currently: single CatBoost model → 12h rebalance → 50 symbols → top/bottom N positions.
Is this too simple? Should I consider:
- Multi-timeframe (separate 1h and 12h models → combine)
- Symbol clustering (trade clusters, not individual coins)
- Regime-conditional models (bull/bear/range → different model)
- Rolling retraining (how often?)

## Technical Details

### CatBoost Training Config (champion)
```
- Ordered boosting, 12h target
- Huber loss (delta=1.0)
- 5 seeds, skip-HPO (default params!)
- ~195 features (no derivatives)
- News: all 8 per-coin + 7 market-level news features
- Walk-forward: 2 windows (R1: test Q4'25, R2: test Q1'26)
- Cross-sectional rank target
- Cost model: 0.03% taker + 0.01% slip + 0.005%/8h funding
```

### Sim Config (best)
```
- 50 symbols from Binance spot
- Capital $5000, leverage 3x
- Kelly fraction 0.8
- Edge-boost sizing: weight ∝ 1 + edge/P75, cap 4x
- 12h rebalance
- No deriv-gate, no DDstop
- Short blocked (long-only futures sim)
```

### Feature Groups
| Group | # Features | Importance |
|---|---|---|
| Price MA ratios | ~15 | HIGH (top-10) |
| Momentum | ~20 | HIGH |
| Volatility (GK, range, std) | ~15 | HIGH |
| Cross-asset (BTC beta, breadth) | ~10 | HIGH |
| Sentiment (FNG, funding) | ~15 | MEDIUM-HIGH |
| News NLP (per-coin + market) | ~15 | MEDIUM (helps CB, hurts LGB) |
| Macro (DXY, rates, gold) | ~38 | LOW-MEDIUM |
| Derivatives (OI, taker CVD, basis) | ~30 | NEGATIVE (hurts all models) |
| Session/calendar | ~5 | NEGLIGIBLE |

I want your honest assessment: what am I missing? What new approaches should I try? Are there fundamental flaws in my methodology? I'm especially worried about running production with a single CatBoost model.
