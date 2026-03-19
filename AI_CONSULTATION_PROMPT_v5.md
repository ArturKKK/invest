# Consultation: Crypto Trading System — After v14–v15 Experiments

## Context
Solo developer, systematic crypto L/S system. OKX perpetual futures, 50 symbols, 3x leverage, 12h rebalance.
15+ rounds of experiments. I previously consulted you (v4 prompt) and got great advice about vol targeting, hysteresis, derivatives overlay, signal smoothing, TS momentum sleeves. Now I tested most of it.

## What Changed Since Last Consultation

### v14: Training Variations (6 CatBoost experiments)

I tested CatBoost variations to beat the old champion (cb_no_deriv, +131.5%, HAC 5.09):

| Experiment | Config | Train Sharpe | Sim Return | Sim HAC |
|---|---|---|---|---|
| cb_noderiv_hpo | no derivs, 50 HPO trials | 1.77 | +121.2% | 4.90 |
| cb_noderiv_residual | no derivs, residual target (excess ret vs BTC) | 1.27 | +92.2% | 4.06 |
| cb_noderiv_hd05 | no derivs, huber delta=0.5 | 1.70 | +111.6% | 4.61 |
| cb_noderiv_hd15 | no derivs, huber delta=1.5 | **1.93** | +127.2% | 5.03 |
| cb_all_hpo | ALL features incl derivatives, 50 HPO | 1.62 | +128.4% | 5.29 |
| **cb_market_noderiv_hpo** | **market-only news, no derivs, 50 HPO** | **1.83** | **+143.8%** | **5.33** |

**NEW CHAMPION**: cb_market_noderiv_hpo. Key insight: per-coin news = noise, market-level news = signal.
Features: ~165 (removed: per-coin OI/funding/taker/basis/news, kept: market news count/sentiment, all macro, all price/vol/momentum).

(Training Sharpe still doesn't predict sim: delta=1.5 had 1.93 train Sharpe but only +127% sim.)

### v15: Execution Layer Optimization (46 sims, zero training)

Tested 6 existing sim flags that were NEVER used. All on new champion, FULL period (5 months, Oct'25–Mar'26):

**Single-flag results (vs baseline +143.8% / HAC 5.33)**:

| Flag | Return | HAC | Verdict |
|---|---|---|---|
| `--vol-target-ann 0.30` | +89.8% | 4.61 | **HURTS** |
| `--vol-target-ann 0.50` | +130.4% | 4.81 | **HURTS** |
| `--vol-target-ann 0.60` | +150.3% | 4.85 | Marginal, worse DD |
| `--hysteresis 3/5/7/10` | +143.8% | 5.33 | **ZERO EFFECT** (all 4 identical) |
| `--smooth-signal 0.2` | +129.3% | 5.16 | **HURTS** |
| `--smooth-signal 0.3` | +105.4% | 4.26 | **HURTS MORE** |
| `--smooth-signal 0.5` | +72.3% | 3.16 | **HURTS A LOT** |
| **`--vol-size`** | **+147.5%** | **5.48** | **ONLY WINNER (+3.7pp, +0.15 HAC)** |
| `--regime-shorts 0.5` | +56.7% | 2.47 | **CATASTROPHE** |
| `--regime-shorts 0.3` | +30.0% | 1.27 | **CATASTROPHE** |
| `--meta-risk` | +184.7% | 5.29 | More return, slightly worse HAC |

**What `--vol-size` does**: inverse-volatility position sizing. Low-vol coins get larger positions, high-vol get smaller. Pre-computed per-coin 24h rolling realized vol, then weight ∝ edge / coin_vol.

**What `--meta-risk` does**: composite risk scaler (0.3x–1.5x gross exposure) from 5 signals:
1. Model confidence (agreement across seeds) — **useless for solo CB** (constant 0.5)
2. Score spread (P90-P10 of predictions) — how differentiated are long vs short
3. Recent win rate (last 20 bars)
4. Current DD depth (deeper → reduce)
5. BTC regime (bull vs bear)

With solo CB: 4 of 5 signals active. Adds +41pp return but HAC drops by 0.04.

**Combos**: All combinations of flags (hysteresis+vol-target, triple combos, etc.) were WORSE than single flags. "Kitchen sink" was worst of all.

**Key surprises**:
- **Hysteresis = zero effect** at 12h rebalance. Top-10 coins barely change between bars.
- **Signal smoothing monotonically hurts**. Model predictions are already precise — EMA dilutes signal.
- **Vol targeting hurts both return AND HAC**. Model already accounts for volatility via features.
- **Short alpha is REAL and strong**. Cutting shorts destroys performance.

### v15 Leverage sweep (with hyst5 + vol-target 0.50)

| Leverage | Return | HAC | Max DD |
|---|---|---|---|
| 1x | +74.8% | **5.66** | -12.6% |
| 2x | +101.9% | 4.91 | -18.3% |
| 3x | +143.8% | 5.33 | -20.5% |
| 5x | +180.1% | 4.63 | -28.2% |

Risk-adjusted optimum = 1x. 3x is a return/risk compromise.

## Current Best Config
```
Model: CatBoost solo (Huber loss delta=1.0, HPO 50 trials, 5 seeds)
Features: ~165 (price/vol/momentum/cross-asset/regime/macro/FNG + market-only news)
  - NO per-coin derivatives (OI, funding, taker, basis)
  - NO per-coin news (count, sentiment by coin)
  - YES market-level news (total count, avg sentiment)
Sim: --leverage 3 --kelly 0.8 --edge-boost --vol-size --no-deriv-gate --no-ddstop
Result: +147.5% in 5 months, HAC 5.48, Max DD -20.4%
```

## What I'm Planning to Deploy
Adding `--vol-size` (inverse-vol sizing) to production. It's the only flag that improved both return AND HAC.

## What I Still Haven't Tried
1. **TS Momentum sleeve** (EMA 48h/192h, non-ML, separate risk budget) — you recommended this before
2. **Per-coin funding z-score as risk overlay** (reduce position when funding extreme) — data available (6 years Binance funding)
3. **Log utility as HPO objective** instead of LS Sharpe
4. **Multi-model allocation** — different alpha sleeves with risk-parity weighting
5. **Rebalance frequency** — never tested 6h or 24h
6. **Rolling retraining** — currently static model
7. **Deflated Sharpe Ratio** for multiple testing correction

## Questions

1. **Why does vol-targeting hurt?** You specifically recommended it. My theory: CatBoost already "sees" volatility through ~15 vol features (gk_vol, ret_std, atr, btc_vol, etc.), so external vol scaling double-counts. Is this plausible? Or is my implementation wrong (simple rolling std, clip 0.2–2.0)?

2. **Meta-risk: +41pp return but -0.04 HAC.** Should I deploy it anyway? The extra return is massive. With solo CB, only 4/5 signals work (no model agreement). Would adding model diversity (e.g., Ridge as second model) make meta-risk work better?

3. **After these results, what's ACTUALLY worth pursuing?** I've now tested: vol targeting (hurts), hysteresis (no effect), signal smoothing (hurts), regime shorts (catastrophe), turnover budget (buggy). The "obvious" execution improvements mostly failed. What's left?

4. **Rebalance frequency**: never tested 6h or 24h. 12h matches target horizon (12h forward return). Theory says shorter rebal = more signal but more costs. Worth testing?

5. **The per-coin news paradox**: per-coin news HELP in v13 (cb_no_deriv 131.5% has per-coin news) but HURT in v14 (cb_market_noderiv_hpo 143.8% drops them). The difference: v14 had HPO. Does HPO learn to overfit per-coin news? Or is it genuinely noise that HP tuning can't save?

6. **Position sizing refinement**: vol-size works. What other sizing methods should I test? Softmax temperature? Dynamic N (vary number of positions based on signal dispersion)?

7. **What would YOU do next if you were me?** Be specific and actionable. I've been running experiments for 2 weeks. Diminishing returns are setting in. Should I optimize further or go to production?
