# Consultation: Feature Engineering & Model Improvement — Crypto L/S System

## System Summary
Solo quant. Systematic crypto L/S on OKX perps. 50 symbols, 3x leverage, 12h
rebalance. Target: predict 12h forward cross-sectional returns. 50 coins ranked
per timestamp, top-N long / bottom-N short.

Model: CatBoost (ordered boosting, huber delta=1.0, 5 seeds, walk-forward 3-fold).
Honest OOS (no leakage): val LS Sharpe ~1.1 for most recent period (H2 2025).

---

## What Features I Currently Have (~165–200 total)

### Per-coin technical (computed per symbol, then cross-sectionally ranked):
- **Returns**: ret_1h, ret_2h, ret_4h, ret_12h, ret_24h, ret_48h, ret_168h
- **Volatility**: GK volatility (gk_vol_24h, gk_vol_48h, gk_vol_168h), ret_std_24h,
  ret_std_48h, ret_std_168h, ATR (atr_14, atr_24, atr_48)
- **Momentum**: mom_12h_zscore, mom_3d, mom_7d, mom_accel_12h, ret_sharpe_168h,
  ret_skew_24h, ret_skew_168h
- **Price structure**: close_ma6/12/24/48/96/168/336/504/720_ratio, bb_high/mid/low
  (Bollinger 24h, 48h), RSI (14, 24, 48), VWAP_12h_dist
- **Volume**: vol_ma12/24/48/96/168_ratio, vol_trend_12_48, vol_12h_cs_rank,
  vol_surge_12/24/48h, range_expansion_12h
- **Special 12h**: vwap_12h_dist, mom_12h_zscore, ret_12h_cs_rank

### Cross-asset (market-level, NOT ranked):
- BTC returns: btc_ret_1h/4h/12h/24h/48h/168h
- ETH returns: eth_ret_1h/4h/12h/24h
- BTC regime: btc_regime_24/72/168 (above MA), btc_vol_24h
- ETH/BTC ratio: eth_btc_ret_24h, eth_btc_ratio
- Market dispersion: market_dispersion (std of ret_1h across coins)
- Per-coin vs BTC: ret_vs_btc_24h, btc_beta_168h

### Regime (market-level, NOT ranked):
- BTC MAs: regime_btc_above_ma336/720, regime_btc_ma720_slope
- BTC drawdown from 720h high: regime_btc_dd_720, regime_btc_not_crashed
- Volatility regime: regime_low_vol
- Breadth: breadth_pct_positive, regime_breadth_bullish
- Composite: regime_composite (weighted sum of above)

### Calendar:
- Cyclical: cal_hour_sin/cos, cal_dow_sin/cos, cal_month_sin/cos
- Binary: cal_is_us_session, cal_is_weekend
- Expiry: cal_days_to_monthly_expiry (distance to last Friday of month)

### Macro (daily → hourly ffill, market-level, NOT ranked):
- Raw: vix_close, spx_close, dxy_close, gold_close, yield_10y_close, hy_spread,
  breakeven_10y, yield_curve_10y2y, fed_funds_rate
- Changes: all of above at 1d/5d/20d
- Z-scores (20d rolling): all of above
- Interactions: risk_aversion (vix_z + hy_spread_z), risk_on_off_ratio, real_rate,
  risk_on_composite

### Sentiment (market-level + per-coin):
- FNG: fng_value, fng_extreme_fear/greed, fng_ma7/30, fng_momentum
- Funding (OKX): funding_rate (per-coin), market_avg_funding, market_funding_std,
  market_funding_skew, funding_vs_market (per-coin deviation from market)
- Long/short ratio: long_short_ratio (per-coin)
- Cross-coin dispersion: cross_coin_dispersion, dispersion_regime

### Derivatives — Binance Futures (optional config, per-coin):
- OI: oi_value_usd, oi_change_1/4/12/24h, oi_zscore_7d, oi_change_12h_cs
- OI interactions: oi_ret_interaction (1h), oi_ret_interaction_12h
- Taker: taker_buy_sell_ratio, taker_imbalance, taker_imbalance_cs
- Top trader L/S: top_ls_ratio, top_long_pct, ls_divergence
- Binance funding: funding_rate_binance, funding_surprise, funding_dispersion
- Market-level: agg_oi_change_12h, agg_taker_imbalance, agg_oi_total_change_12h
- Basis/premium: basis_pct, basis_cs_rank, basis_funding_divergence
- Liquidations: liq_long_usd, liq_short_usd, liq_total_usd, liq_imbalance,
  liq_ret_interaction, agg_liq_zscore

### News (optional config):
- Per-coin: coin_news_count, coin_news_sentiment (if mode=all)
- Market-level: total news count, market avg/std sentiment (market_avg_sentiment,
  market_funding_std)
- Finding: market-level news HELPS, per-coin news HURTS

---

## What I'm Planning to Add (Already in My Pipeline)

### Imminent (P0):
1. **Deribit DVOL** (BTC+ETH options implied volatility):
   - dvol_btc, dvol_btc_change_12h/24h, dvol_btc_z_30d/60d
   - dvol_eth, dvol_spread, dvol_term_ratio, dvol_vol_of_vol
   - Hypothesis: IV spike = regime warning, reduce exposure

2. **TS Z-scores** (per-coin vs its own history, no CS rank):
   - ts_z_funding_60d, ts_z_basis_60d, ts_z_oi_change_7d
   - ts_z_liq_total_30d, ts_z_ret_12h_180d, ts_z_volume_30d
   - Hypothesis: different coordinate system from CS ranks = breaks correlation ceiling

3. **Explicit interaction features**:
   - oi_contra_price = oi_change_12h × (−ret_12h) (squeeze signal)
   - crowded_trend = funding_extreme × trend_strength
   - liq_vol_interaction = liq_imbalance × gk_vol_24h
   - basis_funding_divergence (already have this one)

### Later (P2):
4. **CoinMetrics on-chain** (BTC/ETH): exchange flows, active addressess, NVT
5. **Orderbook depth**: spread, depth imbalance, price impact
6. **Multi-timeframe**: 4h candles resampled from 1h

---

## What I've Learned That Constrains Suggestions

### Things I Tested That Didn't Work:
- Signal smoothing (EMA on predictions) — monotonically hurts. Predictions are
  already precise, smoothing dilutes them
- Hysteresis — zero effect at 12h rebalance. Top coins barely change bar-to-bar
- Vol targeting (external) — hurts. Model already "sees" volatility through 15+
  vol features
- Per-coin news sentiment — hurts. Model can't use noisy coin-level signals
- Derivatives features in the current "weak" regime (WinC) — became noisy,
  no-deriv model outperformed with-deriv in H2 2025 val

### Things That Work:
- Inverse-volatility position sizing (--vol-size) — only execution flag that
  improved both return AND Sharpe
- Short alpha is real and strong — cutting shorts destroys performance
- Market-level features > per-coin features for sentiment
- CatBoost > LGB in regime changes (more robust to distributional shift)

### Known Limitations:
- Training data: 2018→present for full window. H2 2025 performance weaker
  (LS Sharpe ~1.1 vs ~2.3 in earlier periods)
- The WinC degradation likely due to: bull rally chaos in May–Jun 2025, with
  low cross-sectional dispersion (everything moved together)
- ICIR stays high (consistency of ranking is fine) but LS Sharpe drops
  (spread between ranked returns collapses). Alpha magnitude problem, not signal quality

---

## Questions

**1. Feature categories I'm missing entirely**

Looking at my feature list, what important categories are completely absent?
I see I have: price/vol/momentum, cross-asset (BTC/ETH), regime, macro, sentiment
(FNG/funding), derivatives (OI/taker/liquidations), calendar.

What's notably missing from a cross-sectional crypto L/S system perspective?
Candidates I can think of but haven't pursued:
- On-chain metrics beyond what I have
- Social/alternative data (I have news but no Twitter/Reddit volume)
- Microstructure (I have cross-coin but no per-coin, no orderbook)
- Relative value (e.g., sector rotation within crypto — DeFi vs L1s vs L2s)
- Seasonal / structural patterns specific to crypto (halving cycles, etc.)

Which of these are worth pursuing vs likely noise at my scale?

**2. The WinC dispersion collapse problem — feature engineering response**

The core problem: cross-sectional dispersion in forward returns collapsed in H1 2025,
making all models weaker. The model correctly ranks coins but the spread is tiny.

Given this structural problem, what feature engineering approaches could help?
Some ideas:
- Features that predict when dispersion will be HIGH (target periods when alpha
  is actually exploitable)
- Sector/subgroup features (rank within DeFi coins, within L1s, etc.) — maybe
  relative ranking within subcategories is more stable than global ranking
- Volatility-adjusted returns as target instead of raw returns
- Features that capture "idiosyncratic" vs "systematic" moves explicitly

**3. Target engineering**

Currently I train on raw 12h forward cross-sectional returns (rank IC loss).
What alternative targets are worth exploring?

Options I know about:
- Excess return vs BTC (already have as alternative)
- Vol-normalized return (return / rolling_vol) — reward persistence, penalize spikes
- Binary target (will this coin be in top/bottom X%?) — classification instead
- Residualized return (regress out BTC market factor, model idiosyncratic only)
- Smoothed/blended: 50% 12h + 50% 24h (v7 already does this, works OK)

Which of these is most likely to improve robustness in varying regimes?
Is residualization (extract idiosyncratic alpha) theoretically the right approach
when cross-sectional dispersion collapses?

**4. Training objective engineering**

CatBoost uses YetiRank (LambdaRank variant) or huber regression for ranking.
What about:
- Custom loss that directly optimizes LS portfolio Sharpe (not just IC)?
- Weighting recent data more (exponential sample weighting by recency)?
- Two-stage: first classify direction, then rank magnitude?
- Log utility optimization (suggested by researcher colleague)?

Are any of these practically better, or do they introduce more variance than they
reduce? My experiments show training metrics don't reliably predict sim performance,
which suggests the training objective → sim gap is large.

**5. Feature selection approach**

With 165–200 features and only ~50K usable training rows per fold (excluding
lookback), I likely have a curse-of-dimensionality problem.

Current approach: CatBoost built-in feature importance, manual inspection.
The top-30 features from my last training:
btc_vol_24h (4.0), close_ma6_ratio (2.7), close_ma336_ratio (2.7),
gk_vol_24h (2.4), vix_close_chg_1d (2.4), ret_std_48h (2.4), atr_48 (2.3),
close_ma720_ratio (2.2), gk_vol_168h (2.2), vol_price_corr_48h (2.1),
... (sentiment features appear only at rank ~28-30 with 1.0 importance)

The model heavily uses: vol/momentum features and mostly ignores derivatives
and sentiment. Is this expected for cross-sectional 12h systems?

What structured feature selection approach would work best:
- SHAP-based recursive elimination?
- Information-theoretic selection (mutual information vs target)?
- Hold-out IC (does this feature improve IC on val when added/removed)?
- Just trust CatBoost's built-in importance (L2-leaf regularization)?

**6. The "market-level dispersion" feature idea**

My system already computes `cross_coin_dispersion` (std of ret_1h across coins
per timestamp) and `dispersion_regime` (ratio to its MA).

The netsci researcher suggested using realized cross-sectional return dispersion
as a live trading gate. But it could also be a training feature:
- If I include current dispersion level as a feature, could the model learn
  to be more aggressive when dispersion is high and more conservative when low?
- Or would this leak future information (dispersion at t requires returns up to t)?
- What's the right way to add this: lag by 1 bar? Use MA of past 24h dispersion?

**7. Regime-conditional models**

Currently: single model for all regimes.

Alternative: train separate models for bull/bear/sideways regimes, then ensemble
with regime-conditional weights.

Given my data volume (~50K usable rows per fold), would regime-conditional
models have enough data to be useful, or would they overfit?

What's the practical threshold — regime needs at least N months of training data?
I have:
- Bull regimes in 2021, 2023-2024
- Bear/ranging 2022, early 2023
- Mixed 2025 H1 (rapid regime switches)

---

## Context for Practical Suggestions

- I code everything myself, no team
- Each experiment = ~1h training (CatBoost) to ~4h (LGB with full pipeline)
- I have: 1h OHLCV for 50 coins from 2018, funding rates from OKX and Binance,
  binance futures metrics (OI/taker/top-trader/liquidations), macro data (FRED/yfinance),
  FNG, news sentiment
- Feature additions need to be computed at bar open (no look-ahead); daily data
  is forward-filled
- OKX perps market, labels use mark price returns (not spot)
- My biggest constraint: limited variance budget. Each new feature that doesn't
  help dilutes the model. I need *high confidence* additions, not kitchen sink
