# Consultation: New Data Sources & Features for Crypto L/S System

## Role

You are a senior quantitative researcher specializing in alternative data for crypto trading. We need to identify the highest-value NEW data sources and features to break through our current performance ceiling. Analyze what we already have, what's missing, and propose concrete data acquisition + feature engineering plans — prioritized by expected alpha, availability, and implementation cost.

## System Overview

**What it does**: Hourly crypto long/short portfolio — 10 longs + 10 shorts from 50 coins, rebalanced every 12h on OKX (3x leverage).

**Signal**: Ensemble of 4 GBDT groups × 5 seeds = 20 models. Final score = mean → cross-sectional z-score → top/bottom 10.

**Key insight from experiments**: System is **representation-limited, not model-limited**. All 4 model groups see rank-normalized features → converge to identical predictions (correlation 0.93-0.97). LambdaRank, residual targets, meta-labeling all FAILED. New information (features from truly different data sources) is the primary lever for improvement.

**Current production**: Sharpe 8.43, Return +21.7%, Max DD -4.5%, Win Rate 63% on OOS (26 days).

## What We Already Have (Complete Feature Inventory)

### Raw Data Sources (all active, downloading)
| Source | API | Frequency | History | Status |
|--------|-----|-----------|---------|--------|
| **Binance Spot OHLCV** | CCXT (public) | 1h candles | Since 2021 | ✅ Active, 50 coins |
| **Binance Futures Metrics** | data.binance.vision (bulk CSV) | 5min → 1h | Dec 2021+ (1.8M rows) | ✅ Active |
| **Binance Funding Rates** | Binance fapi | 8h | Jan 2020+ (294K rows) | ✅ Active |
| **Binance Premium Index** | Binance fapi | varies | Available | ✅ Active |
| **Binance Liquidation Snapshots** | data.binance.vision | 1h aggregated | Available | ✅ Being downloaded now |
| **OKX Funding Rates** | OKX public | 8h | Available | ✅ Active |
| **OKX OI + Long/Short** | OKX public | 1h | ~100h only (rolling) | ⚠️ Short history, superseded by Binance |
| **Fear & Greed Index** | Alternative.me | Daily | Full history | ✅ Active |
| **CryptoCompare News** | CryptoCompare API | Hourly | ~950K articles | ✅ Active |
| **GDELT Political News** | GDELT DOC API | Hourly | Available | ✅ Active |

### Feature Groups Currently Used (~190 unique features)

**1. Price/Volume Technicals (~65 features)**
- Returns: ret_1h through ret_168h (9 windows)
- Candle patterns: close/open ratio, shadows, body
- MA ratios: close_ma{6,12,24,48,72,168,336,720}_ratio
- Volume momentum: vol_mom_{6,12,24,48}h, vol_surge_{12,24,48}h
- Garman-Klass vol: gk_vol_{12,24,48,168}h
- Rolling stats: ret_std/skew/kurt/mean/sharpe at {24,48,168}h
- VWAP deviation: {12,24,48}h
- Buy pressure, volume-price correlation

**2. Technical Indicators (~25 features)**
- RSI (6,12,14,24), MACD (macd/signal/diff), Bollinger Bands (20,48)
- ATR (14,24,48), ADX (+pos/neg), Stochastic K/D
- CCI (14,48), Williams %R, OBV ratios, MFI

**3. Cross-Asset / Market (~16 features)**
- BTC returns (1h-168h), ETH returns (1h-24h), BTC vol 24h
- ETH/BTC ratio momentum, market dispersion, ret_vs_btc_24h

**4. Regime (~12 features, preserved unranked)**
- BTC above MA{24,72,168,336,720}, BTC MA720 slope
- BTC not crashed (DD from 720h high), regime_low_vol
- Breadth (% positive coins), composite regime score

**5. Calendar (~9 features)**
- Hour sin/cos, DOW sin/cos, month sin/cos
- US session flag, weekend flag, days to monthly expiry

**6. Derivatives (~30 features)**
- OI: change at {1,4,12,24}h + zscore_7d + ret interactions + cross-sectional
- Taker: buy/sell ratio, imbalance, CVD {12,24}h, flow zscore + cross-sectional
- Top trader L/S: ratio, long_pct, change {12,24}h, zscore
- Global L/S: ratio, long_pct, divergence
- Funding: rate, surprise, dispersion
- Basis/Premium: pct, zscore_7d, change {12,24}h, cs_rank, funding divergence
- Liquidations: long/short/total USD, imbalance, cascade {12,24}h, L/I interactions
- Market-wide: agg_oi_change, agg_taker_imbalance, agg_liq_zscore

**7. Sentiment (~17 features)**
- Fear & Greed: value, extreme_fear/greed flags, MA7/MA30, momentum
- News per-coin: count {1h,24h,7d}, sentiment {1h,24h,7d}, momentum, volume zscore
- News market: count_24h, sentiment_24h
- Political: count_24h, sentiment {24h,7d}, shock, volume zscore
- Funding: rate, market avg/std/skew, vs_market

**8. v7-specific enhanced features (~8 extra)**
- Range position, VWPC, trend strength, vol crush ratio, direction quality
- Funding cumulative {24h,72h}, funding cs_rank

### Feature Normalization
**Critical**: ALL features (except regime/calendar) undergo **cross-sectional rank normalization** — percentile within each timestamp. This means temporal dynamics (trend, regime changes) are lost. The system only sees "is this coin's RSI higher than peers right now?" not "is RSI absolutely high?"

### What We Know Doesn't Work
- LambdaRank: ranking objective killed IC (0.111 → 0.006)
- Residual targets: cross-sectional rank already removes market beta
- Meta-labeling: needs 2000+ trades, we only get ~690
- News features in LightGBM: HURT performance by 36-47% (CatBoost handles them OK)
- XGBoost news interactions (23 nx_* features): produced worst model (Sharpe 0.97 vs 1.81)
- Adding more GBDT models does NOT help — they all converge to same predictions

## What We DON'T Have (Gaps)

### 1. On-Chain Data (NO coverage currently)
- Exchange inflows/outflows (BTC, ETH, stablecoins)
- Active address momentum
- NVT ratio (network value / transaction volume)
- Whale transaction count / large transfer alerts
- Stablecoin supply ratio (SSR)
- Miner metrics (hash rate, revenue)

### 2. Orderbook / Microstructure (NO coverage)
- L2 orderbook depth / bid-ask spread snapshots
- Orderbook imbalance (bid vs ask volume at N levels)
- Large order detection / whale walls
- Trade flow: aggressive buyer/seller classification (beyond taker ratio)

### 3. Social Media (NO coverage)
- Twitter/X crypto sentiment / volume per coin
- Reddit (r/CryptoCurrency, coin-specific subs) activity
- Telegram group activity
- YouTube / Google Trends

### 4. Macro / Cross-Market (NO coverage)
- VIX (equity volatility → risk sentiment)
- DXY (dollar strength → crypto headwind)
- US yield curve (2y-10y spread)
- Gold / S&P500 returns (risk-on/risk-off regime)
- FRED economic releases calendar

### 5. Multi-Timeframe (PARTIAL)
- We only use 1h candles. No 15min, 4h, or daily timeframes
- No intraday patterns (first/last hour of session effects)

### 6. Cross-Exchange (NO coverage)
- Binance vs OKX price spread (execution venue arbitrage signal)
- Volume distribution across exchanges
- Exchange-specific OI / funding divergence

## Constraints

1. **Free or cheap APIs preferred** — no $500/month Glassnode subscriptions
2. **Need sufficient history** — at least 6-12 months for model training (if possible, 2+ years)
3. **Hourly frequency** is fine — we rebalance every 12h
4. **50 coins coverage** — BTC/ETH-only data is OK for market-level features, but per-coin is better
5. **VPS resources**: 4 vCPU, 8GB RAM — storage not an issue
6. **GPU cluster (A100 80GB)** available for training overnight

## What I Want From You

1. **Rank the data gaps** by expected alpha contribution. Which missing data source would most improve a rank-based cross-sectional crypto strategy?

2. **For top 3 recommendations**, provide:
   - Specific free/cheap API or data source
   - How to download (endpoint, frequency, auth)
   - What features to compute from it
   - Expected impact hypothesis (WHY will this help?)
   - Risk/caveat (what could go wrong)

3. **Feature engineering ideas from EXISTING data** that we haven't tried:
   - Any cross-feature interactions worth building?
   - Any temporal features being killed by cross-sectional rank normalization?
   - Multi-timeframe features from our 1h candles?

4. **Anti-recommendations**: What popular "alternative data" sources are NOT worth pursuing for this specific strategy and why?

5. **Priority order**: If I can only implement 2 new data sources this week, which 2?
