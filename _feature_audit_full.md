# Complete Feature Audit — Invest Project

**Date:** 2026-04-02
**Source:** Code-level extraction from `run_trading.py`, `run_pipeline_v6.py`, `run_pipeline_xgboost.py`, `_ic_scanner.py`, `_research_r22_models.py`, `_research_r28c_forward.py`

---

## 1. FEATURES_23 (Currently Used by CLS Model)

```
ret_12h, ret_24h, ret_48h,
residual_12h, residual_24h,
mom_z_24h,
oi_chg_12h, oi_chg_24h, oi_zscore,
taker_cvd_12h, taker_cvd_24h,
ls_divergence,
atr_14, rvol_12h, gk_vol_24h, rvol_24h, iv_rv_spread,
pct_coins_up_12h, pct_coins_up_1h,
hour_sin, hour_cos, dow_sin, dow_cos
```
**Count: 23**

---

## 2. R28c CANDIDATES (Forward-Selected, 23 features tested one-at-a-time)

```
top_ls_ratio_zscore, global_ls_ratio_zscore, premium_zscore,
taker_zscore, oi_ret_diverge,
adx, rsi_14, bb_pband_20, mfi_14, ret_skew_24h, ret_kurt_24h,
ret_168h, ret_1h, ret_4h, dist_from_high_24h,
vol_of_vol, rvol_168h, vol_ratio_24h,
premium_zscore_12h, oi_velocity, taker_imb_z,
obv_ma_ratio_24, vwap_dev_24h
```
**Count: 23**

---

## 3. ALL FEATURES — By Source Function

### A. `build_features()` in run_trading.py — Core Per-Coin Features

#### Returns (9)
- `ret_1h`, `ret_2h`, `ret_4h`, `ret_6h`, `ret_12h`, `ret_24h`, `ret_48h`, `ret_72h`, `ret_168h`

#### Price Shape (7)
- `close_open_ratio`, `high_low_ratio`, `high_close_ratio`, `low_close_ratio`
- `upper_shadow`, `lower_shadow`, `body`

#### MA Ratios — close and volume (16)
- `close_ma6_ratio`, `close_ma12_ratio`, `close_ma24_ratio`, `close_ma48_ratio`, `close_ma72_ratio`, `close_ma168_ratio`, `close_ma336_ratio`, `close_ma720_ratio`
- `vol_ma6_ratio`, `vol_ma12_ratio`, `vol_ma24_ratio`, `vol_ma48_ratio`, `vol_ma72_ratio`, `vol_ma168_ratio`, `vol_ma336_ratio`, `vol_ma720_ratio`

#### Garman-Klass Volatility (4)
- `gk_vol_12h`, `gk_vol_24h`, `gk_vol_48h`, `gk_vol_168h`

#### Return Distribution (15)
- `ret_std_24h`, `ret_std_48h`, `ret_std_168h`
- `ret_skew_24h`, `ret_skew_48h`, `ret_skew_168h`
- `ret_kurt_24h`, `ret_kurt_48h`, `ret_kurt_168h`
- `ret_mean_24h`, `ret_mean_48h`, `ret_mean_168h`
- `ret_sharpe_24h`, `ret_sharpe_48h`, `ret_sharpe_168h`

#### Volume Features (11)
- `vol_mom_6h`, `vol_mom_12h`, `vol_mom_24h`, `vol_mom_48h`
- `vwap_dev_12h`, `vwap_dev_24h`, `vwap_dev_48h`
- `vol_price_corr_24h`, `vol_price_corr_48h`, `vol_price_corr_168h`
- `buy_pressure`

#### TA Indicators (27)
- `rsi_6`, `rsi_12`, `rsi_14`, `rsi_24`
- `macd`, `macd_signal`, `macd_diff`
- `bb_high_20`, `bb_low_20`, `bb_width_20`, `bb_pband_20`
- `bb_high_48`, `bb_low_48`, `bb_width_48`, `bb_pband_48`
- `atr_14`, `atr_24`, `atr_48`
- `adx`, `adx_pos`, `adx_neg`
- `stoch_k`, `stoch_d`
- `cci_14`, `cci_48`
- `willr_14`
- `mfi_14`

#### OBV (3)
- `obv_ma_ratio_12`, `obv_ma_ratio_24`, `obv_ma_ratio_48`

#### Cross-Asset (BTC/ETH) — from build_features (17)
- `btc_ret_1h`, `btc_ret_4h`, `btc_ret_12h`, `btc_ret_24h`, `btc_ret_48h`, `btc_ret_168h`
- `eth_ret_1h`, `eth_ret_4h`, `eth_ret_12h`, `eth_ret_24h`
- `btc_vol_24h`
- `market_dispersion`
- `ret_vs_btc_24h`
- `breadth_pct_positive`
- `regime_btc_above_ma720`, `regime_btc_dd_720`, `regime_btc_not_crashed`

#### FNG (from build_features) (6)
- `fng_value`, `fng_extreme_fear`, `fng_extreme_greed`
- `fng_ma7`, `fng_ma30`, `fng_momentum`

#### Synthetic Positioning (from build_features) (3)
- `reversal_4v24`, `reversal_12v48`, `reversal_24v168`

#### Volume Surge (from build_features) (2)
- `vol_surge_12h`, `vol_surge_24h`

#### BTC Beta (from build_features) (2)
- `btc_beta_48h`, `btc_beta_168h`

#### Macro/FRED (from build_features) (30+)
- Raw: `vix_close`, `spx_close`, `dxy_close`, `gold_close`, `yield_10y_close`, `hy_spread`, `breakeven_10y`, `yield_curve_10y2y`, `fed_funds_rate`
- Changes: `vix_close_chg_1d`, `vix_close_chg_5d`, `vix_close_chg_20d`, `spx_close_chg_1d`, `spx_close_chg_5d`, `spx_close_chg_20d`, `dxy_close_chg_1d`, `dxy_close_chg_5d`, `dxy_close_chg_20d`, `gold_close_chg_1d`, `gold_close_chg_5d`, `gold_close_chg_20d`, `hy_spread_chg_1d`, `hy_spread_chg_5d`, `hy_spread_chg_20d`, `breakeven_10y_chg_1d`, `breakeven_10y_chg_5d`, `breakeven_10y_chg_20d`, `yield_curve_10y2y_chg_1d`, `yield_curve_10y2y_chg_5d`, `yield_curve_10y2y_chg_20d`
- Z-scores: `vix_close_z20d`, `hy_spread_z20d`, `breakeven_10y_z20d`, `yield_curve_10y2y_z20d`
- Cross-interactions: `risk_aversion`, `risk_on_off_ratio`, `real_rate`, `real_rate_chg_5d`

**Subtotal from build_features(): ~152 features**

---

### B. `add_cross_asset_features()` in run_pipeline_v6.py (Enrichment)

Recreates/extends BTC/ETH features (overlaps dropped first):
- `btc_ret_1h`...`btc_ret_168h` (6), `eth_ret_1h`...`eth_ret_24h` (4)
- `btc_regime_24`, `btc_regime_72`, `btc_regime_168`
- `btc_vol_24h`
- `eth_btc_ret_24h`
- `market_dispersion`, `ret_vs_btc_24h`

**New unique: ~3** (btc_regime_24, btc_regime_72, btc_regime_168, eth_btc_ret_24h)

---

### C. `add_market_mode_features()` — Market Structure (8)

- `avg_corr_24h`, `avg_corr_72h`
- `pca1_share_72h`, `idio_fraction_72h`
- `beta_dispersion_48h`, `beta_dispersion_168h`
- `dispersion_ret_12h_lag1`, `dispersion_regime_12h`

---

### D. `add_liquidity_features()` — Liquidity Proxies (7)

- `dollar_volume_12h`, `dollar_volume_24h`
- `amihud_illiq_24h`
- `range_per_dv_24h`
- `vol_price_corr_48h` (overlap with build_features)
- `dollar_volume_24h_cs`, `amihud_illiq_24h_cs`

**New unique: ~6** (vol_price_corr_48h already exists)

---

### E. `add_advanced_regime_features()` — Regime Filters (8)

- `regime_btc_above_ma336`, `regime_btc_above_ma720`
- `regime_btc_ma720_slope`, `regime_btc_not_crashed`
- `regime_btc_dd_720`, `regime_low_vol`
- `breadth_pct_positive`, `regime_breadth_bullish`
- `regime_composite`

**New unique: ~4** (regime_btc_above_ma336, regime_btc_ma720_slope, regime_low_vol, regime_breadth_bullish, regime_composite)

---

### F. `add_calendar_features()` — Calendar/Seasonality (9)

- `cal_hour_sin`, `cal_hour_cos`
- `cal_dow_sin`, `cal_dow_cos`
- `cal_month_sin`, `cal_month_cos`
- `cal_is_us_session`, `cal_is_weekend`
- `cal_days_to_monthly_expiry`

---

### G. `add_12h_features()` in run_trading.py — 12h Holding Period (18)

- `mom_12h_zscore`, `vwap_12h_dist`
- `mom_3d`, `mom_7d`
- `mom_accel_12h`
- `vol_trend_12_48`
- `is_asian_session`
- `range_expansion_12h`
- `range_position_12h`
- `vwpc_12h`
- `hh_count_12h`, `ll_count_12h`, `trend_strength_12h`
- `vol_crush_ratio`
- `direction_quality_12h`
- `ret_12h_cs_rank`
- `vol_12h_cs_rank`
- `funding_cs_rank` (if funding_rate present)
- `cum_funding_24h`, `cum_funding_72h` (if funding_rate present)

---

### H. `add_sentiment_features()` — Sentiment/Alternative Data

#### FNG (already created, re-merged) (6)
- `fng_value`, `fng_extreme_fear`, `fng_extreme_greed`, `fng_ma7`, `fng_ma30`, `fng_momentum`

#### OKX Funding Rate (5)
- `funding_rate`
- `market_avg_funding`, `market_funding_std`, `market_funding_skew`
- `funding_vs_market`

#### Long/Short Ratio (1)
- `long_short_ratio`

#### Synthetic Positioning (from add_sentiment) (9)
- `reversal_4v24`, `reversal_12v48`, `reversal_24v168` (overlap with build_features)
- `vol_surge_12h`, `vol_surge_24h`, `vol_surge_48h`
- `cross_coin_dispersion`, `cross_coin_disp_ma24`, `dispersion_regime`

#### Return Skew CS (2)
- `ret_skew_48h_cs`, `ret_skew_168h_cs`

#### BTC Beta (overlap) (2)
- `btc_beta_48h`, `btc_beta_168h`

#### Per-Coin News (8)
- `news_count_1h`, `news_count_24h`, `news_count_7d`
- `news_sentiment_1h`, `news_sentiment_24h`, `news_sentiment_7d`
- `news_sentiment_momentum`, `news_volume_zscore`

#### Market-Level News (2)
- `market_news_count_24h`, `market_news_sentiment_24h`

#### Political News (5)
- `political_news_count_24h`, `political_sentiment_24h`
- `political_sentiment_7d`, `political_sentiment_shock`
- `political_news_volume_zscore`

#### News Flags (2)
- `news_coverage_ok`, `news_event`

#### Funding Surprise (1)
- `funding_surprise`

**New unique from add_sentiment: ~22** (excluding overlaps)

---

### I. `add_derivatives_features()` — Binance Futures Data

#### OI (8)
- `oi_value_usd` (raw)
- `oi_change_1h`, `oi_change_4h`, `oi_change_12h`, `oi_change_24h`
- `oi_zscore_7d`
- `oi_ret_interaction`, `oi_ret_interaction_12h`
- `oi_change_12h_cs`

#### Taker (6)
- `taker_buy_sell_ratio` (raw)
- `taker_imbalance`
- `taker_cvd_12h`, `taker_cvd_24h`
- `taker_flow_zscore`
- `taker_imbalance_cs`

#### Top Trader L/S (5)
- `top_ls_ratio`, `top_long_pct`
- `top_ls_change_12h`, `top_ls_change_24h`
- `top_ls_zscore`

#### Global L/S (3)
- `global_ls_ratio`, `global_long_pct`
- `ls_divergence`

#### Binance Funding (1)
- `funding_rate_binance`

#### Market-Wide Aggregates (4)
- `agg_oi_change_12h`, `agg_taker_imbalance`
- `funding_dispersion`, `agg_oi_total_change_12h`

#### Basis/Premium (7)
- `basis_pct`, `basis_zscore_7d`
- `basis_change_12h`, `basis_change_24h`
- `basis_cs_rank`
- `basis_funding_divergence`
- (premium_index dropped after)

#### Liquidation (11)
- `liq_long_usd`, `liq_short_usd`, `liq_total_usd`
- `liq_imbalance`
- `liq_cascade_12h`, `liq_cascade_24h`
- `liq_imbalance_12h`
- `liq_total_zscore`
- `liq_ret_interaction`
- `agg_liq_zscore`

#### DVOL / Implied Volatility (11)
- `dvol_btc`, `dvol_btc_change_12h`, `dvol_btc_change_24h`, `dvol_btc_z_30d`, `dvol_btc_z_60d`
- `dvol_eth`, `dvol_eth_change_12h`, `dvol_eth_change_24h`, `dvol_eth_z_30d`, `dvol_eth_z_60d`
- `dvol_spread`, `dvol_term_ratio`, `dvol_vol_of_vol`

**Total from add_derivatives: ~56**

---

### J. `add_news_interaction_features()` — News × Price Interactions (24)

- `nx_sent_x_count_1h`, `nx_sent_x_count_24h`, `nx_sent_x_count_7d`
- `nx_burst_ratio`, `nx_is_burst`, `nx_burst_x_sent`
- `nx_sent_price_div`, `nx_sent_ret_product`, `nx_sent_price_div_24h`
- `nx_sent_mom_align`, `nx_sent_mom_3d`
- `nx_sent_vs_market`, `nx_count_vs_market`
- `nx_sent_x_vol`
- `nx_sent_x_fear`
- `nx_high_volume`, `nx_high_vol_positive`, `nx_high_vol_negative`
- `nx_sent_accel`, `nx_sent_accel_7d`
- `nx_funding_x_sent`, `nx_funding_sent_div`
- `nx_news_in_dispersion`

---

### K. `add_ridge_features()` / `add_cls_features()` — Model-Specific

#### Ridge-specific (created if not present):
- `residual_12h`, `residual_24h`
- `mom_z_12h`, `mom_z_24h`
- `dist_from_high_24h`
- Aliases: `oi_chg_12h` ← `oi_change_12h`, `oi_chg_24h` ← `oi_change_24h`, `oi_zscore` ← `oi_zscore_7d`

#### CLS-specific:
- `rvol_12h`, `rvol_24h` (realised vol)
- `iv_rv_spread`
- `pct_coins_up_12h`, `pct_coins_up_1h`
- `hour_sin` ← `cal_hour_sin`, `hour_cos` ← `cal_hour_cos`, `dow_sin` ← `cal_dow_sin`, `dow_cos` ← `cal_dow_cos`

---

## 4. GRAND TOTAL — Unique Feature Columns

### Summary Count by Category

| Category | Count |
|---|---|
| Returns (1h-168h) | 9 |
| Price shape / candle | 7 |
| MA ratios (close + vol) | 16 |
| Garman-Klass vol | 4 |
| Return distribution (std/skew/kurt/sharpe) | 15 |
| Volume features | 11 |
| TA indicators (RSI/MACD/BB/ATR/ADX/Stoch/CCI/WillR/MFI/OBV) | 30 |
| Cross-asset (BTC/ETH returns, ratios, beta) | 20 |
| BTC regime | 9 |
| Market mode / correlation structure | 8 |
| Liquidity proxies | 6 |
| Calendar/seasonality | 9 |
| 12h holding features (zscore/vwap/momentum/range) | 18 |
| FNG sentiment | 6 |
| Funding rate (OKX + Binance) | 7 |
| Long/Short ratio (OKX) | 1 |
| Synthetic positioning (reversal/vol_surge/dispersion) | 8 |
| OI features (Binance) | 9 |
| Taker features (Binance) | 6 |
| Top Trader L/S | 5 |
| Global L/S + divergence | 3 |
| Market-wide aggregates (OI/taker/funding) | 4 |
| Basis/Premium | 7 |
| Liquidation | 11 |
| DVOL (implied vol) | 13 |
| News — per-coin | 8 |
| News — market/political | 9 |
| News interactions (nx_*) | 22 |
| News flags | 2 |
| Macro/FRED raw | 9 |
| Macro changes (1d/5d/20d) | 21 |
| Macro z-scores | 4 |
| Macro cross-interactions | 4 |
| Model-specific aliases (residual/mom_z/dist_from_high) | 5 |
| CS ranks (ret_12h_cs, vol_12h_cs, funding_cs, skew_cs) | 6 |
| **TOTAL** | **~315** |

---

## 5. Classification: What Was Tested vs Not

### ✅ FEATURES_23 (Production — used by CLS model): 23 features
See Section 1 above.

### 🔬 R28c CANDIDATES (Forward-tested, none improved Sh=3.39): 23 features
See Section 2 above.

### ❌ NEVER TESTED in R28c — By Reason

#### A. Excluded as "Toxic" / Known Leakers (Funding Rate)
The R28c comment says: *"excluding toxic funding/market-level"*
- `funding_rate` — OKX raw, known overfitting risk
- `funding_rate_binance` — Binance raw
- `cum_funding_24h`, `cum_funding_72h` — cumulative funding
- `funding_surprise` — funding vs expected
- `funding_cs_rank` — CS rank of funding
- `funding_vs_market` — per-coin vs avg
- `market_avg_funding`, `market_funding_std`, `market_funding_skew`
- `funding_dispersion`
**Count: ~10**

#### B. Market-Level BTC Features (Same for all coins → no CS signal)
- `btc_ret_1h`...`btc_ret_168h` (6)
- `eth_ret_1h`...`eth_ret_24h` (4)
- `btc_vol_24h`
- `btc_regime_24`, `btc_regime_72`, `btc_regime_168`
- `eth_btc_ret_24h`
**Count: ~15**

#### C. BTC Regime Features (Market-level, binary flags)
- `regime_btc_above_ma336`, `regime_btc_above_ma720`, `regime_btc_ma720_slope`
- `regime_btc_not_crashed`, `regime_btc_dd_720`, `regime_low_vol`
- `regime_breadth_bullish`, `regime_composite`
**Count: 8**

#### D. Calendar Features (Already in FEATURES_23 as hour_sin/cos, dow_sin/cos)
- `cal_hour_sin`, `cal_hour_cos`, `cal_dow_sin`, `cal_dow_cos` — aliases of what's in F23
- `cal_month_sin`, `cal_month_cos` — NOT tested
- `cal_is_us_session`, `cal_is_weekend` — NOT tested
- `cal_days_to_monthly_expiry` — NOT tested
**Count untested: 5 (month_sin/cos, us_session, weekend, expiry)**

#### E. Market Mode / Correlation Structure (Market-level)
- `avg_corr_24h`, `avg_corr_72h`
- `pca1_share_72h`, `idio_fraction_72h`
- `beta_dispersion_48h`, `beta_dispersion_168h`
- `dispersion_ret_12h_lag1`, `dispersion_regime_12h`
**Count: 8**

#### F. Liquidity Features (Never tested)
- `dollar_volume_12h`, `dollar_volume_24h`, `dollar_volume_24h_cs`
- `amihud_illiq_24h`, `amihud_illiq_24h_cs`
- `range_per_dv_24h`
**Count: 6**

#### G. News Features (All excluded)
Per-coin: `news_count_1h`, `news_count_24h`, `news_count_7d`, `news_sentiment_1h`, `news_sentiment_24h`, `news_sentiment_7d`, `news_sentiment_momentum`, `news_volume_zscore`
Market: `market_news_count_24h`, `market_news_sentiment_24h`
Political: `political_news_count_24h`, `political_sentiment_24h`, `political_sentiment_7d`, `political_sentiment_shock`, `political_news_volume_zscore`
Flags: `news_coverage_ok`, `news_event`, `has_news_data`
**Count: 18**

#### H. News Interactions (nx_*) (All excluded)
All 22 `nx_*` features were never tested in R28c.
**Count: 22**

#### I. Macro/FRED Features (All excluded — market-level)
Raw (9): `vix_close`, `spx_close`, `dxy_close`, `gold_close`, `yield_10y_close`, `hy_spread`, `breakeven_10y`, `yield_curve_10y2y`, `fed_funds_rate`
Changes (21): `*_chg_1d`, `*_chg_5d`, `*_chg_20d` for 7 instruments
Z-scores (4): `vix_close_z20d`, `hy_spread_z20d`, `breakeven_10y_z20d`, `yield_curve_10y2y_z20d`
Interactions (4): `risk_aversion`, `risk_on_off_ratio`, `real_rate`, `real_rate_chg_5d`
**Count: 38**

#### J. Derivatives — per-coin, PARTIALLY tested
Already in CANDIDATES: `top_ls_ratio_zscore`, `global_ls_ratio_zscore`, `premium_zscore`, `taker_zscore`, `oi_ret_diverge`
NOT in CANDIDATES:
- `oi_change_1h`, `oi_change_4h` — short lookbacks
- `oi_ret_interaction`, `oi_ret_interaction_12h`
- `oi_change_12h_cs` — CS rank
- `taker_flow_zscore`, `taker_imbalance_cs`, `taker_buy_sell_ratio` (raw)
- `top_ls_ratio` (raw), `top_long_pct`, `top_ls_change_12h`, `top_ls_change_24h`, `top_ls_zscore`
- `global_ls_ratio` (raw), `global_long_pct`
- `oi_value_usd` (raw)
- `taker_imbalance` (raw)
**Count untested: ~17**

#### K. Basis/Premium — PARTIALLY tested
In CANDIDATES: `premium_zscore`, `premium_zscore_12h`
NOT in CANDIDATES:
- `basis_pct`, `basis_zscore_7d`, `basis_change_12h`, `basis_change_24h`
- `basis_cs_rank`, `basis_funding_divergence`
**Count untested: 6**

#### L. Liquidation Features (All excluded)
- `liq_long_usd`, `liq_short_usd`, `liq_total_usd`, `liq_imbalance`
- `liq_cascade_12h`, `liq_cascade_24h`, `liq_imbalance_12h`
- `liq_total_zscore`, `liq_ret_interaction`, `agg_liq_zscore`
**Count: 10**

#### M. DVOL (Implied Volatility) — partially overlapping with iv_rv_spread
- `dvol_btc`, `dvol_btc_change_12h`, `dvol_btc_change_24h`, `dvol_btc_z_30d`, `dvol_btc_z_60d`
- `dvol_eth`, `dvol_eth_change_12h`, `dvol_eth_change_24h`, `dvol_eth_z_30d`, `dvol_eth_z_60d`
- `dvol_spread`, `dvol_term_ratio`, `dvol_vol_of_vol`
**Count: 13**

#### N. FNG Sentiment — Partially in model via regime/unranked
- `fng_value`, `fng_extreme_fear`, `fng_extreme_greed`, `fng_ma7`, `fng_ma30`, `fng_momentum`
**Count: 6**

#### O. Price Shape / Candle (Never tested as candidates)
- `close_open_ratio`, `high_low_ratio`, `high_close_ratio`, `low_close_ratio`
- `upper_shadow`, `lower_shadow`, `body`
**Count: 7**

#### P. MA Ratios (Never tested)
- All 8 `close_ma*_ratio` and 8 `vol_ma*_ratio`
**Count: 16**

#### Q. 12h Holding Period Features (Never tested in R28c)
- `mom_12h_zscore`, `vwap_12h_dist`, `mom_3d`, `mom_7d`, `mom_accel_12h`
- `vol_trend_12_48`, `is_asian_session`, `range_expansion_12h`, `range_position_12h`
- `vwpc_12h`, `hh_count_12h`, `ll_count_12h`, `trend_strength_12h`
- `vol_crush_ratio`, `direction_quality_12h`
- `ret_12h_cs_rank`, `vol_12h_cs_rank`
**Count: 17**

#### R. Other Returns Not in F23 or Candidates (Partial)
- `ret_2h`, `ret_6h`, `ret_72h` — not tested
**Count: 3**

#### S. Return Distribution Moments (Partially tested — skew/kurt 24h were CANDIDATES)
- `ret_std_24h/48h/168h`, `ret_mean_24h/48h/168h`, `ret_sharpe_24h/48h/168h` — NOT tested
- `ret_skew_48h`, `ret_skew_168h`, `ret_kurt_48h`, `ret_kurt_168h` — NOT tested
**Count untested: 13**

#### T. Other TA Not in CANDIDATES
- `rsi_6`, `rsi_12`, `rsi_24` — NOT tested (only rsi_14 was)
- `macd`, `macd_signal`, `macd_diff` — NOT tested
- `bb_high_20`, `bb_low_20`, `bb_width_20` — NOT tested (only bb_pband_20)
- `bb_high_48`, `bb_low_48`, `bb_width_48`, `bb_pband_48` — NOT tested
- `atr_24`, `atr_48` — NOT tested (only atr_14 in F23)
- `adx_pos`, `adx_neg` — NOT tested (only adx was)
- `stoch_k`, `stoch_d` — NOT tested
- `cci_14`, `cci_48` — NOT tested
- `willr_14` — NOT tested
- `obv_ma_ratio_12`, `obv_ma_ratio_48` — NOT tested (only obv_ma_ratio_24 was)
**Count: 21**

#### U. Synthetic Positioning / CS Ranks (Partially)
- `reversal_4v24`, `reversal_12v48`, `reversal_24v168` — NOT tested
- `vol_surge_12h`, `vol_surge_24h`, `vol_surge_48h` — NOT tested
- `cross_coin_dispersion`, `cross_coin_disp_ma24`, `dispersion_regime` — NOT tested
- `ret_skew_48h_cs`, `ret_skew_168h_cs` — NOT tested
- `long_short_ratio` — NOT tested
- `market_dispersion` — NOT tested
- `breadth_pct_positive` — NOT tested (pct_coins_up_12h/1h ARE in F23 though)
**Count: ~13**

---

## 6. Summary Statistics

| Set | Count |
|---|---|
| Total unique features in pipeline | ~315 |
| In FEATURES_23 (production) | 23 |
| In R28c CANDIDATES (forward-tested) | 23 |
| **Overlap (in both F23 + CAND)** | **0** (by design) |
| **Total covered (F23 ∪ CAND)** | **46** |
| **NEVER TESTED** | **~269** |

### Breakdown of the ~269 untested features:

| Why Excluded | Count |
|---|---|
| Funding-rate related (toxic/leaker) | 10 |
| Market-level BTC/ETH (no CS signal) | 15 |
| BTC regime flags (binary, unranked) | 8 |
| Calendar (partially covered) | 5 |
| Market mode/correlation  | 8 |
| Liquidity proxies | 6 |
| News per-coin + market + political | 18 |
| News interactions (nx_*) | 22 |
| Macro/FRED (all market-level) | 38 |
| Derivatives per-coin (un-tested subset) | 17 |
| Basis/premium (un-tested subset) | 6 |
| Liquidation | 10 |
| DVOL implied vol | 13 |
| FNG sentiment | 6 |
| Price shape / candle | 7 |
| MA ratios | 16 |
| 12h holding features | 17 |
| Misc returns (2h, 6h, 72h) | 3 |
| Return distribution (untested moments) | 13 |
| TA indicators (untested subset) | 21 |
| Synthetic positioning / CS ranks | 13 |
| **TOTAL** | **~272** |

### Key Insight

R28c tested only **per-coin, non-toxic** features that could serve as additional alpha inputs to a CS-ranked tree model. The ~269 untested features fall into clear categories:

1. **Market-level (same for all coins)** — ~105 features: macro, BTC/ETH returns, regimes, FNG, DVOL, news market-level, market mode. These cannot help a CS-ranked model since they have zero cross-sectional variation after ranking.

2. **Known toxic (funding)** — ~10 features: high leakage risk, funding settlement mechanics create look-ahead bias.

3. **Per-coin but redundant / highly correlated** — ~80 features: multiple MA ratios, TA at different windows, multiple return lookbacks that overlap with existing features.

4. **Per-coin, potentially useful but not tested** — ~75 features: liquidation, basis, 12h holding, price shape, vol_surge, reversals, CS ranks, some TA variants. **These are the best candidates for future R29 testing.**
