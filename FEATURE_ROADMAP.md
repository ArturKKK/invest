# Feature Roadmap — New Data Sources & Features

## Priority Queue

### P0 — This Week (In Progress)

#### 1. Deribit DVOL — Options Implied Volatility (BTC + ETH)
- **Script**: `src/data/download_deribit_dvol.py`
- **Output**: `data/sentiment/deribit_dvol.parquet`
- **API**: `https://www.deribit.com/api/v2/public/get_volatility_index_data` (public, no auth)
- **Frequency**: Hourly candles (OHLC of IV)
- **History**: April 2021+ (~4 years, ~35K rows per currency)
- **Pagination**: 1000 records/request, continuation token
- **Features (UNRANKED, market-level, 10 features)**:
  - `dvol_btc` — BTC implied vol level (close)
  - `dvol_btc_change_12h` — 12h change
  - `dvol_btc_change_24h` — 24h change
  - `dvol_btc_z_30d` — TS z-score over 30d rolling window
  - `dvol_btc_z_60d` — TS z-score over 60d rolling window
  - `dvol_eth` — ETH implied vol level (close)
  - `dvol_eth_change_24h` — 24h change
  - `dvol_spread` — BTC DVOL minus ETH DVOL (regime indicator)
  - `dvol_term_ratio` — DVOL now / DVOL MA(30d) (term structure proxy)
  - `dvol_vol_of_vol` — std(DVOL, 7d) — vol regime turbulence
- **Usage**: UNRANKED regime features (same for all coins per timestamp)
- **Impact hypothesis**: Regime gating — reduce leverage/positions when IV spikes. Should improve MaxDD and WR via risk control.

#### 2. Macro / Cross-Market Data (VIX, DXY, SPX, Gold, Yields)
- **Script**: `src/data/download_macro.py`
- **Output**: `data/sentiment/macro_daily.parquet`
- **API**: yfinance (free, no auth) + FRED (free key optional)
- **Frequency**: Daily (forward-filled to hourly)
- **History**: 2020+ (5+ years)
- **Tickers**: `^VIX`, `DX-Y.NYB` (DXY), `^GSPC` (SPX), `GC=F` (Gold), `^TNX` (10Y yield)
- **Features (UNRANKED, market-level, 12 features)**:
  - `vix_level` — VIX close
  - `vix_change_1d` — 1-day VIX change
  - `vix_z_60d` — TS z-score
  - `dxy_ret_1d` — DXY daily return
  - `dxy_z_60d` — TS z-score
  - `spx_ret_1d` — S&P500 daily return
  - `spx_ret_5d` — 5-day SPX return
  - `gold_ret_1d` — Gold daily return
  - `yield_10y` — 10Y Treasury yield level
  - `yield_change_5d` — 5-day yield change
  - `risk_on_composite` — weighted sum (VIX down + SPX up + DXY down = risk-on)
  - `risk_off_flag` — binary: VIX z-score > 2 (crash/stress)
- **Usage**: UNRANKED regime features
- **Impact hypothesis**: Crypto correlates with risk-on/risk-off. VIX spike → short bias, DXY surge → crypto headwind.

---

### P1 — Next Sprint

#### 3. TS Z-Score Features (from EXISTING data, no new downloads)
- **Where**: Add to `run_pipeline_v6.py` / `run_pipeline_v7.py`
- **Concept**: Per-coin time-series z-scores over rolling 60d window (NOT cross-sectional rank)
- **Features (per-coin, UNRANKED via REGIME_COLS, ~10 features)**:
  - `ts_z_funding_60d` — is this coin's funding at its own historical extreme?
  - `ts_z_basis_60d` — basis vs own history
  - `ts_z_oi_change_7d` — OI momentum vs own norm
  - `ts_z_liq_total_30d` — liquidation level vs own history
  - `ts_z_ret_12h_180d` — current return percentile vs own distribution
  - `ts_z_volume_30d` — volume vs own history
  - `ts_z_taker_imbalance_14d`
  - `ts_z_spread_7d` (if orderbook available)
- **Impact hypothesis**: Breaks correlation ceiling — different coordinate system from CS-rank. Models will see "this coin's funding is extreme for ITSELF" not just "higher than peers".

#### 4. Interaction Features (from EXISTING data)
- **Where**: Add to pipeline feature engineering
- **Features (~6-8)**:
  - `oi_contra_price = oi_change_12h × (-ret_12h)` — positioning against price → squeeze signal
  - `crowded_trend = funding_extreme × trend_strength` — trend + crowd in same direction
  - `liq_vol_interaction = liq_imbalance × gk_vol_24h` — liquidation cascade risk
  - `basis_funding_div = basis_z × (-funding_rate)` — perp mispricing
  - `dvol_ret_interaction = dvol_btc_change_24h × btc_ret_24h` — IV-price divergence
  - `flow_momentum = exchg_netflow_impulse × ret_24h` (if on-chain available)
- **Impact hypothesis**: Non-linear signals that GBDT might miss (or find faster with explicit features).

---

### P2 — Later

#### 5. CoinMetrics On-Chain (Exchange Flows)
- **API**: `https://community-api.coinmetrics.io/v4/timeseries/asset-metrics` (free)
- **Frequency**: Daily
- **Assets**: BTC, ETH (market-level)
- **Metrics**: `FlowInExchgUSD`, `FlowOutExchgUSD`, `AdrActCnt`, `NVTAdj`, `SplyAct1d`
- **Features**: `exchg_netflow_usd`, `netflow_z_180d`, `netflow_impulse`, `active_addr_mom`

#### 6. Orderbook Depth Snapshots
- **API**: OKX/Binance REST or WebSocket
- **Frequency**: 15min snapshots → 1h aggregate
- **Features**: `spread_bps`, `depth_imbalance_top5`, `price_impact_10k`, `orderbook_slope`
- **Risk**: Heavy engineering, storage, synchronization

#### 7. Multi-Timeframe (4h candles from 1h)
- Resample existing 1h data to 4h/12h candles
- Features: range expansion, close-to-high ratio, intrabar patterns

---

## Storage Estimates

| Source | Rows | Columns | Size (parquet) |
|--------|------|---------|----------------|
| Deribit DVOL (BTC+ETH, 4yr hourly) | ~70K | 8 | ~2 MB |
| Macro daily (5 tickers, 5yr) | ~6K | 12 | <1 MB |
| TS z-scores (in-pipeline, no file) | — | — | 0 |
| CoinMetrics daily | ~7K | 10 | <1 MB |
| Orderbook snapshots (50 coins, 1h) | ~2M | 8 | ~50-100 MB |

**Total for P0**: ~3 MB. Trivial.
