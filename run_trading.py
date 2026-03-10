#!/usr/bin/env python3
"""
Production Trading System — OKX Perpetual Swaps

Complete pipeline: fetch data → features → inference → risk management → execute.

Uses:
  - LightGBM v5 (multi-seed ensemble) for fast signal generation
  - Optimal risk config from risk study (vol target, DD stop, Kelly)
  - OKX perpetual swaps with isolated margin, 1x leverage

Modes:
  signal   — Generate signals only, print portfolio (no API needed)
  sim      — Local paper trading: track positions, compute PnL from real prices (no API needed)
  paper    — Paper trade on OKX demo account (needs API key)
  live     — Live trading with real money

Setup:
  1. Train models: python run_pipeline_v5.py (saves to results_v5/)
  2. Run risk study: python run_risk_study.py (saves optimal_config.json)
  3. Set OKX API keys:
     export OKX_API_KEY=xxx OKX_SECRET=xxx OKX_PASSPHRASE=xxx
  4. For paper: export OKX_DEMO=1

Usage:
  # Local paper trading (no API needed):
  python run_trading.py --mode sim --capital 1000 --loop

  # Generate signals only (no API needed):
  python run_trading.py --mode signal

  # Paper trading (single cycle):
  python run_trading.py --mode paper --capital 1000

  # Live continuous (rebalances every 12h by default):
  python run_trading.py --mode live --capital 500 --loop

  # Custom rebalance interval and position count:
  python run_trading.py --mode sim --capital 1000 --loop --rebal 12 --npos 5

  # Override risk params:
  python run_trading.py --mode paper --vol-target 0.01 --kelly 0.3
"""

import os
import sys
import time
import json
import signal
import argparse
import warnings
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# Load .env file if exists
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(_env_path):
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
    except ImportError:
        with open(_env_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith('#') and '=' in _line:
                    _k, _, _v = _line.partition('=')
                    os.environ[_k.strip()] = _v.strip().strip('\'"')

# Import Telegram bot
try:
    from telegram_bot import create_bot
except ImportError:
    create_bot = None

# ============================================================
# CONFIG
# ============================================================
HORIZON = 4
TOP_K_DEFAULT = 10  # will be overridden by risk config
SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
    'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'DOT/USDT', 'LINK/USDT',
    'MATIC/USDT', 'UNI/USDT', 'ATOM/USDT', 'LTC/USDT', 'FIL/USDT',
    'APT/USDT', 'ARB/USDT', 'OP/USDT', 'NEAR/USDT', 'AAVE/USDT',
    'INJ/USDT', 'FTM/USDT', 'ALGO/USDT', 'SAND/USDT', 'MANA/USDT',
    'AXS/USDT', 'THETA/USDT', 'RUNE/USDT', 'EGLD/USDT', 'XTZ/USDT',
    'FLOW/USDT', 'CHZ/USDT', 'CRV/USDT', 'LDO/USDT', 'SNX/USDT',
    'COMP/USDT', 'YFI/USDT', 'SUSHI/USDT', 'ENJ/USDT', 'BAT/USDT',
    'ZIL/USDT', 'ONE/USDT', 'IOTA/USDT', 'ICX/USDT', 'ENS/USDT',
    'IMX/USDT', 'GALA/USDT', 'MKR/USDT', 'GRT/USDT', 'ETC/USDT',
]

# OKX instrument mapping
# Coins unavailable on OKX demo (delisted, rebranded, or restricted)
_OKX_BLOCKED = {
    'MATIC/USDT', 'UNI/USDT', 'APT/USDT', 'FTM/USDT', 'MANA/USDT',
    'RUNE/USDT', 'EGLD/USDT', 'FLOW/USDT', 'SNX/USDT', 'ENJ/USDT',
    'BAT/USDT', 'ONE/USDT', 'ICX/USDT', 'ENS/USDT', 'GALA/USDT',
    'GRT/USDT', 'CHZ/USDT', 'MKR/USDT',  # 51155 compliance (set_leverage OK but orders fail)
    'ZIL/USDT',  # 51202 max market order exceeded on demo
}

SYMBOLS_TO_OKX = {}
for sym in SYMBOLS:
    if sym in _OKX_BLOCKED:
        continue
    # Use ccxt unified swap symbol: "BTC/USDT" → "BTC/USDT:USDT"
    SYMBOLS_TO_OKX[sym] = sym + ':USDT'

# Feature columns to exclude
EXCLUDE_COLS = {
    'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume',
    'target_ret_4h', 'target_ret_12h', 'target_ret_24h',
    'target_cls', 'target_ret', 'target_rank', 'target_excess',
    'hour', 'day_of_week',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'btc_close', 'eth_close',
}

# Columns NOT to rank-normalize (market-level or binary)
# Must match REGIME_COLS in run_pipeline_v6.py to avoid train/live distribution shift
UNRANKED_COLS = {
    'btc_regime_24', 'btc_regime_72', 'btc_regime_168',
    'regime_btc_above_ma336', 'regime_btc_above_ma720',
    'regime_btc_ma720_slope', 'regime_btc_not_crashed',
    'regime_btc_dd_720', 'regime_low_vol',
    'regime_breadth_bullish', 'breadth_pct_positive',
    'regime_composite',
    # Sentiment columns (NOT ranked cross-sectionally)
    'fng_value', 'fng_extreme_fear', 'fng_extreme_greed',
    'fng_ma7', 'fng_ma30', 'fng_momentum',
    'market_avg_funding', 'market_funding_skew',
    # v6: binary features
    'is_asian_session',
    # News sentiment (market-level, should NOT be ranked cross-sectionally)
    'market_news_count_24h', 'market_news_sentiment_24h',
    'news_sentiment_24h', 'news_sentiment_7d', 'news_sentiment_momentum',
    # Political/macro news (market-level, same for all coins)
    'political_news_count_24h', 'political_sentiment_24h',
    'political_sentiment_7d', 'political_sentiment_shock',
    'political_news_volume_zscore',
    # Binary flags (should NOT be ranked)
    'has_news_data',
    'news_coverage_ok', 'news_event',
}

# Default risk config — matches champion backtest (Sharpe 8.04)
# Override via --config JSON or .env
DEFAULT_RISK = {
    'n_long': 5,
    'n_short': 5,
    'leverage': 3,
    'margin_utilization': 0.90,   # use 90% of capital for margin, 10% buffer
    'vol_target': 0.008,
    'vol_lookback': 48,
    'kelly_frac': 1.0,
    'dd_stop': -0.20,
    'dd_resume': -0.08,
    'confidence_threshold': 0.0,
    'min_score': 1.0,             # skip signals with |score| < this (sweep: no loss vs 0.0-0.7)
    'edge_boost': True,           # weight positions proportional to |score| (champion backtest)
}


# ============================================================
# DATA FETCHING
# ============================================================

def fetch_ohlcv(symbols, hours=800):
    """Fetch recent OHLCV from Binance public API."""
    try:
        import ccxt
    except ImportError:
        print(f"❌ pip install ccxt")
        sys.exit(1)

    proxy = os.environ.get('ALL_PROXY') or os.environ.get('HTTPS_PROXY') or os.environ.get('HTTP_PROXY')
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'},
    })
    exchange.session.verify = False
    if proxy:
        exchange.socks_proxy = proxy

    all_dfs = []
    failed = []

    for i, sym in enumerate(symbols):
        try:
            needed = hours + 10
            chunks = []
            since = None  # will fetch most recent first, then paginate backwards
            while needed > 0:
                batch = min(needed, 1000)
                if since is None:
                    # First request: get the most recent candles
                    ohlcv = exchange.fetch_ohlcv(sym, '1h', limit=batch)
                else:
                    ohlcv = exchange.fetch_ohlcv(sym, '1h', since=since, limit=batch)
                if not ohlcv:
                    break
                chunks.append(ohlcv)
                needed -= len(ohlcv)
                if len(ohlcv) < batch:
                    break  # no more data available
                # Next request: go further back
                earliest_ts = ohlcv[0][0]
                since = earliest_ts - batch * 3600 * 1000  # ms
            # Combine chunks (may overlap), deduplicate
            all_candles = []
            seen_ts = set()
            for chunk in chunks:
                for c in chunk:
                    if c[0] not in seen_ts:
                        all_candles.append(c)
                        seen_ts.add(c[0])
            all_candles.sort(key=lambda x: x[0])
            if all_candles:
                df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df['symbol'] = sym
                all_dfs.append(df)
        except Exception as e:
            failed.append(sym)

    if failed:
        print(f"   ⚠️  Failed: {len(failed)} symbols")
    if not all_dfs:
        return None

    df = pd.concat(all_dfs, ignore_index=True)
    return df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)


# ============================================================
# FEATURE ENGINEERING (mirrors training pipeline)
# ============================================================


def add_news_interaction_features(df):
    """Create news × price interaction features for XGBoost model."""
    news_cols = ['news_count_1h', 'news_count_24h', 'news_count_7d',
                 'news_sentiment_1h', 'news_sentiment_24h', 'news_sentiment_7d',
                 'news_sentiment_momentum', 'news_volume_zscore',
                 'market_news_count_24h', 'market_news_sentiment_24h']
    for col in news_cols:
        if col not in df.columns:
            df[col] = 0.0

    # Sentiment × Volume
    df['nx_sent_x_count_1h'] = df['news_sentiment_1h'] * df['news_count_1h']
    df['nx_sent_x_count_24h'] = df['news_sentiment_24h'] * df['news_count_24h']
    df['nx_sent_x_count_7d'] = df['news_sentiment_7d'] * df['news_count_7d']

    # News burst
    df['nx_burst_ratio'] = df['news_count_1h'] / (df['news_count_24h'] / 24 + 1e-6)
    df['nx_is_burst'] = (df['nx_burst_ratio'] > 3).astype(float)
    df['nx_burst_x_sent'] = df['nx_is_burst'] * df['news_sentiment_1h']

    # Sentiment vs Price divergence
    if 'ret_12h' in df.columns:
        df['nx_sent_price_div'] = df['news_sentiment_24h'] * np.sign(-df['ret_12h'])
        df['nx_sent_ret_product'] = df['news_sentiment_24h'] * df['ret_12h']
    if 'ret_24h' in df.columns:
        df['nx_sent_price_div_24h'] = df['news_sentiment_24h'] * np.sign(-df['ret_24h'])

    # Sentiment momentum × price momentum
    if 'mom_12h_zscore' in df.columns:
        df['nx_sent_mom_align'] = df['news_sentiment_momentum'] * df['mom_12h_zscore']
    if 'mom_3d' in df.columns:
        df['nx_sent_mom_3d'] = df['news_sentiment_momentum'] * np.sign(df['mom_3d'])

    # Coin vs Market sentiment
    df['nx_sent_vs_market'] = df['news_sentiment_24h'] - df['market_news_sentiment_24h']
    df['nx_count_vs_market'] = df['news_count_24h'] / (df['market_news_count_24h'] + 1e-6)

    # Sentiment × Volatility
    if 'gk_vol_24h' in df.columns:
        df['nx_sent_x_vol'] = df['news_sentiment_24h'] * df['gk_vol_24h']
    if 'fng_value' in df.columns:
        df['nx_sent_x_fear'] = df['news_sentiment_24h'] * (50 - df['fng_value']) / 50

    # News cluster features
    df['nx_high_volume'] = (df['news_count_1h'] >= 3).astype(float)
    df['nx_high_vol_positive'] = df['nx_high_volume'] * (df['news_sentiment_1h'] > 0.2).astype(float)
    df['nx_high_vol_negative'] = df['nx_high_volume'] * (df['news_sentiment_1h'] < -0.2).astype(float)

    # Sentiment acceleration
    df['nx_sent_accel'] = df['news_sentiment_1h'] - df['news_sentiment_24h']
    df['nx_sent_accel_7d'] = df['news_sentiment_24h'] - df['news_sentiment_7d']

    # Funding × News
    if 'funding_rate' in df.columns:
        df['nx_funding_x_sent'] = df['funding_rate'] * df['news_sentiment_24h']
        df['nx_funding_sent_div'] = df['funding_rate'] * np.sign(-df['news_sentiment_24h'])

    # Cross-coin news asymmetry
    if 'cross_coin_dispersion' in df.columns:
        df['nx_news_in_dispersion'] = df['news_count_24h'] * df['cross_coin_dispersion']

    return df


def build_features(df):
    """
    Full feature engineering — mirrors build_features.py + v5 additions.
    """
    try:
        import ta
    except ImportError:
        print("❌ pip install ta")
        sys.exit(1)

    # Per-symbol features
    result_dfs = []
    for sym, gdf in df.groupby('symbol'):
        g = gdf.sort_values('timestamp').copy()

        c, h, l, o, v = g['close'], g['high'], g['low'], g['open'], g['volume']

        # Returns
        for hr in [1, 2, 4, 6, 12, 24, 48, 72, 168]:
            g[f'ret_{hr}h'] = c.pct_change(hr)

        # Price features
        g['close_open_ratio'] = c / o - 1
        g['high_low_ratio'] = h / l - 1
        g['high_close_ratio'] = h / c - 1
        g['low_close_ratio'] = c / l - 1
        g['upper_shadow'] = (h - np.maximum(c, o)) / (h - l + 1e-10)
        g['lower_shadow'] = (np.minimum(c, o) - l) / (h - l + 1e-10)
        g['body'] = np.abs(c - o) / (h - l + 1e-10)

        for w in [6, 12, 24, 48, 72, 168, 336, 720]:
            ma = c.rolling(w, min_periods=max(w // 2, 1)).mean()
            g[f'close_ma{w}_ratio'] = c / ma - 1
            g[f'vol_ma{w}_ratio'] = v / v.rolling(w, min_periods=max(w // 2, 1)).mean() - 1

        for w in [12, 24, 48, 168]:
            log_hl = np.log(h / l + 1e-10) ** 2
            log_co = np.log(c / o + 1e-10) ** 2
            g[f'gk_vol_{w}h'] = np.sqrt(
                (0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(w).mean().abs()
            )

        for w in [24, 48, 168]:
            r = c.pct_change()
            g[f'ret_std_{w}h'] = r.rolling(w).std()
            g[f'ret_skew_{w}h'] = r.rolling(w).skew()
            g[f'ret_kurt_{w}h'] = r.rolling(w).kurt()
            g[f'ret_mean_{w}h'] = r.rolling(w).mean()
            g[f'ret_sharpe_{w}h'] = g[f'ret_mean_{w}h'] / (g[f'ret_std_{w}h'] + 1e-10)

        # Volume features
        for w in [6, 12, 24, 48]:
            g[f'vol_mom_{w}h'] = v / v.shift(w) - 1
        for w in [12, 24, 48]:
            vwap = (c * v).rolling(w).sum() / (v.rolling(w).sum() + 1e-10)
            g[f'vwap_dev_{w}h'] = c / vwap - 1
        for w in [24, 48, 168]:
            g[f'vol_price_corr_{w}h'] = c.pct_change().rolling(w).corr(v.pct_change())
        g['buy_pressure'] = (c - l) / (h - l + 1e-10)

        # TA indicators
        for p in [6, 12, 14, 24]:
            g[f'rsi_{p}'] = ta.momentum.RSIIndicator(c, window=p).rsi()
        macd = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
        g['macd'] = macd.macd()
        g['macd_signal'] = macd.macd_signal()
        g['macd_diff'] = macd.macd_diff()
        for w in [20, 48]:
            bb = ta.volatility.BollingerBands(c, window=w, window_dev=2)
            g[f'bb_high_{w}'] = (bb.bollinger_hband() - c) / (c + 1e-10)
            g[f'bb_low_{w}'] = (c - bb.bollinger_lband()) / (c + 1e-10)
            g[f'bb_width_{w}'] = bb.bollinger_wband()
            g[f'bb_pband_{w}'] = bb.bollinger_pband()
        for w in [14, 24, 48]:
            g[f'atr_{w}'] = ta.volatility.AverageTrueRange(h, l, c, window=w).average_true_range() / (c + 1e-10)
        adx = ta.trend.ADXIndicator(h, l, c, window=14)
        g['adx'] = adx.adx()
        g['adx_pos'] = adx.adx_pos()
        g['adx_neg'] = adx.adx_neg()
        stoch = ta.momentum.StochasticOscillator(h, l, c, window=14, smooth_window=3)
        g['stoch_k'] = stoch.stoch()
        g['stoch_d'] = stoch.stoch_signal()
        g['cci_14'] = ta.trend.CCIIndicator(h, l, c, window=14).cci()
        g['cci_48'] = ta.trend.CCIIndicator(h, l, c, window=48).cci()
        g['willr_14'] = ta.momentum.WilliamsRIndicator(h, l, c, lbp=14).williams_r()
        obv = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
        for w in [12, 24, 48]:
            g[f'obv_ma_ratio_{w}'] = obv / (obv.rolling(w).mean() + 1e-10) - 1
        g['mfi_14'] = ta.volume.MFIIndicator(h, l, c, v, window=14).money_flow_index()

        result_dfs.append(g)

    df = pd.concat(result_dfs, ignore_index=True)

    # Cross-asset features
    btc = df[df['symbol'] == 'BTC/USDT'][['timestamp', 'close']].rename(
        columns={'close': 'btc_close'}).drop_duplicates('timestamp')
    eth = df[df['symbol'] == 'ETH/USDT'][['timestamp', 'close']].rename(
        columns={'close': 'eth_close'}).drop_duplicates('timestamp')
    df = df.merge(btc, on='timestamp', how='left')
    df = df.merge(eth, on='timestamp', how='left')

    for hr in [1, 4, 12, 24, 48, 168]:
        df[f'btc_ret_{hr}h'] = df.groupby('symbol')['btc_close'].transform(lambda x: x.pct_change(hr))
    for hr in [1, 4, 12, 24]:
        df[f'eth_ret_{hr}h'] = df.groupby('symbol')['eth_close'].transform(lambda x: x.pct_change(hr))
    df['btc_vol_24h'] = df.groupby('symbol')['btc_close'].transform(lambda x: x.pct_change().rolling(24).std())

    # ETH/BTC ratio return (missing in previous version)
    df['eth_btc_ratio'] = df['eth_close'] / (df['btc_close'] + 1e-10)
    df['eth_btc_ret_24h'] = df.groupby('symbol')['eth_btc_ratio'].transform(lambda x: x.pct_change(24))
    df.drop(columns=['eth_btc_ratio'], inplace=True, errors='ignore')

    df['market_dispersion'] = df.groupby('timestamp')['ret_1h'].transform('std')
    if 'ret_24h' in df.columns:
        df['ret_vs_btc_24h'] = df['ret_24h'] - df['btc_ret_24h']

    # Breadth
    breadth = df.groupby('timestamp')['ret_24h'].agg(
        breadth_pct_positive=lambda x: (x > 0).mean()
    ).reset_index()
    breadth['regime_breadth_bullish'] = (breadth['breadth_pct_positive'] > 0.5).astype(float)
    df = df.merge(breadth, on='timestamp', how='left')

    # Regime
    btc_ts = df[df['symbol'] == 'BTC/USDT'][['timestamp', 'btc_close']].drop_duplicates('timestamp').sort_values('timestamp')
    for w in [24, 72, 168]:
        btc_ts[f'btc_ma{w}'] = btc_ts['btc_close'].rolling(w).mean()
    btc_ts['btc_regime_24'] = (btc_ts['btc_close'] > btc_ts['btc_ma24']).astype(float)
    btc_ts['btc_regime_72'] = (btc_ts['btc_close'] > btc_ts['btc_ma72']).astype(float)
    btc_ts['btc_regime_168'] = (btc_ts['btc_close'] > btc_ts['btc_ma168']).astype(float)
    for w in [336, 720]:
        btc_ts[f'btc_ma{w}'] = btc_ts['btc_close'].rolling(w, min_periods=min(w, 100)).mean()
    btc_ts['regime_btc_above_ma336'] = (btc_ts['btc_close'] > btc_ts['btc_ma336']).astype(float)
    btc_ts['regime_btc_above_ma720'] = (btc_ts['btc_close'] > btc_ts['btc_ma720']).astype(float)
    btc_ts['regime_btc_ma720_slope'] = (btc_ts['btc_ma720'] > btc_ts['btc_ma720'].shift(24)).astype(float)
    btc_ts['btc_rolling_high_720'] = btc_ts['btc_close'].rolling(720, min_periods=100).max()
    btc_ts['regime_btc_dd_720'] = btc_ts['btc_close'] / btc_ts['btc_rolling_high_720'] - 1
    btc_ts['regime_btc_not_crashed'] = (btc_ts['regime_btc_dd_720'] > -0.15).astype(float)
    btc_ts['_btc_vol_24'] = btc_ts['btc_close'].pct_change().rolling(24).std()
    btc_ts['_btc_vol_720_med'] = btc_ts['_btc_vol_24'].rolling(720, min_periods=100).median()
    btc_ts['regime_low_vol'] = (btc_ts['_btc_vol_24'] < btc_ts['_btc_vol_720_med'] * 2.0).astype(float)
    regime_cols = ['timestamp', 'btc_regime_24', 'btc_regime_72', 'btc_regime_168',
                   'regime_btc_above_ma336', 'regime_btc_above_ma720',
                   'regime_btc_ma720_slope', 'regime_btc_dd_720',
                   'regime_btc_not_crashed', 'regime_low_vol']
    df = df.merge(btc_ts[regime_cols], on='timestamp', how='left')

    # Regime composite
    df['regime_composite'] = (
        0.25 * df['regime_btc_above_ma720'].fillna(0) +
        0.20 * df['regime_btc_ma720_slope'].fillna(0) +
        0.20 * df['regime_breadth_bullish'].fillna(0) +
        0.20 * df['regime_btc_not_crashed'].fillna(0) +
        0.15 * df['regime_low_vol'].fillna(0)
    )

    # Sentiment: FNG (if available)
    root = os.path.dirname(os.path.abspath(__file__))
    fng_path = os.path.join(root, 'data', 'sentiment', 'fear_greed.parquet')
    if os.path.exists(fng_path):
        fng = pd.read_parquet(fng_path)
        fng['timestamp'] = pd.to_datetime(fng['timestamp'], utc=True)
        fng['date'] = fng['timestamp'].dt.date
        fng_daily = fng[['date', 'fng_value']].drop_duplicates('date')
        df['date'] = df['timestamp'].dt.date
        df = df.merge(fng_daily, on='date', how='left')
        df['fng_value'] = df['fng_value'].ffill().fillna(50)
        df['fng_extreme_fear'] = (df['fng_value'] < 25).astype(float)
        df['fng_extreme_greed'] = (df['fng_value'] > 75).astype(float)
        df['fng_ma7'] = df.groupby('symbol')['fng_value'].transform(lambda x: x.rolling(7*24, min_periods=24).mean())
        df['fng_ma30'] = df.groupby('symbol')['fng_value'].transform(lambda x: x.rolling(30*24, min_periods=24).mean())
        df['fng_momentum'] = df['fng_value'] - df['fng_ma30']
        df.drop(columns=['date'], inplace=True, errors='ignore')

    # Funding rates (critical — used by ALL model groups)
    fund_path = os.path.join(root, 'data', 'sentiment', 'funding_rates.parquet')
    if os.path.exists(fund_path):
        fund = pd.read_parquet(fund_path)
        fund['timestamp'] = pd.to_datetime(fund['timestamp'], utc=True)
        fund = fund.sort_values('timestamp')
        if 'fundingRate' in fund.columns:
            fund = fund.rename(columns={'fundingRate': 'funding_rate'})
        # Match training: merge on 8h floor (funding is 8h, data is hourly)
        fund['ts_8h'] = fund['timestamp'].dt.floor('8h')
        df['ts_8h'] = df['timestamp'].dt.floor('8h')
        fund_melt = fund[['ts_8h', 'symbol', 'funding_rate']].drop_duplicates(['ts_8h', 'symbol'])
        df = df.merge(fund_melt, on=['ts_8h', 'symbol'], how='left')
        df.drop(columns=['ts_8h'], inplace=True)
        df['funding_rate'] = df['funding_rate'].fillna(0)
        # Market-level funding stats (used by CatBoost)
        mf = df.groupby('timestamp')['funding_rate'].agg(['mean', 'std']).reset_index()
        mf.columns = ['timestamp', 'market_avg_funding', 'market_funding_std']
        df = df.merge(mf, on='timestamp', how='left')
        df['funding_vs_market'] = df['funding_rate'] - df['market_avg_funding'].fillna(0)
        df['market_funding_skew'] = df['market_avg_funding'].fillna(0) / (df['market_funding_std'].fillna(0) + 1e-8)
        # Keep market_avg_funding — models use it as a feature
        print(f"   ✅ Funding rates merged")
    else:
        print(f"   ⚠️  No funding data at {fund_path}")
        for c in ['funding_rate', 'market_avg_funding', 'market_funding_std',
                   'market_funding_skew', 'funding_vs_market']:
            df[c] = 0.0

    # Long/Short ratio (used by CatBoost)
    lsr_path = os.path.join(root, 'data', 'sentiment', 'long_short_ratio.parquet')
    if os.path.exists(lsr_path):
        lsr = pd.read_parquet(lsr_path)
        lsr['timestamp'] = pd.to_datetime(lsr['timestamp'], utc=True)
        df = df.merge(lsr[['timestamp', 'symbol', 'long_short_ratio']],
                      on=['timestamp', 'symbol'], how='left')
        df['long_short_ratio'] = df['long_short_ratio'].fillna(1.0)
        print(f"   ✅ L/S ratio merged")
    else:
        df['long_short_ratio'] = 1.0

    # News sentiment features (from fetch_crypto_news.py)
    news_path = os.path.join(root, 'data', 'sentiment', 'crypto_news.parquet')
    if os.path.exists(news_path):
        news = pd.read_parquet(news_path)
        news['timestamp'] = pd.to_datetime(news['timestamp'], utc=True)
        news_per_coin = [
            'news_count_1h', 'news_count_24h', 'news_count_7d',
            'news_sentiment_1h', 'news_sentiment_24h', 'news_sentiment_7d',
            'news_sentiment_momentum', 'news_volume_zscore',
        ]
        news_market = ['market_news_count_24h', 'market_news_sentiment_24h']
        news_political = [
            'political_news_count_24h', 'political_sentiment_24h',
            'political_sentiment_7d', 'political_sentiment_shock',
            'political_news_volume_zscore',
        ]
        merge_cols = ['timestamp', 'symbol'] + [c for c in news_per_coin if c in news.columns]
        df = df.merge(news[merge_cols].drop_duplicates(['timestamp', 'symbol']),
                      on=['timestamp', 'symbol'], how='left')
        all_market = news_market + news_political
        market_merge = ['timestamp'] + [c for c in all_market if c in news.columns]
        df = df.merge(news[market_merge].drop_duplicates('timestamp'),
                      on='timestamp', how='left', suffixes=('', '_dup'))
        dup_cols = [c for c in df.columns if c.endswith('_dup')]
        if dup_cols:
            df.drop(columns=dup_cols, inplace=True)
        for col in news_per_coin + all_market:
            if col in df.columns:
                df[col] = df[col].fillna(0)

    # Synthetic positioning (formula matches v5 training)
    for short, long in [(4, 24), (12, 48), (24, 168)]:
        fr = f'ret_{short}h'
        sr = f'ret_{long}h'
        if fr in df.columns and sr in df.columns:
            df[f'reversal_{short}v{long}'] = df[fr] - df[sr] / (long / short)

    for w in [12, 24, 48]:
        df[f'vol_surge_{w}h'] = df.groupby('symbol')['volume'].transform(
            lambda x: x / x.rolling(w).mean() - 1)

    # Cross-coin dispersion features
    cs_disp = df.groupby('timestamp')['ret_4h'].transform('std') if 'ret_4h' in df.columns else 0
    df['cross_coin_dispersion'] = cs_disp
    df['cross_coin_disp_ma24'] = df.groupby('symbol')['cross_coin_dispersion'].transform(
        lambda x: x.rolling(24, min_periods=6).mean())
    df['dispersion_regime'] = df['cross_coin_dispersion'] / (df['cross_coin_disp_ma24'] + 1e-10)

    # Return skew cross-sectional rank
    for w in [48, 168]:
        if f'ret_skew_{w}h' in df.columns:
            df[f'ret_skew_{w}h_cs'] = df.groupby('timestamp')[f'ret_skew_{w}h'].transform(
                lambda x: x.rank(pct=True) - 0.5)

    # BTC beta
    btc_rets = df[df['symbol'] == 'BTC/USDT'][['timestamp', 'ret_1h']].rename(
        columns={'ret_1h': 'btc_r'}).drop_duplicates('timestamp')
    df = df.merge(btc_rets, on='timestamp', how='left')
    for w in [48, 168]:
        cov = df.groupby('symbol').apply(
            lambda g: g['ret_1h'].rolling(w).cov(g['btc_r'])
        ).reset_index(level=0, drop=True)
        var = df.groupby('symbol')['btc_r'].transform(
            lambda x: x.rolling(w).var() + 1e-10)
        df[f'btc_beta_{w}h'] = cov / var

    # Clean up (keep cross_coin_dispersion — used by CatBoost models)
    df.drop(columns=['btc_close', 'eth_close', 'btc_r'], inplace=True, errors='ignore')

    # Replace inf/nan
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    # ── Derivatives features (Binance Futures: OI, taker, L/S, funding) ──
    # Critical: models trained with these features expect them in live inference.
    # Without this block, 21 features are zeros → model predictions are garbage.
    try:
        from run_pipeline_v6 import add_derivatives_features as _add_deriv
        root = os.path.dirname(os.path.abspath(__file__))
        df = _add_deriv(df, root)
    except Exception as e:
        print(f"   ⚠️  Derivatives features failed: {e}")
        # Fill expected columns with neutral defaults so model doesn't crash
        for col in ['oi_value_usd', 'oi_change_4h', 'oi_change_12h', 'oi_change_24h',
                     'oi_zscore_7d', 'oi_change_12h_cs', 'oi_ret_interaction_12h',
                     'taker_cvd_12h', 'taker_cvd_24h',
                     'top_ls_ratio', 'top_long_pct', 'top_ls_change_12h',
                     'top_ls_change_24h', 'top_ls_zscore',
                     'global_ls_ratio', 'global_long_pct', 'ls_divergence',
                     'funding_rate_binance', 'funding_surprise']:
            if col not in df.columns:
                df[col] = 0.0

    # ── News coverage flags (expected by CatBoost) ──
    if 'news_coverage_ok' not in df.columns:
        coverage_col = None
        if 'news_count_24h' in df.columns:
            coverage_col = 'news_count_24h'
        elif 'market_news_count_24h' in df.columns:
            coverage_col = 'market_news_count_24h'
        if coverage_col:
            has_coverage = ~df[coverage_col].isna() & (df[coverage_col] != 0)
            df['news_coverage_ok'] = has_coverage.astype(float)
            df['news_event'] = np.where(has_coverage, (df[coverage_col] > 0).astype(float), np.nan)
        else:
            df['news_coverage_ok'] = 0.0
            df['news_event'] = np.nan

    # Replace inf/nan in new columns
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    # v6: 12h-specific features
    df = add_12h_features(df)

    # XGBoost news interaction features
    df = add_news_interaction_features(df)

    return df


def add_12h_features(df):
    """
    v7 features optimized for 12h holding period.
    """
    for sym, grp in df.groupby('symbol'):
        c = grp['close']
        v = grp['volume']
        h = grp['high']
        l = grp['low']
        idx = grp.index

        r12 = c.pct_change(12)
        r12_mean = r12.rolling(168).mean()
        r12_std = r12.rolling(168).std() + 1e-10
        df.loc[idx, 'mom_12h_zscore'] = (r12 - r12_mean) / r12_std

        vwap_12 = (c * v).rolling(12).sum() / (v.rolling(12).sum() + 1e-10)
        df.loc[idx, 'vwap_12h_dist'] = c / vwap_12 - 1

        df.loc[idx, 'mom_3d'] = c.pct_change(72)
        df.loc[idx, 'mom_7d'] = c.pct_change(168)

        ret_12 = c.pct_change(12)
        ret_12_prev = c.shift(12).pct_change(12)
        df.loc[idx, 'mom_accel_12h'] = ret_12 - ret_12_prev

        vol_12 = v.rolling(12).mean()
        vol_48 = v.rolling(48).mean() + 1e-10
        df.loc[idx, 'vol_trend_12_48'] = vol_12 / vol_48 - 1

        hours = grp['timestamp'].dt.hour
        df.loc[idx, 'is_asian_session'] = ((hours >= 0) & (hours < 12)).astype(float)

        h12 = h.rolling(12).max()
        l12 = l.rolling(12).min()
        range12 = (h12 - l12) / (c + 1e-10)
        range_avg = range12.rolling(168).mean() + 1e-10
        df.loc[idx, 'range_expansion_12h'] = range12 / range_avg - 1

        # v7 NEW features
        range_pos = (c - l12) / (h12 - l12 + 1e-10)
        df.loc[idx, 'range_position_12h'] = range_pos

        vwap_change = (c * v).rolling(12).sum() / (v.rolling(12).sum() + 1e-10)
        vwap_prev = (c.shift(12) * v.shift(12)).rolling(12).sum() / (v.shift(12).rolling(12).sum() + 1e-10)
        df.loc[idx, 'vwpc_12h'] = vwap_change / (vwap_prev + 1e-10) - 1

        hh = (h > h.shift(1)).rolling(12).sum()
        ll = (l < l.shift(1)).rolling(12).sum()
        df.loc[idx, 'hh_count_12h'] = hh
        df.loc[idx, 'll_count_12h'] = ll
        df.loc[idx, 'trend_strength_12h'] = hh - ll

        vol_recent = c.pct_change().rolling(12).std()
        vol_base = c.pct_change().rolling(72).std() + 1e-10
        df.loc[idx, 'vol_crush_ratio'] = vol_recent / vol_base

        ret_abs = (c - c.shift(12)).abs()
        hi_lo_path = (h - l).rolling(12).sum()
        df.loc[idx, 'direction_quality_12h'] = ret_abs / (hi_lo_path + 1e-10)

    df['ret_12h_temp'] = df.groupby('symbol')['close'].transform(lambda x: x.pct_change(12))
    df['ret_12h_cs_rank'] = df.groupby('timestamp')['ret_12h_temp'].rank(pct=True)
    df.drop(columns=['ret_12h_temp'], inplace=True)

    df['vol_12h_sum'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(12).sum())
    df['vol_12h_cs_rank'] = df.groupby('timestamp')['vol_12h_sum'].rank(pct=True)
    df.drop(columns=['vol_12h_sum'], inplace=True)

    # v7: funding rate cross-sectional rank (if available)
    if 'funding_rate' in df.columns:
        df['funding_cs_rank'] = df.groupby('timestamp')['funding_rate'].rank(pct=True)
        df['cum_funding_24h'] = df.groupby('symbol')['funding_rate'].transform(
            lambda x: x.rolling(3, min_periods=1).sum())
        df['cum_funding_72h'] = df.groupby('symbol')['funding_rate'].transform(
            lambda x: x.rolling(9, min_periods=1).sum())

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    return df


def cross_sectional_rank(df, feat_cols):
    """Rank normalize features cross-sectionally per timestamp."""
    rank_cols = [c for c in feat_cols if c not in UNRANKED_COLS]
    for col in rank_cols:
        df[col] = df.groupby('timestamp')[col].rank(pct=True) - 0.5
    return df


# ============================================================
# SIGNAL GENERATION
# ============================================================

def load_lgb_models(results_dir):
    """Load saved LightGBM model files."""
    import lightgbm as lgb

    model_files = sorted(Path(results_dir).glob('lgb_model_seed_*.txt'))
    if not model_files:
        # Try single model file
        single = Path(results_dir) / 'lgb_model_v5.txt'
        if single.exists():
            model_files = [single]

    models = []
    for f in model_files:
        m = lgb.Booster(model_file=str(f))
        models.append(m)

    return models


def load_catboost_models(results_dir):
    """Load saved CatBoost model files."""
    from catboost import CatBoostRegressor

    model_files = sorted(Path(results_dir).glob('cb_model_seed_*.cbm'))
    if not model_files:
        return []

    models = []
    for f in model_files:
        m = CatBoostRegressor()
        m.load_model(str(f))
        models.append(m)

    return models


def load_xgboost_models(results_dir):
    """Load saved XGBoost model files."""
    import xgboost as xgb

    model_files = sorted(Path(results_dir).glob('xgb_model_seed_*.json'))
    if not model_files:
        return []

    models = []
    for f in model_files:
        m = xgb.XGBRegressor()
        m.load_model(str(f))
        models.append(m)

    return models


def generate_lgb_signal(df, models, feat_cols):
    """Generate signal from LGB model ensemble. Returns (signal_df, raw_preds_matrix)."""
    latest = df.groupby('symbol').last().reset_index()

    # Align features to what model expects
    model_features = models[0].feature_name()
    available = [f for f in model_features if f in latest.columns]
    missing = [f for f in model_features if f not in latest.columns]

    if missing:
        print(f"   ⚠️  LGB missing {len(missing)} features, padding with 0")
        for col in missing:
            latest[col] = 0.0

    X = latest[model_features].values

    all_preds = []
    for m in models:
        all_preds.append(m.predict(X))

    latest['pred_lgb'] = np.mean(all_preds, axis=0)
    # Return raw predictions for confidence calculation
    return latest[['symbol', 'pred_lgb']].copy(), np.array(all_preds)


def generate_signal(df, feat_cols, root):
    """
    Generate ensemble signal.
    Champion: LGB v6 (5 seeds) + LGB v7 (5 seeds) + CatBoost (5 seeds) = 15 models.
    Falls back to LGB v5 or feature-based signal.
    """
    signals = {}

    # Collect raw predictions from all individual models for confidence
    all_raw_preds = []  # list of (n_models, n_coins) arrays
    symbols_order = None

    # ── LGB v6 (12h target, best single model) ──
    # Prefer production models (*_prod) if they exist
    lgb_v6_dir = os.path.join(root, 'results_v6_prod')
    if not os.path.isdir(lgb_v6_dir):
        lgb_v6_dir = os.path.join(root, 'results', 'production', 'lgb_v6_no_news')
    if not os.path.isdir(lgb_v6_dir):
        lgb_v6_dir = os.path.join(root, 'results_v6')
    try:
        models = load_lgb_models(lgb_v6_dir)
        if models:
            sig, raw_preds = generate_lgb_signal(df, models, feat_cols)
            sig = sig.rename(columns={'pred_lgb': 'pred_lgb_v6'})
            signals['lgb_v6'] = sig
            if symbols_order is None:
                symbols_order = sig['symbol'].tolist()
            all_raw_preds.append(raw_preds)
            print(f"   ✅ LGB v6: {len(models)} models, {len(sig)} coins")
    except Exception as e:
        print(f"   ⚠️  LGB v6 failed: {e}")

    # ── LGB v7 (blended 6h+12h+24h target) ──
    lgb_v7_dir = os.path.join(root, 'results_v7_prod')
    if not os.path.isdir(lgb_v7_dir):
        lgb_v7_dir = os.path.join(root, 'results', 'production', 'lgb_v7_no_news')
    if not os.path.isdir(lgb_v7_dir):
        lgb_v7_dir = os.path.join(root, 'results_v7')
    try:
        models = load_lgb_models(lgb_v7_dir)
        if models:
            sig, raw_preds = generate_lgb_signal(df, models, feat_cols)
            sig = sig.rename(columns={'pred_lgb': 'pred_lgb_v7'})
            signals['lgb_v7'] = sig
            all_raw_preds.append(raw_preds)
            print(f"   ✅ LGB v7: {len(models)} models, {len(sig)} coins")
    except Exception as e:
        print(f"   ⚠️  LGB v7 failed: {e}")

    # ── CatBoost ──
    cb_dir = os.path.join(root, 'results_catboost_prod')
    if not os.path.isdir(cb_dir):
        cb_dir = os.path.join(root, 'results', 'production', 'catboost_with_news')
    if not os.path.isdir(cb_dir):
        cb_dir = os.path.join(root, 'results_catboost')
    try:
        cb_models = load_catboost_models(cb_dir)
        if cb_models:
            latest = df.groupby('symbol').last().reset_index()
            # CatBoost uses same features as LGB
            model_features = cb_models[0].feature_names_
            available = [f for f in model_features if f in latest.columns]
            missing = [f for f in model_features if f not in latest.columns]
            if missing:
                print(f"   ⚠️  CatBoost missing {len(missing)} features, padding with 0")
                for col in missing:
                    latest[col] = 0.0
            X = latest[model_features].values
            cb_preds = [m.predict(X) for m in cb_models]
            latest['pred_cb'] = np.mean(cb_preds, axis=0)
            signals['catboost'] = latest[['symbol', 'pred_cb']].copy()
            all_raw_preds.append(np.array(cb_preds))
            print(f"   ✅ CatBoost: {len(cb_models)} models, {len(latest)} coins")
    except Exception as e:
        print(f"   ⚠️  CatBoost failed: {e}")

    # ── XGBoost + News Interactions ──
    xgb_dir = os.path.join(root, 'results_xgboost_prod')
    if not os.path.isdir(xgb_dir):
        xgb_dir = os.path.join(root, 'results_xgboost')
    try:
        xgb_models = load_xgboost_models(xgb_dir)
        if xgb_models:
            latest = df.groupby('symbol').last().reset_index()
            fn_path = os.path.join(xgb_dir, 'feature_names.json')
            if os.path.exists(fn_path):
                with open(fn_path) as _f:
                    model_features = json.load(_f)
            else:
                model_features = xgb_models[0].get_booster().feature_names
            missing = [f for f in model_features if f not in latest.columns]
            if missing:
                print(f"   ⚠️  XGBoost missing {len(missing)} features, padding with 0")
                for col in missing:
                    latest[col] = 0.0
            X = latest[model_features].values
            xgb_preds = [m.predict(X) for m in xgb_models]
            latest['pred_xgb'] = np.mean(xgb_preds, axis=0)
            signals['xgboost'] = latest[['symbol', 'pred_xgb']].copy()
            all_raw_preds.append(np.array(xgb_preds))
            print(f"   ✅ XGBoost: {len(xgb_models)} models, {len(latest)} coins")
    except Exception as e:
        print(f"   ⚠️  XGBoost failed: {e}")

    # ── Fallback: LGB v5 ──
    if not signals:
        lgb_v5_dir = os.path.join(root, 'results_v5')
        try:
            models = load_lgb_models(lgb_v5_dir)
            if models:
                sig = generate_lgb_signal(df, models, feat_cols)
                signals['lgb_v5'] = sig
                print(f"   ✅ LGB v5 (fallback): {len(models)} models")
        except Exception as e:
            print(f"   ⚠️  LGB v5 failed: {e}")

    # ── Fallback: simple feature composite ──
    if not signals:
        print(f"   ⚠️  No trained models found — using signal from features")
        latest = df.groupby('symbol').last().reset_index()

        score_features = ['close_ma720_ratio', 'close_ma336_ratio', 'close_ma24_ratio',
                          'ret_sharpe_168h', 'ret_sharpe_24h', 'breadth_pct_positive']
        avail = [f for f in score_features if f in latest.columns]

        if avail:
            for col in avail:
                latest[col] = (latest[col] - latest[col].mean()) / (latest[col].std() + 1e-10)
            latest['pred_fallback'] = latest[avail].mean(axis=1)
            signals['fallback'] = latest[['symbol', 'pred_fallback']].copy()
            print(f"   ✅ Fallback signal: {len(avail)} features")

    if not signals:
        return None

    # Merge and normalize
    result = list(signals.values())[0]
    for _, other in list(signals.items())[1:]:
        result = result.merge(other, on='symbol', how='inner')

    pred_cols = [c for c in result.columns if c.startswith('pred_')]
    for col in pred_cols:
        result[col] = (result[col] - result[col].mean()) / (result[col].std() + 1e-10)

    result['score'] = sum(result[c] for c in pred_cols) / len(pred_cols)

    # ── Confidence: model agreement (low std → high confidence) ──
    if all_raw_preds:
        # Stack all individual model predictions: shape (n_total_models, n_coins)
        stacked = np.vstack(all_raw_preds)  # e.g. (15, 50)
        # Normalize each model group to same scale before computing std
        for i, raw in enumerate(all_raw_preds):
            for j in range(raw.shape[0]):
                row = raw[j]
                stacked[sum(r.shape[0] for r in all_raw_preds[:i]) + j] = \
                    (row - row.mean()) / (row.std() + 1e-10)
        model_std = np.std(stacked, axis=0)  # per-coin disagreement
        # confidence = 1 / (1 + std), ranges from 0 to 1
        confidence = 1.0 / (1.0 + model_std)
        result['confidence'] = confidence
        result['model_std'] = model_std
        avg_conf = confidence.mean()
        high_conf = (confidence > 0.6).sum()
        print(f"   📊 Confidence: avg={avg_conf:.3f}, high(>0.6)={high_conf}/{len(result)}")
    else:
        result['confidence'] = 0.5
        result['model_std'] = 0.0

    return result.sort_values('score', ascending=False).reset_index(drop=True)


# ============================================================
# PORTFOLIO CONSTRUCTION (risk-managed)
# ============================================================

def construct_portfolio(signals, capital, risk_cfg, state):
    """
    Risk-managed portfolio construction.

    state: dict tracking equity curve for DD stop.
    """
    n_long = risk_cfg['n_long']
    n_short = risk_cfg['n_short']
    kelly = risk_cfg['kelly_frac']

    # DD circuit breaker
    equity = state.get('equity', capital)
    peak = state.get('peak', capital)
    dd = equity / peak - 1

    if state.get('stopped', False):
        if dd > risk_cfg['dd_resume']:
            state['stopped'] = False
            print(f"   🟢 DD recovered to {dd*100:.1f}%, resuming trading")
        else:
            print(f"   🔴 DD stop active ({dd*100:.1f}%), skipping cycle")
            return []

    if dd < risk_cfg['dd_stop']:
        state['stopped'] = True
        print(f"   🔴 DD hit {dd*100:.1f}% (limit {risk_cfg['dd_stop']*100:.0f}%), stopping")
        return []

    # Vol targeting: scale position based on recent realized vol
    vol_history = state.get('recent_rets', [])
    if len(vol_history) >= 6:
        realized_vol = np.std(vol_history[-risk_cfg['vol_lookback']:]) + 1e-10
        vol_scale = np.clip(risk_cfg['vol_target'] / realized_vol, 0.1, 3.0)
    else:
        vol_scale = 1.0

    # Effective allocation
    leverage = risk_cfg.get('leverage', 1)
    margin_util = risk_cfg.get('margin_utilization', 0.90)
    effective_kelly = kelly * vol_scale
    long_capital = capital * margin_util * 0.5 * effective_kelly * leverage
    short_capital = capital * margin_util * 0.5 * effective_kelly * leverage

    # Confidence check
    conf_thresh = risk_cfg.get('confidence_threshold', 0.0)
    if conf_thresh > 0:
        scores = signals['score'].values
        max_spread = scores.max() - scores.min()
        if max_spread < conf_thresh:
            print(f"   ⚠️  Signal too weak (spread={max_spread:.2f} < {conf_thresh}), skipping")
            return []

    # Filter out coins unavailable on exchange
    if SYMBOLS_TO_OKX:
        tradeable = set(SYMBOLS_TO_OKX.keys())
        before = len(signals)
        signals = signals[signals['symbol'].isin(tradeable)].copy()
        dropped = before - len(signals)
        if dropped:
            print(f"   ⚠️  Filtered {dropped} untradeable coins from signals")

    # Min score filter: skip weak signals
    min_score = risk_cfg.get('min_score', 0.0)
    if min_score > 0:
        before = len(signals)
        signals = signals[signals['score'].abs() >= min_score].copy()
        weak = before - len(signals)
        if weak:
            print(f"   ⚠️  Filtered {weak} weak signals (|score| < {min_score})")

    # Build positions
    signals = signals.sort_values('score', ascending=False).reset_index(drop=True)

    # Split by sign: only long positive scores, only short negative scores
    long_candidates = signals[signals['score'] > 0]
    short_candidates = signals[signals['score'] < 0].iloc[::-1]  # most negative first

    actual_n_long = min(n_long, len(long_candidates))
    actual_n_short = min(n_short, len(short_candidates))

    positions = []

    # Edge-boost: weight proportional to |score| (mirrors champion backtest)
    edge_boost = risk_cfg.get('edge_boost', True)

    def _weighted_alloc(candidates, total_capital, side, n_slots):
        """Allocate capital proportional to |score| × confidence with edge-boost.
        Max per position = confidence (prevents concentration while keeping capital at work)."""
        if len(candidates) == 0:
            return []
        scores_abs = candidates['score'].abs().values
        conf_vals = candidates['confidence'].values if 'confidence' in candidates.columns else np.ones(len(candidates))
        if edge_boost and scores_abs.max() > 0:
            p75 = np.percentile(scores_abs, 75) if len(scores_abs) >= 4 else scores_abs.mean()
            p75 = max(p75, 1e-6)
            raw_w = np.array([1.0 + min(s / p75, 3.0) for s in scores_abs])
        else:
            raw_w = np.ones(len(candidates))
        # Multiply by confidence: high-agreement positions get more capital
        raw_w = raw_w * conf_vals
        weights = raw_w / raw_w.sum()
        # Cap per position: max share = confidence, hard ceiling 40%
        # High confidence → allowed more capital; low confidence → capped tighter
        max_weights = np.clip(conf_vals, 0.15, 0.40)  # floor 15%, ceiling 40%
        capped = np.minimum(weights, max_weights)
        # Don't re-normalize — leftover capital stays idle (safer than over-concentrating)
        result = []
        used_capital = 0
        for (_, row), w in zip(candidates.iterrows(), capped):
            usd = round(total_capital * w, 2)
            if usd < 5:  # OKX minimum
                continue
            used_capital += usd
            conf = round(row.get('confidence', 0.5), 3) if 'confidence' in row.index else 0.5
            result.append({
                'symbol': row['symbol'],
                'side': side,
                'usd': usd,
                'score': round(row['score'], 4),
                'confidence': conf,
            })
        unused = total_capital - used_capital
        if unused > 50:
            avg_conf = float(np.mean(conf_vals))
            print(f"   ℹ️  {side}: ${unused:.0f} unallocated (avg conf {avg_conf:.0%}, {len(candidates)} candidates)")
        return result

    if actual_n_long > 0:
        positions.extend(_weighted_alloc(long_candidates.head(actual_n_long), long_capital, 'long', n_long))

    if actual_n_short > 0:
        positions.extend(_weighted_alloc(short_candidates.head(actual_n_short), short_capital, 'short', n_short))

    total_alloc = sum(p['usd'] for p in positions)
    n_l = sum(1 for p in positions if p['side'] == 'long')
    n_s = sum(1 for p in positions if p['side'] == 'short')
    print(f"   📊 Allocating ${total_alloc:.0f} of ${capital:.0f} "
          f"({n_l}L+{n_s}S, kelly={kelly:.0%} × vol_scale={vol_scale:.2f})")

    return positions


# ============================================================
# OKX EXECUTION
# ============================================================

def init_exchange(mode='paper'):
    """Initialize OKX via ccxt."""
    import ccxt

    api_key = os.environ.get('OKX_API_KEY', '')
    secret = os.environ.get('OKX_SECRET', '')
    passphrase = os.environ.get('OKX_PASSPHRASE', '')

    if not api_key:
        return None

    exchange = ccxt.okx({
        'apiKey': api_key,
        'secret': secret,
        'password': passphrase,
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'},
    })

    if mode == 'paper' or os.environ.get('OKX_DEMO', '0') == '1':
        exchange.set_sandbox_mode(True)
        print("   📋 OKX DEMO mode")

    exchange.session.verify = False

    # Verify account config
    try:
        config = exchange.private_get_account_config()
        acct_lv = config.get('data', [{}])[0].get('acctLv', '?')
        pos_mode = config.get('data', [{}])[0].get('posMode', '?')
        mode_names = {'1': 'Simple', '2': 'Single-currency margin',
                      '3': 'Multi-currency', '4': 'Portfolio'}
        print(f"   ✅ Account mode: {mode_names.get(acct_lv, acct_lv)} (acctLv={acct_lv})")
        print(f"   ✅ Position mode: {pos_mode}")
        if acct_lv == '1':
            print("   ⚠️  WARNING: Simple mode — swaps won't work! Switch to Single-currency margin on OKX website")
    except Exception as e:
        print(f"   ⚠️  Config check: {e}")

    try:
        balance = exchange.fetch_balance()
        usdt = balance.get('USDT', {}).get('free', 0)
        print(f"   💰 Balance: ${usdt:.2f} USDT")
    except Exception as e:
        print(f"   ⚠️  Balance check: {e}")

    return exchange


def close_all(exchange):
    """Close all open positions with retry."""
    if not exchange:
        return
    try:
        positions = _api_call_with_retry(exchange.fetch_positions)
        for pos in positions:
            contracts = abs(float(pos.get('contracts', 0)))
            if contracts > 0:
                # Net mode: sell to close long, buy to close short
                side = 'sell' if pos['side'] == 'long' else 'buy'
                _api_call_with_retry(
                    exchange.create_order,
                    symbol=pos['symbol'], type='market', side=side,
                    amount=contracts,
                    params={'tdMode': 'cross'},
                )
                print(f"      ✅ Closed {pos['side']} {pos['symbol']}")
    except Exception as e:
        print(f"      ⚠️  Close failed: {e}")


def _fetch_contract_specs(exchange):
    """Fetch OKX contract specs (ctVal, minSz) for all swap markets."""
    specs = {}
    try:
        markets = exchange.load_markets()
        for sym, mkt in markets.items():
            if mkt.get('swap'):
                try:
                    ct_val = float(mkt.get('contractSize') or 1)
                    min_amt = mkt.get('limits', {}).get('amount', {}).get('min')
                    min_sz = float(min_amt) if min_amt is not None else 1
                    prec_amt = mkt.get('precision', {}).get('amount')
                    prec = int(prec_amt) if prec_amt is not None else 0
                    specs[sym] = {
                        'ctVal': ct_val,
                        'minSz': min_sz,
                        'precision': prec,
                    }
                except (TypeError, ValueError):
                    pass
    except Exception as e:
        print(f"   ⚠️  Failed to fetch market specs: {e}")
    return specs


def _api_call_with_retry(func, *args, max_retries=3, **kwargs):
    """Execute API call with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err_str = str(e).lower()
            # Don't retry on auth errors, insufficient balance, compliance,
            # max order size, or symbol-not-found errors
            if any(x in err_str for x in [
                'auth', 'apikey', 'insufficient', 'invalid',
                'compliance', '51155', 'does not have market',
                'account mode', '51010', '51008',
                'maximum amount', '51202',
            ]):
                raise
            if attempt < max_retries - 1:
                wait = 2 ** attempt * (1 + 0.5 * (hash(str(e)) % 100) / 100)
                print(f"      ⏳ Retry {attempt+1}/{max_retries} in {wait:.1f}s: {e}")
                time.sleep(wait)
            else:
                raise


def execute(exchange, positions, dry_run=True, leverage=3, tracker=None):
    """Execute positions on OKX with proper contract sizing."""
    results = []

    # Load contract specs for USD→contract conversion
    contract_specs = {}
    if exchange and not dry_run:
        contract_specs = _fetch_contract_specs(exchange)

    for pos in positions:
        okx_sym = SYMBOLS_TO_OKX.get(pos['symbol'])
        if not okx_sym:
            continue

        side = 'buy' if pos['side'] == 'long' else 'sell'

        if dry_run:
            conf = pos.get('confidence', 0)
            print(f"      [DRY] {side.upper():4s} ${pos['usd']:>7.0f} {okx_sym} "
                  f"(score: {pos['score']:+.3f}, conf: {conf:.0%})")
            results.append({**pos, 'status': 'dry_run'})
            continue

        try:
            # Set leverage on OKX (must match cross margin mode)
            try:
                _api_call_with_retry(
                    exchange.set_leverage, leverage, okx_sym,
                    params={'mgnMode': 'cross'}
                )
            except Exception:
                pass

            # Convert USD → contracts
            # OKX swaps: 1 contract = ctVal units of base currency
            # Amount needed in contracts = usd_amount / (ctVal * current_price)
            spec = contract_specs.get(okx_sym, {})
            ctVal = spec.get('ctVal', 1)
            minSz = spec.get('minSz', 1)

            # Get current price via ticker
            ticker = _api_call_with_retry(exchange.fetch_ticker, okx_sym)
            price = ticker['last'] or ticker['close']

            contracts = pos['usd'] / (ctVal * price)
            contracts = max(minSz, round(contracts / minSz) * minSz)  # Round to minSz

            # Use ccxt precision (handles OKX step-size format)
            try:
                contracts = float(exchange.amount_to_precision(okx_sym, contracts))
            except Exception:
                contracts = int(contracts)

            if contracts < minSz:
                print(f"      ⏭️  {okx_sym}: ${pos['usd']:.0f} → {contracts} contracts < min {minSz}")
                results.append({**pos, 'status': 'skipped', 'reason': 'below_min'})
                continue

            order = _api_call_with_retry(
                exchange.create_order,
                symbol=okx_sym, type='market', side=side,
                amount=contracts,
                params={
                    'tdMode': 'cross',
                },
            )
            actual_usd = contracts * ctVal * price
            print(f"      ✅ {side.upper():4s} {contracts} cts ≈${actual_usd:.0f} {okx_sym} → {order['id']}")
            results.append({
                **pos, 'status': 'filled', 'order_id': order['id'],
                'contracts': contracts, 'price': price, 'actual_usd': actual_usd,
            })
            # Record trade open
            if tracker:
                tracker.record_open(
                    symbol=pos['symbol'], side=pos['side'], usd=actual_usd,
                    score=pos.get('score', 0), entry_price=price, contracts=contracts,
                )
        except Exception as e:
            print(f"      ❌ {side.upper():4s} ${pos['usd']:>7.0f} {okx_sym} → {e}")
            results.append({**pos, 'status': 'error', 'error': str(e)})

    return results


def diff_rebalance(exchange, target_positions, leverage=3, tracker=None):
    """Diff-based rebalancing: only trade what changed.
    
    Compares current OKX positions to target, produces:
    - Closes for positions no longer wanted
    - Opens for new positions
    - Skips positions that are already correct (saves commissions)
    
    Returns list of results (same format as execute).
    """
    if not exchange:
        return execute(exchange, target_positions, dry_run=True, leverage=leverage, tracker=tracker)

    results = []

    # 1. Fetch current positions from OKX
    current = {}
    try:
        live_positions = _api_call_with_retry(exchange.fetch_positions)
        for pos in live_positions:
            contracts = float(pos.get('contracts', 0))
            if contracts > 0:
                current[pos['symbol']] = {
                    'side': pos['side'],  # 'long' or 'short'
                    'contracts': contracts,
                    'notional': float(pos.get('notional', 0)),
                }
    except Exception as e:
        print(f"   ⚠️  Cannot fetch positions, falling back to close-all: {e}")
        close_all(exchange)
        return execute(exchange, target_positions, dry_run=False, leverage=leverage)

    # 2. Build target map: okx_symbol → {side, usd}
    target = {}
    for pos in target_positions:
        okx_sym = SYMBOLS_TO_OKX.get(pos['symbol'])
        if okx_sym:
            target[okx_sym] = pos

    # 3. Close positions not in target (or side changed)
    for sym, cur in current.items():
        tgt = target.get(sym)
        if tgt is None or tgt['side'] != cur['side']:
            # Need to close this position
            side = 'sell' if cur['side'] == 'long' else 'buy'
            try:
                _api_call_with_retry(
                    exchange.create_order,
                    symbol=sym, type='market', side=side,
                    amount=cur['contracts'],
                    params={'tdMode': 'cross'},
                )
                print(f"      🔄 Closed {cur['side']} {sym} ({cur['contracts']} cts)")
                results.append({'symbol': sym, 'side': cur['side'], 'action': 'close',
                                'status': 'filled'})
                # Record trade close with P&L
                if tracker:
                    # Find original symbol from OKX unified
                    orig_sym = None
                    for s, o in SYMBOLS_TO_OKX.items():
                        if o == sym:
                            orig_sym = s
                            break
                    if orig_sym:
                        try:
                            t = exchange.fetch_ticker(sym)
                            exit_price = t['last'] or t['close']
                            pnl, pnl_pct, result = tracker.record_close(orig_sym, cur['side'], exit_price)
                            if pnl is not None:
                                print(f"         {result} P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)")
                        except Exception:
                            pass
            except Exception as e:
                print(f"      ❌ Close {sym}: {e}")
                results.append({'symbol': sym, 'side': cur.get('side', '?'),
                                'usd': cur.get('notional', 0),
                                'action': 'close', 'status': 'error',
                                'error': str(e)})

    # 4. Open new positions (not currently held, or side changed)
    new_positions = []
    kept = 0
    for okx_sym, pos in target.items():
        cur = current.get(okx_sym)
        if cur and cur['side'] == pos['side']:
            # Already have this position in the right direction — keep it
            kept += 1
            results.append({**pos, 'status': 'kept', 'action': 'hold'})
            continue
        new_positions.append(pos)

    if kept:
        print(f"      ♻️  Kept {kept} existing positions unchanged")

    if new_positions:
        # Check actual available margin before opening new positions
        try:
            time.sleep(0.5)  # let OKX settle after closes
            bal = _api_call_with_retry(exchange.fetch_balance)
            free_usd = float(bal.get('USDT', {}).get('free', 0))
            # Reserve 5% of free as safety buffer
            available_margin = free_usd * 0.95
            # Max notional we can open = available_margin * leverage
            max_new_notional = available_margin * leverage
            total_requested = sum(p['usd'] for p in new_positions)

            if total_requested > max_new_notional and max_new_notional > 0:
                scale = max_new_notional / total_requested
                print(f"      💰 Free margin: ${free_usd:.0f} → "
                      f"scaling new positions to {scale:.0%} "
                      f"(${max_new_notional:.0f}/{total_requested:.0f} notional)")
                for p in new_positions:
                    p['usd'] = round(p['usd'] * scale, 2)
            elif max_new_notional <= 10:
                print(f"      ⚠️  No margin available (${free_usd:.0f} free). Skipping new opens.")
                for p in new_positions:
                    results.append({**p, 'status': 'error', 'error': 'no_margin'})
                new_positions = []
        except Exception as e:
            print(f"      ⚠️  Could not check margin: {e}")

        if new_positions:
            new_results = execute(exchange, new_positions, dry_run=False, leverage=leverage, tracker=tracker)
            results.extend(new_results)

    return results


# ============================================================
# LOCAL PAPER TRADING SIMULATOR
# ============================================================

def fetch_current_prices(symbols):
    """Fetch current prices for all symbols from Binance."""
    try:
        import ccxt
    except ImportError:
        return {}

    exchange = ccxt.binance({'enableRateLimit': True})
    exchange.session.verify = False

    prices = {}
    try:
        tickers = exchange.fetch_tickers([s for s in symbols])
        for sym, tick in tickers.items():
            if tick and tick.get('last'):
                prices[sym] = tick['last']
    except Exception:
        # Fallback: fetch one by one
        for sym in symbols:
            try:
                tick = exchange.fetch_ticker(sym)
                if tick and tick.get('last'):
                    prices[sym] = tick['last']
            except Exception:
                pass
    return prices


def sim_settle_positions(state, prices):
    """
    Settle open positions using current prices.
    Returns PnL for this cycle.
    """
    open_positions = state.get('sim_positions', [])
    if not open_positions:
        return 0.0, []

    settled = []
    total_pnl = 0.0

    for pos in open_positions:
        sym = pos['symbol']
        entry_price = pos.get('entry_price', 0)
        usd = pos['usd']
        side = pos['side']

        current_price = prices.get(sym)
        if not current_price or not entry_price:
            # Can't settle — carry forward
            settled.append({**pos, 'pnl': 0.0, 'status': 'no_price'})
            continue

        # Calculate return
        price_ret = (current_price - entry_price) / entry_price
        if side == 'short':
            price_ret = -price_ret

        # Deduct trading costs (entry + exit)
        cost_rate = 0.0003 + 0.0001  # taker fee + slippage, each side
        net_ret = price_ret - cost_rate * 2  # round-trip

        pnl = usd * net_ret
        total_pnl += pnl

        settled.append({
            'symbol': sym,
            'side': side,
            'usd': usd,
            'entry_price': entry_price,
            'exit_price': current_price,
            'return_%': round(price_ret * 100, 2),
            'net_return_%': round(net_ret * 100, 2),
            'pnl': round(pnl, 2),
        })

    return round(total_pnl, 2), settled


def sim_open_positions(positions, prices):
    """
    Record new positions with entry prices.
    """
    new_positions = []
    for pos in positions:
        sym = pos['symbol']
        price = prices.get(sym)
        if not price:
            print(f"      ⚠️  No price for {sym}, skipping")
            continue
        new_positions.append({
            **pos,
            'entry_price': price,
            'entry_time': datetime.now(timezone.utc).isoformat(),
        })
    return new_positions


def sim_print_summary(state, log_dir):
    """Print sim portfolio summary and save equity curve."""
    equity = state.get('equity', 1000)
    peak = state.get('peak', 1000)
    initial = state.get('initial_capital', 1000)
    dd = equity / peak - 1 if peak > 0 else 0
    total_ret = equity / initial - 1 if initial > 0 else 0
    n_cycles = state.get('n_cycles', 0)
    total_pnl = state.get('total_pnl', 0)

    print(f"\n   {'=' * 50}")
    print(f"   📊 PORTFOLIO SUMMARY (cycle #{n_cycles})")
    print(f"   {'=' * 50}")
    print(f"   Initial:     ${initial:,.0f}")
    print(f"   Current:     ${equity:,.2f}")
    print(f"   Total PnL:   ${total_pnl:+,.2f} ({total_ret:+.1%})")
    print(f"   Peak:        ${peak:,.2f}")
    print(f"   Drawdown:    {dd:.1%}")
    print(f"   Cycles:      {n_cycles}")

    # Win/loss stats
    cycle_pnls = state.get('cycle_pnls', [])
    if cycle_pnls:
        wins = [p for p in cycle_pnls if p > 0]
        losses = [p for p in cycle_pnls if p < 0]
        print(f"   Win rate:    {len(wins)}/{len(cycle_pnls)} ({len(wins)/len(cycle_pnls):.0%})")
        if wins:
            print(f"   Avg win:     ${np.mean(wins):+.2f}")
        if losses:
            print(f"   Avg loss:    ${np.mean(losses):+.2f}")

    # Save equity history
    eq_history = state.get('equity_history', [])
    if eq_history:
        eq_path = os.path.join(log_dir, 'sim_equity.csv')
        pd.DataFrame(eq_history).to_csv(eq_path, index=False)

    print(f"   {'=' * 50}")


# ============================================================
# STATE MANAGEMENT
# ============================================================

def load_state(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_state(state, path):
    # Filter out non-serializable runtime-only keys
    serializable = {k: v for k, v in state.items()
                    if k not in ('_last_signals', '_last_positions')}
    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)


# ============================================================
# DASHBOARD JSON EXPORT
# ============================================================

DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard', 'data')

def _export_dashboard(state, positions, signals, tracker, risk_cfg, args, rebal_hours, exchange=None):
    """Write dashboard.json for the web UI. Safe — never crashes the bot."""
    try:
        os.makedirs(DASHBOARD_DIR, exist_ok=True)
        now = datetime.now(timezone.utc)

        # Equity history from state — append live point every refresh
        eq_hist = state.get('equity_history', [])

        # Trade stats
        stats = tracker.get_stats() if tracker else {}

        # Closed trades (newest first, last 50)
        closed_trades = []
        if tracker and len(tracker.df) > 0:
            closed_df = tracker.df[tracker.df['status'] == 'closed'].copy()
            if len(closed_df) > 0:
                closed_df = closed_df.sort_values('exit_time', ascending=False).head(50)
                for _, row in closed_df.iterrows():
                    closed_trades.append({
                        'symbol': row.get('symbol', ''),
                        'side': row.get('side', ''),
                        'pnl': float(row.get('pnl_usd', 0) or 0),
                        'pnl_pct': float(row.get('pnl_pct', 0) or 0),
                        'usd': float(row.get('usd', 0) or 0),
                        'closed': row.get('exit_time', ''),
                        'hold_h': float(row.get('hold_hours', 0) or 0),
                        'score': float(row.get('score', 0) or 0),
                    })

        # Open positions — try live from OKX, fallback to target
        # Build score + confidence lookup: prefer ENTRY scores (not current signal)
        entry_scores = state.get('entry_scores', {})
        score_lookup = {}
        conf_lookup = {}
        # 1. Entry scores first (from when position was opened)
        for sym, es in entry_scores.items():
            score_lookup[sym] = es.get('score', 0)
            conf_lookup[sym] = es.get('confidence', 0.5)
        # 2. Current cycle positions (overwrite if fresh)
        if positions:
            for p in positions:
                score_lookup[p['symbol']] = p.get('score', 0)
                conf_lookup[p['symbol']] = p.get('confidence', 0.5)
        # 3. Signals as last resort
        if signals is not None and len(signals) > 0:
            for _, row in signals.iterrows():
                sym = row.get('symbol', '')
                if sym and sym not in score_lookup:
                    score_lookup[sym] = round(float(row.get('score', 0)), 4)
                if sym and sym not in conf_lookup and 'confidence' in row.index:
                    conf_lookup[sym] = round(float(row.get('confidence', 0.5)), 3)

        pos_data = []
        if exchange:
            try:
                live = exchange.fetch_positions()
                for p in live:
                    contracts = float(p.get('contracts', 0))
                    if contracts > 0:
                        sym = p.get('symbol', '').replace(':USDT', '')
                        pos_data.append({
                            'symbol': sym,
                            'side': p.get('side', ''),
                            'notional': abs(float(p.get('notional', 0))),
                            'upnl': float(p.get('unrealizedPnl', 0) or 0),
                            'upnl_pct': float(p.get('percentage', 0) or 0),
                            'score': score_lookup.get(sym, 0),
                            'confidence': conf_lookup.get(sym, 0.5),
                        })
            except Exception:
                pass

        if not pos_data and positions:
            for p in positions:
                pos_data.append({
                    'symbol': p['symbol'],
                    'side': p['side'],
                    'notional': p['usd'],
                    'upnl': 0,
                    'upnl_pct': 0,
                    'score': p.get('score', 0),
                    'confidence': p.get('confidence', 0.5),
                })

        # Open/pending orders from OKX
        orders_data = []
        if exchange:
            try:
                open_orders = exchange.fetch_open_orders()
                for o in open_orders:
                    sym = o.get('symbol', '').replace(':USDT', '')
                    orders_data.append({
                        'symbol': sym,
                        'side': o.get('side', ''),
                        'type': o.get('type', 'limit'),
                        'price': float(o.get('price', 0) or 0),
                        'amount': float(o.get('amount', 0) or 0),
                        'filled': float(o.get('filled', 0) or 0),
                        'status': o.get('status', 'open'),
                        'created': o.get('datetime', ''),
                    })
            except Exception:
                pass

        # Signals from latest model run
        sig_data = []
        if signals is not None and len(signals) > 0:
            top = signals.head(10)
            bot = signals.tail(10)
            combined = pd.concat([top, bot]).drop_duplicates()
            for _, row in combined.iterrows():
                sig_data.append({
                    'symbol': row.get('symbol', ''),
                    'score': round(float(row.get('score', 0)), 4),
                    'confidence': round(float(row.get('confidence', 0.5)), 3),
                })

        # Live equity = capital + realized PnL + unrealized PnL
        realized_pnl = stats.get('total_pnl', 0) or 0
        unrealized_pnl = sum(p.get('upnl', 0) for p in pos_data)
        live_equity = args.capital + realized_pnl + unrealized_pnl

        # Max DD based on live equity
        peak = max(args.capital, state.get('peak', args.capital), live_equity)
        max_dd = (live_equity / peak - 1) if peak > 0 else 0

        # Next rebal
        hour_slot = (now.hour // rebal_hours + 1) * rebal_hours
        next_time = now.replace(hour=0, minute=5, second=0, microsecond=0) + timedelta(hours=hour_slot)
        if next_time <= now:
            next_time += timedelta(hours=rebal_hours)
        hours_left = (next_time - now).total_seconds() / 3600

        # Margin used
        margin_used = sum(p.get('notional', 0) for p in pos_data) / risk_cfg.get('leverage', 3)

        dashboard = {
            'updated': now.isoformat(),
            'mode': args.mode,
            'capital': args.capital,
            'equity': round(live_equity, 2),
            'leverage': risk_cfg.get('leverage', 3),
            'margin_used': round(margin_used, 0),
            'rebal_hours': rebal_hours,
            'next_rebal': f"{hours_left:.1f}h",
            'cycle': state.get('cycle', 0),
            'max_dd': round(max_dd, 4),
            'dd_stop': risk_cfg.get('dd_stop', -0.20),
            'models': '15 (LGB v6×5 + v7×5 + CB×5)',
            'min_score': risk_cfg.get('min_score', 0.0),
            'edge_boost': risk_cfg.get('edge_boost', False),

            # Stats
            'total_trades': stats.get('total_trades', 0),
            'win_rate': round(stats.get('win_rate', 0), 3),
            'total_pnl': stats.get('total_pnl', 0),
            'avg_win': stats.get('avg_win', 0),
            'avg_loss': stats.get('avg_loss', 0),

            # Data
            'equity_history': eq_hist[-500:],  # last 500 points
            'positions': pos_data,
            'trades': closed_trades,
            'signals': sig_data,
            'orders': orders_data,
        }

        # Append live equity point to history (every export = every 5min from bg thread)
        eq_hist.append({
            'ts': now.isoformat(),
            'equity': round(live_equity, 2),
            'dd': round(max_dd, 4),
        })
        # Keep last 2000 points (~7 days at 5min interval)
        state['equity_history'] = eq_hist[-2000:]
        dashboard['equity_history'] = eq_hist[-2000:]

        dash_path = os.path.join(DASHBOARD_DIR, 'dashboard.json')
        with open(dash_path, 'w') as f:
            json.dump(dashboard, f, default=str)
        print(f"   📊 Dashboard updated")

    except Exception as e:
        print(f"   ⚠️  Dashboard export failed: {e}")


# ── Background dashboard refresh (every 5 min) ──────────────
_dash_thread = None
_dash_stop = threading.Event()

def _dashboard_bg_loop(state, tracker, risk_cfg, args, rebal_hours, exchange):
    """Background thread: refresh dashboard.json with live OKX data every 5 min."""
    import copy
    while not _dash_stop.is_set():
        _dash_stop.wait(300)  # 5 minutes
        if _dash_stop.is_set():
            break
        try:
            # Use last known signals from state
            signals = state.get('_last_signals', None)
            positions = state.get('sim_positions', [])
            _export_dashboard(
                state=state, positions=positions, signals=signals,
                tracker=tracker, risk_cfg=risk_cfg, args=args,
                rebal_hours=rebal_hours, exchange=exchange,
            )
        except Exception as e:
            print(f"   ⚠️  BG dashboard refresh failed: {e}")


def start_dashboard_bg(state, tracker, risk_cfg, args, rebal_hours, exchange):
    """Start background dashboard refresh thread."""
    global _dash_thread
    _dash_stop.clear()
    _dash_thread = threading.Thread(
        target=_dashboard_bg_loop,
        args=(state, tracker, risk_cfg, args, rebal_hours, exchange),
        daemon=True,
    )
    _dash_thread.start()
    print("   📊 Dashboard background refresh started (every 5 min)")


def stop_dashboard_bg():
    """Stop background dashboard refresh thread."""
    global _dash_thread
    _dash_stop.set()
    if _dash_thread and _dash_thread.is_alive():
        _dash_thread.join(timeout=10)
    _dash_thread = None


# ============================================================
# TRADE TRACKER — per-trade P&L logging
# ============================================================

class TradeTracker:
    """Tracks every trade open/close with P&L, writes to CSV."""

    CSV_COLUMNS = [
        'trade_id', 'symbol', 'side', 'usd', 'score',
        'entry_time', 'entry_price', 'contracts',
        'exit_time', 'exit_price', 'pnl_usd', 'pnl_pct', 'result',
        'hold_hours', 'status',  # status: open / closed
    ]

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self._next_id = 1
        # Load existing trades if file exists
        if os.path.exists(csv_path):
            try:
                self.df = pd.read_csv(csv_path)
                if len(self.df) > 0:
                    self._next_id = int(self.df['trade_id'].max()) + 1
            except Exception:
                self.df = pd.DataFrame(columns=self.CSV_COLUMNS)
        else:
            self.df = pd.DataFrame(columns=self.CSV_COLUMNS)

    def _save(self):
        self.df.to_csv(self.csv_path, index=False)

    def record_open(self, symbol, side, usd, score, entry_price, contracts):
        """Record a new trade open. Returns trade_id."""
        tid = self._next_id
        self._next_id += 1
        now = datetime.now(timezone.utc).isoformat()
        row = {
            'trade_id': tid, 'symbol': symbol, 'side': side, 'usd': usd,
            'score': round(score, 4), 'entry_time': now,
            'entry_price': entry_price, 'contracts': contracts,
            'exit_time': None, 'exit_price': None,
            'pnl_usd': None, 'pnl_pct': None, 'result': None,
            'hold_hours': None, 'status': 'open',
        }
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        self._save()
        return tid

    def record_close(self, symbol, side, exit_price):
        """Close an open trade, compute P&L. Returns (pnl_usd, pnl_pct, result)."""
        mask = (
            (self.df['symbol'] == symbol) &
            (self.df['side'] == side) &
            (self.df['status'] == 'open')
        )
        idx = self.df.index[mask]
        if len(idx) == 0:
            return None, None, None

        row_idx = idx[-1]  # latest open trade for this symbol/side
        row = self.df.loc[row_idx]
        entry_price = float(row['entry_price'])
        usd_alloc = float(row['usd'])
        entry_time = pd.to_datetime(row['entry_time'])

        # P&L calculation
        if side == 'long':
            pnl_pct = (exit_price - entry_price) / entry_price
        else:  # short
            pnl_pct = (entry_price - exit_price) / entry_price
        pnl_usd = pnl_pct * usd_alloc

        now = datetime.now(timezone.utc)
        hold_hours = (now - entry_time).total_seconds() / 3600
        result = '✅' if pnl_usd >= 0 else '❌'

        self.df.loc[row_idx, 'exit_time'] = now.isoformat()
        self.df.loc[row_idx, 'exit_price'] = exit_price
        self.df.loc[row_idx, 'pnl_usd'] = round(pnl_usd, 2)
        self.df.loc[row_idx, 'pnl_pct'] = round(pnl_pct * 100, 2)
        self.df.loc[row_idx, 'result'] = result
        self.df.loc[row_idx, 'hold_hours'] = round(hold_hours, 1)
        self.df.loc[row_idx, 'status'] = 'closed'
        self._save()
        return pnl_usd, pnl_pct * 100, result

    def get_stats(self):
        """Return summary stats dict."""
        closed = self.df[self.df['status'] == 'closed'].copy()
        if len(closed) == 0:
            return {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0}

        wins = closed[closed['pnl_usd'] >= 0]
        losses = closed[closed['pnl_usd'] < 0]
        total_pnl = closed['pnl_usd'].sum()
        avg_win = wins['pnl_usd'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl_usd'].mean() if len(losses) > 0 else 0
        avg_hold = closed['hold_hours'].mean()

        return {
            'total_trades': len(closed),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(closed),
            'total_pnl': round(total_pnl, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'avg_hold_hours': round(avg_hold, 1),
            'open_trades': len(self.df[self.df['status'] == 'open']),
        }

    def get_open_trades(self):
        """Return list of currently open trades."""
        return self.df[self.df['status'] == 'open'].to_dict('records')

    def print_stats(self):
        """Print formatted stats."""
        s = self.get_stats()
        if s['total_trades'] == 0:
            print("   📊 No closed trades yet")
            return
        print(f"   📊 Trade Stats: {s['wins']}W/{s['losses']}L "
              f"({s['win_rate']:.0%} WR) | "
              f"PnL: ${s['total_pnl']:+.2f} | "
              f"Avg win: ${s['avg_win']:+.2f} / loss: ${s['avg_loss']:+.2f} | "
              f"Hold: {s['avg_hold_hours']:.0f}h | "
              f"Open: {s['open_trades']}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Production Trading System')
    parser.add_argument('--mode', choices=['signal', 'sim', 'paper', 'live'], default='signal')
    parser.add_argument('--capital', type=float, default=1000.0)
    parser.add_argument('--loop', action='store_true')
    parser.add_argument('--hours', type=int, default=800, help='Hours of history')
    parser.add_argument('--rebal', type=int, default=12,
                        help='Rebalance interval in hours (default: 12)')
    parser.add_argument('--npos', type=int, default=None,
                        help='Positions per side (overrides config)')
    parser.add_argument('--vol-target', type=float, default=None)
    parser.add_argument('--kelly', type=float, default=None)
    parser.add_argument('--config', type=str, default=None, help='Path to risk config JSON')
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(root, 'trading_logs')
    os.makedirs(log_dir, exist_ok=True)

    # Load risk config
    risk_cfg = DEFAULT_RISK.copy()
    cfg_candidates = [
        args.config,
        os.path.join(root, 'results_risk_study', 'optimal_config.json'),
    ]
    for cfg_path in cfg_candidates:
        if cfg_path and os.path.exists(cfg_path):
            with open(cfg_path) as f:
                loaded = json.load(f)
            risk_cfg.update(loaded)
            print(f"   📋 Loaded risk config from {os.path.basename(cfg_path)}")
            break

    # CLI overrides
    if args.vol_target is not None:
        risk_cfg['vol_target'] = args.vol_target
    if args.kelly is not None:
        risk_cfg['kelly_frac'] = args.kelly
    if args.npos is not None:
        risk_cfg['n_long'] = args.npos
        risk_cfg['n_short'] = args.npos

    rebal_hours = args.rebal

    # Load trading state
    state_path = os.path.join(log_dir, 'trading_state.json')
    state = load_state(state_path)
    if 'equity' not in state:
        state['equity'] = args.capital
        state['peak'] = args.capital
        state['initial_capital'] = args.capital
        state['recent_rets'] = []
        state['cycle_pnls'] = []
        state['equity_history'] = []
        state['total_pnl'] = 0.0
        state['n_cycles'] = 0
        state['sim_positions'] = []

    # Save runtime config into state for Telegram commands
    state['rebal_hours'] = rebal_hours
    state['leverage'] = risk_cfg.get('leverage', 3)
    state['mode'] = args.mode

    print("=" * 70)
    print(f"  PRODUCTION TRADING — {args.mode.upper()}")
    print(f"  Capital: ${args.capital:,.0f}")
    print(f"  Risk: kelly={risk_cfg['kelly_frac']:.0%}, "
          f"vol_target={risk_cfg['vol_target']*100:.1f}%, "
          f"DD_stop={risk_cfg['dd_stop']*100:.0f}%")
    print(f"  Rebalance: every {rebal_hours}h  |  "
          f"N={risk_cfg['n_long']}L+{risk_cfg['n_short']}S")
    print("=" * 70)

    # Init Telegram bot
    tg = None
    if create_bot:
        tg = create_bot()
        if tg and tg.enabled:
            tg.alert_startup(args.mode, args.capital, risk_cfg)
    else:
        print("   ⚠️  Telegram bot not available (telegram_bot.py not found)")

    # Graceful shutdown handler
    _shutdown_flag = False
    def _handle_signal(signum, frame):
        nonlocal _shutdown_flag
        _shutdown_flag = True
        print("\n   🛑 Shutdown signal received. Finishing current cycle...")
        if tg and tg.enabled:
            tg.alert_shutdown(reason=f"Signal {signum}")
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # Init exchange
    exchange = None
    if args.mode in ('paper', 'live'):
        exchange = init_exchange(args.mode)

    # Init trade tracker
    tracker = TradeTracker(os.path.join(log_dir, 'trades.csv'))
    tracker.print_stats()

    # Sync _prev_realized_pnl so bot restart doesn't spike equity
    if '_prev_realized_pnl' not in state:
        state['_prev_realized_pnl'] = tracker.get_stats().get('total_pnl', 0)

    def run_cycle():
        now = datetime.now(timezone.utc)
        print(f"\n{'─' * 70}")
        print(f"  🕐 {now.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"{'─' * 70}")

        # ── Settle previous positions (ALL modes) ──
        def _update_equity(pnl, settled, mode_label=''):
            """Update equity/DD/vol state — shared by sim, paper, and live."""
            state['equity'] = state.get('equity', args.capital) + pnl
            state['peak'] = max(state.get('peak', args.capital), state['equity'])
            state['total_pnl'] = state.get('total_pnl', 0) + pnl
            state['n_cycles'] = state.get('n_cycles', 0) + 1
            state['cycle_pnls'] = state.get('cycle_pnls', []) + [pnl]
            state['equity_history'] = state.get('equity_history', []) + [{
                'timestamp': now.isoformat(),
                'equity': round(state['equity'], 2),
                'pnl': pnl,
                'dd': round(state['equity'] / state['peak'] - 1, 4),
            }]
            # Track recent returns for vol scaling (critical — was missing in paper/live)
            if state['equity'] > 0:
                ret = pnl / (state['equity'] - pnl) if (state['equity'] - pnl) > 0 else 0
                state['recent_rets'] = (state.get('recent_rets', []) + [ret])[-200:]

            dd_pct = state['equity'] / state['peak'] - 1
            print(f"\n      💰 {mode_label}Cycle PnL: ${pnl:+.2f}  |  "
                  f"Equity: ${state['equity']:,.2f}  |  "
                  f"DD: {dd_pct:.1%}")

            if tg and tg.enabled:
                tg.alert_cycle_pnl(pnl, state['equity'], dd_pct, settled)
                if dd_pct < -0.15:
                    tg.alert_dd_warning(dd_pct, state['equity'])
                if dd_pct < risk_cfg.get('dd_stop', -0.20):
                    tg.alert_dd_stop(dd_pct, state['equity'])

        if args.mode == 'sim' and state.get('sim_positions'):
            print(f"\n📤 Settling previous positions...")
            prices = fetch_current_prices(SYMBOLS)
            pnl, settled = sim_settle_positions(state, prices)

            for s in settled:
                icon = '🟢' if s.get('pnl', 0) >= 0 else '🔴'
                print(f"      {icon} {s['side']:>5s} {s['symbol']:<14s} "
                      f"${s.get('entry_price',0):>10.4f} → ${s.get('exit_price',0):>10.4f} "
                      f"  ret={s.get('net_return_%',0):+.2f}%  pnl=${s.get('pnl',0):+.2f}")

            _update_equity(pnl, settled, mode_label='[SIM] ')
            state['sim_positions'] = []

        elif args.mode in ('paper', 'live') and tracker is not None:
            # Use TradeTracker realized PnL for equity tracking
            stats = tracker.get_stats()
            total_realized = stats.get('total_pnl', 0)
            prev_realized = state.get('_prev_realized_pnl', 0)
            cycle_pnl = total_realized - prev_realized
            state['_prev_realized_pnl'] = total_realized
            if state.get('n_cycles', 0) > 0 or cycle_pnl != 0:
                _update_equity(cycle_pnl, [], mode_label=f'[{args.mode.upper()}] ')

        # 1. Fetch data
        print(f"\n📊 Fetching data ({len(SYMBOLS)} symbols, {args.hours}h)...")
        df = fetch_ohlcv(SYMBOLS, args.hours)
        if df is None:
            print("   ❌ Data fetch failed")
            return
        print(f"   Shape: {df.shape}, Symbols: {df['symbol'].nunique()}")

        # 2. Build features
        print(f"\n🔧 Building features...")
        df = build_features(df)
        feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS
                     and not c.startswith('target_')
                     and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
        print(f"   Features: {len(feat_cols)}")

        # 3. Cross-sectional rank
        df = cross_sectional_rank(df, feat_cols)

        # 4. Generate signal
        print(f"\n📡 Generating signal...")
        signals = generate_signal(df, feat_cols, root)
        if signals is None or len(signals) == 0:
            print("   ❌ No signals")
            return

        # 5. Portfolio
        print(f"\n💼 Portfolio construction...")
        positions = construct_portfolio(signals, state.get('equity', args.capital),
                                        risk_cfg, state)

        # Store for background dashboard refresh
        state['_last_signals'] = signals
        state['_last_positions'] = positions

        # Store entry scores/confidence for dashboard (persists across cycles)
        entry_scores = state.get('entry_scores', {})  # {symbol: {score, confidence, time}}
        if positions:
            now_str = datetime.now(timezone.utc).isoformat()
            for p in positions:
                sym = p['symbol']
                if sym not in entry_scores or entry_scores[sym].get('side') != p['side']:
                    # New position or side flipped — record entry score
                    entry_scores[sym] = {
                        'score': p.get('score', 0),
                        'confidence': p.get('confidence', 0.5),
                        'side': p['side'],
                        'time': now_str,
                    }
            state['entry_scores'] = entry_scores

        if not positions:
            print("   (no positions this cycle)")
        else:
            print(f"\n   {'Symbol':<15} {'Side':<6} {'USD':>8} {'Score':>8} {'Conf':>6}")
            print(f"   {'─' * 48}")
            for pos in positions:
                conf = pos.get('confidence', 0)
                print(f"   {pos['symbol']:<15} {pos['side']:<6} ${pos['usd']:>7.0f} "
                      f"{pos['score']:>+8.3f} {conf:>5.0%}")

        # ── Telegram: positions alert ──
        if tg and tg.enabled:
            tg.alert_positions(positions, state.get('equity', args.capital),
                               leverage=risk_cfg.get('leverage', 3))

        # 6. Execute
        if args.mode == 'signal':
            results = execute(None, positions, dry_run=True)
        elif args.mode == 'sim':
            # Record positions with entry prices
            if positions:
                prices = fetch_current_prices(SYMBOLS)
                state['sim_positions'] = sim_open_positions(positions, prices)
                print(f"\n   📌 Opened {len(state['sim_positions'])} sim positions "
                      f"(will settle in {HORIZON}h)")
            sim_print_summary(state, log_dir)
            results = []
        else:
            print(f"\n📋 Diff-based rebalancing...")
            results = diff_rebalance(exchange, positions,
                                     leverage=risk_cfg.get('leverage', 3),
                                     tracker=tracker)
            # Clean up entry_scores for closed positions
            if positions:
                active_syms = {p['symbol'] for p in positions}
                es = state.get('entry_scores', {})
                stale = [k for k in es if k not in active_syms]
                for k in stale:
                    del es[k]
                state['entry_scores'] = es
            # Print trade stats after each cycle
            tracker.print_stats()
            # ── Telegram: fills + stats ──
            if tg and tg.enabled:
                tg.alert_fills(results)
                stats = tracker.get_stats()
                if stats['total_trades'] > 0:
                    tg.send(
                        f"📊 <b>Trade Stats</b>\n"
                        f"Trades: {stats['wins']}W / {stats['losses']}L "
                        f"({stats['win_rate']:.0%} WR)\n"
                        f"Total PnL: ${stats['total_pnl']:+.2f}\n"
                        f"Avg win: ${stats['avg_win']:+.2f} | "
                        f"Avg loss: ${stats['avg_loss']:+.2f}",
                    )

        # 7. Log
        log = {
            'timestamp': now.isoformat(),
            'mode': args.mode,
            'capital': state.get('equity', args.capital),
            'risk_config': risk_cfg,
            'positions': positions,
            'state': {k: v for k, v in state.items()
                      if k not in ('recent_rets', 'equity_history', 'sim_positions',
                                   '_last_signals', '_last_positions')},
            'signals_top5': signals.head(5).to_dict('records') if signals is not None else [],
            'signals_bot5': signals.tail(5).to_dict('records') if signals is not None else [],
        }

        log_path = os.path.join(log_dir, f"trade_{now.strftime('%Y%m%d_%H%M')}.json")
        with open(log_path, 'w') as f:
            json.dump(log, f, indent=2, default=str)

        save_state(state, state_path)
        print(f"\n   📝 Log: {os.path.basename(log_path)}")

        # 8. Dashboard JSON export
        _export_dashboard(
            state=state, positions=positions, signals=signals,
            tracker=tracker, risk_cfg=risk_cfg, args=args,
            rebal_hours=rebal_hours, exchange=exchange,
        )

    # Run
    if args.loop:
        print(f"\n🔄 Continuous mode (every {rebal_hours}h)...")
        # Start background dashboard refresh (every 5 min)
        start_dashboard_bg(state, tracker, risk_cfg, args, rebal_hours, exchange)
        cycle_count = 0
        while not _shutdown_flag:
            try:
                run_cycle()
                cycle_count += 1

                # Daily summary (every ~24h worth of cycles)
                cycles_per_day = max(1, 24 // rebal_hours)
                if tg and tg.enabled and cycle_count % cycles_per_day == 0:
                    tg.alert_daily_summary(state)

            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
                if tg and tg.enabled:
                    tg.alert_error(str(e), context="run_cycle")

            if _shutdown_flag:
                break

            now = datetime.now(timezone.utc)
            # Align to next rebal_hours boundary + 5min
            # Calculate the next slot properly (handles day rollover)
            hour_slot = (now.hour // rebal_hours + 1) * rebal_hours
            next_time = now.replace(hour=0, minute=5, second=0, microsecond=0) + timedelta(hours=hour_slot)
            if next_time <= now:
                next_time += timedelta(hours=rebal_hours)

            sleep = (next_time - now).total_seconds()
            print(f"\n   ⏰ Next: {next_time.strftime('%Y-%m-%d %H:%M UTC')} ({sleep/3600:.1f}h)")

            # Sleep in small chunks to respond to shutdown quickly
            sleep_end = time.time() + max(sleep, 60)
            while time.time() < sleep_end and not _shutdown_flag:
                time.sleep(min(10, sleep_end - time.time()))

        # Graceful shutdown
        print("\n🛑 Shutting down...")
        stop_dashboard_bg()
        if tg and tg.enabled:
            tg.alert_shutdown(reason="graceful")
            tg.stop_polling()
        save_state(state, state_path)

    else:
        try:
            run_cycle()
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            if tg and tg.enabled:
                tg.alert_error(str(e), context="run_cycle")

    # Cleanup
    if tg and tg.enabled:
        tg.stop_polling()

    print(f"\n✅ Done!")


if __name__ == '__main__':
    main()
