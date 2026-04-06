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
  paper    — Paper trade on OKX demo account
  live     — Live trading with real money

Setup:
  1. Train models: python run_pipeline_v5.py (saves to results_v5/)
  2. Run risk study: python run_risk_study.py (saves optimal_config.json)
  3. Set OKX API keys:
     export OKX_API_KEY=xxx OKX_SECRET=xxx OKX_PASSPHRASE=xxx
  4. For paper: export OKX_DEMO=1

Usage:
  # Generate signals only (no API needed):
  python run_trading.py --mode signal

  # Paper trading (single cycle):
  python run_trading.py --mode paper --capital 1000

  # Live continuous (runs every 4h):
  python run_trading.py --mode live --capital 500 --loop

  # Override risk params:
  python run_trading.py --mode paper --vol-target 0.01 --kelly 0.3
"""

import os
import sys
import time
import json
import argparse
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

from telegram_bot import create_bot

# ============================================================
# CONFIG
# ============================================================
DEFAULT_REBAL_HOURS = 12
TOP_K_DEFAULT = 10  # will be overridden by risk config
SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
    'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'DOT/USDT', 'LINK/USDT',
    'UNI/USDT', 'ATOM/USDT', 'LTC/USDT', 'FIL/USDT',
    'APT/USDT', 'ARB/USDT', 'OP/USDT', 'NEAR/USDT', 'AAVE/USDT',
    'INJ/USDT', 'FTM/USDT', 'ALGO/USDT', 'SAND/USDT', 'MANA/USDT',
    'AXS/USDT', 'THETA/USDT', 'RUNE/USDT', 'EGLD/USDT', 'XTZ/USDT',
    'FLOW/USDT', 'CHZ/USDT', 'CRV/USDT', 'LDO/USDT', 'SNX/USDT',
    'COMP/USDT', 'YFI/USDT', 'SUSHI/USDT', 'ENJ/USDT', 'BAT/USDT',
    'ZIL/USDT', 'ONE/USDT', 'IOTA/USDT', 'ICX/USDT', 'ENS/USDT',
    'IMX/USDT', 'GALA/USDT', 'MKR/USDT', 'GRT/USDT', 'ETC/USDT',
]

# OKX instrument mapping
SYMBOLS_TO_OKX = {
    sym: sym.replace('/', '-').replace('USDT', 'USDT-SWAP')
    for sym in SYMBOLS
}

# Symbols that don't exist on OKX Demo swaps or are compliance-restricted
_OKX_BLOCKED = {
    'MATIC/USDT', 'UNI/USDT', 'APT/USDT', 'FTM/USDT', 'MANA/USDT',
    'RUNE/USDT', 'EGLD/USDT', 'FLOW/USDT', 'SNX/USDT', 'ENJ/USDT',
    'BAT/USDT', 'ONE/USDT', 'ICX/USDT', 'ENS/USDT', 'GALA/USDT',
    'GRT/USDT',
    # Compliance-restricted (51155)
    'CHZ/USDT', 'MKR/USDT',
}

# R25 simulation symbols (SYM_35) — CLS mode filters to these
CLS_SYMBOLS = {
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
    'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'DOT/USDT', 'LINK/USDT',
    'MATIC/USDT', 'UNI/USDT', 'ATOM/USDT', 'LTC/USDT', 'NEAR/USDT',
    'FIL/USDT', 'APT/USDT', 'ARB/USDT', 'OP/USDT', 'AAVE/USDT',
    'INJ/USDT', 'FTM/USDT', 'ALGO/USDT', 'SAND/USDT', 'MANA/USDT',
    'AXS/USDT', 'THETA/USDT', 'RUNE/USDT', 'EGLD/USDT', 'XTZ/USDT',
    'FLOW/USDT', 'CHZ/USDT', 'CRV/USDT', 'LDO/USDT', 'SNX/USDT',
}

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
UNRANKED_COLS = {
    'btc_regime_24', 'btc_regime_72', 'btc_regime_168',
    'regime_btc_above_ma336', 'regime_btc_above_ma720',
    'regime_btc_ma720_slope', 'regime_btc_not_crashed',
    'regime_btc_dd_720', 'regime_low_vol',
    'regime_breadth_bullish', 'breadth_pct_positive',
    'regime_composite',
    'fng_value', 'fng_extreme_fear', 'fng_extreme_greed',
    'fng_ma7', 'fng_ma30', 'fng_momentum',
    'market_avg_funding', 'market_funding_skew',
    # v6: binary features that should NOT be ranked
    'is_asian_session',
    # Calendar features (same for all coins at same timestamp)
    'cal_hour_sin', 'cal_hour_cos', 'cal_dow_sin', 'cal_dow_cos',
    'cal_is_us_session', 'cal_is_weekend',
    'cal_days_to_monthly_expiry', 'cal_month_sin', 'cal_month_cos',
    # News sentiment (market-level, should NOT be ranked cross-sectionally)
    'market_news_count_24h', 'market_news_sentiment_24h',
    'news_sentiment_24h', 'news_sentiment_7d', 'news_sentiment_momentum',
    # Political/macro news (market-level, same for all coins)
    'political_news_count_24h', 'political_sentiment_24h',
    'political_sentiment_7d', 'political_sentiment_shock',
    'political_news_volume_zscore',
    # Binary flags (should NOT be ranked)
    'has_news_data', 'news_coverage_ok', 'news_event',
    # Macro / cross-market features (market-level, same for all coins)
    'vix_close', 'spx_close', 'dxy_close', 'gold_close',
    'yield_10y_close', 'hy_spread', 'breakeven_10y',
    'yield_curve_10y2y', 'fed_funds_rate',
    'vix_close_z20d', 'hy_spread_z20d', 'breakeven_10y_z20d',
    'yield_curve_10y2y_z20d', 'risk_aversion', 'real_rate',
    # Macro changes (1d/5d/20d) — market-level, not ranked
    'vix_close_chg_1d', 'vix_close_chg_5d', 'vix_close_chg_20d',
    'spx_close_chg_1d', 'spx_close_chg_5d', 'spx_close_chg_20d',
    'dxy_close_chg_1d', 'dxy_close_chg_5d', 'dxy_close_chg_20d',
    'gold_close_chg_1d', 'gold_close_chg_5d', 'gold_close_chg_20d',
    'hy_spread_chg_1d', 'hy_spread_chg_5d', 'hy_spread_chg_20d',
    'breakeven_10y_chg_1d', 'breakeven_10y_chg_5d', 'breakeven_10y_chg_20d',
    'yield_curve_10y2y_chg_1d', 'yield_curve_10y2y_chg_5d', 'yield_curve_10y2y_chg_20d',
    # Macro cross-interactions
    'risk_on_off_ratio', 'real_rate_chg_5d',
    # CLS champion 31f — market-level features (DO NOT CS-rank)
    'ret_dispersion_12h',  # CS std of ret_12h (same for all symbols at each timestamp)
}

# Default risk config (overridden by optimal_config.json)
# R7 winner: 6L/3S asymmetric (long-heavy)
DEFAULT_RISK = {
    'n_long': 6,
    'n_short': 3,
    'vol_target': 0.008,
    'vol_lookback': 48,
    'kelly_frac': 0.8,
    'dd_stop': -0.15,
    'dd_resume': -0.06,
    'confidence_threshold': 0.0,
}

# ── Partial rebalance settings ──
REBALANCE_THRESHOLD = 0.12  # 12% — don't resize if |target-live|/target < this
LIMIT_ORDER_WAIT = 20       # seconds to wait for limit fill before market fallback
LIMIT_PRICE_AGGRESSION = 0.0003  # cross spread by 0.03% for higher fill rate

# ── R49.1: Maker-First Execution Tiers ──────────────────────
# TIER1: top-5 liquid coins → genuinely try post-only maker fills
# TIER2: mid-cap → use existing aggressive limit (near-taker)
# TIER3: small-cap → plain market orders
_TIER1_SYMS = {'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT'}
_TIER1_OKX = {SYMBOLS_TO_OKX.get(s) for s in _TIER1_SYMS if SYMBOLS_TO_OKX.get(s)}
_TIER3_SYMS = {
    'SAND/USDT', 'LDO/USDT', 'INJ/USDT', 'APT/USDT', 'ARB/USDT',
    'GALA/USDT', 'FTM/USDT', 'MATIC/USDT',
}

# Maker-first parameters (R49.1)
MAKER_TTL_SECONDS = 90        # wait per attempt for post-only fill
MAKER_MAX_RETRIES = 3         # attempts before market fallback
MAKER_MID_OFFSET = 0.00005   # 0.5bp inside mid for initial placement
MAKER_AGGR_STEP = 0.00010    # widen 1bp per retry toward spread
EXEC_LOG_PATH = 'trading_logs/execution_log.csv'
EXEC_LOG_HEADER = (
    'timestamp,symbol,okx_sym,tier,side,order_type,attempt,'
    'bid,ask,mid,limit_px,fill_price,spread_bps,slippage_bps,'
    'effective_bps,filled_qty,cost_usd,fill_time_s,was_maker\n'
)


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

    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'},
    })
    exchange.session.verify = False

    all_dfs = []
    limit = min(hours + 10, 1000)
    failed = []

    for i, sym in enumerate(symbols):
        try:
            ohlcv = exchange.fetch_ohlcv(sym, '1h', limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
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
    df['market_dispersion'] = df.groupby('timestamp')['ret_1h'].transform('std')
    if 'ret_24h' in df.columns:
        df['ret_vs_btc_24h'] = df['ret_24h'] - df['btc_ret_24h']

    # Breadth
    breadth = df.groupby('timestamp')['ret_24h'].agg(
        breadth_pct_positive=lambda x: (x > 0).mean()
    ).reset_index()
    df = df.merge(breadth, on='timestamp', how='left')

    # Regime
    btc_ts = df[df['symbol'] == 'BTC/USDT'][['timestamp', 'btc_close']].drop_duplicates('timestamp').sort_values('timestamp')
    for w in [336, 720]:
        btc_ts[f'btc_ma{w}'] = btc_ts['btc_close'].rolling(w, min_periods=min(w, 100)).mean()
    btc_ts['regime_btc_above_ma720'] = (btc_ts['btc_close'] > btc_ts['btc_ma720']).astype(float)
    btc_ts['btc_rolling_high_720'] = btc_ts['btc_close'].rolling(720, min_periods=100).max()
    btc_ts['regime_btc_dd_720'] = btc_ts['btc_close'] / btc_ts['btc_rolling_high_720'] - 1
    btc_ts['regime_btc_not_crashed'] = (btc_ts['regime_btc_dd_720'] > -0.15).astype(float)
    df = df.merge(btc_ts[['timestamp', 'regime_btc_above_ma720', 'regime_btc_dd_720',
                           'regime_btc_not_crashed']], on='timestamp', how='left')

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
        df['fng_momentum'] = df['fng_value'] - df['fng_ma7']
        df.drop(columns=['date'], inplace=True, errors='ignore')

    # Synthetic positioning
    for fast, slow in [(4, 24), (12, 48), (24, 168)]:
        fr = f'ret_{fast}h'
        sr = f'ret_{slow}h'
        if fr in df.columns and sr in df.columns:
            df[f'reversal_{fast}v{slow}'] = -df[fr] * df[sr].abs()

    for w in [12, 24]:
        df[f'vol_surge_{w}h'] = df.groupby('symbol')['volume'].transform(
            lambda x: x / x.rolling(w).mean() - 1)

    # BTC beta
    btc_rets = df[df['symbol'] == 'BTC/USDT'][['timestamp', 'ret_1h']].rename(
        columns={'ret_1h': 'btc_r'}).drop_duplicates('timestamp')
    df = df.merge(btc_rets, on='timestamp', how='left')
    for w in [48, 168]:
        df[f'btc_beta_{w}h'] = df.groupby('symbol').apply(
            lambda g: g['ret_1h'].rolling(w).corr(g['btc_r']) *
                      (g['ret_1h'].rolling(w).std() / (g['btc_r'].rolling(w).std() + 1e-10))
        ).reset_index(level=0, drop=True)

    # Clean up
    df.drop(columns=['btc_close', 'eth_close', 'btc_r'], inplace=True, errors='ignore')

    # ── Macro / cross-market features (from FRED) ──
    macro_path = os.path.join(root, 'data', 'sentiment', 'macro_daily.parquet')
    if os.path.exists(macro_path):
        macro = pd.read_parquet(macro_path)
        macro['date'] = pd.to_datetime(macro['date']).dt.date
        df['date'] = df['timestamp'].dt.date
        df = df.merge(macro, on='date', how='left')
        df.drop(columns=['date'], inplace=True, errors='ignore')

        raw_cols = ['vix_close', 'spx_close', 'dxy_close', 'gold_close',
                    'yield_10y_close', 'hy_spread', 'breakeven_10y',
                    'yield_curve_10y2y', 'fed_funds_rate']
        for col in raw_cols:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()

        # Changes
        change_cols = ['vix_close', 'spx_close', 'dxy_close', 'gold_close',
                       'hy_spread', 'breakeven_10y', 'yield_curve_10y2y']
        for col in change_cols:
            if col not in df.columns:
                continue
            for hours, suffix in [(24, '1d'), (120, '5d'), (480, '20d')]:
                df[f'{col}_chg_{suffix}'] = df.groupby('symbol')[col].transform(
                    lambda x: x.pct_change(hours))

        # Z-scores
        for col in ['vix_close', 'hy_spread', 'breakeven_10y', 'yield_curve_10y2y']:
            if col not in df.columns:
                continue
            mean = df.groupby('symbol')[col].transform(lambda x: x.rolling(480, min_periods=120).mean())
            std = df.groupby('symbol')[col].transform(lambda x: x.rolling(480, min_periods=120).std())
            df[f'{col}_z20d'] = (df[col] - mean) / (std + 1e-10)

        # Cross-interactions
        if 'vix_close_z20d' in df.columns and 'hy_spread_z20d' in df.columns:
            df['risk_aversion'] = df['vix_close_z20d'] + df['hy_spread_z20d']
        if 'spx_close' in df.columns and 'gold_close' in df.columns:
            df['risk_on_off_ratio'] = df.groupby('symbol').apply(
                lambda g: g['spx_close'].pct_change(24).rolling(120).corr(
                    g['gold_close'].pct_change(24))
            ).reset_index(level=0, drop=True)
        if 'yield_10y_close' in df.columns and 'breakeven_10y' in df.columns:
            df['real_rate'] = df['yield_10y_close'] - df['breakeven_10y']
            df['real_rate_chg_5d'] = df.groupby('symbol')['real_rate'].transform(
                lambda x: x.diff(120))

    # Replace inf/nan
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    return df


# 14 features verified by cross-sectional IC analysis (_cs_model_v3.py)
RIDGE_FEATURES = [
    'ret_12h', 'ret_24h', 'ret_48h',
    'residual_12h', 'residual_24h',
    'mom_z_12h', 'mom_z_24h',
    'dist_from_high_24h',
    'oi_chg_12h', 'oi_chg_24h', 'oi_zscore',
    'taker_cvd_12h', 'taker_cvd_24h',
    'ls_divergence',
]


def cross_sectional_rank(df, feat_cols):
    """Rank normalize features cross-sectionally per timestamp."""
    rank_cols = [c for c in feat_cols if c not in UNRANKED_COLS]
    for col in rank_cols:
        df[col] = df.groupby('timestamp')[col].rank(pct=True) - 0.5
    return df


def add_ridge_features(df):
    """Compute features needed by Ridge model that the standard pipeline doesn't produce."""
    print("   🔧 Adding Ridge-specific features...")

    # 1. BTC beta (168h rolling) + residuals
    if 'btc_ret_1h' in df.columns and 'ret_1h' in df.columns:
        btc_beta = pd.Series(np.nan, index=df.index)
        for sym, g in df.groupby('symbol'):
            cov = g['ret_1h'].rolling(168, min_periods=84).cov(g['btc_ret_1h'])
            var = g['btc_ret_1h'].rolling(168, min_periods=84).var()
            btc_beta.loc[g.index] = cov / (var + 1e-10)
        for h in [12, 24]:
            brc = f'btc_ret_{h}h'
            if brc in df.columns and f'ret_{h}h' in df.columns:
                df[f'residual_{h}h'] = df[f'ret_{h}h'] - btc_beta * df[brc]

    # 2. Momentum z-score: ret / realized_vol
    if 'ret_1h' in df.columns:
        df['_ret_1h_sq'] = df['ret_1h'] ** 2
        for h in [12, 24]:
            rvol = df.groupby('symbol')['_ret_1h_sq'].transform(
                lambda x: x.rolling(h, min_periods=h // 2).mean().pow(0.5))
            df[f'mom_z_{h}h'] = df[f'ret_{h}h'] / (rvol + 1e-10)
        df.drop(columns=['_ret_1h_sq'], inplace=True)

    # 3. Distance from 24h high (mean-reversion signal)
    high_24 = df.groupby('symbol')['high'].transform(lambda x: x.rolling(24).max())
    low_24 = df.groupby('symbol')['low'].transform(lambda x: x.rolling(24).min())
    df['dist_from_high_24h'] = (high_24 - df['close']) / (high_24 - low_24 + 1e-10)

    # 4. Rename pipeline feature names to match Ridge model
    for old, new in [('oi_change_12h', 'oi_chg_12h'),
                     ('oi_change_24h', 'oi_chg_24h'),
                     ('oi_zscore_7d', 'oi_zscore')]:
        if old in df.columns:
            df[new] = df[old]

    n_avail = sum(1 for f in RIDGE_FEATURES if f in df.columns)
    print(f"   ✅ Ridge features: {n_avail}/{len(RIDGE_FEATURES)} available")
    return df


def add_cls_features(df, root):
    """Build features needed by CLS model that the standard pipeline doesn't create.
    Must be called BEFORE cross_sectional_rank() so these get ranked like in simulation."""
    print("   🔧 Adding CLS-specific features...")
    n_added = 0

    # 1. rvol_12h, rvol_24h: realized vol = sqrt(rolling mean of ret_1h²)
    if 'ret_1h' in df.columns:
        for h in [12, 24]:
            col = f'rvol_{h}h'
            if col not in df.columns:
                df[col] = df.groupby('symbol').apply(
                    lambda g: pd.Series(
                        (g['ret_1h'] ** 2).rolling(h, min_periods=h // 2).mean().pow(0.5).values,
                        index=g.index)
                ).droplevel(0)
                n_added += 1

    # 2. iv_rv_spread: btc_dvol/100 - annualized rvol_24h
    if 'iv_rv_spread' not in df.columns:
        sent_dir = os.path.join(root, 'data', 'sentiment')
        dvol_path = os.path.join(sent_dir, 'deribit_dvol.parquet')
        if os.path.exists(dvol_path) and 'rvol_24h' in df.columns:
            try:
                dv = pd.read_parquet(dvol_path)
                btc_dv = dv[dv['currency'] == 'BTC'][['timestamp', 'dvol_close']].rename(
                    columns={'dvol_close': '_btc_dvol_tmp'})
                btc_dv = btc_dv.set_index('timestamp').resample('1h').ffill().reset_index()
                df = df.merge(btc_dv, on='timestamp', how='left')
                df['_btc_dvol_tmp'] = df['_btc_dvol_tmp'].ffill()
                ann_rvol = df.groupby('symbol')['rvol_24h'].transform(
                    lambda x: x * np.sqrt(24 * 365))
                df['iv_rv_spread'] = df['_btc_dvol_tmp'] / 100 - ann_rvol
                df.drop(columns=['_btc_dvol_tmp'], inplace=True, errors='ignore')
                n_added += 1
            except Exception as e:
                print(f"   ⚠️  iv_rv_spread failed: {e}")
                df['iv_rv_spread'] = 0.0
        else:
            df['iv_rv_spread'] = 0.0

    # 3. pct_coins_up_12h, pct_coins_up_1h: cross-sectional breadth
    for ret_col, feat_col in [('ret_12h', 'pct_coins_up_12h'), ('ret_1h', 'pct_coins_up_1h')]:
        if feat_col not in df.columns and ret_col in df.columns:
            df[feat_col] = df.groupby('timestamp')[ret_col].transform(lambda x: (x > 0).mean())
            n_added += 1

    # 4. Calendar features: alias cal_* → plain names (so they get ranked → ~0.0 like in sim)
    for cal, plain in [('cal_hour_sin', 'hour_sin'), ('cal_hour_cos', 'hour_cos'),
                       ('cal_dow_sin', 'dow_sin'), ('cal_dow_cos', 'dow_cos')]:
        if plain not in df.columns and cal in df.columns:
            df[plain] = df[cal]
            n_added += 1

    # 5. rel_volume_cs: log(volume) - cs_mean(log(volume)) — cross-sectional relative volume
    if 'rel_volume_cs' not in df.columns and 'volume' in df.columns:
        df['_log_vol'] = np.log(df['volume'].clip(lower=1))
        df['rel_volume_cs'] = df['_log_vol'] - df.groupby('timestamp')['_log_vol'].transform('mean')
        df.drop(columns=['_log_vol'], inplace=True)
        n_added += 1

    # 6. ret_dispersion_12h: cross-sectional std of ret_12h (market-level, NOT CS-ranked)
    if 'ret_dispersion_12h' not in df.columns and 'ret_12h' in df.columns:
        df['ret_dispersion_12h'] = df.groupby('timestamp')['ret_12h'].transform('std')
        n_added += 1

    # 7. cs_rank_ma_5: rolling mean of CS-ranked ret_12h (cross-sectional momentum persistence)
    if 'cs_rank_ma_5' not in df.columns and 'ret_12h' in df.columns:
        _cs_rank = df.groupby('timestamp')['ret_12h'].rank(pct=True) - 0.5
        df['cs_rank_ma_5'] = _cs_rank.groupby(df['symbol']).transform(
            lambda x: x.rolling(5, min_periods=3).mean()
        )
        n_added += 1

    # 8. cg_taker_imb: CoinGlass taker buy/sell imbalance (daily, shift-1 for lookahead safety)
    if 'cg_taker_imb' not in df.columns:
        cg_taker_path = os.path.join(root, 'data', 'raw', 'coinglass', 'taker.parquet')
        if os.path.exists(cg_taker_path):
            try:
                taker = pd.read_parquet(cg_taker_path)
                taker['timestamp'] = pd.to_datetime(taker['timestamp'], utc=True)
                taker['cg_date'] = taker['timestamp'].dt.normalize()
                taker = taker.drop_duplicates(subset=['symbol', 'cg_date'], keep='last')
                eps = 1e-10
                taker_sum = taker['taker_buy_usd'] + taker['taker_sell_usd']
                taker['cg_taker_imb'] = (taker['taker_buy_usd'] - taker['taker_sell_usd']) / (taker_sum + eps)
                # Shift-1: use cg_date = floor(timestamp, 'D') - 1 day
                df['_cg_date'] = df['timestamp'].dt.normalize() - pd.Timedelta(days=1)
                df = df.merge(
                    taker[['symbol', 'cg_date', 'cg_taker_imb']].rename(columns={'cg_date': '_cg_date'}),
                    on=['symbol', '_cg_date'],
                    how='left',
                )
                df.drop(columns=['_cg_date'], inplace=True)
                n_added += 1
                n_valid = df['cg_taker_imb'].notna().sum()
                print(f"   📊 cg_taker_imb: {n_valid:,}/{len(df):,} valid ({100*n_valid/len(df):.0f}%)")
            except Exception as e:
                print(f"   ⚠️  cg_taker_imb failed: {e}")
                df['cg_taker_imb'] = 0.0
        else:
            print(f"   ⚠️  CG taker data not found: {cg_taker_path}")
            df['cg_taker_imb'] = 0.0

    print(f"   ✅ CLS features: {n_added} added")
    return df


def generate_signal_lgb_cs(df, root):
    """
    Generate trading signal using LightGBM ensemble (R9B production model).

    Key differences from Ridge:
    - Models loaded from results_lgb_prod/lgb_model_seed_*.txt
    - Features: same 14 CS-IC features, already CS-ranked in-place by caller
    - NO EMA smoothing (LGB signal already high quality, EMA hurts)
    - Ensemble of 5 seeds → averaged predictions, reduces seed variance
    - IC: 0.053–0.072 vs Ridge 0.013–0.020 (3-4x better signal quality)
    """
    import lightgbm as lgb_lib

    lgb_dir = os.path.join(root, 'results_lgb_prod')
    if not os.path.isdir(lgb_dir):
        print(f"   ❌ LGB models not found: {lgb_dir}")
        print(f"      Run: python train_lgb_prod.py")
        return None

    model_files = sorted(Path(lgb_dir).glob('lgb_model_seed_*.txt'))
    if not model_files:
        print(f"   ❌ No lgb_model_seed_*.txt files in {lgb_dir}")
        return None

    models = [lgb_lib.Booster(model_file=str(f)) for f in model_files]
    feat_names = models[0].feature_name()

    latest = df.groupby('symbol').last().reset_index()

    missing = [f for f in feat_names if f not in latest.columns]
    if missing:
        print(f"   ⚠️  LGB missing {len(missing)} features: {missing[:5]}{'...' if len(missing)>5 else ''}")
        for f in missing:
            latest[f] = 0.0

    X = latest[feat_names].fillna(0).values
    preds_all = np.mean([m.predict(X) for m in models], axis=0)

    # Z-normalize cross-sectionally
    pred_std = np.std(preds_all)
    if pred_std > 1e-10:
        scores = (preds_all - np.mean(preds_all)) / pred_std
    else:
        scores = preds_all

    # BTC regime data (same computation as Ridge, for construct_portfolio R7 logic)
    btc_data = df[df['symbol'] == 'BTC/USDT'].sort_values('timestamp')
    regime_data = {}
    if len(btc_data) >= 168:
        btc_close = btc_data['close'].values
        btc_ret_7d = btc_close[-1] / btc_close[-168] - 1 if btc_close[-168] > 0 else 0
        btc_hourly_rets = np.diff(btc_close[-169:]) / (btc_close[-169:-1] + 1e-10)
        btc_vol_7d = float(np.std(btc_hourly_rets))
        trend_strength = abs(btc_ret_7d) / (btc_vol_7d * np.sqrt(168) + 1e-10)
        all_hourly_rets = np.diff(btc_close) / (btc_close[:-1] + 1e-10)
        btc_vol_48h = float(np.std(all_hourly_rets[-48:])) if len(all_hourly_rets) >= 48 else btc_vol_7d
        btc_vol_long = float(np.std(all_hourly_rets[-720:])) if len(all_hourly_rets) >= 720 else btc_vol_48h
        vol_regime = btc_vol_48h / (btc_vol_long + 1e-10)
        trend_direction = btc_ret_7d / (btc_vol_7d * np.sqrt(168) + 1e-10)
        regime_data = {
            'trend_strength': trend_strength,
            'vol_regime': vol_regime,
            'trend_direction': trend_direction,
        }
        print(f"   📊 Regime: BTC 7d={btc_ret_7d*100:+.1f}%, "
              f"trend={trend_strength:.2f}, vol_regime={vol_regime:.2f}, "
              f"trend_dir={trend_direction:+.2f}")

    latest['score'] = scores
    latest['confidence'] = np.full(len(latest), 0.5)
    latest['deriv_scale'] = np.full(len(latest), 1.0)

    print(f"   ✅ LGB ensemble: {len(models)} models × {len(feat_names)} feats, "
          f"score range [{scores.min():.2f}, {scores.max():.2f}]")

    result = latest[['symbol', 'score', 'confidence', 'deriv_scale']].sort_values(
        'score', ascending=False).reset_index(drop=True)
    result.attrs['model_info'] = {'n_models': len(models), 'groups': {'lgb_cs': len(models)}}
    result.attrs['regime_data'] = regime_data
    return result


def generate_signal_cls(df, root):
    """
    Generate signal using LGB+XGB binary classification ensemble (R25).
    Features must already be built (add_cls_features) and ranked (cross_sectional_rank).
    """
    import lightgbm as lgb_lib
    import xgboost as xgb_lib

    cls_dir = os.path.join(root, 'results_cls_prod')
    if not os.path.isdir(cls_dir):
        print(f"   ❌ CLS models not found: {cls_dir}")
        print(f"      Run: python train_cls_prod.py")
        return None

    lgb_files = sorted(Path(cls_dir).glob('lgb_cls_seed_*.txt'))
    xgb_files = sorted(Path(cls_dir).glob('xgb_cls_seed_*.json'))
    if not lgb_files or not xgb_files:
        print(f"   ❌ Need both lgb_cls_seed_*.txt and xgb_cls_seed_*.json")
        return None

    lgb_models = [lgb_lib.Booster(model_file=str(f)) for f in lgb_files]
    xgb_models = []
    for f in xgb_files:
        m = xgb_lib.Booster()
        m.load_model(str(f))
        xgb_models.append(m)

    feat_names = lgb_models[0].feature_name()
    print(f"   📦 CLS ensemble: {len(lgb_models)} LGB + {len(xgb_models)} XGB, {len(feat_names)} feats")

    latest = df.groupby('symbol').last().reset_index()

    missing = [f for f in feat_names if f not in latest.columns]
    if missing:
        print(f"   ⚠️  CLS missing {len(missing)} features: {missing}")
        for f in missing:
            latest[f] = 0.0
    else:
        print(f"   ✅ All {len(feat_names)} features present")

    X = latest[feat_names].fillna(0).values

    # LGB predictions (probabilities)
    lgb_probs = np.mean([m.predict(X) for m in lgb_models], axis=0)

    # XGB predictions (probabilities)
    dmat = xgb_lib.DMatrix(X, feature_names=feat_names)
    xgb_probs = np.mean([m.predict(dmat) for m in xgb_models], axis=0)

    # Rank-normalize each, then average (same as R25 research)
    from scipy.stats import rankdata
    lgb_ranked = rankdata(lgb_probs) / len(lgb_probs) - 0.5
    xgb_ranked = rankdata(xgb_probs) / len(xgb_probs) - 0.5
    ensemble = 0.5 * lgb_ranked + 0.5 * xgb_ranked

    # Z-normalize for portfolio construction
    std = np.std(ensemble)
    if std > 1e-10:
        scores = (ensemble - np.mean(ensemble)) / std
    else:
        scores = ensemble

    latest['score'] = scores
    latest['confidence'] = np.full(len(latest), 0.5)
    latest['deriv_scale'] = np.full(len(latest), 1.0)

    print(f"   ✅ CLS ensemble: score range [{scores.min():.2f}, {scores.max():.2f}]")
    print(f"   📊 LGB prob [{lgb_probs.min():.3f}, {lgb_probs.max():.3f}]  "
          f"XGB prob [{xgb_probs.min():.3f}, {xgb_probs.max():.3f}]")

    # BTC regime data for trend_cutoff / dyn_threshold (R25 CFG uses trend_cutoff=0.9)
    btc_data = df[df['symbol'] == 'BTC/USDT'].sort_values('timestamp')
    regime_data = {}
    if len(btc_data) >= 168:
        btc_close = btc_data['close'].values
        btc_ret_7d = btc_close[-1] / btc_close[-168] - 1 if btc_close[-168] > 0 else 0
        btc_hourly_rets = np.diff(btc_close[-169:]) / (btc_close[-169:-1] + 1e-10)
        btc_vol_7d = float(np.std(btc_hourly_rets))
        trend_strength = abs(btc_ret_7d) / (btc_vol_7d * np.sqrt(168) + 1e-10)
        trend_direction = btc_ret_7d / (btc_vol_7d * np.sqrt(168) + 1e-10)
        regime_data = {
            'trend_strength': trend_strength,
            'trend_direction': trend_direction,
        }
        print(f"   📊 BTC regime: 7d={btc_ret_7d*100:+.1f}%, trend_str={trend_strength:.2f}")

    result = latest[['symbol', 'score', 'confidence', 'deriv_scale']].sort_values(
        'score', ascending=False).reset_index(drop=True)
    result.attrs['model_info'] = {'n_models': len(lgb_models) + len(xgb_models),
                                  'groups': {'lgb_cls': len(lgb_models), 'xgb_cls': len(xgb_models)}}
    result.attrs['regime_data'] = regime_data
    return result


def generate_signal_ridge(df, root):
    """Generate trading signal using Ridge mean-reversion model + regime filter."""
    model_path = os.path.join(root, 'results_ridge_prod', 'model.json')
    if not os.path.exists(model_path):
        print(f"   ❌ Ridge model not found: {model_path}")
        print(f"      Run: python train_ridge_prod.py")
        return None

    with open(model_path) as f:
        model_data = json.load(f)

    features = model_data['features']
    coef = np.array(model_data['coef'])
    intercept = model_data['intercept']

    # Latest snapshot per symbol (features already CS-ranked by cross_sectional_rank)
    latest = df.groupby('symbol').last().reset_index()

    missing = [f for f in features if f not in latest.columns]
    if missing:
        print(f"   ⚠️  Ridge missing {len(missing)} features: {missing}")
        for f in missing:
            latest[f] = 0.0

    X = latest[features].fillna(0).values
    scores = X @ coef + intercept

    # Regime filter: scale down in strong BTC trends
    btc_data = df[df['symbol'] == 'BTC/USDT'].sort_values('timestamp')
    regime_data = {}  # R7: pass regime info to construct_portfolio
    if len(btc_data) >= 168:
        btc_close = btc_data['close'].values
        btc_ret_7d = btc_close[-1] / btc_close[-168] - 1 if btc_close[-168] > 0 else 0
        btc_hourly_rets = np.diff(btc_close[-169:]) / (btc_close[-169:-1] + 1e-10)
        btc_vol_7d = float(np.std(btc_hourly_rets))
        trend_strength = abs(btc_ret_7d) / (btc_vol_7d * np.sqrt(168) + 1e-10)
        mr_scale = float(np.clip(1.5 - 0.5 * trend_strength, 0.2, 1.0))

        # R7: vol_regime (48h vol / 720h mean vol) and trend_direction
        all_hourly_rets = np.diff(btc_close) / (btc_close[:-1] + 1e-10)
        btc_vol_48h = float(np.std(all_hourly_rets[-48:])) if len(all_hourly_rets) >= 48 else btc_vol_7d
        btc_vol_long = float(np.std(all_hourly_rets[-720:])) if len(all_hourly_rets) >= 720 else btc_vol_48h
        vol_regime = btc_vol_48h / (btc_vol_long + 1e-10)
        trend_direction = btc_ret_7d / (btc_vol_7d * np.sqrt(168) + 1e-10)

        regime_data = {
            'trend_strength': trend_strength,
            'vol_regime': vol_regime,
            'trend_direction': trend_direction,
        }
        print(f"   📊 Regime: BTC 7d={btc_ret_7d*100:+.1f}%, "
              f"trend={trend_strength:.2f}, mr_scale={mr_scale:.2f}, "
              f"vol_regime={vol_regime:.2f}, trend_dir={trend_direction:+.2f}")
    else:
        mr_scale = 1.0
        print(f"   ⚠️  Regime: insufficient BTC data, mr_scale=1.0")

    scores = scores * mr_scale

    # Z-normalize cross-sectionally
    score_std = np.std(scores)
    if score_std > 1e-10:
        scores = (scores - np.mean(scores)) / score_std

    latest['score'] = scores
    latest['confidence'] = np.full(len(latest), 0.5)
    latest['deriv_scale'] = np.full(len(latest), 1.0)

    print(f"   ✅ Ridge model: {len(features)} feats, "
          f"α={model_data.get('alpha', '?')}, mr_scale={mr_scale:.2f}")

    result = latest[['symbol', 'score', 'confidence', 'deriv_scale']].sort_values(
        'score', ascending=False).reset_index(drop=True)
    result.attrs['model_info'] = {'n_models': 1, 'groups': {'ridge': 1}}
    result.attrs['regime_data'] = regime_data  # R7
    return result


def add_12h_features(df):
    """
    v7 features optimized for 12h holding period.
    Mean-reversion signals, multi-day momentum, overnight effects,
    enhanced funding rate features, exhaustion/breakout signals.
    (Copied from run_pipeline_v7.py for offline fast_sim use.)
    """
    print("   🕐 Adding 12h-specific features (v7 enhanced)...")

    for sym, grp in df.groupby('symbol'):
        c = grp['close']
        v = grp['volume']
        h = grp['high']
        l = grp['low']
        idx = grp.index

        # 12h momentum z-score
        r12 = c.pct_change(12)
        r12_mean = r12.rolling(168).mean()
        r12_std = r12.rolling(168).std() + 1e-10
        df.loc[idx, 'mom_12h_zscore'] = (r12 - r12_mean) / r12_std

        # Mean-reversion: distance from 12h VWAP
        vwap_12 = (c * v).rolling(12).sum() / (v.rolling(12).sum() + 1e-10)
        df.loc[idx, 'vwap_12h_dist'] = c / vwap_12 - 1

        # Multi-day momentum
        df.loc[idx, 'mom_3d'] = c.pct_change(72)
        df.loc[idx, 'mom_7d'] = c.pct_change(168)

        # Momentum acceleration
        ret_12 = c.pct_change(12)
        ret_12_prev = c.shift(12).pct_change(12)
        df.loc[idx, 'mom_accel_12h'] = ret_12 - ret_12_prev

        # Volume trend
        vol_12 = v.rolling(12).mean()
        vol_48 = v.rolling(48).mean() + 1e-10
        df.loc[idx, 'vol_trend_12_48'] = vol_12 / vol_48 - 1

        # Session feature
        hours = grp['timestamp'].dt.hour
        df.loc[idx, 'is_asian_session'] = ((hours >= 0) & (hours < 12)).astype(float)

        # Range expansion
        h12 = h.rolling(12).max()
        l12 = l.rolling(12).min()
        range12 = (h12 - l12) / (c + 1e-10)
        range_avg = range12.rolling(168).mean() + 1e-10
        df.loc[idx, 'range_expansion_12h'] = range12 / range_avg - 1

        # v7: Trend exhaustion — close position in range
        range_pos = (c - l12) / (h12 - l12 + 1e-10)
        df.loc[idx, 'range_position_12h'] = range_pos

        # v7: Volume-weighted price change
        vwap_change = (c * v).rolling(12).sum() / (v.rolling(12).sum() + 1e-10)
        vwap_prev = (c.shift(12) * v.shift(12)).rolling(12).sum() / (v.shift(12).rolling(12).sum() + 1e-10)
        df.loc[idx, 'vwpc_12h'] = vwap_change / (vwap_prev + 1e-10) - 1

        # v7: Higher-high / lower-low counts
        hh = (h > h.shift(1)).rolling(12).sum()
        ll = (l < l.shift(1)).rolling(12).sum()
        df.loc[idx, 'hh_count_12h'] = hh
        df.loc[idx, 'll_count_12h'] = ll
        df.loc[idx, 'trend_strength_12h'] = hh - ll

        # v7: Volatility crush / expansion ratio
        vol_recent = c.pct_change().rolling(12).std()
        vol_base = c.pct_change().rolling(72).std() + 1e-10
        df.loc[idx, 'vol_crush_ratio'] = vol_recent / vol_base

        # v7: Close-to-close vs intraday range (directional quality)
        ret_abs = (c - c.shift(12)).abs()
        hi_lo_path = (h - l).rolling(12).sum()
        df.loc[idx, 'direction_quality_12h'] = ret_abs / (hi_lo_path + 1e-10)

    # Cross-sectional 12h momentum rank
    df['ret_12h_temp'] = df.groupby('symbol')['close'].transform(lambda x: x.pct_change(12))
    df['ret_12h_cs_rank'] = df.groupby('timestamp')['ret_12h_temp'].rank(pct=True)
    df.drop(columns=['ret_12h_temp'], inplace=True)

    # Cross-sectional volume surprise
    df['vol_12h_sum'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(12).sum())
    df['vol_12h_cs_rank'] = df.groupby('timestamp')['vol_12h_sum'].rank(pct=True)
    df.drop(columns=['vol_12h_sum'], inplace=True)

    # Cross-sectional funding rate rank (if available)
    if 'funding_rate' in df.columns:
        df['funding_cs_rank'] = df.groupby('timestamp')['funding_rate'].rank(pct=True)
        df['cum_funding_24h'] = df.groupby('symbol')['funding_rate'].transform(
            lambda x: x.rolling(3, min_periods=1).sum()
        )
        df['cum_funding_72h'] = df.groupby('symbol')['funding_rate'].transform(
            lambda x: x.rolling(9, min_periods=1).sum()
        )

    print(f"   ✅ Added ~18 features for 12h holding period (v7)")
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
    """Load saved CatBoost model files (.cbm)."""
    from catboost import CatBoostRegressor
    model_files = sorted(Path(results_dir).glob('cb_model_seed_*.cbm'))
    models = []
    for f in model_files:
        m = CatBoostRegressor()
        m.load_model(str(f))
        models.append(m)
    return models


def generate_lgb_signal(df, models, feat_cols):
    """Generate signal from LGB model ensemble."""
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
    return latest[['symbol', 'pred_lgb']].copy()


def generate_signal(df, feat_cols, root, use_meta=True, use_deriv_gate=True,
                    use_xgb=True, cb_only=False):
    """
    Generate production ensemble signal.

    Pipeline: LGB v6 + LGB v7 + CatBoost [+ XGB] [→ meta-model] [→ deriv gate] → score.
    Flags allow disabling components based on walk-forward validation results.
    cb_only: if True, skip LGB and XGB models — CatBoost solo mode.
    """
    latest = df.groupby('symbol').last().reset_index()

    # ── Load L0 model groups ──────────────────────────────────
    model_groups = []        # [(models, feature_names), ...]
    model_group_labels = []  # 'v6', 'v7', 'cb'
    loaded_types = set()

    lgb_candidates = [
        ("v6", "results_v6_huber_prod"),
        ("v6", "results/production/lgb_v6_no_news"),
        ("v6", "results_v6_prod"),
        ("v6", "results_v6"),
        ("v7", "results_v7_huber_prod"),
        ("v7", "results/production/lgb_v7_no_news"),
        ("v7", "results_v7_prod"),
        ("v7", "results_v7"),
    ]
    if not cb_only:
        for mtype, d in lgb_candidates:
            if mtype in loaded_types:
                continue
            p = os.path.join(root, d)
            if os.path.isdir(p) and any(f.endswith('.txt') for f in os.listdir(p)):
                ms = load_lgb_models(p)
                if ms:
                    mf_g = ms[0].feature_name()
                    n_missing = sum(1 for c in mf_g if c not in latest.columns)
                    for c in [c for c in mf_g if c not in latest.columns]:
                        latest[c] = 0.0
                    model_groups.append((ms, mf_g))
                    model_group_labels.append(mtype)
                    loaded_types.add(mtype)
                    label = "PROD" if "_prod" in p or "production" in p else "research"
                    warn = f" ⚠️ {n_missing} zero-filled" if n_missing > 3 else ""
                    print(f"   {os.path.basename(p)}: {len(ms)} LGB, {len(mf_g)} feats [{label}]{warn}")

    # CatBoost
    cb_dir = None
    for _cb in ["results_catboost_prod", "results_catboost_huber_prod", "results/production/catboost_with_news", "results_catboost"]:
        _p = os.path.join(root, _cb)
        if os.path.isdir(_p):
            cb_dir = _p
            break
    if cb_dir and os.path.isdir(cb_dir):
        try:
            ms = load_catboost_models(cb_dir)
            if ms:
                fn_path = os.path.join(cb_dir, 'feature_names.json')
                if os.path.exists(fn_path):
                    with open(fn_path) as _f:
                        mf_g = json.load(_f)
                else:
                    mf_g = ms[0].feature_names_
                n_missing = sum(1 for c in mf_g if c not in latest.columns)
                for c in [c for c in mf_g if c not in latest.columns]:
                    latest[c] = 0.0
                model_groups.append((ms, mf_g))
                model_group_labels.append('cb')
                warn = f" ⚠️ {n_missing} zero-filled" if n_missing > 3 else ""
                print(f"   catboost: {len(ms)} CB, {len(mf_g)} feats{warn}")
        except ImportError:
            print("   ⚠️  catboost not installed, skipping")

    # XGBoost
    xgb_dir = None
    for _xd in ["results_xgboost_huber_prod", "results/production/xgboost", "results_xgboost_prod", "results_xgboost"]:
        _p = os.path.join(root, _xd)
        if os.path.isdir(_p) and any(f.endswith('.json') for f in os.listdir(_p)):
            xgb_dir = _p
            break
    if xgb_dir and use_xgb and not cb_only:
        try:
            import xgboost as xgb_lib
            _files = sorted(Path(xgb_dir).glob('xgb_model_seed_*.json'))
            if _files:
                ms = [xgb_lib.Booster(model_file=str(f)) for f in _files]
                fn_path = os.path.join(xgb_dir, 'feature_names.json')
                if os.path.exists(fn_path):
                    with open(fn_path) as _f:
                        mf_g = json.load(_f)
                else:
                    mf_g = ms[0].feature_names
                n_missing = sum(1 for c in mf_g if c not in latest.columns)
                for c in [c for c in mf_g if c not in latest.columns]:
                    latest[c] = 0.0
                # XGBoost Booster.predict() needs DMatrix → wrap
                class _XgbWrapper:
                    def __init__(self, booster, feat_names):
                        self._b = booster
                        self._fn = feat_names
                    def predict(self, X):
                        import xgboost as _xgb
                        dm = _xgb.DMatrix(X, feature_names=self._fn)
                        return self._b.predict(dm)
                ms_wrapped = [_XgbWrapper(m, mf_g) for m in ms]
                model_groups.append((ms_wrapped, mf_g))
                model_group_labels.append('xgb')
                warn = f" ⚠️ {n_missing} zero-filled" if n_missing > 3 else ""
                print(f"   xgboost: {len(ms)} XGB, {len(mf_g)} feats{warn}")
        except ImportError:
            print("   ⚠️  xgboost not installed, skipping")
        except Exception as e:
            print(f"   ⚠️  XGBoost load failed: {e}")

    # MLP
    mlp_dir = None
    for _md in ["results/production/mlp", "results_mlp_prod", "results_mlp"]:
        _p = os.path.join(root, _md)
        if os.path.isdir(_p) and any(f.endswith('.pt') for f in os.listdir(_p)):
            mlp_dir = _p; break
    if mlp_dir and not cb_only:
        try:
            import torch as _torch
            from run_pipeline_mlp import AlphaMLP
            fn_path = os.path.join(mlp_dir, 'feature_names.json')
            if os.path.exists(fn_path):
                with open(fn_path) as _f:
                    mf_g = json.load(_f)
                _pt_files = sorted([f for f in os.listdir(mlp_dir) if f.endswith('.pt')])
                if _pt_files:
                    _mlp_models = []
                    for _pf in _pt_files:
                        ckpt = _torch.load(os.path.join(mlp_dir, _pf),
                                           map_location='cpu', weights_only=False)
                        cfg = ckpt['config']
                        hdims = cfg.get('hidden_dims', (256, 128, 64))
                        if isinstance(hdims, list):
                            hdims = tuple(hdims)
                        m = AlphaMLP(input_dim=ckpt['input_dim'],
                                     hidden_dims=hdims,
                                     dropout=cfg.get('dropout', 0.3))
                        m.load_state_dict(ckpt['model_state_dict'])
                        m.eval()
                        _mlp_models.append(m)
                    n_missing = sum(1 for c in mf_g if c not in latest.columns)
                    for c in [c for c in mf_g if c not in latest.columns]:
                        latest[c] = 0.0
                    class _MlpWrapper:
                        def __init__(self, model, feat_names):
                            self._m = model
                            self._fn = feat_names
                        def predict(self, X):
                            import torch as __torch
                            with __torch.no_grad():
                                t = __torch.FloatTensor(X)
                                return self._m(t).numpy()
                    ms_wrapped = [_MlpWrapper(m, mf_g) for m in _mlp_models]
                    model_groups.append((ms_wrapped, mf_g))
                    model_group_labels.append('mlp')
                    warn = f" ⚠️ {n_missing} zero-filled" if n_missing > 3 else ""
                    print(f"   mlp: {len(_mlp_models)} MLP, {len(mf_g)} feats{warn}")
        except ImportError:
            print("   ⚠️  torch not installed, skipping MLP")
        except Exception as e:
            print(f"   ⚠️  MLP load failed: {e}")

    if not model_groups:
        print("   ❌ No production models found")
        return None

    # ── L0 predictions ────────────────────────────────────────
    all_individual = []
    per_group_scores = []
    for ms, mf_g in model_groups:
        X = latest[mf_g].values
        preds = [m.predict(X) for m in ms]
        all_individual.extend(preds)
        per_group_scores.append(np.mean(preds, axis=0))

    # Confidence = model agreement
    if len(all_individual) > 1:
        normed = [(p - p.mean()) / (p.std() + 1e-10) for p in all_individual]
        model_std = np.std(normed, axis=0)
        confidence = 1.0 / (1.0 + model_std)
    else:
        confidence = np.ones(len(latest)) * 0.5

    # ── Meta-model stacking ───────────────────────────────────
    meta_group_idx = {lbl: i for i, lbl in enumerate(model_group_labels)}
    scores = None

    if use_meta and all(k in meta_group_idx for k in ('v6', 'v7', 'cb')):
        try:
            from src.models.meta_model import MetaModelInference
            # Use lgb_minimal (non-linear LGB meta-model) — ridge compresses
            # scores to a narrow band [0.44, 0.56] making L/S nearly random.
            # lgb_minimal produces wider score range [-2, +1.5] for meaningful
            # portfolio differentiation.
            meta_inf = MetaModelInference.load('auto', variant='lgb_minimal', root=root)
            if meta_inf is not None:
                pred_xgb = per_group_scores[meta_group_idx['xgb']] if 'xgb' in meta_group_idx else None
                scores = meta_inf.predict(
                    latest,
                    pred_v6=per_group_scores[meta_group_idx['v6']],
                    pred_v7=per_group_scores[meta_group_idx['v7']],
                    pred_cb=per_group_scores[meta_group_idx['cb']],
                    pred_xgb=pred_xgb,
                )
                print(f"   ✅ Meta-model lgb_minimal applied")
        except Exception as e:
            print(f"   ⚠️  Meta-model failed: {e}")

    if scores is None:
        scores = np.mean(per_group_scores, axis=0)
        print(f"   ✅ Simple mean ensemble ({len(model_groups)} groups)")

    # ── Deriv risk gate ───────────────────────────────────────
    DERIV_GATE_MIN, DERIV_GATE_MAX = 0.3, 1.0
    deriv_scale = np.ones(len(latest))

    if not use_deriv_gate:
        print("   ⏭️  Deriv gate: disabled via --no-deriv-gate")

    deriv_dir = None
    for _dd in ["results/production/deriv_only", "results_deriv"]:
        _p = os.path.join(root, _dd)
        if os.path.isdir(_p) and any(f.endswith('.txt') for f in os.listdir(_p)):
            deriv_dir = _p
            break

    if deriv_dir and use_deriv_gate:
        try:
            import lightgbm as _lgb
            _files = sorted(Path(deriv_dir).glob('deriv_model_seed_*.txt'))
            if not _files:
                _files = sorted(Path(deriv_dir).glob('lgb_model_seed_*.txt'))
            _d_ms = [_lgb.Booster(model_file=str(f)) for f in _files]
            if _d_ms:
                fn_path = os.path.join(deriv_dir, 'feature_names.json')
                if os.path.exists(fn_path):
                    with open(fn_path) as _f:
                        _d_feats = json.load(_f)
                else:
                    _d_feats = _d_ms[0].feature_name()
                for c in [c for c in _d_feats if c not in latest.columns]:
                    latest[c] = 0.0
                X_d = latest[_d_feats].values
                d_preds = np.mean([m.predict(X_d) for m in _d_ms], axis=0)

                # Rank-based agreement
                n = len(scores)
                ens_rank = np.argsort(np.argsort(scores)).astype(float) / max(n - 1, 1)
                drv_rank = np.argsort(np.argsort(d_preds)).astype(float) / max(n - 1, 1)
                for i in range(n):
                    rank_diff = abs(ens_rank[i] - drv_rank[i])
                    ens_extreme = abs(ens_rank[i] - 0.5) * 2
                    effective_disagree = rank_diff * ens_extreme
                    scale = DERIV_GATE_MAX - effective_disagree * (DERIV_GATE_MAX - DERIV_GATE_MIN) / 0.7
                    deriv_scale[i] = float(np.clip(scale, DERIV_GATE_MIN, DERIV_GATE_MAX))
                print(f"   ✅ Deriv gate: avg scale {deriv_scale.mean():.2f}x")
        except Exception as e:
            print(f"   ⚠️  Deriv gate failed: {e}")

    # ── Build result ──────────────────────────────────────────
    final_scores = scores * deriv_scale

    # Z-normalize scores cross-sectionally: converts raw model output ~[0.44, 0.56]
    # to z-scores ~[-2, +2].  This makes score magnitude meaningful:
    #   |z| > 1  → strong conviction (top/bottom ~16%)
    #   |z| > 2  → very strong conviction (top/bottom ~2%)
    # Without this, all scores cluster near 0.5 and L/S differentiation is poor.
    # The old (working) pipeline z-normalized each L0 model before averaging.
    score_std = np.std(final_scores)
    if score_std > 1e-10:
        final_scores = (final_scores - np.mean(final_scores)) / score_std

    latest['score'] = final_scores
    latest['confidence'] = confidence
    latest['deriv_scale'] = deriv_scale

    n_models = sum(len(ms) for ms, _ in model_groups)
    arch_parts = [f"{len(model_groups)} groups ({n_models} models)"]
    if use_meta and scores is not None:
        arch_parts.append("meta-lgb")
    if use_deriv_gate:
        arch_parts.append("deriv-gate")
    print(f"   🏗️  Architecture: {' + '.join(arch_parts)}")

    # Build model info for dashboard
    group_counts = {}
    for i, (ms, _) in enumerate(model_groups):
        group_counts[model_group_labels[i]] = len(ms)
    model_info = {'n_models': n_models, 'groups': group_counts}

    result = latest[['symbol', 'score', 'confidence', 'deriv_scale']].sort_values(
        'score', ascending=False).reset_index(drop=True)
    result.attrs['model_info'] = model_info
    return result


# ============================================================
# PORTFOLIO CONSTRUCTION (risk-managed)
# ============================================================

def _edge_boost_weights(scores, n_long, n_short, coin_vol=None):
    """Compute per-position weights using edge-proportional boost.

    Mirrors run_fast_sim.py compute_weights(edge_boost=True):
      edge_i  = |score_i - median|
      boost_i = 1 + min(edge_i / P75_edge, 3.0)
      weight  = boost / sum(boosts)   (per side, normalised to 1)

    If coin_vol is provided, applies inverse-vol scaling: weight *= 1/σ.
    Returns list of (symbol, side, weight, score) sorted long-first.
    """
    all_scores = scores['score'].values
    median_score = np.median(all_scores)
    abs_edges = np.abs(all_scores - median_score)
    edge_p75 = float(np.percentile(abs_edges, 75)) + 1e-10

    sorted_df = scores.sort_values('score', ascending=False).reset_index(drop=True)
    long_df = sorted_df.head(n_long)
    short_df = sorted_df.tail(n_short)

    def _side_weights(df):
        edges = np.abs(df['score'].values - median_score)
        boosts = 1.0 + np.minimum(edges / edge_p75, 3.0)
        w = boosts / boosts.sum()
        # Inverse-vol sizing: scale by 1/σ
        if coin_vol:
            vol_arr = np.array([coin_vol.get(row['symbol'], 0.05)
                                for _, row in df.iterrows()])
            vol_arr = np.clip(vol_arr, 0.005, 0.20)
            w = w * (1.0 / vol_arr)
            w = w / w.sum()
        # Cap any single position at 25% of its side
        w = np.minimum(w, 0.25)
        w = w / w.sum()  # re-normalise after cap
        return w

    result = []
    for w, (_, row) in zip(_side_weights(long_df), long_df.iterrows()):
        result.append((row['symbol'], 'long', float(w), round(row['score'], 4)))
    for w, (_, row) in zip(_side_weights(short_df), short_df.iterrows()):
        result.append((row['symbol'], 'short', float(w), round(row['score'], 4)))
    return result


def construct_portfolio(signals, capital, risk_cfg, state, leverage=1, coin_vol=None, regime_data=None):
    """
    Risk-managed portfolio construction with edge-boost sizing.

    R7 enhancements: regime-conditional asymmetry, vol scaling,
    strategy momentum, EQ-MOM boost, dynamic Kelly L/S allocation.

    state: dict tracking equity curve for DD stop.
    leverage: exchange leverage multiplier (positions get leverage × capital).
    coin_vol: optional {symbol: realized_vol} for inverse-vol sizing.
    regime_data: optional dict with trend_strength, vol_regime, trend_direction.
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
            # Reset peak to current equity to avoid deadlock
            state['peak'] = equity
            print(f"   🟢 DD recovered to {dd*100:.1f}%, resuming trading (peak reset to {equity:.2f})")
        else:
            print(f"   🔴 DD stop active ({dd*100:.1f}%), skipping cycle")
            return []

    if dd < risk_cfg['dd_stop']:
        state['stopped'] = True
        print(f"   🔴 DD hit {dd*100:.1f}% (limit {risk_cfg['dd_stop']*100:.0f}%), stopping")
        return []

    # Vol targeting: scale position based on recent realized vol
    # Cap (0.5, 1.2): don't shrink below 0.5x in stress, don't inflate above
    # 1.2x in calm markets (was 0.1–3.0 = 30x spread, now 2.4x).
    # CLS mode: skip vol targeting (not used in R25 simulation)
    vol_history = state.get('recent_rets', [])
    if risk_cfg.get('_cls_mode', False):
        vol_scale = 1.0
    elif len(vol_history) >= 6:
        realized_vol = np.std(vol_history[-risk_cfg['vol_lookback']:]) + 1e-10
        vol_scale = np.clip(risk_cfg['vol_target'] / realized_vol, 0.5, 1.2)
    else:
        vol_scale = 1.0

    # Confidence check
    conf_thresh = risk_cfg.get('confidence_threshold', 0.0)
    if conf_thresh > 0:
        scores = signals['score'].values
        max_spread = scores.max() - scores.min()
        if max_spread < conf_thresh:
            print(f"   ⚠️  Signal too weak (spread={max_spread:.2f} < {conf_thresh}), skipping")
            return []

    # Filter out blocked symbols (only applies to OKX demo, live has all symbols)
    if risk_cfg.get('_demo_mode', False):
        before = len(signals)
        signals = signals[~signals['symbol'].isin(_OKX_BLOCKED)].copy()
        after = len(signals)
        if before != after:
            print(f"   🚫 Filtered {before - after} blocked symbols ({after} tradeable)")

    # Min z-score filter: split by sign to match sim logic
    # Sim: cand_L = [score_z >= min_zscore], cand_S = [score_z <= -min_zscore]
    # Never short a coin the model predicts will go up (positive score)
    min_zs = risk_cfg.get('min_zscore', 0.0)
    if min_zs > 0:
        before_mz = len(signals)
        long_cands = signals[signals['score'] >= min_zs].sort_values('score', ascending=False)
        short_cands = signals[signals['score'] <= -min_zs].sort_values('score', ascending=True)
        n_long = min(n_long, len(long_cands))
        n_short = min(n_short, len(short_cands))
        if n_long + n_short >= 2:
            signals = pd.concat([long_cands.head(n_long), short_cands.head(n_short)])
        n_dropped = before_mz - len(signals)
        if n_dropped > 0:
            print(f"   🔍 Min z-score {min_zs}: dropped {n_dropped} weak signals "
                  f"({n_long}L + {n_short}S = {len(signals)} remain)")
    else:
        # No min_zscore: classic rank-based market-neutral
        signals = signals.sort_values('score', ascending=False).reset_index(drop=True)
        n = len(signals)
        n_long = min(n_long, n // 3)
        n_short = min(n_short, n // 3)

    # ── CLS mode: R26 winner uses trend_cutoff=0.9, dyn_threshold=0.7 ──
    cls_mode = risk_cfg.get('_cls_mode', False)
    dyn_exposure = 1.0
    r7_vol_scale = 1.0
    sm_scale = 1.0
    eq_boost = 1.0

    if cls_mode:
        # R26 CFG: trend_cutoff=0.9 → skip cycle entirely
        if regime_data:
            ts_val = regime_data.get('trend_strength', 0)
            if ts_val > 0.9:
                print(f"   🔴 CLS trend_cutoff: trend_str={ts_val:.2f} > 0.9, skipping cycle")
                return []
            # dyn_threshold=0.7 → scale exposure down linearly
            if ts_val > 0.7:
                dyn_exposure = max(0.1, 1.0 - (ts_val - 0.7) / (0.9 - 0.7) * 0.5)
                print(f"   📊 CLS dyn_exposure: trend_str={ts_val:.2f}, exp={dyn_exposure:.2f}")
        # CLS: no regime-asym, no vol-scale, no sm, no eq-boost (match sim)
    else:
        # ── R7: Regime-conditional asymmetry ──
        if regime_data:
            trend_dir = regime_data.get('trend_direction', 0)
            if not np.isnan(trend_dir):
                n_base_l, n_base_s = n_long, n_short
                if trend_dir >= 0.3:       # mild bull → tilt long
                    n_long = min(n_long + 1, len(signals) // 3)
                    n_short = max(2, n_short - 1)
                elif trend_dir <= -0.3:    # mild bear → tilt short
                    n_long = max(2, n_long - 1)
                    n_short = min(n_short + 1, len(signals) // 3)
                if (n_long, n_short) != (n_base_l, n_base_s):
                    print(f"   📊 Regime-asym: trend_dir={trend_dir:+.2f} → {n_long}L/{n_short}S")

        # ── R7: Vol scaling (BTC vol regime) ──
        if regime_data:
            vol_regime = regime_data.get('vol_regime', 1.0)
            if vol_regime > 0:
                r7_vol_scale = min(1.5, 1.0 / max(0.5, vol_regime))
                if abs(r7_vol_scale - 1.0) > 0.05:
                    print(f"   📊 R7 Vol-scale: vol_regime={vol_regime:.2f}, scale={r7_vol_scale:.2f}")

        # ── R7: Dynamic exposure (reduce in strong trends) ──
        if regime_data:
            ts_val = regime_data.get('trend_strength', 0)
            if ts_val > 0.8:
                dyn_exposure = max(0.5, 1.0 - (ts_val - 0.8) * 0.5)
                print(f"   📊 Dynamic exposure: trend_str={ts_val:.2f}, dyn_exp={dyn_exposure:.2f}")

        # ── R7: Strategy Momentum 48h (4 × 12h cycles) ──
        recent_rets = state.get('recent_rets', [])
        if len(recent_rets) >= 4:
            recent_4 = recent_rets[-4:]
            cum_48h = float(np.prod([1 + r for r in recent_4]))
            if cum_48h < 0.97:
                sm_scale = max(0.3, cum_48h)
                print(f"   📊 Strategy Momentum: cum_48h={cum_48h:.3f}, sm_scale={sm_scale:.2f}")

        # ── R7: EQ-MOM Boost (drawdown scaling + recovery boost) ──
        equity_hist = [x for x in state.get('r7_equity_vals', []) if isinstance(x, (int, float))]
        if len(equity_hist) > 4:
            recent_eq = equity_hist[-1]
            peak_eq = max(equity_hist)
            eq_dd = (recent_eq - peak_eq) / (peak_eq + 1e-10)
            if eq_dd < -0.05:
                eq_boost = max(0.3, 1.0 + eq_dd * 3)
                print(f"   📊 EQ-MOM: DD={eq_dd*100:.1f}%, boost={eq_boost:.2f}")
            elif eq_dd > -0.01:
                lookback = min(8, len(equity_hist))
                min_eq = min(equity_hist[-lookback:])
                recovery = (recent_eq - min_eq) / (min_eq + 1e-10)
                if recovery > 0.05:
                    eq_boost = min(1.5, 1.0 + recovery * 0.5)
                    print(f"   📊 EQ-MOM: recovery={recovery*100:.1f}%, boost={eq_boost:.2f}")

    # Build positions
    signals = signals.sort_values('score', ascending=False).reset_index(drop=True)
    total_positions = n_long + n_short

    if total_positions == 0:
        return []

    # Total allocation: capital × kelly × vol_scale × R7 factors × leverage
    effective_kelly = kelly * vol_scale * r7_vol_scale * dyn_exposure * sm_scale * eq_boost
    total_alloc = capital * effective_kelly * leverage

    # L/S allocation split & position sizing
    if cls_mode:
        # CLS: fixed 50/50 L/S split, equal-weight positions (match R25 simulation)
        long_alloc_frac = 0.5
    else:
        # R7: Dynamic Kelly L/S split (instead of fixed 50/50)
        if n_long > 0 and n_short > 0:
            long_scores = signals.head(n_long)['score']
            short_scores = signals.tail(n_short)['score']
            pred_spread = float(long_scores.mean() - short_scores.mean())
            long_alloc_frac = float(np.clip(0.5 + pred_spread * 5, 0.3, 0.7))
        else:
            long_alloc_frac = 0.5

    long_half = total_alloc * long_alloc_frac
    short_half = total_alloc * (1 - long_alloc_frac)

    # Cap per position at 15% of leveraged capital for diversification
    max_per_pos = capital * leverage * 0.15

    if cls_mode:
        # CLS: equal-weight positions (sim uses .mean() of returns = equal weight)
        sorted_df = signals.sort_values('score', ascending=False).reset_index(drop=True)
        long_df = sorted_df.head(n_long)
        short_df = sorted_df.tail(n_short)
        weighted = []
        if len(long_df) > 0:
            w_l = 1.0 / len(long_df)
            for _, row in long_df.iterrows():
                weighted.append((row['symbol'], 'long', w_l, round(row['score'], 4)))
        if len(short_df) > 0:
            w_s = 1.0 / len(short_df)
            for _, row in short_df.iterrows():
                weighted.append((row['symbol'], 'short', w_s, round(row['score'], 4)))
    else:
        # Compute edge-boost weights (with optional inverse-vol sizing)
        weighted = _edge_boost_weights(signals, n_long, n_short, coin_vol=coin_vol)

    positions = []
    for symbol, side, weight, score in weighted:
        side_alloc = long_half if side == 'long' else short_half
        usd = round(min(side_alloc * weight, max_per_pos), 2)
        if usd < 5:  # OKX minimum
            continue
        positions.append({
            'symbol': symbol,
            'side': side,
            'usd': usd,
            'score': score,
        })

    actual_alloc = sum(p['usd'] for p in positions)
    n_l = sum(1 for p in positions if p['side'] == 'long')
    n_s = sum(1 for p in positions if p['side'] == 'short')
    usds = [p['usd'] for p in positions]
    sizing_mode = 'equal-weight' if cls_mode else 'edge-boost'
    print(f"   📊 Allocating ${actual_alloc:.0f} of ${capital:.0f} "
          f"(kelly={kelly:.0%} × vol={vol_scale:.2f} × r7_vol={r7_vol_scale:.2f} "
          f"× dyn={dyn_exposure:.2f} × sm={sm_scale:.2f} × eqb={eq_boost:.2f} × lev={leverage}x)")
    print(f"   📊 Positions: {n_l}L + {n_s}S = {n_l+n_s} "
          f"[L/S split={long_alloc_frac:.0%}/{1-long_alloc_frac:.0%}, "
          f"{sizing_mode} ${min(usds):.0f}–${max(usds):.0f}]")

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

    # Set net position mode (single direction per symbol)
    try:
        exchange.private_post_account_set_position_mode({'posMode': 'net_mode'})
        print("   📋 Position mode: net")
    except Exception:
        pass  # already set or not supported

    try:
        balance = exchange.fetch_balance()
        usdt = balance.get('USDT', {}).get('free', 0)
        print(f"   💰 Balance: ${usdt:.2f} USDT")
    except Exception as e:
        print(f"   ⚠️  Balance check: {e}")

    return exchange


def close_all(exchange):
    """Close all open positions."""
    if not exchange:
        return
    try:
        positions = exchange.fetch_positions()
        # Sort by notional ascending to free margin from small positions first
        open_pos = [p for p in positions if float(p.get('contracts', 0)) > 0]
        open_pos.sort(key=lambda p: abs(float(p.get('notional', 0))))
        for pos in open_pos:
            side = 'sell' if pos['side'] == 'long' else 'buy'
            try:
                exchange.create_order(
                    symbol=pos['symbol'], type='market', side=side,
                    amount=pos['contracts'],
                    params={'tdMode': 'isolated', 'posSide': 'net', 'reduceOnly': True},
                )
                print(f"      ✅ Closed {pos['side']} {pos['symbol']}")
            except Exception as e:
                print(f"      ⚠️  Close {pos['symbol']}: {str(e)[:120]}")
    except Exception as e:
        print(f"      ⚠️  Close failed: {e}")


def wait_for_margin(exchange, required_usd, timeout=45):
    """Poll balance until enough free USDT is available after closing positions."""
    import time as _time
    start = _time.time()
    _time.sleep(2)  # initial settle time
    while _time.time() - start < timeout:
        try:
            bal = exchange.fetch_balance()
            free = float(bal.get('USDT', {}).get('free', 0))
            if free >= required_usd * 0.9:  # 90% threshold
                print(f"      💰 Margin ready: ${free:.0f} free (need ${required_usd:.0f})")
                return True
            print(f"      ⏳ Waiting for margin: ${free:.0f} free (need ${required_usd:.0f})...")
        except Exception as e:
            print(f"      ⚠️  Balance poll error: {e}")
        _time.sleep(3)
    print(f"      ⚠️  Margin timeout after {timeout}s — proceeding anyway")
    return False


def execute(exchange, positions, dry_run=True, leverage=1):
    """Execute positions on OKX with proper USD→contract conversion and retry sweeps."""
    import time as _time
    results = []

    # Load markets for contract size info (needed for proper amount calculation)
    markets = {}
    if exchange and not dry_run:
        try:
            exchange.load_markets()
            markets = exchange.markets
        except Exception as e:
            print(f"      ⚠️  load_markets: {e}")

    def _usd_to_contracts(okx_sym, usd_amount):
        """Convert USD notional to number of contracts."""
        # Find the unified symbol (e.g., 'BTC/USDT:USDT') from exchange ID ('BTC-USDT-SWAP')
        market = None
        for sym, m in markets.items():
            if m.get('id') == okx_sym:
                market = m
                break
        if not market:
            return usd_amount  # fallback: assume 1:1

        ct_val = float(market.get('contractSize', 1))
        # Get current price
        try:
            ticker = exchange.fetch_ticker(market['symbol'])
            price = ticker['last']
        except Exception:
            return usd_amount  # fallback

        if price <= 0 or ct_val <= 0:
            return usd_amount

        contracts = usd_amount / (ct_val * price)
        # Round down to valid precision
        precision = market.get('precision', {}).get('amount', 0)
        if isinstance(precision, (int, float)) and precision > 0:
            contracts = round(int(contracts / precision) * precision, 10)
        else:
            contracts = int(contracts)

        return max(contracts, precision if isinstance(precision, (int, float)) else 1)

    def _try_order(pos, is_retry=False):
        """Attempt to place a single order. Returns True if filled, False if 51008."""
        okx_sym = SYMBOLS_TO_OKX.get(pos['symbol'])
        if not okx_sym:
            return True  # skip, not a retry candidate
        side = 'buy' if pos['side'] == 'long' else 'sell'
        try:
            if not is_retry:
                try:
                    exchange.set_leverage(leverage, okx_sym, params={'mgnMode': 'isolated'})
                except Exception:
                    pass

            amount = _usd_to_contracts(okx_sym, pos['usd'])
            order = exchange.create_order(
                symbol=okx_sym, type='market', side=side,
                amount=amount,
                params={'tdMode': 'isolated', 'posSide': 'net'},
            )
            tag = "🔄" if is_retry else "✅"
            print(f"      {tag} {side.upper():4s} ${pos['usd']:>7.0f} {okx_sym} ({amount} cts) → {order['id']}")
            results.append({**pos, 'status': 'filled', 'order_id': order['id']})
            return True
        except Exception as e:
            if '51008' in str(e):
                return False  # retry candidate
            print(f"      ❌ {side.upper():4s} ${pos['usd']:>7.0f} {okx_sym} → {e}")
            results.append({**pos, 'status': 'error', 'error': str(e)})
            return True  # non-retryable error

    if dry_run:
        for pos in positions:
            okx_sym = SYMBOLS_TO_OKX.get(pos['symbol'])
            if not okx_sym:
                continue
            side = 'buy' if pos['side'] == 'long' else 'sell'
            print(f"      [DRY] {side.upper():4s} ${pos['usd']:>7.0f} {okx_sym} "
                  f"(score: {pos['score']:+.3f})")
            results.append({**pos, 'status': 'dry_run'})
        return results

    # First pass
    pending = list(positions)
    failed = []
    for pos in pending:
        if not _try_order(pos):
            failed.append(pos)

    # Retry sweeps (up to 2 more passes with 10s waits)
    for sweep in range(2):
        if not failed:
            break
        wait_s = 10
        print(f"      ⏳ {len(failed)} orders need margin — waiting {wait_s}s (sweep {sweep + 1})...")
        _time.sleep(wait_s)
        still_failed = []
        for pos in failed:
            if not _try_order(pos, is_retry=True):
                still_failed.append(pos)
        failed = still_failed

    # Any remaining failures
    for pos in failed:
        okx_sym = SYMBOLS_TO_OKX.get(pos['symbol'], pos['symbol'])
        side = 'buy' if pos['side'] == 'long' else 'sell'
        print(f"      ❌ {side.upper():4s} ${pos['usd']:>7.0f} {okx_sym} → insufficient balance after retries")
        results.append({**pos, 'status': 'error', 'error': '51008 insufficient balance after retries'})

    return results


# ============================================================
# PARTIAL REBALANCE + LIMIT ORDERS
# ============================================================

def _convert_usd_to_contracts(exchange, markets, okx_sym, usd_amount):
    """Convert USD notional to contracts. Returns (contracts, ticker_info)."""
    market = None
    for sym, m in markets.items():
        if m.get('id') == okx_sym:
            market = m
            break
    if not market:
        return usd_amount, None

    ct_val = float(market.get('contractSize', 1))
    try:
        ticker = exchange.fetch_ticker(market['symbol'])
        price = ticker['last']
        bid = ticker.get('bid', price)
        ask = ticker.get('ask', price)
    except Exception:
        return usd_amount, None

    if price <= 0 or ct_val <= 0:
        return usd_amount, None

    contracts = usd_amount / (ct_val * price)
    precision = market.get('precision', {}).get('amount', 0)
    if isinstance(precision, (int, float)) and precision > 0:
        contracts = round(int(contracts / precision) * precision, 10)
    else:
        contracts = int(contracts)
    contracts = max(contracts, precision if isinstance(precision, (int, float)) else 1)
    return contracts, {'last': price, 'bid': bid, 'ask': ask}


def _settle_order(exchange, order, symbol, retries=6):
    """Fetch settled order data to get actual fill price/qty/cost."""
    if not exchange or not order or not order.get('id'):
        return order
    import time as _time
    for i in range(retries):
        try:
            _time.sleep(0.5 * (i + 1))  # 0.5, 1.0, 1.5, 2.0, 2.5, 3.0 = 10.5s max
            settled = exchange.fetch_order(order['id'], symbol)
            if settled.get('status') == 'closed' and settled.get('average'):
                return settled
        except Exception:
            pass
    return order


def _is_order_filled(order):
    """Check if order actually filled (non-zero price and qty)."""
    if not order:
        return False
    avg = order.get('average')
    filled = order.get('filled')
    return (order.get('status') == 'closed'
            and avg is not None and float(avg) > 0
            and filled is not None and float(filled) > 0)


def _log_execution(symbol, okx_sym, side, tier, attempt, order_type,
                   ticker_info, limit_px, order, fill_time_s, was_maker):
    """Write per-trade execution metrics to EXEC_LOG_PATH for R49 pilot analysis."""
    try:
        fill_price = float(order.get('average', 0) or order.get('price', 0) or 0)
        filled_qty = float(order.get('filled', 0) or 0)
        cost_usd = float(order.get('cost', 0) or 0)
        bid = float(ticker_info.get('bid', 0)) if ticker_info else 0
        ask = float(ticker_info.get('ask', 0)) if ticker_info else 0
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else fill_price
        spread_bps = (ask / bid - 1) * 10000 if bid > 0 else 0
        # slippage vs mid (positive = paid more than mid for buy, or received less for sell)
        if mid > 0 and fill_price > 0:
            raw_slip = (fill_price / mid - 1) * 10000
            slippage_bps = raw_slip if side == 'buy' else -raw_slip
        else:
            slippage_bps = 0
        # effective bps = slippage + fee (maker ~-1bp, taker ~+5bp on OKX)
        fee_bps = -1.0 if was_maker else 5.0
        effective_bps = slippage_bps + fee_bps

        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trading_logs')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'execution_log.csv')
        write_header = not os.path.exists(log_path)
        with open(log_path, 'a') as f:
            if write_header:
                f.write(EXEC_LOG_HEADER)
            f.write(
                f"{datetime.now(timezone.utc).isoformat()},"
                f"{symbol},{okx_sym},{tier},{side},{order_type},{attempt},"
                f"{bid:.8g},{ask:.8g},{mid:.8g},{limit_px:.8g if limit_px else ''},"
                f"{fill_price:.8g},{spread_bps:.2f},{slippage_bps:.2f},"
                f"{effective_bps:.2f},{filled_qty},{cost_usd:.4f},"
                f"{fill_time_s:.1f},{1 if was_maker else 0}\n"
            )
    except Exception:
        pass


def _maker_first_limit(exchange, symbol, okx_sym, side, amount, ticker_info, params):
    """
    R49.1 — Post-only maker-first limit order for TIER1 symbols.

    Strategy:
      - Attempt 1: post-only limit at mid +/- MAKER_MID_OFFSET (0.5bp inside spread)
      - Attempt 2: post-only limit at mid +/- (MAKER_MID_OFFSET + MAKER_AGGR_STEP)
      - Attempt 3: aggressive limit at bid/ask (like _limit_with_fallback)
      - Final: market fallback

    Logs: effective_bps, was_maker, fill_time_s to trading_logs/execution_log.csv
    Returns: order dict (filled) or None on failure
    """
    import time as _time

    tier = 'T1'

    if ticker_info is None:
        result = exchange.create_order(symbol=okx_sym, type='market', side=side,
                                       amount=amount, params=params)
        result = _settle_order(exchange, result, okx_sym)
        _log_execution(symbol, okx_sym, side, tier, 0, 'market_no_ticker',
                       ticker_info, None, result, 0, False)
        _log_fill(okx_sym, side, result, ticker_info, 'market')
        return result

    bid = float(ticker_info['bid'])
    ask = float(ticker_info['ask'])
    mid = (bid + ask) / 2

    # Three maker attempts: increasingly aggressive toward taker
    # Attempt 1: deep inside spread (true post-only)
    # Attempt 2: mid (neutral)
    # Attempt 3: just inside bid/ask (near-taker, may still get maker fill)
    offsets = [
        MAKER_MID_OFFSET,
        MAKER_MID_OFFSET + MAKER_AGGR_STEP,
        MAKER_MID_OFFSET + 2 * MAKER_AGGR_STEP,
    ]

    for attempt, offset in enumerate(offsets, start=1):
        if side == 'buy':
            limit_px = round(mid - offset * mid, 8)  # below mid
        else:
            limit_px = round(mid + offset * mid, 8)  # above mid

        # Clamp to not cross spread (would become taker)
        if side == 'buy':
            limit_px = min(limit_px, bid * 0.9999)  # stay below ask
        else:
            limit_px = max(limit_px, ask * 1.0001)  # stay above bid

        maker_params = {**params, 'execType': 'post_only'}
        t0 = _time.time()
        order_id = None
        try:
            order = exchange.create_order(
                symbol=okx_sym, type='limit', side=side,
                amount=amount, price=limit_px, params=maker_params
            )
            order_id = order['id']
        except Exception as e:
            # post_only rejection (would cross spread) → skip to next attempt
            if 'post_only' in str(e).lower() or '51119' in str(e):
                print(f"         [MAKER] Attempt {attempt} post_only rejected for {symbol} → retry")
                continue
            # Other error → fall through to market
            print(f"         [MAKER] Attempt {attempt} failed for {symbol}: {str(e)[:80]}")
            break

        # Poll for fill up to MAKER_TTL_SECONDS
        filled = False
        while _time.time() - t0 < MAKER_TTL_SECONDS:
            _time.sleep(2)
            try:
                check = exchange.fetch_order(order_id, okx_sym)
                if check['status'] == 'closed':
                    fill_time_s = _time.time() - t0
                    # Determine if maker fill (fee negative or very low)
                    fee_rate = abs(float(check.get('fee', {}).get('cost', 0) or 0)) / max(float(check.get('cost', 1e-10) or 1e-10), 1e-10)
                    was_maker = fee_rate < 0.0003  # taker fee ~0.05%, maker ~0% or rebate
                    _log_execution(symbol, okx_sym, side, tier, attempt, 'maker_limit',
                                   ticker_info, limit_px, check, fill_time_s, was_maker)
                    _log_fill(okx_sym, side, check, ticker_info, 'limit')
                    print(f"         [MAKER] {'✅' if was_maker else '🟡'} "
                          f"{'MAKER' if was_maker else 'TAKER'} fill {symbol} "
                          f"attempt={attempt} t={fill_time_s:.0f}s eff_bps≈{(-1 if was_maker else 5):.0f}")
                    return check
                if check['status'] == 'canceled':
                    break
            except Exception:
                pass

        # Cancel and try next attempt
        if order_id:
            try:
                exchange.cancel_order(order_id, okx_sym)
            except Exception:
                pass
            # Check if filled during cancel race
            try:
                check = exchange.fetch_order(order_id, okx_sym)
                if check['status'] == 'closed':
                    fill_time_s = _time.time() - t0
                    _log_execution(symbol, okx_sym, side, tier, attempt, 'maker_limit_racewin',
                                   ticker_info, limit_px, check, fill_time_s, True)
                    _log_fill(okx_sym, side, check, ticker_info, 'limit')
                    return check
            except Exception:
                pass

        print(f"         [MAKER] Attempt {attempt} unfilled for {symbol} → {'retry' if attempt < len(offsets) else 'market fallback'}")

    # All maker attempts exhausted → market fallback
    t0 = _time.time()
    result = exchange.create_order(symbol=okx_sym, type='market', side=side,
                                   amount=amount, params=params)
    result = _settle_order(exchange, result, okx_sym)
    fill_time_s = _time.time() - t0
    _log_execution(symbol, okx_sym, side, tier, len(offsets) + 1, 'market_fallback',
                   ticker_info, None, result, fill_time_s, False)
    _log_fill(okx_sym, side, result, ticker_info, 'market_fallback')
    print(f"         [MAKER] ⏩ Market fallback for {symbol}")
    return result


def _log_fill(symbol, side, order, ticker_info, order_type):
    """Log per-fill execution details to trading_logs/fills.csv."""
    try:
        fill_price = float(order.get('average', 0) or order.get('price', 0) or 0)
        filled_qty = order.get('filled', '')
        cost = order.get('cost', '')
        mid_price = (ticker_info['bid'] + ticker_info['ask']) / 2 if ticker_info else fill_price
        slippage_bps = ((fill_price / mid_price - 1) * 10000
                        if mid_price > 0 and fill_price > 0 else 0)
        # Invert slippage for sells (paying less is good for sells)
        if side == 'sell':
            slippage_bps = -slippage_bps

        row = (f"{datetime.now(timezone.utc).isoformat()},"
               f"{symbol},{side},{order_type},"
               f"{ticker_info.get('bid', '') if ticker_info else ''},"
               f"{ticker_info.get('ask', '') if ticker_info else ''},"
               f"{mid_price:.8g},{fill_price:.8g},"
               f"{slippage_bps:.2f},"
               f"{filled_qty},"
               f"{cost}\n")

        fill_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'trading_logs', 'fills.csv')
        write_header = not os.path.exists(fill_path)
        with open(fill_path, 'a') as f:
            if write_header:
                f.write("timestamp,symbol,side,type,bid,ask,mid,fill_price,slippage_bps,filled_qty,cost\n")
            f.write(row)
    except Exception:
        pass  # never break trading for logging


def _limit_with_fallback(exchange, symbol, side, amount, ticker_info,
                         params, use_limit=True, timeout=LIMIT_ORDER_WAIT):
    """Place limit order at bid/ask for maker fee; fall back to market if not filled."""
    import time as _time

    if not use_limit or ticker_info is None:
        result = exchange.create_order(symbol=symbol, type='market', side=side,
                                     amount=amount, params=params)
        result = _settle_order(exchange, result, symbol)
        _log_fill(symbol, side, result, ticker_info, 'market')
        return result

    # Limit price: cross the spread slightly for higher fill rate
    # buy at ask - small offset (closer to mid), sell at bid + small offset
    bid = ticker_info['bid']
    ask = ticker_info['ask']
    if side == 'buy':
        limit_price = round(ask * (1 + LIMIT_PRICE_AGGRESSION), 8)  # slightly above ask
    else:
        limit_price = round(bid * (1 - LIMIT_PRICE_AGGRESSION), 8)  # slightly below bid
    if not limit_price or limit_price <= 0:
        result = exchange.create_order(symbol=symbol, type='market', side=side,
                                     amount=amount, params=params)
        result = _settle_order(exchange, result, symbol)
        _log_fill(symbol, side, result, ticker_info, 'market')
        return result

    try:
        order = exchange.create_order(
            symbol=symbol, type='limit', side=side,
            amount=amount, price=limit_price, params=params
        )
        order_id = order['id']

        # Poll for fill
        start = _time.time()
        while _time.time() - start < timeout:
            _time.sleep(1.5)
            try:
                check = exchange.fetch_order(order_id, symbol)
                if check['status'] == 'closed':  # filled
                    _log_fill(symbol, side, check, ticker_info, 'limit')
                    return check
                if check['status'] == 'canceled':
                    break
            except Exception:
                pass

        # Cancel unfilled order
        try:
            exchange.cancel_order(order_id, symbol)
        except Exception:
            pass

        # Check final state (may have filled or partially filled during cancel)
        filled_amount = 0
        try:
            check = exchange.fetch_order(order_id, symbol)
            if check['status'] == 'closed':
                _log_fill(symbol, side, check, ticker_info, 'limit')
                return check
            filled_amount = float(check.get('filled', 0))
        except Exception:
            pass

        # Log partial fill if any
        if filled_amount > 0:
            _log_fill(symbol, side, check, ticker_info, 'limit_partial')

        # Market fallback for remaining amount only
        remaining = amount - filled_amount
        if remaining <= 0:
            # Fully filled during cancel race
            return check
        print(f"         ⏩ Limit {'partially ' if filled_amount > 0 else ''}filled for {symbol}"
              f" ({filled_amount}/{amount}), market for remaining {remaining}")
        result = exchange.create_order(symbol=symbol, type='market', side=side,
                                     amount=remaining, params=params)
        result = _settle_order(exchange, result, symbol)
        _log_fill(symbol, side, result, ticker_info, 'market_fallback')
        return result
    except Exception:
        # Limit order placement failed — market fallback
        result = exchange.create_order(symbol=symbol, type='market', side=side,
                                     amount=amount, params=params)
        result = _settle_order(exchange, result, symbol)
        _log_fill(symbol, side, result, ticker_info, 'market_fallback')
        return result


def rebalance_positions(exchange, target_positions, leverage=1, dry_run=True,
                        use_limit=True):
    """
    Partial rebalance: diff live vs target, execute minimal changes.

    Returns (results, actions) where:
      results: list of execution dicts
      actions: dict with 'closed', 'kept', 'resized', 'opened' sets
               and 'pnl_snapshot' dict of {(symbol, side): upnl}
    """
    import time as _time
    results = []
    actions = {
        'closed': set(), 'kept': set(), 'resized': set(), 'opened': set(),
        'pnl_snapshot': {},
        'resize_details': [],  # (symbol, side, old_notional, new_notional, upnl_before)
    }

    if dry_run or not exchange:
        for pos in target_positions:
            okx_sym = SYMBOLS_TO_OKX.get(pos['symbol'])
            if not okx_sym:
                continue
            side = 'buy' if pos['side'] == 'long' else 'sell'
            print(f"      [DRY] {side.upper():4s} ${pos['usd']:>7.0f} {okx_sym} "
                  f"(score: {pos['score']:+.3f})")
            results.append({**pos, 'status': 'dry_run'})
        return results, actions

    # ── 1. Load markets & fetch live positions ──
    try:
        exchange.load_markets()
    except Exception as e:
        print(f"      ⚠️  load_markets: {e}")
    markets = exchange.markets

    live_map = {}  # (our_symbol, side) -> {notional, contracts, upnl}
    pnl_snapshot = {}  # (clean_symbol, side) -> upnl  (for dashboard)
    try:
        live_positions = exchange.fetch_positions()
        for p in live_positions:
            if float(p.get('contracts', 0)) > 0:
                our_sym = p['symbol'].replace('/USDT:USDT', '/USDT')
                key = (our_sym, p['side'])
                live_map[key] = {
                    'notional': abs(float(p.get('notional', 0))),
                    'contracts': float(p.get('contracts', 0)),
                    'upnl': float(p.get('unrealizedPnl', 0)),
                }
                sym_clean = our_sym.replace('/USDT', '')
                pnl_snapshot[(sym_clean, p['side'])] = float(p.get('unrealizedPnl', 0))
    except Exception as e:
        print(f"      ⚠️  Fetch positions failed: {e}")
        print(f"      ⚠️  Falling back to full close → open")
        close_all(exchange)
        total_usd = sum(p['usd'] for p in target_positions)
        wait_for_margin(exchange, total_usd / leverage)
        results = execute(exchange, target_positions, dry_run=False, leverage=leverage)
        actions['opened'] = {(p['symbol'], p['side']) for p in target_positions}
        return results, actions

    actions['pnl_snapshot'] = pnl_snapshot

    # ── 2. Build target map ──
    target_map = {}
    for pos in target_positions:
        key = (pos['symbol'], pos['side'])
        target_map[key] = pos

    # ── 3. Classify positions ──
    to_close = []    # (key, live_info)
    to_resize = []   # (key, live_info, target_pos, delta_usd)
    to_open = []     # target_pos
    kept = []        # key

    for key, live_info in live_map.items():
        if key in target_map:
            target_usd = target_map[key]['usd']
            live_not = live_info['notional']
            diff_pct = abs(target_usd - live_not) / target_usd if target_usd > 5 else 1.0
            if diff_pct > REBALANCE_THRESHOLD:
                delta_usd = target_usd - live_not
                to_resize.append((key, live_info, target_map[key], delta_usd))
                print(f"      🔄 RESIZE {key[0]:<12s} {key[1]:<5s}: "
                      f"${live_not:>7.0f} → ${target_usd:>7.0f} ({diff_pct:>5.0%})")
            else:
                kept.append(key)
                print(f"      ✊ KEEP   {key[0]:<12s} {key[1]:<5s}: "
                      f"${live_not:>7.0f} ≈ ${target_usd:>7.0f} ({diff_pct:>5.0%})")
        else:
            to_close.append((key, live_info))
            print(f"      🗑️  CLOSE  {key[0]:<12s} {key[1]:<5s}: ${live_info['notional']:>7.0f}")

    for key, pos in target_map.items():
        if key not in live_map:
            to_open.append(pos)
            print(f"      🆕 OPEN   {key[0]:<12s} {key[1]:<5s}: ${pos['usd']:>7.0f}")

    # Summary
    n_orders = len(to_close) + len(to_resize) + len(to_open)
    n_orders_old = len(live_map) + len(target_map)
    savings_pct = (1 - n_orders / max(n_orders_old, 1)) * 100
    print(f"\n      📊 Rebalance: {len(kept)} keep, {len(to_resize)} resize, "
          f"{len(to_close)} close, {len(to_open)} open")
    print(f"      💰 Orders: {n_orders} vs {n_orders_old} full rebalance "
          f"(saved {savings_pct:.0f}%)")

    actions['kept'] = set(kept)

    # ── 4. Close removed positions ──
    if to_close:
        print(f"\n      📤 Closing {len(to_close)} removed positions...")
    for key, live_info in to_close:
        sym, side = key
        close_side = 'sell' if side == 'long' else 'buy'
        okx_sym = SYMBOLS_TO_OKX.get(sym)
        if not okx_sym:
            continue
        try:
            _, ticker_info = _convert_usd_to_contracts(exchange, markets, okx_sym, 0)
            if sym in _TIER1_SYMS:
                order = _maker_first_limit(
                    exchange, sym, okx_sym, close_side, live_info['contracts'],
                    ticker_info,
                    params={'tdMode': 'isolated', 'posSide': 'net', 'reduceOnly': True}
                )
            else:
                order = _limit_with_fallback(
                    exchange, okx_sym, close_side, live_info['contracts'], ticker_info,
                    params={'tdMode': 'isolated', 'posSide': 'net', 'reduceOnly': True},
                    use_limit=use_limit
                )
            if _is_order_filled(order):
                print(f"      ✅ Closed {side} {sym} → {order.get('id', '?')}")
                actions['closed'].add(key)
                results.append({'symbol': sym, 'side': side, 'usd': live_info['notional'],
                                'status': 'closed', 'order_id': order.get('id')})
            else:
                print(f"      ⚠️  Close {sym} NOT FILLED (id={order.get('id', '?')})")
                results.append({'symbol': sym, 'side': side, 'usd': live_info['notional'],
                                'status': 'error', 'error': 'not_filled'})
        except Exception as e:
            print(f"      ⚠️  Close {sym}: {str(e)[:120]}")
            results.append({'symbol': sym, 'side': side, 'usd': live_info['notional'],
                            'status': 'error', 'error': str(e)})

    # ── 5. Wait for margin if we freed some and need to open/resize ──
    if to_close and (to_open or any(d > 0 for _, _, _, d in to_resize)):
        need_usd = sum(pos['usd'] for pos in to_open) / leverage
        need_usd += sum(max(d, 0) for _, _, _, d in to_resize) / leverage
        if need_usd > 0:
            wait_for_margin(exchange, need_usd)

    # ── 6. Resize positions ──
    if to_resize:
        print(f"\n      🔄 Resizing {len(to_resize)} positions...")
    for key, live_info, target_pos, delta_usd in to_resize:
        sym, side = key
        okx_sym = SYMBOLS_TO_OKX.get(sym)
        if not okx_sym:
            continue
        try:
            abs_delta = abs(delta_usd)
            contracts_delta, ticker_info = _convert_usd_to_contracts(
                exchange, markets, okx_sym, abs_delta)
            if contracts_delta == 0 or contracts_delta is None:
                print(f"      ⏭️  Skip resize {sym}: delta too small")
                actions['kept'].add(key)
                continue
            if delta_usd > 0:
                order_side = 'buy' if side == 'long' else 'sell'
                params = {'tdMode': 'isolated', 'posSide': 'net'}
                tag = '↑'
            else:
                order_side = 'sell' if side == 'long' else 'buy'
                params = {'tdMode': 'isolated', 'posSide': 'net', 'reduceOnly': True}
                tag = '↓'
            order = _maker_first_limit(
                exchange, sym, okx_sym, order_side, contracts_delta,
                ticker_info, params=params
            ) if sym in _TIER1_SYMS else _limit_with_fallback(
                exchange, okx_sym, order_side, contracts_delta, ticker_info,
                params=params, use_limit=use_limit
            )
            if _is_order_filled(order):
                print(f"      ✅ {tag} {side} {sym}: ${abs_delta:.0f} ({contracts_delta} cts) → {order.get('id', '?')}")
                actions['resized'].add(key)
                actions['resize_details'].append({
                    'symbol': sym.replace('/USDT', ''),
                    'side': side,
                    'old_notional': live_info['notional'],
                    'new_notional': target_pos['usd'],
                    'upnl_before': live_info['upnl'],
                })
                results.append({**target_pos, 'status': 'resized', 'order_id': order.get('id')})
            else:
                print(f"      ⚠️  Resize {sym} NOT FILLED (id={order.get('id', '?')})")
                actions['kept'].add(key)  # treat as kept (old size remains)
        except Exception as e:
            print(f"      ⚠️  Resize {sym}: {str(e)[:120]}")
            results.append({**target_pos, 'status': 'error', 'error': str(e)})

    # ── 7. Open new positions (with retry sweeps) ──
    if to_open:
        print(f"\n      📥 Opening {len(to_open)} new positions...")
    failed = []
    for pos in to_open:
        okx_sym = SYMBOLS_TO_OKX.get(pos['symbol'])
        if not okx_sym:
            continue
        order_side = 'buy' if pos['side'] == 'long' else 'sell'
        try:
            exchange.set_leverage(leverage, okx_sym, params={'mgnMode': 'isolated'})
        except Exception:
            pass
        try:
            contracts, ticker_info = _convert_usd_to_contracts(
                exchange, markets, okx_sym, pos['usd'])
            order = _maker_first_limit(
                exchange, pos['symbol'], okx_sym, order_side, contracts,
                ticker_info, params={'tdMode': 'isolated', 'posSide': 'net'}
            ) if pos['symbol'] in _TIER1_SYMS else _limit_with_fallback(
                exchange, okx_sym, order_side, contracts, ticker_info,
                params={'tdMode': 'isolated', 'posSide': 'net'},
                use_limit=use_limit
            )
            if _is_order_filled(order):
                print(f"      ✅ {order_side.upper():4s} ${pos['usd']:>7.0f} {okx_sym} "
                      f"({contracts} cts) → {order.get('id', '?')}")
                actions['opened'].add((pos['symbol'], pos['side']))
                results.append({**pos, 'status': 'filled', 'order_id': order.get('id')})
            else:
                print(f"      ⚠️  {order_side.upper():4s} ${pos['usd']:>7.0f} {okx_sym} "
                      f"NOT FILLED (id={order.get('id', '?')})")
                failed.append(pos)  # retry as if margin issue
        except Exception as e:
            if '51008' in str(e):
                failed.append(pos)
            else:
                print(f"      ❌ {order_side.upper():4s} ${pos['usd']:>7.0f} {okx_sym} → {e}")
                results.append({**pos, 'status': 'error', 'error': str(e)})

    # Retry sweeps for margin (51008) and unfilled orders
    for sweep in range(2):
        if not failed:
            break
        print(f"      ⏳ {len(failed)} orders need margin — waiting 10s (sweep {sweep + 1})...")
        _time.sleep(10)
        still_failed = []
        for pos in failed:
            okx_sym = SYMBOLS_TO_OKX.get(pos['symbol'])
            order_side = 'buy' if pos['side'] == 'long' else 'sell'
            try:
                contracts, ticker_info = _convert_usd_to_contracts(
                    exchange, markets, okx_sym, pos['usd'])
                order = _maker_first_limit(
                    exchange, pos['symbol'], okx_sym, order_side, contracts,
                    ticker_info, params={'tdMode': 'isolated', 'posSide': 'net'}
                ) if pos['symbol'] in _TIER1_SYMS else _limit_with_fallback(
                    exchange, okx_sym, order_side, contracts, ticker_info,
                    params={'tdMode': 'isolated', 'posSide': 'net'},
                    use_limit=use_limit
                )
                if _is_order_filled(order):
                    print(f"      🔄 {order_side.upper():4s} ${pos['usd']:>7.0f} {okx_sym} → {order.get('id', '?')}")
                    actions['opened'].add((pos['symbol'], pos['side']))
                    results.append({**pos, 'status': 'filled', 'order_id': order.get('id')})
                else:
                    print(f"      ⚠️  {order_side.upper():4s} ${pos['usd']:>7.0f} {okx_sym} "
                          f"NOT FILLED on retry (id={order.get('id', '?')})")
                    still_failed.append(pos)
            except Exception as e:
                if '51008' in str(e):
                    still_failed.append(pos)
                else:
                    print(f"      ❌ {order_side.upper():4s} ${pos['usd']:>7.0f} {okx_sym} → {e}")
                    results.append({**pos, 'status': 'error', 'error': str(e)})
        failed = still_failed

    for pos in failed:
        okx_sym = SYMBOLS_TO_OKX.get(pos['symbol'], pos['symbol'])
        print(f"      ❌ ${pos['usd']:>7.0f} {okx_sym} → insufficient balance after retries")
        results.append({**pos, 'status': 'error', 'error': '51008 after retries'})

    return results, actions


# ============================================================
# STATE MANAGEMENT
# ============================================================

def load_state(path):
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            print(f"   ⚠️  Corrupt state file {path}, starting fresh")
    return {}


def save_state(state, path):
    # Atomic write: write to tmp, then rename (prevents corruption + partial writes)
    tmp_path = path + '.tmp'
    try:
        with open(tmp_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp_path, path)
    except Exception as e:
        print(f"   ⚠️  save_state failed: {e}")
        # Try direct write as fallback
        try:
            with open(path, 'w') as f:
                json.dump(state, f, indent=2, default=str)
        except Exception:
            pass


# ============================================================
# POSITION LEDGER & TRADES.CSV
# ============================================================

def _ledger_key(symbol, side):
    """Canonical key for position ledger: 'BTC|long'."""
    sym = symbol.replace('/USDT', '').replace(':USDT', '')
    return f"{sym}|{side}"


def _update_position_ledger(state, exchange, resize_details=None):
    """Sync position ledger with actual exchange positions.

    Ledger format in state['position_ledger']:
      { "BTC|long": {"entry_price": 65000.0, "opened_at": "...", "notional": 500,
                      "contracts": 0.008, "side": "long", "symbol": "BTC"}, ... }

    Returns (realized_pnl_list, current_ledger) where realized_pnl_list contains
    dicts for each position that was closed since last call.
    """
    ledger = state.get('position_ledger', {})
    realized = []
    fully_closed_keys = set()

    if not exchange:
        return realized, ledger

    try:
        live_positions = exchange.fetch_positions()
    except Exception:
        return realized, ledger

    # Build current position map from exchange
    live_map = {}  # key -> {entry_price, notional, contracts, upnl, side, symbol}
    for p in live_positions:
        if float(p.get('contracts', 0)) > 0:
            sym = p['symbol'].replace('/USDT:USDT', '/USDT')
            sym_clean = sym.replace('/USDT', '')
            key = _ledger_key(sym, p['side'])
            live_map[key] = {
                'entry_price': float(p.get('entryPrice', 0)),
                'notional': abs(float(p.get('notional', 0))),
                'contracts': float(p.get('contracts', 0)),
                'upnl': float(p.get('unrealizedPnl', 0)),
                'mark_price': float(p.get('markPrice', 0)),
                'side': p['side'],
                'symbol': sym_clean,
            }

    # Detect closed positions (in ledger but not in live)
    now_iso = datetime.now(timezone.utc).isoformat()
    for key, entry in list(ledger.items()):
        if key not in live_map:
            # Position was fully closed — record realized PnL
            notional = entry.get('notional', 0)
            last_upnl = entry.get('last_upnl', 0)
            pnl_pct = (last_upnl / notional * 100) if notional > 0 else 0

            hold_hours = 0
            opened_at = entry.get('opened_at', '')
            if opened_at:
                try:
                    opened_dt = datetime.fromisoformat(opened_at.replace('Z', '+00:00'))
                    hold_hours = (datetime.now(timezone.utc) - opened_dt).total_seconds() / 3600
                except Exception:
                    pass

            realized.append({
                'symbol': entry.get('symbol', key.split('|')[0]),
                'side': entry.get('side', ''),
                'usd': round(notional, 2),
                'entry_price': entry.get('entry_price', 0),
                'pnl_usd': round(last_upnl, 4),
                'pnl_pct': round(pnl_pct, 2),
                'opened_at': opened_at,
                'closed_at': now_iso,
                'hold_hours': round(hold_hours, 1),
            })
            fully_closed_keys.add(key)
            del ledger[key]

    # Detect partial closes (resize down): position still exists but smaller
    if resize_details:
        for rd in resize_details:
            lkey = _ledger_key(rd['symbol'], rd['side'])
            if lkey in fully_closed_keys or lkey not in live_map:
                continue
            old_n = rd['old_notional']
            new_n = live_map[lkey].get('notional', rd['new_notional'])
            if new_n >= old_n or old_n <= 0:
                continue  # resize up or zero — no realized PnL
            closed_frac = (old_n - new_n) / old_n
            upnl_before = rd.get('upnl_before', 0)
            realized_upnl = upnl_before * closed_frac
            pnl_pct = (realized_upnl / (old_n * closed_frac) * 100) if (old_n * closed_frac) > 0 else 0
            entry = ledger.get(lkey, {})
            opened_at = entry.get('opened_at', '')
            hold_hours = 0
            if opened_at:
                try:
                    opened_dt = datetime.fromisoformat(opened_at.replace('Z', '+00:00'))
                    hold_hours = (datetime.now(timezone.utc) - opened_dt).total_seconds() / 3600
                except Exception:
                    pass
            realized.append({
                'symbol': rd['symbol'],
                'side': rd['side'],
                'usd': round(old_n * closed_frac, 2),
                'entry_price': entry.get('entry_price', 0),
                'pnl_usd': round(realized_upnl, 4),
                'pnl_pct': round(pnl_pct, 2),
                'opened_at': opened_at,
                'closed_at': now_iso,
                'hold_hours': round(hold_hours, 1),
            })

    # Update / add entries for live positions
    for key, pos in live_map.items():
        if key in ledger:
            # Update mark data; keep original entry time
            ledger[key]['notional'] = pos['notional']
            ledger[key]['contracts'] = pos['contracts']
            ledger[key]['last_upnl'] = pos['upnl']
            ledger[key]['mark_price'] = pos['mark_price']
            # If exchange entry_price changed (partial fill updated avg), sync it
            if pos['entry_price'] > 0:
                ledger[key]['entry_price'] = pos['entry_price']
        else:
            # New position
            ledger[key] = {
                'entry_price': pos['entry_price'],
                'opened_at': now_iso,
                'notional': pos['notional'],
                'contracts': pos['contracts'],
                'last_upnl': pos['upnl'],
                'mark_price': pos['mark_price'],
                'side': pos['side'],
                'symbol': pos['symbol'],
            }

    state['position_ledger'] = ledger
    return realized, ledger


def _write_trades_csv(realized_trades, root):
    """Append closed trades to trading_logs/trades.csv."""
    if not realized_trades:
        return
    csv_path = os.path.join(root, 'trading_logs', 'trades.csv')
    write_header = not os.path.exists(csv_path)
    try:
        with open(csv_path, 'a') as f:
            if write_header:
                f.write("exit_time,symbol,side,usd,entry_price,pnl_usd,pnl_pct,"
                        "opened_at,hold_hours,status\n")
            for t in realized_trades:
                f.write(f"{t['closed_at']},{t['symbol']},{t['side']},"
                        f"{t['usd']},{t['entry_price']},"
                        f"{t['pnl_usd']},{t['pnl_pct']},"
                        f"{t['opened_at']},{t['hold_hours']},closed\n")
        print(f"   📝 Recorded {len(realized_trades)} closed trades to trades.csv")
    except Exception as e:
        print(f"   ⚠️  trades.csv write error: {e}")


# ============================================================
# DASHBOARD JSON UPDATE
# ============================================================

def update_dashboard(exchange, positions, signals, state, results, root,
                     capital, leverage, mode, next_rebal_str='',
                     rebal_hours=DEFAULT_REBAL_HOURS):
    """Write dashboard/data/dashboard.json for the web UI."""
    dashboard_dir = os.path.join(root, 'dashboard', 'data')
    os.makedirs(dashboard_dir, exist_ok=True)
    now = datetime.now(timezone.utc)

    # ── Model info from signals ──
    model_info = None
    if signals is not None and hasattr(signals, 'attrs'):
        model_info = signals.attrs.get('model_info')
    if model_info:
        state['model_info'] = model_info
    else:
        model_info = state.get('model_info')

    # Fetch live positions & balance from exchange
    live_positions = []
    equity = state.get('equity', capital)
    free_usdt = 0
    margin_used = 0
    total_upnl = 0
    exchange_ok = False

    if exchange:
        try:
            bal = exchange.fetch_balance()
            total_usdt = float(bal.get('USDT', {}).get('total', 0))
            free_usdt = float(bal.get('USDT', {}).get('free', 0))

            exch_positions = exchange.fetch_positions()
            open_pos = [p for p in exch_positions if float(p.get('contracts', 0)) > 0]

            for p in open_pos:
                notional = abs(float(p.get('notional', 0)))
                upnl = float(p.get('unrealizedPnl', 0))
                entry_price = float(p.get('entryPrice', 0))
                total_upnl += upnl
                margin_used += notional / leverage if leverage else notional
                live_positions.append({
                    'symbol': p['symbol'].replace('/USDT:USDT', ''),
                    'side': p['side'],
                    'notional': notional,
                    'upnl': round(upnl, 2),
                    'upnl_pct': round(upnl / notional * 100, 2) if notional else 0,
                    'entryPrice': entry_price,
                    'markPrice': float(p.get('markPrice', 0)),
                    'score': 0,
                    'confidence': 0,
                })

            equity = total_usdt + total_upnl
            exchange_ok = True
        except Exception as e:
            print(f"   ⚠️  Dashboard OKX fetch: {e}")
            exchange_ok = False

    # Match scores from signals to live positions
    if signals is not None and len(signals) > 0:
        score_map = dict(zip(signals['symbol'].str.replace('/USDT', ''),
                             zip(signals['score'], signals.get('confidence', pd.Series()))))
        for pos in live_positions:
            sym = pos['symbol']
            if sym in score_map:
                pos['score'] = round(float(score_map[sym][0]), 4)
                pos['confidence'] = round(float(score_map[sym][1]), 3) if pd.notna(score_map[sym][1]) else 0

    # ── Detect closed trades in real-time ──
    # ONLY compare if we successfully fetched exchange positions;
    # otherwise a transient OKX error would mark all trades as closed with PnL=0
    dash_trades = state.get('dash_trades', [])
    if exchange_ok and live_positions:
        live_syms = {(p['symbol'], p['side']) for p in live_positions}
        # Build UPnL lookup from live positions for PnL at close
        upnl_map = {(p['symbol'], p['side']): p.get('upnl', 0) for p in live_positions}
        # Also look up PnL from position ledger (has last_upnl for recently closed)
        ledger = state.get('position_ledger', {})
        for t in dash_trades:
            if t.get('closed') is None and (t['symbol'], t['side']) not in live_syms:
                # Position no longer exists on exchange — mark as closed
                t['closed'] = now.isoformat()
                if t.get('pnl', 0) == 0:
                    # Try live upnl first, then ledger's last_upnl
                    pnl = upnl_map.get((t['symbol'], t['side']), 0)
                    if pnl == 0:
                        lkey = _ledger_key(t['symbol'], t['side'])
                        lentry = ledger.get(lkey)
                        if lentry:
                            pnl = lentry.get('last_upnl', 0)
                    t['pnl'] = round(pnl, 2)
    state['dash_trades'] = dash_trades

    # Build signals list for dashboard
    dash_signals = []
    if signals is not None:
        for _, row in signals.iterrows():
            dash_signals.append({
                'symbol': row['symbol'].replace('/USDT', ''),
                'score': round(float(row['score']), 4),
                'confidence': round(float(row.get('confidence', 0)), 3),
            })

    # Equity history: append to existing
    eq_history = state.get('equity_history', [])
    eq_history.append({
        'timestamp': now.isoformat(),
        'equity': round(equity, 2),
        'pnl': round(equity - capital, 2),
        'dd_pct': round(equity / state.get('peak', capital) - 1, 4),
    })
    # Keep last 2000 points
    eq_history = eq_history[-2000:]
    state['equity_history'] = eq_history

    # Trades from results
    dash_trades = state.get('dash_trades', [])
    if results:
        for r in results:
            if r.get('status') in ('filled', 'dry_run'):
                dash_trades.append({
                    'symbol': r.get('symbol', '?').replace('/USDT', ''),
                    'side': r.get('side', '?'),
                    'usd': r.get('usd', 0),
                    'score': r.get('score', 0),
                    'pnl': 0,  # filled at open; PnL calculated at close
                    'closed': None,
                    'opened': now.isoformat(),
                })
    # Keep last 100 trades
    dash_trades = dash_trades[-100:]
    state['dash_trades'] = dash_trades

    # Win rate from closed trades (actual realized PnL per trade)
    closed_trades = [t for t in dash_trades if t.get('closed') is not None]
    n_wins = sum(1 for t in closed_trades if t.get('pnl', 0) > 0)
    win_rate = n_wins / len(closed_trades) if closed_trades else 0
    max_dd = min((e.get('dd_pct', 0) for e in eq_history), default=0)

    # Build models string from actual loaded groups
    if model_info:
        n_models = model_info['n_models']
        parts = []
        label_map = {'v6': 'LGB v6', 'v7': 'LGB v7', 'cb': 'CB', 'xgb': 'XGB'}
        for lbl, cnt in model_info['groups'].items():
            parts.append(f'{label_map.get(lbl, lbl)}×{cnt}')
        models_str = f"{n_models} ({' + '.join(parts)})"
    else:
        n_models = 20
        models_str = f'{n_models} (LGB v6×5 + v7×5 + CB×5 + XGB×5)'

    dashboard_data = {
        'updated': now.isoformat(),
        'mode': mode,
        'capital': capital,
        'equity': round(equity, 2),
        'leverage': leverage,
        'margin_used': round(margin_used, 2),
        'free_usdt': round(free_usdt, 2),
        'win_rate': round(win_rate, 3),
        'total_trades': len(dash_trades),
        'max_dd': round(max_dd, 4),
        'positions': live_positions,
        'orders': [],
        'trades': dash_trades[-30:],
        'signals': dash_signals,
        'equity_history': eq_history,
        'models': models_str,
        'rebal_hours': rebal_hours,
        'min_score': 0,
        'edge_boost': True,
        'cycle': state.get('n_cycles', 0),
        'next_rebal': next_rebal_str,
    }

    path = os.path.join(dashboard_dir, 'dashboard.json')
    with open(path, 'w') as f:
        json.dump(dashboard_data, f, indent=2, default=str)
    print(f"   📊 Dashboard updated: {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Production Trading System')
    parser.add_argument('--mode', choices=['signal', 'paper', 'live'], default='signal')
    parser.add_argument('--capital', type=float, default=1000.0)
    parser.add_argument('--loop', action='store_true')
    parser.add_argument('--hours', type=int, default=800, help='Hours of history')
    parser.add_argument('--vol-target', type=float, default=None)
    parser.add_argument('--kelly', type=float, default=None)
    parser.add_argument('--config', type=str, default=None, help='Path to risk config JSON')
    parser.add_argument('--leverage', type=int, default=1, help='Exchange leverage (1-10)')
    parser.add_argument('--no-deriv-gate', action='store_true', help='Disable derivative risk gate')
    parser.add_argument('--no-meta', action='store_true', help='Disable meta-model (use simple mean ensemble)')
    parser.add_argument('--no-xgb', action='store_true', help='Exclude XGBoost from ensemble')
    parser.add_argument('--cb-only', action='store_true',
                        help='CatBoost solo mode: skip LGB and XGB models')
    parser.add_argument('--vol-size', action='store_true',
                        help='Inverse-vol position sizing: weight ∝ edge / coin_vol')
    parser.add_argument('--min-zscore', type=float, default=0.0,
                        help='Min |z-score| for position entry (e.g. 0.5). '
                             'Filters out weak signals from portfolio.')
    parser.add_argument('--ridge', action='store_true',
                        help='Use Ridge mean-reversion model instead of GBDT ensemble')
    parser.add_argument('--lgb', action='store_true',
                        help='Use LightGBM ensemble (R9B: Sh=4.29, Wr=-1.2%%, WM=12/13). '
                             'Requires results_lgb_prod/ from train_lgb_prod.py. No EMA.')
    parser.add_argument('--cls', action='store_true',
                        help='Use LGB+XGB classification ensemble (R25: Sh=3.36, Wr=-5.7%%). '
                             'Requires results_cls_prod/ from train_cls_prod.py. '
                             'Simple 5L/3S, no risk stack overlays.')
    parser.add_argument('--rebal-hours', type=int, default=DEFAULT_REBAL_HOURS,
                        choices=[1, 2, 3, 4, 6, 8, 12, 24],
                        help='Live rebalance cadence in hours. Use 24 to match recent sims.')
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
    risk_cfg['leverage'] = args.leverage
    if args.min_zscore > 0:
        risk_cfg['min_zscore'] = args.min_zscore
    risk_cfg['_demo_mode'] = (args.mode == 'paper')

    # CLS mode: 6L/3S, no risk stack overlays (R26 winner: F-6L3S-dt0.7)
    if args.cls:
        risk_cfg['n_long'] = 6
        risk_cfg['n_short'] = 3
        risk_cfg['kelly_frac'] = 1.0       # full allocation (no kelly cut)
        risk_cfg['min_zscore'] = 0.0        # no filtering
        risk_cfg['confidence_threshold'] = 0.0
        risk_cfg['_cls_mode'] = True        # disable vol_scale in construct_portfolio
        print(f"   📋 CLS mode: 6L/3S, kelly=1.0, no min_zscore, no vol_scale")

    # Load trading state
    state_path = os.path.join(log_dir, 'trading_state.json')
    state = load_state(state_path)
    if 'equity' not in state:
        state['equity'] = args.capital
        state['peak'] = args.capital
        state['recent_rets'] = []
    if 'prev_equity' not in state:
        state['prev_equity'] = state.get('equity', args.capital)

    print("=" * 70)
    print(f"  PRODUCTION TRADING — {args.mode.upper()}")
    print(f"  Capital: ${args.capital:,.0f}")
    print(f"  Risk: kelly={risk_cfg['kelly_frac']:.0%}, "
          f"vol_target={risk_cfg['vol_target']*100:.1f}%, "
          f"DD_stop={risk_cfg['dd_stop']*100:.0f}%, "
          f"leverage={risk_cfg['leverage']}x, "
          f"min_zscore={risk_cfg.get('min_zscore', 0.0)}")
    print("=" * 70)

    # Init exchange
    exchange = None
    if args.mode in ('paper', 'live'):
        exchange = init_exchange(args.mode)

    # Init Telegram bot
    bot = create_bot()
    if bot.enabled:
        bot.alert_startup(args.mode, args.capital, risk_cfg)

    def run_cycle():
        now = datetime.now(timezone.utc)
        print(f"\n{'─' * 70}")
        print(f"  🕐 {now.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"{'─' * 70}")

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

        # 2b. Enrich with full pipeline features (matches training pipeline)
        from run_pipeline_v6 import (
            add_multi_horizon_targets, add_cross_asset_features,
            add_market_mode_features, add_liquidity_features,
            add_advanced_regime_features,
            add_derivatives_features, add_sentiment_features,
        )

        # Drop overlap columns that build_features() created partially;
        # pipeline functions will recreate them with correct formulas
        _overlap_prefixes = ('btc_close', 'eth_close',
            'btc_ret_', 'eth_ret_', 'btc_vol_24h', 'btc_ma', 'btc_rolling_high',
            'market_dispersion', 'ret_vs_btc', 'breadth_pct_positive',
            'regime_btc_above_ma720', 'regime_btc_dd_720', 'regime_btc_not_crashed',
            'fng_',
            'reversal_', 'vol_surge_', 'btc_beta_')
        _overlap_cols = [c for c in df.columns if c.startswith(_overlap_prefixes)]
        if _overlap_cols:
            df.drop(columns=_overlap_cols, inplace=True, errors='ignore')
            print(f"   Dropped {len(_overlap_cols)} overlapping cols from build_features")

        print("   🔧 Enriching: targets, cross-asset, regime, 12h, sentiment, derivatives...")
        df = add_multi_horizon_targets(df)
        df = add_cross_asset_features(df)
        df = add_market_mode_features(df)
        df = add_liquidity_features(df)
        df = add_advanced_regime_features(df)
        df = add_12h_features(df)
        from run_pipeline_v6 import add_calendar_features
        df = add_calendar_features(df)
        df = add_sentiment_features(df, root, news_mode='all')

        # Patch derivatives parquet with real-time data before feature build
        try:
            from src.data.fetch_realtime_derivatives import patch_metrics_realtime
            patch_metrics_realtime(root)
        except Exception as e:
            print(f"   ⚠️  Real-time deriv patch failed: {e}")

        df = add_derivatives_features(df, root)

        from run_pipeline_xgboost import add_news_interaction_features
        df = add_news_interaction_features(df)

        # Ridge/LGB-specific features (residuals, mom_z, dist_from_high, OI renames)
        if args.ridge or args.lgb or args.cls:
            df = add_ridge_features(df)

        # CLS: add model-specific features BEFORE ranking (so they get ranked like in sim)
        if args.cls:
            df = add_cls_features(df, root)
            # Filter to SYM_35 for CLS (sim used 35 symbols; ranking on 49 changes distribution)
            before_n = df['symbol'].nunique()
            df = df[df['symbol'].isin(CLS_SYMBOLS)].copy()
            after_n = df['symbol'].nunique()
            if before_n != after_n:
                print(f"   📊 CLS symbol filter: {before_n} → {after_n} symbols")

        feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS
                     and not c.startswith('target_')
                     and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
        print(f"   Features after enrichment: {len(feat_cols)}")

        # Feature health diagnostic (latest snapshot)
        latest_snap = df.groupby('symbol').last()
        zero_cols = [c for c in feat_cols if (latest_snap[c] == 0).all()]
        nan_cols = [c for c in feat_cols if latest_snap[c].isna().all()]
        key_groups = {
            'cross-asset': [c for c in feat_cols if c.startswith(('btc_', 'eth_'))],
            'regime': [c for c in feat_cols if c.startswith('regime_')],
            'sentiment': [c for c in feat_cols if c.startswith(('news_', 'fng_', 'market_news', 'political_'))],
            'derivatives': [c for c in feat_cols if c.startswith(('oi_', 'taker_', 'funding_', 'basis_', 'ls_'))],
            'dvol': [c for c in feat_cols if c.startswith('dvol_')],
        }
        print(f"   🔍 Feature health: {len(zero_cols)} all-zero, {len(nan_cols)} all-NaN of {len(feat_cols)}")
        for gname, gcols in key_groups.items():
            if gcols:
                n_live = sum(1 for c in gcols if c not in zero_cols)
                print(f"      {gname}: {n_live}/{len(gcols)} non-zero")
        if zero_cols:
            print(f"      zero features: {zero_cols[:20]}")

        # 3. Cross-sectional rank (after all enrichment)
        df = cross_sectional_rank(df, feat_cols)

        # Clean infinities & NaN
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[feat_cols] = df[feat_cols].fillna(0)

        # 4. Generate signal
        print(f"\n📡 Generating signal...")
        nonlocal last_signals
        if args.cls:
            signals = generate_signal_cls(df, root)
        elif args.lgb:
            signals = generate_signal_lgb_cs(df, root)
        elif args.ridge:
            signals = generate_signal_ridge(df, root)
        else:
            signals = generate_signal(df, feat_cols, root,
                                         use_meta=not args.no_meta,
                                         use_deriv_gate=not args.no_deriv_gate,
                                         use_xgb=not args.no_xgb,
                                         cb_only=args.cb_only)
        if signals is None or len(signals) == 0:
            print("   ❌ No signals")
            return
        last_signals = signals

        # R7: Signal EMA(2) smoothing — only for Ridge (LGB/CLS=None is optimal)
        if args.ridge and not args.lgb and not args.cls:
            prev_scores = state.get('prev_signal_scores', {})
            if prev_scores:
                alpha_ema = 2.0 / (2 + 1)  # span=2
                for idx, row in signals.iterrows():
                    sym = row['symbol']
                    if sym in prev_scores:
                        signals.at[idx, 'score'] = alpha_ema * row['score'] + (1 - alpha_ema) * prev_scores[sym]
                print(f"   📊 Signal EMA(2): smoothed {len(prev_scores)} symbols")
            state['prev_signal_scores'] = dict(zip(signals['symbol'], signals['score']))

        # 5. Portfolio — trade from FULL equity (compound growth)
        # Refresh equity from exchange before sizing
        trading_capital = state.get('equity', args.capital)
        if exchange:
            try:
                bal = exchange.fetch_balance()
                total_usdt = float(bal.get('USDT', {}).get('total', 0))
                exch_pos = exchange.fetch_positions()
                upnl = sum(float(p.get('unrealizedPnl', 0))
                           for p in exch_pos if float(p.get('contracts', 0)) > 0)
                trading_capital = total_usdt + upnl
                state['equity'] = trading_capital
                state['peak'] = max(state.get('peak', args.capital), trading_capital)
            except Exception as e:
                print(f"   ⚠️  Balance refresh failed: {e}")

        # 5a. Compute per-coin realized vol for inverse-vol sizing
        coin_vol = None
        if args.vol_size:
            coin_vol = {}
            for sym in df['symbol'].unique():
                sym_df = df[df['symbol'] == sym].sort_values('timestamp')
                rets = sym_df['close'].pct_change(1).dropna()
                if len(rets) >= 12:
                    coin_vol[sym] = float(rets.tail(24).std())
            if coin_vol:
                med_v = np.median(list(coin_vol.values()))
                print(f"   📊 Vol-size: {len(coin_vol)} coins, median σ={med_v:.4f}")

        # R7/R9B: Extract regime data from signal
        regime_data = None
        if hasattr(signals, 'attrs'):
            regime_data = signals.attrs.get('regime_data')

        print(f"\n💼 Portfolio construction (equity=${trading_capital:,.0f})...")
        positions = construct_portfolio(signals, trading_capital, risk_cfg, state,
                                          leverage=risk_cfg['leverage'],
                                          coin_vol=coin_vol,
                                          regime_data=regime_data)

        if not positions:
            print("   (no positions this cycle)")
        else:
            print(f"\n   {'Symbol':<15} {'Side':<6} {'USD':>8} {'Score':>8}")
            print(f"   {'─' * 40}")
            for pos in positions:
                print(f"   {pos['symbol']:<15} {pos['side']:<6} ${pos['usd']:>7.0f} "
                      f"{pos['score']:>+8.3f}")

        # 6. Execute (partial rebalance + limit orders)
        resize_details = []  # populated by rebalance_positions if any resizes
        if args.mode == 'signal':
            results = execute(None, positions, dry_run=True, leverage=risk_cfg['leverage'])
        elif not positions and state.get('stopped'):
            # DD stop triggered — close all live positions to limit further loss
            print(f"\n🛑 DD stop: closing all positions...")
            results, actions = rebalance_positions(
                exchange, [], leverage=risk_cfg['leverage'],
                dry_run=False, use_limit=True
            )
            pnl_snapshot = actions.get('pnl_snapshot', {})
            # Mark closed dash_trades with PnL
            dash_trades = state.get('dash_trades', [])
            closed_keys = {(s.replace('/USDT', ''), side)
                           for s, side in actions.get('closed', set())}
            for t in dash_trades:
                key = (t['symbol'], t['side'])
                if t.get('closed') is None and key in closed_keys:
                    t['pnl'] = round(pnl_snapshot.get(key, 0), 2)
                    t['closed'] = now.isoformat()
            state['dash_trades'] = dash_trades
        elif not positions:
            results = execute(None, positions, dry_run=True, leverage=risk_cfg['leverage'])
        else:
            print(f"\n🔄 Partial rebalance (threshold={REBALANCE_THRESHOLD:.0%})...")
            results, actions = rebalance_positions(
                exchange, positions, leverage=risk_cfg['leverage'],
                dry_run=False, use_limit=True
            )

            # PnL snapshot was captured inside rebalance_positions
            pnl_snapshot = actions.get('pnl_snapshot', {})
            resize_details = actions.get('resize_details', [])

            # Mark only closed positions in dash_trades (before ledger sync)
            dash_trades = state.get('dash_trades', [])
            closed_keys = {(s.replace('/USDT', ''), side)
                           for s, side in actions.get('closed', set())}
            for t in dash_trades:
                key = (t['symbol'], t['side'])
                if t.get('closed') is None and key in closed_keys:
                    t['pnl'] = round(pnl_snapshot.get(key, 0), 2)
                    t['closed'] = now.isoformat()
            state['dash_trades'] = dash_trades

        # ── Position ledger sync & trades.csv ──
        if exchange and args.mode in ('paper', 'live'):
            # Pre-seed ledger with pre-close UPnL snapshot (if available)
            # so closed positions get accurate PnL (not stale from last cycle)
            pnl_pre = locals().get('pnl_snapshot', {}) or {}
            if pnl_pre:
                ledger_pre = state.get('position_ledger', {})
                for (sym_clean, side), upnl in pnl_pre.items():
                    lkey = _ledger_key(sym_clean, side)
                    if lkey in ledger_pre:
                        ledger_pre[lkey]['last_upnl'] = upnl
                state['position_ledger'] = ledger_pre

            realized_trades, ledger = _update_position_ledger(state, exchange, resize_details=resize_details)
            if realized_trades:
                _write_trades_csv(realized_trades, root)
                # Use realized PnL from actual closes
                realized_pnl = sum(t['pnl_usd'] for t in realized_trades)
            else:
                realized_pnl = 0

            # Cycle PnL: realized from closes + UPnL delta on open positions
            current_upnl = sum(pos.get('last_upnl', 0) for pos in ledger.values())
            prev_upnl = state.get('prev_upnl_total')
            if prev_upnl is None:
                prev_upnl = current_upnl
            upnl_delta = current_upnl - prev_upnl
            cycle_pnl = realized_pnl + upnl_delta
            state['prev_upnl_total'] = round(current_upnl, 4)

            cycle_pnls = state.get('cycle_pnls', [])
            cycle_pnls.append(round(cycle_pnl, 2))
            cycle_pnls = cycle_pnls[-200:]
            state['cycle_pnls'] = cycle_pnls

            # Refresh equity post-trade for accurate recent_rets
            base_equity = state.get('equity', args.capital)
            try:
                bal = exchange.fetch_balance()
                total_usdt = float(bal.get('USDT', {}).get('total', 0))
                equity_now = total_usdt + current_upnl
                state['equity'] = round(equity_now, 4)
                state['peak'] = max(state.get('peak', args.capital), equity_now)
            except Exception:
                equity_now = base_equity + cycle_pnl
                state['equity'] = round(equity_now, 4)
                state['peak'] = max(state.get('peak', args.capital), equity_now)
            equity_prev = state.get('prev_equity', args.capital)
            if equity_prev > 0:
                cycle_ret = (equity_now - equity_prev) / equity_prev
                recent_rets = state.get('recent_rets', [])
                recent_rets.append(round(cycle_ret, 6))
                recent_rets = recent_rets[-200:]
                state['recent_rets'] = recent_rets
            state['prev_equity'] = round(equity_now, 4)

            # R7: Track equity history for EQ-MOM Boost
            r7_eq = state.get('r7_equity_vals', [])
            r7_eq.append(round(equity_now, 4))
            r7_eq = r7_eq[-200:]
            state['r7_equity_vals'] = r7_eq

            n_wins = sum(1 for p in cycle_pnls if p > 0)
            print(f"   📊 Cycle PnL: ${cycle_pnl:+.2f} "
                  f"(realized: ${realized_pnl:+.2f}, UPnL Δ: ${upnl_delta:+.2f}, "
                  f"win rate: {n_wins}/{len(cycle_pnls)})")

        # 7. Log signals to CSV (for post-hoc analysis: fill quality, signal decay, etc.)
        try:
            sig_csv = os.path.join(log_dir, 'signal_history.csv')
            write_hdr = not os.path.exists(sig_csv)
            with open(sig_csv, 'a') as sf:
                if write_hdr:
                    sf.write('timestamp,symbol,score,confidence,deriv_scale,position_side,position_usd\n')
                pos_map = {p['symbol']: p for p in (positions or [])}
                for _, row in signals.iterrows():
                    sym = row['symbol']
                    pos = pos_map.get(sym, {})
                    sf.write(f"{now.isoformat()},{sym},{row['score']:.6f},"
                             f"{row.get('confidence', 0.5):.4f},{row.get('deriv_scale', 1.0):.4f},"
                             f"{pos.get('side', '')},{pos.get('usd', 0):.2f}\n")
        except Exception as e:
            print(f"   ⚠️  Signal CSV log error: {e}")

        # 7b. Log feature health snapshot (detect zero/NaN features early)
        try:
            feat_csv = os.path.join(log_dir, 'feature_health.csv')
            write_hdr = not os.path.exists(feat_csv)
            latest_snap = df.groupby('symbol').last()
            feat_names_file = os.path.join(root, 'results_cls_prod', 'feature_names.json')
            if os.path.exists(feat_names_file):
                with open(feat_names_file) as ff:
                    check_feats = json.load(ff)
            else:
                check_feats = []
            with open(feat_csv, 'a') as ff:
                if write_hdr:
                    ff.write('timestamp,n_symbols,n_features,n_zero_cols,n_nan_cols,zero_cols,nan_cols\n')
                n_sym = latest_snap.shape[0]
                zero_cols = [c for c in check_feats if c in latest_snap.columns
                             and (latest_snap[c] == 0).all()]
                nan_cols = [c for c in check_feats if c in latest_snap.columns
                            and latest_snap[c].isna().all()]
                ff.write(f"{now.isoformat()},{n_sym},{len(check_feats)},"
                         f"{len(zero_cols)},{len(nan_cols)},"
                         f"\"{';'.join(zero_cols)}\",\"{';'.join(nan_cols)}\"\n")
                if zero_cols:
                    print(f"   ⚠️  ZERO features ({len(zero_cols)}): {zero_cols}")
                if nan_cols:
                    print(f"   ⚠️  NaN features ({len(nan_cols)}): {nan_cols}")
        except Exception as e:
            print(f"   ⚠️  Feature health log error: {e}")

        # 7c. JSON log
        log = {
            'timestamp': now.isoformat(),
            'mode': args.mode,
            'capital': args.capital,
            'risk_config': risk_cfg,
            'positions': positions,
            'state': {k: v for k, v in state.items() if k != 'recent_rets'},
            'signals_top5': signals.head(5).to_dict('records') if signals is not None else [],
            'signals_bot5': signals.tail(5).to_dict('records') if signals is not None else [],
        }

        log_path = os.path.join(log_dir, f"trade_{now.strftime('%Y%m%d_%H%M')}.json")
        with open(log_path, 'w') as f:
            json.dump(log, f, indent=2, default=str)

        print(f"\n   📝 Log: {os.path.basename(log_path)}")

        # 8. Telegram alerts
        try:
            if bot.enabled:
                bot.alert_positions(positions, args.capital, risk_cfg['leverage'])
                bot.alert_fills(results)
        except Exception as e:
            print(f"   ⚠️  Telegram alert error: {e}")

        # 9. Dashboard update (modifies state['dash_trades'] etc.)
        try:
            update_dashboard(exchange, positions, signals, state, results, root,
                             args.capital, risk_cfg['leverage'], args.mode)
        except Exception as e:
            print(f"   ⚠️  Dashboard update error: {e}")

        # 10. Persist state AFTER dashboard so dash_trades are saved
        state['n_cycles'] = state.get('n_cycles', 0) + 1
        save_state(state, state_path)

    # Run
    last_signals = None
    if args.loop:
        print(f"\n🔄 Continuous mode (every {args.rebal_hours}h)...")
        while True:
            try:
                run_cycle()
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
                # Alert error via telegram
                try:
                    if bot.enabled:
                        bot.alert_error(str(e), context="run_cycle")
                except Exception:
                    pass

            now = datetime.now(timezone.utc)
            # Align to next rebalance boundary + 5min
            next_h = ((now.hour // args.rebal_hours) + 1) * args.rebal_hours
            next_time = now.replace(hour=0, minute=5, second=0, microsecond=0) + timedelta(hours=next_h)
            while next_time <= now:
                next_time += timedelta(hours=args.rebal_hours)

            sleep = (next_time - now).total_seconds()
            next_rebal_str = next_time.strftime('%H:%M UTC')
            print(f"\n   ⏰ Next: {next_rebal_str} ({sleep/60:.0f} min)")

            # Sleep in short intervals, refreshing dashboard + DD check each tick
            DASH_INTERVAL = 60  # 1 minute
            remaining = max(sleep, 60)
            while remaining > 0:
                try:
                    update_dashboard(exchange, [], last_signals, state, [], root,
                                     args.capital, risk_cfg['leverage'], args.mode,
                                     next_rebal_str, args.rebal_hours)
                except Exception:
                    pass

                # ── Minute-by-minute DD check ──
                if exchange and not state.get('stopped', False):
                    try:
                        bal = exchange.fetch_balance()
                        total_usdt = float(bal.get('USDT', {}).get('total', 0))
                        exch_pos = exchange.fetch_positions()
                        upnl = sum(float(p.get('unrealizedPnl', 0))
                                   for p in exch_pos if float(p.get('contracts', 0)) > 0)
                        live_equity = total_usdt + upnl
                        peak = state.get('peak', args.capital)
                        dd = live_equity / peak - 1 if peak > 0 else 0

                        if dd < risk_cfg['dd_stop']:
                            state['stopped'] = True
                            state['equity'] = live_equity
                            print(f"\n   🚨 INTRA-CYCLE DD STOP: equity={live_equity:.2f}, "
                                  f"dd={dd*100:.1f}% (limit {risk_cfg['dd_stop']*100:.0f}%)")
                            print(f"   🛑 Emergency close all positions...")
                            close_all(exchange)
                            save_state(state, state_path)
                            if bot.enabled:
                                try:
                                    bot.alert_error(
                                        f"DD STOP: eq=${live_equity:.0f}, dd={dd*100:.1f}%",
                                        context="intra_cycle_dd")
                                except Exception:
                                    pass
                    except Exception as e:
                        pass  # don't break monitoring loop for DD check errors

                chunk = min(DASH_INTERVAL, remaining)
                time.sleep(chunk)
                remaining -= chunk
    else:
        run_cycle()

    print(f"\n✅ Done!")


if __name__ == '__main__':
    main()
