"""
Feature engineering for crypto trading.
Generates ~200+ technical & statistical features from OHLCV data.
"""

import pandas as pd
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import ta


DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
DATA_FEAT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'features')
TIMEFRAME = '1h'


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Price returns at multiple horizons."""
    for h in [1, 2, 4, 6, 12, 24, 48, 72, 168]:
        df[f'ret_{h}h'] = df['close'].pct_change(h)
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalized price features (cross-sectional comparable)."""
    c, h, l, o, v = df['close'], df['high'], df['low'], df['open'], df['volume']

    # Basic ratios
    df['close_open_ratio'] = c / o - 1
    df['high_low_ratio'] = h / l - 1
    df['high_close_ratio'] = h / c - 1
    df['low_close_ratio'] = c / l - 1
    df['upper_shadow'] = (h - np.maximum(c, o)) / (h - l + 1e-10)
    df['lower_shadow'] = (np.minimum(c, o) - l) / (h - l + 1e-10)
    df['body'] = np.abs(c - o) / (h - l + 1e-10)

    # Price relative to moving averages
    for w in [6, 12, 24, 48, 72, 168, 336, 720]:
        ma = c.rolling(w).mean()
        df[f'close_ma{w}_ratio'] = c / ma - 1
        df[f'vol_ma{w}_ratio'] = v / v.rolling(w).mean() - 1

    # Rolling volatility (Garman-Klass)
    for w in [12, 24, 48, 168]:
        log_hl = np.log(h / l) ** 2
        log_co = np.log(c / o) ** 2
        df[f'gk_vol_{w}h'] = np.sqrt(
            (0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(w).mean()
        )

    # Rolling stats
    for w in [24, 48, 168]:
        r = c.pct_change()
        df[f'ret_std_{w}h'] = r.rolling(w).std()
        df[f'ret_skew_{w}h'] = r.rolling(w).skew()
        df[f'ret_kurt_{w}h'] = r.rolling(w).kurt()
        df[f'ret_mean_{w}h'] = r.rolling(w).mean()
        # Sharpe-like ratio
        df[f'ret_sharpe_{w}h'] = df[f'ret_mean_{w}h'] / (df[f'ret_std_{w}h'] + 1e-10)

    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume-based features."""
    v = df['volume']
    c = df['close']

    # Volume momentum
    for w in [6, 12, 24, 48]:
        df[f'vol_mom_{w}h'] = v / v.shift(w) - 1

    # VWAP deviation
    for w in [12, 24, 48]:
        vwap = (c * v).rolling(w).sum() / v.rolling(w).sum()
        df[f'vwap_dev_{w}h'] = c / vwap - 1

    # Volume-price correlation
    for w in [24, 48, 168]:
        df[f'vol_price_corr_{w}h'] = c.pct_change().rolling(w).corr(v.pct_change())

    # Buying/Selling pressure proxy
    df['buy_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)

    return df


def add_ta_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Technical analysis indicators via `ta` library."""
    h, l, c, v = df['high'], df['low'], df['close'], df['volume']

    # RSI at multiple periods
    for p in [6, 12, 14, 24]:
        df[f'rsi_{p}'] = ta.momentum.RSIIndicator(c, window=p).rsi()

    # MACD
    macd = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    # Bollinger Bands
    for w in [20, 48]:
        bb = ta.volatility.BollingerBands(c, window=w, window_dev=2)
        df[f'bb_high_{w}'] = (bb.bollinger_hband() - c) / c
        df[f'bb_low_{w}'] = (c - bb.bollinger_lband()) / c
        df[f'bb_width_{w}'] = bb.bollinger_wband()
        df[f'bb_pband_{w}'] = bb.bollinger_pband()

    # ATR
    for w in [14, 24, 48]:
        df[f'atr_{w}'] = ta.volatility.AverageTrueRange(h, l, c, window=w).average_true_range() / c

    # ADX
    adx = ta.trend.ADXIndicator(h, l, c, window=14)
    df['adx'] = adx.adx()
    df['adx_pos'] = adx.adx_pos()
    df['adx_neg'] = adx.adx_neg()

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(h, l, c, window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    # CCI
    df['cci_14'] = ta.trend.CCIIndicator(h, l, c, window=14).cci()
    df['cci_48'] = ta.trend.CCIIndicator(h, l, c, window=48).cci()

    # Williams %R
    df['willr_14'] = ta.momentum.WilliamsRIndicator(h, l, c, lbp=14).williams_r()

    # OBV momentum
    obv = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    for w in [12, 24, 48]:
        df[f'obv_ma_ratio_{w}'] = obv / obv.rolling(w).mean() - 1

    # MFI
    df['mfi_14'] = ta.volume.MFIIndicator(h, l, c, v, window=14).money_flow_index()

    return df


def add_cross_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Time-of-day and day-of-week features (crypto trades 24/7)."""
    ts = df['timestamp']
    df['hour'] = ts.dt.hour
    df['day_of_week'] = ts.dt.dayofweek

    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    return df


def compute_target(df: pd.DataFrame, horizon: int = 4) -> pd.DataFrame:
    """
    Target: forward return over `horizon` hours.
    - target_ret: raw forward return
    - target_cls: 1 if return > 0, else 0 (for classification)
    """
    df['target_ret'] = df['close'].pct_change(horizon).shift(-horizon)
    df['target_cls'] = (df['target_ret'] > 0).astype(int)
    return df


def process_symbol(filepath: str) -> pd.DataFrame:
    """Full feature pipeline for a single symbol."""
    df = pd.read_parquet(filepath)

    # Ensure sorted
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Add all features
    df = add_returns(df)
    df = add_price_features(df)
    df = add_volume_features(df)
    df = add_ta_indicators(df)
    df = add_cross_time_features(df)
    df = compute_target(df, horizon=4)  # Predict 4h forward

    # Extract symbol name from filename
    basename = os.path.basename(filepath)
    symbol = basename.replace(f'_{TIMEFRAME}.parquet', '').replace('_', '/')
    df['symbol'] = symbol

    return df


def main():
    os.makedirs(DATA_FEAT_DIR, exist_ok=True)

    raw_files = sorted(glob(os.path.join(DATA_RAW_DIR, f'*_{TIMEFRAME}.parquet')))
    if not raw_files:
        print(f"❌ No raw data found in {DATA_RAW_DIR}")
        print("   Run download_crypto.py first!")
        return

    print(f"⚙️  Processing features for {len(raw_files)} symbols...")

    all_dfs = []
    for filepath in tqdm(raw_files, desc="Feature engineering"):
        df = process_symbol(filepath)
        all_dfs.append(df)

    # Combine all symbols
    combined = pd.concat(all_dfs, ignore_index=True)

    # Drop rows with NaN target (last `horizon` rows) or NaN features (warmup period)
    feat_cols = [c for c in combined.columns if c not in ['timestamp', 'symbol', 'target_ret', 'target_cls',
                                                           'open', 'high', 'low', 'close', 'volume']]
    n_before = len(combined)
    combined = combined.dropna(subset=['target_ret'] + feat_cols)
    n_after = len(combined)

    print(f"\n📊 Dataset shape: {combined.shape}")
    print(f"   Symbols: {combined['symbol'].nunique()}")
    print(f"   Features: {len(feat_cols)}")
    print(f"   Rows dropped (NaN): {n_before - n_after:,}")
    print(f"   Date range: {combined['timestamp'].min()} → {combined['timestamp'].max()}")
    print(f"   Target mean return: {combined['target_ret'].mean():.6f}")
    print(f"   Target up ratio: {combined['target_cls'].mean():.4f}")

    # Save
    out_path = os.path.join(DATA_FEAT_DIR, f'crypto_features_{TIMEFRAME}.parquet')
    combined.to_parquet(out_path, index=False)
    print(f"\n✅ Saved to {out_path}")
    print(f"   File size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")

    return combined


if __name__ == '__main__':
    main()
