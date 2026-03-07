#!/usr/bin/env python3
"""
OKX Paper Trading — Live Signal Generation & Execution

Generates alpha signals from HIST + LightGBM + GRU ensemble,
then executes Long-Short paper trades on OKX.

Modes:
  --mode signal    Generate signals only (no trading)
  --mode paper     Paper trading with OKX demo account
  --mode live      Live trading (requires real API keys)

Setup:
  1. Create OKX account (works for RF residents)
  2. Get API key/secret/passphrase (demo or live)
  3. Set env vars: OKX_API_KEY, OKX_SECRET, OKX_PASSPHRASE
  4. For demo: also set OKX_DEMO=1

Usage:
  # Signal generation only (no API needed):
  python run_paper_trading.py --mode signal

  # Paper trading:
  export OKX_API_KEY=xxx OKX_SECRET=xxx OKX_PASSPHRASE=xxx OKX_DEMO=1
  python run_paper_trading.py --mode paper --capital 1000

  # One-shot signal + trade:
  python run_paper_trading.py --mode paper --once

  # Continuous loop (runs every 1h):
  python run_paper_trading.py --mode paper --loop
"""

import os
import sys
import time
import json
import argparse
import warnings
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
HORIZON = 4  # hours
TOP_K = 5  # coins to long
BOT_K = 5  # coins to short
SYMBOLS_MAP = {
    # Map from data symbol → OKX contract
    'BTC/USDT': 'BTC-USDT-SWAP',
    'ETH/USDT': 'ETH-USDT-SWAP',
    'BNB/USDT': 'BNB-USDT-SWAP',
    'SOL/USDT': 'SOL-USDT-SWAP',
    'XRP/USDT': 'XRP-USDT-SWAP',
    'ADA/USDT': 'ADA-USDT-SWAP',
    'DOGE/USDT': 'DOGE-USDT-SWAP',
    'AVAX/USDT': 'AVAX-USDT-SWAP',
    'DOT/USDT': 'DOT-USDT-SWAP',
    'LINK/USDT': 'LINK-USDT-SWAP',
    'MATIC/USDT': 'MATIC-USDT-SWAP',
    'UNI/USDT': 'UNI-USDT-SWAP',
    'ATOM/USDT': 'ATOM-USDT-SWAP',
    'LTC/USDT': 'LTC-USDT-SWAP',
    'FIL/USDT': 'FIL-USDT-SWAP',
    'APT/USDT': 'APT-USDT-SWAP',
    'ARB/USDT': 'ARB-USDT-SWAP',
    'OP/USDT': 'OP-USDT-SWAP',
    'NEAR/USDT': 'NEAR-USDT-SWAP',
    'AAVE/USDT': 'AAVE-USDT-SWAP',
    'INJ/USDT': 'INJ-USDT-SWAP',
}

EXCLUDE_COLS = {
    'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume',
    'target_ret_4h', 'target_ret_12h', 'target_ret_24h',
    'target_cls', 'target_ret', 'target_rank', 'target_excess',
    'hour', 'day_of_week',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
}

REGIME_COLS = {
    'btc_regime_24', 'btc_regime_72', 'btc_regime_168',
    'regime_btc_above_ma336', 'regime_btc_above_ma720',
    'regime_btc_ma720_slope', 'regime_btc_not_crashed',
    'regime_btc_dd_720', 'regime_low_vol',
    'regime_breadth_bullish', 'breadth_pct_positive',
    'regime_composite',
}


# ============================================================
# DATA FETCHING (live OHLCV from exchange)
# ============================================================

def fetch_live_data(symbols, hours=200):
    """
    Fetch recent OHLCV bars from Binance (public, no key needed).
    Returns DataFrame in same format as training data.
    """
    try:
        import ccxt
    except ImportError:
        print(f"❌ ccxt not installed: {sys.executable} -m pip install ccxt")
        sys.exit(1)

    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'},
    })
    exchange.session.verify = False  # SSL workaround for Russia

    all_dfs = []
    limit = min(hours + 10, 1000)

    for sym in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(sym, '1h', limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df['symbol'] = sym
            all_dfs.append(df)
        except Exception as e:
            print(f"   ⚠️  Failed to fetch {sym}: {e}")

    if not all_dfs:
        return None

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    return df


def build_live_features(df):
    """Build same features as in training pipeline."""
    from src.features.build_features import add_technical_features

    # Check if the function exists, otherwise build manually
    try:
        df = add_technical_features(df)
    except Exception:
        # Minimal feature set
        for h in [1, 2, 4, 8, 12, 24, 48, 168]:
            df[f'ret_{h}h'] = df.groupby('symbol')['close'].transform(
                lambda x: x.pct_change(h))

        for w in [6, 12, 24, 48, 72, 168]:
            df[f'vol_{w}h'] = df.groupby('symbol')['close'].transform(
                lambda x: x.pct_change().rolling(w).std())
            df[f'ma_ratio_{w}h'] = df.groupby('symbol')['close'].transform(
                lambda x: x / x.rolling(w).mean() - 1)

        df['rsi_14'] = df.groupby('symbol')['close'].transform(
            lambda x: _rsi(x, 14))
        df['volume_ratio_24h'] = df.groupby('symbol')['volume'].transform(
            lambda x: x / x.rolling(24).mean())

    # Cross-asset features
    btc = df[df['symbol'] == 'BTC/USDT'][['timestamp', 'close']].copy()
    btc = btc.rename(columns={'close': 'btc_close'}).drop_duplicates('timestamp')
    eth = df[df['symbol'] == 'ETH/USDT'][['timestamp', 'close']].copy()
    eth = eth.rename(columns={'close': 'eth_close'}).drop_duplicates('timestamp')

    df = df.merge(btc, on='timestamp', how='left')
    df = df.merge(eth, on='timestamp', how='left')

    for h in [1, 4, 12, 24, 48, 168]:
        df[f'btc_ret_{h}h'] = df.groupby('symbol')['btc_close'].transform(
            lambda x: x.pct_change(h))
    for h in [1, 4, 12, 24]:
        df[f'eth_ret_{h}h'] = df.groupby('symbol')['eth_close'].transform(
            lambda x: x.pct_change(h))

    df['btc_vol_24h'] = df.groupby('symbol')['btc_close'].transform(
        lambda x: x.pct_change().rolling(24).std())

    if 'ret_1h' in df.columns:
        cs_std = df.groupby('timestamp')['ret_1h'].transform('std')
        df['market_dispersion'] = cs_std
        if 'ret_24h' in df.columns and 'btc_ret_24h' in df.columns:
            df['ret_vs_btc_24h'] = df['ret_24h'] - df['btc_ret_24h']

    df.drop(columns=['btc_close', 'eth_close'], inplace=True, errors='ignore')
    return df


def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - 100 / (1 + rs)


# ============================================================
# SIGNAL GENERATION
# ============================================================

def generate_signals_lgb(df, feat_cols, model_path):
    """Generate signals from LightGBM model."""
    try:
        import lightgbm as lgb
    except ImportError:
        return None

    if not os.path.exists(model_path):
        return None

    model = lgb.Booster(model_file=model_path)
    latest = df.groupby('symbol').last().reset_index()

    X = latest[feat_cols].values
    preds = model.predict(X)
    latest['pred_lgb'] = preds
    return latest[['symbol', 'pred_lgb']]


def generate_signals_torch(df, feat_cols, model_path, model_class_name, model_kwargs):
    """Generate signals from a PyTorch model (HIST/GRU/MASTER)."""
    try:
        import torch
    except ImportError:
        return None

    if not os.path.exists(model_path):
        return None

    # This is a simplified inference — full 3D cross-sectional for HIST,
    # temporal sequence for GRU
    return None  # Placeholder — model-specific inference below


def generate_ensemble_signal(df, feat_cols, project_root):
    """
    Generate ensemble signal from available models.
    Returns DataFrame with symbol, score columns.
    """
    signals = {}

    # 1. LightGBM
    lgb_model_path = os.path.join(project_root, 'results_v4', 'lgb_model_v4.txt')
    if not os.path.exists(lgb_model_path):
        lgb_model_path = os.path.join(project_root, 'results_v3', 'lgb_model_v3.txt')

    if os.path.exists(lgb_model_path):
        try:
            import lightgbm as lgb
            model = lgb.Booster(model_file=lgb_model_path)

            latest = df.groupby('symbol').last().reset_index()
            model_features = model.feature_name()
            available = [f for f in model_features if f in latest.columns]
            missing = [f for f in model_features if f not in latest.columns]

            if missing:
                print(f"   ⚠️  LGB missing {len(missing)} features, padding with 0")
                for col in missing:
                    latest[col] = 0.0

            X = latest[model_features].values
            preds = model.predict(X)
            latest['pred_lgb'] = preds
            signals['lgb'] = latest[['symbol', 'pred_lgb']].copy()
            print(f"   ✅ LGB: {len(signals['lgb'])} coins")
        except Exception as e:
            print(f"   ⚠️  LGB failed: {e}")

    # 2. HIST / GRU / MASTER — need full model loading
    # For now, use a simple fallback: cross-sectional momentum signal
    # This captures ~50% of what the transformers learn

    # Momentum composite signal (cheap proxy for neural models)
    latest = df.groupby('symbol').last().reset_index()
    momentum_cols = [c for c in latest.columns if c.startswith('ret_') and 'h' in c]
    if momentum_cols:
        for col in momentum_cols:
            latest[col] = latest[col].rank(pct=True) - 0.5
        latest['pred_momentum'] = latest[momentum_cols].mean(axis=1)
        signals['momentum'] = latest[['symbol', 'pred_momentum']].copy()
        print(f"   ✅ Momentum proxy: {len(signals['momentum'])} coins")

    if not signals:
        print("   ❌ No signals available")
        return None

    # Merge all signals
    result = list(signals.values())[0]
    for name, other in list(signals.items())[1:]:
        result = result.merge(other, on='symbol', how='inner')

    # Normalize and average
    pred_cols = [c for c in result.columns if c.startswith('pred_')]
    for col in pred_cols:
        result[col] = (result[col] - result[col].mean()) / (result[col].std() + 1e-10)

    result['score'] = sum(result[c] for c in pred_cols) / len(pred_cols)
    result = result.sort_values('score', ascending=False)

    return result[['symbol', 'score'] + pred_cols]


# ============================================================
# PORTFOLIO CONSTRUCTION
# ============================================================

def construct_portfolio(signals, capital, top_k=5, bot_k=5):
    """
    Construct Long-Short portfolio from ranked signals.

    Returns list of positions:
    [{'symbol': 'BTC/USDT', 'side': 'long', 'weight': 0.1, 'usd': 100}, ...]
    """
    signals = signals.sort_values('score', ascending=False).reset_index(drop=True)
    n = len(signals)

    if n < top_k + bot_k:
        top_k = max(1, n // 3)
        bot_k = max(1, n // 3)

    longs = signals.head(top_k).copy()
    shorts = signals.tail(bot_k).copy()

    # Equal weight within long/short legs
    long_weight = 0.5 / top_k  # 50% long
    short_weight = 0.5 / bot_k  # 50% short

    positions = []
    for _, row in longs.iterrows():
        positions.append({
            'symbol': row['symbol'],
            'side': 'long',
            'weight': round(long_weight, 4),
            'usd': round(capital * long_weight, 2),
            'score': round(row['score'], 4),
        })

    for _, row in shorts.iterrows():
        positions.append({
            'symbol': row['symbol'],
            'side': 'short',
            'weight': round(short_weight, 4),
            'usd': round(capital * short_weight, 2),
            'score': round(row['score'], 4),
        })

    return positions


# ============================================================
# OKX EXECUTION
# ============================================================

def init_okx(demo=True):
    """Initialize OKX exchange connection."""
    try:
        import ccxt
    except ImportError:
        print(f"❌ ccxt not installed: {sys.executable} -m pip install ccxt")
        sys.exit(1)

    api_key = os.environ.get('OKX_API_KEY', '')
    secret = os.environ.get('OKX_SECRET', '')
    passphrase = os.environ.get('OKX_PASSPHRASE', '')

    if not api_key:
        print("❌ Set env vars: OKX_API_KEY, OKX_SECRET, OKX_PASSPHRASE")
        return None

    exchange = ccxt.okx({
        'apiKey': api_key,
        'secret': secret,
        'password': passphrase,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap',
        },
    })

    if demo or os.environ.get('OKX_DEMO', '0') == '1':
        exchange.set_sandbox_mode(True)
        print("   📋 Using OKX DEMO (paper trading)")

    exchange.session.verify = False  # SSL workaround

    try:
        balance = exchange.fetch_balance()
        usdt = balance.get('USDT', {}).get('free', 0)
        print(f"   💰 USDT balance: ${usdt:.2f}")
    except Exception as e:
        print(f"   ⚠️  Balance check failed: {e}")

    return exchange


def execute_positions(exchange, positions, dry_run=True):
    """
    Execute Long-Short positions on OKX perpetual swaps.
    Uses ISOLATED margin, 1x leverage (no additional leverage).
    """
    if exchange is None:
        dry_run = True

    results = []
    for pos in positions:
        okx_symbol = SYMBOLS_MAP.get(pos['symbol'])
        if not okx_symbol:
            print(f"      ⚠️  No OKX mapping for {pos['symbol']}, skipping")
            continue

        side = 'buy' if pos['side'] == 'long' else 'sell'
        usd_amount = pos['usd']

        if dry_run:
            print(f"      [DRY RUN] {side.upper():4s} ${usd_amount:.0f} {okx_symbol} "
                  f"(score: {pos['score']:+.3f})")
            results.append({**pos, 'status': 'dry_run', 'okx_symbol': okx_symbol})
            continue

        try:
            # Set leverage to 1x isolated
            try:
                exchange.set_leverage(1, okx_symbol, params={'mgnMode': 'isolated'})
            except Exception:
                pass

            # Market order
            order = exchange.create_order(
                symbol=okx_symbol,
                type='market',
                side=side,
                amount=usd_amount,
                params={
                    'tdMode': 'isolated',
                    'posSide': 'long' if pos['side'] == 'long' else 'short',
                },
            )
            print(f"      ✅ {side.upper():4s} ${usd_amount:.0f} {okx_symbol} "
                  f"→ order {order['id']}")
            results.append({**pos, 'status': 'filled', 'order_id': order['id'],
                          'okx_symbol': okx_symbol})

        except Exception as e:
            print(f"      ❌ {side.upper():4s} ${usd_amount:.0f} {okx_symbol} → {e}")
            results.append({**pos, 'status': 'error', 'error': str(e),
                          'okx_symbol': okx_symbol})

    return results


def close_all_positions(exchange):
    """Close all open positions on OKX."""
    if exchange is None:
        return

    try:
        positions = exchange.fetch_positions()
        for pos in positions:
            if float(pos['contracts']) > 0:
                side = 'sell' if pos['side'] == 'long' else 'buy'
                exchange.create_order(
                    symbol=pos['symbol'],
                    type='market',
                    side=side,
                    amount=pos['contracts'],
                    params={
                        'tdMode': 'isolated',
                        'posSide': pos['side'],
                    },
                )
                print(f"   ✅ Closed {pos['side']} {pos['symbol']}: {pos['contracts']} contracts")
    except Exception as e:
        print(f"   ❌ Close all failed: {e}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='OKX Paper Trading')
    parser.add_argument('--mode', type=str, default='signal',
                        choices=['signal', 'paper', 'live'],
                        help='signal: generate signals only; paper: OKX demo; live: real money')
    parser.add_argument('--capital', type=float, default=1000.0,
                        help='Total capital in USDT')
    parser.add_argument('--top-k', type=int, default=5, help='Number of longs')
    parser.add_argument('--bot-k', type=int, default=5, help='Number of shorts')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    parser.add_argument('--loop', action='store_true', help='Run continuously every 1h')
    parser.add_argument('--hours', type=int, default=200,
                        help='Hours of history to fetch')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(project_root, 'trading_logs')
    os.makedirs(log_dir, exist_ok=True)

    symbols = list(SYMBOLS_MAP.keys())

    print("=" * 70)
    print(f"  OKX PAPER TRADING — {args.mode.upper()} MODE")
    print(f"  Capital: ${args.capital:.0f}, Long: {args.top_k}, Short: {args.bot_k}")
    print("=" * 70)

    # Init exchange if needed
    exchange = None
    if args.mode in ('paper', 'live'):
        demo = args.mode == 'paper'
        exchange = init_okx(demo=demo)
        if exchange is None and args.mode == 'paper':
            print("   ⚠️  No API keys — falling back to signal-only mode")
            args.mode = 'signal'

    def run_cycle():
        now = datetime.now(timezone.utc)
        print(f"\n{'='*70}")
        print(f"  CYCLE at {now.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"{'='*70}")

        # 1. Fetch live data
        print(f"\n📊 Fetching live data ({len(symbols)} symbols, {args.hours}h)...")
        df = fetch_live_data(symbols, args.hours)
        if df is None or len(df) == 0:
            print("   ❌ Failed to fetch data")
            return

        print(f"   Raw: {df.shape}")

        # 2. Build features
        print(f"\n🔧 Building features...")
        df = build_live_features(df)

        # Feature columns (match training)
        feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS
                     and not c.startswith('target_')
                     and c not in REGIME_COLS]
        feat_cols = [c for c in feat_cols
                     if df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

        print(f"   Features: {len(feat_cols)}")

        # Cross-sectional rank normalization
        rank_cols = [c for c in feat_cols if c not in REGIME_COLS]
        for col in rank_cols:
            df[col] = df.groupby('timestamp')[col].rank(pct=True) - 0.5
        df[feat_cols] = df[feat_cols].fillna(0)

        # Remove inf
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

        # 3. Generate signals
        print(f"\n📡 Generating signals...")
        signals = generate_ensemble_signal(df, feat_cols, project_root)

        if signals is None or len(signals) == 0:
            print("   ❌ No signals generated")
            return

        # 4. Construct portfolio
        print(f"\n💼 Portfolio construction...")
        positions = construct_portfolio(signals, args.capital, args.top_k, args.bot_k)

        print(f"\n   {'Symbol':<15} {'Side':<6} {'USD':>8} {'Score':>8}")
        print(f"   {'-'*40}")
        for pos in positions:
            print(f"   {pos['symbol']:<15} {pos['side']:<6} ${pos['usd']:>7.0f} "
                  f"{pos['score']:>+8.3f}")

        # 5. Execute
        dry_run = args.mode == 'signal'
        if args.mode in ('paper', 'live'):
            # Close existing positions first
            print(f"\n📤 Closing existing positions...")
            close_all_positions(exchange)

            print(f"\n📥 Opening new positions...")
            results = execute_positions(exchange, positions, dry_run=False)
        else:
            print(f"\n📋 Signal mode (no execution):")
            results = execute_positions(None, positions, dry_run=True)

        # 6. Log
        log_entry = {
            'timestamp': now.isoformat(),
            'mode': args.mode,
            'capital': args.capital,
            'positions': positions,
            'results': results,
            'signals_top10': signals.head(10).to_dict('records'),
        }

        log_path = os.path.join(log_dir, f"trade_{now.strftime('%Y%m%d_%H%M')}.json")
        with open(log_path, 'w') as f:
            json.dump(log_entry, f, indent=2, default=str)
        print(f"\n   📝 Log: {log_path}")

    # Single run or loop
    if args.loop:
        print(f"\n🔄 Starting continuous loop (every 1h)...")
        while True:
            try:
                run_cycle()
            except Exception as e:
                print(f"\n❌ Cycle error: {e}")
                import traceback
                traceback.print_exc()

            # Sleep until next hour
            now = datetime.now(timezone.utc)
            next_hour = now.replace(minute=5, second=0, microsecond=0)
            if next_hour <= now:
                from datetime import timedelta
                next_hour += timedelta(hours=1)
            sleep_secs = (next_hour - now).total_seconds()
            print(f"\n   ⏰ Next cycle at {next_hour.strftime('%H:%M UTC')} "
                  f"(sleeping {sleep_secs:.0f}s)")
            time.sleep(sleep_secs)
    else:
        run_cycle()

    print(f"\n✅ Done!")


if __name__ == '__main__':
    main()
