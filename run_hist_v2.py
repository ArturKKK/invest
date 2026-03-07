#!/usr/bin/env python3
"""
HIST v2 — Transformer with Sentiment Features for Crypto Alpha

Changes from v1:
1. Sentiment features:
   - Fear & Greed Index (daily → hourly)
   - Funding rates from OKX (per-coin, where available)
   - Synthetic positioning proxies (reversal, volume surge, beta, dispersion)
2. Sentiment-aware architecture:
   - Separate sentiment embedding branch
   - Gated fusion of technical and sentiment signals
3. Risk-aware evaluation:
   - Realistic costs (taker + funding + slippage)
   - Vol targeting + drawdown stop metrics
4. Compatible with v5 pipeline for ensemble

Usage:
  python run_hist_v2.py                              # Full training
  python run_hist_v2.py --device cuda                # GPU training
  python run_hist_v2.py --lgb-preds results_v5/test_predictions_v5.parquet

Requirements:
  pip install torch pandas numpy scipy pyarrow tqdm
"""

import sys
import os
import argparse
import json
import warnings
from datetime import datetime

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
TRAIN_END = '2024-06-29'
VAL_START = '2024-07-01'
VAL_END = '2024-12-30'
TEST_START = '2025-01-01'
HORIZON = 4

EXCLUDE_COLS = {
    'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume',
    'target_ret_4h', 'target_ret_12h', 'target_ret_24h',
    'target_cls', 'target_ret', 'target_rank', 'target_excess',
    'hour', 'day_of_week',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
}

CRYPTO_CONCEPTS = {
    'BTC/USDT': 0,
    'ETH/USDT': 1, 'SOL/USDT': 1, 'AVAX/USDT': 1, 'DOT/USDT': 1,
    'NEAR/USDT': 1, 'ATOM/USDT': 1, 'ALGO/USDT': 1, 'FTM/USDT': 1,
    'EGLD/USDT': 1, 'XTZ/USDT': 1, 'FLOW/USDT': 1, 'APT/USDT': 1,
    'AAVE/USDT': 2, 'MKR/USDT': 2, 'CRV/USDT': 2, 'COMP/USDT': 2,
    'SNX/USDT': 2, 'SUSHI/USDT': 2, 'YFI/USDT': 2, 'UNI/USDT': 2,
    'LDO/USDT': 2,
    'AXS/USDT': 3, 'SAND/USDT': 3, 'MANA/USDT': 3, 'GALA/USDT': 3,
    'ENJ/USDT': 3, 'IMX/USDT': 3, 'CHZ/USDT': 3,
    'OP/USDT': 4, 'ARB/USDT': 4,
    'LINK/USDT': 5, 'GRT/USDT': 5, 'FIL/USDT': 5, 'THETA/USDT': 5,
    'ENS/USDT': 5, 'BAT/USDT': 5,
    'DOGE/USDT': 6,
    'BNB/USDT': 7, 'XRP/USDT': 7, 'ADA/USDT': 7, 'LTC/USDT': 7,
    'ETC/USDT': 7, 'RUNE/USDT': 7, 'ZIL/USDT': 7, 'ONE/USDT': 7,
    'IOTA/USDT': 7, 'ICX/USDT': 7, 'INJ/USDT': 7, 'MATIC/USDT': 7,
}
N_CONCEPTS = 8

REGIME_COLS = {
    'btc_regime_24', 'btc_regime_72', 'btc_regime_168',
    'regime_btc_above_ma336', 'regime_btc_above_ma720',
    'regime_btc_ma720_slope', 'regime_btc_not_crashed',
    'regime_btc_dd_720', 'regime_low_vol',
    'regime_breadth_bullish', 'breadth_pct_positive',
    'regime_composite',
    'fng_value', 'fng_extreme_fear', 'fng_extreme_greed',
    'fng_ma7', 'fng_ma30', 'fng_momentum',
    'market_avg_funding', 'market_funding_skew',
}

# Cost model
COST_PER_PERIOD = 0.000225  # blended 0.03% + slip 0.01% × 2sides × 25% turnover + funding 0.0025%


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_multi_horizon_targets(df):
    print("   🎯 Adding targets...")
    for h in [4, 12, 24]:
        df[f'target_ret_{h}h'] = df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change(h).shift(-h))
    return df


def add_cross_asset_features(df):
    print("   🌐 Adding cross-asset features...")
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
    cs_std = df.groupby('timestamp')['ret_1h'].transform('std')
    df['market_dispersion'] = cs_std
    df['ret_vs_btc_24h'] = df['ret_24h'] - df['btc_ret_24h']

    df.drop(columns=['btc_close', 'eth_close'], inplace=True, errors='ignore')
    return df


def add_sentiment_features(df, project_root):
    """Add sentiment features: FNG + funding rates + synthetic proxies."""
    print("   📰 Adding sentiment features...")
    sent_dir = os.path.join(project_root, 'data', 'sentiment')

    # ---- Fear & Greed ----
    fng_path = os.path.join(sent_dir, 'fear_greed.parquet')
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

        fng_ts = df.groupby('timestamp')['fng_value'].first().reset_index()
        fng_ts = fng_ts.sort_values('timestamp')
        fng_ts['fng_ma7'] = fng_ts['fng_value'].rolling(7 * 24, min_periods=24).mean()
        fng_ts['fng_ma30'] = fng_ts['fng_value'].rolling(30 * 24, min_periods=48).mean()
        fng_ts['fng_momentum'] = fng_ts['fng_value'] - fng_ts['fng_ma30']

        df = df.merge(fng_ts[['timestamp', 'fng_ma7', 'fng_ma30', 'fng_momentum']],
                       on='timestamp', how='left')
        df.drop(columns=['date'], inplace=True)
        print(f"      ✅ FNG: mean={df['fng_value'].mean():.1f}")
    else:
        print(f"      ⚠️  No FNG data")

    # ---- Funding Rates ----
    fund_path = os.path.join(sent_dir, 'funding_rates.parquet')
    if os.path.exists(fund_path):
        fund = pd.read_parquet(fund_path)
        fund['timestamp'] = pd.to_datetime(fund['timestamp'], utc=True)
        fund['ts_8h'] = fund['timestamp'].dt.floor('8h')
        df['ts_8h'] = df['timestamp'].dt.floor('8h')

        fund_melt = fund[['ts_8h', 'symbol', 'funding_rate']].drop_duplicates(['ts_8h', 'symbol'])
        df = df.merge(fund_melt, on=['ts_8h', 'symbol'], how='left')
        df['funding_rate'] = df['funding_rate'].fillna(0)

        market_fund = fund.groupby('ts_8h')['funding_rate'].agg(['mean', 'std']).reset_index()
        market_fund.columns = ['ts_8h', 'market_avg_funding', 'market_funding_std']
        df = df.merge(market_fund, on='ts_8h', how='left')
        df['market_avg_funding'] = df['market_avg_funding'].fillna(0)
        df['market_funding_std'] = df['market_funding_std'].fillna(0)
        df['market_funding_skew'] = df['market_avg_funding'] / (df['market_funding_std'] + 1e-8)
        df['funding_vs_market'] = df['funding_rate'] - df['market_avg_funding']

        df.drop(columns=['ts_8h'], inplace=True)
        n_nz = (df['funding_rate'] != 0).sum()
        print(f"      ✅ Funding: {n_nz:,} non-zero ({n_nz/len(df)*100:.1f}%)")
    else:
        print(f"      ⚠️  No funding data")

    # ---- Long/Short Ratio ----
    lsr_path = os.path.join(sent_dir, 'long_short_ratio.parquet')
    if os.path.exists(lsr_path):
        lsr = pd.read_parquet(lsr_path)
        lsr['timestamp'] = pd.to_datetime(lsr['timestamp'], utc=True)
        lsr_merge = lsr[['timestamp', 'symbol', 'long_short_ratio']].drop_duplicates(
            ['timestamp', 'symbol'])
        df = df.merge(lsr_merge, on=['timestamp', 'symbol'], how='left')
        df['long_short_ratio'] = df['long_short_ratio'].fillna(1.0)
        print(f"      ✅ LS ratio merged")
    else:
        print(f"      ⚠️  No LS ratio data")

    # ---- Synthetic Positioning Proxies ----
    print("      Synthetic features...")

    # Reversal scores
    for short, long in [(4, 24), (12, 48), (24, 168)]:
        if f'ret_{short}h' in df.columns and f'ret_{long}h' in df.columns:
            df[f'reversal_{short}v{long}'] = df[f'ret_{short}h'] - df[f'ret_{long}h'] / (long / short)

    # Volume surge
    if 'volume' in df.columns:
        for w in [12, 24, 48]:
            vol_ma = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(w).mean())
            df[f'vol_surge_{w}h'] = df['volume'] / (vol_ma + 1e-10) - 1

    # Cross-coin dispersion
    if 'ret_4h' in df.columns:
        cs_disp = df.groupby('timestamp')['ret_4h'].transform('std')
        df['cross_coin_dispersion'] = cs_disp
        df['cross_coin_disp_ma24'] = df.groupby('symbol')['cross_coin_dispersion'].transform(
            lambda x: x.rolling(24, min_periods=6).mean())
        df['dispersion_regime'] = df['cross_coin_dispersion'] / (df['cross_coin_disp_ma24'] + 1e-10)

    # BTC beta
    if 'ret_1h' in df.columns and 'btc_ret_1h' in df.columns:
        for w in [48, 168]:
            cov = df.groupby('symbol').apply(
                lambda g: g['ret_1h'].rolling(w).cov(g['btc_ret_1h'])
            ).reset_index(level=0, drop=True)
            var = df.groupby('symbol')['btc_ret_1h'].transform(
                lambda x: x.rolling(w).var() + 1e-10)
            df[f'btc_beta_{w}h'] = cov / var

    n_sent = sum(1 for c in df.columns if any(k in c for k in
                  ['fng', 'funding', 'reversal', 'surge', 'dispersion',
                   'long_short', 'btc_beta']))
    print(f"   ✅ {n_sent} sentiment/positioning features added")

    return df


def cross_sectional_rank(df, feat_cols):
    print("   📐 Cross-sectional rank normalization...")
    regime_backup = {}
    for col in REGIME_COLS:
        if col in df.columns:
            regime_backup[col] = df[col].copy()

    rank_cols = [c for c in feat_cols if c not in REGIME_COLS]
    ranked = df.groupby('timestamp')[rank_cols].rank(pct=True)
    df[rank_cols] = ranked - 0.5

    for col, vals in regime_backup.items():
        df[col] = vals
    return df


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_cross_section_data(df, feat_cols, target_col, actual_return_col=None):
    """Convert flat DataFrame to cross-sectional format (T, N, F)."""
    symbols = sorted(df['symbol'].unique())
    sym2idx = {s: i for i, s in enumerate(symbols)}
    N = len(symbols)
    F = len(feat_cols)

    concept_ids = np.array([CRYPTO_CONCEPTS.get(s, 7) for s in symbols])

    result = {}
    for split_name, split_df in [
        ('train', df[df['timestamp'] < TRAIN_END]),
        ('val', df[(df['timestamp'] >= VAL_START) & (df['timestamp'] < VAL_END)]),
        ('test', df[df['timestamp'] >= TEST_START]),
    ]:
        split_df = split_df.copy()
        timestamps = sorted(split_df['timestamp'].unique())
        ts2idx = {ts: i for i, ts in enumerate(timestamps)}
        T = len(timestamps)

        X = np.zeros((T, N, F), dtype=np.float32)
        y_arr = np.full((T, N), np.nan, dtype=np.float32)
        y_actual_arr = np.full((T, N), np.nan, dtype=np.float32)
        mask = np.zeros((T, N), dtype=np.float32)

        split_df['_ti'] = split_df['timestamp'].map(ts2idx)
        split_df['_si'] = split_df['symbol'].map(sym2idx)

        ti = split_df['_ti'].values
        si = split_df['_si'].values

        for i, col in enumerate(feat_cols):
            X[ti, si, i] = split_df[col].values.astype(np.float32)

        y_arr[ti, si] = split_df[target_col].values.astype(np.float32)
        if actual_return_col and actual_return_col in split_df.columns:
            y_actual_arr[ti, si] = split_df[actual_return_col].values.astype(np.float32)
        mask[ti, si] = 1.0

        X = np.nan_to_num(X, nan=0.0)
        y_arr = np.nan_to_num(y_arr, nan=0.0)
        y_actual_arr = np.nan_to_num(y_actual_arr, nan=0.0)

        result[split_name] = {
            'X': X, 'y': y_arr, 'y_actual': y_actual_arr, 'mask': mask,
            'timestamps': timestamps,
        }
        print(f"   {split_name}: {T} timestamps × {N} coins × {F} features")

    return result, symbols, concept_ids


# ============================================================
# HIST v2 MODEL (PyTorch)
# ============================================================

def build_model_and_train(data, concept_ids, feat_cols, args):
    """Build and train HIST v2 model with sentiment-aware architecture."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import Dataset, DataLoader
    except ImportError:
        print("❌ PyTorch not installed: pip install torch")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    print(f"\n   🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # Identify sentiment feature indices for separate processing
    sent_keywords = ['fng', 'funding', 'reversal', 'surge', 'dispersion',
                      'long_short', 'btc_beta', 'market_avg_funding', 'market_funding']
    sent_indices = [i for i, c in enumerate(feat_cols)
                    if any(k in c for k in sent_keywords)]
    tech_indices = [i for i in range(len(feat_cols)) if i not in sent_indices]

    n_sent = len(sent_indices)
    n_tech = len(tech_indices)
    print(f"   Technical features: {n_tech}, Sentiment features: {n_sent}")

    class CrossSectionDataset(Dataset):
        def __init__(self, X, y, mask):
            self.X = torch.FloatTensor(X)
            self.y = torch.FloatTensor(y)
            self.mask = torch.FloatTensor(mask)

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx], self.mask[idx]

    class HISTv2Model(nn.Module):
        """
        HIST v2: Sentiment-aware cross-stock attention model.

        Architecture:
        1. Technical embedding branch (MLP)
        2. Sentiment embedding branch (smaller MLP)
        3. Gated fusion of tech + sentiment
        4. Concept attention (crypto categories)
        5. Cross-stock self-attention
        6. Prediction head

        The separate sentiment branch allows the model to learn
        sentiment-specific representations before fusing with technicals.
        """
        def __init__(self, n_tech, n_sent, d_model=128, n_heads=4, n_layers=2,
                     n_concepts=8, dropout=0.1):
            super().__init__()
            self.d_model = d_model
            self.n_tech = n_tech
            self.n_sent = n_sent

            # Technical feature embedding
            self.tech_embed = nn.Sequential(
                nn.Linear(n_tech, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            # Sentiment feature embedding (separate branch)
            if n_sent > 0:
                sent_hidden = max(32, min(d_model // 2, n_sent * 4))
                self.sent_embed = nn.Sequential(
                    nn.Linear(n_sent, sent_hidden),
                    nn.LayerNorm(sent_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(sent_hidden, d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )

                # Gated fusion: tech + sentiment → fused representation
                self.fusion_gate = nn.Sequential(
                    nn.Linear(d_model * 2, d_model),
                    nn.Sigmoid(),
                )
            else:
                self.sent_embed = None
                self.fusion_gate = None

            # Concept embeddings
            self.concept_embed = nn.Embedding(n_concepts, d_model)
            self.concept_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid(),
            )

            # Cross-stock self-attention
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation='gelu',
            )
            self.cross_attn = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

            # Hidden info gate
            self.hidden_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid(),
            )

            # Prediction head
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )

        def forward(self, x, concept_ids, tech_idx=None, sent_idx=None):
            """
            x: (B, N, F) — batch of cross-sections (all features)
            concept_ids: (N,) — concept index per coin
            tech_idx: list of int — indices of technical features in x
            sent_idx: list of int — indices of sentiment features in x
            """
            B, N, F = x.shape

            if tech_idx is not None and sent_idx is not None and self.sent_embed is not None:
                # Separate technical and sentiment features
                x_tech = x[:, :, tech_idx]  # (B, N, n_tech)
                x_sent = x[:, :, sent_idx]  # (B, N, n_sent)

                h_tech = self.tech_embed(x_tech)  # (B, N, d_model)
                h_sent = self.sent_embed(x_sent)  # (B, N, d_model)

                # Gated fusion
                gate_input = torch.cat([h_tech, h_sent], dim=-1)
                gate = self.fusion_gate(gate_input)
                h = h_tech + gate * h_sent  # Residual: tech + gated sentiment
            else:
                # Fallback: use all features as technical
                h = self.tech_embed(x)

            # Concept attention
            concepts = self.concept_embed(concept_ids)
            concepts = concepts.unsqueeze(0).expand(B, -1, -1)

            gate_input = torch.cat([h, concepts], dim=-1)
            gate = self.concept_gate(gate_input)
            h_with_concept = h + gate * concepts

            # Cross-stock attention
            hidden = self.cross_attn(h_with_concept)

            hidden_gate_input = torch.cat([h_with_concept, hidden], dim=-1)
            hgate = self.hidden_gate(hidden_gate_input)
            h_final = h_with_concept + hgate * (hidden - h_with_concept)

            out = self.head(h_final).squeeze(-1)
            return out

    # ---- Loss functions ----
    def ic_loss(pred, target, mask):
        losses = []
        B = pred.shape[0]
        for i in range(B):
            m = mask[i] > 0.5
            if m.sum() < 5:
                continue
            p = pred[i][m]
            t = target[i][m]
            p = p - p.mean()
            t = t - t.mean()
            corr = (p * t).sum() / (p.norm() * t.norm() + 1e-8)
            losses.append(-corr)
        if not losses:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return torch.stack(losses).mean()

    def mse_loss_masked(pred, target, mask):
        m = mask > 0.5
        if m.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return F.mse_loss(pred[m], target[m])

    # ---- Build ----
    n_features = data['train']['X'].shape[2]

    train_ds = CrossSectionDataset(data['train']['X'], data['train']['y'], data['train']['mask'])
    val_ds = CrossSectionDataset(data['val']['X'], data['val']['y'], data['val']['mask'])
    test_ds = CrossSectionDataset(data['test']['X'], data['test']['y'], data['test']['mask'])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    # Create model with separate branches if we have sentiment features
    if n_sent > 0:
        model = HISTv2Model(
            n_tech=n_tech,
            n_sent=n_sent,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            n_concepts=N_CONCEPTS,
            dropout=args.dropout,
        ).to(device)
    else:
        # Fall back to standard architecture
        model = HISTv2Model(
            n_tech=n_features,
            n_sent=0,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            n_concepts=N_CONCEPTS,
            dropout=args.dropout,
        ).to(device)

    concept_tensor = torch.LongTensor(concept_ids).to(device)

    # Index tensors for feature splitting
    tech_idx_tensor = torch.LongTensor(tech_indices).to(device) if n_sent > 0 else None
    sent_idx_tensor = torch.LongTensor(sent_indices).to(device) if n_sent > 0 else None

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model params: {n_params:,}")
    print(f"   Architecture: tech_embed({n_tech}→{args.d_model}) + "
          f"sent_embed({n_sent}→{args.d_model}) + "
          f"gated_fusion + concept({N_CONCEPTS}) + cross_attn({args.n_layers}L,{args.n_heads}H)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup_epochs = min(5, args.epochs // 10)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Training ----
    print(f"\n{'='*70}")
    print(f"  TRAINING HIST v2 ({args.epochs} epochs, BS={args.batch_size}, LR={args.lr})")
    print(f"{'='*70}")

    best_val_ic = -999
    best_epoch = 0
    patience_counter = 0
    best_state = None
    ic_weight = args.ic_weight
    mse_weight = 1.0 - ic_weight

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch, mask_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)

            pred = model(X_batch, concept_tensor, tech_idx_tensor, sent_idx_tensor)

            loss = mse_weight * mse_loss_masked(pred, y_batch, mask_batch) + \
                   ic_weight * ic_loss(pred, y_batch, mask_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Validate
        model.eval()
        val_preds_all, val_targets_all, val_masks_all = [], [], []
        with torch.no_grad():
            for X_batch, y_batch, mask_batch in val_loader:
                X_batch = X_batch.to(device)
                pred = model(X_batch, concept_tensor, tech_idx_tensor, sent_idx_tensor)
                val_preds_all.append(pred.cpu().numpy())
                val_targets_all.append(y_batch.numpy())
                val_masks_all.append(mask_batch.numpy())

        val_preds = np.concatenate(val_preds_all, axis=0)
        val_targets = np.concatenate(val_targets_all, axis=0)
        val_masks = np.concatenate(val_masks_all, axis=0)

        rank_ics = []
        for t in range(val_preds.shape[0]):
            m = val_masks[t] > 0.5
            if m.sum() < 10:
                continue
            c, _ = spearmanr(val_preds[t][m], val_targets[t][m])
            if not np.isnan(c):
                rank_ics.append(c)

        val_rank_ic = np.mean(rank_ics) if rank_ics else 0
        val_rank_icir = val_rank_ic / (np.std(rank_ics) + 1e-10) if rank_ics else 0

        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"   Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Loss: {np.mean(train_losses):.5f} | "
                  f"Val Rank IC: {val_rank_ic:.4f} | "
                  f"Val ICIR: {val_rank_icir:.3f} | "
                  f"LR: {lr_now:.6f}")

        if val_rank_ic > best_val_ic:
            best_val_ic = val_rank_ic
            best_epoch = epoch + 1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"   ⏹️  Early stopping at epoch {epoch+1} (best: {best_epoch})")
                break

    print(f"\n   ✅ Best epoch: {best_epoch}, Val Rank IC: {best_val_ic:.4f}")

    # Load best model
    model.load_state_dict(best_state)
    model.eval()

    # ---- Test ----
    print(f"\n{'='*70}")
    print(f"  TEST EVALUATION")
    print(f"{'='*70}")

    test_preds_all, test_targets_all, test_masks_all = [], [], []
    with torch.no_grad():
        for X_batch, y_batch, mask_batch in test_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch, concept_tensor, tech_idx_tensor, sent_idx_tensor)
            test_preds_all.append(pred.cpu().numpy())
            test_targets_all.append(y_batch.numpy())
            test_masks_all.append(mask_batch.numpy())

    test_preds = np.concatenate(test_preds_all, axis=0)
    test_targets = np.concatenate(test_targets_all, axis=0)
    test_masks = np.concatenate(test_masks_all, axis=0)

    return model, test_preds, test_targets, test_masks, best_val_ic, best_epoch


# ============================================================
# EVALUATION
# ============================================================

def evaluate_predictions(test_preds, test_targets, test_masks, test_timestamps,
                          symbols, horizon_hours=4, test_actual_rets=None):
    """Evaluate with risk overlay metrics."""
    periods_per_day = 24 // horizon_hours
    periods_per_year = periods_per_day * 365

    rank_ics, ics, ls_rets_raw = [], [], []
    lo5_rets, lo10_rets = [], []
    T, N = test_preds.shape

    for t in range(T):
        m = test_masks[t] > 0.5
        if m.sum() < 10:
            continue

        p = test_preds[t][m]
        a = test_targets[t][m]
        valid = ~(np.isnan(p) | np.isnan(a))
        if valid.sum() < 10:
            continue

        pv, av = p[valid], a[valid]
        ic = np.corrcoef(pv, av)[0, 1]
        ric, _ = spearmanr(pv, av)
        ics.append(ic)
        rank_ics.append(ric)

        # P&L from actual returns
        if test_actual_rets is not None:
            act = test_actual_rets[t][m]
            act_valid = act[valid]
        else:
            act_valid = av

        order = np.argsort(-pv)
        sorted_actual = act_valid[order]

        n_quintile = max(len(pv) // 5, 1)
        long_ret = sorted_actual[:n_quintile].mean()
        short_ret = sorted_actual[-n_quintile:].mean()
        ls_rets_raw.append(long_ret - short_ret)
        lo5_rets.append(sorted_actual[:min(5, len(sorted_actual))].mean())
        lo10_rets.append(sorted_actual[:min(10, len(sorted_actual))].mean())

    rank_ics = np.array(rank_ics)
    ics = np.array(ics)
    ls_rets_raw = np.array(ls_rets_raw)
    ls_rets_net = ls_rets_raw - COST_PER_PERIOD * 2
    lo5 = np.array(lo5_rets) - COST_PER_PERIOD
    lo10 = np.array(lo10_rets) - COST_PER_PERIOD

    def sharpe(r, ppyr):
        return (r.mean() / (r.std() + 1e-10)) * np.sqrt(ppyr)
    def max_dd(r):
        cum = np.cumprod(1 + r)
        return np.min(cum / np.maximum.accumulate(cum) - 1)
    def total_ret(r):
        return np.prod(1 + r) - 1

    # Vol targeting
    vt_rets = np.zeros_like(ls_rets_raw)
    for i in range(len(ls_rets_raw)):
        if i < 48:
            scale = 1.0
        else:
            rv = np.std(ls_rets_raw[max(0, i-48):i])
            scale = np.clip(0.02 / (rv + 1e-8), 0.1, 2.0)
        vt_rets[i] = ls_rets_raw[i] * scale - COST_PER_PERIOD * 2 * scale

    # DD stop
    dd_rets = np.zeros_like(ls_rets_net)
    eq = 1.0
    peak = 1.0
    stopped = False
    for i in range(len(ls_rets_net)):
        if stopped:
            eq *= (1 + ls_rets_net[i])
            if eq / peak - 1 > -0.10:
                stopped = False
                dd_rets[i] = ls_rets_net[i]
        else:
            eq *= (1 + ls_rets_net[i])
            if eq > peak:
                peak = eq
            if eq / peak - 1 < -0.25:
                stopped = True
            else:
                dd_rets[i] = ls_rets_net[i]

    # Daily ICIR
    n_days = len(rank_ics) // periods_per_day
    daily_rics = []
    for d in range(n_days):
        s, e = d * periods_per_day, (d + 1) * periods_per_day
        daily_rics.append(rank_ics[s:e].mean())
    daily_rics = np.array(daily_rics)
    rank_icir = daily_rics.mean() / (daily_rics.std() + 1e-10) if len(daily_rics) > 0 else 0

    metrics = {
        'IC': round(float(ics.mean()), 4),
        'Rank_IC': round(float(rank_ics.mean()), 4),
        'Rank_ICIR': round(float(rank_icir), 4),
        'LS_Sharpe_raw': round(float(sharpe(ls_rets_raw, periods_per_year)), 2),
        'LS_Sharpe_net': round(float(sharpe(ls_rets_net, periods_per_year)), 2),
        'LS_MaxDD_net_%': round(float(max_dd(ls_rets_net) * 100), 1),
        'LS_Total_net_%': round(float(total_ret(ls_rets_net) * 100), 1),
        'LS_VolTarget_Sharpe': round(float(sharpe(vt_rets, periods_per_year)), 2),
        'LS_VolTarget_MaxDD_%': round(float(max_dd(vt_rets) * 100), 1),
        'LS_DDStop_Sharpe': round(float(sharpe(dd_rets, periods_per_year)), 2),
        'LS_DDStop_MaxDD_%': round(float(max_dd(dd_rets) * 100), 1),
        'LO5_Sharpe': round(float(sharpe(lo5, periods_per_year)), 2),
        'LO5_Total_%': round(float(total_ret(lo5) * 100), 1),
        'N_periods': len(ls_rets_raw),
    }
    return metrics


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='HIST v2 — Sentiment-aware Transformer')
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--results', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--ic-weight', type=float, default=0.5)
    parser.add_argument('--lgb-preds', type=str, default=None)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data or os.path.join(project_root, 'data', 'features')
    results_dir = args.results or os.path.join(project_root, 'results_hist_v2')
    os.makedirs(results_dir, exist_ok=True)

    feat_path = os.path.join(data_dir, 'crypto_features_1h.parquet')
    if not os.path.exists(feat_path):
        print(f"❌ Feature file not found: {feat_path}")
        sys.exit(1)

    print("=" * 70)
    print("  HIST v2 — Sentiment-Aware Transformer")
    print("  Tech Embed + Sentiment Embed + Gated Fusion + Cross-Attn")
    print("=" * 70)

    # 1. Load & enrich
    print(f"\n📊 Loading data...")
    df = pd.read_parquet(feat_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    print(f"   Shape: {df.shape}, Symbols: {df['symbol'].nunique()}")

    df = add_multi_horizon_targets(df)
    df = add_cross_asset_features(df)
    df = add_sentiment_features(df, project_root)

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['target_ret_4h'])

    # Feature columns (include sentiment features, exclude regime-level)
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS
                 and not c.startswith('target_')
                 and c not in REGIME_COLS]
    feat_cols = [c for c in feat_cols if df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    print(f"   Features: {len(feat_cols)}")

    # List sentiment features
    sent_keywords = ['fng', 'funding', 'reversal', 'surge', 'dispersion',
                      'long_short', 'btc_beta', 'market_avg_funding', 'market_funding']
    sent_feats = [c for c in feat_cols if any(k in c for k in sent_keywords)]
    print(f"   Sentiment features ({len(sent_feats)}): {sent_feats[:10]}...")

    df[feat_cols] = df[feat_cols].fillna(0)
    df = cross_sectional_rank(df, feat_cols)

    target_col = f'target_ret_{HORIZON}h'
    df['target_rank'] = df.groupby('timestamp')[target_col].rank(pct=True)

    print(f"   Final shape: {df.shape}")

    # 2. Prepare data
    print(f"\n📐 Preparing cross-sectional data...")
    data, symbols, concept_ids = prepare_cross_section_data(
        df, feat_cols, 'target_rank', actual_return_col=target_col
    )

    # 3. Train
    model, test_preds, test_targets, test_masks, best_val_ic, best_epoch = \
        build_model_and_train(data, concept_ids, feat_cols, args)

    # 4. Evaluate
    test_timestamps = data['test']['timestamps']
    test_actual_rets = data['test']['y_actual']

    metrics = evaluate_predictions(
        test_preds, test_targets, test_masks,
        test_timestamps, symbols, HORIZON, test_actual_rets
    )

    print(f"\n   📈 HIST v2 Test Results:")
    for k, v in metrics.items():
        flag = ""
        if k == 'Rank_IC' and abs(v) > 0.02: flag = " ✓"
        if k == 'LS_Sharpe_net' and v > 1.0: flag = " ✓"
        print(f"      {k:30s} {v}{flag}")

    # 5. Save predictions (compatible with ensemble)
    rows = []
    T, N = test_preds.shape
    for t_idx, ts in enumerate(test_timestamps):
        for s_idx, sym in enumerate(symbols):
            if test_masks[t_idx, s_idx] > 0.5:
                rows.append({
                    'timestamp': ts,
                    'symbol': sym,
                    'pred_hist_v2': float(test_preds[t_idx, s_idx]),
                    f'target_ret_{HORIZON}h': float(test_actual_rets[t_idx, s_idx]),
                })

    pred_df = pd.DataFrame(rows)
    pred_df.to_parquet(os.path.join(results_dir, 'test_predictions_hist_v2.parquet'), index=False)

    # Save metrics
    all_results = {
        'metrics': metrics,
        'meta': {
            'timestamp': datetime.now().isoformat(),
            'best_epoch': best_epoch,
            'best_val_rank_ic': round(best_val_ic, 4),
            'n_features': len(feat_cols),
            'n_tech_features': len([c for c in feat_cols if c not in sent_feats]),
            'n_sent_features': len(sent_feats),
            'sent_features': sent_feats,
            'd_model': args.d_model,
            'n_heads': args.n_heads,
            'n_layers': args.n_layers,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'ic_weight': args.ic_weight,
            'device': args.device,
        },
    }

    with open(os.path.join(results_dir, 'results_hist_v2.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    try:
        import torch
        torch.save(model.state_dict(), os.path.join(results_dir, 'hist_v2_model.pt'))
        print(f"\n   💾 Model saved")
    except Exception:
        pass

    # ========================================
    # FINAL SUMMARY
    # ========================================
    print(f"\n{'='*70}")
    print(f"  HIST v2 SUMMARY")
    print(f"{'='*70}")
    print(f"   Rank IC:                  {metrics['Rank_IC']:+.4f}")
    print(f"   Rank ICIR:                {metrics['Rank_ICIR']:+.4f}")
    print(f"   LS Sharpe (raw):          {metrics['LS_Sharpe_raw']:+.2f}")
    print(f"   LS Sharpe (net):          {metrics['LS_Sharpe_net']:+.2f}")
    print(f"   LS MaxDD (net):           {metrics['LS_MaxDD_net_%']:.1f}%")
    print(f"   LS VolTarget Sharpe:      {metrics['LS_VolTarget_Sharpe']:+.2f}")
    print(f"   LS VolTarget MaxDD:       {metrics['LS_VolTarget_MaxDD_%']:.1f}%")
    print(f"   LS DDStop Sharpe:         {metrics['LS_DDStop_Sharpe']:+.2f}")
    print(f"   LS DDStop MaxDD:          {metrics['LS_DDStop_MaxDD_%']:.1f}%")
    print(f"   Best epoch:               {best_epoch}")
    print(f"   Sentiment features:       {len(sent_feats)}")
    print(f"{'='*70}")

    if metrics['LS_Sharpe_net'] > 2.0:
        print("🟢 STRONG — Sentiment boost confirmed!")
        print("   → Combine with LGB v5 for ensemble, then paper trade.")
    elif metrics['LS_Sharpe_net'] > 1.0:
        print("🟡 DECENT — Sentiment helps but marginal.")
    else:
        print("🟠 NEEDS WORK — Try different sentiment feature sets or architecture.")

    print(f"\n✅ Results saved to {results_dir}/")
    print(f"\n💡 To ensemble with LGB v5:")
    print(f"   python run_ensemble_v2.py --hist results_hist_v2/test_predictions_hist_v2.parquet \\")
    print(f"                             --lgb results_v5/test_predictions_v5.parquet")


if __name__ == '__main__':
    main()
