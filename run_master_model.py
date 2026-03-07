#!/usr/bin/env python3
"""
MASTER-style Transformer for Crypto Alpha

Architecture (adapted from Li et al., AAAI 2024 "MASTER: Market-Guided Stock Transformer"):
1. Intra-stock Gated Attention — each coin attends to its own features
2. Inter-stock Routing — dynamic routing between coins (not static concepts like HIST)
3. Market-Guided Modulation — global market state modulates coin representations
4. Temporal Momentum Gate — momentum bias for stability
5. Prediction Head — per-coin alpha score

Key differences from HIST:
- HIST uses predefined concept categories; MASTER learns dynamic routing
- HIST uses standard self-attention; MASTER uses gated attention + routing
- MASTER adds market-guided modulation (global BTC/market state affects all)
- MASTER adds temporal momentum (inertia in predictions)

Usage:
  python run_master_model.py --device cuda --epochs 100
  python run_master_model.py --device cuda --lgb-preds results_v4/test_predictions_v4.parquet
  python run_master_model.py --device cuda --hist-preds results_hist/test_predictions_hist.parquet

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

REGIME_COLS = {
    'btc_regime_24', 'btc_regime_72', 'btc_regime_168',
    'regime_btc_above_ma336', 'regime_btc_above_ma720',
    'regime_btc_ma720_slope', 'regime_btc_not_crashed',
    'regime_btc_dd_720', 'regime_low_vol',
    'regime_breadth_bullish', 'breadth_pct_positive',
    'regime_composite',
}

# Market factor feature names (used for market-guided modulation)
MARKET_FEATURES = [
    'btc_ret_1h', 'btc_ret_4h', 'btc_ret_24h', 'btc_ret_168h',
    'btc_vol_24h', 'market_dispersion', 'eth_ret_24h',
]


# ============================================================
# FEATURE ENGINEERING (shared with HIST/v4)
# ============================================================

def add_multi_horizon_targets(df):
    print("   🎯 Adding targets...")
    for h in [4, 12, 24]:
        df[f'target_ret_{h}h'] = df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change(h).shift(-h)
        )
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

def prepare_cross_section_data(df, feat_cols, target_col, market_feat_cols):
    """
    Convert flat DataFrame to cross-sectional format.
    Also extracts market features as a separate tensor.
    """
    symbols = sorted(df['symbol'].unique())
    sym2idx = {s: i for i, s in enumerate(symbols)}
    N = len(symbols)
    F = len(feat_cols)
    M = len(market_feat_cols)

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
        mask = np.zeros((T, N), dtype=np.float32)
        market = np.zeros((T, M), dtype=np.float32)

        split_df['_ti'] = split_df['timestamp'].map(ts2idx)
        split_df['_si'] = split_df['symbol'].map(sym2idx)

        ti = split_df['_ti'].values
        si = split_df['_si'].values

        for i, col in enumerate(feat_cols):
            X[ti, si, i] = split_df[col].values.astype(np.float32)

        y_arr[ti, si] = split_df[target_col].values.astype(np.float32)
        mask[ti, si] = 1.0

        # Market features: take from BTC/USDT rows (or first available)
        btc_rows = split_df[split_df['symbol'] == 'BTC/USDT'].copy()
        if len(btc_rows) > 0:
            for i, col in enumerate(market_feat_cols):
                if col in btc_rows.columns:
                    btc_map = btc_rows.set_index('_ti')[col].to_dict()
                    for t_idx in range(T):
                        market[t_idx, i] = btc_map.get(t_idx, 0.0)

        X = np.nan_to_num(X, nan=0.0)
        y_arr = np.nan_to_num(y_arr, nan=0.0)
        market = np.nan_to_num(market, nan=0.0)

        result[split_name] = {
            'X': X, 'y': y_arr, 'mask': mask,
            'market': market, 'timestamps': timestamps,
        }

        print(f"   {split_name}: {T} timestamps × {N} coins × {F} features "
              f"+ {M} market features")

    return result, symbols


# ============================================================
# MASTER MODEL (PyTorch)
# ============================================================

def build_model_and_train(data, args):
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import Dataset, DataLoader
    except ImportError:
        print("❌ PyTorch not installed.")
        print(f"   Fix: {sys.executable} -m pip install torch")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    print(f"\n   🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # ---- Dataset ----
    class CrossSectionDataset(Dataset):
        def __init__(self, X, y, mask, market):
            self.X = torch.FloatTensor(X)
            self.y = torch.FloatTensor(y)
            self.mask = torch.FloatTensor(mask)
            self.market = torch.FloatTensor(market)

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx], self.mask[idx], self.market[idx]

    # ---- MASTER Model ----
    class MASTERModel(nn.Module):
        """
        Market-Guided Stock Transformer (adapted for crypto).

        Key components:
        1. Gated Feature Attention: learns which features matter per coin
        2. Dynamic Routing: soft clustering (like Mixture of Experts)
        3. Inter-Stock Transformer: cross-stock attention with routing bias
        4. Market Modulation: global market state conditions all coin representations
        5. Momentum Gate: EMA-like temporal smoothing for stable predictions
        """
        def __init__(self, n_features, n_market, d_model=128, n_heads=4,
                     n_layers=2, n_routes=4, dropout=0.1):
            super().__init__()
            self.d_model = d_model
            self.n_routes = n_routes

            # ---- 1. Gated Feature Attention ----
            # Each coin learns which of its features are important
            self.feat_embed = nn.Sequential(
                nn.Linear(n_features, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.feat_gate = nn.Sequential(
                nn.Linear(n_features, d_model),
                nn.Sigmoid(),
            )
            self.feat_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            # ---- 2. Dynamic Routing (soft clustering) ----
            # Instead of predefined concepts (HIST), learn K routing centroids
            self.route_centroids = nn.Parameter(torch.randn(n_routes, d_model) * 0.02)
            self.route_proj = nn.Linear(d_model, d_model)

            # ---- 3. Inter-Stock Transformer ----
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation='gelu',
            )
            self.inter_stock_attn = nn.TransformerEncoder(
                encoder_layer, num_layers=n_layers
            )

            # ---- 4. Market-Guided Modulation ----
            self.market_encoder = nn.Sequential(
                nn.Linear(n_market, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
            # FiLM-style modulation: market generates (gamma, beta)
            self.market_gamma = nn.Linear(d_model, d_model)
            self.market_beta = nn.Linear(d_model, d_model)

            # ---- 5. Prediction Head ----
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )

            # ---- 6. Momentum Gate ----
            self.momentum_alpha = nn.Parameter(torch.tensor(0.3))

        def forward(self, x, market, prev_pred=None):
            """
            x: (B, N, F) — coin features
            market: (B, M) — global market features
            prev_pred: (B, N) or None — previous prediction for momentum
            Returns: (B, N)
            """
            B, N, F = x.shape

            # 1. Gated Feature Attention
            h = self.feat_embed(x)  # (B, N, d_model)
            gate = self.feat_gate(x)  # (B, N, d_model)
            h = h * gate  # Feature selection
            h = self.feat_proj(h)  # (B, N, d_model)

            # 2. Dynamic Routing
            # Soft assignment of coins to K routes
            h_route = self.route_proj(h)  # (B, N, d_model)
            # (B, N, d_model) @ (d_model, K) => (B, N, K)
            route_scores = torch.matmul(h_route, self.route_centroids.T) / (self.d_model ** 0.5)
            route_probs = torch.softmax(route_scores, dim=-1)  # (B, N, K)

            # Route-aggregated info: each coin gets from its route
            # (B, N, K) @ (K, d_model) => (B, N, d_model) — route context
            route_context = torch.matmul(route_probs, self.route_centroids)
            h = h + route_context  # Add routing info

            # 3. Inter-Stock Transformer
            h = self.inter_stock_attn(h)  # (B, N, d_model)

            # 4. Market-Guided Modulation (FiLM)
            m = self.market_encoder(market)  # (B, d_model)
            gamma = self.market_gamma(m).unsqueeze(1)  # (B, 1, d_model)
            beta = self.market_beta(m).unsqueeze(1)   # (B, 1, d_model)
            h = gamma * h + beta  # Modulate

            # 5. Predict
            pred = self.head(h).squeeze(-1)  # (B, N)

            # 6. Momentum Gate
            if prev_pred is not None:
                alpha = torch.sigmoid(self.momentum_alpha)
                pred = alpha * prev_pred + (1 - alpha) * pred

            return pred

    # ---- Loss functions ----
    def ic_loss(pred, target, mask):
        """Negative Spearman-approx IC loss per cross-section."""
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

    def rank_loss(pred, target, mask, margin=0.01):
        """
        Pairwise ranking loss: pred should rank pairs in same order as target.
        Lightweight: sample random pairs per cross-section.
        """
        losses = []
        B = pred.shape[0]
        for i in range(B):
            m = mask[i] > 0.5
            if m.sum() < 10:
                continue
            p = pred[i][m]
            t = target[i][m]

            # Sample pairs
            n = min(int(m.sum()), 30)
            idx = torch.randperm(m.sum(), device=pred.device)[:n]
            idx2 = torch.randperm(m.sum(), device=pred.device)[:n]

            p_diff = p[idx] - p[idx2]
            t_diff = t[idx] - t[idx2]
            target_sign = torch.sign(t_diff)

            loss = torch.clamp(margin - target_sign * p_diff, min=0).mean()
            losses.append(loss)

        if not losses:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return torch.stack(losses).mean()

    def mse_loss_masked(pred, target, mask):
        m = mask > 0.5
        if m.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return F.mse_loss(pred[m], target[m])

    # ---- Build everything ----
    n_features = data['train']['X'].shape[2]
    n_market = data['train']['market'].shape[1]

    train_ds = CrossSectionDataset(
        data['train']['X'], data['train']['y'],
        data['train']['mask'], data['train']['market'])
    val_ds = CrossSectionDataset(
        data['val']['X'], data['val']['y'],
        data['val']['mask'], data['val']['market'])
    test_ds = CrossSectionDataset(
        data['test']['X'], data['test']['y'],
        data['test']['mask'], data['test']['market'])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    model = MASTERModel(
        n_features=n_features,
        n_market=n_market,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_routes=args.n_routes,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model params: {n_params:,}")
    print(f"   Architecture: gated_feat({n_features}→{args.d_model}) + "
          f"routing({args.n_routes}) + inter_stock({args.n_layers}L,{args.n_heads}H) + "
          f"market_mod({n_market}) + head")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Cosine schedule with warmup
    warmup_epochs = min(5, args.epochs // 10)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Training loop ----
    print(f"\n{'='*70}")
    print(f"  TRAINING MASTER ({args.epochs} epochs, BS={args.batch_size}, LR={args.lr})")
    print(f"{'='*70}")

    best_val_ic = -999
    best_epoch = 0
    patience_counter = 0
    best_state = None

    # Loss weights
    w_ic = args.ic_weight
    w_rank = args.rank_weight
    w_mse = 1.0 - w_ic - w_rank

    for epoch in range(args.epochs):
        model.train()
        train_losses = []

        for X_batch, y_batch, mask_batch, market_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)
            market_batch = market_batch.to(device)

            pred = model(X_batch, market_batch)

            loss = (w_mse * mse_loss_masked(pred, y_batch, mask_batch) +
                    w_ic * ic_loss(pred, y_batch, mask_batch) +
                    w_rank * rank_loss(pred, y_batch, mask_batch))

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
            for X_batch, y_batch, mask_batch, market_batch in val_loader:
                X_batch = X_batch.to(device)
                market_batch = market_batch.to(device)
                pred = model(X_batch, market_batch)
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
        val_icir = val_rank_ic / (np.std(rank_ics) + 1e-10) if rank_ics else 0

        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"   Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Loss: {np.mean(train_losses):.5f} | "
                  f"Val Rank IC: {val_rank_ic:.4f} | "
                  f"Val ICIR: {val_icir:.3f} | "
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

    # Load best
    model.load_state_dict(best_state)
    model.eval()

    # ---- Test ----
    print(f"\n{'='*70}")
    print(f"  TEST EVALUATION")
    print(f"{'='*70}")

    test_preds_all, test_targets_all, test_masks_all = [], [], []
    with torch.no_grad():
        for X_batch, y_batch, mask_batch, market_batch in test_loader:
            X_batch = X_batch.to(device)
            market_batch = market_batch.to(device)
            pred = model(X_batch, market_batch)
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

def evaluate_predictions(test_preds, test_targets, test_masks, horizon_hours=4):
    """Evaluate 3D predictions."""
    periods_per_day = 24 // horizon_hours
    periods_per_year = periods_per_day * 365
    T, N = test_preds.shape

    rank_ics, ics, ls_rets, lo5_rets, lo10_rets = [], [], [], [], []

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
        ics.append(ic if not np.isnan(ic) else 0)
        rank_ics.append(ric if not np.isnan(ric) else 0)

        order = np.argsort(-pv)
        sorted_actual = av[order]
        n_q = max(len(pv) // 5, 1)

        ls_rets.append(sorted_actual[:n_q].mean() - sorted_actual[-n_q:].mean())
        lo5_rets.append(sorted_actual[:min(5, len(sorted_actual))].mean())
        lo10_rets.append(sorted_actual[:min(10, len(sorted_actual))].mean())

    rank_ics = np.array(rank_ics)
    ics = np.array(ics)
    ls_rets = np.array(ls_rets)
    lo5 = np.array(lo5_rets) - 0.0008
    lo10 = np.array(lo10_rets) - 0.0008

    def sharpe(r, ppyr):
        if len(r) == 0 or r.std() < 1e-12:
            return 0.0
        return (r.mean() / (r.std() + 1e-10)) * np.sqrt(ppyr)

    def max_dd(r):
        if len(r) == 0:
            return 0.0
        cum = np.cumprod(1 + r)
        running_max = np.maximum.accumulate(cum)
        dd = cum / running_max - 1
        return np.min(dd) if len(dd) > 0 else 0.0

    # Daily IC for ICIR
    n_per_day = periods_per_day
    daily_rics = []
    for i in range(0, len(rank_ics) - n_per_day + 1, n_per_day):
        daily_rics.append(rank_ics[i:i+n_per_day].mean())
    daily_rics = np.array(daily_rics)
    rank_icir = (daily_rics.mean() / (daily_rics.std() + 1e-10)) if len(daily_rics) > 1 else 0

    metrics = {
        'IC': round(float(ics.mean()), 4),
        'Rank_IC': round(float(rank_ics.mean()), 4),
        'ICIR': round(float(ics.mean() / (ics.std() + 1e-10)), 4),
        'Rank_ICIR': round(float(rank_icir), 4),
        'LS_Sharpe': round(float(sharpe(ls_rets, periods_per_year)), 2),
        'LS_Ann_Return_%': round(float(ls_rets.mean() * periods_per_year * 100), 1),
        'LS_MaxDD_%': round(float(max_dd(ls_rets) * 100), 1),
        'LO5_Sharpe': round(float(sharpe(lo5, periods_per_year)), 2),
        'LO5_Total_%': round(float(np.prod(1 + lo5) * 100 - 100), 1) if len(lo5) > 0 else 0,
        'LO10_Sharpe': round(float(sharpe(lo10, periods_per_year)), 2),
        'N_periods': len(ls_rets),
    }
    return metrics


def flatten_predictions(test_preds, test_targets, test_masks, test_timestamps,
                        symbols, target_col):
    """Convert 3D predictions back to flat DataFrame."""
    rows = []
    T, N = test_preds.shape
    for t_idx, ts in enumerate(test_timestamps):
        for s_idx, sym in enumerate(symbols):
            if test_masks[t_idx, s_idx] > 0.5:
                rows.append({
                    'timestamp': ts,
                    'symbol': sym,
                    'pred_master': float(test_preds[t_idx, s_idx]),
                    target_col: float(test_targets[t_idx, s_idx]),
                })
    return pd.DataFrame(rows)


def multi_model_ensemble(master_df, lgb_path=None, hist_path=None, horizon=4):
    """
    Combine MASTER + HIST + LGB predictions.
    Normalize each to z-scores, then weighted average.
    """
    target_col = f'target_ret_{horizon}h'

    dfs = {'master': master_df[['timestamp', 'symbol', 'pred_master', target_col]].copy()}

    # Load LGB
    if lgb_path and os.path.exists(lgb_path):
        lgb_df = pd.read_parquet(lgb_path)
        lgb_df['timestamp'] = pd.to_datetime(lgb_df['timestamp'], utc=True)
        # Find best pred column
        for col in ['pred_ensemble', 'pred_hpo', 'pred_selected', 'pred_baseline']:
            if col in lgb_df.columns:
                lgb_df = lgb_df.rename(columns={col: 'pred_lgb'})
                break
        if 'pred_lgb' in lgb_df.columns:
            dfs['lgb'] = lgb_df[['timestamp', 'symbol', 'pred_lgb']].copy()
            print(f"   LGB: {len(dfs['lgb']):,} rows")

    # Load HIST
    if hist_path and os.path.exists(hist_path):
        hist_df = pd.read_parquet(hist_path)
        hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'], utc=True)
        if 'pred_hist' in hist_df.columns:
            dfs['hist'] = hist_df[['timestamp', 'symbol', 'pred_hist']].copy()
            print(f"   HIST: {len(dfs['hist']):,} rows")

    # Merge all
    merged = dfs['master'].copy()
    for name, other_df in dfs.items():
        if name == 'master':
            continue
        merged = merged.merge(other_df, on=['timestamp', 'symbol'], how='inner')

    pred_cols = [c for c in merged.columns if c.startswith('pred_')]
    print(f"   Models: {pred_cols}")
    print(f"   Merged rows: {len(merged):,}")

    # Z-score normalize each
    for col in pred_cols:
        merged[col] = (merged[col] - merged[col].mean()) / (merged[col].std() + 1e-10)

    # Weighted ensemble
    n_models = len(pred_cols)
    merged['pred_final_ensemble'] = sum(merged[c] for c in pred_cols) / n_models

    # Evaluate each
    periods_per_year = (24 // horizon) * 365
    results = {}
    for col in pred_cols + ['pred_final_ensemble']:
        rics = []
        ls_rets = []
        for ts, grp in merged.groupby('timestamp'):
            if len(grp) < 10:
                continue
            c, _ = spearmanr(grp[col].values, grp[target_col].values)
            if not np.isnan(c):
                rics.append(c)
            grp_sorted = grp.sort_values(col, ascending=False)
            n_q = max(len(grp_sorted) // 5, 1)
            lr = grp_sorted.head(n_q)[target_col].mean()
            sr = grp_sorted.tail(n_q)[target_col].mean()
            ls_rets.append(lr - sr)

        rics = np.array(rics)
        ls_rets = np.array(ls_rets)
        ls_sharpe = (ls_rets.mean() / (ls_rets.std() + 1e-10)) * np.sqrt(periods_per_year)

        results[col] = {
            'Rank_IC': round(float(rics.mean()), 4),
            'LS_Sharpe': round(float(ls_sharpe), 2),
        }
        print(f"   {col}: Rank IC={rics.mean():.4f}, LS Sharpe={ls_sharpe:.2f}")

    return merged, results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='MASTER Transformer for Crypto Alpha')
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--results', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--n-routes', type=int, default=4,
                        help='Number of dynamic routing centroids')
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--ic-weight', type=float, default=0.4)
    parser.add_argument('--rank-weight', type=float, default=0.2)
    parser.add_argument('--lgb-preds', type=str, default=None)
    parser.add_argument('--hist-preds', type=str, default=None)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data or os.path.join(project_root, 'data', 'features')
    results_dir = args.results or os.path.join(project_root, 'results_master')
    os.makedirs(results_dir, exist_ok=True)

    feat_path = os.path.join(data_dir, 'crypto_features_1h.parquet')
    if not os.path.exists(feat_path):
        print(f"❌ Feature file not found: {feat_path}")
        sys.exit(1)

    print("=" * 70)
    print("  MASTER TRANSFORMER — Crypto Alpha Model")
    print("  Market-Guided + Dynamic Routing + Gated Attention + Rank Loss")
    print("=" * 70)

    # ========================================
    # 1. LOAD DATA
    # ========================================
    print(f"\n📊 Loading data...")
    df = pd.read_parquet(feat_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    print(f"   Shape: {df.shape}, Symbols: {df['symbol'].nunique()}")

    df = add_multi_horizon_targets(df)
    df = add_cross_asset_features(df)

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=['target_ret_4h'])

    # Feature columns
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS
                 and not c.startswith('target_')
                 and c not in REGIME_COLS]
    feat_cols = [c for c in feat_cols if df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

    # Market feature columns (will be extracted separately)
    market_feat_cols = [c for c in MARKET_FEATURES if c in df.columns]
    print(f"   Features: {len(feat_cols)}, Market features: {len(market_feat_cols)}")

    df[feat_cols] = df[feat_cols].fillna(0)
    df = cross_sectional_rank(df, feat_cols)

    target_col = f'target_ret_{HORIZON}h'
    df['target_rank'] = df.groupby('timestamp')[target_col].rank(pct=True)
    print(f"   Final shape: {df.shape}")

    # ========================================
    # 2. PREPARE DATA
    # ========================================
    print(f"\n📐 Preparing cross-sectional data...")
    data, symbols = prepare_cross_section_data(
        df, feat_cols, 'target_rank', market_feat_cols
    )

    # ========================================
    # 3. TRAIN
    # ========================================
    model, test_preds, test_targets, test_masks, best_val_ic, best_epoch = \
        build_model_and_train(data, args)

    # ========================================
    # 4. EVALUATE
    # ========================================
    test_timestamps = data['test']['timestamps']

    metrics = evaluate_predictions(test_preds, test_targets, test_masks, HORIZON)

    print(f"\n   📈 MASTER Test Results:")
    for k, v in metrics.items():
        flag = ""
        if k == 'Rank_IC' and abs(v) > 0.02: flag = " ✓"
        if k == 'LS_Sharpe' and v > 1.0: flag = " ✓"
        if k == 'ICIR' and abs(v) > 0.3: flag = " ✓"
        print(f"      {k:25s} {v}{flag}")

    # ========================================
    # 5. MULTI-MODEL ENSEMBLE
    # ========================================
    master_flat = flatten_predictions(
        test_preds, test_targets, test_masks,
        test_timestamps, symbols, target_col
    )

    # Auto-discover prediction files
    lgb_path = args.lgb_preds
    hist_path = args.hist_preds
    if lgb_path is None:
        for candidate in ['results_v4/test_predictions_v4.parquet',
                          'results_v3/test_predictions_v3.parquet']:
            if os.path.exists(os.path.join(project_root, candidate)):
                lgb_path = os.path.join(project_root, candidate)
                break
    if hist_path is None:
        candidate = os.path.join(project_root, 'results_hist/test_predictions_hist.parquet')
        if os.path.exists(candidate):
            hist_path = candidate

    ensemble_results = {}
    if lgb_path or hist_path:
        print(f"\n{'='*70}")
        print(f"  MULTI-MODEL ENSEMBLE")
        print(f"{'='*70}")
        merged, ensemble_results = multi_model_ensemble(
            master_flat, lgb_path, hist_path, HORIZON
        )
        # Save ensemble predictions
        merged.to_parquet(
            os.path.join(results_dir, 'test_predictions_ensemble_final.parquet'),
            index=False
        )

    # ========================================
    # 6. SAVE
    # ========================================
    master_flat.to_parquet(
        os.path.join(results_dir, 'test_predictions_master.parquet'), index=False
    )

    all_results = {
        'master_metrics': metrics,
        'ensemble_results': ensemble_results,
        'meta': {
            'timestamp': datetime.now().isoformat(),
            'best_epoch': best_epoch,
            'best_val_rank_ic': round(best_val_ic, 4),
            'n_features': len(feat_cols),
            'n_market_features': len(market_feat_cols),
            'd_model': args.d_model,
            'n_heads': args.n_heads,
            'n_layers': args.n_layers,
            'n_routes': args.n_routes,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'ic_weight': args.ic_weight,
            'rank_weight': args.rank_weight,
            'device': args.device,
            'symbols': symbols,
        },
    }

    with open(os.path.join(results_dir, 'results_master.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    try:
        import torch
        torch.save(model.state_dict(), os.path.join(results_dir, 'master_model.pt'))
        print(f"\n   💾 Model saved to {results_dir}/master_model.pt")
    except Exception:
        pass

    # ========================================
    # FINAL
    # ========================================
    print(f"\n{'='*70}")
    print(f"  MASTER RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"   Rank IC:           {metrics['Rank_IC']:+.4f}")
    print(f"   Rank ICIR:         {metrics['Rank_ICIR']:+.4f}")
    print(f"   LS Sharpe:         {metrics['LS_Sharpe']:+.2f}")
    print(f"   LS Ann Return:     {metrics['LS_Ann_Return_%']:+.1f}%")
    print(f"   LS Max Drawdown:   {metrics['LS_MaxDD_%']:.1f}%")
    print(f"   Best epoch:        {best_epoch}")

    if ensemble_results and 'pred_final_ensemble' in ensemble_results:
        er = ensemble_results['pred_final_ensemble']
        print(f"   ---")
        print(f"   Final Ensemble:    Rank IC={er['Rank_IC']}, LS Sharpe={er['LS_Sharpe']}")

    print(f"{'='*70}")

    if metrics['LS_Sharpe'] > 2.0:
        print("🟢 STRONG — MASTER transformer works!")
    elif metrics['LS_Sharpe'] > 1.0:
        print("🟡 DECENT — Try different hyperparams.")
    else:
        print("🟠 NEEDS TUNING — Adjust d_model, routing, loss weights.")

    print(f"\n✅ Results saved to {results_dir}/")

    # Instructions
    print(f"\n💡 Run with all 3 models:")
    print(f"   python run_master_model.py --device cuda \\")
    print(f"     --lgb-preds results_v4/test_predictions_v4.parquet \\")
    print(f"     --hist-preds results_hist/test_predictions_hist.parquet")


if __name__ == '__main__':
    main()
