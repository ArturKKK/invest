#!/usr/bin/env python3
"""
HIST-style Transformer for Crypto Alpha

Architecture (adapted from Xu et al., KDD 2021):
1. Feature Embedding — MLP per coin → d_model
2. Concept Attention — predefined crypto categories (L1, DeFi, Gaming, etc.)
3. Cross-Stock Self-Attention — learn hidden relationships between 50 coins
4. Prediction Head — per-coin alpha score

Key differences from original HIST:
- Crypto-specific concept categories instead of stock sectors
- Pre-computed features (98 TA + cross-asset) instead of raw OHLCV
- Ranking loss (IC-based) combined with MSE

Usage:
  python run_hist_model.py                              # Full training
  python run_hist_model.py --epochs 100 --batch-size 128
  python run_hist_model.py --device cuda                # GPU training
  python run_hist_model.py --data /path/to/features
  python run_hist_model.py --lgb-preds results_v4/test_predictions_v4.parquet  # ensemble with LGB

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

# Crypto concept categories (predefined, for HIST concept attention)
# 8 categories: 0=BTC, 1=L1, 2=DeFi, 3=Gaming, 4=L2, 5=Infra, 6=Meme, 7=Other
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
}


# ============================================================
# FEATURE ENGINEERING (same as v4, but also reusable here)
# ============================================================

def add_multi_horizon_targets(df):
    print("   🎯 Adding targets...")
    for h in [4, 12, 24]:
        df[f'target_ret_{h}h'] = df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change(h).shift(-h)
        )
    return df


def add_cross_asset_features(df):
    """Minimal cross-asset features for HIST (BTC/ETH)."""
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
# DATA PREPARATION (flat → 3D tensors)
# ============================================================

def prepare_cross_section_data(df, feat_cols, target_col):
    """
    Convert flat DataFrame to cross-sectional format.

    Returns:
        X: dict {split: (T, N, F)} — features per timestamp
        y: dict {split: (T, N)} — targets per timestamp
        masks: dict {split: (T, N)} — valid coin mask
        timestamps: dict {split: list}
        symbols: list of symbol names (sorted)
        concept_ids: (N,) — concept index per coin
    """
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
        split_df['ts_idx_local'] = split_df.groupby('symbol').cumcount()  # not used

        # Get unique timestamps sorted
        timestamps = sorted(split_df['timestamp'].unique())
        ts2idx = {ts: i for i, ts in enumerate(timestamps)}
        T = len(timestamps)

        X = np.zeros((T, N, F), dtype=np.float32)
        y_arr = np.full((T, N), np.nan, dtype=np.float32)
        mask = np.zeros((T, N), dtype=np.float32)

        # Vectorized fill
        split_df['_ti'] = split_df['timestamp'].map(ts2idx)
        split_df['_si'] = split_df['symbol'].map(sym2idx)

        ti = split_df['_ti'].values
        si = split_df['_si'].values

        for i, col in enumerate(feat_cols):
            X[ti, si, i] = split_df[col].values.astype(np.float32)

        y_arr[ti, si] = split_df[target_col].values.astype(np.float32)
        mask[ti, si] = 1.0

        # Replace NaN in X with 0
        X = np.nan_to_num(X, nan=0.0)
        y_arr = np.nan_to_num(y_arr, nan=0.0)

        result[split_name] = {
            'X': X, 'y': y_arr, 'mask': mask,
            'timestamps': timestamps,
        }

        print(f"   {split_name}: {T} timestamps × {N} coins × {F} features = {T*N*F:,} values")

    return result, symbols, concept_ids


# ============================================================
# HIST MODEL (PyTorch)
# ============================================================

def build_model_and_train(data, concept_ids, args):
    """Build and train HIST model."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import Dataset, DataLoader
    except ImportError:
        print("❌ PyTorch not installed. Install with:")
        print("   pip install torch")
        print("   or: conda install pytorch -c pytorch")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    print(f"\n   🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # ---- Dataset ----
    class CrossSectionDataset(Dataset):
        def __init__(self, X, y, mask):
            self.X = torch.FloatTensor(X)
            self.y = torch.FloatTensor(y)
            self.mask = torch.FloatTensor(mask)

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx], self.mask[idx]

    # ---- HIST Model ----
    class HISTModel(nn.Module):
        def __init__(self, n_features, d_model=128, n_heads=4, n_layers=2,
                     n_concepts=8, dropout=0.1):
            super().__init__()
            self.d_model = d_model

            # Feature embedding
            self.embed = nn.Sequential(
                nn.Linear(n_features, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            # Concept embeddings (predefined crypto categories)
            self.concept_embed = nn.Embedding(n_concepts, d_model)
            self.concept_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid(),
            )

            # Cross-stock self-attention (hidden relationships)
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

        def forward(self, x, concept_ids):
            """
            x: (B, N, F) — batch of cross-sections
            concept_ids: (N,) — concept index per coin
            Returns: (B, N) — predicted score per coin
            """
            B, N, F = x.shape

            # 1. Feature embedding
            h = self.embed(x)  # (B, N, d_model)

            # 2. Concept attention (shared information)
            # Each coin gets info from its concept group
            concepts = self.concept_embed(concept_ids)  # (N, d_model)
            concepts = concepts.unsqueeze(0).expand(B, -1, -1)  # (B, N, d_model)

            # Gate: how much concept info to incorporate
            gate_input = torch.cat([h, concepts], dim=-1)  # (B, N, d_model*2)
            gate = self.concept_gate(gate_input)  # (B, N, d_model)
            h_with_concept = h + gate * concepts  # (B, N, d_model)

            # 3. Cross-stock self-attention (hidden relationships)
            hidden = self.cross_attn(h_with_concept)  # (B, N, d_model)

            # Gate: how much hidden info to incorporate
            hidden_gate_input = torch.cat([h_with_concept, hidden], dim=-1)
            hgate = self.hidden_gate(hidden_gate_input)
            h_final = h_with_concept + hgate * (hidden - h_with_concept)

            # 4. Predict
            out = self.head(h_final).squeeze(-1)  # (B, N)
            return out

    # ---- Loss functions ----
    def ic_loss(pred, target, mask):
        """Negative Pearson correlation loss (per cross-section)."""
        # Only compute on valid coins
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
        """MSE loss only on valid coins."""
        m = mask > 0.5
        if m.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return F.mse_loss(pred[m], target[m])

    # ---- Build everything ----
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

    model = HISTModel(
        n_features=n_features,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_concepts=N_CONCEPTS,
        dropout=args.dropout,
    ).to(device)

    concept_tensor = torch.LongTensor(concept_ids).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model params: {n_params:,}")
    print(f"   Architecture: embed({n_features}→{args.d_model}) + "
          f"concept({N_CONCEPTS}) + cross_attn({args.n_layers}L,{args.n_heads}H) + head")

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
    print(f"  TRAINING HIST ({args.epochs} epochs, BS={args.batch_size}, LR={args.lr})")
    print(f"{'='*70}")

    best_val_ic = -999
    best_epoch = 0
    patience_counter = 0
    best_state = None

    ic_weight = args.ic_weight
    mse_weight = 1.0 - ic_weight

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_losses = []
        for X_batch, y_batch, mask_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)

            pred = model(X_batch, concept_tensor)

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
                pred = model(X_batch, concept_tensor)
                val_preds_all.append(pred.cpu().numpy())
                val_targets_all.append(y_batch.numpy())
                val_masks_all.append(mask_batch.numpy())

        val_preds = np.concatenate(val_preds_all, axis=0)
        val_targets = np.concatenate(val_targets_all, axis=0)
        val_masks = np.concatenate(val_masks_all, axis=0)

        # Compute Rank IC per cross-section, then average
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

    # ---- Test evaluation ----
    print(f"\n{'='*70}")
    print(f"  TEST EVALUATION")
    print(f"{'='*70}")

    test_preds_all, test_targets_all, test_masks_all = [], [], []
    with torch.no_grad():
        for X_batch, y_batch, mask_batch in test_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch, concept_tensor)
            test_preds_all.append(pred.cpu().numpy())
            test_targets_all.append(y_batch.numpy())
            test_masks_all.append(mask_batch.numpy())

    test_preds = np.concatenate(test_preds_all, axis=0)
    test_targets = np.concatenate(test_targets_all, axis=0)
    test_masks = np.concatenate(test_masks_all, axis=0)

    return model, test_preds, test_targets, test_masks, best_val_ic, best_epoch


# ============================================================
# EVALUATION (same metrics as LightGBM pipeline)
# ============================================================

def evaluate_hist_predictions(test_preds, test_targets, test_masks, test_timestamps,
                               symbols, target_col_name, horizon_hours=4):
    """
    Evaluate HIST predictions using same metrics as LightGBM pipeline.
    test_preds: (T, N)
    test_targets: (T, N)
    test_masks: (T, N)
    """
    periods_per_day = 24 // horizon_hours
    periods_per_year = periods_per_day * 365

    # per-cross-section metrics
    rank_ics = []
    ics = []
    ls_rets = []
    lo5_rets = []
    lo10_rets = []

    T, N = test_preds.shape

    for t in range(T):
        m = test_masks[t] > 0.5
        if m.sum() < 10:
            continue

        p = test_preds[t][m]
        a = test_targets[t][m]

        # IC
        valid = ~(np.isnan(p) | np.isnan(a))
        if valid.sum() < 10:
            continue

        pv, av = p[valid], a[valid]
        ic = np.corrcoef(pv, av)[0, 1]
        ric, _ = spearmanr(pv, av)
        ics.append(ic)
        rank_ics.append(ric)

        # Sort by prediction (descending)
        order = np.argsort(-pv)
        sorted_actual = av[order]

        n_quintile = max(len(pv) // 5, 1)
        long_ret = sorted_actual[:n_quintile].mean()
        short_ret = sorted_actual[-n_quintile:].mean()
        ls_rets.append(long_ret - short_ret)

        lo5_rets.append(sorted_actual[:5].mean())
        lo10_rets.append(sorted_actual[:10].mean())

    rank_ics = np.array(rank_ics)
    ics = np.array(ics)
    ls_rets = np.array(ls_rets)
    lo5 = np.array(lo5_rets) - 0.0008  # commission
    lo10 = np.array(lo10_rets) - 0.0008

    def sharpe(r, ppyr):
        return (r.mean() / (r.std() + 1e-10)) * np.sqrt(ppyr)

    def max_dd(r):
        cum = np.cumprod(1 + r)
        return np.min(cum / np.maximum.accumulate(cum) - 1)

    def total_ret(r):
        return np.prod(1 + r) - 1

    # Daily aggregation for ICIR
    # Group by every `periods_per_day` periods
    n_days = len(rank_ics) // periods_per_day
    daily_rics = []
    for d in range(n_days):
        start = d * periods_per_day
        end = start + periods_per_day
        daily_rics.append(rank_ics[start:end].mean())
    daily_rics = np.array(daily_rics)
    rank_icir = daily_rics.mean() / (daily_rics.std() + 1e-10) if len(daily_rics) > 0 else 0

    metrics = {
        'IC': round(float(ics.mean()), 4),
        'Rank_IC': round(float(rank_ics.mean()), 4),
        'ICIR': round(float(ics.mean() / (ics.std() + 1e-10)), 4),
        'Rank_ICIR': round(float(rank_icir), 4),
        'LS_Sharpe': round(float(sharpe(ls_rets, periods_per_year)), 2),
        'LS_Ann_Return_%': round(float(ls_rets.mean() * periods_per_year * 100), 1),
        'LS_MaxDD_%': round(float(max_dd(ls_rets) * 100), 1),
        'LO5_Sharpe': round(float(sharpe(lo5, periods_per_year)), 2),
        'LO5_Total_%': round(float(total_ret(lo5) * 100), 1),
        'LO10_Sharpe': round(float(sharpe(lo10, periods_per_year)), 2),
        'LO10_Total_%': round(float(total_ret(lo10) * 100), 1),
        'N_periods': len(ls_rets),
    }
    return metrics


def create_ensemble_with_lgb(test_preds_hist, test_masks, test_timestamps,
                              symbols, df_test, feat_cols, target_col,
                              lgb_pred_path=None, horizon_hours=4):
    """
    Combine HIST predictions with LightGBM predictions.
    Weight: 50% HIST + 50% LGB (or configurable).
    """
    if lgb_pred_path and os.path.exists(lgb_pred_path):
        print(f"\n   📦 Loading LGB predictions from {lgb_pred_path}")
        lgb_df = pd.read_parquet(lgb_pred_path)

        # Find prediction column
        pred_cols = [c for c in lgb_df.columns if c.startswith('pred_')]
        if not pred_cols:
            print("   ⚠️  No prediction columns found in LGB file")
            return None

        # Prefer ensemble > hpo > baseline
        for pref in ['pred_ensemble', 'pred_hpo', 'pred_baseline']:
            if pref in pred_cols:
                lgb_pred_col = pref
                break
        else:
            lgb_pred_col = pred_cols[0]

        print(f"   Using LGB column: {lgb_pred_col}")

        # Reconstruct HIST predictions as flat DataFrame
        sym2idx = {s: i for i, s in enumerate(symbols)}
        T, N = test_preds_hist.shape

        rows = []
        for t_idx, ts in enumerate(test_timestamps):
            for s_idx, sym in enumerate(symbols):
                if test_masks[t_idx, s_idx] > 0.5:
                    rows.append({
                        'timestamp': ts,
                        'symbol': sym,
                        'pred_hist': float(test_preds_hist[t_idx, s_idx]),
                    })

        hist_df = pd.DataFrame(rows)
        hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'], utc=True)
        lgb_df['timestamp'] = pd.to_datetime(lgb_df['timestamp'], utc=True)

        # Merge
        merged = hist_df.merge(
            lgb_df[['timestamp', 'symbol', lgb_pred_col]],
            on=['timestamp', 'symbol'],
            how='inner',
        )

        if len(merged) == 0:
            print("   ⚠️  No matching timestamps between HIST and LGB")
            return None

        # Normalize predictions to same scale
        for col in ['pred_hist', lgb_pred_col]:
            merged[col] = (merged[col] - merged[col].mean()) / (merged[col].std() + 1e-10)

        # Ensemble: equal weight
        merged['pred_ensemble'] = 0.5 * merged['pred_hist'] + 0.5 * merged[lgb_pred_col]

        print(f"   Ensemble: {len(merged):,} rows, "
              f"HIST+LGB combined")

        return merged
    else:
        print("   ⚠️  No LGB predictions file provided or found")
        return None


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='HIST Transformer for Crypto Alpha')
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--results', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda',
                        help='cuda or cpu')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--ic-weight', type=float, default=0.5,
                        help='Weight for IC loss (1-weight for MSE)')
    parser.add_argument('--lgb-preds', type=str, default=None,
                        help='Path to LGB test predictions for ensemble')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data or os.path.join(project_root, 'data', 'features')
    results_dir = args.results or os.path.join(project_root, 'results_hist')
    os.makedirs(results_dir, exist_ok=True)

    feat_path = os.path.join(data_dir, 'crypto_features_1h.parquet')
    if not os.path.exists(feat_path):
        print(f"❌ Feature file not found: {feat_path}")
        sys.exit(1)

    print("=" * 70)
    print("  HIST TRANSFORMER — Crypto Alpha Model")
    print("  Cross-Stock Attention + Concept Attention + IC Loss")
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
    print(f"   Features: {len(feat_cols)}")

    df[feat_cols] = df[feat_cols].fillna(0)
    df = cross_sectional_rank(df, feat_cols)

    # Rank target
    target_col = f'target_ret_{HORIZON}h'
    df['target_rank'] = df.groupby('timestamp')[target_col].rank(pct=True)

    print(f"   Final shape: {df.shape}")

    # ========================================
    # 2. PREPARE CROSS-SECTION DATA
    # ========================================
    print(f"\n📐 Preparing cross-sectional data...")
    data, symbols, concept_ids = prepare_cross_section_data(
        df, feat_cols, 'target_rank'
    )

    # ========================================
    # 3. TRAIN
    # ========================================
    model, test_preds, test_targets, test_masks, best_val_ic, best_epoch = \
        build_model_and_train(data, concept_ids, args)

    # ========================================
    # 4. EVALUATE
    # ========================================
    test_timestamps = data['test']['timestamps']

    metrics = evaluate_hist_predictions(
        test_preds, test_targets, test_masks,
        test_timestamps, symbols, target_col, HORIZON
    )

    print(f"\n   📈 HIST Test Results:")
    for k, v in metrics.items():
        flag = ""
        if k == 'Rank_IC' and abs(v) > 0.02: flag = " ✓"
        if k == 'LS_Sharpe' and v > 1.0: flag = " ✓"
        if k == 'ICIR' and abs(v) > 0.3: flag = " ✓"
        print(f"      {k:25s} {v}{flag}")

    # ========================================
    # 5. ENSEMBLE WITH LGB (optional)
    # ========================================
    ensemble_metrics = None
    lgb_pred_path = args.lgb_preds
    if lgb_pred_path is None:
        # Try default paths
        for candidate in ['results_v4/test_predictions_v4.parquet',
                          'results_v3/test_predictions_v3.parquet']:
            candidate_full = os.path.join(project_root, candidate)
            if os.path.exists(candidate_full):
                lgb_pred_path = candidate_full
                break

    if lgb_pred_path:
        merged = create_ensemble_with_lgb(
            test_preds, test_masks, test_timestamps,
            symbols, df, feat_cols, target_col,
            lgb_pred_path, HORIZON
        )

        if merged is not None:
            # Evaluate ensemble on merged data
            # Need to add target column
            target_map = df.set_index(['timestamp', 'symbol'])[target_col].to_dict()
            merged['target'] = merged.apply(
                lambda row: target_map.get((row['timestamp'], row['symbol']), np.nan), axis=1
            )
            merged = merged.dropna(subset=['target'])

            # Eval each: HIST alone, LGB alone, ensemble
            for pred_name in ['pred_hist', merged.columns[-3], 'pred_ensemble']:
                if pred_name not in merged.columns:
                    continue

                ens_ics = []
                ens_ls = []
                for ts, grp in merged.groupby('timestamp'):
                    if len(grp) < 10:
                        continue
                    c, _ = spearmanr(grp[pred_name].values, grp['target'].values)
                    if not np.isnan(c):
                        ens_ics.append(c)
                    grp = grp.sort_values(pred_name, ascending=False)
                    n = max(len(grp) // 5, 1)
                    lr = grp.head(n)['target'].mean()
                    sr = grp.tail(n)['target'].mean()
                    ens_ls.append(lr - sr)

                ens_ics = np.array(ens_ics)
                ens_ls = np.array(ens_ls)
                ppy = (24 // HORIZON) * 365

                ric = np.mean(ens_ics) if len(ens_ics) > 0 else 0
                ls_sharpe = (ens_ls.mean() / (ens_ls.std() + 1e-10)) * np.sqrt(ppy) if len(ens_ls) > 0 else 0

                print(f"\n   {pred_name}: Rank IC={ric:.4f}, LS Sharpe={ls_sharpe:.2f}")

            ensemble_metrics = {'note': 'see above'}

    # ========================================
    # 6. SAVE
    # ========================================

    # Save predictions as flat DataFrame (same format as LGB)
    rows = []
    T, N = test_preds.shape
    for t_idx, ts in enumerate(test_timestamps):
        for s_idx, sym in enumerate(symbols):
            if test_masks[t_idx, s_idx] > 0.5:
                rows.append({
                    'timestamp': ts,
                    'symbol': sym,
                    'pred_hist': float(test_preds[t_idx, s_idx]),
                    f'target_ret_{HORIZON}h': float(test_targets[t_idx, s_idx]),
                })

    pred_df = pd.DataFrame(rows)
    pred_df.to_parquet(os.path.join(results_dir, 'test_predictions_hist.parquet'), index=False)

    # Save metrics
    all_results = {
        'hist_metrics': metrics,
        'ensemble_metrics': ensemble_metrics,
        'meta': {
            'timestamp': datetime.now().isoformat(),
            'best_epoch': best_epoch,
            'best_val_rank_ic': round(best_val_ic, 4),
            'n_features': len(feat_cols),
            'd_model': args.d_model,
            'n_heads': args.n_heads,
            'n_layers': args.n_layers,
            'n_concepts': N_CONCEPTS,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'ic_weight': args.ic_weight,
            'device': args.device,
            'symbols': symbols,
        },
    }

    with open(os.path.join(results_dir, 'results_hist.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save model checkpoint
    try:
        import torch
        torch.save(model.state_dict(), os.path.join(results_dir, 'hist_model.pt'))
        print(f"\n   💾 Model saved to {results_dir}/hist_model.pt")
    except Exception:
        pass

    # ========================================
    # FINAL
    # ========================================
    print(f"\n{'='*70}")
    print(f"  HIST RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"   Rank IC:           {metrics['Rank_IC']:+.4f}")
    print(f"   Rank ICIR:         {metrics['Rank_ICIR']:+.4f}")
    print(f"   LS Sharpe:         {metrics['LS_Sharpe']:+.2f}")
    print(f"   LS Ann Return:     {metrics['LS_Ann_Return_%']:+.1f}%")
    print(f"   LS Max Drawdown:   {metrics['LS_MaxDD_%']:.1f}%")
    print(f"   Best epoch:        {best_epoch}")
    print(f"{'='*70}")

    if metrics['LS_Sharpe'] > 2.0:
        print("🟢 STRONG — HIST transformer works!")
    elif metrics['LS_Sharpe'] > 1.0:
        print("🟡 DECENT — Try more layers/heads/epochs or MASTER model.")
    else:
        print("🟠 NEEDS TUNING — Try different d_model, n_layers, IC weight.")

    print(f"\n✅ Results saved to {results_dir}/")
    print(f"\n💡 Next: combine with LightGBM:")
    print(f"   python run_hist_model.py --lgb-preds results_v4/test_predictions_v4.parquet")


if __name__ == '__main__':
    main()
