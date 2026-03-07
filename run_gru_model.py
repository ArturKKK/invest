#!/usr/bin/env python3
"""
Temporal GRU Model for Crypto Alpha

Architecture — fundamentally different from HIST/MASTER:
- HIST/MASTER: cross-sectional ("who's better among all coins at time T?")
- GRU: temporal ("based on this coin's recent trajectory, where is it going?")

1. Per-coin feature window (lookback 48h) → GRU encoder
2. GRU captures momentum, mean-reversion, volatility clustering per coin
3. Output = per-coin alpha score
4. Cross-sectional ranking for relative alpha

This gives ORTHOGONAL signal to HIST/MASTER — better ensemble diversity.

Usage:
  python run_gru_model.py --device cuda
  python run_gru_model.py --device cuda --lookback 72
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


# ============================================================
# FEATURE ENGINEERING (shared with other models)
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
# TEMPORAL SEQUENCE DATA PREPARATION
# ============================================================

def prepare_temporal_data(df, feat_cols, target_col, actual_return_col, lookback):
    """
    Build temporal sequences per coin.

    Returns dict for each split:
        X: (N_samples, lookback, n_features) — feature sequences
        y: (N_samples,) — rank target at end of sequence
        y_actual: (N_samples,) — actual return at end of sequence
        meta: DataFrame with timestamp, symbol for each sample
    """
    print(f"   Building temporal sequences (lookback={lookback})...")
    symbols = sorted(df['symbol'].unique())

    result = {}
    for split_name, (start, end) in [
        ('train', (None, TRAIN_END)),
        ('val', (VAL_START, VAL_END)),
        ('test', (TEST_START, None)),
    ]:
        all_X, all_y, all_y_actual, all_meta = [], [], [], []

        for sym in symbols:
            sym_df = df[df['symbol'] == sym].sort_values('timestamp').reset_index(drop=True)

            # Determine split range (need lookback history before split start)
            if start:
                split_mask = sym_df['timestamp'] >= start
            else:
                split_mask = pd.Series(True, index=sym_df.index)
            if end:
                split_mask &= sym_df['timestamp'] < end

            split_indices = sym_df.index[split_mask]
            if len(split_indices) == 0:
                continue

            feats = sym_df[feat_cols].values.astype(np.float32)
            targets = sym_df[target_col].values.astype(np.float32)
            actuals = sym_df[actual_return_col].values.astype(np.float32)
            timestamps = sym_df['timestamp'].values

            for idx in split_indices:
                if idx < lookback:
                    continue
                if np.isnan(targets[idx]):
                    continue

                seq = feats[idx - lookback:idx]  # (lookback, F)
                if np.isnan(seq).sum() > seq.size * 0.3:
                    continue

                seq = np.nan_to_num(seq, nan=0.0)
                all_X.append(seq)
                all_y.append(targets[idx])
                all_y_actual.append(actuals[idx])
                all_meta.append({
                    'timestamp': timestamps[idx],
                    'symbol': sym,
                })

        if not all_X:
            print(f"   ⚠️  No samples for {split_name}")
            result[split_name] = None
            continue

        X_arr = np.stack(all_X, axis=0)
        y_arr = np.array(all_y, dtype=np.float32)
        y_actual_arr = np.array(all_y_actual, dtype=np.float32)
        meta_df = pd.DataFrame(all_meta)

        # Replace any remaining NaN
        X_arr = np.nan_to_num(X_arr, nan=0.0)
        y_arr = np.nan_to_num(y_arr, nan=0.0)
        y_actual_arr = np.nan_to_num(y_actual_arr, nan=0.0)

        result[split_name] = {
            'X': X_arr,
            'y': y_arr,
            'y_actual': y_actual_arr,
            'meta': meta_df,
        }

        print(f"   {split_name}: {X_arr.shape[0]:,} samples × "
              f"{X_arr.shape[1]} steps × {X_arr.shape[2]} features")

    return result


# ============================================================
# GRU MODEL (PyTorch)
# ============================================================

def build_and_train_gru(data, args):
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import Dataset, DataLoader
    except ImportError:
        print(f"❌ PyTorch not installed. Fix: {sys.executable} -m pip install torch")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    print(f"\n   🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    class TemporalDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.FloatTensor(X)
            self.y = torch.FloatTensor(y)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    class TemporalGRUModel(nn.Module):
        """
        Temporal GRU for per-coin alpha prediction.

        Architecture:
        1. Input projection (F → d_model) with LayerNorm
        2. Bidirectional GRU (captures both recent and longer-term patterns)
        3. Temporal attention (weighted aggregation of GRU outputs)
        4. Feature gate (learn which timesteps matter most)
        5. Prediction head with residual connection
        """
        def __init__(self, n_features, d_model=96, n_gru_layers=2,
                     dropout=0.1, bidirectional=True):
            super().__init__()
            self.d_model = d_model
            self.bidirectional = bidirectional
            n_dirs = 2 if bidirectional else 1

            # 1. Input projection
            self.input_proj = nn.Sequential(
                nn.Linear(n_features, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            # 2. GRU encoder
            self.gru = nn.GRU(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=n_gru_layers,
                batch_first=True,
                dropout=dropout if n_gru_layers > 1 else 0,
                bidirectional=bidirectional,
            )

            gru_out_dim = d_model * n_dirs

            # 3. Temporal attention
            self.attn_query = nn.Linear(gru_out_dim, 1)

            # 4. Feature gate
            self.feat_gate = nn.Sequential(
                nn.Linear(gru_out_dim, gru_out_dim),
                nn.Sigmoid(),
            )

            # 5. Prediction head
            self.head = nn.Sequential(
                nn.Linear(gru_out_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )

            # Last hidden shortcut
            self.shortcut = nn.Linear(gru_out_dim, 1)

        def forward(self, x):
            """
            x: (B, T, F) — feature sequences
            Returns: (B,) — alpha scores
            """
            B, T, F = x.shape

            # Input projection
            h = self.input_proj(x)  # (B, T, d_model)

            # GRU
            gru_out, _ = self.gru(h)  # (B, T, d_model*n_dirs)

            # Temporal attention
            attn_scores = self.attn_query(gru_out).squeeze(-1)  # (B, T)
            attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, T)
            context = torch.bmm(attn_weights.unsqueeze(1), gru_out).squeeze(1)  # (B, gru_out_dim)

            # Feature gate
            gate = self.feat_gate(context)
            context = context * gate

            # Head + residual from last timestep
            pred = self.head(context).squeeze(-1)  # (B,)
            shortcut = self.shortcut(gru_out[:, -1, :]).squeeze(-1)  # (B,)
            pred = pred + 0.1 * shortcut

            return pred

    # ---- Loss ----
    def combined_loss(pred, target, ic_weight=0.4):
        """MSE + IC loss."""
        mse = F.mse_loss(pred, target)

        # IC loss (negative correlation)
        p = pred - pred.mean()
        t = target - target.mean()
        corr = (p * t).sum() / (p.norm() * t.norm() + 1e-8)
        ic_loss = -corr

        return (1 - ic_weight) * mse + ic_weight * ic_loss

    # ---- Build ----
    n_features = data['train']['X'].shape[2]

    train_ds = TemporalDataset(data['train']['X'], data['train']['y'])
    val_ds = TemporalDataset(data['val']['X'], data['val']['y'])

    # Larger batch for temporal model (each sample is one coin at one timestamp)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=0, pin_memory=True)

    model = TemporalGRUModel(
        n_features=n_features,
        d_model=args.d_model,
        n_gru_layers=args.n_gru_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model params: {n_params:,}")
    print(f"   Architecture: proj({n_features}→{args.d_model}) → "
          f"GRU({args.n_gru_layers}L, {'bi' if args.bidirectional else 'uni'}) → "
          f"temp_attn → gate → head")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Cosine schedule with warmup
    warmup_epochs = min(5, args.epochs // 10)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Training ----
    print(f"\n{'='*70}")
    print(f"  TRAINING GRU ({args.epochs} epochs, BS={args.batch_size}, "
          f"LR={args.lr}, lookback={args.lookback})")
    print(f"{'='*70}")

    best_val_ic = -999
    best_epoch = 0
    patience_counter = 0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(X_batch)
            loss = combined_loss(pred, y_batch, ic_weight=args.ic_weight)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()

        # Validate
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                pred = model(X_batch)
                val_preds.append(pred.cpu().numpy())
                val_targets.append(y_batch.numpy())

        vp = np.concatenate(val_preds)
        vt = np.concatenate(val_targets)

        # Compute rank IC per timestamp for proper eval
        val_meta = data['val']['meta']
        val_df_eval = val_meta.copy()
        val_df_eval['pred'] = vp
        val_df_eval['target'] = vt

        rank_ics = []
        for ts, grp in val_df_eval.groupby('timestamp'):
            if len(grp) < 10:
                continue
            c, _ = spearmanr(grp['pred'].values, grp['target'].values)
            if not np.isnan(c):
                rank_ics.append(c)

        val_rank_ic = np.mean(rank_ics) if rank_ics else 0

        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"   Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Loss: {np.mean(train_losses):.5f} | "
                  f"Val Rank IC: {val_rank_ic:.4f} | "
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

    # ---- Test predictions ----
    test_ds = TemporalDataset(data['test']['X'], data['test']['y'])
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=0, pin_memory=True)

    test_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            test_preds.append(pred.cpu().numpy())

    test_preds = np.concatenate(test_preds)

    return model, test_preds, best_val_ic, best_epoch


# ============================================================
# EVALUATION (flat — proper, no rank-as-return bug)
# ============================================================

def evaluate_flat(pred_df, target_col, actual_return_col, horizon=4):
    """
    Evaluate predictions in flat format — correct P&L using actual returns.

    pred_df must have: timestamp, symbol, pred, target_col, actual_return_col
    """
    periods_per_day = 24 // horizon
    periods_per_year = periods_per_day * 365

    rank_ics, ics, ls_rets, lo5_rets, lo10_rets = [], [], [], [], []

    for ts, grp in pred_df.groupby('timestamp'):
        if len(grp) < 10:
            continue

        p = grp['pred'].values
        t = grp[target_col].values
        actual = grp[actual_return_col].values

        valid = ~(np.isnan(p) | np.isnan(t) | np.isnan(actual))
        if valid.sum() < 10:
            continue

        pv, tv, av = p[valid], t[valid], actual[valid]

        ic = np.corrcoef(pv, tv)[0, 1]
        ric, _ = spearmanr(pv, tv)
        ics.append(ic if not np.isnan(ic) else 0)
        rank_ics.append(ric if not np.isnan(ric) else 0)

        # Sort by prediction (descending), use ACTUAL RETURNS for P&L
        order = np.argsort(-pv)
        sorted_actual = av[order]
        n_q = max(len(pv) // 5, 1)

        ls_rets.append(sorted_actual[:n_q].mean() - sorted_actual[-n_q:].mean())
        lo5_rets.append(sorted_actual[:min(5, len(sorted_actual))].mean())
        lo10_rets.append(sorted_actual[:min(10, len(sorted_actual))].mean())

    rank_ics = np.array(rank_ics)
    ics = np.array(ics)
    ls_rets = np.array(ls_rets)
    lo5 = np.array(lo5_rets) - 0.0005
    lo10 = np.array(lo10_rets) - 0.0005

    def sharpe(r, ppyr):
        if len(r) == 0 or r.std() < 1e-12:
            return 0.0
        return (r.mean() / (r.std() + 1e-10)) * np.sqrt(ppyr)

    def max_dd(r):
        if len(r) == 0:
            return 0.0
        cum = np.cumprod(1 + np.clip(r, -0.99, None))
        running_max = np.maximum.accumulate(cum)
        dd = cum / running_max - 1
        return float(np.min(dd))

    def total_ret(r):
        return float(np.prod(1 + np.clip(r, -0.99, None)) - 1)

    # Daily IC for ICIR
    n_per_day = periods_per_day
    daily_rics = []
    for i in range(0, max(1, len(rank_ics) - n_per_day + 1), n_per_day):
        daily_rics.append(rank_ics[i:i+n_per_day].mean())
    daily_rics = np.array(daily_rics) if daily_rics else np.array([0.0])
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
        'LO5_Total_%': round(float(total_ret(lo5) * 100), 1),
        'LO10_Sharpe': round(float(sharpe(lo10, periods_per_year)), 2),
        'LO10_Total_%': round(float(total_ret(lo10) * 100), 1),
        'N_periods': len(ls_rets),
    }
    return metrics


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Temporal GRU for Crypto Alpha')
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--results', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--d-model', type=int, default=96)
    parser.add_argument('--n-gru-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--lookback', type=int, default=48,
                        help='Hours of history per sample')
    parser.add_argument('--bidirectional', action='store_true', default=True)
    parser.add_argument('--no-bidirectional', dest='bidirectional', action='store_false')
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--ic-weight', type=float, default=0.4)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data or os.path.join(project_root, 'data', 'features')
    results_dir = args.results or os.path.join(project_root, 'results_gru')
    os.makedirs(results_dir, exist_ok=True)

    feat_path = os.path.join(data_dir, 'crypto_features_1h.parquet')
    if not os.path.exists(feat_path):
        print(f"❌ Feature file not found: {feat_path}")
        sys.exit(1)

    print("=" * 70)
    print("  TEMPORAL GRU — Per-Coin Sequence Model")
    print("  Orthogonal to cross-sectional HIST/MASTER")
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

    # For GRU: normalize features per-coin using rolling z-score (temporal normalization)
    # This is different from cross-sectional rank used in HIST/MASTER
    print("   📐 Per-coin rolling z-score normalization...")
    for col in feat_cols:
        rolling_mean = df.groupby('symbol')[col].transform(lambda x: x.rolling(168, min_periods=24).mean())
        rolling_std = df.groupby('symbol')[col].transform(lambda x: x.rolling(168, min_periods=24).std())
        df[col] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
    df[feat_cols] = df[feat_cols].clip(-5, 5).fillna(0)

    target_col = f'target_ret_{HORIZON}h'
    actual_return_col = target_col
    # Rank target for IC-based training
    df['target_rank'] = df.groupby('timestamp')[target_col].rank(pct=True)
    print(f"   Final shape: {df.shape}")

    # ========================================
    # 2. PREPARE TEMPORAL DATA
    # ========================================
    print(f"\n📐 Preparing temporal sequences...")
    data = prepare_temporal_data(
        df, feat_cols, 'target_rank', actual_return_col, args.lookback
    )

    if data['train'] is None or data['val'] is None or data['test'] is None:
        print("❌ Insufficient data for temporal model")
        sys.exit(1)

    # ========================================
    # 3. TRAIN
    # ========================================
    model, test_preds_raw, best_val_ic, best_epoch = build_and_train_gru(data, args)

    # ========================================
    # 4. EVALUATE
    # ========================================
    print(f"\n{'='*70}")
    print(f"  TEST EVALUATION")
    print(f"{'='*70}")

    test_meta = data['test']['meta'].copy()
    test_meta['pred'] = test_preds_raw
    test_meta['target_rank'] = data['test']['y']
    test_meta[target_col] = data['test']['y_actual']

    metrics = evaluate_flat(test_meta, 'target_rank', target_col, HORIZON)

    print(f"\n   📈 GRU Test Results:")
    for k, v in metrics.items():
        flag = ""
        if k == 'Rank_IC' and abs(v) > 0.02: flag = " ✓"
        if k == 'LS_Sharpe' and v > 1.0: flag = " ✓"
        if k == 'ICIR' and abs(v) > 0.3: flag = " ✓"
        print(f"      {k:25s} {v}{flag}")

    # ========================================
    # 5. SAVE
    # ========================================
    pred_df = test_meta[['timestamp', 'symbol', 'pred', target_col]].copy()
    pred_df = pred_df.rename(columns={'pred': 'pred_gru'})
    pred_df.to_parquet(
        os.path.join(results_dir, 'test_predictions_gru.parquet'), index=False
    )

    all_results = {
        'gru_metrics': metrics,
        'meta': {
            'timestamp': datetime.now().isoformat(),
            'best_epoch': best_epoch,
            'best_val_rank_ic': round(best_val_ic, 4),
            'n_features': len(feat_cols),
            'lookback': args.lookback,
            'd_model': args.d_model,
            'n_gru_layers': args.n_gru_layers,
            'bidirectional': args.bidirectional,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'ic_weight': args.ic_weight,
            'device': args.device,
        },
    }

    with open(os.path.join(results_dir, 'results_gru.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    try:
        import torch
        torch.save(model.state_dict(), os.path.join(results_dir, 'gru_model.pt'))
        print(f"\n   💾 Model saved to {results_dir}/gru_model.pt")
    except Exception:
        pass

    # ========================================
    # SUMMARY
    # ========================================
    print(f"\n{'='*70}")
    print(f"  GRU RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"   Rank IC:           {metrics['Rank_IC']:+.4f}")
    print(f"   Rank ICIR:         {metrics['Rank_ICIR']:+.4f}")
    print(f"   LS Sharpe:         {metrics['LS_Sharpe']:+.2f}")
    print(f"   LS Ann Return:     {metrics['LS_Ann_Return_%']:+.1f}%")
    print(f"   LS Max Drawdown:   {metrics['LS_MaxDD_%']:.1f}%")
    print(f"   LO5 Total:         {metrics['LO5_Total_%']:+.1f}%")
    print(f"   Best epoch:        {best_epoch}")
    print(f"{'='*70}")

    if metrics['LS_Sharpe'] > 2.0:
        print("🟢 STRONG — GRU temporal model works!")
    elif metrics['LS_Sharpe'] > 1.0:
        print("🟡 DECENT — Try different lookback/d_model.")
    else:
        print("🟠 NEEDS TUNING")

    print(f"\n✅ Results saved to {results_dir}/")
    print(f"\n💡 Next: run final ensemble:")
    print(f"   python run_ensemble.py")


if __name__ == '__main__':
    main()
