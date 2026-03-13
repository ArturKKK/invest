#!/usr/bin/env python3
"""
MLP (Multi-Layer Perceptron) Model — Ensemble diversity via neural network.

Uses the same data pipeline and features as LGB v6 (12h target),
but trains an MLP with different inductive bias:
  - Smooth non-linear decision boundaries (vs piecewise-constant in trees)
  - Implicit feature interactions via hidden layers
  - Dropout + weight decay as regularization
  - Different seed → different initialization → genuine diversity

Adds real model diversity to GBDT ensemble (tree splits vs smooth manifolds).

Saved models are loaded alongside LGB/CB/XGB in run_fast_sim.py --ensemble.

Usage:
  python run_pipeline_mlp.py                         # Full walk-forward
  python run_pipeline_mlp.py --production            # Production mode
  python run_pipeline_mlp.py --production --gpu      # GPU training
  python run_pipeline_mlp.py --skip-hpo              # Skip Optuna
  python run_pipeline_mlp.py --seeds 3               # Fewer seeds

Requirements:
  pip install torch pandas numpy scipy pyarrow
  Optional: pip install optuna (for HPO)
"""

import sys
import os
import argparse
import json
import warnings
from datetime import datetime
from copy import deepcopy

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

# Import shared feature engineering from v6
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_pipeline_v6 import (
    add_multi_horizon_targets, add_cross_asset_features,
    add_advanced_regime_features, add_12h_features, add_calendar_features,
    add_sentiment_features, add_derivatives_features,
    cross_sectional_rank, create_rank_target, add_residual_targets,
    evaluate_model, vol_target_returns, drawdown_stop_returns,
    compute_costs_per_period,
    EXCLUDE_COLS, REGIME_COLS, WALK_FORWARD_WINDOWS, PRODUCTION_WINDOW,
    HORIZON, SEEDS, COST_MODEL, PURGE_DAYS,
)

N_SEEDS = 5


# ============================================================
# MLP ARCHITECTURE
# ============================================================

import torch
import torch.nn as nn

class AlphaMLP(nn.Module):
    """
    Simple but effective MLP for cross-sectional alpha prediction.

    Architecture:
    - Input → BatchNorm → [Linear → SiLU → Dropout] × N → Linear → output
    - Skip connections between blocks (ResNet-style)
    - SiLU activation (smoother than ReLU, works well for tabular)
    """
    def __init__(self, input_dim, hidden_dims=(256, 128, 64), dropout=0.3):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            block = nn.Sequential(
                nn.Linear(prev_dim, h_dim),
                nn.SiLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(dropout),
            )
            layers.append(block)
            prev_dim = h_dim

        self.blocks = nn.ModuleList(layers)

        # Skip-connection projections (when dims don't match)
        self.skips = nn.ModuleList()
        prev_dim = input_dim
        for h_dim in hidden_dims:
            if prev_dim != h_dim:
                self.skips.append(nn.Linear(prev_dim, h_dim, bias=False))
            else:
                self.skips.append(nn.Identity())
            prev_dim = h_dim

        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        x = self.input_bn(x)
        for block, skip in zip(self.blocks, self.skips):
            identity = skip(x)
            x = block(x) + identity
        return self.head(x).squeeze(-1)


# ============================================================
# TRAINING UTILITIES
# ============================================================

class RankICLoss(nn.Module):
    """
    Differentiable approximation of Rank IC loss.
    Uses soft ranking (sigmoid-based) for gradient flow.
    Combined with MSE for stability.
    """
    def __init__(self, mse_weight=0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)

        # Soft Spearman correlation approximation
        # Center both
        p = pred - pred.mean()
        t = target - target.mean()
        # Pearson as proxy (on centered data, correlated with Spearman)
        corr = (p * t).sum() / (torch.sqrt((p**2).sum() * (t**2).sum()) + 1e-8)
        rank_loss = 1.0 - corr  # minimize → maximize correlation

        return self.mse_weight * mse_loss + (1 - self.mse_weight) * rank_loss


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n_batches = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    n_batches = 0
    all_preds, all_targets = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        total_loss += loss.item()
        n_batches += 1
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())
    avg_loss = total_loss / max(n_batches, 1)
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    ic = spearmanr(preds, targets)[0] if len(preds) > 10 else 0
    return avg_loss, ic


@torch.no_grad()
def predict(model, X, device, batch_size=8192):
    model.eval()
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    preds = []
    for (X_batch,) in loader:
        X_batch = X_batch.to(device)
        p = model(X_batch)
        preds.append(p.cpu().numpy())
    return np.concatenate(preds)


def train_mlp(X_train, y_train, X_val, y_val, config, seed=42, device='cpu'):
    """Train a single MLP model. Returns (model, best_val_ic)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_features = X_train.shape[1]
    model = AlphaMLP(
        input_dim=n_features,
        hidden_dims=config.get('hidden_dims', (256, 128, 64)),
        dropout=config.get('dropout', 0.3),
    ).to(device)

    train_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val), torch.FloatTensor(y_val))

    batch_size = config.get('batch_size', 4096)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('lr', 1e-3),
        weight_decay=config.get('weight_decay', 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.get('epochs', 50), eta_min=1e-6)

    criterion = RankICLoss(mse_weight=config.get('mse_weight', 0.3))

    best_ic = -999
    patience = config.get('patience', 10)
    no_improve = 0
    best_state = None

    epochs = config.get('epochs', 50)
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_ic = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        if val_ic > best_ic:
            best_ic = val_ic
            no_improve = 0
            best_state = deepcopy(model.state_dict())
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"         ep {epoch+1:3d}: train_loss={train_loss:.5f} "
                  f"val_loss={val_loss:.5f} val_IC={val_ic:.4f} lr={lr:.2e}")

        if no_improve >= patience:
            print(f"         Early stop at epoch {epoch+1} (best IC={best_ic:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_ic


# ============================================================
# HPO
# ============================================================

def run_optuna_hpo(X_train, y_train, X_val, y_val, val_dates,
                   n_trials=30, device='cpu'):
    """Optuna HPO for MLP hyperparameters."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("   ⚠️  Optuna not installed, using defaults")
        return {}

    print(f"   🔍 Running MLP HPO ({n_trials} trials)...")
    unique_dates = np.unique(val_dates)

    def objective(trial):
        # Architecture
        n_layers = trial.suggest_int('n_layers', 2, 4)
        first_dim = trial.suggest_categorical('first_dim', [128, 256, 512])
        hidden_dims = []
        d = first_dim
        for _ in range(n_layers):
            hidden_dims.append(d)
            d = max(d // 2, 32)
        hidden_dims = tuple(hidden_dims)

        config = {
            'hidden_dims': hidden_dims,
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'lr': trial.suggest_float('lr', 1e-4, 5e-3, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [2048, 4096, 8192]),
            'mse_weight': trial.suggest_float('mse_weight', 0.1, 0.7),
            'epochs': 30,  # fewer for HPO
            'patience': 8,
        }

        model, val_ic = train_mlp(X_train, y_train, X_val, y_val,
                                   config, seed=42, device=device)
        return val_ic

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    # Reconstruct hidden_dims from HPO params
    n_layers = best.pop('n_layers', 3)
    first_dim = best.pop('first_dim', 256)
    dims = []
    d = first_dim
    for _ in range(n_layers):
        dims.append(d)
        d = max(d // 2, 32)
    best['hidden_dims'] = tuple(dims)

    print(f"   ✅ Best val IC: {study.best_value:.4f}")
    print(f"      Config: {best}")
    return best


# ============================================================
# FEATURE SELECTION (coefficient-based for MLP)
# ============================================================

def feature_selection_mlp(model, feat_cols, X_val, y_val, device='cpu',
                          threshold_pct=15):
    """
    Gradient-based feature importance for MLP.
    Compute |grad(output) / grad(input)| averaged over val set.
    """
    model.eval()
    X_tensor = torch.FloatTensor(X_val).to(device)
    X_tensor.requires_grad_(True)

    output = model(X_tensor)
    output.sum().backward()

    grad_importance = X_tensor.grad.abs().mean(dim=0).cpu().numpy()
    imp = pd.Series(grad_importance, index=feat_cols)
    threshold = np.percentile(imp.values, threshold_pct)
    keep = imp[imp > threshold].index.tolist()
    print(f"   🔪 Feature selection: {len(feat_cols)} → {len(keep)} "
          f"(gradient-based, drop bottom {threshold_pct}%)")
    return keep


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="MLP model for crypto alpha")
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--results', type=str, default=None)
    parser.add_argument('--hpo-trials', type=int, default=30)
    parser.add_argument('--skip-hpo', action='store_true')
    parser.add_argument('--single-window', action='store_true')
    parser.add_argument('--production', action='store_true')
    parser.add_argument('--train-end', type=str, default=None)
    parser.add_argument('--val-end', type=str, default=None)
    parser.add_argument('--seeds', type=int, default=N_SEEDS)
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU (CUDA) for training')
    parser.add_argument('--residual-target', action='store_true')
    parser.add_argument('--hybrid-norm', action='store_true')
    parser.add_argument('--no-news', action='store_true')
    parser.add_argument('--news-mode', type=str, default='all',
                        choices=['all', 'market-only', 'coin-only', 'none'])
    parser.add_argument('--no-derivatives', action='store_true')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    if args.no_news:
        args.news_mode = 'none'

    # Device
    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"   🔥 Using GPU: {torch.cuda.get_device_name(0)}")
    elif args.gpu and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"   🍎 Using Apple MPS")
    else:
        device = torch.device('cpu')
        if args.gpu:
            print("   ⚠️  GPU requested but not available, using CPU")

    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data or os.path.join(project_root, 'data', 'features')
    if args.production:
        results_dir = args.results or os.path.join(project_root, 'results_mlp_prod')
    else:
        results_dir = args.results or os.path.join(project_root, 'results_mlp')
    os.makedirs(results_dir, exist_ok=True)

    feat_path = os.path.join(data_dir, 'crypto_features_1h.parquet')
    if not os.path.exists(feat_path):
        print(f"❌ Feature file not found: {feat_path}")
        sys.exit(1)

    print("=" * 70)
    print("  MLP CRYPTO ALPHA MODEL")
    print("  12h Target + Neural Network + Walk-Forward")
    print("=" * 70)

    # ========================================
    # 1. LOAD & ENRICH DATA (same pipeline as v6)
    # ========================================
    print(f"\n📊 Loading data...")
    df = pd.read_parquet(feat_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    print(f"   Shape: {df.shape}, Symbols: {df['symbol'].nunique()}")

    df = add_multi_horizon_targets(df)
    df = add_cross_asset_features(df)
    if args.residual_target:
        df = add_residual_targets(df, beta_window=168)
    df = add_advanced_regime_features(df)
    df = add_12h_features(df)
    df = add_calendar_features(df)
    df = add_sentiment_features(df, project_root, news_mode=args.news_mode)
    if not args.no_derivatives:
        df = add_derivatives_features(df, project_root)

    # Clean infinities
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=['target_ret_12h'])

    # Feature columns
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS
                 and not c.startswith('target_')]
    feat_cols = [c for c in feat_cols if df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    print(f"   Features: {len(feat_cols)}")

    df[feat_cols] = df[feat_cols].fillna(0)

    # Cross-sectional rank normalization (same as GBDT)
    df = cross_sectional_rank(df, feat_cols, hybrid=args.hybrid_norm)
    df = create_rank_target(df, HORIZON, use_excess=args.residual_target)

    print(f"   Final shape: {df.shape}")
    print(f"   Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

    # ========================================
    # 2. ROLLING WALK-FORWARD
    # ========================================
    if args.production:
        prod_win = deepcopy(PRODUCTION_WINDOW)
        if args.train_end:
            prod_win['train_end'] = args.train_end
            te = pd.Timestamp(args.train_end)
            prod_win['val_start'] = (te + pd.Timedelta(days=PURGE_DAYS)).strftime('%Y-%m-%d')
        if args.val_end:
            prod_win['val_end'] = args.val_end
            prod_win['test_start'] = args.val_end
        windows = [prod_win]
        print(f"\n🔴 PRODUCTION MODE")
        print(f"   Train: start → {prod_win['train_end']}")
        print(f"   Val:   {prod_win['val_start']} → {prod_win['val_end']}")
    else:
        windows = WALK_FORWARD_WINDOWS
        if args.single_window:
            windows = [windows[-1]]

    print(f"\n{'='*70}")
    print(f"  ROLLING WALK-FORWARD ({len(windows)} windows)")
    print(f"{'='*70}")

    target_col = f'target_ret_{HORIZON}h'
    all_window_metrics = []
    combined_ls_rets = []
    combined_timestamps = []

    for w_idx, window in enumerate(windows):
        print(f"\n{'─'*70}")
        print(f"  Window {w_idx+1}/{len(windows)}: {window['name']}")
        print(f"  Train: → {window['train_end']}")
        print(f"  Val:   {window['val_start']} → {window['val_end']}")
        print(f"  Test:  {window.get('test_start','N/A')} → {window.get('test_end','N/A')}")
        print(f"{'─'*70}")

        train = df[df['timestamp'] < window['train_end']].copy()
        val = df[(df['timestamp'] >= window['val_start']) &
                 (df['timestamp'] < window['val_end'])].copy()
        test = df[(df['timestamp'] >= window.get('test_start', '2099-01-01')) &
                  (df['timestamp'] <= window.get('test_end', '2099-01-01'))].copy()

        has_test = len(test) > 0
        if not has_test and not args.production:
            print(f"   ⚠️  No test data, skipping")
            continue

        print(f"   Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

        X_train = train[feat_cols].values.astype(np.float32)
        y_train = train[target_col].values.astype(np.float32)
        X_val = val[feat_cols].values.astype(np.float32)
        y_val = val[target_col].values.astype(np.float32)
        val_dates = val['timestamp'].dt.date.values

        # --- Feature selection (gradient-based on quick base model) ---
        print("   🔧 Base model for feature selection...")
        base_config = {
            'hidden_dims': (256, 128, 64),
            'dropout': 0.3, 'lr': 1e-3, 'weight_decay': 1e-4,
            'batch_size': 4096, 'epochs': 15, 'patience': 5,
            'mse_weight': 0.3,
        }
        base_model, _ = train_mlp(X_train, y_train, X_val, y_val,
                                   base_config, seed=42, device=device)
        selected_feats = feature_selection_mlp(
            base_model, feat_cols, X_val[:50000], y_val[:50000],
            device=device, threshold_pct=15
        )
        sel_idx = [feat_cols.index(f) for f in selected_feats]

        X_train_sel = X_train[:, sel_idx]
        X_val_sel = X_val[:, sel_idx]

        # --- HPO ---
        best_config = {
            'hidden_dims': (256, 128, 64),
            'dropout': 0.3, 'lr': args.lr, 'weight_decay': 1e-4,
            'batch_size': args.batch_size, 'epochs': args.epochs,
            'patience': 10, 'mse_weight': 0.3,
        }
        if not args.skip_hpo and w_idx == 0:
            hpo_config = run_optuna_hpo(
                X_train_sel, y_train, X_val_sel, y_val, val_dates,
                n_trials=args.hpo_trials, device=device,
            )
            if hpo_config:
                best_config.update(hpo_config)
                best_config['epochs'] = args.epochs  # full epochs for final
                best_config['patience'] = 10

        # --- Multi-seed train ---
        X_pred = test[selected_feats].values.astype(np.float32) if has_test else X_val_sel

        print(f"\n   🌱 MLP multi-seed ensemble ({args.seeds} seeds)...")
        all_preds = []
        all_models = []
        for i, seed in enumerate(SEEDS[:args.seeds]):
            print(f"      Seed {seed} ({i+1}/{args.seeds}):")
            model, val_ic = train_mlp(
                X_train_sel, y_train, X_val_sel, y_val,
                best_config, seed=seed, device=device,
            )
            preds = predict(model, X_pred, device)
            all_preds.append(preds)
            all_models.append(model.cpu())
            print(f"      → val IC={val_ic:.4f}")

        ensemble_pred = np.mean(all_preds, axis=0)
        if has_test:
            test['pred_mlp'] = ensemble_pred

        # --- Save models ---
        for i, mdl in enumerate(all_models):
            seed = SEEDS[:args.seeds][i]
            model_path = os.path.join(results_dir, f'mlp_model_seed_{seed}.pt')
            torch.save({
                'model_state_dict': mdl.state_dict(),
                'config': best_config,
                'input_dim': len(selected_feats),
            }, model_path)

        # Save feature names
        with open(os.path.join(results_dir, 'feature_names.json'), 'w') as f:
            json.dump(selected_feats, f)

        # Save metadata
        meta = {
            'model_type': 'mlp',
            'architecture': str(best_config.get('hidden_dims', (256, 128, 64))),
            'n_features': len(selected_feats),
            'n_seeds': args.seeds,
            'train_rows': len(train),
            'val_rows': len(val),
            'config': {k: v if not isinstance(v, tuple) else list(v)
                      for k, v in best_config.items()},
            'timestamp': datetime.now().isoformat(),
        }
        if args.production:
            meta['mode'] = 'production'
            meta['train_end'] = window['train_end']
            meta['val_end'] = window['val_end']

        with open(os.path.join(results_dir, 'production_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"\n   💾 Saved {len(all_models)} MLP models + {len(selected_feats)} features → {results_dir}")

        if not has_test:
            print(f"   ✅ Production models saved")
            # Val metrics for sanity
            val['pred_mlp'] = np.mean([predict(m.to(device), X_val_sel, device)
                                       for m in all_models], axis=0)
            metrics, _, _, _, _ = evaluate_model(val, 'pred_mlp', target_col,
                                                  HORIZON, label=f"VAL {window['name']}")
            all_window_metrics.append(metrics)
            continue

        # --- Evaluate ---
        metrics, ls_net, ls_vt, ls_dd, timestamps = evaluate_model(
            test, 'pred_mlp', target_col, HORIZON, label=window['name']
        )
        all_window_metrics.append(metrics)
        combined_ls_rets.extend(ls_net.tolist())
        combined_timestamps.extend(timestamps)

    # ========================================
    # 3. SUMMARY
    # ========================================
    if all_window_metrics:
        print(f"\n{'='*70}")
        print("  COMBINED RESULTS")
        print(f"{'='*70}")
        avg_metrics = {}
        for key in all_window_metrics[0]:
            vals = [m[key] for m in all_window_metrics]
            avg_metrics[key] = round(np.mean(vals), 4)
        for k, v in avg_metrics.items():
            print(f"   {k}: {v}")

        # Save
        results_file = os.path.join(results_dir, 'mlp_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'per_window': all_window_metrics,
                'average': avg_metrics,
                'config': {k: v if not isinstance(v, tuple) else list(v)
                          for k, v in best_config.items()},
            }, f, indent=2)
        print(f"\n   📊 Results: {results_file}")


if __name__ == '__main__':
    main()
