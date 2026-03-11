"""
Meta-model stacking inference module.

Shared between run_trading.py (production) and run_fast_sim.py (backtesting).
Trained by run_meta_stack.py, loaded from results/meta_stack/meta_model.pkl.

Usage:
    from src.models.meta_model import MetaModelInference

    meta = MetaModelInference.load('results/meta_stack/meta_model.pkl', variant='lgb_minimal')
    scores = meta.predict(snap_df, pred_v6, pred_v7, pred_cb)
"""

import os
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────
# Feature constants — must match run_meta_stack.py exactly
# ──────────────────────────────────────────────────────────────────────

META_FEATURES_FULL = [
    # L0 predictions (3–4, xgb present only in exp15+)
    'pred_v6', 'pred_v7', 'pred_cb', 'pred_xgb',
    # Pairwise spreads (3–6)
    'spread_v6_v7', 'spread_v6_cb', 'spread_v6_xgb',
    'spread_v7_cb', 'spread_v7_xgb', 'spread_cb_xgb',
    # Cross-model aggregate stats (5)
    'pred_mean', 'pred_std', 'pred_min', 'pred_max', 'pred_range',
    # Cross-sectional ranks (3–4)
    'pred_v6_rank', 'pred_v7_rank', 'pred_cb_rank', 'pred_xgb_rank',
    # Cross-sectional z-scores (3–4)
    'pred_v6_zscore', 'pred_v7_zscore', 'pred_cb_zscore', 'pred_xgb_zscore',
    # Rank agreement (4)
    'rank_std', 'rank_min', 'all_top_q', 'all_bot_q',
    # Per-symbol context (3)
    'gk_vol_24h', 'rsi_14', 'adx',
    # BTC market context (4)
    'btc_vol_24h', 'btc_vol_ratio', 'btc_trend', 'btc_rsi',
    # Market aggregates (2)
    'market_dispersion', 'market_breadth',
    # Time features (3)
    'hour_sin', 'hour_cos', 'dow_sin',
]

META_FEATURES_MINIMAL = [
    'pred_v6', 'pred_v7', 'pred_cb', 'pred_xgb',
    'spread_v6_v7', 'spread_v6_cb', 'spread_v6_xgb',
    'spread_v7_cb', 'spread_v7_xgb', 'spread_cb_xgb',
    'pred_mean', 'pred_std', 'pred_min', 'pred_max', 'pred_range',
    'pred_v6_rank', 'pred_v7_rank', 'pred_cb_rank', 'pred_xgb_rank',
    'pred_v6_zscore', 'pred_v7_zscore', 'pred_cb_zscore', 'pred_xgb_zscore',
    'rank_std', 'rank_min', 'all_top_q', 'all_bot_q',
]

VALID_VARIANTS = ('lgb', 'lgb_minimal', 'ridge', 'ridge_all')


def build_meta_features_live(snap_df, pred_v6, pred_v7, pred_cb, pred_xgb=None):
    """
    Build meta-features from L0 predictions at inference time.
    Must match run_meta_stack.py build_meta_features() exactly.

    Args:
        snap_df: DataFrame with one row per symbol (latest snapshot).
                 Expected columns: symbol, timestamp, and optionally
                 gk_vol_24h, gk_vol_168h, rsi_14, adx, close_ma336_ratio,
                 ret_24h, ret_168h.
        pred_v6: np.ndarray of v6 predictions (one per symbol)
        pred_v7: np.ndarray of v7 predictions
        pred_cb: np.ndarray of CatBoost predictions
        pred_xgb: np.ndarray of XGBoost predictions (optional, None if not available)

    Returns:
        pd.DataFrame with all META_FEATURES_FULL columns.
    """
    n = len(snap_df)
    mf = pd.DataFrame(index=range(n))

    # ── Raw predictions ──
    mf['pred_v6'] = pred_v6
    mf['pred_v7'] = pred_v7
    mf['pred_cb'] = pred_cb
    has_xgb = pred_xgb is not None
    if has_xgb:
        mf['pred_xgb'] = pred_xgb

    # ── Pairwise spreads (dynamic) ──
    pred_cols = ['pred_v6', 'pred_v7', 'pred_cb'] + (['pred_xgb'] if has_xgb else [])
    from itertools import combinations
    for c1, c2 in combinations(pred_cols, 2):
        name = f'spread_{c1.replace("pred_", "")}_{c2.replace("pred_", "")}'
        mf[name] = np.abs(mf[c1].values - mf[c2].values)

    # ── Cross-model stats ──
    preds_list = [pred_v6, pred_v7, pred_cb] + ([pred_xgb] if has_xgb else [])
    preds = np.column_stack(preds_list)
    mf['pred_mean'] = preds.mean(axis=1)
    mf['pred_std'] = preds.std(axis=1)
    mf['pred_min'] = preds.min(axis=1)
    mf['pred_max'] = preds.max(axis=1)
    mf['pred_range'] = mf['pred_max'] - mf['pred_min']

    # ── Cross-sectional ranks (per snapshot = single timestamp) ──
    for col in pred_cols:
        mf[f'{col}_rank'] = pd.Series(mf[col].values).rank(pct=True).values

    # ── Cross-sectional z-scores ──
    for col in pred_cols:
        vals = mf[col].values
        mu, sigma = vals.mean(), vals.std() + 1e-10
        mf[f'{col}_zscore'] = (vals - mu) / sigma

    # ── Rank agreement ──
    rank_cols = [f'{c}_rank' for c in pred_cols]
    rank_vals = mf[rank_cols].values
    mf['rank_std'] = rank_vals.std(axis=1)
    mf['rank_min'] = rank_vals.min(axis=1)
    mf['all_top_q'] = (rank_vals > 0.75).all(axis=1).astype(float)
    mf['all_bot_q'] = (rank_vals < 0.25).all(axis=1).astype(float)

    # ── Per-symbol context ──
    for ctx_col in ['gk_vol_24h', 'rsi_14', 'adx']:
        if ctx_col in snap_df.columns:
            mf[ctx_col] = snap_df[ctx_col].values
        else:
            mf[ctx_col] = 0.0

    # ── BTC market context ──
    syms = snap_df['symbol'].values if 'symbol' in snap_df.columns else np.array([])
    btc_mask = syms == 'BTC/USDT'
    if btc_mask.any():
        btc_idx = np.where(btc_mask)[0][0]
        btc_vol = snap_df['gk_vol_24h'].values[btc_idx] if 'gk_vol_24h' in snap_df.columns else 0.0
        btc_vol_168 = snap_df['gk_vol_168h'].values[btc_idx] if 'gk_vol_168h' in snap_df.columns else btc_vol
        mf['btc_vol_24h'] = btc_vol
        mf['btc_vol_ratio'] = btc_vol / (btc_vol_168 + 1e-10)
        mf['btc_trend'] = snap_df['close_ma336_ratio'].values[btc_idx] if 'close_ma336_ratio' in snap_df.columns else 1.0
        mf['btc_rsi'] = snap_df['rsi_14'].values[btc_idx] if 'rsi_14' in snap_df.columns else 50.0
    else:
        mf['btc_vol_24h'] = 0.0
        mf['btc_vol_ratio'] = 1.0
        mf['btc_trend'] = 1.0
        mf['btc_rsi'] = 50.0

    # ── Market dispersion & breadth ──
    if 'ret_24h' in snap_df.columns:
        mf['market_dispersion'] = snap_df['ret_24h'].std()
    else:
        mf['market_dispersion'] = 0.0
    if 'ret_168h' in snap_df.columns:
        mf['market_breadth'] = (snap_df['ret_168h'].values > 0).mean()
    else:
        mf['market_breadth'] = 0.5

    # ── Time features ──
    if 'timestamp' in snap_df.columns:
        ts = snap_df['timestamp'].iloc[0]
        hour = ts.hour if hasattr(ts, 'hour') else 0
        dow = ts.dayofweek if hasattr(ts, 'dayofweek') else 0
    else:
        hour, dow = 0, 0
    mf['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    mf['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    mf['dow_sin'] = np.sin(2 * np.pi * dow / 7)

    return mf


class MetaModelInference:
    """
    Meta-model wrapper for inference.

    Loads trained meta-model from pkl and provides predict() method
    that takes L0 predictions + market snapshot and returns meta-scores.
    """

    def __init__(self, models, feature_cols, variant, is_ridge=False):
        self.models = models
        self.feature_cols = feature_cols
        self.variant = variant
        self.is_ridge = is_ridge

    @classmethod
    def load(cls, pkl_path, variant='lgb_minimal', root=None):
        """
        Load meta-model from pkl file.

        Args:
            pkl_path: Path to meta_model.pkl, or 'auto' to search in results/meta_stack/
            variant: One of 'lgb', 'lgb_minimal', 'ridge', 'ridge_all'
            root: Project root directory (used when pkl_path='auto')

        Returns:
            MetaModelInference instance, or None if auto-search fails
        """
        import joblib

        # Resolve 'auto' path
        if pkl_path == 'auto':
            if root is None:
                root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            auto_path = os.path.join(root, 'results', 'meta_stack', 'meta_model.pkl')
            if os.path.exists(auto_path):
                pkl_path = auto_path
                print(f"   Meta-model: {auto_path}")
            else:
                print(f"   ⚠️  Meta-model not found at {auto_path}")
                return None

        if not os.path.exists(pkl_path):
            print(f"   ⚠️  Meta-model not found: {pkl_path}")
            return None

        obj = joblib.load(pkl_path)

        if variant == 'lgb':
            models = obj['lgb_models']
            feat_cols = obj['feature_cols']
            is_ridge = False
        elif variant == 'lgb_minimal':
            models = obj['lgb_models_minimal']
            feat_cols = obj['minimal_cols']
            is_ridge = False
        elif variant == 'ridge':
            models = [obj['ridge_model_3']]
            feat_cols = obj['ridge_cols_3']
            is_ridge = True
        elif variant == 'ridge_all':
            models = [obj['ridge_model_all']]
            feat_cols = obj['ridge_cols_all']
            is_ridge = True
        else:
            raise ValueError(f"Unknown variant: {variant}. Must be one of {VALID_VARIANTS}")

        return cls(models=models, feature_cols=feat_cols, variant=variant, is_ridge=is_ridge)

    def predict(self, snap_df, pred_v6, pred_v7, pred_cb, pred_xgb=None):
        """
        Generate meta-model predictions.

        Args:
            snap_df: DataFrame with latest snapshot (one row per symbol)
            pred_v6: np.ndarray — L0 v6 group mean predictions
            pred_v7: np.ndarray — L0 v7 group mean predictions
            pred_cb: np.ndarray — L0 CatBoost group mean predictions
            pred_xgb: np.ndarray — L0 XGBoost group mean predictions (optional)

        Returns:
            np.ndarray of meta-model scores (one per symbol)
        """
        # Build meta-features
        mf = build_meta_features_live(snap_df, pred_v6, pred_v7, pred_cb, pred_xgb=pred_xgb)

        # Ensure all required features exist (fill missing with 0)
        for col in self.feature_cols:
            if col not in mf.columns:
                mf[col] = 0.0

        X = mf[self.feature_cols].values

        if self.is_ridge:
            return self.models[0].predict(X)
        else:
            # Multi-seed LGB: average predictions
            return np.mean([m.predict(X) for m in self.models], axis=0)

    def __repr__(self):
        return (f"MetaModelInference(variant={self.variant!r}, "
                f"features={len(self.feature_cols)}, "
                f"models={len(self.models)})")
