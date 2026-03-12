#!/usr/bin/env python3
"""L0 vs Meta-model score diagnostic — run on VPS.
Replicates EXACT pipeline from run_trading.py run_cycle().
"""
import sys, os, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.abspath(__file__))

from run_trading import (
    fetch_ohlcv, build_features, cross_sectional_rank,
    load_lgb_models, load_catboost_models, SYMBOLS,
    add_12h_features,
)

EXCLUDE_COLS = {'symbol', 'timestamp', 'open_time', 'open', 'high', 'low',
                'close', 'volume', 'date', 'hour', 'coin'}


def build_latest():
    """Replicate run_cycle() feature pipeline exactly."""
    print("  Fetching OHLCV...")
    df = fetch_ohlcv(SYMBOLS, 800)
    if df is None:
        raise RuntimeError("fetch_ohlcv failed")
    print(f"  Raw shape: {df.shape}")

    df = build_features(df)

    from run_pipeline_v6 import (
        add_multi_horizon_targets, add_cross_asset_features,
        add_advanced_regime_features,
        add_derivatives_features, add_sentiment_features,
    )

    # Drop overlaps (same as run_cycle)
    _overlap_prefixes = ('btc_close', 'eth_close',
        'btc_ret_', 'eth_ret_', 'btc_vol_24h', 'btc_ma', 'btc_rolling_high',
        'market_dispersion', 'ret_vs_btc', 'breadth_pct_positive',
        'regime_btc_above_ma720', 'regime_btc_dd_720', 'regime_btc_not_crashed',
        'fng_', 'reversal_', 'vol_surge_', 'btc_beta_')
    _overlap_cols = [c for c in df.columns if c.startswith(_overlap_prefixes)]
    if _overlap_cols:
        df.drop(columns=_overlap_cols, inplace=True, errors='ignore')

    df = add_multi_horizon_targets(df)
    df = add_cross_asset_features(df)
    df = add_advanced_regime_features(df)
    df = add_12h_features(df)
    df = add_sentiment_features(df, ROOT, news_mode='all')
    df = add_derivatives_features(df, ROOT)

    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS
                 and not c.startswith('target_')
                 and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

    df = cross_sectional_rank(df, feat_cols)

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df[feat_cols] = df[feat_cols].fillna(0)

    latest = df.groupby('symbol').last().reset_index()
    return latest, feat_cols


def load_model_groups():
    """Load model groups exactly like generate_signal."""
    import lightgbm as lgb
    from pathlib import Path

    groups = {}  # label -> (models_list, feature_names)

    for name, dirs in [
        ('v6', ["results/production/lgb_v6_no_news", "results_v6_prod", "results_v6"]),
        ('v7', ["results/production/lgb_v7_no_news", "results_v7_prod", "results_v7"]),
    ]:
        for d in dirs:
            p = os.path.join(ROOT, d)
            if os.path.isdir(p):
                files = sorted(Path(p).glob('lgb_model_seed_*.txt'))
                if files:
                    ms = [lgb.Booster(model_file=str(f)) for f in files]
                    fn = json.load(open(os.path.join(p, 'feature_names.json')))
                    groups[name] = (ms, fn)
                    print(f"  Loaded {name}: {len(ms)} models, {len(fn)} feats from {d}")
                break

    for d in ["results/production/catboost_with_news", "results_catboost_prod", "results_catboost"]:
        p = os.path.join(ROOT, d)
        if os.path.isdir(p):
            try:
                from catboost import CatBoostRegressor
                files = sorted(Path(p).glob('cb_model_seed_*.cbm'))
                if files:
                    ms = [CatBoostRegressor().load_model(str(f)) for f in files]
                    fn_path = os.path.join(p, 'feature_names.json')
                    fn = json.load(open(fn_path)) if os.path.exists(fn_path) else ms[0].feature_names_
                    groups['cb'] = (ms, fn)
                    print(f"  Loaded cb: {len(ms)} models, {len(fn)} feats from {d}")
            except ImportError:
                print("  CatBoost not installed")
            break

    return groups


def main():
    print("=" * 70)
    print("L0 vs META-MODEL DIAGNOSTIC")
    print("=" * 70)

    # 1. Load models
    print("\n1. Loading models...")
    groups = load_model_groups()
    if not groups:
        print("ERROR: No models loaded!")
        return

    # 2. Build features
    print("\n2. Building features...")
    latest, feat_cols = build_latest()
    print(f"   Coins: {len(latest)}, Features: {len(feat_cols)}")

    # 3. L0 predictions per model group
    print("\n3. L0 PREDICTIONS PER MODEL GROUP:")
    print("-" * 70)

    per_group_scores = {}
    per_seed_scores = {}

    for label, (models, feats) in groups.items():
        missing = [c for c in feats if c not in latest.columns]
        for c in missing:
            latest[c] = 0.0
        if missing:
            print(f"   {label}: {len(missing)} MISSING features zero-filled: {missing[:10]}")

        X = latest[feats].values
        seed_preds = [m.predict(X) for m in models]
        avg_pred = np.mean(seed_preds, axis=0)
        per_group_scores[label] = avg_pred
        per_seed_scores[label] = seed_preds

        print(f"\n   {label.upper()} ({len(models)} seeds):")
        print(f"     Mean pred: {avg_pred.mean():.6f}")
        print(f"     Std pred:  {avg_pred.std():.6f}")
        print(f"     Min pred:  {avg_pred.min():.6f}")
        print(f"     Max pred:  {avg_pred.max():.6f}")
        print(f"     Spread:    {avg_pred.max() - avg_pred.min():.6f}")

        seed_spreads = [p.max() - p.min() for p in seed_preds]
        print(f"     Per-seed spreads: {[f'{s:.4f}' for s in seed_spreads]}")

        ranked_idx = np.argsort(-avg_pred)
        syms = latest['symbol'].values
        top5 = [(syms[i], f'{avg_pred[i]:.5f}') for i in ranked_idx[:5]]
        bot5 = [(syms[i], f'{avg_pred[i]:.5f}') for i in ranked_idx[-5:]]
        print(f"     Top 5: {top5}")
        print(f"     Bot 5: {bot5}")

    # 4. Model agreement
    print("\n4. MODEL AGREEMENT:")
    print("-" * 70)

    labels = list(per_group_scores.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            l1, l2 = labels[i], labels[j]
            rank1 = np.argsort(np.argsort(-per_group_scores[l1])).astype(float)
            rank2 = np.argsort(np.argsort(-per_group_scores[l2])).astype(float)
            n = len(rank1)
            d = rank1 - rank2
            rho = 1 - 6 * np.sum(d ** 2) / (n * (n ** 2 - 1))
            print(f"   {l1} vs {l2}: Rank corr = {rho:.4f}")

    # 5. Meta-model Ridge
    print("\n5. META-MODEL RIDGE:")
    print("-" * 70)

    if all(k in per_group_scores for k in ['v6', 'v7', 'cb']):
        try:
            from src.models.meta_model import MetaModelInference
            meta_inf = MetaModelInference.load('auto', variant='ridge', root=ROOT)
            if meta_inf is not None:
                meta_scores = meta_inf.predict(
                    latest,
                    pred_v6=per_group_scores['v6'],
                    pred_v7=per_group_scores['v7'],
                    pred_cb=per_group_scores['cb'],
                )
                print(f"   Meta scores mean: {meta_scores.mean():.6f}")
                print(f"   Meta scores std:  {meta_scores.std():.6f}")
                print(f"   Meta scores min:  {meta_scores.min():.6f}")
                print(f"   Meta scores max:  {meta_scores.max():.6f}")
                print(f"   Meta spread:      {meta_scores.max() - meta_scores.min():.6f}")

                ranked_idx = np.argsort(-meta_scores)
                syms = latest['symbol'].values
                top5 = [(syms[i], f'{meta_scores[i]:.5f}') for i in ranked_idx[:5]]
                bot5 = [(syms[i], f'{meta_scores[i]:.5f}') for i in ranked_idx[-5:]]
                print(f"   Top 5: {top5}")
                print(f"   Bot 5: {bot5}")
            else:
                print("   Meta-model not found")
        except Exception as e:
            print(f"   Meta-model error: {e}")
            import traceback; traceback.print_exc()

    # 6. Simple mean (no meta)
    print("\n6. SIMPLE MEAN (no meta, no ridge):")
    print("-" * 70)
    all_preds = list(per_group_scores.values())
    simple_mean = np.mean(all_preds, axis=0)
    print(f"   Mean: {simple_mean.mean():.6f}")
    print(f"   Std:  {simple_mean.std():.6f}")
    print(f"   Min:  {simple_mean.min():.6f}")
    print(f"   Max:  {simple_mean.max():.6f}")
    print(f"   Spread: {simple_mean.max() - simple_mean.min():.6f}")

    ranked_idx = np.argsort(-simple_mean)
    syms = latest['symbol'].values
    top5 = [(syms[i], f'{simple_mean[i]:.5f}') for i in ranked_idx[:5]]
    bot5 = [(syms[i], f'{simple_mean[i]:.5f}') for i in ranked_idx[-5:]]
    print(f"   Top 5: {top5}")
    print(f"   Bot 5: {bot5}")

    # 7. Data freshness check
    print("\n7. DATA FRESHNESS:")
    print("-" * 70)

    data_dir = os.path.join(ROOT, 'data', 'raw')
    if os.path.isdir(data_dir):
        for fn in sorted(os.listdir(data_dir)):
            if fn.endswith('.parquet'):
                fpath = os.path.join(data_dir, fn)
                try:
                    d = pd.read_parquet(fpath)
                    ts_col = None
                    for c in ['timestamp', 'open_time']:
                        if c in d.columns:
                            ts_col = c
                            break
                    if ts_col is None:
                        ts_col = d.columns[0]
                    last_ts = pd.to_datetime(d[ts_col]).max()
                    try:
                        age_h = (pd.Timestamp.now(tz='UTC') - last_ts).total_seconds() / 3600
                    except TypeError:
                        age_h = (pd.Timestamp.now() - last_ts.tz_localize(None)).total_seconds() / 3600
                    status = "STALE" if age_h > 24 else "OK"
                    print(f"   {fn}: last={last_ts}, age={age_h:.1f}h [{status}]")
                except Exception as e:
                    print(f"   {fn}: ERROR: {e}")

    # 8. Zero features
    print("\n8. ZERO FEATURES ANALYSIS:")
    print("-" * 70)

    all_feats = set()
    for label, (models, feats) in groups.items():
        all_feats.update(feats)

    zero_feats = []
    for f in sorted(all_feats):
        if f in latest.columns:
            vals = latest[f].values
            if np.all(vals == 0) or np.all(np.isnan(vals)):
                zero_feats.append(f)

    print(f"   Total unique features across models: {len(all_feats)}")
    print(f"   Zero/NaN features: {len(zero_feats)}")
    if zero_feats:
        print(f"   Zero features: {zero_feats}")
        for label, (models, feats) in groups.items():
            model_zero = [f for f in feats if f in zero_feats]
            if model_zero:
                print(f"   {label} uses {len(model_zero)} zero features: {model_zero}")

    # 9. Trading logs
    print("\n9. RECENT TRADING LOGS:")
    print("-" * 70)

    log_dir = os.path.join(ROOT, 'trading_logs')
    if os.path.isdir(log_dir):
        files = sorted(os.listdir(log_dir))[-5:]
        for fn in files:
            fp = os.path.join(log_dir, fn)
            try:
                d = json.load(open(fp))
                if isinstance(d, dict):
                    positions = d.get('positions', [])
                    if positions:
                        scores = [p.get('score', 0) for p in positions]
                        print(f"   {fn}: {len(positions)} positions, "
                              f"score range [{min(scores):.4f}, {max(scores):.4f}], "
                              f"spread={max(scores) - min(scores):.4f}")
                    else:
                        n_long = len(d.get('long', []))
                        n_short = len(d.get('short', []))
                        print(f"   {fn}: {n_long}L/{n_short}S")
            except Exception:
                print(f"   {fn}: can't parse")
    else:
        print("   No trading_logs directory")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
