#!/usr/bin/env python3
"""
Fast Historical Simulation — backtest the live pipeline on recent data.

Walks through recent data in configurable steps, generates LGB signals,
and tracks portfolio PnL with realistic costs and hold-aware execution.

Key features:
  - Rebalance every N hours (default 12h, optimal from sweep analysis)
  - HOLD positions that remain in portfolio (save on costs)
  - Only pay transaction costs on position CHANGES
  - Realistic cost model: taker + slippage per side + funding for leverage
  - Vol scaling + DD circuit-breaker
  - Edge-based selectivity: filter by |score − median| (P75/P90)
  - Leverage support for futures trading

Usage:
  python run_fast_sim.py                                  # 14d, $1000, 12h rebal
  python run_fast_sim.py --days 30 --capital 500          # more days
  python run_fast_sim.py --days 30 --rebal 8 --npos 3    # custom params
  python run_fast_sim.py --leverage 3 --edge-pct 75       # 3x leverage, P75 edge filter
  python run_fast_sim.py --ensemble --leverage 3 --rebal 24 --deriv-gate  # recommended
"""

import os, sys, json, argparse, warnings, glob
import pandas as pd, numpy as np
from datetime import datetime, timezone

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from src.models.meta_model import MetaModelInference, build_meta_features_live
except ImportError:
    MetaModelInference = None
from run_trading import (
    SYMBOLS, EXCLUDE_COLS, DEFAULT_RISK, HORIZON,
    fetch_ohlcv, build_features, cross_sectional_rank, load_lgb_models,
    load_catboost_models,
)
try:
    from run_trading import _OKX_BLOCKED
except ImportError:
    _OKX_BLOCKED = set()

COST_SIDE = 0.0003 + 0.0001          # taker 3bps + slippage 1bp
FUNDING_PER_8H = 0.0001              # ~1bp per 8h funding cost for leveraged positions

# ── Vol-targeting + meta-risk defaults ─────────────────────────────
DEFAULT_VOL_TARGET_ANN = 0.30        # 30% annual portfolio vol target
META_RISK_MIN  = 0.3                 # minimum risk scale (never zero)
META_RISK_MAX  = 1.5                 # allow slight upscale on ideal conditions


def compute_hac_sharpe(returns, rebal_h):
    """Newey-West HAC-adjusted Sharpe ratio.

    Standard Sharpe assumes IID returns, which overstates significance
    when consecutive returns are autocorrelated (e.g. overlapping hold
    periods).  HAC variance accounts for this via Bartlett kernel.
    """
    a = np.array(returns)
    n = len(a)
    if n < 10:
        return np.nan
    mean_r = np.mean(a)
    # Bandwidth: Newey-West rule of thumb = int(n^(1/3)), min 2
    max_lag = max(int(n ** (1 / 3)), 2)
    demeaned = a - mean_r
    gamma_0 = np.mean(demeaned ** 2)
    hac_var = gamma_0
    for k in range(1, max_lag + 1):
        bartlett_weight = 1.0 - k / (max_lag + 1)
        gamma_k = np.mean(demeaned[k:] * demeaned[:-k])
        hac_var += 2.0 * bartlett_weight * gamma_k
    hac_var = max(hac_var, 1e-20)  # floor to prevent sqrt of negative
    periods_per_year = 365 * 24 / rebal_h
    return mean_r / np.sqrt(hac_var) * np.sqrt(periods_per_year)

# ── Macro event calendar (FOMC, CPI, major crypto events) ─────────
# These are UTC dates of high-impact events where we reduce/skip positions
# to avoid tail risk.  Updated periodically.
MACRO_EVENTS = {
    # 2025 FOMC rate decisions (announcement ~18:00 UTC)
    '2025-01-29', '2025-03-19', '2025-05-07', '2025-06-18',
    '2025-07-30', '2025-09-17', '2025-10-29', '2025-12-17',
    # 2025 US CPI releases (~12:30 UTC)
    '2025-01-15', '2025-02-12', '2025-03-12', '2025-04-10',
    '2025-05-13', '2025-06-11', '2025-07-11', '2025-08-12',
    '2025-09-10', '2025-10-14', '2025-11-12', '2025-12-10',
    # 2026 FOMC (projected)
    '2026-01-28', '2026-03-18', '2026-04-29', '2026-06-17',
    '2026-07-29', '2026-09-16', '2026-10-28', '2026-12-16',
    # 2026 US CPI (projected)
    '2026-01-14', '2026-02-11', '2026-03-11', '2026-04-14',
    '2026-05-12', '2026-06-10', '2026-07-14', '2026-08-12',
    '2026-09-11', '2026-10-13', '2026-11-12', '2026-12-10',
}

def is_near_event(ts, hours_before=18, hours_after=6):
    """Check if timestamp is within danger zone around a macro event.
    Default: skip 18h before event (day before) to 6h after.
    With 24h rebalance, this means we go flat for the step spanning the event.
    """
    ts_dt = pd.Timestamp(ts).tz_localize(None) if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts)
    for evt_str in MACRO_EVENTS:
        evt = pd.Timestamp(evt_str)
        delta_h = (ts_dt - evt).total_seconds() / 3600
        if -hours_before <= delta_h <= hours_after:
            return True, evt_str
    return False, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days",    type=int,   default=14)
    ap.add_argument("--capital", type=float, default=1000.0)
    ap.add_argument("--config",  type=str,   default=None)
    ap.add_argument("--model-dir", type=str, default=None,
                    help="Model directory (default: results_v8 > results_v7 > results_v6)")
    ap.add_argument("--rebal",   type=int,   default=12,
                    help="Rebalance interval in hours (default: 12)")
    ap.add_argument("--npos",    type=int,   default=None,
                    help="Positions per side (overrides config)")
    ap.add_argument("--kelly",   type=float, default=None,
                    help="Kelly fraction (overrides config)")
    ap.add_argument("--leverage", type=float, default=1.0,
                    help="Leverage multiplier (default: 1.0, e.g. 3 for 3x)")
    ap.add_argument("--edge-pct", type=int,  default=0, choices=[0, 50, 75, 90],
                    help="Edge percentile filter: 0=off, 75=P75 (recommended)")
    ap.add_argument("--min-edge", type=float, default=0.0,
                    help="Manual min edge threshold (overrides --edge-pct)")
    ap.add_argument("--ensemble", action="store_true",
                    help="Ensemble v6+v7+CB models (avg scores) + deriv risk gate")
    ap.add_argument("--edge-boost", action="store_true",
                    help="Edge-proportional sizing: high-edge positions get more weight")
    ap.add_argument("--no-conf", action="store_true",
                    help="Disable confidence weighting (for A/B testing)")
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="Min confidence filter: skip signals with confidence < threshold (e.g. 0.90)")
    ap.add_argument("--adaptive-rebal", action="store_true",
                    help="Adaptive rebalance: base period + early rebal on strong signals")
    ap.add_argument("--dynamic-lev", action="store_true",
                    help="Dynamic leverage: base lev normally, scale up on strong edge")
    ap.add_argument("--max-lev", type=float, default=7.0,
                    help="Max leverage for dynamic-lev mode (default: 7)")
    ap.add_argument("--event-filter", action="store_true",
                    help="Reduce positions near FOMC/CPI events to avoid tail risk")
    ap.add_argument("--smooth-signal", type=float, default=0.0,
                    help="Signal smoothing: EMA weight on previous scores (0=off, 0.4=recommended)")
    ap.add_argument("--vol-size", action="store_true",
                    help="Vol-adjusted sizing: weight ∝ edge / coin_vol (inverse-vol)")
    ap.add_argument("--regime-shorts", type=float, default=0.0,
                    help="Regime short scaling: in bull regime, scale short allocation to X (e.g. 0.5)")
    ap.add_argument("--short-blocked", action="store_true",
                    help="Block shorting OKX-restricted symbols (19 coins). "
                         "Simulates real OKX constraints for realistic backtest.")
    ap.add_argument("--vol-target-ann", type=float, default=0.0,
                    help="Portfolio vol targeting: annualized target vol (e.g. 0.30 = 30%%). "
                         "Scales gross exposure inversely to recent realized vol. 0=off.")
    ap.add_argument("--meta-risk", action="store_true",
                    help="Meta-model risk scaler: adjust gross exposure (0.3x-1.5x) based on "
                         "model agreement, regime, recent performance, score spread.")
    ap.add_argument("--deriv-gate", action="store_true", default=False,
                    help="Use derivatives-only model as risk gate (scale positions 0.3x-1.0x "
                         "based on deriv model disagreement). Auto-enabled if deriv models found.")
    ap.add_argument("--no-deriv-gate", action="store_true",
                    help="Force disable deriv risk gate even if models exist.")
    ap.add_argument("--no-xgb", action="store_true",
                    help="Exclude XGBoost from ensemble (for A/B testing).")
    ap.add_argument("--softmax-temp", type=float, default=0.0,
                    help="Softmax temperature for score-proportional sizing (e.g. 2.5). "
                         "0=off (uses default exp(score*2) or edge-boost). "
                         "Higher = more concentration on strong signals.")
    ap.add_argument("--no-ddstop", action="store_true",
                    help="Disable drawdown stop completely (for diagnostic runs).")
    ap.add_argument("--hysteresis", type=int, default=0,
                    help="Position hysteresis: keep position until rank drops below N+hysteresis. "
                         "E.g. --hysteresis 5: enter top-10, keep until rank > 15. 0=off.")
    ap.add_argument("--turnover-budget", type=int, default=0,
                    help="Max replacements per side per rebalance (e.g. 3-5). 0=unlimited.")
    ap.add_argument("--min-zscore", type=float, default=0.0,
                    help="Min |z-score| to open a position. Implements dynamic N: "
                         "only trade coins with |z| > threshold. 0=off (fixed N).")
    ap.add_argument("--dynamic-n", action="store_true",
                    help="Dynamic N: vary positions per side based on cross-sectional dispersion. "
                         "Low dispersion → fewer positions (min 3), high → up to N.")
    ap.add_argument("--meta-model", type=str, default=None, nargs='?', const='auto',
                    help="Use meta-model stacking instead of simple mean. "
                         "Pass path to meta_model.pkl or 'auto' to find in results/meta_stack/")
    ap.add_argument("--meta-variant", type=str, default="lgb_minimal",
                    choices=["lgb", "lgb_minimal", "ridge"],
                    help="Meta-model variant: lgb (full 33 feat), lgb_minimal (20 feat), ridge")
    ap.add_argument("--start-date", type=str, default=None,
                    help="Force sim to start from this date (YYYY-MM-DD). Trims earlier data.")
    ap.add_argument("--end-date", type=str, default=None,
                    help="Force sim to end at this date (YYYY-MM-DD). Trims later data.")
    ap.add_argument("--data", type=str, default=None,
                    help="Path to pre-built features parquet (offline mode). "
                         "If set, skips live fetch + feature engineering.")
    ap.add_argument("--warmup",  type=int,   default=720)
    args = ap.parse_args()
    root = os.path.dirname(os.path.abspath(__file__))

    # ── risk cfg ──────────────────────────────────────────────────
    risk = DEFAULT_RISK.copy()
    for p in [args.config,
              os.path.join(root, "results_risk_study", "optimal_config.json")]:
        if p and os.path.exists(p):
            with open(p) as f: risk.update(json.load(f))
            print(f"   Risk config: {os.path.basename(p)}")
            break

    n_pos   = args.npos  or risk["n_long"]
    kelly   = args.kelly or risk["kelly_frac"]
    vol_tgt = risk["vol_target"]
    dd_stop = risk["dd_stop"] if not args.no_ddstop else -9.99
    dd_resume = risk["dd_resume"] if not args.no_ddstop else -9.99
    vol_lb  = risk.get("vol_lookback", 50)
    rebal_h = args.rebal
    leverage = args.leverage
    min_edge = args.min_edge            # will be calibrated later if edge_pct > 0

    total_h = args.warmup + args.days * 24
    sim_h   = args.days * 24

    lev_str = f"{leverage:.0f}x" if leverage >= 1 else f"{leverage:.1f}x"
    edge_str = f"P{args.edge_pct}" if args.edge_pct > 0 else (
        f"edge>{min_edge:.4f}" if min_edge > 0 else "off")
    boost_str = "boost" if args.edge_boost else ""
    softmax_str = f"softmax{args.softmax_temp:.1f}" if args.softmax_temp > 0 else ""
    adapt_str = "adaptive" if args.adaptive_rebal else ""
    dynlev_str = f"dynlev→{args.max_lev:.0f}x" if args.dynamic_lev else ""
    evtfilt_str = "evtfilt" if args.event_filter else ""
    voltgt_str = f"voltgt{args.vol_target_ann:.0%}" if args.vol_target_ann > 0 else ""
    metarisk_str = "meta-risk" if args.meta_risk else ""
    derivgate_str = "deriv-gate" if (not args.no_deriv_gate) else ""
    mode_parts = [s for s in [edge_str if edge_str != 'off' else '', boost_str, softmax_str,
                               adapt_str,
                               dynlev_str, evtfilt_str, voltgt_str, metarisk_str,
                               derivgate_str] if s]
    mode_str = '+'.join(mode_parts) if mode_parts else 'baseline'
    print("=" * 70)
    print(f"  FAST SIMULATION")
    print(f"  {args.days}d | ${args.capital:,.0f} | rebal={rebal_h}h | "
          f"N={n_pos}+{n_pos} | kelly={kelly:.0%} | cost={COST_SIDE*1e4:.0f}bp/side")
    print(f"  leverage={lev_str} | mode={mode_str}")
    print("=" * 70)

    # ── 1  data ───────────────────────────────────────────────────
    if args.data:
        # Offline mode: load pre-built features from parquet
        print(f"\n📊 Loading offline data: {args.data}")
        df = pd.read_parquet(args.data)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        # Ensure symbol column exists
        if 'symbol' not in df.columns and df.index.names and 'symbol' in df.index.names:
            df = df.reset_index()
        print(f"   {df.shape}, {df['symbol'].nunique()} symbols")
        print(f"   Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

        # If raw OHLCV (no feature columns), build basic features first
        built_basic = False
        if 'ret_1h' not in df.columns:
            print("   🔧 Raw OHLCV detected — building basic features...")
            df = build_features(df)
            built_basic = True

        # Enrich with features computed at runtime (same as pipeline)
        from run_pipeline_v6 import (
            add_multi_horizon_targets, add_cross_asset_features,
            add_advanced_regime_features,
            add_derivatives_features, add_sentiment_features,
            add_calendar_features, add_macro_features,
        )
        from run_trading import add_12h_features
        from run_pipeline_xgboost import add_news_interaction_features

        if built_basic:
            # build_features created partial cross-asset/regime/FNG features;
            # drop them so pipeline functions can recreate the full set cleanly
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

            print("   Enriching features (full pipeline: cross-asset, regime, 12h, calendar, macro, sentiment, derivatives)...")
            df = add_multi_horizon_targets(df)
            df = add_cross_asset_features(df)
            df = add_advanced_regime_features(df)
            df = add_12h_features(df)
            df = add_calendar_features(df)
            df = add_macro_features(df, root)
            df = add_sentiment_features(df, root, news_mode='all')
            df = add_derivatives_features(df, root)
            df = add_news_interaction_features(df)
        else:
            # Pre-built features parquet — enrich all
            print("   Enriching features (cross-asset, regime, 12h+v7, calendar, macro, sentiment, derivatives)...")
            df = add_multi_horizon_targets(df)
            df = add_cross_asset_features(df)
            df = add_advanced_regime_features(df)
            df = add_12h_features(df)
            df = add_calendar_features(df)
            df = add_macro_features(df, root)
            df = add_sentiment_features(df, root, news_mode='all')
            df = add_derivatives_features(df, root)
            df = add_news_interaction_features(df)

        # Cross-sectional rank (after all features built)
        fc = [c for c in df.columns if c not in EXCLUDE_COLS
              and not c.startswith("target_")
              and df[c].dtype in ("float64","float32","int64","int32")]
        df = cross_sectional_rank(df, fc)

        print("   ⚠️  offline mode: some features may be zero-filled by models")

        # Clean infinities
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        # Feature columns + fillna
        fc = [c for c in df.columns if c not in EXCLUDE_COLS
              and not c.startswith("target_")
              and df[c].dtype in ("float64", "float32", "int64", "int32")]
        df[fc] = df[fc].fillna(0)
        print(f"   Final: {df.shape}, {len(fc)} features")

    else:
        print(f"\n📊 Fetching {total_h}h …")
        raw = fetch_ohlcv(SYMBOLS, total_h)
        if raw is None or len(raw) == 0:
            print("❌ fetch failed"); return
        print(f"   {raw.shape}, {raw['symbol'].nunique()} symbols")

        # ── 2  features ───────────────────────────────────────────
        print("🔧 Features …")
        df = build_features(raw)

        # Save raw snapshot for reproducible offline runs
        raw.to_parquet(os.path.join(root, "trading_logs", "frozen_raw.parquet"), index=False)
        print(f"   💾 Raw OHLCV saved: trading_logs/frozen_raw.parquet ({raw.shape})")
        print(f"      Date range: {raw['timestamp'].min()} → {raw['timestamp'].max()}")
        print(f"      Per-symbol candles: {raw.groupby('symbol').size().describe()[['min','max']].to_dict()}")

        # Enrich with pipeline features (same as offline built_basic path)
        from run_pipeline_v6 import (
            add_multi_horizon_targets, add_cross_asset_features,
            add_advanced_regime_features,
            add_derivatives_features, add_sentiment_features,
            add_macro_features,
        )
        from run_trading import add_12h_features

        # build_features() created partial cross-asset/regime/FNG features;
        # drop them so pipeline functions can recreate the full set cleanly
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

        print("   🔧 Enriching: targets, cross-asset, regime, 12h, calendar, macro, sentiment, derivatives...")
        df = add_multi_horizon_targets(df)
        df = add_cross_asset_features(df)
        df = add_advanced_regime_features(df)
        df = add_12h_features(df)
        from run_pipeline_v6 import add_calendar_features as _acf
        df = _acf(df)
        df = add_macro_features(df, root)
        df = add_sentiment_features(df, root, news_mode='all')
        df = add_derivatives_features(df, root)
        from run_pipeline_xgboost import add_news_interaction_features as _anif
        df = _anif(df)

        # Cross-sectional rank (after all features built)
        fc = [c for c in df.columns if c not in EXCLUDE_COLS
              and not c.startswith("target_")
              and df[c].dtype in ("float64","float32","int64","int32")]
        df = cross_sectional_rank(df, fc)

        # Clean infinities
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        # Feature columns + fillna
        fc = [c for c in df.columns if c not in EXCLUDE_COLS
              and not c.startswith("target_")
              and df[c].dtype in ("float64", "float32", "int64", "int32")]
        df[fc] = df[fc].fillna(0)
        print(f"   Final: {df.shape}, {len(fc)} features")

    # ── 3  models ─────────────────────────────────────────────────
    print("📡 Models …")
    model_groups = []   # list of (models, feature_names) tuples
    model_group_labels = []  # parallel list: 'v6', 'v7', 'cb' — for meta-model mapping
    deriv_models = None  # (models, feature_names) — for risk gate, NOT ensemble
    use_deriv_gate = False

    # Auto-enable ensemble when meta-model is requested (meta needs v6+v7+cb)
    if getattr(args, 'meta_model', None) and not args.ensemble:
        args.ensemble = True
        print("   ℹ️  --meta-model requires ensemble → auto-enabling --ensemble")

    if args.ensemble:
        # Load LGB v6, v7 — one directory per model type (first match wins)
        # Prevents duplicate loading if both production/ and results_v6/ exist
        loaded_types = set()  # track 'v6', 'v7' to prevent duplicates
        lgb_candidates = [
            ("v6", "results/production/lgb_v6_no_news"),
            ("v6", "results_v6_prod"),
            ("v6", "results_v6"),
            ("v7", "results/production/lgb_v7_no_news"),
            ("v7", "results_v7_prod"),
            ("v7", "results_v7"),
        ]
        for mtype, d in lgb_candidates:
            if mtype in loaded_types:
                continue
            p = os.path.join(root, d)
            if os.path.isdir(p) and any(f.endswith('.txt') for f in os.listdir(p)):
                ms = load_lgb_models(p)
                if ms:
                    mf_g = ms[0].feature_name()
                    n_missing = 0
                    for c in [c for c in mf_g if c not in df.columns]:
                        df[c] = 0.0
                        n_missing += 1
                    model_groups.append((ms, mf_g))
                    model_group_labels.append(mtype)  # 'v6' or 'v7'
                    loaded_types.add(mtype)
                    label = "PROD" if "_prod" in p or "production" in p else "research"
                    warn = f" ⚠️ {n_missing} features zero-filled" if n_missing > 3 else ""
                    print(f"   {os.path.basename(p)}: {len(ms)} LGB models, {len(mf_g)} feats [{label}]{warn}")
        # CatBoost ensemble member
        cb_dir = None
        for _cb in ["results/production/catboost_with_news", "results_catboost_prod", "results_catboost"]:
            _p = os.path.join(root, _cb)
            if os.path.isdir(_p):
                cb_dir = _p; break
        if not cb_dir:
            cb_dir = os.path.join(root, "results/production/catboost_with_news")
        if os.path.isdir(cb_dir):
            try:
                ms = load_catboost_models(cb_dir)
                if ms:
                    fn_path = os.path.join(cb_dir, 'feature_names.json')
                    if os.path.exists(fn_path):
                        with open(fn_path) as _f:
                            mf_g = json.load(_f)
                    else:
                        mf_g = ms[0].feature_names_
                    n_missing = sum(1 for c in mf_g if c not in df.columns)
                    for c in [c for c in mf_g if c not in df.columns]:
                        df[c] = 0.0
                    model_groups.append((ms, mf_g))
                    model_group_labels.append('cb')
                    warn = f" ⚠️ {n_missing} features zero-filled" if n_missing > 3 else ""
                    print(f"   catboost: {len(ms)} CB models, {len(mf_g)} feats{warn}")
            except ImportError:
                print("   ⚠️  catboost not installed, skipping CatBoost models")

        # XGBoost ensemble member
        if not args.no_xgb:
            xgb_dir = None
            for _xd in ["results/production/xgboost", "results_xgboost_prod", "results_xgboost"]:
                _p = os.path.join(root, _xd)
                if os.path.isdir(_p) and any(f.endswith('.json') for f in os.listdir(_p)):
                    xgb_dir = _p; break
            if xgb_dir:
                try:
                    import xgboost as _xgb_lib
                    from pathlib import Path as _XPath
                    _xfiles = sorted(_XPath(xgb_dir).glob('xgb_model_seed_*.json'))
                    if _xfiles:
                        _xms = [_xgb_lib.Booster(model_file=str(f)) for f in _xfiles]
                        fn_path = os.path.join(xgb_dir, 'feature_names.json')
                        if os.path.exists(fn_path):
                            with open(fn_path) as _f:
                                mf_g = json.load(_f)
                        else:
                            mf_g = _xms[0].feature_names
                        n_missing = sum(1 for c in mf_g if c not in df.columns)
                        for c in [c for c in mf_g if c not in df.columns]:
                            df[c] = 0.0
                        class _XgbWrapper:
                            def __init__(self, booster, feat_names):
                                self._b = booster
                                self._fn = feat_names
                            def predict(self, X):
                                import xgboost as __xgb
                                dm = __xgb.DMatrix(X, feature_names=self._fn)
                                return self._b.predict(dm)
                        ms_wrapped = [_XgbWrapper(m, mf_g) for m in _xms]
                        model_groups.append((ms_wrapped, mf_g))
                        model_group_labels.append('xgb')
                        warn = f" ⚠️ {n_missing} features zero-filled" if n_missing > 3 else ""
                        print(f"   xgboost: {len(_xms)} XGB models, {len(mf_g)} feats{warn}")
                except ImportError:
                    print("   ⚠️  xgboost not installed, skipping XGBoost models")
                except Exception as e:
                    print(f"   ⚠️  XGBoost load failed: {e}")

        # MLP ensemble member
        mlp_dir = None
        for _md in ["results/production/mlp", "results_mlp_prod", "results_mlp"]:
            _p = os.path.join(root, _md)
            if os.path.isdir(_p) and any(f.endswith('.pt') for f in os.listdir(_p)):
                mlp_dir = _p; break
        if mlp_dir:
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
                        n_missing = sum(1 for c in mf_g if c not in df.columns)
                        for c in [c for c in mf_g if c not in df.columns]:
                            df[c] = 0.0
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
                        warn = f" ⚠️ {n_missing} features zero-filled" if n_missing > 3 else ""
                        print(f"   mlp: {len(_mlp_models)} MLP models, {len(mf_g)} feats{warn}")
            except ImportError:
                print("   ⚠️  torch not installed, skipping MLP models")
            except Exception as e:
                print(f"   ⚠️  MLP load failed: {e}")

        # Multi-horizon LGB models (e.g. results_v6_4h_prod)
        for _hz_dir in sorted(glob.glob(os.path.join(root, "results_v6_*h_prod"))):
            if os.path.isdir(_hz_dir) and any(f.endswith('.txt') for f in os.listdir(_hz_dir)):
                _hz_name = os.path.basename(_hz_dir)  # e.g. results_v6_4h_prod
                _hz_label = _hz_name.replace("results_", "").replace("_prod", "")  # v6_4h
                ms = load_lgb_models(_hz_dir)
                if ms:
                    mf_g = ms[0].feature_name()
                    n_missing = 0
                    for c in [c for c in mf_g if c not in df.columns]:
                        df[c] = 0.0
                        n_missing += 1
                    model_groups.append((ms, mf_g))
                    model_group_labels.append(_hz_label)
                    warn = f" ⚠️ {n_missing} features zero-filled" if n_missing > 3 else ""
                    print(f"   {_hz_label}: {len(ms)} LGB models, {len(mf_g)} feats{warn}")

        if not model_groups:
            print("❌ no models for ensemble"); return
    else:
        model_dir = args.model_dir
        if not model_dir:
            for d in ["results/production/lgb_v7_no_news", "results/production/lgb_v6_no_news",
                      "results_v7", "results_v6"]:
                p = os.path.join(root, d)
                if os.path.isdir(p) and any(f.endswith('.txt') for f in os.listdir(p)):
                    model_dir = p; break
        if not model_dir:
            model_dir = os.path.join(root, "results/production/lgb_v6_no_news")
        models = load_lgb_models(model_dir)
        if not models:
            print("❌ no models"); return
        mf = models[0].feature_name()
        for c in [c for c in mf if c not in df.columns]:
            df[c] = 0.0
        model_groups.append((models, mf))
        model_group_labels.append('single')
        print(f"   {len(models)} models, {len(mf)} feats")

    # ── Derivatives-Only model → RISK GATE (works in both single & ensemble) ──
    deriv_dir = None
    for _dd in ["results/production/deriv_only", "results_deriv"]:
        _p = os.path.join(root, _dd)
        if os.path.isdir(_p) and any(f.endswith('.txt') for f in os.listdir(_p)):
            deriv_dir = _p; break
    use_deriv_gate = (args.deriv_gate or deriv_dir is not None) and not args.no_deriv_gate
    if deriv_dir and use_deriv_gate:
        import lightgbm as _lgb
        from pathlib import Path as _Path
        _deriv_files = sorted(_Path(deriv_dir).glob('deriv_model_seed_*.txt'))
        if not _deriv_files:
            _deriv_files = sorted(_Path(deriv_dir).glob('lgb_model_seed_*.txt'))
        _d_ms = [_lgb.Booster(model_file=str(f)) for f in _deriv_files]
        if _d_ms:
            fn_path = os.path.join(deriv_dir, 'feature_names.json')
            if os.path.exists(fn_path):
                with open(fn_path) as _f:
                    _d_feats = json.load(_f)
            else:
                _d_feats = _d_ms[0].feature_name()
            for c in [c for c in _d_feats if c not in df.columns]:
                df[c] = 0.0
            deriv_models = (_d_ms, _d_feats)
            print(f"   deriv_only (risk gate): {len(_d_ms)} LGB models, {len(_d_feats)} feats")
    elif deriv_dir and not use_deriv_gate:
        print(f"   deriv_only: found but --no-deriv-gate, skipping")

    # Architecture summary
    n_ensemble = len(model_groups)
    n_models_total = sum(len(ms) for ms, _ in model_groups)
    arch_parts = [f"{n_ensemble} groups ({n_models_total} models)"]
    if deriv_models is not None and use_deriv_gate:
        arch_parts.append(f"deriv gate ({len(deriv_models[0])} models)")

    # ── Meta-model loading ────────────────────────────────────
    _meta_model_inf = None
    _meta_group_idx = {}  # maps 'v6','v7','cb' → index in model_groups
    if getattr(args, 'meta_model', None) and MetaModelInference is not None:
        _meta_model_inf = MetaModelInference.load(
            args.meta_model, variant=args.meta_variant, root=root
        )
        if _meta_model_inf is not None:
            # Build label→index mapping for correct v6/v7/cb identification
            for _i, _lbl in enumerate(model_group_labels):
                _meta_group_idx[_lbl] = _i
            if all(k in _meta_group_idx for k in ('v6', 'v7', 'cb')):
                arch_parts.append(f"meta-{args.meta_variant}")
                mode_str = mode_str.rstrip('+') + f"+meta-{args.meta_variant}"
            else:
                missing = [k for k in ('v6', 'v7', 'cb') if k not in _meta_group_idx]
                print(f"   ⚠️  Meta-model needs v6+v7+cb but missing: {missing} → disabled")
                _meta_model_inf = None

    print(f"   \U0001f3d7\ufe0f  Architecture: {' + '.join(arch_parts)}")

    # ── Deriv risk gate constants ─────────────────────────────
    DERIV_GATE_MIN = 0.3   # minimum scale (never fully zero out)
    DERIV_GATE_MAX = 1.0   # at full agreement → no scaling

    def predict_ensemble(snap_df):
        """Predict using L0 models, optionally refined by meta-model stacking."""
        # Step 1: Get per-group L0 predictions (always needed)
        all_scores = []
        all_individual = []
        per_group_scores = []  # [v6_mean, v7_mean, cb_mean]
        for ms, mf_g in model_groups:
            X = snap_df[mf_g].values
            preds = [m.predict(X) for m in ms]
            all_individual.extend(preds)
            scores = np.mean(preds, axis=0)
            all_scores.append(scores)
            per_group_scores.append(scores)

        # Confidence = model agreement (always computed for other modules)
        if len(all_individual) > 1:
            normed = []
            for p in all_individual:
                normed.append((p - p.mean()) / (p.std() + 1e-10))
            model_std = np.std(normed, axis=0)
            confidence = 1.0 / (1.0 + model_std)
        else:
            confidence = np.ones(len(snap_df)) * 0.5

        # Step 2: Meta-model stacking (if enabled and v6+v7+cb all present)
        if _meta_model_inf is not None and all(k in _meta_group_idx for k in ('v6', 'v7', 'cb')):
            _xgb_scores = per_group_scores[_meta_group_idx['xgb']] if 'xgb' in _meta_group_idx else None
            meta_scores = _meta_model_inf.predict(
                snap_df,
                pred_v6=per_group_scores[_meta_group_idx['v6']],
                pred_v7=per_group_scores[_meta_group_idx['v7']],
                pred_cb=per_group_scores[_meta_group_idx['cb']],
                pred_xgb=_xgb_scores,
            )
            return meta_scores, confidence

        # Fallback: simple mean (original behavior)
        mean_scores = np.mean(all_scores, axis=0)
        return mean_scores, confidence

    def predict_deriv_gate(snap_df, ensemble_scores):
        """Compute per-symbol risk scale (0.3–1.0) from derivatives-only model.

        Logic: deriv model predicts cross-sectional rank (like main models).
        If deriv model AGREES with ensemble direction → scale=1.0 (full position).
        If deriv model DISAGREES → scale down to 0.3 (reduce position).
        Agreement measured by rank correlation per position.

        Returns: dict {symbol: scale_factor}
        """
        if deriv_models is None:
            return {}  # no deriv model → no gating

        d_ms, d_feats = deriv_models
        X_d = snap_df[d_feats].values
        d_preds = np.mean([m.predict(X_d) for m in d_ms], axis=0)

        syms = snap_df['symbol'].values
        scale_dict = {}

        # Compute ranks (higher = more bullish)
        ens_rank = np.argsort(np.argsort(ensemble_scores)).astype(float)
        drv_rank = np.argsort(np.argsort(d_preds)).astype(float)
        n = len(ens_rank)
        if n < 5:
            return {s: 1.0 for s in syms}

        # Normalize ranks to [0, 1]
        ens_rank /= (n - 1)
        drv_rank /= (n - 1)

        for i, sym in enumerate(syms):
            # How different is this symbol's rank in ensemble vs deriv?
            # 0 = perfect agreement, 1 = complete disagreement
            rank_diff = abs(ens_rank[i] - drv_rank[i])

            # Focus on extremes: only penalize when ensemble says strong long/short
            # but deriv says the opposite
            ens_extreme = abs(ens_rank[i] - 0.5) * 2  # 0=middle, 1=extreme

            # Disagreement matters more for extreme positions
            effective_disagree = rank_diff * ens_extreme

            # Map: 0 disagree → scale=1.0, 0.7+ disagree → scale=0.3
            scale = DERIV_GATE_MAX - effective_disagree * (DERIV_GATE_MAX - DERIV_GATE_MIN) / 0.7
            scale = float(np.clip(scale, DERIV_GATE_MIN, DERIV_GATE_MAX))
            scale_dict[sym] = scale

        return scale_dict

    # ── 4  timestamps (rebal_h apart) ─────────────────────────────
    all_ts = sorted(df["timestamp"].unique())
    sim_start = max(0, len(all_ts) - sim_h)
    steps = all_ts[sim_start::rebal_h]      # every rebal_h hours

    # Apply --start-date / --end-date filters
    if args.start_date:
        sd = pd.Timestamp(args.start_date, tz='UTC')
        steps = [t for t in steps if t >= sd]
    if args.end_date:
        ed = pd.Timestamp(args.end_date, tz='UTC')
        steps = [t for t in steps if t <= ed]
    if len(steps) < 5:
        print(f"❌ Only {len(steps)} steps after date filters — too few"); return

    print(f"   {steps[0]} → {steps[-1]}  ({len(steps)} steps, {rebal_h}h apart)")

    # ── 4b  calibrate edge threshold ──────────────────────────────
    edge_p75 = 0.0   # used by edge-boost sizing
    need_calibrate = (args.edge_pct > 0 and min_edge == 0.0) or args.edge_boost or args.adaptive_rebal
    if need_calibrate:
        label = f"P{args.edge_pct}" if args.edge_pct > 0 else "P75 (for boost/adaptive)"
        print(f"\n📐 Calibrating edge distribution ({label}) ...")
        edge_samples = []
        # Calibrate on FIRST 30 steps (warmup) to avoid lookahead bias
        # (was: last 30 steps = future data the sim will trade on)
        cal_steps = steps[:min(30, len(steps))]
        for ts in cal_steps:
            snap = df[df["timestamp"] == ts]
            if len(snap) < 20:
                continue
            scores, _ = predict_ensemble(snap)
            median_s = np.median(scores)
            edges_abs = np.abs(scores - median_s)
            edge_samples.extend(edges_abs.tolist())
        if edge_samples:
            if args.edge_pct > 0:
                min_edge = float(np.percentile(edge_samples, args.edge_pct))
            edge_p75 = float(np.percentile(edge_samples, 75))
            edge_p90 = float(np.percentile(edge_samples, 90))
            print(f"   P75 edge = {edge_p75:.5f}, P90 = {edge_p90:.5f}")
            if min_edge > 0:
                print(f"   Filter threshold (P{args.edge_pct}) = {min_edge:.5f}")
            print(f"   ({len(edge_samples)} samples, {len(cal_steps)} steps)")
        else:
            print("   ⚠️  No calibration data, edge features disabled")
            min_edge = 0.0
            edge_p75 = 0.0
            edge_p90 = 0.0

    # ── 5  pre-compute per-coin realized vol (for vol-adjusted sizing) ──
    coin_vol: dict[str, float] = {}
    if args.vol_size:
        print(f"\n📊 Pre-computing per-coin realized vol (24h rolling std)...")
        df = df.sort_values(['symbol', 'timestamp'])
        df['_ret_1h'] = df.groupby('symbol')['close'].pct_change(1)
        df['_rvol_24h'] = df.groupby('symbol')['_ret_1h'].transform(
            lambda x: x.rolling(24, min_periods=12).std()
        )
        latest = df.dropna(subset=['_rvol_24h']).groupby('symbol')['_rvol_24h'].last()
        coin_vol = latest.to_dict()
        df.drop(columns=['_ret_1h', '_rvol_24h'], inplace=True)
        print(f"   Vol computed for {len(coin_vol)} symbols, "
              f"median={np.median(list(coin_vol.values())):.4f}")

    # ── 5b  pre-compute regime indicator (for regime shorts / meta-risk) ──
    regime_col = None
    if args.regime_shorts > 0 or args.meta_risk:
        for rcol in ['btc_regime_168', 'btc_above_ma720', 'btc_trend_ma_168']:
            if rcol in df.columns:
                regime_col = rcol
                break
        if regime_col:
            print(f"   Regime column: '{regime_col}'")
        else:
            print("   ⚠️  No regime column found, regime features disabled")

    # ── 5c  vol targeting setup ───────────────────────────────────
    vol_target_ann = args.vol_target_ann
    if vol_target_ann > 0:
        # Convert annual vol target to per-step vol target
        steps_per_year = 365 * 24 / rebal_h
        vol_target_step = vol_target_ann / np.sqrt(steps_per_year)
        print(f"   Vol targeting: {vol_target_ann:.0%} annual → "
              f"{vol_target_step:.4f} per-step, lookback={vol_lb}")
    else:
        vol_target_step = 0.0

    # ── 5d  meta-risk scaler setup ────────────────────────────────
    if args.meta_risk:
        print(f"   Meta-risk scaler: ON (range {META_RISK_MIN:.1f}x–{META_RISK_MAX:.1f}x)")

    def compute_meta_risk(confidence_arr, scores_arr, ret_buf, equity, peak,
                          snap_df=None, regime_col=None):
        """Compute risk scaling factor (0.3 – 1.5) from multiple risk signals.

        Factors (each contributes a 0-1 score, then combined):
        1. Model agreement  — mean confidence of top/bottom selections
        2. Score spread     — how differentiated are long vs short scores
        3. Recent perf      — rolling win rate from ret_buf
        4. Current DD depth — deeper DD → reduce risk
        5. Regime           — bull regime → slightly higher base
        """
        signals = []
        weights = []

        # 1. Model agreement (higher = better, range ~0.3-0.8 typically)
        if len(confidence_arr) > 0:
            mean_conf = np.mean(confidence_arr)
            # Map 0.3-0.7 → 0-1
            conf_score = np.clip((mean_conf - 0.3) / 0.4, 0.0, 1.0)
            signals.append(conf_score)
            weights.append(0.25)

        # 2. Score spread (higher = model has clear opinions)
        if len(scores_arr) > 5:
            spread = np.percentile(scores_arr, 90) - np.percentile(scores_arr, 10)
            # Map spread: low (<0.01) = uncertain, high (>0.05) = confident
            spread_score = np.clip(spread / 0.05, 0.0, 1.0)
            signals.append(spread_score)
            weights.append(0.20)

        # 3. Recent performance (rolling win rate, last 20 steps)
        if len(ret_buf) >= 5:
            recent = ret_buf[-20:]
            recent_wr = sum(1 for r in recent if r > 0) / len(recent)
            # Map: 40% WR → 0.0, 60% → 1.0
            perf_score = np.clip((recent_wr - 0.40) / 0.20, 0.0, 1.0)
            signals.append(perf_score)
            weights.append(0.25)
        else:
            # Not enough history → neutral
            signals.append(0.5)
            weights.append(0.15)

        # 4. DD depth — deeper DD means we should be cautious
        dd_now = equity / peak - 1 if peak > 0 else 0
        # Map: 0% DD → 1.0, -15%→ 0.3, -20% → 0.0
        dd_score = np.clip(1.0 + dd_now / 0.20, 0.0, 1.0)
        signals.append(dd_score)
        weights.append(0.20)

        # 5. Regime (optional)
        if snap_df is not None and regime_col and regime_col in snap_df.columns:
            btc_snap = snap_df[snap_df['symbol'] == 'BTC/USDT']
            if len(btc_snap) > 0:
                regime_val = btc_snap[regime_col].values[0]
                # Bull regime → slightly favor risk (0.6), bear → cautious (0.3)
                regime_score = 0.3 + 0.4 * regime_val  # maps 0→0.3, 1→0.7
                signals.append(regime_score)
                weights.append(0.10)

        # Weighted combination → scale
        if not signals:
            return 1.0
        w = np.array(weights)
        w = w / w.sum()
        composite = np.dot(signals, w)
        # Map composite (0-1) to risk scale (META_RISK_MIN - META_RISK_MAX)
        risk_scale = META_RISK_MIN + composite * (META_RISK_MAX - META_RISK_MIN)
        return float(np.clip(risk_scale, META_RISK_MIN, META_RISK_MAX))

    # ── 6  simulate ───────────────────────────────────────────────
    print(f"\n{'─'*70}")
    equity   = args.capital
    shadow_equity = args.capital      # tracks market while DDStop is active
    peak     = args.capital
    stopped  = False
    ret_buf: list[float] = []
    skip_count = 0                    # steps where edge filter blocked all positions
    early_rebal_count = 0             # adaptive early rebalances triggered
    event_reduce_count = 0            # steps where event filter reduced leverage
    meta_risk_sum = 0.0               # for reporting avg meta-risk
    meta_risk_count = 0
    vol_scale_sum = 0.0               # for reporting avg vol scale
    vol_scale_count = 0
    deriv_gate_sum = 0.0              # for reporting avg deriv gate scale
    deriv_gate_count = 0

    held_L: dict[str, float] = {}     # symbol → entry_price
    held_S: dict[str, float] = {}
    prev_alloc_L: dict[str, float] = {}   # symbol → dollar allocation (prev step)
    prev_alloc_S: dict[str, float] = {}
    prev_scores: dict[str, float] = {}    # symbol → previous score (for signal smoothing)
    results: list[dict] = []
    cum_cost = 0.0
    cum_dollar_turnover = 0.0
    tot_trades = 0

    # Build step schedule for adaptive rebalance
    if args.adaptive_rebal:
        # Base rebalance every rebal_h; also check at half-intervals for P90+ opportunities
        check_interval = max(rebal_h // 2, 4)  # check every half-period (min 4h)
        all_check_ts = all_ts[sim_start::check_interval]
        # Mark which are "base" rebalance times vs "check" times
        base_set = set(all_ts[sim_start::rebal_h])
        step_schedule = [(ts, ts in base_set) for ts in all_check_ts]
    else:
        step_schedule = [(ts, True) for ts in steps]  # all are base rebalances

    for si in range(len(step_schedule) - 1):
        ts0 = step_schedule[si][0]
        is_base = step_schedule[si][1]

        # Find next timestamp in schedule
        ts1 = step_schedule[si + 1][0]

        snap0 = df[df["timestamp"] == ts0]
        snap1 = df[df["timestamp"] == ts1]
        if len(snap0) < 20 or len(snap1) < 20:
            continue

        px0 = dict(zip(snap0["symbol"], snap0["close"]))
        px1 = dict(zip(snap1["symbol"], snap1["close"]))

        # ── predict & rank at ts0 ────────────────────────────────
        scores, confidence = predict_ensemble(snap0)
        syms = snap0["symbol"].values

        # ── Signal smoothing: EMA blend with previous step scores ─
        if args.smooth_signal > 0 and prev_scores:
            alpha = args.smooth_signal  # weight on previous
            smoothed = np.copy(scores)
            for i_s, sym in enumerate(syms):
                if sym in prev_scores:
                    smoothed[i_s] = (1 - alpha) * scores[i_s] + alpha * prev_scores[sym]
            scores = smoothed
        # Store for next step
        prev_scores = dict(zip(syms, scores))

        median_score = np.median(scores)
        edges = scores - median_score
        abs_edges = np.abs(edges)

        # Adaptive rebalance check: skip non-base steps unless strong signal
        if args.adaptive_rebal and not is_base:
            max_edge = np.max(abs_edges)
            if max_edge < edge_p90:
                # No exceptional signal → don't rebalance, just hold
                # Still compute PnL for held positions
                mtm_pnl = 0.0
                n_held = len(held_L) + len(held_S)
                if n_held > 0:
                    alloc_per = (equity * kelly * leverage) / n_held
                    for sym, ep in held_L.items():
                        p0 = px0.get(sym); p1 = px1.get(sym, p0)
                        if p0 and p1 and ep:
                            mtm_pnl += alloc_per * (p1 - p0) / p0
                    for sym, ep in held_S.items():
                        p0 = px0.get(sym); p1 = px1.get(sym, p0)
                        if p0 and p1 and ep:
                            mtm_pnl -= alloc_per * (p1 - p0) / p0
                    # Funding cost for held period
                    if leverage > 1:
                        hours_held = check_interval
                        fc = (equity * kelly * leverage) * FUNDING_PER_8H * (hours_held / 8.0)
                        mtm_pnl -= fc
                    equity += mtm_pnl
                    peak = max(peak, equity)
                    # Update entry prices
                    held_L = {s: px1.get(s, 0) for s in held_L}
                    held_S = {s: px1.get(s, 0) for s in held_S}
                continue
            else:
                early_rebal_count += 1

        # Edge filtering: select positions based on edge
        order_desc = np.argsort(-scores)
        order_asc  = np.argsort(scores)

        # ── Dynamic leverage: scale leverage based on edge strength ──
        if args.dynamic_lev and edge_p75 > 0 and edge_p90 > 0:
            max_abs_edge = np.max(abs_edges)
            # Require: edge > P90 AND recent returns positive (momentum)
            recent_ok = len(ret_buf) >= 3 and np.mean(ret_buf[-3:]) > 0
            # Also: current DD must be shallow (not in drawdown recovery)
            dd_now = equity / peak - 1 if peak > 0 else 0
            dd_ok = dd_now > -0.08  # only scale up if DD < 8%

            if max_abs_edge >= edge_p90 and recent_ok and dd_ok:
                # Gradual scale: P90 edge → base+25%, P90×1.5 → max_lev
                overshoot = (max_abs_edge - edge_p90) / (edge_p90 * 0.5 + 1e-10)
                lev_ratio = min(overshoot, 1.0)
                step_leverage = leverage + lev_ratio * (args.max_lev - leverage)
            else:
                step_leverage = leverage
        else:
            step_leverage = leverage

        # ── Event filter: reduce leverage near macro events ──────
        if args.event_filter:
            near_evt, evt_name = is_near_event(ts0)
            if near_evt:
                step_leverage = max(1.0, step_leverage * 0.3)  # reduce to 30% of planned lev
                event_reduce_count += 1

        # ── Dynamic N: adjust positions per side based on dispersion ──
        if args.dynamic_n:
            score_z = (scores - np.mean(scores)) / (np.std(scores) + 1e-10)
            disp = np.std(score_z)
            # Low dispersion → fewer positions, high → up to n_pos
            # disp typical range: 0.7-1.5. Thresholds tuned empirically.
            if disp < 0.85:
                step_n_pos = max(3, n_pos // 2)  # min 3 per side
            elif disp < 1.0:
                step_n_pos = max(5, int(n_pos * 0.7))
            else:
                step_n_pos = n_pos
        else:
            step_n_pos = n_pos

        # ── Min z-score filter: skip weak signals ──
        if args.min_zscore > 0:
            score_z = (scores - np.mean(scores)) / (np.std(scores) + 1e-10)

        if min_edge > 0:
            # Select only positions with sufficient edge
            long_idx = []
            for idx in order_desc:
                if edges[idx] >= min_edge:
                    # Min z-score filter
                    if args.min_zscore > 0:
                        if score_z[idx] < args.min_zscore:
                            continue
                    long_idx.append(idx)
                if len(long_idx) >= step_n_pos:
                    break
            short_idx = []
            for idx in order_asc:
                if edges[idx] <= -min_edge:
                    # Min z-score filter
                    if args.min_zscore > 0:
                        if score_z[idx] > -args.min_zscore:
                            continue
                    # Skip OKX-blocked symbols when --short-blocked is on
                    if args.short_blocked and _OKX_BLOCKED and syms[idx] in _OKX_BLOCKED:
                        continue
                    short_idx.append(idx)
                if len(short_idx) >= step_n_pos:
                    break
            new_L = set(syms[long_idx]) if long_idx else set()
            new_S = set(syms[short_idx]) if short_idx else set()
            nl = len(long_idx)
        else:
            n = len(syms)
            nl = min(step_n_pos, n // 3)
            if args.min_zscore > 0:
                score_z_loc = (scores - np.mean(scores)) / (np.std(scores) + 1e-10)
                cand_L = [i for i in order_desc if score_z_loc[i] >= args.min_zscore]
                cand_S = [i for i in order_asc if score_z_loc[i] <= -args.min_zscore]
                new_L = set(syms[cand_L[:nl]])
                new_S = set(syms[cand_S[:nl]])
            else:
                new_L = set(syms[order_desc[:nl]])
                new_S = set(syms[order_asc[:nl]])

        # ── Short-blocked filter: skip OKX-restricted symbols for shorts ──
        if args.short_blocked and _OKX_BLOCKED:
            blocked_in_S = new_S & _OKX_BLOCKED
            if blocked_in_S:
                # Remove blocked symbols, backfill from next-worst candidates
                new_S -= blocked_in_S
                shortable_idx = [i for i in order_asc
                                 if syms[i] not in new_S
                                 and syms[i] not in new_L
                                 and syms[i] not in _OKX_BLOCKED]
                for idx in shortable_idx:
                    if len(new_S) >= nl:
                        break
                    if min_edge > 0 and edges[idx] > -min_edge:
                        continue
                    new_S.add(syms[idx])

        # ── Hysteresis: keep incumbents unless they fall below threshold ──
        if args.hysteresis > 0 and (held_L or held_S):
            sym_to_rank_desc = {syms[order_desc[i]]: i for i in range(len(order_desc))}  # 0 = best long
            sym_to_rank_asc = {syms[order_asc[i]]: i for i in range(len(order_asc))}  # 0 = best short
            keep_threshold = step_n_pos + args.hysteresis  # e.g. 10 + 5 = 15

            # Longs: keep incumbent if rank < keep_threshold, even if not in new top-N
            sticky_L = set()
            for s in held_L:
                if s in sym_to_rank_desc and sym_to_rank_desc[s] < keep_threshold:
                    sticky_L.add(s)
            # Merge: incumbents that still qualify + new entries
            merged_L = sticky_L | new_L
            # If too many, trim worst by score
            if len(merged_L) > step_n_pos:
                sym_list = list(syms)
                ranked = sorted(merged_L, key=lambda s: -scores[sym_list.index(s)])
                merged_L = set(ranked[:step_n_pos])

            # Shorts: symmetric
            sticky_S = set()
            for s in held_S:
                if s in sym_to_rank_asc and sym_to_rank_asc[s] < keep_threshold:
                    sticky_S.add(s)
            merged_S = sticky_S | new_S
            if len(merged_S) > step_n_pos:
                ranked = sorted(merged_S, key=lambda s: scores[list(syms).index(s)])
                merged_S = set(ranked[:step_n_pos])

            new_L = merged_L
            new_S = merged_S

        # ── Turnover budget: cap replacements per side ──
        if args.turnover_budget > 0 and (held_L or held_S):
            budget = args.turnover_budget

            # Longs: limit exits
            exits_L = set(held_L) - new_L
            enters_L = new_L - set(held_L)
            if len(exits_L) > budget:
                # Keep the best-scoring exits (they're least bad)
                exits_sorted = sorted(exits_L, key=lambda s: -scores[list(syms).index(s)])
                forced_keep = set(exits_sorted[:len(exits_L) - budget])
                new_L = new_L | forced_keep
                # If too many now, drop worst new entries
                if len(new_L) > step_n_pos:
                    ranked = sorted(new_L, key=lambda s: -scores[list(syms).index(s)])
                    new_L = set(ranked[:step_n_pos])

            # Shorts: symmetric
            exits_S = set(held_S) - new_S
            enters_S = new_S - set(held_S)
            if len(exits_S) > budget:
                exits_sorted = sorted(exits_S, key=lambda s: scores[list(syms).index(s)])
                forced_keep = set(exits_sorted[:len(exits_S) - budget])
                new_S = new_S | forced_keep
                if len(new_S) > step_n_pos:
                    ranked = sorted(new_S, key=lambda s: scores[list(syms).index(s)])
                    new_S = set(ranked[:step_n_pos])

        if len(new_L) == 0 and len(new_S) == 0:
            # No positions pass the edge filter — skip step
            skip_count += 1
            results.append(dict(step=si, ts=str(ts0), pnl=0,
                                eq=round(equity, 2), dd=round(equity/peak-1, 4),
                                nL=0, nS=0, turn=0, skipped=True))
            held_L.clear(); held_S.clear()
            prev_alloc_L.clear(); prev_alloc_S.clear()
            continue

        # ── Min confidence filter will be applied after conf_dict is built ──

        # Confidence-weighted sizing with optional edge boost
        score_dict = dict(zip(syms, scores))
        edge_dict = dict(zip(syms, abs_edges))
        conf_dict = dict(zip(syms, confidence))

        # ── Min confidence filter: remove low-agreement signals ──
        if args.min_conf > 0:
            new_L = {s for s in new_L if conf_dict.get(s, 0) >= args.min_conf}
            new_S = {s for s in new_S if conf_dict.get(s, 0) >= args.min_conf}
            if len(new_L) == 0 and len(new_S) == 0:
                skip_count += 1
                results.append(dict(step=si, ts=str(ts0), pnl=0,
                                    eq=round(equity, 2), dd=round(equity/peak-1, 4),
                                    nL=0, nS=0, turn=0, skipped=True))
                held_L.clear(); held_S.clear()
                prev_alloc_L.clear(); prev_alloc_S.clear()
                continue

        def compute_weights(symbols, is_long=True):
            """Compute position weights with optional edge-boost × confidence.
            Cap per position = its confidence (high conf → more capital allowed).
            Optional vol-adjustment: weight ∝ 1/σ (inverse-volatility sizing)."""
            if len(symbols) == 0:
                return {}
            syms_list = list(symbols)

            # --- Base weights ---
            if args.softmax_temp > 0:
                # Softmax score-proportional: weight ∝ exp(score × temp)
                if is_long:
                    arr = np.array([score_dict[s] for s in syms_list])
                else:
                    arr = np.array([-score_dict[s] for s in syms_list])
                arr = arr - arr.mean()  # numerical stability
                w = np.exp(arr * args.softmax_temp)
                w = w / w.sum()
            elif args.edge_boost and edge_p75 > 0:
                raw_w = []
                conf_arr = []
                for s in syms_list:
                    e = edge_dict.get(s, 0)
                    ratio = e / edge_p75
                    boost = 1.0 + min(ratio, 3.0)
                    c = conf_dict.get(s, 0.5) if not getattr(args, 'no_conf', False) else 1.0
                    raw_w.append(boost * c)
                    conf_arr.append(c)
                raw_w = np.array(raw_w)
                w = raw_w / raw_w.sum()
                max_w = np.clip(np.array(conf_arr), 0.15, 0.40)
                w = np.minimum(w, max_w)
            else:
                if is_long:
                    arr = np.array([score_dict[s] for s in syms_list])
                else:
                    arr = np.array([-score_dict[s] for s in syms_list])
                arr = arr - arr.mean()
                w = np.exp(arr * 2)
                w = w / w.sum()

            # --- Vol-adjusted sizing: scale by 1/σ ---
            if args.vol_size:
                vol_arr = np.array([coin_vol.get(s, 0.05) for s in syms_list])
                vol_arr = np.clip(vol_arr, 0.005, 0.20)  # cap extremes
                inv_vol = 1.0 / vol_arr
                w = w * inv_vol
                w = w / w.sum()  # re-normalise

            return dict(zip(syms_list, w))

        weight_L = compute_weights(new_L, is_long=True)
        weight_S = compute_weights(new_S, is_long=False)

        # ── Deriv risk gate: scale per-symbol weights by deriv agreement ──
        # NO per-side renormalisation: if deriv disagrees on some longs,
        # the long side naturally gets less capital (sum(weight_L) < 1.0).
        # Each side is scaled independently → no separate gross scaling needed.
        if deriv_models is not None and use_deriv_gate:
            deriv_scale = predict_deriv_gate(snap0, scores)
            if deriv_scale:
                for s in weight_L:
                    weight_L[s] *= deriv_scale.get(s, 1.0)
                for s in weight_S:
                    weight_S[s] *= deriv_scale.get(s, 1.0)

                # Track avg gate for logging
                all_selected = list(new_L | new_S)
                avg_gate = np.mean([deriv_scale.get(s, 1.0) for s in all_selected])
                deriv_gate_sum += float(np.clip(avg_gate, DERIV_GATE_MIN, DERIV_GATE_MAX))
                deriv_gate_count += 1

        # ── Regime short scaling: reduce shorts in bull regime ────
        regime_scale = 1.0
        if args.regime_shorts > 0 and regime_col:
            # Get regime value from current snapshot (BTC row)
            btc_snap = snap0[snap0['symbol'] == 'BTC/USDT']
            if len(btc_snap) > 0 and regime_col in btc_snap.columns:
                regime_val = btc_snap[regime_col].values[0]
                if regime_val > 0.5:  # bullish regime
                    regime_scale = args.regime_shorts  # e.g. 0.5 = halve short alloc

        # ── compute changes ───────────────────────────────────────
        open_L  = new_L - set(held_L)
        close_L = set(held_L) - new_L
        open_S  = new_S - set(held_S)
        close_S = set(held_S) - new_S

        # Compute actual hours between ts0→ts1 for funding calc
        hours_between = max(1, int((ts1 - ts0).total_seconds() / 3600)) if hasattr(ts1, 'total_seconds') else rebal_h

        total_alloc = equity * kelly * step_leverage

        # ── Vol targeting: scale gross exposure to target portfolio vol ──
        if vol_target_step > 0 and len(ret_buf) >= max(10, vol_lb // 2):
            recent_rets = np.array(ret_buf[-vol_lb:])
            realized_vol = np.std(recent_rets)
            if realized_vol > 1e-8:
                vol_scale = vol_target_step / realized_vol
                vol_scale = np.clip(vol_scale, 0.2, 2.0)  # don't go crazy
            else:
                vol_scale = 1.0
            total_alloc *= vol_scale
            vol_scale_sum += vol_scale
            vol_scale_count += 1

        # ── Meta-risk scaler: adjust gross exposure based on multi-signal ──
        if args.meta_risk:
            # Use confidence of selected positions (long + short)
            selected_idx = list(new_L | new_S)
            sel_conf = np.array([conf_dict.get(s, 0.5) for s in selected_idx])
            sel_scores = scores  # all scores for spread calc
            risk_scale = compute_meta_risk(
                sel_conf, sel_scores, ret_buf, equity, peak,
                snap_df=snap0, regime_col=regime_col
            )
            total_alloc *= risk_scale
            meta_risk_sum += risk_scale
            meta_risk_count += 1

        half_alloc = total_alloc / 2  # half for longs, half for shorts
        short_alloc = half_alloc * regime_scale  # reduced in bull regime

        # ── Dollar-turnover cost model ────────────────────────────
        # Compute target dollar allocation for each symbol
        new_alloc_L = {s: half_alloc * weight_L.get(s, 0) for s in new_L}
        new_alloc_S = {s: short_alloc * weight_S.get(s, 0) for s in new_S}

        # Dollar turnover = sum |new_alloc − prev_alloc| per symbol
        dollar_turnover = 0.0
        all_L = set(new_alloc_L) | set(prev_alloc_L)
        for s in all_L:
            dollar_turnover += abs(new_alloc_L.get(s, 0) - prev_alloc_L.get(s, 0))
        all_S = set(new_alloc_S) | set(prev_alloc_S)
        for s in all_S:
            dollar_turnover += abs(new_alloc_S.get(s, 0) - prev_alloc_S.get(s, 0))

        step_cost = dollar_turnover * COST_SIDE
        cum_dollar_turnover += dollar_turnover

        # Funding cost for leveraged positions (proportional to hold time)
        if leverage > 1:
            funding_periods = rebal_h / 8.0  # how many 8h funding intervals
            funding_cost = total_alloc * FUNDING_PER_8H * funding_periods
            step_cost += funding_cost
        cum_cost += step_cost
        tot_trades += len(open_L) + len(close_L) + len(open_S) + len(close_S)

        # ── dd breaker (checked BEFORE PnL to avoid phantom trades) ─
        if stopped:
            # Equity stays flat while stopped; track shadow to decide resume
            shadow_pnl = 0.0
            for sym in new_L:
                p0 = px0.get(sym, 0); p1 = px1.get(sym, p0)
                w = weight_L.get(sym, 1.0 / max(len(new_L), 1))
                if p0 > 0: shadow_pnl += half_alloc * w * (p1 - p0) / p0
            for sym in new_S:
                p0 = px0.get(sym, 0); p1 = px1.get(sym, p0)
                w = weight_S.get(sym, 1.0 / max(len(new_S), 1))
                if p0 > 0: shadow_pnl += short_alloc * w * (-(p1 - p0) / p0)
            # Update shadow equity to track market recovery
            shadow_equity += shadow_pnl - step_cost
            shadow_dd = shadow_equity / peak - 1
            if shadow_dd > dd_resume:
                stopped = False
                # Resume from NEXT step — this step we're still flat
            results.append(dict(step=si, ts=str(ts0), pnl=0,
                                eq=round(equity, 2), dd=round(equity / peak - 1, 4),
                                nL=0, nS=0, turn=0, stopped=True))
            held_L.clear(); held_S.clear()
            prev_alloc_L.clear(); prev_alloc_S.clear()
            continue

        # ── PnL from ts0→ts1 for NEW portfolio ───────────────────
        fwd_pnl = 0.0
        for sym in new_L:
            p0 = px0.get(sym, 0); p1 = px1.get(sym, p0)
            w = weight_L.get(sym, 1.0 / max(len(new_L), 1))
            if p0 > 0: fwd_pnl += half_alloc * w * (p1 - p0) / p0
        for sym in new_S:
            p0 = px0.get(sym, 0); p1 = px1.get(sym, p0)
            w = weight_S.get(sym, 1.0 / max(len(new_S), 1))
            if p0 > 0: fwd_pnl += short_alloc * w * (-(p1 - p0) / p0)

        equity += fwd_pnl - step_cost
        shadow_equity = equity  # keep shadow in sync while trading
        peak = max(peak, equity)
        dd = equity / peak - 1

        if fwd_pnl != 0:
            prev_eq = equity - fwd_pnl + step_cost
            if prev_eq > 0:
                ret_buf.append((fwd_pnl - step_cost) / prev_eq)
                ret_buf = ret_buf[-200:]

        if dd < dd_stop:
            stopped = True
            held_L.clear(); held_S.clear()
            prev_alloc_L.clear(); prev_alloc_S.clear()
            continue

        turn = len(open_L) + len(close_L) + len(open_S) + len(close_S)
        results.append(dict(step=si, ts=str(ts0),
                            pnl=round(fwd_pnl - step_cost, 2),
                            eq=round(equity, 2), dd=round(dd, 4),
                            nL=len(new_L), nS=len(new_S), turn=turn))

        # Update held for next step mtm
        held_L = {s: px1.get(s, 0) for s in new_L}
        held_S = {s: px1.get(s, 0) for s in new_S}
        prev_alloc_L = new_alloc_L
        prev_alloc_S = new_alloc_S

        if si % 5 == 0 or si == len(step_schedule) - 2:
            print(f"   {si:>4d}/{len(step_schedule)-1} | ${equity:>8,.2f} | "
                  f"DD {dd:>6.1%} | L{len(new_L)} S{len(new_S)} | "
                  f"Δ{turn}")

    # ── 6  summary ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  RESULTS — {args.days}d, rebal={rebal_h}h, N={n_pos}+{n_pos}, "
          f"lev={lev_str}, mode={mode_str}")
    print(f"{'='*70}")

    tot_ret = equity / args.capital - 1
    max_dd  = min(r["dd"] for r in results) if results else 0
    pnls    = [r["pnl"] for r in results if r["pnl"] != 0]

    if pnls:
        w = [p for p in pnls if p > 0]
        l = [p for p in pnls if p < 0]
        wr = len(w) / len(pnls)
        a  = np.array(pnls)
        sh = np.mean(a) / (np.std(a) + 1e-10) * np.sqrt(365 * 24 / rebal_h)

        ann_ret = tot_ret * (365 / args.days)
        calmar = ann_ret / (abs(max_dd) + 1e-10) if max_dd < 0 else 999

        print(f"\n   Start:      ${args.capital:,.0f}")
        print(f"   End:        ${equity:,.2f}")
        print(f"   Return:     {tot_ret:+.1%}  (ann. ~{ann_ret:+.0%})")
        print(f"   Max DD:     {max_dd:.1%}")
        print(f"   Sharpe:     {sh:+.2f}")
        sh_hac = compute_hac_sharpe(pnls, rebal_h)
        print(f"   Sharpe HAC: {sh_hac:+.2f}  (Newey-West adjusted)")
        print(f"   Calmar:     {calmar:.2f}")
        print(f"   Win Rate:   {wr:.0%}  ({len(w)}W / {len(l)}L)")
        if w: print(f"   Avg Win:    ${np.mean(w):+.2f}")
        if l: print(f"   Avg Loss:   ${np.mean(l):+.2f}")
        if l: print(f"   PF:         {sum(w)/(abs(sum(l))+1e-10):.2f}")
        print(f"   Trades:     {tot_trades}")
        turnover_rate = cum_dollar_turnover / (args.capital + 1e-10)
        print(f"   Turnover:   ${cum_dollar_turnover:,.0f}  ({turnover_rate:.1f}x capital)")
        print(f"   Costs:      ${cum_cost:,.2f}  ({cum_cost/args.capital*100:.1f}%)")
        if skip_count > 0:
            print(f"   Skipped:    {skip_count} steps (no edge)")
        if early_rebal_count > 0:
            print(f"   Early rebal:{early_rebal_count} (adaptive P90+ triggers)")
        if event_reduce_count > 0:
            print(f"   Event filt: {event_reduce_count} steps (leverage reduced near FOMC/CPI)")
        if vol_scale_count > 0:
            avg_vs = vol_scale_sum / vol_scale_count
            print(f"   Vol target: {vol_target_ann:.0%} ann, avg scale {avg_vs:.2f}x")
        if meta_risk_count > 0:
            avg_mr = meta_risk_sum / meta_risk_count
            print(f"   Meta-risk:  avg scale {avg_mr:.2f}x ({META_RISK_MIN:.1f}–{META_RISK_MAX:.1f})")
        if deriv_gate_count > 0:
            avg_dg = deriv_gate_sum / deriv_gate_count
            print(f"   Deriv gate: avg scale {avg_dg:.2f}x ({DERIV_GATE_MIN:.1f}–{DERIV_GATE_MAX:.1f})")
        if leverage > 1:
            print(f"   Leverage:   {lev_str}")
            liq_dd = -1.0 / leverage  # approximate liquidation DD
            print(f"   Liq. level: {liq_dd:.0%} DD (approx)")
    else:
        print("\n   No trades.")

    # ── save ──────────────────────────────────────────────────────
    outdir = os.path.join(root, "trading_logs"); os.makedirs(outdir, exist_ok=True)
    ep = os.path.join(outdir, "fast_sim_equity.csv")
    pd.DataFrame(results).to_csv(ep, index=False)

    # ── ascii chart ───────────────────────────────────────────────
    if len(results) > 3:
        eqs = [r["eq"] for r in results]
        s = eqs[::max(1, len(eqs)//50)]
        mn, mx = min(s), max(s)
        rng = mx - mn or 1
        print(f"\n   📈 Equity (${mn:.0f}–${mx:.0f}):")
        for i, e in enumerate(s):
            f_ = int((e - mn) / rng * 40)
            if i % max(1, len(s)//8) == 0 or i == len(s)-1:
                print(f"      ${e:>8,.0f} |{'█'*f_}{'░'*(40-f_)}|")

    print(f"\n   Saved: {ep}")
    print(f"\n{'='*70}")
    if tot_ret > 0.02 and max_dd > -0.15:
        print("   🟢 PROFITABLE — go live")
    elif tot_ret > 0 and max_dd > (-1.0 / leverage if leverage > 1 else -0.3):
        print("   🟡 Marginal — keep monitoring")
    elif leverage > 1 and max_dd < -1.0 / leverage:
        print("   💀 LIQUIDATED — reduce leverage!")
    else:
        print("   🔴 Unprofitable")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
