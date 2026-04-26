"""Independent re-verification of OOS Sharpe finding.
Opus audit: do not trust prior pipelines, recompute from raw parquets.
"""
import pandas as pd, numpy as np, sys, os

OOS_START = pd.Timestamp("2026-03-18", tz="UTC")
OOS_END   = pd.Timestamp("2026-04-26", tz="UTC")  # exclusive

FILES = {
    "R133_r128style": "cache/r133_r128style_preds.parquet",
    "R132_oos":       "cache/r132_oos_preds.parquet",
    "R134_fresh":     "cache/r134_fresh_preds.parquet",
    "PROD_deployed":  "cache/current_prod_cls_oos_preds.parquet",
}


def load(p):
    df = pd.read_parquet(p)
    tcol = 'ts' if 'ts' in df.columns else 'timestamp'
    df = df.rename(columns={tcol: 'ts'})
    df['ts'] = pd.to_datetime(df['ts'], utc=True)
    return df


def section(name):
    print(f"\n{'='*78}\n  {name}\n{'='*78}")


def stat_ic(df):
    d = df.dropna(subset=['pred', 'fwd_ret']).copy()
    if d.empty:
        return None
    ic = d.groupby('ts').apply(
        lambda g: g['pred'].corr(g['fwd_ret'], method='spearman') if len(g) > 3 else np.nan,
        include_groups=False,
    )
    ic = ic.dropna()
    return {
        'ic_mean': float(ic.mean()),
        'ic_std':  float(ic.std()),
        'ic_n':    int(len(ic)),
        'ic_t':    float(ic.mean() / (ic.std() / np.sqrt(len(ic)) + 1e-12)),
    }


def quintile_spread(df):
    d = df.dropna(subset=['pred', 'fwd_ret']).copy()
    d['rk'] = d.groupby('ts')['pred'].rank(pct=True)
    top = d.loc[d['rk'] >= 0.8, 'fwd_ret'].mean() * 1e4
    bot = d.loc[d['rk'] <= 0.2, 'fwd_ret'].mean() * 1e4
    return float(top), float(bot), float(top - bot)


def naive_long_short(df, n_long=4, n_short=2):
    """No regime, no cost, no hysteresis. Pure raw signal->forward return.
    For each ts: long top-n_long by pred, short bottom-n_short. Equal-weight
    inside long & short legs; long leg weight = +1, short leg = -1 (gross 2x).
    """
    d = df.dropna(subset=['pred', 'fwd_ret']).sort_values(['ts', 'pred'])
    rets = []
    tss = []
    for ts, g in d.groupby('ts'):
        if len(g) < n_long + n_short:
            continue
        gs = g.sort_values('pred')
        short = gs.head(n_short)['fwd_ret'].mean()
        longs = gs.tail(n_long)['fwd_ret'].mean()
        rets.append(longs - short)
        tss.append(ts)
    s = pd.Series(rets, index=pd.DatetimeIndex(tss, name='ts')).sort_index()
    return s


def sharpe(r, ppy=730):
    r = np.asarray(r, dtype=float)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(ppy))


def report(name, df_full):
    df = df_full[(df_full['ts'] >= OOS_START) & (df_full['ts'] < OOS_END)].copy()
    print(f"\n--- {name} ---")
    print(f"  rows={len(df)}  ts={df['ts'].nunique()}  syms={df['symbol'].nunique()}")
    print(f"  pred mean={df['pred'].mean():+.5f}  std={df['pred'].std():.5f}  "
          f"min={df['pred'].min():+.3f}  max={df['pred'].max():+.3f}")
    print(f"  fwd_ret(bp) mean={df['fwd_ret'].mean()*1e4:+.2f}  "
          f"std={df['fwd_ret'].std()*1e4:.2f}")
    ic = stat_ic(df)
    if ic:
        print(f"  IC spearman: mean={ic['ic_mean']:+.4f}  t={ic['ic_t']:+.2f}  n_ts={ic['ic_n']}")
    top, bot, spr = quintile_spread(df)
    print(f"  top-Q={top:+.2f}bp  bot-Q={bot:+.2f}bp  spread(t-b)={spr:+.2f}bp")
    # naive long/short, every timestamp, no overlay, no cost
    s = naive_long_short(df, 4, 2)
    print(f"  NAIVE 4L/2S all-ts (no cost): n={len(s)}  "
          f"mean={s.mean()*1e4:+.2f}bp  std={s.std()*1e4:.2f}bp  "
          f"sum={s.sum()*100:+.2f}%  Sharpe={sharpe(s.values):+.3f}")
    # naive every 12h slice
    s12 = s.iloc[::12]
    print(f"  NAIVE 4L/2S 12h-stride: n={len(s12)}  "
          f"mean={s12.mean()*1e4:+.2f}bp  std={s12.std()*1e4:.2f}bp  "
          f"sum={s12.sum()*100:+.2f}%  Sharpe={sharpe(s12.values):+.3f}")
    # INVERTED signal (sanity): if model-flipped is hugely positive => sign error
    df_inv = df.copy()
    df_inv['pred'] = -df_inv['pred']
    s_inv = naive_long_short(df_inv, 4, 2).iloc[::12]
    print(f"  INVERTED 4L/2S 12h:        Sharpe={sharpe(s_inv.values):+.3f}  "
          f"sum={s_inv.sum()*100:+.2f}%")
    # RANDOM baseline
    rng = np.random.default_rng(42)
    df_r = df.copy()
    df_r['pred'] = rng.standard_normal(len(df_r))
    s_r = naive_long_short(df_r, 4, 2).iloc[::12]
    print(f"  RANDOM   4L/2S 12h:        Sharpe={sharpe(s_r.values):+.3f}  "
          f"sum={s_r.sum()*100:+.2f}%")


def market_baseline_btc():
    """Compute realized BTC returns over OOS window for sanity."""
    candidates = [
        "data/raw/BTC_USDT_1h.parquet",
        "data/raw/BTCUSDT_1h.parquet",
    ]
    for path in candidates:
        if os.path.exists(path):
            d = pd.read_parquet(path)
            tcol = 'ts' if 'ts' in d.columns else ('timestamp' if 'timestamp' in d.columns else d.columns[0])
            d = d.rename(columns={tcol: 'ts'})
            d['ts'] = pd.to_datetime(d['ts'], utc=True)
            d = d[(d['ts'] >= OOS_START) & (d['ts'] < OOS_END)].sort_values('ts')
            if 'close' in d.columns:
                px = d['close'].values
            else:
                continue
            if len(px) < 2:
                continue
            r1h = pd.Series(np.diff(px) / px[:-1], index=d['ts'].iloc[1:])
            r12h = r1h.iloc[::12]
            tot = (px[-1] / px[0] - 1.0) * 100
            print(f"\n--- BTC market baseline ({path}) ---")
            print(f"  hours={len(d)}  total_return={tot:+.2f}%  "
                  f"hourly Sharpe={sharpe(r1h.values, 24*365):+.3f}  "
                  f"12h Sharpe={sharpe(r12h.values, 730):+.3f}")
            return
    print("  [no BTC raw parquet found]")


if __name__ == "__main__":
    section("LOAD & DESCRIBE EACH PRED FILE")
    cache = {}
    for k, p in FILES.items():
        if not os.path.exists(p):
            print(f"  SKIP {k}: file missing")
            continue
        df = load(p)
        cache[k] = df
        print(f"  {k}: full rows={len(df)}  ts {df['ts'].min()} -> {df['ts'].max()}  "
              f"uniq_ts={df['ts'].nunique()}  cols={list(df.columns)}")

    section("OOS-WINDOW DEEP STATS PER MODEL")
    for k, df in cache.items():
        report(k, df)

    section("MARKET BASELINE")
    market_baseline_btc()

    section("CHECK fwd_ret CONSISTENCY ACROSS FILES")
    keys = list(cache.keys())
    if len(keys) >= 2:
        a = cache[keys[0]][['ts', 'symbol', 'fwd_ret']].rename(columns={'fwd_ret': 'fa'})
        b = cache[keys[1]][['ts', 'symbol', 'fwd_ret']].rename(columns={'fwd_ret': 'fb'})
        m = a.merge(b, on=['ts', 'symbol'], how='inner')
        diff = (m['fa'] - m['fb']).abs()
        print(f"  {keys[0]} vs {keys[1]}: matched={len(m)}  max|Δfwd|={diff.max():.2e}  mean|Δfwd|={diff.mean():.2e}")
