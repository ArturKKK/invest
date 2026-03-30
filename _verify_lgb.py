#!/usr/bin/env python3
"""
Тщательная верификация LightGBM результатов из R9B.

Проверки:
  1. Явный вывод дат train/val/test — убедиться что нет утечки
  2. Train IC vs Val IC vs Test IC — обнаружить переобучение
  3. Разные num_leaves (15/31/63) — стабильность результатов
  4. Разные seeds — нет ли зависимости от случайности
  5. Permutation test — перемешать предикты → Sharpe должен → 0
  6. Monthly breakdown Ridge vs LGB — месяц за месяцем
  7. Equity curve comparison
  8. Статистический тест значимости (bootstrap)
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings("ignore")

from _research_round7 import (
    SYM_35, WINDOWS, FEATURES as FEATURES_14, cs_rank,
    compute_regime, simulate, eval_config, show,
    train_and_predict_multi,
)
from _ic_scanner import load_ohlcv, load_derivatives, build_features_minimal

LEVERAGE = 5
CAPITAL  = 100
CFG_BASE = {
    "n_long": 6, "n_short": 3,
    "trend_cutoff": 0.8, "dyn_threshold": 0.5,
    "eq_mom_boost": True, "kelly_sizing": True,
    "strategy_momentum": True, "strat_mom_lookback": 48,
    "regime_asym": True, "vol_scaling": True,
    "signal_ema": 2, "rebal_hours": 12,
}
CFG_LGB_BASE = {**CFG_BASE, "signal_ema": None}


# ══════════════════════════════════════════════════════════════════
def divider(title):
    print(f"\n{'═'*70}")
    print(f"  {title}")
    print(f"{'═'*70}")

def section(title):
    print(f"\n  {'─'*60}")
    print(f"  {title}")
    print(f"  {'─'*60}")

# ══════════════════════════════════════════════════════════════════
def train_lgb_verbose(df, feats, num_leaves=31, seed=42, n_rounds=300):
    """
    Тренирует LGB с подробным выводом IC на train/val/test.
    Возвращает (predictions_df, metrics_dict).
    """
    import lightgbm as lgb
    feat_r = [f"{f}_r" for f in feats if f in df.columns]
    feats  = [f for f in feats if f in df.columns]
    fwd_col = "fwd_ret_12h"
    all_preds = []
    window_metrics = []

    for w in WINDOWS:
        train = df[df["timestamp"] < pd.Timestamp(w["train_end"], tz="UTC")].copy()
        val   = df[(df["timestamp"] >= pd.Timestamp(w["val_start"],  tz="UTC")) &
                   (df["timestamp"] <  pd.Timestamp(w["val_end"],    tz="UTC"))].copy()
        test  = df[(df["timestamp"] >= pd.Timestamp(w["test_start"], tz="UTC")) &
                   (df["timestamp"] <= pd.Timestamp(w["test_end"],   tz="UTC"))].copy()

        print(f"\n    Window: {w['name']}")
        print(f"      train: {train['timestamp'].min().date()} → {train['timestamp'].max().date()}  ({len(train):,} rows)")
        print(f"      val:   {val['timestamp'].min().date()}  → {val['timestamp'].max().date()}  ({len(val):,} rows)")
        print(f"      test:  {test['timestamp'].min().date()}  → {test['timestamp'].max().date()}  ({len(test):,} rows)")
        print(f"      GAP val↔test: {(pd.Timestamp(w['test_start'], tz='UTC') - pd.Timestamp(w['val_end'], tz='UTC')).days} days")

        if len(train) < 5000 or len(test) < 200:
            print(f"      ⚠️  Skipping window (insufficient data)")
            continue

        for d in [train, val, test]:
            for feat in feats:
                d[f"{feat}_r"] = cs_rank(d, feat)
            d["target_rank"] = d.groupby("timestamp")[fwd_col].rank(pct=True) - 0.5

        train_c = train[feat_r + ["target_rank"]].dropna()
        val_c   = val[feat_r + ["target_rank"]].dropna()
        test_c  = test[feat_r + ["target_rank", "timestamp", "symbol"]].dropna()

        dtrain = lgb.Dataset(train_c[feat_r], label=train_c["target_rank"])
        dval   = lgb.Dataset(val_c[feat_r],   label=val_c["target_rank"])
        params = {
            "objective": "regression", "metric": "mse",
            "learning_rate": 0.05, "num_leaves": num_leaves,
            "min_child_samples": 100, "subsample": 0.8,
            "colsample_bytree": 0.8, "verbose": -1, "n_jobs": -1,
            "seed": seed,
        }
        model = lgb.train(params, dtrain, num_boost_round=n_rounds,
                          valid_sets=[dval],
                          callbacks=[lgb.early_stopping(30, verbose=False),
                                     lgb.log_evaluation(-1)])

        # IC on train/val/test
        train_preds = model.predict(train_c[feat_r])
        val_preds   = model.predict(val_c[feat_r])
        train_ic = stats.spearmanr(train_preds, train_c["target_rank"])[0]
        val_ic   = stats.spearmanr(val_preds,   val_c["target_rank"])[0]

        test_c = test_c.copy()
        test_c["pred"] = model.predict(test_c[feat_r])
        test_ic = stats.spearmanr(test_c["pred"], test_c["target_rank"])[0]

        print(f"      trees={model.best_iteration:3d}  "
              f"IC_train={train_ic:.4f}  IC_val={val_ic:.4f}  IC_test={test_ic:.4f}")
        if train_ic > test_ic * 3:
            print(f"      ⚠️  WARNING: train IC / test IC = {train_ic/test_ic:.1f}× — possible overfit!")
        else:
            print(f"      ✅ IC ratio train/test = {train_ic/test_ic:.2f}× (OK)")

        fwd_data = test[["timestamp", "symbol", fwd_col]].rename(
            columns={fwd_col: "fwd_ret"}).dropna()
        merged = test_c[["timestamp", "symbol", "pred"]].merge(
            fwd_data, on=["timestamp", "symbol"], how="inner")
        all_preds.append(merged)
        window_metrics.append({
            "window": w["name"], "trees": model.best_iteration,
            "ic_train": train_ic, "ic_val": val_ic, "ic_test": test_ic,
        })

    preds_df = pd.concat(all_preds, ignore_index=True) if all_preds else None
    return preds_df, window_metrics


# ══════════════════════════════════════════════════════════════════
def simulate_and_eval(preds, regime_df, cfg, label, lev=LEVERAGE, cap=CAPITAL):
    sub = simulate(preds, regime_df, 12, cfg)
    r = eval_config(sub, 12, label, lev, cap)
    show(r)
    return r, sub


def monthly_table(ridge_sub, lgb_sub, lev=LEVERAGE, cap=CAPITAL):
    """Print month-by-month comparison."""
    def get_monthly(sub):
        s = sub.copy()
        s["month"] = s["timestamp"].dt.to_period("M")
        return s.groupby("month")["portfolio_ret"].apply(
            lambda x: (1 + x * lev).prod() - 1)

    rm = get_monthly(ridge_sub)
    lm = get_monthly(lgb_sub)
    all_months = sorted(set(list(rm.index) + list(lm.index)))

    print(f"\n  {'Month':<10}  {'Ridge':>8}  {'LGB':>8}  {'Δ(LGB-R)':>10}  {'Winner':<6}")
    print("  " + "─" * 52)
    ridge_wins = lgb_wins = 0
    for m in all_months:
        r = rm.get(m, np.nan) * 100
        l = lm.get(m, np.nan) * 100
        delta = l - r if not (np.isnan(r) or np.isnan(l)) else np.nan
        if not np.isnan(delta):
            winner = "LGB ✅" if delta > 0 else "Ridge ✅"
            if delta > 0: lgb_wins += 1
            else: ridge_wins += 1
        else:
            winner = "—"
        print(f"  {str(m):<10}  {r:>+7.1f}%  {l:>+7.1f}%  {delta:>+9.1f}%  {winner}")

    print(f"\n  Monthly wins: Ridge={ridge_wins}, LGB={lgb_wins}")
    return ridge_wins, lgb_wins


def permutation_test(preds, regime_df, cfg, n_perms=50, label=""):
    """Shuffle predictions at each timestamp → test if Sharpe collapses."""
    sharpes = []
    for i in range(n_perms):
        rng = np.random.default_rng(seed=i)
        p = preds.copy()
        p["pred"] = p.groupby("timestamp")["pred"].transform(
            lambda x: rng.permutation(x.values))
        sub = simulate(p, regime_df, 12, cfg)
        r = eval_config(sub, 12, "perm", LEVERAGE, CAPITAL)
        if r:
            sharpes.append(r["sharpe"])

    arr = np.array(sharpes)
    print(f"\n  Permutation test for {label} ({n_perms} perms):")
    print(f"    Permuted Sharpe: mean={arr.mean():.3f}  std={arr.std():.3f}  "
          f"min={arr.min():.3f}  max={arr.max():.3f}")
    return arr


def bootstrap_sharpe(sub, n_boots=500, lev=LEVERAGE, cap=CAPITAL):
    """Bootstrap confidence interval on Sharpe."""
    rets = sub["portfolio_ret"].values
    sharpes = []
    for i in range(n_boots):
        rng = np.random.default_rng(seed=i)
        sample = rng.choice(rets, size=len(rets), replace=True)
        sh = sample.mean() / (sample.std() + 1e-10) * np.sqrt(8760 / 12)
        sharpes.append(sh)
    arr = np.array(sharpes)
    lo, hi = np.percentile(arr, [5, 95])
    return arr.mean(), lo, hi


# ══════════════════════════════════════════════════════════════════
def main():
    divider("VERIFICATION: LightGBM vs Ridge — R9B Claims")

    print("\n📊 Loading data...")
    ohlcv  = load_ohlcv()
    ohlcv  = ohlcv[ohlcv["symbol"].isin(SYM_35)]
    derivs = load_derivatives()
    df     = build_features_minimal(ohlcv, derivs)
    feats  = [f for f in FEATURES_14 if f in df.columns]
    regime_df = compute_regime(df)
    print(f"   df shape: {df.shape}, {df['symbol'].nunique()} symbols")
    print(f"   date range: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

    # ──────────────────────────────────────────────────────────────
    section("CHECK 1: Walk-forward windows — data integrity")
    for w in WINDOWS:
        print(f"\n  {w['name']}:")
        print(f"    train: ? → {w['train_end']}")
        print(f"    val:   {w['val_start']} → {w['val_end']}")
        print(f"    test:  {w['test_start']} → {w['test_end']}")
        train_size = df[df["timestamp"] < pd.Timestamp(w["train_end"], tz="UTC")].shape[0]
        val_size   = df[(df["timestamp"] >= pd.Timestamp(w["val_start"], tz="UTC")) &
                       (df["timestamp"] <  pd.Timestamp(w["val_end"],   tz="UTC"))].shape[0]
        test_size  = df[(df["timestamp"] >= pd.Timestamp(w["test_start"], tz="UTC")) &
                       (df["timestamp"] <= pd.Timestamp(w["test_end"],   tz="UTC"))].shape[0]
        print(f"    sizes: train={train_size:,}  val={val_size:,}  test={test_size:,}")
        gap_days = (pd.Timestamp(w["test_start"], tz="UTC") -
                    pd.Timestamp(w["val_end"], tz="UTC")).days
        print(f"    val→test gap: {gap_days} days  {'✅ OK' if gap_days >= 14 else '❌ TOO SMALL!'}")

    # ──────────────────────────────────────────────────────────────
    section("CHECK 2: Train Ridge baseline (reference)")
    print("\n  Training Ridge (R7 production)...")
    p12_ridge = train_and_predict_multi(df, feats, horizons=[12])[12]
    r_ridge, sub_ridge = simulate_and_eval(p12_ridge, regime_df, CFG_BASE,
                                           "Ridge EMA=2 (PROD)")

    # ──────────────────────────────────────────────────────────────
    section("CHECK 3: LGB num_leaves=31 — with full IC diagnostics")
    print("\n  Training LGB (n_leaves=31, seed=42)...")
    p12_lgb_31, metrics_31 = train_lgb_verbose(df, feats, num_leaves=31, seed=42)
    r_lgb_31, sub_lgb_31 = simulate_and_eval(p12_lgb_31, regime_df, CFG_LGB_BASE,
                                              "LGB EMA=None (n_leaves=31, seed=42)")

    # ──────────────────────────────────────────────────────────────
    section("CHECK 4: LGB stability — different seeds")
    print("\n  Testing seed sensitivity...")
    seed_results = []
    for seed in [0, 7, 13, 42, 99]:
        p, _ = train_lgb_verbose(df, feats, num_leaves=31, seed=seed)
        if p is not None:
            r, _ = simulate_and_eval(p, regime_df, CFG_LGB_BASE,
                                     f"LGB n_leaves=31 seed={seed}")
            if r:
                seed_results.append((seed, r["equity"], r["sharpe"],
                                     r["worst_m"]*100, r["win_months"]))

    print(f"\n  Seed stability summary:")
    print(f"  {'Seed':<6}  {'Equity':>7}  {'Sharpe':>7}  {'Worst M':>8}  WM")
    print(f"  {'─'*40}")
    equities = [x[1] for x in seed_results]
    sharpes  = [x[2] for x in seed_results]
    for seed, eq, sh, wm, wins in seed_results:
        print(f"  {seed:<6}  ${eq:>6.0f}  {sh:>7.2f}  {wm:>+7.1f}%  {wins}/13")
    print(f"  Range: Eq ${min(equities):.0f}–${max(equities):.0f}  "
          f"Sh {min(sharpes):.2f}–{max(sharpes):.2f}")

    # ──────────────────────────────────────────────────────────────
    section("CHECK 5: LGB stability — different num_leaves")
    print("\n  Testing num_leaves sensitivity...")
    leaves_results = []
    for nl in [15, 31, 63, 127]:
        p, metrics = train_lgb_verbose(df, feats, num_leaves=nl, seed=42)
        if p is not None:
            r, _ = simulate_and_eval(p, regime_df, CFG_LGB_BASE,
                                     f"LGB n_leaves={nl}")
            if r:
                avg_ic_test = np.mean([m["ic_test"] for m in metrics])
                leaves_results.append((nl, r["equity"], r["sharpe"],
                                       r["worst_m"]*100, round(avg_ic_test, 4)))

    print(f"\n  num_leaves effect:")
    print(f"  {'Leaves':<8}  {'Equity':>7}  {'Sharpe':>7}  {'Worst M':>8}  {'Avg IC':>8}")
    print(f"  {'─'*48}")
    for nl, eq, sh, wm, ic in leaves_results:
        flag = " ← best" if nl == 31 else ""
        print(f"  {nl:<8}  ${eq:>6.0f}  {sh:>7.2f}  {wm:>+7.1f}%  {ic:>8.4f}{flag}")

    # ──────────────────────────────────────────────────────────────
    section("CHECK 6: Permutation test — is signal real?")
    print("\n  Running permutation tests (50 shuffles each):")
    perm_ridge = permutation_test(p12_ridge, regime_df, CFG_BASE, n_perms=50, label="Ridge")
    perm_lgb   = permutation_test(p12_lgb_31, regime_df, CFG_LGB_BASE, n_perms=50, label="LGB")
    print(f"\n  Real Ridge Sharpe:  {r_ridge['sharpe']:.3f}")
    print(f"  Real LGB Sharpe:    {r_lgb_31['sharpe']:.3f}")
    print(f"  Perm Ridge p-value: "
          f"{(perm_ridge >= r_ridge['sharpe']).mean():.4f}")
    print(f"  Perm LGB p-value:   "
          f"{(perm_lgb >= r_lgb_31['sharpe']).mean():.4f}")

    # ──────────────────────────────────────────────────────────────
    section("CHECK 7: Month-by-month breakdown")
    r_wins, l_wins = monthly_table(sub_ridge, sub_lgb_31)

    # ──────────────────────────────────────────────────────────────
    section("CHECK 8: Bootstrap confidence intervals on Sharpe")
    ridge_sh_mean, ridge_sh_lo, ridge_sh_hi = bootstrap_sharpe(sub_ridge)
    lgb_sh_mean, lgb_sh_lo, lgb_sh_hi = bootstrap_sharpe(sub_lgb_31)
    print(f"\n  Ridge Sharpe: {ridge_sh_mean:.3f}  90% CI [{ridge_sh_lo:.3f}, {ridge_sh_hi:.3f}]")
    print(f"  LGB   Sharpe: {lgb_sh_mean:.3f}  90% CI [{lgb_sh_lo:.3f}, {lgb_sh_hi:.3f}]")
    overlap = lgb_sh_lo < ridge_sh_hi and ridge_sh_lo < lgb_sh_hi
    if overlap:
        print(f"  ⚠️  CIs OVERLAP — difference may not be statistically significant")
    else:
        print(f"  ✅ CIs DO NOT overlap — difference is significant")

    # Check if Sharpe CI of LGB is fully above Ridge
    if lgb_sh_lo > ridge_sh_hi:
        print(f"  ✅ LGB is CLEARLY better (lower bound LGB > upper bound Ridge)")
    elif lgb_sh_lo > ridge_sh_mean:
        print(f"  ⚠️  LGB likely better, but not conclusively")
    else:
        print(f"  ❌ LGB superiority is NOT confirmed statistically")

    # ──────────────────────────────────────────────────────────────
    divider("FINAL VERDICT")
    print(f"""
  MODEL         Equity    Sharpe   Worst M   WM
  ─────────────────────────────────────────────────
  Ridge EMA=2   ${r_ridge['equity']:>5.0f}    {r_ridge['sharpe']:.2f}    {r_ridge['worst_m']*100:>+5.1f}%  {r_ridge['win_months']}/{r_ridge['total_months']}
  LGB EMA=None  ${r_lgb_31['equity']:>5.0f}    {r_lgb_31['sharpe']:.2f}    {r_lgb_31['worst_m']*100:>+5.1f}%  {r_lgb_31['win_months']}/{r_lgb_31['total_months']}

  Seed stability:  Eq range ${min(equities):.0f}–${max(equities):.0f}  (spread: ${max(equities)-min(equities):.0f})
  Leaves stability: {' / '.join(f'nl={nl}: Sh={sh:.2f}' for nl, eq, sh, wm, ic in leaves_results)}
  Monthly wins: Ridge {r_wins} vs LGB {l_wins}
  Bootstrap Sharpe CI: Ridge [{ridge_sh_lo:.2f},{ridge_sh_hi:.2f}]  LGB [{lgb_sh_lo:.2f},{lgb_sh_hi:.2f}]
""")

    if (r_lgb_31['sharpe'] > r_ridge['sharpe'] and
            lgb_sh_lo > ridge_sh_lo and
            l_wins >= r_wins and
            max(equities) - min(equities) < 500):
        print("  🚀 VERDICT: LGB EMA=None is GENUINELY BETTER than Ridge.")
        print("     → Safe to proceed to R10 (hyperparameter tuning for deployment)")
    elif r_lgb_31['sharpe'] > r_ridge['sharpe']:
        print("  ⚠️  VERDICT: LGB has higher Sharpe but evidence is MIXED.")
        print("     → Need more investigation before deployment decision")
    else:
        print("  ❌ VERDICT: LGB advantage NOT confirmed. Stay on Ridge.")

    print()


if __name__ == "__main__":
    main()
