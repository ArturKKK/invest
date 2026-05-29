from __future__ import annotations

import gc
import sys
import time

sys.path.insert(0, ".")

from _preflight_check import check_versions

check_versions()

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

import _research_r68_continuous_wf as r68
from _research_r22_models import SEEDS
from _research_r35_new_features import MARKET_LEVEL_FEATURES
from _research_r68_continuous_wf import (
    CHAMPION_FEAT_31,
    EARLY_STOP,
    LGB_PARAMS,
    N_ROUNDS,
    XGB_PARAMS,
)

OOS_START = pd.Timestamp("2026-03-18", tz="UTC")
OOS_END_EXCL = pd.Timestamp("2026-04-26", tz="UTC")

WINDOWS = [
    (
        "R132",
        "cache/r132_oos_preds.parquet",
        dict(
            name="W4_OOS",
            train_end="2026-01-01",
            val_start="2026-01-01",
            val_end="2026-03-15",
            test_start="2026-03-18",
            test_end="2026-04-25",
        ),
    ),
    (
        "R134",
        "cache/r134_fresh_preds.parquet",
        dict(
            name="W4_FRESH",
            train_end="2026-03-15",
            val_start="2026-03-15",
            val_end="2026-03-17",
            test_start="2026-03-18",
            test_end="2026-04-25",
        ),
    ),
]


def rank_inplace(df: pd.DataFrame, feats: list[str]) -> None:
    print(f"RANK_ALL_START features={len(feats)} rows={len(df):,}", flush=True)
    for idx, feat in enumerate(feats, 1):
        started = time.time()
        df[feat] = df.groupby("timestamp")[feat].rank(pct=True) - 0.5
        print(f"RANK_ALL {idx:02d}/{len(feats):02d} {feat} {time.time() - started:.1f}s", flush=True)
    print("RANK_ALL_DONE", flush=True)


def clean_sets(train_: pd.DataFrame, val_: pd.DataFrame, test_: pd.DataFrame, avail: list[str]):
    for data in (train_, val_, test_):
        data["target_binary"] = (data["fwd_ret_12h"] > 0).astype(int)
    for col in avail:
        for data in (train_, val_, test_):
            if data[col].isna().any():
                data[col] = data[col].fillna(0)
    tr = train_[avail + ["target_binary"]].dropna()
    va = val_[avail + ["target_binary"]].dropna()
    te = test_[avail + ["target_binary", "timestamp", "symbol"]].dropna()
    fwd = test_[["timestamp", "symbol", "fwd_ret_12h"]].rename(columns={"fwd_ret_12h": "fwd_ret"}).dropna()
    for data in (tr, va, te):
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
    return tr.dropna(), va.dropna(), te.dropna(), fwd


def train_one(df: pd.DataFrame, avail: list[str], label: str, out_path: str, window: dict) -> pd.DataFrame:
    tz = df["timestamp"].dt.tz
    tr_end = pd.Timestamp(window["train_end"], tz=tz)
    va_start = pd.Timestamp(window["val_start"], tz=tz)
    va_end = pd.Timestamp(window["val_end"], tz=tz)
    te_start = pd.Timestamp(window["test_start"], tz=tz)
    te_end = pd.Timestamp(window["test_end"], tz=tz)

    started = time.time()
    train_ = df[df["timestamp"] < tr_end].copy()
    val_ = df[(df["timestamp"] >= va_start) & (df["timestamp"] < va_end)].copy()
    test_ = df[(df["timestamp"] >= te_start) & (df["timestamp"] <= te_end)].copy()
    print(f"{label}_SPLIT train_raw={len(train_):,} val_raw={len(val_):,} test_raw={len(test_):,}", flush=True)
    if len(train_) < 5000 or len(test_) < 200:
        raise SystemExit(f"STOP {label}: insufficient rows")

    tr, va, te, fwd = clean_sets(train_, val_, test_, avail)
    del train_, val_, test_
    gc.collect()
    if len(te) == 0:
        raise SystemExit(f"STOP {label}: empty test after clean")
    print(f"{label}_CLEAN train={len(tr):,} val={len(va):,} test={len(te):,}", flush=True)

    all_lgb = []
    all_xgb = []
    for seed in SEEDS:
        seed_started = time.time()
        p_lgb = {**LGB_PARAMS, "seed": seed}
        p_xgb = {**XGB_PARAMS, "seed": seed}

        dt = lgb.Dataset(tr[avail], label=tr["target_binary"])
        dv = lgb.Dataset(va[avail], label=va["target_binary"])
        model_lgb = lgb.train(
            p_lgb,
            dt,
            num_boost_round=N_ROUNDS,
            valid_sets=[dv],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(-1)],
        )
        rec = te[["timestamp", "symbol"]].copy()
        rec["pred_lgb"] = model_lgb.predict(te[avail])
        rec = rec.merge(fwd, on=["timestamp", "symbol"], how="inner")
        rec["window"] = window["name"]
        rec["seed"] = seed
        all_lgb.append(rec)

        dt_x = xgb.DMatrix(tr[avail], label=tr["target_binary"])
        dv_x = xgb.DMatrix(va[avail], label=va["target_binary"])
        model_xgb = xgb.train(
            p_xgb,
            dt_x,
            num_boost_round=N_ROUNDS,
            evals=[(dv_x, "val")],
            early_stopping_rounds=EARLY_STOP,
            verbose_eval=False,
        )
        rec2 = te[["timestamp", "symbol"]].copy()
        rec2["pred_xgb"] = model_xgb.predict(xgb.DMatrix(te[avail]))
        rec2 = rec2.merge(fwd, on=["timestamp", "symbol"], how="inner")
        rec2["window"] = window["name"]
        rec2["seed"] = seed
        all_xgb.append(rec2)
        print(f"{label}_SEED_DONE seed={seed} elapsed={time.time() - seed_started:.1f}s", flush=True)
        gc.collect()

    lgb_df = pd.concat(all_lgb)
    xgb_df = pd.concat(all_xgb)
    lgb_avg = lgb_df.groupby(["timestamp", "symbol"]).agg(
        pred_lgb=("pred_lgb", "mean"), fwd_ret=("fwd_ret", "first"), window=("window", "first")
    ).reset_index()
    xgb_avg = xgb_df.groupby(["timestamp", "symbol"]).agg(pred_xgb=("pred_xgb", "mean")).reset_index()
    merged = lgb_avg.merge(xgb_avg, on=["timestamp", "symbol"], how="inner")
    merged["raw_prob"] = 0.5 * merged["pred_lgb"] + 0.5 * merged["pred_xgb"]
    merged["rank_lgb"] = merged.groupby("timestamp")["pred_lgb"].rank(pct=True) - 0.5
    merged["rank_xgb"] = merged.groupby("timestamp")["pred_xgb"].rank(pct=True) - 0.5
    merged["pred"] = 0.5 * merged["rank_lgb"] + 0.5 * merged["rank_xgb"]
    preds = merged[["timestamp", "symbol", "pred", "raw_prob", "fwd_ret", "window"]]
    preds.to_parquet(out_path, index=False)
    print(
        f"{label}_DONE rows={len(preds):,} ts={preds.timestamp.nunique()} syms={preds.symbol.nunique()} "
        f"range={preds.timestamp.min()}->{preds.timestamp.max()} elapsed={time.time() - started:.1f}s out={out_path}",
        flush=True,
    )
    return preds


def metric_row(path: str, label: str, regime: pd.DataFrame) -> dict:
    import _r130_validate_r129 as r130
    import _r131_prod_sim_validate as r131

    preds = pd.read_parquet(path)
    port = r131.simulate_prod(preds, regime, n_long=4, n_short=2)
    port = port[(port["timestamp"] >= OOS_START) & (port["timestamp"] < OOS_END_EXCL)]
    returns = port["net_ret"].values
    return dict(
        label=label,
        sharpe=r130.sharpe(returns),
        sortino=r130.sortino(returns),
        ret=float(np.sum(returns)),
        maxDD=r130.max_drawdown(returns),
        cvar5=r131.cvar_5pct(returns),
        n=len(port),
        n_act=int((~port["risk_off"]).sum()),
    )


def print_table(regime: pd.DataFrame) -> None:
    rows = [
        metric_row("cache/r133_r128style_preds.parquet", "R128_2025-07-01", regime),
        metric_row("cache/r132_oos_preds.parquet", "R132_2026-01-01", regime),
        metric_row("cache/r134_fresh_preds.parquet", "R134_2026-03-15", regime),
    ]
    print("FINAL_TABLE")
    print(f"{'Model':<16} {'Sharpe':>8} {'Sortino':>9} {'Ret%':>8} {'maxDD':>8} {'CVaR5':>9} {'n':>4} {'act':>4}")
    for row in rows:
        print(
            f"{row['label']:<16} {row['sharpe']:+8.3f} {row['sortino']:+9.3f} "
            f"{row['ret']*100:+7.2f}% {row['maxDD']*100:+7.2f}% {row['cvar5']*1e4:+8.1f}bp "
            f"{row['n']:4d} {row['n_act']:4d}"
        )
    base = rows[0]["sharpe"]
    for row in rows[1:]:
        print(f"DELTA_VS_R128 {row['label']} {row['sharpe'] - base:+.3f}")
    if all(row["sharpe"] <= 0 or row["ret"] <= 0 for row in rows):
        print("ABSOLUTE_GUARD all variants have negative OOS Sharpe/return; deltas are not a deploy signal")


def main() -> None:
    started = time.time()
    df, regime = r68.load_data()
    regime.to_parquet("cache/r132_regime_oos.parquet")
    feats = [feat for feat in CHAMPION_FEAT_31 if feat in df.columns]
    no_rank = [feat for feat in feats if feat in MARKET_LEVEL_FEATURES]
    rank_feats = [feat for feat in feats if feat not in set(no_rank)]
    oos = df[(df.timestamp >= OOS_START) & (df.timestamp < OOS_END_EXCL)]
    print(
        f"DATA_OK rows={len(df):,} oos_rows={len(oos):,} oos_ts={oos.timestamp.nunique()} "
        f"oos_syms={oos.symbol.nunique()} feats={len(feats)} no_rank={len(no_rank)} rank={len(rank_feats)}",
        flush=True,
    )
    assert len(feats) == 31 and len(oos) > 30000
    rank_inplace(df, rank_feats)
    for label, out_path, window in WINDOWS:
        train_one(df, feats, label, out_path, window)
    print_table(regime)
    print(f"TOTAL_ELAPSED {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()