"""Replacement features for the 6 "dead" market-level slots.

Baseline (F10_F20, 6 feats in CS-rank): Net Sharpe 3.777
F10_F21 (6 feats raw, FIX2 ON):         Net Sharpe 2.135 (-1.642)

6 dead slots: pct_coins_up_12h, pct_coins_up_1h, hour_sin, hour_cos, dow_sin, dow_cos

Goal: replace each pair with per-symbol variant that has real cross-sectional variance,
so CS-rank produces meaningful signal instead of zeros.

Each function returns List[str] of column names ADDED to df.
All features are PER-SYMBOL (vary across symbols at same timestamp) → CS-rank safe.

Usage:
    import _replacement_features as rf
    added = rf.EXP_A_seasonal_x_symbol(df)
    # then add `added` to CHAMPION_FEAT_31 and remove the 6 dead ones
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT A: Seasonal × per-symbol interactions
# Hypothesis: hour/dow carry info only when multiplied by per-symbol signals
# ═══════════════════════════════════════════════════════════════

def EXP_A_seasonal_x_symbol(df: pd.DataFrame) -> list[str]:
    """6 replacements: seasonal harmonics × per-symbol momentum/volume.

    Replaces hour_sin/cos, dow_sin/cos, pct_coins_up_1h/12h with:
      - hour_sin × premium_zscore_12h
      - hour_cos × rel_volume_cs
      - dow_sin  × ret_12h
      - dow_cos  × oi_chg_12h (or 0 if missing)
      - hour_sin × ret_1h
      - dow_sin  × rel_volume_cs
    Each varies per symbol → survives CS-rank.
    """
    ts = df["timestamp"]
    hs = np.sin(2 * np.pi * ts.dt.hour / 24)
    hc = np.cos(2 * np.pi * ts.dt.hour / 24)
    ds = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
    dc = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
    added: list[str] = []
    mapping = {
        "seas_hsin_x_premz": hs * df.get("premium_zscore_12h", 0.0),
        "seas_hcos_x_relvol": hc * df.get("rel_volume_cs", 0.0),
        "seas_dsin_x_ret12": ds * df.get("ret_12h", 0.0),
        "seas_dcos_x_oichg": dc * df.get("oi_chg_12h", 0.0),
        "seas_hsin_x_ret1": hs * df.get("ret_1h", 0.0),
        "seas_dsin_x_relvol": ds * df.get("rel_volume_cs", 0.0),
    }
    for name, series in mapping.items():
        df[name] = pd.Series(series, index=df.index).replace([np.inf, -np.inf], np.nan)
        added.append(name)
    return added


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT B: Per-symbol hour-of-day volatility rank
# Hypothesis: each symbol has its own "high-vol hours" — rank among peers
# ═══════════════════════════════════════════════════════════════

def EXP_B_hod_vol_rank(df: pd.DataFrame) -> list[str]:
    """Per-symbol historical hour-of-day relative volume/volatility.

    For each (symbol, hour-of-day) compute expanding-mean of |ret_1h|
    then z-score across symbols at same timestamp (CS-rank-safe).
    """
    df = df.sort_values(["symbol", "timestamp"])
    added: list[str] = []
    ts = df["timestamp"]
    hod = ts.dt.hour
    dow = ts.dt.dayofweek

    if "ret_1h" in df.columns:
        abs_ret = df["ret_1h"].abs()
        # Expanding mean |ret_1h| per (symbol, hour)
        key_h = df["symbol"].astype(str) + "_" + hod.astype(str)
        df["_sym_hod_vol"] = (
            abs_ret.groupby(key_h)
                   .transform(lambda s: s.shift(1).expanding(min_periods=30).mean())
        )
        df["hod_vol_z"] = df["_sym_hod_vol"]
        added.append("hod_vol_z")

        key_d = df["symbol"].astype(str) + "_" + dow.astype(str)
        df["_sym_dow_vol"] = (
            abs_ret.groupby(key_d)
                   .transform(lambda s: s.shift(1).expanding(min_periods=50).mean())
        )
        df["dow_vol_z"] = df["_sym_dow_vol"]
        added.append("dow_vol_z")

    # Per-symbol hour-of-day mean return (expanding, lagged)
    if "ret_1h" in df.columns:
        key_h = df["symbol"].astype(str) + "_" + hod.astype(str)
        df["hod_ret_mean"] = (
            df["ret_1h"].groupby(key_h)
                       .transform(lambda s: s.shift(1).expanding(min_periods=30).mean())
        )
        added.append("hod_ret_mean")

        key_d = df["symbol"].astype(str) + "_" + dow.astype(str)
        df["dow_ret_mean"] = (
            df["ret_1h"].groupby(key_d)
                       .transform(lambda s: s.shift(1).expanding(min_periods=50).mean())
        )
        added.append("dow_ret_mean")

    # Drop helpers
    for c in ["_sym_hod_vol", "_sym_dow_vol"]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # Pad to 6 with per-symbol breadth analogues
    added += _breadth_per_symbol(df)
    return added


def _breadth_per_symbol(df: pd.DataFrame) -> list[str]:
    """2 features: symbol's own recent 'up-ratio' (analog to pct_coins_up)."""
    added: list[str] = []
    if "ret_1h" in df.columns:
        up1 = (df["ret_1h"] > 0).astype(float)
        df["up_rate_12h_sym"] = (
            up1.groupby(df["symbol"]).transform(lambda s: s.rolling(12, min_periods=6).mean())
        )
        added.append("up_rate_12h_sym")
        df["up_rate_48h_sym"] = (
            up1.groupby(df["symbol"]).transform(lambda s: s.rolling(48, min_periods=24).mean())
        )
        added.append("up_rate_48h_sym")
    return added


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT C: Symbol breadth relative-to-market
# Hypothesis: replace pct_coins_up (market constant) with per-symbol contribution
# ═══════════════════════════════════════════════════════════════

def EXP_C_relative_breadth(df: pd.DataFrame) -> list[str]:
    """Per-symbol 'contribution to breadth' — rank-style but direct."""
    added: list[str] = []
    ts = df["timestamp"]
    if "ret_1h" in df.columns:
        # Market pct up
        mkt_up_1 = df.groupby("timestamp")["ret_1h"].transform(lambda s: (s > 0).mean())
        # Symbol deviation: did THIS symbol outperform market up-rate?
        df["breadth_dev_1h"] = (df["ret_1h"] > 0).astype(float) - mkt_up_1
        added.append("breadth_dev_1h")
    if "ret_12h" in df.columns:
        mkt_up_12 = df.groupby("timestamp")["ret_12h"].transform(lambda s: (s > 0).mean())
        df["breadth_dev_12h"] = (df["ret_12h"] > 0).astype(float) - mkt_up_12
        added.append("breadth_dev_12h")
    # How many std from median symbol return
    if "ret_12h" in df.columns:
        mkt_med = df.groupby("timestamp")["ret_12h"].transform("median")
        mkt_mad = df.groupby("timestamp")["ret_12h"].transform(
            lambda s: (s - s.median()).abs().median() + 1e-10
        )
        df["ret_12h_mad_z"] = (df["ret_12h"] - mkt_med) / mkt_mad
        added.append("ret_12h_mad_z")
    # Rolling per-symbol rank in top-K
    if "ret_12h" in df.columns:
        df["ret12_cs_rank_ma24"] = (
            df.groupby("timestamp")["ret_12h"].rank(pct=True).sub(0.5)
              .groupby(df["symbol"]).transform(lambda s: s.rolling(24, min_periods=12).mean())
        )
        added.append("ret12_cs_rank_ma24")
    # Symbol's share of market absolute volume
    if "volume" in df.columns:
        tot = df.groupby("timestamp")["volume"].transform("sum") + 1e-10
        df["vol_share"] = df["volume"] / tot
        added.append("vol_share")
        df["vol_share_chg_12h"] = (
            df.groupby("symbol")["vol_share"]
              .transform(lambda s: s - s.shift(12))
        )
        added.append("vol_share_chg_12h")
    return added


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT D: Session/regime indicators per symbol
# Hypothesis: Asian/EU/US session × symbol liquidity profile
# ═══════════════════════════════════════════════════════════════

def EXP_D_session_regime(df: pd.DataFrame) -> list[str]:
    """Per-symbol session × activity."""
    added: list[str] = []
    hour = df["timestamp"].dt.hour
    # Sessions (UTC): Asia 0-8, EU 7-16, US 13-22
    is_asia = ((hour >= 0) & (hour < 8)).astype(float)
    is_eu = ((hour >= 7) & (hour < 16)).astype(float)
    is_us = ((hour >= 13) & (hour < 22)).astype(float)
    if "rel_volume_cs" in df.columns:
        df["asia_x_relvol"] = is_asia * df["rel_volume_cs"]
        df["eu_x_relvol"] = is_eu * df["rel_volume_cs"]
        df["us_x_relvol"] = is_us * df["rel_volume_cs"]
        added += ["asia_x_relvol", "eu_x_relvol", "us_x_relvol"]
    if "ret_12h" in df.columns:
        df["weekend_x_ret12"] = (df["timestamp"].dt.dayofweek >= 5).astype(float) * df["ret_12h"]
        added.append("weekend_x_ret12")
    if "ret_1h" in df.columns:
        # 8-hour-block returns per symbol (captures session momentum)
        df["ret_session_8h"] = (
            df.groupby("symbol")["ret_1h"]
              .transform(lambda s: s.rolling(8, min_periods=4).sum())
        )
        added.append("ret_session_8h")
    # Per-symbol Monday-effect
    if "ret_12h" in df.columns:
        is_mon = (df["timestamp"].dt.dayofweek == 0).astype(float)
        df["monday_x_ret12"] = is_mon * df["ret_12h"]
        added.append("monday_x_ret12")
    return added


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT E: Market regime × symbol beta
# Hypothesis: pct_coins_up signal = market breadth regime — replace with per-symbol regime fit
# ═══════════════════════════════════════════════════════════════

def EXP_E_regime_x_beta(df: pd.DataFrame) -> list[str]:
    """Per-symbol regime markers."""
    added: list[str] = []
    if "ret_12h" in df.columns:
        # Market up-strength (not constant across symbols after we multiply)
        mkt_up = df.groupby("timestamp")["ret_12h"].transform(lambda s: (s > 0).mean())
        df["breadth_x_ret12"] = mkt_up * df["ret_12h"]
        added.append("breadth_x_ret12")
        df["breadth_x_ret12_sign"] = mkt_up * np.sign(df["ret_12h"].fillna(0.0))
        added.append("breadth_x_ret12_sign")
    if "btc_corr_168h" in df.columns and "ret_12h" in df.columns:
        # Symbols highly correlated to BTC behave differently during breadth regimes
        mkt_up = df.groupby("timestamp")["ret_12h"].transform(lambda s: (s > 0).mean())
        df["btc_corr_x_breadth"] = df["btc_corr_168h"] * mkt_up
        added.append("btc_corr_x_breadth")
    if "ret_dispersion_12h" in df.columns and "ret_12h" in df.columns:
        # Dispersion × own return (low-dispersion = trending, pays the leader)
        df["disp_x_ret12"] = df["ret_dispersion_12h"] * df["ret_12h"]
        added.append("disp_x_ret12")
    if "ret_1h" in df.columns:
        # Per-symbol momentum persistence in current breadth regime
        mkt_up = df.groupby("timestamp")["ret_1h"].transform(lambda s: (s > 0).mean())
        df["breadth_x_ret1_sym"] = (mkt_up - 0.5) * np.sign(df["ret_1h"].fillna(0.0))
        added.append("breadth_x_ret1_sym")
    if "ret_12h" in df.columns:
        # Own-vs-market sign alignment
        mkt_med = df.groupby("timestamp")["ret_12h"].transform("median")
        df["aligned_with_mkt"] = (np.sign(df["ret_12h"] - mkt_med)).astype(float)
        added.append("aligned_with_mkt")
    return added


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT F: Remove the 6 dead slots entirely (control)
# Hypothesis: maybe CHAMPION_FEAT_25 (25 real feats) is just as good
# ═══════════════════════════════════════════════════════════════

def EXP_F_drop_only(df: pd.DataFrame) -> list[str]:
    """Control: drop the 6 dead slots, add nothing. Shows pure removal cost."""
    return []


# ═══════════════════════════════════════════════════════════════
# Experiment registry
# ═══════════════════════════════════════════════════════════════

DEAD_FEATS = [
    "pct_coins_up_12h", "pct_coins_up_1h",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]

EXPERIMENTS = {
    "A_seasonal_x_symbol": EXP_A_seasonal_x_symbol,
    "B_hod_vol_rank": EXP_B_hod_vol_rank,
    "C_relative_breadth": EXP_C_relative_breadth,
    "D_session_regime": EXP_D_session_regime,
    "E_regime_x_beta": EXP_E_regime_x_beta,
    "F_drop_only": EXP_F_drop_only,
}
