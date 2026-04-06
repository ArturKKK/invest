#!/usr/bin/env python3
"""Quick audit: which features are untested."""
FEATURES_23 = [
    'ret_12h','ret_24h','ret_48h','residual_12h','residual_24h',
    'mom_z_24h','oi_chg_12h','oi_chg_24h','oi_zscore',
    'taker_cvd_12h','taker_cvd_24h','ls_divergence',
    'atr_14','rvol_12h','gk_vol_24h','rvol_24h','iv_rv_spread',
    'pct_coins_up_12h','pct_coins_up_1h','hour_sin','hour_cos','dow_sin','dow_cos',
]
R28C_TESTED = [
    'top_ls_ratio_zscore','global_ls_ratio_zscore','premium_zscore','taker_zscore','oi_ret_diverge',
    'adx','rsi_14','bb_pband_20','mfi_14','ret_skew_24h','ret_kurt_24h',
    'ret_168h','ret_1h','ret_4h','dist_from_high_24h',
    'vol_of_vol','rvol_168h','vol_ratio_24h',
    'premium_zscore_12h','oi_velocity','taker_imb_z','obv_ma_ratio_24','vwap_dev_24h',
]
TOXIC = [
    'funding_rate_binance','cum_funding_24h','cum_funding_72h','cum_funding_168h','funding_zscore',
    'fng_value','fng_zscore','vix_close','vix_zscore','dxy_close','dxy_ret_7d',
    'btc_dvol','dvol_zscore','btc_close',
    'btc_ret','btc_ret_1h','btc_ret_4h','btc_ret_12h','btc_ret_24h',
]
NOT_FEATURES = ['oi_value_usd','fwd_ret_1h','premium_index','coin_ret']

ALL_85 = [
    'adx','atr_14','bb_pband_20','btc_beta_168h','btc_close','btc_dvol','btc_outperform',
    'btc_ret','btc_ret_12h','btc_ret_1h','btc_ret_24h','btc_ret_4h','coin_ret',
    'cum_funding_168h','cum_funding_24h','cum_funding_72h','dist_from_high_24h',
    'dow_cos','dow_sin','dvol_zscore','dxy_close','dxy_ret_7d','fng_value','fng_zscore',
    'funding_rate_binance','funding_x_mom_12h','funding_x_mom_24h','funding_zscore',
    'fwd_ret_1h','gk_vol_24h','global_ls_ratio','global_ls_ratio_zscore','hour_cos',
    'hour_sin','iv_rv_spread','ls_divergence','mfi_14','mom_z_12h','mom_z_24h',
    'obv_ma_ratio_24','oi_chg_12h','oi_chg_1h','oi_chg_24h','oi_chg_4h','oi_ret_diverge',
    'oi_value_usd','oi_velocity','oi_zscore','pct_coins_up_12h','pct_coins_up_1h',
    'premium_index','premium_zscore','premium_zscore_12h','range_24h','residual_12h',
    'residual_24h','ret_12h','ret_168h','ret_1h','ret_24h','ret_48h','ret_4h',
    'ret_kurt_24h','ret_skew_24h','reversal_12v48','rsi_14','rvol_12h','rvol_168h',
    'rvol_24h','taker_buy_sell_ratio','taker_cvd_12h','taker_cvd_24h','taker_cvd_4h',
    'taker_imb_z','taker_imbalance','taker_zscore','top_ls_ratio','top_ls_ratio_zscore',
    'vix_close','vix_zscore','vol_crush','vol_of_vol','vol_ratio_12h','vol_ratio_24h',
    'vwap_dev_24h',
]

used = set(FEATURES_23) | set(R28C_TESTED) | set(TOXIC) | set(NOT_FEATURES)
new = [f for f in ALL_85 if f not in used]
print(f"Total: {len(ALL_85)}, Used/tested/toxic: {len(used)}, NEW: {len(new)}")
for f in sorted(new):
    print(f"  {f}")
