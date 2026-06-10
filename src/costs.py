"""Unified cost models for backtest/research simulations.

Single source of truth — import from here, do NOT copy-paste _cost_for_sym.
Ported verbatim from _research_r121_realistic_costs.py (R121 audit, validated).

History / why this exists:
- r68._cost_for_sym is the LENIENT pre-R121 model (Tier1 = -0.36bp net maker
  rebate, T2 2.5bp, T3 7bp). R121 measured it inflates Net Sharpe by ~0.44 vs
  realistic execution. Every R128-R135 overlay/OOS result inherited it.
- The validated production model is S6 "prod_blended": Tier1 2.4bp / Tier2
  5.5bp / Tier3 10bp effective per side, funding 1.2bp per 12h period.
- The canonical champion number (Net Sharpe 2.831 on 1013 periods) is an S6
  number. Any overlay delta must be measured under S6 to be comparable.

D6 orderbook calibration (2026-06-10, 65 days of Binance spot spreads):
actual spreads say Tier2 is understated for ~18/23 symbols (median half-spread
~4bp vs 2bp modeled), THETA/SNX/EGLD/RUNE真 cost 16-29bp/side (3-5x model),
INJ/LDO over-penalized in Tier3, DOGE/AAVE are Tier1-quality. Re-tiering is a
separate calibration step — S6 numbers below are kept as validated baseline.
"""

TIER1_SYMS = {"BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"}
TIER3_SYMS = {
    "SAND/USDT", "LDO/USDT", "INJ/USDT", "APT/USDT", "ARB/USDT",
    "GALA/USDT", "FTM/USDT", "MATIC/USDT",
}


def cost_lenient_r68(sym):
    """Pre-R121 lenient model (r68._cost_for_sym). KEPT ONLY FOR COMPARISON.

    Tier1 nets a -0.36bp maker REBATE — unrealistic. Do not use for decisions.
    """
    if sym in TIER1_SYMS:
        return 0.92 * (-0.0001) + 0.08 * 0.0007   # -0.36 bps
    elif sym in TIER3_SYMS:
        return 0.0005 + 0.0002                      # 7 bps
    else:
        return 0.75 * 0.0001 + 0.25 * 0.0007        # 2.5 bps


def cost_prod_blended(sym):
    """S6: validated prod execution mix. THE default for all research sims.

    Tier1 (BTC,ETH,SOL,BNB,XRP): maker-first, ~90% maker 2bp + 10% taker 6bp = 2.4bp
    Tier2 (mid-cap): aggressive limit, 50/50 maker-like/taker = 5.5bp
    Tier3 (small-cap): pure market = 10bp
    """
    if sym in TIER1_SYMS:
        return 0.90 * 0.0002 + 0.10 * 0.0006   # 2.4 bps blended
    elif sym in TIER3_SYMS:
        return 0.0005 + 0.0005                   # 10 bps (pure taker)
    else:
        return 0.50 * 0.0004 + 0.50 * 0.0007     # 5.5 bps blended


# ---------------------------------------------------------------------------
# D6 RE-TIERING (2026-06-11, POINT 6)
# ---------------------------------------------------------------------------
# Source: data_vps_d6/binance_orderbook_depth_features_JUN10.parquet
#   65 days (2026-04-05 .. 2026-06-10), 1563 hourly obs/symbol, all 35 SYM_35
#   symbols present. Per-symbol MEDIAN full spread_bps (Binance spot proxy):
#
#     THETA 48.66 | SNX 32.84 | EGLD 24.97 | RUNE 22.70   <- WIDE (new Tier0)
#     SAND 12.59 | MANA 11.20 | FIL 10.63 | APT 10.53 | AVAX 10.50 |
#     LINK 10.27 | ALGO 8.91 | ARB 8.61 | AXS 8.60 | OP 8.11 | DOT 7.99 |
#     FLOW 6.69 | NEAR 6.68 | XTZ 5.32 | ATOM 5.26 | CRV 4.42 | ADA 4.02 |
#     UNI 3.07 | CHZ 2.58                                  <- mid (Tier2, 5.5bp)
#     LDO 2.80 | INJ 2.58                                  <- were Tier3(10), now 6bp
#     LTC 1.82 | SOL 1.18 | AAVE 1.08 | DOGE 1.00 | XRP 0.73 |
#     BNB 0.16 | ETH 0.04 | BTC 0.00                       <- cheap (Tier1, 2.4bp)
#     MATIC, FTM: 0 obs on Binance spot (renamed POL / S) -> keep legacy.
#
# v2 re-tiering vs S6 cost_prod_blended (all PER-SIDE bps):
#   * Tier0_wide: median full spread > 15bp -> half_spread + 5bp taker cushion.
#       THETA 29.3 | SNX 21.4 | EGLD 17.5 | RUNE 16.4  (was 5.5; 3-5x understated).
#   * Tier1_major (cheap majors): median full spread < 2.5bp -> 2.4bp.
#       Adds DOGE, AAVE, LTC to the original BTC/ETH/SOL/BNB/XRP set.
#   * Tier2_mid: everything else -> 5.5bp (unchanged).
#       SAND/APT/ARB drop from Tier3(10) -> 5.5 (their real median spread is mid-tier).
#   * Tier2b_inj_ldo: INJ, LDO -> 6.0bp (over-penalized in Tier3; audit target ~6bp).
#   * No-data fallback (MATIC, FTM, GALA): keep S6 tier cost (cannot remeasure).
#
# CAVEAT: Binance-spot proxy; prod is OKX perp. Treat as RELATIVE guidance, not
# absolute calibration. NOT yet backtested — separate VM job.
# ---------------------------------------------------------------------------

# Cheap majors at Tier1-quality (median D6 full spread < 2.5bp): originals + DOGE/AAVE/LTC.
V2_TIER1_MAJOR = TIER1_SYMS | {"DOGE/USDT", "AAVE/USDT", "LTC/USDT"}

# Wide / illiquid (median D6 full spread > 15bp): proposed per-side = half_spread + 5bp.
V2_TIER0_WIDE = {
    "THETA/USDT": 0.00293,   # median full 48.66bp -> 29.3bp/side
    "SNX/USDT":   0.00214,   # 32.84 -> 21.4
    "EGLD/USDT":  0.00175,   # 24.97 -> 17.5
    "RUNE/USDT":  0.00164,   # 22.70 -> 16.4
}

# Tight-spread names mis-filed in S6 Tier3(10bp); audit target ~6bp.
V2_TIER2B_INJ_LDO = {"INJ/USDT", "LDO/USDT"}

# No D6 spread data on Binance spot (renamed/delisted) -> fall back to S6 tier cost.
V2_NO_DATA = {"MATIC/USDT", "FTM/USDT", "GALA/USDT"}


def cost_prod_blended_v2(sym):
    """D6-recalibrated re-tiering of S6 cost_prod_blended (per-side bps).

    Fixes the three errors the June-2026 D6 orderbook audit found in S6:
      1. WIDE underpriced: THETA/SNX/EGLD/RUNE sat in Tier2(5.5bp) but their
         true taker is 16-29bp/side (3-5x). -> Tier0_wide (half_spread+5bp).
      2. Cheap majors overpriced: DOGE/AAVE/LTC are Tier1-quality (median full
         spread ~1bp) but sat in Tier2(5.5bp). -> 2.4bp.
      3. INJ/LDO over-penalized: median full spread ~2.6/2.8bp yet S6 put them
         in Tier3(10bp). -> 6.0bp.
    SAND/APT/ARB also drop Tier3(10)->Tier2(5.5) (real median spread is mid-tier).
    MATIC/FTM/GALA have no D6 data -> keep S6 cost via cost_prod_blended fallback.

    Binance-spot proxy; prod is OKX perp. Relative guidance, not yet backtested.
    """
    if sym in V2_TIER0_WIDE:
        return V2_TIER0_WIDE[sym]
    if sym in V2_TIER1_MAJOR:
        return 0.90 * 0.0002 + 0.10 * 0.0006   # 2.4 bps blended (Tier1-quality)
    if sym in V2_TIER2B_INJ_LDO:
        return 0.0006                            # 6.0 bps (de-tiered from Tier3)
    if sym in V2_NO_DATA:
        return cost_prod_blended(sym)            # keep S6 tier (no D6 measurement)
    # everything else -> Tier2 mid-cap, unchanged from S6
    return 0.50 * 0.0004 + 0.50 * 0.0007         # 5.5 bps blended


# ---------------------------------------------------------------------------
# REF20 — S6 with 20% OKX referral fee cashback (2026-06-11, TRACK C)
# ---------------------------------------------------------------------------
# Cashback applies to EXCHANGE FEES only; spread/impact components unchanged.
# S6 fee/spread decomposition (per side):
#   T1: 90% maker(2bp fee) + 10% taker(5bp fee + 1bp spread)
#       -> ref20: 0.90*(0.8*2.0) + 0.10*(0.8*5.0 + 1.0)
#                = 0.90*1.6 + 0.10*5.0 = 1.44 + 0.50 = 1.94bp  (S6: 2.4bp)
#   T2: 50%(2bp fee + 2bp spread) + 50%(5bp fee + 2bp spread)
#       -> ref20: 0.50*(0.8*2.0 + 2.0) + 0.50*(0.8*5.0 + 2.0)
#                = 0.50*3.6 + 0.50*6.0 = 1.80 + 3.00 = 4.80bp  (S6: 5.5bp)
#   T3: 5bp fee + 5bp spread
#       -> ref20: 0.8*5.0 + 5.0 = 4.0 + 5.0 = 9.0bp            (S6: 10bp)
# Funding is NOT a fee -> unchanged (0.00012 per 12h in COST_MODELS).
# ---------------------------------------------------------------------------


def cost_prod_blended_ref20(sym):
    """S6 cost_prod_blended with 20% OKX referral cashback on FEE components.

    Identical tiers/mix to cost_prod_blended; fee parts x0.8, spread parts
    unchanged (cashback rebates exchange fees, not market impact):
      Tier1 2.4 -> 1.94bp | Tier2 5.5 -> 4.8bp | Tier3 10 -> 9.0bp per side.
    See decomposition block above. Funding unchanged.
    """
    if sym in TIER1_SYMS:
        return 0.90 * (0.8 * 0.0002) + 0.10 * (0.8 * 0.0005 + 0.0001)  # 1.94 bps
    elif sym in TIER3_SYMS:
        return 0.8 * 0.0005 + 0.0005                                    # 9.0 bps
    else:
        return (0.50 * (0.8 * 0.0002 + 0.0002)
                + 0.50 * (0.8 * 0.0005 + 0.0002))                       # 4.8 bps


def cost_okx_taker(sym):
    """OKX Futures Lv1, 100% market orders."""
    if sym in TIER1_SYMS:
        return 0.0005 + 0.0001   # 6 bps
    elif sym in TIER3_SYMS:
        return 0.0005 + 0.0005   # 10 bps
    else:
        return 0.0005 + 0.0002   # 7 bps


def cost_pessimistic(sym):
    """Worst case: taker + wide spreads + stress slippage."""
    if sym in TIER1_SYMS:
        return 0.0005 + 0.0002 + 0.0001   # 8 bps
    elif sym in TIER3_SYMS:
        return 0.0005 + 0.0010 + 0.0003   # 18 bps
    else:
        return 0.0005 + 0.0004 + 0.0002   # 11 bps


# (cost_fn, funding per 12h period)
COST_MODELS = {
    "lenient_r68":  (cost_lenient_r68,  0.00008),   # 0.8bp/12h — legacy, comparison only
    "okx_taker":    (cost_okx_taker,    0.00012),
    "pessimistic":  (cost_pessimistic,  0.00015),
    "prod_blended": (cost_prod_blended, 0.00012),   # S6 — THE default
    "prod_blended_v2": (cost_prod_blended_v2, 0.00012),  # D6-recalibrated re-tier (POINT 6, not yet backtested)
    "prod_blended_ref20": (cost_prod_blended_ref20, 0.00012),  # S6 + 20% OKX referral cashback on fees
}

# Default for new research code
cost_for_sym = cost_prod_blended
FUNDING_PER_12H = 0.00012
