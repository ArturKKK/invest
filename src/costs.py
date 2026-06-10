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
}

# Default for new research code
cost_for_sym = cost_prod_blended
FUNDING_PER_12H = 0.00012
