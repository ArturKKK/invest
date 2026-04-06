# Consultation: Next Steps for Crypto L/S System — Full Research Report (R34–R46)

## System Summary

Solo quant. Systematic crypto long/short on OKX perpetual swaps. 35 coins, 5× leverage, 12h rebalance.

- **Model**: LightGBM + XGBoost binary ensemble, `P(fwd_ret_12h > 0)`, cross-sectional ranking
- **Portfolio**: 6 long / 3 short, 12h rebalance
- **Execution**: EMA smoothing (alpha=0.5) + hysteresis (band=3) for turnover reduction
- **Cost model**: 7 bps one-way (taker + slippage) in sim; ~18 bps effective on live OKX (2.6× gap)
- **Evaluation**: expanding-window walk-forward, 5 seeds × 3 OOS windows:
  - W1 = Oct 2024 – Jan 2025 (BTC rally, kills mean-reversion)
  - W2 = May – Aug 2025 (choppy, weakest period)
  - W3 = Nov 2025 – Mar 2026 (most alpha-rich)
  - ALL = full history combined

**Currently deployed live with real money ($86, 5× leverage).**

---

## Current Champion

| Metric | Value |
|--------|-------|
| **Feature set** | A_28f + `ret_dispersion_12h` + `cs_rank_ma_5` (30 features) |
| **ALL Sharpe** | **1.13** (gross 1.89) |
| **W2 Sharpe** | **3.22** (was -0.98 at baseline) |
| **W3 Sharpe** | **2.50** |
| **W1 Sharpe** | 0.01 (flat — acceptable) |
| **Cost drag** | 19.22% of gross equity (turnover 4.5) |
| **Max DD** | -54.2% |
| **Equity (sim)** | $228 from $100 start |

Key features of the champion: `ret_dispersion_12h` (market-level, excluded from CS ranking) + `cs_rank_ma_5` (smoothed rank). This specific pair fixed W2 from -0.98 to +3.22 while keeping W3 alive.

---

## Full Experiment Matrix (R34–R46)

### Feature & Architecture Experiments

| Experiment | Hypothesis | ALL Sharpe | Verdict |
|------------|-----------|------------|---------|
| **R34** W2 attribution | Diagnose W2 failure | — | Long-leg Sharpe -1.71, shorts OK (+0.38). Toxic longs: XRP, ADA, SAND, APT |
| **R35** New features scan | CS second-order features | 0.64 | `r35a_cs_second_order` = best durable bundle, interactions unstable |
| **R36** Regime gating | Hard regime switch between experts | 0.92 (standalone stability expert) | Gates fix W2 (3.72) but destroy ALL |
| **R37** Cost-aware execution | Liquidity filter, score-edge gates | 0.74 | `liq70` cuts cost to 4%, but kills W1→ALL |
| **R38** Target engineering | Threshold, excess-vs-BTC, temporal decay | 0.71 | **All worse than baseline** |
| **R39** Stablecoin features | `stable_flow4` as direct features | 0.17 | Fixed W2 (2.46) but killed W3/ALL. Regime signal only |
| **R41** Consolidation matrix | Combine all prior winners | 0.74 | No combo beat `A_28f` on ALL |
| **R42** Ablation R35a | Find minimal durable subset | **1.13** | **NEW CHAMPION**: `dispersion + rankma` |
| **R43** Dynamic exposure | 5L/4S, 4L/4S rules | 0.87 | Coarse rules give back too much alpha |
| **R44** Dynamic universe | Rolling quality/liquidity filters | 0.82 | Hard pruning too blunt |
| **R45** Calibrated soft gate | Soft blend R42 × stability × stablecoin experts | 0.92 (standalone) | Blends didn't beat standalone expert |
| **R46** Separate long/short models | Different targets for L vs S | 0.54 | Alpha loss >> cost savings |

### R42 Ablation Detail (the winning experiment)

Ablated the 5-feature R35a bundle (`ret_dispersion_12h, cs_rank_ma_5, oi_chg_12h_cs, taker_cvd_12h_cs, cum_funding_24h_cs`) down to every subset:

| Config | W1 | W2 | W3 | ALL | Cost% |
|--------|-----|------|------|------|-------|
| **A_28f + dispersion + rankma** | 0.01 | **3.22** | **2.50** | **1.13** | 19.2% |
| A_28f + dispersion + funding_cs | 0.21 | 3.09 | 2.89 | 0.98 | 19.3% |
| A_28f + rankma + taker_cs + funding_cs | 0.10 | 1.34 | 2.92 | 0.95 | 18.6% |
| A_28f + dispersion + rankma + taker_cs | -0.65 | 3.74 | 3.03 | 0.91 | 18.9% |
| Full R35a (all 5) | -0.45 | 3.03 | 1.91 | 0.64 | 18.6% |
| A_28f baseline | 0.05 | 0.21 | 2.35 | 0.74 | 18.4% |

### R45 Soft Gate Detail

| Config | W1 | W2 | W3 | ALL |
|--------|-----|------|------|------|
| expert_stability (30f) | -0.47 | 1.06 | 2.84 | **0.92** |
| soft_tri blend | -0.54 | 2.76 | 1.35 | 0.61 |
| soft_r42_vs_flow | -0.25 | 2.84 | 2.14 | 0.60 |
| expert_r42 | -0.32 | 3.02 | 3.42 | 0.50 |
| expert_stable_flow | -0.61 | 2.76 | 1.97 | 0.37 |

### R46 Asymmetric Long/Short Detail

| Config | W1 | W2 | W3 | ALL | Cost% | Turnover |
|--------|-----|------|------|------|-------|----------|
| **Unified baseline** | 0.01 | **3.22** | **2.50** | **1.13** | 19.2% | 4.5 |
| Asymmetric L/S | -1.17 | 1.80 | 2.08 | 0.54 | 12.2% | 2.4 |

---

## Data Sources & Status

| Source | Status | Finding |
|--------|--------|---------|
| **D1** Liquidations (CoinGlass) | ❌ BLOCKED | Binance endpoints 404; need paid CoinGlass API key ($29/mo) |
| **D5** DefiLlama stablecoins | ✅ Downloaded | 360 assets, 3K+ global history rows. Used in R39 |
| **D6** Orderbook depth (Binance) | ⏳ Accumulating | Infra ready, daemon collecting hourly. Need 2-3 weeks for IC scan |
| **D7** Social/Trends | ✅ Scanned | Reddit subs IC≈0.05, gtrends IC≈0.04. Real signal, but 17/35 symbol coverage |
| On-chain (CoinMetrics) | ✅ Scanned | `TxCnt_chg7d` ICIR=0.145 but only 9 coins. Exchange flows = zero signal |

---

## Proven Effective

1. **12h binary classification** — optimal target, all alternatives worse
2. **LGB+XGB ensemble** — optimal model architecture
3. **EMA + hysteresis** — critical, cuts cost drag from 57% → 35%
4. **`ret_dispersion_12h`** — key W2 repair lever (market-level, needs `cs_rank_exclude`)
5. **`cs_rank_ma_5`** — stabilizer, without it dispersion alone fails on ALL
6. **Regime filter** — BTC trend_strength scaling

## Proven Useless

- ❌ Target engineering (threshold, excess-vs-BTC, temporal decay)
- ❌ Stablecoin/macro features directly in CS model
- ❌ Hard regime switching
- ❌ Feature sets >30 (noise dominates)
- ❌ Multi-horizon / meta-stacking
- ❌ Dynamic exposure rules (5L4S/4L4S)
- ❌ Hard quality/liquidity universe pruning
- ❌ Separate long/short models

---

## Core Problem

**Cost gap sim→live = 2.6×.** Sim ALL=1.13 at 7bps → live estimate ≈ 0.4–0.7 Sharpe. With $86 capital and 5× leverage, this is borderline. Need either:
- (a) Substantially higher gross alpha (ALL gross > 3.0)
- (b) Substantially cheaper execution (maker fills)
- (c) More capital
- (d) New orthogonal data sources that add real incremental IC

## Open Questions for Consultation

1. **D6 orderbook features** will be ready in 2-3 weeks — microstructure features are theoretically orthogonal to everything we have. Priority to test.
2. **D1 liquidations** — should we buy CoinGlass API key ($29/mo)? Liquidations are the strongest theoretically orthogonal signal we haven't tested.
3. **Social/Reddit features** (D7) — real IC=0.04-0.05 but only 17/35 coverage. Worth a bounded WF experiment on covered coins?
4. **Maker execution** — if we solve limit order fill rates, cost gap drops from 2.6× to ~1.5×, making current Sharpe 1.13 viable at live ~0.8.
5. **Capital** — with $86 it's impossible to properly size 9 positions. What's the minimum viable capital for this strategy?
6. Given the full experiment history above, **what direction would you prioritize** for the next research cycle?
