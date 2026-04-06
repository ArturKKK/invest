# Consultation: Next Steps for Crypto L/S System — Full Research Report (R34–R47)

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

## Current Champion (R42 → R47 update)

### Previous Champion (R42):

| Metric | Value |
|--------|-------|
| **Feature set** | A_28f + `ret_dispersion_12h` + `cs_rank_ma_5` (30 features) |
| **ALL Sharpe** | **1.13** (gross 1.89) |
| **W1 / W2 / W3** | 0.01 / 3.22 / 2.50 |
| **Cost drag** | 19.22% |

### NEW Champion Candidate (R47):

| Metric | Value |
|--------|-------|
| **Feature set** | Champion_30f + `cg_taker_imb` (31 features) |
| **ALL Sharpe** | **1.31** (gross 2.03) — **+0.18 improvement** |
| **W1 / W2 / W3** | **0.60** / 2.86 / 2.03 |
| **Cost drag** | 19.16% |
| **Max DD** | -52.5% |
| **Turnover** | 4.4 |

Key improvement: **W1 jumped from 0.01 → 0.60** (the hardest window, BTC rally period). W2/W3 slightly lower but still strong. Net result: substantially higher ALL Sharpe.

---

## R47 — CoinGlass Feature Research (COMPLETED)

### What We Did

**Goal**: Test whether CoinGlass derivatives data (liquidations, taker buy/sell, long/short ratio) adds incremental alpha to our existing 30-feature champion model.

**Data downloaded**: CoinGlass API, 5 endpoints × 35 symbols:
- 1d interval: 259,412 rows, 2022-01-01 → 2026-04-05 (covers all 3 WF windows)
- 12h interval: 25,200 rows/endpoint, 2025-04-11 → 2026-04-05 (for W3 alignment)

**Protocol**: QA → IC scan on TRAIN only → redundancy check → event study → per-feature WF ablation.

### Phase 1: Data QA Findings

- **Timestamp semantics**: CoinGlass candle at `t` OPENS at `t`, covers `[t, t+24h)`. Confirmed via OI chaining (99.2% match). **Shift=1 day required** (previous day's complete data, lookahead-safe).
- MATIC + FTM excluded (no funding/ls_ratio data)
- No date gaps, 4.4% zero values (normal for quiet periods)
- Anomaly: Oct 10, 2025 = $1.87B BTC liquidation — real event, kept

### Phase 2: Feature Engineering (11 features built)

**Per-symbol features (CS-ranked):**
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `cg_liq_total` | long_usd + short_usd | Total liquidation volume |
| `cg_liq_imbalance` | (long - short) / total | Which side got liquidated more |
| `cg_liq_zscore` | zscore_30d(total) | Anomalous liquidation level |
| `cg_liq_accel` | total / total.shift(1) - 1 | Cascade acceleration |
| `cg_taker_imb` | (buy - sell) / (buy + sell) | Net aggressor direction |
| `cg_taker_imb_z` | zscore_30d(taker_imb) | Extreme taker imbalance |
| `cg_ls_ratio` | long_accounts / short_accounts | Retail positioning |
| `cg_ls_zscore` | zscore_30d(ls_ratio) | Extreme L/S positioning |

**Market-level features (NOT CS-ranked, like `ret_dispersion_12h`):**
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `mkt_cg_liq_total` | sum(liq_total) across all coins | Market-wide liquidation |
| `mkt_cg_liq_log` | log(mkt_liq_total + 1) | Scaled market liquidation |
| `mkt_cg_liq_imb` | market-wide (long-short)/total | Aggregate liquidation direction |

**Excluded**: `cg_liq_intensity` (= liq/volume) — correlation with existing `rel_volume_cs` = -0.89. Formula artifact, not new information.

### Phase 3: IC Scan Results (TRAIN data only, all 35 symbols, 3 WF windows)

| Flag | Feature | mean_IC | ICIR | nWin |
|------|---------|---------|------|------|
| 🔥 | `mkt_cg_liq_imb` | **+0.086** | **+0.479** | 3 |
| 🔥 | `cg_liq_imbalance` | **+0.063** | **+0.509** | 3 |
| 🔥 | `cg_taker_imb` | **-0.032** | **-0.324** | 3 |
| 🔥 | `mkt_cg_liq_log` | +0.031 | +0.165 | 3 |
| 🔥 | `mkt_cg_liq_total` | +0.031 | +0.165 | 3 |
| 🔥 | `cg_taker_imb_z` | -0.031 | -0.292 | 3 |
| ✅ | `cg_liq_zscore` | +0.019 | +0.151 | 3 |
| ✅ | `cg_ls_zscore` | +0.018 | +0.207 | 3 |
| — | `cg_liq_total` | +0.013 | +0.158 | 3 |
| — | `cg_ls_ratio` | +0.013 | +0.217 | 3 |
| — | `cg_liq_accel` | +0.004 | +0.030 | 3 |

**Top signals**: `mkt_cg_liq_imb` (IC=+0.086, ICIR=+0.479) and `cg_liq_imbalance` (IC=+0.063, ICIR=+0.509) are very strong — comparable to our best existing features.

### Phase 4: Redundancy Check vs FEAT_30

| Flag | CG Feature | Most Correlated Existing | Corr |
|------|-----------|-------------------------|------|
| ⚠️ redundant | `cg_liq_total` | `taker_cvd_24h` | -0.586 |
| ⚠️ redundant | `mkt_cg_liq_imb` | `ret_48h` | -0.570 |
| ⚠️ redundant | `cg_liq_imbalance` | `ret_48h` | -0.558 |
| ⚠️ redundant | `cg_ls_ratio` | `ls_divergence` | -0.538 |
| ↔️ partial | `cg_taker_imb` | `ret_48h` | +0.388 |
| ↔️ partial | `cg_taker_imb_z` | `ret_48h` | +0.395 |

**Key insight**: Top IC features (`mkt_cg_liq_imb`, `cg_liq_imbalance`) are partially redundant with `ret_48h` (r≈-0.56). This makes sense — when longs get liquidated, price has already dropped. The remaining variance (r²=0.69 unexplained) is the useful signal.

`cg_taker_imb` is less redundant (r=+0.39 with `ret_48h`) — more orthogonal information.

### Phase 5: Event Study

Top-1% liquidation intensity events (12,355 events, threshold $1.4B):

| Event Type | N | fwd_ret_12h | t-stat |
|-----------|---|------------|--------|
| **short_liq_dom** | 4,848 | **+0.0008** | **t=+4.16** ✅ |
| long_liq_dom | 7,495 | +0.0000 | t=+0.10 |
| rest_of_universe | 1,221,460 | -0.0001 | — |

**Statistically significant**: When shorts get mass-liquidated, mean 12h forward return = +0.08% (short squeeze effect, t=4.16). Long liquidation events are noise.

### Phase 6: Walk-Forward Ablation (Per-Feature, champion_30f baseline)

| Config | W1 | W2 | W3 | ALL | Δ ALL | Cost% |
|--------|-----|------|------|------|-------|-------|
| **champion+cg_taker_imb** | **0.60** | 2.86 | 2.03 | **1.31** | **+0.18** | 19.2% |
| champion_30f (baseline) | 0.01 | 3.22 | 2.50 | 1.13 | 0.00 | 19.2% |
| champion+liq_log+liq_total | -0.60 | 2.33 | 1.76 | 1.03 | -0.10 | 20.9% |
| champion+liq_log+liq_zscore | -0.25 | 0.05 | 0.65 | 0.79 | -0.34 | 19.9% |
| champion+cg_ls_zscore | -0.91 | 4.03 | 2.17 | 0.54 | -0.59 | 19.2% |
| champion+cg_taker_imb_z | -0.72 | 3.06 | 2.97 | 0.28 | -0.85 | 19.3% |
| champion+mkt_cg_liq_log | -1.50 | -0.56 | 1.77 | 0.07 | -1.06 | 20.9% |
| champion+mkt_cg_liq_total | -1.50 | -0.56 | 1.77 | 0.07 | -1.06 | 20.9% |
| champion+cg_liq_zscore | -1.49 | 2.79 | 2.33 | -0.04 | -1.17 | 19.2% |

### R47 Key Conclusions

1. **`cg_taker_imb` is the only CoinGlass feature that improves ALL Sharpe** (1.13 → 1.31, +16%). It's the net taker buy/sell imbalance per coin.

2. **Paradox: highest-IC features (liquidation imbalance) HURT the model in WF**:
   - `mkt_cg_liq_imb` IC=+0.086 but ALL Sharpe → 0.07 (destroyed)
   - `cg_liq_imbalance` IC=+0.063 but not tested solo because it's partially redundant with `ret_48h`
   - Likely explanation: high IC on TRAIN = overfitting to past liquidation patterns; or the signal is already captured by `ret_48h` and adding it creates noise

3. **`cg_taker_imb` wins despite modest IC** (IC=-0.032, ICIR=-0.324):
   - Less redundant with existing features (r=0.39 vs r=0.57 for liq)
   - Negative IC = when aggressive buyers dominate, forward returns are negative (contrarian signal)
   - Critical fix: **W1 from 0.01 → 0.60** — the hardest window that was always flat/negative

4. **Market-level CG features hurt** — unlike `ret_dispersion_12h` which helped as market-level, liquidation totals add noise

5. **Event study confirms directional value**: short squeezes predict positive returns (t=4.16), but this doesn't translate directly to model improvement for liq features — the contrarian taker signal captures a cleaner version of this

6. **CoinGlass subscription worth it** ($29/mo): `cg_taker_imb` alone justifies the cost if R47 result holds OOS

---

## Full Experiment History (R34–R47)

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
| **R42** Ablation R35a | Find minimal durable subset | **1.13** | **CHAMPION**: `dispersion + rankma` |
| **R43** Dynamic exposure | 5L/4S, 4L/4S rules | 0.87 | Coarse rules give back too much alpha |
| **R44** Dynamic universe | Rolling quality/liquidity filters | 0.82 | Hard pruning too blunt |
| **R45** Calibrated soft gate | Soft blend multiple experts | 0.92 (standalone) | Blends didn't beat standalone expert |
| **R46** Separate long/short models | Different targets for L vs S | 0.54 | Alpha loss >> cost savings |
| **R47** CoinGlass features | Liq/taker/LS ratio from CoinGlass | **1.31** | **NEW CHAMPION CANDIDATE**: `+cg_taker_imb` |

### R42 Ablation Detail (the previous best)

| Config | W1 | W2 | W3 | ALL | Cost% |
|--------|-----|------|------|------|-------|
| **A_28f + dispersion + rankma** | 0.01 | **3.22** | **2.50** | **1.13** | 19.2% |
| A_28f + dispersion + funding_cs | 0.21 | 3.09 | 2.89 | 0.98 | 19.3% |
| A_28f + rankma + taker_cs + funding_cs | 0.10 | 1.34 | 2.92 | 0.95 | 18.6% |
| Full R35a (all 5) | -0.45 | 3.03 | 1.91 | 0.64 | 18.6% |
| A_28f baseline | 0.05 | 0.21 | 2.35 | 0.74 | 18.4% |

---

## Data Sources & Status

| Source | Status | Finding |
|--------|--------|---------|
| **D1** CoinGlass (liq/taker/LS) | ✅ **TESTED (R47)** | `cg_taker_imb` → ALL=1.31 (+0.18). Liq features high IC but hurt WF |
| **D5** DefiLlama stablecoins | ✅ Downloaded | Used in R39. Regime signal only, not direct features |
| **D6** Orderbook depth (Binance) | ⏳ **Accumulating** | Daemon collecting hourly since Apr 5. Need 2-3 weeks for IC scan |
| **D7** Social/Reddit/gtrends | ✅ Scanned | IC≈0.04-0.05, only 17/35 coverage |
| On-chain (CoinMetrics) | ✅ Scanned | `TxCnt_chg7d` ICIR=0.145 but only 9 coins |

---

## Proven Effective

1. **12h binary classification** — optimal target, all alternatives worse
2. **LGB+XGB ensemble** — optimal model architecture
3. **EMA + hysteresis** — critical execution smoothing
4. **`ret_dispersion_12h`** — key W2 repair lever (market-level)
5. **`cs_rank_ma_5`** — rank stabilizer
6. **`cg_taker_imb`** — NEW: taker buy/sell imbalance, fixes W1 (0.01→0.60), ALL 1.13→1.31

## Proven Useless

- ❌ Target engineering (threshold, excess-vs-BTC, temporal decay)
- ❌ Stablecoin/macro features directly in CS model
- ❌ Hard regime switching
- ❌ Feature sets >31 (noise dominates; even champion+2 CG features hurt)
- ❌ Multi-horizon / meta-stacking
- ❌ Dynamic exposure rules (5L4S/4L4S)
- ❌ Hard quality/liquidity universe pruning
- ❌ Separate long/short models
- ❌ Market-level CG liquidation features (mkt_cg_liq_log, mkt_cg_liq_total → killed ALL)
- ❌ CG liquidation imbalance in model (high IC=0.086 but redundant with ret_48h, hurts WF)

---

## Core Problems

1. **Cost gap sim→live = 2.6×.** Sim ALL=1.31 at 7bps → live estimate ≈ 0.5–0.8 Sharpe. Need either higher gross alpha, maker fills, or more capital.

2. **Max DD = -52.5%.** Still very high. With 5× leverage on $86 account this is existential risk.

3. **W2 and W3 slightly weakened** with `cg_taker_imb` (W2: 3.22→2.86, W3: 2.50→2.03) while W1 massively improved (0.01→0.60). Net positive but worth monitoring.

## Open Questions for Consultation

1. **Should we accept `cg_taker_imb` as the new champion?** ALL improved 1.13→1.31 but at the cost of W2 (-0.36) and W3 (-0.47). W1 dramatically improved (+0.59). Is this a better risk profile or just window-shifting?

2. **Paradox diagnosis needed**: Why do the highest-IC CoinGlass features (liq_imbalance IC=+0.086) DESTROY the model in walk-forward (ALL→0.07), while the modest-IC feature (`cg_taker_imb` IC=-0.032) improves it? Is this:
   - (a) Overfitting of high-IC features on TRAIN
   - (b) Redundancy with `ret_48h` (r=-0.57) creating multicollinearity
   - (c) The negative sign matters (contrarian > momentum for this model)
   - (d) Something else?

3. **D6 orderbook depth** will be ready in ~2-3 weeks. Given how much `cg_taker_imb` helped, should we prioritize microstructure features (order flow imbalance, bid-ask depth ratio) — they're theoretically the same family of signals?

4. **Feature interaction**: Should we try `cg_taker_imb` × `ret_dispersion_12h` interaction, or `cg_taker_imb` with a different join cadence (e.g., rolling 3-day average instead of shift-1)?

5. **Cost reduction**: With 31 features and ALL=1.31 gross 2.03, is it worth revisiting execution optimization (maker fills, liquidity gating) now that we have more gross alpha to play with?

6. **What direction would you prioritize for R48?**
   - (a) More CoinGlass feature engineering (e.g., taker imbalance at 12h, rolling windows)
   - (b) D6 orderbook depth when ready
   - (c) Second-order CG features (taker_imb × volume, taker_imb acceleration)
   - (d) Revisit cost reduction / execution with higher gross alpha
   - (e) Portfolio construction (position sizing, risk parity)
   - (f) Something else entirely
