# Project Progress — AI Crypto Trading System

**Последнее обновление:** 2026-06-11  
**Статус:** Phase 5 — LIVE trading. Honest S6 retest cycle (R136/R137) complete: canonical baseline Net Sharpe **2.831** (S6 prod_blended costs, include-flat 1013-period accounting). Only **GATED_A1** overlay survives honest retest (+0.268 → 3.099, bootstrap P=0.90) — pending pristine OOS confirmation. Ridge α=1000 (R7) still the deployed VPS model.

---

## R128–R137 (June 2026) — Honest S6 Retest of R128 Overlays + Hysteresis Cutoff Grid

> **Why this section exists**: R128 overlays (A1 asymmetric-kelly, G2 vol-weighting, persistence gate) were originally measured with the **LENIENT pre-R121 cost model** (Tier1 −0.36bp rebate) **AND** the old skip-mode period accounting (drop risk-off periods instead of holding flat). Both inflated the headline numbers. R136/R137 re-ran everything on the canonical cache under **S6 prod_blended costs** with **include-flat 1013-period accounting**. Most overlays collapsed.

### R128 — Original overlay sweep (⚠️ measured with LENIENT costs + old skip accounting → INFLATED)
- Three overlays proposed: **A1** asymmetric-Kelly (size longs/shorts differently), **G2** volatility-weighting of positions, **persistence/hysteresis gate** (require a signal to persist L hours, quantile q).
- Original headline gains were positive but **not trustworthy**: the Tier1 −0.36bp maker rebate (pre-R121) and the SKIP risk-off accounting (which dropped risk-off periods entirely rather than holding the prior book flat and recording fwd_ret) both biased the Sharpe upward. Treated as candidates pending an honest retest.

### R136 — Honest S6 retest on canonical cache (commit ~`9ad3755`)
- **Setup**: S6 prod_blended per-trade costs (Tier1 2.4bp / Tier2 5.5bp / Tier3 10bp), include-flat **1013-period** accounting (risk-off periods held flat, fwd_ret recorded), bootstrap P(improve) vs baseline.
- **Baseline Net Sharpe = 2.831.**

| Overlay | Net Sharpe | Δ vs base | Bootstrap P(improve) | Verdict |
|---|---:|---:|---:|---|
| Baseline (no overlay) | 2.831 | — | — | reference |
| **GATED_A1** (persistence L=720, q=0.20 + A1 kelly) | **3.099** | **+0.268** | **0.90** | ✅ **survives** |
| A1 plain (asymmetric kelly, no gate) | ~2.83 | ~0 | <0.80 | ❌ not significant |
| G2 vol-weighting (as originally coded) | (inflated) | — | — | ⚠️ **LOOKAHEAD LEAK** |
| G2 vol-weighting (honest, leak fixed) | ~2.74–2.78 | −0.05..−0.09 | — | ❌ dead |
| SKIP risk-off mode | ~0.77 | **−2.06** | — | ❌ **dead** |

- **GATED_A1** (persistence gate L=720, q=0.20 combined with A1 asymmetric kelly) is the **only** overlay that survives: +0.268 Net Sharpe → **3.099**, bootstrap **P(improve)=0.90**.
- **SKIP risk-off mode is DEAD**: −2.06 Sharpe under honest include-flat accounting. The old skip behavior (dropping risk-off periods) had silently flattered it.
- **G2 vol-weighting had a LOOKAHEAD LEAK** in the original implementation. Honest (leak-fixed) version is **−0.05..−0.09** → dead.
- **Plain A1** (asymmetric kelly without the persistence gate) is **not significant** — the gate is what carries GATED_A1.

### R137 — Hysteresis cutoff grid (proper state-machine test)
- **Motive**: the May-29 working tree claimed "0.95 cutoff wins" for the trend filter. R137 tested this honestly as a **state machine** with separate on/off thresholds (`cutoff_on` to enter risk-off, `cutoff_off` to exit), not a single threshold.
- **Grid**: `cutoff_on ∈ {0.90, 0.95, 1.00}` × `cutoff_off ∈ {−0.05, −0.10}`, portfolio 4L/2S, S6 prod_blended costs.
- **Result**: **NO cell beats the baseline 0.90/0.80 at P(improve) ≥ 0.80.** Higher `cutoff_on` values (0.95, 1.00) **worsen drawdown** without a compensating Sharpe gain.
- **VERDICT**: **keep 0.90/0.80.** The May-29 "0.95 wins" was a **methodology artifact** — it used a single threshold (not a hysteresis state machine), a 6L/3S book (not 4L/2S), and the lenient pre-R121 cost model. Under the honest setup the effect disappears. **Cutoff tuning CLOSED.**

### Data refresh (state as of 2026-06-10/11)
- OHLCV / futures / funding / premium / dvol / fng all current through **2026-06-10**; macro through **2026-06-09**.
- ⚠️ **CoinGlass subscription DEAD** ("Upgrade plan" response). `cg_taker_imb` — the **only** CoinGlass-derived feature (1 of 31) — is **frozen at 2026-05-06**. All other 30 features remain live.

### In progress (MLC VM)
- **Pristine OOS test** on the held-out window **2026-04-26 → 2026-06-08** (data never seen during R128–R137 tuning), to confirm whether GATED_A1's +0.268 / P=0.90 holds out of sample.
- **`cg_taker_imb` ablation**: quantify the model's dependence on the now-frozen CoinGlass feature (drop it, retrain, compare) given the subscription is dead.

### Net verdicts (R128–R137)
- ❌ **CLOSED**: cutoff/hysteresis tuning (keep 0.90/0.80, R137); SKIP risk-off mode (−2.06, dead); G2 vol-weighting (lookahead leak; honest = dead).
- 🟡 **PROMISING, pending OOS**: **GATED_A1** (L=720, q=0.20) — +0.268 → 3.099, P=0.90 on the canonical cache; awaiting pristine-OOS confirmation before any deploy.

---

## R127 (2026-04-23) — Ablation 2×2 + 6-dead-feature replacement research

**Motive**: user challenged "Sharpe 3.777 is fake" claim from R126. Ran systematic 2×2 ablation to isolate which of 3 "bug fixes" from commit `b325afc` destroyed Sharpe.

### Bugs under test
- **Fix#1** (`_ic_scanner.py build_features_minimal`): inf→nan cleanup of `oi_chg` etc.
- **Fix#2** (`_research_r35_new_features.py`): 6 "dead" feats added to `MARKET_LEVEL_FEATURES` set (skip CS-rank): `pct_coins_up_12h`, `pct_coins_up_1h`, `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`.
- **Fix#3** (`run_trading.py`): `cum_funding_24h` Binance override — LIVE-only, no-op for R68 backtest. Matrix reduced to 2².

### Results (Net Sharpe 4L/2S continuous, 688 periods, cef6e2f simulate skip-mode)
| Config | Fix1 | Fix2 | GrossSh | **NetSh** | Δ vs baseline |
|---|:---:|:---:|---:|---:|---:|
| F10_F20 (baseline) | ❌ | ❌ | 4.297 | **3.777** ✓ | — |
| F10_F21 | ❌ | ✅ | 2.611 | **2.135** | **−1.642** 🔥 |
| F11_F20 | ✅ | ❌ | 3.754 | **3.224** ✓ | −0.553 |
| F11_F21 | ✅ | ✅ | 2.831 | **2.356** ✓ | −1.42 |

- Baseline **3.777 reproduces exactly** → R126's "fake Sharpe" conclusion was WRONG.
- **Fix#2 is primary culprit** (−1.64 Sharpe alone).
- F10_F20 & F11_F20 match memory [mem_2fd5b69cc98e, mem_53b864c12268].

### Interpretation — why Fix#2 hurts
The 6 feats ARE market-wide constants per timestamp (hour_sin is same for all coins at t=now). Without Fix#2 they go through CS-rank → become identically-zero → effectively occupy a "dead slot" that the model ignores. With Fix#2 they pass raw values → model latches onto time-of-day / breadth regime hints → overfits seasonal spurious patterns → live degradation.

Paradoxically: **"dead weight" slot > "real information" slot** when the information lacks per-symbol variation.

### Decision (user directive verbatim)
> «нужно их оставить, конечно, и начать ресерч на тему на что их можно заменить»

- Keep 6 feats in CS-rank path (revert Fix#2).
- Research replacements with **per-symbol variance** so CS-rank produces real signal.

### Replacement experiments (launched on invest-1v40rg)
All 5 candidate sets each replace the 6 dead slots with 6 new per-symbol feats.

| Exp | Idea | Added feats |
|---|---|---|
| A_seasonal_x_symbol | hour/dow harmonics × per-symbol signals | `seas_hsin_x_premz`, `seas_hcos_x_relvol`, `seas_dsin_x_ret12`, `seas_dcos_x_oichg`, `seas_hsin_x_ret1`, `seas_dsin_x_relvol` |
| B_hod_vol_rank | per-(symbol, hour-of-day / dow) expanding vol & return means + own up-rate | `hod_vol_z`, `dow_vol_z`, `hod_ret_mean`, `dow_ret_mean`, `up_rate_12h_sym`, `up_rate_48h_sym` |
| C_relative_breadth | per-symbol deviation from market breadth | `breadth_dev_1h`, `breadth_dev_12h`, `ret_12h_mad_z`, `ret12_cs_rank_ma24`, `vol_share`, `vol_share_chg_12h` |
| D_session_regime | Asia/EU/US session × per-symbol rel_volume + weekend/monday effects | `asia_x_relvol`, `eu_x_relvol`, `us_x_relvol`, `weekend_x_ret12`, `ret_session_8h`, `monday_x_ret12` |
| E_regime_x_beta | market-breadth × own return + BTC-corr × breadth | `breadth_x_ret12`, `breadth_x_ret12_sign`, `btc_corr_x_breadth`, `disp_x_ret12`, `breadth_x_ret1_sym`, `aligned_with_mkt` |
| F_drop_only | control: just drop the 6 dead feats | — |
| BASELINE | sanity (no changes) | — |

Scripts (in repo root):
- [_replacement_features.py](_replacement_features.py) — 6 experiments, each returns 6 per-symbol feats.
- [_run_replacement.py](_run_replacement.py) — wrapper: monkey-patches `r68.load_data` to inject features + rewrite `CHAMPION_FEAT_31` (drops 6 DEAD, appends new).
- [_replacement_all.sh](_replacement_all.sh) — sweep runner writing `/data/datasets/replacement_${EXP}.csv` + summary CSV.

Target: beat F10_F20 baseline Net Sharpe **3.777**. Anything >3.8 wins; >4.0 would be substantial improvement.

### Replacement sweep results (completed 2026-04-23T16:02+03:00)

| Exp | 4L/2S cont NetSh | 4L/2S orig | 6L/3S cont | 6L/3S orig | Δ vs BASELINE (4L cont) |
|---|---:|---:|---:|---:|---:|
| **BASELINE** | **3.777** ✓ | 2.984 | 2.509 | 1.779 | — |
| B_hod_vol_rank | 2.898 | 2.561 | 2.630 | 2.175 | **−0.879** |
| F_drop_only | 2.898 | 2.561 | 2.630 | 2.175 | −0.879 |
| E_regime_x_beta | 2.282 | 2.028 | **2.951** | 2.213 | −1.495 (but 6L best!) |
| D_session_regime | 2.108 | 1.554 | 2.097 | 1.504 | −1.669 |
| A_seasonal_x_symbol | 2.102 | 1.827 | 2.378 | 1.713 | −1.675 |
| C_relative_breadth | 1.889 | 1.276 | 1.811 | 0.899 | −1.888 |

**Verdict: no replacement beats baseline 3.777.**

Key findings:
1. **B_hod_vol_rank ≡ F_drop_only bit-for-bit** — B's 6 new feats had zero effect (likely all NaN on early WF folds where expanding window min_periods=30 fails). Both effectively = "just drop 6 dead feats" = **2.898**.
2. **Drop-only (2.898) > Fix#2=F21 raw (2.356) > F21 no-Fix#1 (2.135)**: the ranking is clear:
   - keep 6 as dead CS-rank slot = **3.777** (best)
   - drop them entirely = 2.898 (−0.88)
   - add per-symbol replacements = 1.89-2.90 (same or worse)
   - pass them raw bypassing CS-rank = 2.14 (worst)
3. **Dead slot > real information** paradox confirmed. The CHAMPION_FEAT_31 set appears to be a **tightly-coupled optimum**: removing any slot (even a "dead" one) breaks model calibration.
4. **E_regime_x_beta 6L/3S cont = 2.951** vs baseline 2.509: the only bright spot. For 6L/3S config, breadth × own-return interactions add value. Not a 4L winner though, so moot for prod (4L is deployed size).

Decision: **revert to F10_F20 state, do NOT replace 6 feats**. Fix#2 reverted (critical, -1.64 Sharpe bug). Fix#1 reverted (costs 0.55 Sharpe, and VPS never had it — prod already matches 3.777).

### R127.7 — LIVE vs Backtest Parity Gap 🚨
Checked VPS at commit `ccb3bc2` (before b325afc): Fix#1/Fix#2 NEVER deployed. VPS was already at baseline F10_F20. Both local fixes were divergence, now synced.

**LIVE reality** (bot.log 2026-03-16 → 2026-04-23, 38 days, $110→$80 = −27.3%):

| Metric | Backtest F10_F20 | LIVE | Δ |
|---|---:|---:|---:|
| Net Sharpe (ann) | +3.777 | **−4.10** | −7.9 🔥 |
| Win rate | 59.7% | 23.7% (9/38) | −36pp |
| Ret total | +179% | −27.3% | −206pp |
| Trades | 688 WF per | 38 real | small sample |

Not a code bug — identical code. **Classic sim-prod parity gap** per R25 checklist:
- $80 capital × 1x lev = $5-10 positions → min-order fees eat pnl
- 51 bot restarts in 5 weeks, `trading_state.json` = `{}` → state lost each restart
- Sim-prod parity bugs (edge_boost, dynamic L/S, sm_scale overlays) possibly still active in CLS mode
- Downloaded d6 data → [data_vps_d6/](data_vps_d6): 21200 OB snapshots (50 syms, 2 weeks), 7 coinglass parquets, bot.log

**No deploy needed** — VPS already at target state. Next: **R128 sim-prod parity audit** (walk checklist against `run_trading.py`, save 100 LIVE predictions, diff offline).


Artifacts:
- [_replacement_summary.csv](_replacement_summary.csv) — all 7×4=28 rows of results.
- [data_vps_d6/](data_vps_d6) — 13MB of VPS prod data (orderbook, coinglass, bot log) pulled 2026-04-23.
- Memory: [mem_c2e2190b313d] ablation matrix, [mem_007762304796] replacement plan, [mem_68baf6123cc3] replacement results, repo memory `r127_live_parity_gap.md`.

---

## 0. Ключевые результаты (TL;DR)

### Baseline v1 (ПРОВАЛ)
- IC = 0.005, Sharpe = -1.0 — модель хуже рандома
- Проблемы: time features leakage, нет cross-sectional normalization, переобучение

### Baseline v2 (УСПЕХ) ← текущий
- **Rank IC = 0.031, ICIR = 0.36, LS Sharpe = 3.87** (rank model)
- **Ensemble LS Sharpe = 4.21** (среднее 3 моделей)
- Long-Short работает отлично, Long-Only нет (рынок падал в test периоде)
- Top фичи: MA-ratios, GK-volatility, Sharpe-ratios, CCI

### Baseline v3 (CLUSTER RUN)
- **Best horizon: 4h** (LS Sharpe = 3.82)
- Rank IC = 0.0287, ICIR = 0.3366, Rank ICIR = 0.5789
- Cross-asset фичи работают (btc_vol_24h в Top-4)
- Regime filter (72h MA) бесполезен: 49.9% ON = монетка

### v4 LightGBM (CLUSTER RUN) ←
- **LS Sharpe = 4.00** (5-seed ensemble, 94 features)
- Rank IC = 0.0290, ICIR = 0.3535, Rank ICIR = 0.5552
- Feature selection 118 → 94 (+0.05 LS Sharpe)
- ⚠️ Optuna HPO не запустился (Python 3.11 на кластере не видит optuna)
- Advanced regime ХУЖЕ v3 для LO: $2.21 vs $14.05 (v3 regime)
- breadth_pct_positive — топ фича (#1), regime_btc_dd_720 тоже важна

### HIST Transformer (GPU, H100) ← ЛУЧШИЙ
- **Rank IC = 0.0752** (2.6× лучше LightGBM!)
- **Rank ICIR = 0.5296**, ICIR = 0.3482
- Val Rank IC = 0.0708 (best epoch 9/80, early stop at 24)
- 502K params, embed(105→128) + concept(8) + cross_attn(2L,4H)
- ✅ Eval bug FIXED — теперь LS Sharpe считается по actual returns, не по ranks

### MASTER Transformer (GPU, H100) ← НЕ ДАЛ ПРИРОСТА
- Rank IC = 0.0738 (≈ HIST 0.0752, нет прироста)
- Best epoch = 2 (мгновенное переобучение)
- Вывод: архитектуры слишком похожи на HIST, заменён на GRU

### GRU Temporal Model (ПРОВАЛ — слабый сигнал)
- Архитектура: proj → BiGRU(2L) → temporal_attention → gate → head
- Принципиально другой подход: temporal per-coin (не cross-sectional как HIST/MASTER)
- Per-coin rolling z-score нормализация (не cross-sectional rank)
- **Результат: Rank IC = 0.035, LS Sharpe = 1.16, MaxDD = -68%**
- Вывод: temporal per-coin подход значительно слабее cross-sectional. Убран из ансамбля.

### Ensemble (HIST+LGB) ← ЛУЧШИЙ РЕЗУЛЬТАТ
- **Rank IC = 0.078, LS Sharpe = 4.38, MaxDD = -55.2%**
- HIST+LGB = лучшая комбинация (MASTER/GRU не помогают)
- Все LS Sharpe — GROSS (без учёта комиссий)

### Phase 4: Sentiment (NEW)

#### Sentiment Data (OKX + Alternative.me)
- **Fear & Greed Index:** 2953 дня (с 2018), mean=46.6
- **Funding Rates (OKX):** 15,241 строк, 46 символов, ~3 мес. истории
- **Open Interest (OKX):** 1,900 строк, 19 символов, ~100ч истории
- **Long/Short Ratio (OKX):** 1,900 строк, 19 символов, ~100ч истории
- ⚠️ Binance Futures API заблокирован из России → заменён на OKX public API
- ⚠️ OKX даёт только ~3 мес. funding и ~100ч OI/LSR → используем synthetic proxies для backtest

#### LGB v5 с Sentiment + Risk Overlay
**Скрипт:** `run_pipeline_v5.py`
- 143 фичи (98 оригинальных + 25 cross-asset + 20 sentiment/positioning)
- **Sentiment фичи:** FNG (value, MA7, MA30, momentum, extreme zones), funding rates, synthetic proxies (reversal scores, volume surge, cross-coin dispersion, BTC beta)
- **Risk overlay:** vol targeting (target=2%), drawdown stop (-25% DD circuit breaker, resume at -10%)
- **Rolling walk-forward:** 3 окна (W1→2024-12, W2→2025-03, W3→latest)
- **Реалистичная модель комиссий:** 0.03% blended maker + 0.01% slip + 0.005%/8h funding, 25% turnover

**Результаты кластер (3 walk-forward окна, 5 seeds, Optuna не установлен):**

| Window | Rank IC | Rank ICIR | LS Sharpe raw | LS Sharpe net | MaxDD net | DDStop Sharpe | DDStop MaxDD |
|--------|---------|-----------|---------------|---------------|-----------|---------------|------ -------|
| W1 (→2024-12) | 0.026 | 0.415 | 2.77 | 0.39 | -67.5% | -0.06 | -43.8% |
| W2 (→2025-03) | 0.026 | 0.503 | 4.04 | 1.45 | -70.1% | 1.05 | -37.6% |
| W3 (→latest) | 0.029 | 0.565 | 4.20 | 1.70 | -67.9% | 0.99 | -41.2% |
| **AVG** | **0.027** | **0.494** | **3.67** | **1.18** | **-68.5%** | **0.66** | **-40.9%** |

**Combined (all windows):** LS Net Sharpe=1.35, LS DDStop Sharpe=0.87 (MaxDD -46.2%), Cost=2.2 bps/period

⚠️ W1 слабее (Sharpe net 0.39, DDStop -0.06) — меньше train данных, короткий test.
W2/W3 стабильные: net Sharpe 1.45-1.70, DDStop ~1.0.

**7/30 top фичей — sentiment:** fng_ma30 (#4), fng_momentum (#7), fng_ma7 (#8), btc_beta_168h (#15), btc_beta_48h (#18), fng_value (#20), reversal_24v168 (#24)

#### HIST v2 с Sentiment-Aware Architecture (CLUSTER RUN ✅)
**Скрипт:** `run_hist_v2.py`
- Отдельная sentiment embedding ветка: `sent_embed(15→128)` + gated fusion
- 544K params (vs 502K v1), best epoch 7/80, early stop at 22
- Val Rank IC: 0.0723

**Результаты кластер (H100, 80 epochs):**

| Metric | HIST v2 | HIST v1 |
|--------|---------|---------|
| Rank IC | **0.074** | 0.067 |
| Rank ICIR | **0.533** | 0.530 |
| LS Sharpe raw | 3.91 | 4.25 |
| LS Sharpe net | **1.23** | N/A (gross) |
| LS MaxDD net | -64.9% | -51% (gross) |
| LS DDStop Sharpe | 0.70 | N/A |
| LS DDStop MaxDD | **-33.6%** | N/A |

✅ Rank IC улучшился: 0.074 vs 0.067 (v1). Sentiment branch добавляет сигнал.
⚠️ LS Sharpe raw упал: 3.91 vs 4.25 — вероятно шум от ограниченных sentiment данных.
✅ DDStop MaxDD -33.6% — лучший показатель по просадке среди всех моделей.

### Сводная таблица (все модели, актуальные результаты)

| Model | Rank IC | LS Sharpe (gross) | LS Sharpe (net) | DDStop Sharpe | DDStop MaxDD |
|-------|---------|-------------------|-----------------|---------------|------ -------|
| LGB v4 | 0.029 | 4.00 | ~1.5* | N/A | N/A |
| HIST v1 | 0.067 | 4.25 | ~1.7* | N/A | N/A |
| HIST+LGB v1 ensemble | 0.078 | 4.38 | ~1.8* | N/A | N/A |
| **LGB v5 (avg 3 win)** | **0.027** | **3.67** | **1.18** | **0.66** | **-40.9%** |
| **HIST v2** | **0.074** | **3.91** | **1.23** | **0.70** | **-33.6%** |

*оценка, пересчёт с теми же 2.2 bps/period

### Что нужно дальше
1. **Запустить ensemble v2** на кластере: `python run_ensemble_v2.py`
2. **Оптимизировать rebalance частоту** — 8h/12h вместо 4h (снижение костов)
3. **Paper trading** на OKX (run_paper_trading.py)

---

## 1. Что уже сделано

### ✅ RFC и Architecture Design
- Написан подробный RFC: [RFC_TRADING_SYSTEM.md](RFC_TRADING_SYSTEM.md)
- Выбрана архитектура: Multi-model ensemble (MASTER + HIST + LightGBM)
- Рынок: Crypto, 1h intraday, top 50 койнов, OKX (основная биржа)
- Капитал: $100-1K, spot only

### ✅ Project Setup
- Python 3.13, venv в `./venv`
- macOS M3 Pro, `libomp` установлен через brew для LightGBM
- Установлены пакеты: `ccxt pandas numpy lightgbm xgboost scikit-learn plotly ta vectorbt pyarrow tqdm joblib scipy`
- Структура:
```
invest/
├── RFC_TRADING_SYSTEM.md          # Детальный RFC
├── PROGRESS.md                    # Этот файл
├── run_pipeline.py                # v1 pipeline (baseline, плохие результаты)
├── run_pipeline_v2.py             # v2 pipeline (cross-sectional rank, 3 модели + ensemble)
├── run_pipeline_v3.py             # v3 pipeline (multi-horizon + cross-asset + regime)
├── run_pipeline_v4.py             # v4 pipeline (HPO ICIR + advanced regime + multi-seed)
├── run_pipeline_v5.py             # v5 pipeline (sentiment + risk overlay + walk-forward)
├── run_hist_model.py              # HIST transformer v1 (cross-stock attention)
├── run_hist_v2.py                 # HIST transformer v2 (sentiment-aware, gated fusion)
├── run_master_model.py            # MASTER transformer (≈HIST, dropped)
├── run_gru_model.py               # GRU temporal model (слабый, dropped)
├── run_ensemble.py                # Final multi-model ensemble evaluator
├── run_paper_trading.py           # OKX paper trading (signal → execution)
├── requirements-cluster.txt       # Зависимости для кластера (CPU)
├── requirements-gpu.txt           # Зависимости для GPU (torch + всё остальное)
├── data/
│   ├── raw/                       # 50 parquet файлов, 65 MB
│   ├── features/                  # crypto_features_1h.parquet, 1.5 GB
│   └── sentiment/                 # Fear&Greed, funding rates, OI, LSR (parquet)
├── results/                       # v1 results (плохие)
├── results_v2/                    # v2 results (хорошие)
├── results_v3/                    # v3 results (cluster run)
├── results_v4/                    # v4 results (HPO + regime + ensemble)
├── results_v5/                    # v5 results (sentiment + risk overlay)
├── results_hist/                  # HIST transformer results
│   ├── all_results.json           # Метрики всех 4 моделей
│   ├── feature_importance_v2.csv  # Важность фичей
│   ├── test_predictions_v2.parquet# Предсказания на тест
│   └── equity_curve_v2.parquet    # Кривая капитала
├── src/
│   ├── data/download_crypto.py    # Загрузка OHLCV данных с Binance
│   ├── data/download_sentiment.py # Загрузка sentiment данных (OKX + Alternative.me)
│   ├── features/build_features.py # Генерация 98 фичей
│   ├── models/baseline_lgbm.py    # LightGBM baseline (v1)
│   └── backtest/simple_backtest.py# Long-only Top-K бэктест
└── venv/                          # Python virtual environment
```

### ✅ Data Download (50 symbols, 65 MB)
- Загружены 1h OHLCV данные с Binance (public API, без ключа)
- 50 пар: BTC, ETH, BNB, SOL, XRP, ADA, DOGE, AVAX, DOT, LINK, MATIC, UNI, ATOM, LTC, ETC, FIL, APT, ARB, OP, NEAR, AAVE, MKR, GRT, INJ, FTM, ALGO, SAND, MANA, AXS, THETA, RUNE, EGLD, XTZ, FLOW, CHZ, CRV, LDO, SNX, COMP, YFI, SUSHI, ENJ, BAT, ZIL, ONE, IOTA, ICX, ENS, IMX, GALA
- Период: с 2021-01-01 по 2026-03-06, ~45K строк на монету
- Формат: parquet, по файлу на символ в `data/raw/`
- Workaround: `exchange.session.verify = False` (SSL проблема из России)

### ✅ Feature Engineering (2.1M rows, 98 features)
- Обработаны все 50 символов
- Сгенерированы 98 фичей (описание ниже)
- Результат: `data/features/crypto_features_1h.parquet` (1.5 GB)
- Shape: 2,102,405 строк × 107 столбцов (98 фичей + meta)
- Period: 2021-01-30 → 2026-03-06
- Target: 4h forward return (`target_ret`), binary classification (`target_cls`)
- Target stats: mean return = 0.000143, up ratio = 49.11%

### ❌ → ✅ Baseline v1 обучена, ПРОВАЛ
- LightGBM обучился за ~30 секунд на M3 Pro
- IC = 0.005, Sharpe = -1.0, Backtest $1K → $6 
- Проблемы: time features leakage (hour_sin/dow_cos доминировали), нет CS normalization
- Early stopping на 51 итерации из 2000 — переобучение

### ✅ Baseline v2 обучена, УСПЕХ
**Скрипт:** `run_pipeline_v2.py`

**Ключевые исправления (v1 → v2):**
1. **Cross-sectional rank normalization** — фичи ранжируются внутри каждого timestamp [0,1] → центрируются [-0.5, 0.5]
2. **Time features удалены** — hour_sin, hour_cos, dow_sin, dow_cos убраны из модели
3. **3 варианта target:** rank (CS rank), excess return (vs mean), raw return
4. **Ensemble** — среднее нормализованных предсказаний 3 моделей
5. **Лучшие гиперпараметры:** LR 0.01 (было 0.05), depth 6 (было 8), min_samples 200 (было 50), L1/L2 = 1.0 (было 0.1)

**Результаты на тесте (2025-07 → 2026-03, 282K rows):**

| Model | Rank IC | ICIR | LS Sharpe | LS Ann Return | Max DD |
|-------|---------|------|-----------|---------------|--------|
| **Rank Model** | **0.031** | **0.36** | **3.87** | 158% | -56% |
| Excess Model | 0.005 | 0.04 | 1.10 | 34% | -61% |
| Raw Model | 0.006 | 0.09 | 2.95 | 87% | -47% |
| **Ensemble** | **0.025** | **0.27** | **4.21** | 161% | -49% |

**⚠️ Long-Only НЕ работает** (Top5 $1K → $18-25). Причина: bear market в test периоде.
**✅ Long-Short РАБОТАЕТ** (Sharpe 3.87-4.21). Модель отличает winners от losers.

**Top 20 фичей (v2):**
```
close_ma24_ratio, close_ma720_ratio, low_close_ratio, close_ma336_ratio,
ret_sharpe_24h, ret_sharpe_168h, gk_vol_168h, close_ma6_ratio, ret_2h,
gk_vol_24h, close_ma12_ratio, cci_48, gk_vol_48h, vol_price_corr_168h,
macd, gk_vol_12h, atr_48, vol_price_corr_48h, bb_width_20, high_close_ratio
```

### ✅ v3 обучена на кластере
**Скрипт:** `run_pipeline_v3.py`

**Новое в v3 vs v2:**
1. Multi-horizon targets (4h, 12h, 24h)
2. Cross-asset features: btc_ret_*, eth_ret_*, btc_vol_24h, market_dispersion, ret_vs_btc_24h
3. BTC regime filter (btc_regime_72)
4. Optuna HPO slot (не запустился — optuna не был установлен)
5. 109 фичей (было 92)

**Результаты на тесте (2025-01 → 2026-03):**

| Horizon | Rank IC | ICIR | Rank ICIR | LS Sharpe | LS Ann Ret | LS MaxDD |
|---------|---------|------|-----------|-----------|------------|----------|
| **4h**  | **0.0287** | **0.337** | **0.579** | **3.82** | 152% | -55.6% |
| 12h     | 0.0302 | 0.262 | 0.451 | 2.01 | 78.5% | -91.5% |
| 24h     | 0.0245 | 0.178 | 0.318 | 0.78 | 31.9% | -99.7% |

**Regime filter НЕ работает:** 49.9% ON = монетка. BTC MA(72) = 3 дня слишком короткая.
**Long-Only = -100%** — катастрофа. Все монеты упали в bear market.

**Top фичи (v3):**
```
close_ma336_ratio, close_ma720_ratio, vol_price_corr_168h, btc_vol_24h,
gk_vol_168h, macd_signal, ret_skew_168h, ret_sharpe_168h, ret_std_168h,
ret_kurt_168h, atr_48, gk_vol_48h, vol_price_corr_48h
```

### ✅ v4 обучена на кластере
**Скрипт:** `run_pipeline_v4.py`

**Результаты (сравнительная таблица):**

| Model | Rank IC | ICIR | Rank ICIR | LS Sharpe | LS Ann Ret | LS MaxDD |
|-------|---------|------|-----------|-----------|------------|----------|
| baseline | 0.0285 | 0.347 | 0.550 | 3.89 | 155% | -57.8% |
| feat_selected (94 feat) | 0.0291 | 0.358 | 0.564 | 3.94 | 158% | -55.9% |
| **ensemble (5 seeds)** | **0.0290** | **0.354** | **0.555** | **4.00** | **159%** | **-56.3%** |

**Regime comparison (Long-Only Top-5):**
| Strategy | Sharpe | Final $ |
|----------|--------|---------|
| No filter | -2.00 | $0.11 |
| v3 regime (BTC 72h MA) | -1.69 | $14.05 |
| v4 regime (composite) | -2.41 | $2.21 |

⚠️ v4 composite regime ХУЖЕ v3 simple regime. Слишком агрессивный фильтр.
⚠️ Optuna HPO skipped — optuna не найден в Python 3.11 на кластере.

**Regime Breakdown:** Full 38.6%, Partial60 7.7%, Partial25 19.2%, Out 34.5%

**Top 10 фичей (v4):**
```
breadth_pct_positive (607), close_ma720_ratio (477), close_ma24_ratio (457),
ret_sharpe_168h (427), close_ma336_ratio (398), ret_sharpe_24h (330),
regime_btc_dd_720 (324), vol_price_corr_168h (323), ret_std_168h (316), btc_vol_24h (307)
```

### ✅ HIST Transformer обучен на H100
**Скрипт:** `run_hist_model.py`

**Архитектура:**
- 502K params, embed(105→128) + concept(8 crypto categories) + cross_attn(2L,4H) + head
- Loss: 0.5×MSE + 0.5×IC_loss
- Best epoch: 9/80, early stop at 24
- Val Rank IC: 0.0708

**Test Results (eval bug fixed — actual returns для P&L):**

| Model | Rank IC | LS Sharpe |
|-------|---------|-----------|
| HIST standalone | 0.067 | 4.25 |
| LGB v4 ensemble | 0.081 | 4.00 |
| **HIST+LGB ensemble** | **0.078** | **4.38** |

✅ HIST Rank IC = 0.067 — cross-stock attention работает.
✅ Ensemble HIST+LGB = Sharpe 4.38 ← лучший результат (GROSS, без комиссий).

---

## 2. Детальное описание фичей (98 шт.)

### Returns (9 фичей)
`ret_1h, ret_2h, ret_4h, ret_6h, ret_12h, ret_24h, ret_48h, ret_72h, ret_168h`

### Price Features (~40 фичей)
- Candle ratios: `close_open_ratio, high_low_ratio, high_close_ratio, low_close_ratio, upper_shadow, lower_shadow, body`
- MA ratios (8 окон × 2): `close_ma{6,12,24,48,72,168,336,720}_ratio, vol_ma{...}_ratio`
- Garman-Klass volatility: `gk_vol_{12,24,48,168}h`
- Rolling stats (3 окна × 5): `ret_std_{24,48,168}h, ret_skew_{...}h, ret_kurt_{...}h, ret_mean_{...}h, ret_sharpe_{...}h`

### Volume Features (~14 фичей)
- Volume momentum: `vol_mom_{6,12,24,48}h`
- VWAP deviation: `vwap_dev_{12,24,48}h`
- Volume-price correlation: `vol_price_corr_{24,48,168}h`
- Buying pressure: `buy_pressure`

### Technical Indicators (~30 фичей)
- RSI: `rsi_{6,12,14,24}`
- MACD: `macd, macd_signal, macd_diff`
- Bollinger Bands: `bb_high_{20,48}, bb_low_{20,48}, bb_width_{20,48}, bb_pband_{20,48}`
- ATR: `atr_{14,24,48}`
- ADX: `adx, adx_pos, adx_neg`
- Stochastic: `stoch_k, stoch_d`
- CCI: `cci_14, cci_48`
- Williams %R: `willr_14`
- OBV: `obv_ma_ratio_{12,24,48}`
- MFI: `mfi_14`

### Time Features (4 фичи)
- Cyclical: `hour_sin, hour_cos, dow_sin, dow_cos`

---

## 3. Модель: Что обучать

### 3.1 Baseline: LightGBM (первый приоритет)

**Скрипт:** `src/models/baseline_lgbm.py`

**Walk-forward split:**
- Train: < 2024-07-01 (3.5 года)
- Val: 2024-07 — 2025-07 (1 год)  
- Test: >= 2025-07 (8 месяцев OOS)

**Гиперпараметры:**
```python
{
    'objective': 'regression',
    'metric': 'mse',
    'n_estimators': 2000,
    'learning_rate': 0.05,
    'max_depth': 8,
    'num_leaves': 63,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 50,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'early_stopping': 50 rounds
}
```

**Метрики (ожидаем):**
- IC > 0.02 — хороший знак для baseline
- ICIR > 0.3
- Direction Accuracy > 51%
- Long-Short Sharpe > 0.5

**Output:**
- `results/baseline_results.json` — метрики
- `results/feature_importance.csv` — важность фичей
- `results/test_predictions.parquet` — предсказания на тест

### 3.2 Backtest

**Скрипт:** `src/backtest/simple_backtest.py`

Запускается после обучения модели, читает `results/test_predictions.parquet`.
- Long-only, Top-5 монет каждые 4 часа
- Commission: 0.1% (OKX taker fee)
- Initial capital: $1,000

---

## 4. Инструкция для кластера

### Что загрузить на S3:
```
data/features/crypto_features_1h.parquet    # 1.5 GB — основные данные
src/models/baseline_lgbm.py                 # скрипт обучения
src/backtest/simple_backtest.py             # скрипт бэктеста
```

### Зависимости:
```bash
pip install pandas numpy lightgbm scikit-learn scipy pyarrow
```

### Запуск:
```bash
# 1. Обучение LightGBM (1-3 минуты)
python baseline_lgbm.py

# 2. Бэктест (секунды)
python simple_backtest.py
```

### Ожидаемый результат:
- `results/baseline_results.json` — основные метрики (IC, Sharpe, Drawdown)
- `results/feature_importance.csv` — топ фичей
- `results/test_predictions.parquet` — предсказания
- `results/backtest_results.json` — результаты симуляции торговли
- `results/equity_curve.parquet` — кривая капитала

---

## 5. Следующие шаги (приоритезированы)

### Immediate:
1. **Запустить v5 full** на кластере — 3 walk-forward окна + Optuna HPO (50 trials)
2. **Запустить HIST v2** на GPU — sentiment-aware architecture (80 epochs, H100)
3. **Ensemble v5 LGB + HIST v2** — ожидаем Sharpe net > 2.0
4. **Исследовать 12h/24h rebalance** — меньше комиссий, возможно лучший net Sharpe

### Phase 5: Production:
5. **Paper trading** на OKX sandbox (run_paper_trading.py)
6. **Live trading** с $100 deposit — maker orders для 0.02% комиссий
7. **Streaming pipeline** — real-time data + inference каждые 4h

---

## 6. Ключевые решения и контекст

- **Почему крипто, а не акции:** Доступно из РФ, торговля 24/7, старт с $100, бесплатные данные
- **Почему OKX:** Работает для резидентов РФ (в отличие от Binance). Bybit — запасная.
- **Почему 1h:** На минутках побеждает HFT (latency). На daily — мало данных. 1h — оптимальный баланс.
- **Почему 4h target:** Forward return на 4 часа — достаточно для значимого сигнала при 1h данных
- **Почему 50 монет:** Top по ликвидности на Binance, достаточно для cross-sectional ranking
- **SSL workaround:** В России SSL проблема с Binance — отключаем verify в CCXT
- **libomp на macOS:** LightGBM не работает без OpenMP. Fix: `brew install libomp` (сделано).
- **v1 vs v2:** v1 провалился из-за time feature leakage и отсутствия CS normalization. v2 с cross-sectional ranks дал Rank IC 0.031 и LS Sharpe 3.87.
- **Long-only vs Long-short:** Модель хорошо РАНЖИРУЕТ (LS Sharpe 4.2), но long-only не работает в bear market. Нужен short-side для реальной прибыли.
- **Rank Model лучше всех:** Предсказание cross-sectional rank (а не raw return) даёт самый стабильный сигнал (ICIR 0.36).

---

## 7. Technical Notes

### Data Schema (crypto_features_1h.parquet)
```
Columns: timestamp, open, high, low, close, volume, 
         [98 feature columns], 
         target_ret, target_cls, symbol, hour, day_of_week
Rows: 2,102,405
Symbols: 50
Period: 2021-01-30 to 2026-03-06
```

### Feature exclusion list (NOT used as model input)
```python
exclude = {'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume',
           'target_ret', 'target_cls', 'hour', 'day_of_week'}
```

### Walk-forward dates
```python
TRAIN_END = '2024-07-01'  # Train: all before this
VAL_END = '2025-07-01'    # Val: TRAIN_END to VAL_END
                           # Test: after VAL_END
```

---

## Phase 5 — Production System (в работе)

### Что сделано

1. **`run_risk_study.py`** — 4-фазная оптимизация риск-параметров:
   - Phase 1: Portfolio construction (top/bot K sweep)
   - Phase 2: Vol target × Kelly fraction grid
   - Phase 3: DD stop / resume threshold sweep
   - Phase 4: Confidence filter threshold
   - Оптимизирует по **Calmar ratio** (return / max drawdown)
   - Сохраняет `optimal_config.json` для загрузки live-системой

2. **`run_trading.py`** — полный production trading pipeline:
   - Fetch live OHLCV (Binance spot API, 800h history)
   - Feature engineering (100+ фичей, тот же пайплайн что и при обучении)
   - Cross-sectional rank normalization
   - Signal generation: LGB v5 multi-seed ensemble (загружает сохранённые .txt модели)
   - Risk overlay: vol targeting, Kelly fraction, DD stop/resume, confidence filter
   - Execution через OKX CCXT (isolated margin, 1x leverage)
   - Три режима: `signal` (только сигналы), `paper` (OKX demo), `live`
   - `--loop` режим: автоматический цикл каждые 4h
   - State persistence: отслеживает equity, peak, drawdown между циклами
   - Logging: JSON логи каждого trade cycle

3. **Model saving в LGB v5** (`run_pipeline_v5.py`):
   - `train_multi_seed()` теперь возвращает все модели (не только последнюю)
   - Сохраняет `lgb_model_seed_{seed}.txt` для каждого seed
   - Сохраняет `feature_names.json` со списком отобранных фичей
   - Backward-compatible: feature importance по-прежнему использует last_model

### Критическая проблема: DDStop Sharpe

| Ensemble | LS raw | LS net | MaxDD | DDStop Sharpe |
|----------|--------|--------|-------|---------------|
| HIST v1 + LGB v5 | 4.42 | 2.93 | -56.5% | -0.19 |
| HIST v1 standalone | 4.20 | 2.57 | -51.1% | +0.12 |
| LGB v5 standalone | 4.20 | 2.78 | -59.0% | -0.08 |

Стратегия зарабатывает, но через болезненные просадки (-56%).
`run_risk_study.py` должен найти параметры, при которых Calmar > 1.0.

### Roadmap
1. Переобучить LGB v5 на кластере (нужно для сохранения моделей)
2. Запустить `run_risk_study.py` на ensemble predictions
3. Paper trading на OKX demo (минимум 2 недели)
4. Live с $100-500 при DDStop Sharpe > 1.0

---

## LIVE Trading — Production (30 марта 2026)

**Модель:** Ridge α=1000, 14 CS-IC features, 12h horizon  
**VPS:** root@185.42.163.63 (SOCKS5 proxy)  
**Exchange:** OKX LIVE (NOT demo), isolated margin  
**Capital:** $100, leverage 3x  
**Service:** systemd `run_trading.py --mode live --loop --capital 100 --leverage 3 --no-deriv-gate --no-meta --ridge --vol-size --min-zscore 0.8`  
**Dashboard:** invest.arturt.com

### R7 Deployed Stack
- Ridge regression α=1000, 14 CS-IC features, 12h horizon
- Walk-forward: 3 windows, 15-day gaps, HPO alpha on val
- BTC regime filter (trend_strength cutoff=0.8, dynamic threshold=0.5)
- Regime-conditional asymmetry (bull → 7L/2S, bear → 5L/4S)
- Vol scaling (exposure ∝ 1/vol)
- Signal EMA(2) smoothing
- EQ-MOM Boost + Kelly sizing + Strategy Momentum 48h
- Asymmetric base: 6L/3S

### Позиции (30 марта 2026)
6 open: ATOM long, CHZ short, COMP long, EGLD long, ENS long, IMX long  
(FTM-USDT-SWAP и MKR-USDT-SWAP не существуют на OKX → 2 шорта не открылись)

---

## R8 Research — New Feature Discovery (31 марта 2026)

**Скрипт:** `_research_round8.py`  
**Результат:** TOP-3 combo (17 features) = Sharpe 3.97, Equity $223, Worst month -1.9%

Новые фичи для модели: `range_24h`, `btc_beta_168h`, `global_ls_ratio_zscore`

IC scan выявил 21 фичу с |IC| > 0.015 из 79 кандидатов. DVOL и макро-фичи не прошли порог.
Ablation подтвердил: добавление >3 фич ухудшает модель (Ridge предпочитает меньше orthogonal features).

~~**Pending:** переобучить модель на 17 фичах и задеплоить на VPS.~~ **ОТМЕНЕНО**: R8 < R7 при прямом сравнении.

---

## R9 + R9B Research — Targeted Improvements (31 марта 2026)

**Скрипты:** `_research_round9.py`, `_research_round9b.py`  
**Итог:** найдено 2 key-finding + 1 quick-win

### Quick Win (Ready to Deploy)
- **`pred_shrinkage=0.05`**: Eq=-6 (-0.2%), Sh +0.01, Worst month -5.6% → -6.4% (улучшение на 0.8pp). Добавить в VPS args.

### Key Finding: LGB EMA=None is Superior
- LightGBM без EMA-сглаживания: Sh=4.21 (+0.62!), Eq=$2916 (-2.6%), Wr=-5.6%, WM=11/13 vs 9/13
- IC теста: 0.060-0.072 vs Ridge 0.013-0.020 (3-4× лучше)
- Вывод: Ridge нуждается в EMA(2) для подавления шума; LGB уже имеет качественный сигнал
- Deploy: требует R10 validation (hyperparameter tuning, out-of-sample robustness check)

### Отвергнутые идеи
- Multi-horizon блендинг: все комбинации хуже 12h. 12h оптимален.
- Position counts: 6L/3S текущий оптимален по equity
- EMA=3: лучший worst month (-4.2%), но -$222 equity и -0.05 Sharpe
- Rebalancing sweep: тест невалиден (overlapping fwd_ret)
- vol_target param: баг — параметр не подключён в sim

### Next: R10 — LightGBM Hyperparameter Tuning
- num_leaves sweep: 15/31/63/127
- n_estimators + early stopping
- Feature importance analysis  
- Ridge+LGB ensemble test
- VPS deployment если результаты подтвердятся
