# Research Execution Log

Started: 2026-04-05

This file is the durable night-run log for plan execution. Every completed research step should leave:
- what was run
- where raw outputs were saved
- the concise result
- the next action

## Status

| Step | Status | Notes |
|------|--------|-------|
| R34.1-R34.5 | completed | `results_r34.log` plus 3 CSV artifacts; trace cross-check matches canonical sim closely |
| R39.1 | completed | inspection saved to `results_r39.log` |
| R39 stablecoin pilot | completed | `stable_flow4` fixes W2 strongly but hurts W3/ALL; best interpreted as regime signal, not direct champion add-on |
| Market-level feature support | completed | `train_ensemble()` now supports `cs_rank_exclude` so regime features are not zeroed by CS ranking |
| R35 | completed | `r35a_cs_second_order` was the best durable bundle; interaction bundle strongest only in W3 |
| R36 | partial | gated blends help W2 sharply, but standalone `expert_stability` remains best on `ALL` |
| R37 | partial | liquidity filter and edge gate reduce turnover/cost; explicit `R37.5` not isolated separately |
| R38 | completed | current binary target remains best overall; threshold/excess/decay variants underperform |
| R41 | completed | canonical consolidation matrix run; no combo beat `A_28f` on `ALL`, while `A_28f+r35a` stayed best on `W2` and `A_28f+liq70` led `W3` |
| R42 | completed | ablation found a smaller `R35a` subset that beats the canonical `ALL` baseline: `A_28f + ret_dispersion_12h + cs_rank_ma_5` |
| R43 | completed | dynamic exposure rules improved `W3` in places but none beat the static `R42` baseline on `ALL` |
| R44 | completed | dynamic universe / quality filters reduced costs but none beat the static `R42` baseline on `ALL` |
| R45 | completed | corrected rerun fixed missing `cs_rank_exclude` for `ret_dispersion_12h`; soft gates still did not beat standalone `expert_stability` on `ALL` |
| R46 | completed | separate long/short models reduced turnover/costs but did not beat the unified `R42` candidate on `ALL` |
| D1 source search | blocked | no CoinGlass API key; Binance liquidation snapshot substitute returned `404` on tested URLs |
| D6 orderbook bootstrap | partial | hourly depth collector, feature builder, deploy cron entry, bootstrap snapshots, and a local unattended daemon are running; only history accumulation / IC / WF remains |
| D5 public stablecoins | completed | `stablecoins.llama.fi` downloader finished and saved 4 parquet artifacts |
| D7.1 dead-data inventory | completed | `google_trends.parquet` and `cc_social_daily.parquet` both exist locally, but there are no Python code references to either dataset |
| D7.2 social/search scan | completed | conservative `+1 day` merge produced usable ICs, strongest on Reddit/social features, but only partial universe coverage |

## Run Log

### 2026-04-05 — Session Start

- Created durable execution log.
- First implementation target: `R34` infrastructure and `R39.1` dead-data inspection.
- Planned output files:
  - `results_r34.log`
  - `results_r34_w2_conditional_ic.csv`
  - `results_r34_w2_coin_contrib.csv`
  - `results_r39.log`

### 2026-04-05 — Environment Repair

- Configured Python environment: `pyenv 3.10.14`.
- Initial runs failed because the selected environment did not contain `pandas`, `pyarrow`, `requests`, `xgboost`, `catboost`.
- `pip` was also blocked by SSL verification on `pypi.org`; install worked only with `--trusted-host pypi.org --trusted-host files.pythonhosted.org`.
- Installed missing packages successfully into the configured environment.

### 2026-04-05 — R39.1 Dead-Data Inspection

- Command: `python _research_r39_dead_data.py > results_r39.log`
- Output file: `results_r39.log`
- Findings:
  - `stablecoin_supply.parquet`: 2285 rows, 16 cols, daily, full coverage, strongest immediate candidate for regime features.
  - `defi_tvl_daily.parquet`: 39978 rows, 8 cols, 20 symbols, daily, but ~50% nulls because chain/protocol coverage is sparse by symbol.
  - `onchain_daily.parquet`: 20780 rows, 21 cols, 10 symbols only, exchange flow fields mostly BTC/ETH-only and ~78% null overall.
- Immediate conclusion:
  - next cheap test = stablecoin market-level features;
  - on-chain is usable only as a partial-universe / regime source;
  - DeFi TVL is more likely breadth/regime than direct per-coin alpha.

### 2026-04-05 — R34 Launch

- Command: `PYTHONUNBUFFERED=1 python -u _research_r34_w2_attribution.py --feature-set baseline > results_r34.log 2>&1`
- Status: completed
- Expected artifacts:
  - `results_r34.log`
  - `results_r34_w2_conditional_ic.csv`
  - `results_r34_w2_coin_contrib.csv`

### 2026-04-05 — R34 Completed

- Artifacts confirmed:
  - `results_r34.log`
  - `results_r34_w2_conditional_ic.csv`
  - `results_r34_w2_coin_contrib.csv`
  - `results_r34_feature_importance.csv`
- Canonical simulator cross-check:
  - `W1`: official `-0.69` vs trace `-0.651`
  - `W2`: official `-0.98` vs trace `-1.012`
  - `W3`: official `2.88` vs trace `2.877`
  - Conclusion: attribution trace is close enough to trust for diagnosis.
- Main findings:
  - `W2` break is led by a weak long leg: long-leg Sharpe `-1.71` vs short-leg Sharpe `+0.38`.
  - Rank stability is not the main failure on the long side, but short-book churn is materially worse in `W2` than `W3` (`0.664` vs `0.577`).
  - Worst `W2` gross contributors: `XTZ` short, `UNI` short, `OP` short, `XRP` long, `ADA` long.
  - Best `W2` gross contributors: `LDO` short, `INJ` short, `SNX` short, `SOL` long.
  - Regime-conditioned IC still favors `dist_from_high_24h` and `iv_rv_spread`, especially in low-liquidity / mid-dispersion bins.
  - Per-window importance exposes a mismatch in `W2`: high-gain features like `cum_funding_24h`, `mom_z_24h`, `ret_24h`, `gk_vol_24h`, `ret_168h` all show negative post-hoc `W2` test IC, while `iv_rv_spread` remains positive.
- Immediate interpretation:
  - The baseline is over-allocating to unstable market/regime features in `W2` while one of the few robust signals is still vol-relative (`iv_rv_spread`).
  - The cheapest next intervention is to test a new market-level regime source from dead data rather than add more cross-sectional complexity.

### 2026-04-05 — R39 Stablecoin Pilot Completed

- Command: `PYTHONUNBUFFERED=1 python -u _research_r39_stablecoin_features.py > results_r39_stablecoin.log 2>&1`
- Artifacts confirmed:
  - `results_r39_stablecoin.log`
  - `results_r39_stablecoin_summary.csv`
- Implementation note:
  - While wiring this branch, a pipeline issue surfaced: market-level features were being cross-sectionally ranked and therefore collapsed to zero within each timestamp.
  - Fixed in `_research_r30b_fixed.py` by adding `cs_rank_exclude` to `train_ensemble()`.
- Results:
  - `stable_flow4` (`stable_total_supply_chg7d`, `stable_total_supply_chg30d`, `stable_supply_accel`, `stable_usdt_vs_usdc_chg7d`) changed `W2` net Sharpe from `-0.98` to `+2.46`.
  - But the same bundle weakened `W3` from `2.88` to `1.78` and lowered `ALL` from `0.47` to `0.17`.
  - `stable_regime6` was even less stable: `W2=2.27`, `W3=0.14`, `ALL=-0.72`.
- Interpretation:
  - Stablecoin flow data looks real and powerful, but only conditionally.
  - This is stronger evidence for `R36` regime gating / expert blending than for directly appending the features to the base model.
- Next action:
  - Run `R35`, `R37`, `R38` scripts overnight and then revisit `R36` using the stablecoin signal as a gate candidate.

### 2026-04-05 — Overnight Batch Expanded

- Added new research scripts:
  - `_research_r35_new_features.py`
  - `_research_r36_regime_gating.py`
  - `_research_r37_cost_aware.py`
  - `_research_r38_targets.py`
- Added `run_research_plan_batch.sh` to execute `R35→R38` sequentially with durable logs.
- Important runtime fix:
  - the first batch attempt picked an older shell `python` and failed immediately on `R35` type-hint syntax.
  - fixed by pinning `/Users/a.s.tabakov/.pyenv/versions/3.10.14/bin/python` in the batch script.
- Current batch state:
  - `results_r35.log` confirms `R35` is running past data load and into train-only IC scan.

### 2026-04-05 — External Data Source Check

- `D1 / CoinGlass`:
  - existing `run_coinglass.sh probe` was executed.
  - result: no `COINGLASS_API_KEY` is configured, so paid CoinGlass download cannot start automatically.
- `D1 / Binance liquidation snapshots`:
  - public `data.binance.vision` liquidation snapshot URLs were probed directly for representative BTC dates in 2024–2026.
  - result: tested URLs returned `404`; this path is not currently usable as a historical substitute.
- `D5 / DefiLlama stablecoins`:
  - public endpoints were confirmed on `https://stablecoins.llama.fi`, while `api.llama.fi` stablecoin routes returned `404`.
  - added `src/data/download_defillama_stablecoins.py` to fetch asset snapshot, chain snapshot, global history, and per-chain history.
  - download launched with durable log `results_d5_defillama_stablecoins.log`.

### 2026-04-05 — R35 Completed

- Artifacts:
  - `results_r35.log`
  - `results_r35_feature_ic.csv`
  - `results_r35_summary.csv`
- Main scan findings:
  - strongest `TRAIN` signals came from `ret_dispersion_12h`, `mkt_funding_dispersion`, `mkt_oi_extreme_pct`, `ret_168h_x_disp`, `cum_funding_24h_cs`, `oi_chg_12h_cs`.
- Main WF findings:
  - `r35a_cs_second_order` was the best durable bundle: `W2=3.03`, `W3=1.91`, `ALL=0.64`.
  - `r35b_interactions` was strongest in `W3` (`3.34`) but collapsed on `ALL` (`-0.00`).
  - `r35c_temporal` was decent but weaker than `r35a` on `ALL` (`0.62`).
  - `r35d_market` improved turnover/costs but was not durable on `ALL` (`-0.13`).
- Interpretation:
  - `R35a` is the cleanest next promotion candidate.
  - interactions matter, but mostly as late-regime / `W3` helpers rather than universal additions.

### 2026-04-05 — R36 Pilot Completed

- Artifacts:
  - `results_r36.log`
  - `results_r36_gating_summary.csv`
- Experts tested:
  - `expert_base = FEAT_28`
  - `expert_stability = FEAT_30`
  - `expert_stable_flow = FEAT_28 + stable_flow4`
- Results:
  - best `ALL`: `expert_stability` with `ALL=0.92`, `W3=2.84`.
  - best `W2`: `gate_stable_base_vs_flow` with `W2=3.72`, but `ALL=0.29`.
  - `gate_tri_regime` improved `W2` (`3.17`) and kept `W3` healthy (`2.24`), but still did not beat `expert_stability` on `ALL`.
- Interpretation:
  - regime gating is real, especially for repairing `W2`.
  - the current hard gates are still too brittle to beat the more stable 30-feature expert on full history.
  - next R36 step should be proper calibration on `W1+W2 train` and/or feature scaling by regime, not more ad hoc thresholds.

### 2026-04-05 — R37 Completed (1-4)

- Artifacts:
  - `results_r37.log`
  - `results_r37_execution_summary.csv`
- Findings:
  - `liq70` sharply reduced turnover and costs (`W2 cost=3.86%`, `W3 cost=4.05%`) and improved `W2/W3` to `1.47 / 2.87`.
  - `edge_x1` and `edge_x2` materially reduced turnover and cost drag.
  - best `ALL` inside this execution sweep remained the baseline execution path (`0.74`), so lower costs alone did not dominate full-history performance.
- Caveat:
  - `R37` used a custom execution sweep simulator for relative comparisons, so absolute numbers here should be compared within-script rather than against `R34` diagnostics one-to-one.

### 2026-04-05 — R38 Completed

- Artifacts:
  - `results_r38.log`
  - `results_r38_target_summary.csv`
- Findings:
  - current binary baseline remained best on `ALL` (`0.71`).
  - threshold targets `P(ret > 0.5%-2.0%)` were broadly destructive, especially in `W2`.
  - `excess_vs_btc` also underperformed on `ALL` (`-0.56`).
  - temporal decay (`90d`, `180d`) improved `W2` somewhat, but hurt `ALL` and never beat the baseline target overall.
- Interpretation:
  - `R38` re-confirmed the older conclusion: target engineering is not the main path to alpha here.

### 2026-04-05 — D5 Public Download Completed

- Artifacts:
  - `results_d5_defillama_stablecoins.log`
  - `data/sentiment/llama_stablecoins_assets.parquet`
  - `data/sentiment/llama_stablecoin_chains.parquet`
  - `data/sentiment/llama_stablecoin_chart_all.parquet`
  - `data/sentiment/llama_stablecoin_chart_by_chain.parquet`
- Download result:
  - asset snapshot: `360` rows
  - chain snapshot: `175` rows
  - global history: `3,049` rows
  - per-chain history: `158,318` rows
  - chain fetch failures: `0`
- Interpretation:
  - `D5` is now materially unblocked; the raw data exists locally and can feed feature engineering without any paid key.

### 2026-04-05 — R41 Consolidation Completed

- Command: `PYTHONUNBUFFERED=1 /Users/a.s.tabakov/.pyenv/versions/3.10.14/bin/python _research_r41_consolidation.py > results_r41.log 2>&1`
- Artifacts:
  - `results_r41.log`
  - `results_r41_summary.csv`
- Main matrix results under the canonical `6L3S_ema05_h3` execution setup:
  - best `ALL`: `A_28f` with `0.74`
  - best `W2`: `A_28f+r35a` with `3.03`
  - best `W3`: `A_28f+liq70` with `2.81`
  - best simple blend on `ALL`: `blend_r35a_d30f` with `0.70` (close, but still below baseline)
- Key takeaways:
  - the known winners do not stack linearly under one honest simulator.
  - `R35a` remains the strongest repair lever for `W2`, but still weakens full-history stability.
  - `liq70` cuts costs/turnover materially and improves `W2/W3`, but drags `W1` enough to make `ALL` worse.
  - naive `D_30f` blending does not beat the base `A_28f` champion on `ALL`.
- Runtime note:
  - the redirected run produced complete artifacts and reached the `Saved artifacts` block in `results_r41.log`; terminal status later surfaced as `130`, but the output files are complete and usable.
- Next action:
  - run `R42` ablation inside the `R35a` bundle to isolate the minimal `W2` repair subset that preserves `W3/ALL`.

### 2026-04-05 — D7.1 Dead-Data Inventory Completed

- Files confirmed:
  - `data/features/google_trends.parquet`
  - `data/features/cc_social_daily.parquet`
- Coverage snapshot:
  - `google_trends.parquet`: `760 x 5`, `2020-01-01 -> 2026-04-01`, `10` symbols
  - `cc_social_daily.parquet`: `23,413 x 13`, `2020-10-12 -> 2026-04-04`, `17` symbols
- Code search result:
  - no Python references to `google_trends`, `cc_social`, `google_trends.parquet`, or `cc_social_daily.parquet` were found in the repo.
- Interpretation:
  - these are currently dead data sources: present on disk, absent from the active feature/research pipeline.
  - they are not yet proven useless; they are simply not wired into any experiment.
- Next action:
  - after `R42`, do `D7.2`: inspect per-symbol coverage/mergeability and build a minimal feature scan instead of leaving them as dormant files.

### 2026-04-05 — R42 Ablation Completed

- Command: `PYTHONUNBUFFERED=1 /Users/a.s.tabakov/.pyenv/versions/3.10.14/bin/python _research_r42_ablation.py > results_r42.log 2>&1`
- Artifacts:
  - `results_r42.log`
  - `results_r42_summary.csv`
- Main result:
  - best pair and best overall strict passer = `A_28f + ret_dispersion_12h + cs_rank_ma_5`
  - metrics: `W2=3.22`, `W3=2.50`, `ALL=1.13`, `ALL cost=19.22%`
- Important comparisons:
  - `A_28f + ret_dispersion_12h + cum_funding_24h_cs` also passed strictly with `ALL=0.98`, but remained below the `dispersion+rankma` pair.
  - several larger triples improved `W2/W3` further (`dispersion+rankma+taker_cs` reached `W2=3.74`, `W3=3.03`), but they gave back too much on `ALL`.
  - the full `R35a` bundle re-confirmed the older profile: `W2=3.03`, `W3=1.91`, `ALL=0.64`.
- Interpretation:
  - the `R35a` alpha is not coming from the whole 5-feature bundle; it is concentrated in a smaller subset.
  - `ret_dispersion_12h` looks like the core W2 repair lever, but only becomes durable on `ALL` when paired with `cs_rank_ma_5`.
  - adding more CS-derivative features often boosts `W2/W3` but tends to overfit and reduce full-history stability.
- Next action:
  - run `R43` dynamic exposure using the new `R42` candidate subset (`A_28f + dispersion + rankma`) as the base feature set.

### 2026-04-05 — R43 Dynamic Exposure Launched

- Command: `PYTHONUNBUFFERED=1 /Users/a.s.tabakov/.pyenv/versions/3.10.14/bin/python _research_r43_dynamic_exposure.py --feature-set r42_candidate > results_r43.log 2>&1`
- Current state:
  - log file created: `results_r43.log`
  - startup verified through data-load stage
- Goal:
  - test whether dynamic `n_long/n_short` rules can keep the new `R42` subset's `ALL` improvement while reducing the W2 long-leg failure mode diagnosed in `R34`.

### 2026-04-05 — R43 Dynamic Exposure Completed

- Artifacts:
  - `results_r43.log`
  - `results_r43_dynamic_exposure_summary.csv`
- Base feature set:
  - `R42` winner = `A_28f + ret_dispersion_12h + cs_rank_ma_5`
- Results:
  - baseline/static book remained best: `W2=3.22`, `W3=2.50`, `ALL=1.13`
  - `breadth_5L4S` improved `W3` to `3.60`, but reduced `ALL` to `0.87`
  - `dispersion_4L4S` also landed at `ALL=0.87`
  - more aggressive rules (`breadth_4L4S`, `stress_4L5S`, `breadth_or_disp_4L4S`) degraded `ALL` further
- Interpretation:
  - the current alpha appears more sensitive to symbol selection than to simple gross/net long rebalancing rules.
  - `R34` correctly identified the long-leg issue, but these coarse dynamic exposure heuristics are too blunt and give back too much outside `W2`.
- Next action:
  - move to `R44` dynamic universe / coin quality filtering on top of the new `R42` subset.

### 2026-04-05 — R44 Dynamic Universe Launched

- Command: `PYTHONUNBUFFERED=1 /Users/a.s.tabakov/.pyenv/versions/3.10.14/bin/python _research_r44_dynamic_universe.py --feature-set r42_candidate > results_r44.log 2>&1`
- Current state:
  - log file created: `results_r44.log`
  - startup verified through data-load stage
- Goal:
  - test whether rolling liquidity/OI filters and validation-derived long-quality filters can remove toxic names more effectively than the failed `R43` exposure heuristics.

### 2026-04-05 — R44 Dynamic Universe Completed

- Artifacts:
  - `results_r44.log`
  - `results_r44_dynamic_universe_summary.csv`
  - `results_r44_quality_filter.csv`
  - `results_r44_toxic_coin_check.csv`
- Results:
  - baseline/static `R42` subset remained best: `W2=3.22`, `W3=2.50`, `ALL=1.13`
  - `quality_long_only` reduced costs/turnover materially (`ALL cost=16.01%`, `turn=3.5`) but still fell to `ALL=0.82`
  - `liq30_combo35` also cut costs (`15.52%`) but dropped to `ALL=0.80`
  - combined filtering (`liq30_plus_quality`) over-pruned the universe and weakened both `W2` and `W3`
- Toxic-name check:
  - the validation-derived quality filter caught some known bad names (`ADA` in `W2`, `XRP/SAND/APT` in `W3`) but not enough of the full `W2` toxic-long set.
- Interpretation:
  - cheaper execution and cleaner universe alone are not enough; the static `R42` signal remains stronger than these pruning rules.
  - next step should move from hard pruning to calibrated expert blending (`R45`) rather than more blunt filters.

### 2026-04-05 — R45 Calibrated Soft Gate Launched

- Command: `PYTHONUNBUFFERED=1 /Users/a.s.tabakov/.pyenv/versions/3.10.14/bin/python _research_r45_soft_gate.py > results_r45.log 2>&1`
- Current state:
  - log file created: `results_r45.log`
  - startup verified through data-load stage
- Goal:
  - test whether validation-calibrated soft blending between the new `R42` winner and regime experts can preserve `ALL` while borrowing conditional uplift from `R36`-style regime behavior.

### 2026-04-05 — D7.2 Social/Search Scan Completed

- Command: `PYTHONUNBUFFERED=1 /Users/a.s.tabakov/.pyenv/versions/3.10.14/bin/python _research_d7_social_search_scan.py > results_d7.log 2>&1`
- Artifacts:
  - `results_d7.log`
  - `results_d7_social_search_ic.csv`
- Coverage / overlap:
  - conservative `+1 day` shift before hourly forward-fill
  - overlap universe = `17` active symbols for social data, `10` symbols for Google Trends
  - effective merged coverage after lagging is mostly `~47-57%` for social features and `~49-54%` for the main Trends fields
- Strongest train-only ICs were consistent across windows:
  - `reddit_subscribers`: `W1=0.0417`, `W2=0.0492`, `W3=0.0487`
  - `social_reddit_activity`: `W1=0.0346`, `W2=0.0401`, `W3=0.0399`
  - `reddit_active_users`: `W2=0.0383`, `W3=0.0379`
  - `gtrends`: `W1=0.0384`, `W2=0.0352`, `W3=0.0335`
- Interpretation:
  - this data is not dead; it contains real signal after a conservative lag-aware merge.
  - the main limitation is coverage, not raw IC: social features span only `17/35` active symbols and Trends only `10/35`.
  - this looks promising as a partial-universe side branch or additive overlay, but it is not yet stronger than the main `R45/R46` alpha path.
- Next action:
  - keep `D7` as a secondary branch; after the main architecture tests, promote the best 1-2 social features into a bounded WF experiment rather than a full-universe replacement.

### 2026-04-05 — R46 Separate Long/Short Script Prepared

- File created: `_research_r46_asymmetric.py`
- Scope implemented:
  - unified baseline on the same feature set for head-to-head comparison
  - `model_long`: `P(fwd_ret_12h > cs_median)`
  - `model_short`: `P(fwd_ret_12h < cs_p25)`
  - dual-book simulator with separate long/short ranks, EMA smoothing, hysteresis, and canonical cost accounting
  - feature-importance export for long-vs-short asymmetry analysis
- Validation completed:
  - `py_compile` passed
  - import smoke test passed
  - editor diagnostics: no errors
- Planned artifacts:
  - `results_r46.log`
  - `results_r46_asymmetric_summary.csv`
  - `results_r46_feature_importance.csv`
- Runtime decision:
  - heavy launch is intentionally deferred until `R45` completes, to respect the sequential-run constraint on `M3 Pro / 18 GB`.

### 2026-04-05 — R45 Calibrated Soft Gate Completed (Corrected Rerun)

- Bug found after the first run:
  - `expert_r42` inside `_research_r45_soft_gate.py` was missing `cs_rank_exclude` for market-level `ret_dispersion_12h`, so the in-script comparison was not canonical.
- Fix applied:
  - imported `MARKET_LEVEL_FEATURES`
  - passed `cs_rank_exclude=R42_NO_RANK` to both `expert_r42` and `expert_r42_val`
- Safety backup:
  - pre-fix artifacts were copied to `results_r45_prefix.log`, `results_r45_soft_gate_summary_prefix.csv`, `results_r45_soft_gate_calibration_prefix.csv`
- Corrected command:
  - `PYTHONUNBUFFERED=1 /Users/a.s.tabakov/.pyenv/versions/3.10.14/bin/python _research_r45_soft_gate.py > results_r45.log 2>&1`
- Corrected artifacts:
  - `results_r45.log`
  - `results_r45_soft_gate_summary.csv`
  - `results_r45_soft_gate_calibration.csv`
- Corrected results:
  - best `W2`: `expert_r42` with `3.02`
  - best `W3`: `expert_r42` with `3.42`
  - best `ALL`: `expert_stability` with `0.92`
  - best soft blend on `ALL`: `soft_tri` with `0.61`
- Interpretation:
  - even after fixing the market-level feature handling, soft gating still does not produce a new full-history winner.
  - the blends help conditionally, but the best `ALL` inside this script remains standalone `expert_stability`.
  - there is still residual mismatch between this script's `expert_r42` and the standalone `R42` ablation baseline, so the safest takeaway is directional: `R45` did not beat the existing durable leaders.
- Next action:
  - launch `R46` asymmetric long/short models as the next heavy architecture test.

### 2026-04-05 — D6 Orderbook Bootstrap Completed (Initial Infra)

- Implemented files:
  - `src/data/download_binance_depth.py`
  - `src/features/build_orderbook_depth_features.py`
  - `deploy/update_orderbook_depth.sh`
  - updated `deploy/crontab.txt`
- Runtime notes:
  - default `api.binance.com` depth endpoint is geo-restricted in the current environment.
  - switched collector to `data-api.binance.vision/api/v3/depth` with `verify=False`, which works here.
- Bootstrap command:
  - `PYTHONUNBUFFERED=1 /Users/a.s.tabakov/.pyenv/versions/3.10.14/bin/python src/data/download_binance_depth.py > results_d6_bootstrap.log 2>&1 && PYTHONUNBUFFERED=1 /Users/a.s.tabakov/.pyenv/versions/3.10.14/bin/python src/features/build_orderbook_depth_features.py >> results_d6_bootstrap.log 2>&1`
- Bootstrap artifacts:
  - `results_d6_bootstrap.log`
  - `data/raw/orderbook_depth/binance_orderbook_depth_snapshots.parquet`
  - `data/features/binance_orderbook_depth_features.parquet`
- Bootstrap outcome:
  - initial full-universe snapshot saved: `50` symbols, `50` rows
  - first feature parquet built: `50` rows, hourly timestamp `2026-04-05 11:00:00+00:00`
  - D6.1 and D6.2 are now implemented and bootstrapped; only `D6.3` remains time-blocked because it needs weeks of accumulation.

### 2026-04-05 — R46 Asymmetric Long/Short Run Launched

- Command:
  - `PYTHONUNBUFFERED=1 /Users/a.s.tabakov/.pyenv/versions/3.10.14/bin/python _research_r46_asymmetric.py > results_r46.log 2>&1`
- Expected artifacts:
  - `results_r46.log`
  - `results_r46_asymmetric_summary.csv`
  - `results_r46_feature_importance.csv`
- Goal:
  - test whether separate long/short models can beat the unified ranker under the same cost-aware execution framework.

### 2026-04-05 — R46 Asymmetric Long/Short Run Completed

- Artifacts:
  - `results_r46.log`
  - `results_r46_asymmetric_summary.csv`
  - `results_r46_feature_importance.csv`
- Main comparison (`r42_candidate` feature set):
  - unified baseline: `W2=3.22`, `W3=2.50`, `ALL=1.13`, `ALL cost=19.22%`, `turn=4.5`
  - asymmetric long/short: `W2=1.80`, `W3=2.08`, `ALL=0.54`, `ALL cost=12.15%`, `turn=2.4`
- Interpretation:
  - asymmetric modeling cuts turnover and costs materially, but the alpha loss is too large.
  - the unified `R42` candidate remains clearly superior on both `W2` and `ALL`.
  - this architecture is not the next durable winner.
- Feature-importance asymmetry notes:
  - long-skewed features were led by `cum_funding_24h`, `pct_coins_up_12h`, `mom_z_24h`, `rel_volume_cs`, `ret_skew_168h`
  - short-skew importance concentrated much more heavily in `atr_14`, `rvol_24h`, and `ret_12h`
- Next action:
  - no further heavy architecture experiment remains in the active `R41+` plan; only blocked or time-bound data branches remain.

### 2026-04-05 — D6 Local Collector Daemon Launched

- Command:
  - `PYTHONUNBUFFERED=1 /Users/a.s.tabakov/.pyenv/versions/3.10.14/bin/python src/data/run_orderbook_depth_daemon.py > results_d6_daemon.log 2>&1`
- Runtime state at launch:
  - daemon started cleanly and is sleeping until the next top-of-hour snapshot window
  - startup log: `results_d6_daemon.log`
- Purpose:
  - keep accumulating hourly depth history locally while the user is away, so `D6.3` can progress without another manual launch.
