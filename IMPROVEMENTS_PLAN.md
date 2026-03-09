# План улучшений — 9 марта 2026

## Закодить сегодня (без новых данных)

### Pipeline (требует retrain после)
- [x] **Residual target** — `ret_excess = ret_coin_12h - β*ret_btc_12h` (rolling beta 168h). Добавить как опцию `--residual-target` в v6/v7. Ранжирование по excess return вместо raw return.
- [x] **Гибридная нормализация** — CS-rank для relative фичей (momentum ranks, MA ratios, etc.) + TS-zscore per symbol для burst/spike фичей (vol_surge, news_burst, funding, range_expansion). Список фичей для TS-zscore определяется в TSZSCORE_COLS.
- [x] **LambdaRank objective** — добавить `--lambdarank` флаг в v6. При включении: objective='lambdarank', group=timestamp, ndcg позиция. Альтернатива MSE на rank target.
- [x] **Null importance FS** — новая функция для сравнения gain importance на реальном vs shuffled target. Помогает отсеять шумные фичи.

### Sim (можно тестить на текущих моделях)
- [x] **Signal smoothing** — EMA(2): `score_final = 0.6*score_t + 0.4*score_{t-1}`. Флаг `--smooth-signal`.
- [x] **Vol-adjusted sizing** — `weight ∝ edge / σ_i` (Garman-Klass vol 24h). Флаг `--vol-size`.
- [x] **Regime short scaling** — если btc_above_ma720 + breadth > 0.5 → shorts = 50-70% от лонгов. Флаг `--regime-shorts`.
- [x] **npos 8/8 и 10/10** — добавить в sim grid для тестирования.

## Требует новых данных (Binance Futures API есть)
- [ ] **Open Interest** fetcher → фичи: oi_change_1h/12h/24h, oi_zscore_7d, ret*oi_change interaction
- [ ] **Basis / perp premium** = perp_price - spot_price, zscore
- [ ] **Taker buy/sell volume** → taker_imbalance_1h/12h, CVD
- [ ] **Liquidations** → long/short liq 1h/12h, zscore (контрарный сигнал)
- [ ] **Funding surprise** = actual_funding - rolling_mean(funding)

## После retrain
- [ ] **Meta-model (OOF стэкинг)** — логрег как бейзлайн, маленький GBDT. Label = "profitable rebalance". Использовать как risk scaler, не как binary filter.
- [ ] **Permutation importance** по OOS окнам (медиана дельты метрики при перемешивании фичи)
- [ ] **Group ablation** — выключать целые семейства фичей (returns, MA, vol, regime, news)
- [ ] **Dynamic universe** — фильтр по ликвидности (24h volume > threshold) для уменьшения survivorship bias
- [ ] **Gaussian rank** вместо linear rank (ppf(rank)) — тестировать после гибридной нормы

## Замечания по осторожности
- WR легко поднять "обрезав хвосты" — но цель max growth, не max WR
- FNG daily → hourly: проверить что нет мягкого leakage (присваивать со сдвигом?)
- При residual target: beta rolling window не должен быть слишком коротким (шум)
- LambdaRank: нужны group_id в LightGBM, группировка по timestamp

## Рекомендуемый sim grid (тестовые команды)

```bash
# Baseline (текущий лучший)
python run_fast_sim.py --days 365 --rebal 12 --ensemble --edge-boost

# + Signal smoothing (EMA 0.4)
python run_fast_sim.py --days 365 --rebal 12 --ensemble --edge-boost --smooth-signal 0.4

# + Vol-adjusted sizing
python run_fast_sim.py --days 365 --rebal 12 --ensemble --edge-boost --vol-size

# + Regime short scaling (50% shorts in bull)
python run_fast_sim.py --days 365 --rebal 12 --ensemble --edge-boost --regime-shorts 0.5

# + All sim improvements
python run_fast_sim.py --days 365 --rebal 12 --ensemble --edge-boost --smooth-signal 0.4 --vol-size --regime-shorts 0.5

# npos 8/8
python run_fast_sim.py --days 365 --rebal 12 --ensemble --edge-boost --npos 8

# npos 10/10
python run_fast_sim.py --days 365 --rebal 12 --ensemble --edge-boost --npos 10

# Combo: smooth + vol + npos 8
python run_fast_sim.py --days 365 --rebal 12 --ensemble --edge-boost --smooth-signal 0.4 --vol-size --npos 8
```

## Pipeline grid (после retrain на GPU)

```bash
# Residual target
python run_pipeline_v6.py --skip-hpo --residual-target

# Hybrid normalization
python run_pipeline_v6.py --skip-hpo --hybrid-norm

# LambdaRank
python run_pipeline_v6.py --skip-hpo --lambdarank

# Null importance
python run_pipeline_v6.py --skip-hpo --null-importance

# All improvements
python run_pipeline_v6.py --skip-hpo --residual-target --hybrid-norm --null-importance

# Residual + LambdaRank (test separately — different objective)
python run_pipeline_v6.py --skip-hpo --residual-target --lambdarank
```
