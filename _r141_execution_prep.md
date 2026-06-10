# R141 — Execution-readiness prep (POINT 5)

Pure code-reading + arithmetic. No data loads. Source: `run_trading.py`, `src/costs.py`,
`deploy/crypto-trader.service`, `.env`, live log `data_vps_d6/bot_live.log`.

## 1. Adaptive position sizing — how count/size shrink with small capital

`construct_portfolio()` (run_trading.py ~L1416–1689). Champion = `DEFAULT_RISK`/CLS:
`n_long=4, n_short=2` (6 positions), `kelly_frac=0.8` (CLS overrides to 1.0), lev set by CLI.

Flow:
- `total_alloc = capital × effective_kelly × leverage` (L1611). In CLS mode vol_scale and
  other factors are disabled, so `total_alloc ≈ capital × leverage`.
- L/S split: CLS forces 50/50 (`long_alloc_frac=0.5`, L1616). So
  `long_half = 0.5·total_alloc`, `short_half = 0.5·total_alloc`.
- **Per-sleeve fraction of capital** (lev=1):
  - long sleeve = `0.5·cap / 4 = 0.125·cap`  ← smallest
  - short sleeve = `0.5·cap / 2 = 0.25·cap`
- **Adaptive count reduction** (L1630–1641): `MIN_ORDER = 5.0`.
  `max_long = max(1, int(long_half / 5))`; if `max_long < n_long` it drops the long count
  (prints "Adaptive sizing: 4L → kL"). Same for shorts. So when capital is small the bot
  silently trades **fewer than 6 positions**.
- **Hard floor** (L1668): after edge-boost weighting, any sleeve with `usd < 5` is `continue`d
  (dropped entirely). Combined with the count-reduction this means small capital → a degenerate
  1–3 position book, not the 4L/2S champion that was backtested.
- Cap per position 15% of leveraged capital (L1644), not binding at low capital.

## 2. Minimum viable capital (4L/2S champion, OKX perps, lev=1, CLS kelly=1)

Smallest sleeve is a **long** sleeve = `0.125 × capital`. Min order = $5 (OKX min-notional,
hard-coded both as MIN_ORDER and the `usd < 5` floor).

| Constraint | Smallest long sleeve | Required capital | Short sleeves |
|---|---|---|---|
| C1 hard $5 floor (sleeves just clear $5) | $5 | **$40** | $10 |
| C2/C3 practical (lot-rounding + 12% rebal threshold safe) | $20 | **$160** | $40 |
| Comfortable (fee << edge, alt min-lots non-binding) | $30 | $240 | $60 |

Arithmetic: `capital = smallest_sleeve / 0.125`.
- Absolute floor = $5 / 0.125 = **$40** (below this the bot drops long positions and is no
  longer the 6-position champion).
- **Recommended minimum viable = $160–200** (long sleeve $20–25, short sleeve $40–50).
  Reasoning: REBALANCE_THRESHOLD=0.12 (L177) means a sleeve only re-trades when target moves
  >12%; on a $5 sleeve that is $0.60 — below OKX lot quantization for most alts, so rebalances
  either no-op or round to a full lot = large relative slippage. At $20–25 sleeves the $5 floor
  and lot rounding stop being binding and per-trade fee (Tier1 ~2.4bp blended, Tier2 ~5.5bp; RT
  ≈ 0.05–0.11% of sleeve) is a small fraction of edge.

**Why April failed at $80:** `--capital 80` → long sleeves `0.125·80 = $10`, short sleeves $20.
After balance drifted to ~$28 (see log), long sleeves fell to ~$3.5 → below $5 → positions
dropped, book collapsed to 1–2 names, and fixed min-order + lot rounding dominated returns.
$80 is below even a safe-ish floor; the prior "$500" guess is far higher than required — **$160–200
is the honest minimum** for the 6-position champion to run as designed.

## 3. State-persistence bug — root cause + fix

Restarted 51 times in 5 weeks (51 "PRODUCTION TRADING" banners in bot_live.log; 7 LIVE startups
at `--capital 80`/`100`). systemd `Restart=on-failure, RestartSec=30`
(`deploy/crypto-trader.service`).

**What's already correct in current tree (committed):**
- `save_state()` (L2578) IS atomic: write `.tmp` then `os.replace()`.
- `load_state()` (L2566) loads on start; startup reads `trading_state.json` (L3082–3083).
- DD/peak resume keeps the true peak on recovery (L1440, "don't reset it").

**Root cause of the wipe (still live):** the live log shows
`⚠️ Corrupt state file /home/trader/invest/trading_logs/trading_state.json, starting fresh`
followed immediately by a banner with **balance $28.35** but capital reset. Two compounding
problems:
1. **Corrupt-file → silent fresh `{}`**: `load_state` catches `JSONDecodeError`/`ValueError`
   and returns `{}`. The fresh branch (L3084–3089) then sets
   `state['peak'] = args.capital` (=$80/$100) — i.e. **peak is reset to the CLI capital, not to
   the real account equity**. So DD is computed against a fictitious peak and the equity curve /
   recent_rets / cycle history are all lost. (Atomic write was added Mar-31 / commit f7e1357,
   but the file can still be empty/partial if the very first write was SIGTERM-killed, and
   crucially the *recovery path resets peak to capital*.)
2. **Bootstrap only fires when equity == capital**: the exchange-bootstrap (L3114)
   `if state.get('equity', args.capital) == args.capital:` — after a fresh reset equity==capital
   so it *should* fire, but it only overwrites `equity`/`prev_equity` and does
   `peak = max(peak, live_equity)`. Since live_equity ($28) < reset peak ($80), **peak stays at
   the inflated $80**, so the bot believes it is in a −65% drawdown on every restart and the
   DD circuit breaker (`dd_stop=-0.15`) can trip immediately.

**Concrete fix (3 parts):**
- (a) Keep atomic write (already done) + add an `os.fsync` before `os.replace` for durability,
  and write a backup `trading_state.json.bak` after each successful save so a corrupt primary
  can fall back instead of starting fresh.
- (b) In `load_state`, on corrupt/missing primary, try `.bak` before returning `{}`; only
  return `{}` if both fail.
- (c) On fresh/recovered state, **derive peak from the exchange, never from `--capital`**:
  always `fetch_balance()`+`fetch_positions()` first, set `equity = live_equity`, and
  `peak = max(persisted_peak_if_any, live_equity)` — and do NOT seed `peak = args.capital`. i.e.
  move the exchange-bootstrap *before* the fresh-defaults block, and gate it on
  `'equity' not in state` rather than `equity == capital`.

## 4. OKX referral 20% — status

**Purely manual. Not wired anywhere in runtime.** No referral/rebate key in `.env` or
`.env.example` (keys: OKX_API_KEY/SECRET/PASSPHRASE/DEMO, TELEGRAM_*, CAPITAL, MODE,
REBAL_HOURS, LEVERAGE, CC/FRED/ALPHAVANTAGE/COINGLASS/CRYPTOQUANT). The only "referral 20%"
references are in research scaffolding (`_research_r124_fee_optimization.py`,
`_research_r124b_taker_baseline.py`) that *model* a 20% fee cashback as a scenario — they do not
touch execution. The live cost model in `src/costs.py` does NOT subtract any referral discount.
Status: referral discount is applied by OKX server-side on the account (manual enrolment); the
bot/cost-model does not know about it and does not double-count it. If you want backtests to
reflect it, multiply the fee terms in `src/costs.py` by 0.8 — but that is an explicit, manual
edit, not currently present.

## 5. Maker-fill measurement — what's needed

The infrastructure already exists: `trading_logs/execution_log.csv`
(`EXEC_LOG_PATH`, L197) with header (L198–202):
`timestamp,symbol,okx_sym,tier,side,order_type,attempt,bid,ask,mid,limit_px,fill_price,`
`spread_bps,slippage_bps,effective_bps,filled_qty,cost_usd,fill_time_s,was_maker`.
`was_maker` is inferred at L2104–2106 from the actual OKX fill fee:
`fee_rate = |fee.cost| / fill.cost; was_maker = fee_rate < 0.0003` (taker ~5bp, maker ~0/rebate).

To validate the cost model's 90% (Tier1) / 50% (Tier2) maker assumptions, you need this CSV
populated from a **live** run (paper fills don't exercise the maker-limit path). Then per tier:
`maker_rate = mean(was_maker) grouped by tier`, and cross-check `effective_bps`/`slippage_bps`
distributions vs the assumed 2.4bp (Tier1) / 5.5bp (Tier2) blends in `src/costs.py` (L45–52).
Current gap: no live execution_log.csv has accumulated enough fills to measure this — that is the
single artifact to collect on the next live restart before trusting the blended cost numbers.
