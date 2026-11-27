# wrsi_trader — configuration reference

Every configurable parameter, grouped by file, with allowed values, the value currently in the repo
(`master`, Nov 2026) and examples. Generated from the config files **and** the code that reads them, so
keys that exist only as code defaults are listed too (marked *code default*).

Conventions used below
- **Percentages are plain percent**: `1.5` means 1.5 %, not 0.015.
- **`_note` / `_xxx` keys** are comments. The code ignores every key that starts with `_`.
- **`null`** in JSON = "not set"; the code then uses its default or switches the feature off.
- **Pandas resample rules**: `weekly_rule` / `timeframe_rule` take a weekly anchor such as `"W-FRI"`
  (week ending Friday — NSE's last session), `"W-THU"`, …; `monthly_rule` takes `"ME"` (month end).
- **Timeframes**: `"daily"`, `"weekly"`, `"monthly"`.
- **`fail_mode`** (filters/regime checks): `"allow"` = missing/short data passes (fail open);
  `"block"` = missing data blocks (fail closed).
- **Filter `mode`**: `"rank"` = never blocks, only adds to the tie-break score when slots are scarce;
  `"filter"` = hard gate (blocks candidates that fail). `sentiment` also has `"shadow"`.

Contents
1. [How to override without editing files](#1-how-to-override-without-editing-files)
2. [risk_config.json](#2-risk_configjson) — capital, sizing, position limits, exposure caps
3. [system_config.json](#3-system_configjson) — paper/live, paths, Upstox data, slippage
4. [fee_config.json](#4-fee_configjson)
5. [Strategies (`strategies/*.json`)](#5-strategies) — common keys, then each strategy
6. [Filters (`filters/*.json`)](#6-filters) — common keys, then each filter
7. [Support/resistance: sr_engine.json, levels.json, fractal_sr.json](#7-supportresistance)
8. [candlestick_patterns.json](#8-candlestick_patternsjson)
9. [Market-trend engines: market_trend.json, trend_structure.json](#9-market-trend-engines)
10. [Data files: universe, sector_map, holidays, corporate_actions](#10-data-files)
11. [Recipes](#11-recipes)

---

## 1. How to override without editing files

`scripts/regression_check.py --as-is --set <path>=<value>` changes a value **in memory for one run**;
the JSON files are not touched. Values are parsed as JSON (`true`, `80`, `null`, `"text"`); anything
that is not valid JSON is used as a plain string.

| Path form | Points at | Example |
|---|---|---|
| `risk.<key>...` | `risk_config.json` | `--set risk.capital.mode=equity` |
| `<filter>.<key>...` | `filters/<filter>.json` | `--set candlestick.enabled=true` |
| `<strategy>.<key>...` | `strategies/<strategy>.json` | `--set breakout_trend.exit.atr_mult=2.5` |

```powershell
python scripts\regression_check.py --as-is --start 2021-01-01 --end 2026-11-13 --label my_test `
  --set risk.capital.mode=equity `
  --set risk.portfolio.regime_gross_exposure.enabled=true `
  --set risk.portfolio.regime_gross_exposure.bear_pct=80
```
`python -m trader backtest|eod|open` read the files as they are. `--strategy a,b` runs only those
strategies (overrides each strategy's `enabled` for that run).

---

## 2. risk_config.json

Read by the engine (`trader/engine/orchestrator.py`), `trader/risk/capital.py`, `trader/risk/sizing.py`.

### 2.1 `capital`
| Key | Values | Current | Meaning |
|---|---|---|---|
| `base_capital` | number (₹) | `1000000` | Starting cash, and the risk base in `fixed` mode. |
| `mode` | `"fixed"` \| `"equity"` | `"fixed"` | **fixed**: every trade risks a % of `base_capital` forever — profits pile up as idle cash. **equity**: risk % of *current* equity — profits are reinvested (compounding). |
| `allocation_pct.<strategy>` | number (%) | `100` each | Share of the base each strategy sizes from (`base × pct/100`). **Every enabled strategy needs an entry** (missing → error). `50` halves that strategy's position sizes. |

### 2.2 `per_trade`
| Key | Values | Current | Meaning |
|---|---|---|---|
| `risk_pct_of_base` | number (%) | `0.8` | ₹ risked per entry = base × this %. Quantity = risk ₹ ÷ (entry − stop). |
| `max_position_pct_of_base` | number (%) | `20.0` | Cap on one position's value as % of the base. |
| `min_notional` | number (₹) | `15000` | Orders smaller than this are skipped (fees would dominate). |
| `min_stop_distance_pct` | number (%) | `3.0` | Stops closer than this are treated as this far **for sizing** (prevents huge quantities on tight stops). |
| `cash_buffer_pct` | number (%) | `1.0` | Keep this % of cash unspent on each buy. |

### 2.3 `portfolio`
| Key | Values | Current | Meaning |
|---|---|---|---|
| `max_positions` | integer | `20` | Maximum open positions across all strategies (slot auction: highest score first). |
| `max_total_open_risk_pct_of_base` | number (%) | `8.0` | Cap on the sum of open risk (entry→stop, profit-locked lots count 0) across positions. |
| `max_gross_exposure_pct` | number (%) \| `null` | `null` | **Fixed gross-exposure cap**: open positions + a new buy may not exceed this % of current equity; the order is shrunk to fit, or skipped below `min_notional`. Buy-time only — winners are never trimmed. `null` = off. Example: `80`. |
| `regime_gross_exposure` | object | see 2.4 | **Trend-aware gross cap** — overrides `max_gross_exposure_pct` when enabled. |

### 2.4 `portfolio.regime_gross_exposure` (trend-aware gross cap)
| Key | Values | Current | Meaning |
|---|---|---|---|
| `enabled` | `true` \| `false` | `false` | On/off. When on, the cap is `bull_pct` or `bear_pct` depending on the signal, read as of the session **before** each fill. Journals `GROSS_REGIME` on days with buys (and every EOD once `breadth-live-cache.patch` is applied). |
| `signal` | `"breadth"` \| `"index"` | `"breadth"` | **breadth** = % of the strategies' universe closing above its own SMA (below). **index** = benchmark weekly close vs weekly SMA (same test as `market_regime`; tested much worse). |
| `bull_pct` | number (%) | `100` | Max % of equity invested while the signal is bullish (`100` = effectively uncapped). |
| `bear_pct` | number (%) | `70` | Max % invested while not bullish. Tested best: `80`. |
| `breadth.sma_period` | integer | `50` | Each stock is "above" when its close > its own SMA of this length. |
| `breadth.smooth_days` | integer | `5` | Moving average of the daily breadth % before the state test. |
| `breadth.bull_on_pct` | number (%) | `55` | Turns bullish when smoothed breadth ≥ this. |
| `breadth.bull_off_pct` | number (%) | `45` | Turns not-bullish when smoothed breadth < this. Between the two, the previous state holds (hysteresis). |
| `breadth.min_symbols` | integer | `50` | Fewer stocks with a valid SMA → signal fails open (bullish). |
| `index.benchmark_symbol` | symbol | `"NIFTY500"` | Index for `signal: "index"`. |
| `index.timeframe_rule` / `sma_period` / `buffer_pct` / `min_bars` | as `market_regime` | `"W-FRI"` / `20` / `0.0` / `30` | See [market_regime](#market_regimejson). |

Robustness notes (Nov 2026 sweeps): `bear_pct` 60–80 and thresholds 60/40 hold up; slower signals
(`smooth_days` 10, `sma_period` 100) performed as badly as the index signal. Visualize first with
`python scripts\trend_signals.py --start 2021-01-01`.

### 2.5 `pyramiding`
| Key | Values | Current | Meaning |
|---|---|---|---|
| `max_adds` | integer | `3` | Max ADD orders per position (only strategies with an `add_signal`, i.e. `weekly_rsi_sma20`). |
| `max_position_risk_pct_of_base` | number (%) | `0.8` | Cap on one position's total open risk after adds. |

### 2.6 Per-strategy `sizing` override (*code default: absent*)
A strategy config may contain a `sizing` block to override the shared sizing for that strategy only
(`trader/risk/sizing.py`). Keys: `risk_pct`, `max_position_pct`, `min_stop_distance_pct`,
`min_notional`, `cash_buffer_pct` (override the `per_trade` keys), `capital_mode`
(`"fixed"`/`"equity"`), `portfolio_open_risk_cap` (`true`/`false`, default `true`).
```json
"sizing": { "risk_pct": 0.4, "capital_mode": "equity" }
```
Note: in a cash-bound book, a smaller `risk_pct` mostly means *more, smaller* positions, not less exposure.

**Example — the Nov 2026 top pick (equity sizing, breadth cap 100 / 80):**
```json
"capital":   { "base_capital": 1000000, "mode": "equity", "allocation_pct": { ...unchanged... } },
"portfolio": { "max_positions": 20, "max_total_open_risk_pct_of_base": 8.0, "max_gross_exposure_pct": null,
               "regime_gross_exposure": { "enabled": true, "signal": "breadth", "bull_pct": 100, "bear_pct": 80,
                 "breadth": {"sma_period": 50, "smooth_days": 5, "bull_on_pct": 55, "bull_off_pct": 45, "min_symbols": 50},
                 "index": { ...unchanged... } } }
```

---

## 3. system_config.json

| Key | Values | Current | Meaning |
|---|---|---|---|
| `mode` | `"paper"` \| `"live"` | `"paper"` | `live` places real orders via Upstox and **also requires** env var `ALLOW_LIVE=1`. |
| `timezone` | IANA zone | `"Asia/Kolkata"` | Used for "today" and Upstox timestamps. |
| `paths.state_dir` | folder | `"state"` | Portfolio, pending orders, journal, equity, backtest/regression outputs. |
| `paths.cache_dir` | folder | `"data_cache"` | Daily-bar CSV cache (`<SYMBOL>.csv` + `.meta.json`). |
| `data.base_url` | URL | `"https://api.upstox.com"` | Upstox API root. |
| `data.history_years` | integer | `10` | Years fetched on a full refresh. |
| `data.chunk_days` | integer | `1800` | Days per historical-candle request. |
| `data.overlap_days` | integer | `7` | Incremental fetch re-pulls this many days to catch revisions. |
| `data.full_refresh_days` | integer | `7` | Full re-pull every N days (picks up split/bonus back-adjustments). |
| `data.http_timeout_s` | seconds | `20` | Per-request timeout. |
| `data.retries` | integer | `3` | Retries on 429/5xx/network errors. |
| `data.retry_backoff_s` | seconds | `2.0` | Backoff × attempt number. |
| `data.quote_batch_size` | integer | `100` | Symbols per market-quote call (open prices). |
| `data.fetch_today_intraday` | `true`/`false` | *code default `true`* | After 15:35, append today's final daily candle from the intraday endpoint. |
| `execution.slippage_bps` | number (bps) | `10` | Paper/backtest slippage on every fill (10 = 0.10 %); also used to size on the expected fill price. |
| `execution.stale_order_policy` | `"drop"` | `"drop"` | **Not read by the code** — stale pending orders are always dropped (journal `STALE_DROPPED`); exits are retried next session. |
| `logging.level` | `"DEBUG"`/`"INFO"`/`"WARNING"`/`"ERROR"` | `"INFO"` | `-v` on the CLI forces DEBUG. |

---

## 4. fee_config.json
Delivery-equity charges used by the paper broker (and so by every backtest). **Verify against the
current Upstox/NSE/SEBI schedule.**

| Key | Values | Current | Meaning |
|---|---|---|---|
| `brokerage_flat` | ₹ | `20.0` | Flat brokerage per order… |
| `brokerage_pct_cap` | % | `2.5` | …capped at this % of order value. |
| `stt_pct` | % | `0.1` | Securities transaction tax (buy and sell, delivery). |
| `exchange_txn_pct` | % | `0.00297` | NSE transaction charge. |
| `sebi_pct` | % | `0.0001` | SEBI turnover fee. |
| `stamp_duty_pct_buy` | % | `0.015` | Stamp duty, buy side only. |
| `gst_pct` | % | `18.0` | GST on brokerage + exchange + SEBI charges. |
| `dp_charge_sell` | ₹ | `15.34` | Depository charge per sell. |

---

## 5. Strategies

Files: `strategies/<name>.json`. Names: `weekly_rsi_sma20`, `breakout_trend`, `new_high_momentum`,
`pullback_ma`, `sr_reversal`, `inside_bar_breakout`. Entries fill at the **next session's open**;
level exits trigger on the close and fill at the next open (unless `execution.mode` is `intraday`).

### 5.1 Keys common to every strategy
| Key | Values | Meaning |
|---|---|---|
| `enabled` | `true`/`false` | On/off. Current: all `true` except `inside_bar_breakout`. |
| `universe` | list name from `universe.json → lists` | Symbols scanned. Current: `"default"` (Nifty 500). |
| `exclude_filters` | list of filter names | Globally-enabled filters this strategy skips. Names: `candlestick`, `market_efficiency`, `market_regime`, `quality`, `regime_exposure`, `regime_match`, `resistance`, `rs_rank`, `sector_regime`, `sentiment`, `trend_strength`. |
| `market_regime_tag` | `"UPTREND"` \| `"SIDEWAYS"` \| `"DOWNTREND"` \| absent | Read only by the `regime_match` filter (currently off). |
| `candlestick_timeframe` | `"daily"`/`"weekly"`/`"monthly"` | Which candle the `candlestick` filter checks (default: that filter's `default_timeframe`). |
| `market_efficiency_override` | `{ "min_efficiency": <0–1> }` | Per-strategy threshold for the `market_efficiency` filter. |
| `sizing` | object | Per-strategy sizing override — see [2.6](#26-per-strategy-sizing-override-code-default-absent). |
| `execution` | `{ "mode": "intraday" }` \| absent | *code default absent.* Resting-order simulation on the session's own OHLC. **Do not use with the current strategies**: entries would fill at the open of the same session whose close fired the signal (look-ahead). |

### 5.2 weekly_rsi_sma20 — weekly RSI crosses above 60, monthly RSI confirms
| Key | Values | Current | Meaning |
|---|---|---|---|
| `timeframes.weekly_rule` / `monthly_rule` | resample rules | `"W-FRI"` / `"ME"` | Bar construction. |
| `indicators.rsi_period` | integer | `14` | RSI length (weekly and monthly). |
| `indicators.sma_period` | integer | `20` | Weekly SMA used for entry check and exit. |
| `entry.weekly_bar` | `"completed"` \| anything else | `"completed"` | `completed` = only finished weeks; otherwise the forming week counts. |
| `entry.weekly_rsi_threshold` | number | `60` | Weekly RSI must cross above this. |
| `entry.sustain_weeks` | integer ≥1 | `1` | Weeks RSI must stay above before entry. |
| `entry.monthly_rsi_threshold` | number | `60` | Monthly RSI must be ≥ this − tolerance. |
| `entry.monthly_rsi_near_tolerance` | number | `3` | Allowed shortfall (60 − 3 = 57 passes). |
| `entry.monthly_include_partial` | `true`/`false` | `true` | Include the forming month. |
| `entry.min_weekly_bars` / `min_monthly_bars` | integer | `60` / `48` | History required. |
| `entry.require_close_above_sma` | `true`/`false` | `true` | Weekly close must be above SMA20. |
| `entry.min_stop_distance_pct` / `max_stop_distance_pct` | % | `1.0` / `25.0` | Reject entries whose stop is closer/farther than this. |
| `exit.method` | `"sma20"` \| `"supertrend"` | `"sma20"` | Trailing exit line. |
| `exit.trigger` | `"wick"` \| `"close"` | `"wick"` | `wick` = weekly LOW touches the line; `close` = weekly close below. |
| `exit.buffer_pct` | % | `0.0` | Line moved down by this %. |
| `exit.mode` | `"weekly_close"` | *code default* | Only supported value. |
| `exit.supertrend.period` / `multiplier` | integer / number | `10` / `3.0` | Used when `method` = `supertrend`. |
| `exit.fast_trail.enabled` | `true`/`false` | `false` | Ratchet the stop up to a faster SMA once earned. |
| `exit.fast_trail.trigger` | `"min_holding_weeks"` \| `"min_unrealized_gain_pct"` | `"min_holding_weeks"` | What earns the faster trail. |
| `exit.fast_trail.min_holding_weeks` / `min_unrealized_gain_pct` / `sma_period` | integer / % / integer | `8` / `15.0` / `10` | Thresholds; only the one matching `trigger` is read. |
| `pyramiding.enabled` | `true`/`false` | `true` | Allow ADD orders. |
| `pyramiding.weekly_bar` | as `entry.weekly_bar` | `"completed"` | |
| `pyramiding.proximity_pct` | % | `3.0` | Add when the weekly SMA is within this % of the last buy price. |
| `pyramiding.min_weekly_rsi` | number | `50` | |
| `pyramiding.min_weeks_between_adds` | integer | `2` | |
| `pyramiding.require_close_above_sma` | `true`/`false` | `true` | |
| `open_checks.apply_to` | list of `"ENTRY"`, `"ADD"` | both | Which orders get the open-price checks below. |
| `open_checks.gap_down_tolerance_pct` | % | `0.0` | Skip if the open is more than this % below yesterday's close. |
| `open_checks.skip_if_open_below_sma` | `true`/`false` | `true` | Skip if the open is below the order's stop. |
| `open_checks.max_gap_up_pct` | % \| `null` | `null` | Skip if the open gaps up more than this %. `null` = no limit. |

### 5.3 breakout_trend — daily close breaks a resistance zone above SMA50
| Key | Values | Current | Meaning |
|---|---|---|---|
| `levels_timeframe` | `"daily"`/`"weekly"`/`"monthly"` | `"daily"` | Timeframe of the zones. |
| `levels_engine` | `null` \| `"legacy"` \| `"fractal"` | `null` | `null` = `sr_engine.json`. |
| `levels_config_path` | path | `"config/levels.json"` | Legacy-engine params. |
| `entry.min_bars` | integer | `260` | History required. |
| `entry.trend_sma_period` | integer | `50` | Close must be above this SMA. |
| `entry.breakout_buffer_pct` | % | `0.3` | Close must clear the zone top by this %. |
| `entry.require_volume_confirmation` | `true`/`false` | `true` | Volume ≥ `volume_mult` × SMA(`volume_sma_period`). |
| `entry.volume_sma_period` / `volume_mult` | integer / number | `20` / `1.2` | |
| `entry.atr_period` / `atr_mult` | integer / number | `14` / `1.0` | Initial stop = entry − `atr_mult` × ATR. |
| `exit.atr_period` / `atr_mult` | integer / number | `14` / `3.0` | Chandelier trail: highest close since entry − `atr_mult` × ATR (ratchets up only). Tested 2.5: worse. |
| `exit.buffer_pct` | % | `0.0` | Extra room below the trail. |

### 5.4 new_high_momentum — close is a 50-bar high above SMA50 on volume
| Key | Values | Current | Meaning |
|---|---|---|---|
| `entry.min_bars` | integer | `260` | |
| `entry.new_high_lookback` | integer | `50` | Close must be the highest close of this many bars. |
| `entry.trend_sma_period` | integer | `50` | |
| `entry.require_volume_confirmation` / `volume_sma_period` / `volume_mult` | bool / int / number | `true` / `20` / `1.2` | |
| `entry.atr_period` / `atr_mult` | int / number | `14` / `2.0` | Initial stop. |
| `exit.atr_period` / `atr_mult` / `buffer_pct` | int / number / % | `14` / `3.5` / `0.0` | Chandelier trail. Tested 3.0 and 2.5: worse. |
| `exit.trend_fail_sma_period` / `trend_fail_buffer_pct` | int / % | `50` / `1.0` | Also exit when the close falls this % below this SMA. |

### 5.5 pullback_ma — dip to the SMA20 inside an SMA50 uptrend
| Key | Values | Current | Meaning |
|---|---|---|---|
| `candlestick_config_path` | path | `"config/candlestick_patterns.json"` | |
| `entry.min_bars` | integer | `260` | |
| `entry.trend_sma_period` | integer | `50` | Trend filter. |
| `entry.pullback_sma_period` | integer | `20` | The MA the pullback must touch. |
| `entry.touch_tolerance_pct` | % | `0.5` | How close counts as a touch. |
| `entry.atr_period` / `atr_mult` | int / number | `14` / `1.0` | Initial stop. |
| `exit.trail_sma_period` | integer | `50` | Exit when the close breaks this SMA. Tested 30: worse. |
| `exit.buffer_pct` | % | `0.0` | |

### 5.6 sr_reversal — bullish candle off a support zone (long only)
| Key | Values | Current | Meaning |
|---|---|---|---|
| `levels_timeframe` / `levels_engine` / `levels_config_path` / `candlestick_config_path` | as breakout_trend | `"daily"` / `null` / … | |
| `entry.min_bars` | integer | `260` | |
| `entry.proximity_pct` | % | `1.0` | Price must be within this % of the support zone. |
| `entry.momentum_lookback` | integer | `3` | Bars used for the turn-up check. |
| `entry.atr_period` / `atr_mult` | int / number | `14` / `1.0` | Stop below the zone. |
| `exit.target_tolerance_pct` | % | `1.0` | Exit when within this % of the next resistance (target). |
| `entry.min_reward_risk`, `entry.min_support_touches` | — | removed | Tested Nov 2026 and removed: they cut the best trades. |

### 5.7 inside_bar_breakout — inside bar in an uptrend, break of its high (currently disabled)
| Key | Values | Current | Meaning |
|---|---|---|---|
| `entry.min_bars` / `trend_sma_period` | integer | `60` / `50` | |
| `entry.atr_period` / `atr_mult` | int / number | `14` / `1.0` | |
| `entry.breakout_buffer_pct` | % | `0.0` | |
| `entry.prefer_small_range` / `max_range_atr_mult` | bool / number | `true` / `1.0` | Only inside bars with range ≤ this × ATR. |
| `exit.atr_period` / `atr_mult` / `buffer_pct` | int / number / % | `10` / `1.75` / `0.0` | Tight chandelier trail. |

---

## 6. Filters

Files: `filters/<name>.json`. A filter applies to every enabled strategy except those listing it in
their `exclude_filters`. Scores from `rank`-mode filters are normalised and averaged into the
candidate's ranking score.

### 6.1 Keys common to most filters
| Key | Values | Meaning |
|---|---|---|
| `enabled` | `true`/`false` | Global on/off. |
| `mode` | `"rank"` \| `"filter"` (sentiment: `"filter"` \| `"shadow"`) | See conventions at the top. |
| `fail_mode` | `"allow"` \| `"block"` | Missing-data behaviour. |
| `min_score` | integer | In `filter` mode, minimum number of sub-checks that must pass. |
| `benchmark_symbol` | symbol in `universe.json → instruments` | Usually `"NIFTY500"`. |

### market_regime.json — block entries unless the benchmark is in a weekly uptrend
Current: **enabled**, but every strategy except `weekly_rsi_sma20` excludes it.
| Key | Values | Current | Meaning |
|---|---|---|---|
| `timeframe_rule` | resample rule | `"W-FRI"` | Weekly bars. Note: the current (unfinished) week is included, so the state can flip mid-week. |
| `sma_period` | integer | `20` | Bullish when weekly close > SMA × (1 + buffer). |
| `buffer_pct` | % | `0.0` | |
| `min_bars` | integer | `30` | |
| `fail_mode` | allow/block | `"block"` | |

### resistance.json — block entries just below overhead resistance
Current: **enabled** (applies to `new_high_momentum` and `weekly_rsi_sma20`).
| Key | Values | Current | Meaning |
|---|---|---|---|
| `timeframe` | daily/weekly/monthly | `"weekly"` | Zone timeframe. |
| `levels_config_path` | path | `"config/levels.json"` | |
| `proximity_pct` | % | `3.0` | Block if price is within this % below a zone. |
| `engine` | `null`/`"legacy"`/`"fractal"` | `null` | `null` = `sr_engine.json`. |
| `proximity_atr` | number \| `null` | `null` | Fractal engine only: proximity in ATRs instead of %. |
| `block_inside_zone` | `true`/`false` | `true` | Fractal engine only: block when price is already inside the zone. |
| `fail_mode` | allow/block | `"allow"` | |

### rs_rank.json — relative strength vs benchmark
Current: **enabled, rank** (excluded by `pullback_ma`, `weekly_rsi_sma20`).
| Key | Values | Current | Meaning |
|---|---|---|---|
| `lookback_weeks` | integer | `13` | Return window. |
| `weekly_rule` | rule | `"W-FRI"` | |
| `min_percentile` | 0–1 | `0.5` | `filter` mode: block below this cross-sectional percentile. Tested `filter`: worse. |
| `sector_map_path` | path | `"config/sector_map.json"` | Universe for percentiles. |
| `fail_mode` | allow/block | `"allow"` | |

### sector_regime.json — the stock's sector index must be in a weekly uptrend
Current: **enabled, rank** (applies to `pullback_ma`, `weekly_rsi_sma20`).
| Key | Values | Current | Meaning |
|---|---|---|---|
| `timeframe_rule` / `sma_period` / `buffer_pct` / `min_bars` | as market_regime | `"W-FRI"` / `20` / `0.0` / `30` | Gate test per sector index. |
| `sector_map_path` | path | `"config/sector_map.json"` | Symbol → industry. |
| `industry_to_benchmark` | `{ industry: index symbol }` | 20 industries | Each index must exist in `universe.json → instruments`. Example: `"Information Technology": "NIFTY IT"`. |

### trend_strength.json — ADX and SMA50 trend checks
Current: **enabled, rank** (applies to `weekly_rsi_sma20`, `sr_reversal`).
| Key | Values | Current | Meaning |
|---|---|---|---|
| `weekly_rule`, `sector_map_path` | | `"W-FRI"`, sector map | |
| `min_score` | integer 0–2 | `1` | `filter` mode: sub-checks that must pass. |
| `adx.enabled` / `period` / `min_adx` | bool / int / number | `true` / `14` / `20` | |
| `sma_trend.enabled` / `period` / `buffer_pct` | bool / int / % | `true` / `50` / `0.0` | |

### candlestick.json — the deciding candle must be bullish
Current: **disabled** (tested Nov 2026: inconsistent).
| Key | Values | Current | Meaning |
|---|---|---|---|
| `mode` | rank/filter | `"rank"` | |
| `min_net_sentiment` | number | `0.0` | `filter` mode: net pattern score must exceed this. |
| `score_ceiling` | number | `5.0` | Normalisation cap for ranking. |
| `default_timeframe` | daily/weekly/monthly | `"daily"` | Overridden per strategy by `candlestick_timeframe`. |
| `pattern_cfg_path` | path | `"config/candlestick_patterns.json"` | |

### market_efficiency.json — benchmark trending vs chopping (Kaufman efficiency ratio)
Current: **disabled** (tested as `filter`: worse).
| Key | Values | Current | Meaning |
|---|---|---|---|
| `lookback_days` / `min_bars` | integer | `20` / `25` | |
| `default_min_efficiency` | 0–1 | `0.10` | Per-strategy override: `market_efficiency_override`. |
| `mode` / `min_score` / `fail_mode` | | `"rank"` / `1` / `"allow"` | |

### quality.json — near all-time high and/or cup shape
Current: **disabled**.
| Key | Values | Current | Meaning |
|---|---|---|---|
| `weekly_rule`, `mode`, `min_score` | | `"W-FRI"`, `"filter"`, `1` | |
| `ath.enabled` / `basis` / `max_distance_pct` | bool / `"close"`\|`"high"` / % | `true` / `"close"` / `5.0` | Within this % of the all-time high. |
| `cup.enabled` | bool | `true` | Cup-shape check: |
| `cup.lookback_weeks` / `left_rim_search_frac` | int / 0–1 | `52` / `0.5` | |
| `cup.min_depth_pct` / `max_depth_pct` | % | `12` / `35` | |
| `cup.min_leg_weeks` / `rim_tolerance_pct` | int / % | `8` / `5` | |
| `cup.min_symmetry` / `max_symmetry` | number | `0.5` / `2.0` | Left/right leg length ratio. |
| `cup.recent_uptick_weeks` | integer | `0` | |

### regime_match.json — strategy's `market_regime_tag` must match the swing-structure trend
Current: **disabled**. Keys: `benchmark_symbol` (`"NIFTY500"`), `trend_structure_config_path`
(`"config/trend_structure.json"`), `fail_mode` (`"block"`).

### regime_exposure.json — fewer slots for new entries in a bear regime (index signal)
Current: **disabled** (tested Nov 2026: worse with equity sizing). Never gates candidates.
| Key | Values | Current | Meaning |
|---|---|---|---|
| `benchmark_symbol` / `timeframe_rule` / `sma_period` / `buffer_pct` / `min_bars` / `fail_mode` | as market_regime | `"NIFTY500"` / `"W-FRI"` / `20` / `0.0` / `30` / `"allow"` | |
| `bear_max_positions_pct` | % | `40` | New-entry slots in a bear regime = floor(max_positions × pct / 100). |
| `bear_risk_scale` | number | `1.0` | Multiplies each throttled entry's risk. |

### sentiment.json — news-sentiment veto
Current: **disabled**.
| Key | Values | Current | Meaning |
|---|---|---|---|
| `mode` | `"filter"` \| `"shadow"` | `"filter"` | `shadow` logs what it would veto without blocking. |
| `path` / `glob` | folder / pattern | `"sentiment_gate"` / `"*.csv"` | Daily sentiment CSVs. |
| `lookback_days` | integer | `1` | |
| `veto_score` / `veto_weight` | number | `-0.25` / `1.0` | Veto when weighted score ≤ this. |
| `require_more_negative` | bool | `true` | |
| `fail_mode` | allow/block | `"allow"` | Unscraped symbol-days pass. |

---

## 7. Support/resistance

### sr_engine.json
| Key | Values | Current | Meaning |
|---|---|---|---|
| `engine` | `"legacy"` \| `"fractal"` | `"legacy"` | Default S/R engine for `breakout_trend`, `sr_reversal` (`levels_engine`) and the `resistance` filter (`engine`). Their own key overrides it. |

### levels.json — legacy engine, one section per timeframe (`daily`, `weekly`, `monthly`)
| Key | Values | Current (d / w / m) | Meaning |
|---|---|---|---|
| `pivot_left` / `pivot_right` | integer (bars of that timeframe) | 5/5 · 3/3 · 2/2 | Swing high/low = extreme of left+right bars. |
| `cluster_pct` | % | 1.5 · 2.5 · 4.0 | Pivots within this % merge into one zone. |
| `min_touches` | integer | 2 | Touches required for a zone. |
| `lookback_bars` | integer | 252 · 104 · 60 | History scanned. |
| `weekly_rule` / `monthly_rule` | rules | `"W-FRI"` / `"ME"` | |

### fractal_sr.json — fractal engine and `scripts/fractal_sr.py`
| Key | Values | Current | Meaning |
|---|---|---|---|
| `data.universe_list` | list name | `"default"` | |
| `data.weekly_rule` / `monthly_rule` | rules | `"W-FRI"` / `"ME"` | |
| `data.completed_bars_only` | bool | `true` | Drop the forming week/month. |
| `data.corporate_action_truncate_types` | list | `["demerger","spinoff"]` | Cut history at these events from `corporate_actions.json`. |
| `output.dir` / `prefix` | folder / text | `"state"` / `"sr_levels"` | Script output. |
| `output.levels_per_side` | integer | `1` | S1/R1 (2 adds S2/R2…). |
| `output.major_level` | bool | `true` | Also output the strongest zone per side. |
| `weights.fractal_only` / `window_only` / `both` / `polarity_bonus` | number | `1.0` / `0.75` / `1.5` / `1.25` | Pivot weights; zone score = Σ weight × 0.5^(age/half-life). |
| `timeframes.<tf>.lookback_months` | integer (0 = all) | 6 · 12 · 36 | Only swings inside this window count. |
| `timeframes.<tf>.min_bars` | integer | 60 · 30 · 18 | |
| `timeframes.<tf>.fractal_n` | integer | 2 | 2 = classic 5-candle Williams fractal. |
| `timeframes.<tf>.window` / `persist` | integer | 5/3 · 5/3 · 6/3 | Window-shifting detector. |
| `timeframes.<tf>.atr_period` | integer | 14 | |
| `timeframes.<tf>.cluster_atr_mult` | number | 0.75 · 0.75 · 0.35 | Clustering tolerance in ATRs… |
| `timeframes.<tf>.tol_min_pct` / `tol_max_pct` | % | 0.6–4.0 · 1.0–6.0 · 2.0–8.0 | …clamped to this % range. |
| `timeframes.<tf>.min_width_atr_mult` | number | 0.25 | Minimum zone width. |
| `timeframes.<tf>.recency_half_life_bars` | integer | 126 · 52 · 36 | Score decay. |
| `timeframes.<tf>.min_touches` | integer | 2 | |
| `timeframes.<tf>.accept_confirmed_single` / `single_pivot_min_score` | bool / number | `true` / `0.75` | Accept a fresh lone pivot found by both detectors. |
| `timeframes.<tf>.edge_fallback` | bool | `true` | Use the window's extreme as an unconfirmed zone when none exists. |

---

## 8. candlestick_patterns.json
Shape thresholds (ratios, so they work on any timeframe) and weights (relative conviction). Used by
`sr_reversal`, `pullback_ma` and the `candlestick` filter.

| Pattern | Keys (current) | Meaning |
|---|---|---|
| `marubozu` | `min_body_pct` 90, `weight` 3.0 | Body ≥ 90 % of range. |
| `doji` | `max_body_pct` 5, `weight` 0.0 | |
| `spinning_top` | `max_body_pct` 35, `weight` 0.0 | Small body, balanced shadows. |
| `hammer_family` | `max_body_pct` 35, `min_long_shadow_ratio` 2.0, `max_short_shadow_pct` 10, `trend_lookback` 5, `weight` 2.0 | Hammer / hanging man / shooting star / inverted hammer — trend picks the label. |
| `engulfing` | `weight` 2.5 | |
| `harami` | `min_first_body_pct` 50, `max_second_body_ratio` 50, `weight` 1.5 | |
| `piercing_dark_cloud` | `min_first_body_pct` 60, `require_gap` false, `weight` 2.0 | `require_gap: true` = strict textbook. |
| `star` | `max_star_body_pct` 30, `min_first_body_pct` 60, `require_gap` false, `weight` 3.5 | Morning/evening star. |

A weight of `0` makes a pattern informational only.

---

## 9. Market-trend engines
Neither drives trading directly today: `market_trend.json` feeds `scripts/market_trend.py` and the IBD
strip in `scripts/trend_signals.py`; `trend_structure.json` feeds the `regime_match` filter (off).

### market_trend.json — IBD-style distribution-day / follow-through-day state machine
| Key | Values | Current | Meaning |
|---|---|---|---|
| `benchmark_symbol` | symbol | `"NIFTY500"` | |
| `fall_pct` | % | `0.25` | Down day on higher volume ≥ this = distribution day. |
| `rally_pct` | % | `1.25` | Rally-attempt threshold. |
| `ftd_pct` | % | `1.0` | Follow-through-day gain. |
| `vd_window_calendar_days` | calendar days | `30` | Distribution days counted in this window. |
| `neutral_vd_count` / `downtrend_vd_count` | integer | `4` / `8` | Counts that move UPTREND → NEUTRAL → DOWNTREND. |
| `breach_check_trading_days` / `ftd_search_trading_days` | sessions | `3` / `7` | |
| `initial_state` | `"UPTREND"`/`"NEUTRAL"`/`"DOWNTREND"` | `"NEUTRAL"` | |

### trend_structure.json — swing-structure trend (HH+HL / LH+LL)
| Key | Values | Current | Meaning |
|---|---|---|---|
| `benchmark_symbol` | symbol | `"NIFTY500"` | |
| `timeframe` | daily/weekly/monthly | `"daily"` | |
| `pivot_left` / `pivot_right` | integer | `2` / `2` | |
| `min_confirmed_swings` | integer | `2` | |
| `initial_state` | `"UPTREND"`/`"SIDEWAYS"`/`"DOWNTREND"` | `"SIDEWAYS"` | |
| `weekly_rule` / `monthly_rule` | rules | `"W-FRI"` / `"ME"` | |

---

## 10. Data files
Data, not tuning knobs — edit when the market data changes.

| File | Shape | Example | Notes |
|---|---|---|---|
| `universe.json` | `instruments: {SYMBOL: "NSE_EQ\|ISIN"}`, `lists: {name: [SYMBOL, …]}` | `"TCS": "NSE_EQ\|INE467B01029"`; `"lists": {"default": ["360ONE", …]}` | 514 instruments incl. indices (`"NIFTY500": "NSE_INDEX\|Nifty 500"`). A strategy's `universe` names a list. Rebuild with `scripts/build_universe.py`. Breadth uses today's list → survivorship bias in early years. |
| `sector_map.json` | `symbol_to_industry: {SYMBOL: industry}` | `"ABB": "Capital Goods"` | Industries must match `sector_regime.industry_to_benchmark`. Written by `build_universe.py`. |
| `holidays.json` | `holidays: ["YYYY-MM-DD", …]` | `"2026-01-26"` | NSE trading holidays (weekends are automatic). Currently holds **2026 only** (16 dates). **Add 2027 before 26 Jan 2027** — `eod`/`open` treat a missing holiday as a trading day. `scripts/fetch_holidays.py`. |
| `corporate_actions.json` | `events: [{symbol, date, type, note}]` | `{"symbol": "VEDL", "date": "2026-04-30", "type": "demerger", "note": "4-way demerger ex-date"}` | `type`: `demerger`, `spinoff`, `split`, … Only types in `fractal_sr.json → corporate_action_truncate_types` cut history. |

---

## 11. Recipes

**Switch on the Nov 2026 top pick** (equity sizing + breadth cap 100/80): see the example in
[section 2](#2-risk_configjson). Check: a 2021-01-01 → 2026-11-13 backtest should end at ₹60,44,803.

**Fixed 80 % gross cap instead** (simpler, no signal):
```json
"capital":   { "mode": "equity", ... },
"portfolio": { "max_gross_exposure_pct": 80, "regime_gross_exposure": { "enabled": false, ... } }
```

**Most cautious** (fixed sizing, max 10 positions — ~₹11 L in the market however large equity grows):
```json
"capital":   { "mode": "fixed", ... },
"portfolio": { "max_positions": 10, ... }
```

**Run one strategy in isolation:**
```powershell
python -m trader backtest --strategy breakout_trend --start 2021-01-01 --end 2026-11-13 --label breakout_only
```

**Try a filter as a hard gate for one run:**
```powershell
python scripts\regression_check.py --as-is --start 2021-01-01 --end 2026-11-13 --label rs_gate --set rs_rank.mode=filter
```

**Compare runs:** `python scripts\rank_runs.py --pattern "state\regression_2021-01-01_2026-11-13_*" --baseline f_base`
(add `--segment 2024-07-01` to score only the weak period). Full history of tested settings: the
"wrsi_trader backtest run ledger" sheet and `claude/capital-preservation-plan-2026-11-17.md`.
