# Swing Strategies Drop — apply into wrsi_trader_new

No push access to the repo, so this is delivered as files to copy in manually.
Every path below is relative to your repo root — copy each file over the
matching path (new files are created, the 3 "modified" files below REPLACE
the existing ones, don't merge by hand).

## New files
- trader/strategies/_swing_common.py      (shared helpers: levels_cfg, candlestick_cfg,
                                            trailing_level, ratchet, atr_buffer, nearest_zones)
- trader/strategies/breakout_trend.py     (tag: UPTREND)
- trader/strategies/pullback_ma.py        (tag: UPTREND)
- trader/strategies/sr_reversal.py        (tag: SIDEWAYS)
- trader/strategies/inside_bar_breakout.py (tag: UPTREND)
- trader/filters/regime_match.py          (enforces the tags above against
                                            trader.core.trend_structure's state — see below)
- trader/core/trend_structure.py          (NEW: course-based swing-structure trend classifier)
- scripts/trend_structure.py              (NEW: CLI to inspect it against real cached data,
                                            same shape as your existing scripts/market_trend.py)
- tests/test_trend_structure.py           (NEW: 11 deterministic unit tests for it)
- config/strategies/breakout_trend.json
- config/strategies/pullback_ma.json
- config/strategies/sr_reversal.json
- config/strategies/inside_bar_breakout.json
- config/filters/regime_match.json
- config/trend_structure.json

## Modified (replace in place)
- trader/strategies/base.py   (registers the 4 new strategies in build_strategy())
- trader/filters/base.py      (registers regime_match in build_filter())
- trader/cli.py               (adds --strategy flag, see below)
- config/risk_config.json     (adds allocation_pct entries for the 4 new strategy names
                                — all set to 100, same as weekly_rsi_sma20; rebalance if you
                                ever enable more than one strategy at once alongside it)

All 4 new strategies default to `"enabled": false` in their own config file, so
nothing changes for your existing default-config backtests until you opt in.

## Running them in isolation, to compare

New `--strategy` CLI flag overrides which strategy configs are enabled for one run,
without writing back to the JSON files:

    python -m trader.cli backtest --strategy breakout_trend \
        --start 2020-01-01 --end 2024-12-31 --label breakout_trend_only

    python -m trader.cli backtest --strategy pullback_ma \
        --start 2020-01-01 --end 2024-12-31 --label pullback_ma_only

    python -m trader.cli backtest --strategy sr_reversal \
        --start 2020-01-01 --end 2024-12-31 --label sr_reversal_only

    python -m trader.cli backtest --strategy inside_bar_breakout \
        --start 2020-01-01 --end 2024-12-31 --label inside_bar_breakout_only

(Confirm the exact flag names your cli.py's backtest subcommand expects for
start/end/label — I added only --strategy; the rest already existed.)

You can pass a comma-separated list (`--strategy breakout_trend,pullback_ma`) to run
several together, or omit the flag entirely to use whatever each config's own
`enabled` value says (unchanged behavior).

## Market trend, rebuilt per the course (trader/core/trend_structure.py)

You said you weren't confident in the existing `trader/core/market_trend.py` — that module
is an IBD-style "volume distribution day / follow-through day" mechanism, which is **not**
what the course teaches. I left it completely untouched (still there, still used by
`scripts/market_trend.py`, still importable) and built a **separate, new module** instead:
`trader/core/trend_structure.py`, implementing the course's own method:

- **UPTREND** = 2+ consecutive higher highs **and** higher lows (confirmed only after 2+
  up-swings, per the course).
- **DOWNTREND** = the mirror — 2+ consecutive lower highs and lower lows.
- **SIDEWAYS** = everything else, including a confirmed trend's earliest warning sign: a
  close breaking the most recent protecting swing low (uptrend) or swing high (downtrend) —
  "break of market structure" drops straight to SIDEWAYS, not to the opposite trend (that
  needs its own fresh 2+ swing confirmation, exactly as the course describes it as two
  separate events).
- Swing highs/lows are detected with the **same** `pivot_highs`/`pivot_lows` functions your
  S/R zone builder (`trader/core/levels.py`) already uses and is already unit-tested —
  reused, not reimplemented, so "swing point" means the same thing to both building blocks.
- The protecting level only ever **ratchets** in the trend's favor, same tighten-only
  discipline as every trailing stop elsewhere in this codebase.

Try it against your real cached NIFTY500 data the same way you visually sanity-checked the
old engine:

    python scripts/trend_structure.py --history 15

On the real data in your cache right now it transitions fairly often (roughly every 1–3
weeks over the last 6 months) with the default tight `pivot_left=2, pivot_right=2` daily
window — that's the course's method working as literally specified, but it may be noisier
than you want for a general market-regime gate. If so, the easiest knobs in
`config/trend_structure.json` are: widen `pivot_left`/`pivot_right` (fewer, more significant
swings), or set `"timeframe": "weekly"` (same convention `config/levels.json` already uses
for S/R) to classify off weekly swings instead of daily ones.

`regime_match` now reads this new engine instead of the old one:

    "applicable_regimes": {
      "breakout_trend":        ["UPTREND"],
      "pullback_ma":           ["UPTREND"],
      "inside_bar_breakout":   ["UPTREND"],
      "sr_reversal":           ["SIDEWAYS"]
    }

- UPTREND / SIDEWAYS / DOWNTREND are trader.core.trend_structure's states, the course's own
  words (no more NEUTRAL-means-Sideways mapping needed).
- weekly_rsi_sma20 has no entry in this map, so it is NOT gated by regime at all
  (opt-in design — untagged strategies always pass this filter unchanged).
- Re-tagging a strategy later is a one-line edit to this JSON, not a code change.
- fail_mode: "block" (default) means if the benchmark data needed to compute the regime
  isn't available on a given day, tagged strategies are blocked that day rather than
  trading blind; set to "allow" to fail open instead.

## Testing each building block individually

You asked for market trend, candlesticks and S/R to each be testable on their own, since
nearly every strategy leans on them. Here's where each one's tests live:

- **Market trend** (new): `tests/test_trend_structure.py` — 11 tests, all against hand-built
  OHLC sequences with known, exact turning points (a "zigzag" helper that walks linearly
  through a list of swing prices), asserting: uptrend/downtrend confirmation after exactly 2
  swings, staying SIDEWAYS through an equal-highs/equal-lows range, break-of-structure
  dropping an uptrend to SIDEWAYS (never straight to DOWNTREND), the protecting level
  ratcheting up and never loosening, no-lookahead on the last `pivot_right` bars, the weekly-
  timeframe path, and the empty-history edge case.
- **Support/Resistance** (already existed, untouched): `tests/test_levels.py` — pivot
  detection, zone clustering, nearest-support/resistance picking, the timeframe wrapper.
- **Candlesticks** (already existed, untouched): `tests/test_candlestick.py` — every pattern
  detector (marubozu, doji, hammer family incl. trend-context labeling, engulfing, harami,
  piercing/dark-cloud, morning/evening star), `net_sentiment`, and the timeframe wrapper.

Run any one in isolation, e.g. `python -m pytest tests/test_trend_structure.py -v`, to check
that specific building block without running the whole suite.

## Strategy summaries (course mapping)

- **breakout_trend** — close breaks above a pivot-clustered daily resistance ZONE
  (not just a prior high), confirmed by a full-candle close beyond the zone's near
  edge + above-average volume, while already above a trend SMA. Stop sits just below
  the broken zone, ATR-buffered; trails up on a faster SMA (ratchet-only).
- **pullback_ma** — in an established uptrend (close > trend SMA), price pulls back
  to touch a faster MA, prints a bullish reversal candle there, then closes back above
  the MA. Stop = MA − ATR buffer; trails the same MA.
- **sr_reversal** — price dips into (wicks below) a daily support ZONE and closes back
  above it with a bullish reversal candle (the course's "fade the level" setup, hence
  tagged SIDEWAYS rather than trending). Fixed target at the next resistance
  zone; stop = min(day's low, zone's near edge) − ATR buffer; no trailing (static
  target trade).
- **inside_bar_breakout** — an inside bar (lower high/higher low than the prior "mother"
  candle) inside an uptrend, breakout confirmed on a close beyond the inside bar's own
  high. Stop = trend SMA − ATR buffer; trails a configurable SMA.

## ⚠️ Separate, pre-existing issue found while testing — not something I introduced today

`trader/filters/resistance.py` in your working tree already had an **uncommitted**
rewrite in it before I started today's work (confirmed via `git diff` — it replaced
the old single-rolling-max-high check with the new zone-based `support_resistance`/
`nearest_levels` logic). I didn't touch that file, but while smoke-testing the new
strategies I found it is now blocking almost every single trade, including on your
existing `weekly_rsi_sma20`:

- On the repo's own synthetic test fixture (`tests/conftest.py`), running the full
  backtest with resistance.py as-is in your tree produces **0 trades**. Reverting
  *only* that one file (`git checkout -- trader/filters/resistance.py`, i.e. back to
  its last committed version) on the exact same data/dates immediately restores
  **41 trades** — nothing else changed.
- `pytest tests/` currently shows 14 failing / 56 passing. I bisected all 14: 10 are
  pre-existing and unrelated (config/CLI edge cases, an Upstox env-var test), and the
  other 4 are caused solely by this resistance.py rewrite. None of the 14 are caused
  by anything I added today (the 4 new strategies, regime_match, or the --strategy flag).
- Sampling the new zone logic directly (synthetic data, weekly timeframe, every 5th
  bar from day 300 onward): out of 394 samples, 141 found no resistance zone at all,
  and 103 of the remaining 253 sat within the configured 3% "too close to break out"
  band. On its own that's a plausible, not absurd, block rate — the near-total
  collapse in the full backtest looks like it's `resistance` and the existing
  `quality` filter *compounding* (between them blocking almost every raw candidate
  that reaches them), rather than one filter being flatly broken.

I haven't touched this file and didn't try to fix it — it's unrelated to today's
scope and I didn't want to silently patch code you were already mid-editing. But
it will make your planned comparative backtests (including on the strategies in
this drop) show close to zero trades across the board until it's addressed, so
flagging it now rather than after you run the comparisons. Worth deciding whether
to revert it, loosen the proximity/clustering thresholds in `config/levels.json`,
or dig into it further — happy to help with whichever direction you pick.
