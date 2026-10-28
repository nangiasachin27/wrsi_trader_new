from __future__ import annotations

import logging
from datetime import date

from trader.core.candlestick import net_sentiment, patterns_for
from trader.core.models import Signal
from trader.filters.base import Filter, FilterResult, register
from trader.strategies._swing_common import candlestick_cfg
from trader.strategies.base import SymbolContext

log = logging.getLogger(__name__)


@register("candlestick")
class CandlestickFilter(Filter):
    """Confirms the DECIDING candle itself is a genuine bullish candle, not just a technically-
    qualifying one - built 2026-10-27 in direct response to a measured finding: on every single
    trading day across a 5.75-year backtest, the candidate pool exceeded free slots (median 43
    qualifying candidates/day), while barely any ENTRY_FILTERED events fired (median 1/day) -
    every other filter here (trend_strength/rs_rank/sector_regime/market_regime) is derived from
    the SAME underlying trend/relative-return data the core strategy rule already used to flag
    the candidate, so none of them narrow the pool much; they mostly just re-ask the same
    question. Candle shape is a genuinely independent signal none of those look at.

    `sr_reversal` already has an equivalent, STRICTER check hard-coded into its own core
    entry_signal() (net_sentiment(matches) <= 0 -> no signal, built 2026-10-21/24, well before
    this filter existed) - it should stay excluded from this filter via its own exclude_filters
    rather than being double-gated or re-scored by a second, more permissive copy of the same
    idea. This filter is for the other 4 strategies, which have no candle-shape check at all
    today.

    Reads the DAILY bars but checks a per-strategy TIMEFRAME, because strategies decide on
    different bars: weekly_rsi_sma20's own entry fires off a completed WEEKLY bar, so what
    matters is whether THAT weekly candle is bullish, not today's daily candle - set via that
    strategy's own `candlestick_timeframe` config key (read through `_strategy_cfgs`, same
    per-strategy-parameter mechanism market_efficiency_override uses - see base.py's class
    docstring). Defaults to `default_timeframe` (daily) for every strategy that doesn't override
    it - breakout_trend/new_high_momentum/pullback_ma all decide off the daily close.

    mode 'rank' (default) never excludes - net_sentiment only adds to score for tie-breaking.
    mode 'filter' is a hard gate: net_sentiment below `min_net_sentiment` excludes the candidate
    outright, regardless of free slots - this is the mode that actually shrinks the oversupplied
    candidate pool, which was the point of building this.

    Score: net_sentiment(), continuous, scaled by `score_ceiling` (the net_sentiment value that
    saturates to a full 1.0) and clamped to [0, 1] - same continuous-scoring convention trend_
    strength/rs_rank were just moved to on 2026-10-27, so a strong marubozu+engulfing combo
    outranks a single weak doji-adjacent candle instead of tying with it.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._pattern_cfg: dict | None = None

    def _patterns_cfg(self) -> dict:
        if self._pattern_cfg is None:
            try:
                self._pattern_cfg = candlestick_cfg(self.cfg.get("pattern_cfg_path", "config/candlestick_patterns.json"))
            except Exception as exc:  # noqa: BLE001 - one bad config file shouldn't crash the backtest
                log.warning("candlestick: could not load pattern config: %s", exc)
                self._pattern_cfg = {}
        return self._pattern_cfg

    def describe(self) -> str:
        c = self.cfg
        return (f"mode={c.get('mode', 'rank')}, default_timeframe={c.get('default_timeframe', 'daily')}, "
                f"min_net_sentiment={c.get('min_net_sentiment', 0.0)}, score_ceiling={c.get('score_ceiling', 5.0)}")

    def evaluate(self, signal: Signal, ctx: SymbolContext) -> FilterResult:
        if not self.enabled:
            return FilterResult(True, "disabled")

        override = self._strategy_cfgs.get(signal.strategy, {})
        timeframe = override.get("candlestick_timeframe", self.cfg.get("default_timeframe", "daily"))

        matches = patterns_for(ctx.daily, timeframe, self._patterns_cfg())
        ns = net_sentiment(matches)
        names = ", ".join(f"{m.name}({m.direction})" for m in matches) or "no pattern matched"

        min_ns = float(self.cfg.get("min_net_sentiment", 0.0))
        mode = self.cfg.get("mode", "rank")
        passed = not (mode == "filter" and ns < min_ns)

        ceiling = float(self.cfg.get("score_ceiling", 5.0))
        score = max(0.0, min(1.0, ns / ceiling)) if ceiling > 0 else 0.0

        reason = f"{timeframe} candle net_sentiment {ns:+.1f} ({names})"
        return FilterResult(passed, reason, score=score, meta={"net_sentiment": round(ns, 2), "timeframe": timeframe})

    def max_score(self) -> float:
        """Continuous net_sentiment/score_ceiling ratio, clamped to [0, 1] in evaluate() - 1.0
        whenever enabled, 0 otherwise. See Filter.max_score()'s own docstring."""
        return 1.0 if self.enabled else 0.0
