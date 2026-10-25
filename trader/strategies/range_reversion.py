from __future__ import annotations

from datetime import date

from trader.core.candlestick import detect_patterns, net_sentiment
from trader.core.indicators import bollinger_bands, efficiency_ratio
from trader.core.models import Position, Signal
from trader.strategies._swing_common import atr_buffer, candlestick_cfg
from trader.strategies.base import Strategy, SymbolContext, register

# Market-trend tag: SIDEWAYS (trader.core.trend_structure's own word for it - see module note in
# config/filters/regime_match.json). Long-only here (mirrors the rest of this codebase), so it
# targets a bounce off the LOWER band specifically.


@register("range_reversion")
class RangeReversion(Strategy):
    """Daily-bar statistical range-fade strategy - the backlog item from the post-2023
    edge-decay diagnosis (strategy_performance_log / efficiency-ratio work): a genuinely
    different mean-reversion mechanism from sr_reversal, deliberately NOT built on
    trader.core.levels' pivot-clustered S/R zones. Where sr_reversal fades a specific
    *structural* level (a cluster of prior swing highs/lows), this strategy fades a
    *statistical* extreme - price closing below a Bollinger Band (SMA +/- std-dev) - and is
    the only strategy in this codebase whose own entry logic directly measures and requires low
    trendiness, rather than relying on an external regime tag or filter to keep it out of
    trending names.

    The low-efficiency requirement is a HARD GATE inside entry_signal itself: Kaufman
    Efficiency Ratio (trader.core.indicators.efficiency_ratio) on THIS SYMBOL's own daily
    closes - not the market-wide benchmark the `market_efficiency` filter reads - must sit at
    or below `max_efficiency`, or there is no trade at all, full stop. Per-symbol (not
    market-wide) on purpose: whether THIS stock is actually range-bound right now is what makes
    a band-fade trade valid, independent of what the broader index is doing. This makes the
    `market_efficiency` filter unnecessary for this strategy (it already self-gates more
    strictly than that filter's own 'rank'/'filter' modes), so it is deliberately left out of
    that filter's `applicable_strategies` map.

    Entry requires: (a) efficiency ratio <= max_efficiency (the tape is genuinely choppy, not
    trending in either direction - efficiency_ratio uses the ABSOLUTE net move, so this also
    excludes a strong, choppy downtrend, not just excluding strong uptrends), (b) a pierce of
    the lower Bollinger Band on the day's low, (c) a close back above that band (rejection, not
    a breakdown), (d) a bullish reversal candle (hammer / bullish engulfing / morning star /
    bullish marubozu - same registry as pullback_ma/sr_reversal) confirming the bounce.

    Like sr_reversal, this is a static TARGET trade, not a trailing one: stop is set once at
    entry (below the band pierce, ATR-buffered) and never trailed; the profit target is the
    band's own middle line (the SMA) - "reversion to the mean" is the whole trade thesis, so the
    exit is the mean itself, not the opposite band. Exempt from `quality`/`resistance` (see their
    own exempt_strategies) for the same reason sr_reversal is: both filters are built for
    trend/breakout setups and structurally fight a range-bound entry signature."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._candle_cfg = candlestick_cfg(cfg.get("candlestick_config_path", "config/candlestick_patterns.json"))

    def entry_signal(self, ctx: SymbolContext) -> Signal | None:
        c = self.cfg["entry"]
        d = ctx.daily
        if len(d) < c["min_bars"]:
            return None

        er = efficiency_ratio(d["close"], c["efficiency_period"]).iloc[-1]
        if er != er or float(er) > c["max_efficiency"]:
            return None  # too trendy (or not enough history to tell) for a range trade
        er = float(er)

        lower, mid, _ = bollinger_bands(d["close"], c["bb_period"], c["bb_std"])
        lo, md = lower.iloc[-1], mid.iloc[-1]
        if lo != lo or md != md:
            return None
        lo, md = float(lo), float(md)

        px = ctx.last_close
        low_today = float(d["low"].iloc[-1])
        pierced = low_today <= lo * (1 + c["proximity_pct"] / 100.0)
        rejected = px > lo
        if not (pierced and rejected):
            return None

        matches = detect_patterns(d, len(d) - 1, self._candle_cfg)
        bullish = [m for m in matches if m.direction == "bullish"]
        if not bullish or net_sentiment(matches) <= 0:
            return None

        buf = atr_buffer(d, c["atr_period"], c["atr_mult"])
        stop = min(low_today, lo) - buf
        if stop >= px:
            return None

        names = ", ".join(m.name for m in bullish)
        return Signal(ctx.symbol, self.name, "ENTRY", ctx.asof.isoformat(),
                      f"rejection of lower Bollinger Band {lo:.2f} (mid {md:.2f}) by {names}, "
                      f"efficiency ratio {er:.2f} <= {c['max_efficiency']} (range-bound)",
                      score=0, stop=stop, ref_price=px,
                      meta={"bb_lower": lo, "bb_mid": md, "efficiency_ratio": er, "patterns": names})

    def stop_level(self, ctx: SymbolContext, pos: Position) -> float | None:
        return None  # static target trade - keep whatever stop was set at entry, no trailing

    def exit_signal(self, ctx: SymbolContext, pos: Position) -> Signal | None:
        if pos.opened and ctx.asof < date.fromisoformat(pos.opened):
            return None
        d = ctx.daily
        px = float(d["close"].iloc[-1])
        if px <= pos.stop:
            return Signal(ctx.symbol, self.name, "EXIT", ctx.asof.isoformat(),
                          f"close {px:.2f} <= stop {pos.stop:.2f}", ref_price=px, stop=pos.stop)

        x = self.cfg["exit"]
        c = self.cfg["entry"]
        _, mid, _ = bollinger_bands(d["close"], c["bb_period"], c["bb_std"])
        md = mid.iloc[-1]
        if md == md and px >= float(md) * (1 - x["target_tolerance_pct"] / 100.0):
            return Signal(ctx.symbol, self.name, "EXIT", ctx.asof.isoformat(),
                          f"close {px:.2f} reached mid Bollinger Band {float(md):.2f} (mean-reversion target)",
                          ref_price=px, stop=pos.stop)
        return None

    def add_signal(self, ctx: SymbolContext, pos: Position) -> Signal | None:
        return None  # no pyramiding into a mean-reversion trade - the thesis is one bounce back to the mean