from __future__ import annotations

from datetime import date

from trader.core.candlestick import detect_patterns, net_sentiment
from trader.core.indicators import sma
from trader.core.models import Position, Signal
from trader.strategies._swing_common import atr_buffer, candlestick_cfg, ratchet, trailing_level
from trader.strategies.base import Strategy, SymbolContext, register

# Market-trend tag: UPTREND only - a pullback trade is, by definition, an entry WITH an
# existing trend on its retracement; there is no "pullback" to trade without one.


@register("pullback_ma")
class PullbackMA(Strategy):
    """Daily-bar moving-average pullback (course ep.10): in an established uptrend (close
    above a slower trend SMA), wait for price to retrace down INTO a faster "pullback" MA and
    show a bullish rejection candle there (hammer / bullish engulfing / morning star / bullish
    marubozu, via trader.core.candlestick's existing pattern registry) before entering - never
    just buying the dip on touch alone.

    Stop sits below the pullback MA with an ATR buffer (course note: price often overshoots an
    MA before truly reversing, so the buffer needs real room). Exit trails the SAME pullback MA
    - once price is riding the trend again, a close back through that MA is the course's own
    "market structure changed" signal."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._candle_cfg = candlestick_cfg(cfg.get("candlestick_config_path", "config/candlestick_patterns.json"))

    def entry_signal(self, ctx: SymbolContext) -> Signal | None:
        c = self.cfg["entry"]
        if len(ctx.daily) < c["min_bars"]:
            return None

        trend_sma = sma(ctx.daily["close"], c["trend_sma_period"]).iloc[-1]
        if trend_sma != trend_sma:
            return None
        px = ctx.last_close
        if px <= float(trend_sma):
            return None  # not in an uptrend by the slower SMA - no trend to pull back within

        ma = sma(ctx.daily["close"], c["pullback_sma_period"]).iloc[-1]
        if ma != ma:
            return None
        ma = float(ma)

        low_today = float(ctx.daily["low"].iloc[-1])
        touched_ma = low_today <= ma * (1 + c["touch_tolerance_pct"] / 100.0)
        if not touched_ma:
            return None

        matches = detect_patterns(ctx.daily, len(ctx.daily) - 1, self._candle_cfg)
        bullish = [m for m in matches if m.direction == "bullish"]
        if not bullish or net_sentiment(matches) <= 0:
            return None
        if px <= ma:
            return None  # the rejection candle itself must already close back above the MA

        buf = atr_buffer(ctx.daily, c["atr_period"], c["atr_mult"])
        stop = ma - buf
        if stop >= px:
            return None

        names = ", ".join(m.name for m in bullish)
        return Signal(ctx.symbol, self.name, "ENTRY", ctx.asof.isoformat(),
                      f"pullback to SMA{c['pullback_sma_period']} {ma:.2f} rejected by {names}, "
                      f"close {px:.2f} back above it, trend SMA{c['trend_sma_period']} intact",
                      score=0, stop=stop, ref_price=px, meta={"pullback_sma": ma, "patterns": names})

    def stop_level(self, ctx: SymbolContext, pos: Position) -> float | None:
        x = self.cfg["exit"]
        return ratchet(pos.stop, trailing_level(ctx.daily["close"], x["trail_sma_period"]))

    def exit_signal(self, ctx: SymbolContext, pos: Position) -> Signal | None:
        if pos.opened and ctx.asof < date.fromisoformat(pos.opened):
            return None
        x = self.cfg["exit"]
        trail = trailing_level(ctx.daily["close"], x["trail_sma_period"])
        if trail is None:
            return None
        px = float(ctx.daily["close"].iloc[-1])
        level = max(trail, pos.stop) if pos.stop else trail
        if px <= level * (1 - x["buffer_pct"] / 100.0):
            return Signal(ctx.symbol, self.name, "EXIT", ctx.asof.isoformat(),
                          f"close {px:.2f} <= SMA{x['trail_sma_period']} {level:.2f} (market structure change)",
                          ref_price=px, stop=level)
        return None

    def add_signal(self, ctx: SymbolContext, pos: Position) -> Signal | None:
        return None
