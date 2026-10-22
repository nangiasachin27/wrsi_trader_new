from __future__ import annotations

from datetime import date

from trader.core.indicators import atr, sma
from trader.core.models import Position, Signal
from trader.strategies._swing_common import chandelier_trail, ratchet
from trader.strategies.base import Strategy, SymbolContext, register

# Market-trend tag: UPTREND only - this is specifically the course's trend-continuation use of
# an inside bar (ep.12's third strategy: inside bar as a pullback-pause marker), not the
# reversal use (that belongs at a swing/S/R level and is closer in spirit to sr_reversal).


@register("inside_bar_breakout")
class InsideBarBreakout(Strategy):
    """Daily-bar inside-bar trend continuation (course ep.12): in a strong uptrend (close
    above a trend SMA), a pullback that stalls into an inside bar (yesterday's high/low both
    inside the candle before it) is the entry trigger - go long on today's close breaking
    above the inside bar's high. Prefers SMALL-range inside bars (tighter stop, and low
    volatility tends to precede an expansion move) via `max_range_atr_mult`.

    Stop sits ~1 ATR beyond the trend SMA (per the course: the market is expected to find
    support there), not just below the inside bar's own low - a plain inside-bar low stop is
    exactly what a Hikkake-style false breakout would hit first.

    This is a short-horizon expansion play by construction (a SMALL-range inside bar is coiled
    volatility, not a multi-month trend thesis), so the trail is a TIGHT Chandelier Exit
    (shorter ATR lookback, smaller multiple than breakout_trend's) - the goal is locking in the
    quick expansion leg once it starts, not giving it room to ride for months."""

    def entry_signal(self, ctx: SymbolContext) -> Signal | None:
        c = self.cfg["entry"]
        d = ctx.daily
        if len(d) < max(c["min_bars"], 3):
            return None

        trend_sma = sma(d["close"], c["trend_sma_period"]).iloc[-1]
        if trend_sma != trend_sma:
            return None
        trend_sma = float(trend_sma)
        px = ctx.last_close
        if px <= trend_sma:
            return None

        mother_high, mother_low = float(d["high"].iloc[-3]), float(d["low"].iloc[-3])
        ib_high, ib_low = float(d["high"].iloc[-2]), float(d["low"].iloc[-2])
        is_inside = ib_high <= mother_high and ib_low >= mother_low
        if not is_inside:
            return None

        a = atr(d["high"], d["low"], d["close"], c["atr_period"]).iloc[-1]
        if a != a:
            return None
        a = float(a)
        if c.get("prefer_small_range", True) and (ib_high - ib_low) > a * c["max_range_atr_mult"]:
            return None

        if px <= ib_high * (1 + c["breakout_buffer_pct"] / 100.0):
            return None  # today hasn't actually closed back above the inside bar's high yet

        stop = trend_sma - a * c["atr_mult"]
        if stop >= px:
            return None

        return Signal(ctx.symbol, self.name, "ENTRY", ctx.asof.isoformat(),
                      f"close {px:.2f} broke inside bar high {ib_high:.2f} (range {ib_high - ib_low:.2f}) "
                      f"with the trend, above SMA{c['trend_sma_period']} {trend_sma:.2f}",
                      score=0, stop=stop, ref_price=px,
                      meta={"inside_bar_high": ib_high, "inside_bar_low": ib_low})

    def stop_level(self, ctx: SymbolContext, pos: Position) -> float | None:
        x = self.cfg["exit"]
        return ratchet(pos.stop, chandelier_trail(ctx.daily, pos.opened, x["atr_period"], x["atr_mult"]))

    def exit_signal(self, ctx: SymbolContext, pos: Position) -> Signal | None:
        if pos.opened and ctx.asof < date.fromisoformat(pos.opened):
            return None
        x = self.cfg["exit"]
        trail = chandelier_trail(ctx.daily, pos.opened, x["atr_period"], x["atr_mult"])
        if trail is None:
            return None
        px = float(ctx.daily["close"].iloc[-1])
        level = max(trail, pos.stop) if pos.stop else trail
        if px <= level * (1 - x["buffer_pct"] / 100.0):
            return Signal(ctx.symbol, self.name, "EXIT", ctx.asof.isoformat(),
                          f"close {px:.2f} <= chandelier trail {level:.2f} "
                          f"(highest close since entry - {x['atr_mult']}x ATR{x['atr_period']})",
                          ref_price=px, stop=level)
        return None

    def add_signal(self, ctx: SymbolContext, pos: Position) -> Signal | None:
        return None