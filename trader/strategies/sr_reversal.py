from __future__ import annotations

from datetime import date

from trader.core.candlestick import detect_patterns, net_sentiment
from trader.core.models import Position, Signal
from trader.strategies._swing_common import atr_buffer, candlestick_cfg, levels_cfg, nearest_zones, ratchet
from trader.strategies.base import Strategy, SymbolContext, register

# Market-trend tag: SIDEWAYS (trader.core.trend_structure's own word for it - see module
# note in config/filters/regime_match.json) - a fade-the-level reversal trade wants a market that
# is ranging/consolidating, not already trending hard against the zone. Long-only here
# (mirrors the rest of this codebase), so it targets a bounce off SUPPORT specifically.


@register("sr_reversal")
class SrReversal(Strategy):
    """Daily-bar support-zone reversal (course ep.6): wants a real momentum move DOWN into a
    daily support zone (trader.core.levels), followed by a strong bullish rejection candle
    there (hammer / bullish engulfing / morning star / bullish marubozu). Unlike the trend
    strategies, this is a target-based (not trailing-stop) trade - it exits either at the
    next resistance zone (profit target) or on a stop breach below the rejection candle's own
    low, buffered by ATR, per the course's zone-not-line stop placement.

    `momentum_lookback` guards against trading a slow, step-like grind into support (the
    course explicitly prefers a clean momentum thrust over a staircase of swing points for
    this setup) - it's a simple closes-fell check, not a full swing-structure classifier."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._zones_cfg = levels_cfg(cfg.get("levels_timeframe", "daily"), cfg.get("levels_config_path", "config/levels.json"))
        self._candle_cfg = candlestick_cfg(cfg.get("candlestick_config_path", "config/candlestick_patterns.json"))

    def entry_signal(self, ctx: SymbolContext) -> Signal | None:
        c = self.cfg["entry"]
        if len(ctx.daily) < c["min_bars"]:
            return None

        px = ctx.last_close
        zones = nearest_zones(ctx.daily, self.cfg.get("levels_timeframe", "daily"), self._zones_cfg, px)
        sup = zones["support"]
        if sup is None:
            return None

        low_today = float(ctx.daily["low"].iloc[-1])
        pierced = low_today <= sup.price_high * (1 + c["proximity_pct"] / 100.0)
        rejected = px > sup.price_low
        if not (pierced and rejected):
            return None

        look = c["momentum_lookback"]
        if len(ctx.daily) < look + 2:
            return None
        fell_into_level = float(ctx.daily["close"].iloc[-2]) < float(ctx.daily["close"].iloc[-2 - look])
        if not fell_into_level:
            return None

        matches = detect_patterns(ctx.daily, len(ctx.daily) - 1, self._candle_cfg)
        bullish = [m for m in matches if m.direction == "bullish"]
        if not bullish or net_sentiment(matches) <= 0:
            return None

        buf = atr_buffer(ctx.daily, c["atr_period"], c["atr_mult"])
        stop = min(low_today, sup.price_low) - buf
        if stop >= px:
            return None

        names = ", ".join(m.name for m in bullish)
        return Signal(ctx.symbol, self.name, "ENTRY", ctx.asof.isoformat(),
                      f"rejection of daily support zone {sup.price_low:.2f}-{sup.price_high:.2f} "
                      f"(strength {sup.strength}) by {names} after a fall into the level",
                      score=0, stop=stop, ref_price=px,
                      meta={"zone_low": sup.price_low, "zone_high": sup.price_high, "patterns": names})

    def stop_level(self, ctx: SymbolContext, pos: Position) -> float | None:
        return None  # static target trade - keep whatever stop was set at entry, no trailing

    def exit_signal(self, ctx: SymbolContext, pos: Position) -> Signal | None:
        if pos.opened and ctx.asof < date.fromisoformat(pos.opened):
            return None
        px = float(ctx.daily["close"].iloc[-1])
        if px <= pos.stop:
            return Signal(ctx.symbol, self.name, "EXIT", ctx.asof.isoformat(),
                          f"close {px:.2f} <= stop {pos.stop:.2f}", ref_price=px, stop=pos.stop)

        x = self.cfg["exit"]
        zones = nearest_zones(ctx.daily, self.cfg.get("levels_timeframe", "daily"), self._zones_cfg, px)
        res = zones["resistance"]
        if res is not None and px >= res.price_low * (1 - x["target_tolerance_pct"] / 100.0):
            return Signal(ctx.symbol, self.name, "EXIT", ctx.asof.isoformat(),
                          f"close {px:.2f} reached next resistance zone {res.price_low:.2f}-{res.price_high:.2f}",
                          ref_price=px, stop=pos.stop)
        return None

    def add_signal(self, ctx: SymbolContext, pos: Position) -> Signal | None:
        return None
