from __future__ import annotations

from datetime import date

from trader.core.indicators import sma
from trader.core.models import Position, Signal
from trader.strategies._swing_common import atr_buffer, levels_cfg, nearest_zones, ratchet, trailing_level
from trader.strategies.base import Strategy, SymbolContext, register

# Market-trend tag (see config/filters/regime_match.json): UPTREND only - the course is
# explicit that a breakout should only ever be traded WITH the prevailing trend. Taking this
# setup in a SIDEWAYS/DOWNTREND tape is exactly the "counter-trend breakout" the source material
# warns produces only short, fast-reversing moves.


@register("breakout_trend")
class BreakoutTrend(Strategy):
    """Daily-bar breakout-near-structure strategy (course ep.8): price closes above the
    nearest daily resistance ZONE (trader.core.levels - the same pivot-clustered zones
    scripts/levels.py and the resistance filter read), confirmed by (a) a candle CLOSE beyond
    the zone, not just a wick poke-through, (b) price already above a slower trend SMA, and
    optionally (c) above-average volume - the course's two fastest, cheapest fakeout filters
    (close-confirmation + volume) short of waiting for a full retest.

    Stop sits just below the broken zone's LOWER edge (not its midpoint), buffered by ATR -
    "trading near structure" is the whole point of this setup over a bare swing-high breakout:
    the zone's own lower edge is the tightest logical invalidation level. Once in the trade,
    the stop only trails up via a faster SMA (ratchet-only, never loosens) - riding the trend
    per the course's 'exit on a close below the trail, or when market structure breaks' rule,
    approximated here by the trailing SMA alone (no explicit swing-structure exit yet)."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._zones_cfg = levels_cfg(cfg.get("levels_timeframe", "daily"), cfg.get("levels_config_path", "config/levels.json"))

    def _trend_ok(self, ctx: SymbolContext) -> tuple[bool, float | None]:
        c = self.cfg["entry"]
        s = sma(ctx.daily["close"], c["trend_sma_period"]).iloc[-1]
        if s != s:
            return False, None
        return float(ctx.last_close) > float(s), float(s)

    def entry_signal(self, ctx: SymbolContext) -> Signal | None:
        c = self.cfg["entry"]
        if len(ctx.daily) < c["min_bars"]:
            return None
        trend_ok, trend_sma = self._trend_ok(ctx)
        if not trend_ok:
            return None

        px = ctx.last_close
        if len(ctx.daily) < 2:
            return None
        # Find the zone that was resistance as of YESTERDAY's close, not today's - asking
        # "nearest resistance above today's price" would be tautologically unbreakable (by
        # construction it's always above today's close), so the zone has to be anchored to
        # where price stood BEFORE today's candle for a breakout above it to even be possible.
        prev_px = float(ctx.daily["close"].iloc[-2])
        zones = nearest_zones(ctx.daily, self.cfg.get("levels_timeframe", "daily"), self._zones_cfg, prev_px)
        res = zones["resistance"]
        if res is None:
            return None
        # Near (lower) edge, same convention the resistance filter itself uses - the first
        # price the stock actually has to clear, not the zone's far edge (which can sit well
        # above the near edge once several loosely-clustered touches widen the band).
        if px <= res.price_low * (1 + c["breakout_buffer_pct"] / 100.0):
            return None

        if c.get("require_volume_confirmation", True):
            vol_sma = sma(ctx.daily["volume"], c["volume_sma_period"]).iloc[-1]
            if vol_sma == vol_sma and float(ctx.daily["volume"].iloc[-1]) < vol_sma * c["volume_mult"]:
                return None

        buf = atr_buffer(ctx.daily, c["atr_period"], c["atr_mult"])
        stop = res.price_low - buf
        if stop >= px:
            return None

        return Signal(ctx.symbol, self.name, "ENTRY", ctx.asof.isoformat(),
                      f"close {px:.2f} broke daily resistance zone {res.price_low:.2f}-{res.price_high:.2f} "
                      f"(strength {res.strength}), above trend SMA{c['trend_sma_period']} {trend_sma:.2f}",
                      score=0, stop=stop, ref_price=px,
                      meta={"zone_low": res.price_low, "zone_high": res.price_high, "zone_strength": res.strength})

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
                          f"close {px:.2f} <= trailing SMA{x['trail_sma_period']} {level:.2f}",
                          ref_price=px, stop=level)
        return None

    def add_signal(self, ctx: SymbolContext, pos: Position) -> Signal | None:
        return None
