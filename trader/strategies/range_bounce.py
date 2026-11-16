from datetime import date

import pandas as pd

from trader.core.fractal_levels import fractal_result
from trader.core.indicators import adx
from trader.core.models import Position, Signal
from trader.strategies._swing_common import atr_buffer
from trader.strategies.base import Strategy, SymbolContext, register


# Market-trend tag: SIDEWAYS (see config/strategies/range_bounce.json `market_regime_tag`).
@register("range_bounce")
class RangeBounce(Strategy):
    """Sideways-market specialist (added 2026-11-16): buy the FLOOR of a qualified horizontal
    range, exit at its ceiling. Motivated by the 2026-11-16 finding that ~42% of recent sessions
    are off-regime (NIFTY500 below weekly SMA20) and every trend strategy runs PF < 1 there.

    How this differs from sr_reversal (the other fade-the-level strategy): sr_reversal fades ANY
    support after a momentum thrust down into it, in any structure - including downtrends.
    range_bounce qualifies the RANGE first and only trades inside it:
      - both walls must be real: nearest support zone >= min_support_touches, nearest resistance
        zone >= min_resistance_touches (fractal zones - trader/core/fractal_levels.py, the same
        engine as scripts/fractal_sr.py; this strategy is zone-native and does NOT follow the
        legacy/fractal sr_engine switch, which exists to preserve old consumers' behaviour);
      - the tape must be flat: daily ADX(adx_period) <= max_adx (trend strategies REQUIRE the
        opposite - trend_strength gates on ADX >= 20);
      - the range must pay: width (res.low - sup.high) >= min_range_width_atr x ATR, and the
        entry's reward:risk to the ceiling >= min_reward_risk (MANDATORY here, unlike
        sr_reversal's optional gate - a range trade's target is known by definition);
      - no momentum-thrust or candlestick-pattern requirement: ranges drift to the floor, they
        don't thrust (the candlestick filter stays available as a rank overlay).
    Exits: ceiling reached (target), range resolved upward (no resistance left above - take the
    profit, this is not a trend ride), engine stop below the floor zone, or a time stop -
    a range trade that goes nowhere for max_holding_days is dead capital, not a winner-in-waiting.
    Static stop (no trailing): the floor invalidates the trade, nothing else does."""

    # ---- entry -------------------------------------------------------------------------------
    def entry_signal(self, ctx: SymbolContext) -> Signal | None:
        c = self.cfg["entry"]
        if len(ctx.daily) < c["min_bars"]:
            return None
        px = ctx.last_close
        r = fractal_result(ctx.daily, self.cfg.get("levels_timeframe", "daily"), px, ctx.asof)
        if not r.sup or not r.res:
            return None
        sup, res = r.sup[0], r.res[0]

        # -- range qualification --
        if sup.edge or res.edge:
            return None                                   # unconfirmed window extremes are not walls
        if sup.touches < c["min_support_touches"] or res.touches < c["min_resistance_touches"]:
            return None
        width_pct = (res.low - sup.high) / px * 100.0
        if width_pct < c["min_range_width_atr"] * r.atr_pct:
            return None
        d = ctx.daily
        a = adx(d["high"], d["low"], d["close"], c["adx_period"]).iloc[-1]
        if pd.isna(a) or float(a) > c["max_adx"]:
            return None                                   # trending tape - wrong strategy

        # -- floor touch + rejection today --
        low_today = float(d["low"].iloc[-1])
        touched = low_today <= sup.high * (1 + c["touch_proximity_pct"] / 100.0)
        back_above = px > sup.high
        bullish = (not c["require_bullish_close"]) or px > float(d["open"].iloc[-1])
        if not (touched and back_above and bullish):
            return None

        # -- stop below the floor zone, mandatory reward:risk to the ceiling --
        stop = sup.low - atr_buffer(d, c["atr_period"], c["atr_mult"])
        if stop >= px:
            return None
        rr = (res.low - px) / (px - stop)
        if rr < c["min_reward_risk"]:
            return None

        return Signal(ctx.symbol, self.name, "ENTRY", ctx.asof.isoformat(),
                      f"bounce off range floor {sup.low:.2f}-{sup.high:.2f} ({sup.touches} touches), "
                      f"ceiling {res.low:.2f}-{res.high:.2f} ({res.touches} touches), "
                      f"ADX {float(a):.1f}, R:R {rr:.2f}",
                      score=0, stop=stop, ref_price=px,
                      meta={"floor_low": round(sup.low, 2), "floor_high": round(sup.high, 2),
                            "ceiling_low": round(res.low, 2), "ceiling_high": round(res.high, 2),
                            "reward_risk": round(rr, 2), "adx": round(float(a), 1),
                            "range_width_pct": round(width_pct, 2)})

    # ---- exits -------------------------------------------------------------------------------
    def stop_level(self, ctx: SymbolContext, pos: Position) -> float | None:
        return None  # static stop: only the floor breaking invalidates the trade

    def exit_signal(self, ctx: SymbolContext, pos: Position) -> Signal | None:
        if pos.opened and ctx.asof < date.fromisoformat(pos.opened):
            return None
        px = float(ctx.daily["close"].iloc[-1])
        if px <= pos.stop:
            return Signal(ctx.symbol, self.name, "EXIT", ctx.asof.isoformat(),
                          f"close {px:.2f} <= stop {pos.stop:.2f} (range floor broken)",
                          ref_price=px, stop=pos.stop)
        x = self.cfg["exit"]
        if pos.opened and (ctx.asof - date.fromisoformat(pos.opened)).days >= x["max_holding_days"]:
            return Signal(ctx.symbol, self.name, "EXIT", ctx.asof.isoformat(),
                          f"time stop: {x['max_holding_days']} days without reaching the ceiling",
                          ref_price=px, stop=pos.stop)
        r = fractal_result(ctx.daily, self.cfg.get("levels_timeframe", "daily"), px, ctx.asof)
        res = r.res[0] if r.res else None
        if res is None:
            return Signal(ctx.symbol, self.name, "EXIT", ctx.asof.isoformat(),
                          f"range resolved upward (no resistance left above {px:.2f}) - taking profit",
                          ref_price=px, stop=pos.stop)
        if px >= res.low * (1 - x["target_tolerance_pct"] / 100.0):
            return Signal(ctx.symbol, self.name, "EXIT", ctx.asof.isoformat(),
                          f"close {px:.2f} reached range ceiling {res.low:.2f}-{res.high:.2f}",
                          ref_price=px, stop=pos.stop)
        return None

    def add_signal(self, ctx: SymbolContext, pos: Position) -> Signal | None:
        return None  # no pyramiding into a range trade - the target caps the payoff
