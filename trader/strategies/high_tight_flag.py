from __future__ import annotations

from datetime import date

import pandas as pd

from trader.core.indicators import sma
from trader.core.models import PendingOrder, Position, Signal
from trader.strategies._breakout_common import (Setup, bars_held, check_open, entry_price,
                                                hard_stop_hit, triggered)
from trader.strategies.base import Strategy, SymbolContext, register


def momentum_surge(daily: pd.DataFrame, lookback: int) -> tuple[float | None, dict]:
    """Spec Part 2.1 on the last `lookback` completed bars: Low40 = lowest low in the window,
    HighMax = highest high on the bars strictly AFTER Low40's bar. Returns (surge_pct, details);
    surge_pct is None when the low is the window's last bar (nothing after it to measure)."""
    w = daily.iloc[-lookback:]
    pos_low = int(w["low"].to_numpy().argmin())          # first occurrence of the minimum
    low40 = float(w["low"].iloc[pos_low])
    after = w.iloc[pos_low + 1:]
    if after.empty or low40 <= 0:
        return None, {"low40": low40, "low40_date": str(w.index[pos_low].date())}
    high_max = float(after["high"].max())
    return (high_max - low40) / low40 * 100.0, {
        "low40": low40, "low40_date": str(w.index[pos_low].date()),
        "high_max": high_max, "high_max_date": str(after["high"].idxmax().date())}


@register("high_tight_flag")
class HighTightFlag(Strategy):
    """Kristjan Kullamagi High Tight Flag / daily momentum breakout (spec Part 2, 2026-11-12).

    Setup (completed bars; the last completed bar is the spec's "yesterday"):
      - Momentum surge over the last `surge_lookback` (40) bars >= min_surge_pct (30%).
      - Flag = last `flag_bars` (5) completed bars:
          A) every flag close > its own SMA10 or SMA20 (`ma_support`: either|sma10|sma20|both;
             spec says "SMA10 or SMA20" -> either);
          B) (Flag_High5 - Flag_Low5) / Flag_Low5 * 100 <= max_flag_range_pct (8%).
      - pivot = yesterday's high + tick.
      - stop = yesterday's low - tick (always the TIGHTER of the spec's two choices, since
        yesterday is inside the flag); if that leaves less than `min_room_pct` between pivot and
        stop ("room to breathe"), fall back to Flag_Low5 - tick.

    Trigger: next session's high reaches the pivot (`trigger_on: "high"`, spec) - or closes above
    it (`"close"`). Fill is the following session's open in this EOD engine (see config _note).

    Exits:
      - hard stop: close (or low) at/below pos.stop -> EXIT.
      - trail: close strictly below SMA10 -> EXIT (spec: "only when a daily candle closes below
        SMA10"). `trail_before_trim: true` also applies it before the partial (the spec is silent
        on a trade that is never positive on day 1-2; without this it would have no trail at all).
      - partial: if the close is above entry on day 1 or day 2 (`partial_max_day`), TRIM
        `trim_fraction` (33%) and move the stop to breakeven. At most once.
    No pyramiding (spec has none).
    """

    def setup(self, daily: pd.DataFrame) -> Setup | None:
        c = self.cfg["entry"]
        need = max(c["surge_lookback"], c["sma_slow"] + c["flag_bars"], c.get("min_bars", 0))
        if len(daily) < need:
            return None
        surge, sd = momentum_surge(daily, c["surge_lookback"])
        if surge is None or surge < c["min_surge_pct"]:
            return None

        n = c["flag_bars"]
        flag = daily.iloc[-n:]
        s10 = sma(daily["close"], c["sma_fast"]).iloc[-n:]
        s20 = sma(daily["close"], c["sma_slow"]).iloc[-n:]
        if s10.isna().any() or s20.isna().any():
            return None
        above10, above20 = flag["close"] > s10, flag["close"] > s20
        support = {"either": above10 | above20, "sma10": above10,
                   "sma20": above20, "both": above10 & above20}[c["ma_support"]]
        if not bool(support.all()):
            return None
        fh, fl = float(flag["high"].max()), float(flag["low"].min())
        if fl <= 0:
            return None
        flag_range = (fh - fl) / fl * 100.0
        if flag_range > c["max_flag_range_pct"]:
            return None

        tick, y = c["tick"], daily.iloc[-1]
        pivot = round(float(y["high"]) + tick, 2)
        stop, stop_basis = round(float(y["low"]) - tick, 2), "yesterday_low"
        if (pivot - stop) / pivot * 100.0 < c["min_room_pct"]:
            stop, stop_basis = round(fl - tick, 2), "flag_low"
        cap = c.get("max_initial_risk_pct")
        if cap is not None and (pivot - stop) / pivot * 100.0 > cap:
            stop, stop_basis = round(pivot * (1 - cap / 100.0), 2), stop_basis + "_capped"
        return Setup(daily.index[-1].date(), pivot, stop,
                     {"surge_pct": round(surge, 1), **sd, "flag_high": fh, "flag_low": fl,
                      "flag_range_pct": round(flag_range, 2), "stop_basis": stop_basis})

    def entry_signal(self, ctx: SymbolContext) -> Signal | None:
        d = ctx.daily
        if len(d) < 2:
            return None
        st = self.setup(d.iloc[:-1])
        if st is None:
            return None
        live, trig = d.iloc[-1], self.cfg["entry"]["trigger_on"]
        if not triggered(live, st.pivot, trig):
            return None
        return Signal(ctx.symbol, self.name, "ENTRY", ctx.asof.isoformat(),
                      f"HTF breakout: {trig} {float(live[trig]):.2f} >= pivot {st.pivot:.2f} "
                      f"(surge {st.details['surge_pct']:.0f}%, flag range {st.details['flag_range_pct']:.1f}%), "
                      f"stop {st.stop:.2f} [{st.details['stop_basis']}]",
                      score=0, stop=st.stop, ref_price=st.pivot,
                      meta={"pivot": st.pivot, "surge_pct": st.details["surge_pct"],
                            "flag_range_pct": st.details["flag_range_pct"], "stop_basis": st.details["stop_basis"]})

    def validate_open(self, order: PendingOrder, open_px: float, prev_close: float) -> tuple[bool, str]:
        return check_open(order, open_px, self.cfg["open_checks"],
                          self.cfg["entry"].get("max_initial_risk_pct"))

    def stop_level(self, ctx: SymbolContext, pos: Position) -> float | None:
        return None   # stop only moves once, to breakeven, on the TRIM fill (engine applies new_stop)

    def exit_signal(self, ctx: SymbolContext, pos: Position) -> Signal | None:
        if pos.opened and ctx.asof < date.fromisoformat(pos.opened):
            return None
        x, d = self.cfg["exit"], ctx.daily
        live = d.iloc[-1]
        px = float(live["close"])
        if hard_stop_hit(live, pos.stop, x["stop_trigger"]):
            return Signal(ctx.symbol, self.name, "EXIT", ctx.asof.isoformat(),
                          f"{x['stop_trigger']} {float(live[x['stop_trigger']]):.2f} <= stop {pos.stop:.2f}"
                          + (" (breakeven)" if pos.meta.get("trims") else ""),
                          ref_price=px, stop=pos.stop)
        trimmed = bool(pos.meta.get("trims"))
        if trimmed or x["trail_before_trim"]:
            s10 = sma(d["close"], x["trail_sma_period"]).iloc[-1]
            if s10 == s10 and px < float(s10):
                return Signal(ctx.symbol, self.name, "EXIT", ctx.asof.isoformat(),
                              f"close {px:.2f} < SMA{x['trail_sma_period']} {float(s10):.2f}",
                              ref_price=px, stop=pos.stop)
        if not trimmed:
            entry, held = entry_price(pos), bars_held(d, pos.opened)
            if 1 <= held <= x["partial_max_day"] and px > entry:
                return Signal(ctx.symbol, self.name, "TRIM", ctx.asof.isoformat(),
                              f"day {held} close {px:.2f} > entry {entry:.2f}: sell {x['trim_fraction']:.0%}, "
                              f"stop -> breakeven", ref_price=px, stop=entry,
                              meta={"fraction": x["trim_fraction"], "new_stop": round(entry, 2)})
        return None

    def add_signal(self, ctx: SymbolContext, pos: Position) -> Signal | None:
        return None
