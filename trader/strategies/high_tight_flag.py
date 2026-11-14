from __future__ import annotations

from datetime import date

import pandas as pd

from trader.core.indicators import sma
from trader.core.models import PendingOrder, Position, Signal
from trader.strategies._breakout_common import (Setup, bars_held, buy_stop_fill, check_open,
                                                entry_price, hard_stop_hit, triggered)
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
    """Kristjan Kullamagi High Tight Flag / daily momentum breakout - spec Part 2 (user document,
    2026-11-12). Default config = the spec's rules exactly; every non-spec choice is listed below.

    Setup (setup(), on COMPLETED bars; the last completed bar is the spec's "yesterday"):
      - Momentum surge: last 40 bars, Low40 = lowest low, HighMax = highest high AFTER Low40,
        (HighMax - Low40) / Low40 * 100 >= 30.
      - Flag = last 5 completed bars: (A) every close > its own SMA10 or SMA20; (B) (Flag_High5 -
        Flag_Low5) / Flag_Low5 * 100 <= 8.
      - pivot = Yesterday_High + 0.01.
      - SL = the tighter of Yesterday_Low - 0.01 and Flag_Low5 - 0.01 (yesterday is inside the
        flag, so this is always Yesterday_Low - 0.01), "provided it gives the trade room to
        breathe". The spec does not quantify that: `min_room_pct` (2%) - if the tighter stop is
        closer than that to the pivot, Flag_Low5 - 0.01 is used. This is the only interpretive
        number in the entry.

    Execution (`execution.mode: "intraday"`, simulated resting orders - see Strategy.intraday):
      - "Market buy immediately upon trigger detection" = buy-stop at the pivot, filled on the
        session whose high reaches it at max(open, pivot).
      - Hard stop-loss: resting; fills at min(open, stop) on the first session the low reaches it.
      - Day 1 = the entry session. If the trade CLOSES above entry on day 1 or day 2, sell 33% and
        move the remaining stop to breakeven. The condition is only known at the close, so the
        sale is a market order at the next open (once per trade).
      - Trailing exit: a close strictly below SMA10 -> market sell of the entire position at the
        next open. Read as applying to the whole position whether or not the partial happened
        ("close the entire position"); `trail_before_trim: false` restricts it to after the
        partial instead.

    With daily bars the intrabar order is unknown, so an entry session that also reaches the stop
    is booked as stopped out (conservative). No risk ceiling (the spec gives none for HTF).
    `execution.mode: "next_open"` restores the old EOD -> next-open model. No pyramiding.
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
        return check_open(order, open_px, self.cfg.get("open_checks", {}),
                          self.cfg["entry"].get("max_initial_risk_pct"))

    def entry_fill_price(self, sig: Signal, bar: pd.Series) -> float:
        return buy_stop_fill(bar, sig.ref_price, self.cfg["entry"]["trigger_on"])

    def stop_level(self, ctx: SymbolContext, pos: Position) -> float | None:
        return None   # stop only moves once, to breakeven, on the TRIM fill (engine applies new_stop)

    def exit_signal(self, ctx: SymbolContext, pos: Position) -> Signal | None:
        if pos.opened and ctx.asof < date.fromisoformat(pos.opened):
            return None
        x, d = self.cfg["exit"], ctx.daily
        live = d.iloc[-1]
        px = float(live["close"])
        # Hard stop: a resting order handled by the engine in intraday mode; close-checked here
        # only in the legacy next_open mode.
        if not self.intraday and hard_stop_hit(live, pos.stop, x.get("stop_trigger", "close")):
            trig = x.get("stop_trigger", "close")
            return Signal(ctx.symbol, self.name, "EXIT", ctx.asof.isoformat(),
                          f"{trig} {float(live[trig]):.2f} <= stop {pos.stop:.2f}"
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
