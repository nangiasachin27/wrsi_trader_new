from __future__ import annotations

from datetime import date

import pandas as pd

from trader.core.indicators import ema, sma
from trader.core.models import PendingOrder, Position, Signal
from trader.strategies._breakout_common import (Setup, buy_stop_fill, check_open, entry_price,
                                                hard_stop_hit, triggered)
from trader.strategies._swing_common import ratchet
from trader.strategies.base import Strategy, SymbolContext, register


def trend_template(daily: pd.DataFrame, cfg: dict) -> tuple[bool, dict]:
    """Minervini Trend Template, conditions 1-7, on the LAST row of `daily` (C = its close).
    Condition 8 (RS rating >= 70) is cross-sectional - it needs every symbol's return on the same
    day - so it is enforced by the rs_rank filter via this strategy's `rs_rank_min_percentile`,
    not here. Returns (all_passed, details) where details lists every condition's result, so the
    watchlist can show WHY a name failed, not just that it did."""
    f, m, s = cfg["sma_fast"], cfg["sma_mid"], cfg["sma_slow"]
    look, n52 = cfg["sma_slow_rising_lookback"], cfg["lookback_52w"]
    if len(daily) < max(s + look, n52):
        return False, {"error": f"need {max(s + look, n52)} bars, have {len(daily)}"}
    c = daily["close"]
    sf, sm, ss = sma(c, f), sma(c, m), sma(c, s)
    C, SF, SM, SS = float(c.iloc[-1]), float(sf.iloc[-1]), float(sm.iloc[-1]), float(ss.iloc[-1])
    SS_prev = float(ss.iloc[-1 - look])
    w = daily.iloc[-n52:]
    low52, high52 = float(w["low"].min()), float(w["high"].max())
    conds = {
        "c1_close_gt_sma50_and_sma150": C > SF and C > SM,
        "c2_sma50_gt_sma150": SF > SM,
        "c3_sma150_gt_sma200": SM > SS,
        "c4_sma200_rising": SS > SS_prev,
        "c5_close_gt_sma200": C > SS,
        "c6_above_52w_low": C >= low52 * (1 + cfg["min_above_52w_low_pct"] / 100.0),
        "c7_near_52w_high": C >= high52 * (1 - cfg["max_below_52w_high_pct"] / 100.0),
    }
    details = {**conds, "close": C, f"sma{f}": SF, f"sma{m}": SM, f"sma{s}": SS,
               f"sma{s}_{look}d_ago": SS_prev, "low52": low52, "high52": high52,
               "pct_above_low52": (C / low52 - 1) * 100, "pct_below_high52": (1 - C / high52) * 100}
    return all(conds.values()), details


@register("vcp_breakout")
class VCPBreakout(Strategy):
    """Mark Minervini Volatility Contraction Pattern breakout - spec Part 1 (user document,
    2026-11-12). Default config = the spec's rules exactly; every non-spec choice is listed below.

    Daily Filter Routine (setup(), on COMPLETED bars - the pre-market view):
      - Trend Template conditions 1-7 on the last completed close (trend_template()).
      - Condition 8, RS rating >= 70: rs_rank filter, hard-gated for this strategy only at
        `rs_rank_min_percentile` (0.70). Proxy as the spec allows: 13-week return vs NIFTY500
        (the Indian stand-in for the S&P 500), as a percentile across the universe.
      - VCP: last 4 completed bars, (High4 - Low4) / Low4 * 100 < 4.0.
      - pivot = High4 + 0.01, SL = Low4 - 0.01.

    Execution (`execution.mode: "intraday"`, simulated resting orders - see Strategy.intraday):
      - Buy-stop at the pivot: fills on the session whose high reaches it, at max(open, pivot).
      - Hard Risk Ceiling: if (Entry - SL) / Entry > 5%, SL = Entry * 0.95, against the ACTUAL
        fill (validate_open).
      - Hard stop-loss: resting; fills at min(open, stop) on the first session the low reaches it.
      - Profit taking: resting limit sell of 50% at Entry + 2 * (Entry - SL) (1:2 R:R, R from the
        stop the trade was sized against); the remaining stop moves to the exact entry price.
      - Remainder trailed with the 20-day EMA. The spec does not say close-based or a stop level:
        `trail_mode: "stop"` (default) = resting stop at max(entry, EMA20), ratchet-only, updated
        each close; `"close"` = exit at the next open after a close below EMA20 (stop stays at
        breakeven meanwhile).

    Non-spec details (all config, all documented): price data/ticks are NSE's, the spec's $0.01
    is used as-is (0.01); with daily bars the intrabar order is unknown, so a session that
    reaches both the stop and the target (or the entry and the stop) is booked as the stop
    (conservative). `execution.mode: "next_open"` restores the old EOD -> next-open model.
    No pyramiding (spec has none).
    """

    # ---- daily filter routine -------------------------------------------------------------
    def setup(self, daily: pd.DataFrame) -> Setup | None:
        c = self.cfg["entry"]
        if len(daily) < c["min_bars"]:
            return None
        ok, tt = trend_template(daily, c["trend_template"])
        if not ok:
            return None
        n = c["contraction_bars"]
        w = daily.iloc[-n:]
        hi, lo = float(w["high"].max()), float(w["low"].min())
        if lo <= 0:
            return None
        spread = (hi - lo) / lo * 100.0
        if not spread < c["max_spread_pct"]:
            return None
        tick = c["tick"]
        pivot, stop = round(hi + tick, 2), round(lo - tick, 2)
        cap = c.get("max_initial_risk_pct")
        capped = False
        if cap is not None and (pivot - stop) / pivot * 100.0 > cap:
            stop, capped = round(pivot * (1 - cap / 100.0), 2), True
        return Setup(daily.index[-1].date(), pivot, stop,
                     {"vcp_high": hi, "vcp_low": lo, "spread_pct": round(spread, 2),
                      "stop_capped": capped, "trend_template": tt})

    # ---- execution engine -----------------------------------------------------------------
    def entry_signal(self, ctx: SymbolContext) -> Signal | None:
        d = ctx.daily
        if len(d) < 2:
            return None
        st = self.setup(d.iloc[:-1])
        if st is None:
            return None
        live = d.iloc[-1]
        trig = self.cfg["entry"]["trigger_on"]
        if not triggered(live, st.pivot, trig):
            return None
        return Signal(ctx.symbol, self.name, "ENTRY", ctx.asof.isoformat(),
                      f"VCP breakout: {trig} {float(live[trig]):.2f} >= pivot {st.pivot:.2f} "
                      f"({self.cfg['entry']['contraction_bars']}-bar spread {st.details['spread_pct']:.2f}%), "
                      f"stop {st.stop:.2f}",
                      score=0, stop=st.stop, ref_price=st.pivot,
                      meta={"pivot": st.pivot, "spread_pct": st.details["spread_pct"],
                            "stop_capped": st.details["stop_capped"]})

    def validate_open(self, order: PendingOrder, open_px: float, prev_close: float) -> tuple[bool, str]:
        return check_open(order, open_px, self.cfg.get("open_checks", {}),
                          self.cfg["entry"].get("max_initial_risk_pct"))

    def entry_fill_price(self, sig: Signal, bar: pd.Series) -> float:
        return buy_stop_fill(bar, sig.ref_price, self.cfg["entry"]["trigger_on"])

    def _target(self, pos: Position) -> tuple[float, float, float] | None:
        """(entry, R, target) for the 1:2 scale-out, from information fixed at entry."""
        entry = entry_price(pos)
        r = entry - float(pos.meta.get("initial_stop", pos.stop))
        if r <= 0:
            return None
        return entry, r, entry + self.cfg["exit"]["target_r"] * r

    def take_profit_order(self, ctx: SymbolContext, pos: Position) -> dict | None:
        if pos.meta.get("trims"):
            return None
        t = self._target(pos)
        if t is None:
            return None
        entry, r, target = t
        x = self.cfg["exit"]
        return {"price": round(target, 2), "fraction": x["trim_fraction"], "new_stop": round(entry, 2),
                "reason": f"{x['target_r']}R target (entry {entry:.2f}, R {r:.2f}): sell "
                          f"{x['trim_fraction']:.0%}, stop -> breakeven"}

    def stop_level(self, ctx: SymbolContext, pos: Position) -> float | None:
        if not pos.meta.get("trims"):
            return None                       # fixed initial stop until the 2R scale-out
        if self.cfg["exit"].get("trail_mode", "stop") == "close":
            return None                       # stop stays at breakeven; EMA exit is close-based
        e = ema(ctx.daily["close"], self.cfg["exit"]["trail_ema_period"]).iloc[-1]
        trail = max(entry_price(pos), float(e)) if e == e else entry_price(pos)
        return ratchet(pos.stop, trail)

    def exit_signal(self, ctx: SymbolContext, pos: Position) -> Signal | None:
        if pos.opened and ctx.asof < date.fromisoformat(pos.opened):
            return None
        x, live = self.cfg["exit"], ctx.daily.iloc[-1]
        px = float(live["close"])
        trimmed = bool(pos.meta.get("trims"))
        if trimmed and x.get("trail_mode", "stop") == "close":
            e = ema(ctx.daily["close"], x["trail_ema_period"]).iloc[-1]
            if e == e and px < float(e):
                return Signal(ctx.symbol, self.name, "EXIT", ctx.asof.isoformat(),
                              f"close {px:.2f} < EMA{x['trail_ema_period']} {float(e):.2f}",
                              ref_price=px, stop=pos.stop)
        if self.intraday:
            return None                       # hard stop and 2R limit are resting orders (engine)

        # ---- execution.mode "next_open" (legacy EOD model) ----
        strig, ttrig = x.get("stop_trigger", "close"), x.get("target_trigger", "high")
        if hard_stop_hit(live, pos.stop, strig):
            what = "trailing stop (max(entry, EMA%d))" % x["trail_ema_period"] if trimmed else "initial stop"
            return Signal(ctx.symbol, self.name, "EXIT", ctx.asof.isoformat(),
                          f"{strig} {float(live[strig]):.2f} <= {what} {pos.stop:.2f}",
                          ref_price=px, stop=pos.stop)
        t = None if trimmed else self._target(pos)
        if t and float(live[ttrig]) >= t[2]:
            entry, r, target = t
            return Signal(ctx.symbol, self.name, "TRIM", ctx.asof.isoformat(),
                          f"{ttrig} {float(live[ttrig]):.2f} >= {x['target_r']}R target "
                          f"{target:.2f} (entry {entry:.2f}, R {r:.2f}): sell {x['trim_fraction']:.0%}, stop -> breakeven",
                          ref_price=px, stop=entry,
                          meta={"fraction": x["trim_fraction"], "new_stop": round(entry, 2), "target": target})
        return None

    def add_signal(self, ctx: SymbolContext, pos: Position) -> Signal | None:
        return None
