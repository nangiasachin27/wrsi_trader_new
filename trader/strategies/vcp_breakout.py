from __future__ import annotations

from datetime import date

import pandas as pd

from trader.core.indicators import ema, sma
from trader.core.models import PendingOrder, Position, Signal
from trader.strategies._breakout_common import Setup, check_open, entry_price, hard_stop_hit, triggered
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
    """Mark Minervini Volatility Contraction Pattern breakout (spec Part 1, 2026-11-12).

    Setup (on completed bars, i.e. as the spec's pre-market Daily Filter Routine sees them):
      - Trend Template conditions 1-7 on the last completed close (see trend_template()).
      - Condition 8, RS >= 70: rs_rank filter, gated per-strategy at `rs_rank_min_percentile`
        (13-week return vs NIFTY500, percentile across the universe - the spec's own proxy,
        with NIFTY500 in place of the S&P 500).
      - Last `contraction_bars` (4) completed bars: spread = (High4 - Low4) / Low4 * 100 must be
        < max_spread_pct (4.0).
      - pivot = High4 + tick, stop = Low4 - tick, capped at max_initial_risk_pct (5%) below entry.

    Trigger: the next session's high reaches the pivot (`trigger_on: "high"`, spec) - or closes
    above it (`"close"`, stricter). This engine is EOD -> next open, so the fill is the FOLLOWING
    session's open, not the intraday buy-stop the spec describes; see the config _note.
    validate_open re-applies the 5% cap against the real open and skips extended gaps.

    Exits:
      - hard stop: close (or low, `stop_trigger`) at/below pos.stop -> EXIT.
      - target: high (or close, `target_trigger`) >= entry + target_r * R, R = entry - initial
        stop -> TRIM `trim_fraction` (50%) and move the stop to breakeven (entry).
      - after the trim: the stop trails max(entry, EMA20), ratchet-only; close below it -> EXIT.
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
        return check_open(order, open_px, self.cfg["open_checks"],
                          self.cfg["entry"].get("max_initial_risk_pct"))

    def stop_level(self, ctx: SymbolContext, pos: Position) -> float | None:
        if not pos.meta.get("trims"):
            return None                       # fixed initial stop until the 2R scale-out
        e = ema(ctx.daily["close"], self.cfg["exit"]["trail_ema_period"]).iloc[-1]
        trail = max(entry_price(pos), float(e)) if e == e else entry_price(pos)
        return ratchet(pos.stop, trail)

    def exit_signal(self, ctx: SymbolContext, pos: Position) -> Signal | None:
        if pos.opened and ctx.asof < date.fromisoformat(pos.opened):
            return None
        x, live = self.cfg["exit"], ctx.daily.iloc[-1]
        px = float(live["close"])
        if hard_stop_hit(live, pos.stop, x["stop_trigger"]):
            what = "trailing stop (max(entry, EMA%d))" % x["trail_ema_period"] if pos.meta.get("trims") else "initial stop"
            return Signal(ctx.symbol, self.name, "EXIT", ctx.asof.isoformat(),
                          f"{x['stop_trigger']} {float(live[x['stop_trigger']]):.2f} <= {what} {pos.stop:.2f}",
                          ref_price=px, stop=pos.stop)
        if pos.meta.get("trims"):
            return None
        entry = entry_price(pos)
        r = entry - float(pos.meta.get("initial_stop", pos.stop))
        if r <= 0:
            return None
        target = entry + x["target_r"] * r
        if float(live[x["target_trigger"]]) >= target:
            return Signal(ctx.symbol, self.name, "TRIM", ctx.asof.isoformat(),
                          f"{x['target_trigger']} {float(live[x['target_trigger']]):.2f} >= {x['target_r']}R target "
                          f"{target:.2f} (entry {entry:.2f}, R {r:.2f}): sell {x['trim_fraction']:.0%}, stop -> breakeven",
                          ref_price=px, stop=entry,
                          meta={"fraction": x["trim_fraction"], "new_stop": round(entry, 2), "target": target})
        return None

    def add_signal(self, ctx: SymbolContext, pos: Position) -> Signal | None:
        return None
