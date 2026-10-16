from __future__ import annotations

from datetime import date

import pandas as pd

from trader.core.indicators import rsi, sma, supertrend
from trader.core.models import PendingOrder, Position, Signal
from trader.strategies.base import Strategy, SymbolContext, register


@register("weekly_rsi_sma20")
class WeeklyRsiSma20(Strategy):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        if cfg["exit"].get("mode", "weekly_close") != "weekly_close":
            raise ValueError("exit.mode 'daily_close' was removed: exits are judged on COMPLETED weekly "
                             "candles only (a daily candle is never compared against the weekly SMA20). "
                             "Delete the 'mode' key from the exit config.")
        if cfg["exit"].get("trigger", "wick") not in ("wick", "close"):
            raise ValueError("exit.trigger must be 'wick' or 'close'")
        if cfg["exit"].get("method", "sma20") not in ("sma20", "supertrend"):
            raise ValueError("exit.method must be 'sma20' or 'supertrend'")
        ft = cfg["exit"].get("fast_trail", {})
        if ft.get("enabled") and ft.get("trigger", "min_holding_weeks") not in (
                "min_holding_weeks", "min_unrealized_gain_pct"):
            raise ValueError("exit.fast_trail.trigger must be 'min_holding_weeks' or 'min_unrealized_gain_pct'")

    # ---- helpers -------------------------------------------------------------------------
    def _w(self, ctx: SymbolContext, partial: bool) -> pd.DataFrame:
        return ctx.weekly(self.cfg["timeframes"]["weekly_rule"], partial)

    def _sma_last(self, w: pd.DataFrame) -> float | None:
        s = sma(w["close"], self.cfg["indicators"]["sma_period"]).iloc[-1]
        return None if pd.isna(s) else float(s)

    def _supertrend_last(self, w: pd.DataFrame) -> tuple[float | None, int | None]:
        sc = self.cfg["exit"]["supertrend"]
        line, direction = supertrend(w["high"], w["low"], w["close"], sc["period"], sc["multiplier"])
        if pd.isna(line.iloc[-1]):
            return None, None
        return float(line.iloc[-1]), int(direction.iloc[-1])

    def _fast_trail_triggered(self, pos: Position, ctx: SymbolContext, w: pd.DataFrame, ft: dict) -> bool:
        """Has this specific position earned the faster trail yet? Checked fresh every week
        against the position's own age/gain - a position that hasn't earned it yet is still
        governed purely by the normal (slower) stop, unaffected by this feature at all."""
        trigger = ft.get("trigger", "min_holding_weeks")
        if trigger == "min_holding_weeks":
            if not pos.opened:
                return False
            weeks_held = (ctx.asof - date.fromisoformat(pos.opened)).days / 7.0
            return weeks_held >= ft.get("min_holding_weeks", 8)
        # min_unrealized_gain_pct
        if not pos.avg_cost:
            return False
        px = float(w["close"].iloc[-1])
        gain_pct = (px - pos.avg_cost) / pos.avg_cost * 100.0
        return gain_pct >= ft.get("min_unrealized_gain_pct", 15.0)

    def _apply_fast_trail(self, w: pd.DataFrame, pos: Position, ctx: SymbolContext, base_stop: float) -> float:
        """Ratchet only: once triggered, the stop can tighten to the faster SMA, but never
        loosens below whatever the normal (slower) method already guarantees. Disabled by
        default - with fast_trail.enabled false this is a no-op and base_stop passes through
        unchanged, so existing backtests/configs are byte-for-byte unaffected."""
        ft = self.cfg["exit"].get("fast_trail", {})
        if not ft.get("enabled"):
            return base_stop
        if not self._fast_trail_triggered(pos, ctx, w, ft):
            return base_stop
        fast = sma(w["close"], ft.get("sma_period", 10)).iloc[-1]
        if pd.isna(fast):
            return base_stop
        return max(base_stop, float(fast))

    def stop_level(self, ctx: SymbolContext, pos: Position) -> float | None:
        # The reported/tracked stop always reflects whichever exit method is actually governing
        # exits, so Rule 3's open-check and the position's displayed stop stay consistent with
        # what will really trigger a sell. fast_trail only participates when exit.method is
        # sma20 - see _apply_fast_trail's docstring and this class's module-level note.
        w = self._w(ctx, partial=False)
        if self.cfg["exit"].get("method", "sma20") == "supertrend":
            return self._supertrend_last(w)[0]
        base_stop = self._sma_last(w)
        if base_stop is None:
            return None
        return self._apply_fast_trail(w, pos, ctx, base_stop)

    # ---- Rule 1 (+ Rule 4 ranking) --------------------------------------------------------
    def entry_signal(self, ctx: SymbolContext) -> Signal | None:
        c, ind = self.cfg["entry"], self.cfg["indicators"]
        completed = c["weekly_bar"] == "completed"
        if completed and not ctx.calendar.is_week_end(ctx.asof):
            return None
        w = self._w(ctx, partial=not completed)
        m = ctx.monthly(self.cfg["timeframes"]["monthly_rule"], c["monthly_include_partial"])
        if len(w) < c["min_weekly_bars"] or len(m) < c["min_monthly_bars"]:
            return None

        wr = rsi(w["close"], ind["rsi_period"])
        mr = rsi(m["close"], ind["rsi_period"])
        rw, rm = float(wr.iloc[-1]), float(mr.iloc[-1])
        if pd.isna(rw) or pd.isna(rm):
            return None

        # "sustain_weeks" (default 1) guards against a one-week RSI spike that reverses right
        # after entry: with 1, this fires the moment RSI crosses (today's behavior, unchanged).
        # With N>1, it only fires once RSI has stayed above the threshold for N consecutive
        # completed weeks - i.e. it waits for confirmation instead of buying the first tick
        # above 60. Trade-off: entries are later by (N-1) weeks, which also gives up that much
        # of any real move - this filters false starts at the cost of some of your best winners
        # too, so A/B it via backtest rather than assuming a higher number is strictly better.
        sustain = c.get("sustain_weeks", 1)
        if len(wr) < sustain + 1:
            return None
        sustained_above = bool((wr.iloc[-sustain:] > c["weekly_rsi_threshold"]).all())
        was_below_before = bool(wr.iloc[-(sustain + 1)] <= c["weekly_rsi_threshold"])
        crossed = sustained_above and was_below_before
        monthly_ok = rm >= c["monthly_rsi_threshold"] - c["monthly_rsi_near_tolerance"]
        if not (crossed and monthly_ok):
            return None

        sma_now, px = self._sma_last(w), float(w["close"].iloc[-1])
        if sma_now is None:
            return None
        if c["require_close_above_sma"] and px <= sma_now:
            return None
        stop_dist = (px - sma_now) / px * 100.0
        if stop_dist > c["max_stop_distance_pct"]:
            return None

        # ATH/cup quality scoring moved to the "quality" filter (trader/filters/quality.py) -
        # it's strategy-agnostic ranking, applied to every strategy's candidates uniformly,
        # not something this strategy needs to compute itself.
        return Signal(ctx.symbol, self.name, "ENTRY", ctx.asof.isoformat(),
                      f"wRSI {rw:.1f} crossed {c['weekly_rsi_threshold']}, mRSI {rm:.1f}",
                      score=0, stop=sma_now, ref_price=px,
                      meta={"rsi_w": rw, "rsi_m": rm, "sma": sma_now})

    # ---- Rule 2 ---------------------------------------------------------------------------
    def exit_signal(self, ctx: SymbolContext, pos: Position) -> Signal | None:
        x = self.cfg["exit"]
        # Only COMPLETED weekly candles are ever compared with the weekly SMA20 - never a daily
        # candle, and never the still-forming week. Because the last completed candle doesn't
        # change between Fridays, it is safe to evaluate this every day: a missed run simply
        # re-fires the next day instead of being lost until the following Friday.
        w = self._w(ctx, partial=False)
        bar_end = w.index[-1].date()
        # Only a candle that completed on/after the entry day can trigger an exit. Without this,
        # the candle that produced the BUY signal (whose low may well have touched the SMA20/
        # Supertrend even though it closed on the right side) would be re-read on the first eod
        # after the buy and sell the position the same evening it was bought.
        if pos.opened and bar_end < date.fromisoformat(pos.opened):
            return None

        method = x.get("method", "sma20")
        if method == "supertrend":
            # Supertrend's own recursive band logic is defined on the CLOSE, not the low/high -
            # that's intrinsic to the indicator, so exit.trigger (wick/close) does not apply
            # here; it fires purely on the direction flip, same as the indicator's own signal.
            # fast_trail does not apply to this branch (see module note) - a direction flip
            # isn't a level to ratchet against.
            level, direction = self._supertrend_last(w)
            if level is None:
                return None
            if direction == -1:
                return Signal(ctx.symbol, self.name, "EXIT", ctx.asof.isoformat(),
                              f"weekly Supertrend({x['supertrend']['period']},{x['supertrend']['multiplier']}) "
                              f"turned DOWN, line {level:.2f} (week ending {bar_end})",
                              ref_price=float(w["close"].iloc[-1]), stop=level)
            return None

        # method == "sma20" (default)
        s = self._sma_last(w)
        if s is None:
            return None
        fast_triggered = False
        ft = x.get("fast_trail", {})
        if ft.get("enabled") and self._fast_trail_triggered(pos, ctx, w, ft):
            fast = sma(w["close"], ft.get("sma_period", 10)).iloc[-1]
            if not pd.isna(fast) and float(fast) > s:
                s, fast_triggered = float(fast), True
        # trigger "wick" (default): exit when the weekly candle's LOW touches or goes below the
        # SMA20 (or the ratcheted fast-trail level, once earned), even if it recovered by the
        # close. "close": only the weekly close is compared. Touching counts (<=), not just
        # piercing.
        trigger = x.get("trigger", "wick")
        col = "low" if trigger == "wick" else "close"
        px = float(w[col].iloc[-1])
        if px <= s * (1 - x["buffer_pct"] / 100.0):
            label = f"fast SMA{ft.get('sma_period', 10)}" if fast_triggered \
                else f"SMA{self.cfg['indicators']['sma_period']}"
            return Signal(ctx.symbol, self.name, "EXIT", ctx.asof.isoformat(),
                          f"weekly {trigger} {px:.2f} <= weekly {label} {s:.2f} (week ending {bar_end})",
                          ref_price=px, stop=s)
        return None

    # ---- Rule 5 ---------------------------------------------------------------------------
    def add_signal(self, ctx: SymbolContext, pos: Position) -> Signal | None:
        a = self.cfg["pyramiding"]
        if not a["enabled"]:
            return None
        completed = a["weekly_bar"] == "completed"
        if completed and not ctx.calendar.is_week_end(ctx.asof):
            return None
        w = self._w(ctx, partial=not completed)
        s = self._sma_last(w)
        if s is None:
            return None
        px = float(w["close"].iloc[-1])
        rw = float(rsi(w["close"], self.cfg["indicators"]["rsi_period"]).iloc[-1])
        weeks_since = (ctx.asof - pd.Timestamp(pos.last_buy_date).date()).days / 7.0
        near = s >= pos.last_buy_price * (1 - a["proximity_pct"] / 100.0)
        if (near and rw >= a["min_weekly_rsi"] and weeks_since >= a["min_weeks_between_adds"]
                and (px > s or not a["require_close_above_sma"])):
            return Signal(ctx.symbol, self.name, "ADD", ctx.asof.isoformat(),
                          f"wSMA {s:.2f} within {a['proximity_pct']}% of last buy {pos.last_buy_price:.2f}",
                          stop=s, ref_price=px, meta={"rsi_w": rw, "sma": s})
        return None

    # ---- Rule 3 ---------------------------------------------------------------------------
    def validate_open(self, order: PendingOrder, open_px: float, prev_close: float) -> tuple[bool, str]:
        o = self.cfg["open_checks"]
        if order.action not in o["apply_to"]:
            return True, "ok"
        if open_px < prev_close * (1 - o["gap_down_tolerance_pct"] / 100.0):
            return False, f"gap down: open {open_px:.2f} < prev close {prev_close:.2f}"
        if o["skip_if_open_below_sma"] and order.stop is not None and open_px < order.stop:
            return False, f"open {open_px:.2f} below SMA {order.stop:.2f}"
        if o["max_gap_up_pct"] is not None and open_px > prev_close * (1 + o["max_gap_up_pct"] / 100.0):
            return False, f"gap up beyond {o['max_gap_up_pct']}%"
        return True, "ok"