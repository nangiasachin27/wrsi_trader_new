"""Shared pieces for the two spec-driven breakout strategies (vcp_breakout, high_tight_flag).

Both strategies split their logic the same way, mirroring the spec's two routines:
  setup(completed_bars)  - the "Daily Filter Routine": everything that is knowable before the
                           next session opens. Returns a Setup with the buy-stop trigger price
                           (pivot) and the initial stop, or None.
  entry_signal(ctx)      - the "Execution Engine": runs setup() on every bar EXCEPT the latest,
                           then checks whether the latest (the "live" candle in the spec; the
                           just-completed session in this EOD engine) crossed the pivot.

scripts/swing_watchlist.py calls setup() directly on all bars to list tomorrow's ARMED buy-stops,
so the watchlist and the backtest can never drift apart - they run literally the same code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from trader.core.models import PendingOrder, Position

EPS = 1e-9


@dataclass
class Setup:
    asof: date              # date of the last COMPLETED bar the setup was evaluated on
    pivot: float            # buy-stop trigger price for the next session
    stop: float             # initial stop (before any validate_open re-cap against the real fill)
    details: dict = field(default_factory=dict)

    @property
    def risk_pct(self) -> float:
        return (self.pivot - self.stop) / self.pivot * 100.0 if self.pivot > 0 else float("nan")


def triggered(live: pd.Series, pivot: float, trigger_on: str) -> bool:
    """'high' = spec: the session's high reached the pivot (a buy-stop would have filled).
    'close' = stricter: the session also CLOSED above it (filters failed intraday breakouts)."""
    px = float(live["high"]) if trigger_on == "high" else float(live["close"])
    return px + EPS >= pivot if trigger_on == "high" else px > pivot + EPS


def bars_held(daily: pd.DataFrame, opened: str) -> int:
    """Trading bars since (and including) the entry day - 1 on the entry session itself."""
    if not opened:
        return 0
    return int((daily.index >= pd.Timestamp(opened)).sum())


def entry_price(pos: Position) -> float:
    """Original average entry cost - unchanged by partial sells (the engine records it at the
    first TRIM; before any TRIM it is simply pos.avg_cost)."""
    return float(pos.meta.get("entry_avg_cost") or pos.avg_cost)


def hard_stop_hit(live: pd.Series, stop: float | None, stop_trigger: str) -> bool:
    if not stop:
        return False
    px = float(live["low"]) if stop_trigger == "low" else float(live["close"])
    return px <= stop + EPS


def check_open(order: PendingOrder, open_px: float, open_cfg: dict,
               max_risk_pct: float | None) -> tuple[bool, str]:
    """Shared open-time validation. Mutates order.stop when the actual fill makes the stop wider
    than max_risk_pct (the spec's caps are defined against the real Entry, which an EOD engine
    only knows at the open). The engine sizes and records the position from order.stop AFTER this
    call, so the cap is what the trade is actually sized and stopped against."""
    if order.action != "ENTRY":
        return True, "ok"
    if order.stop is not None and open_px <= order.stop:
        return False, f"open {open_px:.2f} at/below stop {order.stop:.2f}"
    pivot = order.ref_price
    chase = open_cfg.get("max_chase_pct")
    if pivot and chase is not None and open_px > pivot * (1 + chase / 100.0):
        return False, f"open {open_px:.2f} more than {chase}% above pivot {pivot:.2f} (extended)"
    if max_risk_pct is not None and order.stop is not None:
        floor = round(open_px * (1 - max_risk_pct / 100.0), 2)
        if floor > order.stop:
            order.meta["stop_before_cap"] = order.stop
            order.stop = floor
    return True, "ok"
