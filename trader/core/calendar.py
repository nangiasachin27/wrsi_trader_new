from __future__ import annotations

from bisect import bisect_right
from datetime import date, timedelta


class TradingCalendar:
    """Live: weekdays minus configured holidays. Backtest: pass the explicit trading_days list."""

    def __init__(self, holidays: set[date], trading_days: list[date] | None = None):
        self.holidays = holidays
        self.days = sorted(trading_days) if trading_days else None

    def is_trading_day(self, d: date) -> bool:
        if self.days is not None:
            i = bisect_right(self.days, d)
            return i > 0 and self.days[i - 1] == d
        return d.weekday() < 5 and d not in self.holidays

    def next_trading_day(self, d: date) -> date:
        if self.days is not None:
            i = bisect_right(self.days, d)
            if i < len(self.days):
                return self.days[i]
        n = d + timedelta(days=1)
        while n.weekday() >= 5 or n in self.holidays:
            n += timedelta(days=1)
        return n

    def is_week_end(self, d: date) -> bool:
        return self.next_trading_day(d).isocalendar()[:2] != d.isocalendar()[:2]

    def is_month_end(self, d: date) -> bool:
        return self.next_trading_day(d).month != d.month
