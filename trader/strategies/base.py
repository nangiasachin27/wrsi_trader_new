from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

from trader.core.calendar import TradingCalendar
from trader.core.indicators import resample_ohlc
from trader.core.models import PendingOrder, Position, Signal


class SymbolContext:
    """Everything a strategy may look at for one symbol on one date. Data is strictly <= asof."""

    def __init__(self, symbol: str, daily: pd.DataFrame, asof: date, calendar: TradingCalendar):
        self.symbol, self.daily, self.asof, self.calendar = symbol, daily, asof, calendar
        self._cache: dict = {}

    @property
    def last_close(self) -> float:
        return float(self.daily["close"].iloc[-1])

    def weekly(self, rule: str, include_partial: bool) -> pd.DataFrame:
        key = ("w", rule, include_partial)
        if key not in self._cache:
            w = resample_ohlc(self.daily, rule)
            if not include_partial and not self.calendar.is_week_end(self.asof):
                w = w.iloc[:-1]
            self._cache[key] = w
        return self._cache[key]

    def monthly(self, rule: str, include_partial: bool) -> pd.DataFrame:
        key = ("m", rule, include_partial)
        if key not in self._cache:
            m = resample_ohlc(self.daily, rule)
            if not include_partial and not self.calendar.is_month_end(self.asof):
                m = m.iloc[:-1]
            self._cache[key] = m
        return self._cache[key]


class Strategy(ABC):
    """Plug-in contract. A strategy decides WHEN; sizing/risk/execution belong to the engine."""

    name: str = ""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    @abstractmethod
    def entry_signal(self, ctx: SymbolContext) -> Signal | None: ...

    @abstractmethod
    def exit_signal(self, ctx: SymbolContext, pos: Position) -> Signal | None: ...

    @abstractmethod
    def add_signal(self, ctx: SymbolContext, pos: Position) -> Signal | None: ...

    @abstractmethod
    def stop_level(self, ctx: SymbolContext, pos: Position) -> float | None: ...

    def validate_open(self, order: PendingOrder, open_px: float, prev_close: float) -> tuple[bool, str]:
        return True, "ok"

    # ---- resting-order execution (opt-in; added 2026-11-12) --------------------------------
    # Default for every strategy is the original model: signals are computed after the close and
    # filled at the NEXT session's open. A strategy whose config has "execution": {"mode":
    # "intraday"} instead gets resting broker orders simulated on the session's own OHLC by the
    # engine (Engine._intraday_fills / _enter_intraday):
    #   - entry: a buy-stop filled on the trigger session at entry_fill_price(),
    #   - hard stop: pos.stop is a resting stop-loss, filled at min(open, stop) on the first
    #     session whose low reaches it,
    #   - take_profit_order(): a resting limit sell for part of the position.
    # Close-conditioned rules ("if it CLOSES above/below X") still fill at the next open, which is
    # when a market order placed after the close actually executes.
    @property
    def intraday(self) -> bool:
        return self.cfg.get("execution", {}).get("mode") == "intraday"

    def entry_fill_price(self, sig: Signal, bar: pd.Series) -> float:
        """Intraday mode only: fill price of `sig` on its own trigger session `bar`."""
        return float(bar["open"])

    def take_profit_order(self, ctx: SymbolContext, pos: Position) -> dict | None:
        """Intraday mode only: a resting limit sell in force for the session being evaluated -
        {"price", "fraction", "new_stop", "reason"} or None. Must only use information known
        BEFORE that session (entry, initial stop, earlier bars) - never ctx.daily's last row."""
        return None


REGISTRY: dict[str, type[Strategy]] = {}


def register(name: str):
    def deco(cls):
        cls.name = name
        REGISTRY[name] = cls
        return cls
    return deco


def build_strategy(name: str, cfg: dict) -> Strategy:
    import trader.strategies.weekly_rsi_sma20    # noqa: F401  (import registers; add new strategies here)
    import trader.strategies.breakout_trend      # noqa: F401
    import trader.strategies.pullback_ma         # noqa: F401
    import trader.strategies.sr_reversal         # noqa: F401
    import trader.strategies.inside_bar_breakout # noqa: F401
    import trader.strategies.new_high_momentum   # noqa: F401
    import trader.strategies.vcp_breakout        # noqa: F401
    import trader.strategies.high_tight_flag     # noqa: F401
    if name not in REGISTRY:
        raise KeyError(f"No strategy class registered for config '{name}'. Known: {list(REGISTRY)}")
    return REGISTRY[name](cfg)