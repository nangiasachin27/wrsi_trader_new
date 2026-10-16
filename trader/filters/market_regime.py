from __future__ import annotations

import logging
from datetime import date

from trader.core.models import Signal
from trader.data.base import DataProvider
from trader.filters.base import Filter, FilterResult, register
from trader.risk.regime import RegimeFilter, RegimeStatus
from trader.strategies.base import SymbolContext

log = logging.getLogger(__name__)


@register("market_regime")
class MarketRegimeFilter(Filter):
    """Portfolio-wide: blocks ALL new entries when the benchmark isn't itself trending up.
    Computed once per eod (prepare), then the same cached status is returned for every
    candidate that eod - this is a portfolio-wide check, not a per-symbol one."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._engine = RegimeFilter(cfg)
        self._status: RegimeStatus | None = None

    def describe(self) -> str:
        c = self.cfg
        return (f"benchmark_symbol={c.get('benchmark_symbol')}, sma_period={c.get('sma_period')}, "
                f"fail_mode={c.get('fail_mode')}")

    def prepare(self, asof: date, provider: DataProvider) -> None:
        if not self.enabled:
            self._status = None
            return
        bench = provider.get_daily(self.cfg["benchmark_symbol"], asof)
        self._status = self._engine.status(bench, asof, self.cfg["benchmark_symbol"])
        log.info("FILTER market_regime %s | %s", asof,
                 "BULLISH - entries allowed" if self._status.bullish else "NOT BULLISH - new entries blocked")
        log.info("  %s", self._status.reason)

    def evaluate(self, signal: Signal, ctx: SymbolContext) -> FilterResult:
        if not self.enabled:
            return FilterResult(True, "disabled")
        if self._status is None:
            return FilterResult(False, "prepare() was not called before evaluate()")
        return FilterResult(self._status.bullish, self._status.reason)