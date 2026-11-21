from __future__ import annotations

import logging
from datetime import date

from trader.core.models import Signal
from trader.data.base import DataProvider
from trader.filters.base import Filter, FilterResult, register
from trader.risk.regime import RegimeFilter, RegimeStatus
from trader.strategies.base import SymbolContext

log = logging.getLogger(__name__)


@register("regime_exposure")
class RegimeExposureThrottle(Filter):
    """Portfolio-level EXPOSURE THROTTLE, not a per-candidate gate (added 2026-11-15).

    Motivation: every recent full-window run is net negative in 2025-2026 while 2021/2023 carry
    the CAGR - the long-only trend strategies keep deploying full risk into choppy/corrective
    tape. market_regime already BLOCKS all entries for the strategies that include it, but it is
    binary and most daily strategies exclude it (it hurt their sweeps). This throttle is the
    graded middle ground, applied to EVERY strategy uniformly: in a bear regime it lowers the
    effective max_positions for NEW entries and scales down each new entry's risk amount.
    It never touches exits, stops, pyramids on held positions, or anything already open.

    Regime test: same RegimeFilter as market_regime/sector_regime (weekly close of
    benchmark_symbol vs its weekly SMA). While the benchmark is NOT bullish:
      - effective max positions = floor(max_positions * bear_max_positions_pct / 100); new
        entries are only granted slots down to that cap (held positions above the cap are
        untouched - they just mean no new slots until exits bring the count back under).
      - every granted ENTRY is stamped meta["risk_scale"] = bear_risk_scale, which _buy()
        multiplies into the per-trade risk amount at execution (position caps still apply).

    evaluate() always passes and never scores: candidates are not gated here (that's
    market_regime's job, where a strategy's sweep earned it). The engine reads the throttle via
    throttled() / max_positions_cap() / risk_scale() after prepare(). fail_mode "allow" (default)
    means no benchmark data -> no throttle. Ships disabled; A/B it per window with
    --set regime_exposure.enabled=true before trusting it."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._engine = RegimeFilter(cfg)
        self._status: RegimeStatus | None = None

    def describe(self) -> str:
        c = self.cfg
        return (f"benchmark_symbol={c.get('benchmark_symbol')}, sma_period={c.get('sma_period')}, "
                f"bear_max_positions_pct={c.get('bear_max_positions_pct')}, "
                f"bear_risk_scale={c.get('bear_risk_scale')}")

    def prepare(self, asof: date, provider: DataProvider) -> None:
        if not self.enabled:
            self._status = None
            return
        try:
            bench = provider.get_daily(self.cfg["benchmark_symbol"], asof)
        except Exception as e:  # noqa: BLE001 - a data problem must not abort the whole EOD
            log.warning("regime_exposure: benchmark %s fetch failed (%s) - throttle off today",
                        self.cfg.get("benchmark_symbol"), e)
            self._status = None
            return
        self._status = self._engine.status(bench, asof, self.cfg["benchmark_symbol"])
        log.info("FILTER regime_exposure %s | %s", asof,
                 "BULLISH - full exposure" if self._status.bullish else
                 f"NOT BULLISH - new entries throttled ({self.reason()})")

    def evaluate(self, signal: Signal, ctx: SymbolContext) -> FilterResult:
        # Never gates and never scores (max_score stays 0): the throttle acts on slots and
        # sizing in the engine, not on individual candidates.
        if not self.enabled:
            return FilterResult(True, "disabled")
        return FilterResult(True, "portfolio exposure throttle - does not gate candidates")

    # ---- throttle API (read by Engine.run_eod after prepare) ---------------------------------
    def throttled(self) -> bool:
        return self.enabled and self._status is not None and not self._status.bullish

    def max_positions_cap(self, max_positions: int) -> int:
        if not self.throttled():
            return max_positions
        return int(max_positions * float(self.cfg.get("bear_max_positions_pct", 100.0)) / 100.0)

    def risk_scale(self) -> float:
        return float(self.cfg.get("bear_risk_scale", 1.0)) if self.throttled() else 1.0

    def reason(self) -> str:
        return self._status.reason if self._status is not None else "no regime status"
