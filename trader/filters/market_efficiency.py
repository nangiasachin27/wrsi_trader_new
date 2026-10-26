from __future__ import annotations

import logging
from datetime import date

from trader.core.indicators import efficiency_ratio
from trader.data.base import DataProvider
from trader.core.models import Signal
from trader.filters.base import Filter, FilterResult, register
from trader.strategies.base import SymbolContext

log = logging.getLogger(__name__)


@register("market_efficiency")
class MarketEfficiencyFilter(Filter):
    """Portfolio-wide "is the market actually trending, or just chopping sideways" check,
    computed ONCE per eod (prepare()) from `benchmark_symbol`'s own daily closes - a Kaufman
    Efficiency Ratio (trader.core.indicators.efficiency_ratio): net move over `lookback_days`
    divided by the total distance traveled over the same window. 1.0 = a straight line, near 0
    = pure back-and-forth chop that nets nowhere.

    Built in response to a concrete, measured finding, not a hunch: NIFTY500's own efficiency
    ratio was far higher in 2023 (~0.20 on a yearly basis) than in 2024-2026 (~0.04-0.09) - and
    that is exactly the window where breakout_trend/inside_bar_breakout/weekly_rsi_sma20's
    win-rate and avg-pnl-per-trade all decayed in lockstep, together with rising cross-strategy
    correlation. A shared market-regime effect, not something specific to any one strategy's
    exit logic - so the fix belongs at the market-regime layer, not in another round of
    per-strategy exit tuning.

    Whether a strategy participates in this filter at all is decided upstream by that
    strategy's own `exclude_filters` list (see trader/filters/base.py's class docstring and
    trader/engine/orchestrator.py's candidate loop) - backlog item from the 2026-10-25
    session, replacing the old `applicable_strategies` map that used to live here. sr_reversal
    deliberately excludes this filter: it is a range/reversal play that specifically wants a
    non-trending tape, so penalizing it for low efficiency would be exactly backwards. A
    future iteration might reward sr_reversal for LOW efficiency instead - that is a separate,
    deliberate design choice (tracked separately), not something this filter does on its own.

    A participating strategy can override `default_min_efficiency` for itself via its own
    config's `"market_efficiency_override": {"min_efficiency": ...}` (read via
    bind_strategy_cfgs() - see base.py), instead of a per-strategy dict living here.

    mode 'rank' (default, and the only mode this should run in until it has been backtested
    against real history) NEVER blocks a trade - it only nudges ranking score, exactly like
    quality/trend_strength's own 'rank' mode. mode 'filter' is a hard gate (score < min_score
    -> excluded outright) - do not switch to it without first comparing both modes' actual
    equity-curve impact per strategy; a brand-new filter should not get hard-gating power over
    real trades until its effect has actually been measured, which is the entire point of
    shipping this disabled-by-default in 'rank' mode."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._efficiency: float | None = None
        self._reason: str = "prepare() not yet called"

    def describe(self) -> str:
        c = self.cfg
        return (f"benchmark_symbol={c.get('benchmark_symbol')}, lookback_days={c.get('lookback_days')}, "
                f"mode={c.get('mode')}, default_min_efficiency={c.get('default_min_efficiency')}")

    def prepare(self, asof: date, provider: DataProvider) -> None:
        if not self.enabled:
            self._efficiency = None
            return
        c = self.cfg
        bench_symbol = c.get("benchmark_symbol", "NIFTY500")
        period = c.get("lookback_days", 20)
        try:
            bench = provider.get_daily(bench_symbol, asof)
        except Exception as exc:  # noqa: BLE001
            log.warning("market_efficiency: benchmark '%s' fetch failed: %s", bench_symbol, exc)
            self._efficiency, self._reason = None, f"benchmark fetch failed: {exc}"
            return

        need = max(c.get("min_bars", period + 5), period + 1)
        if bench is None or len(bench) < need:
            self._efficiency = None
            self._reason = f"only {0 if bench is None else len(bench)} benchmark bars, need {need}"
            log.info("FILTER market_efficiency %s | %s", asof, self._reason)
            return

        er = efficiency_ratio(bench["close"], period)
        val = er.iloc[-1]
        if val != val:  # NaN - not warmed up or a flat window (zero path length)
            self._efficiency, self._reason = None, "efficiency ratio not available (NaN)"
        else:
            self._efficiency = float(val)
            self._reason = (f"{bench_symbol} {period}d efficiency ratio {self._efficiency:.3f} "
                            f"(net move / path length, last {period} days)")
        log.info("FILTER market_efficiency %s | %s", asof, self._reason)

    def evaluate(self, signal: Signal, ctx: SymbolContext) -> FilterResult:
        if not self.enabled:
            return FilterResult(True, "disabled")

        # Participation is decided upstream (strategy's own exclude_filters, see
        # trader/engine/orchestrator.py) - evaluate() only runs at all for a strategy that
        # opted in, so there's no "not applicable" branch needed here any more. Just read
        # this strategy's own override (if any) for the one parameter that can vary per
        # strategy - bound via bind_strategy_cfgs(), not a map living in this filter's config.
        override = self._strategy_cfgs.get(signal.strategy, {}).get("market_efficiency_override", {})

        mode = self.cfg.get("mode", "rank")
        if self._efficiency is None:
            fail_open = self.cfg.get("fail_mode", "allow") == "allow"
            passed = True if mode == "rank" else fail_open
            return FilterResult(passed, f"market efficiency unavailable this eod ({self._reason})")

        min_eff = override.get("min_efficiency", self.cfg.get("default_min_efficiency", 0.10))
        ok = self._efficiency >= min_eff
        score = int(ok)
        passed = not (mode == "filter" and score < self.cfg.get("min_score", 1))
        reason = (f"market efficiency {self._efficiency:.3f} {'>=' if ok else '<'} "
                 f"required {min_eff:.3f} for {signal.strategy}")
        return FilterResult(passed, reason, score=score,
                            meta={"efficiency": round(self._efficiency, 3), "min_efficiency": min_eff})