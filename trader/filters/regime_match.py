from __future__ import annotations

import json
import logging
from datetime import date

from trader.core.config import ROOT
from trader.core.models import Signal
from trader.core.trend_structure import classify_trend_structure
from trader.filters.base import Filter, FilterResult, register
from trader.strategies.base import SymbolContext

log = logging.getLogger(__name__)


@register("regime_match")
class RegimeMatchFilter(Filter):
    """Blocks a candidate unless the CURRENT general-market trend (trader.core.trend_structure's
    UPTREND/SIDEWAYS/DOWNTREND state machine - the course's own swing-structure method: higher
    highs + higher lows confirmed after 2+ swings, mirror for downtrend, everything else
    including a break of market structure is SIDEWAYS) is one this candidate's own STRATEGY
    declared itself applicable to. Each new swing strategy was written with a specific market
    phase in mind per the course (breakouts and pullbacks want an UPTREND; a support-zone fade
    wants a ranging/SIDEWAYS tape) - this filter is what actually enforces that tag at runtime,
    read from this file's own config rather than hard-coded in each strategy class, so
    re-tagging a strategy later is a one-line config edit, not a code change.

    NOTE: this used to run on trader.core.market_trend's IBD-style VD-day/follow-through-day
    engine. That module is untouched and still importable (scripts/market_trend.py still uses
    it), but this filter now reads trader.core.trend_structure instead - a different, simpler
    algorithm that matches what the price-action course actually teaches, built directly on
    the same pivot_highs/pivot_lows swing-detection trader.core.levels already uses for S/R, so
    "swing high/low" means the same thing to both.

    Computes the benchmark's swing-structure state ONCE per eod (prepare()) by reading
    `benchmark_symbol`'s own daily bars - no universe-wide volume aggregation needed here
    (trend_structure's classifier is price/swing-based, not volume-based), unlike the old
    market_trend-based version of this filter.

    The regime tag itself (`"market_regime_tag": "UPTREND"`/`"SIDEWAYS"`/`"DOWNTREND"`) now
    lives in each STRATEGY's own config, not in a map here (backlog item from the 2026-10-25
    session - see trader/filters/base.py's class docstring on bind_strategy_cfgs()). A strategy
    with no `market_regime_tag` in its own config is NOT gated (always passes) - tagging is
    opt-in per strategy, so existing strategies (e.g. weekly_rsi_sma20) are unaffected unless
    their own config sets a tag. Whether a strategy participates in this filter AT ALL is a
    separate, upstream decision made via that strategy's own `exclude_filters` list (see
    trader/engine/orchestrator.py's candidate loop) - a strategy only needs a `market_regime_tag`
    if it hasn't excluded this filter."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._state: str | None = None
        self._reason: str = ""

    def describe(self) -> str:
        c = self.cfg
        return f"benchmark_symbol={c.get('benchmark_symbol')}, fail_mode={c.get('fail_mode')}"

    def _trend_structure_cfg(self) -> dict:
        p = ROOT / self.cfg.get("trend_structure_config_path", "config/trend_structure.json")
        return json.loads(p.read_text())

    def prepare(self, asof: date, provider) -> None:
        if not self.enabled:
            self._state = None
            return

        ts_cfg = self._trend_structure_cfg()
        bench_symbol = ts_cfg.get("benchmark_symbol", self.cfg.get("benchmark_symbol", "NIFTY500"))
        try:
            bench_daily = provider.get_daily(bench_symbol, asof)
        except Exception as exc:  # noqa: BLE001
            log.warning("regime_match: benchmark '%s' fetch failed: %s", bench_symbol, exc)
            self._state = None
            return

        out, events = classify_trend_structure(bench_daily, ts_cfg)
        if out.empty:
            self._state = ts_cfg.get("initial_state", "SIDEWAYS")
            self._reason = "no classified history yet"
        else:
            self._state = str(out["state"].iloc[-1])
            self._reason = events[-1].reason if events else "no transition yet"
        log.info("FILTER regime_match %s | state=%s (%s)", asof, self._state, self._reason)

    def evaluate(self, signal: Signal, ctx: SymbolContext) -> FilterResult:
        if not self.enabled:
            return FilterResult(True, "disabled")

        # Check the tag FIRST, before caring whether the market state could even be computed -
        # an untagged strategy (e.g. weekly_rsi_sma20) must never be gated or need a benchmark
        # at all, regardless of whether NIFTY500/the universe happen to be available this run
        # (small/synthetic test fixtures in particular have neither). Tag now comes from the
        # ORIGINATING STRATEGY's own config (bound via bind_strategy_cfgs()), not a map here.
        strat_cfg = self._strategy_cfgs.get(signal.strategy, {})
        allowed = strat_cfg.get("market_regime_tag")
        if allowed is None:
            return FilterResult(True, f"{signal.strategy} has no regime tag - not gated")
        if isinstance(allowed, str):
            allowed = [allowed]  # a strategy's own config states one tag, not a list - normalize here

        if self._state is None:
            fail_open = self.cfg.get("fail_mode", "block") == "allow"
            return FilterResult(fail_open, "market regime unavailable this eod")

        passed = self._state in allowed
        return FilterResult(passed, f"market regime {self._state} (strategy wants {allowed})",
                            meta={"state": self._state})
