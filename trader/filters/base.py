from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date

from trader.core.models import Signal
from trader.data.base import DataProvider
from trader.strategies.base import SymbolContext


@dataclass
class FilterResult:
    passed: bool
    reason: str
    score: float = 0.0          # contributes to ranking; 0 for pure hard-gates
    meta: dict = field(default_factory=dict)


class Filter(ABC):
    """A pluggable overlay applied to a strategy's raw candidate signal, AFTER the strategy's
    own core rule has already decided a symbol qualifies. Strategies stay strategy-specific
    (their own entry condition); filters are strategy-agnostic and apply identically to every
    strategy's candidates, so adding a second strategy later gets every existing filter for
    free without re-implementing any of it.

    STRATEGY-OWNED applicability (backlog item from the 2026-10-25 session, replacing the old
    per-filter exempt_strategies/applicable_strategies/applicable_regimes maps): whether a
    given strategy participates in a given filter at all is now decided by that STRATEGY's own
    config (`exclude_filters: [...]` - see trader/strategies/base.py and each strategy's own
    config _note), not by a map living inside the filter's config. trader.engine.orchestrator's
    candidate loop skips calling evaluate() for any filter named in the originating strategy's
    `exclude_filters` before a filter is ever consulted - so by the time evaluate() runs for a
    given signal, the strategy has already opted in, full stop. This is what makes the
    "filters are strategy-agnostic" claim above actually true in code again: a filter no
    longer needs (and should not re-implement) its own opt-in/opt-out logic keyed on
    `signal.strategy`.

    A filter that still needs a per-strategy PARAMETER (not just a yes/no) - e.g. regime_match
    needs each strategy's own market-regime tag, market_efficiency can take a per-strategy
    efficiency-threshold override - reads it from the ORIGINATING STRATEGY's own config via
    `self._strategy_cfgs[signal.strategy]`, bound once by bind_strategy_cfgs() below (called by
    Engine.__init__ right after both the strategies and filters dicts exist). That data lives
    in each strategy's own config too (e.g. `"market_regime_tag": "UPTREND"`), for the same
    reason exclude_filters does: one strategy config is the single place to see everything
    that strategy runs with, instead of needing to cross-reference every filter's own config.

    prepare() runs ONCE per eod, before any candidate is evaluated - use it to compute or
    cache anything shared (a market-wide regime check, a sector benchmark's status per
    industry) so evaluate() doesn't refetch the same data once per symbol. A filter that
    ignores `signal`/`ctx` in evaluate() and only reads what prepare() cached is, in effect,
    a portfolio-wide filter; one that looks at `signal.symbol` or `ctx` per call is a
    per-candidate filter. Same contract either way - no separate base classes needed.

    A candidate is excluded if ANY enabled, non-excluded filter's evaluate() returns
    passed=False. A candidate's final ranking score (used only to break ties for scarce
    max_positions slots - see trader.risk.portfolio) is the AVERAGE, across every non-excluded
    filter that actually scores (max_score() > 0), of that filter's score/max_score() - NOT a
    raw sum (see max_score()'s own docstring below for why summing was a cross-strategy bias).
    """

    name: str = ""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._strategy_cfgs: dict[str, dict] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.get("enabled", False))

    def bind_strategy_cfgs(self, strategy_cfgs: dict[str, dict]) -> None:
        """Called once by Engine.__init__, after both the strategies and filters dicts exist,
        with {strategy_name: strategy.cfg} for every currently-ENABLED strategy (a disabled
        strategy never produces a candidate signal, so it never needs to be looked up here).
        Only filters that read a per-strategy PARAMETER (not just applicability, which is
        already handled upstream by exclude_filters - see the class docstring) need to use
        this; most filters can ignore it entirely. Default: just stores it."""
        self._strategy_cfgs = strategy_cfgs

    def prepare(self, asof: date, provider: DataProvider) -> None:
        """Optional: precompute/cache shared state once per eod run. Default: no-op."""

    def max_score(self) -> float:
        """The highest value this filter's own evaluate() can return as FilterResult.score,
        GIVEN this filter's own current config (sub-checks it has turned on, enabled flag).
        Added 2026-10-26 to fix a cross-strategy slot-ranking bias (see
        strategy-performance-log.md / filter-tuning-master-table.md): orchestrator.run_eod
        used to sum every non-excluded filter's raw score straight into signal.score, which
        systematically favored whichever strategy's winning combo happened to include MORE
        scoring filters, or filters with a WIDER native scale (e.g. trend_strength's 0-2 vs
        sector_regime's 0-1) - nothing to do with actual signal quality. A strategy with a
        single max-1 scoring filter (pullback_ma: sector_regime alone) was getting crushed in
        slot competition against one summing two max-2 filters (sr_reversal: trend_strength +
        rs_rank), independent of which candidate was actually the better pick that day.
        orchestrator.run_eod now normalizes each scoring filter's contribution to score/max_score
        before AVERAGING (not summing) across a strategy's own scoring filters - so every
        strategy's final ranking score lands in the same [0, 1] range regardless of how many
        scoring filters it uses or their native scales. Default: 0.0 - a filter that never sets
        FilterResult.score (a pure pass/fail gate, e.g. resistance/market_regime/regime_match)
        correctly contributes NOTHING to the average (excluded, not counted as a zero) rather
        than dragging it down; override this in any filter whose evaluate() DOES set a score,
        returning 0.0 whenever this filter is currently disabled (same reasoning: a disabled
        filter must not drag the average down with a spurious zero either) or whenever a
        sub-check that contributes to its own max is itself turned off."""
        return 0.0

    def describe(self) -> str:
        """One-line summary for the startup 'FILTERS ON' log (Engine._log_filter_status).
        Default: surface whichever of these common knobs this filter's cfg actually has, so
        every filter gets a reasonable line for free; override in a subclass for something
        more specific (e.g. naming the benchmark symbol or which sub-checks are on). Per-
        strategy applicability is no longer part of a filter's own description - see each
        strategy's own config for its `exclude_filters` list instead."""
        c = self.cfg
        bits = []
        if "mode" in c:
            bits.append(f"mode={c['mode']}")
        if "min_score" in c:
            bits.append(f"min_score={c['min_score']}")
        for key in ("benchmark_symbol", "proximity_pct", "sma_period", "timeframe_rule", "fail_mode"):
            if key in c:
                bits.append(f"{key}={c[key]}")
        return ", ".join(bits) if bits else "enabled"

    @abstractmethod
    def evaluate(self, signal: Signal, ctx: SymbolContext) -> FilterResult: ...


REGISTRY: dict[str, type[Filter]] = {}


def register(name: str):
    def deco(cls):
        cls.name = name
        REGISTRY[name] = cls
        return cls
    return deco


def build_filter(name: str, cfg: dict) -> Filter:
    import trader.filters.market_regime  # noqa: F401  (import registers; add new filters here)
    import trader.filters.quality        # noqa: F401
    import trader.filters.resistance     # noqa: F401
    import trader.filters.sector_regime  # noqa: F401
    import trader.filters.trend_strength # noqa: F401
    import trader.filters.rs_rank        # noqa: F401
    import trader.filters.regime_match   # noqa: F401
    import trader.filters.market_efficiency  # noqa: F401
    if name not in REGISTRY:
        raise KeyError(f"No filter class registered for config '{name}'. Known: {list(REGISTRY)}")
    return REGISTRY[name](cfg)