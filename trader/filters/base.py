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

    prepare() runs ONCE per eod, before any candidate is evaluated - use it to compute or
    cache anything shared (a market-wide regime check, a sector benchmark's status per
    industry) so evaluate() doesn't refetch the same data once per symbol. A filter that
    ignores `signal`/`ctx` in evaluate() and only reads what prepare() cached is, in effect,
    a portfolio-wide filter; one that looks at `signal.symbol` or `ctx` per call is a
    per-candidate filter. Same contract either way - no separate base classes needed.

    A candidate is excluded if ANY enabled filter's evaluate() returns passed=False.
    A candidate's final ranking score is its own signal.score plus every filter's score.
    """

    name: str = ""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.get("enabled", False))

    def prepare(self, asof: date, provider: DataProvider) -> None:
        """Optional: precompute/cache shared state once per eod run. Default: no-op."""

    def describe(self) -> str:
        """One-line summary for the startup 'FILTERS ON' log (Engine._log_filter_status).
        Default: surface whichever of these common knobs this filter's cfg actually has, so
        every filter gets a reasonable line for free; override in a subclass for something
        more specific (e.g. naming the benchmark symbol or which sub-checks are on)."""
        c = self.cfg
        bits = []
        if "mode" in c:
            bits.append(f"mode={c['mode']}")
        if "min_score" in c:
            bits.append(f"min_score={c['min_score']}")
        for key in ("benchmark_symbol", "proximity_pct", "sma_period", "timeframe_rule", "fail_mode"):
            if key in c:
                bits.append(f"{key}={c[key]}")
        if c.get("exempt_strategies"):
            bits.append(f"exempt_strategies={c['exempt_strategies']}")
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