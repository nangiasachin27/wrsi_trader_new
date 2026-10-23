from __future__ import annotations

from trader.core.indicators import cup_shape, near_ath
from trader.core.models import Signal
from trader.filters.base import Filter, FilterResult, register
from trader.strategies.base import SymbolContext


@register("quality")
class QualityFilter(Filter):
    """Ranking (not a hard gate, by default): scores a candidate on ATH proximity and cup
    formation. mode 'rank' (default) lets a zero-score candidate still be taken if there's a
    free slot - it only loses the tiebreak to a higher-scoring one. mode 'filter' turns this
    into a hard gate: a candidate below min_score is excluded outright, regardless of slots.

    `exempt_strategies` (opt-OUT, default empty - every other filter in this codebase that
    distinguishes strategies is opt-IN instead, e.g. regime_match/market_efficiency's
    `applicable_strategies`; this one is deliberately the other way round because quality has
    always applied to every strategy uniformly, so flipping the default here would silently
    change behavior for strategies nobody asked to touch): a strategy named here is NEVER
    scored or gated by this filter, full stop, regardless of mode. Added for sr_reversal -
    this filter's ATH-proximity/cup-formation checks are near-ATH, trend-continuation
    signatures. sr_reversal's whole premise is buying a stock that just fell AWAY from its
    highs into a support zone and bounced - almost the opposite of "near its all-time high" -
    so this filter was structurally fighting that strategy's own logic, not complementing it,
    and is very likely why sr_reversal's surviving trades looked trend-correlated instead of
    showing genuine reversal character."""

    def evaluate(self, signal: Signal, ctx: SymbolContext) -> FilterResult:
        if not self.enabled:
            return FilterResult(True, "disabled")
        c = self.cfg
        if signal.strategy in c.get("exempt_strategies", []):
            return FilterResult(True, f"{signal.strategy} is exempt from the quality filter")
        score, notes = 0, {}
        if c["ath"]["enabled"]:
            ath_ok, dist = near_ath(ctx.daily, c["ath"])
            notes["ath_ok"], notes["ath_dist_pct"] = ath_ok, dist
            score += int(ath_ok)
        if c["cup"]["enabled"]:
            w = ctx.weekly(c.get("weekly_rule", "W-FRI"), include_partial=False)
            cup_ok, info = cup_shape(w["close"], c["cup"])
            notes["cup_ok"], notes["cup"] = cup_ok, (info if cup_ok else None)
            score += int(cup_ok)
        passed = not (c["mode"] == "filter" and score < c["min_score"])
        return FilterResult(passed, f"quality score {score}", score=score, meta=notes)