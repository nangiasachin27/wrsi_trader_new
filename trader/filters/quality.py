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

    Per-strategy applicability (e.g. sr_reversal's exemption - this filter's ATH-proximity/
    cup-formation checks are near-ATH, trend-continuation signatures, which is close to the
    OPPOSITE of sr_reversal's own entry: a stock that just fell AWAY from its highs into a
    support zone and bounced) is no longer decided here - it lives in sr_reversal's own
    config as `"exclude_filters": ["quality", ...]` (see trader/filters/base.py's class
    docstring and trader/engine/orchestrator.py's candidate loop, which skips calling this
    filter at all for an excluded strategy's candidates - this filter never sees them)."""

    def evaluate(self, signal: Signal, ctx: SymbolContext) -> FilterResult:
        if not self.enabled:
            return FilterResult(True, "disabled")
        c = self.cfg
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

    def max_score(self) -> float:
        """ath + cup, each worth 1 point when its own sub-check is turned on - see
        Filter.max_score()'s docstring for why this must match evaluate()'s own ceiling
        exactly (0 whenever disabled, so a disabled filter never drags the cross-strategy
        ranking average down with a spurious zero)."""
        if not self.enabled:
            return 0.0
        c = self.cfg
        return float(int(c["ath"]["enabled"]) + int(c["cup"]["enabled"]))