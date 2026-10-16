from __future__ import annotations

from trader.core.models import Signal
from trader.filters.base import Filter, FilterResult, register
from trader.strategies.base import SymbolContext


@register("resistance")
class ResistanceFilter(Filter):
    """Blocks a candidate whose current price sits just below a prior swing high (a level
    the stock previously failed to break through), rather than clear of it or already above
    it. The resistance level is the highest 'basis' price over a lookback window that
    EXCLUDES the most recent exclude_recent_weeks weeks - excluding recent weeks matters:
    without it, a stock's own current rally would be measured against its own high, which is
    circular (nothing is ever 'near resistance' relative to itself). Price already above the
    old resistance (a genuine breakout) passes - only the narrow band just underneath it is
    blocked. Complements the quality filter's ATH check: a stock already near its all-time
    high has, by definition, no old resistance overhead - this mainly matters for candidates
    that AREN'T near ATH."""

    def evaluate(self, signal: Signal, ctx: SymbolContext) -> FilterResult:
        if not self.enabled:
            return FilterResult(True, "disabled")
        c = self.cfg
        w = ctx.weekly(c.get("weekly_rule", "W-FRI"), include_partial=False)
        need = c["lookback_weeks"] + c["exclude_recent_weeks"]
        if len(w) < need:
            fail_open = c.get("fail_mode", "block") == "allow"
            return FilterResult(fail_open, f"only {len(w)} weekly bars, need {need}")

        end = -c["exclude_recent_weeks"] if c["exclude_recent_weeks"] else None
        window = w.iloc[-need:end]
        resistance = float(window[c["basis"]].max())
        px = ctx.last_close
        dist_pct = (resistance - px) / resistance * 100.0
        near = 0 <= dist_pct <= c["proximity_pct"]
        reason = (f"price {px:.2f} vs resistance {resistance:.2f} "
                 f"({dist_pct:+.1f}% {'below' if dist_pct >= 0 else 'above'})")
        return FilterResult(not near, reason, meta={"resistance": resistance, "dist_pct": round(dist_pct, 1)})
