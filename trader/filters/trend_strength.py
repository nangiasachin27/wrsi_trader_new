from __future__ import annotations

import pandas as pd

from trader.core.indicators import adx, sma
from trader.core.models import Signal
from trader.filters.base import Filter, FilterResult, register
from trader.strategies.base import SymbolContext


@register("trend_strength")
class TrendStrengthFilter(Filter):
    """Per-candidate confirmation that a weekly-RSI>60 entry is riding a real, strong trend -
    not a choppy-range spike that the lagging weekly-wick-vs-SMA20 exit won't catch until it's
    already given back several percent. Two independently toggleable sub-checks, combined the
    same way as the quality filter's ath/cup checks:

      - adx: weekly ADX(period) >= min_adx. ADX measures trend STRENGTH, not direction, so it's
        a genuine complement to the strategy's own RSI>60 (direction), not a duplicate of it -
        it filters out weak/choppy RSI pops that happen without a real directional move behind
        them.
      - sma_trend: weekly close above weekly SMA(period) * (1+buffer_pct/100). A higher-timeframe
        trend-alignment check layered on top of the strategy's own weekly/monthly RSI>60 -
        typically run with period=50 to require the stock to already be in a multi-month uptrend,
        not just popping RSI off a short-term low.

    mode 'rank' (default) only affects the tiebreak for a free slot when candidates outnumber
    slots; mode 'filter' is a hard gate - any candidate scoring below min_score is excluded
    outright, regardless of free slots. Same contract as config/filters/quality.json, for
    consistency across filters.
    """

    def describe(self) -> str:
        c = self.cfg
        bits = [f"mode={c['mode']}", f"min_score={c['min_score']}"]
        if c["adx"]["enabled"]:
            bits.append(f"adx>={c['adx']['min_adx']} (period {c['adx']['period']})")
        if c["sma_trend"]["enabled"]:
            bits.append(f"close>SMA{c['sma_trend']['period']}")
        return ", ".join(bits)

    def evaluate(self, signal: Signal, ctx: SymbolContext) -> FilterResult:
        if not self.enabled:
            return FilterResult(True, "disabled")
        c = self.cfg
        w = ctx.weekly(c.get("weekly_rule", "W-FRI"), include_partial=False)
        score, notes = 0, {}

        if c["adx"]["enabled"]:
            period = c["adx"]["period"]
            if len(w) < period * 2:
                notes["adx_ok"], notes["adx_val"] = False, None
            else:
                a = adx(w["high"], w["low"], w["close"], period)
                val = float(a.iloc[-1]) if len(a) and not pd.isna(a.iloc[-1]) else None
                ok = val is not None and val >= c["adx"]["min_adx"]
                notes["adx_ok"], notes["adx_val"] = ok, (round(val, 1) if val is not None else None)
                score += int(ok)

        if c["sma_trend"]["enabled"]:
            period = c["sma_trend"]["period"]
            if len(w) < period:
                notes["sma_trend_ok"], notes["sma_dist_pct"] = False, None
            else:
                s = sma(w["close"], period)
                level = float(s.iloc[-1])
                px = float(w["close"].iloc[-1])
                ok = bool(level and px > level * (1 + c["sma_trend"]["buffer_pct"] / 100.0))
                dist_pct = (px - level) / level * 100.0 if level else None
                notes["sma_trend_ok"] = ok
                notes["sma_dist_pct"] = round(dist_pct, 1) if dist_pct is not None else None
                score += int(ok)

        passed = not (c["mode"] == "filter" and score < c["min_score"])
        parts = []
        if "adx_val" in notes:
            parts.append(f"ADX {notes['adx_val']}" if notes["adx_val"] is not None else "ADX n/a")
        if "sma_dist_pct" in notes:
            parts.append(f"price {notes['sma_dist_pct']:+.1f}% vs SMA{c['sma_trend']['period']}"
                        if notes["sma_dist_pct"] is not None else "SMA n/a")
        reason = f"trend_strength score {score}" + (f" ({', '.join(parts)})" if parts else "")
        return FilterResult(passed, reason, score=score, meta=notes)

    def max_score(self) -> float:
        """adx + sma_trend, each worth 1 point when its own sub-check is turned on - see
        Filter.max_score()'s docstring."""
        if not self.enabled:
            return 0.0
        c = self.cfg
        return float(int(c["adx"]["enabled"]) + int(c["sma_trend"]["enabled"]))