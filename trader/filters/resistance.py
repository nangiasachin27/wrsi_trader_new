from __future__ import annotations

import json

from trader.core.config import ROOT
from trader.core.levels import nearest_levels, support_resistance
from trader.core.models import Signal
from trader.filters.base import Filter, FilterResult, register
from trader.strategies.base import SymbolContext


@register("resistance")
class ResistanceFilter(Filter):
    """Blocks a candidate whose current price sits just below the nearest overhead
    support/resistance ZONE - pivot-clustered swing highs that have been touched
    >= min_touches times - rather than clear of it or already above it.

    Previously this filter's 'resistance' was the single highest weekly high over a
    52-week lookback (excluding the most recent weeks). That's a coarse, single-point
    measure: it can sit far above price (irrelevant to an entry happening now) and
    doesn't require the level to have actually been tested more than once. Swapping
    in trader.core.levels' pivot-clustering means this filter now checks the SAME
    nearest zone a human would read off `scripts/levels.py` for this stock on this
    date - the thing actually meant by "resistance" in this project - instead of a
    simple rolling max.

    Zone-finding params (pivot window, cluster tolerance, min_touches, lookback_bars)
    come from config/levels.json's section for `timeframe`, shared with
    scripts/levels.py and the candlestick module rather than duplicated here. Only
    `timeframe` and `proximity_pct` (how close below the zone counts as "too close")
    live in this filter's own config.

    No explicit exclude_recent_weeks is needed here (unlike the old rolling-max
    version): a pivot can't be confirmed until `pivot_right` bars after it exist, so
    the most recent bars are automatically unable to form a pivot yet - a stock's own
    current rally can't become "resistance" against itself just by prepare()/evaluate()
    running on it.

    Price already ABOVE the nearest zone, or with no zone above it at all (fail_mode
    decides), always passes - only the narrow band just underneath a real, multi-touch
    level is blocked."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._levels_cfg_by_timeframe: dict[str, dict] = {}

    def _levels_cfg(self) -> dict:
        timeframe = self.cfg.get("timeframe", "weekly")
        if timeframe not in self._levels_cfg_by_timeframe:
            p = ROOT / self.cfg.get("levels_config_path", "config/levels.json")
            all_cfg = json.loads(p.read_text())
            if timeframe not in all_cfg:
                raise KeyError(f"resistance filter: timeframe '{timeframe}' not in {p}")
            self._levels_cfg_by_timeframe[timeframe] = all_cfg[timeframe]
        return self._levels_cfg_by_timeframe[timeframe]

    def describe(self) -> str:
        c = self.cfg
        return (f"timeframe={c.get('timeframe', 'weekly')}, "
                f"proximity_pct={c.get('proximity_pct')}, fail_mode={c.get('fail_mode', 'block')}")

    def evaluate(self, signal: Signal, ctx: SymbolContext) -> FilterResult:
        if not self.enabled:
            return FilterResult(True, "disabled")
        c = self.cfg
        timeframe = c.get("timeframe", "weekly")
        levels_cfg = self._levels_cfg()

        zones = support_resistance(ctx.daily, timeframe, levels_cfg)
        px = ctx.last_close
        resistance = nearest_levels(zones, px)["resistance"]

        if resistance is None:
            fail_open = c.get("fail_mode", "block") == "allow"
            return FilterResult(fail_open, f"no {timeframe} resistance zone found above price {px:.2f}")

        # compare against the zone's near (lower) edge - the first price the stock would
        # actually have to clear, not the zone's midpoint or far edge
        dist_pct = (resistance.price_low - px) / resistance.price_low * 100.0
        near = 0 <= dist_pct <= c["proximity_pct"]
        reason = (f"price {px:.2f} vs nearest {timeframe} resistance zone "
                 f"{resistance.price_low:.2f}-{resistance.price_high:.2f} (strength {resistance.strength}) "
                 f"({dist_pct:+.1f}% {'below' if dist_pct >= 0 else 'above'})")
        return FilterResult(not near, reason,
                            meta={"resistance_low": resistance.price_low,
                                  "resistance_high": resistance.price_high,
                                  "strength": resistance.strength,
                                  "dist_pct": round(dist_pct, 1)})