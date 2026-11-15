from __future__ import annotations

import json

from trader.core.config import ROOT
from trader.core.fractal_levels import fractal_result, resolve_engine
from trader.core.levels import nearest_levels, support_resistance
from trader.core.models import Signal
from trader.filters.base import Filter, FilterResult, register
from trader.strategies.base import SymbolContext


@register("room_to_resistance")
class RoomToResistance(Filter):
    """Blocks an ENTRY whose reward:risk to the nearest overhead resistance is too small.

        reward = nearest resistance zone's near (lower) edge - entry reference price
        risk   = entry reference price - the signal's own stop
        block if reward / risk < min_reward_risk

    Unlike the `resistance` filter (a flat 'too close below a zone' band), this uses each
    strategy's actual stop, so a tight-stop setup is allowed closer to resistance than a
    wide-stop one - the same R:R a trader would compute before clicking buy.

    Strategy-agnostic gate (max_score 0, never ranks). A strategy that BUYS THROUGH resistance
    (breakout_trend, inside_bar_breakout, vcp_breakout, high_tight_flag) excludes it via its own
    `exclude_filters` - for those, the zone overhead is the trigger, not an obstacle.

    Entry reference price = signal.ref_price (the close the signal was computed on; the real fill
    is the next open). A signal without a usable stop (None or >= price) is not judged (passes).
    No resistance above price = open upside -> fail_mode decides (default allow).

    Engine: "engine" in this filter's config, else config/sr_engine.json. Legacy = nearest zone
    from trader/core/levels.py with config/levels.json[timeframe]; fractal = trader/core/
    fractal_levels.py (a zone price is already inside counts, giving reward <= 0 -> blocked).
    Ships disabled (added 2026-11-14) - sweep it per strategy with scripts/tune_strategy.py first."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._legacy_cfg: dict | None = None

    def _levels_cfg(self, timeframe: str) -> dict:
        if self._legacy_cfg is None:
            p = ROOT / self.cfg.get("levels_config_path", "config/levels.json")
            all_cfg = json.loads(p.read_text())
            if timeframe not in all_cfg:
                raise KeyError(f"room_to_resistance: timeframe '{timeframe}' not in {p}")
            self._legacy_cfg = all_cfg[timeframe]
        return self._legacy_cfg

    def describe(self) -> str:
        c = self.cfg
        return (f"engine={resolve_engine(c.get('engine'))}, timeframe={c.get('timeframe', 'daily')}, "
                f"min_reward_risk={c.get('min_reward_risk')}, fail_mode={c.get('fail_mode', 'allow')}")

    def evaluate(self, signal: Signal, ctx: SymbolContext) -> FilterResult:
        if not self.enabled:
            return FilterResult(True, "disabled")
        c = self.cfg
        timeframe = c.get("timeframe", "daily")
        px = float(signal.ref_price) if signal.ref_price else ctx.last_close
        stop = signal.stop
        if stop is None or stop >= px:
            return FilterResult(True, f"no usable stop ({stop}) below price {px:.2f} - not judged")

        engine = resolve_engine(c.get("engine"))
        if engine == "fractal":
            r = fractal_result(ctx.daily, timeframe, px, ctx.asof)
            z = r.res[0] if r.res else None
            res_low, res_high = (z.low, z.high) if z else (None, None)
        else:
            lv = nearest_levels(support_resistance(ctx.daily, timeframe, self._levels_cfg(timeframe)), px)["resistance"]
            res_low, res_high = (lv.price_low, lv.price_high) if lv else (None, None)

        if res_low is None:
            return FilterResult(c.get("fail_mode", "allow") == "allow",
                                f"[{engine}] no {timeframe} resistance above {px:.2f} (open upside)")
        reward, risk = res_low - px, px - stop
        rr = reward / risk
        need = float(c["min_reward_risk"])
        return FilterResult(rr >= need,
                            f"[{engine}] R:R {rr:.2f} vs min {need:g}: reward {reward:.2f} to {timeframe} "
                            f"resistance {res_low:.2f}-{res_high:.2f}, risk {risk:.2f} to stop {stop:.2f}",
                            meta={"reward_risk": round(rr, 2), "resistance_low": res_low,
                                  "resistance_high": res_high, "engine": engine})
