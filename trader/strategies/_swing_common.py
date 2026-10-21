from __future__ import annotations

import json

from trader.core.config import ROOT
from trader.core.indicators import atr, sma
from trader.core.levels import nearest_levels, support_resistance


def levels_cfg(timeframe: str, path: str = "config/levels.json") -> dict:
    """Shared loader so every new strategy reads the SAME S/R zone parameters
    scripts/levels.py and the resistance filter already use, instead of each
    re-typing its own pivot/cluster numbers."""
    all_cfg = json.loads((ROOT / path).read_text())
    if timeframe not in all_cfg:
        raise KeyError(f"levels config has no '{timeframe}' section in {path}")
    return all_cfg[timeframe]


def candlestick_cfg(path: str = "config/candlestick_patterns.json") -> dict:
    return json.loads((ROOT / path).read_text())


def trailing_level(close_series, period: int) -> float | None:
    """Last value of a simple trailing SMA, or None while it's still warming up."""
    s = sma(close_series, period).iloc[-1]
    return None if s != s else float(s)  # NaN check without importing pandas here


def ratchet(prev_stop: float | None, candidate: float | None) -> float | None:
    """Stops only ever tighten, never loosen - same discipline as weekly_rsi_sma20's
    fast_trail. Returning None (rather than prev_stop) when candidate is also None lets
    the engine's `pos.stop = strat.stop_level(...) or pos.stop` fallback keep whatever
    stop was already recorded."""
    if candidate is None:
        return None
    if prev_stop is None:
        return candidate
    return max(prev_stop, candidate)


def atr_buffer(daily, period: int, mult: float) -> float:
    a = atr(daily["high"], daily["low"], daily["close"], period).iloc[-1]
    return float(a) * mult if a == a else 0.0  # NaN -> no buffer rather than crashing


def nearest_zones(daily, timeframe: str, cfg: dict, price: float) -> dict:
    zones = support_resistance(daily, timeframe, cfg)
    return nearest_levels(zones, price)
