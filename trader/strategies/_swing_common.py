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


def chandelier_trail(daily, opened: str | None, period: int, atr_mult: float) -> float | None:
    """Highest close SINCE ENTRY, minus atr_mult * ATR(period) - a volatility-adaptive trailing
    stop (classic trend-following "Chandelier Exit"), used instead of a fixed-period SMA trail
    for strategies meant to ride a trend: it widens automatically for a volatile name and
    tightens for a calm one, rather than clipping every stock on the same fixed lookback
    regardless of how it actually moves. Highest close is capped to bars on/after `opened`
    (pos.opened, an ISO date string) so a huge pre-entry spike can't set an unreachable trail -
    the trail only ever reflects what happened WHILE the position has been open."""
    a = atr(daily["high"], daily["low"], daily["close"], period).iloc[-1]
    if a != a:
        return None
    window = daily if not opened else daily.loc[opened:]
    if window.empty:
        return None
    highest_close = float(window["close"].max())
    return highest_close - float(a) * atr_mult


def nearest_zones(daily, timeframe: str, cfg: dict, price: float) -> dict:
    zones = support_resistance(daily, timeframe, cfg)
    return nearest_levels(zones, price)