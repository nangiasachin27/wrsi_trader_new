from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from trader.core.indicators import candles_for_timeframe


@dataclass
class PatternMatch:
    name: str
    direction: str       # "bullish" | "bearish" | "neutral" (doji: indecision, not a side)
    weight: float         # from config - relative conviction, NOT a probability
    bars_used: int         # 1, 2, or 3 candles, ending at (and including) the queried bar


def _shape(bars: pd.DataFrame, i: int) -> dict:
    o, h, l, c = (float(bars[col].iloc[i]) for col in ("open", "high", "low", "close"))
    rng = h - l
    body = abs(c - o)
    return {
        "o": o, "h": h, "l": l, "c": c, "range": rng, "body": body,
        "body_pct": (body / rng * 100.0) if rng > 0 else 0.0,
        "upper_shadow": h - max(o, c), "lower_shadow": min(o, c) - l,
        "bullish": c > o, "bearish": c < o,
    }


def _prior_trend(bars: pd.DataFrame, i: int, lookback: int) -> str:
    """Heuristic, not a trend-strength indicator: compares the close `lookback` bars before
    candle i to the close just before i. Good enough to disambiguate hammer-vs-hanging-man
    shape (which direction is only known relative to the preceding trend) - not a substitute
    for a real trend filter if a strategy needs one."""
    if i - lookback < 0:
        return "flat"
    before, just_before = bars["close"].iloc[i - lookback], bars["close"].iloc[i - 1]
    if just_before < before:
        return "down"
    if just_before > before:
        return "up"
    return "flat"


PATTERN_REGISTRY: dict[str, Callable] = {}


def register_pattern(name: str):
    def deco(fn):
        PATTERN_REGISTRY[name] = fn
        return fn
    return deco


# ---- single-candle ------------------------------------------------------------------------

@register_pattern("marubozu")
def _marubozu(bars: pd.DataFrame, i: int, cfg: dict) -> PatternMatch | None:
    c = cfg["marubozu"]
    s = _shape(bars, i)
    if s["range"] <= 0 or s["body_pct"] < c["min_body_pct"]:
        return None
    return PatternMatch("marubozu", "bullish" if s["bullish"] else "bearish", c["weight"], 1)


@register_pattern("doji")
def _doji(bars: pd.DataFrame, i: int, cfg: dict) -> PatternMatch | None:
    c = cfg["doji"]
    s = _shape(bars, i)
    if s["range"] <= 0 or s["body_pct"] > c["max_body_pct"]:
        return None
    return PatternMatch("doji", "neutral", c["weight"], 1)


@register_pattern("spinning_top")
def _spinning_top(bars: pd.DataFrame, i: int, cfg: dict) -> PatternMatch | None:
    """Built 2026-10-29, found via a categorization pass on the candlestick filter's own
    "no pattern matched" bucket (the ~80-95% majority case the 2026-10-29 contributes_to_score
    fix abstains on) - see strategy-performance-log.md's "Candlestick unrecognized-pattern
    categorization" section for the full analysis. A small-to-moderate body (bigger than doji's
    very tight max_body_pct, smaller than a trending/plain candle) with roughly BALANCED shadows
    on both sides is a real, conventionally-named pattern (indecision/consolidation, same
    family-spirit as doji but less extreme) that was falling into "no pattern" purely because
    hammer_family specifically requires one DOMINANT long shadow (min_long_shadow_ratio) and the
    OTHER shadow to be short (max_short_shadow_pct) - a spinning top fails that asymmetry test by
    construction, not because it isn't a real shape. Distinguishing test: body_pct in
    (doji's max_body_pct, max_body_pct] AND neither shadow is at least `min_long_shadow_ratio`
    times the other (reuses hammer_family's own ratio knob so the two patterns partition the
    small-body space without a gap or overlap between them)."""
    c = cfg["spinning_top"]
    hf = cfg["hammer_family"]
    s = _shape(bars, i)
    if s["range"] <= 0 or not (cfg["doji"]["max_body_pct"] < s["body_pct"] <= c["max_body_pct"]):
        return None
    upper, lower = s["upper_shadow"], s["lower_shadow"]
    lo, hi = min(upper, lower), max(upper, lower)
    if hi <= 0 or lo <= 0:
        return None  # one-sided shadow with the other truly zero isn't "balanced" - belongs to
                     # hammer_family's own asymmetric test instead, not this pattern
    if hi / lo >= hf["min_long_shadow_ratio"]:
        return None  # asymmetric enough to be hammer_family's territory, not this one
    return PatternMatch("spinning_top", "neutral", c["weight"], 1)


def _hammer_family(bars: pd.DataFrame, i: int, cfg: dict) -> PatternMatch | None:
    """Hammer, Hanging Man, Shooting Star, and Inverted Hammer are all the SAME shape (small
    body, one long shadow, one short shadow) - only the preceding trend decides which of the
    four names (and which direction) applies. One detector, four possible labels."""
    c = cfg["hammer_family"]
    s = _shape(bars, i)
    if s["range"] <= 0 or s["body_pct"] > c["max_body_pct"]:
        return None
    max_short = c["max_short_shadow_pct"] / 100.0 * s["range"]
    trend = _prior_trend(bars, i, c["trend_lookback"])

    hammer_shape = s["lower_shadow"] >= c["min_long_shadow_ratio"] * max(s["body"], 1e-9) and s["upper_shadow"] <= max_short
    star_shape = s["upper_shadow"] >= c["min_long_shadow_ratio"] * max(s["body"], 1e-9) and s["lower_shadow"] <= max_short
    if hammer_shape and trend == "down":
        return PatternMatch("hammer", "bullish", c["weight"], 1)
    if hammer_shape and trend == "up":
        return PatternMatch("hanging_man", "bearish", c["weight"], 1)
    if star_shape and trend == "up":
        return PatternMatch("shooting_star", "bearish", c["weight"], 1)
    if star_shape and trend == "down":
        return PatternMatch("inverted_hammer", "bullish", c["weight"], 1)
    return None


register_pattern("hammer_family")(_hammer_family)


# ---- two-candle -----------------------------------------------------------------------------

@register_pattern("engulfing")
def _engulfing(bars: pd.DataFrame, i: int, cfg: dict) -> PatternMatch | None:
    if i < 1:
        return None
    c = cfg["engulfing"]
    p, s = _shape(bars, i - 1), _shape(bars, i)
    if p["bearish"] and s["bullish"] and s["o"] <= p["c"] and s["c"] >= p["o"]:
        return PatternMatch("bullish_engulfing", "bullish", c["weight"], 2)
    if p["bullish"] and s["bearish"] and s["o"] >= p["c"] and s["c"] <= p["o"]:
        return PatternMatch("bearish_engulfing", "bearish", c["weight"], 2)
    return None


@register_pattern("harami")
def _harami(bars: pd.DataFrame, i: int, cfg: dict) -> PatternMatch | None:
    """The mirror of engulfing: the SECOND candle's body sits inside the first's. Classed by
    the first (larger) candle's color only, per common practical convention (e.g. TA-Lib) -
    not gated on a separate trend check the way the hammer family is."""
    if i < 1:
        return None
    c = cfg["harami"]
    p, s = _shape(bars, i - 1), _shape(bars, i)
    if p["body"] <= 0 or p["body_pct"] < c["min_first_body_pct"]:
        return None
    contained = min(s["o"], s["c"]) >= min(p["o"], p["c"]) and max(s["o"], s["c"]) <= max(p["o"], p["c"])
    if not contained or s["body"] > c["max_second_body_ratio"] / 100.0 * p["body"]:
        return None
    return PatternMatch("bullish_harami" if p["bearish"] else "bearish_harami",
                        "bullish" if p["bearish"] else "bearish", c["weight"], 2)


@register_pattern("piercing_dark_cloud")
def _piercing_dark_cloud(bars: pd.DataFrame, i: int, cfg: dict) -> PatternMatch | None:
    if i < 1:
        return None
    c = cfg["piercing_dark_cloud"]
    p, s = _shape(bars, i - 1), _shape(bars, i)
    if p["body"] <= 0 or p["body_pct"] < c["min_first_body_pct"]:
        return None
    mid = (p["o"] + p["c"]) / 2.0
    gap_ok_bull = s["o"] < p["l"] if c["require_gap"] else s["o"] < p["c"]
    gap_ok_bear = s["o"] > p["h"] if c["require_gap"] else s["o"] > p["c"]
    if p["bearish"] and s["bullish"] and gap_ok_bull and p["c"] < s["c"] < p["o"] and s["c"] > mid:
        return PatternMatch("piercing_line", "bullish", c["weight"], 2)
    if p["bullish"] and s["bearish"] and gap_ok_bear and p["o"] < s["c"] < p["c"] and s["c"] < mid:
        return PatternMatch("dark_cloud_cover", "bearish", c["weight"], 2)
    return None


# ---- three-candle ---------------------------------------------------------------------------

@register_pattern("star")
def _star(bars: pd.DataFrame, i: int, cfg: dict) -> PatternMatch | None:
    if i < 2:
        return None
    c = cfg["star"]
    p1, p2, p3 = _shape(bars, i - 2), _shape(bars, i - 1), _shape(bars, i)
    if p2["body"] <= 0:
        return None
    star_small = p2["body_pct"] <= c["max_star_body_pct"]
    mid1 = (p1["o"] + p1["c"]) / 2.0

    if (p1["bearish"] and p1["body_pct"] >= c["min_first_body_pct"] and star_small
            and (max(p2["o"], p2["c"]) < p1["c"] or not c["require_gap"])
            and p3["bullish"] and p3["c"] > mid1):
        return PatternMatch("morning_star", "bullish", c["weight"], 3)
    if (p1["bullish"] and p1["body_pct"] >= c["min_first_body_pct"] and star_small
            and (min(p2["o"], p2["c"]) > p1["c"] or not c["require_gap"])
            and p3["bearish"] and p3["c"] < mid1):
        return PatternMatch("evening_star", "bearish", c["weight"], 3)
    return None


def detect_patterns(bars: pd.DataFrame, i: int, cfg: dict) -> list[PatternMatch]:
    """Runs every registered detector against the candle(s) ending at index i (i is 'today';
    i-1, i-2 are the preceding candles for multi-candle patterns). Several patterns can match
    at once - e.g. a strong bullish engulfing candle can ALSO be a marubozu - so this returns
    every match, not just the first, letting the caller combine or pick the strongest."""
    return [m for fn in PATTERN_REGISTRY.values() if (m := fn(bars, i, cfg)) is not None]


def net_sentiment(matches: list[PatternMatch]) -> float:
    """Sum of bullish weights minus bearish weights; doji/neutral patterns contribute 0.
    A simple combined score, not a prediction - two weak matches summing to the same score as
    one strong one are NOT equivalent in reliability, just in this one number."""
    total = 0.0
    for m in matches:
        if m.direction == "bullish":
            total += m.weight
        elif m.direction == "bearish":
            total -= m.weight
    return total


def patterns_for(daily: pd.DataFrame, timeframe: str, cfg: dict, i: int = -1,
                 weekly_rule: str = "W-FRI", monthly_rule: str = "ME") -> list[PatternMatch]:
    """The timeframe-parameterized entry point: 'pass a stock's daily bars + "weekly" and get
    back the candlestick pattern(s) matching its most recent completed weekly candle' (i=-1,
    the default). Pass a different i to check an earlier candle, e.g. the exact bar an entry
    fired on."""
    bars = candles_for_timeframe(daily, timeframe, weekly_rule, monthly_rule)
    idx = i if i >= 0 else len(bars) + i
    if idx < 0 or idx >= len(bars):
        return []
    return detect_patterns(bars, idx, cfg)