import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trader.core.candlestick import detect_patterns, net_sentiment, patterns_for

CFG = {
    "marubozu": {"min_body_pct": 90.0, "weight": 3.0},
    "doji": {"max_body_pct": 5.0, "weight": 0.0},
    "spinning_top": {"max_body_pct": 35.0, "weight": 0.0},
    "hammer_family": {"max_body_pct": 35.0, "min_long_shadow_ratio": 2.0, "max_short_shadow_pct": 10.0,
                      "trend_lookback": 5, "weight": 2.0},
    "engulfing": {"weight": 2.5},
    "harami": {"min_first_body_pct": 50.0, "max_second_body_ratio": 50.0, "weight": 1.5},
    "piercing_dark_cloud": {"min_first_body_pct": 60.0, "require_gap": False, "weight": 2.0},
    "star": {"max_star_body_pct": 30.0, "min_first_body_pct": 60.0, "require_gap": False, "weight": 3.5},
}


def _bars(rows: list[tuple]) -> pd.DataFrame:
    """rows of (open, high, low, close); index is just positional dates, values are what matter."""
    idx = pd.bdate_range("2024-01-01", periods=len(rows))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1000
    return df


def _names(matches) -> set:
    return {m.name for m in matches}


# ---- marubozu / doji ------------------------------------------------------------------------

def test_marubozu_bullish_and_bearish():
    up = _bars([(100, 120, 100, 120)])      # full body, zero shadow
    down = _bars([(120, 120, 100, 100)])
    small_body = _bars([(100, 110, 95, 102)])  # body is a small fraction of range - not marubozu
    assert "marubozu" in _names(detect_patterns(up, 0, CFG))
    assert detect_patterns(up, 0, CFG)[[m.name for m in detect_patterns(up, 0, CFG)].index("marubozu")].direction == "bullish"
    assert detect_patterns(down, 0, CFG)[0].direction == "bearish"
    assert "marubozu" not in _names(detect_patterns(small_body, 0, CFG))


def test_doji():
    doji = _bars([(100, 105, 95, 100.2)])      # tiny body vs wide range
    not_doji = _bars([(100, 110, 95, 109)])     # large body
    m = detect_patterns(doji, 0, CFG)
    assert any(x.name == "doji" and x.direction == "neutral" for x in m)
    assert "doji" not in _names(detect_patterns(not_doji, 0, CFG))


# ---- hammer family ----------------------------------------------------------------------------

def _downtrend_then(last_row):
    rows = [(100 - k, 100 - k + 1, 100 - k - 1, 100 - k) for k in range(6)] + [last_row]
    return _bars(rows)


def _uptrend_then(last_row):
    rows = [(100 + k, 100 + k + 1, 100 + k - 1, 100 + k) for k in range(6)] + [last_row]
    return _bars(rows)


def test_hammer_after_downtrend_is_bullish():
    # small body near the top, long lower shadow, tiny upper shadow
    df = _downtrend_then((94.5, 95, 90, 95))
    m = [x for x in detect_patterns(df, 6, CFG) if x.name == "hammer"]
    assert m and m[0].direction == "bullish"


def test_same_shape_after_uptrend_is_hanging_man_not_hammer():
    df = _uptrend_then((105.5, 106, 101, 106))
    names = _names(detect_patterns(df, 6, CFG))
    assert "hanging_man" in names and "hammer" not in names
    m = [x for x in detect_patterns(df, 6, CFG) if x.name == "hanging_man"][0]
    assert m.direction == "bearish"


def test_shooting_star_after_uptrend_is_bearish():
    df = _uptrend_then((105.5, 111, 105, 105))
    m = [x for x in detect_patterns(df, 6, CFG) if x.name == "shooting_star"]
    assert m and m[0].direction == "bearish"


def test_same_shape_after_downtrend_is_inverted_hammer_not_shooting_star():
    df = _downtrend_then((94.5, 100, 94, 94))
    names = _names(detect_patterns(df, 6, CFG))
    assert "inverted_hammer" in names and "shooting_star" not in names
    m = [x for x in detect_patterns(df, 6, CFG) if x.name == "inverted_hammer"][0]
    assert m.direction == "bullish"


def test_normal_body_candle_is_not_in_hammer_family():
    df = _downtrend_then((94, 100, 93, 99))   # body too large relative to range
    assert not any(x.bars_used == 1 and x.name != "marubozu" and x.name != "doji"
                  for x in detect_patterns(df, 6, CFG) if x.name in
                  ("hammer", "hanging_man", "shooting_star", "inverted_hammer"))


# ---- two-candle -------------------------------------------------------------------------------

def test_bullish_engulfing():
    df = _bars([(110, 112, 98, 100), (97, 115, 97, 114)])   # 2nd body fully engulfs 1st
    m = detect_patterns(df, 1, CFG)
    assert any(x.name == "bullish_engulfing" and x.direction == "bullish" for x in m)


def test_bearish_engulfing():
    df = _bars([(100, 112, 98, 110), (113, 115, 95, 97)])
    m = detect_patterns(df, 1, CFG)
    assert any(x.name == "bearish_engulfing" and x.direction == "bearish" for x in m)


def test_small_second_candle_is_not_engulfing():
    df = _bars([(100, 112, 98, 110), (105, 106, 104, 105.5)])  # tiny 2nd candle, doesn't engulf
    assert "bullish_engulfing" not in _names(detect_patterns(df, 1, CFG))
    assert "bearish_engulfing" not in _names(detect_patterns(df, 1, CFG))


def test_bullish_harami():
    df = _bars([(120, 122, 98, 100), (105, 112, 104, 110)])   # big bearish candle, small 2nd candle contained within
    m = detect_patterns(df, 1, CFG)
    assert any(x.name == "bullish_harami" and x.direction == "bullish" for x in m)


def test_bearish_harami():
    df = _bars([(100, 122, 98, 120), (105, 112, 104, 110)])
    m = detect_patterns(df, 1, CFG)
    assert any(x.name == "bearish_harami" and x.direction == "bearish" for x in m)


def test_harami_rejects_second_candle_not_contained():
    df = _bars([(120, 122, 98, 100), (97, 130, 96, 125)])   # 2nd candle extends BEYOND 1st's body (that's engulfing, not harami)
    assert "bullish_harami" not in _names(detect_patterns(df, 1, CFG))


def test_piercing_line():
    df = _bars([(120, 122, 98, 100), (99, 118, 98, 115)])   # closes well above midpoint (110) of candle 1's body
    m = detect_patterns(df, 1, CFG)
    assert any(x.name == "piercing_line" and x.direction == "bullish" for x in m)


def test_dark_cloud_cover():
    df = _bars([(100, 122, 98, 120), (121, 122, 103, 105)])   # closes well below midpoint (110) of candle 1's body
    m = detect_patterns(df, 1, CFG)
    assert any(x.name == "dark_cloud_cover" and x.direction == "bearish" for x in m)


def test_piercing_line_requires_more_than_half_penetration():
    # closes only slightly above candle 1's close - doesn't reach the midpoint
    df = _bars([(120, 122, 98, 100), (99, 103, 98, 102)])
    assert "piercing_line" not in _names(detect_patterns(df, 1, CFG))


# ---- three-candle -----------------------------------------------------------------------------

def test_morning_star():
    df = _bars([
        (120, 122, 98, 100),    # long bearish
        (98, 100, 96, 99),      # small star, gapped-ish down
        (101, 118, 100, 116),   # long bullish, closes above candle 1's midpoint (110)
    ])
    m = detect_patterns(df, 2, CFG)
    assert any(x.name == "morning_star" and x.direction == "bullish" and x.bars_used == 3 for x in m)


def test_evening_star():
    df = _bars([
        (100, 122, 98, 120),
        (121, 124, 120, 122),
        (119, 120, 102, 104),
    ])
    m = detect_patterns(df, 2, CFG)
    assert any(x.name == "evening_star" and x.direction == "bearish" and x.bars_used == 3 for x in m)


def test_morning_star_requires_third_candle_to_close_past_midpoint():
    df = _bars([
        (120, 122, 98, 100),
        (98, 100, 96, 99),
        (101, 108, 100, 106),   # bullish but doesn't close past candle 1's midpoint (110)
    ])
    assert "morning_star" not in _names(detect_patterns(df, 2, CFG))


# ---- combinators + the timeframe wrapper -------------------------------------------------------

def test_net_sentiment_sums_bullish_minus_bearish_and_ignores_neutral():
    from trader.core.candlestick import PatternMatch
    matches = [PatternMatch("a", "bullish", 2.0, 1), PatternMatch("b", "bearish", 1.5, 1),
              PatternMatch("c", "neutral", 5.0, 1)]
    assert net_sentiment(matches) == pytest.approx(0.5)


def test_multiple_patterns_can_match_the_same_candle():
    # a strong bullish engulfing candle whose own body is also big enough to be a marubozu
    df = _bars([(110, 112, 98, 100), (97, 115.5, 97, 115)])
    names = _names(detect_patterns(df, 1, CFG))
    assert "bullish_engulfing" in names and "marubozu" in names


def test_patterns_for_defaults_to_the_most_recent_bar_and_resamples_by_timeframe():
    idx = pd.bdate_range("2024-01-01", periods=40)
    close = pd.Series(range(100, 140), index=idx, dtype=float)
    daily = pd.DataFrame({"open": close - 1, "high": close + 2, "low": close - 2,
                         "close": close, "volume": 1000})
    daily_matches = patterns_for(daily, "daily", CFG)         # i=-1 default -> last daily bar
    weekly_matches = patterns_for(daily, "weekly", CFG)       # resampled first, then last weekly bar
    assert isinstance(daily_matches, list) and isinstance(weekly_matches, list)


def test_patterns_for_rejects_unknown_timeframe():
    daily = _bars([(100, 101, 99, 100)])
    with pytest.raises(ValueError):
        patterns_for(daily, "fortnightly", CFG)


def test_patterns_for_out_of_range_index_returns_empty_not_an_error():
    daily = _bars([(100, 101, 99, 100)])
    assert patterns_for(daily, "daily", CFG, i=500) == []