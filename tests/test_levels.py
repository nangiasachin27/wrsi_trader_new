import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trader.core.levels import Level, nearest_levels, pivot_highs, pivot_lows, support_resistance, support_resistance_zones

CFG = {"pivot_left": 2, "pivot_right": 2, "cluster_pct": 1.0, "min_touches": 2, "lookback_bars": None}


def _zigzag(points: list[float], bars_between: int = 10) -> pd.DataFrame:
    """Build a clean OHLC series that walks linearly through `points` - gives known, exact
    swing highs/lows at the turning points for deterministic assertions."""
    closes = []
    for a, b in zip(points, points[1:]):
        closes.extend(np.linspace(a, b, bars_between, endpoint=False))
    closes.append(points[-1])
    closes = np.array(closes, dtype=float)  # plain array, not a Series - avoids label-based
    idx = pd.bdate_range("2020-01-01", periods=len(closes))  # alignment against the index= below
    return pd.DataFrame({"open": closes, "high": closes + 0.2, "low": closes - 0.2,
                         "close": closes, "volume": 1000}, index=idx)


def test_pivot_highs_and_lows_find_known_turning_points():
    df = _zigzag([100, 120, 90, 130, 95])
    ph = pivot_highs(df["high"], left=2, right=2)
    pl = pivot_lows(df["low"], left=2, right=2)
    # the two interior peaks (120, 130) and the two interior troughs (90, 95's predecessor) should be flagged
    peak_prices = df.loc[ph, "high"].round(0).tolist()
    trough_prices = df.loc[pl, "low"].round(0).tolist()
    assert any(abs(p - 120.2) < 1 for p in peak_prices)
    assert any(abs(p - 130.2) < 1 for p in peak_prices)
    assert any(abs(p - 89.8) < 1 for p in trough_prices)


def test_last_right_bars_are_never_confirmed_pivots():
    df = _zigzag([100, 150, 100])
    ph = pivot_highs(df["high"], left=3, right=3)
    assert not ph.iloc[-3:].any()   # can't confirm a pivot without `right` bars of future data


def test_touches_near_each_other_cluster_into_one_zone_with_combined_strength():
    # two separate touches near price 150, one touch far away near 90 - should yield 2 zones,
    # the 150-ish one with strength 2, not two separate 1-touch zones
    df = _zigzag([100, 150, 110, 151, 90, 140])
    zones = support_resistance_zones(df, CFG)
    near_150 = [z for z in zones if 148 <= z.mid <= 153]
    assert len(near_150) == 1
    assert near_150[0].strength >= 2


def test_single_untested_touch_is_not_reported_as_a_zone():
    cfg = {**CFG, "min_touches": 3}
    df = _zigzag([100, 150, 100, 160, 100])   # two different, non-clustering peaks - neither reaches 3 touches
    zones = support_resistance_zones(df, cfg)
    assert all(z.strength >= 3 for z in zones)
    assert len(zones) == 0


def test_nearest_levels_picks_support_below_and_resistance_above_current_price():
    zones = [Level(95, 96, 3), Level(104, 105, 2), Level(150, 151, 4), Level(40, 41, 2)]
    picked = nearest_levels(zones, current_price=100)
    assert picked["support"].mid == pytest_approx(95.5)
    assert picked["resistance"].mid == pytest_approx(104.5)


def pytest_approx(x):
    import pytest
    return pytest.approx(x, abs=0.01)


def test_nearest_levels_returns_none_when_nothing_on_one_side():
    zones = [Level(150, 151, 3)]
    picked = nearest_levels(zones, current_price=100)
    assert picked["support"] is None
    assert picked["resistance"].mid == pytest_approx(150.5)


def test_support_resistance_timeframe_wrapper_matches_manual_resample():
    from trader.core.indicators import resample_ohlc
    df = _zigzag([100, 150, 90, 160, 80, 170], bars_between=25)
    cfg = {"pivot_left": 2, "pivot_right": 2, "cluster_pct": 2.0, "min_touches": 1, "lookback_bars": None}
    via_wrapper = support_resistance(df, "weekly", cfg)
    via_manual = support_resistance_zones(resample_ohlc(df, "W-FRI"), cfg)
    assert [z.mid for z in via_wrapper] == [z.mid for z in via_manual]
    assert len(via_wrapper) > 0


def test_support_resistance_weekly_uses_far_fewer_bars_than_daily():
    from trader.core.indicators import candles_for_timeframe
    df = _zigzag([100, 150, 90, 160, 80, 170], bars_between=25)
    assert len(candles_for_timeframe(df, "weekly")) < len(df) / 3   # ~5 trading days per week


def test_support_resistance_rejects_unknown_timeframe():
    import pytest
    df = _zigzag([100, 110])
    with pytest.raises(ValueError):
        support_resistance(df, "fortnightly", CFG)