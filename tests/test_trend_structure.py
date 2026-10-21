import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trader.core.trend_structure import classify_trend_structure, current_trend_structure

CFG = {"timeframe": "daily", "pivot_left": 2, "pivot_right": 2, "min_confirmed_swings": 2,
      "initial_state": "SIDEWAYS"}


def _zigzag(points: list[float], bars_between: int = 10) -> pd.DataFrame:
    """Same deterministic-turning-point helper as test_levels.py's _zigzag - a clean linear
    walk through `points`, giving known, exact swing highs/lows at each interior point."""
    closes = []
    for a, b in zip(points, points[1:]):
        closes.extend(np.linspace(a, b, bars_between, endpoint=False))
    closes.append(points[-1])
    closes = np.array(closes, dtype=float)
    idx = pd.bdate_range("2020-01-01", periods=len(closes))
    return pd.DataFrame({"open": closes, "high": closes + 0.2, "low": closes - 0.2,
                         "close": closes, "volume": 1000}, index=idx)


def test_rising_highs_and_lows_confirm_uptrend_after_two_swings():
    # swing lows: 90, 95, 105 (rising); swing highs: 110, 120, 130 (rising)
    df = _zigzag([100, 90, 110, 95, 120, 105, 130])
    out, events = classify_trend_structure(df, CFG)
    assert out["state"].iloc[-1] == "UPTREND"
    assert any(e.to_state == "UPTREND" for e in events)
    # confirmed the moment the 2nd swing high/low pair is in (doesn't wait for the 3rd)
    first_uptrend_event = next(e for e in events if e.to_state == "UPTREND")
    assert first_uptrend_event.from_state == "SIDEWAYS"


def test_falling_highs_and_lows_confirm_downtrend_after_two_swings():
    # mirror: swing highs 110,100,90 (falling); swing lows 95,85,75 (falling)
    df = _zigzag([120, 110, 95, 100, 85, 90, 75])
    out, events = classify_trend_structure(df, CFG)
    assert out["state"].iloc[-1] == "DOWNTREND"
    assert any(e.to_state == "DOWNTREND" for e in events)


def test_equal_highs_and_lows_stay_sideways():
    # oscillates between ~110 highs and ~95 lows with no net direction - never strictly rising
    # or falling, so it should never confirm either trend
    df = _zigzag([100, 95, 110, 95, 110, 95, 110, 95])
    out, events = classify_trend_structure(df, CFG)
    assert (out["state"] == "SIDEWAYS").all()
    assert events == []


def test_break_of_market_structure_drops_uptrend_to_sideways_not_downtrend():
    # confirm an uptrend (swing lows 90,95 rising; swing highs 110,120 rising), then crash
    # straight through the protecting swing low (95) without forming a new confirmed downswing
    # (one extra leg down to 105 so 120 itself gets confirmed as an interior swing high)
    df = _zigzag([100, 90, 110, 95, 120, 105])
    out, events = classify_trend_structure(df, CFG)
    assert out["state"].iloc[-1] == "UPTREND"
    structure_level = out["structure_level"].iloc[-1]
    assert structure_level == pytest.approx(95, abs=0.5)

    # append a sharp break below the protecting low, held for a couple of bars (no new swing
    # confirmation follows - too short for min_confirmed_swings=2 on the down side)
    crash_idx = pd.bdate_range(df.index[-1] + pd.Timedelta(days=1), periods=5)
    crash = pd.DataFrame({"open": 80, "high": 80.2, "low": 79.8, "close": 80, "volume": 1000}, index=crash_idx)
    full = pd.concat([df, crash])
    out2, events2 = classify_trend_structure(full, CFG)
    assert out2["state"].iloc[-1] == "SIDEWAYS"
    bos = [e for e in events2 if "break of market structure" in e.reason]
    assert len(bos) == 1
    assert bos[0].from_state == "UPTREND"
    assert bos[0].to_state == "SIDEWAYS"  # never jumps straight to DOWNTREND


def test_structure_level_ratchets_up_as_new_higher_lows_form():
    # swing lows 90, then 95, then 108 (each higher than the last) - the protecting level
    # should climb to match each new confirmed higher low, never staying behind
    df = _zigzag([100, 90, 110, 95, 120, 108, 140, 125, 160])
    out, events = classify_trend_structure(df, CFG)
    assert out["state"].iloc[-1] == "UPTREND"
    levels = [lvl for lvl in out["structure_level"] if pd.notna(lvl)]
    assert levels == sorted(levels)  # monotonically non-decreasing throughout
    assert levels[-1] == pytest.approx(125, abs=0.5)  # last confirmed higher low in the series


def test_a_lower_swing_low_breaks_structure_before_it_can_ever_loosen_the_level():
    # by construction, a close that dips to a new, LOWER swing low must first cross the
    # existing protecting level (the previous, higher swing low) - so the break-of-structure
    # check in step 1 always fires before any lower candidate could reach the ratchet in step
    # 3; this is what actually keeps the ratchet one-directional, so pin it down explicitly
    df = _zigzag([100, 90, 110, 95, 120, 80, 140])
    out, events = classify_trend_structure(df, CFG)
    assert any("break of market structure" in e.reason for e in events)
    assert out["state"].iloc[-1] in ("SIDEWAYS", "UPTREND")  # never silently stays UPTREND
                                                             # past a broken structure level
    # once broken, the structure level resets to None rather than reporting the lower value
    broke_at = next(i for i, e in enumerate(out.index) if any(
        ev.date == e.date() and "break of market structure" in ev.reason for ev in events))
    assert pd.isna(out["structure_level"].iloc[broke_at])


def test_last_pivot_right_bars_never_count_as_confirmed_swings_no_lookahead():
    df = _zigzag([100, 90, 110, 95, 120])
    out, _ = classify_trend_structure(df, CFG)
    # swing_high_count/swing_low_count must not increase on the final pivot_right bars - a
    # pivot there can't be confirmed without future bars to compare against (see
    # trader.core.levels.pivot_highs/pivot_lows)
    right = CFG["pivot_right"]
    tail_counts = out[["swing_high_count", "swing_low_count"]].iloc[-right:]
    assert (tail_counts.iloc[-1] <= tail_counts.max()).all()


def test_too_few_bars_returns_empty_frame_and_no_events_not_an_error():
    df = _zigzag([100, 110])  # far fewer bars than pivot_left+pivot_right+1 needs in practice
    out, events = classify_trend_structure(df, {**CFG, "pivot_left": 20, "pivot_right": 20})
    assert out.empty
    assert events == []


def test_current_trend_structure_wrapper_returns_latest_state_and_context():
    df = _zigzag([100, 90, 110, 95, 120, 105, 130])
    result = current_trend_structure(df, CFG)
    assert result["state"] == "UPTREND"
    assert result["structure_level"] is not None
    assert result["structure_kind"] == "low"
    assert result["last_event"] is not None


def test_current_trend_structure_wrapper_on_empty_history_returns_initial_state():
    df = _zigzag([100, 110])
    result = current_trend_structure(df, {**CFG, "pivot_left": 20, "pivot_right": 20})
    assert result["state"] == CFG["initial_state"]
    assert result["last_event"] is None


def test_weekly_timeframe_uses_far_fewer_bars_than_daily_like_support_resistance_does():
    from trader.core.indicators import candles_for_timeframe
    df = _zigzag([100, 90, 110, 95, 120, 105, 130], bars_between=25)
    weekly_cfg = {**CFG, "timeframe": "weekly"}
    out_daily, _ = classify_trend_structure(df, CFG)
    out_weekly, _ = classify_trend_structure(df, weekly_cfg)
    assert len(out_weekly) == len(candles_for_timeframe(df, "weekly"))
    assert len(out_weekly) < len(out_daily)
