from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from trader.core.indicators import candles_for_timeframe
from trader.core.levels import pivot_highs, pivot_lows


@dataclass
class StructureEvent:
    date: date
    from_state: str
    to_state: str
    reason: str


def classify_trend_structure(daily: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list]:
    """Market trend EXACTLY as the price-action course teaches it, not the IBD-style VD-day/
    follow-through-day mechanism in trader.core.market_trend.py (which this module does not
    touch or depend on - that one stays available for comparison, this is a separate engine).

    Course rules (section 1, "Market Trend" - see the course-notes doc):
      - An uptrend is higher highs + higher lows, confirmed only after 2+ up-swings
        (`min_confirmed_swings`, default 2). A downtrend is the mirror: lower highs + lower
        lows.
      - Anything else - including a range, and a confirmed trend whose newest swing doesn't
        (yet) extend it - is SIDEWAYS. SIDEWAYS is the course's own word for this state, not
        a stand-in for anything else.
      - "The earliest warning of a trend change is the first break of the most recent higher
        low (uptrend) or lower high (downtrend) - a break of market structure." That break
        drops the state straight to SIDEWAYS on the spot (not to the opposite trend - the
        course treats losing a trend and confirming the NEW opposite one as two separate
        events, the second needing its own 2+ swings).

    Swing points are the SAME pivot_highs/pivot_lows pivot-detection already used by
    trader.core.levels for support/resistance (reused, not reimplemented) - so "swing high/
    low" means the same thing here as it does to the S/R zone builder: a bar whose high/low is
    the extreme within `pivot_left` bars before and `pivot_right` bars after it. The last
    `pivot_right` bars of any window are therefore never a confirmed swing yet (see
    pivot_highs/pivot_lows docstrings) - that's correct, not a gap, and this function performs
    no lookahead because of it.

    `cfg["timeframe"]` picks which bars the swings are read from ("daily"/"weekly"/"monthly" -
    same convention as trader.core.levels.support_resistance); the course draws S/R and
    structure on a swing-appropriate timeframe (weeks/months for swing trading), and the
    sensible pivot window differs by timeframe the same way it does for S/R, hence this taking
    its own small cfg dict rather than being hard-coded to daily bars.

    The protecting level ("structure_level": the swing low/high whose break would end the
    current trend) only ever RATCHETS while a trend holds - a new, less-favorable swing never
    loosens it - the same tighten-only discipline every trailing stop in this codebase already
    follows (see trader.strategies._swing_common.ratchet). It resets to None on every SIDEWAYS
    bar and is set fresh the moment a trend is (re)confirmed.

    Returns (per-bar DataFrame indexed by date with columns state/structure_level/
    structure_kind/swing_high_count/swing_low_count, list of StructureEvent transitions) - the
    same (DataFrame, events-list) shape trader.core.market_trend.classify_market_trend returns,
    so callers (e.g. a regime filter) can treat the two engines interchangeably.
    """
    timeframe = cfg.get("timeframe", "daily")
    if timeframe == "daily":
        bars = daily
    else:
        bars = candles_for_timeframe(daily, timeframe, cfg.get("weekly_rule", "W-FRI"), cfg.get("monthly_rule", "ME"))

    left = cfg.get("pivot_left", 2)
    right = cfg.get("pivot_right", 2)
    min_swings = cfg.get("min_confirmed_swings", 2)
    state = cfg.get("initial_state", "SIDEWAYS")

    if len(bars) < left + right + 1:
        return pd.DataFrame(columns=["state", "structure_level", "structure_kind",
                                     "swing_high_count", "swing_low_count"]), []

    ph = pivot_highs(bars["high"], left, right)
    pl = pivot_lows(bars["low"], left, right)

    swing_highs: list[tuple[int, pd.Timestamp, float]] = []
    swing_lows: list[tuple[int, pd.Timestamp, float]] = []
    structure_level: float | None = None
    structure_kind: str | None = None

    events: list[StructureEvent] = []
    rows: list[dict] = []
    idx = bars.index

    for i in range(len(bars)):
        today = idx[i]
        close = float(bars["close"].iloc[i])

        new_high_pivot = bool(ph.iloc[i])
        new_low_pivot = bool(pl.iloc[i])
        if new_high_pivot:
            swing_highs.append((i, today, float(bars["high"].iloc[i])))
        if new_low_pivot:
            swing_lows.append((i, today, float(bars["low"].iloc[i])))

        # 1) Break of market structure - losing a confirmed trend takes priority over any other
        # check this bar, including re-confirming a direction from the newest swings below.
        if state == "UPTREND" and structure_level is not None and close < structure_level:
            events.append(StructureEvent(today.date(), state, "SIDEWAYS",
                f"close {close:.2f} broke the protecting swing low {structure_level:.2f} - "
                f"break of market structure"))
            state = "SIDEWAYS"
            structure_level, structure_kind = None, None
        elif state == "DOWNTREND" and structure_level is not None and close > structure_level:
            events.append(StructureEvent(today.date(), state, "SIDEWAYS",
                f"close {close:.2f} broke the protecting swing high {structure_level:.2f} - "
                f"break of market structure"))
            state = "SIDEWAYS"
            structure_level, structure_kind = None, None

        # 2) (Re)confirm a direction from the most recent `min_swings` swings on each side -
        # only when THIS bar actually added a new confirmed swing. Without this guard, a bar
        # that just broke an uptrend (step 1, above) would immediately re-evaluate the SAME,
        # now-stale swing list that originally confirmed it and flip straight back to
        # UPTREND on the very same bar - an infinite same-bar oscillation. A trend's
        # confirmation state only has reason to change when the swing structure itself changes.
        if (new_high_pivot or new_low_pivot) and len(swing_highs) >= min_swings and len(swing_lows) >= min_swings:
            last_highs = [p for _, _, p in swing_highs[-min_swings:]]
            last_lows = [p for _, _, p in swing_lows[-min_swings:]]
            rising = (all(b > a for a, b in zip(last_highs, last_highs[1:])) and
                     all(b > a for a, b in zip(last_lows, last_lows[1:])))
            falling = (all(b < a for a, b in zip(last_highs, last_highs[1:])) and
                      all(b < a for a, b in zip(last_lows, last_lows[1:])))
            if rising and state != "UPTREND":
                events.append(StructureEvent(today.date(), state, "UPTREND",
                    f"{min_swings} consecutive higher highs + higher lows"))
                state = "UPTREND"
                structure_level = None
            elif falling and state != "DOWNTREND":
                events.append(StructureEvent(today.date(), state, "DOWNTREND",
                    f"{min_swings} consecutive lower highs + lower lows"))
                state = "DOWNTREND"
                structure_level = None

        # 3) Ratchet the protecting level while a trend holds - never loosens.
        if state == "UPTREND" and swing_lows:
            candidate = swing_lows[-1][2]
            if structure_level is None or candidate > structure_level:
                structure_level = candidate
            structure_kind = "low"
        elif state == "DOWNTREND" and swing_highs:
            candidate = swing_highs[-1][2]
            if structure_level is None or candidate < structure_level:
                structure_level = candidate
            structure_kind = "high"
        else:
            structure_kind = None

        rows.append({"date": today, "state": state,
                     "structure_level": structure_level, "structure_kind": structure_kind,
                     "swing_high_count": len(swing_highs), "swing_low_count": len(swing_lows)})

    out = pd.DataFrame(rows).set_index("date")
    return out, events


def current_trend_structure(daily: pd.DataFrame, cfg: dict) -> dict:
    """Convenience wrapper mirroring trader.core.market_trend.current_trend: run the full
    classification and return just the latest bar's state plus enough context to explain it."""
    out, events = classify_trend_structure(daily, cfg)
    if out.empty:
        return {"state": cfg.get("initial_state", "SIDEWAYS"), "structure_level": None,
                "structure_kind": None, "last_event": None}
    last = out.iloc[-1]
    return {
        "state": last["state"],
        "structure_level": last["structure_level"],
        "structure_kind": last["structure_kind"],
        "last_event": events[-1] if events else None,
    }
