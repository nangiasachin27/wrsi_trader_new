from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass
class TrendEvent:
    date: date
    from_state: str
    to_state: str
    reason: str


def aggregate_universe_volume(cache_dir, universe: list[str]) -> pd.Series:
    """Sum of daily volume across `universe`'s cached CSVs, aligned by date - a market-breadth
    volume proxy for an index like NIFTY500 whose own 'volume' field is always zero (an index
    isn't itself traded, so Upstox reports no volume for it). Missing or unreadable symbols are
    skipped rather than failing the whole aggregate; a symbol's volume for a date it has no row
    for simply contributes nothing that day."""
    totals: dict = {}
    for symbol in universe:
        f = cache_dir / f"{symbol}.csv"
        if not f.exists():
            continue
        try:
            d = pd.read_csv(f, parse_dates=["date"], usecols=["date", "volume"])
        except Exception:  # noqa: BLE001 - one bad file shouldn't break the aggregate
            continue
        for dt, vol in zip(d["date"], d["volume"]):
            totals[dt] = totals.get(dt, 0.0) + float(vol)
    return pd.Series(totals).sort_index()


def classify_market_trend(daily: pd.DataFrame, cfg: dict, volume: pd.Series | None = None
                          ) -> tuple[pd.DataFrame, list[TrendEvent]]:
    """Day-by-day general-market-trend state machine over UPTREND / NEUTRAL / DOWNTREND, run
    against a single benchmark's daily bars (NIFTY500 by default - same index already used
    elsewhere in this project for rs_rank/sector_regime) for PRICE moves.

    `volume`, if given, is a date-indexed Series used for the volume side of every VD-day/FTD
    check INSTEAD OF daily['volume'] - needed because an index's own volume field is always
    zero (it isn't itself traded). Pass aggregate_universe_volume(...)'s output here. If
    omitted, daily['volume'] is used as-is (useful for testing against an individual stock,
    which does have real volume).

    DOWN-MOVE (UPTREND -> NEUTRAL -> DOWNTREND): counts 'volume distribution days' - close down
    more than `fall_pct` vs the previous close, AND volume higher than the previous day's - in
    a trailing `vd_window_calendar_days`-day window (calendar days, not trading days). Reaching
    `neutral_vd_count` such days in that window demotes UPTREND -> NEUTRAL. Reaching
    `downtrend_vd_count` demotes straight to DOWNTREND, from either UPTREND or NEUTRAL.

    UP-MOVE (DOWNTREND -> NEUTRAL -> UPTREND), only searched while the state is DOWNTREND:
    watches for a 'Day 1' rally - close up more than `rally_pct` vs the previous close - and
    records that day's LOW. If Day 1's low survives `breach_check_trading_days` TRADING days
    without any day's low falling below it, the market moves to NEUTRAL. From there, within
    `ftd_search_trading_days` TRADING days, a 'follow-through day' (close up more than
    `ftd_pct` vs previous close, AND volume higher than the previous day's) confirms UPTREND.

    At ANY point while a Day 1 is being tracked - during the breach-check window OR the later
    FTD search - a day's low falling below Day 1's low fails the attempt: state reverts to
    DOWNTREND and the search restarts from scratch, looking for a brand new Day 1. This also
    covers the FTD-search window simply running out with no FTD day and no breach, which the
    source rules didn't specify an outcome for - reverting-and-restarting (rather than waiting
    in NEUTRAL indefinitely with a stale, unconfirmed rally) was the most internally consistent
    choice available; flag it if a different behavior is wanted.

    ASYMMETRY IN THE RULES AS GIVEN: UPTREND -> NEUTRAL via VD days is a ONE-WAY gate as
    specified - that NEUTRAL only ever continues on to DOWNTREND (more VD days) or persists;
    nothing in the source rules lets it recover directly back to UPTREND, because the recovery
    mechanism (Day 1 / breach-check / FTD) is only specified to run "when market is in
    downtrend." So as written, a mere correction must deepen all the way into DOWNTREND before
    the recovery mechanism can ever fire - implemented literally; say if that's not intended.

    Returns (per-day DataFrame indexed by date, list of TrendEvent transitions).
    """
    fall_pct = cfg.get("fall_pct", 0.25) / 100.0
    rally_pct = cfg.get("rally_pct", 1.25) / 100.0
    ftd_pct = cfg.get("ftd_pct", 1.0) / 100.0
    vd_window_days = cfg.get("vd_window_calendar_days", 30)
    neutral_vd_count = cfg.get("neutral_vd_count", 4)
    downtrend_vd_count = cfg.get("downtrend_vd_count", 8)
    breach_check_days = cfg.get("breach_check_trading_days", 3)
    ftd_search_days = cfg.get("ftd_search_trading_days", 7)
    state = cfg.get("initial_state", "NEUTRAL")

    daily = daily.sort_index()
    idx = daily.index

    vd_dates: list = []
    day1_date, day1_low, day1_idx, neutral_since_idx = None, None, None, None

    rows: list[dict] = []
    events: list[TrendEvent] = []

    def reset_day1() -> None:
        nonlocal day1_date, day1_low, day1_idx, neutral_since_idx
        day1_date, day1_low, day1_idx, neutral_since_idx = None, None, None, None

    for i in range(len(daily)):
        today = idx[i]
        close = float(daily["close"].iloc[i])
        low = float(daily["low"].iloc[i])
        prev_close = float(daily["close"].iloc[i - 1]) if i > 0 else None
        if volume is not None:
            vol = float(volume.get(today, 0.0))
            prev_vol = float(volume.get(idx[i - 1], 0.0)) if i > 0 else None
        else:
            vol = float(daily["volume"].iloc[i])
            prev_vol = float(daily["volume"].iloc[i - 1]) if i > 0 else None

        is_vd_day = False
        if prev_close:
            fall = (prev_close - close) / prev_close
            is_vd_day = fall > fall_pct and vol > prev_vol
        if is_vd_day:
            vd_dates.append(today)
        cutoff = today - pd.Timedelta(days=vd_window_days)
        vd_dates = [d for d in vd_dates if d > cutoff]
        vd_count_30d = len(vd_dates)

        # ---- down-move: VD-day counting -------------------------------------------------
        if state == "UPTREND":
            if vd_count_30d >= downtrend_vd_count:
                events.append(TrendEvent(today.date(), state, "DOWNTREND",
                                         f"{vd_count_30d} VD days in trailing {vd_window_days}d"))
                state = "DOWNTREND"
                reset_day1()
            elif vd_count_30d >= neutral_vd_count:
                events.append(TrendEvent(today.date(), state, "NEUTRAL",
                                         f"{vd_count_30d} VD days in trailing {vd_window_days}d"))
                state = "NEUTRAL"
        elif state == "NEUTRAL" and day1_date is None:
            # a correction-NEUTRAL (from mechanism A) can still deepen into DOWNTREND
            if vd_count_30d >= downtrend_vd_count:
                events.append(TrendEvent(today.date(), state, "DOWNTREND",
                                         f"{vd_count_30d} VD days in trailing {vd_window_days}d"))
                state = "DOWNTREND"

        # ---- up-move: Day1 / breach-check / FTD -------------------------------------------
        if state == "DOWNTREND":
            if day1_date is None:
                if prev_close:
                    rally = (close - prev_close) / prev_close
                    if rally > rally_pct:
                        day1_date, day1_low, day1_idx = today, low, i
            else:
                if low < day1_low:
                    events.append(TrendEvent(today.date(), state, "DOWNTREND",
                                             f"Day1 low {day1_low:.2f} (from {day1_date.date()}) breached - restarting search"))
                    reset_day1()
                elif i - day1_idx >= breach_check_days:
                    neutral_since_idx = i
                    events.append(TrendEvent(today.date(), state, "NEUTRAL",
                                             f"Day1 low held for {breach_check_days} trading days since {day1_date.date()}"))
                    state = "NEUTRAL"
        elif state == "NEUTRAL" and day1_date is not None:
            if low < day1_low:
                events.append(TrendEvent(today.date(), state, "DOWNTREND",
                                         f"Day1 low {day1_low:.2f} (from {day1_date.date()}) breached during FTD search - restarting"))
                state = "DOWNTREND"
                reset_day1()
            elif prev_close and (close - prev_close) / prev_close > ftd_pct and vol > prev_vol:
                gain = (close - prev_close) / prev_close
                events.append(TrendEvent(today.date(), state, "UPTREND",
                                         f"follow-through day: +{gain*100:.2f}% on rising volume, "
                                         f"{i - neutral_since_idx} trading days after NEUTRAL"))
                state = "UPTREND"
                reset_day1()
            elif i - neutral_since_idx >= ftd_search_days:
                events.append(TrendEvent(today.date(), state, "DOWNTREND",
                                         f"no follow-through day within {ftd_search_days} trading days of NEUTRAL - restarting"))
                state = "DOWNTREND"
                reset_day1()

        rows.append({
            "date": today, "state": state, "vd_count_30d": vd_count_30d, "is_vd_day": is_vd_day,
            "day1_date": day1_date, "day1_low": day1_low,
        })

    out = pd.DataFrame(rows).set_index("date")
    return out, events


def current_trend(daily: pd.DataFrame, cfg: dict, volume: pd.Series | None = None) -> dict:
    """Convenience wrapper: run the full classification and return just the latest day's
    state plus enough context to explain it, for a filter or CLI to use without re-deriving
    the whole history itself."""
    out, events = classify_market_trend(daily, cfg, volume)
    if out.empty:
        return {"state": cfg.get("initial_state", "NEUTRAL"), "vd_count_30d": 0,
                "day1_date": None, "day1_low": None, "last_event": None}
    last = out.iloc[-1]
    return {
        "state": last["state"],
        "vd_count_30d": int(last["vd_count_30d"]),
        "day1_date": last["day1_date"],
        "day1_low": last["day1_low"],
        "last_event": events[-1] if events else None,
    }