"""range_bounce: range qualification, floor-rejection entry, mandatory R:R, exits, and an
end-to-end synthetic backtest. All data synthetic and deterministic."""
from __future__ import annotations

import copy
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from trader.backtest.runner import run_backtest
from trader.core.calendar import TradingCalendar
from trader.core.config import load_settings
from trader.core.fractal_levels import fractal_result, reset_settings_cache
from trader.core.models import Position
from trader.strategies.base import SymbolContext, build_strategy

S = load_settings()


@pytest.fixture(autouse=True)
def _fresh_zone_cache():
    reset_settings_cache()
    yield


def ranged(n=300, lo=100.0, hi=120.0, period=12, seed=11, start="2025-01-01"):
    rng = np.random.default_rng(seed)
    mid, amp = (lo + hi) / 2, (hi - lo) / 2
    c = mid + amp * np.sin(np.arange(n) * 2 * np.pi / period) + rng.normal(0, 0.6, n)
    o = np.r_[c[0], c[:-1]]
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.004,
                         "low": np.minimum(o, c) * 0.996, "close": c, "volume": 1e6}, index=idx)


def ctx(df, sym="X"):
    cal = TradingCalendar(set(), trading_days=[t.date() for t in df.index])
    return SymbolContext(sym, df, df.index[-1].date(), cal)


def strat(**entry_over):
    cfg = copy.deepcopy(S.strategies["range_bounce"])
    cfg["entry"].update(entry_over)
    return build_strategy("range_bounce", cfg)


def with_last_bar(df, o, h, l, c):
    nxt = df.index[-1] + pd.offsets.BDay(1)
    row = pd.DataFrame({"open": [o], "high": [h], "low": [l], "close": [c], "volume": [1e6]}, index=[nxt])
    return pd.concat([df, row])


def floor_touch_df(close_mult=1.012):
    """Range df ending with a bullish rejection bar at the floor zone: low pierces the zone's
    top edge, close back above it (close_mult x zone top)."""
    base = ranged()
    sup = fractal_result(base, "daily", float(base.close.iloc[-1])).sup[-1]  # deepest support zone
    # use the NEAREST support zone of the final frame instead: compute after appending
    r = fractal_result(base, "daily", 110.0)
    floor = min(r.sup, key=lambda z: z.low) if r.sup else sup
    top = floor.high
    return with_last_bar(base, o=top * 1.002, h=top * close_mult * 1.002, l=top * 0.998, c=top * close_mult)


def test_entry_fires_on_floor_rejection_in_qualified_range():
    df = floor_touch_df()
    sig = strat().entry_signal(ctx(df))
    assert sig is not None
    assert sig.stop < float(df.close.iloc[-1])
    assert sig.meta["reward_risk"] >= 1.5
    assert sig.meta["ceiling_low"] > sig.meta["floor_high"]
    assert "touches" in sig.reason


def test_no_entry_without_floor_touch():
    base = ranged()
    # mid-range bullish day, no touch of the floor
    df = with_last_bar(base, o=109.0, h=111.5, l=108.5, c=111.0)
    assert strat().entry_signal(ctx(df)) is None


def test_no_entry_in_trending_tape():
    c = np.linspace(100, 200, 300)
    o = np.r_[c[0], c[:-1]]
    idx = pd.bdate_range("2025-01-01", periods=300)
    df = pd.DataFrame({"open": o, "high": c * 1.004, "low": np.minimum(o, c) * 0.996,
                       "close": c, "volume": 1e6}, index=idx)
    assert strat().entry_signal(ctx(df)) is None


def test_gates_adx_width_rr_and_bearish_close():
    df = floor_touch_df()
    assert strat().entry_signal(ctx(df)) is not None          # control
    assert strat(max_adx=0.1).entry_signal(ctx(df)) is None   # flat-tape gate
    assert strat(min_range_width_atr=50).entry_signal(ctx(df)) is None   # width gate
    assert strat(min_reward_risk=50).entry_signal(ctx(df)) is None       # R:R gate
    # same touch but close at mid-range: R:R to the ceiling too small
    base = ranged()
    r = fractal_result(base, "daily", 110.0)
    top = min(r.sup, key=lambda z: z.low).high
    mid = with_last_bar(base, o=top * 1.001, h=111.0, l=top * 0.998, c=110.0)
    assert strat().entry_signal(ctx(mid)) is None
    # bearish close on the touch day
    bear = with_last_bar(base, o=top * 1.03, h=top * 1.035, l=top * 0.998, c=top * 1.012)
    assert strat().entry_signal(ctx(bear)) is None
    assert strat(require_bullish_close=False).entry_signal(ctx(bear)) is not None


def _pos(stop, asof, days_held):
    opened = asof - timedelta(days=days_held)
    return Position("X", "range_bounce", [], stop, 0.0, opened=opened.isoformat())


def test_exits():
    st = strat()
    base = ranged()
    asof = (base.index[-1] + pd.offsets.BDay(1)).date()
    r = fractal_result(base, "daily", 110.0)
    ceiling = max(z.low for z in r.res) if r.res else 119.0

    at_ceiling = with_last_bar(base, 117.0, ceiling * 1.001, 116.5, ceiling * 0.995)
    sig = st.exit_signal(ctx(at_ceiling), _pos(97.0, asof, 10))
    assert sig is not None and "ceiling" in sig.reason

    mid = with_last_bar(base, 109.0, 111.0, 108.5, 110.0)
    assert st.exit_signal(ctx(mid), _pos(97.0, asof, 10)) is None                 # nothing yet
    sig = st.exit_signal(ctx(mid), _pos(97.0, asof, 60))
    assert sig is not None and "time stop" in sig.reason
    sig = st.exit_signal(ctx(mid), _pos(115.0, asof, 10))
    assert sig is not None and "stop" in sig.reason                                # close <= stop
    # opened in the future (order filled tomorrow) -> never exits today
    assert st.exit_signal(ctx(mid), _pos(97.0, asof + timedelta(days=3), 0)) is None


def test_backtest_round_trips_on_ranging_symbols(tmp_path):
    s = copy.deepcopy(S)
    syms = ["R1", "R2", "R3"]
    data = {f"R{i}": ranged(n=500, period=10 + 2 * i, seed=i, start="2024-01-01") for i in (1, 2, 3)}
    s.universe["lists"]["default"] = syms
    s.universe["instruments"] = {k: k for k in syms}
    for name, cfg in s.strategies.items():
        cfg["enabled"] = name == "range_bounce"
    s.filters = {}
    idx = data["R1"].index
    eq, trades = run_backtest(s, data, idx[90].date(), idx[-1].date(), str(tmp_path / "bt"))
    sells = [t for t in trades if t["side"] == "SELL"]
    assert sells, "expected at least one closed range trade"
    assert all(t["strategy"] == "range_bounce" for t in trades)
    reasons = " | ".join(t.get("reason", "") for t in sells)
    assert "ceiling" in reasons or "time stop" in reasons


def test_shipped_disabled_and_wired():
    assert S.strategies["range_bounce"]["enabled"] is False
    assert S.risk["capital"]["allocation_pct"]["range_bounce"] == 100
    assert S.strategies["range_bounce"]["market_regime_tag"] == "SIDEWAYS"
    assert "room_to_resistance" in S.strategies["range_bounce"]["exclude_filters"]
