"""Support/resistance engine switch (legacy <-> fractal), the gates that use it, and rollback safety.

Every test that flips the engine restores the global default afterwards, so the rest of the suite
keeps running on config/sr_engine.json's value ("legacy" as shipped)."""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

import trader.core.fractal_levels as FL
import trader.strategies._swing_common as SC
from trader.core.calendar import TradingCalendar
from trader.core.config import ROOT, load_settings
from trader.core.levels import Level, nearest_levels, support_resistance
from trader.core.models import Signal
from trader.filters.base import build_filter
from trader.strategies.base import SymbolContext, build_strategy

S = load_settings()
LEVELS = json.loads((ROOT / "config" / "levels.json").read_text())


@pytest.fixture(autouse=True)
def _restore_engine():
    FL.reset_settings_cache()
    yield
    FL.reset_settings_cache()


def set_global(engine: str) -> None:
    FL.reset_settings_cache()
    FL._SETTINGS["engine"] = engine


def ctx(df, sym="X"):
    cal = TradingCalendar(set(), trading_days=[t.date() for t in df.index])
    return SymbolContext(sym, df, df.index[-1].date(), cal)


def ranged(n=300, lo=100.0, hi=120.0, period=40, tail=None, seed=7):
    """Range-bound daily bars (floor ~100, ceiling ~120); optional closing path appended."""
    rng = np.random.default_rng(seed)
    mid, amp = (lo + hi) / 2, (hi - lo) / 2
    c = mid + amp * np.sin(np.arange(n) * 2 * np.pi / period) + rng.normal(0, 0.3, n)
    if tail is not None:
        c = np.r_[c, tail]
    o = np.r_[c[0], c[:-1]]
    idx = pd.bdate_range("2025-01-01", periods=len(c))
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.003, "low": np.minimum(o, c) * 0.997,
                         "close": c, "volume": 1e6}, index=idx)


# ---- switch resolution ------------------------------------------------------------------------
def test_shipped_default_is_legacy_and_overrides_resolve():
    assert json.loads((ROOT / "config" / "sr_engine.json").read_text())["engine"] == "legacy"
    assert FL.default_engine() == "legacy"
    assert FL.resolve_engine(None) == "legacy"
    assert FL.resolve_engine("fractal") == "fractal"
    with pytest.raises(ValueError):
        FL.resolve_engine("fractals")


def test_shipped_consumer_overrides_are_null():
    for name in ("breakout_trend", "sr_reversal"):
        assert S.strategies[name]["levels_engine"] is None
    assert S.filters["resistance"]["engine"] is None


# ---- nearest_zones ----------------------------------------------------------------------------
def test_legacy_path_is_the_old_code_and_never_touches_fractal(monkeypatch):
    df = ranged()
    px = float(df.close.iloc[-1])
    monkeypatch.setattr(SC, "fractal_nearest", lambda *a, **k: pytest.fail("fractal engine called in legacy mode"))
    got = SC.nearest_zones(df, "daily", LEVELS["daily"], px)
    want = nearest_levels(support_resistance(df, "daily", LEVELS["daily"]), px)
    for side in ("support", "resistance"):
        assert (got[side] is None) == (want[side] is None)
        if got[side] is not None:
            assert (got[side].price_low, got[side].price_high, got[side].strength) == \
                   (want[side].price_low, want[side].price_high, want[side].strength)


@pytest.mark.parametrize("how", ["global", "override"])
def test_fractal_path_via_global_or_override_returns_levels(how):
    df = ranged()
    px = float(df.close.iloc[-1])
    if how == "global":
        set_global("fractal")
        z = SC.nearest_zones(df, "daily", LEVELS["daily"], px)
    else:
        z = SC.nearest_zones(df, "daily", LEVELS["daily"], px, engine="fractal")
    assert isinstance(z["support"], Level) and isinstance(z["resistance"], Level)
    assert z["support"].price_high <= px or z["support"].role == "S"
    assert z["result"].atr_pct > 0
    assert abs(z["support"].mid - 99.7) < 3 and abs(z["resistance"].mid - 120.3) < 3


def test_fractal_zone_containing_price_is_resistance_when_rallied_into():
    # rally from the floor up INTO the ceiling band (~119.7-120.5): legacy ignores the straddling zone,
    # fractal reports it as resistance with price inside
    df = ranged(n=260, tail=np.linspace(101, 120.0, 12))
    px = float(df.close.iloc[-1])
    z = SC.nearest_zones(df, "daily", LEVELS["daily"], px, engine="fractal")
    r = z["resistance"]
    assert r is not None and r.price_low <= px <= r.price_high and r.role == "R"


def test_cached_build_is_shared_and_classification_does_not_leak_between_prices():
    df = ranged()
    a = FL.fractal_result(df, "daily", 99.0)
    b = FL.fractal_result(df, "daily", 125.0)
    assert FL.zones_build(df, "daily") is FL.zones_build(df, "daily")          # one build per day
    roles_a = {(z.low, z.high): z.role for z in a.zones}
    roles_b = {(z.low, z.high): z.role for z in b.zones}
    assert roles_a.keys() == roles_b.keys() and roles_a != roles_b               # same zones, own roles
    assert all(z.role == "" for z in FL.zones_build(df, "daily").zones)        # cache never mutated


# ---- breakout_trend ---------------------------------------------------------------------------
def _breakout_ctx(prev_close, close):
    df = ranged(n=300, lo=90, hi=110)
    df = pd.concat([df.iloc[:-2], df.iloc[-2:].assign(close=[prev_close, close], high=[prev_close, close],
                                                         open=[prev_close, prev_close], volume=[1e6, 5e6])])
    return ctx(df)


def test_breakout_needs_far_edge_when_prev_close_was_inside_zone(monkeypatch):
    zone = Level(price_low=100.0, price_high=104.0, strength=3)
    monkeypatch.setattr("trader.strategies.breakout_trend.nearest_zones",
                        lambda *a, **k: {"support": None, "resistance": zone})
    strat = build_strategy("breakout_trend", copy.deepcopy(S.strategies["breakout_trend"]))
    strat._trend_ok = lambda c: (True, 90.0)
    inside_then_mid = _breakout_ctx(prev_close=101.0, close=103.0)     # still inside the zone
    inside_then_out = _breakout_ctx(prev_close=101.0, close=105.0)     # cleared the far edge
    below_then_mid = _breakout_ctx(prev_close=98.0, close=103.0)       # legacy case: near edge suffices
    assert strat.entry_signal(inside_then_mid) is None
    assert strat.entry_signal(inside_then_out) is not None
    assert strat.entry_signal(below_then_mid) is not None


# ---- resistance filter --------------------------------------------------------------------------
def _sig(px, stop=None, strat="weekly_rsi_sma20"):
    return Signal("X", strat, "ENTRY", "2026-01-01", "t", stop=stop, ref_price=px)


def test_resistance_filter_legacy_branch_unchanged(monkeypatch):
    cfg = dict(S.filters["resistance"], timeframe="daily")
    f = build_filter("resistance", cfg)
    df = ranged(n=260, tail=np.linspace(101, 120.0, 12))
    monkeypatch.setattr("trader.filters.resistance.fractal_result", lambda *a, **k: pytest.fail("fractal in legacy"))
    r = f.evaluate(_sig(float(df.close.iloc[-1])), ctx(df))
    assert "[fractal]" not in r.reason


def test_resistance_filter_fractal_blocks_inside_zone_and_can_be_relaxed():
    df = ranged(n=260, tail=np.linspace(101, 120.0, 12))
    px = float(df.close.iloc[-1])
    f = build_filter("resistance", dict(S.filters["resistance"], timeframe="daily", engine="fractal"))
    r = f.evaluate(_sig(px), ctx(df))
    assert not r.passed and "inside" in r.reason and r.meta["engine"] == "fractal"
    f2 = build_filter("resistance", dict(S.filters["resistance"], timeframe="daily", engine="fractal",
                                         block_inside_zone=False))
    assert f2.evaluate(_sig(px), ctx(df)).passed


def test_resistance_filter_fractal_proximity_in_atr():
    df = ranged(n=300)                                      # mid-range, ceiling some ATRs away
    px = float(df.close.iloc[-1])
    base = dict(S.filters["resistance"], timeframe="daily", engine="fractal")
    r_far = build_filter("resistance", dict(base, proximity_atr=0.1)).evaluate(_sig(px), ctx(df))
    r_wide = build_filter("resistance", dict(base, proximity_atr=50)).evaluate(_sig(px), ctx(df))
    assert r_far.passed and not r_wide.passed and "limit" in r_wide.reason
