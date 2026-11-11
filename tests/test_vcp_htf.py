"""vcp_breakout / high_tight_flag: setup rules, triggers, exits, and engine partial-exit (TRIM)
accounting. Deterministic hand-built OHLC, no real market data needed."""
from __future__ import annotations

import copy
from datetime import date

import numpy as np
import pandas as pd
import pytest

from trader.backtest.runner import run_backtest, summarize, trade_log
from trader.core.calendar import TradingCalendar
from trader.core.config import load_settings
from trader.core.models import Lot, PendingOrder, Position, Signal
from trader.filters.base import build_filter
from trader.strategies.base import SymbolContext, build_strategy
from trader.strategies.high_tight_flag import momentum_surge
from trader.strategies.vcp_breakout import trend_template

S = load_settings()


def bars(closes, spread=0.01, start="2022-01-03"):
    c = np.asarray(closes, dtype=float)
    idx = pd.bdate_range(start, periods=len(c))
    return pd.DataFrame({"open": c, "high": c * (1 + spread), "low": c * (1 - spread),
                         "close": c, "volume": 1e6}, index=idx)


def append(df, rows):
    """rows: list of (open, high, low, close)."""
    idx = pd.bdate_range(df.index[-1] + pd.offsets.BDay(), periods=len(rows))
    add = pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close"])
    add["volume"] = 1e6
    return pd.concat([df, add])


def ctx(df, sym="X"):
    cal = TradingCalendar(set(), trading_days=[t.date() for t in df.index])
    return SymbolContext(sym, df, df.index[-1].date(), cal)


def vcp_base():
    """300-bar steady uptrend (Trend Template passes), then a 4-bar 2.1% contraction."""
    df = bars(100 * 1.003 ** np.arange(300))
    top = float(df["close"].iloc[-1])
    return append(df, [(top, top * 1.005, top * 0.985, top)] * 4), top


def vcp():
    return build_strategy("vcp_breakout", copy.deepcopy(S.strategies["vcp_breakout"]))


def htf():
    return build_strategy("high_tight_flag", copy.deepcopy(S.strategies["high_tight_flag"]))


# ---- VCP -------------------------------------------------------------------------------------
def test_trend_template_passes_uptrend_fails_downtrend():
    cfg = S.strategies["vcp_breakout"]["entry"]["trend_template"]
    ok, det = trend_template(bars(100 * 1.003 ** np.arange(300)), cfg)
    assert ok, det
    ok, det = trend_template(bars(100 * 0.998 ** np.arange(300)), cfg)
    assert not ok and not det["c4_sma200_rising"]


def test_vcp_setup_pivot_and_stop():
    df, top = vcp_base()
    st = vcp().setup(df)
    assert st is not None
    assert st.pivot == round(top * 1.005 + 0.05, 2)
    assert st.stop == round(top * 0.985 - 0.05, 2)
    assert st.details["spread_pct"] < 4.0


def test_vcp_rejects_wide_contraction():
    df = bars(100 * 1.003 ** np.arange(300))
    top = float(df["close"].iloc[-1])
    df = append(df, [(top, top * 1.03, top * 0.98, top)] * 4)          # ~5.1% spread
    assert vcp().setup(df) is None


def test_vcp_trigger_on_high_and_close():
    df, top = vcp_base()
    pivot = vcp().setup(df).pivot
    hit = append(df, [(top, pivot + 0.5, top * 0.99, top)])          # high crosses, closes back
    assert vcp().entry_signal(ctx(hit)) is not None
    strict = vcp()
    strict.cfg["entry"]["trigger_on"] = "close"
    assert strict.entry_signal(ctx(hit)) is None
    miss = append(df, [(top, pivot - 0.5, top * 0.99, top)])
    assert vcp().entry_signal(ctx(miss)) is None


def test_vcp_risk_cap_in_setup_and_at_open():
    st = vcp()
    st.cfg["entry"]["max_spread_pct"] = 50.0                          # allow a wide base
    df = bars(100 * 1.003 ** np.arange(300))
    top = float(df["close"].iloc[-1])
    df = append(df, [(top, top * 1.02, top * 0.94, top)] * 4)
    s = st.setup(df)
    assert s.details["stop_capped"] and s.stop == pytest.approx(s.pivot * 0.95, abs=0.01)
    o = PendingOrder("X", "vcp_breakout", "ENTRY", "", "", "", stop=s.stop, ref_price=s.pivot)
    ok, _ = st.validate_open(o, s.pivot * 1.03, s.pivot)               # gap-up 3% widens risk
    assert ok and o.stop == pytest.approx(s.pivot * 1.03 * 0.95, abs=0.01)
    ok, why = st.validate_open(copy.copy(o), s.pivot * 1.06, s.pivot)  # > max_chase_pct
    assert not ok and "extended" in why


def _pos(entry, stop, opened, trims=0, extra=None):
    meta = {"initial_stop": stop, **({"trims": trims} if trims else {}), **(extra or {})}
    return Position("X", "s", [Lot(100, entry, opened)], stop, entry, opened=opened, meta=meta)


def test_vcp_exit_sequence_trim_then_ema_trail():
    df, top = vcp_base()
    s = vcp()
    entry, stop = top, top * 0.97
    opened = str(df.index[-1].date())
    r = entry - stop
    day = append(df, [(entry, entry + 2 * r + 0.1, entry, entry + r)])
    sig = s.exit_signal(ctx(day), _pos(entry, stop, opened))
    assert sig.action == "TRIM" and sig.meta["fraction"] == 0.5 and sig.meta["new_stop"] == round(entry, 2)
    below = append(df, [(entry, entry, stop - 1, stop - 0.5)])
    assert s.exit_signal(ctx(below), _pos(entry, stop, opened)).action == "EXIT"
    # after the trim: stop trails max(entry, EMA20); a close under it exits
    p = _pos(entry, entry, opened, trims=1)
    assert s.stop_level(ctx(day), p) >= entry
    drop = append(day, [(entry, entry, entry * 0.98, entry * 0.99)])
    p.stop = s.stop_level(ctx(drop), p)
    assert s.exit_signal(ctx(drop), p).action == "EXIT"


# ---- HTF -------------------------------------------------------------------------------------
def htf_base():
    """Flat 60 bars at 100, 15-bar surge to ~160 (+60%), then a tight 5-bar flag ~157-162."""
    df = bars([100.0] * 60)
    df = append(df, [(100 * 1.032 ** i,) * 4 for i in range(1, 16)])
    fl = [(158, 161, 157.5, 159), (159, 161.5, 158, 160), (160, 162, 158.5, 160.5),
          (160, 161.8, 158.2, 160.2), (160.2, 161.6, 158.8, 161)]
    return append(df, fl)


def test_momentum_surge_measures_high_after_low():
    df = htf_base()
    pct, det = momentum_surge(df, 40)
    assert pct == pytest.approx((162 - 99) / 99 * 100, rel=1e-6)   # flat-bar low = 100 * 0.99
    assert det["low40_date"] < det["high_max_date"]


def test_htf_setup_and_stop_choice():
    df = htf_base()
    st = htf().setup(df)
    assert st is not None and st.pivot == round(161.6 + 0.05, 2)
    assert st.details["stop_basis"] == "flag_low"        # yesterday's low is only 1.7% below pivot
    s2 = htf()
    s2.cfg["entry"]["min_room_pct"] = 1.0
    assert s2.setup(df).stop == round(158.8 - 0.05, 2) and s2.setup(df).details["stop_basis"] == "yesterday_low"


def test_htf_rejects_weak_surge_and_loose_flag():
    s = htf()
    s.cfg["entry"]["min_surge_pct"] = 80.0
    assert s.setup(htf_base()) is None
    loose = append(htf_base().iloc[:-5], [(158, 170, 150, 160)] * 5)
    assert htf().setup(loose) is None


def test_htf_trigger_and_day1_trim_and_sma10_exit():
    df = htf_base()
    pivot = htf().setup(df).pivot
    hit = append(df, [(161, pivot + 1, 160.5, 162.5)])
    sig = htf().entry_signal(ctx(hit))
    assert sig and sig.ref_price == pivot
    entry = 163.0
    d1 = append(hit, [(163, 166, 162.5, 165)])
    p = _pos(entry, 157.5, str(d1.index[-1].date()))
    t = htf().exit_signal(ctx(d1), p)
    assert t.action == "TRIM" and t.meta["fraction"] == 0.33
    # day 3 positive: no trim any more (only day 1-2)
    d3 = append(d1, [(165, 166, 164, 165.5), (165.5, 167, 165, 166)])
    p3 = _pos(entry, 157.5, str(d1.index[-1].date()))
    assert htf().exit_signal(ctx(d3), p3) is None
    crash = append(d3, [(160, 160, 150, 151)])                         # close < SMA10
    assert htf().exit_signal(ctx(crash), _pos(entry, 140, str(d1.index[-1].date()), trims=1)).action == "EXIT"


# ---- rs_rank per-strategy gate ---------------------------------------------------------------
def test_rs_rank_strategy_gate_only_applies_to_that_strategy():
    f = build_filter("rs_rank", dict(S.filters["rs_rank"], enabled=True, mode="rank"))
    f.bind_strategy_cfgs({"vcp_breakout": {"rs_rank_min_percentile": 0.7}, "other": {}})
    f._percentile_by_symbol = {"A": 0.65, "B": 0.85}
    sig = lambda sym, st: Signal(sym, st, "ENTRY", "2024-01-01", "")
    assert not f.evaluate(sig("A", "vcp_breakout"), None).passed
    assert f.evaluate(sig("B", "vcp_breakout"), None).passed
    assert f.evaluate(sig("A", "other"), None).passed                 # global rank mode untouched
    assert f.evaluate(sig("A", "vcp_breakout"), None).score == 0.65


# ---- engine integration: TRIM accounting -----------------------------------------------------
def _engineered_vcp_symbol():
    """VCP base -> breakout -> run to well past 2R -> roll over below EMA20."""
    df, top = vcp_base()
    path = [(top, top * 1.012, top * 0.998, top * 1.01)]                     # trigger day
    p = top * 1.01
    for _ in range(12):                                                       # rally ~ +15%
        path.append((p, p * 1.014, p * 0.996, p * 1.012)); p *= 1.012
    for _ in range(15):                                                       # roll over
        path.append((p, p * 1.002, p * 0.985, p * 0.988)); p *= 0.988
    return append(df, path)


def test_engine_trim_then_exit_accounting(tmp_path):
    s = load_settings()
    for n, c in s.strategies.items():
        c["enabled"] = n == "vcp_breakout"
    s.strategies["vcp_breakout"]["rs_rank_min_percentile"] = None
    for c in s.filters.values():
        c["enabled"] = False
    data = {"VCP1": _engineered_vcp_symbol()}
    s.universe["lists"]["default"] = list(data)
    s.universe["instruments"] = {k: k for k in data}
    eq, trades = run_backtest(s, data, data["VCP1"].index[0].date(), data["VCP1"].index[-1].date(), str(tmp_path))
    sides = [t["side"] for t in trades]
    assert sides == ["BUY", "TRIM", "SELL"], trades
    buy, trim, sell = trades
    assert trim["qty"] == buy["qty"] // 2 and sell["qty"] == buy["qty"] - trim["qty"]
    assert sell["partial_qty"] == trim["qty"] and sell["partial_pnl"] == trim["pnl"]
    assert sell["avg_cost"] == round(buy["price"], 2)
    # cash identity: flat at the end -> cash = initial + round-trip pnl (fees all included)
    base = s.risk["capital"]["base_capital"]
    assert eq["cash"].iloc[-1] == pytest.approx(base + sell["pnl"], abs=1.0)
    assert summarize(eq, trades)["closed_trades"] == 1
    tl = trade_log(trades)
    assert tl["pnl_pct"].iloc[0] == pytest.approx(sell["pnl"] / (buy["price"] * buy["qty"]) * 100, abs=0.05)


def test_engine_smoke_both_strategies_on_random_data(data, tmp_path):
    s = load_settings()
    for n, c in s.strategies.items():
        c["enabled"] = n in ("vcp_breakout", "high_tight_flag")
    s.strategies["vcp_breakout"]["rs_rank_min_percentile"] = None
    for c in s.filters.values():
        c["enabled"] = False
    s.universe["lists"]["default"] = list(data)
    s.universe["instruments"] = {k: k for k in data}
    eq, trades = run_backtest(s, data, date(2018, 1, 1), date(2024, 12, 31), str(tmp_path))
    print(summarize(eq, trades), {k: sum(t["side"] == k for t in trades) for k in ("BUY", "TRIM", "SELL")},
          {st: sum(t["side"] == "BUY" and t["strategy"] == st for t in trades) for st in ("vcp_breakout", "high_tight_flag")})
    assert any(t["side"] == "TRIM" for t in trades)
    assert (eq["cash"] >= 0).all()
    base = s.risk["capital"]["base_capital"]
    # every round trip: sold qty + trimmed qty == bought qty, and risk budget respected at entry
    bought = {}
    for t in trades:
        if t["side"] == "BUY":
            bought[(t["symbol"], t["date"])] = t["qty"]
            assert t["qty"] * max(t["price"] - t["stop"], 0) <= base * 0.01 * 1.02
            assert t["qty"] * t["price"] <= base * 0.20 * 1.01
        elif t["side"] == "SELL":
            assert bought[(t["symbol"], t["entry_date"])] == t["qty"] + t.get("partial_qty", 0)
