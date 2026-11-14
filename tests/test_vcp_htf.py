"""vcp_breakout / high_tight_flag: spec rules (setup, trigger, exits), the engine's intraday
resting-order execution (buy-stop entry, hard stop, take-profit limit), per-strategy spec sizing,
and partial-exit (TRIM) accounting. Deterministic hand-built OHLC, no real market data needed."""
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
from trader.risk.capital import CapitalManager
from trader.risk.sizing import effective_risk, size_entry
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
    """300-bar steady uptrend with 3% daily ranges (Trend Template passes, but no 4-bar window is
    tight enough for a VCP), then a 4-bar 2.1% contraction - the only setup in the series."""
    df = bars(100 * 1.003 ** np.arange(300), spread=0.03)
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


SLIP = S.system["execution"]["slippage_bps"] / 1e4


def buy_px(p):
    return round(p * (1 + SLIP), 2)


def sell_px(p):
    return round(p * (1 - SLIP), 2)


def test_spec_defaults_in_config():
    """Guards against non-spec defaults creeping back into either strategy's config."""
    for n in ("vcp_breakout", "high_tight_flag"):
        c = S.strategies[n]
        assert c["execution"]["mode"] == "intraday"
        assert c["entry"]["tick"] == 0.01 and c["entry"]["trigger_on"] == "high"
        assert not c.get("open_checks", {}).get("max_chase_pct")
        assert "stop_trigger" not in c["exit"]
        sz = c["sizing"]
        assert (sz["capital_mode"], sz["risk_pct"], sz["max_position_pct"]) == ("equity", 1.0, 20.0)
        assert sz["min_stop_distance_pct"] == 0 and sz["min_notional"] == 0 and sz["cash_buffer_pct"] == 0
        assert sz["portfolio_open_risk_cap"] is False
    v, h = S.strategies["vcp_breakout"], S.strategies["high_tight_flag"]
    assert v["entry"]["contraction_bars"] == 4 and v["entry"]["max_spread_pct"] == 4.0
    assert v["entry"]["max_initial_risk_pct"] == 5.0 and v["rs_rank_min_percentile"] == 0.70
    assert (v["exit"]["target_r"], v["exit"]["trim_fraction"], v["exit"]["trail_ema_period"]) == (2.0, 0.5, 20)
    assert (h["entry"]["surge_lookback"], h["entry"]["min_surge_pct"]) == (40, 30.0)
    assert (h["entry"]["flag_bars"], h["entry"]["max_flag_range_pct"]) == (5, 8.0)
    assert h["entry"]["max_initial_risk_pct"] is None
    assert (h["exit"]["partial_max_day"], h["exit"]["trim_fraction"], h["exit"]["trail_sma_period"]) == (2, 0.33, 10)


def test_vcp_setup_pivot_and_stop():
    df, top = vcp_base()
    st = vcp().setup(df)
    assert st is not None
    assert st.pivot == round(top * 1.005 + 0.01, 2)
    assert st.stop == round(top * 0.985 - 0.01, 2)
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


def test_buy_stop_fill_price():
    df, top = vcp_base()
    s = vcp()
    pivot = s.setup(df).pivot
    sig = Signal("X", "vcp_breakout", "ENTRY", "", "", stop=1.0, ref_price=pivot)
    assert s.entry_fill_price(sig, pd.Series({"open": top, "high": pivot + 1, "low": top, "close": top})) == pivot
    assert s.entry_fill_price(sig, pd.Series({"open": pivot + 2, "high": pivot + 3, "low": pivot,
                                              "close": pivot})) == pivot + 2      # gap above pivot
    s.cfg["entry"]["trigger_on"] = "close"
    assert s.entry_fill_price(sig, pd.Series({"open": top, "high": pivot + 1, "low": top, "close": pivot + 0.5})) == pivot + 0.5


def test_vcp_risk_cap_in_setup_and_at_fill_no_chase_rule():
    st = vcp()
    st.cfg["entry"]["max_spread_pct"] = 50.0                          # allow a wide base
    df = bars(100 * 1.003 ** np.arange(300))
    top = float(df["close"].iloc[-1])
    df = append(df, [(top, top * 1.02, top * 0.94, top)] * 4)
    s = st.setup(df)
    assert s.details["stop_capped"] and s.stop == pytest.approx(s.pivot * 0.95, abs=0.01)
    o = PendingOrder("X", "vcp_breakout", "ENTRY", "", "", "", stop=s.stop, ref_price=s.pivot)
    ok, _ = st.validate_open(o, s.pivot * 1.03, s.pivot)               # gap-up fill widens risk
    assert ok and o.stop == pytest.approx(s.pivot * 1.03 * 0.95, abs=0.01)
    ok, _ = st.validate_open(copy.copy(o), s.pivot * 1.08, s.pivot)    # spec has no chase limit
    assert ok


def _pos(entry, stop, opened, trims=0, extra=None):
    meta = {"initial_stop": stop, **({"trims": trims} if trims else {}), **(extra or {})}
    return Position("X", "s", [Lot(100, entry, opened)], stop, entry, opened=opened, meta=meta)


def test_vcp_take_profit_order_and_trail_modes():
    df, top = vcp_base()
    s = vcp()
    entry, stop = 100.0, 97.0
    opened = str(df.index[-1].date())
    tp = s.take_profit_order(ctx(df), _pos(entry, stop, opened))
    assert tp == {"price": 106.0, "fraction": 0.5, "new_stop": 100.0, "reason": tp["reason"]}
    assert s.take_profit_order(ctx(df), _pos(entry, stop, opened, trims=1)) is None
    assert s.exit_signal(ctx(df), _pos(entry, stop, opened)) is None    # stops/targets are resting orders
    # trail 'stop': resting stop = max(entry, EMA20), ratchet only
    p = _pos(top * 0.9, top * 0.9, opened, trims=1)
    lvl = s.stop_level(ctx(df), p)
    assert lvl > top * 0.9 and s.exit_signal(ctx(df), p) is None
    # trail 'close': stop stays at breakeven, close < EMA20 exits at next open
    s.cfg["exit"]["trail_mode"] = "close"
    assert s.stop_level(ctx(df), p) is None
    drop = append(df, [(top, top, top * 0.9, top * 0.92)])
    assert s.exit_signal(ctx(drop), p).action == "EXIT"


def test_vcp_legacy_next_open_mode_still_signals():
    df, top = vcp_base()
    s = vcp()
    s.cfg["execution"]["mode"] = "next_open"
    entry, stop = top, top * 0.97
    opened = str(df.index[-1].date())
    r = entry - stop
    day = append(df, [(entry, entry + 2 * r + 0.1, entry, entry + r)])
    assert s.exit_signal(ctx(day), _pos(entry, stop, opened)).action == "TRIM"
    below = append(df, [(entry, entry, stop - 1, stop - 0.5)])
    assert s.exit_signal(ctx(below), _pos(entry, stop, opened)).action == "EXIT"


# ---- HTF -------------------------------------------------------------------------------------
def htf_base():
    """Flat 60 bars at 100, 15-bar surge to ~160 (+60%), then a tight 5-bar flag ~157-162."""
    df = bars([100.0] * 60)
    df = append(df, [(100 * 1.032 ** i,) * 4 for i in range(1, 16)])
    # descending flag highs: no flag bar takes out the previous high, so the only HTF trigger
    # is the one each test appends
    fl = [(158, 162, 157.5, 159), (159, 161.8, 158, 160), (160, 161.6, 158.5, 160.5),
          (160, 161.5, 158.2, 160.2), (160.2, 161.4, 158.8, 161)]
    return append(df, fl)


def test_momentum_surge_measures_high_after_low():
    df = htf_base()
    pct, det = momentum_surge(df, 40)
    assert pct == pytest.approx((162 - 99) / 99 * 100, rel=1e-6)   # flat-bar low = 100 * 0.99
    assert det["low40_date"] < det["high_max_date"]


def test_htf_setup_and_stop_choice():
    df = htf_base()
    st = htf().setup(df)
    assert st is not None and st.pivot == round(161.4 + 0.01, 2)
    assert st.details["stop_basis"] == "flag_low"        # yesterday's low is only 1.7% below pivot
    assert st.stop == round(157.5 - 0.01, 2)
    s2 = htf()
    s2.cfg["entry"]["min_room_pct"] = 1.0
    assert s2.setup(df).stop == round(158.8 - 0.01, 2) and s2.setup(df).details["stop_basis"] == "yesterday_low"


def test_htf_rejects_weak_surge_and_loose_flag():
    s = htf()
    s.cfg["entry"]["min_surge_pct"] = 80.0
    assert s.setup(htf_base()) is None
    loose = append(htf_base().iloc[:-5], [(158, 170, 150, 160)] * 5)
    assert htf().setup(loose) is None


def test_htf_day1_trim_day3_none_sma10_exit_no_close_stop_signal():
    df = htf_base()
    pivot = htf().setup(df).pivot
    hit = append(df, [(161, pivot + 1, 160.5, 162.5)])
    sig = htf().entry_signal(ctx(hit))
    assert sig and sig.ref_price == pivot
    entry = pivot
    p = _pos(entry, 157.49, str(hit.index[-1].date()))               # day 1 = the entry session
    t = htf().exit_signal(ctx(hit), p)
    assert t.action == "TRIM" and t.meta == {"fraction": 0.33, "new_stop": round(entry, 2)}
    d3 = append(hit, [(162, 163, 161.8, 162.2), (162.2, 164, 162, 163)])
    assert htf().exit_signal(ctx(d3), _pos(entry, 157.49, str(hit.index[-1].date()))) is None
    # a close below the stop is NOT a close-based exit any more (the stop is a resting order)
    sub = append(d3, [(158, 158.2, 150, 151)])
    sig = htf().exit_signal(ctx(sub), _pos(entry, 157.49, str(hit.index[-1].date()), trims=1))
    assert sig.action == "EXIT" and "SMA10" in sig.reason


# ---- sizing (spec Part 3) --------------------------------------------------------------------
def test_spec_sizing_override_equity_mode_and_caps():
    risk, mode, cap = effective_risk(S.risk, S.strategies["vcp_breakout"])
    assert mode == "equity" and cap is False and risk is not S.risk
    assert risk["per_trade"]["min_stop_distance_pct"] == 0 and risk["per_trade"]["min_notional"] == 0
    assert S.risk["per_trade"]["min_stop_distance_pct"] == 3.0                 # shared config untouched
    assert effective_risk(S.risk, S.strategies["weekly_rsi_sma20"]) == (S.risk, None, True)
    cm = CapitalManager(S.risk)
    eq = 2_000_000.0
    base, amt = cm.strategy_base("vcp_breakout", eq, mode), cm.risk_amount("vcp_breakout", eq, mode, 1.0)
    assert (base, amt) == (eq, 20_000.0)
    # 10% stop: risk-bound -> floor(20000 / 10) = 2000 shares (notional 200k < 20% of 2M)
    assert size_entry(amt, 100.0, 90.0, base, 10e6, risk) == (2000, "ok")
    # 2% stop: risk says 10000 shares (1M) -> 20% cap -> 4000 shares (400k)
    assert size_entry(amt, 100.0, 98.0, base, 10e6, risk) == (4000, "ok")
    # cash is still a physical limit
    assert size_entry(amt, 100.0, 98.0, base, 150_000.0, risk) == (1500, "ok")


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


# ---- engine integration: intraday resting orders ---------------------------------------------
def _settings(only, **strategy_overrides):
    s = load_settings()
    for n, c in s.strategies.items():
        c["enabled"] = n in only
    s.strategies["vcp_breakout"]["rs_rank_min_percentile"] = None
    for k, v in strategy_overrides.items():
        s.strategies[only[0]][k] = v
    for c in s.filters.values():
        c["enabled"] = False
    return s


def _run(s, data, tmp, end=None):
    s.universe["lists"]["default"] = list(data)
    s.universe["instruments"] = {k: k for k in data}
    d = next(iter(data.values()))
    return run_backtest(s, data, d.index[0].date(), end or d.index[-1].date(), str(tmp))


def _vcp_path(trigger_bar=None, after=None):
    """VCP base, then the trigger session (default: opens below the pivot, trades through it,
    low stays above the stop), then `after` bars (default: rally past 2R then roll over)."""
    df, top = vcp_base()
    st = vcp().setup(df)
    trig = trigger_bar or (top, st.pivot * 1.01, top * 0.995, st.pivot * 1.005)
    path = [trig]
    if after is None:
        p = st.pivot * 1.005
        after = []
        for _ in range(12):                                      # rally ~ +15%
            after.append((p, p * 1.014, p * 0.996, p * 1.012)); p *= 1.012
        for _ in range(25):                                      # roll over through EMA20
            after.append((p, p * 1.002, p * 0.985, p * 0.988)); p *= 0.988
    return append(df, path + after), st


def test_engine_vcp_buy_stop_fill_same_session_then_2r_limit_then_ema_trail(tmp_path):
    data_df, st = _vcp_path()
    trig_day = str(data_df.index[304].date())
    eq, trades = _run(_settings(["vcp_breakout"]), {"VCP1": data_df}, tmp_path)
    assert [t["side"] for t in trades] == ["BUY", "TRIM", "SELL"], trades
    buy, trim, sell = trades
    assert buy["date"] == trig_day and buy["price"] == buy_px(st.pivot)       # filled AT the pivot, same day
    entry, risk = buy["price"], buy["price"] - buy["stop"]
    target = round(entry + 2 * risk, 2)
    assert trim["price"] == sell_px(max(target, float(data_df.loc[trim["date"], "open"])))
    assert trim["qty"] == buy["qty"] // 2 and trim["new_stop"] == round(entry, 2)
    assert "hard stop" in sell["reason"] and sell["qty"] == buy["qty"] - trim["qty"]
    bar = data_df.loc[sell["date"]]
    assert float(bar["low"]) <= sell["price"] / (1 - SLIP) + 0.01           # filled within the session
    assert sell["pnl"] > 0 and sell["partial_pnl"] == trim["pnl"]
    base = S.risk["capital"]["base_capital"]
    assert eq["cash"].iloc[-1] == pytest.approx(base + sell["pnl"], abs=1.0)
    # spec sizing: 20% of equity cap binds for a ~2% stop (risk qty would be 1% / 2% = 50% notional)
    assert buy["qty"] == int(base * 0.20 // (st.pivot * (1 + SLIP)))


def test_engine_gap_up_entry_fills_at_open(tmp_path):
    df, top = vcp_base()
    st = vcp().setup(df)
    o = st.pivot * 1.02
    data_df, _ = _vcp_path(trigger_bar=(o, o * 1.01, o * 0.995, o * 1.005), after=[(o, o * 1.005, o * 0.99, o)] * 3)
    eq, trades = _run(_settings(["vcp_breakout"]), {"VCP1": data_df}, tmp_path)
    assert trades[0]["side"] == "BUY" and trades[0]["price"] == buy_px(o)


def test_engine_same_session_stop_assumed_hit(tmp_path):
    df, top = vcp_base()
    st = vcp().setup(df)
    data_df, _ = _vcp_path(trigger_bar=(top, st.pivot * 1.01, st.stop - 1, top),
                           after=[(top, top * 1.005, top * 0.99, top)] * 3)
    eq, trades = _run(_settings(["vcp_breakout"]), {"VCP1": data_df}, tmp_path)
    assert [t["side"] for t in trades] == ["BUY", "SELL"]
    assert trades[0]["date"] == trades[1]["date"] and trades[1]["price"] == sell_px(trades[0]["stop"])
    s = _settings(["vcp_breakout"])
    s.strategies["vcp_breakout"]["execution"]["same_bar_stop"] = "ignore"
    eq, trades = _run(s, {"VCP1": data_df}, tmp_path / "ignore")
    assert [t["side"] for t in trades] == ["BUY"]


def test_engine_hard_stop_gap_down_fills_at_open(tmp_path):
    df, top = vcp_base()
    st = vcp().setup(df)
    gap = st.stop * 0.97
    data_df, _ = _vcp_path(after=[(top, top * 1.004, top * 0.999, top),
                                  (gap, gap * 1.01, gap * 0.99, gap)] + [(gap,) * 4] * 2)
    eq, trades = _run(_settings(["vcp_breakout"]), {"VCP1": data_df}, tmp_path)
    assert [t["side"] for t in trades] == ["BUY", "SELL"]
    assert trades[1]["price"] == sell_px(gap) and "gap below stop" in trades[1]["reason"]


def test_engine_htf_day1_trim_next_open_breakeven_then_stop(tmp_path):
    df = htf_base()
    st = htf().setup(df)
    d1 = (161, st.pivot + 1.5, 160.6, 162.8)                       # day 1 closes above entry
    d2 = (162.0, 162.4, 158.0, 158.5)                              # opens above entry, trades below it
    data_df = append(df, [d1, d2] + [(158.5, 159, 158, 158.5)] * 3)
    eq, trades = _run(_settings(["high_tight_flag"]), {"HTF1": data_df}, tmp_path)
    assert [t["side"] for t in trades] == ["BUY", "TRIM", "SELL"], trades
    buy, trim, sell = trades
    assert buy["price"] == buy_px(st.pivot) and buy["date"] == str(data_df.index[-5].date())
    assert trim["date"] == str(data_df.index[-4].date()) and trim["price"] == sell_px(162.0)   # next open
    assert trim["qty"] == int(buy["qty"] * 0.33)
    assert sell["date"] == trim["date"] and "hard stop" in sell["reason"]                 # breakeven stop
    assert sell["price"] == sell_px(buy["price"])


def test_engine_legacy_next_open_mode(tmp_path):
    """execution.mode next_open: the old model still runs end to end (fill at next open)."""
    s = _settings(["vcp_breakout"])
    s.strategies["vcp_breakout"]["execution"]["mode"] = "next_open"
    data_df, st = _vcp_path()
    eq, trades = _run(s, {"VCP1": data_df}, tmp_path)
    assert trades[0]["side"] == "BUY" and trades[0]["date"] == str(data_df.index[305].date())
    assert trades[0]["price"] == buy_px(float(data_df["open"].iloc[305]))


def test_pnl_breakdown_counts_trims_on_still_open_positions(tmp_path):
    """Stop the run on the TRIM day: that partial is banked in cash but has no SELL row yet - it
    must show up in realized_pnl, and net = realized + unrealized."""
    s = _settings(["vcp_breakout"])
    data_df, _ = _vcp_path()
    _, full = _run(s, {"VCP1": data_df}, tmp_path / "full")
    trim_day = date.fromisoformat(next(t["date"] for t in full if t["side"] == "TRIM"))
    eq, trades = _run(_settings(["vcp_breakout"]), {"VCP1": data_df}, tmp_path / "cut", end=trim_day)
    assert [t["side"] for t in trades] == ["BUY", "TRIM"]
    base = S.risk["capital"]["base_capital"]
    res = summarize(eq, trades, base)
    assert res["closed_trades"] == 0
    assert res["realized_pnl"] == pytest.approx(trades[1]["pnl"], abs=1.0)
    assert res["net_pnl"] == pytest.approx(res["realized_pnl"] + res["unrealized_pnl"], abs=1.0)
    assert res["net_pnl"] == pytest.approx(eq["equity"].iloc[-1] - base, abs=1.0)


def test_engine_smoke_both_strategies_on_random_data(data, tmp_path):
    s = _settings(["vcp_breakout", "high_tight_flag"])
    eq, trades = _run(s, data, tmp_path)
    print(summarize(eq, trades, s.risk["capital"]["base_capital"]),
          {k: sum(t["side"] == k for t in trades) for k in ("BUY", "TRIM", "SELL")})
    assert any(t["side"] == "TRIM" for t in trades)
    assert (eq["cash"] >= -1e-6).all()
    prev_eq = eq["equity"].shift(1).fillna(s.risk["capital"]["base_capital"])
    bought = {}
    for t in trades:
        if t["side"] == "BUY":
            bought[(t["symbol"], t["date"])] = t["qty"]
            e = float(prev_eq.loc[pd.Timestamp(t["date"])]) * 1.10       # equity moves intraday
            assert t["qty"] * max(t["price"] - t["stop"], 0) <= e * 0.01 + t["price"]
            assert t["qty"] * t["price"] <= e * 0.20 + t["price"]
        elif t["side"] == "SELL":
            assert bought[(t["symbol"], t["entry_date"])] == t["qty"] + t.get("partial_qty", 0)
            if "hard stop" in t["reason"]:
                bar = data[t["symbol"]].loc[t["date"]]
                # a resting stop fills inside the session's range...
                assert float(bar["low"]) * (1 - SLIP) - 0.01 <= t["price"] <= float(bar["high"]) + 0.01
                # ...and, on a session that started with the position held, never above the open
                if "entry session" not in t["reason"]:
                    assert t["price"] <= float(bar["open"]) + 0.01
