import json
import pytest
from datetime import date

import numpy as np
import pandas as pd

from trader.brokers.paper import compute_fees
from trader.core.calendar import TradingCalendar
from trader.core.config import load_settings
from trader.core.indicators import rsi, sma
from trader.core.models import Lot, PendingOrder, Position
from trader.risk.sizing import position_open_risk, size_add, size_entry
from trader.strategies.base import build_strategy

S = load_settings()


def test_rsi_monotonic_up_is_100_and_bounded():
    up = pd.Series(np.arange(1.0, 60.0))
    assert rsi(up, 14).iloc[-1] == 100.0
    x = pd.Series(np.random.default_rng(0).normal(0, 1, 300).cumsum() + 500)
    r = rsi(x, 14).dropna()
    assert r.between(0, 100).all()


def test_calendar_week_end_with_friday_holiday():
    cal = TradingCalendar({date(2026, 1, 2)})       # Fri holiday
    assert cal.is_week_end(date(2026, 1, 1)) is True    # Thu is last trading day
    assert cal.is_week_end(date(2026, 1, 5)) is False   # Mon


def test_size_entry_risk_math():
    qty, why = size_entry(10_000, 100.0, 90.0, 1_000_000, 1_000_000, S.risk)
    assert qty == 1000 and why == "ok"                  # 1% of 10L / Rs10 stop distance


def test_size_entry_respects_min_stop_floor_and_position_cap():
    qty, _ = size_entry(10_000, 100.0, 99.5, 1_000_000, 1_000_000, S.risk)
    assert qty == 2000                                   # capped at 20% of base, not 1% / 0.5


def test_size_add_uses_freed_risk_budget():
    pos = Position("X", "weekly_rsi_sma20", [Lot(1000, 100.0, "2026-01-02")], stop=90.0, last_price=110.0)
    assert position_open_risk(pos, 90.0) == 10_000       # budget fully used at 1% -> no add
    assert size_add(pos, 110.0, 90.0, 1_000_000, 1_000_000, S.risk)[0] == 0
    assert position_open_risk(pos, 101.0) == 0.0         # stop above cost: profit locked
    qty, _ = size_add(pos, 110.0, 101.0, 1_000_000, 1_000_000, S.risk)
    assert qty == 818                                    # risk budget allows 1111; 20%-of-base position cap binds


def test_fees_positive_and_sell_has_dp():
    assert compute_fees("SELL", 100_000, S.fees) > compute_fees("BUY", 100_000, S.fees) - 20


def test_rule3_open_checks():
    st = build_strategy("weekly_rsi_sma20", S.strategies["weekly_rsi_sma20"])
    o = PendingOrder("X", "weekly_rsi_sma20", "ENTRY", "2026-01-02", "2026-01-05", "t", stop=95.0)
    assert st.validate_open(o, 101.0, 100.0)[0] is True
    assert st.validate_open(o, 99.0, 100.0)[0] is False   # gap down
    o2 = PendingOrder("X", "weekly_rsi_sma20", "ENTRY", "2026-01-02", "2026-01-05", "t", stop=105.0)
    assert st.validate_open(o2, 104.0, 103.0)[0] is False  # open below SMA


def test_regime_filter_bullish_and_bearish():
    from trader.risk.regime import RegimeFilter
    cfg = dict(enabled=True, benchmark_symbol="BENCH", timeframe_rule="W-FRI",
              sma_period=5, buffer_pct=0.0, fail_mode="block", min_bars=6)
    rf = RegimeFilter(cfg)
    assert rf.enabled is True

    up = pd.DataFrame({"open": range(60), "high": range(60), "low": range(60),
                       "close": range(1, 61), "volume": [1000] * 60},
                      index=pd.bdate_range("2025-01-01", periods=60))
    rs = rf.status(up, date(2025, 3, 20))
    assert rs.bullish is True

    down = pd.DataFrame({"open": range(60), "high": range(60), "low": range(60),
                         "close": list(range(60, 0, -1)), "volume": [1000] * 60},
                        index=pd.bdate_range("2025-01-01", periods=60))
    rs = rf.status(down, date(2025, 3, 20))
    assert rs.bullish is False

    # fail_mode: too little history -> block by default, allow if configured
    thin = up.iloc[:3]
    assert rf.status(thin, date(2025, 1, 5)).bullish is False
    rf2 = RegimeFilter({**cfg, "fail_mode": "allow"})
    assert rf2.status(thin, date(2025, 1, 5)).bullish is True


def test_regime_filter_blocks_entries_in_backtest(data, tmp_path):
    from trader.backtest.runner import run_backtest
    s = load_settings()
    s.universe["instruments"] = {k: k for k in data}
    s.universe["lists"]["default"] = list(data)

    # a benchmark that's flat-then-down across the whole window - should block ~all entries
    bench = data["S1"].copy()
    bench["close"] = list(range(2000, 2000 - len(bench), -1))
    data_with_bench = {**data, "BENCH": bench}
    s.universe["instruments"]["BENCH"] = "BENCH"

    s.filters["market_regime"] = {**s.filters["market_regime"], "enabled": False,
                                 "benchmark_symbol": "BENCH", "min_bars": 20}
    _, trades_baseline = run_backtest(s, data, date(2020, 1, 1), date(2021, 12, 31), str(tmp_path / "baseline"))
    s.filters["market_regime"]["enabled"] = True
    _, trades_gated = run_backtest(s, data_with_bench, date(2020, 1, 1), date(2021, 12, 31), str(tmp_path / "gated"))

    entries_baseline = [t for t in trades_baseline if t.get("action") == "ENTRY"]
    entries_gated = [t for t in trades_gated if t.get("action") == "ENTRY"]
    assert len(entries_gated) < len(entries_baseline)


def _exit_fixture():
    """Uptrend ending on a Friday; returns (strategy factory, daily df, sma, friday)."""
    idx = pd.bdate_range(end="2026-03-20", periods=300)          # 2026-03-20 is a Friday
    close = pd.Series(np.linspace(100, 300, len(idx)), index=idx)
    df = pd.DataFrame({"open": close, "high": close * 1.002, "low": close * 0.998,
                       "close": close, "volume": 1000})
    return df, idx[-1].date()


def _pos(opened: str):
    return Position("X", "weekly_rsi_sma20", [Lot(10, 100.0, opened)], stop=0.0, last_price=100.0, opened=opened)


def _strat(trigger: str):
    cfg = {**S.strategies["weekly_rsi_sma20"], "exit": {"buffer_pct": 0.0, "trigger": trigger}}
    return build_strategy("weekly_rsi_sma20", cfg)


def test_exit_uses_weekly_wick_touch_and_completed_candle_only():
    from trader.strategies.base import SymbolContext
    df, fri = _exit_fixture()
    cal = TradingCalendar(set())
    sma_now = _strat("wick")._sma_last(SymbolContext("X", df, fri, cal).weekly("W-FRI", False))

    # This week's candle dips to EXACTLY the SMA20 (a touch) but closes far above it
    df.loc[df.index[-3:], "low"] = sma_now
    ctx = SymbolContext("X", df, fri, cal)
    assert _strat("wick").exit_signal(ctx, _pos("2026-01-05")) is not None   # touch -> exit
    assert _strat("close").exit_signal(ctx, _pos("2026-01-05")) is None      # close never went below

    # Mid-week (Wednesday) the same dip must NOT be judged: the week isn't complete yet
    wed = pd.Timestamp("2026-03-18").date()
    ctx_wed = SymbolContext("X", df.loc[:pd.Timestamp(wed)], wed, cal)
    assert _strat("wick").exit_signal(ctx_wed, _pos("2026-01-05")) is None


def test_candle_before_entry_cannot_trigger_exit():
    from trader.strategies.base import SymbolContext
    df, fri = _exit_fixture()
    cal = TradingCalendar(set())
    sma_now = _strat("wick")._sma_last(SymbolContext("X", df, fri, cal).weekly("W-FRI", False))
    df.loc[df.index[-3:], "low"] = sma_now * 0.95                            # candle pierced SMA20
    # Bought the Monday AFTER that candle completed -> the signal candle must be ignored
    ctx_mon = SymbolContext("X", df, fri, cal)
    assert _strat("wick").exit_signal(ctx_mon, _pos("2026-03-23")) is None
    # Bought earlier in that same week -> this candle is post-entry and does count
    assert _strat("wick").exit_signal(ctx_mon, _pos("2026-03-16")) is not None


def test_daily_close_mode_is_rejected():
    import pytest
    cfg = {**S.strategies["weekly_rsi_sma20"], "exit": {"mode": "daily_close", "buffer_pct": 0.0}}
    with pytest.raises(ValueError):
        build_strategy("weekly_rsi_sma20", cfg)


def test_no_free_slot_journal_names_rank_and_winners(data, tmp_path):
    from trader.backtest.runner import run_backtest
    s = load_settings()
    s.universe["instruments"] = {k: k for k in data}
    s.universe["lists"]["default"] = list(data)
    s.risk["portfolio"]["max_positions"] = 1            # force slot competition
    work = tmp_path / "slots"
    run_backtest(s, data, date(2020, 1, 1), date(2022, 12, 31), str(work))
    import json
    rows = [json.loads(l) for l in (work / "journal.jsonl").read_text().splitlines()]
    skipped = [r["why"] for r in rows if r["event"] == "ENTRY_SKIPPED" and r["why"].startswith("no free slot")]
    assert skipped, "expected at least one slot-competition skip with max_positions=1"
    assert all("ranked #" in w and "went to:" in w for w in skipped)


def test_monte_carlo_reshuffle_preserves_total_pnl_exactly():
    from trader.backtest.monte_carlo import run_monte_carlo
    pnl = np.array([500.0, -300.0, 1200.0, -800.0, 200.0, -150.0])
    res = run_monte_carlo(pnl, starting_capital=100_000, n_sims=500, method="reshuffle", seed=1)
    expected_final = 100_000 + pnl.sum()
    assert np.allclose(res["equity_paths"][:, -1], expected_final)   # reorder, not resample -> same total, always
    assert res["horizon"] == len(pnl)


def test_monte_carlo_all_losses_means_certain_ruin():
    from trader.backtest.monte_carlo import run_monte_carlo
    pnl = np.array([-1000.0] * 10)
    res = run_monte_carlo(pnl, starting_capital=10_000, n_sims=300, horizon=10,
                          method="bootstrap", ruin_pct=50.0, seed=2)
    assert res["prob_loss_pct"] == 100.0
    assert res["prob_ruin_pct"] == 100.0        # 10 * -1000 = -10,000 -> wipes out 100% of a 10k start


def test_monte_carlo_all_gains_means_zero_ruin_and_loss():
    from trader.backtest.monte_carlo import run_monte_carlo
    pnl = np.array([100.0, 200.0, 50.0])
    res = run_monte_carlo(pnl, starting_capital=10_000, n_sims=300, method="bootstrap", seed=3)
    assert res["prob_loss_pct"] == 0.0
    assert res["prob_ruin_pct"] == 0.0
    assert res["final_equity"]["p5"] > 10_000


def test_monte_carlo_seed_is_reproducible():
    from trader.backtest.monte_carlo import run_monte_carlo
    pnl = np.array([500.0, -300.0, 1200.0, -800.0, 200.0])
    r1 = run_monte_carlo(pnl, 100_000, n_sims=200, method="bootstrap", seed=42)
    r2 = run_monte_carlo(pnl, 100_000, n_sims=200, method="bootstrap", seed=42)
    assert np.array_equal(r1["equity_paths"], r2["equity_paths"])


def test_monte_carlo_percentile_rank_and_fan_chart_shape():
    from trader.backtest.monte_carlo import equity_fan_chart, percentile_rank, run_monte_carlo
    pnl = np.array([500.0, -300.0, 1200.0, -800.0, 200.0, -150.0, 900.0, -400.0])
    res = run_monte_carlo(pnl, 100_000, n_sims=1000, method="bootstrap", seed=7)
    actual_final = 100_000 + pnl.sum()
    rank = percentile_rank(res, actual_final)
    assert 0.0 <= rank <= 100.0
    fan = equity_fan_chart(res)
    assert len(fan) == res["horizon"] + 1          # +1 for the t=0 starting-capital row
    assert list(fan.columns) == ["p5", "p25", "p50", "p75", "p95"]
    assert (fan["p5"] <= fan["p50"]).all() and (fan["p50"] <= fan["p95"]).all()


def test_trade_log_includes_strategy_column():
    from trader.backtest.runner import trade_log
    trades = [{"date": "2026-01-01", "symbol": "X", "strategy": "weekly_rsi_sma20", "side": "BUY",
              "qty": 10, "price": 100.0, "fees": 1.0, "action": "ENTRY"},
             {"date": "2026-02-01", "symbol": "X", "strategy": "weekly_rsi_sma20", "side": "SELL",
              "qty": 10, "price": 110.0, "fees": 1.0, "pnl": 98.0, "avg_cost": 100.0,
              "adds": 0, "reason": "test", "entry_date": "2026-01-01", "holding_days": 31}]
    tl = trade_log(trades)
    assert "strategy" in tl.columns
    assert tl.iloc[0]["strategy"] == "weekly_rsi_sma20"


def test_open_positions_summary_reads_real_portfolio_schema(tmp_path):
    from trader.backtest.monte_carlo import open_positions_summary
    from trader.core.models import Lot, Position
    from trader.portfolio.state import Portfolio, StateStore
    store = StateStore(tmp_path)
    pf = Portfolio(cash=50_000, positions={
        "WIN": Position("WIN", "s", [Lot(100, 50.0, "2026-01-01")], stop=45.0, last_price=70.0),
        "LOSS": Position("LOSS", "s", [Lot(50, 200.0, "2026-01-01")], stop=180.0, last_price=190.0),
    })
    store.save_portfolio(pf)
    summary = open_positions_summary(store.pf_path)
    assert summary["n_open"] == 2
    assert summary["unrealized_pnl"] == pytest.approx((70 - 50) * 100 + (190 - 200) * 50)
    names = {p["symbol"] for p in summary["positions"]}
    assert names == {"WIN", "LOSS"}


def test_open_positions_summary_missing_file_is_empty_not_an_error(tmp_path):
    from trader.backtest.monte_carlo import open_positions_summary
    summary = open_positions_summary(tmp_path / "does_not_exist.json")
    assert summary == {"n_open": 0, "unrealized_pnl": 0.0, "positions": []}


def _write_json(p, obj):
    import json
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def test_load_account_settings_overrides_capital_and_strategy_subset(tmp_path):
    from trader.core.config import list_accounts, load_account_settings
    cfg = tmp_path / "config"
    _write_json(cfg / "system_config.json", {"mode": "paper", "timezone": "Asia/Kolkata",
               "paths": {"state_dir": "state", "cache_dir": "data_cache"},
               "data": {}, "execution": {"slippage_bps": 10}, "logging": {"level": "INFO"}})
    _write_json(cfg / "risk_config.json", {"capital": {"base_capital": 1_000_000, "mode": "fixed",
               "allocation_pct": {"strat_a": 100, "strat_b": 100}}, "per_trade": {}, "portfolio": {}, "pyramiding": {}})
    _write_json(cfg / "fee_config.json", {})
    _write_json(cfg / "holidays.json", {"holidays": []})
    _write_json(cfg / "universe.json", {"instruments": {}, "lists": {"default": []}})
    _write_json(cfg / "strategies" / "strat_a.json", {"enabled": True, "universe": "default"})
    _write_json(cfg / "strategies" / "strat_b.json", {"enabled": True, "universe": "default"})
    _write_json(cfg / "accounts" / "cousin_a.json",
               {"base_capital": 250_000, "strategies_enabled": ["strat_a"],
                "upstox_token_env": "UPSTOX_ACCESS_TOKEN_COUSIN_A"})

    assert list_accounts(cfg) == ["cousin_a"]
    s = load_account_settings("cousin_a", cfg)
    assert s.account_id == "cousin_a"
    assert s.risk["capital"]["base_capital"] == 250_000
    assert set(s.strategies) == {"strat_a"}           # strat_b excluded, not just disabled
    assert s.upstox_token_env == "UPSTOX_ACCESS_TOKEN_COUSIN_A"
    assert s.state_dir_override == "state/accounts/cousin_a"
    # base config on disk is untouched by the override (no mutation of shared files)
    assert json.loads((cfg / "risk_config.json").read_text())["capital"]["base_capital"] == 1_000_000


def test_load_account_settings_default_token_env_and_unknown_strategy(tmp_path):
    import pytest
    from trader.core.config import load_account_settings
    cfg = tmp_path / "config"
    _write_json(cfg / "system_config.json", {"mode": "paper", "timezone": "Asia/Kolkata",
               "paths": {"state_dir": "state", "cache_dir": "data_cache"},
               "data": {}, "execution": {}, "logging": {"level": "INFO"}})
    _write_json(cfg / "risk_config.json", {"capital": {"base_capital": 1_000_000, "mode": "fixed",
               "allocation_pct": {"strat_a": 100}}, "per_trade": {}, "portfolio": {}, "pyramiding": {}})
    _write_json(cfg / "fee_config.json", {})
    _write_json(cfg / "holidays.json", {"holidays": []})
    _write_json(cfg / "universe.json", {"instruments": {}, "lists": {"default": []}})
    _write_json(cfg / "strategies" / "strat_a.json", {"enabled": True, "universe": "default"})
    _write_json(cfg / "accounts" / "friend_b.json", {})   # no overrides at all - still must work

    s = load_account_settings("friend_b", cfg)
    assert s.upstox_token_env == "UPSTOX_ACCESS_TOKEN_FRIEND_B"   # derived default
    assert s.risk["capital"]["base_capital"] == 1_000_000          # unchanged (no override given)
    assert set(s.strategies) == {"strat_a"}                        # unchanged (no override given)

    _write_json(cfg / "accounts" / "bad.json", {"strategies_enabled": ["does_not_exist"]})
    with pytest.raises(KeyError):
        load_account_settings("bad", cfg)

    with pytest.raises(FileNotFoundError):
        load_account_settings("nobody", cfg)


def test_upstox_provider_requires_its_named_token_env_var(tmp_path, monkeypatch):
    import pytest
    from trader.data.upstox import UpstoxProvider
    monkeypatch.delenv("UPSTOX_ACCESS_TOKEN_SOMEONE", raising=False)
    with pytest.raises(KeyError):
        UpstoxProvider({}, "Asia/Kolkata", {}, tmp_path, token_env_var="UPSTOX_ACCESS_TOKEN_SOMEONE")
    monkeypatch.setenv("UPSTOX_ACCESS_TOKEN_SOMEONE", "dummy")
    UpstoxProvider({}, "Asia/Kolkata", {}, tmp_path, token_env_var="UPSTOX_ACCESS_TOKEN_SOMEONE")  # no raise


def test_apply_overrides_sets_nested_path_without_mutating_baseline():
    from trader.backtest.sweep import apply_overrides
    s = load_settings()
    original_mode = s.filters["quality"]["mode"]
    original_enabled = s.filters["market_regime"]["enabled"]

    changed = apply_overrides(s, {"filters.quality.mode": "filter",
                                  "strategies.weekly_rsi_sma20.entry.sustain_weeks": 3})
    assert changed.filters["quality"]["mode"] == "filter"
    assert changed.strategies["weekly_rsi_sma20"]["entry"]["sustain_weeks"] == 3
    # baseline Settings object must be completely untouched
    assert s.filters["quality"]["mode"] == original_mode
    assert s.filters["market_regime"]["enabled"] == original_enabled
    assert "sustain_weeks" not in s.strategies["weekly_rsi_sma20"]["entry"] or \
           s.strategies["weekly_rsi_sma20"]["entry"]["sustain_weeks"] != 3


def test_apply_overrides_two_calls_from_same_baseline_dont_see_each_other():
    from trader.backtest.sweep import apply_overrides
    s = load_settings()
    a = apply_overrides(s, {"filters.resistance.enabled": True})
    b = apply_overrides(s, {"filters.resistance.enabled": False})
    assert a.filters["resistance"]["enabled"] is True
    assert b.filters["resistance"]["enabled"] is False   # proves 'a's change didn't leak into 'b'


def test_apply_overrides_rejects_unknown_root():
    from trader.backtest.sweep import apply_overrides
    import pytest
    s = load_settings()
    with pytest.raises(ValueError):
        apply_overrides(s, {"not_a_real_root.foo": 1})


def test_expand_one_at_a_time_is_linear_not_combinatorial():
    from trader.backtest.sweep import expand_one_at_a_time
    space = {"filters.quality.mode": ["rank", "filter"],
            "filters.market_regime.enabled": [False, True],
            "strategies.weekly_rsi_sma20.entry.sustain_weeks": [1, 2, 3]}
    combos = expand_one_at_a_time(space)
    assert len(combos) == 2 + 2 + 3                       # sum, not product
    assert all(len(c) == 1 for c in combos)                # exactly one knob changed per combo


def test_expand_grid_is_full_cartesian_product():
    from trader.backtest.sweep import expand_grid
    space = {"filters.quality.mode": ["rank", "filter"], "filters.market_regime.enabled": [False, True]}
    combos = expand_grid(space)
    assert len(combos) == 4                                # 2 x 2
    assert all(len(c) == 2 for c in combos)                # every combo sets BOTH knobs


def test_run_sweep_end_to_end_produces_one_row_per_combo_plus_baseline(data, tmp_path):
    from trader.backtest.sweep import run_sweep
    s = load_settings()
    s.universe["instruments"] = {k: k for k in data}
    s.universe["lists"]["default"] = list(data)
    space = {"filters.quality.mode": ["rank", "filter"]}
    df = run_sweep(s, data, date(2020, 1, 1), date(2021, 12, 31), space, tmp_path / "sweep",
                   mode="one_at_a_time", mc_sims=200, mc_seed=1)
    assert len(df) == 1 + 2                                # baseline + 2 values for the one knob
    assert "baseline" in df["combo"].tolist()
    assert "closed_trades" in df.columns and "mc_prob_loss_pct" in df.columns
    assert (tmp_path / "sweep" / "sweep_results.csv").exists()


def test_supertrend_flips_direction_from_up_to_down():
    from trader.core.indicators import supertrend
    idx = pd.bdate_range("2020-01-01", periods=120)
    # clean, low-noise uptrend for 80 bars, then a clean, sustained downtrend
    up = 100 + np.arange(80) * 1.5
    down = up[-1] - np.arange(40) * 2.0
    close = pd.Series(np.concatenate([up, down]), index=idx)
    high, low = close * 1.004, close * 0.996
    line, direction = supertrend(high, low, close, period=10, multiplier=3.0)
    assert direction.iloc[30] == 1                 # solidly mid-uptrend
    assert direction.iloc[-1] == -1                 # solidly into the downtrend by the end
    assert not line.iloc[30:].isna().any()           # warmed up well before either check point


def test_supertrend_warmup_is_nan_not_an_error():
    from trader.core.indicators import supertrend
    idx = pd.bdate_range("2020-01-01", periods=5)
    close = pd.Series([100.0, 101, 99, 102, 100], index=idx)
    line, direction = supertrend(close * 1.01, close * 0.99, close, period=10, multiplier=3.0)
    assert line.isna().all() and (direction == 0).all()   # fewer bars than the ATR period


def test_exit_method_rejects_invalid_value():
    import pytest
    cfg = {**S.strategies["weekly_rsi_sma20"],
          "exit": {"buffer_pct": 0.0, "trigger": "wick", "method": "not_a_real_method"}}
    with pytest.raises(ValueError):
        build_strategy("weekly_rsi_sma20", cfg)


def _strat_method(method: str):
    cfg = {**S.strategies["weekly_rsi_sma20"],
          "exit": {"buffer_pct": 0.0, "trigger": "wick", "method": method,
                   "supertrend": {"period": 10, "multiplier": 3.0}}}
    return build_strategy("weekly_rsi_sma20", cfg)


def test_supertrend_exit_fires_on_downtrend():
    from trader.strategies.base import SymbolContext
    idx = pd.bdate_range(end="2026-05-15", periods=300)   # ends on a Friday
    up = 100 + np.arange(len(idx) - 40) * 0.6
    down = up[-1] - np.arange(40) * 2.5                     # sharp, sustained drop into the end
    close = pd.Series(np.concatenate([up, down]), index=idx)
    df = pd.DataFrame({"open": close, "high": close * 1.004, "low": close * 0.996,
                       "close": close, "volume": 1000})
    cal = TradingCalendar(set())
    ctx = SymbolContext("X", df, idx[-1].date(), cal)
    pos = Position("X", "weekly_rsi_sma20", [Lot(10, close.iloc[-60], "2024-01-05")],
                   stop=0.0, last_price=100.0, opened="2024-01-05")

    st_sig = _strat_method("supertrend").exit_signal(ctx, pos)
    assert st_sig is not None and "Supertrend" in st_sig.reason
    level, direction = _strat_method("supertrend")._supertrend_last(ctx.weekly("W-FRI", False))
    assert direction == -1


def test_stop_level_reflects_active_exit_method():
    from trader.strategies.base import SymbolContext
    idx = pd.bdate_range(end="2026-05-15", periods=300)
    close = pd.Series(np.linspace(100, 300, len(idx)), index=idx)
    df = pd.DataFrame({"open": close, "high": close * 1.004, "low": close * 0.996,
                       "close": close, "volume": 1000})
    ctx = SymbolContext("X", df, idx[-1].date(), TradingCalendar(set()))
    sma_stop = _strat_method("sma20").stop_level(ctx)
    st_stop = _strat_method("supertrend").stop_level(ctx)
    assert sma_stop is not None and st_stop is not None
    assert sma_stop != st_stop                       # two different indicators, genuinely different levels


def test_backtest_runs_with_supertrend_exit_and_differs_from_sma20(data, tmp_path):
    from trader.backtest.runner import run_backtest
    s = load_settings()
    s.universe["instruments"] = {k: k for k in data}
    s.universe["lists"]["default"] = list(data)
    s.strategies["weekly_rsi_sma20"]["exit"] = {
        "buffer_pct": 0.0, "trigger": "wick", "method": "supertrend",
        "supertrend": {"period": 10, "multiplier": 3.0},
    }
    eq, trades = run_backtest(s, data, date(2020, 1, 1), date(2021, 12, 31), str(tmp_path))
    sells = [t for t in trades if t["side"] == "SELL"]
    assert len(sells) > 0
    assert all("Supertrend" in t["reason"] for t in sells)