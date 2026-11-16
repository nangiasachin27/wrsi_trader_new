"""rank_weight (per-strategy slot-ranking multiplier) and the regime_exposure throttle,
exercised end-to-end through run_backtest on deterministic synthetic bars with two dummy
strategies and one fake scoring filter - no real market data, no real strategies."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trader.backtest.runner import run_backtest
from trader.core.config import load_settings
from trader.core.models import Signal
from trader.filters.base import Filter, FilterResult, register as register_filter
from trader.strategies.base import Strategy, register as register_strategy

S = load_settings()


@register_strategy("dummy_a")
@register_strategy("dummy_b")
class DummyStrategy(Strategy):
    """Signals ENTRY on its one symbol every day, stop 10% below close, quality from cfg."""

    def entry_signal(self, ctx):
        px = ctx.last_close
        return Signal(ctx.symbol, self.name, "ENTRY", ctx.asof.isoformat(), "test entry",
                      stop=px * 0.9, ref_price=px, meta={"q": self.cfg["q"]})

    def exit_signal(self, ctx, pos):
        return None

    def add_signal(self, ctx, pos):
        return None

    def stop_level(self, ctx, pos):
        return None


@register_filter("fake_quality")
class FakeQuality(Filter):
    def evaluate(self, signal, ctx):
        return FilterResult(True, "fake quality", score=float(signal.meta.get("q", 0.0)))

    def max_score(self) -> float:
        return 1.0 if self.enabled else 0.0


def bars(closes, start="2024-01-01"):
    c = np.asarray(closes, dtype=float)
    idx = pd.bdate_range(start, periods=len(c))
    return pd.DataFrame({"open": c, "high": c * 1.004, "low": c * 0.996, "close": c,
                         "volume": 1e6}, index=idx)


N = 90  # enough weekly benchmark bars for the throttle's SMA in every test


def make_data(bench="falling"):
    data = {"AAA": bars(np.full(N, 100.0)), "BBB": bars(np.full(N, 100.0))}
    data["BENCH"] = bars(np.linspace(200, 100, N) if bench == "falling" else np.linspace(100, 200, N))
    return data


def make_settings(max_positions=2, regime=None, rank_weight_b=1.0):
    s = copy.deepcopy(S)
    s.universe["lists"]["list_a"], s.universe["lists"]["list_b"] = ["AAA"], ["BBB"]
    s.strategies = {
        "dummy_a": {"enabled": True, "universe": "list_a", "exclude_filters": [], "q": 0.6},
        "dummy_b": {"enabled": True, "universe": "list_b", "exclude_filters": [], "q": 0.5,
                    "rank_weight": rank_weight_b},
    }
    s.filters = {"fake_quality": {"enabled": True}}
    if regime is not None:
        s.filters["regime_exposure"] = {"enabled": True, "benchmark_symbol": "BENCH",
                                        "timeframe_rule": "W-FRI", "sma_period": 4, "buffer_pct": 0.0,
                                        "min_bars": 5, "fail_mode": "allow", **regime}
    s.risk["portfolio"]["max_positions"] = max_positions
    s.risk["capital"]["allocation_pct"].update({"dummy_a": 100, "dummy_b": 100})
    return s


def run(s, data, tmp_path, days=3):
    idx = data["AAA"].index
    wd = tmp_path / "bt"
    eq, trades = run_backtest(s, data, idx[-days].date(), idx[-1].date(), str(wd))
    journal = [json.loads(l) for l in (wd / "journal.jsonl").read_text().splitlines()] \
        if (wd / "journal.jsonl").exists() else []
    return [t for t in trades if t["side"] == "BUY"], journal


# ---- rank_weight ------------------------------------------------------------------------------
def test_rank_weight_default_higher_score_wins(tmp_path):
    buys, _ = run(make_settings(max_positions=1), make_data(), tmp_path)
    assert buys and buys[0]["symbol"] == "AAA"          # q 0.6 beats q 0.5


def test_rank_weight_flips_slot_winner_and_fill_order(tmp_path):
    buys, _ = run(make_settings(max_positions=1, rank_weight_b=1.5), make_data(), tmp_path)
    assert buys and buys[0]["symbol"] == "BBB"          # 0.5 * 1.5 = 0.75 beats 0.6


# ---- regime_exposure throttle -----------------------------------------------------------------
def test_throttle_pct0_blocks_all_new_entries_and_journals(tmp_path):
    s = make_settings(regime={"bear_max_positions_pct": 0, "bear_risk_scale": 1.0})
    buys, journal = run(s, make_data("falling"), tmp_path)
    assert buys == []
    throttles = [r for r in journal if r.get("event") == "REGIME_THROTTLE"]
    assert throttles and throttles[0]["max_positions"] == 0


def test_throttle_scales_entry_risk_but_not_slots(tmp_path):
    base_s = make_settings()
    base_buys, _ = run(base_s, make_data("falling"), tmp_path / "a")
    s = make_settings(regime={"bear_max_positions_pct": 100, "bear_risk_scale": 0.5})
    buys, journal = run(s, make_data("falling"), tmp_path / "b")
    assert {b["symbol"] for b in buys} == {b["symbol"] for b in base_buys} == {"AAA", "BBB"}
    q0 = {b["symbol"]: b["qty"] for b in base_buys}
    q1 = {b["symbol"]: b["qty"] for b in buys}
    assert q1["AAA"] * 2 == q0["AAA"] and q1["BBB"] * 2 == q0["BBB"]
    assert any(r.get("event") == "REGIME_THROTTLE" and r["risk_scale"] == 0.5 for r in journal)


def test_bullish_benchmark_means_no_throttle(tmp_path):
    s = make_settings(regime={"bear_max_positions_pct": 0, "bear_risk_scale": 0.5})
    buys, journal = run(s, make_data("rising"), tmp_path)
    assert {b["symbol"] for b in buys} == {"AAA", "BBB"}
    assert not any(r.get("event") == "REGIME_THROTTLE" for r in journal)


def test_missing_benchmark_fails_open(tmp_path):
    s = make_settings(regime={"benchmark_symbol": "NOPE", "bear_max_positions_pct": 0})
    data = make_data()
    del data["BENCH"]
    buys, journal = run(s, data, tmp_path)
    assert {b["symbol"] for b in buys} == {"AAA", "BBB"}      # throttle off, no crash
    assert not any(r.get("event") == "REGIME_THROTTLE" for r in journal)


def test_shipped_defaults_are_inert():
    assert S.filters["regime_exposure"]["enabled"] is False
    assert S.strategies["weekly_rsi_sma20"]["rank_weight"] == 1.0
