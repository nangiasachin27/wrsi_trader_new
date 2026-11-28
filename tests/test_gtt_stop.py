"""GTT stop-loss simulation (system_config execution.stop_orders = "gtt")."""
from __future__ import annotations

import copy
import json
from datetime import date

import numpy as np
import pandas as pd

from trader.backtest.runner import run_backtest
from trader.core.config import load_settings
from trader.core.models import Signal
from trader.strategies.base import Strategy, register as register_strategy

S = load_settings()
STOP = 95.0


@register_strategy("buy_once_gtt")
class BuyOnce(Strategy):
    """Buys AAA on the first scan with a fixed stop; exits only on a CLOSE below the stop."""
    def entry_signal(self, ctx):
        if getattr(self, "_done", False):
            return None
        self._done = True
        return Signal("AAA", self.name, "ENTRY", ctx.asof.isoformat(), "test", 1.0, STOP, 100.0)
    def exit_signal(self, ctx, pos):
        px = float(ctx.daily["close"].iloc[-1])
        if px <= pos.stop:
            return Signal("AAA", self.name, "EXIT", ctx.asof.isoformat(), "close below stop", stop=pos.stop)
        return None
    def add_signal(self, ctx, pos): return None
    def stop_level(self, ctx, pos): return pos.stop


def _settings(mode):
    s = copy.deepcopy(S)
    s.universe["lists"]["one"] = ["AAA"]
    s.strategies = {"buy_once_gtt": {"enabled": True, "universe": "one", "exclude_filters": []}}
    s.filters = {}
    s.risk["capital"]["allocation_pct"]["buy_once_gtt"] = 100
    s.risk["capital"]["mode"] = "fixed"
    s.risk["portfolio"]["regime_gross_exposure"]["enabled"] = False
    s.system["universe_history"] = {"enabled": False}
    s.system["execution"] = {**s.system["execution"], "slippage_bps": 0, "stop_orders": mode}
    return s


def _data(path):
    """30 flat bars at 100, then the given (open, low, close) path."""
    idx = pd.bdate_range("2024-01-01", periods=30 + len(path))
    o = [100.0] * 30 + [p[0] for p in path]
    lo = [99.5] * 30 + [p[1] for p in path]
    c = [100.0] * 30 + [p[2] for p in path]
    hi = [max(a, b) + 0.5 for a, b in zip(o, c)]
    return {"AAA": pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": 1e6}, index=idx)}


def _run(mode, path, tmp_path):
    data = _data(path)
    idx = data["AAA"].index
    _, trades = run_backtest(_settings(mode), data, idx[29].date(), idx[-1].date(), str(tmp_path / mode))
    j = tmp_path / mode / "journal.jsonl"
    journal = [json.loads(l) for l in j.read_text().splitlines()] if j.exists() else []
    return [t for t in trades if t["side"] == "SELL"], journal


# entry signal on bar 29 -> bought at bar 30's open (100); then a dip that recovers by the close
DIP = [(100, 99, 100), (100, 94, 99), (99, 98, 99), (99, 98, 99)]


def test_gtt_sells_intraday_dip_that_close_based_rule_ignores(tmp_path):
    sells_close, _ = _run("close", DIP, tmp_path)
    sells_gtt, journal = _run("gtt", DIP, tmp_path)
    assert sells_close == []                                   # close 99 > 95: close rule holds on
    assert len(sells_gtt) == 1 and sells_gtt[0]["price"] == STOP
    assert "GTT stop" in sells_gtt[0]["reason"]
    assert any(r.get("event") == "GTT_STOP" for r in journal)


def test_gtt_gap_below_stop_fills_at_open(tmp_path):
    sells, _ = _run("gtt", [(100, 99, 100), (90, 88, 91), (91, 90, 91)], tmp_path)
    assert len(sells) == 1 and sells[0]["price"] == 90.0 and "gapped below" in sells[0]["reason"]


def test_gtt_triggers_on_the_purchase_session(tmp_path):
    sells, _ = _run("gtt", [(100, 94, 99), (99, 98, 99)], tmp_path)  # bought at 100, low 94 same day
    assert len(sells) == 1 and sells[0]["price"] == STOP


def test_close_mode_is_the_default():
    assert S.system["execution"].get("stop_orders", "close") == "close"
