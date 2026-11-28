"""Fail-loud live runs: TOTP generation, Upstox token preflight, and the strict data-health checks
that stop EOD/open from 'succeeding' on stale data (an expired token silently fell back to cached
bars for every symbol in Nov 2026)."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trader.brokers.paper import PaperBroker
from trader.core.calendar import TradingCalendar
from trader.core.config import ROOT, load_settings
from trader.core.models import PendingOrder
from trader.data.historical import HistoricalProvider
from trader.engine.orchestrator import DataHealthError, Engine
from trader.portfolio.state import StateStore
from trader.strategies.base import Strategy, register as register_strategy

sys.path.insert(0, str(ROOT / "scripts"))
from upstox_token import totp  # noqa: E402

S = load_settings()


@register_strategy("never_signals")
class Never(Strategy):
    def entry_signal(self, ctx): return None
    def exit_signal(self, ctx, pos): return None
    def add_signal(self, ctx, pos): return None
    def stop_level(self, ctx, pos): return None


def _bars(n, end):
    idx = pd.bdate_range(end=end, periods=n)
    c = np.linspace(100, 110, n)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e5}, index=idx)


def _engine(tmp_path, fresh: int, total: int = 10, day="2026-11-27"):
    s = copy.deepcopy(S)
    syms = [f"S{i}" for i in range(total)]
    s.universe["lists"]["t"] = syms
    s.strategies = {"never_signals": {"enabled": True, "universe": "t", "exclude_filters": []}}
    s.filters = {}
    s.risk["capital"]["allocation_pct"]["never_signals"] = 100
    s.risk["portfolio"]["regime_gross_exposure"]["enabled"] = False
    d = pd.Timestamp(day)
    data = {x: _bars(30, d if i < fresh else d - pd.Timedelta(days=7)) for i, x in enumerate(syms)}
    days = sorted({t.date() for f in data.values() for t in f.index})
    eng = Engine(s, HistoricalProvider(data), PaperBroker(s.fees, 0), StateStore(tmp_path),
                 TradingCalendar(set(), trading_days=days))
    return eng, d.date()


def test_totp_matches_rfc6238_vector():
    secret = "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ"          # base32("12345678901234567890")
    assert totp(secret, at=59) == "287082"                 # RFC 6238 T=59 -> 94287082
    assert totp(secret, at=1111111109) == "081804"         # RFC 6238 -> 07081804
    assert totp(secret.lower()[:16] + " " + secret.lower()[16:], at=59) == "287082"   # spaces/case


def test_strict_eod_fails_and_saves_nothing_on_stale_data(tmp_path):
    eng, d = _engine(tmp_path, fresh=3)                    # 30% fresh < 90%
    eng.strict_data = True
    with pytest.raises(DataHealthError, match="3/10 symbols"):
        eng.run_eod(d)
    assert not (tmp_path / "daily_equity.csv").exists()   # no stale equity row
    assert not (tmp_path / "portfolio.json").exists()


def test_strict_eod_passes_and_journals_health(tmp_path):
    eng, d = _engine(tmp_path, fresh=10)
    eng.strict_data = True
    eng.run_eod(d)
    j = [json.loads(l) for l in (tmp_path / "journal.jsonl").read_text().splitlines()]
    h = [r for r in j if r.get("event") == "DATA_HEALTH"]
    assert h and h[0]["fresh"] == 10 and h[0]["fresh_pct"] == 100.0
    assert (tmp_path / "daily_equity.csv").exists()


def test_non_strict_backtest_behaviour_unchanged(tmp_path):
    eng, d = _engine(tmp_path, fresh=3)                    # strict_data defaults to False
    eng.run_eod(d)                                         # no raise, as before
    j = (tmp_path / "journal.jsonl").read_text() if (tmp_path / "journal.jsonl").exists() else ""
    assert "DATA_HEALTH" not in j


def test_strict_open_fails_without_quotes_and_keeps_orders(tmp_path):
    eng, d = _engine(tmp_path, fresh=10)
    eng.strict_data = True
    orders = [PendingOrder(f"S{i}", "never_signals", "ENTRY", "2026-11-26", d.isoformat(),
                           "test", 1.0, 95.0, 100.0, {}) for i in range(4)]
    eng.store.save_pending(orders)
    eng.provider.get_open_quotes = lambda syms, asof: {}   # quotes API down / token rejected
    with pytest.raises(DataHealthError, match="4/4 symbols"):
        eng.run_open(d)
    assert len(eng.store.load_pending()) == 4              # nothing executed, orders kept


def test_provider_rejects_missing_token(monkeypatch, tmp_path):
    from trader.data.upstox import TokenError, UpstoxProvider
    monkeypatch.delenv("UPSTOX_ACCESS_TOKEN", raising=False)
    with pytest.raises(TokenError, match="not set"):
        UpstoxProvider(S.system["data"], "Asia/Kolkata", S.universe["instruments"], tmp_path)


def test_check_token_raises_on_401(monkeypatch, tmp_path):
    from trader.data.upstox import TokenError, UpstoxProvider
    monkeypatch.setenv("UPSTOX_ACCESS_TOKEN", "expired")
    p = UpstoxProvider(S.system["data"], "Asia/Kolkata", S.universe["instruments"], tmp_path)

    class R:
        status_code, text = 401, '{"status":"error","errors":[{"message":"Invalid token"}]}'
    seen = {}
    def fake_get(url, *a, **k):
        seen["url"] = url
        return R()
    monkeypatch.setattr(p.session, "get", fake_get)
    with pytest.raises(TokenError, match="expired"):
        p.check_token()
    assert "/market-quote/" in seen["url"]        # market data: works with the Analytics Token


def test_cli_exits_2_without_token():
    env = {k: v for k, v in os.environ.items() if k != "UPSTOX_ACCESS_TOKEN"}
    r = subprocess.run([sys.executable, "-m", "trader", "eod", "--date", "2026-11-27", "--force"],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 2, r.stderr[-500:]
    assert "UPSTOX TOKEN" in r.stderr
