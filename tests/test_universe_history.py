"""Point-in-time Nifty 500 membership (survivorship-bias fix) and its builder's CSV parsing."""
from __future__ import annotations

import copy
import json
import sys
from datetime import date

import numpy as np
import pandas as pd

from trader.backtest.runner import run_backtest
from trader.core.config import ROOT, load_settings
from trader.core.models import Signal
from trader.core.universe_history import UniverseHistory
from trader.risk.market_breadth import breadth_frame
from trader.strategies.base import Strategy, register as register_strategy

sys.path.insert(0, str(ROOT / "scripts"))
from build_universe_history import parse_constituents  # noqa: E402

S = load_settings()
INST = {"OLDCO": "NSE_EQ|INE000000001", "NEWCO": "NSE_EQ|INE000000002", "STAY": "NSE_EQ|INE000000003",
        "NIFTY500": "NSE_INDEX|Nifty 500"}


def _hist(tmp_path):
    p = tmp_path / "uh.json"
    p.write_text(json.dumps({"snapshots": [
        {"date": "2024-01-01", "isins": ["INE000000001", "INE000000003", "INE999999999"]},   # OLDCO in
        {"date": "2024-03-01", "isins": ["INE000000002", "INE000000003"]},                   # swapped
    ]}))
    return UniverseHistory(p, INST)


def test_membership_by_date_and_isin(tmp_path):
    h = _hist(tmp_path)
    assert h.members(date(2024, 2, 15)) == {"OLDCO", "STAY"}
    assert h.members(date(2024, 3, 1)) == {"NEWCO", "STAY"}       # effective on the snapshot date
    assert h.members(date(2023, 6, 1)) == {"OLDCO", "STAY"}       # before the first: first is used
    assert h.unmapped == {"INE999999999"}                         # no instrument: reported, not traded
    assert h.all_symbols == ["NEWCO", "OLDCO", "STAY"]


def test_breadth_counts_members_only(tmp_path):
    h = _hist(tmp_path)
    idx = pd.bdate_range("2024-01-01", "2024-04-30")
    up = pd.DataFrame({"close": np.linspace(100, 150, len(idx))}, index=idx)
    down = pd.DataFrame({"close": np.linspace(150, 100, len(idx))}, index=idx)
    frames = {"OLDCO": down, "NEWCO": up, "STAY": up}
    b = breadth_frame(frames, list(frames), 5, h.membership_frame)
    assert b.loc["2024-02-15", "n_valid"] == 2 and b.loc["2024-02-15", "pct_above"] == 50.0   # OLDCO+STAY
    assert b.loc["2024-04-15", "n_valid"] == 2 and b.loc["2024-04-15", "pct_above"] == 100.0  # NEWCO+STAY


@register_strategy("always_buy_pit")
class AlwaysBuy(Strategy):
    def entry_signal(self, ctx):
        c = float(ctx.daily["close"].iloc[-1])
        return Signal(ctx.symbol, self.name, "ENTRY", ctx.asof.isoformat(), "test", 1.0, c * 0.9, c)
    def exit_signal(self, ctx, pos): return None
    def add_signal(self, ctx, pos): return None
    def stop_level(self, ctx, pos): return pos.stop


def test_backtest_enters_only_members_of_the_day(tmp_path, monkeypatch):
    h = _hist(tmp_path)
    s = copy.deepcopy(S)
    s.universe["instruments"] = dict(INST)
    s.universe["lists"]["pit"] = ["OLDCO", "NEWCO", "STAY"]
    s.strategies = {"always_buy_pit": {"enabled": True, "universe": "pit", "exclude_filters": []}}
    s.filters = {}
    s.risk["capital"]["allocation_pct"]["always_buy_pit"] = 100
    s.risk["capital"]["mode"] = "fixed"
    s.risk["portfolio"]["regime_gross_exposure"]["enabled"] = False
    s.system["universe_history"] = {"enabled": True, "path": str(tmp_path / "uh.json"), "scan_list": "pit"}
    idx = pd.bdate_range("2024-01-01", "2024-04-30")
    c = np.linspace(100, 120, len(idx))
    bar = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e6}, index=idx)
    data = {k: bar.copy() for k in ("OLDCO", "NEWCO", "STAY", "NIFTY500")}
    _, trades = run_backtest(s, data, date(2024, 3, 4), date(2024, 3, 8), str(tmp_path / "bt"))
    bought = {t["symbol"] for t in trades if t["side"] == "BUY"}
    assert bought == {"NEWCO", "STAY"}                          # OLDCO left the index on 2024-03-01


def test_parse_constituents_accepts_nse_csv_and_rejects_junk():
    rows = "\n".join(f"Co {i} Ltd.,Industry,SYM{i},EQ,INE{str(i).zfill(9)}" for i in range(1, 501))
    df = parse_constituents("Company Name,Industry,Symbol,Series,ISIN Code\n" + rows)
    assert df is not None and len(df) == 500 and df.iloc[0]["isin"] == "INE000000001"
    assert parse_constituents("<html>Wayback Machine error</html>") is None
    assert parse_constituents("Company Name,Industry,Symbol,Series,ISIN Code\nA,B,C,EQ,INE000000001") is None
