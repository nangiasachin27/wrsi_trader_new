import csv
import logging
from datetime import date

import pytest

from trader.core.models import Signal
from trader.filters.base import build_filter

FIELDS = ["date", "symbol", "decision_time", "score", "weight", "n", "n_neg", "n_pos", "date_only_share", "negative"]


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({"decision_time": "15:15", "date_only_share": 0, "negative": "", **r})


NEG = dict(score=-0.6, weight=2.0, n=3, n_neg=3, n_pos=0)
NONE = dict(score="", weight=0.0, n=0, n_neg=0, n_pos=0)


@pytest.fixture
def gate(tmp_path):
    _write(tmp_path / "g" / "a.csv", [
        dict(date="2026-03-02", symbol="AAA", **NEG),
        dict(date="2026-03-03", symbol="AAA", **NONE),
        dict(date="2026-03-03", symbol="BBB", **NEG),
        dict(date="2026-03-03", symbol="CCC", score=-0.6, weight=0.5, n=1, n_neg=1, n_pos=0),   # too light
        dict(date="2026-03-03", symbol="DDD", score=-0.3, weight=2.0, n=4, n_neg=2, n_pos=2),   # not more negative
        dict(date="2026-03-03", symbol="M&M", **NEG),
    ])

    def make(**over):
        f = build_filter("sentiment", {"enabled": True, "mode": "filter", "path": str(tmp_path / "g"), **over})
        return f

    return make


def _eval(f, d, sym):
    f.prepare(d, None)
    return f.evaluate(Signal(sym, "s", "ENTRY", d.isoformat(), ""), None)


def test_rule_and_coverage(gate):
    f, tue = gate(), date(2026, 3, 3)
    assert not _eval(f, tue, "BBB").passed
    assert not _eval(f, tue, "m&m").passed                       # symbols compared upper-case
    r = _eval(f, tue, "AAA")                                     # flagged yesterday, quiet today
    assert r.passed and r.meta["coverage"] == "no_news"
    assert _eval(f, tue, "CCC").passed and _eval(f, tue, "DDD").passed
    r = _eval(f, tue, "ZZZ")                                     # never scraped
    assert r.passed and r.meta["coverage"] == "missing"
    assert not _eval(gate(fail_mode="block"), tue, "ZZZ").passed
    assert f.max_score() == 0.0                                  # pure gate, never ranks


def test_lookback_and_shadow(gate, caplog):
    tue = date(2026, 3, 3)
    assert not _eval(gate(lookback_days=2), tue, "AAA").passed   # Monday's flag still counts
    with caplog.at_level(logging.INFO, logger="trader.filters.sentiment"):
        r = _eval(gate(mode="shadow"), tue, "BBB")
    assert r.passed and r.meta["would_block"] and "SHADOW" in caplog.text


def test_day_without_any_rows_fails_open(gate, caplog):
    with caplog.at_level(logging.WARNING, logger="trader.filters.sentiment"):
        r = _eval(gate(), date(2026, 3, 4), "BBB")
    assert r.passed and r.meta["coverage"] == "missing" and "no sentiment rows" in caplog.text


def test_later_file_overrides_earlier(gate, tmp_path):
    _write(tmp_path / "g" / "b.csv", [dict(date="2026-03-03", symbol="BBB", **NONE)])
    assert _eval(gate(), date(2026, 3, 3), "BBB").passed


def test_disabled_and_unprepared(gate):
    assert build_filter("sentiment", {"enabled": False}).evaluate(Signal("X", "s", "ENTRY", "", ""), None).passed
    assert not gate().evaluate(Signal("X", "s", "ENTRY", "", ""), None).passed


def test_backtest_blocks_flagged_entries(data, tmp_path):
    from trader.backtest.runner import run_backtest
    from trader.core.config import load_settings

    from conftest import synth

    data = {**data, "NIFTY500": synth(99)}
    syms = [k for k in data if k != "NIFTY500"]

    def run(sent_cfg, tag):
        s = load_settings()
        s.universe["instruments"] = {k: k for k in data}
        s.universe["lists"]["default"] = syms
        for c in s.filters.values():
            c["enabled"] = False
        s.filters["sentiment"] = sent_cfg
        eq, trades = run_backtest(s, data, date(2020, 1, 1), date(2021, 12, 31), str(tmp_path / tag))
        return [t for t in trades if t["side"] == "BUY" and t.get("action") == "ENTRY"]

    base = run({"enabled": False}, "base")
    assert base, "baseline needs entries for this test to mean anything"
    # flag the signal day of the first baseline entry: that entry must disappear
    first = base[0]
    days = sorted({ts.date() for ts in data[first["symbol"]].index})
    sig_day = max(d for d in days if d < date.fromisoformat(first["date"]))
    _write(tmp_path / "gate" / "s.csv", [dict(date=sig_day.isoformat(), symbol=first["symbol"], **NEG)])
    gated = run({"enabled": True, "mode": "filter", "path": str(tmp_path / "gate")}, "gated")
    assert not any(t["symbol"] == first["symbol"] and t["date"] == first["date"] for t in gated)
    shadow = run({"enabled": True, "mode": "shadow", "path": str(tmp_path / "gate")}, "shadow")
    assert [(t["symbol"], t["date"]) for t in shadow] == [(t["symbol"], t["date"]) for t in base]
