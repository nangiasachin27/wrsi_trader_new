from datetime import date

from trader.backtest.runner import run_backtest, summarize
from trader.core.config import load_settings


def test_backtest_smoke_and_invariants(data, tmp_path):
    s = load_settings()
    s.universe["instruments"] = {k: k for k in data}
    s.universe["lists"]["default"] = list(data)
    eq, trades = run_backtest(s, data, date(2020, 1, 1), date(2024, 12, 31), str(tmp_path))
    res = summarize(eq, trades)
    print(res)
    assert len(trades) > 0
    assert (eq["cash"] >= 0).all()
    assert (eq["n_positions"] <= s.risk["portfolio"]["max_positions"]).all()
    buys = [t for t in trades if t["side"] == "BUY"]
    # every entry respected the 1% risk budget (allowing for the slippage/min-stop floor)
    base, pct = s.risk["capital"]["base_capital"], s.risk["per_trade"]["risk_pct_of_base"]
    for t in buys:
        if t["action"] == "ENTRY":
            assert t["qty"] * max(t["price"] - t["stop"], 0) <= base * pct / 100 * 1.02
