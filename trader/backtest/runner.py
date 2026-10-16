from __future__ import annotations

import tempfile
from datetime import date

import numpy as np
import pandas as pd

from trader.brokers.paper import PaperBroker
from trader.core.calendar import TradingCalendar
from trader.core.config import Settings
from trader.data.historical import HistoricalProvider
from trader.engine.orchestrator import Engine
from trader.portfolio.state import StateStore


def run_backtest(settings: Settings, data: dict[str, pd.DataFrame], start: date, end: date,
                 work_dir: str | None = None):
    """Replays the exact live code path (Engine.run_open -> run_eod per day) over stored bars."""
    days = sorted({ts.date() for df in data.values() for ts in df.index})
    cal = TradingCalendar(set(), trading_days=days)
    store = StateStore(work_dir or tempfile.mkdtemp(prefix="bt_"))
    broker = PaperBroker(settings.fees, settings.system["execution"]["slippage_bps"])
    eng = Engine(settings, HistoricalProvider(data), broker, store, cal)
    for d in days:
        if start <= d <= end:
            eng.run_open(d)
            eng.run_eod(d)
    return store.equity_frame(), store.load_portfolio(0.0).trades


def summarize(eq: pd.DataFrame, trades: list[dict]) -> dict:
    e = eq["equity"]
    years = max((e.index[-1] - e.index[0]).days / 365.25, 1e-9)
    sells = [t for t in trades if t["side"] == "SELL"]
    wins = [t["pnl"] for t in sells if t["pnl"] > 0]
    losses = [-t["pnl"] for t in sells if t["pnl"] <= 0]
    holds = [t["holding_days"] for t in sells if t.get("holding_days") is not None]
    return {
        "cagr_pct": round(((e.iloc[-1] / e.iloc[0]) ** (1 / years) - 1) * 100, 2),
        "max_dd_pct": round(float(((e / e.cummax()) - 1).min() * 100), 2),
        "closed_trades": len(sells),
        "win_rate_pct": round(len(wins) / len(sells) * 100, 1) if sells else None,
        "profit_factor": round(sum(wins) / sum(losses), 2) if losses else None,
        "avg_holding_days": round(sum(holds) / len(holds), 1) if holds else None,
        "total_fees": round(sum(t["fees"] for t in trades), 0),
    }


def trade_log(trades: list[dict]) -> pd.DataFrame:
    """One row per closed round-trip: entry date, exit date, holding period, P&L. This is the
    trade-by-trade record for eyeballing whether exits are firing at sensible points, not just
    aggregate stats. Pyramided adds are folded into the position (avg_cost already blends them);
    'adds' shows how many times it was pyramided before the exit closed the whole position."""
    sells = [t for t in trades if t["side"] == "SELL"]
    if not sells:
        return pd.DataFrame(columns=["symbol", "strategy", "entry_date", "exit_date", "holding_days",
                                     "qty", "avg_cost", "exit_price", "pnl", "pnl_pct", "adds", "exit_reason"])
    rows = [{
        "symbol": t["symbol"], "strategy": t.get("strategy"), "entry_date": t.get("entry_date"),
        "exit_date": t["date"], "holding_days": t.get("holding_days"), "qty": t["qty"], "avg_cost": t["avg_cost"],
        "exit_price": t["price"], "pnl": t["pnl"],
        "pnl_pct": round(t["pnl"] / (t["avg_cost"] * t["qty"]) * 100, 2) if t["avg_cost"] * t["qty"] else None,
        "adds": t["adds"], "exit_reason": t["reason"],
    } for t in sells]
    return pd.DataFrame(rows).sort_values("exit_date").reset_index(drop=True)