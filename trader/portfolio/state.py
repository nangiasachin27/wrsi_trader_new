from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from trader.core.models import PendingOrder, Position


@dataclass
class Portfolio:
    cash: float
    positions: dict = field(default_factory=dict)
    trades: list = field(default_factory=list)

    def positions_value(self) -> float:
        return sum(p.qty * p.last_price for p in self.positions.values())

    def equity(self) -> float:
        return self.cash + self.positions_value()


class StateStore:
    """JSON/CSV persistence committed back to git by the workflows (same pattern as T_Raider)."""

    def __init__(self, state_dir: Path | str):
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pf_path = self.dir / "portfolio.json"
        self.pending_path = self.dir / "pending_orders.json"
        self.equity_path = self.dir / "daily_equity.csv"
        self.journal_path = self.dir / "journal.jsonl"

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    def load_portfolio(self, initial_cash: float) -> Portfolio:
        if not self.pf_path.exists():
            return Portfolio(cash=initial_cash)
        d = json.loads(self.pf_path.read_text(encoding="utf-8"))
        return Portfolio(cash=d["cash"], trades=d["trades"],
                         positions={k: Position.from_dict(v) for k, v in d["positions"].items()})

    def save_portfolio(self, pf: Portfolio) -> None:
        d = {"cash": round(pf.cash, 2), "trades": pf.trades,
             "positions": {k: p.to_dict() for k, p in pf.positions.items()}}
        self._atomic_write(self.pf_path, json.dumps(d, indent=2))

    def load_pending(self) -> list[PendingOrder]:
        if not self.pending_path.exists():
            return []
        return [PendingOrder(**o) for o in json.loads(self.pending_path.read_text(encoding="utf-8"))]

    def save_pending(self, orders: list[PendingOrder]) -> None:
        self._atomic_write(self.pending_path, json.dumps([o.to_dict() for o in orders], indent=2))

    def append_equity(self, asof: date, pf: Portfolio) -> None:
        row = f"{asof.isoformat()},{pf.cash:.2f},{pf.positions_value():.2f},{pf.equity():.2f},{len(pf.positions)}\n"
        new = not self.equity_path.exists()
        lines = [] if new else self.equity_path.read_text().splitlines(keepends=True)
        lines = [l for l in lines if not l.startswith(asof.isoformat())]  # idempotent re-runs
        if new or not lines:
            lines = ["date,cash,positions_value,equity,n_positions\n"]
        self._atomic_write(self.equity_path, "".join(lines) + row)

    def equity_frame(self) -> pd.DataFrame:
        return pd.read_csv(self.equity_path, parse_dates=["date"], index_col="date")

    def journal(self, **rec) -> None:
        with open(self.journal_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
