from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from trader.data.base import DataProvider


def load_csv_dir(path: Path | str) -> dict[str, pd.DataFrame]:
    out = {}
    for p in sorted(Path(path).glob("*.csv")):
        df = pd.read_csv(p, parse_dates=["date"], index_col="date").sort_index()
        out[p.stem] = df[["open", "high", "low", "close", "volume"]]
    return out


class HistoricalProvider(DataProvider):
    """Replays stored bars with strict no-lookahead slicing; used by the backtester."""

    def __init__(self, data: dict[str, pd.DataFrame]):
        self.data = data

    def get_daily(self, symbol: str, end: date) -> pd.DataFrame:
        return self.data[symbol].loc[: pd.Timestamp(end)]

    def get_open_quotes(self, symbols: list[str], on: date) -> dict[str, float]:
        ts, out = pd.Timestamp(on), {}
        for s in symbols:
            df = self.data[s]
            if ts in df.index:
                out[s] = float(df.at[ts, "open"])
        return out
