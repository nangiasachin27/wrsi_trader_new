from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class DataProvider(ABC):
    @abstractmethod
    def get_daily(self, symbol: str, end: date) -> pd.DataFrame:
        """Daily OHLCV, DatetimeIndex, columns open/high/low/close/volume, rows <= end."""

    @abstractmethod
    def get_open_quotes(self, symbols: list[str], on: date) -> dict[str, float]:
        """Today's official open per symbol; symbols without an open yet are omitted."""
