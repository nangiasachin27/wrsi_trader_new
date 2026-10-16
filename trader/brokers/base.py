from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

from trader.core.models import Fill


class Broker(ABC):
    @abstractmethod
    def execute(self, side: str, symbol: str, qty: int, ref_price: float, on: date) -> Fill: ...
