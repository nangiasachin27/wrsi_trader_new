from __future__ import annotations

from datetime import date

from trader.brokers.base import Broker
from trader.core.models import Fill


class UpstoxLiveBroker(Broker):
    """Deliberately unimplemented until paper trading has run its course. Before enabling:
    idempotent order tags, order-status polling + partial fills, broker-side position/cash
    reconciliation against portfolio.json, a kill-switch, and per-day loss limits."""

    def execute(self, side: str, symbol: str, qty: int, ref_price: float, on: date) -> Fill:
        raise NotImplementedError("Live execution not implemented (paper phase).")
