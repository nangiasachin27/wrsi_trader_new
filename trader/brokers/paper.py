from __future__ import annotations

from datetime import date

from trader.brokers.base import Broker
from trader.core.models import Fill


def compute_fees(side: str, value: float, cfg: dict) -> float:
    brk = min(cfg["brokerage_flat"], value * cfg["brokerage_pct_cap"] / 100.0)
    stt = value * cfg["stt_pct"] / 100.0
    txn = value * cfg["exchange_txn_pct"] / 100.0
    sebi = value * cfg["sebi_pct"] / 100.0
    stamp = value * cfg["stamp_duty_pct_buy"] / 100.0 if side == "BUY" else 0.0
    gst = (brk + txn + sebi) * cfg["gst_pct"] / 100.0
    dp = cfg["dp_charge_sell"] if side == "SELL" else 0.0
    return brk + stt + txn + sebi + stamp + gst + dp


class PaperBroker(Broker):
    def __init__(self, fee_cfg: dict, slippage_bps: float):
        self.fee_cfg, self.slip = fee_cfg, slippage_bps

    def execute(self, side: str, symbol: str, qty: int, ref_price: float, on: date) -> Fill:
        adj = ref_price * self.slip / 1e4
        px = ref_price + adj if side == "BUY" else ref_price - adj
        return Fill(symbol, side, qty, round(px, 2), round(compute_fees(side, qty * px, self.fee_cfg), 2), on.isoformat())
