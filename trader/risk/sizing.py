from __future__ import annotations

from trader.core.models import Position


def position_open_risk(pos: Position, stop: float) -> float:
    """Loss vs cost if the whole position exits at `stop` (floored at 0: profit-locked lots
    do not create negative risk)."""
    return max(0.0, sum(l.qty * (l.price - stop) for l in pos.lots))


def total_open_risk(positions: dict[str, Position]) -> float:
    return sum(position_open_risk(p, p.stop) for p in positions.values())


def _caps(qty: int, price: float, base: float, cash: float, held_value: float, cfg: dict) -> int:
    pt = cfg["per_trade"]
    qty = min(qty, int((base * pt["max_position_pct_of_base"] / 100.0 - held_value) // price))
    qty = min(qty, int(cash * (1 - pt["cash_buffer_pct"] / 100.0) // price))
    return max(qty, 0)


def size_entry(risk_amount: float, price: float, stop: float, base: float, cash: float, cfg: dict):
    pt = cfg["per_trade"]
    dist = max(price - stop, price * pt["min_stop_distance_pct"] / 100.0)
    qty = _caps(int(risk_amount // dist), price, base, cash, 0.0, cfg)
    if qty * price < pt["min_notional"]:
        return 0, "below min notional / caps"
    return qty, "ok"


def size_add(pos: Position, price: float, stop: float, base: float, cash: float, cfg: dict):
    """Pyramid add: total position risk (vs cost, at the new stop) is capped, so as the stop
    rises toward/above earlier lots, that freed-up risk budget funds the add."""
    pt = cfg["per_trade"]
    budget = base * cfg["pyramiding"]["max_position_risk_pct_of_base"] / 100.0 - position_open_risk(pos, stop)
    if budget <= 0:
        return 0, "no risk budget left"
    dist = max(price - stop, price * pt["min_stop_distance_pct"] / 100.0)
    qty = _caps(int(budget // dist), price, base, cash, pos.qty * price, cfg)
    if qty * price < pt["min_notional"]:
        return 0, "below min notional / caps"
    return qty, "ok"
