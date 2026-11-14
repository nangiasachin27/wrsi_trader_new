from __future__ import annotations

import copy

from trader.core.models import Position

# strategy config `sizing` key -> risk_config.json per_trade key it overrides
_SIZING_KEYS = {"risk_pct": "risk_pct_of_base", "max_position_pct": "max_position_pct_of_base",
                "min_stop_distance_pct": "min_stop_distance_pct", "min_notional": "min_notional",
                "cash_buffer_pct": "cash_buffer_pct"}


def effective_risk(risk_cfg: dict, strategy_cfg: dict) -> tuple[dict, str | None, bool]:
    """Per-strategy sizing override (added 2026-11-12 so vcp_breakout/high_tight_flag can size
    exactly per their spec's Part 3 without changing the shared risk_config.json the other
    strategies use). A strategy config with a `sizing` block overrides the listed per_trade keys,
    capital.mode (`capital_mode`), and whether the portfolio open-risk cap applies
    (`portfolio_open_risk_cap`). No `sizing` block -> (risk_cfg, None, True): unchanged behaviour.
    Returns (risk_cfg_to_use, capital_mode_or_None, apply_portfolio_open_risk_cap)."""
    sz = strategy_cfg.get("sizing")
    if not sz:
        return risk_cfg, None, True
    cfg = copy.deepcopy(risk_cfg)
    for k, rk in _SIZING_KEYS.items():
        if k in sz:
            cfg["per_trade"][rk] = sz[k]
    return cfg, sz.get("capital_mode"), bool(sz.get("portfolio_open_risk_cap", True))


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
    if dist <= 0:
        return 0, "stop at/above entry"
    qty = _caps(int(risk_amount // dist), price, base, cash, 0.0, cfg)
    if qty <= 0 or qty * price < pt["min_notional"]:   # qty<=0 only matters when min_notional is 0
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
