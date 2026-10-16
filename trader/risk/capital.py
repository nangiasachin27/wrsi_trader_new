from __future__ import annotations


class CapitalManager:
    """Base capital -> per-strategy base -> risk amount per trade.
    mode 'fixed': risk is a % of the configured base (does not compound).
    mode 'equity': base tracks current equity (compounds)."""

    def __init__(self, risk_cfg: dict):
        self.cap = risk_cfg["capital"]
        self.risk_pct = risk_cfg["per_trade"]["risk_pct_of_base"]

    def initial_cash(self) -> float:
        return float(self.cap["base_capital"])

    def strategy_base(self, strategy: str, equity: float) -> float:
        root = self.cap["base_capital"] if self.cap["mode"] == "fixed" else equity
        return root * self.cap["allocation_pct"][strategy] / 100.0

    def risk_amount(self, strategy: str, equity: float) -> float:
        return self.strategy_base(strategy, equity) * self.risk_pct / 100.0
