from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Lot:
    qty: int
    price: float
    date: str


@dataclass
class Position:
    symbol: str
    strategy: str
    lots: list
    stop: float
    last_price: float
    fees_paid: float = 0.0
    adds: int = 0
    opened: str = ""

    @property
    def qty(self) -> int:
        return sum(l.qty for l in self.lots)

    @property
    def cost(self) -> float:
        return sum(l.qty * l.price for l in self.lots)

    @property
    def avg_cost(self) -> float:
        return self.cost / self.qty

    @property
    def last_buy_price(self) -> float:
        return self.lots[-1].price

    @property
    def last_buy_date(self) -> str:
        return self.lots[-1].date

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        d = dict(d)
        d["lots"] = [Lot(**l) for l in d["lots"]]
        return cls(**d)


@dataclass
class Signal:
    symbol: str
    strategy: str
    action: str  # ENTRY | ADD | EXIT
    asof: str
    reason: str
    score: float = 0.0
    stop: float | None = None
    ref_price: float | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class PendingOrder:
    symbol: str
    strategy: str
    action: str
    created: str
    exec_date: str
    reason: str
    score: float = 0.0
    stop: float | None = None
    ref_price: float | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Fill:
    symbol: str
    side: str
    qty: int
    price: float
    fees: float
    date: str
