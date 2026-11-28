"""Point-in-time index membership, so backtests trade only stocks that were in the Nifty 500 AT
THE TIME - not today's list projected back to 2021 (survivorship bias: winners that grew into
the index are in, losers that dropped out are missing).

config/universe_history.json (built by scripts/build_universe_history.py):
    {"snapshots": [{"date": "2021-03-31", "isins": ["INE...", ...]}, ...],
     "names": {"INE...": {"symbol": "ZOMATO", "company": "Zomato Ltd."}}, ...}

Each snapshot is the constituent list as captured on `date`; it is taken to hold until the next
snapshot. Membership is keyed by ISIN and mapped to today's trading symbol through
universe.json's instruments ("NSE_EQ|<ISIN>"), so symbol renames (ZOMATO -> ETERNAL) do not
break the history. Before the first snapshot, the first snapshot is used (logged once).
"""
from __future__ import annotations

import bisect
import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


class UniverseHistory:
    def __init__(self, path: Path | str, instruments: dict[str, str]):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        isin_to_sym = {v.split("|", 1)[1]: k for k, v in instruments.items() if v.startswith("NSE_EQ|")}
        snaps = sorted(raw.get("snapshots", []), key=lambda x: x["date"])
        if not snaps:
            raise ValueError(f"{path}: no snapshots")
        self.dates = [date.fromisoformat(x["date"]) for x in snaps]
        self.sets: list[frozenset[str]] = []
        self.unmapped: set[str] = set()
        for x in snaps:
            syms = set()
            for isin in x["isins"]:
                sym = isin_to_sym.get(isin)
                if sym:
                    syms.add(sym)
                else:
                    self.unmapped.add(isin)
            self.sets.append(frozenset(syms))
        self.all_symbols = sorted(set().union(*self.sets))
        self._warned_early = False
        if self.unmapped:
            log.warning("UNIVERSE HISTORY: %d historical ISINs have no instrument in universe.json "
                        "(not tradable in backtests - re-run scripts/build_universe_history.py)",
                        len(self.unmapped))

    def members(self, d: date) -> frozenset[str]:
        i = bisect.bisect_right(self.dates, d) - 1
        if i < 0:
            if not self._warned_early:
                log.warning("UNIVERSE HISTORY: %s is before the first snapshot (%s) - using it",
                            d, self.dates[0])
                self._warned_early = True
            i = 0
        return self.sets[i]

    def is_member(self, symbol: str, d: date) -> bool:
        return symbol in self.members(d)

    def membership_frame(self, index: pd.DatetimeIndex, symbols: list[str]) -> pd.DataFrame:
        """Boolean dates x symbols mask (for breadth): True where the symbol was a member."""
        out = {}
        pos = [max(bisect.bisect_right(self.dates, t.date()) - 1, 0) for t in index]
        for s in symbols:
            out[s] = [s in self.sets[p] for p in pos]
        return pd.DataFrame(out, index=index)
