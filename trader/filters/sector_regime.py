from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

from trader.core.config import ROOT
from trader.core.models import Signal
from trader.filters.base import Filter, FilterResult, register
from trader.risk.regime import RegimeFilter, RegimeStatus
from trader.strategies.base import SymbolContext

log = logging.getLogger(__name__)


@register("sector_regime")
class SectorRegimeFilter(Filter):
    """Per-symbol version of market_regime: blocks an entry if the STOCK'S OWN sector
    benchmark isn't itself trending up, even when the broad market is fine. Needs two things
    populated with real data before enabling, same discipline as the equity universe and the
    market-regime benchmark - never hand-typed:
      1. config/sector_map.json - symbol -> industry label (built by scripts/build_universe.py
         from NSE's own 'Industry' column; written on every run, no separate flag needed).
      2. this file's 'industry_to_benchmark' map - industry label -> a benchmark symbol you've
         added to config/universe.json's 'instruments' (found via
         scripts/find_index_instrument.py --search "NIFTY BANK" etc.) and fetched history for.
    scripts/finish_sector_regime.py automates both of step 2's lookups (industry -> candidate
    benchmark, benchmark -> instrument_key), adds the result to universe.json, fetches its
    history, writes this map, and flips enabled to true - run it after build_universe.py has
    populated config/sector_map.json.
    A symbol whose industry has no mapped benchmark, or whose benchmark has no cached data,
    follows fail_mode (default 'block' - conservative until the mapping is actually filled in).

    mode 'filter' (default - matches this filter's original, pre-existing behavior) hard-blocks
    ENTRY whenever the stock's sector benchmark isn't bullish, full stop, regardless of how
    strong the individual candidate's own signal is. mode 'rank' NEVER blocks on this check -
    it only adds 1 to the candidate's score when the sector IS bullish (0 otherwise), same
    contract as quality/trend_strength's scoring. In rank mode a stock in a weak sector can
    still take a free slot, and can still outrank a sector-bullish-but-otherwise-weaker
    candidate on quality/trend_strength score or wRSI - the sector check becomes a tiebreak
    nudge rather than a trapdoor that can reject an individually strong setup outright.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._engine = RegimeFilter(cfg)
        self._sector_map: dict[str, str] | None = None
        self._status_by_industry: dict[str, RegimeStatus] = {}

    def describe(self) -> str:
        return f"mode={self.cfg.get('mode', 'filter')}, fail_mode={self.cfg.get('fail_mode')}, " \
               f"sma_period={self.cfg.get('sma_period')}, buffer_pct={self.cfg.get('buffer_pct')}"

    def _load_sector_map(self) -> dict[str, str]:
        if self._sector_map is None:
            p = ROOT / self.cfg.get("sector_map_path", "config/sector_map.json")
            self._sector_map = json.loads(p.read_text())["symbol_to_industry"] if p.exists() else {}
        return self._sector_map

    def prepare(self, asof: date, provider) -> None:
        if not self.enabled:
            self._status_by_industry = {}
            return
        self._status_by_industry = {}
        for industry, bench_symbol in self.cfg.get("industry_to_benchmark", {}).items():
            try:
                df = provider.get_daily(bench_symbol, asof)
                st = self._engine.status(df, asof, bench_symbol)
            except Exception as exc:  # noqa: BLE001 - one bad benchmark shouldn't break the rest
                fail_open = self.cfg.get("fail_mode", "block") == "allow"
                st = RegimeStatus(fail_open, f"benchmark '{bench_symbol}' fetch failed: {exc}")
            self._status_by_industry[industry] = st
            log.info("FILTER sector_regime [%s] %s | %s", industry,
                     "BULLISH" if st.bullish else "NOT BULLISH", st.reason)

    def evaluate(self, signal: Signal, ctx: SymbolContext) -> FilterResult:
        if not self.enabled:
            return FilterResult(True, "disabled")
        mode = self.cfg.get("mode", "filter")
        fail_open = self.cfg.get("fail_mode", "block") == "allow"

        industry = self._load_sector_map().get(signal.symbol)
        if industry is None:
            ok = fail_open
            passed = True if mode == "rank" else ok
            return FilterResult(passed, f"no sector mapping for {signal.symbol}", score=int(ok))

        st = self._status_by_industry.get(industry)
        if st is None:
            ok = fail_open
            passed = True if mode == "rank" else ok
            return FilterResult(passed, f"no benchmark configured for industry '{industry}'", score=int(ok))

        passed = True if mode == "rank" else st.bullish
        return FilterResult(passed, f"[{industry}] {st.reason}", score=int(st.bullish))