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
    it only adds to the candidate's score when the sector IS bullish, same contract as
    quality/trend_strength's scoring. In rank mode a stock in a weak sector can still take a
    free slot, and can still outrank a sector-bullish-but-otherwise-weaker candidate on
    quality/trend_strength score or wRSI - the sector check becomes a tiebreak nudge rather
    than a trapdoor that can reject an individually strong setup outright.

    Scoring: went through TWO iterations on 2026-10-27, both fixing the same real regression.
    It was a flat 0/1 on `bullish` originally. rs_rank/trend_strength moved to continuous
    scoring on 2026-10-27, which meant a strategy using THOSE filters almost never scored an
    exact 1.0 any more. pullback_ma's only scoring filter is this one - so while every other
    strategy's typical score sat below 1.0, pullback_ma kept hitting a clean, binary 1.0
    whenever its sector was simply bullish at all, and WON almost every slot contest as a
    direct result: measured on the user's own 2026-10-27 backtest, pullback_ma's trade count
    exploded from 74 (previous, bucketed-scoring run) to 591 (62% of ALL trades across 5
    strategies), with the worst win rate (23.4%) and PF (1.26) of any strategy, dragging
    portfolio CAGR 31.07%->28.22%, maxDD -14.74%->-21.69%, and total fees up ~46%.

    FIRST fix attempt (abandoned the same day): used the raw benchmark distance
    (`RegimeStatus.benchmark_close`/`benchmark_sma`, already computed by
    `trader.risk.regime.RegimeFilter.status()` and previously thrown away) scaled against a
    fixed `score_ceiling_pct`. This did NOT work - verified with a real 3-month backtest
    (2021-01 to 2021-03, a strongly trending window): most sector benchmarks were comfortably
    past that fixed ceiling for the whole window, so pullback_ma kept scoring a near-1.0
    anyway. A fixed absolute ceiling is regime-dependent (too easy to clear in a strong bull
    market, too hard in a weak one) - it just moves the saturation point, it doesn't remove it.

    SECOND fix (current): a CROSS-SECTIONAL PERCENTILE of each industry's own distance-from-
    SMA against every OTHER industry's distance that same day (computed once in prepare(),
    same construction as rs_rank.py's `_percentile_by_symbol`) - self-calibrating to whatever
    today's actual distribution looks like, so only the industries that are relatively
    strongest TODAY get a percentile near 1.0, regardless of whether the whole market is
    roaring or crawling. The pass/fail GATE (`mode: filter`, `buffer_pct`) is untouched -
    still keyed off `bullish` exactly as before, only the ranking score changed.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._engine = RegimeFilter(cfg)
        self._sector_map: dict[str, str] | None = None
        self._status_by_industry: dict[str, RegimeStatus] = {}
        self._percentile_by_industry: dict[str, float] = {}

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
            self._percentile_by_industry = {}
            return
        self._status_by_industry = {}
        dist_by_industry: dict[str, float] = {}
        for industry, bench_symbol in self.cfg.get("industry_to_benchmark", {}).items():
            try:
                df = provider.get_daily(bench_symbol, asof)
                st = self._engine.status(df, asof, bench_symbol)
            except Exception as exc:  # noqa: BLE001 - one bad benchmark shouldn't break the rest
                fail_open = self.cfg.get("fail_mode", "block") == "allow"
                st = RegimeStatus(fail_open, f"benchmark '{bench_symbol}' fetch failed: {exc}")
            self._status_by_industry[industry] = st
            if st.benchmark_close is not None and st.benchmark_sma:
                dist_by_industry[industry] = (st.benchmark_close / st.benchmark_sma - 1.0) * 100.0
            log.info("FILTER sector_regime [%s] %s | %s", industry,
                     "BULLISH" if st.bullish else "NOT BULLISH", st.reason)

        # Cross-sectional percentile across today's industries with valid data - see the
        # class docstring's "SECOND fix" section for why this replaced a fixed ceiling.
        self._percentile_by_industry = {}
        if dist_by_industry:
            ranked = sorted(dist_by_industry.items(), key=lambda kv: kv[1])
            n = len(ranked)
            for i, (industry, _) in enumerate(ranked):
                self._percentile_by_industry[industry] = (i + 1) / n

    def evaluate(self, signal: Signal, ctx: SymbolContext) -> FilterResult:
        if not self.enabled:
            return FilterResult(True, "disabled")
        mode = self.cfg.get("mode", "filter")
        fail_open = self.cfg.get("fail_mode", "block") == "allow"

        industry = self._load_sector_map().get(signal.symbol)
        if industry is None:
            ok = fail_open
            passed = True if mode == "rank" else ok
            # No benchmark data at all for this symbol's industry - nothing continuous to
            # compute, fall back to the old flat 0/1 (matches fail_open, same as before).
            return FilterResult(passed, f"no sector mapping for {signal.symbol}", score=float(ok))

        st = self._status_by_industry.get(industry)
        if st is None:
            ok = fail_open
            passed = True if mode == "rank" else ok
            return FilterResult(passed, f"no benchmark configured for industry '{industry}'", score=float(ok))

        passed = True if mode == "rank" else st.bullish
        # Cross-sectional percentile looked up from prepare() - falls back to the old flat
        # 0/1 only if this industry had no valid benchmark_close/benchmark_sma that day (so
        # never made it into dist_by_industry/the percentile map at all).
        score = self._percentile_by_industry.get(industry, float(st.bullish))
        return FilterResult(passed, f"[{industry}] {st.reason} (percentile {score:.0%} of "
                            f"{len(self._percentile_by_industry)} sectors today)", score=score)

    def max_score(self) -> float:
        """Cross-sectional percentile, already bounded [0, 1] - 1.0 whenever enabled, 0
        otherwise. Second iteration as of 2026-10-27 (was a flat 0/1 bit, then briefly a
        fixed-ceiling ratio that didn't work - see the class docstring). See
        Filter.max_score()'s own docstring."""
        return 1.0 if self.enabled else 0.0