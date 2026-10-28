from __future__ import annotations

import json
import logging
from datetime import date

import pandas as pd

from trader.core.config import ROOT
from trader.core.models import Signal
from trader.filters.base import Filter, FilterResult, register
from trader.strategies.base import SymbolContext

log = logging.getLogger(__name__)


def _weekly_return(daily: pd.DataFrame, asof: date, weekly_rule: str, lookback_weeks: int) -> float | None:
    """Trailing N-week close-to-close return, using the same Friday-anchored weekly
    resample convention as the strategy's own weekly signals, as of the most
    recently completed week on or before `asof`."""
    if daily is None or daily.empty:
        return None
    idx = daily.index
    d = daily[idx.date <= asof] if hasattr(idx, "date") else daily[idx <= pd.Timestamp(asof)]
    if d.empty:
        return None
    weekly_close = d["close"].resample(weekly_rule).last().dropna()
    if len(weekly_close) <= lookback_weeks:
        return None
    end = weekly_close.iloc[-1]
    start = weekly_close.iloc[-1 - lookback_weeks]
    if start is None or start <= 0:
        return None
    return (end / start) - 1.0


@register("rs_rank")
class RelativeStrengthFilter(Filter):
    """Ranks today's qualifying candidates by relative strength - trailing N-week
    return vs a market benchmark - instead of any single stock's own absolute
    trend indicators (that's what trend_strength does, and empirically it hasn't
    been a good tiebreaker: ADX/SMA50 largely re-asks a question the core
    weekly-RSI>60 + monthly-RSI>60 entry rule already answered).

    Intent: when max_positions forces a choice between several candidates that
    already pass quality/resistance/the core rule, prefer whichever one has been
    outperforming the market the most over the recent past - closer to an IBD/
    Minervini-style RS rating than a trend-presence check. This is a relative,
    cross-sectional measure, so unlike every other filter here it needs the WHOLE
    universe's returns on the same day to produce a meaningful percentile, not
    just the one candidate being evaluated - that's why prepare() loops the full
    universe instead of a small industry_to_benchmark map.

    Needs:
      1. config/sector_map.json's symbol_to_industry keys as the universe list -
         reusing this (already populated for sector_regime) instead of
         re-parsing config/universe.json's own schema.
      2. a benchmark with cached daily history (NIFTY500 by default, same
         fallback sector_regime's finish script uses for unmapped industries).

    mode 'rank' (default) never excludes a candidate - RS percentile only adds to
    score for tie-breaking when slots are scarce, same contract as
    trend_strength/sector_regime in rank mode. mode 'filter' hard-blocks any
    candidate below min_percentile.

    Score: the candidate's own RS percentile (0, 1.0], used directly - no bucketing.
    This was a fixed 0/1/2 three-value bucket until 2026-10-27: with that few possible
    values, any day with more than a handful of candidates produced large tied groups at
    the max score, and ties were then broken by an arbitrary, undocumented ordering
    (strategy-config filename, then symbol, alphabetically - see orchestrator.py's sort
    key) rather than by anything resembling quality. The percentile itself is already a
    clean continuous (0, 1.0] value (computed in prepare() below) - real ties among
    distinct symbols are now vanishingly rare, since no two symbols are likely to log the
    exact same relative return down to float precision.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._rs_by_symbol: dict[str, float] = {}
        self._percentile_by_symbol: dict[str, float] = {}
        self._universe: list[str] | None = None

    def describe(self) -> str:
        c = self.cfg
        bits = [f"mode={c.get('mode', 'rank')}", f"lookback_weeks={c.get('lookback_weeks')}",
                f"benchmark_symbol={c.get('benchmark_symbol')}"]
        if c.get("mode") == "filter":
            bits.append(f"min_percentile={c.get('min_percentile')}")
        return ", ".join(bits)

    def _load_universe(self) -> list[str]:
        if self._universe is None:
            p = ROOT / self.cfg.get("sector_map_path", "config/sector_map.json")
            try:
                data = json.loads(p.read_text()) if p.exists() else {}
                self._universe = sorted(data.get("symbol_to_industry", {}).keys())
            except Exception as exc:  # noqa: BLE001 - one bad file shouldn't crash the backtest
                log.warning("rs_rank: could not load universe from %s: %s", p, exc)
                self._universe = []
        return self._universe

    def prepare(self, asof: date, provider) -> None:
        # Deliberately recomputes every call, with no "same week, skip it" caching. A caching
        # attempt was tried and reverted: _weekly_return's trailing window is calculated from
        # whatever bars exist <= asof, so within a still-forming week its answer keeps drifting
        # day to day as new closes arrive - it is NOT a constant that only changes once the
        # week's Friday bar completes. A "same calendar week -> skip" cache ends up serving a
        # STALE snapshot from early in the week (e.g. Monday's) on the one day that actually
        # matters for ranking - the Friday decision day itself - because pandas' W-FRI resample
        # labels every day of a forming week with that same upcoming Friday. Confirmed on a real
        # case: TARIL's RS percentile was 74.7% using a Monday snapshot vs 90.0% recomputed
        # fresh on the Friday decision - enough to drop its composite score from 4 to 3 and lose
        # a contested slot to a worse candidate, materially changing backtest results. If this
        # needs to be faster again, the safe angle is having the ENGINE skip calling prepare()
        # on days with no entry candidates to rank at all, not caching inside the filter.
        self._rs_by_symbol = {}
        self._percentile_by_symbol = {}
        if not self.enabled:
            return

        weekly_rule = self.cfg.get("weekly_rule", "W-FRI")
        lookback = int(self.cfg.get("lookback_weeks", 13))
        bench_symbol = self.cfg.get("benchmark_symbol", "NIFTY500")

        try:
            bench_daily = provider.get_daily(bench_symbol, asof)
            bench_ret = _weekly_return(bench_daily, asof, weekly_rule, lookback)
        except Exception as exc:  # noqa: BLE001
            log.warning("rs_rank: benchmark '%s' fetch failed: %s", bench_symbol, exc)
            bench_ret = None

        if bench_ret is None:
            log.warning("rs_rank: no benchmark return available as of %s, skipping today (no RS tiebreak)", asof)
            return

        universe = self._load_universe()
        raw: dict[str, float] = {}
        for symbol in universe:
            try:
                daily = provider.get_daily(symbol, asof)
                sym_ret = _weekly_return(daily, asof, weekly_rule, lookback)
            except Exception:  # noqa: BLE001 - one bad symbol shouldn't break the rest
                sym_ret = None
            if sym_ret is None:
                continue
            raw[symbol] = sym_ret - bench_ret

        self._rs_by_symbol = raw
        if not raw:
            return

        ranked = sorted(raw.items(), key=lambda kv: kv[1])  # ascending: weakest RS first
        n = len(ranked)
        for i, (symbol, _) in enumerate(ranked):
            self._percentile_by_symbol[symbol] = (i + 1) / n  # (0, 1.0], higher = stronger RS

        log.info("FILTER rs_rank prepared RS for %d/%d universe symbols vs %s (%d-week trailing)",
                  len(raw), len(universe), bench_symbol, lookback)

    def evaluate(self, signal: Signal, ctx: SymbolContext) -> FilterResult:
        if not self.enabled:
            return FilterResult(True, "disabled")

        mode = self.cfg.get("mode", "rank")
        fail_open = self.cfg.get("fail_mode", "allow") == "allow"
        pct = self._percentile_by_symbol.get(signal.symbol)

        if pct is None:
            passed = True if mode == "rank" else fail_open
            return FilterResult(passed, f"no RS data for {signal.symbol}", score=0)

        min_pct = float(self.cfg.get("min_percentile", 0.0))
        passed = True if mode == "rank" else pct >= min_pct
        rs = self._rs_by_symbol.get(signal.symbol, 0.0)
        return FilterResult(passed, f"RS percentile {pct:.0%} (rel. return {rs:+.1%} vs benchmark)", score=pct)

    def max_score(self) -> float:
        """Percentile is already (0, 1.0] (see prepare()) - 1.0 whenever enabled, 0
        otherwise. Changed from a fixed 2.0 on 2026-10-27 when the 0/1/2 bucket scheme was
        dropped in favor of using the raw percentile directly - see evaluate()'s docstring.
        See Filter.max_score()'s own docstring for why this ratio matters."""
        return 1.0 if self.enabled else 0.0