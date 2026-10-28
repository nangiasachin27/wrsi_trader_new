from __future__ import annotations

import json
import logging
from datetime import date

import pandas as pd

from trader.core.config import ROOT
from trader.core.indicators import adx, resample_ohlc, sma
from trader.core.models import Signal
from trader.filters.base import Filter, FilterResult, register
from trader.strategies.base import SymbolContext

log = logging.getLogger(__name__)


def _percentile_map(raw: dict[str, float]) -> dict[str, float]:
    """Cross-sectional percentile, ascending (highest value -> percentile closest to 1.0) -
    same construction as rs_rank.py's own `_percentile_by_symbol`. Self-calibrating to
    whatever today's actual distribution looks like, unlike a fixed absolute ceiling: in a
    raging bull market where every stock's ADX is elevated, only the TOP ones still get a
    percentile near 1.0, and the same is true in a weak/choppy market where every stock's ADX
    is low - there's no absolute number that works in both regimes, see sector_regime.py's own
    2026-10-27 lesson (first fix attempt used a fixed score_ceiling_pct=10%, which stayed too
    easy to clear during the whole 2021-01 to 2021-03 strongly-trending window and did NOT fix
    the mass-tie problem it was built to fix)."""
    if not raw:
        return {}
    ranked = sorted(raw.items(), key=lambda kv: kv[1])
    n = len(ranked)
    return {sym: (i + 1) / n for i, (sym, _) in enumerate(ranked)}


@register("trend_strength")
class TrendStrengthFilter(Filter):
    """Per-candidate confirmation that a weekly-RSI>60 entry is riding a real, strong trend -
    not a choppy-range spike that the lagging weekly-wick-vs-SMA20 exit won't catch until it's
    already given back several percent. Two independently toggleable sub-checks, combined the
    same way as the quality filter's ath/cup checks:

      - adx: weekly ADX(period) >= min_adx. ADX measures trend STRENGTH, not direction, so it's
        a genuine complement to the strategy's own RSI>60 (direction), not a duplicate of it -
        it filters out weak/choppy RSI pops that happen without a real directional move behind
        them.
      - sma_trend: weekly close above weekly SMA(period) * (1+buffer_pct/100). A higher-timeframe
        trend-alignment check layered on top of the strategy's own weekly/monthly RSI>60 -
        typically run with period=50 to require the stock to already be in a multi-month uptrend,
        not just popping RSI off a short-term low.

    mode 'rank' (default) only affects the tiebreak for a free slot when candidates outnumber
    slots; mode 'filter' is a hard gate - any candidate scoring below min_score is excluded
    outright, regardless of free slots. Same contract as config/filters/quality.json, for
    consistency across filters.

    Scoring, each sub-check's contribution (0.0-1.0): went through TWO iterations on
    2026-10-27. First was a flat 0/1 per sub-check (int(ok)) - with only 2 sub-checks and a
    500-symbol universe, most days produced a large pile of candidates tied at the max score
    (2), decided only by an arbitrary tie-break (see rs_rank.py's docstring for the same
    finding there). The FIRST fix used each sub-check's raw value scaled against a fixed
    absolute `score_ceiling`/`score_ceiling_pct` - this was abandoned the same day once the
    equivalent fix in sector_regime.py demonstrably failed to resolve the mass-tie problem in
    a real backtest: a fixed absolute ceiling is regime-dependent (too easy to clear in a
    strong bull market, too hard in a weak one), so it just moved the saturation point rather
    than removing it. Replaced with a CROSS-SECTIONAL PERCENTILE, same construction as
    rs_rank's own `_percentile_by_symbol`: each sub-check's raw value (ADX, SMA-distance-%) is
    ranked against the SAME metric computed for every other symbol in the universe that day
    (see prepare() below), self-calibrating to whatever today's actual distribution looks
    like rather than a hand-picked number. The pass/fail gate (`adx_ok`/`sma_trend_ok`, used
    for 'filter' mode and the reason text) is unchanged and still keyed off `min_adx`/
    `buffer_pct`, computed per-candidate exactly as before - only the ranking score changed.

    KNOWN APPROXIMATION in the percentile computation: prepare()'s universe-wide scan uses
    each symbol's latest resampled weekly bar as-is (which may be a still-forming/partial week
    on a non-week-boundary day), while each individual candidate's own GATE in evaluate() uses
    `ctx.weekly(..., include_partial=False)` (completed bars only) - done this way because
    filters' `prepare(asof, provider)` has no access to the engine's `TradingCalendar` needed
    to replicate the exact completed-vs-partial distinction universe-wide. This only affects
    how precisely "fair" the cross-sectional comparison is (comparing a few symbols' forming-
    week reading against others' completed-week readings), not the gate/exclusion logic, and
    ADX(14)/SMA(50) move little within one partial week - acceptable for a ranking tiebreak,
    not acceptable if this value were ever used for a hard gate.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._adx_pct_by_symbol: dict[str, float] = {}
        self._sma_pct_by_symbol: dict[str, float] = {}
        self._universe: list[str] | None = None

    def describe(self) -> str:
        c = self.cfg
        bits = [f"mode={c['mode']}", f"min_score={c['min_score']}"]
        if c["adx"]["enabled"]:
            bits.append(f"adx>={c['adx']['min_adx']} (period {c['adx']['period']})")
        if c["sma_trend"]["enabled"]:
            bits.append(f"close>SMA{c['sma_trend']['period']}")
        return ", ".join(bits)

    def _load_universe(self) -> list[str]:
        # Reuses rs_rank's own universe source (config/sector_map.json's symbol_to_industry
        # keys) rather than inventing a second one - same reasoning rs_rank's own docstring
        # gives: this is already populated for sector_regime, no new file to maintain.
        if self._universe is None:
            p = ROOT / self.cfg.get("sector_map_path", "config/sector_map.json")
            try:
                data = json.loads(p.read_text()) if p.exists() else {}
                self._universe = sorted(data.get("symbol_to_industry", {}).keys())
            except Exception as exc:  # noqa: BLE001 - one bad file shouldn't crash the backtest
                log.warning("trend_strength: could not load universe from %s: %s", p, exc)
                self._universe = []
        return self._universe

    def prepare(self, asof: date, provider) -> None:
        self._adx_pct_by_symbol = {}
        self._sma_pct_by_symbol = {}
        if not self.enabled:
            return
        c = self.cfg
        weekly_rule = c.get("weekly_rule", "W-FRI")
        adx_raw: dict[str, float] = {}
        sma_raw: dict[str, float] = {}

        for symbol in self._load_universe():
            try:
                daily = provider.get_daily(symbol, asof)
            except Exception:  # noqa: BLE001 - one bad symbol shouldn't break the rest
                continue
            if daily is None or daily.empty:
                continue
            w = resample_ohlc(daily, weekly_rule)

            if c["adx"]["enabled"]:
                period = c["adx"]["period"]
                if len(w) >= period * 2:
                    a = adx(w["high"], w["low"], w["close"], period)
                    val = float(a.iloc[-1]) if len(a) and not pd.isna(a.iloc[-1]) else None
                    if val is not None:
                        adx_raw[symbol] = val

            if c["sma_trend"]["enabled"]:
                period = c["sma_trend"]["period"]
                if len(w) >= period:
                    s = sma(w["close"], period)
                    level = float(s.iloc[-1]) if not pd.isna(s.iloc[-1]) else None
                    if level:
                        px = float(w["close"].iloc[-1])
                        sma_raw[symbol] = (px - level) / level * 100.0

        self._adx_pct_by_symbol = _percentile_map(adx_raw)
        self._sma_pct_by_symbol = _percentile_map(sma_raw)
        log.info("FILTER trend_strength prepared ADX pct for %d, SMA-dist pct for %d of %d universe symbols",
                  len(adx_raw), len(sma_raw), len(self._load_universe()))

    def evaluate(self, signal: Signal, ctx: SymbolContext) -> FilterResult:
        if not self.enabled:
            return FilterResult(True, "disabled")
        c = self.cfg
        w = ctx.weekly(c.get("weekly_rule", "W-FRI"), include_partial=False)
        score, gate_score, notes = 0.0, 0, {}

        if c["adx"]["enabled"]:
            period = c["adx"]["period"]
            if len(w) < period * 2:
                notes["adx_ok"], notes["adx_val"] = False, None
            else:
                a = adx(w["high"], w["low"], w["close"], period)
                val = float(a.iloc[-1]) if len(a) and not pd.isna(a.iloc[-1]) else None
                ok = val is not None and val >= c["adx"]["min_adx"]
                notes["adx_ok"], notes["adx_val"] = ok, (round(val, 1) if val is not None else None)
                gate_score += int(ok)
                score += self._adx_pct_by_symbol.get(signal.symbol, 0.0)

        if c["sma_trend"]["enabled"]:
            period = c["sma_trend"]["period"]
            if len(w) < period:
                notes["sma_trend_ok"], notes["sma_dist_pct"] = False, None
            else:
                s = sma(w["close"], period)
                level = float(s.iloc[-1])
                px = float(w["close"].iloc[-1])
                ok = bool(level and px > level * (1 + c["sma_trend"]["buffer_pct"] / 100.0))
                dist_pct = (px - level) / level * 100.0 if level else None
                notes["sma_trend_ok"] = ok
                notes["sma_dist_pct"] = round(dist_pct, 1) if dist_pct is not None else None
                gate_score += int(ok)
                score += self._sma_pct_by_symbol.get(signal.symbol, 0.0)

        # Gate/reason logic stays on the old boolean gate_score (unaffected by the percentile
        # scoring above) - min_score in 'filter' mode still means "N sub-checks passed", not a
        # fractional threshold on the new continuous score.
        passed = not (c["mode"] == "filter" and gate_score < c["min_score"])
        parts = []
        if "adx_val" in notes:
            parts.append(f"ADX {notes['adx_val']}" if notes["adx_val"] is not None else "ADX n/a")
        if "sma_dist_pct" in notes:
            parts.append(f"price {notes['sma_dist_pct']:+.1f}% vs SMA{c['sma_trend']['period']}"
                        if notes["sma_dist_pct"] is not None else "SMA n/a")
        reason = f"trend_strength score {score:.2f}" + (f" ({', '.join(parts)})" if parts else "")
        return FilterResult(passed, reason, score=score, meta=notes)

    def max_score(self) -> float:
        """adx + sma_trend, each worth 1 point when its own sub-check is turned on - see
        Filter.max_score()'s docstring."""
        if not self.enabled:
            return 0.0
        c = self.cfg
        return float(int(c["adx"]["enabled"]) + int(c["sma_trend"]["enabled"]))