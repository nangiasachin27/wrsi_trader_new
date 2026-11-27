"""Market breadth: share of the trading universe closing above its own N-day SMA.

A faster, broader market-trend signal than one index vs its weekly SMA (trader.risk.regime):
breadth turns as soon as most stocks roll over, before a cap-weighted index does, and it sees
the small/mid-cap tape this system actually trades.

    pct_above(t) = 100 * #(close > SMA_N, among symbols with >= N bars by t) / #(those symbols)
    smoothed(t)  = mean of pct_above over the last `smooth_days` sessions
    state        : hysteresis - BULL once smoothed >= bull_on_pct, stays BULL until smoothed
                   < bull_off_pct, then NOT-BULL until it climbs back to bull_on_pct. The band in
                   between holds the previous state, so a reading oscillating around one level
                   does not flip the signal every few days.

Everything at date t uses closes up to and including t only (no look-ahead); the engine reads
the state as of the session BEFORE a fill. Caveat: the universe is today's constituent list, so
early-history breadth carries survivorship bias (stocks that later dropped out are missing).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULTS = {"sma_period": 50, "smooth_days": 5, "bull_on_pct": 55.0, "bull_off_pct": 45.0,
            "min_symbols": 50}


def breadth_frame(frames: dict[str, pd.DataFrame], symbols: list[str], sma_period: int = 50) -> pd.DataFrame:
    """Per-date pct of symbols above their SMA, and how many symbols had a valid SMA."""
    closes = {s: frames[s]["close"] for s in symbols if s in frames and not frames[s].empty}
    if not closes:
        return pd.DataFrame(columns=["pct_above", "n_valid"])
    wide = pd.DataFrame(closes).sort_index()
    # each symbol's SMA over ITS OWN bars (a symbol missing a day is not filled forward)
    sma = wide.apply(lambda c: c.dropna().rolling(sma_period, min_periods=sma_period).mean()
                     .reindex(c.index))
    valid = sma.notna() & wide.notna()
    above = (wide > sma) & valid
    n = valid.sum(axis=1)
    pct = (above.sum(axis=1) / n.replace(0, np.nan)) * 100.0
    return pd.DataFrame({"pct_above": pct, "n_valid": n})


def classify(pct: pd.Series, smooth_days: int = 5, bull_on_pct: float = 55.0,
             bull_off_pct: float = 45.0) -> pd.DataFrame:
    """Hysteresis state machine on the smoothed breadth. First state = smoothed >= midpoint."""
    sm = pct.rolling(smooth_days, min_periods=1).mean()
    state, out = None, []
    mid = (bull_on_pct + bull_off_pct) / 2.0
    for v in sm.to_numpy():
        if np.isnan(v):
            out.append(np.nan)
            continue
        if state is None:
            state = v >= mid
        elif state and v < bull_off_pct:
            state = False
        elif not state and v >= bull_on_pct:
            state = True
        out.append(state)
    return pd.DataFrame({"smoothed": sm, "bullish": pd.Series(out, index=pct.index, dtype=object)})


@dataclass
class BreadthStatus:
    bullish: bool
    reason: str
    pct_above: float | None = None
    smoothed: float | None = None


class MarketBreadth:
    """Breadth regime for the engine. With a HistoricalProvider (backtest: it holds every frame
    in .data) the whole series is computed once and looked up; otherwise (live) it is computed
    from each symbol's recent history ending at `asof` - enough bars for the SMA, the smoothing
    and ~1 year for the hysteresis state to settle."""

    def __init__(self, cfg: dict | None):
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self._series: pd.DataFrame | None = None

    def _compute(self, frames: dict[str, pd.DataFrame], symbols: list[str]) -> pd.DataFrame:
        c = self.cfg
        b = breadth_frame(frames, symbols, int(c["sma_period"]))
        b = b[b["n_valid"] >= int(c["min_symbols"])]
        st = classify(b["pct_above"], int(c["smooth_days"]), float(c["bull_on_pct"]), float(c["bull_off_pct"]))
        return b.join(st)

    def status(self, asof: date, provider, symbols: list[str]) -> BreadthStatus:
        ts = pd.Timestamp(asof)
        data = getattr(provider, "data", None)
        if isinstance(data, dict):
            if self._series is None:
                self._series = self._compute(data, symbols)
            ser = self._series.loc[:ts]
        else:
            keep = int(self.cfg["sma_period"]) + int(self.cfg["smooth_days"]) + 260
            cache = getattr(provider, "cache", None)
            frames = {}
            for s in symbols:
                try:
                    if cache is not None:
                        # live: read the CSV cache that the previous EOD run refreshed - no API
                        # calls (get_daily would re-fetch every symbol in a fresh `open` process)
                        f = Path(cache) / f"{s}.csv"
                        if not f.exists():
                            continue
                        df = pd.read_csv(f, parse_dates=["date"], index_col="date", usecols=["date", "close"])
                        frames[s] = df.loc[:ts].tail(keep)
                    else:
                        frames[s] = provider.get_daily(s, asof).tail(keep)
                except Exception:  # noqa: BLE001 - one missing symbol must not kill the regime
                    continue
            ser = self._compute(frames, symbols).loc[:ts]
        if ser.empty or pd.isna(ser["bullish"].iloc[-1]):
            return BreadthStatus(True, "breadth: not enough data - treated as bullish (fail open)")
        row = ser.iloc[-1]
        bull = bool(row["bullish"])
        return BreadthStatus(bull, (f"breadth {row['pct_above']:.1f}% above SMA{self.cfg['sma_period']} "
                                    f"(smoothed {row['smoothed']:.1f}%, bull >= {self.cfg['bull_on_pct']:g}, "
                                    f"off < {self.cfg['bull_off_pct']:g}) -> {'BULL' if bull else 'NOT BULL'}"),
                             float(row["pct_above"]), float(row["smoothed"]))
