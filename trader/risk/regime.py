from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd

from trader.core.indicators import resample_ohlc, sma

log = logging.getLogger(__name__)


@dataclass
class RegimeStatus:
    bullish: bool
    reason: str
    benchmark_close: float | None = None
    benchmark_sma: float | None = None


class RegimeFilter:
    """Gate on the *broader market*, not any one stock: config/regime_filter.json's
    benchmark_symbol (e.g. a Nifty 500 index instrument, held in config/universe.json's
    'instruments' map but deliberately in no tradeable 'list') must itself be trending up
    before NEW entries are allowed. Exits and management of already-open positions are never
    gated - this only stops opening new risk into a market that's already turning down,
    which is the one thing a purely per-stock momentum rule can't see for itself.

    Shared by market_regime (one benchmark, cfg['benchmark_symbol']) and sector_regime (one
    benchmark PER INDUSTRY, no single cfg['benchmark_symbol']) - status() takes the benchmark
    name as an explicit argument rather than reading it off self.cfg, so it works for both
    callers regardless of which cfg shape they were built with."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.get("enabled", False))

    def status(self, benchmark_daily: pd.DataFrame, asof: date, benchmark_symbol: str = "?") -> RegimeStatus:
        c = self.cfg
        if benchmark_daily is None or benchmark_daily.empty:
            fail_open = c.get("fail_mode", "block") == "allow"
            return RegimeStatus(fail_open, "no benchmark data (fail_mode=%s)" % c.get("fail_mode", "block"))

        weekly = resample_ohlc(benchmark_daily, c["timeframe_rule"])
        if len(weekly) < c["min_bars"]:
            fail_open = c.get("fail_mode", "block") == "allow"
            return RegimeStatus(fail_open, f"only {len(weekly)} benchmark bars, need {c['min_bars']}")

        s = sma(weekly["close"], c["sma_period"])
        if pd.isna(s.iloc[-1]):
            fail_open = c.get("fail_mode", "block") == "allow"
            return RegimeStatus(fail_open, "SMA not yet warmed up")

        close, level = float(weekly["close"].iloc[-1]), float(s.iloc[-1])
        bullish = close > level * (1 + c["buffer_pct"] / 100.0)
        reason = (f"benchmark {benchmark_symbol} close {close:.2f} "
                 f"{'>' if bullish else '<='} weekly SMA{c['sma_period']} {level:.2f}")
        return RegimeStatus(bullish, reason, close, level)