from __future__ import annotations

import csv
import logging
from datetime import date
from pathlib import Path

from trader.core.config import ROOT
from trader.core.models import Signal
from trader.data.base import DataProvider
from trader.filters.base import Filter, FilterResult, register
from trader.strategies.base import SymbolContext

log = logging.getLogger(__name__)


@register("sentiment")
class SentimentFilter(Filter):
    """Per-candidate news-sentiment veto: blocks an ENTRY when the stock had clearly negative
    news before today's scan. A pure gate (max_score() == 0): it never adds to the ranking,
    because strongly POSITIVE sentiment showed no edge in the news_scraper backtest - only the
    negative side did.

    Data: daily CSVs from news_scraper's `python -m sentiment_backtest.export`, one row per
    (date, symbol) that was actually scraped. The row for day T uses only news known before
    15:15 IST on T, so an eod scan on T (after the close) can use it without lookahead; the
    order then executes at T+1's open. `negative` in the file is the frozen rule
    (score <= -0.25, weight >= 1.0, n_neg > n_pos); this filter re-applies the same rule from
    the raw columns using the thresholds in config, which default to exactly those values.
    They were fixed BEFORE the backtest that validated them - sweeping them against wrsi's own
    backtest would be fitting noise, so leave them alone.

    Coverage is explicit. A symbol-day with a row but no news (n = 0) passes as "no news". A
    symbol-day WITHOUT a row was never scraped: that's "unknown" and follows fail_mode
    ('allow' by default - a missing scrape should never stop the system trading, it just
    trades without this filter that day). Every result carries meta.coverage so the share of
    candidates actually checked shows up in the candidate meta.

    Modes:
      filter  block flagged candidates
      shadow  never block; flagged candidates pass with meta.would_block=True and an INFO log
              line - for running it live alongside the system before trusting it
    lookback_days: block if ANY of the last N scraped dates up to and including today is
    flagged (default 1 = today's row only, the horizon that was validated).
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._rows: dict[tuple[str, str], dict] = {}
        self._dates: list[str] = []
        self._loaded_sig: tuple | None = None
        self._asof: str | None = None
        self._recent: list[str] = []
        self._day_covered = False

    # ---- data --------------------------------------------------------------------------
    def _files(self) -> list[Path]:
        p = Path(self.cfg.get("path", "sentiment_gate"))
        p = p if p.is_absolute() else ROOT / p
        if p.is_file():
            return [p]
        return sorted(p.glob(self.cfg.get("glob", "*.csv"))) if p.is_dir() else []

    def _load(self) -> None:
        files = self._files()
        sig = tuple((str(f), f.stat().st_mtime_ns) for f in files)
        if sig == self._loaded_sig:
            return
        rows: dict[tuple[str, str], dict] = {}
        for f in files:  # sorted by name: a later file overrides an earlier one for the same key
            with open(f, newline="", encoding="utf-8-sig") as fh:
                for r in csv.DictReader(fh):
                    d, s = (r.get("date") or "").strip()[:10], (r.get("symbol") or "").strip().upper()
                    if d and s:
                        rows[(d, s)] = r
        self._rows, self._loaded_sig = rows, sig
        self._dates = sorted({d for d, _ in rows})
        log.info("FILTER sentiment: %d symbol-day rows from %d file(s), %s .. %s", len(rows), len(files),
                 self._dates[0] if self._dates else "-", self._dates[-1] if self._dates else "-")
        if not files:
            log.warning("FILTER sentiment: no CSV files at %s - every candidate is 'unknown' (fail_mode=%s)",
                        self.cfg.get("path", "sentiment_gate"), self.cfg.get("fail_mode", "allow"))

    @staticmethod
    def _f(v, default=0.0) -> float:
        try:
            return float(v) if v not in (None, "") else default
        except ValueError:
            return default

    def _negative(self, r: dict) -> bool:
        c = self.cfg
        score, weight = self._f(r.get("score"), float("nan")), self._f(r.get("weight"))
        n_neg, n_pos = int(self._f(r.get("n_neg"))), int(self._f(r.get("n_pos")))
        return (score <= c.get("veto_score", -0.25) and weight >= c.get("veto_weight", 1.0)
                and (n_neg > n_pos if c.get("require_more_negative", True) else True))

    # ---- Filter API ----------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (f"mode={c.get('mode', 'filter')}, fail_mode={c.get('fail_mode', 'allow')}, "
                f"lookback_days={c.get('lookback_days', 1)}, path={c.get('path', 'sentiment_gate')}")

    def prepare(self, asof: date, provider: DataProvider) -> None:
        if not self.enabled:
            return
        self._load()
        self._asof = asof.isoformat()
        n = max(1, int(self.cfg.get("lookback_days", 1)))
        self._recent = [d for d in self._dates if d <= self._asof][-n:]
        self._day_covered = bool(self._recent) and self._recent[-1] == self._asof
        if not self._day_covered:
            log.warning("FILTER sentiment %s: no sentiment rows for this date - candidates are 'unknown' "
                        "(fail_mode=%s)", asof, self.cfg.get("fail_mode", "allow"))

    def evaluate(self, signal: Signal, ctx: SymbolContext) -> FilterResult:
        if not self.enabled:
            return FilterResult(True, "disabled")
        if self._asof is None:
            return FilterResult(False, "prepare() was not called before evaluate()")
        sym = signal.symbol.upper()
        today = self._rows.get((self._asof, sym))
        if today is None:
            allow = self.cfg.get("fail_mode", "allow") == "allow"
            return FilterResult(allow, f"no sentiment row for {self._asof} ({'allowed' if allow else 'blocked'})",
                                meta={"coverage": "missing"})
        flagged = [d for d in self._recent if (r := self._rows.get((d, sym))) is not None and self._negative(r)]
        meta = {"coverage": "news" if int(self._f(today.get("n"))) > 0 else "no_news",
                "sent_score": today.get("score") or None, "sent_weight": self._f(today.get("weight")),
                "sent_n": int(self._f(today.get("n"))), "sent_flagged_dates": flagged}
        if not flagged:
            return FilterResult(True, f"sentiment ok ({meta['coverage']})", meta=meta)
        r = self._rows[(flagged[-1], sym)]
        why = (f"negative news on {flagged[-1]}: score {self._f(r.get('score')):+.2f}, weight "
               f"{self._f(r.get('weight')):.2f}, {r.get('n_neg')} neg vs {r.get('n_pos')} pos")
        if self.cfg.get("mode", "filter") == "shadow":
            meta["would_block"] = True
            log.info("FILTER sentiment SHADOW %s %s would be blocked: %s", self._asof, sym, why)
            return FilterResult(True, f"shadow: {why}", meta=meta)
        return FilterResult(False, why, meta=meta)
