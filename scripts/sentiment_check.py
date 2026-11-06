"""Sentiment filter helpers, driven by a backtest's own output folder
(state/backtest_<start>_<end>[_label]/ with journal.jsonl and trade_log.csv).

  # 1. which (date, symbol) pairs need sentiment data? (all post-filter candidates)
  python scripts/sentiment_check.py pairs state/backtest_2026-01-01_2026-10-31_base --out pairs.csv
  #    only the ones actually taken (smallest set, enough for step 3):
  python scripts/sentiment_check.py pairs state/backtest_2026-01-01_2026-10-31_base --taken-only --out pairs.csv

  # 2. scrape them in news_scraper:  python scrape_pairs.py pairs.csv --skip-existing
  #    then export:                 python -m sentiment_backtest.export --start ... --end ... --out <wrsi>/sentiment_gate/...

  # 3. would the filter have cut losers? closed trades split by the sentiment at their signal date
  python scripts/sentiment_check.py trades state/backtest_2026-01-01_2026-10-31_base

Step 3 answers "were the flagged trades losers?" without re-running anything. The real
test is the A/B: the same backtest with the filter enabled, which also lets the freed slot
go to the next candidate - for that, step 1 should cover ALL shortlisted candidates, and
a second `pairs` pass over the filtered run picks up candidates that only appear there.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trader.core.config import ROOT  # noqa: E402
from trader.core.models import Signal  # noqa: E402
from trader.filters.sentiment import SentimentFilter  # noqa: E402


def _shortlists(bt_dir: Path):
    j = bt_dir / "journal.jsonl"
    if not j.exists():
        raise SystemExit(f"{j} not found - pass a backtest output folder (state/backtest_...)")
    with open(j, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("event") == "SHORTLIST":
                yield str(rec["date"])[:10], rec.get("candidates", [])


def _filter(path: str | None) -> SentimentFilter:
    cfg_path = ROOT / "config" / "filters" / "sentiment.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    cfg = {**cfg, "enabled": True, "mode": "filter"}
    if path:
        cfg["path"] = path
    return SentimentFilter(cfg)


def cmd_pairs(a) -> int:
    pairs: set[tuple[str, str]] = set()
    for bt in a.backtest_dirs:
        for d, cands in _shortlists(Path(bt)):
            pairs |= {(d, c["symbol"]) for c in cands if c.get("taken") or not a.taken_only}
    have = 0
    if a.missing_only:
        f = _filter(a.sentiment_path)
        f._load()
        have = len(pairs & set(f._rows))
        pairs -= set(f._rows)
    rows = sorted(pairs)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["date", "symbol"])
        w.writerows(rows)
    print(f"{len(rows)} pairs over {len({d for d, _ in rows})} dates, {len({s for _, s in rows})} symbols -> {out}"
          + (f" ({have} already covered, left out)" if a.missing_only else ""))
    return 0


def cmd_trades(a) -> int:
    bt = Path(a.backtest_dir)
    tl = bt / "trade_log.csv"
    if not tl.exists():
        raise SystemExit(f"{tl} not found")
    taken: dict[str, list[str]] = {}
    for d, cands in _shortlists(bt):
        for c in cands:
            if c.get("taken"):
                taken.setdefault(c["symbol"], []).append(d)
    logging.getLogger("trader.filters.sentiment").setLevel(logging.ERROR)  # per-date notices
    f = _filter(a.sentiment_path)
    groups: dict[str, list[float]] = {}
    out_rows = []
    with open(tl, newline="", encoding="utf-8") as fh:
        trades = list(csv.DictReader(fh))
    for t in trades:
        entry = str(t["entry_date"])[:10]
        sig_dates = [d for d in taken.get(t["symbol"], []) if d < entry]
        if not sig_dates:
            grp, why = "no_signal_date", ""
        else:
            sd = sig_dates[-1]
            f.prepare(date.fromisoformat(sd), None)
            r = f.evaluate(Signal(t["symbol"], t.get("strategy") or "", "ENTRY", sd, ""), None)
            grp = "flagged" if not r.passed else r.meta.get("coverage", "?")
            why = r.reason
        pnl = float(t["pnl_pct"]) if t.get("pnl_pct") not in (None, "") else 0.0
        groups.setdefault(grp, []).append(pnl)
        out_rows.append({**t, "sentiment_group": grp, "sentiment_reason": why})

    def stats(xs):
        wins = [x for x in xs if x > 0]
        losses = [-x for x in xs if x <= 0]
        pf = sum(wins) / sum(losses) if sum(losses) else float("inf")
        return len(xs), 100 * len(wins) / len(xs), sum(xs) / len(xs), pf

    print(f"{len(trades)} closed trades in {tl}")
    print(f"  {'group':16s} {'trades':>6s} {'win%':>6s} {'avg pnl%':>9s} {'PF*':>6s}")
    order = ["flagged", "news", "no_news", "missing", "no_signal_date"]
    for g in order + sorted(set(groups) - set(order)):
        if g in groups:
            n, w, avg, pf = stats(groups[g])
            print(f"  {g:16s} {n:6d} {w:6.1f} {avg:9.2f} {pf:6.2f}")
    kept = [x for g, xs in groups.items() if g != "flagged" for x in xs]
    if groups.get("flagged") and kept:
        n0, w0, a0, _ = stats([x for xs in groups.values() for x in xs])
        n1, w1, a1, _ = stats(kept)
        print(f"\n  all trades      win {w0:.1f}%  avg {a0:+.2f}%   ->   without flagged: win {w1:.1f}%  avg {a1:+.2f}%")
    covered = sum(len(groups.get(g, [])) for g in ("flagged", "news", "no_news"))
    print(f"\n  coverage: {covered}/{len(trades)} trades had a sentiment row at their signal date"
          "  (*PF on pnl% - equal-weighted, not rupees)")
    if covered < len(trades):
        print("  -> run `pairs --taken-only --missing-only`, scrape, export, and re-run this for full coverage")
    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(out_rows)
        print(f"  per-trade detail -> {a.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("pairs", help="(date, symbol) pairs a backtest needs sentiment for")
    pr.add_argument("backtest_dirs", nargs="+")
    pr.add_argument("--out", default="pairs.csv")
    pr.add_argument("--taken-only", action="store_true", help="only candidates that got a slot")
    pr.add_argument("--missing-only", action="store_true", help="leave out pairs already in the sentiment CSVs")
    pr.add_argument("--sentiment-path", help="CSV file or folder (default: the filter config's path)")
    tr = sub.add_parser("trades", help="closed trades grouped by sentiment at their signal date")
    tr.add_argument("backtest_dir")
    tr.add_argument("--sentiment-path", help="CSV file or folder (default: the filter config's path)")
    tr.add_argument("--out", help="also write per-trade detail CSV")
    a = p.parse_args(argv)
    return cmd_pairs(a) if a.cmd == "pairs" else cmd_trades(a)


if __name__ == "__main__":
    sys.exit(main())
