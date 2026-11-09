"""Markdown summary of one eod/open run, built from the state files the run just wrote.

Used by the GitHub workflows to fill the run's Summary page:
    python scripts/run_summary.py eod  >> "$GITHUB_STEP_SUMMARY"
    python scripts/run_summary.py open >> "$GITHUB_STEP_SUMMARY"

Locally it prints the same markdown to the terminal. --date defaults to the latest date in the
journal; --state-dir defaults to ./state.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _load(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _journal(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def _cell(v) -> str:
    return str(v).replace("|", "\\|").replace("\n", " ")


def _table(headers: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(_cell(c) for c in r) + " |" for r in rows]
    return out


def _f(x, nd=2) -> str:
    return f"{x:,.{nd}f}" if isinstance(x, (int, float)) else "-"


def build(kind: str, state: Path, day: str | None, max_shortlist: int) -> str:
    journal = _journal(state / "journal.jsonl")
    if day is None:
        day = max((str(r.get("date"))[:10] for r in journal if r.get("date")), default="n/a")
    today = [r for r in journal if str(r.get("date"))[:10] == day]
    pending = _load(state / "pending_orders.json", [])
    pf = _load(state / "portfolio.json", {})
    positions = pf.get("positions", {}) or {}

    md = [f"## {kind.upper()} {day}", ""]

    # ---- account --------------------------------------------------------------------------
    equity_line = ""
    eq_file = state / "daily_equity.csv"
    if eq_file.exists():
        lines = [l for l in eq_file.read_text().splitlines() if l.strip()]
        if len(lines) > 1:
            equity_line = lines[-1]
    md.append(f"**Cash** ₹{_f(pf.get('cash'), 0)} · **Open positions** {len(positions)} · "
              f"**Pending orders** {len(pending)} · **Closed trades (all time)** {len(pf.get('trades', []))}")
    if equity_line:
        md.append(f"\n`daily_equity.csv` last row: `{equity_line}`")
    md.append("")

    # ---- pending orders (what the next session will try to execute) ----------------------
    md.append(f"### Pending orders ({len(pending)})")
    if pending:
        rows = []
        for o in pending:
            ref, stop = o.get("ref_price"), o.get("stop")
            risk = f"{(ref - stop) / ref * 100:.1f}%" if isinstance(ref, (int, float)) and isinstance(stop, (int, float)) and ref else "-"
            rows.append([o.get("action"), o.get("symbol"), o.get("strategy"), o.get("exec_date"),
                         _f(o.get("score"), 3), _f(ref), _f(stop), risk, o.get("reason", "")])
        md += _table(["Action", "Symbol", "Strategy", "Exec", "Score", "Ref", "Stop", "Stop dist", "Reason"], rows)
        by_strat = Counter(o.get("strategy") for o in pending)
        md.append("\nBy strategy: " + ", ".join(f"{k} {v}" for k, v in by_strat.most_common()))
    else:
        md.append("_none_")
    md.append("")

    # ---- open positions ------------------------------------------------------------------
    md.append(f"### Open positions ({len(positions)})")
    if positions:
        rows, total_upnl = [], 0.0
        for s, p in positions.items():
            lots = p.get("lots", [])
            qty = sum(l.get("qty", 0) for l in lots)
            avg = sum(l.get("qty", 0) * l.get("price", 0) for l in lots) / qty if qty else None
            last = p.get("last_price")
            upnl = (last - avg) * qty if isinstance(last, (int, float)) and avg is not None else None
            if upnl is not None:
                total_upnl += upnl
            pct = f"{(last / avg - 1) * 100:+.1f}%" if upnl is not None and avg else "-"
            rows.append([p.get("symbol", s), p.get("strategy", "-"), qty, _f(avg), _f(last),
                         _f(p.get("stop")), _f(upnl, 0), pct, p.get("adds", 0), p.get("opened", "-")])
        md += _table(["Symbol", "Strategy", "Qty", "Avg cost", "Last", "Stop", "Unreal. P&L", "%",
                      "Adds", "Opened"], rows)
        md.append(f"\nTotal unrealized P&L: ₹{_f(total_upnl, 0)}")
    else:
        md.append("_none_")
    md.append("")

    # ---- shortlist (eod) -----------------------------------------------------------------
    shortlist = next((r for r in reversed(today) if r.get("event") == "SHORTLIST"), None)
    if shortlist:
        cands = shortlist.get("candidates", [])
        md.append(f"### Shortlist: {shortlist.get('n_candidates', len(cands))} candidate(s), "
                  f"{shortlist.get('n_slots', 0)} free slot(s)")
        rows = [[c.get("rank"), c.get("symbol"), c.get("strategy"), _f(c.get("score"), 3),
                 "✅ taken" if c.get("taken") else "no slot"] for c in cands[:max_shortlist]]
        md += _table(["#", "Symbol", "Strategy", "Score", "Result"], rows)
        if len(cands) > max_shortlist:
            md.append(f"\n_…{len(cands) - max_shortlist} more not shown (see `state/journal.jsonl`)_")
        by_strat = Counter(c.get("strategy") for c in cands)
        md.append("\nCandidates by strategy: " + ", ".join(f"{k} {v}" for k, v in by_strat.most_common()))
        md.append("")

    # ---- everything else the run journaled today (fills, exits, filter rejections, skips) --
    other = [r for r in today if r.get("event") not in ("SHORTLIST", "ENTRY_SKIPPED")]
    counts = Counter(r.get("event") for r in today)
    md.append(f"### Journal events ({len(today)})")
    if counts:
        md.append(", ".join(f"`{k}` {v}" for k, v in counts.most_common()))
    if other:
        md.append("")
        rows = [[r.get("event"), r.get("symbol", "-"), r.get("strategy", "-"),
                 r.get("filter", "-"),
                 r.get("reason") or r.get("why") or ", ".join(f"{k}={v}" for k, v in r.items()
                                                              if k not in ("date", "event", "symbol", "strategy"))]
                for r in other]
        md += _table(["Event", "Symbol", "Strategy", "Filter", "Detail"], rows)
    elif not counts:
        md.append("_none_")
    md.append("")
    return "\n".join(md)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["eod", "open"])
    ap.add_argument("--date", help="YYYY-MM-DD (default: latest date in the journal)")
    ap.add_argument("--state-dir", default="state")
    ap.add_argument("--max-shortlist", type=int, default=60)
    a = ap.parse_args(argv)
    print(build(a.kind, Path(a.state_dir), a.date, a.max_shortlist))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
