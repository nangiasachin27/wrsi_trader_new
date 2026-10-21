"""
Course-based swing-structure market-trend classifier (UPTREND / SIDEWAYS / DOWNTREND) run
against a benchmark index's daily bars - see trader/core/trend_structure.py for the full rules.
Separate from scripts/market_trend.py, which still runs the older IBD-style VD-day/FTD engine.

    python scripts/trend_structure.py
    python scripts/trend_structure.py --benchmark NIFTY500 --date 2026-09-30
    python scripts/trend_structure.py --history 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trader.core.config import ROOT, load_settings
from trader.core.trend_structure import classify_trend_structure


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", help="override config/trend_structure.json's benchmark_symbol")
    ap.add_argument("--date", help="YYYY-MM-DD - classify as of this date (default: latest cached bar)")
    ap.add_argument("--history", type=int, default=10, help="how many recent transition events to print")
    a = ap.parse_args()

    cfg = json.loads((ROOT / "config" / "trend_structure.json").read_text())
    symbol = a.benchmark or cfg["benchmark_symbol"]

    s = load_settings()
    cache = ROOT / s.system["paths"]["cache_dir"]
    f = cache / f"{symbol}.csv"
    if not f.exists():
        sys.exit(f"No cached bars for {symbol} - run scripts/fetch_history.py first")
    daily = pd.read_csv(f, parse_dates=["date"], index_col="date").sort_index()
    if a.date:
        daily = daily.loc[:pd.Timestamp(a.date)]
    if daily.empty:
        sys.exit(f"No bars for {symbol} on or before {a.date}")

    out, events = classify_trend_structure(daily, cfg)
    if out.empty:
        sys.exit(f"Not enough bars to confirm any swings yet (need pivot_left+pivot_right+1="
                 f"{cfg['pivot_left'] + cfg['pivot_right'] + 1}, have {len(daily)})")
    last = out.iloc[-1]
    as_of = out.index[-1].date()

    print(f"{symbol} | as of {as_of} | timeframe={cfg.get('timeframe', 'daily')}\n")
    print(f"STATE: {last['state']}")
    if pd.notna(last["structure_level"]):
        print(f"Protecting {last['structure_kind']}: {last['structure_level']:.2f} "
              f"(a close {'below' if last['structure_kind'] == 'low' else 'above'} this is "
              f"the earliest warning of a trend change)")
    print(f"Confirmed swing highs so far: {int(last['swing_high_count'])}, "
          f"swing lows: {int(last['swing_low_count'])}")

    print(f"\nLast {a.history} transition(s):")
    for ev in events[-a.history:]:
        print(f"  {ev.date}  {ev.from_state:>9s} -> {ev.to_state:<9s}  {ev.reason}")
    if not events:
        print("  (none in this history)")


if __name__ == "__main__":
    main()
