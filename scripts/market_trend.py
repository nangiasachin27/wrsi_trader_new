"""
General market-trend classifier (UPTREND / NEUTRAL / DOWNTREND) run against a benchmark
index's daily bars - see trader/core/market_trend.py for the full rules.

    python scripts/market_trend.py
    python scripts/market_trend.py --benchmark NIFTY500 --date 2026-09-30
    python scripts/market_trend.py --history 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trader.core.config import ROOT, load_settings
from trader.core.market_trend import aggregate_universe_volume, classify_market_trend


def _load_universe(cfg: dict) -> list[str]:
    p = ROOT / cfg.get("universe_sector_map_path", "config/sector_map.json")
    data = json.loads(p.read_text()) if p.exists() else {}
    return sorted(data.get("symbol_to_industry", {}).keys())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", help="override config/market_trend.json's benchmark_symbol")
    ap.add_argument("--date", help="YYYY-MM-DD - classify as of this date (default: latest cached bar)")
    ap.add_argument("--history", type=int, default=10, help="how many recent transition events to print")
    ap.add_argument("--own-volume", action="store_true",
                    help="use the benchmark's own 'volume' column instead of aggregate universe "
                         "volume - only meaningful if --benchmark names a real traded instrument, "
                         "not an index (which has no volume of its own)")
    a = ap.parse_args()

    cfg = json.loads((ROOT / "config" / "market_trend.json").read_text())
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

    volume = None
    if not a.own_volume:
        universe = _load_universe(cfg)
        print(f"Building aggregate volume from {len(universe)} universe symbols "
              f"(config/sector_map.json)...", file=sys.stderr)
        volume = aggregate_universe_volume(cache, universe)

    out, events = classify_market_trend(daily, cfg, volume)
    last = out.iloc[-1]
    as_of = out.index[-1].date()

    print(f"{symbol} | as of {as_of}\n")
    print(f"STATE: {last['state']}")
    print(f"VD days in trailing {cfg['vd_window_calendar_days']} calendar days: {int(last['vd_count_30d'])} "
          f"(neutral at {cfg['neutral_vd_count']}, downtrend at {cfg['downtrend_vd_count']})")
    if pd.notna(last["day1_date"]):
        days_tracked = (out.index.get_loc(out.index[-1]) - out.index.get_loc(pd.Timestamp(last["day1_date"])))
        print(f"Tracking Day1: {last['day1_date'].date()} (low {last['day1_low']:.2f}), "
              f"{days_tracked} trading day(s) since")

    print(f"\nLast {a.history} transition(s):")
    for ev in events[-a.history:]:
        print(f"  {ev.date}  {ev.from_state:>9s} -> {ev.to_state:<9s}  {ev.reason}")
    if not events:
        print("  (none in this history)")


if __name__ == "__main__":
    main()