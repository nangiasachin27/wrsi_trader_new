"""
Pass a stock to this script and get back its support/resistance zones for a given timeframe.

    python scripts/levels.py RELIANCE --timeframe weekly
    python scripts/levels.py RELIANCE --timeframe daily --date 2026-09-22
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trader.core.config import ROOT, load_settings
from trader.core.levels import nearest_levels, support_resistance


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol")
    ap.add_argument("--timeframe", choices=["daily", "weekly", "monthly"], default="weekly")
    ap.add_argument("--date", help="YYYY-MM-DD - analyze as of this date (default: latest cached bar)")
    a = ap.parse_args()

    s = load_settings()
    cache = ROOT / s.system["paths"]["cache_dir"]
    f = cache / f"{a.symbol}.csv"
    if not f.exists():
        sys.exit(f"No cached bars for {a.symbol} - run scripts/fetch_history.py first")
    daily = pd.read_csv(f, parse_dates=["date"], index_col="date").sort_index()
    if a.date:
        daily = daily.loc[:pd.Timestamp(a.date)]
    if daily.empty:
        sys.exit(f"No bars for {a.symbol} on or before {a.date}")

    import json
    levels_cfg = json.loads((ROOT / "config" / "levels.json").read_text())[a.timeframe]

    zones = support_resistance(daily, a.timeframe, levels_cfg)
    current_price = float(daily["close"].iloc[-1])
    as_of = daily.index[-1].date()

    print(f"{a.symbol} | {a.timeframe} | as of {as_of} | current price {current_price:.2f}\n")
    if not zones:
        print("No zones found - not enough history, or no pivots clustered to min_touches yet "
              f"(lookback_bars={levels_cfg.get('lookback_bars')}, min_touches={levels_cfg['min_touches']}).")
        return

    print(f"{'ZONE':>20s}   {'STRENGTH':>8s}   ROLE (relative to current price)")
    near = nearest_levels(zones, current_price)
    for z in zones:
        role = ("<- NEAREST SUPPORT" if near["support"] is z else
               "<- NEAREST RESISTANCE" if near["resistance"] is z else
               "support" if z.mid < current_price else "resistance")
        print(f"{z.price_low:9.2f} - {z.price_high:8.2f}   {z.strength:8d}   {role}")

    print(f"\n{len(zones)} zone(s) total.")
    if near["support"]:
        s_ = near["support"]
        print(f"Nearest support:    {s_.price_low:.2f}-{s_.price_high:.2f} "
              f"(strength {s_.strength}, {(current_price - s_.mid) / current_price * 100:.1f}% below price)")
    if near["resistance"]:
        r_ = near["resistance"]
        print(f"Nearest resistance: {r_.price_low:.2f}-{r_.price_high:.2f} "
              f"(strength {r_.strength}, {(r_.mid - current_price) / current_price * 100:.1f}% above price)")


if __name__ == "__main__":
    main()