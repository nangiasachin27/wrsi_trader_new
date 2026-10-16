"""
Pass a stock to this script and get back which candlestick pattern(s) match its candle on a
given timeframe - the one you'd check when deciding whether an entry candle looks bullish,
bearish, or something more specific (marubozu, morning star, bullish harami, ...).

    python scripts/candlestick.py RELIANCE --timeframe weekly
    python scripts/candlestick.py RELIANCE --timeframe daily --date 2026-09-22
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trader.core.candlestick import net_sentiment, patterns_for
from trader.core.config import ROOT, load_settings
from trader.core.indicators import candles_for_timeframe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol")
    ap.add_argument("--timeframe", choices=["daily", "weekly", "monthly"], default="daily")
    ap.add_argument("--date", help="YYYY-MM-DD - the candle ending on/before this date (default: latest)")
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

    pattern_cfg = json.loads((ROOT / "config" / "candlestick_patterns.json").read_text())
    bars = candles_for_timeframe(daily, a.timeframe)
    if bars.empty:
        sys.exit(f"No {a.timeframe} bars available for {a.symbol}")
    matches = patterns_for(daily, a.timeframe, pattern_cfg)
    bar_end = bars.index[-1].date()
    o, h, l, c = (float(bars[col].iloc[-1]) for col in ("open", "high", "low", "close"))

    print(f"{a.symbol} | {a.timeframe} candle ending {bar_end} | O {o:.2f} H {h:.2f} L {l:.2f} C {c:.2f}\n")
    if not matches:
        print("No registered pattern matched this candle.")
        return

    print(f"{'PATTERN':20s} {'DIRECTION':>10s} {'WEIGHT':>7s} {'BARS':>5s}")
    for m in sorted(matches, key=lambda m: -m.weight):
        print(f"{m.name:20s} {m.direction:>10s} {m.weight:7.1f} {m.bars_used:5d}")
    print(f"\nnet_sentiment: {net_sentiment(matches):+.1f} "
          f"({'bullish bias' if net_sentiment(matches) > 0 else 'bearish bias' if net_sentiment(matches) < 0 else 'neutral'})")


if __name__ == "__main__":
    main()