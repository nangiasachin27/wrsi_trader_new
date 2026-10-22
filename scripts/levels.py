"""
Pass a stock to this script and get back its support/resistance zones for a given timeframe,
plus (by default) a chart of price with those zones drawn as horizontal bands, so you can
visually sanity-check them the same way scripts/trend_structure.py plots its trend state.

    python scripts/levels.py RELIANCE --timeframe weekly
    python scripts/levels.py RELIANCE --timeframe daily --date 2026-09-22
    python scripts/levels.py RELIANCE --timeframe weekly --start 2025-01-01 --out test/reliance_levels.png
    python scripts/levels.py RELIANCE --timeframe weekly --no-plot

Zones are always computed over the FULL configured lookback_bars window (config/levels.json),
same as before - --start only narrows what's DRAWN, it never changes which pivots/zones get
found (same safeguard as trend_structure.py's --start, so a zoomed-in chart can't quietly
change the underlying classification).

Role (support vs. resistance) is ALWAYS relative to one reference price: the close on --date
(or the latest cached bar if --date is omitted) - NOT relative to where price was trading at
each point across the plotted window. A zone that price was ranging ABOVE for months can show
as "resistance" if price has since broken below it - that's not a misclassification, it's the
standard "broken support becomes resistance" flip. To see a zone's role as it stood at some
earlier point (e.g. during a consolidation, before a later breakdown), pass that date to
--date; --start then only trims the chart's x-axis, it does not change which bars feed the
role/zone calculation unless you also move --date.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trader.core.config import ROOT, load_settings
from trader.core.indicators import candles_for_timeframe
from trader.core.levels import Level, nearest_levels, support_resistance


def _plot(bars: pd.DataFrame, zones: list[Level], near: dict, current_price: float,
         symbol: str, timeframe: str, start: pd.Timestamp | None, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_bars = bars.loc[start:] if start is not None else bars
    if plot_bars.empty:
        plot_bars = bars  # --start cut off everything - fall back to full range rather than an empty chart

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(plot_bars.index, plot_bars["close"], color="black", linewidth=1.0, label="close", zorder=3)

    max_strength = max((z.strength for z in zones), default=1)
    for z in zones:
        is_nearest = (z is near["support"]) or (z is near["resistance"])
        role = "support" if z.mid < current_price else "resistance"
        color = "tab:green" if role == "support" else "tab:red"
        alpha = 0.12 + 0.28 * (z.strength / max_strength)  # stronger zone (more touches) = darker band
        ax.axhspan(z.price_low, z.price_high, facecolor=color, alpha=alpha, zorder=1,
                  linewidth=1.6 if is_nearest else 0,
                  edgecolor=color if is_nearest else "none")
        label_price = z.price_high if role == "resistance" else z.price_low
        tag = f"{role.upper()}{' (nearest)' if is_nearest else ''} {z.mid:.1f} x{z.strength}"
        ax.text(plot_bars.index[-1], label_price, f"  {tag}", fontsize=7, va="center",
                color=color, fontweight="bold" if is_nearest else "normal")
        touch_dates = [ts for ts, _ in z.touches if ts >= plot_bars.index[0]]
        touch_prices = [p for ts, p in z.touches if ts >= plot_bars.index[0]]
        if touch_dates:
            ax.scatter(touch_dates, touch_prices, color=color, s=28, zorder=4, marker="o",
                      edgecolor="black", linewidth=0.4)

    ax.axhline(current_price, color="tab:blue", linestyle="--", linewidth=1.0, alpha=0.8, zorder=2)
    ax.text(plot_bars.index[0], current_price, f"current {current_price:.1f}  ", fontsize=8,
           va="center", ha="right", color="tab:blue")

    ax.set_title(f"{symbol}: {timeframe} support/resistance zones, as of {plot_bars.index[-1].date()}")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"\nChart saved to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("symbol")
    ap.add_argument("--timeframe", choices=["daily", "weekly", "monthly"], default="weekly")
    ap.add_argument("--date", help="YYYY-MM-DD - analyze as of this date (default: latest cached bar). "
                                   "This is the reference price that decides each zone's support/resistance role.")
    ap.add_argument("--start", help="YYYY-MM-DD - only narrows what's PLOTTED, zones/role are still "
                                    "computed as of --date (or the latest bar) regardless")
    ap.add_argument("--out", help="chart path (default: test/levels_<symbol>_<timeframe>.png)")
    ap.add_argument("--no-plot", action="store_true", help="skip the chart, print only")
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

    if not a.no_plot:
        bars = candles_for_timeframe(daily, a.timeframe, levels_cfg.get("weekly_rule", "W-FRI"),
                                    levels_cfg.get("monthly_rule", "ME"))
        start_ts = pd.Timestamp(a.start) if a.start else None
        out_path = Path(a.out) if a.out else Path(f"test/levels_{a.symbol}_{a.timeframe}.png")
        _plot(bars, zones, near, current_price, a.symbol, a.timeframe, start_ts, out_path)


if __name__ == "__main__":
    main()