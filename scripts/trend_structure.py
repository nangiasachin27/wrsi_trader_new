"""
Course-based swing-structure market-trend classifier (UPTREND / SIDEWAYS / DOWNTREND) run
against a benchmark index's daily bars - see trader/core/trend_structure.py for the full rules.
Separate from scripts/market_trend.py, which still runs the older IBD-style VD-day/FTD engine.

Also saves a visual-test chart: the benchmark's own price curve (NIFTY500 by default - this
classifies market trend off an index, there's no "equity curve" here since no strategy or
backtest is involved) with the background shaded by classified state (green=UPTREND,
red=DOWNTREND, gray=SIDEWAYS), the confirmed swing highs/lows marked, and the ratcheting
protecting level drawn - everything needed to eyeball whether a transition looks right.

--start only trims what's PRINTED/PLOTTED, not what's classified: the classifier always runs
over the FULL cached history up to --date/--end first, so swings/states near your --start date
have correct left-context (prior swings to compare against) rather than being misclassified
from a cold "initial_state" start. Use --date (or the new --end, same thing) for the upper
bound and --start for the lower bound of the WINDOW you want to look at.

    python scripts/trend_structure.py
    python scripts/trend_structure.py --benchmark NIFTY500 --date 2026-09-30
    python scripts/trend_structure.py --start 2026-01-01 --end 2026-09-22
    python scripts/trend_structure.py --history 20
    python scripts/trend_structure.py --out test/trend_structure_test.png
    python scripts/trend_structure.py --no-plot
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
from trader.core.levels import pivot_highs, pivot_lows
from trader.core.trend_structure import classify_trend_structure

_STATE_COLOR = {"UPTREND": "#16a34a", "DOWNTREND": "#dc2626", "SIDEWAYS": "#9ca3af"}


def _plot(bars: pd.DataFrame, out: pd.DataFrame, cfg: dict, symbol: str, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")  # headless - saves to file, doesn't need a display
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(14, 7))

    # ---- background shading: one axvspan per contiguous run of the same state -------------
    states = out["state"]
    run_start = states.index[0]
    for i in range(1, len(states) + 1):
        changed = i == len(states) or states.iloc[i] != states.iloc[i - 1]
        if changed:
            run_end = states.index[i] if i < len(states) else states.index[-1]
            ax.axvspan(run_start, run_end, color=_STATE_COLOR[states.iloc[i - 1]], alpha=0.12, linewidth=0)
            if i < len(states):
                run_start = states.index[i]

    # ---- price line -------------------------------------------------------------------------
    ax.plot(bars.index, bars["close"], color="#111827", linewidth=1.1, label=f"{symbol} close")

    # ---- confirmed swing highs/lows (same pivot detection trend_structure.py itself uses) --
    left, right = cfg.get("pivot_left", 2), cfg.get("pivot_right", 2)
    ph = pivot_highs(bars["high"], left, right)
    pl = pivot_lows(bars["low"], left, right)
    ax.scatter(bars.index[ph], bars.loc[ph, "high"], marker="^", color="#2563eb", s=28,
              zorder=5, label="Confirmed swing high")
    ax.scatter(bars.index[pl], bars.loc[pl, "low"], marker="v", color="#ea580c", s=28,
              zorder=5, label="Confirmed swing low")

    # ---- the ratcheting protecting level - where a break of market structure would fire ----
    level = out["structure_level"].astype(float)
    ax.step(out.index, level, where="post", color="#7c3aed", linewidth=1.0, linestyle="--",
            label="Protecting level (structure_level)")

    ax.set_title(f"{symbol} | trend_structure classification ({cfg.get('timeframe', 'daily')} bars, "
                f"pivot {left}/{right}, min_confirmed_swings={cfg.get('min_confirmed_swings', 2)})")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.3)

    legend_handles = ax.get_legend_handles_labels()[0] + [
        Line2D([0], [0], color=_STATE_COLOR[s], alpha=0.3, linewidth=10) for s in ("UPTREND", "SIDEWAYS", "DOWNTREND")
    ]
    legend_labels = ax.get_legend_handles_labels()[1] + ["UPTREND", "SIDEWAYS", "DOWNTREND"]
    ax.legend(legend_handles, legend_labels, loc="upper left", fontsize=8, ncol=2)

    fig.autofmt_xdate()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"\nChart saved to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", help="override config/trend_structure.json's benchmark_symbol")
    ap.add_argument("--date", "--end", dest="date",
                    help="YYYY-MM-DD - classify (and plot/print up to) this date (default: latest cached bar)")
    ap.add_argument("--start", help="YYYY-MM-DD - only PRINT/PLOT from this date onward; classification "
                                    "still runs over the full history before it, so states near --start "
                                    "have correct context instead of starting cold")
    ap.add_argument("--history", type=int, default=10, help="how many recent transition events to print "
                                                             "(within --start.. if given)")
    ap.add_argument("--out", default="test/trend_structure_test.png",
                    help="chart output path (default: test/trend_structure_test.png)")
    ap.add_argument("--no-plot", action="store_true", help="skip the chart, print-only")
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

    out, events = classify_trend_structure(daily, cfg)   # ALWAYS the full history up to --date
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

    start_ts = pd.Timestamp(a.start) if a.start else None
    events_in_window = [e for e in events if start_ts is None or pd.Timestamp(e.date) >= start_ts]
    window_label = f" within {a.start}..{as_of}" if a.start else ""
    print(f"\nLast {a.history} transition(s){window_label}:")
    for ev in events_in_window[-a.history:]:
        print(f"  {ev.date}  {ev.from_state:>9s} -> {ev.to_state:<9s}  {ev.reason}")
    if not events_in_window:
        print("  (none in this window)" if a.start else "  (none in this history)")

    if not a.no_plot:
        timeframe = cfg.get("timeframe", "daily")
        bars = daily if timeframe == "daily" else candles_for_timeframe(
            daily, timeframe, cfg.get("weekly_rule", "W-FRI"), cfg.get("monthly_rule", "ME"))
        out_display, bars_display = out, bars
        if start_ts is not None:
            out_display = out.loc[start_ts:]
            bars_display = bars.loc[start_ts:]
            if out_display.empty:
                sys.exit(f"--start {a.start} is after the last available bar ({as_of}) - nothing to plot")
        _plot(bars_display, out_display, cfg, symbol, Path(a.out))


if __name__ == "__main__":
    main()