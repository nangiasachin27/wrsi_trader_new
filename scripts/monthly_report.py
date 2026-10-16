"""
Monthly performance report from a backtest's trade_log.csv and daily_equity.csv: a table of
trades/win-rate/realized P&L per month, the all-time equity peak, and the worst drawdown window
(peak date/value -> trough date/value). Saves a two-panel chart (equity + running peak +
drawdown shading on top, monthly realized P&L bars on bottom) and prints the monthly table to
stdout.

    python scripts/monthly_report.py state/backtest_.../trade_log.csv state/backtest_.../daily_equity.csv
    python scripts/monthly_report.py trade_log.csv daily_equity.csv --out monthly_report.png
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")  # headless - saves to file, doesn't need a display
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trade_log", help="path to trade_log.csv")
    ap.add_argument("daily_equity", help="path to daily_equity.csv")
    ap.add_argument("--out", default="monthly_report.png", help="output chart path")
    a = ap.parse_args()

    tl = pd.read_csv(a.trade_log, parse_dates=["entry_date", "exit_date"])
    eq = pd.read_csv(a.daily_equity, parse_dates=["date"]).set_index("date").sort_index()

    # ---- monthly realized P&L / trade stats, attributed to the month a trade CLOSED ------
    tl["month"] = tl["exit_date"].dt.to_period("M")
    monthly = tl.groupby("month").agg(
        trades=("pnl", "size"),
        wins=("pnl", lambda s: (s > 0).sum()),
        pnl=("pnl", "sum"),
    )
    monthly["win_rate_pct"] = (monthly["wins"] / monthly["trades"] * 100).round(1)

    # ---- equity curve, running peak, drawdown ---------------------------------------------
    eq["peak"] = eq["equity"].cummax()
    eq["dd_pct"] = (eq["equity"] - eq["peak"]) / eq["peak"] * 100

    peak_val = eq["equity"].max()
    peak_date = eq["equity"].idxmax()
    trough_idx = eq["dd_pct"].idxmin()
    trough_val = eq.loc[trough_idx, "equity"]
    dd_pct = eq.loc[trough_idx, "dd_pct"]
    dd_peak_date = eq.loc[:trough_idx, "equity"].idxmax()  # the peak this worst drawdown fell FROM
    dd_peak_val = eq.loc[dd_peak_date, "equity"]

    print("=" * 78)
    print(f"EQUITY: start {eq['equity'].iloc[0]:,.0f} -> end {eq['equity'].iloc[-1]:,.0f} "
          f"({(eq['equity'].iloc[-1] / eq['equity'].iloc[0] - 1) * 100:+.1f}%)")
    print(f"All-time high: {peak_val:,.0f} on {peak_date.date()}")
    print(f"Worst drawdown: {dd_pct:.1f}%  (from {dd_peak_val:,.0f} on {dd_peak_date.date()} "
          f"to {trough_val:,.0f} on {trough_idx.date()})")
    print("=" * 78)

    print(f"\n{'MONTH':9s} {'TRADES':>7s} {'WINRATE%':>9s} {'REALIZED PNL':>14s}")
    for m, row in monthly.iterrows():
        print(f"{str(m):9s} {int(row.trades):7d} {row.win_rate_pct:8.1f}% {row.pnl:14,.0f}")
    print(f"{'TOTAL':9s} {int(monthly.trades.sum()):7d} "
          f"{monthly.wins.sum() / monthly.trades.sum() * 100:8.1f}% {monthly.pnl.sum():14,.0f}")

    # ---- chart: equity + running peak + drawdown shading (top), monthly P&L bars (bottom) --
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(eq.index, eq["equity"], color="#2563eb", linewidth=1.3, label="Equity")
    ax1.plot(eq.index, eq["peak"], color="#94a3b8", linewidth=0.8, linestyle="--", label="Running peak")
    ax1.scatter([peak_date], [peak_val], color="#16a34a", zorder=5, s=50,
               label=f"All-time high ({peak_date.date()})")
    ax1.scatter([trough_idx], [trough_val], color="#dc2626", zorder=5, s=50,
               label=f"Worst drawdown trough ({dd_pct:.1f}%)")
    ax1.set_title("Equity curve")
    ax1.set_ylabel("Equity")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.fill_between(eq.index, eq["dd_pct"], 0, color="#dc2626", alpha=0.12)
    ax2.set_ylabel("Drawdown %")
    ax2.set_ylim(eq["dd_pct"].min() * 1.3, 5)

    months = monthly.index.to_timestamp()
    colors = ["#16a34a" if v >= 0 else "#dc2626" for v in monthly["pnl"]]
    ax3.bar(months, monthly["pnl"], width=20, color=colors)
    ax3.axhline(0, color="black", linewidth=0.6)
    ax3.set_title("Monthly realized P&L")
    ax3.set_ylabel("P&L")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.grid(alpha=0.3)
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(a.out, dpi=140)
    print(f"\nChart saved to {a.out}")


if __name__ == "__main__":
    main()