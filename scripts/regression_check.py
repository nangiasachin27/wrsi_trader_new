"""Regression check: re-run the existing strategies with temporary in-memory config overrides
(config files are never edited) and print the summary plus a per-strategy table, for comparison
against a known baseline.

Default = the 2026-10-31 reference run: 2022-01-01 -> 2023-12-31, the 5 enabled strategies,
candlestick filter ON (rank mode), sentiment filter OFF - the config as it stood then. Expected:
163 closed trades, net closed P&L 549,665, CAGR 44.81%, PF 1.86 (if data_cache is unchanged).

  python scripts/regression_check.py                       # 31-Oct reference config
  python scripts/regression_check.py --as-is               # today's config files, untouched
  python scripts/regression_check.py --set sentiment.enabled=false --set candlestick.enabled=true

Each run also saves equity_vs_<benchmark>.png in its output folder: the run's equity curve vs the
benchmark index scaled to the same starting capital, on one axis (--benchmark NIFTY50 by default,
falling back to NIFTY500 if NIFTY50 bars are not cached; --no-plot to skip; needs matplotlib).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trader.backtest.runner import run_backtest, summarize, trade_log  # noqa: E402
from trader.core.config import ROOT, load_settings  # noqa: E402
from trader.data.historical import load_csv_dir  # noqa: E402

REFERENCE = ["candlestick.enabled=true", "sentiment.enabled=false"]


def apply(settings, spec: str) -> None:
    """filter_or_strategy.key=value, value parsed as JSON (true/false/numbers/"strings").
    Dotted keys reach into nested config dicts (2026-11-15), e.g.
    --set sr_reversal.entry.proximity_pct=1.5 sets strategies["sr_reversal"]["entry"]["proximity_pct"]."""
    path, raw = spec.split("=", 1)
    name, key = path.split(".", 1)
    target = settings.filters.get(name) or settings.strategies.get(name)
    if target is None:
        raise SystemExit(f"--set {spec}: no filter or strategy called {name!r}")
    try:
        val = json.loads(raw)
    except json.JSONDecodeError:
        val = raw
    *parents, leaf = key.split(".")
    for k in parents:
        target = target.setdefault(k, {})
        if not isinstance(target, dict):
            raise SystemExit(f"--set {spec}: {name}.{k} is not a dict")
    target[leaf] = val


def save_equity_plot(eq, data: dict, benchmark: str, out: Path, title: str,
                     start_capital: float) -> Path | None:
    """One chart, one axis: the run's equity curve vs the benchmark index scaled to the same
    starting capital. Returns the written path, or None (missing benchmark bars / matplotlib),
    never raising - a plot problem must not fail the regression run itself."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ImportError:
        print("  plot skipped      : matplotlib not installed (pip install matplotlib)")
        return None

    e = eq["equity"]
    bench_sym = next((b for b in (benchmark, "NIFTY 50", "NIFTY500") if b in data), None)
    if bench_sym is None:
        print(f"  plot skipped      : no cached bars for {benchmark!r} (or NIFTY500) in data_cache -"
              f" add it to universe.json instruments and run scripts/fetch_history.py")
        return None
    if bench_sym != benchmark:
        print(f"  plot note         : no {benchmark!r} bars cached - using {bench_sym} instead")
    bench = data[bench_sym]["close"].loc[e.index[0]:e.index[-1]].dropna()
    if len(bench) < 2:
        print(f"  plot skipped      : {bench_sym} has no bars inside the run window")
        return None
    scaled = bench * (float(e.iloc[0]) / float(bench.iloc[0]))

    years = max((e.index[-1] - e.index[0]).days / 365.25, 1e-9)
    cagr = lambda x: ((float(x.iloc[-1]) / float(x.iloc[0])) ** (1 / years) - 1) * 100  # noqa: E731
    ink, muted, grid = "#0b0b0b", "#52514e", "#e6e5e0"
    sys_col, bench_col = "#1a7f5a", "#8a8781"

    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(e.index, e, color=sys_col, lw=1.6, label=f"System  {cagr(e):+.1f}%/yr")
    ax.plot(scaled.index, scaled, color=bench_col, lw=1.2,
            label=f"{bench_sym} (scaled)  {cagr(scaled):+.1f}%/yr")
    for series, col in ((e, sys_col), (scaled, bench_col)):
        ax.annotate(f"{float(series.iloc[-1]):,.0f}", (series.index[-1], float(series.iloc[-1])),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=col)
    ax.grid(axis="y", color=grid, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(grid)
    ax.tick_params(colors=muted, labelsize=9)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e5:,.1f}L"))
    ax.set_ylabel("Equity (\u20b9 lakh)", color=muted, fontsize=9)
    ax.set_title(f"Equity vs {bench_sym} \u2014 {title}", loc="left", fontsize=11,
                 color=ink, fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    png = out / f"equity_vs_{bench_sym.replace(' ', '_')}.png"
    fig.savefig(png, dpi=120)
    plt.close(fig)
    return png


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--as-is", action="store_true", help="no overrides: current config files")
    ap.add_argument("--set", action="append", default=[], help="filter_or_strategy.key=value (repeatable)")
    ap.add_argument("--data-dir")
    ap.add_argument("--label", default=None)
    ap.add_argument("--benchmark", default="NIFTY50",
                    help="index symbol for the equity-vs-benchmark plot (falls back to NIFTY500)")
    ap.add_argument("--no-plot", action="store_true", help="skip the equity-vs-benchmark plot")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    s = load_settings()
    for spec in ([] if a.as_is else REFERENCE) + a.set:
        apply(s, spec)
    enabled = sorted(n for n, c in s.strategies.items() if c["enabled"])
    filters = sorted(n for n, c in s.filters.items() if c.get("enabled"))

    data = load_csv_dir(Path(a.data_dir) if a.data_dir else ROOT / s.system["paths"]["cache_dir"])
    label = a.label or ("as_is" if a.as_is else "reference")
    out = ROOT / s.system["paths"]["state_dir"] / f"regression_{a.start}_{a.end}_{label}"
    if out.exists():
        import shutil
        shutil.rmtree(out)
    eq, trades = run_backtest(s, data, date.fromisoformat(a.start), date.fromisoformat(a.end), str(out))
    tl = trade_log(trades)
    tl.to_csv(out / "trade_log.csv", index=False)

    print(f"\nREGRESSION {a.start} -> {a.end}  [{label}]")
    print(f"  strategies: {', '.join(enabled)}")
    print(f"  filters on: {', '.join(filters)}")
    for k, v in summarize(eq, trades, s.risk["capital"]["base_capital"]).items():
        print(f"  {k:18s}: {v}")
    print(f"  closed P&L (sum)  : {tl['pnl'].sum():,.0f}" if not tl.empty else "  no closed trades")
    if not tl.empty:
        g = tl.groupby("strategy")["pnl"]
        per = (g.agg(trades="count", net="sum")
               .assign(win_pct=g.apply(lambda x: round((x > 0).mean() * 100, 1)),
                       pf=g.apply(lambda x: round(x[x > 0].sum() / -x[x < 0].sum(), 2) if (x < 0).any() else None))
               .round({"net": 0}))
        print("\n" + per.to_string())
    if not a.no_plot:
        png = save_equity_plot(eq, data, a.benchmark, out,
                               f"{a.start} \u2192 {a.end}  [{label}]", s.risk["capital"]["base_capital"])
        if png:
            print(f"  plot              : {png.name}")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()