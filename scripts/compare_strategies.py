"""
Run several strategies' backtests in PARALLEL (each in its own isolated subprocess, via
`python -m trader.cli backtest --strategy <name> ...` - same isolation mechanism as running
them one at a time), then overlay all their equity curves on one chart alongside NIFTY500 (or
any other cached benchmark), rebased the same way scripts/monthly_report.py already does for a
single strategy - directly comparable since every strategy shares the same base_capital
(config/risk_config.json), so no further rebasing is needed between strategies, only for the
benchmark.

    python scripts/compare_strategies.py --start 2020-01-01 --end 2024-12-31
    python scripts/compare_strategies.py --start 2020-01-01 --end 2024-12-31 \
        --strategies weekly_rsi_sma20,breakout_trend,pullback_ma,sr_reversal,inside_bar_breakout
    python scripts/compare_strategies.py --start 2020-01-01 --end 2024-12-31 --workers 3
    python scripts/compare_strategies.py --start 2020-01-01 --end 2024-12-31 --no-benchmark
    python scripts/compare_strategies.py --start 2020-01-01 --end 2024-12-31 --out compare.png

Each strategy still runs through the normal --strategy isolation (overrides every OTHER
strategy's "enabled" to false for that subprocess only - nothing is written back to the JSON
configs), so this is exactly equivalent to running the --strategy commands from the README one
at a time, just in parallel and with the results plotted together afterward.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless - saves to file, doesn't need a display
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trader.core.config import ROOT, load_settings

# A fixed, readable color per strategy so repeated runs/comparisons stay visually consistent
# rather than matplotlib picking a different cycle order depending on which strategies are
# included this time.
_COLORS = {
    "weekly_rsi_sma20": "#2563eb", "breakout_trend": "#16a34a", "pullback_ma": "#9333ea",
    "sr_reversal": "#ea580c", "inside_bar_breakout": "#0891b2",
}
_FALLBACK_COLORS = ["#4b5563", "#be123c", "#65a30d", "#0369a1", "#a16207"]


def _run_one(strategy: str, start: str, end: str, label_suffix: str) -> tuple[str, Path | None, str]:
    """Runs one strategy's backtest in a fresh subprocess (own Python process, own state dir -
    no shared mutable state with any other strategy's run, so these are genuinely safe to run
    concurrently). Returns (strategy, work_dir or None on failure, stderr tail)."""
    label = f"{strategy}{label_suffix}"
    cmd = [sys.executable, "-m", "trader.cli", "backtest", "--strategy", strategy,
          "--start", start, "--end", end, "--label", label]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    s = load_settings()
    work_dir = ROOT / s.system["paths"]["state_dir"] / f"backtest_{start}_{end}_{label}"
    if proc.returncode != 0 or not (work_dir / "daily_equity.csv").exists():
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return strategy, None, tail[-1] if tail else f"exit code {proc.returncode}, no output captured"
    return strategy, work_dir, ""


def _load_benchmark(path: str, start_idx: pd.DatetimeIndex, start_capital: float) -> pd.Series | None:
    """Same rebasing scripts/monthly_report.py uses: the benchmark's close, reindexed to the
    strategies' own date range and rebased to the same starting capital, so it shares one
    meaningful y-axis with every equity curve rather than an incomparable index-level scale."""
    p = Path(path)
    if not p.exists():
        print(f"(benchmark file not found at {path} - skipping overlay)")
        return None
    bench = pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()["close"]
    bench = bench.loc[(bench.index >= start_idx.min()) & (bench.index <= start_idx.max())]
    if bench.empty:
        print(f"(benchmark {path} has no bars in this date range - skipping overlay)")
        return None
    bench = bench.reindex(start_idx, method="ffill").dropna()
    return bench / bench.iloc[0] * start_capital


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--strategies",
                    help="comma-separated strategy names to compare (default: every strategy "
                         "under config/strategies/)")
    ap.add_argument("--label-suffix", default="_compare",
                    help="appended to each strategy's own name for its work dir, e.g. "
                         "breakout_trend_compare -> state/backtest_<start>_<end>_breakout_trend_compare/ "
                         "(default: _compare, kept separate from any _only runs you did by hand)")
    ap.add_argument("--workers", type=int, help="max parallel backtests (default: one per strategy)")
    ap.add_argument("--benchmark", default="data_cache/NIFTY500.csv",
                    help="cached daily OHLC csv to overlay, rebased to the same starting "
                         "capital (default: data_cache/NIFTY500.csv)")
    ap.add_argument("--no-benchmark", action="store_true", help="skip the benchmark overlay")
    ap.add_argument("--out", default="compare_strategies.png", help="output chart path")
    a = ap.parse_args()

    s = load_settings()
    strategies = ([x.strip() for x in a.strategies.split(",") if x.strip()]
                 if a.strategies else sorted(s.strategies))
    unknown = set(strategies) - set(s.strategies)
    if unknown:
        sys.exit(f"Unknown strategy name(s) {sorted(unknown)}; known: {sorted(s.strategies)}")

    print(f"Running {len(strategies)} backtest(s) in parallel: {', '.join(strategies)}\n")
    results: dict[str, Path] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=a.workers or len(strategies)) as pool:
        futures = {pool.submit(_run_one, name, a.start, a.end, a.label_suffix): name for name in strategies}
        for fut in as_completed(futures):
            strategy, work_dir, err = fut.result()
            if work_dir is None:
                failures[strategy] = err
                print(f"  [FAILED]  {strategy:22s} {err}")
            else:
                results[strategy] = work_dir
                print(f"  [done]    {strategy:22s} -> {work_dir}")

    if not results:
        sys.exit("\nNo strategy produced a usable backtest - nothing to plot. See the [FAILED] lines above.")
    if failures:
        print(f"\n{len(failures)} strategy(ies) failed or produced no trades and are excluded "
              f"from the chart: {', '.join(failures)}")

    # ---- load each strategy's equity curve, and NIFTY500 (or whatever --benchmark names) ---
    curves: dict[str, pd.Series] = {}
    for name, work_dir in results.items():
        eq = pd.read_csv(work_dir / "daily_equity.csv", parse_dates=["date"]).set_index("date").sort_index()
        curves[name] = eq["equity"]

    common_start_capital = s.risk["capital"]["base_capital"]  # same for every strategy - no rebasing needed
    full_idx = pd.DatetimeIndex(sorted(set().union(*[c.index for c in curves.values()])))
    bench = None if a.no_benchmark else _load_benchmark(a.benchmark, full_idx, common_start_capital)

    # ---- summary table (printed + usable even without opening the chart) -------------------
    print(f"\n{'STRATEGY':22s} {'START':>12s} {'END':>12s} {'RETURN %':>10s} {'MAX DD %':>10s}")
    rows = []
    for name, eq in curves.items():
        ret = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
        dd = ((eq - eq.cummax()) / eq.cummax() * 100).min()
        rows.append((name, eq.iloc[0], eq.iloc[-1], ret, dd))
        print(f"{name:22s} {eq.iloc[0]:12,.0f} {eq.iloc[-1]:12,.0f} {ret:+10.1f} {dd:10.1f}")
    if bench is not None:
        bret = (bench.iloc[-1] / bench.iloc[0] - 1) * 100
        bdd = ((bench - bench.cummax()) / bench.cummax() * 100).min()
        print(f"{Path(a.benchmark).stem:22s} {bench.iloc[0]:12,.0f} {bench.iloc[-1]:12,.0f} "
              f"{bret:+10.1f} {bdd:10.1f}")

    # ---- the comparison chart ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 7))
    for i, (name, eq) in enumerate(sorted(curves.items())):
        color = _COLORS.get(name, _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)])
        ret = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
        ax.plot(eq.index, eq.values, color=color, linewidth=1.4, label=f"{name} ({ret:+.1f}%)")
    if bench is not None:
        bret = (bench.iloc[-1] / bench.iloc[0] - 1) * 100
        ax.plot(bench.index, bench.values, color="#9ca3af", linewidth=1.2, linestyle="--",
                label=f"{Path(a.benchmark).stem} ({bret:+.1f}%)")

    ax.set_title(f"Strategy comparison: {a.start} -> {a.end}")
    ax.set_ylabel(f"Equity (starting capital {common_start_capital:,.0f})")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(a.out, dpi=140)
    print(f"\nChart saved to {a.out}")


if __name__ == "__main__":
    main()