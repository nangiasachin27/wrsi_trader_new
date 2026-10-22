"""
Runs ONE strategy's backtest once for every on/off combination of a chosen set of filters (a
full 2^N grid), over the SAME loaded universe data, ranking results by CAGR (ties broken by
smaller drawdown). This version adds three things on top of the original:

  - INCREMENTAL RESULTS: every combo's row is appended to tune_results.csv the moment that
    combo finishes, not held in memory until the whole sweep ends - so results are never lost
    if you stop the run partway through (Ctrl+C, closing the terminal, the machine sleeping).

  - RESUME (--resume): re-running the exact same --strategy/--start/--end/--filters with
    --resume skips every combo whose row is already in that CSV and only computes what's left.
    Interrupt it and resume it as many times as you like. Running it again WITHOUT --resume or
    --fresh refuses to proceed if that CSV already has rows, so you can't accidentally mix
    results from two different runs without saying so explicitly.

  - PARALLEL WORKERS (--parallel N): runs N combos at once across N worker processes instead
    of one at a time. Each worker loads the universe data ONCE when it starts (a few seconds,
    not per combo) and reuses it for every combo that worker is given - so this scales close to
    linearly with N up to roughly your CPU's core count, not N times the data-loading cost.

    python scripts/tune_strategy.py --strategy breakout_trend --start 2022-01-01 --end 2024-12-31 \
        --filters resistance,quality,sector_regime --parallel 4
    python scripts/tune_strategy.py --strategy breakout_trend --start 2022-01-01 --end 2024-12-31 \
        --filters resistance,quality,sector_regime --parallel 4 --resume   # continue after Ctrl+C
    python scripts/tune_strategy.py --strategy weekly_rsi_sma20 --start 2022-01-01 --end 2023-12-31 \
        --filters resistance,quality --out test/tune_weekly_rsi.png

By default every filter under config/filters/ is varied - with N filters that's 2^N FULL
backtests, so a safety cap (--max-combos, default 32) refuses to run more than that unless you
pass --force. Narrowing --filters to the 2-4 filters you actually suspect matter keeps a run
fast and the results table readable; each extra filter doubles the run count - that's still the
single biggest lever on wall-clock time, --parallel just lets you spend fewer wall-clock hours
on whatever count you do decide to run.

Pick --parallel based on your machine's core count (e.g. 4-6 on a typical laptop). Each worker
is a full backtest process with its own memory - going far past your physical core count won't
make it faster and can thrash memory/CPU instead.
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trader.backtest.runner import run_backtest, summarize
from trader.core.config import ROOT, load_settings
from trader.data.historical import load_csv_dir

_DATA = None   # set once per WORKER PROCESS by _init_worker (loaded fresh in each worker, a few
               # seconds), then reused for every combo that worker runs - never reloaded per combo.


def _init_worker(data_dir: str) -> None:
    global _DATA
    _DATA = load_csv_dir(Path(data_dir))


def _combo_label(filter_names: list[str], combo: tuple[bool, ...]) -> str:
    return "_".join(f"{name}{'ON' if v else 'OFF'}" for name, v in zip(filter_names, combo))


def _run_one(task: tuple) -> dict:
    """Runs inside a worker process. Takes only small, plain-picklable args - never the loaded
    universe data itself, which lives in that worker's own _DATA global (see _init_worker)."""
    label, strategy, filter_names, combo, base_settings, start, end, work_dir = task
    settings = copy.deepcopy(base_settings)
    for name, cfg in settings.strategies.items():
        cfg["enabled"] = (name == strategy)
    for name, v in zip(filter_names, combo):
        settings.filters[name]["enabled"] = v
    try:
        eq, trades = run_backtest(settings, _DATA, start, end, work_dir)
    except Exception as e:
        return {"label": label, "combo": combo, "ok": False, "error": f"{type(e).__name__}: {e}"}
    if eq.empty or len(eq) < 2:
        return {"label": label, "combo": combo, "ok": False, "error": "no equity curve (0 bars?)"}
    return {"label": label, "combo": combo, "ok": True, "stats": summarize(eq, trades)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", required=True, help="strategy name to tune, e.g. breakout_trend")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--filters", help="comma-separated filter names to toggle on/off "
                                      "(default: every filter under config/filters/)")
    ap.add_argument("--max-combos", type=int, default=32,
                    help="refuse to run more than this many combinations unless --force (default 32 = 2^5)")
    ap.add_argument("--force", action="store_true", help="run even if combos exceed --max-combos")
    ap.add_argument("--data-dir", help="CSV dir to backtest against (default: cache_dir)")
    ap.add_argument("--out", help="chart path for the top combos' equity curves "
                                  "(default: test/tune_<strategy>.png)")
    ap.add_argument("--no-plot", action="store_true", help="skip the chart")
    ap.add_argument("--top", type=int, default=5, help="how many best combos to plot/print in full (default 5)")
    ap.add_argument("--parallel", type=int, default=1,
                    help="number of combos to run at once, across worker processes (default 1 = sequential)")
    ap.add_argument("--resume", action="store_true",
                    help="reuse an existing run's results folder, skip combos already completed "
                         "in it, and only run what's left")
    ap.add_argument("--fresh", action="store_true",
                    help="wipe any existing results for this exact --strategy/--start/--end and start over")
    a = ap.parse_args()

    base = load_settings()
    if a.strategy not in base.strategies:
        sys.exit(f"Unknown strategy '{a.strategy}'; known: {sorted(base.strategies)}")

    filter_names = ([x.strip() for x in a.filters.split(",") if x.strip()]
                    if a.filters else sorted(base.filters))
    unknown = set(filter_names) - set(base.filters)
    if unknown:
        sys.exit(f"Unknown filter name(s) {sorted(unknown)}; known: {sorted(base.filters)}")

    n_combos = 2 ** len(filter_names)
    if n_combos > a.max_combos and not a.force:
        sys.exit(f"{len(filter_names)} filters -> {n_combos} combinations, over --max-combos "
                 f"({a.max_combos}). Narrow --filters, raise --max-combos, or pass --force "
                 f"(each extra filter DOUBLES the run count).")

    data_dir = Path(a.data_dir) if a.data_dir else ROOT / base.system["paths"]["cache_dir"]
    state_root = ROOT / base.system["paths"]["state_dir"] / f"tune_{a.strategy}_{a.start}_{a.end}"
    csv_path = state_root / "tune_results.csv"
    fieldnames = ["combo"] + [f"filter.{n}" for n in filter_names] + \
        ["cagr_pct", "max_dd_pct", "closed_trades", "win_rate_pct", "profit_factor",
         "avg_holding_days", "total_fees"]

    if a.fresh and state_root.exists():
        shutil.rmtree(state_root)

    done: dict[str, dict] = {}
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        if list(existing.columns) != fieldnames:
            sys.exit(f"{csv_path} exists from a run with different filters/columns. Pass --fresh "
                     f"to discard it and start over, or point --filters at the same set used before.")
        if not a.resume:
            sys.exit(f"{csv_path} already has {len(existing)} result(s) from a previous run. "
                     f"Pass --resume to continue it, or --fresh to discard it and start over.")
        for _, row in existing.iterrows():
            done[row["combo"]] = row.to_dict()
        print(f"Resuming: {len(done)}/{n_combos} combination(s) already completed in {csv_path}")

    state_root.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(a.start).date()
    end = pd.Timestamp(a.end).date()

    tasks = []
    for combo in itertools.product([False, True], repeat=len(filter_names)):
        label = _combo_label(filter_names, combo)
        if label in done:
            continue
        tasks.append((label, a.strategy, filter_names, combo, base, start, end, str(state_root / label)))

    print(f"Tuning '{a.strategy}' over {a.start}..{a.end}: {n_combos} combination(s) of "
         f"{', '.join(filter_names)} ({len(done)} already done, {len(tasks)} left to run), "
         f"{max(1, a.parallel)} worker(s)\n"
         f"(every other filter stays exactly as configured; only these {len(filter_names)} are toggled)\n")

    write_header = not csv_path.exists()
    completed_count = len(done)
    newly_written = 0

    def _write_row(label: str, combo: tuple[bool, ...], stats: dict) -> None:
        nonlocal write_header
        row = {"combo": label, **{f"filter.{n}": v for n, v in zip(filter_names, combo)}, **stats}
        with open(csv_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
                write_header = False
            w.writerow(row)

    if not tasks:
        print("Nothing left to run - every combination is already in the results file.")
    else:
        executor = ProcessPoolExecutor(max_workers=max(1, a.parallel),
                                       initializer=_init_worker, initargs=(str(data_dir),))
        try:
            futures = {executor.submit(_run_one, t): t[0] for t in tasks}
            for fut in as_completed(futures):
                label = futures[fut]
                completed_count += 1
                try:
                    result = fut.result()
                except Exception as e:
                    print(f"  [{completed_count}/{n_combos}] {label} ... CRASHED: {type(e).__name__}: {e}")
                    continue
                if not result["ok"]:
                    print(f"  [{completed_count}/{n_combos}] {label} ... {result['error']} - skipped")
                    continue
                stats = result["stats"]
                _write_row(label, result["combo"], stats)
                newly_written += 1
                print(f"  [{completed_count}/{n_combos}] {label} ... CAGR {stats['cagr_pct']:+.1f}%  "
                     f"maxDD {stats['max_dd_pct']:.1f}%  trades {stats['closed_trades']}")
        except KeyboardInterrupt:
            print(f"\nInterrupted - {newly_written} new result(s) written this run (plus {len(done)} "
                 f"from before), all saved in {csv_path}.")
            print("Re-run the exact same command with --resume to pick up where this left off.")
            executor.shutdown(wait=False, cancel_futures=True)
            sys.exit(130)
        executor.shutdown(wait=True)

    if not csv_path.exists():
        sys.exit("\nNo combination produced a usable equity curve - nothing to rank.")

    results = pd.read_csv(csv_path).sort_values(["cagr_pct", "max_dd_pct"], ascending=[False, False]).reset_index(drop=True)
    print(f"\n{'=' * 100}\nRANKED RESULTS ({len(results)} combination(s), best first) - saved to {csv_path}\n{'=' * 100}")
    display_cols = ["combo", "cagr_pct", "max_dd_pct", "closed_trades", "win_rate_pct", "profit_factor", "total_fees"]
    print(results[display_cols].to_string(index=False))

    best = results.iloc[0]
    print(f"\nBEST: {best['combo']}  ->  CAGR {best['cagr_pct']:+.1f}%, max drawdown {best['max_dd_pct']:.1f}%, "
         f"{int(best['closed_trades'])} closed trades")

    if len(results) < n_combos:
        print(f"\n({n_combos - len(results)} combination(s) not yet run or skipped - re-run with "
             f"--resume to fill in the rest.)")

    if not a.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top_labels = results["combo"].head(a.top).tolist()
        curves = {}
        for label in top_labels:
            eq_path = state_root / label / "daily_equity.csv"
            if eq_path.exists():
                curves[label] = pd.read_csv(eq_path, parse_dates=["date"], index_col="date")["equity"]

        fig, ax = plt.subplots(figsize=(13, 7))
        colors = plt.cm.viridis([i / max(len(top_labels) - 1, 1) for i in range(len(top_labels))])
        for label, color in zip(top_labels, colors):
            if label not in curves:
                continue
            eq = curves[label]
            row = results[results["combo"] == label].iloc[0]
            ax.plot(eq.index, eq.values, color=color, linewidth=1.3,
                   label=f"{label} ({row['cagr_pct']:+.1f}% CAGR, {row['max_dd_pct']:.1f}% DD)")
        ax.set_title(f"{a.strategy}: top {len(top_labels)} filter combo(s) by CAGR, {a.start} -> {a.end}")
        ax.set_ylabel("Equity")
        ax.legend(loc="upper left", fontsize=7)
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        out_path = Path(a.out) if a.out else Path(f"test/tune_{a.strategy}.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140)
        print(f"\nChart (top {len(top_labels)}) saved to {out_path}")


if __name__ == "__main__":
    main()