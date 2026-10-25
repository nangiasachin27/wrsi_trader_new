"""
Runs ONE strategy's backtest over many on/off filter combinations, over the SAME loaded
universe data, ranking results by CAGR (ties broken by smaller drawdown). Two modes:

  --mode full (default - UNCHANGED from the original version of this script, byte-for-byte
    the same behavior, CLI flags, CSV schema, resume/parallel mechanics and output): tries
    EVERY combination of a chosen set of filters (a full 2^N grid). Guaranteed to find the
    true best combo in that grid, including combos where two filters only help TOGETHER (an
    interaction effect a greedy search can miss) - but cost doubles with every extra filter,
    so N=7-8 filters means 128-256 full backtests.

  --mode smart (NEW - greedy forward selection): starts from every named filter OFF, tries
    turning ON each remaining filter ONE AT A TIME on top of whatever is already locked in,
    keeps whichever single addition improves the ranking metric (CAGR, then smaller drawdown
    as tiebreak - same ordering --mode full sorts by) the most, locks it in, and repeats over
    the filters still remaining. Stops the moment no remaining filter improves on the current
    best. Cost is roughly N + (N-1) + (N-2) + ... down to however many rounds it takes to stop
    (at most N*(N+1)/2, far below 2^N for N>=5) - but it CAN miss a combo that only helps as a
    pair/triple and does nothing or even hurts alone, since it never reconsiders a filter once
    a round passes it over. Writes the exact same row schema as --mode full into a SEPARATE
    results file (tune_results_smart.csv, alongside full mode's tune_results.csv in the same
    state_root) so you can run both on the same --strategy/--start/--end and directly diff the
    two result sets - which is the recommended way to decide whether smart search is trustworthy
    enough to rely on for a given strategy before skipping straight to it for the others.

  - INCREMENTAL RESULTS: every combo's row is appended to its CSV the moment that combo
    finishes, not held in memory until the whole run ends - results are never lost if you stop
    partway through (Ctrl+C, closing the terminal, the machine sleeping).

  - RESUME (--resume): re-running the exact same --strategy/--start/--end/--filters/--mode with
    --resume picks up where a previous run of THAT MODE left off (full mode: skips combos
    already in its CSV; smart mode: reloads its last completed round from a small state file
    and continues the search). Running again WITHOUT --resume or --fresh refuses to proceed if
    that mode's results already exist, so you can't accidentally mix two different runs without
    saying so explicitly. The two modes never collide with or resume each other's progress.

  - PARALLEL WORKERS (--parallel N): runs N backtests at once across N worker processes.
    Each worker loads the universe data ONCE when it starts (a few seconds, not per combo) and
    reuses it for every combo that worker is given. In --mode smart, this parallelizes WITHIN
    each round (every remaining filter's trial that round is independent of the others), not
    across rounds (each round's winner has to be known before the next round's trials are
    built) - so --parallel still helps, just capped at however many filters remain that round.

    python scripts/tune_strategy.py --strategy breakout_trend --start 2022-01-01 --end 2024-12-31 \
        --filters resistance,quality,sector_regime --parallel 4
    python scripts/tune_strategy.py --strategy breakout_trend --start 2022-01-01 --end 2024-12-31 \
        --filters resistance,quality,sector_regime --parallel 4 --resume   # continue after Ctrl+C
    python scripts/tune_strategy.py --strategy sr_reversal --start 2021-01-01 --end 2026-09-22 \
        --mode smart --parallel 4   # greedy search, same strategy/range as a --mode full run
    python scripts/tune_strategy.py --strategy weekly_rsi_sma20 --start 2022-01-01 --end 2023-12-31 \
        --filters resistance,quality --out test/tune_weekly_rsi.png

By default every filter under config/filters/ is varied - with N filters that's 2^N FULL
backtests in --mode full, so a safety cap (--max-combos, default 32) refuses to run more than
that unless you pass --force. This cap does NOT apply to --mode smart, since its cost grows
quadratically, not exponentially, and is reported up front before it starts.

Pick --parallel based on your machine's core count (e.g. 4-6 on a typical laptop). Each worker
is a full backtest process with its own memory - going far past your physical core count won't
make it faster and can thrash memory/CPU instead.
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
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

_RESULT_FIELDS = ["cagr_pct", "max_dd_pct", "closed_trades", "win_rate_pct", "profit_factor",
                  "avg_holding_days", "total_fees"]
_RANK_KEYS = ["cagr_pct", "max_dd_pct"]  # higher is better for both (max_dd_pct is a negative
                                         # number, e.g. -19.79, so "higher" = less negative = a
                                         # smaller drawdown) - same convention used everywhere
                                         # below, including --mode smart's round-over-round
                                         # "did this improve" comparison.


def _init_worker(data_dir: str) -> None:
    global _DATA
    _DATA = load_csv_dir(Path(data_dir))


def _combo_label(filter_names: list[str], combo: tuple[bool, ...]) -> str:
    return "_".join(f"{name}{'ON' if v else 'OFF'}" for name, v in zip(filter_names, combo))


def _rank_tuple(stats: dict) -> tuple[float, float]:
    return (stats["cagr_pct"], stats["max_dd_pct"])


def _run_one(task: tuple) -> dict:
    """Runs inside a worker process. Takes only small, plain-picklable args - never the loaded
    universe data itself, which lives in that worker's own _DATA global (see _init_worker)."""
    label, strategy, filter_names, combo, base_settings, start, end, work_dir = task
    settings = copy.deepcopy(base_settings)
    for name, cfg in settings.strategies.items():
        cfg["enabled"] = (name == strategy)
    for name, v in zip(filter_names, combo):
        settings.filters[name]["enabled"] = v
    # StateStore.load_portfolio()/append_equity() read and append to whatever already sits in
    # work_dir (portfolio.json, daily_equity.csv, journal.jsonl) rather than starting fresh -
    # so if this exact combo label's directory is left over from an earlier invocation (e.g.
    # a prior --mode full run in the same state_root, or an interrupted previous attempt),
    # run_backtest() would silently resume from that stale state instead of a clean start. Wipe
    # it first, every time, so every backtest - in either mode, --fresh or --resume, first run
    # or re-run - always starts from a truly empty StateStore. Mirrors trader/cli.py's own
    # backtest command, which does the same rmtree before run_backtest().
    wd = Path(work_dir)
    if wd.exists():
        shutil.rmtree(wd)
    try:
        eq, trades = run_backtest(settings, _DATA, start, end, work_dir)
    except Exception as e:
        return {"label": label, "combo": combo, "ok": False, "error": f"{type(e).__name__}: {e}"}
    if eq.empty or len(eq) < 2:
        return {"label": label, "combo": combo, "ok": False, "error": "no equity curve (0 bars?)"}
    return {"label": label, "combo": combo, "ok": True, "stats": summarize(eq, trades)}


def _fieldnames(filter_names: list[str]) -> list[str]:
    return ["combo"] + [f"filter.{n}" for n in filter_names] + _RESULT_FIELDS


def _csv_writer(csv_path: Path, filter_names: list[str]):
    """Returns a _write_row(label, combo, stats) closure that appends one row to csv_path the
    moment it's called, writing the header on first use - shared by both modes so a result is
    on disk (never just in memory) the instant a backtest finishes, exactly like the original
    --mode full behavior."""
    fieldnames = _fieldnames(filter_names)
    state = {"write_header": not csv_path.exists()}

    def _write_row(label: str, combo: tuple[bool, ...], stats: dict) -> None:
        row = {"combo": label, **{f"filter.{n}": v for n, v in zip(filter_names, combo)}, **stats}
        with open(csv_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            if state["write_header"]:
                w.writeheader()
                state["write_header"] = False
            w.writerow(row)

    return _write_row


def _load_done(csv_path: Path, filter_names: list[str]) -> dict[str, dict]:
    if not csv_path.exists():
        return {}
    existing = pd.read_csv(csv_path)
    fieldnames = _fieldnames(filter_names)
    if list(existing.columns) != fieldnames:
        sys.exit(f"{csv_path} exists from a run with different filters/columns. Pass --fresh "
                 f"to discard it and start over, or point --filters at the same set used before.")
    return {row["combo"]: row.to_dict() for _, row in existing.iterrows()}


# ============================================================================================
# --mode full: UNCHANGED from the original script - same CSV, same combos, same everything.
# ============================================================================================

def run_full(a, base, filter_names: list[str], start, end, data_dir: Path, state_root: Path) -> None:
    n_combos = 2 ** len(filter_names)
    if n_combos > a.max_combos and not a.force:
        sys.exit(f"{len(filter_names)} filters -> {n_combos} combinations, over --max-combos "
                 f"({a.max_combos}). Narrow --filters, raise --max-combos, or pass --force "
                 f"(each extra filter DOUBLES the run count).")

    csv_path = state_root / "tune_results.csv"
    fieldnames = _fieldnames(filter_names)

    if a.fresh and csv_path.exists():
        csv_path.unlink()

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

    _write_row = _csv_writer(csv_path, filter_names)
    completed_count = len(done)
    newly_written = 0

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

    results = pd.read_csv(csv_path).sort_values(_RANK_KEYS, ascending=[False, False]).reset_index(drop=True)
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
        _plot_top(results, a, state_root, title=f"{a.strategy}: top {{n}} filter combo(s) by CAGR, {a.start} -> {a.end}",
                  default_out=f"test/tune_{a.strategy}.png")


# ============================================================================================
# --mode smart: greedy forward selection. Reuses _run_one/the same worker executor, but drives
# it round-by-round instead of submitting the whole 2^N grid at once.
# ============================================================================================

def run_smart(a, base, filter_names: list[str], start, end, data_dir: Path, state_root: Path) -> None:
    csv_path = state_root / "tune_results_smart.csv"
    state_path = state_root / "tune_smart_state.json"

    if a.fresh:
        for p in (csv_path, state_path):
            if p.exists():
                p.unlink()

    done = _load_done(csv_path, filter_names)

    if state_path.exists():
        if not a.resume:
            sys.exit(f"{state_path} already has a smart search in progress. Pass --resume to "
                     f"continue it, or --fresh to discard it and start over.")
        st = json.loads(state_path.read_text())
        locked = {n: bool(st["locked"][n]) for n in filter_names}
        decided = list(st["decided"])
        best_score = tuple(st["best_score"]) if st["best_score"] is not None else None
        round_num = st["round"]
        print(f"Resuming smart search for '{a.strategy}': round {round_num}, "
             f"already locked in: {decided or '(none yet)'}")
    else:
        locked = {n: False for n in filter_names}
        decided: list[str] = []
        best_score = None
        round_num = 0

    def _save_state() -> None:
        state_path.write_text(json.dumps({"locked": locked, "decided": decided,
                                          "best_score": list(best_score) if best_score else None,
                                          "round": round_num}, indent=2))

    _write_row = _csv_writer(csv_path, filter_names)

    def _combo_of(locked_map: dict[str, bool]) -> tuple[bool, ...]:
        return tuple(locked_map[n] for n in filter_names)

    def _score_combo(locked_map: dict[str, bool]) -> tuple[float, float] | None:
        """Runs (or reuses an already-done row for) exactly one combo, sequentially - used only
        for the one-off baseline point, not inside a parallel round."""
        combo = _combo_of(locked_map)
        label = _combo_label(filter_names, combo)
        if label in done:
            r = done[label]
            return (r["cagr_pct"], r["max_dd_pct"])
        task = (label, a.strategy, filter_names, combo, base, start, end, str(state_root / label))
        _init_worker(str(data_dir))
        result = _run_one(task)
        if not result["ok"]:
            print(f"  baseline {label} ... {result['error']} - cannot continue")
            return None
        stats = result["stats"]
        _write_row(label, combo, stats)
        done[label] = {"combo": label, **{f"filter.{n}": v for n, v in zip(filter_names, combo)}, **stats}
        print(f"  baseline {label} ... CAGR {stats['cagr_pct']:+.1f}%  maxDD {stats['max_dd_pct']:.1f}%  "
             f"trades {stats['closed_trades']}")
        return _rank_tuple(stats)

    max_rounds_possible = len(filter_names) * (len(filter_names) + 1) // 2
    print(f"Smart search for '{a.strategy}' over {a.start}..{a.end} across {len(filter_names)} filter(s): "
         f"{', '.join(filter_names)} - up to {max_rounds_possible} backtest(s) worst case "
         f"(vs {2 ** len(filter_names)} for the equivalent --mode full grid), "
         f"{max(1, a.parallel)} worker(s) per round\n")

    if best_score is None:
        print("Scoring baseline (every filter OFF):")
        best_score = _score_combo(locked)
        if best_score is None:
            sys.exit("Baseline combo failed to produce a usable equity curve - nothing to search from.")
        round_num = 0
        _save_state()

    remaining = [n for n in filter_names if n not in decided]

    while remaining:
        round_num += 1
        print(f"\nRound {round_num}: trying {len(remaining)} candidate(s) on top of "
             f"{decided or '(baseline)'} ...")

        trial_tasks = []
        trial_label_by_cand = {}
        preloaded_scores: dict[str, tuple[float, float]] = {}
        for cand in remaining:
            trial_locked = dict(locked)
            trial_locked[cand] = True
            combo = _combo_of(trial_locked)
            label = _combo_label(filter_names, combo)
            trial_label_by_cand[cand] = label
            if label in done:
                r = done[label]
                preloaded_scores[cand] = (r["cagr_pct"], r["max_dd_pct"])
            else:
                trial_tasks.append((cand, label, combo))

        round_scores: dict[str, tuple[float, float]] = dict(preloaded_scores)
        for cand, score in preloaded_scores.items():
            print(f"  {trial_label_by_cand[cand]} ... (already done) CAGR {score[0]:+.1f}%  maxDD {score[1]:.1f}%")

        if trial_tasks:
            executor = ProcessPoolExecutor(max_workers=max(1, a.parallel),
                                           initializer=_init_worker, initargs=(str(data_dir),))
            try:
                futures = {}
                for cand, label, combo in trial_tasks:
                    task = (label, a.strategy, filter_names, combo, base, start, end, str(state_root / label))
                    futures[executor.submit(_run_one, task)] = cand
                for fut in as_completed(futures):
                    cand = futures[fut]
                    label = trial_label_by_cand[cand]
                    try:
                        result = fut.result()
                    except Exception as e:
                        print(f"  {label} ... CRASHED: {type(e).__name__}: {e} - treated as no improvement")
                        continue
                    if not result["ok"]:
                        print(f"  {label} ... {result['error']} - treated as no improvement")
                        continue
                    stats = result["stats"]
                    _write_row(label, result["combo"], stats)
                    done[label] = {"combo": label,
                                   **{f"filter.{n}": v for n, v in zip(filter_names, result["combo"])}, **stats}
                    round_scores[cand] = _rank_tuple(stats)
                    print(f"  {label} ... CAGR {stats['cagr_pct']:+.1f}%  maxDD {stats['max_dd_pct']:.1f}%  "
                         f"trades {stats['closed_trades']}")
            except KeyboardInterrupt:
                print(f"\nInterrupted mid-round. Re-run the exact same command with --resume - "
                     f"the last COMPLETED round is saved in {state_path}, this partial round is not.")
                executor.shutdown(wait=False, cancel_futures=True)
                sys.exit(130)
            executor.shutdown(wait=True)

        if not round_scores:
            print("  No candidate in this round produced a usable result - stopping search.")
            break

        best_cand = max(round_scores, key=lambda c: round_scores[c])
        cand_score = round_scores[best_cand]

        if cand_score > best_score:
            locked[best_cand] = True
            decided.append(best_cand)
            best_score = cand_score
            remaining.remove(best_cand)
            _save_state()
            print(f"  -> {best_cand} ON improves to CAGR {cand_score[0]:+.1f}% / maxDD {cand_score[1]:.1f}% "
                 f"- locking it in.")
        else:
            print(f"  -> best candidate this round ({best_cand}, CAGR {cand_score[0]:+.1f}% / "
                 f"maxDD {cand_score[1]:.1f}%) does not improve on the current best "
                 f"(CAGR {best_score[0]:+.1f}% / maxDD {best_score[1]:.1f}%) - stopping search.")
            break

    print(f"\n{'=' * 100}\nSMART SEARCH RESULT for '{a.strategy}' - saved to {csv_path}\n{'=' * 100}")
    final_combo = _combo_of(locked)
    final_label = _combo_label(filter_names, final_combo)
    print(f"Adopted path: {decided or '(none - baseline was already best)'}")
    print(f"BEST: {final_label}  ->  CAGR {best_score[0]:+.1f}%, max drawdown {best_score[1]:.1f}%")
    print(f"({len(done)} backtest(s) run in total this search, vs {2 ** len(filter_names)} for the "
         f"equivalent --mode full grid)")

    if not a.no_plot:
        results = pd.read_csv(csv_path).sort_values(_RANK_KEYS, ascending=[False, False]).reset_index(drop=True)
        _plot_top(results, a, state_root,
                  title=f"{a.strategy}: smart-search top {{n}} combo(s) tried, {a.start} -> {a.end}",
                  default_out=f"test/tune_{a.strategy}_smart.png")


def _plot_top(results: pd.DataFrame, a, state_root: Path, title: str, default_out: str) -> None:
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
    ax.set_title(title.format(n=len(top_labels)))
    ax.set_ylabel("Equity")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path = Path(a.out) if a.out else Path(default_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"\nChart (top {len(top_labels)}) saved to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", required=True, help="strategy name to tune, e.g. breakout_trend")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--mode", choices=["full", "smart"], default="full",
                    help="'full' (default): every on/off combination (2^N), unchanged from the "
                         "original script. 'smart': greedy forward selection - far fewer runs, "
                         "can miss a combo that only helps as a pair/triple. See the module "
                         "docstring (--help) for the full tradeoff.")
    ap.add_argument("--filters", help="comma-separated filter names to toggle on/off "
                                      "(default: every filter under config/filters/)")
    ap.add_argument("--max-combos", type=int, default=32,
                    help="--mode full only: refuse to run more than this many combinations "
                         "unless --force (default 32 = 2^5)")
    ap.add_argument("--force", action="store_true", help="--mode full only: run even if combos exceed --max-combos")
    ap.add_argument("--data-dir", help="CSV dir to backtest against (default: cache_dir)")
    ap.add_argument("--out", help="chart path for the top combos' equity curves "
                                  "(default: test/tune_<strategy>.png, or _smart.png in --mode smart)")
    ap.add_argument("--no-plot", action="store_true", help="skip the chart")
    ap.add_argument("--top", type=int, default=5, help="how many best combos to plot/print in full (default 5)")
    ap.add_argument("--parallel", type=int, default=1,
                    help="number of backtests to run at once, across worker processes (default 1 = sequential)")
    ap.add_argument("--resume", action="store_true",
                    help="reuse an existing run's results for THIS --mode, skip what's already "
                         "completed, and only run what's left")
    ap.add_argument("--fresh", action="store_true",
                    help="wipe any existing results for this exact --strategy/--start/--end/--mode and start over")
    a = ap.parse_args()

    base = load_settings()
    if a.strategy not in base.strategies:
        sys.exit(f"Unknown strategy '{a.strategy}'; known: {sorted(base.strategies)}")

    filter_names = ([x.strip() for x in a.filters.split(",") if x.strip()]
                    if a.filters else sorted(base.filters))
    unknown = set(filter_names) - set(base.filters)
    if unknown:
        sys.exit(f"Unknown filter name(s) {sorted(unknown)}; known: {sorted(base.filters)}")

    data_dir = Path(a.data_dir) if a.data_dir else ROOT / base.system["paths"]["cache_dir"]
    state_root = ROOT / base.system["paths"]["state_dir"] / f"tune_{a.strategy}_{a.start}_{a.end}"
    state_root.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(a.start).date()
    end = pd.Timestamp(a.end).date()

    if a.mode == "full":
        run_full(a, base, filter_names, start, end, data_dir, state_root)
    else:
        run_smart(a, base, filter_names, start, end, data_dir, state_root)


if __name__ == "__main__":
    main()