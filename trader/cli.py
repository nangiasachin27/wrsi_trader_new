from __future__ import annotations

import argparse
import logging
import os
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from trader.brokers.paper import PaperBroker
from trader.brokers.upstox_live import UpstoxLiveBroker
from trader.core.calendar import TradingCalendar
from trader.core.config import ROOT, load_settings
from trader.data.historical import load_csv_dir
from trader.data.upstox import UpstoxProvider
from trader.engine.orchestrator import Engine
from trader.portfolio.state import StateStore


def _engine(s) -> Engine:
    sysc = s.system
    store = StateStore(ROOT / sysc["paths"]["state_dir"])
    cal = TradingCalendar({date.fromisoformat(h) for h in s.holidays})
    provider = UpstoxProvider(sysc["data"], sysc["timezone"], s.universe["instruments"],
                              ROOT / sysc["paths"]["cache_dir"])
    if sysc["mode"] == "live":
        if os.environ.get("ALLOW_LIVE") != "1":
            raise SystemExit("mode=live requires ALLOW_LIVE=1")
        broker = UpstoxLiveBroker()
    else:
        broker = PaperBroker(s.fees, sysc["execution"]["slippage_bps"])
    return Engine(s, provider, broker, store, cal)


def main() -> None:
    ap = argparse.ArgumentParser(prog="trader")
    ap.add_argument("cmd", choices=["eod", "open", "backtest", "status", "montecarlo"])
    ap.add_argument("--date", help="YYYY-MM-DD (default: today in configured timezone)")
    ap.add_argument("--start"), ap.add_argument("--end")
    ap.add_argument("--data-dir", help="CSV dir for backtest (default: cache_dir)")
    ap.add_argument("--strategy", help="backtest/eod/open: comma-separated strategy name(s) to run in "
                                       "ISOLATION, overriding every strategy config's own 'enabled' flag for "
                                       "this run only (nothing is written back to the json files). Use this to "
                                       "backtest each new swing strategy separately and compare, e.g. "
                                       "--strategy breakout_trend, then --strategy pullback_ma, etc. Omit to "
                                       "run whatever combination each config's own 'enabled' says (unchanged "
                                       "default behavior).")
    ap.add_argument("--label", help="backtest: suffix for the output folder, e.g. --label strategyA_only "
                                    "-> state/backtest_<start>_<end>_strategyA_only/. Use this whenever you "
                                    "run the same date range more than once (e.g. comparing strategies in "
                                    "isolation) - without it, a second run silently overwrites the first.")
    ap.add_argument("--force", action="store_true", help="run even on a non-trading day")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="show per-symbol scan/hold detail (DEBUG level), not just fired signals")
    ap.add_argument("--trade-log", help="montecarlo: path to a trade_log.csv from a prior backtest")
    ap.add_argument("--sims", type=int, default=2000, help="montecarlo: number of simulated paths")
    ap.add_argument("--mc-horizon", type=int,
                    help="montecarlo: trades per simulated path (bootstrap only; default = trades in the log)")
    ap.add_argument("--mc-method", choices=["bootstrap", "reshuffle"], default="bootstrap",
                    help="montecarlo: 'bootstrap' resamples with replacement (can extrapolate horizon); "
                         "'reshuffle' permutes the real trades only (sequence risk, same trade count)")
    ap.add_argument("--ruin-pct", type=float, default=50.0,
                    help="montecarlo: %% equity drop from the start counted as 'ruin' (default 50)")
    ap.add_argument("--mc-seed", type=int, help="montecarlo: fix the random seed for reproducible output")
    ap.add_argument("--by-strategy", action="store_true",
                    help="montecarlo: also run a separate simulation per strategy, not just combined")
    ap.add_argument("--include-open", action="store_true",
                    help="montecarlo: include positions still open at backtest end as mark-to-market "
                         "pseudo-trades (at their last recorded price) in the resampling pool - their "
                         "REAL eventual P&L is still unknown and may differ from this snapshot")
    a = ap.parse_args()

    s = load_settings()
    if a.strategy:
        wanted = {n.strip() for n in a.strategy.split(",") if n.strip()}
        unknown = wanted - set(s.strategies)
        if unknown:
            raise SystemExit(f"--strategy: unknown strategy name(s) {sorted(unknown)}; "
                             f"known: {sorted(s.strategies)}")
        for name, cfg in s.strategies.items():
            cfg["enabled"] = name in wanted
    level = "DEBUG" if a.verbose else s.system["logging"]["level"]
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")
    today = datetime.now(ZoneInfo(s.system["timezone"])).date()
    d = date.fromisoformat(a.date) if a.date else today

    if a.cmd == "backtest":
        if not a.start or not a.end:
            raise SystemExit("backtest requires --start YYYY-MM-DD --end YYYY-MM-DD")
        from trader.backtest.runner import run_backtest, summarize, trade_log
        data = load_csv_dir(a.data_dir or ROOT / s.system["paths"]["cache_dir"])
        start, end = date.fromisoformat(a.start), date.fromisoformat(a.end)
        suffix = f"_{a.label}" if a.label else ""
        work_dir = ROOT / s.system["paths"]["state_dir"] / f"backtest_{a.start}_{a.end}{suffix}"
        if work_dir.exists():
            import shutil
            shutil.rmtree(work_dir)  # backtests must start from base_capital, never inherit a prior run's state
        eq, trades = run_backtest(s, data, start, end, str(work_dir))
        tl = trade_log(trades)

        print(f"\n{'='*90}\nBACKTEST {a.start} -> {a.end}  ({len(data)} symbols loaded)\n{'='*90}")
        for k, v in summarize(eq, trades).items():
            print(f"  {k:18s}: {v}")

        if not tl.empty:
            print(f"\n{len(tl)} closed trades:")
            print(tl.to_string(index=False))
            tl.to_csv(work_dir / "trade_log.csv", index=False)
            print(f"\nTrade log saved to {work_dir / 'trade_log.csv'}")
            print(f"Equity curve saved to {work_dir / 'daily_equity.csv'}")

            losers = tl[tl["pnl"] < 0].copy()
            if not losers.empty:
                losers = losers.rename(columns={"avg_cost": "entry_price"})
                cols = ["symbol", "entry_date", "entry_price", "exit_date", "exit_price",
                       "holding_days", "pnl", "pnl_pct"]
                print(f"\n{'-'*90}\n{len(losers)} LOSING trade(s):")
                print(losers[cols].to_string(index=False))
        else:
            print("\nNo trades closed in this window - check universe/date range, or that "
                  "data_cache has enough pre-start history for the indicators to warm up "
                  "(min_weekly_bars/min_monthly_bars in the strategy config).")

        pf = StateStore(str(work_dir)).load_portfolio(s.risk["capital"]["base_capital"])
        if pf.positions:
            unreal = sum((p.last_price - p.avg_cost) * p.qty for p in pf.positions.values())
            print(f"\n{len(pf.positions)} position(s) STILL OPEN at {a.end} "
                  f"(unrealized P&L: {unreal:+,.0f} - this is NOT in the closed-trade stats above, "
                  f"but IS reflected in cagr_pct/max_dd_pct since those use the full equity curve):")
            print(f"  {'symbol':12s} {'qty':>6s} {'avg_cost':>10s} {'last_price':>10s} {'unrealized':>12s} {'entry_date':>12s}")
            for p in pf.positions.values():
                print(f"  {p.symbol:12s} {p.qty:6d} {p.avg_cost:10.2f} {p.last_price:10.2f} "
                      f"{(p.last_price - p.avg_cost) * p.qty:+12,.0f} {p.opened:>12s}")
        else:
            print(f"\nNo positions open at {a.end} - closed-trade stats above are the complete picture.")
        return
    if a.cmd == "montecarlo":
        if not a.trade_log:
            raise SystemExit("montecarlo requires --trade-log path/to/trade_log.csv (from a prior backtest run)")
        import numpy as np
        import pandas as pd
        from trader.backtest.monte_carlo import (equity_fan_chart, open_positions_summary,
                                                  percentile_rank, run_monte_carlo)

        tl = pd.read_csv(a.trade_log)
        if "pnl" not in tl.columns or tl.empty:
            raise SystemExit(f"{a.trade_log} has no 'pnl' column or no rows - nothing to simulate")
        starting_capital = s.risk["capital"]["base_capital"]
        out_dir = Path(a.trade_log).parent

        # trade_log.csv is ONLY closed trades - a position still open when the backtest ended
        # never appears in it. Surface that gap every time, unconditionally: it's easy to read
        # a grim "probability of loss" number and not realize open positions (possibly very
        # profitable ones) were never part of the simulation at all.
        open_pf = open_positions_summary(out_dir / "portfolio.json")
        if open_pf["n_open"]:
            print(f"\n{'!' * 90}\nNOTE: {open_pf['n_open']} position(s) were still OPEN when this backtest "
                  f"ended - trade_log.csv (and therefore this Monte Carlo) does NOT include them.")
            print(f"Unrealized P&L in those open positions (mark-to-market at last recorded price): "
                  f"{open_pf['unrealized_pnl']:+,.0f}")
            for p in open_pf["positions"]:
                print(f"  {p['symbol']:12s} qty {p['qty']:>6} avg {p['avg_cost']:>10.2f} "
                      f"last {p['last_price']:>10.2f}  unrealized {p['unrealized_pnl']:+,.0f}")
            print(f"Pass --include-open to add these as mark-to-market pseudo-trades in the simulation "
                  f"below (their REAL eventual outcome is still unknown and may differ).\n{'!' * 90}")
        else:
            print("\n(no portfolio.json found next to this trade log, or no positions were open at "
                  "the end of that backtest - this simulation is the complete picture either way)")

        def _run_and_print(label: str, pnl: "pd.Series") -> None:
            pnl_values = pnl.to_numpy()
            note = ""
            if a.include_open and open_pf["n_open"]:
                pnl_values = np.concatenate([pnl_values, [p["unrealized_pnl"] for p in open_pf["positions"]]])
                note = f"  (+{open_pf['n_open']} open position(s) included as mark-to-market pseudo-trades)"
            realized_final = starting_capital + pnl.sum()
            full_final = realized_final + (open_pf["unrealized_pnl"] if not a.include_open else 0)
            res = run_monte_carlo(pnl_values, starting_capital, n_sims=a.sims, horizon=a.mc_horizon,
                                  method=a.mc_method, ruin_pct=a.ruin_pct, seed=a.mc_seed)
            print(f"\n{'=' * 90}\nMONTE CARLO: {label}{note}  "
                  f"({res['n_source_trades']} source trades, {res['n_sims']} sims, "
                  f"method={res['method']}, horizon={res['horizon']} trades/path)\n{'=' * 90}")
            print(f"  starting capital           : {starting_capital:,.0f}")
            print(f"  YOUR realized result        : {realized_final:,.0f}  "
                  f"(sits at the {percentile_rank(res, realized_final):.0f}th percentile of this simulation)")
            if open_pf["n_open"] and not a.include_open:
                print(f"  + unrealized (open, excluded above): {open_pf['unrealized_pnl']:+,.0f}  "
                      f"-> full actual equity: {full_final:,.0f}")
            fe = res["final_equity"]
            print(f"  simulated final equity : p5 {fe['p5']:,.0f}  p25 {fe['p25']:,.0f}  "
                  f"p50 {fe['p50']:,.0f}  p75 {fe['p75']:,.0f}  p95 {fe['p95']:,.0f}")
            dd = res["max_drawdown_pct"]
            print(f"  simulated max drawdown  : p5 {dd['p5']:.1f}%  p25 {dd['p25']:.1f}%  "
                  f"p50 {dd['p50']:.1f}%  p75 {dd['p75']:.1f}%  p95 {dd['p95']:.1f}%")
            print(f"  probability of a loss   : {res['prob_loss_pct']:.1f}%")
            print(f"  probability of ruin     : {res['prob_ruin_pct']:.1f}%  "
                  f"(equity ever falling {res['ruin_threshold_pct']:.0f}% below the starting capital)")
            fan = equity_fan_chart(res)
            fan_path = out_dir / f"monte_carlo_{label.replace(' ', '_')}.csv"
            fan.to_csv(fan_path)
            print(f"  fan-chart data saved to {fan_path}")

        if a.by_strategy and "strategy" in tl.columns and tl["strategy"].notna().any():
            for strat_name, grp in tl.groupby("strategy"):
                _run_and_print(strat_name, grp["pnl"])
        _run_and_print("combined_portfolio", tl["pnl"])
        return
    if a.cmd == "status":
        pf = StateStore(ROOT / s.system["paths"]["state_dir"]).load_portfolio(s.risk["capital"]["base_capital"])
        print(f"cash {pf.cash:,.0f} | equity {pf.equity():,.0f} | positions {len(pf.positions)}")
        for p in pf.positions.values():
            print(f"  {p.symbol:12s} qty {p.qty:6d} avg {p.avg_cost:9.2f} stop {p.stop:9.2f} adds {p.adds}")
        return

    eng = _engine(s)
    if not a.force and not eng.cal.is_trading_day(d):
        print(f"{d} is not a trading day - nothing to do")
        return

    if a.cmd == "eod":
        orders = eng.run_eod(d)
        if orders:
            print(f"\n{len(orders):d} pending order(s) for next session:")
            for o in orders:
                print(f"  {o.action:5s} {o.symbol:12s} exec {o.exec_date}  "
                      f"stop {o.stop or 0:8.2f}  {o.reason}")
    else:
        eng.run_open(d)


if __name__ == "__main__":
    main()