"""
Trace why a symbol was (or wasn't) picked in a given window, gate by gate:

  1. CORE RULE   - the real strategy.entry_signal() (RSI cross, monthly RSI, close > SMA ...).
                   The columns beside it show WHICH sub-condition failed on weeks it didn't fire.
  2. FILTERS     - every enabled filter (market regime, quality, resistance, sector ...) is run
                   for real on weeks the core rule fired, with its pass/fail, score and reason.
  3. WHAT THE BACKTEST ACTUALLY DID - read from the backtest's own records: BUY/SELL fills,
                   and journal events (ENTRY_FILTERED, ENTRY_SKIPPED: no free slot, gap-down,
                   sizing ...). Slot competition and Monday open checks can't be replayed from
                   price history alone, so those come from what really happened in the run.

    python scripts/inspect_entry.py SOLARINDS --start 2026-03-01 --end 2026-05-15
    python scripts/inspect_entry.py SOLARINDS --start 2026-03-01 --end 2026-05-15 ^
           --backtest-dir state/backtest_2026-02-01_2026-09-22
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trader.core.calendar import TradingCalendar
from trader.core.config import ROOT, load_settings
from trader.core.indicators import rsi, sma
from trader.data.historical import HistoricalProvider
from trader.filters.base import build_filter
from trader.strategies.base import SymbolContext, build_strategy


def _read_bars(cache: Path, name: str) -> pd.DataFrame | None:
    f = cache / f"{name}.csv"
    if not f.exists():
        return None
    return pd.read_csv(f, parse_dates=["date"], index_col="date").sort_index()


def _benchmarks(filter_cfgs: dict) -> set[str]:
    names = set()
    for c in filter_cfgs.values():
        if c.get("benchmark_symbol"):
            names.add(c["benchmark_symbol"])
        names.update(c.get("industry_to_benchmark", {}).values())
    return names


def _gates(cfg: dict, ctx: SymbolContext) -> dict | None:
    """Explain the core rule's sub-conditions (mirrors the strategy's own config keys)."""
    e, ind = cfg["entry"], cfg["indicators"]
    w = ctx.weekly(cfg["timeframes"]["weekly_rule"], e["weekly_bar"] != "completed")
    m = ctx.monthly(cfg["timeframes"]["monthly_rule"], e["monthly_include_partial"])
    if len(w) < e["min_weekly_bars"] or len(m) < e["min_monthly_bars"]:
        return None
    wr, mr = rsi(w["close"], ind["rsi_period"]), rsi(m["close"], ind["rsi_period"])
    n = e.get("sustain_weeks", 1)
    thr = e["weekly_rsi_threshold"]
    crossed = bool((wr.iloc[-n:] > thr).all()) and bool(wr.iloc[-(n + 1)] <= thr)
    s_now, px = sma(w["close"], ind["sma_period"]).iloc[-1], float(w["close"].iloc[-1])
    return {
        "close": px, "wrsi": float(wr.iloc[-1]), "mrsi": float(mr.iloc[-1]), "sma": float(s_now),
        "crossed": crossed,
        "m_ok": float(mr.iloc[-1]) >= e["monthly_rsi_threshold"] - e["monthly_rsi_near_tolerance"],
        "above": bool(px > s_now), "stop_pct": (px - s_now) / px * 100.0,
    }


def _backtest_records(bt_dir: Path, symbol: str, lo: str, hi: str) -> None:
    print(f"\n{'=' * 100}\nWHAT THE BACKTEST ACTUALLY DID  ({bt_dir})\n{'=' * 100}")
    pf = bt_dir / "portfolio.json"
    if pf.exists():
        trades = [t for t in json.loads(pf.read_text())["trades"] if t["symbol"] == symbol]
        print(f"\n{symbol} fills across the WHOLE backtest (was a position already open in your window?):")
        if not trades:
            print("  none - this symbol was never bought in this run")
        for t in sorted(trades, key=lambda t: t["date"]):
            extra = f"  pnl {t['pnl']:+,.0f}" if "pnl" in t else f"  ({t.get('action', '')})"
            print(f"  {t['date']}  {t['side']:4s} qty {t['qty']:>5} @ {t['price']:>10.2f}{extra}")
    else:
        print(f"  no portfolio.json in {bt_dir}")

    jr = bt_dir / "journal.jsonl"
    if jr.exists():
        rows = [json.loads(l) for l in jr.read_text().splitlines() if l.strip()]
        rows = [r for r in rows if r.get("symbol") == symbol and lo <= str(r["date"])[:10] <= hi]
        print(f"\nJournal events for {symbol}, {lo} -> {hi}:")
        if not rows:
            print("  none - no filter block, slot skip, or open-check skip was recorded in this window")
        for r in rows:
            detail = r.get("why") or f"{r.get('filter', '')}: {r.get('reason', '')}"
            print(f"  {str(r['date'])[:10]}  {r['event']:15s} {detail}")
    else:
        print(f"  no journal.jsonl in {bt_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol")
    ap.add_argument("--strategy", default="weekly_rsi_sma20")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--backtest-dir", help="a state/backtest_* folder to read real fills/journal from")
    a = ap.parse_args()

    s = load_settings()
    cfg = s.strategies[a.strategy]
    strat = build_strategy(a.strategy, cfg)
    filters = {n: build_filter(n, c) for n, c in s.filters.items()}
    cache = ROOT / s.system["paths"]["cache_dir"]

    data = {}
    for name in {a.symbol} | _benchmarks(s.filters):
        df = _read_bars(cache, name)
        if df is not None:
            data[name] = df
    if a.symbol not in data:
        sys.exit(f"No cached bars for {a.symbol} in {cache} - run scripts/fetch_history.py first")
    provider = HistoricalProvider(data)
    # Same calendar the backtest uses: real trading days taken from the bars themselves. Using
    # holidays.json instead silently drops any week whose Friday was a holiday (week-end is then
    # the Thursday, which an empty holiday list doesn't know about).
    cal = TradingCalendar(set(), trading_days=[t.date() for t in data[a.symbol].index])

    on = [n for n, f in filters.items() if f.enabled]
    print(f"{a.symbol} | strategy {a.strategy} | window {a.start} -> {a.end}")
    print(f"filters ON: {', '.join(on) or 'none'}  |  OFF: {', '.join(n for n in filters if n not in on) or 'none'}")
    print("Entries are only evaluated on the LAST trading day of each week (completed weekly candle).\n")
    print(f"{'WEEK-END':11s} {'CLOSE':>9s} {'wRSI':>6s} {'mRSI':>6s} {'SMA20':>9s} "
          f"{'RSI-X':>6s} {'mRSI-OK':>8s} {'>SMA':>5s} {'STOP%':>6s}  CORE RULE")

    idx = data[a.symbol].index
    for ts in pd.bdate_range(a.start, a.end):
        d = ts.date()
        if ts not in idx or not cal.is_week_end(d):
            continue
        ctx = SymbolContext(a.symbol, provider.get_daily(a.symbol, d), d, cal)
        g = _gates(cfg, ctx)
        if g is None:
            print(f"{d}   (not enough history yet)")
            continue
        sig = strat.entry_signal(ctx)
        print(f"{d}  {g['close']:9.2f} {g['wrsi']:6.1f} {g['mrsi']:6.1f} {g['sma']:9.2f} "
              f"{'YES' if g['crossed'] else 'no':>6s} {'YES' if g['m_ok'] else 'no':>8s} "
              f"{'YES' if g['above'] else 'no':>5s} {g['stop_pct']:6.1f}  "
              f"{'>>> FIRED' if sig else 'not fired'}")
        if not sig:
            continue

        blocked = []
        for fname, filt in filters.items():
            if not filt.enabled:
                continue
            try:
                filt.prepare(d, provider)
                res = filt.evaluate(sig, ctx)
            except KeyError as exc:
                print(f"      filter {fname:14s} ERROR: no cached bars for {exc} - fetch that benchmark first")
                blocked.append(fname)
                continue
            print(f"      filter {fname:14s} {'pass' if res.passed else 'BLOCKED':7s} score {res.score:+.0f} | {res.reason}")
            if not res.passed:
                blocked.append(fname)
        print(f"      => {'BLOCKED by ' + ', '.join(blocked) if blocked else 'survives the filters (still needs a free slot + Monday open checks)'}")

    bt = Path(a.backtest_dir) if a.backtest_dir else None
    if bt is None:
        found = sorted((ROOT / s.system["paths"]["state_dir"]).glob("backtest_*"), key=lambda p: p.stat().st_mtime)
        bt = found[-1] if found else None
    if bt and bt.exists():
        _backtest_records(bt, a.symbol, a.start, (pd.Timestamp(a.end) + pd.Timedelta(days=7)).date().isoformat())
    else:
        print("\n(no backtest folder found - run a backtest to see fills/journal events)")


if __name__ == "__main__":
    main()