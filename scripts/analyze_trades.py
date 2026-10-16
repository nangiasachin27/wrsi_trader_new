"""
Diagnose a backtest trade_log.csv against the cached daily bars: for every trade, walk the
actual daily closes from entry to exit and report:
  - max adverse excursion (MAE): the worst intra-hold drawdown from avg_cost, vs. the pnl_pct
    the trade actually booked. If MAE and final pnl_pct are close, the exit fired promptly.
    If MAE is far worse than pnl_pct, that's fine (the exit protected you). If pnl_pct is far
    worse than any single day-to-day move implies, that's the anomaly signature below.
  - biggest single-session % move during the hold. A move beyond --anomaly-threshold (default
    25%) in one session is either real (circuit-filter news day) or an unadjusted corporate
    action / bad tick - this script can't tell you which, but it tells you exactly which
    session to go verify against a second source (NSE bhavcopy, a charting site) before you
    trust the trade's P&L.

    python scripts/analyze_trades.py state/backtest_2026-02-01_2026-09-22/trade_log.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trader.core.config import ROOT, load_settings


def load_corporate_actions() -> dict[str, list]:
    f = ROOT / "config" / "corporate_actions.json"
    if not f.exists():
        return {}
    import json
    events = json.loads(f.read_text())["events"]
    out: dict[str, list] = {}
    for e in events:
        out.setdefault(e["symbol"], []).append(pd.Timestamp(e["date"]))
    return out


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/analyze_trades.py <trade_log.csv>")
    tl = pd.read_csv(sys.argv[1], parse_dates=["entry_date", "exit_date"])
    s = load_settings()
    cache = ROOT / s.system["paths"]["cache_dir"]
    corp_actions = load_corporate_actions()

    def is_corp_action(row) -> bool:
        return any(row.entry_date <= d <= row.exit_date for d in corp_actions.get(row.symbol, []))

    tl["corp_action"] = tl.apply(is_corp_action, axis=1)
    clean = tl[~tl["corp_action"]]

    def stats(df, label):
        if df.empty:
            return
        wr = (df['pnl'] > 0).mean() * 100
        pf = df.loc[df.pnl > 0, 'pnl'].sum() / -df.loc[df.pnl <= 0, 'pnl'].sum() if (df.pnl <= 0).any() else float("inf")
        print(f"{label}: {len(df)} trades | win rate {wr:.1f}% | profit factor {pf:.2f} | net {df['pnl'].sum():,.0f}")

    stats(tl, "ALL (including corporate actions)")
    if tl["corp_action"].any():
        stats(clean, "EXCLUDING corporate-action trades")
        print(f"  ({tl['corp_action'].sum()} trade(s) flagged: "
              f"{', '.join(tl.loc[tl.corp_action, 'symbol'] + ' ' + tl.loc[tl.corp_action, 'exit_date'].dt.date.astype(str))})")
    print()

    flagged, rows = [], []
    for t in tl.itertuples():
        f = cache / f"{t.symbol}.csv"
        if not f.exists():
            print(f"  {t.symbol:12s}: no cached bars, skipping (re-run scripts/fetch_history.py)")
            continue
        bars = pd.read_csv(f, parse_dates=["date"], index_col="date")
        hold = bars.loc[t.entry_date:t.exit_date, "close"]
        if hold.empty:
            continue
        mae_pct = (hold.min() / t.avg_cost - 1) * 100
        day_moves = hold.pct_change().dropna() * 100
        if len(day_moves):
            worst_idx = day_moves.abs().idxmax()
            worst_session, worst_session_date = float(day_moves.loc[worst_idx]), worst_idx.date()
        else:
            worst_session, worst_session_date = 0.0, None
        rows.append({"symbol": t.symbol, "pnl_pct": t.pnl_pct, "mae_pct": round(mae_pct, 1),
                     "worst_session_pct": round(worst_session, 1), "worst_session_date": worst_session_date,
                     "holding_days": t.holding_days})
        if abs(worst_session) > 25:
            flagged.append((t.symbol, worst_session_date, worst_session, t.pnl_pct))

    report = pd.DataFrame(rows).sort_values("pnl_pct")
    print(report.to_string(index=False))

    if flagged:
        print(f"\n{'='*70}\nFLAGGED - single-session move > 25% during the hold:")
        for sym, d, mv, pnl in flagged:
            tag = " [in config/corporate_actions.json]" if any(
                sym == s_ and d == cd.date() for s_ in corp_actions for cd in corp_actions[s_]
            ) else " [UNRECOGNIZED - verify against bhavcopy/corporate-action calendar]"
            print(f"  {sym:12s}  {d}  {mv:+.1f}% in one session  (trade finished at {pnl:+.1f}%){tag}")
        print("A move this size is almost always a demerger/split/bonus ex-date, not a real crash -\n"
              "add unrecognized ones to config/corporate_actions.json once confirmed, so future runs\n"
              "exclude them from win-rate/profit-factor automatically.")

    losers = report[report["pnl_pct"] < 0]
    if len(losers):
        fast = (losers["holding_days"] <= 10).sum()
        print(f"\n{len(losers)} losing trades: {fast} exited within 10 days (whipsaw - entered "
              f"then immediately reversed), {len(losers) - fast} took longer (a trend that formed, "
              f"then broke).")


if __name__ == "__main__":
    main()