"""Regression check: re-run the existing strategies with temporary in-memory config overrides
(config files are never edited) and print the summary plus a per-strategy table, for comparison
against a known baseline.

Default = the 2026-10-31 reference run: 2022-01-01 -> 2023-12-31, the 5 enabled strategies,
candlestick filter ON (rank mode), sentiment filter OFF - the config as it stood then. Expected:
163 closed trades, net closed P&L 549,665, CAGR 44.81%, PF 1.86 (if data_cache is unchanged).

  python scripts/regression_check.py                       # 31-Oct reference config
  python scripts/regression_check.py --as-is               # today's config files, untouched
  python scripts/regression_check.py --set sentiment.enabled=false --set candlestick.enabled=true
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
NEW = {"vcp_breakout", "high_tight_flag"}


def apply(settings, spec: str) -> None:
    """filter_or_strategy.key=value, value parsed as JSON (true/false/numbers/"strings")."""
    path, raw = spec.split("=", 1)
    name, key = path.split(".", 1)
    target = settings.filters.get(name) or settings.strategies.get(name)
    if target is None:
        raise SystemExit(f"--set {spec}: no filter or strategy called {name!r}")
    try:
        val = json.loads(raw)
    except json.JSONDecodeError:
        val = raw
    target[key] = val


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--as-is", action="store_true", help="no overrides: current config files")
    ap.add_argument("--set", action="append", default=[], help="filter_or_strategy.key=value (repeatable)")
    ap.add_argument("--data-dir")
    ap.add_argument("--label", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    s = load_settings()
    for spec in ([] if a.as_is else REFERENCE) + a.set:
        apply(s, spec)
    for n in NEW:                                   # never part of the old-strategy baseline
        s.strategies[n]["enabled"] = False
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
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()