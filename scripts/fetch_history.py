"""
Populate data_cache/*.csv with daily bars for every symbol in config/universe.json,
via the same UpstoxProvider (and its incremental/full-refresh caching logic) that the
live eod/open commands use. Run this once before backtesting, and re-run occasionally
to refresh (it's incremental after the first full pull, per system_config.json's
data.full_refresh_days).

    export UPSTOX_ACCESS_TOKEN=...     # PowerShell: $env:UPSTOX_ACCESS_TOKEN = "..."
    python scripts/fetch_history.py
    python scripts/fetch_history.py --list nifty500     # a specific universe list
    python scripts/fetch_history.py --symbols RELIANCE,TCS,INFY   # a subset, for a quick test
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trader.core.config import ROOT, load_settings
from trader.data.upstox import UpstoxProvider

log = logging.getLogger("fetch_history")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="default", help="universe list name from config/universe.json")
    ap.add_argument("--symbols", help="comma-separated override, ignores --list")
    ap.add_argument("--sleep", type=float, default=0.25, help="seconds between symbols (rate-limit courtesy)")
    a = ap.parse_args()

    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    s = load_settings()
    symbols = a.symbols.split(",") if a.symbols else s.universe["lists"][a.list]
    missing = [sym for sym in symbols if sym not in s.universe["instruments"]]
    if missing:
        sys.exit(f"{len(missing)} symbol(s) not in config/universe.json instruments: {missing[:10]}...")

    provider = UpstoxProvider(s.system["data"], s.system["timezone"], s.universe["instruments"],
                              ROOT / s.system["paths"]["cache_dir"])
    today = datetime.now(ZoneInfo(s.system["timezone"])).date()

    ok, failed = 0, []
    for i, sym in enumerate(symbols, 1):
        try:
            df = provider.get_daily(sym, today)
            log.info("[%d/%d] %-12s %d bars, %s -> %s", i, len(symbols), sym, len(df),
                     df.index[0].date() if len(df) else "-", df.index[-1].date() if len(df) else "-")
            ok += 1
        except Exception as e:  # noqa: BLE001 - one bad symbol shouldn't kill the whole pull
            log.error("[%d/%d] %-12s FAILED: %s", i, len(symbols), sym, e)
            failed.append(sym)
        time.sleep(a.sleep)

    print(f"\n{ok}/{len(symbols)} symbols cached to {ROOT / s.system['paths']['cache_dir']}")
    if failed:
        print(f"{len(failed)} failed (delisted/renamed/rate-limited - re-run to retry): {failed}")


if __name__ == "__main__":
    main()