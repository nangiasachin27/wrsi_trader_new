"""
Populate config/holidays.json from NSE's own trading-holiday calendar (equity/CM segment),
rather than hand-typing dates. NSE blocks non-browser requests without a cookie warm-up first
(same pattern as scripts/build_universe.py's NSE fetch).

    python scripts/fetch_holidays.py

Re-run this once a year (NSE typically publishes the next calendar year's holidays in
December) - this script OVERWRITES config/holidays.json each run, so re-running doesn't
accumulate stale years; it always reflects whatever NSE's API currently returns.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import requests

HOLIDAY_URL = "https://www.nseindia.com/api/holiday-master?type=trading"
OUT_PATH = Path(__file__).resolve().parents[1] / "config" / "holidays.json"
SEGMENT = "CM"  # Capital Market (equity cash) segment - what this trading system actually trades

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/resources/exchange-communication-holidays",
}


def main() -> None:
    with requests.Session() as s:
        s.headers.update(NSE_HEADERS)
        s.get("https://www.nseindia.com", timeout=10)  # sets cookies NSE requires
        r = s.get(HOLIDAY_URL, timeout=15)
        r.raise_for_status()
    data = r.json()

    print(f"Segments returned: {list(data.keys())}")
    if SEGMENT not in data:
        sys.exit(f"'{SEGMENT}' segment not in response - inspect the segments above and "
                 f"adjust SEGMENT at the top of this script.")

    rows = data[SEGMENT]
    print(f"  sample row: {rows[0] if rows else '(empty)'}")

    dates = []
    for row in rows:
        # NSE's date format has varied historically (e.g. "26-Jan-2026"); try the common ones
        # rather than assuming, so a format change fails loudly instead of silently miscounting.
        raw = row.get("tradingDate") or row.get("date")
        parsed = None
        for fmt in ("%d-%b-%Y", "%d-%b-%y", "%Y-%m-%d"):
            try:
                parsed = datetime.strptime(raw, fmt).date()
                break
            except (ValueError, TypeError):
                continue
        if parsed is None:
            sys.exit(f"Could not parse date '{raw}' in row {row} - NSE's format may have "
                     f"changed; add it to the fmt list above.")
        if parsed.weekday() < 5:  # a holiday that happens to fall on a weekend changes nothing
            dates.append(parsed.isoformat())

    dates = sorted(set(dates))
    out = {
        "_note": f"NSE {SEGMENT}-segment trading holidays, weekday-only, fetched "
                 f"{datetime.now().date().isoformat()} via scripts/fetch_holidays.py. "
                 f"Re-run when NSE publishes a new year's calendar.",
        "holidays": dates,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {len(dates)} weekday holidays to {OUT_PATH}:")
    for d in dates:
        print(f"  {d}")


if __name__ == "__main__":
    main()