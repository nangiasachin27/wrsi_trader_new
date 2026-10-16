"""
Find the Upstox instrument_key for a benchmark index (Nifty 50, Nifty 500, etc.) in the
NSE_INDEX segment, so you can add it to config/universe.json's "instruments" map without
guessing. Indices aren't tradeable instruments, so they never go in universe.json's "lists" -
only in "instruments", where the regime filter (and only the regime filter) will look them up.

    python scripts/find_index_instrument.py
    python scripts/find_index_instrument.py --search "NIFTY 50"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trader.core.config import load_settings

UPSTOX_INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", default="NIFTY", help="substring to match in tradingsymbol/name")
    a = ap.parse_args()

    print("Fetching Upstox NSE instrument master...")
    df = pd.read_csv(UPSTOX_INSTRUMENTS_URL, compression="gzip")
    idx = df[df["exchange"] == "NSE_INDEX"].copy()
    print(f"  {len(idx)} NSE_INDEX instruments total\n")

    hits = idx[idx["tradingsymbol"].str.contains(a.search, case=False, na=False)
              | idx["name"].str.contains(a.search, case=False, na=False)]
    if hits.empty:
        sys.exit(f"No NSE_INDEX instrument matched '{a.search}'. Try --search with a shorter substring.")

    print(f"Matches for '{a.search}':")
    for _, r in hits.iterrows():
        print(f"  {r['tradingsymbol']:20s} {r['name']:30s} -> {r['instrument_key']}")

    print("\nAdd the one you want to config/universe.json's \"instruments\" map, e.g.:")
    print('  "instruments": { ..., "NIFTY500": "<instrument_key from above>" }')
    print("Do NOT add it to any list under \"lists\" - that would make it a tradeable symbol.")
    print("Then set that same name as \"benchmark_symbol\" in config/regime_filter.json.")


if __name__ == "__main__":
    main()