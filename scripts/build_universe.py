"""
Build config/universe.json for the Nifty 500 from two live official sources,
joined on ISIN (not symbol/name - those drift; ISIN is the stable key):

  1. NSE's own Nifty 500 constituent list (symbol + ISIN)
  2. Upstox's instrument master (ISIN -> instrument_key)

Run this locally (needs real internet access - nseindia.com blocks non-browser
UAs without a warm-up request, and this repo's own CI network policy shouldn't
need to reach either host):

    pip install pandas requests
    python scripts/build_universe.py

Re-run whenever the Nifty 500 is rebalanced (semi-annually) or when Upstox
reissues instrument keys (rare, but check the "unmatched" list on every run -
that's your early warning).
"""
from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

NIFTY500_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
UPSTOX_INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz"
OUT_PATH = Path(__file__).resolve().parents[1] / "config" / "universe.json"
SECTOR_OUT_PATH = Path(__file__).resolve().parents[1] / "config" / "sector_map.json"

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/market-data/live-equity-market",
}


def fetch_nifty500() -> pd.DataFrame:
    with requests.Session() as s:
        s.headers.update(NSE_HEADERS)
        s.get("https://www.nseindia.com", timeout=10)  # sets cookies NSE requires
        r = s.get(NIFTY500_URL, timeout=15)
        r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df.columns = [c.strip() for c in df.columns]
    print(f"  columns: {list(df.columns)}")
    need = {"Symbol", "ISIN Code", "Industry"}
    if not need.issubset(df.columns):
        sys.exit(f"Unexpected NSE CSV columns: {list(df.columns)} (expected {need})")
    out = df.rename(columns={"ISIN Code": "isin"})[["Symbol", "isin", "Company Name", "Industry"]]
    out["Symbol"] = out["Symbol"].str.strip()
    out["isin"] = out["isin"].str.strip()
    out["Industry"] = out["Industry"].str.strip()
    print(f"  sample: {out.head(3).to_dict('records')}")
    return out


def fetch_upstox_instruments() -> pd.DataFrame:
    df = pd.read_csv(UPSTOX_INSTRUMENTS_URL, compression="gzip")
    need = {"instrument_key", "tradingsymbol", "exchange", "instrument_type"}
    if not need.issubset(df.columns):
        sys.exit(f"Unexpected Upstox CSV columns: {list(df.columns)} (expected {need})")

    print(f"  raw file: {len(df)} rows")
    print(f"  exchange values: {sorted(df['exchange'].dropna().unique().tolist())}")
    print(f"  instrument_type values: {sorted(df['instrument_type'].dropna().unique().tolist())}")
    print("  sample rows:")
    print(df.head(3).to_string())

    eq = df[(df["exchange"] == "NSE_EQ") & (df["instrument_type"] == "EQUITY")].copy()
    print(f"  after exchange==NSE & instrument_type==EQ filter: {len(eq)} rows")
    if eq.empty:
        sys.exit("No rows survived the exchange/instrument_type filter - adjust it to match the "
                 "values printed above, then re-run.")

    # This file ships without an isin/segment column despite Upstox's own docs - but NSE_EQ
    # instrument_keys are documented as "NSE_EQ|<ISIN>", so extract it and sanity-check the shape.
    eq["isin"] = eq["instrument_key"].str.split("|", n=1).str[1]
    print(f"  sample instrument_key -> isin: {eq[['instrument_key', 'isin']].head(3).to_dict('records')}")
    bad_isin = eq[~eq["isin"].str.match(r"^IN[A-Z0-9]{10}$", na=True)]
    if len(bad_isin) == len(eq):
        print("  WARNING: none of the extracted 'isin' values look like real ISINs - "
              "instrument_key format differs from docs; ISIN join will be skipped, relying on symbol fallback only.")
        eq["isin"] = pd.NA
    return eq[["instrument_key", "tradingsymbol", "isin"]].assign(
        tradingsymbol=lambda d: d["tradingsymbol"].str.strip()
    )


def main() -> None:
    print("Fetching Nifty 500 constituent list from NSE...")
    nifty = fetch_nifty500()
    print(f"  {len(nifty)} constituents")

    print("Fetching Upstox NSE instrument master...")
    upstox = fetch_upstox_instruments()
    print(f"  {len(upstox)} NSE_EQ instruments")

    # Primary join: ISIN (derived from instrument_key). Fallback: exact trading symbol match,
    # for the rare instrument_key that isn't ISIN-suffixed.
    by_isin = nifty.merge(upstox[["instrument_key", "isin"]], on="isin", how="left")
    still_missing = by_isin[by_isin["instrument_key"].isna()].drop(columns="instrument_key")
    by_symbol = still_missing.merge(
        upstox[["instrument_key", "tradingsymbol"]], left_on="Symbol", right_on="tradingsymbol", how="left"
    )
    matched_isin = by_isin.dropna(subset=["instrument_key"])
    matched_symbol = by_symbol.dropna(subset=["instrument_key"])
    unmatched = by_symbol[by_symbol["instrument_key"].isna()]

    if len(matched_symbol):
        print(f"\n{len(matched_symbol)} symbols matched by trading-symbol fallback (ISIN mismatch - verify):")
        for _, row in matched_symbol.iterrows():
            print(f"    {row['Symbol']:15s} {row['isin']}  -> {row['instrument_key']}")
    if len(unmatched):
        print(f"\nWARNING: {len(unmatched)} symbols had no match at all (delisted, ISIN changed, "
              f"or newly listed) - excluded from universe.json:")
        for _, row in unmatched.iterrows():
            print(f"    {row['Symbol']:15s} {row['isin']}  {row['Company Name']}")

    matched = pd.concat([matched_isin[["Symbol", "instrument_key"]],
                         matched_symbol[["Symbol", "instrument_key"]]], ignore_index=True)
    equity_instruments = dict(zip(matched["Symbol"], matched["instrument_key"]))

    # Preserve index benchmarks you've added by hand (e.g. NIFTY500 for the market-regime
    # filter, sector indices for sector_regime). They aren't in the Nifty 500 constituent list,
    # so without this, re-running this script would silently delete them from universe.json.
    carried = {}
    if OUT_PATH.exists():
        prior = json.loads(OUT_PATH.read_text(encoding="utf-8")).get("instruments", {})
        carried = {k: v for k, v in prior.items()
                   if v.startswith("NSE_INDEX|") and k not in equity_instruments}
    if carried:
        print(f"\nPreserved {len(carried)} index benchmark(s) from the existing universe.json: {sorted(carried)}")

    instruments = {**equity_instruments, **carried}
    universe = {
        "_note": f"Nifty 500 as of NSE's current published list, {len(equity_instruments)}/{len(nifty)} "
                 f"matched to Upstox instrument keys, plus {len(carried)} preserved index benchmark(s). "
                 f"Regenerate with scripts/build_universe.py.",
        "instruments": instruments,
        "lists": {"nifty500": sorted(equity_instruments), "default": sorted(equity_instruments)},
    }
    OUT_PATH.write_text(json.dumps(universe, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(instruments)} instruments to {OUT_PATH}")

    symbol_to_industry = dict(zip(nifty["Symbol"], nifty["Industry"]))
    sector_map = {
        "_note": "symbol -> NSE's own 'Industry' classification, from the same Nifty 500 list "
                 "above. This is NOT yet a benchmark mapping - config/filters/sector_regime.json's "
                 "industry_to_benchmark still needs each of the industries below mapped to an "
                 "actual sectoral index you've added to universe.json and fetched history for.",
        "symbol_to_industry": symbol_to_industry,
    }
    SECTOR_OUT_PATH.write_text(json.dumps(sector_map, indent=2, ensure_ascii=False), encoding="utf-8")
    industries = sorted(set(symbol_to_industry.values()))
    print(f"Wrote {len(symbol_to_industry)} symbol->industry mappings to {SECTOR_OUT_PATH}")
    print(f"\n{len(industries)} distinct industries found - map each of these in "
          f"config/filters/sector_regime.json's \"industry_to_benchmark\":")
    for ind in industries:
        print(f"    \"{ind}\": \"<benchmark symbol you add to universe.json>\",")


if __name__ == "__main__":
    main()