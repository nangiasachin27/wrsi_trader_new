"""
Finish wiring up the sector_regime filter end-to-end:

  1. Reads config/sector_map.json (symbol -> NSE industry label). Run scripts/build_universe.py
     first if this is still empty - this script does not fetch the Nifty 500 list itself.
  2. Maps each distinct industry label to a candidate NSE sectoral benchmark index via a
     curated keyword table below. This table is a naming convention, not fetched market data -
     it's safe to hand-edit. The dry run (default, no flag) only prints the proposed mapping so
     you can review/override anything before it's applied.
  3. Resolves each chosen index name to its Upstox instrument_key (NSE_INDEX segment) and adds
     it to config/universe.json's "instruments" map (never to a tradeable "list" - same rule
     scripts/find_index_instrument.py already documents).
  4. Fetches daily history for each of those benchmark instruments via the same UpstoxProvider
     scripts/fetch_history.py uses.
  5. Writes the resulting industry_to_benchmark map into config/filters/sector_regime.json and
     sets "enabled": true.

Needs real internet access (assets.upstox.com for the instrument master and index history) and
UPSTOX_ACCESS_TOKEN set - run this locally, same as build_universe.py and fetch_history.py.

    export UPSTOX_ACCESS_TOKEN=...
    python scripts/build_universe.py            # only if config/sector_map.json is still empty
    python scripts/finish_sector_regime.py       # dry run: prints the proposed mapping, writes nothing
    python scripts/finish_sector_regime.py --apply   # resolves, fetches history, writes config, enables
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from trader.core.config import ROOT, load_settings
from trader.data.upstox import UpstoxProvider

UPSTOX_INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz"
SECTOR_REGIME_PATH = ROOT / "config" / "filters" / "sector_regime.json"
UNIVERSE_PATH = ROOT / "config" / "universe.json"

# Best-effort keyword -> NSE sectoral index tradingsymbol lookup, checked in order (first match
# wins). This is a curated naming table, not fetched market data - the dry run prints what each
# industry resolves to; edit this list and re-run if any mapping looks wrong before using --apply.
# Unmatched industries fall back to FALLBACK_BENCHMARK (the same broad-market index
# market_regime.json already uses) rather than being left unmapped, which would just block every
# trade in that industry via fail_mode instead of giving it a sensible default.
FALLBACK_BENCHMARK = "NIFTY500"
INDUSTRY_KEYWORDS: list[tuple[str, str]] = [
    ("private bank", "NIFTY PVT BANK"),
    ("psu bank", "NIFTY PSU BANK"),
    ("bank", "NIFTY BANK"),
    ("insurance", "NIFTY FIN SERVICE"),
    ("financ", "NIFTY FIN SERVICE"),          # catches "Finance" and "Financial ..."
    ("information technology", "NIFTY IT"),
    ("it services", "NIFTY IT"),
    ("software", "NIFTY IT"),
    ("biotechnology", "NIFTY PHARMA"),
    ("pharma", "NIFTY PHARMA"),
    ("healthcare", "NIFTY HEALTHCARE"),
    ("automobile", "NIFTY AUTO"),
    ("auto component", "NIFTY AUTO"),
    ("passenger car", "NIFTY AUTO"),
    ("two wheeler", "NIFTY AUTO"),
    ("commercial vehicle", "NIFTY AUTO"),
    ("vehicle", "NIFTY AUTO"),
    ("fast moving consumer", "NIFTY FMCG"),
    ("fmcg", "NIFTY FMCG"),
    ("metal", "NIFTY METAL"),
    ("mining", "NIFTY METAL"),
    ("oil", "NIFTY OIL AND GAS"),
    ("gas", "NIFTY OIL AND GAS"),
    ("power", "NIFTY ENERGY"),
    ("energy", "NIFTY ENERGY"),
    ("realty", "NIFTY REALTY"),
    ("real estate", "NIFTY REALTY"),
    ("media", "NIFTY MEDIA"),
    ("entertainment", "NIFTY MEDIA"),
    ("consumer durable", "NIFTY CONSR DURBL"),
    ("consumer electronics", "NIFTY CONSR DURBL"),
    ("textile", "NIFTY500"),                  # no standalone liquid sectoral index - left at fallback deliberately
    ("cement", "NIFTY INFRA"),
    ("retail", "NIFTY500"),                   # ditto
    ("chemical", "NIFTY500"),                 # ditto - edit if your broker lists a usable index
    ("electrical equipment", "NIFTY INFRA"),
    ("infrastructure", "NIFTY INFRA"),
    ("construction", "NIFTY INFRA"),
    ("capital goods", "NIFTY INFRA"),
    ("telecommunication", "NIFTY SERV SECTOR"),  # no standalone liquid NSE telecom index - nearest broad bucket
    ("telecom", "NIFTY SERV SECTOR"),
    ("services", "NIFTY SERV SECTOR"),
    ("commodities", "NIFTY COMMODITIES"),
    ("diversified", "NIFTY500"),               # conglomerates spanning sectors - no single sector fits
]


def propose_mapping(industries: list[str]) -> dict[str, str]:
    out = {}
    for ind in industries:
        low = ind.lower()
        out[ind] = next((bench for kw, bench in INDUSTRY_KEYWORDS if kw in low), FALLBACK_BENCHMARK)
    return out


def _norm(s: str) -> str:
    """Upstox's index tradingsymbols aren't consistently spaced ('NIFTY FIN SERVICE' vs
    'NIFTYFINSERVICE' vs 'NIFTY_FIN_SERVICE') - compare on letters only so spacing/punctuation
    differences never cause a miss."""
    return "".join(ch for ch in s.upper() if ch.isalnum())


def resolve_instrument_keys(index_names: set[str]) -> dict[str, str]:
    print("Fetching Upstox NSE instrument master to resolve index instrument_keys...")
    df = pd.read_csv(UPSTOX_INSTRUMENTS_URL, compression="gzip")
    idx = df[df["exchange"] == "NSE_INDEX"].copy()
    has_name_col = "name" in idx.columns
    idx["_sym_norm"] = idx["tradingsymbol"].astype(str).map(_norm)
    idx["_name_norm"] = idx["name"].astype(str).map(_norm) if has_name_col else idx["_sym_norm"]

    out, missing = {}, []
    for name in sorted(index_names):
        want = _norm(name)
        # 1. exact match on tradingsymbol or name, spacing/punctuation ignored
        hit = idx[(idx["_sym_norm"] == want) | (idx["_name_norm"] == want)]
        # 2. either side contains the other (handles a trailing "INDEX" or "25"/"50" suffix)
        if hit.empty:
            hit = idx[idx["_sym_norm"].str.contains(want, na=False)
                     | idx["_name_norm"].str.contains(want, na=False)
                     | idx["_sym_norm"].apply(lambda s: s in want)
                     | idx["_name_norm"].apply(lambda s: s in want)]
        # 3. loose substring match on the original (non-normalized) columns, as a last resort
        if hit.empty:
            hit = idx[idx["tradingsymbol"].str.contains(name, case=False, na=False)]
            if hit.empty and has_name_col:
                hit = idx[idx["name"].str.contains(name, case=False, na=False)]
        if hit.empty:
            missing.append(name)
            continue
        out[name] = hit.iloc[0]["instrument_key"]
    if missing:
        sys.exit(f"Could not resolve instrument_key for: {missing}. Run "
                 f"scripts/find_index_instrument.py --search '<substring>' to find the right "
                 f"tradingsymbol, add/fix it in INDUSTRY_KEYWORDS above, and re-run.\n"
                 f"(Everything else resolved fine - only these need a manual look.)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="resolve instrument_keys, fetch benchmark history, and write config "
                         "(default: dry run - only prints the proposed mapping)")
    a = ap.parse_args()

    s = load_settings()
    sector_map_path = ROOT / "config" / "sector_map.json"
    sector_map = json.loads(sector_map_path.read_text())["symbol_to_industry"]
    if not sector_map:
        sys.exit(f"{sector_map_path} is empty - run scripts/build_universe.py first "
                 f"(it fetches the Nifty 500 list and writes this file alongside universe.json).")

    industries = sorted(set(sector_map.values()))
    mapping = propose_mapping(industries)
    print(f"{len(industries)} industries found in {sector_map_path}. Proposed benchmark mapping:")
    for ind, bench in mapping.items():
        flag = "  <- fallback, no keyword matched: review this one" if bench == FALLBACK_BENCHMARK else ""
        print(f"  {ind:40s} -> {bench}{flag}")

    if not a.apply:
        print("\nDry run only - nothing written. Edit INDUSTRY_KEYWORDS in this script if any "
              "mapping above looks wrong, then re-run with --apply to resolve instrument_keys, "
              "fetch benchmark history, write config/filters/sector_regime.json, and enable it.")
        return

    index_names = set(mapping.values())
    resolved = resolve_instrument_keys(index_names)
    universe = json.loads(UNIVERSE_PATH.read_text())
    added = sum(universe["instruments"].get(name) != key for name, key in resolved.items())
    universe["instruments"].update(resolved)
    UNIVERSE_PATH.write_text(json.dumps(universe, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nAdded/updated {added} benchmark instrument_key(s) in {UNIVERSE_PATH}")

    provider = UpstoxProvider(s.system["data"], s.system["timezone"], universe["instruments"],
                              ROOT / s.system["paths"]["cache_dir"])
    today = datetime.now(ZoneInfo(s.system["timezone"])).date()
    for name in sorted(index_names):
        df = provider.get_daily(name, today)
        print(f"  fetched {name}: {len(df)} bars, "
              f"{df.index[0].date() if len(df) else '-'} -> {df.index[-1].date() if len(df) else '-'}")
        time.sleep(0.25)

    cfg = json.loads(SECTOR_REGIME_PATH.read_text())
    cfg["industry_to_benchmark"] = mapping
    cfg["enabled"] = True
    SECTOR_REGIME_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote industry_to_benchmark ({len(mapping)} industries) to {SECTOR_REGIME_PATH} "
          f"and set enabled: true.")
    print("fail_mode stays 'block', so any industry whose benchmark fails to fetch at runtime "
          "blocks entries in that industry rather than silently letting them through. "
          "Re-run your backtest to see its effect.")


if __name__ == "__main__":
    main()