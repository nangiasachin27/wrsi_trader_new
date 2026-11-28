"""Build config/universe_history.json - the Nifty 500 constituent list as it stood at many points
since 2019 - from archived copies of NSE's own `ind_nifty500list.csv` in the Internet Archive
(web.archive.org). Each snapshot carries Symbol + ISIN, so membership is keyed by ISIN and
survives symbol renames.

Run locally (needs internet; the Internet Archive does not block scripts the way nseindia.com does):

    python scripts/build_universe_history.py                 # 2019-06 -> today
    python scripts/build_universe_history.py --dry-run       # coverage report only, writes nothing

What it writes
  config/universe_history.json  snapshots [{date, isins}], names {isin: {symbol, company, industry}}
  config/universe.json          + an instrument ("NSE_EQ|<ISIN>") for every historical member not
                                already present, + list "nifty500_pit" = every stock ever a member
                                (the "default" list is not touched)
  config/sector_map.json        + industry for historical members not already mapped

Then fetch their bars and enable the point-in-time universe for backtests:
    python scripts/fetch_history.py --list nifty500_pit
    (system_config.json -> universe_history.enabled: true)

Read the report it prints: snapshot dates, the longest gap between snapshots (membership is
assumed unchanged inside a gap), and how many historical ISINs are missing from Upstox's
instrument master (usually delisted/merged companies - their bars may not be fetchable, which
leaves a little survivorship bias in; the report says how much).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config"
SOURCES = [   # NSE has served the same file from several hosts over the years
    "archives.nseindia.com/content/indices/ind_nifty500list.csv",
    "nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    "www1.nseindia.com/content/indices/ind_nifty500list.csv",
    "www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    "niftyindices.com/IndexConstituent/ind_nifty500list.csv",
]
CDX = "https://web.archive.org/cdx/search/cdx"
RAW = "https://web.archive.org/web/{ts}id_/http://{url}"
UPSTOX_INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz"
UA = {"User-Agent": "wrsi_trader universe-history builder (research; contact via GitHub)"}


def parse_constituents(text: str) -> pd.DataFrame | None:
    """NSE constituent CSV -> DataFrame[symbol, isin, company, industry], or None if it isn't one."""
    try:
        df = pd.read_csv(StringIO(text))
    except Exception:  # noqa: BLE001 - an archived error page is not a CSV
        return None
    df.columns = [str(c).strip() for c in df.columns]
    if not {"Symbol", "ISIN Code"} <= set(df.columns):
        return None
    out = pd.DataFrame({
        "symbol": df["Symbol"].astype(str).str.strip(),
        "isin": df["ISIN Code"].astype(str).str.strip(),
        "company": df.get("Company Name", pd.Series([""] * len(df))).astype(str).str.strip(),
        "industry": df.get("Industry", pd.Series([""] * len(df))).astype(str).str.strip(),
    })
    out = out[out["isin"].str.match(r"^IN[A-Z0-9]{10}$")]
    return out if 400 <= len(out) <= 520 else None       # a Nifty 500 list, not a truncated page


def list_captures(session: requests.Session, url: str, start: str) -> list[str]:
    r = session.get(CDX, params={"url": url, "output": "json", "from": start.replace("-", "")[:8],
                                 "filter": "statuscode:200", "fl": "timestamp,digest",
                                 "collapse": "digest"}, timeout=60)
    r.raise_for_status()
    rows = r.json() if r.text.strip() else []
    return [row[0] for row in rows[1:]]                   # first row is the header


def build_snapshots(session, start: str, sleep: float) -> tuple[list[dict], dict]:
    by_day: dict[str, pd.DataFrame] = {}
    for url in SOURCES:
        try:
            stamps = list_captures(session, url, start)
        except requests.RequestException as e:
            print(f"  {url}: CDX query failed ({e})")
            continue
        print(f"  {url}: {len(stamps)} distinct captures")
        for ts in stamps:
            day = datetime.strptime(ts[:8], "%Y%m%d").date().isoformat()
            if day in by_day:
                continue
            try:
                r = session.get(RAW.format(ts=ts, url=url), timeout=60)
                df = parse_constituents(r.text) if r.ok else None
            except requests.RequestException:
                df = None
            if df is not None:
                by_day[day] = df
            time.sleep(sleep)
    snaps, names = [], {}
    prev = None
    for day in sorted(by_day):
        df = by_day[day]
        isins = sorted(set(df["isin"]))
        for _, r in df.iterrows():                        # later snapshots overwrite: latest name wins
            names[r["isin"]] = {"symbol": r["symbol"], "company": r["company"], "industry": r["industry"]}
        if isins != prev:                                 # keep only snapshots that changed something
            snaps.append({"date": day, "isins": isins})
            prev = isins
    return snaps, names


def coverage_report(snaps: list[dict], start: str) -> None:
    if not snaps:
        print("\nNO SNAPSHOTS FOUND - nothing to build.")
        return
    dates = [date.fromisoformat(s["date"]) for s in snaps]
    gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
    print(f"\n{len(snaps)} distinct constituent lists, {dates[0]} -> {dates[-1]}")
    for s in snaps:
        print(f"  {s['date']}  {len(s['isins'])} stocks")
    if gaps:
        worst = max(gaps)
        i = gaps.index(worst)
        print(f"longest gap: {worst} days ({dates[i]} -> {dates[i + 1]}); membership is assumed "
              f"unchanged inside a gap (NSE rebalances end-March and end-September)")
    if dates[0] > date.fromisoformat(start):
        print(f"WARNING: first snapshot {dates[0]} is after --start {start}: backtests before it use "
              f"the first snapshot (still survivorship-biased before that date)")
    ever = set().union(*(set(s["isins"]) for s in snaps))
    print(f"stocks ever in the index over the period: {len(ever)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2019-06-01")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between archive downloads")
    ap.add_argument("--dry-run", action="store_true", help="report coverage, write nothing")
    a = ap.parse_args()

    s = requests.Session()
    s.headers.update(UA)
    print("Querying the Internet Archive for archived Nifty 500 constituent lists...")
    snaps, names = build_snapshots(s, a.start, a.sleep)
    coverage_report(snaps, a.start)
    if not snaps or a.dry_run:
        return 0 if snaps else 1

    print("\nFetching Upstox NSE instrument master (ISIN -> instrument_key)...")
    m = pd.read_csv(UPSTOX_INSTRUMENTS_URL, compression="gzip")
    m = m[(m["exchange"] == "NSE_EQ") & (m["instrument_type"] == "EQUITY")]
    upstox = {k.split("|", 1)[1]: k for k in m["instrument_key"] if "|" in str(k)}

    uni_path, sec_path = CFG / "universe.json", CFG / "sector_map.json"
    uni = json.loads(uni_path.read_text(encoding="utf-8"))
    inst = uni["instruments"]
    by_isin = {v.split("|", 1)[1]: k for k, v in inst.items() if v.startswith("NSE_EQ|")}
    sectors = json.loads(sec_path.read_text(encoding="utf-8")) if sec_path.exists() else {"symbol_to_industry": {}}

    ever = sorted(set().union(*(set(x["isins"]) for x in snaps)))
    added, not_in_master, pit = [], [], []
    for isin in ever:
        sym = by_isin.get(isin)
        if sym is None:                                   # historical member we don't have yet
            sym = names[isin]["symbol"]
            if sym in inst:                               # symbol reused by another company
                sym = f"{sym}_{isin[-4:]}"
            inst[sym] = upstox.get(isin, f"NSE_EQ|{isin}")
            by_isin[isin] = sym
            added.append(sym)
        if isin not in upstox:
            not_in_master.append(f"{sym} ({names[isin]['company']})")
        ind = names[isin].get("industry")
        if ind and sym not in sectors["symbol_to_industry"]:
            sectors["symbol_to_industry"][sym] = ind
        pit.append(sym)
    uni["lists"]["nifty500_pit"] = sorted(pit)

    out = {"_note": "Built by scripts/build_universe_history.py from archived NSE ind_nifty500list.csv "
                    "snapshots (web.archive.org). Each snapshot holds until the next.",
           "built": date.today().isoformat(), "start": a.start, "snapshots": snaps, "names": names}
    (CFG / "universe_history.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    uni_path.write_text(json.dumps(uni, indent=2), encoding="utf-8")
    sec_path.write_text(json.dumps(sectors, indent=2), encoding="utf-8")

    print(f"\nwrote config/universe_history.json ({len(snaps)} snapshots)")
    print(f"universe.json: +{len(added)} historical instruments, list nifty500_pit = {len(pit)} stocks")
    if not_in_master:
        print(f"{len(not_in_master)} historical members are NOT in Upstox's current instrument master "
              f"(delisted/merged/renamed ISIN) - their bars may not be fetchable:")
        for x in not_in_master[:60]:
            print(f"    {x}")
    print("\nnext: python scripts/fetch_history.py --list nifty500_pit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
