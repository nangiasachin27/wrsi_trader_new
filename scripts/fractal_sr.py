"""
Standalone multi-timeframe Support/Resistance zones: Fractal (local extrema) + Window-Shifting.

Independent of trader/core/levels.py, trader/core/indicators.py and config/levels.json - it only
reads the same daily-bar cache (data_cache/<SYM>.csv) and config/universe.json. Writes one CSV
per run: one row per ticker, nearest support and resistance ZONE (low-high range) for the
daily, weekly and monthly charts.

    python scripts/fractal_sr.py                          # whole 'default' list, as of latest cached bar
    python scripts/fractal_sr.py --symbols RELIANCE,TCS   # subset
    python scripts/fractal_sr.py --date 2026-10-30        # as of a past date (data sliced <= date, no lookahead)
    python scripts/fractal_sr.py --refresh                # top up data_cache from Upstox first (needs UPSTOX_ACCESS_TOKEN)
    python scripts/fractal_sr.py --zones-out state/sr_zones.csv   # also dump EVERY zone (long format) for chart checks

Algorithm, per symbol and per timeframe (weekly/monthly bars are resampled from daily, W-FRI / ME):

 0. Lookback window: only swings dated within lookback_months of the as-of date count (default
    daily 6, weekly 12, monthly 36; 0 = all history). Detection also scans a short warm-up margin before
    the window so a swing on the window's first bars can still be confirmed on its left side.

 1. Candidate swing points from two detectors, run on highs (swing highs) and lows (swing lows):
    a. Fractal, order n: bar i is a swing high if high[i] >  high[i-k] for k=1..n
                                              and high[i] >= high[i+k] for k=1..n.
       Strict-left / non-strict-right means a run of equal highs yields exactly ONE pivot (the first).
    b. Window shifting, (W, P): slide a W-bar window forward one bar at a time and record the
       argmax at each position. A bar that stays the window's argmax for P consecutive shifts is a
       swing high. In effect: the bar is the extreme of some (W+P-1)-bar span with >= P-1 bars on
       EACH side - but not necessarily centred. So it also catches lower-highs / higher-lows sitting
       next to a bigger swing, which a symmetric fractal rejects.
    Swing lows: mirror image (min / argmin).
 2. Merge: the same bar+side found by both detectors is a "confirmed" pivot (highest weight).
 3. Cluster all pivot prices - highs AND lows together, because a broken resistance becomes support -
    in log-price space. Tolerance = cluster_atr_mult * ATR% of that timeframe (clamped to
    [tol_min_pct, tol_max_pct]). Density-greedy: the pivot with the highest recency-weighted
    neighbour mass seeds a zone, takes every unassigned pivot within +-tol, repeat. Zone width is
    therefore bounded to 2*tol (no single-linkage chaining).
 4. Zone = [min, max] of member pivot prices, widened symmetrically to >= min_width_atr_mult * ATR%.
    score = sum(weight * 0.5^(age_bars / half_life)), x polarity_bonus if the zone holds both swing
    highs and swing lows. Valid zone: touches >= min_touches, OR a single pivot found by both
    detectors that is still fresh (score >= single_pivot_min_score) - lone old swings don't count.
 5. Role vs the latest DAILY close: a zone wholly below it is support, wholly above it is resistance.
    A zone the close sits INSIDE takes its role from the side price approached it from before the
    current visit began (rallied into it = resistance being tested; fell into it = support being
    tested) - a one-day poke through the far side doesn't flip it.
    S1/R1 = nearest valid zone on each side; SMAJ/RMAJ = strongest (highest score) zone on each side.
 6. Edge fallback: a side with no valid zone gets the lookback's extreme high/low as an UNCONFIRMED
    zone (score 0, note '{tf}:r1_edge' / '{tf}:s1_edge') if price has pulled back from it by at least
    the cluster tolerance - the latest top/bottom is invisible to fractals until right-side bars form.

Output columns (tf = d / w / m, k = 1..levels_per_side):
    date, ticker, close,
    {tf}_atr_pct,
    {tf}_s{k}_low, {tf}_s{k}_high, {tf}_s{k}_touches, {tf}_s{k}_score, {tf}_s{k}_dist_pct,
    {tf}_smaj_low, ... {tf}_smaj_dist_pct          (major support: strongest zone below / being tested)
    {tf}_r{k}_low, {tf}_r{k}_high, {tf}_r{k}_touches, {tf}_r{k}_score, {tf}_r{k}_dist_pct,
    {tf}_rmaj_low, ... {tf}_rmaj_dist_pct          (major resistance)
    notes
  touches: swing points (5-candle fractal / window-shift pivots) that formed the zone.
  dist_pct: support = (close - zone_high)/close, resistance = (zone_low - close)/close, in %.
  NEGATIVE dist_pct = price is currently INSIDE that zone. Blank = no zone on that side even after
  the edge fallback (blank resistance = trading within one cluster tolerance of the lookback high, i.e. at
  ATH for monthly). notes: ca_cut:<date>, stale:<last bar>, {tf}:insufficient_bars(n), {tf}:s1_edge/r1_edge,
  {tf}:no_sup_below/no_res_above.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("TRADER_ROOT", Path(__file__).resolve().parents[1]))
TF_PREFIX = {"daily": "d", "weekly": "w", "monthly": "m"}

log = logging.getLogger("fractal_sr")


# Algorithm lives in trader/core/fractal_levels.py (shared with the strategies/filters' "fractal"
# levels engine) - this script is the CLI around it. Names re-exported for tests/back-compat.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trader.core.fractal_levels import (  # noqa: E402,F401
    OHLC, TFResult, Zone, ZoneBuild, analyze_timeframe, approach_side, build_zones, classify_zones,
    cluster_zones, detect_pivots, fractal_pivots, last_session_on_or_before, lookback_window,
    resample, timeframe_bars, wilder_atr, window_shift_pivots,
)


# ============================================================================================
# I/O
# ============================================================================================
def load_json(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_daily(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"], index_col="date").sort_index()
    df = df[~df.index.duplicated(keep="last")][OHLC].astype(float).dropna()
    return df if len(df) else None


def ca_cutoffs(cfg_dir: Path, types: list[str]) -> dict[str, list[date]]:
    f = cfg_dir / "corporate_actions.json"
    if not f.exists() or not types:
        return {}
    out: dict[str, list[date]] = {}
    for e in load_json(f).get("events", []):
        if e.get("type") in types:
            out.setdefault(e["symbol"], []).append(date.fromisoformat(e["date"]))
    return out


def refresh_cache(symbols: list[str], instruments: dict, cache_dir: Path, cfg_dir: Path) -> None:
    """Optional: reuse the system's Upstox provider purely as a data fetcher (no S/R logic)."""
    sys.path.insert(0, str(ROOT))
    from zoneinfo import ZoneInfo
    from trader.data.upstox import UpstoxProvider
    sysc = load_json(cfg_dir / "system_config.json")
    prov = UpstoxProvider(sysc["data"], sysc["timezone"], instruments, cache_dir)
    today = datetime.now(ZoneInfo(sysc["timezone"])).date()
    for i, s in enumerate(symbols, 1):
        try:
            prov.get_daily(s, today)
        except Exception as e:  # noqa: BLE001 - one bad symbol must not stop the run
            log.error("[%d/%d] refresh %s failed: %s", i, len(symbols), s, e)
        time.sleep(0.25)


def zone_cols(tf: str, n: int, major: bool = True) -> list[str]:
    t = TF_PREFIX[tf]
    cols = [f"{t}_atr_pct"]
    for side in ("s", "r"):
        keys = [f"{side}{k}" for k in range(1, n + 1)] + ([f"{side}maj"] if major else [])
        for key in keys:
            cols += [f"{t}_{key}_{f}" for f in ("low", "high", "touches", "score", "dist_pct")]
    return cols


# ============================================================================================
# Per symbol
# ============================================================================================
def process_symbol(sym: str, daily: pd.DataFrame, asof: date, cfg: dict, holidays: set[date],
                   cuts: list[date], want_zones: bool) -> tuple[dict, list[dict]]:
    dcfg, wts, n_side = cfg["data"], cfg["weights"], cfg["output"]["levels_per_side"]
    want_major = cfg["output"].get("major_level", True)
    notes: list[str] = []

    daily = daily.loc[: pd.Timestamp(asof)]
    applicable = [c for c in cuts if c <= asof]
    if applicable:
        cut = max(applicable)
        daily = daily.loc[pd.Timestamp(cut):]
        notes.append(f"ca_cut:{cut.isoformat()}")
    last = daily.index[-1].date()
    if last < asof:
        notes.append(f"stale:{last.isoformat()}")
    ref = float(daily["close"].iloc[-1])

    row: dict = {"date": asof.isoformat(), "ticker": sym, "close": round(ref, 2)}
    zrows: list[dict] = []
    for tf, p in cfg["timeframes"].items():
        t = TF_PREFIX[tf]
        bars, _ = timeframe_bars(daily, tf, dcfg, asof, holidays)
        r = analyze_timeframe(bars, p, wts, ref, asof)
        if r.note:
            notes.append(f"{t}:{r.note}")
            continue
        row[f"{t}_atr_pct"] = round(r.atr_pct, 2)

        def put(key: str, z: Zone, side: str) -> None:
            dist = (ref - z.high) / ref if side == "s" else (z.low - ref) / ref
            row.update({f"{t}_{key}_low": round(z.low, 2), f"{t}_{key}_high": round(z.high, 2),
                        f"{t}_{key}_touches": z.touches, f"{t}_{key}_score": round(z.score, 2),
                        f"{t}_{key}_dist_pct": round(100 * dist, 2)})

        for side, lst, major in (("s", r.sup, r.sup_major), ("r", r.res, r.res_major)):
            for k, z in enumerate(lst[:n_side], 1):
                put(f"{side}{k}", z, side)
            if want_major and major is not None:
                put(f"{side}maj", major, side)
        for side, lst, empty in (("s", r.sup, "no_sup_below"), ("r", r.res, "no_res_above")):
            if not lst:
                notes.append(f"{t}:{empty}")
            elif lst[0].edge:
                notes.append(f"{t}:{side}1_edge")
        if want_zones:
            rank: dict[int, list[str]] = {}
            for k, z in enumerate(r.sup, 1):
                rank.setdefault(id(z), []).append(f"S{k}")
            for k, z in enumerate(r.res, 1):
                rank.setdefault(id(z), []).append(f"R{k}")
            for z, lab in ((r.sup_major, "SMAJ"), (r.res_major, "RMAJ")):
                if z is not None:
                    rank.setdefault(id(z), []).append(lab)
            extra = [z for z in r.sup[:1] + r.res[:1] if z.edge]
            for z in r.zones + extra:
                zrows.append({"date": asof.isoformat(), "ticker": sym, "tf": tf, "rank": "|".join(rank.get(id(z), [])),
                              "role": {"S": "support", "R": "resistance"}.get(z.role, ""),
                              "low": round(z.low, 2), "high": round(z.high, 2), "touches": z.touches,
                              "score": round(z.score, 3), "valid": z.valid, "confirmed": z.confirmed,
                              "polarity": z.polarity, "edge": z.edge,
                              "last_touch": r.bars.index[z.last_idx].date().isoformat()})
    row["notes"] = ";".join(notes)
    return row, zrows


# ============================================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(ROOT / "config" / "fractal_sr.json"))
    ap.add_argument("--list", help="universe list name (default: config data.universe_list)")
    ap.add_argument("--symbols", help="comma-separated override, ignores --list")
    ap.add_argument("--date", help="as-of date YYYY-MM-DD (default: latest bar in cache)")
    ap.add_argument("--out", help="output CSV path (default: <output.dir>/<prefix>_<asof>.csv)")
    ap.add_argument("--zones-out", help="optional long-format CSV of every zone, for verification")
    ap.add_argument("--refresh", action="store_true", help="top up data_cache from Upstox before computing")
    ap.add_argument("--include-partial", action="store_true", help="keep the still-forming week/month")
    a = ap.parse_args()

    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_json(Path(a.config))
    if a.include_partial:
        cfg["data"]["completed_bars_only"] = False
    cfg_dir = ROOT / "config"
    universe = load_json(cfg_dir / "universe.json")
    instruments = universe["instruments"]
    try:
        cache_dir = ROOT / load_json(cfg_dir / "system_config.json")["paths"]["cache_dir"]
    except (FileNotFoundError, KeyError):
        cache_dir = ROOT / "data_cache"
    holidays = set()
    if (cfg_dir / "holidays.json").exists():
        holidays = {date.fromisoformat(d) for d in load_json(cfg_dir / "holidays.json")["holidays"]}

    if a.symbols:
        symbols = [s.strip() for s in a.symbols.split(",") if s.strip()]
    else:
        symbols = universe["lists"][a.list or cfg["data"]["universe_list"]]
    unknown = [s for s in symbols if s not in instruments]
    if unknown:
        log.warning("%d symbol(s) not in universe.json instruments (skipped): %s", len(unknown), unknown[:10])
        symbols = [s for s in symbols if s in instruments]

    if a.refresh:
        log.info("refreshing %d symbols from Upstox ...", len(symbols))
        refresh_cache(symbols, instruments, cache_dir, cfg_dir)

    data, missing = {}, []
    for s in symbols:
        df = load_daily(cache_dir / f"{s}.csv")
        if df is None:
            missing.append(s)
        else:
            data[s] = df
    if not data:
        sys.exit(f"no cached bars found in {cache_dir} - run scripts/fetch_history.py or use --refresh")
    if missing:
        log.warning("%d symbol(s) have no cached data: %s", len(missing), missing[:10])

    asof = date.fromisoformat(a.date) if a.date else max(df.index[-1].date() for df in data.values())
    cuts = ca_cutoffs(cfg_dir, cfg["data"].get("corporate_action_truncate_types", []))

    rows, zrows, failed = [], [], []
    for s, df in data.items():
        if df.index[0].date() > asof:
            continue
        try:
            r, z = process_symbol(s, df, asof, cfg, holidays, cuts.get(s, []), bool(a.zones_out))
            rows.append(r)
            zrows += z
        except Exception as e:  # noqa: BLE001
            log.error("%s failed: %s", s, e)
            failed.append(s)

    n_side = cfg["output"]["levels_per_side"]
    major = cfg["output"].get("major_level", True)
    cols = ["date", "ticker", "close"] + [c for tf in cfg["timeframes"] for c in zone_cols(tf, n_side, major)] + ["notes"]
    out_df = pd.DataFrame(rows).reindex(columns=cols)
    for c in out_df.columns:
        if c.endswith("_touches"):
            out_df[c] = out_df[c].astype("Int64")
    out = Path(a.out) if a.out else ROOT / cfg["output"]["dir"] / f"{cfg['output']['prefix']}_{asof.isoformat()}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    if a.zones_out:
        Path(a.zones_out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(zrows).to_csv(a.zones_out, index=False)

    print(f"\nS/R zones as of {asof}: {len(out_df)} tickers -> {out}")
    for tf in cfg["timeframes"]:
        t = TF_PREFIX[tf]
        if f"{t}_s1_low" in out_df:
            print(f"  {tf:<8} support found {out_df[f'{t}_s1_low'].notna().sum():>4}   "
                  f"resistance found {out_df[f'{t}_r1_low'].notna().sum():>4}")
    if failed:
        print(f"  {len(failed)} failed: {failed[:20]}")


if __name__ == "__main__":
    main()
