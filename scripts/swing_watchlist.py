"""Daily actionable watchlist for the spec-driven breakout strategies (vcp_breakout,
high_tight_flag) - or any strategy exposing a setup(completed_bars) method.

Two lists per strategy, both computed by the SAME strategy code the engine/backtester runs:

  TRIGGERED  entry_signal() fired on the as-of session (the pivot was crossed today). For an
             intraday-execution strategy the engine books the buy-stop fill on that session;
             otherwise `trader.cli eod` turns it into a pending order for the next open.
  ARMED      setup() is valid on the as-of session's completed bars: tomorrow is the "live
             candle". Place a buy-stop at PIVOT with the listed STOP - this is the spec's
             intraday execution, which the EOD engine can only approximate.

Every row is sized exactly as Engine._buy does (effective_risk + size_entry: each strategy's own
`sizing` block where it has one - for vcp_breakout/high_tight_flag the spec's 1% of current
equity, 20% max exposure - else config/risk_config.json), so the quantities are what the engine
would buy with the current equity and cash. vcp_breakout's Trend Template condition 8
(RS >= 70) is evaluated by the real rs_rank filter (prepare + evaluate), not a re-implementation.

Usage:
  python scripts/swing_watchlist.py                                  # data_cache, latest date
  python scripts/swing_watchlist.py --date 2026-11-11
  python scripts/swing_watchlist.py --source yfinance --years 2      # fetch NSE data via yfinance
  python scripts/swing_watchlist.py --symbols TCS,INFY --strategies vcp_breakout
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trader.core.calendar import TradingCalendar  # noqa: E402
from trader.core.config import ROOT, load_settings  # noqa: E402
from trader.core.models import Signal  # noqa: E402
from trader.data.historical import HistoricalProvider, load_csv_dir  # noqa: E402
from trader.filters.base import build_filter  # noqa: E402
from trader.risk.capital import CapitalManager  # noqa: E402
from trader.risk.sizing import effective_risk, size_entry  # noqa: E402
from trader.strategies.base import SymbolContext, build_strategy  # noqa: E402

log = logging.getLogger("watchlist")

DEFAULT_STRATEGIES = ["vcp_breakout", "high_tight_flag"]
YF_INDEX = {"NIFTY500": "^CRSLDX", "NIFTY50": "^NSEI", "NIFTY100": "^CNX100"}
COLS = ["list", "strategy", "symbol", "asof", "pivot", "stop", "risk_pct", "qty", "notional",
        "risk_rs", "risk_pct_of_base", "rs_pct", "held", "sizing", "action", "detail"]


# ---- data -----------------------------------------------------------------------------------
def load_yfinance(symbols: list[str], years: int) -> dict[str, pd.DataFrame]:
    try:
        import yfinance as yf
    except ImportError as e:
        raise SystemExit("yfinance not installed: pip install yfinance") from e
    tick = {s: YF_INDEX.get(s, f"{s}.NS") for s in symbols}
    raw = yf.download(list(tick.values()), period=f"{years}y", interval="1d", group_by="ticker",
                      auto_adjust=False, threads=True, progress=False)
    out: dict[str, pd.DataFrame] = {}
    for sym, t in tick.items():
        try:
            df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna(
                subset=["open", "high", "low", "close"])
        except (KeyError, TypeError):
            log.warning("%s (%s): no data from yfinance", sym, t)
            continue
        if df.empty:
            log.warning("%s (%s): empty frame from yfinance", sym, t)
            continue
        idx = pd.to_datetime(df.index)
        df.index = idx.tz_localize(None) if idx.tz is not None else idx
        out[sym] = df.sort_index().astype(float)
    log.info("yfinance: %d/%d symbols loaded", len(out), len(symbols))
    return out


def resolve_asof(data: dict[str, pd.DataFrame], bench: str, wanted: str | None) -> date:
    if wanted:
        return date.fromisoformat(wanted)
    if bench in data and not data[bench].empty:
        return data[bench].index[-1].date()
    last = pd.Series([df.index[-1].date() for df in data.values() if not df.empty])
    if last.empty:
        raise SystemExit("no data loaded")
    return last.mode().max()          # most common last bar = the latest complete session


def held_symbols(state_dir: Path) -> set[str]:
    p = state_dir / "portfolio.json"
    try:
        return set(json.loads(p.read_text())["positions"]) if p.exists() else set()
    except Exception as e:  # noqa: BLE001 - a corrupt state file must not stop the scan
        log.warning("could not read %s: %s", p, e)
        return set()


# ---- core -----------------------------------------------------------------------------------
def size(settings, capital: CapitalManager, strategy: str, entry: float, stop: float,
         cash: float, equity: float, slippage_bps: float) -> dict:
    """Engine sizing (effective_risk + size_entry, exactly as Engine._buy) plus which cap bound."""
    cfg_risk, mode, _ = effective_risk(settings.risk, settings.strategies[strategy])
    est = entry * (1 + slippage_bps / 1e4)
    base = capital.strategy_base(strategy, equity, mode)
    amt = capital.risk_amount(strategy, equity, mode, cfg_risk["per_trade"]["risk_pct_of_base"])
    qty, why = size_entry(amt, est, stop, base, cash, cfg_risk)
    pt = cfg_risk["per_trade"]
    risk_qty = int(amt // max(est - stop, est * pt["min_stop_distance_pct"] / 100.0, 1e-9))
    cap_qty = int(base * pt["max_position_pct_of_base"] / 100.0 // est)
    bound = "risk" if qty == risk_qty else ("20%-exposure cap" if qty == cap_qty else "cash/min-stop")
    return {"qty": qty, "notional": round(qty * est, 0), "risk_rs": round(qty * max(est - stop, 0), 0),
            "risk_pct_of_base": round(qty * max(est - stop, 0) / base * 100, 2) if base else None,
            "sizing": "ok, bound by " + bound if qty else why}


def scan(settings, data, strategies, asof, cash, equity, held) -> list[dict]:
    days = sorted({ts.date() for df in data.values() for ts in df.index if ts.date() <= asof})
    cal = TradingCalendar(set(), trading_days=days)
    provider = HistoricalProvider(data)
    capital = CapitalManager(settings.risk)
    slip = settings.system["execution"]["slippage_bps"]
    nxt = cal.next_trading_day(asof) if days and asof < days[-1] else "next session"

    rs = None
    if "rs_rank" in settings.filters:
        rs = build_filter("rs_rank", settings.filters["rs_rank"])
        rs.bind_strategy_cfgs({n: s.cfg for n, s in strategies.items()})
        try:
            rs.prepare(asof, provider)
        except Exception as e:  # noqa: BLE001
            log.warning("rs_rank prepare failed (%s) - RS gate falls back to its fail_mode", e)

    rows: list[dict] = []
    for name, strat in strategies.items():
        if not hasattr(strat, "setup"):
            log.warning("%s has no setup() - only TRIGGERED rows will be listed", name)
        universe = settings.universe["lists"].get(strat.cfg.get("universe", "default"), [])
        n_ok = n_err = 0
        for sym in universe:
            if sym not in data:
                continue
            try:
                d = data[sym].loc[: pd.Timestamp(asof)]
                if d.empty or d.index[-1].date() != asof:
                    continue                                  # stale / no bar for asof
                ctx = SymbolContext(sym, d, asof, cal)
                found = []
                sig = strat.entry_signal(ctx)
                if sig:
                    if strat.intraday:   # buy-stop filled on this session (Engine._enter_intraday)
                        fill = strat.entry_fill_price(sig, d.iloc[-1])
                        act = f"buy-stop filled {asof} @ {fill:.2f} (engine books it on this session)"
                    else:
                        act = f"engine BUY at {nxt} open (pending order)"
                    found.append(("TRIGGERED", sig.ref_price, sig.stop, sig.reason, act))
                st = strat.setup(d) if hasattr(strat, "setup") else None
                if st:
                    det = {k: v for k, v in st.details.items() if k != "trend_template"}
                    found.append(("ARMED", st.pivot, st.stop, json.dumps(det, default=str),
                                  f"BUY-STOP {st.pivot:.2f} / SL {st.stop:.2f} for {nxt}"))
                n_ok += 1
            except Exception as e:  # noqa: BLE001 - one bad symbol never aborts the scan
                n_err += 1
                log.debug("%s/%s: %s", name, sym, e)
                continue

            for lst, pivot, stop, detail, action in found:
                rs_pct, rs_ok = None, True
                if rs is not None and rs.enabled:
                    res = rs.evaluate(Signal(sym, name, "ENTRY", asof.isoformat(), "", stop=stop,
                                             ref_price=pivot), ctx)
                    rs_pct, rs_ok = rs._percentile_by_symbol.get(sym), res.passed
                if not rs_ok:
                    continue                                  # fails Trend Template condition 8
                rows.append({"list": lst, "strategy": name, "symbol": sym, "asof": asof.isoformat(),
                             "pivot": round(pivot, 2), "stop": round(stop, 2),
                             "risk_pct": round((pivot - stop) / pivot * 100, 2),
                             "rs_pct": None if rs_pct is None else round(rs_pct * 100),
                             "held": sym in held, "action": action, "detail": detail,
                             **size(settings, capital, name, pivot, stop, cash, equity, slip)})
        log.info("%s: scanned %d symbols (%d errors)", name, n_ok, n_err)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES))
    ap.add_argument("--date", help="as-of session (YYYY-MM-DD); default = latest available")
    ap.add_argument("--source", choices=["cache", "yfinance"], default="cache")
    ap.add_argument("--data-dir", help="CSV dir for --source cache (default: system_config cache_dir)")
    ap.add_argument("--years", type=int, default=2, help="--source yfinance history length (>=2 for SMA200+20)")
    ap.add_argument("--symbols", help="comma-separated subset of the universe")
    ap.add_argument("--cash", type=float, help="available cash for sizing (default: portfolio.json cash, else base)")
    ap.add_argument("--out", help="CSV path (default: <state_dir>/watchlist_<asof>.csv)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("trader", "yfinance"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    s = load_settings()
    names = [n.strip() for n in a.strategies.split(",") if n.strip()]
    unknown = [n for n in names if n not in s.strategies]
    if unknown:
        raise SystemExit(f"unknown strategy {unknown}; have {sorted(s.strategies)}")
    strategies = {n: build_strategy(n, s.strategies[n]) for n in names}   # ignores "enabled"

    if a.symbols:
        subset = [x.strip() for x in a.symbols.split(",") if x.strip()]
        for lst in s.universe["lists"]:
            s.universe["lists"][lst] = subset
    bench = s.filters.get("rs_rank", {}).get("benchmark_symbol", "NIFTY500")
    universe = sorted({sym for st in strategies.values()
                       for sym in s.universe["lists"].get(st.cfg.get("universe", "default"), [])})

    if a.source == "yfinance":
        data = load_yfinance(universe + [bench], a.years)
    else:
        path = Path(a.data_dir) if a.data_dir else ROOT / s.system["paths"]["cache_dir"]
        if not path.exists():
            raise SystemExit(f"data dir {path} not found - run scripts/fetch_history.py or use --source yfinance")
        data = load_csv_dir(path)
    if not data:
        raise SystemExit("no price data loaded")
    if bench not in data:
        log.warning("benchmark %s not loaded - rs_rank has no RS data; condition 8 follows its fail_mode", bench)

    asof = resolve_asof(data, bench, a.date)
    state_dir = ROOT / s.system["paths"]["state_dir"]
    held = held_symbols(state_dir)
    cash, equity = a.cash, None
    try:
        pf = json.loads((state_dir / "portfolio.json").read_text())
        equity = float(pf["cash"]) + sum(sum(l["qty"] for l in p["lots"]) * p["last_price"]
                                         for p in pf["positions"].values())
        cash = float(pf["cash"]) if cash is None else cash
    except Exception:  # noqa: BLE001 - no live state yet: size off base capital
        pass
    base = float(s.risk["capital"]["base_capital"])
    cash = base if cash is None else cash
    equity = cash if equity is None else equity

    rows = scan(s, data, strategies, asof, cash, equity, held)
    df = pd.DataFrame(rows, columns=COLS)
    out = Path(a.out) if a.out else state_dir / f"watchlist_{asof.isoformat()}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\nWATCHLIST as of {asof}  (sizing: equity {equity:,.0f}, cash {cash:,.0f})")
    if df.empty:
        print("  no setups.")
    show = ["symbol", "pivot", "stop", "risk_pct", "qty", "notional", "risk_rs", "rs_pct", "held", "action"]
    for (lst, strat), g in df.sort_values(["list", "strategy", "risk_pct"]).groupby(["list", "strategy"], sort=False):
        print(f"\n== {lst}  [{strat}]  ({len(g)})")
        print(g[show].to_string(index=False))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
