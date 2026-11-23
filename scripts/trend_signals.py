"""Compare market-trend signals and plot them, to judge by eye whether a signal identifies the
market's trend (and, with --equity, the periods when THIS system makes or loses money).

Signals (all computed from closes up to each day only):
  index   : NIFTY500 weekly close vs its weekly SMA (default 20) - today's market_regime test
  breadth : % of the trading universe closing above its own SMA (default 50), smoothed, with
            hysteresis (bullish from --bull-on, not bullish below --bull-off) - trader.risk.market_breadth
  ibd     : the IBD-style distribution-day / follow-through-day state machine in
            trader.core.market_trend (UPTREND / NEUTRAL / DOWNTREND; only UPTREND counts as bullish)

  python scripts/trend_signals.py
  python scripts/trend_signals.py --start 2021-01-01 ^
      --equity eqmode=state/regression_2021-01-01_2026-11-13_f_eqmode/daily_equity.csv ^
      --equity gross80=state/regression_2021-01-01_2026-11-13_f_eq_gross80/daily_equity.csv
  python scripts/trend_signals.py --bull-on 60 --bull-off 40 --smooth 10      # try other thresholds

Output (default state/trend_signals/): trend_signals.png, trend_signals.csv (one row per day),
and a printed table per signal:
  bull days %        share of sessions the signal says bullish
  switches/yr        how often it flips (whipsaw)
  median spell       median length of a bull or not-bull stretch, in sessions
  NIFTY fwd20 bull / not   average NIFTY500 return over the NEXT 20 sessions after bull vs
                     not-bull days (evaluation only - a useful signal separates these)
  <equity> bull / not      the system's average daily return (bp/day) on days whose signal, as of
                     the PREVIOUS session, was bull vs not-bull - the number that matters for a
                     trend-aware exposure cap: it should be clearly lower on not-bull days.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trader.core.config import load_settings  # noqa: E402
from trader.core.market_trend import classify_market_trend  # noqa: E402
from trader.data.historical import load_csv_dir  # noqa: E402
from trader.risk.market_breadth import DEFAULTS, breadth_frame, classify  # noqa: E402
from trader.risk.regime import RegimeFilter  # noqa: E402

BULL, NOT, NEUTRAL = "#1f6fb4", "#e66100", "#f2b134"     # blue / orange / amber (CVD-safe pair)


def index_signal(bench: pd.DataFrame, sma_period: int, days: pd.DatetimeIndex) -> pd.Series:
    rf = RegimeFilter({"timeframe_rule": "W-FRI", "sma_period": sma_period, "buffer_pct": 0.0,
                       "min_bars": sma_period + 2, "fail_mode": "allow"})
    out = {}
    for d in days:
        st = rf.status(bench.loc[:d], d.date(), "NIFTY500")
        out[d] = st.bullish if st.benchmark_sma is not None else np.nan
    return pd.Series(out, dtype=object)


def spells(b: pd.Series) -> list[int]:
    b = b.dropna().astype(bool)
    if b.empty:
        return []
    grp = (b != b.shift()).cumsum()
    return b.groupby(grp).size().tolist()


def stats(name: str, bull: pd.Series, bench_close: pd.Series, equities: dict[str, pd.Series]) -> dict:
    b = bull.dropna().astype(bool)
    yrs = max((b.index[-1] - b.index[0]).days / 365.25, 1e-9)
    fwd = (bench_close.shift(-20) / bench_close - 1).reindex(b.index) * 100
    row = {"signal": name, "bull days %": round(b.mean() * 100, 1),
           "switches/yr": round(((b != b.shift()).sum() - 1) / yrs, 1),
           "median spell": int(np.median(spells(b))) if len(b) else 0,
           "NIFTY fwd20 bull %": round(fwd[b].mean(), 2), "NIFTY fwd20 not %": round(fwd[~b].mean(), 2)}
    for lab, eq in equities.items():
        r = eq.pct_change() * 1e4                                   # bp/day
        prev = b.shift(1).reindex(r.index)                          # signal as of the session before
        ok = prev.notna()
        row[f"{lab} bull bp/d"] = round(r[ok & (prev == True)].mean(), 1)   # noqa: E712
        row[f"{lab} not bp/d"] = round(r[ok & (prev == False)].mean(), 1)   # noqa: E712
    return row


def shade(ax, bull: pd.Series, alpha=0.13):
    b = bull.dropna().astype(bool)
    if b.empty:
        return
    grp = (b != b.shift()).cumsum()
    for _, g in b.groupby(grp):
        ax.axvspan(g.index[0], g.index[-1] + pd.Timedelta(days=1),
                   color=BULL if g.iloc[0] else NOT, alpha=alpha, lw=0)


def strip(ax, y, states: pd.Series, colors: dict):
    s = states.dropna()
    if s.empty:
        return
    grp = (s != s.shift()).cumsum()
    for _, g in s.groupby(grp):
        x0 = g.index[0]
        x1 = g.index[-1] + pd.Timedelta(days=1)
        ax.broken_barh([(x0, x1 - x0)], (y - 0.38, 0.76), color=colors[g.iloc[0]], lw=0)


def main() -> None:
    s = load_settings()
    bcfg = {**DEFAULTS, **((s.risk.get("portfolio", {}).get("regime_gross_exposure") or {}).get("breadth") or {})}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--benchmark", default="NIFTY500")
    ap.add_argument("--sma", type=int, default=int(bcfg["sma_period"]), help="breadth: per-stock SMA")
    ap.add_argument("--smooth", type=int, default=int(bcfg["smooth_days"]))
    ap.add_argument("--bull-on", type=float, default=float(bcfg["bull_on_pct"]))
    ap.add_argument("--bull-off", type=float, default=float(bcfg["bull_off_pct"]))
    ap.add_argument("--index-sma", type=int, default=20, help="index: weekly SMA period")
    ap.add_argument("--equity", action="append", default=[], metavar="LABEL=PATH",
                    help="a daily_equity.csv to overlay and score (repeatable)")
    ap.add_argument("--no-ibd", action="store_true", help="skip the IBD state machine")
    ap.add_argument("--out", default=str(ROOT / "state" / "trend_signals"))
    a = ap.parse_args()

    data = load_csv_dir(Path(a.data_dir) if a.data_dir else ROOT / s.system["paths"]["cache_dir"])
    if a.benchmark not in data:
        sys.exit(f"no cached bars for {a.benchmark!r} in the data cache")
    lists = s.universe["lists"]
    syms = sorted({x for c in s.strategies.values() if c.get("enabled") for x in lists.get(c.get("universe", "default"), [])})
    syms = [x for x in syms if x in data]
    bench = data[a.benchmark]
    start = pd.Timestamp(a.start)
    end = pd.Timestamp(a.end) if a.end else bench.index[-1]
    print(f"universe {len(syms)} symbols with cached bars, benchmark {a.benchmark}, {start.date()} -> {end.date()}")

    # breadth over the FULL history (SMA + hysteresis warm-up), then cut to the window
    bf = breadth_frame(data, syms, a.sma)
    bf = bf[bf["n_valid"] >= int(bcfg["min_symbols"])]
    bf = bf.join(classify(bf["pct_above"], a.smooth, a.bull_on, a.bull_off))
    days = bench.loc[start:end].index
    bf = bf.reindex(days)

    sig = pd.DataFrame(index=days)
    sig["nifty"] = bench["close"].reindex(days)
    sig["breadth_pct"], sig["breadth_smoothed"], sig["breadth_bull"] = bf["pct_above"], bf["smoothed"], bf["bullish"]
    sig["index_bull"] = index_signal(bench, a.index_sma, days)
    if not a.no_ibd:
        cfg = json.loads((ROOT / "config" / "market_trend.json").read_text())
        vol = pd.DataFrame({x: data[x]["volume"] for x in syms}).sum(axis=1)   # NIFTY500 has no volume
        ibd, _ = classify_market_trend(bench.loc[:end], cfg, vol)
        sig["ibd_state"] = ibd["state"].reindex(days)
        sig["ibd_bull"] = sig["ibd_state"].map({"UPTREND": True, "NEUTRAL": False, "DOWNTREND": False})

    equities = {}
    for spec in a.equity:
        lab, path = spec.split("=", 1)
        e = pd.read_csv(path, parse_dates=["date"]).set_index("date")["equity"]
        equities[lab] = e.loc[start:end]
        sig[f"equity_{lab}"] = e.reindex(days)

    rows = [stats(f"index (weekly SMA{a.index_sma})", sig["index_bull"], bench["close"], equities),
            stats(f"breadth (>SMA{a.sma}, {a.bull_on:g}/{a.bull_off:g}, smooth {a.smooth})",
                  sig["breadth_bull"], bench["close"], equities)]
    if not a.no_ibd:
        rows.append(stats("ibd (UPTREND only)", sig["ibd_bull"], bench["close"], equities))
    table = pd.DataFrame(rows).set_index("signal")
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\n" + table.T.to_string() + "\n")
    print("Read it as: a useful signal has a LOWER NIFTY fwd20 and a LOWER system bp/day on not-bull days,")
    print("with few switches/yr. Bull bp/d >> not bp/d is what a trend-aware exposure cap needs.\n")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sig.to_csv(out / "trend_signals.csv")
    table.to_csv(out / "trend_signals_stats.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    n = 4 if equities else 3
    ratios = [3, 2, 1.1] + ([3] if equities else [])
    fig, axes = plt.subplots(n, 1, figsize=(15, 3.1 * n), sharex=True,
                             gridspec_kw={"height_ratios": ratios, "hspace": 0.28})
    ax = axes[0]
    shade(ax, sig["breadth_bull"])
    ax.plot(days, sig["nifty"], color="#222", lw=1.2)
    ax.set_yscale("log")
    ax.set_ylabel(a.benchmark)
    ax.set_title(f"{a.benchmark}, shaded by the breadth signal  (blue = bullish, orange = not bullish)", loc="left")

    ax = axes[1]
    ax.plot(days, sig["breadth_pct"], color="#9a9a9a", lw=0.8, label=f"% above SMA{a.sma} (daily)")
    ax.plot(days, sig["breadth_smoothed"], color="#222", lw=1.6, label=f"smoothed ({a.smooth}d)")
    ax.axhline(a.bull_on, color=BULL, lw=1, ls="--", label=f"bull on {a.bull_on:g}%")
    ax.axhline(a.bull_off, color=NOT, lw=1, ls="--", label=f"bull off {a.bull_off:g}%")
    ax.set_ylim(0, 100)
    ax.set_ylabel("breadth %")
    ax.legend(loc="lower left", fontsize=8, ncol=4, frameon=False)

    ax = axes[2]
    strip(ax, 2, sig["index_bull"].map({True: "bull", False: "not"}), {"bull": BULL, "not": NOT})
    strip(ax, 1, sig["breadth_bull"].map({True: "bull", False: "not"}), {"bull": BULL, "not": NOT})
    labels = {2: f"index (wkly SMA{a.index_sma})", 1: "breadth"}
    if not a.no_ibd:
        strip(ax, 0, sig["ibd_state"], {"UPTREND": BULL, "NEUTRAL": NEUTRAL, "DOWNTREND": NOT})
        labels[0] = "IBD state"
    ax.set_yticks(list(labels), list(labels.values()))
    ax.set_ylim(-0.6 if not a.no_ibd else 0.4, 2.6)
    ax.legend(handles=[Patch(color=BULL, label="bullish / UPTREND"), Patch(color=NEUTRAL, label="IBD NEUTRAL"),
                       Patch(color=NOT, label="not bullish / DOWNTREND")],
              loc="lower right", bbox_to_anchor=(1, 1.0), ncol=3, fontsize=8, frameon=False)

    if equities:
        ax = axes[3]
        shade(ax, sig["breadth_bull"])
        for lab, e in equities.items():
            ax.plot(e.index, e / 1e5, lw=1.4, label=lab)
        ax.set_yscale("log")
        ax.set_ylabel("system equity (Rs lakh)")
        ax.legend(loc="upper left", fontsize=9, frameon=False)
        ax.set_title("System equity, same breadth shading - losses should cluster in orange", loc="left")

    from matplotlib.ticker import FuncFormatter, NullFormatter
    plain = FuncFormatter(lambda v, _: f"{v:,.0f}")
    for x in (axes[0], axes[3]) if equities else (axes[0],):
        x.yaxis.set_major_formatter(plain)
        x.yaxis.set_minor_formatter(plain)
    for x in axes:
        x.grid(axis="y", color="#e6e6e6", lw=0.6)
        for side in ("top", "right"):
            x.spines[side].set_visible(False)
    fig.savefig(out / "trend_signals.png", dpi=130, bbox_inches="tight")
    print(f"saved: {out / 'trend_signals.png'}\n       {out / 'trend_signals.csv'}\n       {out / 'trend_signals_stats.csv'}")


if __name__ == "__main__":
    main()
