"""Rank regression_check runs by drawdown-adjusted return, against a baseline run.

Reads each run folder's daily_equity.csv + trade_log.csv (as written by regression_check.py)
and prints one row per run, best MAR (CAGR / |maxDD|) first (losing runs below, by CAGR):

  python scripts/rank_runs.py --pattern "state/regression_2024-07-01_2026-11-13_s_*" --baseline s_base
  python scripts/rank_runs.py --pattern "state/regression_2021-01-01_2026-11-13_f_*" --baseline f_base ^
         --segment 2024-07-01                # also score the 2024-07 -> end slice of continuous runs

Columns (whole run, or the --segment slice when given):
  CAGR / maxDD / MAR   - MAR = CAGR / |maxDD|; risk_pct scales CAGR and maxDD roughly together,
                         so MAR is the config's edge and risk_pct is the dial you set afterwards
  ulcer                - RMS of the drawdown series (%): depth AND duration of time under water
  dd>10%               - share of sessions spent more than 10% below the running peak
  worst_m              - worst calendar month (%)
  trades / PF / win%   - closed trades exiting inside the scored window
  dCAGR / dDD          - vs --baseline (dDD > 0 = shallower drawdown)
  verdict              - BETTER: dDD >= +1pp and dCAGR >= -0.5pp  (the stated goal)
                         TRADE  : MAR up (CAGR up if baseline loses money), but not BETTER
                         worse  : everything else
Full-window runs (start <= 2021-06) also get per-calendar-year returns, so "did it hurt the
2021/2023 gainers" is answered directly. Writes rank_runs.csv next to the first run folder.
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _metrics(eq: pd.Series, tl: pd.DataFrame) -> dict:
    eq = eq.dropna()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = ((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) * 100
    dd = (eq / eq.cummax() - 1) * 100
    mdd = dd.min()
    monthly = eq.resample("ME").last().pct_change().dropna() * 100
    first_m = (eq.resample("ME").last().iloc[0] / eq.iloc[0] - 1) * 100
    worst_m = min([first_m, *monthly.tolist()]) if len(monthly) else first_m
    pnl = tl["pnl"] if not tl.empty else pd.Series(dtype=float)
    gross_loss = -pnl[pnl < 0].sum()
    return {
        "CAGR": round(cagr, 2), "maxDD": round(mdd, 2),
        "MAR": round(cagr / abs(mdd), 2) if mdd < 0 else np.nan,
        "ulcer": round(float(np.sqrt((dd ** 2).mean())), 2),
        "dd>10%": round(float((dd < -10).mean() * 100), 1),
        "worst_m": round(float(worst_m), 2),
        "final": round(float(eq.iloc[-1])),
        "trades": int(len(pnl)),
        "PF": round(pnl[pnl > 0].sum() / gross_loss, 2) if gross_loss > 0 else np.nan,
        "win%": round(float((pnl > 0).mean() * 100), 1) if len(pnl) else np.nan,
    }


def _per_strategy(tl: pd.DataFrame) -> str:
    if tl.empty:
        return ""
    out = []
    for s, g in tl.groupby("strategy"):
        gl = -g.loc[g.pnl < 0, "pnl"].sum()
        pf = g.loc[g.pnl > 0, "pnl"].sum() / gl if gl > 0 else float("inf")
        out.append(f"{s[:6]} {len(g)}/{pf:.2f}/{g.pnl.sum() / 1e5:+.1f}L")
    return "  ".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", action="append", required=True, help="glob of run folders (repeatable)")
    ap.add_argument("--baseline", required=True, help="label of the baseline run (folder name suffix)")
    ap.add_argument("--segment", default=None, help="score only from this date on (YYYY-MM-DD)")
    a = ap.parse_args()

    dirs = sorted({Path(p) for pat in a.pattern for p in glob.glob(pat) if Path(p).is_dir()})
    if not dirs:
        sys.exit(f"no run folders match {a.pattern}")
    rows, years = [], {}
    for d in dirs:
        m = re.match(r"regression_(\d{4}-\d\d-\d\d)_(\d{4}-\d\d-\d\d)_(.+)$", d.name)
        label = m.group(3) if m else d.name
        eqf, tlf = d / "daily_equity.csv", d / "trade_log.csv"
        if not eqf.exists():
            print(f"  skip {label}: no daily_equity.csv (run failed or still running)")
            continue
        eq = pd.read_csv(eqf, parse_dates=["date"]).set_index("date")["equity"]
        tl = pd.read_csv(tlf, parse_dates=["exit_date"]) if tlf.exists() and tlf.stat().st_size > 5 else pd.DataFrame()
        if a.segment:
            seg = pd.Timestamp(a.segment)
            prior = eq[eq.index < seg]
            eq = pd.concat([prior.iloc[-1:], eq[eq.index >= seg]]) if len(prior) else eq[eq.index >= seg]
            if not tl.empty:
                tl = tl[tl.exit_date >= seg]
        r = {"run": label, **_metrics(eq, tl), "per strategy n/PF/net": _per_strategy(tl)}
        rows.append(r)
        if eq.index[0] <= pd.Timestamp("2021-06-30") and not a.segment:
            ye = eq.resample("YE").last()
            yr = pd.concat([eq.iloc[:1], ye]).pct_change().dropna() * 100
            years[label] = {str(i.year): round(v, 1) for i, v in yr.items()}

    df = pd.DataFrame(rows).set_index("run")
    if a.baseline not in df.index:
        sys.exit(f"baseline {a.baseline!r} not among: {', '.join(df.index)}")
    b = df.loc[a.baseline]
    df["dCAGR"] = (df["CAGR"] - b["CAGR"]).round(2)
    df["dDD"] = (df["maxDD"] - b["maxDD"]).round(2)
    # sort key: MAR for profitable runs; losing runs (where CAGR/|DD| misorders) all rank below
    # every profitable one, ordered by CAGR
    key = np.where(df.CAGR > 0, df.MAR, -1000 + df.CAGR)
    df["_key"] = key
    bkey = df.loc[a.baseline, "_key"]
    df["verdict"] = np.where((df.dDD >= 1.0) & (df.dCAGR >= -0.5), "BETTER",
                             np.where(df._key > bkey, "TRADE", "worse"))
    df.loc[a.baseline, "verdict"] = "baseline"
    df = df.sort_values("_key", ascending=False).drop(columns="_key")

    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 90)
    scope = f"segment {a.segment} -> end" if a.segment else "whole run"
    print(f"\nRANKED by MAR ({scope}), baseline = {a.baseline}\n")
    print(df.drop(columns=["per strategy n/PF/net"]).to_string())
    print("\nper strategy (trades / PF / net):")
    for run, s in df["per strategy n/PF/net"].items():
        print(f"  {run:22s} {s}")
    if years:
        print("\ncalendar-year returns (%):")
        print(pd.DataFrame(years).T.reindex(df.index).to_string())
    out = dirs[0].parent / ("rank_runs" + (f"_seg{a.segment}" if a.segment else "") + ".csv")
    try:
        df.to_csv(out)
    except PermissionError:          # the CSV is open in Excel (Windows locks it): don't lose the run
        out = out.with_name(out.stem + pd.Timestamp.now().strftime("_%Y%m%d_%H%M%S") + ".csv")
        df.to_csv(out)
        print(f"\n(rank_runs.csv is locked - probably open in Excel - wrote a new file instead)")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
