"""
Fractal (local extrema) + Window-Shifting support/resistance ZONES - the "fractal" levels engine.

Shared by scripts/fractal_sr.py (daily CSV of S/R per ticker) and, when switched on, by the
strategies/filters that previously called trader/core/levels.py (the "legacy" engine):
breakout_trend, sr_reversal and the resistance filter.

Engine switch (added 2026-11-14 so the old behaviour is always one setting away):
  config/sr_engine.json          {"engine": "legacy" | "fractal"}   global default ("legacy")
  strategy config "levels_engine": null | "legacy" | "fractal"       per-strategy override
  filter config   "engine":        null | "legacy" | "fractal"       per-filter override
null/absent = use the global default. With every setting on "legacy" the code paths are exactly
the pre-2026-11-14 ones (trader/core/levels.py + config/levels.json), untouched.

Algorithm (per timeframe; weekly/monthly resampled from daily, W-FRI / ME, completed bars only):
 0. Lookback window in calendar months (config/fractal_sr.json); a warm-up margin before it lets a
    swing on the window's first bars still be confirmed on its left side.
 1. Swing points from two detectors on highs and lows:
    a. Fractal order n: high[i] > high[i-k] and >= high[i+k], k=1..n (n=2 = 5-candle Williams fractal).
    b. Window shifting (W, P): a bar that stays a sliding W-bar window's argmax for P shifts = the
       extreme of a (W+P-1)-bar span with >= P-1 bars each side, not necessarily centred.
 2. Found by both = "confirmed" (highest weight).
 3. Highs and lows clustered together in log price (broken resistance becomes support); tolerance =
    cluster_atr_mult x ATR% clamped; density-greedy seeding so zone width <= 2 x tolerance.
 4. score = sum(weight x 0.5^(age/half_life)) x polarity bonus; valid if touches >= min_touches or a
    fresh confirmed single swing.
 5. Role vs a reference price: wholly below = support, wholly above = resistance; a zone the price
    sits INSIDE takes its role from the side price approached it from (rallied into it = resistance).
 6. S1/R1 = nearest valid zone; SMAJ/RMAJ = strongest. Edge fallback for an empty side.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import date, timedelta

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from trader.core.config import ROOT

OHLC = ["open", "high", "low", "close"]
ENGINES = ("legacy", "fractal")


# ============================================================================================
# Pivot detectors
# ============================================================================================
def fractal_pivots(x: np.ndarray, n: int, kind: str) -> np.ndarray:
    """Indices of order-n fractal highs (kind='high') or lows (kind='low').
    Needs n bars on both sides, so the last n bars can never qualify (unconfirmed)."""
    if n < 1 or len(x) < 2 * n + 1:
        return np.empty(0, dtype=int)
    w = sliding_window_view(x, 2 * n + 1)
    c, left, right = w[:, n], w[:, :n], w[:, n + 1:]
    if kind == "high":
        ok = (c > left.max(axis=1)) & (c >= right.max(axis=1))
    else:
        ok = (c < left.min(axis=1)) & (c <= right.min(axis=1))
    return np.flatnonzero(ok) + n


def window_shift_pivots(x: np.ndarray, window: int, persist: int, kind: str) -> np.ndarray:
    """Indices that remain the argmax (kind='high') / argmin (kind='low') of a sliding
    `window`-bar window for >= `persist` consecutive one-bar shifts.
    np.argmax/argmin return the FIRST occurrence, so ties resolve to the earliest bar,
    consistent with fractal_pivots()."""
    if window < 2 or persist < 1 or len(x) < window + persist - 1:
        return np.empty(0, dtype=int)
    w = sliding_window_view(x, window)
    pos = w.argmax(axis=1) if kind == "high" else w.argmin(axis=1)
    idx = pos + np.arange(len(w))                     # absolute bar index of each window's extreme
    starts = np.r_[True, idx[1:] != idx[:-1]]         # run-length encode consecutive identical idx
    run_ids = np.cumsum(starts) - 1
    run_len = np.bincount(run_ids)
    return np.unique(idx[starts][run_len >= persist])


def detect_pivots(bars: pd.DataFrame, p: dict, wts: dict) -> pd.DataFrame:
    """All swing highs/lows from both detectors, merged. Columns: idx, price, side ('H'/'L'),
    src ('F' fractal only, 'W' window only, 'FW' both), weight."""
    rows = []
    for side, col, kind in (("H", "high", "high"), ("L", "low", "low")):
        x = bars[col].to_numpy(dtype=float)
        f = set(fractal_pivots(x, p["fractal_n"], kind).tolist())
        w = set(window_shift_pivots(x, p["window"], p["persist"], kind).tolist())
        for i in sorted(f | w):
            src = "FW" if (i in f and i in w) else ("F" if i in f else "W")
            weight = {"FW": wts["both"], "F": wts["fractal_only"], "W": wts["window_only"]}[src]
            rows.append((i, x[i], side, src, weight))
    return pd.DataFrame(rows, columns=["idx", "price", "side", "src", "weight"])


# ============================================================================================
# Zones
# ============================================================================================
@dataclass
class Zone:
    low: float
    high: float
    touches: int        # distinct bars that formed the zone
    score: float
    confirmed: bool     # contains at least one pivot found by BOTH detectors
    polarity: bool      # contains both swing highs and swing lows (role reversal seen)
    last_idx: int       # bar index of the most recent member pivot
    valid: bool = False
    edge: bool = False  # unconfirmed lookback extreme (edge_fallback), not a detected pivot cluster
    role: str = ""      # "S" / "R" relative to the reference close (set by analyze_timeframe)

    @property
    def mid(self) -> float:
        return (self.low + self.high) / 2.0


def wilder_atr(bars: pd.DataFrame, n: int) -> float:
    h, l, c = (bars[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    return float(pd.Series(tr).ewm(alpha=1.0 / n, adjust=False).mean().iloc[-1])


def cluster_zones(piv: pd.DataFrame, n_bars: int, tol_pct: float, min_width_pct: float,
                  p: dict, wts: dict) -> list[Zone]:
    if piv.empty:
        return []
    price = piv["price"].to_numpy(dtype=float)
    bar = piv["idx"].to_numpy(dtype=int)
    is_high = (piv["side"] == "H").to_numpy()
    is_both = (piv["src"] == "FW").to_numpy()
    lp = np.log(price)
    decayed = piv["weight"].to_numpy(dtype=float) * 0.5 ** (((n_bars - 1) - bar) / p["recency_half_life_bars"])
    ltol = np.log1p(tol_pct / 100.0)
    near_all = np.abs(lp[:, None] - lp[None, :]) <= ltol      # pairwise, computed once

    remaining = np.ones(len(piv), dtype=bool)
    zones: list[Zone] = []
    while remaining.any():
        cand = np.flatnonzero(remaining)
        near = near_all[np.ix_(cand, cand)]
        mass = np.round(near @ decayed[cand], 12)
        # densest seed wins; ties -> most recent pivot
        top = np.flatnonzero(mass == mass.max())
        seed = top[np.argmax(bar[cand[top]])]
        members = cand[near[seed]]
        remaining[members] = False

        lo, hi = float(price[members].min()), float(price[members].max())
        mid = (lo + hi) / 2.0
        half = mid * min_width_pct / 200.0
        if hi - lo < 2 * half:
            lo, hi = mid - half, mid + half
        polarity = bool(is_high[members].any() and (~is_high[members]).any())
        score = float(decayed[members].sum()) * (wts["polarity_bonus"] if polarity else 1.0)
        touches = int(np.unique(bar[members]).size)
        confirmed = bool(is_both[members].any())
        z = Zone(lo, hi, touches, score, confirmed, polarity, int(bar[members].max()))
        z.valid = touches >= p["min_touches"] or (
            p["accept_confirmed_single"] and confirmed and score >= p.get("single_pivot_min_score", 0.0))
        zones.append(z)
    return sorted(zones, key=lambda z: z.mid)


def lookback_window(bars: pd.DataFrame, p: dict, asof: date | None) -> tuple[pd.DataFrame, int]:
    """Slice to the lookback window plus a warm-up margin in front of it.
    Returns (sliced bars, start) where bars.iloc[start:] is the window proper. Swings are only
    accepted inside the window, but a swing on the window's first bar still needs fractal_n /
    persist-1 bars to its LEFT to be confirmed - the warm-up supplies them (and seeds the ATR)."""
    months = p.get("lookback_months") or 0
    if not months or bars.empty:
        return bars, 0
    end = pd.Timestamp(asof) if asof else bars.index[-1]
    w_start = int(bars.index.searchsorted(end - pd.DateOffset(months=months), side="left"))
    warm = max(p["fractal_n"], p["window"] + p["persist"] - 1, 2 * p["atr_period"])
    s0 = max(0, w_start - warm)
    return bars.iloc[s0:], w_start - s0


def approach_side(bars: pd.DataFrame, z: Zone) -> int:
    """Side price came FROM into a zone it is currently inside: +1 above, -1 below, 0 unknown.
    Walks back over the bars whose range overlaps the zone (the current visit) and returns the
    side of the last close before that visit started - so a one-day close beyond the zone inside
    the visit (a failed breakout) does not flip the role."""
    h, l, c = (bars[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
    i = len(c) - 1
    while i >= 0 and l[i] <= z.high and h[i] >= z.low:
        i -= 1
    if i < 0:
        return 0
    return 1 if c[i] > z.high else (-1 if c[i] < z.low else 0)


@dataclass
class TFResult:
    atr_pct: float | None
    sup: list            # valid support zones, nearest first
    res: list            # valid resistance zones, nearest first
    sup_major: Zone | None   # strongest (highest score) support zone
    res_major: Zone | None   # strongest resistance zone
    bars: pd.DataFrame | None
    zones: list          # every zone found (valid or not), roles set for this reference price
    note: str | None = None


@dataclass
class ZoneBuild:
    """Reference-price-independent part of the analysis (cacheable per symbol/day/timeframe)."""
    bars: pd.DataFrame
    start: int
    atr_pct: float
    tol_pct: float
    min_width_pct: float
    zones: list


def build_zones(bars: pd.DataFrame, p: dict, wts: dict, asof: date | None = None) -> ZoneBuild | str:
    """Detect + cluster. Returns a ZoneBuild, or an 'insufficient_bars(n)' note string."""
    bars, start = lookback_window(bars, p, asof)
    if len(bars) - start < p["min_bars"]:
        return f"insufficient_bars({len(bars) - start})"
    atr_pct = 100.0 * wilder_atr(bars, p["atr_period"]) / float(bars["close"].iloc[-1])
    tol_pct = float(np.clip(p["cluster_atr_mult"] * atr_pct, p["tol_min_pct"], p["tol_max_pct"]))
    min_width_pct = p["min_width_atr_mult"] * atr_pct
    piv = detect_pivots(bars, p, wts)
    piv = piv[piv["idx"] >= start].reset_index(drop=True)       # swings inside the window only
    zones = cluster_zones(piv, len(bars), tol_pct, min_width_pct, p, wts)
    return ZoneBuild(bars, start, atr_pct, tol_pct, min_width_pct, zones)


def classify_zones(zb: ZoneBuild, p: dict, ref_close: float) -> TFResult:
    """Roles, nearest + major, edge fallback - for one reference price. Never mutates zb (the
    same cached build is classified against different prices, e.g. breakout_trend's previous
    close and a filter's current close)."""
    bars, start = zb.bars, zb.start
    zones = []
    for z0 in zb.zones:
        if z0.high < ref_close:
            role = "S"
        elif z0.low > ref_close:
            role = "R"
        else:
            side = approach_side(bars, z0)
            role = "R" if side < 0 else ("S" if side > 0 else ("S" if z0.mid < ref_close else "R"))
        zones.append(replace(z0, role=role))
    valid = [z for z in zones if z.valid]
    sup = sorted((z for z in valid if z.role == "S"), key=lambda z: -z.high)
    res = sorted((z for z in valid if z.role == "R"), key=lambda z: z.low)

    if p.get("edge_fallback"):
        # Fractals need bars on the right, so the most recent top/bottom is invisible to them.
        # If a side is empty and price has pulled back from the window extreme by at least the
        # cluster tolerance, expose that extreme as an UNCONFIRMED zone (score 0, edge=True).
        w, pull = zb.min_width_pct / 100.0, zb.tol_pct / 100.0
        if not res:
            i = start + int(np.argmax(bars["high"].to_numpy()[start:]))
            hi = float(bars["high"].iat[i])
            if hi >= ref_close * (1 + pull):
                res = [Zone(hi * (1 - w), hi, 1, 0.0, False, False, i, valid=True, edge=True, role="R")]
        if not sup:
            i = start + int(np.argmin(bars["low"].to_numpy()[start:]))
            lo = float(bars["low"].iat[i])
            if lo <= ref_close * (1 - pull):
                sup = [Zone(lo, lo * (1 + w), 1, 0.0, False, False, i, valid=True, edge=True, role="S")]

    smaj = max(sup, key=lambda z: (z.score, z.high)) if sup else None
    rmaj = max(res, key=lambda z: (z.score, -z.low)) if res else None
    return TFResult(zb.atr_pct, sup, res, smaj, rmaj, bars, zones)


def analyze_timeframe(bars: pd.DataFrame, p: dict, wts: dict, ref_close: float,
                      asof: date | None = None) -> TFResult:
    zb = build_zones(bars, p, wts, asof)
    if isinstance(zb, str):
        return TFResult(None, [], [], None, None, None, [], zb)
    return classify_zones(zb, p, ref_close)


# ============================================================================================
# Bars / calendar
# ============================================================================================
def last_session_on_or_before(d: date, holidays: set[date]) -> date:
    while d.weekday() >= 5 or d in holidays:
        d -= timedelta(days=1)
    return d


def resample(daily: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    return daily[OHLC].resample(rule).agg(agg).dropna(subset=["close"])


def timeframe_bars(daily: pd.DataFrame, tf: str, dcfg: dict, asof: date,
                   holidays: set[date]) -> tuple[pd.DataFrame, bool]:
    """Bars for tf. If completed_bars_only, drop the last weekly/monthly bar when its period's
    final trading session is after `asof` (still forming). Returns (bars, dropped_partial)."""
    if tf == "daily":
        return daily, False
    bars = resample(daily, dcfg["weekly_rule"] if tf == "weekly" else dcfg["monthly_rule"])
    if dcfg["completed_bars_only"] and len(bars):
        period_end = bars.index[-1].date()          # W-FRI label = that Friday, ME label = month end
        if last_session_on_or_before(period_end, holidays) > asof:
            return bars.iloc[:-1], True
    return bars, False


# ============================================================================================
# Engine-facing API (strategies / filters)
# ============================================================================================
_SETTINGS: dict = {}


def _read_json(name: str) -> dict:
    with open(ROOT / "config" / name, encoding="utf-8") as f:
        return json.load(f)


def default_engine() -> str:
    """Global default from config/sr_engine.json ("legacy" if the file or key is missing)."""
    if "engine" not in _SETTINGS:
        try:
            eng = _read_json("sr_engine.json").get("engine", "legacy")
        except FileNotFoundError:
            eng = "legacy"
        if eng not in ENGINES:
            raise ValueError(f"config/sr_engine.json: engine must be one of {ENGINES}, got {eng!r}")
        _SETTINGS["engine"] = eng
    return _SETTINGS["engine"]


def resolve_engine(override: str | None) -> str:
    """Per-consumer override if set, else the global default."""
    if override is None:
        return default_engine()
    if override not in ENGINES:
        raise ValueError(f"levels engine must be one of {ENGINES} or null, got {override!r}")
    return override


def fractal_config() -> dict:
    if "cfg" not in _SETTINGS:
        _SETTINGS["cfg"] = _read_json("fractal_sr.json")
    return _SETTINGS["cfg"]


def _holidays() -> set[date]:
    if "holidays" not in _SETTINGS:
        try:
            _SETTINGS["holidays"] = {date.fromisoformat(d) for d in _read_json("holidays.json")["holidays"]}
        except FileNotFoundError:
            _SETTINGS["holidays"] = set()
    return _SETTINGS["holidays"]


def reset_settings_cache() -> None:
    """Forget cached config + zone builds (tests, or after editing config in a long-lived process)."""
    _SETTINGS.clear()
    _BUILD_CACHE.clear()


_BUILD_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_BUILD_CACHE_MAX = 4096


def zones_build(daily: pd.DataFrame, timeframe: str, asof: date | None = None) -> ZoneBuild | str:
    """Cached build_zones for one symbol's daily bars on one day. The cache value holds a reference
    to `daily` so its id() cannot be recycled while the entry is alive."""
    asof = asof or daily.index[-1].date()
    key = (id(daily), len(daily), daily.index[-1], timeframe, asof)
    hit = _BUILD_CACHE.get(key)
    if hit is not None and hit[0] is daily:
        _BUILD_CACHE.move_to_end(key)
        return hit[1]
    cfg = fractal_config()
    bars, _ = timeframe_bars(daily, timeframe, cfg["data"], asof, _holidays())
    zb = build_zones(bars, cfg["timeframes"][timeframe], cfg["weights"], asof)
    _BUILD_CACHE[key] = (daily, zb)
    if len(_BUILD_CACHE) > _BUILD_CACHE_MAX:
        _BUILD_CACHE.popitem(last=False)
    return zb


def fractal_result(daily: pd.DataFrame, timeframe: str, price: float, asof: date | None = None) -> TFResult:
    zb = zones_build(daily, timeframe, asof)
    if isinstance(zb, str):
        return TFResult(None, [], [], None, None, None, [], zb)
    return classify_zones(zb, fractal_config()["timeframes"][timeframe], price)


def as_level(z: Zone | None):
    """Adapt a Zone to trader.core.levels.Level - the type every legacy consumer reads
    (.price_low / .price_high / .strength / .mid) - so strategies need no other changes."""
    if z is None:
        return None
    from trader.core.levels import Level
    lv = Level(price_low=float(z.low), price_high=float(z.high), strength=int(z.touches))
    lv.score, lv.edge, lv.role = float(z.score), bool(z.edge), z.role   # extra info, ignored by legacy code
    return lv


def fractal_nearest(daily: pd.DataFrame, timeframe: str, price: float, asof: date | None = None) -> dict:
    """Same shape as trader.core.levels.nearest_levels(): {"support": Level|None, "resistance": Level|None},
    plus "result": the full TFResult (majors, ATR) for consumers that want more."""
    r = fractal_result(daily, timeframe, price, asof)
    return {"support": as_level(r.sup[0] if r.sup else None),
            "resistance": as_level(r.res[0] if r.res else None),
            "result": r}
