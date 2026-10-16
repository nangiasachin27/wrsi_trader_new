from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from trader.core.indicators import candles_for_timeframe


@dataclass
class Level:
    """A support/resistance ZONE, not a single line - price_low/price_high bracket the band.
    'kind' is NOT stored here: the same physical zone acts as resistance when price is below it
    and support once price breaks above it (role-flipping is normal, real S/R behavior), so
    role is decided relative to a current price by nearest_levels(), not baked into the zone."""
    price_low: float
    price_high: float
    strength: int                 # how many pivot touches clustered into this zone - higher = more significant
    touches: list = field(default_factory=list)   # [(timestamp, price), ...] - the pivots that formed this zone

    @property
    def mid(self) -> float:
        return (self.price_low + self.price_high) / 2.0


def pivot_highs(high: pd.Series, left: int, right: int) -> pd.Series:
    """True at bar i if high[i] is the max within [i-left, i+right] - a local swing high.
    The last `right` bars can never be confirmed yet (need future bars to compare against),
    so they're always False - that's correct, not a bug: a pivot isn't known until it's held."""
    n = len(high)
    out = pd.Series(False, index=high.index)
    vals = high.to_numpy()
    for i in range(left, n - right):
        window = vals[i - left: i + right + 1]
        if vals[i] == window.max() and (window == vals[i]).sum() == 1:
            out.iloc[i] = True
    return out


def pivot_lows(low: pd.Series, left: int, right: int) -> pd.Series:
    n = len(low)
    out = pd.Series(False, index=low.index)
    vals = low.to_numpy()
    for i in range(left, n - right):
        window = vals[i - left: i + right + 1]
        if vals[i] == window.min() and (window == vals[i]).sum() == 1:
            out.iloc[i] = True
    return out


def support_resistance_zones(bars: pd.DataFrame, cfg: dict) -> list[Level]:
    """Find swing pivots over the lookback window, then cluster touches within cluster_pct of
    each other into bands. A zone needs >= min_touches to be reported - a single untested swing
    point isn't support/resistance yet, it's just a high or a low."""
    window = bars.iloc[-cfg["lookback_bars"]:] if cfg.get("lookback_bars") else bars
    left, right = cfg["pivot_left"], cfg["pivot_right"]
    if len(window) < left + right + 1:
        return []

    highs = pivot_highs(window["high"], left, right)
    lows = pivot_lows(window["low"], left, right)
    touches = sorted(
        [(ts, float(window.at[ts, "high"])) for ts in window.index[highs]] +
        [(ts, float(window.at[ts, "low"])) for ts in window.index[lows]],
        key=lambda t: t[1],
    )
    if not touches:
        return []

    tol = cfg["cluster_pct"] / 100.0
    clusters: list[list[tuple]] = [[touches[0]]]
    for ts, price in touches[1:]:
        anchor = clusters[-1][0][1]  # cluster against the FIRST member's price, not a running
        if price <= anchor * (1 + tol):  # average, so a zone can't "drift" far from where it started
            clusters[-1].append((ts, price))
        else:
            clusters.append([(ts, price)])

    zones = []
    for c in clusters:
        if len(c) < cfg["min_touches"]:
            continue
        prices = [p for _, p in c]
        zones.append(Level(price_low=min(prices), price_high=max(prices), strength=len(c), touches=c))
    return sorted(zones, key=lambda z: z.mid)


def nearest_levels(zones: list[Level], current_price: float) -> dict:
    """The two zones a strategy almost always actually wants: the closest support BELOW price
    and the closest resistance ABOVE it. A zone straddling current_price (price sits inside the
    band) counts as neither - it's in the middle of being tested, not yet broken either way."""
    below = [z for z in zones if z.price_high < current_price]
    above = [z for z in zones if z.price_low > current_price]
    return {
        "support": max(below, key=lambda z: z.mid) if below else None,
        "resistance": min(above, key=lambda z: z.mid) if above else None,
    }


def support_resistance(daily: pd.DataFrame, timeframe: str, cfg: dict) -> list[Level]:
    """The timeframe-parameterized entry point: 'pass a stock's daily bars + "weekly" and get
    back its weekly support/resistance zones'. cfg is the config section FOR THAT TIMEFRAME
    (config/levels.json is keyed by timeframe, since a sensible pivot window and cluster
    tolerance genuinely differ between daily/weekly/monthly bars - a 5-bar pivot window means
    5 days on a daily chart but 5 weeks on a weekly one)."""
    bars = candles_for_timeframe(daily, timeframe, cfg.get("weekly_rule", "W-FRI"), cfg.get("monthly_rule", "ME"))
    return support_resistance_zones(bars, cfg)