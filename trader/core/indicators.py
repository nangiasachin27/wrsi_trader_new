from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder RSI (RMA smoothing). Needs several multiples of `period` of history to converge
    to charting-platform values; seeding differs slightly from TradingView's SMA-seeded RMA."""
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    au = up.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    ad = dn.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return 100.0 - 100.0 / (1.0 + au / ad)


def sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period, min_periods=period).mean()


def efficiency_ratio(close: pd.Series, period: int) -> pd.Series:
    """Kaufman Efficiency Ratio: net directional move over `period` bars, divided by the total
    distance traveled over the same window (sum of each bar's absolute change). 1.0 = price
    moved in a straight line (pure trend, zero wasted motion); near 0 = price churned back and
    forth and netted almost nowhere (pure chop). Unlike ADX (trend/core/indicators.adx), which
    needs 2x period bars to warm up and is itself smoothed, this is a direct, un-smoothed ratio -
    deliberately simple so a market_efficiency-style filter's threshold means exactly what it
    says ("net moved >= X% of the distance traveled"), not an opaque oscillator scale."""
    net_move = (close - close.shift(period)).abs()
    path_length = close.diff().abs().rolling(period, min_periods=period).sum()
    return net_move / path_length.replace(0.0, np.nan)


def bollinger_bands(close: pd.Series, period: int, num_std: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Classic Bollinger Bands: a simple-moving-average midline plus/minus `num_std` rolling
    standard deviations (population std, ddof=0 - matches the common charting-platform
    convention). Returns (lower, mid, upper) in that order, each a pd.Series aligned to `close`.
    Used by range_reversion for a statistics-based range-fade signal, deliberately independent
    of the pivot-clustered S/R zones trader.core.levels builds (sr_reversal's own mechanism) -
    two structurally different ways of saying "this looks stretched", not the same idea twice."""
    mid = sma(close, period)
    std = close.rolling(period, min_periods=period).std(ddof=0)
    return mid - num_std * std, mid, mid + num_std * std


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Wilder's ATR (same RMA smoothing as rsi() above, for consistency)."""
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Wilder's ADX: trend STRENGTH (0-100), not direction - pairs with a directional signal
    like the strategy's own RSI>60 rather than replacing it. Same RMA smoothing (alpha=1/period)
    as rsi()/atr() above for consistency. +DM/-DM use the standard "larger, positive move wins"
    rule (a bar can't count as both up-move and down-move), smoothed the same way as true range,
    then DX = 100*|+DI - -DI|/(+DI + -DI) is itself smoothed once more to get ADX. All three
    stages need `period` bars to warm up, so ADX needs roughly 2x period bars before trusting it -
    mirrors how rsi()/atr() above are also NaN until min_periods is met."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = atr(high, low, close, period)
    plus_dm_s = plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    minus_dm_s = minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    plus_di = 100.0 * plus_dm_s / tr
    minus_di = 100.0 * minus_dm_s / tr
    denom = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denom
    return dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int, multiplier: float) -> tuple[pd.Series, pd.Series]:
    """Standard Supertrend: the recursive final-band logic (not just an instantaneous
    hl2 +/- multiplier*ATR band) is what gives it its characteristic stickiness - a naive
    non-recursive version would flip direction far more often than the real indicator does.
    Returns (line, direction) where direction is +1 (uptrend, line below price) or
    -1 (downtrend, line above price); both are NaN until ATR has enough bars to warm up."""
    atr_ = atr(high, low, close, period)
    hl2 = (high + low) / 2.0
    basic_upper = hl2 + multiplier * atr_
    basic_lower = hl2 - multiplier * atr_

    n = len(close)
    final_upper = [float("nan")] * n
    final_lower = [float("nan")] * n
    direction = [0] * n
    line = [float("nan")] * n

    start = atr_.first_valid_index()
    if start is None:
        idx = close.index
        return pd.Series(line, index=idx), pd.Series(direction, index=idx)
    i0 = close.index.get_loc(start)
    final_upper[i0], final_lower[i0] = basic_upper.iloc[i0], basic_lower.iloc[i0]
    direction[i0] = 1  # arbitrary seed for the first bar; the recursion settles within a few bars
    line[i0] = final_lower[i0]

    for i in range(i0 + 1, n):
        bu, bl = basic_upper.iloc[i], basic_lower.iloc[i]
        final_upper[i] = bu if (bu < final_upper[i - 1] or close.iloc[i - 1] > final_upper[i - 1]) else final_upper[i - 1]
        final_lower[i] = bl if (bl > final_lower[i - 1] or close.iloc[i - 1] < final_lower[i - 1]) else final_lower[i - 1]
        if direction[i - 1] == 1:
            direction[i] = -1 if close.iloc[i] < final_lower[i] else 1
        else:
            direction[i] = 1 if close.iloc[i] > final_upper[i] else -1
        line[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

    idx = close.index
    return pd.Series(line, index=idx), pd.Series(direction, index=idx)


def resample_ohlc(daily: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = daily.resample(rule).agg(agg)
    return out.dropna(subset=["close"])


def candles_for_timeframe(daily: pd.DataFrame, timeframe: str,
                          weekly_rule: str = "W-FRI", monthly_rule: str = "ME") -> pd.DataFrame:
    """Shared by the support/resistance and candlestick modules so 'pass daily/weekly/monthly'
    always means exactly the same resampling a strategy's ctx.weekly()/ctx.monthly() would see -
    one place to get this right instead of three."""
    if timeframe == "daily":
        return daily
    if timeframe == "weekly":
        return resample_ohlc(daily, weekly_rule)
    if timeframe == "monthly":
        return resample_ohlc(daily, monthly_rule)
    raise ValueError(f"timeframe must be 'daily', 'weekly', or 'monthly', got '{timeframe}'")


def cup_shape(weekly_close: pd.Series, cfg: dict) -> tuple[bool, dict]:
    """Heuristic rounded-base ('cup') detector on weekly closes. Not a pattern-recognition
    oracle - backtest it before letting it gate trades."""
    n = cfg["lookback_weeks"]
    if len(weekly_close) < n:
        return False, {}
    v = weekly_close.iloc[-n:].to_numpy(dtype=float)
    left = int(np.argmax(v[: max(2, int(n * cfg["left_rim_search_frac"]))]))
    rim = v[left]
    if left >= len(v) - 2:
        return False, {}
    bot = left + 1 + int(np.argmin(v[left + 1:]))
    depth = (rim - v[bot]) / rim * 100.0
    left_len, right_len = bot - left, len(v) - 1 - bot
    if left_len <= 0 or right_len <= 0:
        return False, {}
    sym = right_len / left_len
    # "recent_uptick_weeks" (default 0 = off) confirms the RIGHT side of the cup is still
    # climbing at the point of detection, not just that it once reached back near the rim and
    # has since gone flat or started rolling over again. 0 keeps the original behavior exactly
    # (only "near the rim" is checked, not direction); a positive value requires today's close
    # to be above the close that many weeks ago - i.e. still-recovering, not merely recovered.
    up_weeks = cfg.get("recent_uptick_weeks", 0)
    recent_up = True
    if up_weeks > 0:
        if len(v) < up_weeks + 1:
            return False, {}
        recent_up = bool(v[-1] > v[-1 - up_weeks])
    ok = (
        cfg["min_depth_pct"] <= depth <= cfg["max_depth_pct"]
        and left_len >= cfg["min_leg_weeks"]
        and right_len >= cfg["min_leg_weeks"]
        and v[-1] >= rim * (1 - cfg["rim_tolerance_pct"] / 100.0)
        and cfg["min_symmetry"] <= sym <= cfg["max_symmetry"]
        and recent_up
    )
    return bool(ok), {"depth_pct": round(depth, 1), "left_wks": left_len, "right_wks": right_len}


def near_ath(daily: pd.DataFrame, cfg: dict) -> tuple[bool, float]:
    """ATH over the *available* history window (10y by default), not necessarily true all-time."""
    col = "high" if cfg["basis"] == "high" else "close"
    ath = float(daily[col].max())
    dist = (ath - float(daily["close"].iloc[-1])) / ath * 100.0
    return dist <= cfg["max_distance_pct"], round(dist, 2)