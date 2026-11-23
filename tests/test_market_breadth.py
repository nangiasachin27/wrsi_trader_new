"""Market breadth (% of universe above its SMA, with hysteresis) and the trend-aware gross cap."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trader.risk.market_breadth import MarketBreadth, breadth_frame, classify


def _f(closes, start="2024-01-01"):
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e6},
                        index=pd.bdate_range(start, periods=len(c)))


def test_breadth_frame_counts_symbols_above_their_own_sma():
    up, down, flat = _f(np.arange(1, 11)), _f(np.arange(10, 0, -1)), _f(np.full(10, 5.0))
    b = breadth_frame({"U": up, "D": down, "F": flat}, ["U", "D", "F", "MISSING"], sma_period=3)
    assert b["n_valid"].iloc[:2].tolist() == [0, 0]          # SMA3 not warm for 2 bars
    assert b["n_valid"].iloc[-1] == 3
    assert abs(b["pct_above"].iloc[-1] - 100 / 3) < 1e-9     # only U is above (flat == SMA is not)


def test_classify_hysteresis_holds_state_inside_the_band():
    pct = pd.Series([70, 50, 46, 44, 50, 54, 56, 50], dtype=float,
                    index=pd.bdate_range("2024-01-01", periods=8))
    st = classify(pct, smooth_days=1, bull_on_pct=55, bull_off_pct=45)["bullish"].tolist()
    #   70 bull | 50,46 stay bull (band) | 44 off | 50,54 stay off | 56 bull | 50 stays bull
    assert st == [True, True, True, False, False, False, True, True]


def test_status_uses_only_data_up_to_asof():
    class P:   # live-style provider (no .data) - must slice itself
        def __init__(self, frames): self.f = frames
        def get_daily(self, s, end): return self.f[s].loc[: pd.Timestamp(end)]
    rise_then_fall = np.r_[np.linspace(10, 20, 60), np.linspace(20, 10, 40)]
    frames = {f"S{i}": _f(rise_then_fall + i * 0.01) for i in range(60)}
    mb = MarketBreadth({"sma_period": 10, "smooth_days": 1})
    idx = frames["S0"].index
    assert mb.status(idx[55].date(), P(frames), list(frames)).bullish is True     # still rising
    assert mb.status(idx[-1].date(), P(frames), list(frames)).bullish is False    # rolled over
    class H(P):  # backtest-style provider: same answers from the precomputed series
        data = frames
    assert MarketBreadth({"sma_period": 10, "smooth_days": 1}).status(idx[-1].date(), H(frames),
                                                                       list(frames)).bullish is False


def test_status_fails_open_without_enough_symbols():
    class H:
        data = {"A": _f(np.arange(1, 80))}
    st = MarketBreadth({"sma_period": 10}).status(H.data["A"].index[-1].date(), H(), ["A"])
    assert st.bullish is True and "fail open" in st.reason      # 1 symbol < min_symbols 50
