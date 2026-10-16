import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def synth(seed: int, years: int = 9, start: str = "2016-01-01") -> pd.DataFrame:
    """Regime-switching random walk so weekly/monthly RSI actually cycles through 60."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=int(252 * years))
    drift, out, price = 0.0, [], 100.0
    for i in range(len(idx)):
        if i % 120 == 0:
            drift = rng.choice([-0.0008, 0.0002, 0.0012, 0.0020])
        r = drift + rng.normal(0, 0.013)
        o = price * (1 + rng.normal(0, 0.004))
        price = price * (1 + r)
        hi, lo = max(o, price) * 1.006, min(o, price) * 0.994
        out.append((o, hi, lo, price, 1_000_000))
    return pd.DataFrame(out, index=idx, columns=["open", "high", "low", "close", "volume"])


@pytest.fixture(scope="session")
def data():
    return {f"S{i}": synth(i) for i in range(1, 9)}
