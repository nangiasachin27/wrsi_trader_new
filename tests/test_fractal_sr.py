"""Tests for scripts/fractal_sr.py (standalone Fractal + Window-Shifting S/R)."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("fractal_sr", ROOT / "scripts" / "fractal_sr.py")
fsr = importlib.util.module_from_spec(_spec)
sys.modules["fractal_sr"] = fsr
_spec.loader.exec_module(fsr)

CFG = json.loads((ROOT / "config" / "fractal_sr.json").read_text())


def test_fractal_basic():
    x = np.array([1, 2, 3, 2, 1, 2, 5, 2, 1], dtype=float)
    assert fsr.fractal_pivots(x, 2, "high").tolist() == [2, 6]
    assert fsr.fractal_pivots(x, 2, "low").tolist() == [4]


def test_fractal_equal_highs_give_one_pivot():
    x = np.array([1, 2, 5, 5, 2, 1, 0], dtype=float)
    assert fsr.fractal_pivots(x, 2, "high").tolist() == [2]


def test_window_shift_catches_asymmetric_lower_high():
    # bar 6 (8.0) is a lower high 3 bars after the big high at bar 3 (10.0):
    # a symmetric order-4 fractal rejects it, window-shifting (W=6, P=3) keeps it
    x = np.array([1, 2, 3, 10, 4, 5, 8, 6, 5, 4, 3, 2, 1, 0.5])
    assert 6 not in fsr.fractal_pivots(x, 4, "high")
    ws = fsr.window_shift_pivots(x, 6, 3, "high").tolist()
    assert 3 in ws and 6 in ws


def test_window_shift_needs_right_side_confirmation():
    x = np.arange(30, dtype=float)                       # straight up: last bar is not a pivot yet
    assert fsr.window_shift_pivots(x, 10, 5, "high").size == 0
    assert fsr.fractal_pivots(x, 5, "high").size == 0


def _range_bound(n=400, lo=100.0, hi=120.0, period=40, seed=7):
    rng = np.random.default_rng(seed)
    mid, amp = (lo + hi) / 2, (hi - lo) / 2
    c = mid + amp * np.sin(np.arange(n) * 2 * np.pi / period) + rng.normal(0, 0.3, n)
    o = np.r_[c[0], c[:-1]]
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.003,
                         "low": np.minimum(o, c) * 0.997, "close": c}, index=idx)


def test_range_bound_series_finds_the_range_edges():
    df = _range_bound(n=400)                              # ends mid-cycle, close ~110
    ref = float(df["close"].iloc[-1])
    r = fsr.analyze_timeframe(df, CFG["timeframes"]["daily"], CFG["weights"], ref)
    sup, res = r.sup, r.res
    assert r.note is None and sup and res
    assert abs(sup[0].mid - 99.7) < 1.5, sup[0]
    assert abs(res[0].mid - 120.3) < 1.5, res[0]
    assert sup[0].touches >= 3 and res[0].touches >= 3 and sup[0].polarity is False   # ~3 cycles in 6 months
    assert all(z.mid < ref for z in sup) and all(z.mid >= ref for z in res)


def test_zone_width_is_bounded_and_min_width_applied():
    df = _range_bound(n=400)
    ref = float(df["close"].iloc[-1])
    p = CFG["timeframes"]["daily"]
    r = fsr.analyze_timeframe(df, p, CFG["weights"], ref)
    atr, zones = r.atr_pct, r.zones
    tol = float(np.clip(p["cluster_atr_mult"] * atr, p["tol_min_pct"], p["tol_max_pct"]))
    for z in zones:
        width_pct = 100 * (z.high - z.low) / z.mid
        assert width_pct <= 2 * tol + 1e-6 or width_pct <= p["min_width_atr_mult"] * atr + 1e-6
        assert width_pct >= p["min_width_atr_mult"] * atr - 1e-6


def test_partial_week_and_month_dropped():
    idx = pd.bdate_range("2026-01-01", "2026-11-11")      # Wed 11-Nov-2026
    daily = pd.DataFrame(100.0, index=idx, columns=fsr.OHLC)
    w, dropped = fsr.timeframe_bars(daily, "weekly", CFG["data"], date(2026, 11, 11), set())
    assert dropped and w.index[-1] == pd.Timestamp("2026-11-06")
    m, dropped = fsr.timeframe_bars(daily, "monthly", CFG["data"], date(2026, 11, 11), set())
    assert dropped and m.index[-1] == pd.Timestamp("2026-10-31")
    # Friday holiday: Thursday's bar completes the week
    idx2 = pd.bdate_range("2026-01-01", "2026-11-12")
    daily2 = pd.DataFrame(100.0, index=idx2, columns=fsr.OHLC)
    w2, dropped2 = fsr.timeframe_bars(daily2, "weekly", CFG["data"], date(2026, 11, 12), {date(2026, 11, 13)})
    assert not dropped2 and w2.index[-1] == pd.Timestamp("2026-11-13")


def test_process_symbol_no_lookahead_and_ca_cut(data):
    df = data["S1"]
    asof = df.index[-200].date()
    cut = df.index[-1500].date()
    row, zrows = fsr.process_symbol("S1", df, asof, CFG, set(), [cut], want_zones=True)
    assert row["close"] == round(float(df["close"].loc[pd.Timestamp(asof)]), 2)
    assert f"ca_cut:{cut.isoformat()}" in row["notes"]
    assert all(z["last_touch"] <= asof.isoformat() and z["last_touch"] >= cut.isoformat() for z in zrows)
    for t in ("d", "w", "m"):
        if pd.notna(row.get(f"{t}_s1_high")):
            assert row[f"{t}_s1_low"] <= row[f"{t}_s1_high"]
            assert row[f"{t}_s1_low"] <= row["close"]       # below price, or price inside (fell into it)
        if pd.notna(row.get(f"{t}_r1_low")):
            assert row[f"{t}_r1_high"] >= row["close"]      # above price, or price inside (rallied into it)


def test_short_history_flagged_not_crashing():
    df = _range_bound(n=45)
    row, _ = fsr.process_symbol("NEW", df, df.index[-1].date(), CFG, set(), [], want_zones=False)
    assert "d:insufficient_bars(45)" in row["notes"] and "m:insufficient_bars" in row["notes"]


def test_cli_end_to_end(tmp_path, data):
    (tmp_path / "config").mkdir()
    (tmp_path / "data_cache").mkdir()
    shutil.copy(ROOT / "config" / "fractal_sr.json", tmp_path / "config")
    syms = ["S1", "S2", "S3"]
    (tmp_path / "config" / "universe.json").write_text(json.dumps(
        {"instruments": {s: f"NSE_EQ|X{s}" for s in syms}, "lists": {"default": syms}}))
    (tmp_path / "config" / "system_config.json").write_text(json.dumps({"paths": {"cache_dir": "data_cache"}}))
    for s in syms:
        data[s].to_csv(tmp_path / "data_cache" / f"{s}.csv", index_label="date")
    env = {**os.environ, "TRADER_ROOT": str(tmp_path)}
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "fractal_sr.py"),
                        "--zones-out", str(tmp_path / "zones.csv")],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    asof = data["S1"].index[-1].date().isoformat()
    out = pd.read_csv(tmp_path / "state" / f"sr_levels_{asof}.csv")
    assert out["ticker"].tolist() == syms and len(out.columns) == 3 + 3 * (1 + 4 * 5) + 1   # s1,smaj,r1,rmaj
    assert (tmp_path / "zones.csv").exists()


def test_lone_old_pivot_is_not_valid_but_fresh_one_is():
    p = dict(CFG["timeframes"]["daily"])
    piv = pd.DataFrame({"idx": [10, 490], "price": [100.0, 150.0], "side": ["L", "L"],
                        "src": ["FW", "FW"], "weight": [1.5, 1.5]})
    zones = fsr.cluster_zones(piv, 500, 1.0, 0.5, p, CFG["weights"])
    by_price = {round(z.mid): z for z in zones}
    assert not by_price[100].valid          # 490 bars old, score ~1.5*0.5^(489/126) = 0.1
    assert by_price[150].valid              # 9 bars old, score ~1.43


def test_edge_fallback_exposes_recent_unconfirmed_top():
    # steady rise to a top on the previous bar, then a 3% drop: the top has 1 right-side bar, too
    # few for either detector (5-candle fractal needs 2), so without the fallback there is no resistance
    up = np.linspace(100, 200, 300)
    c = np.r_[up, [194]]
    o = np.r_[c[0], c[:-1]]
    idx = pd.bdate_range("2024-01-01", periods=len(c))
    df = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.002, "low": np.minimum(o, c) * 0.998,
                       "close": c}, index=idx)
    ref = float(df["close"].iloc[-1])
    p = dict(CFG["timeframes"]["daily"])
    res = fsr.analyze_timeframe(df, {**p, "edge_fallback": False}, CFG["weights"], ref).res
    assert res == []
    res = fsr.analyze_timeframe(df, p, CFG["weights"], ref).res
    assert res and res[0].edge and res[0].score == 0.0
    assert abs(res[0].high - df["high"].max()) < 1e-9
    row, _ = fsr.process_symbol("X", df, df.index[-1].date(), CFG, set(), [], want_zones=False)
    assert "d:r1_edge" in row["notes"]


def test_edge_fallback_not_used_when_price_is_at_the_high():
    c = np.linspace(100, 200, 300)
    o = np.r_[c[0], c[:-1]]
    idx = pd.bdate_range("2024-01-01", periods=len(c))
    df = pd.DataFrame({"open": o, "high": c * 1.001, "low": o * 0.999, "close": c}, index=idx)
    res = fsr.analyze_timeframe(df, CFG["timeframes"]["daily"], CFG["weights"], float(c[-1])).res
    assert res == []


def test_lookback_window_excludes_old_swings_but_confirms_swings_at_its_start():
    # flat ~100 tape with two sharp lows: 8 months ago (outside the 6-month window) and on the
    # window's 3rd session (inside, but needs pre-window bars on its left to be confirmed)
    idx = pd.bdate_range("2025-01-01", "2026-11-13")
    rng = np.random.default_rng(3)
    c = 100 + rng.normal(0, 0.4, len(idx))
    asof = idx[-1].date()
    cutoff = pd.Timestamp(asof) - pd.DateOffset(months=6)
    old = int(idx.searchsorted(cutoff - pd.DateOffset(months=2)))
    new = int(idx.searchsorted(cutoff)) + 2
    c[old], c[new] = 80.0, 90.0
    o = np.r_[c[0], c[:-1]]
    df = pd.DataFrame({"open": o, "high": np.maximum(o, c) + 0.2, "low": np.minimum(o, c) - 0.2,
                       "close": c}, index=idx)
    p = CFG["timeframes"]["daily"]
    r = fsr.analyze_timeframe(df, p, CFG["weights"], float(c[-1]), asof)
    bars, zones = r.bars, r.zones
    lows = [round(z.low) for z in zones]
    assert 90 in lows or 89 in lows                     # swing at window start detected
    assert all(z.low > 85 for z in zones)               # 8-month-old swing excluded
    assert all(bars.index[z.last_idx] >= cutoff for z in zones)
    # with no window the old swing comes back
    zones_all = fsr.analyze_timeframe(df, {**p, "lookback_months": 0}, CFG["weights"], float(c[-1]), asof).zones
    assert any(z.low < 85 for z in zones_all)


def _bars(closes, spread=0.004):
    c = np.asarray(closes, dtype=float)
    o = np.r_[c[0], c[:-1]]
    idx = pd.bdate_range("2026-01-01", periods=len(c))
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * (1 + spread),
                         "low": np.minimum(o, c) * (1 - spread), "close": c}, index=idx)


def test_zone_containing_price_takes_role_from_approach_side():
    z = fsr.Zone(100.0, 104.0, 3, 3.0, True, False, 0)
    up_into = _bars([90, 92, 94, 96, 98, 101, 102, 103])      # rallied up into 100-104 -> resistance
    down_into = _bars([115, 113, 110, 108, 106, 103, 102, 101])   # fell into it -> support
    assert fsr.approach_side(up_into, z) == -1
    assert fsr.approach_side(down_into, z) == 1
    # a one-bar close above the zone INSIDE the visit doesn't flip it (failed breakout)
    poke = _bars([90, 92, 94, 96, 98, 101, 104.2, 102.5])
    poke.iloc[-2, poke.columns.get_loc("low")] = 101.0             # still overlapping the zone
    assert fsr.approach_side(poke, z) == -1


def test_major_is_strongest_not_nearest():
    # weak lone swing just under price, strong multi-touch floor further down
    p = {**CFG["timeframes"]["daily"], "lookback_months": 0, "edge_fallback": False}
    piv = pd.DataFrame({"idx": [10, 30, 50, 70, 95], "price": [90.0, 90.5, 89.8, 90.2, 99.0],
                        "side": ["L"] * 5, "src": ["FW"] * 5, "weight": [1.5] * 5})
    zones = fsr.cluster_zones(piv, 100, 1.5, 0.5, p, CFG["weights"])
    for z in zones:
        z.role = "S"
    sup = sorted((z for z in zones if z.valid), key=lambda z: -z.high)
    smaj = max(sup, key=lambda z: (z.score, z.high))
    assert round(sup[0].mid) == 99 and round(smaj.mid) == 90 and smaj.touches == 4


def test_real_world_shape_360one_like():
    # range floor tested repeatedly, then a rally back into the range ceiling: ceiling must be
    # RESISTANCE (being tested from below) and the floor must be the major support
    seg = [1050, 1040, 1032, 1045, 1060, 1080, 1100, 1120, 1150, 1170, 1180, 1175, 1165, 1150, 1120,
           1090, 1060, 1040, 1030, 1042, 1065, 1090, 1110, 1140, 1165, 1178, 1172, 1160, 1130, 1100,
           1070, 1045, 1033, 1048, 1070, 1100, 1130, 1160, 1172, 1176]
    df = _bars(seg * 2 + [1080, 1055, 1036, 1050, 1080, 1115, 1150, 1168, 1174, 1177])
    r = fsr.analyze_timeframe(df, {**CFG["timeframes"]["daily"], "lookback_months": 0}, CFG["weights"],
                              float(df.close.iloc[-1]))
    assert r.res and r.res[0].low <= 1182 and r.res[0].high >= 1165          # ceiling = resistance
    assert r.sup_major is not None and 1025 <= r.sup_major.mid <= 1055         # floor = major support
