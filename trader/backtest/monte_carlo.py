from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def open_positions_summary(portfolio_path: Path | str) -> dict:
    """Unrealized P&L sitting in positions still open at the end of a backtest -
    trade_log.csv NEVER contains these (it's built only from closed SELL trades), so Monte
    Carlo run on trade_log.csv alone is blind to them. This reads the backtest's own
    portfolio.json (written by the same run) to surface what's being left out, mark-to-market
    at each position's last recorded price - the actual eventual P&L is still unknown and
    could differ once those positions are actually closed."""
    p = Path(portfolio_path)
    if not p.exists():
        return {"n_open": 0, "unrealized_pnl": 0.0, "positions": []}
    pf = json.loads(p.read_text(encoding="utf-8"))
    rows = []
    for sym, pos in pf.get("positions", {}).items():
        cost = sum(l["qty"] * l["price"] for l in pos["lots"])
        qty = sum(l["qty"] for l in pos["lots"])
        mtm_pnl = pos["last_price"] * qty - cost
        rows.append({"symbol": sym, "qty": qty, "avg_cost": cost / qty if qty else 0.0,
                     "last_price": pos["last_price"], "unrealized_pnl": mtm_pnl})
    return {"n_open": len(rows), "unrealized_pnl": sum(r["unrealized_pnl"] for r in rows), "positions": rows}


def run_monte_carlo(pnl: np.ndarray, starting_capital: float, n_sims: int = 2000,
                    horizon: int | None = None, method: str = "bootstrap",
                    ruin_pct: float = 50.0, seed: int | None = None) -> dict:
    """Resample a backtest's REAL per-trade dollar P&L (fees already included) to show the
    range of equity outcomes the same edge could plausibly have produced under a different
    trade sequence or sample - not just the one path the backtest happened to generate.

    Equity paths are built ADDITIVELY (starting_capital + cumulative resampled P&L), which
    matches config/risk_config.json's default capital.mode "fixed" exactly: risk per trade is
    a constant % of a fixed base capital, not of current equity, so position sizing does not
    compound and neither should this simulation. If capital.mode is "equity" (compounding),
    treat these results as approximate - true path-dependent position sizing isn't modeled
    here and would require re-running the sizing engine inside each simulated path.

    method "bootstrap": draws `horizon` trades WITH replacement each simulation - can extrapolate
        beyond the historical trade count (horizon > len(pnl)) to ask "what does the next N
        trades look like", at the cost of assuming future trades resemble the historical sample.
    method "reshuffle": permutes the ACTUAL trades WITHOUT replacement - isolates sequence risk
        only (does the order losses/wins arrived in matter), never extrapolates; horizon is
        forced to len(pnl) since a permutation can't be longer or shorter than the original.
    """
    pnl = np.asarray(pnl, dtype=float)
    n = len(pnl)
    if n == 0:
        raise ValueError("No trades to simulate from - need at least one closed trade.")

    rng = np.random.default_rng(seed)
    if method == "reshuffle":
        horizon = n
        paths = np.array([rng.permutation(pnl) for _ in range(n_sims)])
    elif method == "bootstrap":
        horizon = horizon or n
        paths = rng.choice(pnl, size=(n_sims, horizon), replace=True)
    else:
        raise ValueError(f"unknown method '{method}' (use 'bootstrap' or 'reshuffle')")

    equity = starting_capital + np.cumsum(paths, axis=1)
    equity = np.concatenate([np.full((n_sims, 1), starting_capital), equity], axis=1)  # include t=0

    final_equity = equity[:, -1]
    running_max = np.maximum.accumulate(equity, axis=1)
    max_dd_pct = ((equity - running_max) / running_max * 100.0).min(axis=1)
    ruin_level = starting_capital * (1 - ruin_pct / 100.0)
    ruin_hit = equity.min(axis=1) <= ruin_level

    pct = lambda a, q: float(np.percentile(a, q))
    return {
        "n_sims": n_sims, "horizon": horizon, "method": method, "starting_capital": starting_capital,
        "n_source_trades": n,
        "final_equity": {"p5": pct(final_equity, 5), "p25": pct(final_equity, 25), "p50": pct(final_equity, 50),
                         "p75": pct(final_equity, 75), "p95": pct(final_equity, 95),
                         "mean": float(final_equity.mean())},
        "max_drawdown_pct": {"p5": pct(max_dd_pct, 5), "p25": pct(max_dd_pct, 25), "p50": pct(max_dd_pct, 50),
                             "p75": pct(max_dd_pct, 75), "p95": pct(max_dd_pct, 95)},
        "prob_loss_pct": float((final_equity < starting_capital).mean() * 100.0),
        "prob_ruin_pct": float(ruin_hit.mean() * 100.0),
        "ruin_threshold_pct": ruin_pct,
        "equity_paths": equity,  # kept for equity_fan_chart(); not printed by the CLI summary
    }


def equity_fan_chart(result: dict) -> pd.DataFrame:
    """Percentile equity curve (p5/p25/p50/p75/p95) at every simulated trade-step - the data
    behind a Monte Carlo 'fan chart', for exporting and plotting outside this tool."""
    paths = result["equity_paths"]
    qs = [5, 25, 50, 75, 95]
    return pd.DataFrame({f"p{q}": np.percentile(paths, q, axis=0) for q in qs},
                        index=pd.RangeIndex(paths.shape[1], name="trade_step"))


def percentile_rank(result: dict, actual_final_equity: float) -> float:
    """Where the backtest's OWN actual result sits within the simulated distribution (0-100).
    A low number means the real backtest did worse than most resampled alternatives would have
    - i.e. the real run may have gotten a favorable trade sequence rather than a robust edge."""
    paths = result["equity_paths"][:, -1]
    return float((paths <= actual_final_equity).mean() * 100.0)