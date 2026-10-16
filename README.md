# wrsi_trader — modular multi-strategy NSE swing system (paper first)

## Daily flow (both phases idempotent, state committed to git)
| Phase | When (IST) | What |
|---|---|---|
| `python -m trader eod`  | ~16:00 | refresh bars → manage positions (exit / pyramid) → scan universe → write `state/pending_orders.json` for next session |
| `python -m trader open` | ~09:20 | fetch today's open → **Rule 3 checks** → size (risk % of base) → execute via broker → update `state/portfolio.json` |

Weekly/monthly bars are resampled from daily bars (single source, single adjustment basis).

## Layout
- `config/` — **all tunables** (system, risk, fees, universe, holidays, `strategies/<name>.json`). No thresholds in `.py`.
- `trader/strategies/` — plug-in strategies (`@register("name")`, file stem of the JSON config must match). Strategy decides *when*; never sizes or executes.
- `trader/risk/` — `CapitalManager` (base capital, per-strategy allocation, risk %), sizing, pyramiding budget.
- `trader/brokers/` — `Broker` ABC; `PaperBroker` (slippage + Indian delivery fees); `UpstoxLiveBroker` stub.
- `trader/data/` — `DataProvider` ABC; Upstox (V3 candles + V3 quote OHLC, incremental CSV cache); historical replay.
- `trader/engine/orchestrator.py` — the only place that wires everything together.
- `trader/backtest/` — replays the *same* `run_open → run_eod` code path over stored bars.

## Rule → config map (`config/strategies/weekly_rsi_sma20.json`)
1. Entry: `entry.*` (weekly RSI cross of 60, monthly ≥ 60 − tolerance)
2. Exit: `exit.mode` (`weekly_close` | `daily_close`), `exit.buffer_pct`
3. Gap/SMA skip: `open_checks.*`
4. ATH / cup: `quality.*` (`rank` or `filter`)
5. Pyramiding trigger: `pyramiding.*`; budget/limits: `risk_config.json → pyramiding`
6. Risk: `risk_config.json → per_trade.risk_pct_of_base` (+ `capital.mode` fixed|equity)

## Run
```
pip install -r requirements.txt && pytest -q
export UPSTOX_ACCESS_TOKEN=...
python -m trader eod            # after close
python -m trader open           # right after open next day
python -m trader status
python -m trader backtest --start 2020-01-01 --end 2025-12-31   # uses data_cache/*.csv
```
Add secret `UPSTOX_ACCESS_TOKEN` in GitHub (it expires daily — same OAuth wiring problem as candle_trader).
Fill `config/holidays.json` (needed for correct week-end detection) and replace the sample universe.

## Adding strategy #2
1. `config/strategies/<name>.json` (+ allocation in `risk_config.json → capital.allocation_pct`)
2. `trader/strategies/<name>.py` subclassing `Strategy`, decorated with `@register("<name>")`
3. import it inside `build_strategy()`
