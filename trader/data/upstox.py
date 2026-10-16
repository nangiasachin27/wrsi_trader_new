from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from trader.data.base import DataProvider

log = logging.getLogger(__name__)


class UpstoxProvider(DataProvider):
    """Daily bars via V3 historical-candle API (cached to CSV, incremental); today's open via
    V3 market-quote OHLC (live_ohlc.open). Weekly/monthly bars are built from daily bars so all
    timeframes share one source and one adjustment basis."""

    def __init__(self, data_cfg: dict, tz: str, instruments: dict[str, str], cache_dir: Path | str):
        self.c, self.tz = data_cfg, ZoneInfo(tz)
        self.instruments = instruments
        self.by_key = {v: k for k, v in instruments.items()}
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json",
                                     "Authorization": f"Bearer {os.environ['UPSTOX_ACCESS_TOKEN']}"})
        self._fresh: set[tuple[str, date]] = set()

    # ---- HTTP ---------------------------------------------------------------------------
    def _get(self, url: str, params: dict | None = None) -> dict:
        for attempt in range(self.c["retries"]):
            try:
                r = self.session.get(url, params=params, timeout=self.c["http_timeout_s"])
                if r.status_code == 429 or r.status_code >= 500:
                    raise requests.HTTPError(f"{r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                return r.json()
            except requests.RequestException as e:
                if attempt == self.c["retries"] - 1:
                    raise
                time.sleep(self.c["retry_backoff_s"] * (attempt + 1))
                log.warning("retry %s after %s", url, e)
        raise RuntimeError("unreachable")

    def _fetch(self, key: str, start: date, end: date) -> pd.DataFrame:
        rows, cur = [], start
        step = timedelta(days=self.c["chunk_days"])
        while cur <= end:
            to = min(cur + step, end)
            url = f"{self.c['base_url']}/v3/historical-candle/{quote(key, safe='')}/days/1/{to:%Y-%m-%d}/{cur:%Y-%m-%d}"
            rows += self._get(url)["data"]["candles"]
            cur = to + timedelta(days=1)
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows).iloc[:, :6]
        df.columns = ["date", "open", "high", "low", "close", "volume"]
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(self.tz).dt.tz_localize(None).dt.normalize()
        return df.drop_duplicates("date").set_index("date").sort_index()

    # ---- daily cache ----------------------------------------------------------------------
    def get_daily(self, symbol: str, end: date) -> pd.DataFrame:
        f, meta = self.cache / f"{symbol}.csv", self.cache / f"{symbol}.meta.json"
        today = datetime.now(self.tz).date()
        cached = pd.read_csv(f, parse_dates=["date"], index_col="date") if f.exists() else None
        last_full = date.fromisoformat(json.loads(meta.read_text())["full"]) if meta.exists() else None
        need_full = cached is None or last_full is None or (today - last_full).days >= self.c["full_refresh_days"]

        if (symbol, today) not in self._fresh:
            key = self.instruments[symbol]
            if need_full:  # periodic full re-pull picks up corporate-action adjustments
                start = today - timedelta(days=365 * self.c["history_years"])
                cached = self._fetch(key, start, today)
                meta.write_text(json.dumps({"full": today.isoformat()}))
            else:
                start = cached.index[-1].date() - timedelta(days=self.c["overlap_days"])
                new = self._fetch(key, start, today)
                cached = pd.concat([cached[cached.index < new.index[0]], new]) if len(new) else cached
            cached.to_csv(f, index_label="date")
            self._fresh.add((symbol, today))
        return cached.loc[: pd.Timestamp(end)]

    # ---- open quotes ----------------------------------------------------------------------
    def get_open_quotes(self, symbols: list[str], on: date) -> dict[str, float]:
        out, keys = {}, [self.instruments[s] for s in symbols]
        n = self.c["quote_batch_size"]
        for i in range(0, len(keys), n):
            batch = ",".join(keys[i:i + n])
            data = self._get(f"{self.c['base_url']}/v3/market-quote/ohlc",
                             {"instrument_key": batch, "interval": "1d"})["data"]
            for v in data.values():  # response keys are 'NSE_EQ:SYMBOL' - map via instrument_token
                sym = self.by_key.get(v.get("instrument_token"))
                live = v.get("live_ohlc")
                if not sym or not live:
                    continue
                if datetime.fromtimestamp(live["ts"] / 1000, self.tz).date() != on:
                    continue  # stale candle: market not open yet / holiday
                out[sym] = float(live["open"])
        return out
