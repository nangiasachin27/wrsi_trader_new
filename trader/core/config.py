"""Config loading. Python files hold logic only; every tunable lives in config/*.json."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(os.environ.get("TRADER_ROOT", Path(__file__).resolve().parents[2]))


def _read(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@dataclass
class Settings:
    system: dict
    risk: dict
    fees: dict
    universe: dict
    holidays: list
    strategies: dict  # name (file stem) -> strategy config
    filters: dict      # name (file stem) -> filter config, applied to every strategy's candidates


def load_settings(config_dir: Path | str | None = None) -> Settings:
    d = Path(config_dir) if config_dir else ROOT / "config"
    strategies = {p.stem: _read(p) for p in sorted((d / "strategies").glob("*.json"))}
    filters = {p.stem: _read(p) for p in sorted((d / "filters").glob("*.json"))}
    return Settings(
        system=_read(d / "system_config.json"),
        risk=_read(d / "risk_config.json"),
        fees=_read(d / "fee_config.json"),
        universe=_read(d / "universe.json"),
        holidays=_read(d / "holidays.json")["holidays"],
        strategies=strategies,
        filters=filters,
    )
