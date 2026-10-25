from __future__ import annotations

from datetime import date

from trader.core.indicators import sma
from trader.core.models import Position, Signal
from trader.strategies._swing_common import atr_buffer, chandelier_trail, ratchet
from trader.strategies.base import Strategy, SymbolContext, register

# Market-trend tag (see config/filters/regime_match.json): UPTREND only - this is a momentum/
# leadership strategy, the deliberate complement to this codebase's dip-buy (pullback_ma,
# weekly_rsi_sma20) and base-breakout (breakout_trend, inside_bar_breakout) trend strategies and
# to its mean-reversion strategies (sr_reversal, range_reversion). Every one of those waits for
# either weakness/compression (dip/base) or a statistical/structural extreme (reversion) before
# entering - none of them buys a name that is simply already strong and making fresh highs. In a
# genuinely sideways or down market this strategy is EXPECTED to sit mostly idle (new highs don't
# form, or fail fast) - that is correct, not a bug: sr_reversal/range_reversion are what should be
# doing the work in that regime instead. See this module's class docstring for the full rationale.


@register("new_high_momentum")
class NewHighMomentum(Strategy):
    """New-high leadership/momentum strategy - added to fill a regime gap in this codebase's
    strategy book (backlog item from the 2026-10-24 session: every existing strategy waits for
    weakness, compression, or a statistical extreme before entering; nothing rides a name that is
    simply already strong). The edge this targets is the well-documented "52-week-high effect"
    (George & Hwang and decades of trend-following practice before that): a stock closing at a
    fresh multi-month high, on real volume, tends to keep going more than chance alone would
    suggest - the mechanism is behavioral (anchoring, slow information diffusion), not a
    statistical-arbitrage effect, which is exactly why it tends to survive regime changes that
    break mean-reversion edges (mean-reversion needs the stock to stay range-bound; this needs the
    opposite).

    Entry requires: (a) today's close is the highest close in `new_high_lookback` bars (a genuine
    fresh high, not a one-day wick spike against a declining series of closes), (b) today's volume
    is at least `volume_mult` times its own `volume_sma_period`-day average (real participation,
    not a quiet drift to a new high on thin volume), (c) price is above a longer `trend_sma_period`
    SMA (confirms the broader structure is up, not just a short squeeze inside a longer downtrend).
    Deliberately NOT exempt from `quality`/`resistance` (unlike the two reversal strategies) -
    both filters' bullish-base/no-overhead-resistance checks are aligned with, not opposed to,
    a momentum-continuation thesis, same as breakout_trend/pullback_ma.

    Unlike every other strategy in this book, there is NO fixed profit target - a fixed target
    caps exactly the trades this edge is meant to pay off biggest on. The stop trails via a
    Chandelier Exit (highest close since entry - atr_mult*ATR, ratchet-only, same mechanism as
    breakout_trend) to let a genuine trend run. A faster secondary bail (close below
    `trend_fail_sma_period`'s own SMA by more than `trend_fail_buffer_pct`) exits sooner than the
    wider ATR trail would when momentum stalls outright, rather than waiting for the full
    volatility-adjusted trail to catch up. No pyramiding in this first version - add_signal is a
    placeholder, not a design conclusion; scaling into strength is a natural extension once this
    base version is validated.

    STATUS: newly built 2026-10-24, NOT YET BACKTESTED against this repo's actual price history -
    see config/strategies/new_high_momentum.json's _note. Disabled by default. Validate with a
    direct backtest / scripts/tune_strategy.py before enabling."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)

    def _trend_ok(self, ctx: SymbolContext) -> tuple[bool, float | None]:
        c = self.cfg["entry"]
        s = sma(ctx.daily["close"], c["trend_sma_period"]).iloc[-1]
        if s != s:
            return False, None
        return float(ctx.last_close) > float(s), float(s)

    def entry_signal(self, ctx: SymbolContext) -> Signal | None:
        c = self.cfg["entry"]
        d = ctx.daily
        if len(d) < c["min_bars"]:
            return None

        trend_ok, trend_sma = self._trend_ok(ctx)
        if not trend_ok:
            return None

        look = c["new_high_lookback"]
        if len(d) < look:
            return None
        px = ctx.last_close
        recent_close = d["close"].iloc[-look:]
        # Strict fresh high: today's close must be the highest in the window (ties count - a
        # flat double-top still clears a prior high, it just doesn't exceed it by much).
        if px < float(recent_close.max()):
            return None

        if c.get("require_volume_confirmation", True):
            vol_sma = sma(d["volume"], c["volume_sma_period"]).iloc[-1]
            if vol_sma == vol_sma and float(d["volume"].iloc[-1]) < vol_sma * c["volume_mult"]:
                return None

        buf = atr_buffer(d, c["atr_period"], c["atr_mult"])
        stop = px - buf
        if stop >= px or buf <= 0:
            return None

        return Signal(ctx.symbol, self.name, "ENTRY", ctx.asof.isoformat(),
                      f"close {px:.2f} is a {look}-bar high, above trend SMA{c['trend_sma_period']} "
                      f"{trend_sma:.2f}, on confirming volume",
                      score=0, stop=stop, ref_price=px,
                      meta={"new_high_lookback": look, "trend_sma": trend_sma})

    def stop_level(self, ctx: SymbolContext, pos: Position) -> float | None:
        x = self.cfg["exit"]
        return ratchet(pos.stop, chandelier_trail(ctx.daily, pos.opened, x["atr_period"], x["atr_mult"]))

    def exit_signal(self, ctx: SymbolContext, pos: Position) -> Signal | None:
        if pos.opened and ctx.asof < date.fromisoformat(pos.opened):
            return None
        d = ctx.daily
        x = self.cfg["exit"]
        px = float(d["close"].iloc[-1])

        trail = chandelier_trail(d, pos.opened, x["atr_period"], x["atr_mult"])
        level = max(trail, pos.stop) if (trail is not None and pos.stop) else (trail or pos.stop)
        if level is not None and px <= level * (1 - x["buffer_pct"] / 100.0):
            return Signal(ctx.symbol, self.name, "EXIT", ctx.asof.isoformat(),
                          f"close {px:.2f} <= chandelier trail {level:.2f} "
                          f"(highest close since entry - {x['atr_mult']}x ATR{x['atr_period']})",
                          ref_price=px, stop=level)

        # Faster secondary bail: don't wait for the (deliberately wide) ATR trail to catch up if
        # the broader trend has already failed outright - this is meant to fire BEFORE the
        # chandelier trail on a genuine trend failure, not duplicate it.
        fail_sma = sma(d["close"], x["trend_fail_sma_period"]).iloc[-1]
        if fail_sma == fail_sma and px < float(fail_sma) * (1 - x["trend_fail_buffer_pct"] / 100.0):
            return Signal(ctx.symbol, self.name, "EXIT", ctx.asof.isoformat(),
                          f"close {px:.2f} broke trend SMA{x['trend_fail_sma_period']} {float(fail_sma):.2f} "
                          f"(momentum failure)", ref_price=px, stop=pos.stop)
        return None

    def add_signal(self, ctx: SymbolContext, pos: Position) -> Signal | None:
        return None  # no pyramiding in this first version - see class docstring