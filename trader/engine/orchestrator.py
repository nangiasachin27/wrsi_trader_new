from __future__ import annotations

import logging
from datetime import date, timedelta

from trader.brokers.base import Broker
from trader.core.calendar import TradingCalendar
from trader.core.config import Settings
from trader.core.models import Lot, PendingOrder, Position, Signal
from trader.data.base import DataProvider
from trader.filters.base import Filter, build_filter
from trader.portfolio.state import Portfolio, StateStore
from trader.risk.capital import CapitalManager
from trader.risk.sizing import size_add, size_entry, total_open_risk
from trader.strategies.base import SymbolContext, build_strategy

log = logging.getLogger(__name__)


class Engine:
    """Two phases per trading day, both idempotent:
       run_eod(d)  : after close - manage positions, scan universe, write pending orders for d+1
       run_open(d) : right after open - validate (Rule 3), size, execute pending orders for d"""

    def __init__(self, settings: Settings, provider: DataProvider, broker: Broker,
                 store: StateStore, calendar: TradingCalendar):
        self.s, self.provider, self.broker, self.store, self.cal = settings, provider, broker, store, calendar
        self.capital = CapitalManager(settings.risk)
        self.strategies = {n: build_strategy(n, c) for n, c in settings.strategies.items() if c["enabled"]}
        self.filters: dict[str, Filter] = {n: build_filter(n, c) for n, c in settings.filters.items()}
        # Backlog item from the 2026-10-25 session: a filter that needs a per-strategy
        # PARAMETER (regime_match's tag, market_efficiency's threshold override) reads it from
        # the originating strategy's own config, bound here once - see Filter.bind_strategy_cfgs
        # (trader/filters/base.py) for the full rationale. Only enabled strategies are included
        # (a disabled one never produces a candidate signal, so it's never looked up).
        strategy_cfgs = {n: s.cfg for n, s in self.strategies.items()}
        for filt in self.filters.values():
            filt.bind_strategy_cfgs(strategy_cfgs)
        self._status_logged = False

    # ---- helpers ---------------------------------------------------------------------------
    def _ctx(self, sym: str, asof: date) -> SymbolContext | None:
        df = self.provider.get_daily(sym, asof)
        if df.empty or df.index[-1].date() != asof:
            log.warning("%s: no bar for %s (last=%s) - skipped", sym, asof, None if df.empty else df.index[-1].date())
            return None
        return SymbolContext(sym, df, asof, self.cal)

    def _order(self, sig: Signal, exec_date: date) -> PendingOrder:
        log.info("SIGNAL %-5s %-12s %s -> exec %s | %s",
                 sig.action, sig.symbol, sig.asof, exec_date, sig.reason)
        return PendingOrder(sig.symbol, sig.strategy, sig.action, sig.asof, exec_date.isoformat(),
                            sig.reason, sig.score, sig.stop, sig.ref_price, sig.meta)

    def _universe(self, name: str) -> list[str]:
        return self.s.universe["lists"][self.s.strategies[name]["universe"]]

    # ---- EOD -------------------------------------------------------------------------------
    def _log_filter_status(self) -> None:
        if self._status_logged:  # backtests call run_eod once per simulated day - print the header once
            return
        self._status_logged = True
        log.info("STRATEGIES: %s", ", ".join(self.strategies) or "none")
        on = [(n, f) for n, f in self.filters.items() if f.enabled]
        off = [n for n, f in self.filters.items() if not f.enabled]
        log.info("FILTERS ON (%d):", len(on))
        for n, f in on:
            log.info("  + %-15s %s", n, f.describe())
        log.info("FILTERS OFF (%d): %s", len(off), ", ".join(off) or "none")

    def run_eod(self, asof: date) -> list[PendingOrder]:
        self._log_filter_status()
        pf = self.store.load_portfolio(self.capital.initial_cash())
        nxt, orders, exiting = self.cal.next_trading_day(asof), [], set()
        max_adds = self.s.risk["pyramiding"]["max_adds"]

        for sym, pos in list(pf.positions.items()):
            ctx = self._ctx(sym, asof)
            if ctx is None:
                continue
            strat = self.strategies[pos.strategy]
            pos.last_price = ctx.last_close
            pos.stop = strat.stop_level(ctx, pos) or pos.stop
            sig = strat.exit_signal(ctx, pos)
            if sig:
                orders.append(self._order(sig, nxt))
                exiting.add(sym)
            elif pos.adds < max_adds and (sig := strat.add_signal(ctx, pos)):
                orders.append(self._order(sig, nxt))
            else:
                log.debug("HOLD  %-12s qty %d avg %.2f stop %.2f last %.2f",
                         sym, pos.qty, pos.avg_cost, pos.stop, pos.last_price)

        slots = self.s.risk["portfolio"]["max_positions"] - (len(pf.positions) - len(exiting))
        for filt in self.filters.values():
            filt.prepare(asof, self.provider)

        cands: list[Signal] = []
        for name, strat in self.strategies.items():
            for sym in self._universe(name):
                if sym in pf.positions or any(c.symbol == sym for c in cands):
                    continue
                ctx = self._ctx(sym, asof)
                if not ctx:
                    continue
                sig = strat.entry_signal(ctx)
                if not sig:
                    log.debug("SCAN  %-12s no entry signal (%s)", sym, name)
                    continue
                # Overlay filters, applied identically regardless of which strategy produced
                # this candidate: every enabled, non-excluded filter must pass. `exclude_filters`
                # is the strategy's OWN config (backlog item from the 2026-10-25 session,
                # replacing the old per-filter exempt_strategies/applicable_strategies/
                # applicable_regimes maps) - a filter named there is skipped entirely for this
                # strategy's candidates: it never runs, never scores, never gates. See
                # trader/filters/base.py's class docstring.
                #
                # Ranking score (backlog item from the 2026-10-26 session - see
                # Filter.max_score()'s docstring for the full rationale): signal.score is the
                # AVERAGE of score/max_score() across every non-excluded filter that actually
                # scores (max_score() > 0) - NOT a raw sum. Summing used to systematically favor
                # whichever strategy's winning combo happened to include more scoring filters or
                # filters with a wider native scale (e.g. trend_strength's 0-2 vs sector_regime's
                # 0-1), independent of actual signal quality - confirmed as the dominant driver
                # of pullback_ma's near-total starvation (max possible raw score of 1, vs
                # sr_reversal's 4) when all 5 strategies ran together sharing max_positions. A
                # pure pass/fail gate (resistance/market_regime/regime_match, max_score()==0) is
                # excluded from the average entirely, not counted as a zero - it was never meant
                # to carry ranking weight, only gate.
                excluded = strat.cfg.get("exclude_filters", [])
                blocked_by, results, score_fracs = None, {}, []
                for fname, filt in self.filters.items():
                    if fname in excluded:
                        continue
                    res = filt.evaluate(sig, ctx)
                    results[fname] = res
                    fmax = filt.max_score()
                    # contributes_to_score (2026-10-29): a per-CANDIDATE opt-out, distinct from
                    # fmax>0 (a per-FILTER one). A filter can have a genuine "no information"
                    # case for this specific candidate (e.g. candlestick.py when no pattern
                    # matched at all) - see FilterResult's own docstring for why averaging that
                    # in as a flat 0 is the same bias already fixed twice elsewhere, confirmed by
                    # a controlled A/B. Default True, so every other filter's behavior is
                    # unchanged.
                    if fmax > 0 and res.contributes_to_score:
                        score_fracs.append(res.score / fmax)
                    if not res.passed and blocked_by is None:
                        blocked_by = fname
                sig.score = sum(score_fracs) / len(score_fracs) if score_fracs else 0.0
                sig.meta["filters"] = {k: {"passed": r.passed, "reason": r.reason, "score": r.score, **r.meta}
                                       for k, r in results.items()}
                if blocked_by:
                    log.debug("SCAN  %-12s blocked by filter '%s': %s", sym, blocked_by, results[blocked_by].reason)
                    self.store.journal(date=asof, symbol=sym, event="ENTRY_FILTERED", strategy=name,
                                       filter=blocked_by, reason=results[blocked_by].reason)
                    continue
                cands.append(sig)

        cands.sort(key=lambda s: (-s.score, -s.meta.get("rsi_w", 0.0)))
        # Full post-filter, pre-slot shortlist for this eod, one record per candidate, each
        # tagged with its own strategy and rank - added 2026-10-27 specifically so the day's
        # ~N qualifying candidates can be audited by hand (symbol + strategy + score) without
        # reconstructing them from the ENTRY_SKIPPED winners text, which only ever carried a
        # symbol and was never tagged with which strategy produced it. This event fires even on
        # a day with zero competition (candidates <= free slots), unlike ENTRY_SKIPPED.
        if cands:
            self.store.journal(date=asof, event="SHORTLIST", n_candidates=len(cands), n_slots=max(slots, 0),
                               candidates=[{"rank": i, "symbol": s.symbol, "strategy": s.strategy,
                                            "score": round(s.score, 4), "taken": i <= max(slots, 0)}
                                           for i, s in enumerate(cands, 1)])
        if cands:
            log.info("ENTRY RANKING %s: %d candidate(s), %d slot(s) available",
                     asof, len(cands), max(slots, 0))
            log.info("  %-3s %-12s %6s %6s %6s %10s %5s  %s",
                     "#", "SYMBOL", "SCORE", "wRSI", "mRSI", "ATH_DIST%", "CUP", "RESULT")
            for i, sig in enumerate(cands, 1):
                q = sig.meta.get("filters", {}).get("quality", {})
                ath = f"{q['ath_dist_pct']:.1f}" if "ath_dist_pct" in q else "n/a"
                cup = "Y" if q.get("cup_ok") else ("N" if "cup_ok" in q else "n/a")
                taken = "TAKEN" if i <= max(slots, 0) else "no free slot"
                log.info("  %-3d %-12s %6.2f %6.1f %6.1f %10s %5s  %s",
                         i, sig.symbol, sig.score, sig.meta.get("rsi_w", 0), sig.meta.get("rsi_m", 0),
                         ath, cup, taken)
        for sig in cands[:max(slots, 0)]:
            orders.append(self._order(sig, nxt))
        n_slots = max(slots, 0)
        winners = ", ".join(f"{c.symbol}[{c.strategy}](score {c.score:g}, wRSI {c.meta.get('rsi_w', 0):.1f})"
                            for c in cands[:n_slots]) or f"none - {len(pf.positions)} positions already held"
        for rank, sig in enumerate(cands[n_slots:], start=n_slots + 1):
            why = (f"no free slot: ranked #{rank} of {len(cands)} (score {sig.score:g}, "
                   f"wRSI {sig.meta.get('rsi_w', 0):.1f}); {n_slots} slot(s) went to: {winners}")
            log.info("SIGNAL ENTRY %-12s skipped: %s", sig.symbol, why)
            self.store.journal(date=asof, symbol=sig.symbol, event="ENTRY_SKIPPED", strategy=sig.strategy, why=why)

        self.store.save_pending(orders)
        self.store.save_portfolio(pf)
        self.store.append_equity(asof, pf)
        log.info("EOD %s: %d pending orders, equity %.0f", asof, len(orders), pf.equity())
        return orders

    # ---- OPEN ------------------------------------------------------------------------------
    def run_open(self, asof: date) -> None:
        pf = self.store.load_portfolio(self.capital.initial_cash())
        pending = self.store.load_pending()
        iso = asof.isoformat()
        due = [o for o in pending if o.exec_date == iso]
        keep = [o for o in pending if o.exec_date > iso]
        for o in (o for o in pending if o.exec_date < iso):
            self.store.journal(date=asof, symbol=o.symbol, event="STALE_DROPPED", action=o.action)
        if not due:
            log.info("OPEN %s: no pending orders due", asof)
            self.store.save_pending(keep)
            return

        opens = self.provider.get_open_quotes(sorted({o.symbol for o in due}), asof)
        prev_day = asof - timedelta(days=1)

        for o in (o for o in due if o.action == "EXIT"):
            px = opens.get(o.symbol)
            if px is None or o.symbol not in pf.positions:
                if o.symbol in pf.positions:  # never lose an exit: retry next session
                    o.exec_date = self.cal.next_trading_day(asof).isoformat()
                    keep.append(o)
                continue
            pos = pf.positions.pop(o.symbol)
            fill = self.broker.execute("SELL", o.symbol, pos.qty, px, asof)
            proceeds = fill.qty * fill.price - fill.fees
            pnl = proceeds - pos.cost - pos.fees_paid
            hold_days = (asof - date.fromisoformat(pos.opened)).days if pos.opened else None
            log.info("FILL  SELL %-12s qty %-6d @ %8.2f  fees %6.2f  pnl %+9.2f  held %sd  (%s)",
                     o.symbol, fill.qty, fill.price, fill.fees, pnl, hold_days, o.reason)
            pf.cash += proceeds
            pf.trades.append({"date": iso, "symbol": o.symbol, "strategy": o.strategy, "side": "SELL",
                              "qty": fill.qty, "price": fill.price, "fees": fill.fees,
                              "avg_cost": round(pos.avg_cost, 2), "pnl": round(pnl, 2),
                              "adds": pos.adds, "reason": o.reason,
                              "entry_date": pos.opened, "holding_days": hold_days})

        for o in sorted((o for o in due if o.action in ("ENTRY", "ADD")), key=lambda x: -x.score):
            self._buy(pf, o, opens.get(o.symbol), prev_day, asof)

        self.store.save_portfolio(pf)
        self.store.save_pending(keep)
        log.info("OPEN %s: processed %d due orders, equity %.0f", asof, len(due), pf.equity())

    def _buy(self, pf: Portfolio, o: PendingOrder, px: float | None, prev_day: date, asof: date) -> None:
        def skip(why: str) -> None:
            log.info("skip %s %s: %s", o.action, o.symbol, why)
            self.store.journal(date=asof, symbol=o.symbol, event=f"{o.action}_SKIPPED", why=why)

        if px is None:
            return skip("no open quote")
        prev = self.provider.get_daily(o.symbol, prev_day)
        if prev.empty:
            return skip("no previous close")
        ok, why = self.strategies[o.strategy].validate_open(o, px, float(prev["close"].iloc[-1]))
        if not ok:
            return skip(why)

        risk = self.s.risk
        est = px * (1 + self.s.system["execution"]["slippage_bps"] / 1e4)  # size on the price we will actually pay
        eq = pf.equity()
        base, amt = self.capital.strategy_base(o.strategy, eq), self.capital.risk_amount(o.strategy, eq)
        if o.action == "ENTRY":
            if o.symbol in pf.positions or len(pf.positions) >= risk["portfolio"]["max_positions"]:
                return skip("held or max positions")
            qty, why = size_entry(amt, est, o.stop, base, pf.cash, risk)
            new_risk = qty * max(est - o.stop, 0.0)
            cap = base * risk["portfolio"]["max_total_open_risk_pct_of_base"] / 100.0
            if qty and total_open_risk(pf.positions) + new_risk > cap:
                return skip("portfolio open-risk cap")
        else:
            pos = pf.positions.get(o.symbol)
            if pos is None:
                return skip("no position to add to")
            qty, why = size_add(pos, est, o.stop, base, pf.cash, risk)
        if qty <= 0:
            return skip(why)

        fill = self.broker.execute("BUY", o.symbol, qty, px, asof)
        log.info("FILL  BUY  %-12s qty %-6d @ %8.2f  fees %6.2f  (%s, stop %.2f)",
                 o.symbol, fill.qty, fill.price, fill.fees, o.action, o.stop)
        pf.cash -= fill.qty * fill.price + fill.fees
        lot = Lot(fill.qty, fill.price, fill.date)
        if o.action == "ENTRY":
            pf.positions[o.symbol] = Position(o.symbol, o.strategy, [lot], o.stop, fill.price,
                                              fees_paid=fill.fees, opened=fill.date)
        else:
            p = pf.positions[o.symbol]
            p.lots.append(lot)
            p.adds += 1
            p.fees_paid += fill.fees
            p.stop, p.last_price = o.stop, fill.price
        pf.trades.append({"date": fill.date, "symbol": o.symbol, "strategy": o.strategy,
                          "side": "BUY", "action": o.action, "qty": fill.qty, "price": fill.price,
                          "fees": fill.fees, "stop": round(o.stop, 2), "reason": o.reason})