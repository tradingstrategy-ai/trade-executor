"""Define a live strategy execution model."""

import abc
import datetime
from contextlib import AbstractContextManager
import logging
from io import StringIO

from typing import List

from tradeexecutor.strategy.approval import ApprovalModel
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.state.revaluation import RevaluationMethod
from tradeexecutor.state.sync import SyncMethod
from tradeexecutor.strategy.pricing_model import PricingModelFactory
from tradeexecutor.strategy.universe_model import TradeExecutorTradingUniverse

from tradeexecutor.state.state import State, TradeExecution, ReservePosition, TradingPosition, Portfolio
from tradingstrategy.analysis.tradeanalyzer import TradePosition
from tradingstrategy.universe import Universe

logger = logging.getLogger(__name__)


class PreflightCheckFailed(Exception):
    """Something was wrong with the datafeeds."""


class StrategyRunner(abc.ABC):
    """A base class for a strategy live trade executor."""

    def __init__(self,
                 timed_task_context_manager: AbstractContextManager,
                 execution_model: ExecutionModel,
                 approval_model: ApprovalModel,
                 revaluation_method: RevaluationMethod,
                 sync_method: SyncMethod,
                 pricing_model_factory: PricingModelFactory):
        self.timed_task_context_manager = timed_task_context_manager
        self.execution_model = execution_model
        self.approval_model = approval_model
        self.revaluation_method = revaluation_method
        self.sync_method = sync_method
        self.pricing_model_factory = pricing_model_factory

    @abc.abstractmethod
    def pretick_check(self, ts: datetime.datetime, universe: TradeExecutorTradingUniverse):
        """Called when the trade executor instance is started.

        :param client: Trading Strategy client to check server versions etc.

        :param universe: THe currently constructed universe

        :param ts: Real-time clock signal or past clock timestamp in the case of unit testing

        :raise PreflightCheckFailed: In the case we cannot go live
        """
        pass

    def sync_portfolio(self, ts: datetime.datetime, universe: TradeExecutorTradingUniverse, state: State, debug_details: dict):
        """Adjust portfolio balances based on the external events.

        External events include
        - Deposits
        - Withdrawals
        - Interest accrued
        - Token rebases
        """
        assert isinstance(universe, TradeExecutorTradingUniverse), f"Universe was {universe}"
        reserve_assets = universe.reserve_assets
        assert len(reserve_assets) > 0, "No reserve assets available"
        assert len(reserve_assets) == 1, f"We only support strategies with a single reserve asset, got {self.reserve_assets}"
        reserve_update_events = self.sync_method(state.portfolio, ts, reserve_assets)
        assert type(reserve_update_events) == list
        debug_details["reserve_update_events"] = reserve_update_events
        debug_details["total_equity_at_start"] = state.portfolio.get_total_equity()
        debug_details["total_cash_at_start"] = state.portfolio.get_current_cash()

    def revalue_portfolio(self, ts: datetime.datetime, state: State):
        """Revalue portfolio based on the data."""
        state.revalue_positions(ts, self.revaluation_method)
        logger.info("After revaluation at %s our equity is %f", ts, state.portfolio.get_total_equity())

    def on_data_signal(self):
        pass

    def on_clock(self, clock: datetime.datetime, universe: TradeExecutorTradingUniverse, state: State, debug_details: dict) -> List[TradeExecution]:
        return []

    def format_position(self, position: TradingPosition, up_symbol="🌲", down_symbol="🔻") -> List[str]:
        """Write a position status line to logs.

        Position can be open/closed.
        """
        symbol = up_symbol if position.get_total_profit_percent() >= 0 else down_symbol
        if position.pair.info_url:
            link = position.pair.info_url
        else:
            link = ""

        lines =[
            f"{symbol} #{position.position_id} {position.pair.get_human_description()} size:${position.get_value():,.2f}, profit:{position.get_total_profit_usd():.2f}% ({position.get_total_profit_usd():,.4f} USD)"
        ]

        if position.has_executed_trades():
            price_diff = position.get_current_price() - position.get_opening_price()
            lines.append(f"   current price:${position.get_current_price():,.8f}, open price:${position.get_opening_price():,.8f}, diff:{price_diff:,.8f} USD")

        if position.is_frozen():
            last_trade = "buy" if position.get_last_trade().is_buy() else "sell"
            lines.append(f"   last trade: {last_trade}, freeze reason: {position.get_freeze_reason()}")

        if link:
            lines.append(f"   link: {link}")

        return lines

    def format_trade(self, portfolio: Portfolio, trade: TradeExecution) -> List[str]:
        """Write a trade status line to logs."""
        pair = trade.pair
        if pair.info_url:
            link = pair.info_url
        else:
            link = ""

        if trade.is_buy():
            trade_type = "Buy"
        else:
            trade_type = "Sell"

        existing_position = portfolio.get_existing_open_position_by_trading_pair(trade.pair)
        if existing_position:
            amount = abs(trade.planned_quantity / existing_position.get_net_quantity())
            existing_text = f", {amount*100:,.2f}% of existing position"
        else:
            existing_text = ""

        lines = [
            f"{trade_type:5} #{trade.trade_id} {pair.get_human_description()} ${trade.get_planned_value():,.2f} ({abs(trade.get_position_quantity())} {pair.base.token_symbol}){existing_text}",
        ]

        if link:
            lines.append(f"      link: {link}")

        return lines

    def report_after_sync_and_revaluation(self, clock: datetime.datetime, universe: TradeExecutorTradingUniverse, state: State, debug_details: dict):
        buf = StringIO()
        portfolio = state.portfolio
        print("Portfolio status (before rebalance)", file=buf)
        print("", file=buf)
        print(f"Total equity: ${portfolio.get_total_equity():,.2f}, in cash: ${portfolio.get_current_cash():,.2f}", file=buf)
        print(f"Life-time positions: {portfolio.next_position_id - 1}, trades: {portfolio.next_trade_id - 1}", file=buf)
        print("", file=buf)

        open_positions = list(portfolio.open_positions.values())

        if open_positions:
            print(f"Current positions:", file=buf)
            print("", file=buf)
            position: TradingPosition
            for position in open_positions:
                for line in self.format_position(position):
                    print("    " + line, file=buf)
                print("", file=buf)
        else:
            print(f"No open positions.", file=buf)

        frozen_positions = list(portfolio.frozen_positions.values())

        if frozen_positions:
            print(f"Frozen positions (${portfolio.get_frozen_position_equity():,.2f}):", file=buf)
            print("", file=buf)
            position: TradingPosition
            for position in frozen_positions:
                assert position.is_frozen()
                for line in self.format_position(position):
                    print("    " + line, file=buf)
                print("", file=buf)
        else:
            # Should not happen in the first place,
            # so let's not advertise this
            pass

        print(f"", file=buf)
        print("Reserves:", file=buf)
        print("", file=buf)
        reserve: ReservePosition
        for reserve in state.portfolio.reserves.values():
            print(f"    {reserve.quantity:,.2f} {reserve.asset.token_symbol}", file=buf)

        logger.trade(buf.getvalue())

    def report_before_execution(self, clock: datetime.datetime, universe: TradeExecutorTradingUniverse, state: State, trades: List[TradeExecution], debug_details: dict):
        buf = StringIO()
        print("New trades to be executed", file=buf)
        print("", file=buf)
        position: TradingPosition
        portfolio = state.portfolio
        for t in trades:
            for line in self.format_trade(portfolio, t):
                print("    " + line, file=buf)
            print("", file=buf)
        logger.trade(buf.getvalue())

    def report_after_execution(self, clock: datetime.datetime, universe: TradeExecutorTradingUniverse, state: State, debug_details: dict):
        buf = StringIO()
        portfolio = state.portfolio
        print("Portfolio status (after rebalance)", file=buf)
        print("", file=buf)
        print(f"Total equity: ${portfolio.get_total_equity():,.2f}, Cash: ${portfolio.get_current_cash():,.2f}", file=buf)
        print("", file=buf)
        print(f"Opened/open positions:", file=buf)
        print("", file=buf)
        position: TradingPosition
        for position in portfolio.open_positions.values():
            for line in self.format_position(position):
                print("    " + line, file=buf)
            print("", file=buf)

        print("", file=buf)
        closed_positions = list(portfolio.get_positions_closed_at(clock))
        if len(closed_positions) > 0:
            print(f"Closed positions:", file=buf)
            print("", file=buf)
            position: TradingPosition
            for position in portfolio.get_positions_closed_at(clock):
                for line in self.format_position(position):
                    print("    " + line, file=buf)
                print("", file=buf)
        else:
            print("No positions were closed on this tick.", file=buf)
        print(f"", file=buf)
        print("Reserves:", file=buf)
        print("", file=buf)
        reserve: ReservePosition
        for reserve in state.portfolio.reserves.values():
            print(f"    {reserve.quantity:,.2f} {reserve.asset.token_symbol}", file=buf)

        logger.trade(buf.getvalue())

    def report_strategy_thinking(self, clock: datetime.datetime, universe: TradeExecutorTradingUniverse, state: State, trades: List[TradeExecution], debug_details: dict):
        """Strategy runner subclass can fill in.

        By default, no-op.
        """

    def tick(self, clock: datetime.datetime, universe: TradeExecutorTradingUniverse, state: State, debug_details: dict) -> dict:
        """Perform the strategy main tick.

        :return: Debug details dictionary where different subsystems can write their diagnostics information what is happening during the dict.
            Mostly useful for integration testing.
        """

        with self.timed_task_context_manager("strategy_tick", clock=clock):

            with self.timed_task_context_manager("sync_portfolio"):
                self.sync_portfolio(clock, universe, state, debug_details)

            with self.timed_task_context_manager("revalue_portfolio"):
                self.revalue_portfolio(clock, state)

            self.report_after_sync_and_revaluation(clock, universe, state, debug_details)

            with self.timed_task_context_manager("decide_trades"):
                rebalance_trades = self.on_clock(clock, universe, state, debug_details)
                assert type(rebalance_trades) == list
                debug_details["rebalance_trades"] = rebalance_trades
                logger.info("We have %d trades", len(rebalance_trades))

            self.report_strategy_thinking(clock, universe, state, rebalance_trades, debug_details)

            with self.timed_task_context_manager("confirm_trades"):
                approved_trades = self.approval_model.confirm_trades(state, rebalance_trades)
                assert type(approved_trades) == list
                logger.info("After approval we have %d trades left", len(approved_trades))
                debug_details["approved_trades"] = approved_trades

            self.report_before_execution(clock, universe, state, approved_trades, debug_details)

            with self.timed_task_context_manager("execute_trades"):
                succeeded_trades, failed_trades = self.execution_model.execute_trades(clock, state, approved_trades)
                debug_details["succeeded_trades"] = succeeded_trades
                debug_details["failed_trades"] = failed_trades

            self.report_after_execution(clock, universe, state, debug_details)

        return debug_details
