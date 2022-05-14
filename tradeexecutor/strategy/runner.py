"""Define a live strategy execution model."""

import abc
import datetime
from contextlib import AbstractContextManager
import logging
from io import StringIO

from typing import List, Optional

from tradeexecutor.strategy.approval import ApprovalModel
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.state.revaluation import RevaluationMethod
from tradeexecutor.state.sync import SyncMethod
from tradeexecutor.strategy.output import output_positions, DISCORD_BREAK_CHAR, output_trades
from tradeexecutor.strategy.pricing_model import PricingModelFactory
from tradeexecutor.strategy.routing import RoutingModel
from tradeexecutor.strategy.universe_model import TradeExecutorTradingUniverse

from tradeexecutor.state.state import State
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.reserve import ReservePosition

logger = logging.getLogger(__name__)


class PreflightCheckFailed(Exception):
    """Something was wrong with the datafeeds."""


class StrategyRunner(abc.ABC):
    """A base class for a strategy live trade executor.

    TODO: Make routing_model non-optional after eliminating legacy code.
    """

    def __init__(self,
                 timed_task_context_manager: AbstractContextManager,
                 execution_model: ExecutionModel,
                 approval_model: ApprovalModel,
                 revaluation_method: RevaluationMethod,
                 sync_method: SyncMethod,
                 pricing_model_factory: PricingModelFactory,
                 routing_model: Optional[RoutingModel]=None):
        self.timed_task_context_manager = timed_task_context_manager
        self.execution_model = execution_model
        self.approval_model = approval_model
        self.revaluation_method = revaluation_method
        self.sync_method = sync_method
        self.pricing_model_factory = pricing_model_factory
        self.routing_model = routing_model

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

    def report_after_sync_and_revaluation(self, clock: datetime.datetime, universe: TradeExecutorTradingUniverse, state: State, debug_details: dict):
        buf = StringIO()
        portfolio = state.portfolio
        tick = debug_details.get("cycle", 1)
        print(f"Portfolio status (before rebalance), tick #{tick}", file=buf)
        print("", file=buf)
        print(f"Total equity: ${portfolio.get_total_equity():,.2f}, in cash: ${portfolio.get_current_cash():,.2f}", file=buf)
        print(f"Life-time positions: {portfolio.next_position_id - 1}, trades: {portfolio.next_trade_id - 1}", file=buf)
        print(DISCORD_BREAK_CHAR, file=buf)

        print(f"Currently open positions:", file=buf)
        print("", file=buf)
        output_positions(portfolio.open_positions.values(), buf)

        print(DISCORD_BREAK_CHAR, file=buf)

        print(f"Frozen positions (${portfolio.get_frozen_position_equity():,.2f}):", file=buf)
        print("", file=buf)
        output_positions(portfolio.frozen_positions.values(), buf)

        print(DISCORD_BREAK_CHAR, file=buf)

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
        output_trades(trades, portfolio, buf)
        logger.trade(buf.getvalue())

    def report_after_execution(self, clock: datetime.datetime, universe: TradeExecutorTradingUniverse, state: State, debug_details: dict):
        buf = StringIO()
        portfolio = state.portfolio
        
        print("Portfolio status (after rebalance)", file=buf)
        print("", file=buf)
        print(f"Total equity: ${portfolio.get_total_equity():,.2f}, Cash: ${portfolio.get_current_cash():,.2f}", file=buf)

        print(DISCORD_BREAK_CHAR, file=buf)

        print(f"Opened/open positions:", file=buf)
        print("", file=buf)
        output_positions(portfolio.open_positions.values(), buf)

        print(DISCORD_BREAK_CHAR, file=buf)

        closed_positions = list(portfolio.get_positions_closed_at(clock))
        print(f"Closed positions:", file=buf)
        output_positions(closed_positions, buf)

        print(DISCORD_BREAK_CHAR, file=buf)

        print("Reserves:", file=buf)
        print("", file=buf)
        reserve: ReservePosition
        for reserve in state.portfolio.reserves.values():
            print(f"    {reserve.quantity:,.2f} {reserve.asset.token_symbol}", file=buf)
        logger.trade(buf.getvalue())

    def report_strategy_thinking(self, clock: datetime.datetime, universe: TradeExecutorTradingUniverse, state: State, trades: List[TradeExecution], debug_details: dict):
        """Strategy runner subclass can fill in.

        By default, no-op. Override in the subclass.
        """
        pass

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
