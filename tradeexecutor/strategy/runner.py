"""Define a live strategy execution model."""

import abc
import datetime
from contextlib import AbstractContextManager
import logging

from typing import List

from tradeexecutor.strategy.approval import ApprovalModel
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.state.revaluation import RevaluationMethod
from tradeexecutor.state.sync import SyncMethod
from tradeexecutor.strategy.pricing_model import PricingModelFactory
from tradeexecutor.strategy.universe_model import TradeExecutorTradingUniverse

from tradeexecutor.state.state import State, TradeExecution


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

            with self.timed_task_context_manager("decide_trades"):
                rebalance_trades = self.on_clock(clock, universe, state, debug_details)
                assert type(rebalance_trades) == list
                debug_details["rebalance_trades"] = rebalance_trades
                logger.info("We have %d trades", len(rebalance_trades))

            with self.timed_task_context_manager("confirm_trades"):
                approved_trades = self.approval_model.confirm_trades(state, rebalance_trades)
                assert type(approved_trades) == list
                logger.info("After approval we have %d trades left", len(approved_trades))
                debug_details["approved_trades"] = approved_trades

            with self.timed_task_context_manager("execute_trades"):
                self.execution_model.execute_trades(clock, state, approved_trades)

        return debug_details
