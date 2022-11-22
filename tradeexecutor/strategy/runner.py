"""Strategy execution core.

Define the runner model for different strategy types.
"""

import abc
import datetime
from contextlib import AbstractContextManager
import logging
from io import StringIO

from typing import List, Optional, Tuple

from tradeexecutor.strategy.approval import ApprovalModel
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.state.sync import SyncMethod
from tradeexecutor.strategy.output import output_positions, DISCORD_BREAK_CHAR, output_trades
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.pricing_model import PricingModelFactory, PricingModel
from tradeexecutor.strategy.routing import RoutingModel, RoutingState
from tradeexecutor.strategy.stoploss import check_position_triggers
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.strategy.universe_model import StrategyExecutionUniverse

from tradeexecutor.state.state import State
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.strategy.valuation import ValuationModelFactory, ValuationModel


logger = logging.getLogger(__name__)


class PreflightCheckFailed(Exception):
    """Something was wrong with the datafeeds."""


class StrategyRunner(abc.ABC):
    """A base class for a strategy executor.

    Each different strategy type needs its own runner.
    Currently we have

    - :py:class:`tradeexecutor.strategy.pandas_trader.runner.PandasTraderRunner`

    - :py:class:`tradeexecutor.strategy.qstrader.runner.QSTraderRunner`

    TODO: Make user_supplied_routing_model non-optional after eliminating legacy code.
    """

    def __init__(self,
                 timed_task_context_manager: AbstractContextManager,
                 execution_model: ExecutionModel,
                 approval_model: ApprovalModel,
                 valuation_model_factory: ValuationModelFactory,
                 sync_method: SyncMethod,
                 pricing_model_factory: PricingModelFactory,
                 routing_model: Optional[RoutingModel]=None):
        self.timed_task_context_manager = timed_task_context_manager
        self.execution_model = execution_model
        self.approval_model = approval_model
        self.valuation_model_factory = valuation_model_factory
        self.sync_method = sync_method
        self.pricing_model_factory = pricing_model_factory
        self.routing_model = routing_model

    @abc.abstractmethod
    def pretick_check(self, ts: datetime.datetime, universe: StrategyExecutionUniverse):
        """Called when the trade executor instance is started.

        :param client: Trading Strategy client to check server versions etc.

        :param universe: THe currently constructed universe

        :param ts: Real-time clock signal or past clock timestamp in the case of unit testing

        :raise PreflightCheckFailed: In the case we cannot go live
        """
        pass

    def sync_portfolio(self, ts: datetime.datetime, universe: StrategyExecutionUniverse, state: State, debug_details: dict):
        """Adjust portfolio balances based on the external events.

        External events include
        - Deposits
        - Withdrawals
        - Interest accrued
        - Token rebases
        """
        assert isinstance(universe, StrategyExecutionUniverse), f"Universe was {universe}"
        reserve_assets = universe.reserve_assets
        assert len(reserve_assets) > 0, "No reserve assets available"
        assert len(reserve_assets) == 1, f"We only support strategies with a single reserve asset, got {self.reserve_assets}"
        token = reserve_assets[0]
        assert token.decimals and token.decimals > 0, f"Reserve asset lacked decimals"
        reserve_update_events = self.sync_method(state.portfolio, ts, reserve_assets)
        assert type(reserve_update_events) == list
        debug_details["reserve_update_events"] = reserve_update_events
        debug_details["total_equity_at_start"] = state.portfolio.get_total_equity()
        debug_details["total_cash_at_start"] = state.portfolio.get_current_cash()

    def revalue_portfolio(self, ts: datetime.datetime, state: State, valuation_method: ValuationModel):
        """Revalue portfolio based on the data."""
        state.revalue_positions(ts, valuation_method)
        logger.info("After revaluation at %s our equity is %f", ts, state.portfolio.get_total_equity())

    def on_clock(self,
                 clock: datetime.datetime,
                 universe: StrategyExecutionUniverse,
                 pricing_model: PricingModel,
                 state: State,
                 debug_details: dict) -> List[TradeExecution]:
        """Perform the core strategy decision cycle.

        :param clock:
            The current cycle timestamp

        :param universe:
            Our trading pairs and such. Refreshed before the cycle.

        :param pricing_model:
            When constructing trades, uses pricing model to estimate the cost of a trade.

        :param state:
            The current trade execution and portfolio status

        :return:
            List of new trades to execute
        """
        return []

    def report_after_sync_and_revaluation(self, clock: datetime.datetime, universe: StrategyExecutionUniverse, state: State, debug_details: dict):
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

    def report_before_execution(self, clock: datetime.datetime, universe: StrategyExecutionUniverse, state: State, trades: List[TradeExecution], debug_details: dict):
        buf = StringIO()
        print("New trades to be executed", file=buf)
        print("", file=buf)
        position: TradingPosition
        portfolio = state.portfolio
        output_trades(trades, portfolio, buf)
        logger.trade(buf.getvalue())

    def report_after_execution(self, clock: datetime.datetime, universe: StrategyExecutionUniverse, state: State, debug_details: dict):
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

    def report_strategy_thinking(self, clock: datetime.datetime, universe: StrategyExecutionUniverse, state: State, trades: List[TradeExecution], debug_details: dict):
        """Strategy runner subclass can fill in.

        By default, no-op. Override in the subclass.
        """
        pass

    def setup_routing(self, universe: StrategyExecutionUniverse) -> Tuple[RoutingState, PricingModel, ValuationModel]:
        """Setups routing state for this cycle.

        :param universe:
            The currently tradeable universe

        :return:
            Tuple(routing state, pricing model, valuation model)
        """

        assert self.routing_model, "Routing model not set"

        # Get web3 connection, hot wallet
        routing_state_details = self.execution_model.get_routing_state_details()

        # Initialise the current routing state with execution details
        routing_state = self.routing_model.create_routing_state(universe, routing_state_details)

        # Create a pricing model for assets
        pricing_model = self.pricing_model_factory(self.execution_model, universe, self.routing_model)

        assert pricing_model, "pricing_model_factory did not return a value"

        # Create a valuation model for positions
        valuation_model = self.valuation_model_factory(pricing_model)

        return routing_state, pricing_model, valuation_model

    def tick(self,
             clock: datetime.datetime,
             universe: StrategyExecutionUniverse,
             state: State,
             debug_details: dict,
             cycle_duration: Optional[CycleDuration]=None,
        ) -> dict:
        """Execute the core functions of a strategy.

        :param clock:
            Current timestamp of the execution cycle

        :param universe:
            Loaded trading data

        :param state:
            The current state of the strategy (open position, past trades, visualisation)

        :param debug_details:
            Internal bunch of data used in unit testing

        :param cycle_duration:
            The currenct cycle duration (time between ticks).
            This may be specific in a strategy module, but also overridden for testing.
            This is used only for logging purposes.

        :return: Debug details dictionary where different subsystems can write their diagnostics information what is happening during the dict.
            Mostly useful for integration testing.
        """

        assert isinstance(universe, StrategyExecutionUniverse)

        friendly_cycle_duration = cycle_duration.value if cycle_duration else "-"
        with self.timed_task_context_manager("strategy_tick", clock=clock, cycle_duration=friendly_cycle_duration):

            routing_state, pricing_model, valuation_model = self.setup_routing(universe)
            assert pricing_model, "Routing did not provide pricing_model"

            # Watch incoming deposits
            with self.timed_task_context_manager("sync_portfolio"):
                self.sync_portfolio(clock, universe, state, debug_details)

            # Assing a new value for every existing position
            with self.timed_task_context_manager("revalue_portfolio"):
                self.revalue_portfolio(clock, state, valuation_model)

            # Log output
            self.report_after_sync_and_revaluation(clock, universe, state, debug_details)

            # Run the strategy cycle
            with self.timed_task_context_manager("decide_trades"):
                rebalance_trades = self.on_clock(clock, universe, pricing_model, state, debug_details)
                assert type(rebalance_trades) == list
                debug_details["rebalance_trades"] = rebalance_trades
                logger.info("We have %d trades", len(rebalance_trades))

            # Log output
            self.report_strategy_thinking(clock, universe, state, rebalance_trades, debug_details)

            # Ask user confirmation for any trades
            with self.timed_task_context_manager("confirm_trades"):
                approved_trades = self.approval_model.confirm_trades(state, rebalance_trades)
                assert type(approved_trades) == list
                logger.info("After approval we have %d trades left", len(approved_trades))
                debug_details["approved_trades"] = approved_trades

            # Log output
            self.report_before_execution(clock, universe, state, approved_trades, debug_details)

            # Physically execute the trades
            with self.timed_task_context_manager("execute_trades", trade_count=len(approved_trades)):

                # Unit tests can turn this flag to make it easier to see why trades fail
                check_balances = debug_details.get("check_balances", False)

                self.execution_model.execute_trades(
                    clock,
                    state,
                    approved_trades,
                    self.routing_model,
                    routing_state,
                    check_balances=check_balances)

            # Log output
            self.report_after_execution(clock, universe, state, debug_details)

        return debug_details

    def check_position_triggers(self,
        clock: datetime.datetime,
        state: State,
        universe: StrategyExecutionUniverse,
        stop_loss_pricing_model: PricingModel,
        routing_state: RoutingState,
        ) -> List[TradeExecution]:
        """Check stop loss/take profit for positions.

        Unlike trade balancing in tick()

        - Stop loss/take profit can occur only to any existing positions.
          No new positions are opened.

        - Trading Universe cannot change for these triggers,
          but remains stable between main ticks.

        - check_position_triggers() is much more lightweight and can be called much more frequently,
          even once per minute

        :return:
            List of generated stop loss trades
        """

        assert isinstance(routing_state, RoutingState)
        assert isinstance(stop_loss_pricing_model, PricingModel)

        with self.timed_task_context_manager("check_position_triggers"):

            # We use PositionManager.close_position()
            # to generate trades to close stop loss positions
            position_manager = PositionManager(
                clock,
                universe,
                state,
                stop_loss_pricing_model,
            )

            stop_loss_trades = check_position_triggers(position_manager)

            approved_trades = self.approval_model.confirm_trades(state, stop_loss_trades)

            if approved_trades:
                logger.info("Executing %d stop loss/take profit trades at %s", len(approved_trades), clock)
                self.execution_model.execute_trades(
                    clock,
                    state,
                    approved_trades,
                    self.routing_model,
                    routing_state,
                    check_balances=False)

            return approved_trades

