"""Strategy execution core.

Define the runner model for different strategy types.
"""

import abc
import datetime
import time
from contextlib import AbstractContextManager
import logging
from io import StringIO
from pprint import pformat
from types import NoneType

from typing import List, Optional, Tuple, cast, Callable

from eth_defi.provider.anvil import is_anvil, mine
from tradeexecutor.backtest.backtest_execution import BacktestExecutionFailed
from tradeexecutor.cli.double_position import check_double_position
from tradeexecutor.ethereum.ethereum_protocol_adapters import EthereumPairConfigurator
from tradeexecutor.ethereum.tx import TransactionBuilder
from tradeexecutor.state.store import StateStore
from tradeexecutor.state.types import BlockNumber, Percent
from tradeexecutor.statistics.core import update_statistics
from tradeexecutor.statistics.statistics_table import StatisticsTable
from tradeexecutor.strategy.account_correction import check_accounts, UnexpectedAccountingCorrectionIssue
from tradeexecutor.strategy.approval import ApprovalModel
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricing
from tradeexecutor.strategy.generic.generic_router import GenericRouting
from tradeexecutor.strategy.generic.generic_valuation import GenericValuation
from tradeexecutor.strategy.pandas_trader.indicator import CreateIndicatorsProtocol
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInputIndicators
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.sync_model import SyncModel
from tradeexecutor.strategy.run_state import RunState
from tradeexecutor.strategy.output import output_positions, DISCORD_BREAK_CHAR, output_trades
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.pricing_model import PricingModelFactory, PricingModel
from tradeexecutor.strategy.routing import RoutingModel, RoutingState
from tradeexecutor.strategy.stop_loss import check_position_triggers
from tradeexecutor.strategy.trade_pricing import PriceImpactToleranceExceeded
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.strategy.universe_model import StrategyExecutionUniverse

from tradeexecutor.state.state import State
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.trade import TradeExecution, TradeFlag
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.strategy.valuation import ValuationModelFactory, ValuationModel, revalue_state

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

    def __init__(
        self,
        timed_task_context_manager: AbstractContextManager,
        execution_model: ExecutionModel,
        approval_model: ApprovalModel,
        valuation_model_factory: ValuationModelFactory,
        sync_model: Optional[SyncModel],
        pricing_model_factory: PricingModelFactory,
        execution_context: ExecutionContext,
        routing_model: Optional[RoutingModel] = None,
        routing_model_factory: Callable[[], RoutingModel] = None,
        run_state: Optional[RunState] = None,
        accounting_checks=False,
        unit_testing=False,
        trade_settle_wait=None,
        parameters: StrategyParameters = None,
        create_indicators: CreateIndicatorsProtocol = None,\
        visualisation=True,
        max_price_impact: Percent | None = None,
    ):
        """
        :param engine_version:
            Strategy execution version.

            Changes function arguments based on this.
            See `StrategyModuleInformation.trading_strategy_engine_version`.
        """
        assert isinstance(execution_context, ExecutionContext)

        if sync_model is not None:
            assert isinstance(sync_model, SyncModel)

        self.timed_task_context_manager = timed_task_context_manager
        self.execution_model = execution_model
        self.approval_model = approval_model
        self.valuation_model_factory = valuation_model_factory
        self.sync_model = sync_model
        self.pricing_model_factory = pricing_model_factory
        self.routing_model = routing_model
        self.run_state = run_state
        self.execution_context = execution_context
        self.accounting_checks = accounting_checks
        self.unit_testing = unit_testing
        self.routing_model_factory = routing_model_factory
        self.parameters = parameters
        self.create_indicators = create_indicators
        self.visualisation = visualisation
        self.max_price_impact = max_price_impact

        # Avoid fat fingering
        if self.max_price_impact:
            assert self.max_price_impact < 0.10, f"max_price_impact set to {self.max_price_impact * 100}%, are you sure this is correct?"

        # We need 60 seconds wait to read balances
        # after trades only on a real trading,
        # Anvil and test nodes are immune for this AFAIK
        if not trade_settle_wait:
            if unit_testing or not execution_context.mode.is_live_trading():
                self.trade_settle_wait = datetime.timedelta(0)
            else:
                self.trade_settle_wait = datetime.timedelta(seconds=60)
        else:
            self.trade_settle_wait = trade_settle_wait

        logger.info(
            "Created strategy runner: %s, engine version: %s, running mode: %s, max_price_impact: %s, routing model: %s",
            self.__class__.__name__,
            self.execution_context.engine_version,
            self.execution_context.mode.name,
            self.max_price_impact,
            routing_model,
        )

        # If planned and executed price is % off then
        # make a warning in the post execution output
        self.execution_warning_tolerance = 0.01

    def __repr__(self):
        """Get a long presentation of internal runner state."""
        dump = pformat(self.__dict__)
        return f"<{self.__class__.__name__}\n" \
               f"{dump}\n" \
               f">"

    @abc.abstractmethod
    def pretick_check(self, ts: datetime.datetime, universe: StrategyExecutionUniverse):
        """Check the universe for good data before a strategy tick is executed.

        If there are data errors, then log and abort with helpful error messages.

        Only relevant for live trading; if backtesting data fails
        it can be diagnosed in the backtesting itself.

        :param client: Trading Strategy client to check server versions etc.

        :param universe: THe currently constructed universe

        :param ts: Real-time clock signal or past clock timestamp in the case of unit testing

        :raise PreflightCheckFailed: In the case we cannot go live
        """
        pass

    def is_progress_report_needed(self) -> bool:
        """Do we log the strategy steps to logger?

        - Disabled for backtesting to speed up

        - Can be enabled by hacking this function if backtesting needs debugging
        """
        # return self.execution_context.mode.is_live_trading() or self.execution_context.mode.is_unit_testing()
        return self.execution_context.mode.is_live_trading() or self.execution_context.mode == ExecutionMode.unit_testing_trading

    def sync_portfolio(
        self,
        strategy_cycle_or_trigger_check_ts: datetime.datetime,
        universe: StrategyExecutionUniverse,
        state: State,
        debug_details: dict,
        end_block: BlockNumber | NoneType = None,
        long_short_metrics_latest: StatisticsTable | None = None,
        post_valuation=True,
    ):
        """Adjust portfolio balances based on the external events.

        External events include

        - Deposits

        - Withdrawals

        - Interest accrued

        - Token rebases

        :param strategy_cycle_or_trigger_check_ts:
            Timestamp for the event trigger

        :param universe:
            Loaded universe

        :param state:
            Currnet strategy state

        :param end_block:
            Sync until this block.

            If not given sync to the lateshish.

        :param debug_details:
            Dictionary of debug data that will be passed down to the callers
            
        :param long_short_metrics_latest:
            Latest long/short statistics table, if available

        """
        assert isinstance(universe, StrategyExecutionUniverse), f"Universe was {universe}"
        reserve_assets = list(universe.reserve_assets)
        assert len(reserve_assets) > 0, "No reserve assets available"
        assert len(reserve_assets) == 1, f"We only support strategies with a single reserve asset, got {self.reserve_assets}"
        token = reserve_assets[0]
        assert token.decimals and token.decimals > 0, f"Reserve asset lacked decimals"

        routing_state, pricing_model, valuation_model = self.setup_routing(universe)

        if end_block is not None:
            # Only msg in live executoin
            logger.info("sync_portfolio() starting at block %s", end_block)

        balance_update_events = self.sync_model.sync_treasury(
            strategy_cycle_or_trigger_check_ts,
            state,
            supported_reserves=reserve_assets,
            end_block=end_block,
            post_valuation=post_valuation,
        )
        assert type(balance_update_events) == list
        if end_block is not None:
            logger.info("Received %d balance update events from the sync", len(balance_update_events))
        for e in balance_update_events:
            logger.trade("Funding flow event: %s", e)

        # Update the debug data for tests with our events
        debug_details["reserve_update_events"] = balance_update_events
        debug_details["total_equity_at_start"] = state.portfolio.calculate_total_equity()
        debug_details["total_cash_at_start"] = state.portfolio.get_cash()

        if self.sync_model.has_position_sync():
            logger.info("Performing sync_positions() for %s", self.sync_model)
            sync_position_events = self.sync_model.sync_positions(
                strategy_cycle_or_trigger_check_ts,
                state,
                cast(TradingStrategyUniverse, universe),
                pricing_model,
            )
            debug_details["sync_position_events"] = sync_position_events
        else:
            logger.info("No sync_positions() needed")

        # If we have any new deposits, let's refresh our stats right away
        # to reflect the new balances
        if len(balance_update_events) > 0:

            with self.timed_task_context_manager("sync_portfolio_stats_refresh"):

                timestamp = strategy_cycle_or_trigger_check_ts

                # Re-value the portfolio with new deposits
                self.revalue_state(
                    timestamp,
                    state,
                    valuation_model,
                )

                update_statistics(
                    timestamp,
                    state.stats,
                    state.portfolio,
                    self.execution_context.mode,
                    strategy_cycle_or_wall_clock=timestamp,
                    long_short_metrics_latest=long_short_metrics_latest,
                    treasury = state.sync.treasury,
                )

    def revalue_state(self, ts: datetime.datetime, state: State, valuation_model: ValuationModel):
        """Revalue portfolio based on the latest prices."""
        revalue_state(state, ts, valuation_model)
        logger.info("After revaluation at %s our portfolio value is %f USD", ts, state.portfolio.calculate_total_equity())

    def collect_post_execution_data(
            self,
            execution_context: ExecutionContext,
            pricing_model: PricingModel,
            trades: List[TradeExecution]):
        """Collect post execution data for all trades.

        - Collect prices after the execution
        - Mostly matters for failed execution only, but we collect for everything
        """

        # Rerun price estimations for the latest block data
        # after the trade has been executed
        for t in trades:

            if execution_context.mode.is_live_trading():
                # In live trading, use the current UTC time to fetch
                # the post execution price info
                ts = datetime.datetime.utcnow()
            else:
                # Backtesting does not yet have a way
                # to simulate slippage
                ts = t.strategy_cycle_at

            logger.info("Fetching post-execution price data for %s at %s", t.get_short_label(), ts)

            # Credit supply pairs do not have pricing ATM
            if t.pair.is_spot() or t.pair.is_vault():
                if t.is_buy():
                    t.post_execution_price_structure = pricing_model.get_buy_price(ts, t.pair, t.planned_reserve)
                else:
                    t.post_execution_price_structure = pricing_model.get_sell_price(ts, t.pair, -t.planned_quantity)
            elif t.pair.is_leverage() and t.is_short():
                spot_pair = t.pair.underlying_spot_pair
                if t.is_sell():
                    t.post_execution_price_structure = pricing_model.get_buy_price(ts, spot_pair, t.planned_collateral_consumption)
                else:
                    t.post_execution_price_structure = pricing_model.get_sell_price(ts, spot_pair, t.planned_quantity)
            elif t.pair.is_credit_supply():
                # For credit supply, no swaps are executed
                t.post_execution_price_structure = None
            else:
                raise AssertionError(f"Unsupported: {t}")

            #
            # Check if we got so bad trade execution we should worry about it
            #

            if t.planned_reserve and t.executed_reserve:
                reserve_drift = abs((t.executed_reserve - t.planned_reserve) / t.planned_reserve)
            else:
                reserve_drift = 0

            if t.planned_quantity and t.executed_quantity:
                quantity_drift = abs((t.executed_quantity - t.planned_quantity) / t.planned_quantity)
            else:
                quantity_drift = 0

            if reserve_drift >= self.execution_warning_tolerance or quantity_drift >= self.execution_warning_tolerance:
                log_level = logging.WARNING
            else:
                log_level = logging.INFO

            logger.log(
                log_level,
                "Trade quantity and reserve match for pre an post-execution for: %s\n  Estimated reserve %s, executed reserve %s\n  Estimated quantity %s, executed quantity %s\n  Reserve drift %f %%, quantity drift %f %%",
                t.get_short_label(),
                t.planned_reserve,
                t.executed_reserve,
                t.planned_quantity,
                t.executed_quantity,
                reserve_drift * 100,
                quantity_drift * 100,
            )

    def on_clock(
        self,
        clock: datetime.datetime,
        universe: StrategyExecutionUniverse,
        pricing_model: PricingModel,
        state: State,
        debug_details: dict,
        indicators:StrategyInputIndicators | None = None,
        routing_state: RoutingState = None,
        routing_model: RoutingModel = None,
    ) -> List[TradeExecution]:
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
        total_equity = portfolio.calculate_total_equity()
        print(f"Portfolio status (before rebalance), tick #{tick}", file=buf)
        print("", file=buf)
        print(f"Total equity: ${portfolio.calculate_total_equity():,.2f}, in cash: ${portfolio.get_cash():,.2f}", file=buf)
        print(f"Life-time positions: {portfolio.next_position_id - 1}, trades: {portfolio.next_trade_id - 1}", file=buf)

        print(DISCORD_BREAK_CHAR, file=buf)

        if len(portfolio.open_positions) > 0:
            print(f"Currently open positions:", file=buf)
            print("", file=buf)
            output_positions(portfolio.open_positions.values(), total_equity, buf)

            print(DISCORD_BREAK_CHAR, file=buf)
        else:
            logger.info("No open positions")

        if portfolio.get_frozen_position_equity() > 0:
            print(f"Frozen positions (${portfolio.get_frozen_position_equity():,.2f}):", file=buf)
            print("", file=buf)
            output_positions(portfolio.frozen_positions.values(), total_equity, buf)

            print(DISCORD_BREAK_CHAR, file=buf)
        else:
            logger.info("No frozen positions")

        print("Reserves:", file=buf)
        print("", file=buf)
        reserve: ReservePosition
        for reserve in state.portfolio.reserves.values():
            print(f"    {reserve.quantity:,.2f} {reserve.asset.token_symbol}", file=buf)

        logger.trade(buf.getvalue())

    def report_before_execution(self, clock: datetime.datetime, universe: StrategyExecutionUniverse, state: State, trades: List[TradeExecution], debug_details: dict):
        buf = StringIO()

        if len(trades) > 0:
            print("New trades to be executed", file=buf)
            print("", file=buf)
            position: TradingPosition
            portfolio = state.portfolio
            output_trades(trades, portfolio, buf)
        else:
            print("No new trades", file=buf)
        logger.trade(buf.getvalue())

    def report_after_execution(
        self,
        cycle: int,
        clock: datetime.datetime,
        universe: StrategyExecutionUniverse,
        state: State,
        debug_details: dict,
        trades: list[TradeExecution],
    ):

        if cycle is None:
            # Legacy unit test path
            cycle = 0

        assert type(cycle) == int, f"Got: {type(cycle)}: {cycle}"
        assert isinstance(clock, datetime.datetime)
        assert type(trades) == list

        buf = StringIO()
        portfolio = state.portfolio

        total_equity = portfolio.calculate_total_equity()
        if total_equity == 0:
            # Should not never happen, but just be prepared
            total_equity = 0.00001

        total_volume = sum(t.get_value() for t in trades)

        print(f"Strategy status after rebalance #{cycle} ({clock})", file=buf)
        print("", file=buf)
        print(f"- Trades made: {len(trades)} trades, with volume of ${total_volume:,.2f}", file = buf)
        print("", file=buf)
        print(f"- Total equity: ${total_equity:,.2f}", file = buf)
        print("", file=buf)
        print(f"- Cash: ${portfolio.get_cash():,.2f} ({portfolio.get_cash() / total_equity * 100:.2f}%)", file=buf)
        print("", file=buf)

        # print(DISCORD_BREAK_CHAR, file=buf)

        if len(portfolio.open_positions) > 0:
            print(f"Opened/open positions:", file=buf)
            print("", file=buf)
            output_positions(portfolio.open_positions.values(), total_equity, buf)

            # print(DISCORD_BREAK_CHAR, file=buf)
        else:
            logger.info("No positions opened")

        closed_positions = list(portfolio.get_positions_closed_at(clock))
        if len(closed_positions) > 0:
            print(f"Closed positions:", file=buf)
            output_positions(closed_positions, total_equity, buf)

            # print(DISCORD_BREAK_CHAR, file=buf)
        else:
            logger.info("The clock tick %s did not close any positions", clock)

        logger.trade_high(buf.getvalue())

    def report_strategy_thinking(
        self,
         strategy_cycle_timestamp: datetime.datetime,
         cycle: int,
         universe: TradingStrategyUniverse,
         state: State,
         trades: List[TradeExecution],
         debug_details: dict
    ):
        """Strategy admin helpers to understand a live running strategy.

        - Post latest variables

        - Draw the single pair strategy visualisation.

        :param strategy_cycle_timestamp:
            real time lock

        :param cycle:
            Cycle number

        :param universe:
            Currnet trading universe

        :param trades:
            Trades executed on this cycle

        :param state:
            Current execution state

        :param debug_details:
            Dict of random debug stuff
        """

    def setup_routing(
        self,
        universe: StrategyExecutionUniverse,
    ) -> Tuple[RoutingState, PricingModel, ValuationModel]:
        """Setups routing state for this cycle.

        :param universe:
            The currently tradeable universe

        :return:
            Tuple(routing state, pricing model, valuation model)
        """

        #
        # Todo: a bit of mess as we support legacy and new generic routing style routing here
        #

        routing_model = self.routing_model

        if routing_model is None:
            # The new style routing initialisation
            logger.info("Lazily initialised the default routing model, as routing model has not been set up earlier")
            self.routing_model = routing_model = self.execution_model.create_default_routing_model(universe)
            logger.info("Lazy routing model initialised: %s", self.routing_model)
        else:
            logger.info("Using passed routing model: %s", self.routing_model)

        assert routing_model, "Routing model not set"

        # Get web3 connection, hot wallet
        routing_state_details = self.execution_model.get_routing_state_details()

        # Lazily initialised routing model
        # TODO: Add a specific app startup step that downloads
        # the universe for the first time and initialises here
        if isinstance(routing_model, GenericRouting):
            assert self.execution_context.is_version_greater_or_equal_than(0, 3, 0), f"Strategy modules need to be at least 0.3 to support GenericRouting, we got version {self.execution_context.engine_version}"
            if not routing_model.is_initialised():
                tx_builder = cast(TransactionBuilder, routing_state_details["tx_builder"])
                web3 = tx_builder.web3
                pair_configurator = EthereumPairConfigurator(
                    web3,
                    cast(TradingStrategyUniverse, universe)
                )
                routing_model.initialise(pair_configurator)

            # Update the pair configuration universe to the latest.
            # This will be referred when creating
            # pricing_model and valuation model
            routing_model.pair_configurator.strategy_universe = universe

        routing_state = self.routing_model.create_routing_state(universe, routing_state_details)

        if isinstance(routing_model, GenericRouting):
            pricing_model = GenericPricing(routing_model.pair_configurator)
            valuation_model = GenericValuation(routing_model.pair_configurator)
        else:
            # Legacy routing logic

            # Create a pricing model for assets
            pricing_model = self.pricing_model_factory(self.execution_model, universe, self.routing_model)

            assert pricing_model, "pricing_model_factory did not return a value"

            # Create a valuation model for positions
            valuation_model = self.valuation_model_factory(pricing_model)

        logger.debug(
            "setup_routing(): routing_state: %s, pricing_model: %s, valuation_model: %s",
             routing_state,
             pricing_model,
             valuation_model
        )

        return routing_state, pricing_model, valuation_model

    def check_balances_post_execution(
        self,
        universe: StrategyExecutionUniverse,
        state: State,
        cycle: int,
    ):
        """Check that on-chain balances matches our internal accounting after executing trades.

        - Crash the execution if the on-chain balance is not what we expect

        - Call after we have stored the execution state in the database
        """

        # We cannot call account check right after the trades,
        # as meny low quality nodes might still report old token balances
        # from eth_call
        logger.info("Waiting on-chain balances to settle for %s before performing accounting checks", self.trade_settle_wait)
        time.sleep(self.trade_settle_wait.total_seconds())

        # Double check we handled incoming trade balances correctly
        with self.timed_task_context_manager("check_accounts_post_trade"):
            # end_block = self.execution_model.get_safe_latest_block()
            # Always use the latest block here, not safe block,
            # to work around anvil + mainnet fork issues
            web3 = self.execution_model.web3
            if is_anvil(web3):
                mine(web3)
                end_block = self.execution_model.web3.eth.block_number
            else:
                end_block = self.execution_model.get_safe_latest_block()
            logger.info("Post-trade accounts balance check for block %s, cycle %d", end_block, cycle)
            self.check_accounts(
                universe,
                state,
                end_block=end_block,
                cycle=cycle,
            )

    def tick(
        self,
        strategy_cycle_timestamp: datetime.datetime,
        universe: StrategyExecutionUniverse,
        state: State,
        debug_details: dict,
        cycle_duration: Optional[CycleDuration] = None,
        cycle: Optional[int] = None,
        store: Optional[StateStore] = None,
        long_short_metrics_latest: StatisticsTable | None = None,
        indicators: StrategyInputIndicators | None = None,
    ) -> dict:
        """Execute the core functions of a strategy.

        TODO: This function is vulnerable to balance changes in the middle of execution.
        It's not possible to fix this until we have atomic rebalances.

        :param strategy_cycle_timestamp:
            Current timestamp of the execution cycle.

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

        :param cycle:
            Strategy cycle number

        :param execution_context:
            Live or backtesting

        :param indicators:
            Precalculated backtest or live calculated indicator values.

        :return: Debug details dictionary where different subsystems can write their diagnostics information what is happening during the dict.
            Mostly useful for integration testing.
        """

        assert isinstance(universe, StrategyExecutionUniverse)

        assert isinstance(strategy_cycle_timestamp, datetime.datetime)

        if cycle_duration not in (CycleDuration.cycle_unknown, CycleDuration.cycle_1s, None):
            assert strategy_cycle_timestamp.second == 0, f"Cycle duration {cycle_duration}: Does not look like a cycle timestamp: {strategy_cycle_timestamp}, should be even minutes"

        end_block = self.execution_model.get_safe_latest_block()

        logger.info("tick() at block %s", end_block)

        friendly_cycle_duration = cycle_duration.value if cycle_duration else "-"
        with self.timed_task_context_manager("strategy_tick", clock=strategy_cycle_timestamp, cycle_duration=friendly_cycle_duration):

            routing_state, pricing_model, valuation_model = self.setup_routing(universe)
            assert pricing_model, "Routing did not provide pricing_model"

            # Needed for Lagoon.
            # sync_portfolio() posts valuation onchain to
            # process the deposit/redemption queue and
            # for this we need up-to-date valuation
            # if self.sync_model.has_async_deposits():
            with self.timed_task_context_manager("revalue_portfolio_before_sync"):
                self.revalue_state(strategy_cycle_timestamp, state, valuation_model)

            # Watch incoming deposits
            with self.timed_task_context_manager("sync_portfolio"):
                self.sync_portfolio(
                    strategy_cycle_timestamp,
                    universe,
                    state,
                    debug_details,
                    end_block,
                    long_short_metrics_latest=long_short_metrics_latest,
                    post_valuation=True,
                )

            # Double check we handled deposits correctly
            with self.timed_task_context_manager("check_accounts_pre_trade"):
                logger.info("Pre-trade accounts balance check")
                self.check_accounts(universe, state, end_block)

            # Assing a new value for every existing position
            # with self.timed_task_context_manager("revalue_portfolio"):
            #    self.revalue_state(strategy_cycle_timestamp, state, valuation_model)

            # Log output
            if self.is_progress_report_needed():
                self.report_after_sync_and_revaluation(strategy_cycle_timestamp, universe, state, debug_details)

            # Check if we do have any money yo trade or not.
            # Otherwise we are going to crash with "not enough USDC to open a trade" errors
            execution_context = self.execution_context

            # TODO: Due to the legacy some tests assume they run with zero capital,
            # and we have a flag to check it for here
            if state.portfolio.has_trading_capital() or execution_context.mode.is_unit_testing():

                old_position_ids = set(state.portfolio.open_positions.keys())

                # Run the strategy cycle main trading decision cycle
                with self.timed_task_context_manager("decide_trades"):
                    rebalance_trades = self.on_clock(
                        strategy_cycle_timestamp,
                        universe,
                        pricing_model,
                        state,
                        debug_details,
                        indicators=indicators,
                        routing_state=routing_state,
                        routing_model=self.routing_model,
                    )
                    assert type(rebalance_trades) == list
                    debug_details["rebalance_trades"] = rebalance_trades

                    # Make some useful diagnostics output for log files to troubleshoot if something
                    # when wrong internally
                    # if self.execution_context.live_trading:
                    #     _, last_point_at = state.visualisation.get_timestamp_range()
                    #     logger.info("We have %d new trades, %d total visualisation points, last visualisation point at %s",
                    #                 len(rebalance_trades),
                    #                 state.visualisation.get_total_points(),
                    #                 last_point_at
                    #                 )

                    # Check that we did not get duplicate trades for some reason,
                    # like API bugs
                    trade_set = set()
                    for t in rebalance_trades:
                        assert t not in trade_set, f"decide_trades() returned a duplicate trade: {t}"
                        trade_set.add(t)

                    # logger.info("decide_trades() returned %d trades", len(rebalance_trades))

                new_position_ids = set(state.portfolio.open_positions.keys())
                if old_position_ids != new_position_ids and len(trade_set) == 0:
                    # Handle user error within decide_trades() that they forget to return trades list.
                    # 1. PositionManager.open_spot() called
                    # 2. Trade is created
                    # 3. This trade is not returned
                    if not state.portfolio.pending_positions:  # The strategy might have created market limit positions that do not open on this cycle
                        raise RuntimeError(f"decide_trades() returned empty trade list, but new positions were added.\n"
                                           f"This is likely a bug in your decide_trades() - remember to return trades list for any position modifications you do,\n"
                                           f"and do not accidentally return empty list when you have made trades.\n"
                                           f"Old positions before decide_trades(): {old_position_ids}\n"
                                           f"New positions after decide_trades(): {new_position_ids}\n")

                rebalance_trades = post_process_trade_decision(
                    state,
                    execution_context,
                    rebalance_trades,
                    max_price_impact=self.max_price_impact,
                )

                # If we have no trades, then we are done
                # Log what our strategy decided
                if self.is_progress_report_needed():
                    self.report_strategy_thinking(
                        strategy_cycle_timestamp=strategy_cycle_timestamp,
                        cycle=cycle,
                        universe=universe,
                        state=state,
                        trades=rebalance_trades,
                        debug_details=debug_details,)

                # Shortcut quit here if no trades are needed
                if len(rebalance_trades) == 0:
                    logger.trade_high(
                        "Rebalance: #%d (%s), No action taken: strategy decided not to open or close any positions",
                        cycle or 0,
                        strategy_cycle_timestamp
                    )
                    return debug_details

                # Ask user confirmation for any trades
                with self.timed_task_context_manager("confirm_trades"):
                    approved_trades = self.approval_model.confirm_trades(state, rebalance_trades)
                    assert type(approved_trades) == list
                    logger.info("After approval we have %d trades left", len(approved_trades))
                    debug_details["approved_trades"] = approved_trades

                with self.timed_task_context_manager("sort_trades"):
                    # Order trades to a natural order, so we have always the correct cash in hand.
                    # Any credit market withdraw will be executed first, then sells, then buys and Aave deposits last.
                    sorted_approved_trades = prepare_sorted_trades(approved_trades)

                    if approved_trades != sorted_approved_trades:
                        logger.info(
                            "Cycle %d, trades re-sorted to to: %s",
                            cycle or 0,
                            sorted_approved_trades,
                        )

                # Log output
                if self.is_progress_report_needed():
                    self.report_before_execution(strategy_cycle_timestamp, universe, state, sorted_approved_trades, debug_details)

                # Unit tests can turn this flag to make it easier to see why trades fail
                check_balances = debug_details.get("check_balances", False)

                # Physically execute the trades
                with self.timed_task_context_manager("execute_trades", trade_count=len(sorted_approved_trades), check_balances=check_balances):

                    # Make sure our hot wallet nonce is up to date
                    self.sync_model.resync_nonce()

                    # Bail out if we generated double position
                    logger.info("Double open position tripwire check before executing trades")
                    check_double_position(
                        state,
                        printer=logger.error,
                        crash=True,
                    )

                    # Sync state before broadcasting,
                    # so we have generated tx hashes on the disk
                    # and trades flagged with broadcasting/broadcasted status.
                    # This allows us to recover and rebroadcast,
                    # if the execution crashes e.g. due to blockchain being down,
                    # node issues, or gas fee spikes
                    if self.execution_context.mode.is_live_trading():
                        if store is not None:
                            logger.info("Syncing state file before the trade execution")
                            store.sync(state)

                    try:
                        self.execution_model.execute_trades(
                            strategy_cycle_timestamp,
                            state,
                            sorted_approved_trades,
                            self.routing_model,
                            routing_state,
                            check_balances=check_balances)
                    except BacktestExecutionFailed as e:
                        # Add more diagnostics info to the exception
                        # to make strategy development easier
                        msg = "execute_trades() failed - likely trying to have larger trades than having cash in hand.\n"
                        msg += "Currently open positions:\n"
                        for p in state.portfolio.open_positions.values():
                            msg += f"{p}, quantity: {p.get_quantity()}, planned quantiy: {p.get_quantity(planned=True)}\n"
                        msg += "Trade queue:\n"
                        for t in approved_trades:
                            msg += f"{t} - execution order {t.get_execution_sort_position() + 1}.\n"
                        raise BacktestExecutionFailed(msg) from e

                with self.timed_task_context_manager("post_execution"):
                    self.collect_post_execution_data(
                        self.execution_context,
                        pricing_model,
                        approved_trades,
                    )

            else:
                equity = state.portfolio.calculate_total_equity()
                logger.trade("Strategy has no trading capital and trade decision step was skipped. The total equity is %f USD, execution mode is %s", equity, execution_context.mode.name)

            # Log output
            if self.is_progress_report_needed():
                self.report_after_execution(
                    cycle=cycle,
                    clock=strategy_cycle_timestamp,
                    universe=universe,
                    state=state,
                    debug_details=debug_details,
                    trades=approved_trades,
                )

        return debug_details

    def check_position_triggers(self,
        clock: datetime.datetime,
        state: State,
        universe: StrategyExecutionUniverse,
        stop_loss_pricing_model: PricingModel,
        routing_state: RoutingState,
        long_short_metrics_latest: StatisticsTable | None = None,
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

        if routing_state is None:
            # Dummy executoin model
            return

        assert isinstance(routing_state, RoutingState)
        assert isinstance(stop_loss_pricing_model, PricingModel)

        debug_details = {}

        end_block = self.execution_model.get_safe_latest_block()
        if end_block is not None:
            # Only log in live execution
            logger.info(f"check_position_triggers() using block %s", end_block)

        timestamp = datetime.datetime.utcnow()
        universe = cast(TradingStrategyUniverse, universe)

        with self.timed_task_context_manager("check_position_triggers"):

            # Don't post valuation on Lagoon when we check the triggers
            if not self.sync_model.has_async_deposits():
                with self.timed_task_context_manager("sync_portfolio_before_triggers"):
                    # Sync treasure before the trigger checks
                    self.sync_portfolio(
                        clock,
                        universe,
                        state,
                        debug_details,
                        end_block=end_block,
                        long_short_metrics_latest=long_short_metrics_latest,
                        # Stop loss checks should not trigger any valuation updates
                        post_valuation=True,
                    )

            # We need to sync interest before we can run check accounts
            # but after sync treasury since new deposit / redemption can break the interest calculations
            with self.timed_task_context_manager("sync_interest_before_triggers"):
                if not self.execution_context.mode.is_backtesting():
                    # Only run in live execution to speed up backtesting
                    interest_events = self.sync_model.sync_interests(
                        timestamp,
                        state,
                        universe,
                        stop_loss_pricing_model,
                    )
                    logger.info("Generated %d sync interest events", len(interest_events))

            # Check that our accounting is intact before proceeding
            with self.timed_task_context_manager("check_accounts_before_triggers"):
                self.check_accounts(
                    universe,
                    state,
                    report_only=True,
                    end_block=end_block
                )

            # We use PositionManager.close_position()
            # to generate trades to close stop loss positions
            position_manager = PositionManager(
                clock,
                universe,
                state,
                stop_loss_pricing_model,
                default_slippage_tolerance=self.parameters.get("slippage_tolerance") if self.parameters else None,
            )

            triggered_trades = check_position_triggers(position_manager, self.execution_context)
            triggered_trades = post_process_trade_decision(
                state,
                self.execution_context,
                triggered_trades,
                max_price_impact=self.max_price_impact,
            )
            approved_trades = self.approval_model.confirm_trades(state, triggered_trades)
            if approved_trades:
                logger.info("Executing %d stop loss/take profit trades at %s", len(approved_trades), clock)
                self.execution_model.execute_trades(
                    clock,
                    state,
                    approved_trades,
                    self.routing_model,
                    routing_state,
                    check_balances=False,
                    triggered=True,
                )

            return approved_trades

    def repair_state(self, state: State) -> List[TradeExecution]:
        """Repair unclean state issues.

        Currently supports

        - Fixing unfinished trades

        :return:
            List of fixed trades
        """

        logger.info("Reparing the state")

        repaired = []
        repaired += self.execution_model.repair_unconfirmed_trades(state)
        return repaired

    def refresh_visualisations(self, state: State, universe: TradingStrategyUniverse):
        """Update the visualisations in the run state.

        This will update `RunState.visualisations` for the current strategy.

        - In-process memory charts are served by webhook

        - In-process memory charts are posted to Discord, etc.

        - This is called on the startup, so that we have immediately good visualisation
          to show over the webhook when the web server boots up

        - This is called after each strategy thinking cycle is complete.

        The function is overridden by the child class for actual strategy runner specific implementation.
        """

    def check_accounts(
        self,
        universe: TradingStrategyUniverse,
        state: State,
        report_only=False,
        end_block: BlockNumber | NoneType = None,
        cycle: int | None = None,
    ):
        """Perform extra accounting checks on live trading startup.

        Must be enabled in the settings. Enabled by default for live trading.

        :param report_only:
            Don't crash if we get problems in accounts

        :param end_block:
            Check specifically at this block.

            If not given use the lateshish block.

        :raise UnexpectedAccountingCorrectionIssue:
            Aborting execution.

        """

        assert isinstance(universe, TradingStrategyUniverse)

        # Enzyme tests
        if len(state.portfolio.reserves) == 0:
            logger.info("No reserves, skipping accounting checks")
            return

        if self.accounting_checks:
            clean, df = check_accounts(
                universe.data_universe.pairs,
                [universe.get_reserve_asset()],
                state,
                self.sync_model,
                block_identifier=end_block,
            )

            log_level = logging.INFO if report_only else logging.ERROR

            address = self.execution_model.get_balance_address()

            if not clean:
                block_message = f"{end_block:,}" if end_block else "<latest>"
                logger.log(
                    log_level,
                    f"Accounting differences detected for: %s at block {block_message}, cycle {cycle}\n"                    
                    "Differences are:\n"
                    "%s",
                    address,
                    df.to_string()
                )

                if not report_only:
                    logger.error("Aborting execution as we cannot reliable trade with incorrect balances.")
                    raise UnexpectedAccountingCorrectionIssue("Aborting execution as we cannot reliable trade with incorrect balances.")
        else:
            # Path taken by some legacy tests
            # Too noisy
            # logger.info("Accounting checks disabled - skipping")
            pass


def post_process_trade_decision(
    state: State,
    execution_context: ExecutionContext,
    trades: List[TradeExecution],
    max_price_impact: Percent | None = None,
):
    """Set any extra flags and do extra checks on trades.

    - Called before trades are send to the execution

    - Mainly to deal with the fact that if trades close a final position on lending

    :param max_price_impact:
        What is the allowed maximum price impact of a single trade.

        Trades must have their `trade.price_structure` data filled
        to detect.

    :raise PriceImpactToleranceExceeded:
        If any of the trades is detected to have too much price impact.
    """

    if not execution_context.mode.is_backtesting():
        # Too noisy for backtesting
        logger.info(
            "post_process_trade_decision(): Post-processing %d trades, max_price_impact: %s",
            len(trades),
            max_price_impact,
        )

    # TODO: Write a full logic here, only supports closing shorts now,
    # assuming everything lending is short
    lending_positions_open = [p for p in state.portfolio.open_positions.values() if p.is_leverage()]
    lending_position_closing_trades = [t for t in trades if t.pair.is_leverage() and TradeFlag.close in t.flags]
    assert len(lending_positions_open) >= len(lending_position_closing_trades), "We cannot close more than we have open"

    # TODO: Currently we do not pass enough information in :py:class:`TradingPairIdentifier`
    # so here we take a hack shortcut to set close_protocol_all.
    if len(lending_position_closing_trades) == len(lending_positions_open) and len(lending_positions_open) > 0:
        lending_position_closing_trades[-1].flags.add(TradeFlag.close_protocol_last)

    if max_price_impact is not None:
        for t in trades:

            if t.is_credit_supply():
                # Credit supply positions do not have price structure
                continue

            price_structure = t.price_structure

            if execution_context.live_trading:
                # TODO: Why not always filled?
                # Make sure PositionManager passes trade.price_structure along
                assert price_structure is not None, f"Live trading running, but trade missing price structure {t}"

            if price_structure is not None:
                impact = t.price_structure.get_price_impact()
                if impact > max_price_impact:
                    raise PriceImpactToleranceExceeded(
                        f"Trade {t} has too much price impact\n"
                        f"Maximum allowed: {max_price_impact * 100} %\n"
                        f"Trade impact: {impact * 100} %\n"
                        f"Trade pricing: {t.price_structure}\n"
                    )
                t.price_impact_tolerance = max_price_impact


    return trades


def prepare_sorted_trades(approved_trades: list[TradeExecution]) -> list[TradeExecution]:
    """Set correct execution order of trades.

    - Take trades in any order
    - Sells must go before buys, so that we have always enough cash in hand
    """
    sorted_approved_trades = sorted(approved_trades, key=lambda t: t.get_execution_sort_position())
    # Store sort_index for later debugging
    for idx, trade in enumerate(sorted_approved_trades):
        trade.sort_index = idx
    return sorted_approved_trades