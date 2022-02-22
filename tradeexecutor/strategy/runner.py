"""Define a live strategy execution model."""

import abc
import datetime
import textwrap
from contextlib import AbstractContextManager
import logging
from dataclasses import dataclass

import pandas as pd
from typing import List, Optional

from tradeexecutor.strategy.approval import ApprovalModel
from tradeexecutor.strategy.execution import ExecutionModel
from tradeexecutor.state.revaluation import RevaluationMethod
from tradeexecutor.state.sync import SyncMethod
from tradingstrategy.client import Client

from tradeexecutor.state.state import State, TradeExecution, AssetIdentifier
from tradingstrategy.exchange import ExchangeUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.strategy.pricingmethod import PricingMethod

logger = logging.getLogger(__name__)


class PreflightCheckFailed(Exception):
    """Something was wrong with the datafeeds."""


@dataclass
class Dataset:
    """Contain raw loaded datasets."""

    time_frame: TimeBucket
    exchanges: ExchangeUniverse
    pairs: pd.DataFrame
    candles: pd.DataFrame
    liquidity: pd.DataFrame


class StrategyRunner(abc.ABC):
    """A base class for a strategy live trade executor."""

    def __init__(self,
                 timed_task_context_manager: AbstractContextManager,
                 execution_model: ExecutionModel,
                 approval_model: ApprovalModel,
                 revaluation_method: RevaluationMethod,
                 sync_method: SyncMethod,
                 pricing_method: PricingMethod,
                 reserve_assets: List[AssetIdentifier]):
        self.timed_task_context_manager = timed_task_context_manager
        self.execution_model = execution_model
        self.approval_model = approval_model
        self.revaluation_method = revaluation_method
        self.sync_method = sync_method
        self.pricing_method = pricing_method
        #: TODO: Make something more sensible how to the list of reseve assets are managed
        self.reserve_assets = reserve_assets

    def load_data(self, time_frame: TimeBucket, client: Client, lookback: Optional[datetime.timedelta]=None) -> Optional[Dataset]:
        """Loads the server-side data using the client.

        :param client: Client instance. Note that this cannot be stable across ticks, as e.g. API keys can change. Client is recreated for every tick.

        :param lookback: how long to the past load data e.g. 1 year, 1 month. **Not implemented yet**.

        :return: None if not dataset for the strategy required
        """
        with self.timed_task_context_manager("load_data", time_frame=time_frame.value):
            exchanges = client.fetch_exchange_universe()
            pairs = client.fetch_pair_universe().to_pandas()
            candles = client.fetch_all_candles(time_frame).to_pandas()
            liquidity = client.fetch_all_liquidity_samples(time_frame).to_pandas()
            return Dataset(
                time_frame=time_frame,
                exchanges=exchanges,
                pairs=pairs,
                candles=candles,
                liquidity=liquidity,
            )

    def setup_universe_timed(self, dataset: Optional[Dataset]) -> Optional[Universe]:
        """Time the setting up of the universe.

        :return: None if no universe for the strategy required
        """
        if dataset:
            with self.timed_task_context_manager("setup_universe", time_frame=dataset.time_frame.value):
                universe = self.construct_universe(dataset)
                data_start, data_end = universe.candles.get_timestamp_range()
                logger.info(textwrap.dedent(f"""
                        Universe constructed.                    
                        
                        Time periods
                        - Time frame {universe.time_frame.value}
                        - Candle data: {data_start} - {data_end}
                        
                        The size of our trading universe is
                        - {len(universe.exchanges)} exchanges
                        - {universe.pairs.get_count()} pairs
                        - {universe.candles.get_sample_count()} candles
                        - {universe.liquidity.get_sample_count()} liquidity samples                
                        """))
                return universe
        else:
            logger.info("Strategy did not load dataset, universe not generated")
            return None

    @abc.abstractmethod
    def construct_universe(self, dataset: Dataset) -> Universe:
        """Sets up pairs, candles and liquidity samples.

        :param client: Client instance. Note that this cannot be stable across ticks, as e.g. API keys can change. Client is recreated for every tick.
        :return:
        """

    @abc.abstractmethod
    def preflight_check(self, client: Client, universe: Universe, now_: datetime.datetime):
        """Called when the trade executor instance is started.

        :param client: Trading Strategy client to check server versions etc.

        :param universe: THe currently constructed universe

        :param now_: Real-time clock signal or past clock timestamp in the case of unit testing

        :raise PreflightCheckFailed: In the case we cannot go live
        """
        pass

    @abc.abstractmethod
    def get_strategy_time_frame(self) -> Optional[TimeBucket]:
        """Return the default candle time bucket used in this strategy.

        :return: None if not relevant for the strategy
        """

    def sync_portfolio(self, ts: datetime.datetime, state: State, debug_details: dict):
        """Adjust portfolio balances based on the external events.

        External events include
        - Deposits
        - Withdrawals
        - Interest accrued
        - Token rebases
        """
        reserve_update_events = self.sync_method(state.portfolio, ts, self.reserve_assets)
        assert type(reserve_update_events) == list
        debug_details["reserve_update_events"] = reserve_update_events
        debug_details["total_equity_at_start"] = state.portfolio.get_total_equity()
        debug_details["total_cash_at_start"] = state.portfolio.get_current_cash()


    def revalue_portfolio(self, ts: datetime.datetime, state: State):
        """Revalue portfolio based on the data."""
        state.revalue_positions(ts, self.revaluation_method)

    def on_data_signal(self):
        pass

    def on_clock(self, clock: datetime.datetime, universe: Universe, state: State, debug_details: dict) -> List[TradeExecution]:
        return []

    def tick(self, clock: datetime.datetime, universe: Universe, state: State) -> dict:
        """Perform the strategy main tick.

        :return: Debug details dictionary where different subsystems can write their diagnostics information what is happening during the dict.
            Mostly useful for integration testing.
        """

        debug_details = {"clock": clock}

        with self.timed_task_context_manager("strategy_tick", clock=clock):

            with self.timed_task_context_manager("sync_portfolio"):
                self.sync_portfolio(clock, state, debug_details)

            with self.timed_task_context_manager("revalue_portfolio"):
                self.revalue_portfolio(clock, state)

            with self.timed_task_context_manager("decide_trades"):
                new_trades = self.on_clock(clock, universe, state, debug_details)
                assert type(new_trades) == list
                logger.info("We have %d trades", len(new_trades))

            with self.timed_task_context_manager("confirm_trades"):
                approved_trades = self.approval_model.confirm_trades(state, new_trades)
                assert type(approved_trades) == list
                logger.info("After approval we have %d trades left", len(new_trades))

            with self.timed_task_context_manager("execute_trades"):
                self.execution_model.execute_trades(clock, approved_trades)

        logger.info("Tick complete %s", clock)
        return debug_details
