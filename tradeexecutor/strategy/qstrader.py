from abc import ABC, abstractmethod
import datetime
from contextlib import AbstractContextManager
from typing import List, Type
import logging

import pandas as pd

from qstrader.alpha_model.alpha_model import AlphaModel
from qstrader.asset.universe.static import StaticUniverse
from qstrader.data.backtest_data_handler import BacktestDataHandler
from tradeexecutor.state.state import State, TradeExecution
from tradeexecutor.strategy.qstrader.pcm import PortfolioConstructionModel
from tradeexecutor.strategy.runner import StrategyRunner, PreflightCheckFailed
from tradingstrategy.client import Client
from tradingstrategy.frameworks.qstrader import TradingStrategyDataSource
from tradingstrategy.universe import Universe


logger = logging.getLogger(__name__)


class QSTraderLiveTrader(StrategyRunner):
    """A live trading executor for QSTrade based algorithm."""

    def __init__(self, alpha_model_factory: Type, timed_task_context_manager: AbstractContextManager, max_data_age: datetime.timedelta):
        """
        :param alpha_model:
        :param timed_task_context_manager:
        :param max_data_age: Allow to unit test on old datasets
        """
        super().__init__(timed_task_context_manager)
        self.alpha_model_factory = alpha_model_factory
        self.max_data_age = max_data_age

    def on_data_signal(self):
        pass

    def on_clock(self, clock: datetime.datetime, universe: Universe, state: State) -> List[TradeExecution]:
        raise NotImplementedError()

    def preflight_check(self, client: Client, universe: Universe, now_: datetime.datetime):
        """Check the data looks more or less sane."""

        if len(universe.exchanges) == 0:
            raise PreflightCheckFailed("Exchange count zero")

        if universe.pairs.get_count() == 0:
            raise PreflightCheckFailed("Pair count zero")

        start, end = universe.get_candle_availability()

        if now_ - end > self.max_data_age:
            raise PreflightCheckFailed(f"We do not have up-to-date data for candles. Last candles are at {end}")

    def tick(self, clock: datetime.datetime, universe: Universe, state: State) -> List[TradeExecution]:
        """Run one strategy tick."""

        logger.info("Processing tick %s", clock)

        # TODO: Most of QSTrader execuion parts need to be rewritten to support these things better
        data_source = TradingStrategyDataSource(
            universe.exchanges,
            universe.pairs,
            universe.candles)

        strategy_assets = list(data_source.asset_bar_frames.keys())
        strategy_universe = StaticUniverse(strategy_assets)

        data_handler = BacktestDataHandler(strategy_universe, data_sources=[data_source])

        # TODO: PortfolioConstructionModel is using trading pairs instead of underlying assets
        # - adding routing support for AlphaModel is going to be major work
        alpha_model = self.alpha_model_factory(universe)

        pcm = PortfolioConstructionModel(universe, state, alpha_model)
        rebalance_orders = pcm(pd.Timestamp(clock))







