from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from datetime import datetime
from typing import List, Type

from qstrader.alpha_model.alpha_model import AlphaModel
from tradeexecutor.state.state import State, TradeExecution
from tradeexecutor.strategy.runner import StrategyRunner, PreflightCheckFailed
from tradingstrategy.client import Client
from tradingstrategy.universe import Universe


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

    def on_clock(self, clock: datetime.datetime, state: State) -> List[TradeExecution]:
        pass

    def preflight_check(self, client: Client, universe: Universe, now_: datetime.datetime):
        """Check the data looks more or less sane."""

        if len(universe.exchanges) == 0:
            raise PreflightCheckFailed("Exchange count zero")

        if universe.pairs.get_count() == 0:
            raise PreflightCheckFailed("Pair count zero")

        start, end = universe.get_candle_availability()

        if now_ - end > self.max_data_age:
            raise PreflightCheckFailed(f"We do not have up-to-date data for candles. Last candles are at {end}")







