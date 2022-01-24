from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from datetime import datetime
from typing import List

from qstrader.alpha_model.alpha_model import AlphaModel
from tradeexecutor.state.state import State, TradeExecution
from tradeexecutor.strategy.runner import StrategyRunner


class QSTraderLiveTrader(StrategyRunner):
    """A live trading executor for QSTrade based algorithm."""

    def __init__(self, alpha_model: AlphaModel, timed_task_context_manager: AbstractContextManager, max_data_age=datetime.timedelta(days=2)):
        """

        :param alpha_model:
        :param timed_task_context_manager:
        :param max_data_age: Allow to unit test on old datasets
        """
        super().__init__(timed_task_context_manager)
        self.alpha_model = alpha_model

    def on_data_signal(self):
        pass

    def on_clock(self, clock: datetime.datetime, state: State) -> List[TradeExecution]:
        pass

    def preflight_check(self, client: Client, universe: Universe, now_: datetime.datetime):
        """Check the data looks more or less sane."""
        pass






