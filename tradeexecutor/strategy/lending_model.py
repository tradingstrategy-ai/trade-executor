"""Asset pricing model."""

import abc
import datetime
from logging import getLogger
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Callable, Optional

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.types import USDollarPrice, BlockNumber
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.routing import RoutingModel
from tradeexecutor.strategy.universe_model import StrategyExecutionUniverse
from tradeexecutor.strategy.trade_pricing import TradePricing


class LendingModel(abc.ABC):
    """Calculate interests of various activities.

    - This is an abstract class. Both backtesting and live trading
      will have their own implementation.
    """

    @abc.abstractmethod
    def update_interests(self,
            pair: TradingPairIdentifier,
       ) -> :
        """How much interest has accumulated during the period.

        :param pair:
            Trading pair we are intereted in

        :return:
            Multiplier for the deposit
        """
