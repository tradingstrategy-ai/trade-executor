"""How a strategy will calculate the transaction costs when deciding buying/selling assets."""

import abc
import datetime
from decimal import Decimal, ROUND_DOWN
from typing import Callable

from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.universe import TradeExecutorTradingUniverse


class PricingModel(abc.ABC):
    """A helper class to calculate asset prices within the strategy backtesting and live execution.

    Timestamp is passed to the pricing method. However we expect it only be honoured during
    the backtesting - live execution may always use the latest price.
    """

    @abc.abstractmethod
    def get_simple_sell_price(self, ts: datetime.datetime, pair_id: int) -> USDollarAmount:
        """Get simple buy price without the quantity identified.
        """
        pass

    @abc.abstractmethod
    def get_simple_buy_price(self, ts: datetime.datetime, pair_id: int) -> USDollarAmount:
        """Get simple sell price without the quantity identified.
        """
        pass

    @abc.abstractmethod
    def quantize_quantity(self, pair_id: int, quantity: float, rounding=ROUND_DOWN) -> Decimal:
        """Convert any base token quantity to the native token units by its ERC-20 decimals."""


#: Pricing model depends on the trading universe that may change for each strategy tick.
#: Thus, we need to reconstruct pricing model as the start of the each tick.
PricingModelFactory = Callable[[ExecutionModel, TradeExecutorTradingUniverse], PricingModel]

