import abc
import datetime

from tradeexecutor.state.state import TradingPairIdentifier
from tradeexecutor.state.types import USDollarAmount


class PricingMethod(abc.ABC):
    """A helper class to calculate asset prices within the strategy backtesting and live execution.

    Timestamp is passed to the pricing method. However we expect it only be honoured during
    the backtesting - live execution may always use the latest price.
    """

    @abc.abstractmethod
    def get_simple_sell_price(self, ts: datetime.datetime, pair: TradingPairIdentifier) -> USDollarAmount:
        """Get simple buy price without the quantity identified.
        """
        pass

    @abc.abstractmethod
    def get_simple_buy_price(self, ts: datetime.datetime, pair: TradingPairIdentifier) -> USDollarAmount:
        """Get simple sell price without the quantity identified.
        """
        pass