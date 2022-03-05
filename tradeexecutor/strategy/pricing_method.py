import abc
import datetime
from decimal import Decimal, ROUND_DOWN

from tradeexecutor.state.state import TradingPairIdentifier
from tradeexecutor.state.types import USDollarAmount


class PricingMethod(abc.ABC):
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